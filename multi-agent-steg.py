import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
import wandb
from typing import Dict, Tuple, Optional, List

class StegEnv():
    def __init__(self, 
                 tokenizer: AutoTokenizer,
                 batch_size: int = 16,
                 ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.key_length = 4 # must be >= 4 for now because of ppo_train

        self.prompts = [" 0 7 3 8", "This morning I went to the ", "The weather today is ", "What is your favorite "]
        self.key_tokens = [' 0', ' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9']

        self.enc_response_len = 4
        self.dec_response_len = self.key_length

        self.prompt_batch = None
        self.key_batch = None


    def _get_queries(self):
        return ["key:" + key + " prompt:" + prompt for key, prompt in zip(self.key_batch, self.prompt_batch)]
    
    def _get_obs(self):
        queries = self._get_queries()
        query_pt = [self.tokenizer.encode(query, return_tensors="pt").squeeze() for query in queries]
        prompt_pt = [self.tokenizer.encode(prompt, return_tensors="pt").squeeze() for prompt in self.prompt_batch]
        return {
            "query": queries,
            "query_pt": query_pt,
            "prompt": self.prompt_batch,
            "prompt_pt": prompt_pt,
            "key": self.key_batch
        }

    def reset(self, ):
        prompt_idxs = torch.randint(len(self.prompts), size=(self.batch_size,))
        key_idxs = torch.randint(len(self.key_tokens), size=(self.batch_size, self.key_length))

        self.prompt_batch = [self.prompts[idx] for idx in prompt_idxs]
        self.key_batch = ["".join([self.key_tokens[idx] for idx in idxs]) for idxs in key_idxs]

        print(self.prompt_batch)
        print(self.key_batch)

        obs = self._get_obs()

        return obs

    def _reward_function(self, responses_encoder, responses_decoder):

        rewards_encoder = []
        rewards_decoder = []

        for response_enc, response_dec, key in zip(responses_encoder, responses_decoder, self.key_batch):
            key_pt = self.tokenizer.encode(key, return_tensors="pt").squeeze()

            # if response_dec_txt == key:
            #     reward_enc, reward_dec = 1.0, 1.0
            # else:
            #     reward_enc, reward_dec = 0.0, 0.0

            reward_enc = reward_dec = (key_pt == response_dec).sum().float()

            rewards_encoder.append(reward_enc)
            rewards_decoder.append(reward_dec)

        return (rewards_encoder, rewards_decoder)

    def step(self, responses_encoder, responses_decoder):

        reward = self._reward_function(responses_encoder, responses_decoder)
        return reward

class StegPPOTrainer():
    def __init__(self,
            multi_agent = True,
        ):
        
        self.multi_agent = multi_agent
        batch_size = 16
        self.num_episodes = 100

        # 1. load a pretrained model
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
        self.model_ref = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')

        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # initialize environment
        self.env = StegEnv(
            tokenizer = self.tokenizer,
            batch_size = batch_size
        )

        # get tokens to suppress
        key_token_ids = self.tokenizer(self.env.key_tokens)['input_ids']
        self.suppress_tokens = [i for i in range(self.tokenizer.vocab_size) if [i] not in key_token_ids]

        # 2. initialize trainer
        config = PPOConfig(
            batch_size=batch_size * 2, # double for encoder + decoder responses
            learning_rate=1e-5,
            steps=5000,
            )
        self.ppo_trainer = PPOTrainer(config, self.model, self.model_ref, self.tokenizer)

        wandb.init(
            # Set the project where this run will be logged
            project="my-awesome-project",
            # Track hyperparameters and run metadata
            )
        
    def log_stats(
        self,
        stats: dict,
        batch: dict,
        rewards: List[torch.FloatTensor],
    ):
        """
        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]):
                A dictionary of training stats.
            batch (dict[str, Any]):
                A dictionary of batch data, this contains the queries and responses.
            rewards (`List[torch.FloatTensor]`):
                A tensor of rewards.
        """
        logs = {}

        # Log stats
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards) #.to(self.current_device)

        logs.update(stats)

        # manually cast in fp32 for bf16 torch tensors
        for k, v in logs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                logs[k] = v.float()

        logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
        logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
        logs["env/reward_dist"] = rewards.cpu().numpy()

        wandb.log(logs)

    def get_generation_kwargs(self):

        encoder_generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": self.env.enc_response_len
        }

        decoder_generation_kwargs = {"suppress_tokens": self.suppress_tokens, 
                                    "max_new_tokens": self.env.dec_response_len, 
                                    **encoder_generation_kwargs}
        
        return encoder_generation_kwargs, decoder_generation_kwargs

    def get_model_responses(self, obs):
        
        encoder_queries = []
        decoder_queries = []
        encoder_responses = []
        decoder_responses = []

        enc_gen_kwargs, dec_gen_kwargs = self.get_generation_kwargs()

        for query_encoder_pt, prompt_encoder_pt in zip(obs['query_pt'], obs['prompt_pt']):

            # get encoder response on prompt
            response_encoder_pt = self.ppo_trainer.generate(query_encoder_pt, **enc_gen_kwargs)
            response_encoder_pt = response_encoder_pt.squeeze()[-enc_gen_kwargs["max_new_tokens"]:]
            encoder_queries.append(query_encoder_pt)
            encoder_responses.append(response_encoder_pt)

            # construct decoder query which exclues key
            query_decoder_pt = torch.cat((prompt_encoder_pt, response_encoder_pt))

            # get decoder response
            response_decoder_pt = self.ppo_trainer.generate(query_decoder_pt, **dec_gen_kwargs)
            response_decoder_pt = response_decoder_pt.squeeze()[-dec_gen_kwargs["max_new_tokens"]:]
            
            decoder_queries.append(query_decoder_pt)
            decoder_responses.append(response_decoder_pt)

        return encoder_queries, encoder_responses, decoder_queries, decoder_responses

    def train(self):

        for _ in range(self.num_episodes):
            obs = self.env.reset()
            encoder_queries, encoder_responses, decoder_queries, decoder_responses = self.get_model_responses(obs)
            encoder_rewards, decoder_rewards = self.env.step(encoder_responses, decoder_responses)

            print(obs['query'])
            print('encoder respsonses', [self.tokenizer.decode(res) for res in encoder_responses])
            print('encoder', encoder_rewards)
            print('decoder respsonses', [self.tokenizer.decode(res) for res in decoder_responses])
            print('rewards: ', decoder_rewards)
            print()

            if self.multi_agent:

                queries = encoder_queries + decoder_queries
                responses = encoder_responses + decoder_responses
                rewards = encoder_rewards + decoder_rewards

                stats = self.ppo_trainer.step(queries, responses, rewards)

            else:
                self.ppo_trainer.step(encoder_queries, encoder_responses, encoder_rewards,
                                      decoder_queries, decoder_responses, decoder_rewards)

            # ppo_encoder_trainer.log_stats(stats1, {'query':encoder_queries, 'response': encoder_responses}, encoder_rewards)
            self.log_stats(stats, {'query':decoder_queries, 'response': decoder_responses}, decoder_rewards)

if __name__ == '__main__':
    steg_trainer = StegPPOTrainer(multi_agent = True)
    steg_trainer.train()