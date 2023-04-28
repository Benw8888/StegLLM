import torch
from transformers import AutoTokenizer, AutoModel
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
import wandb
from typing import Dict, Tuple, Optional, List

class StegEnv():
    def __init__(self, 
                tokenizer: AutoTokenizer,
                batch_size: int = 16,
                device: str = 'cpu',
            ):
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.key_length = 4 # must be >= 4 for now because of ppo_train

        self.key_tokens = [' 0', ' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9']
        self.prompts = [" 0 7 3 8 4", "Yesterday I went to ", "The weather today is ", "What is your favorite "]

        self.prompts_pt = self.tokenizer(self.prompts, return_tensors='pt', padding=True)['input_ids'].to(self.device)
        self.key_tokens_pt = self.tokenizer(self.key_tokens, return_tensors='pt', padding=True)['input_ids'].squeeze().to(self.device)

        self.enc_response_len = 4
        self.dec_response_len = self.key_length

        self.prompt_batch = None
        self.key_batch = None


    def _get_queries(self):
        """
        Generates queries of the form "$key:[key]$prompt:[prompt]"
        """
        key_buff = "$key:"
        prompt_buff = "$prompt:"

        key_buff_batch = self.tokenizer(key_buff, return_tensors='pt')['input_ids'].repeat(self.batch_size,1).to(self.device)
        prompt_buff_batch = self.tokenizer(prompt_buff, return_tensors='pt')['input_ids'].repeat(self.batch_size,1).to(self.device)

        self.query_batch = torch.cat((key_buff_batch, self.key_batch, prompt_buff_batch, self.prompt_batch), dim=-1)
    
    def _get_obs(self):

        return {
            "query": self.query_batch,
            "prompt": self.prompt_batch,
            "key": self.key_batch
        }

    def reset(self, ):
        
        prompt_idxs = torch.randint(len(self.prompts_pt), size=(self.batch_size,))
        key_idxs = torch.randint(len(self.key_tokens_pt), size=(self.batch_size, self.key_length))

        self.prompt_batch = self.prompts_pt[prompt_idxs]
        self.key_batch = self.key_tokens_pt[key_idxs]
        self._get_queries()
        
        obs = self._get_obs()

        return obs

    def _reward_function(self, encoder_response, decoder_response):

        reward_encoder = reward_decoder = (decoder_response == self.key_batch).sum(dim=-1).float()
        return (reward_encoder, reward_decoder)

    def step(self, encoder_response, decoder_response):

        reward = self._reward_function(encoder_response, decoder_response)
        return reward


class StegPPOTrainer():
    def __init__(self,
            model: AutoModel,
            model_ref: AutoModel,
            tokenizer: AutoTokenizer,
            device: str = 'cpu',
            multi_agent = True,
        ):
        
        self.model = model
        self.model_ref = model_ref
        self.tokenizer = tokenizer
        self.device = device
        self.multi_agent = multi_agent

        batch_size = 16
        self.num_episodes = 100

        # initialize environment
        self.env = StegEnv(
            tokenizer = self.tokenizer,
            device=device,
            batch_size=batch_size,
        )

        self.enc_gen_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": self.env.enc_response_len
        }

        # get tokens to suppress
        self.suppress_tokens = [i for i in range(self.tokenizer.vocab_size) if i not in self.env.key_tokens_pt]
        
        self.dec_gen_kwargs = {
            "suppress_tokens": self.suppress_tokens, 
            "max_new_tokens": self.env.dec_response_len, 
            **self.enc_gen_kwargs
        }

        config = PPOConfig(
            batch_size= batch_size * 2 if multi_agent else batch_size, # double for encoder + decoder responses
            learning_rate=1e-5,
            steps=5000,
            )
            
        self.ppo_trainer = PPOTrainer(config, self.model, self.model_ref, self.tokenizer)
        
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
            rewards = torch.tensor(rewards).to(self.device)

        logs.update(stats)

        # manually cast in fp32 for bf16 torch tensors
        for k, v in logs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                logs[k] = v.float()

        logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
        logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
        logs["env/reward_dist"] = rewards.cpu().numpy()

        wandb.log(logs)

    def get_model_responses(self, obs):

        encoder_query = obs['query']
        encoder_response = self.model.generate(encoder_query, **self.enc_gen_kwargs) # should this be ppo_trainer.generate????
        encoder_response = encoder_response[:, -self.enc_gen_kwargs["max_new_tokens"]:]

        decoder_query = torch.cat((obs['prompt'], encoder_response), dim=-1)
        decoder_response = self.model.generate(decoder_query, **self.dec_gen_kwargs)
        decoder_response = decoder_response[:, -self.dec_gen_kwargs["max_new_tokens"]:]

        return encoder_query, encoder_response, decoder_query, decoder_response

    def train(self):

        for _ in range(self.num_episodes):
            obs = self.env.reset()
            enc_query, enc_response, dec_query, dec_response = self.get_model_responses(obs)
            enc_reward, dec_reward = self.env.step(enc_response, dec_response)

            print('-----------------------------------------------------------------------')
            print('Prompt and keys:')
            print(self.tokenizer.batch_decode(obs['prompt']))
            print(self.tokenizer.batch_decode(obs['key']))
            print('Encoder:')
            print(self.tokenizer.batch_decode(enc_response))
            print(enc_reward)
            print('Decoder:')
            print(self.tokenizer.batch_decode(dec_response))
            print(dec_reward)
            print()

            if self.multi_agent:

                query = list(enc_query) + list(dec_query)
                response = list(enc_response) + list(dec_response)
                reward = list(enc_reward) + list(dec_reward)

                stats = self.ppo_trainer.step(query, response, reward)

            else:
                stats = self.ppo_trainer.step(enc_query, enc_response, enc_reward, dec_query, dec_response, dec_reward)

            self.log_stats(stats, {}, dec_reward)


if __name__ == "__main__":

    multi_agent = True

    if multi_agent:
        from trl import PPOTrainer
    else:
        from trl_custom import PPOTrainer

    device = 0 if torch.cuda.is_available() else 'cpu'

    model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2').to(device)
    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2').to(device)

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    wandb.init(project="my-awesome-project")

    steg_trainer = StegPPOTrainer(model, model_ref, tokenizer, multi_agent=multi_agent, device=device)
    steg_trainer.train()