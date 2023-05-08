import torch
from transformers import AutoTokenizer, AutoModel
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
import wandb
from peft import PeftConfig, PeftModel, LoraConfig
from typing import Dict, Tuple, Optional, List
from accelerate import Accelerator

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

class StegEnv():
    def __init__(self, 
                tokenizer: AutoTokenizer,
                batch_size: int = 16,
                device: str = 'cpu',
                trainer = None
            ):
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.trainer = trainer
        
        self.key_length = 4
        self.enc_response_len = 4
        self.dec_response_len = max(self.key_length, 4) # must be >= 4 for now because of ppo_train

        if self.key_length < self.dec_response_len:
            print("Warning: key length is less than decoder response length. Reward will only use first [key_length] tokens.")

        # self.key_tokens = [' 0', ' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9']
        self.key_tokens = ['a', 'b', 'c', 'd',]
        self.message_tokens = ['+', '-', '<', '>',]
        # self.prompts = [" 0 7 3 8 4", "Yesterday I went to ", "The weather today is ", "What is your favorite "]

        # self.prompts_pt = self.tokenize_batch(self.prompts)
        self.key_tokens_pt = self.tokenize_batch(self.key_tokens, squeeze=True)
        self.message_tokens_pt = self.tokenize_batch(self.message_tokens, squeeze=True)

        self.prompt_batch = None
        self.key_batch = None

        encode_prompt = "I am a spy who needs to encode a secret key into a message. keys have the letters abcd while messages have the characters +-<> " #+\
                        # "Example: key:b d a c message:+ > - <"
        
        decode_prompt = "I am a spy who needs to recover a secret key from a message. keys have the letters abcd while messages have the characters +-<> " #+\
                        # "Example: message:+ > - < key:b d a c"
        key_buff = "key:"
        # prompt_buff = " prompt:"
        # encode_buff = ". Now repeat the key exactly as is. The key is:"
        # encode_buff = ". message:"
        message_buff = "message:"
        # decode_buff = ". key:"
        
        self.buffs = dict()

        self.buffs["encode_prompt"] = self.tokenize_batch(encode_prompt, repeat=True, special_tokens=True)
        self.buffs["decode_prompt"] = self.tokenize_batch(decode_prompt, repeat=True, special_tokens=True)
        self.buffs["key"] = self.tokenize_batch(key_buff, repeat=True)
        # self.buffs["prompt"] = self.tokenize_batch(prompt_buff, repeat=True)
        # self.buffs["encode"] = self.tokenize_batch(encode_buff, repeat=True)
        self.buffs["message"] = self.tokenize_batch(message_buff, repeat=True)
        # self.buffs["decode"] = self.tokenize_batch(decode_buff, repeat=True)
    
    def tokenize_batch(self, string, repeat=False, squeeze=False, special_tokens=False):
        tokenized = self.tokenizer(string, return_tensors='pt', padding=True, add_special_tokens=special_tokens)['input_ids']
        if repeat:
            tokenized = tokenized.repeat(self.batch_size,1)
        if squeeze:
            tokenized = tokenized.squeeze()
        return tokenized.to(self.device)

    def _get_encoder_query(self, key, prompt):
        """
        Generates queries of the form "$key:[key]$prompt:[prompt]"
        """
        # print((self.buffs["key"], key, self.buffs["encode"]))
        # print(self.buffs["encode_prompt"], self.buffs["key"], key, self.buffs["message"])
        return torch.cat((self.buffs["encode_prompt"], self.buffs["key"], key, self.buffs["message"]), dim=-1) #self.buffs["encode"]
        # return torch.cat((self.key_buff_batch, key, prompt_buff_batch, self.prompt_batch), dim=-1)

    def _get_decoder_query(self, prompt, response):
        #return torch.cat((response, self.key_buff_batch), dim=-1)
        # return torch.cat((self.buffs["message"], response, self.buffs["decode"]), dim=-1)
        return torch.cat((self.buffs["decode_prompt"], self.buffs["message"], response[:, :self.key_length], self.buffs["key"]), dim=-1)
    
    def _get_obs(self):

        return {
            "query": self.query_batch,
            "prompt": self.prompt_batch,
            "key": self.key_batch
        }

    def reset(self, ):
        
        # prompt_idxs = torch.randint(len(self.prompts_pt), size=(self.batch_size,))
        key_idxs = torch.randint(len(self.key_tokens_pt), size=(self.batch_size, self.key_length))

        # self.prompt_batch = self.prompts_pt[prompt_idxs]
        self.key_batch = self.key_tokens_pt[key_idxs]
        self.query_batch = self._get_encoder_query(self.key_batch, self.prompt_batch)
        
        obs = self._get_obs()

        return obs

    def _reward_function(self, enc_query, enc_response, dec_query, dec_response):

        dec_response = dec_response[:, :self.key_length] # only use first [key_length] tokens
        #reward_encoder = self.trainer.model_ref(enc_query, enc_response)
        # ref_logprobs, _, _, _ = self.batched_forward_pass(self.trainer.ref_model, queries, responses, model_inputs)
        reward_encoder = reward_decoder = (dec_response == self.key_batch).sum(dim=-1).float()
        return (reward_encoder, reward_decoder)

    def step(self, enc_query, enc_response, dec_query, dec_response):

        reward = self._reward_function(enc_query, enc_response, dec_query, dec_response)
        return reward


class StegPPOTrainer():
    def __init__(self,
            config: dict,
            model: AutoModel,
            model_ref: AutoModel,
            tokenizer: AutoTokenizer,
        ):
        
        self.model = model
        self.model_ref = model_ref
        self.tokenizer = tokenizer

        batch_size = config['batch_size']
        self.episodes = config['episodes']
        self.device = config['device']
        self.multi_agent = config['multi_agent']
        self.use_wandb = config['use_wandb']

        # initialize environment
        self.env = StegEnv(
            tokenizer = self.tokenizer,
            batch_size=batch_size,
            device=self.device,
            trainer=self,
        )

        # get tokens to suppress
        enc_suppress_tokens = [i for i in range(self.tokenizer.vocab_size) if i not in self.env.message_tokens_pt]
        dec_suppress_tokens = [i for i in range(self.tokenizer.vocab_size) if i not in self.env.key_tokens_pt]

        self.enc_gen_kwargs = {
            "min_length": -1,
            "top_k": 0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": self.env.enc_response_len,
            "suppress_tokens": enc_suppress_tokens, 
        }

        self.dec_gen_kwargs = {
            **self.enc_gen_kwargs,
            # "top_k": 4,
            "max_new_tokens": self.env.dec_response_len, 
            "suppress_tokens": dec_suppress_tokens, 
        }

        if config['multi_agent']: from trl import PPOTrainer
        else: from trl_custom import PPOTrainer

        ppo_config = PPOConfig(
            batch_size= batch_size if self.multi_agent else batch_size, # double for encoder + decoder responses
            learning_rate=config['learning_rate'],
            steps=config['steps'],
            optimize_cuda_cache=True,
            )
            
        # print(self.model, type(self.model))
        self.ppo_trainer = PPOTrainer(ppo_config, self.model, None, self.tokenizer)
        
    def log_stats(
        self,
        stats: dict,
        rewards: List[torch.FloatTensor],
    ):
        """
        A function that logs all the training stats. Call it at the end of each epoch.
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

        if self.use_wandb: wandb.log(logs)

    def get_model_responses(self, obs):

        encoder_query = obs['query']
        encoder_response = self.ppo_trainer.generate(list(encoder_query), return_prompt=False, **self.enc_gen_kwargs) # should this be ppo_trainer.generate????
        encoder_response = torch.stack(encoder_response)
        # encoder_response = encoder_response[:, -self.enc_gen_kwargs["max_new_tokens"]:]
        decoder_query = self.env._get_decoder_query(obs['prompt'], encoder_response)
        decoder_response = self.ppo_trainer.generate(list(decoder_query), return_prompt=False, **self.dec_gen_kwargs)
        decoder_response = torch.stack(decoder_response)
        # decoder_response = decoder_response[:, -self.dec_gen_kwargs["max_new_tokens"]:]

        return encoder_query, encoder_response, decoder_query, decoder_response

    def train(self):

        for _ in range(self.episodes):
            obs = self.env.reset()
            enc_query, enc_response, dec_query, dec_response = self.get_model_responses(obs)
            enc_reward, dec_reward = self.env.step(enc_query, enc_response, dec_query, dec_response)

            print('-----------------------------------------------------------------------')
            print('prompt, keys:')
            # print(self.tokenizer.batch_decode(obs['prompt']))
            print(self.tokenizer.batch_decode(obs['key']))
            print('\nencoder:')
            print(self.tokenizer.batch_decode(obs['query']))
            print(self.tokenizer.batch_decode(enc_response))
            print(enc_reward)
            print('\ndecoder:')
            print(self.tokenizer.batch_decode(dec_query))
            print(self.tokenizer.batch_decode(dec_response))
            print(dec_reward)
            print()

            if self.multi_agent:

                # query = list(enc_query) + list(dec_query)
                # response = list(enc_response) + list(dec_response)
                # reward = list(enc_reward) + list(dec_reward)

                # stats = self.ppo_trainer.step(query, response, reward)

                # llama error if I don't split up. Not sure if this has any bad effects.
                stats = self.ppo_trainer.step(list(enc_query), list(enc_response), list(enc_reward))
                self.ppo_trainer.step(list(dec_query), list(dec_response), list(dec_reward))

            else:
                stats = self.ppo_trainer.step(list(enc_query), list(enc_response), list(enc_reward), list(dec_query), list(dec_response), list(dec_reward))

            self.log_stats(stats, dec_reward)


def main():
    current_device = Accelerator().local_process_index

    print('device', current_device)

    config = {
        'model_name':  '/om2/user/juliany/llama7B/', #'gpt2-xl', #
        'batch_size': 8,
        'learning_rate': 1e-5,
        'steps': 100,
        'episodes': 1000,
        'device': current_device,
        'multi_agent': True,
        'use_wandb': True
    }

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config['model_name'],
        peft_config=lora_config,
        load_in_8bit=True,
        device_map={"": current_device},
        # layer_norm_names=[],
    )

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    if "llama" in config['model_name']:
        # required for llama
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )
    else:
        # required for gpt2
        tokenizer.pad_token = tokenizer.eos_token

    if config['use_wandb']: wandb.init(project="julian-steg-runs")

    steg_trainer = StegPPOTrainer(config, model, None, tokenizer)  
    steg_trainer.train()  


if __name__ == "__main__":
    main()
