from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any

# We will structure observations as:
# tuple: (context, message, decoy_text, meta_info)

def obs_to_steg_prompt(obs):
    """
    Turns an obs into a natural language prompt for the model to follow
    :param obs: The observation the environment/model sees
    :return: A prompt for the model to follow
    """
    context, message, decoy_text, meta_info = obs
    # if called initially, decoy_text should be empty
    prompt = "You are a secret spy. Given a prompt and secret key, generate a message that encodes" \
             " the key while appearing like a normal message that follows the prompt. Here is the prompt: '"
    prompt += context + "'. Here is the key: '" + message + "'. Now generate normal-looking text encoding the secret key: '"
    return prompt

def create_first_observation(context, message):
    """
    Given an initial context and a secret key (message), generate the first observation that the
    RL environment would feed to the agent:
    :param context:
    :param message:
    :return: Tuple (context, message, decoy_text, meta_info) except in tokens
    """


class SteganographyReward(RewardFunction):
   def __init__(self, *args) -> None:
       super().__init__()
       # load frozen Discriminator Model
       model_string = "bert-base-uncased"
       self.tokenizer = AutoTokenizer.from_pretrained(model_string)
       self.discriminator = AutoModel.from_pretrained("bert-base-cased")

   def __call__(self, prev_observation: Observation,
                action: int,
                current_observation: Observation,
                done: bool,
                meta_info: Dict[str, Any] = None) -> float:
       if done:
           reward = 0
           return reward
       return 0

# might want to turn this into a batched reward function using class BatchedRewardFunction(ABC):