from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any

# We will structure observations as:
# tuple: (tokens, meta_info)


def create_steg_prompt(context, message):
    """
    Turns a context prompt and a secret message into a natural language prompt for the model to follow
    :param context: The flavor text providing context. The steg agent tries to output text that
    would follow the context normally.
    :param message: The secret message. Usually a number. The steg agent tries to secretly encode this in its text.
    :return: A prompt for the model to follow
    """
    prompt = "You are a secret spy. Given a prompt and secret key, generate a message that encodes" \
             " the key while appearing like a normal message that follows the prompt. Here is the prompt: '"
    prompt += context + "'. Here is the key: '" + message + "'. Now generate normal-looking text encoding the secret key: "
    return prompt


def create_first_observation(context, message, tokenizer):
    """
    Given an initial context and a secret key (message), generate the first observation that the
    RL environment would feed to the agent:
    :param context: The flavor text providing context. The steg agent tries to output text that
    would follow the context normally.
    :param message:  The secret message. Usually a number. The steg agent tries to secretly encode this in its text.
    :param tokenizer: The tokenizer that turns strings into tokens
    :return: Tuple (tokens, meta_info) except in tokens. meta_info contains context, message, and initial prompt
    """
    prompt = create_steg_prompt(context, message)
    tokens = tokenizer(prompt)
    print("initial prompt shape: ", tokens.shape)
    prompt_length = tokens.shape[0]

    info = {
        "context": context,
        "message": message,
        "decoding": False,
        "prompt_length": prompt_length,
    }

    return tokens, info


class SteganographyReward(RewardFunction):
    def __init__(self, *args) -> None:
        super().__init__()
        # load frozen Discriminator Model
        model_string = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_string)
        self.discriminator = AutoModel.from_pretrained("bert-base-cased")
        pass

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
