# StegLLM

Large Language Models and Beyond Class Project

Our goal is to understand whether language models will learn to hide information in plain sight given reward pressure in a reinforcement learning environment.

Abstract:

Steganography, the art of conveying a secret
message while concealing its existence in plain
sight, is a capability that would be concerning
to see in language models. Language models capable of steganography would be able to
communicate and plan without our supervision,
posing a safety concern. Therefore, we investigate to what extent current language models
can perform steganography in an end-to-end reinforcement learning setting. We compare the
performance of LLaMA (Touvron et al., 2023),
a language model with 6 billion parameters,
with GPT-2 (small) (Radford et al., 2018), a 124
million parameter model. We train the models
in a multi-agent setting using Proximal Policy
Optimization (Schulman et al., 2017), a popular
reinforcement learning technique, and employ
optimizations for training large language models including ZeRO (Rajbhandari et al., 2020)
and LoRA (Hu et al., 2021). We show that
while GPT-2 (small) fails at the steganography task, LLaMA is able to succesfully en-
code and decode secret messages, although the
steganography scheme employed is limited in
complexity. Our work lays the foundation for
steganographic capability evaluations for future language models and paints a concerning
trend regarding our ability to supervise scaled
up models.
