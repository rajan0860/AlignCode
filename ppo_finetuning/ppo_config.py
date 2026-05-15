"""
ppo_finetuning/ppo_config.py

PPO-specific hyperparameters for RLHF fine-tuning.

These values are tuned for Qwen2.5-Coder-1.5B on Apple M4 16GB (MPS).
The LoRA config mirrors the DPO setup for a fair comparison.

Key parameters:
  ppo_epochs      : Number of PPO optimization epochs per batch
  mini_batch_size : Size of minibatches within each PPO epoch
  init_kl_coef    : Initial KL penalty coefficient
  target_kl       : Target KL divergence — training stops if exceeded
  learning_rate   : Optimizer learning rate
"""

from peft import LoraConfig, TaskType


def get_lora_config() -> LoraConfig:
    """Return the LoRA configuration for PPO fine-tuning (mirrors DPO config)."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,
    )


# PPO-specific training parameters
PPO_CONFIG = {
    "ppo_epochs": 4,                # PPO optimization epochs per batch
    "mini_batch_size": 1,           # Minibatch size (small for 16GB RAM)
    "batch_size": 4,                # Number of prompts per PPO step
    "learning_rate": 1.41e-5,       # PPO learning rate
    "init_kl_coef": 0.2,            # Initial KL penalty coefficient
    "target_kl": 6.0,               # Target KL — early stops if exceeded
    "max_new_tokens": 512,          # Max tokens per generation
    "temperature": 0.7,             # Sampling temperature during generation
    "top_p": 0.9,                   # Nucleus sampling threshold
}
