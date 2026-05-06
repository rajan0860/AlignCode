"""
dpo_finetuning/lora_config.py

LoRA (Low-Rank Adaptation) hyperparameters for DPO fine-tuning.

These values are tuned for Qwen2.5-Coder-1.5B on Apple M4 16GB (MPS).
Adjust batch size and gradient checkpointing if running on smaller hardware.

Key parameters:
  r          : LoRA rank — higher = more capacity, more memory (8–16 typical)
  lora_alpha : Scaling factor — usually 2x rank
  target_modules : Attention projection layers to adapt
  lora_dropout   : Regularization dropout on LoRA layers
"""

from peft import LoraConfig, TaskType


def get_lora_config() -> LoraConfig:
    """Return the LoRA configuration for Qwen2.5-Coder-1.5B DPO fine-tuning."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                          # LoRA rank
        lora_alpha=32,                 # Scaling factor (2x rank)
        lora_dropout=0.05,             # Dropout on LoRA layers
        bias="none",                   # Don't adapt bias terms
        target_modules=[               # Attention projection layers
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
