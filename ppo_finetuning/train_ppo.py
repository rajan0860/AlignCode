"""
ppo_finetuning/train_ppo.py

Fine-tunes Qwen2.5-Coder 1.5B using Proximal Policy Optimization (PPO).

This implements the classic RLHF loop:
  1. Generate code from the policy model given a prompt
  2. Score the generation using the trained reward model
  3. Update the policy using PPO to maximize reward while staying
     close to the reference model (KL penalty)

Setup:
  - Base model : Qwen/Qwen2.5-Coder-1.5B-Instruct
  - Reward     : reward_model/model/reward_model.pt (trained CodeBERT)
  - Method     : LoRA via peft (mirrors DPO config for fair comparison)
  - Trainer    : PPOTrainer from HuggingFace trl
  - Hardware   : Apple M4 MPS (~8-12GB RAM, ~2-3 hours)

Usage:
    python train_ppo.py --data ../data/evaluations.jsonl
    python train_ppo.py --data ../data/evaluations.jsonl --steps 200
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer

# Allow imports from sibling packages
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppo_config import get_lora_config, PPO_CONFIG
from reward_model.train import MODEL_NAME as REWARD_BASE_MODEL, RewardModel
from reward_model.evaluate import score_solution

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
REWARD_MODEL_PATH = Path(__file__).parents[1] / "reward_model" / "model" / "reward_model.pt"
OUTPUT_DIR = Path(__file__).parent / "model"
DEFAULT_STEPS = 100


def load_prompts(path: Path) -> list[str]:
    """Extract unique problem prompts from evaluations.jsonl."""
    prompts = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            prompt = r["problem"]
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
    return prompts


def load_reward_model(device: torch.device):
    """Load the trained reward model and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(REWARD_BASE_MODEL)
    model = RewardModel()
    model.load_state_dict(torch.load(REWARD_MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model, tokenizer


def train(data_path: Path, max_steps: int) -> None:
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Using device: {device}")

    # Load policy model with LoRA
    logger.info(f"Loading base model: {BASE_MODEL}")
    policy_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)
    lora_config = get_lora_config()
    policy_model = get_peft_model(policy_model, lora_config)
    policy_model.print_trainable_parameters()

    # Load reward model
    logger.info("Loading reward model...")
    reward_model, reward_tokenizer = load_reward_model(device)

    # Load prompts
    prompts = load_prompts(data_path)
    logger.info(f"Loaded {len(prompts)} unique prompts")

    # Build dataset — PPOTrainer expects a dataset with a "query" column
    dataset = Dataset.from_dict({"query": prompts})

    # Configure PPO
    ppo_config = PPOConfig(
        model_name=BASE_MODEL,
        learning_rate=PPO_CONFIG["learning_rate"],
        ppo_epochs=PPO_CONFIG["ppo_epochs"],
        mini_batch_size=PPO_CONFIG["mini_batch_size"],
        batch_size=PPO_CONFIG["batch_size"],
        init_kl_coef=PPO_CONFIG["init_kl_coef"],
        target_kl=PPO_CONFIG["target_kl"],
        log_with=None,
        accelerator_kwargs={"step_scheduler_with_optimizer": False},
    )

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        tokenizer=policy_tokenizer,
        dataset=dataset,
    )

    # Training loop
    logger.info(f"Starting PPO training for {max_steps} steps...")
    step = 0
    for epoch in range(max_steps // len(prompts) + 1):
        for batch in ppo_trainer.dataloader:
            if step >= max_steps:
                break

            # Tokenize prompts
            query_tensors = [
                policy_tokenizer.encode(q, return_tensors="pt").squeeze(0)
                for q in batch["query"]
            ]

            # Generate responses
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_new_tokens=PPO_CONFIG["max_new_tokens"],
                temperature=PPO_CONFIG["temperature"],
                top_p=PPO_CONFIG["top_p"],
                do_sample=True,
            )

            # Decode responses and compute rewards
            responses = [
                policy_tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                for r in response_tensors
            ]

            rewards = []
            for resp in responses:
                score = score_solution(reward_model, reward_tokenizer, resp, device)
                rewards.append(torch.tensor(score, dtype=torch.float32))

            # PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            mean_reward = sum(r.item() for r in rewards) / len(rewards)
            kl = stats.get("objective/kl", 0.0)
            logger.info(
                f"Step {step + 1}/{max_steps} — "
                f"mean_reward: {mean_reward:.4f} — "
                f"kl: {kl:.4f}"
            )
            step += 1

        if step >= max_steps:
            break

    # Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ppo_trainer.save_pretrained(str(OUTPUT_DIR))
    policy_tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info(f"PPO fine-tuned model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO fine-tuning for AlignCode")
    parser.add_argument("--data", type=Path, required=True, help="Path to evaluations.jsonl")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Number of PPO steps")
    args = parser.parse_args()

    train(args.data, args.steps)
