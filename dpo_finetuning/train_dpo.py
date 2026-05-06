"""
dpo_finetuning/train_dpo.py

Fine-tunes Qwen2.5-Coder 1.5B using Direct Preference Optimization (DPO).

Setup:
  - Base model : Qwen/Qwen2.5-Coder-1.5B-Instruct
  - Method     : LoRA via peft (see lora_config.py)
  - Trainer    : DPOTrainer from HuggingFace trl
  - Dataset    : preference pairs from data/evaluations.jsonl
  - Hardware   : Apple M4 MPS (~8–12GB RAM, ~2 hours)

Usage:
    python train_dpo.py --data ../data/evaluations.jsonl
    python train_dpo.py --data ../data/evaluations.jsonl --epochs 2 --batch-size 2
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

from lora_config import get_lora_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 2
DEFAULT_LR = 5e-5
OUTPUT_DIR = Path(__file__).parent / "model"


def load_preference_dataset(path: Path) -> Dataset:
    """Load evaluations.jsonl and convert to DPO format (prompt / chosen / rejected)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            winner = r.get("winner", "B")
            rows.append(
                {
                    "prompt": r["problem"],
                    "chosen": r["solution_b"] if winner == "B" else r["solution_a"],
                    "rejected": r["solution_a"] if winner == "B" else r["solution_b"],
                }
            )
    return Dataset.from_list(rows)


def train(data_path: Path, epochs: int, batch_size: int, lr: float) -> None:
    logger.info(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)

    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_preference_dataset(data_path)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    logger.info(f"Dataset: {len(split['train'])} train / {len(split['test'])} eval pairs")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        beta=0.1,  # DPO temperature — controls deviation from reference policy
    )

    logger.info("Starting DPO fine-tuning...")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info(f"Fine-tuned model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO fine-tuning for AlignCode")
    parser.add_argument("--data", type=Path, required=True, help="Path to evaluations.jsonl")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size, args.lr)
