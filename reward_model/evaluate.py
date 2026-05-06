"""
reward_model/evaluate.py

Evaluates the trained reward model on held-out preference pairs.

Reports:
  - Accuracy: % of pairs where reward(chosen) > reward(rejected)
  - Mean reward gap: average difference between chosen and rejected scores
  - Per-language breakdown (Python vs Go)

Usage:
    python evaluate.py --data ../data/evaluations.jsonl --model model/reward_model.pt
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from train import MODEL_NAME, RewardModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def score_solution(
    model: RewardModel,
    tokenizer: AutoTokenizer,
    code: str,
    device: torch.device,
    max_length: int = 512,
) -> float:
    """Return the reward score (0–1) for a single code solution."""
    enc = tokenizer(
        code, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    )
    with torch.no_grad():
        score = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
    return score.item()


def run_evaluation(data_path: Path, model_path: Path) -> None:
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = RewardModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    correct = total = 0
    reward_gaps: list[float] = []
    per_language: dict[str, dict[str, int]] = {}

    for r in records:
        winner = r.get("winner", "B")
        chosen = r["solution_b"] if winner == "B" else r["solution_a"]
        rejected = r["solution_a"] if winner == "B" else r["solution_b"]
        lang = r.get("language", "unknown")

        r_chosen = score_solution(model, tokenizer, chosen, device)
        r_rejected = score_solution(model, tokenizer, rejected, device)

        is_correct = r_chosen > r_rejected
        correct += int(is_correct)
        total += 1
        reward_gaps.append(r_chosen - r_rejected)

        if lang not in per_language:
            per_language[lang] = {"correct": 0, "total": 0}
        per_language[lang]["total"] += 1
        per_language[lang]["correct"] += int(is_correct)

    accuracy = correct / total if total > 0 else 0.0
    mean_gap = sum(reward_gaps) / len(reward_gaps) if reward_gaps else 0.0

    logger.info(f"\n{'─' * 40}")
    logger.info(f"Overall Accuracy : {accuracy:.2%} ({correct}/{total})")
    logger.info(f"Mean Reward Gap  : {mean_gap:+.4f}")
    logger.info("Per-language breakdown:")
    for lang, stats in per_language.items():
        lang_acc = stats["correct"] / stats["total"]
        logger.info(f"  {lang:12s}: {lang_acc:.2%} ({stats['correct']}/{stats['total']})")
    logger.info(f"{'─' * 40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the AlignCode reward model")
    parser.add_argument("--data", type=Path, required=True, help="Path to evaluations.jsonl")
    parser.add_argument(
        "--model", type=Path,
        default=Path(__file__).parent / "model" / "reward_model.pt",
        help="Path to saved reward model weights",
    )
    args = parser.parse_args()
    run_evaluation(args.data, args.model)
