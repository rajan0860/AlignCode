"""
reward_model/train.py

Trains a code-aware reward model on collected preference pairs.

Architecture:
  - Base model: microsoft/codebert-base (125M encoder)
  - Fine-tuning: Bradley-Terry preference learning
  - Loss: -log(sigmoid(reward_chosen - reward_rejected))
  - Output: scalar score 0-1 representing code quality

Usage:
    python train.py --data ../data/evaluations.jsonl
    python train.py --data ../data/evaluations.jsonl --epochs 5 --batch-size 8

Target: >80% accuracy on held-out preference pairs (~30 min on M4 MPS).
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "microsoft/codebert-base"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 2e-5
TRAIN_SPLIT = 0.8


# ── Model ─────────────────────────────────────────────────────────────────────

class RewardModel(nn.Module):
    """CodeBERT encoder with a scalar reward head."""

    def __init__(self, base_model_name: str = MODEL_NAME) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.reward_head(cls_embedding).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PreferenceDataset(Dataset):
    """Loads preference pairs from evaluations.jsonl."""

    def __init__(self, path: Path, tokenizer: AutoTokenizer, max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs: list[dict] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                winner = record.get("winner", "B")
                self.pairs.append(
                    {
                        "chosen": record["solution_b"] if winner == "B" else record["solution_a"],
                        "rejected": record["solution_a"] if winner == "B" else record["solution_b"],
                    }
                )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        chosen_enc = self.tokenizer(
            pair["chosen"], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            pair["rejected"], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


# ── Training ──────────────────────────────────────────────────────────────────

def bradley_terry_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry preference loss: -log(sigmoid(r_chosen - r_rejected))."""
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()


def train(data_path: Path, epochs: int, batch_size: int, lr: float) -> None:
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = PreferenceDataset(data_path, tokenizer)

    train_size = int(len(dataset) * TRAIN_SPLIT)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = RewardModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            r_chosen = model(chosen_ids, chosen_mask)
            r_rejected = model(rejected_ids, rejected_mask)
            loss = bradley_terry_loss(r_chosen, r_rejected)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch}/{epochs} — loss: {avg_loss:.4f} — val_acc: {val_acc:.2%}")

    output_dir = Path(__file__).parent / "model"
    output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output_dir / "reward_model.pt")
    logger.info(f"Model saved to {output_dir / 'reward_model.pt'}")


def evaluate(model: RewardModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            r_chosen = model(
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
            )
            r_rejected = model(
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
            )
            correct += (r_chosen > r_rejected).sum().item()
            total += len(r_chosen)
    return correct / total if total > 0 else 0.0


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AlignCode reward model")
    parser.add_argument("--data", type=Path, required=True, help="Path to evaluations.jsonl")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size, args.lr)
