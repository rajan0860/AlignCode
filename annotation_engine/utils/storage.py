"""
annotation_engine/utils/storage.py

Handles reading and writing preference pairs to/from evaluations.jsonl.

Each line in evaluations.jsonl is a complete JSON object representing
one annotated preference pair. This format is directly compatible with
HuggingFace datasets.

Schema supports both single-turn and multi-turn evaluations:
  - Single-turn: standard fields (solution_a, solution_b, scores, winner, etc.)
  - Multi-turn: adds an optional "turns" list containing refinement history:
      "turns": [
          {
              "turn": 1,
              "feedback": "Handle the None edge case",
              "refined_code": "def foo(x): ...",
              "refined_test_results": {"passed": 4, "failed": 0, "total": 4}
          }
      ]
    Old single-turn records without "turns" remain fully compatible.

Usage:
    from annotation_engine.utils.storage import save_evaluation, load_evaluations

    save_evaluation(record)
    records = load_evaluations()
"""

import json
from pathlib import Path
from typing import Any

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "evaluations.jsonl"


def save_evaluation(record: dict[str, Any], path: Path = DEFAULT_DATA_PATH) -> None:
    """
    Append a single preference pair record to the JSONL file.

    Args:
        record: The completed annotation dict.
        path: Path to the JSONL file (defaults to data/evaluations.jsonl).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_evaluations(path: Path = DEFAULT_DATA_PATH) -> list[dict[str, Any]]:
    """
    Load all preference pairs from the JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of annotation dicts. Returns empty list if file doesn't exist.
    """
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count_evaluations(path: Path = DEFAULT_DATA_PATH) -> int:
    """Return the number of saved preference pairs."""
    return len(load_evaluations(path))
