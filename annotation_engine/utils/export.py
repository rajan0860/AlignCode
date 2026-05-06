"""
annotation_engine/utils/export.py

Exports the collected preference pairs to HuggingFace datasets format.

The exported dataset maps AlignCode fields to the standard DPO format:
  - "prompt"  → the problem statement
  - "chosen"  → the winning solution
  - "rejected" → the losing solution

Usage:
    from annotation_engine.utils.export import export_to_hf_dataset

    dataset = export_to_hf_dataset("data/evaluations.jsonl")
    dataset.push_to_hub("your-username/aligncode-preference-dataset")
"""

from pathlib import Path

from datasets import Dataset

from annotation_engine.utils.storage import load_evaluations


def export_to_hf_dataset(path: str | Path | None = None) -> Dataset:
    """
    Convert evaluations.jsonl to a HuggingFace Dataset in DPO format.

    Args:
        path: Path to evaluations.jsonl. Uses default data path if None.

    Returns:
        HuggingFace Dataset with columns: prompt, chosen, rejected, language,
        winner, justification, scores, bugs_found, complexity.
    """
    kwargs = {"path": Path(path)} if path else {}
    records = load_evaluations(**kwargs)

    rows = []
    for r in records:
        winner = r.get("winner", "B")
        chosen = r["solution_b"] if winner == "B" else r["solution_a"]
        rejected = r["solution_a"] if winner == "B" else r["solution_b"]

        rows.append(
            {
                "prompt": r["problem"],
                "chosen": chosen,
                "rejected": rejected,
                "language": r.get("language", ""),
                "winner": winner,
                "justification": r.get("justification", ""),
                "scores": r.get("scores", {}),
                "bugs_found": r.get("bugs_found", {}),
                "complexity": r.get("complexity", {}),
            }
        )

    return Dataset.from_list(rows)
