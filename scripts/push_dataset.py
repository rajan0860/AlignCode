"""
scripts/push_dataset.py

Exports evaluations.jsonl and pushes it to HuggingFace Hub.
"""

import sys
from pathlib import Path

# Allow imports from parent directory
sys.path.append(str(Path(__file__).resolve().parents[1]))

from annotation_engine.utils.export import export_to_hf_dataset
import argparse

def main():
    parser = argparse.ArgumentParser(description="Push AlignCode dataset to HuggingFace")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace repo (e.g., username/repo)")
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    args = parser.parse_args()

    print(f"Exporting dataset...")
    dataset = export_to_hf_dataset()
    
    print(f"Pushing to {args.repo}...")
    dataset.push_to_hub(args.repo, private=args.private)
    print("Done!")

if __name__ == "__main__":
    main()
