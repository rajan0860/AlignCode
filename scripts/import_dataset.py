"""
scripts/import_dataset.py

Imports 80 problems from the MBPP dataset on HuggingFace and saves them as 
JSON to be loaded by the problem bank.
"""

import json
import re
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    exit(1)

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "annotation_engine" / "problems" / "mbpp_problems.json"

def main():
    print("Loading MBPP dataset from HuggingFace...")
    # Using 'sanitized' subset for higher quality
    ds = load_dataset("mbpp", "sanitized", split="test")
    
    problems = []
    for i, row in enumerate(ds):
        if i >= 80:
            break
            
        task_id = row["task_id"]
        text = row["prompt"]
        tests = row["test_list"]
        
        # Simple title generation from the text
        words = text.split()
        title = " ".join(words[4:8]).title() if len(words) > 8 else "MBPP Problem"
        title = re.sub(r'[^a-zA-Z\s]', '', title).strip()
        if not title:
            title = f"MBPP Task {task_id}"
            
        # Format the tests using the new "assert" format for the sandbox
        test_cases = [{"assert": t} for t in tests]
        
        problems.append({
            "id": f"mbpp_{task_id}",
            "language": "python",
            "difficulty": "medium",
            "title": title,
            "statement": text,
            "test_cases": test_cases
        })
        
    print(f"Generated {len(problems)} problems. Saving to {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    main()
