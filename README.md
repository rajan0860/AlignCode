# AlignCode
### An End-to-End RLHF Pipeline for Code Quality — From Human Annotations to a Fine-Tuned LLM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-orange)](https://huggingface.co/datasets/rajan/aligncode-preference-dataset)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLMs-black)](https://ollama.ai/)

---

## Table of Contents
- [What is AlignCode?](#what-is-aligncode)
- [Why This Project Exists](#why-this-project-exists)
- [Quick Start](#quick-start)
- [The Pipeline](#the-pipeline)
- [Annotation Engine](#annotation-engine)
- [Reward Model Training](#reward-model-training)
- [DPO Fine-Tuning](#dpo-fine-tuning)
- [Evaluation Dashboard](#evaluation-dashboard)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Running](#setup--running)
- [Dataset](#dataset)
- [Troubleshooting](#troubleshooting)
- [Roadmap & Future Work](#roadmap--future-work)
- [License](#license)

---

## What is AlignCode?

AlignCode is a complete, locally-run RLHF (Reinforcement Learning from Human Feedback) pipeline focused on code quality evaluation.

Most people studying RLHF read about it. AlignCode **implements it** — every stage, end to end:

1. **Collect** structured human preference data on AI-generated code (~4 min per annotation)
2. **Train** a reward model that learns what "good code" means
3. **Fine-tune** a code LLM using those preferences (DPO)
4. **Measure** the improvement with a live evaluation dashboard

Everything runs **fully locally** using Ollama. No API keys. No cloud costs. No data leaving your machine.

---

## Why This Project Exists

When companies like Anthropic, OpenAI, and Google fine-tune coding LLMs, they rely on human annotators to compare AI-generated solutions and judge which one is better. That human preference data becomes the training signal that makes models more helpful, correct, and safe.

AlignCode demonstrates that full pipeline end to end:
- A real preference dataset collected through structured human annotation — not a toy example
- A reward model trained on those annotations that can score new code without human input
- A fine-tuned LLM that measurably improves on the annotated quality criteria
- The complete RLHF loop in a single, reproducible local project

---

## Quick Start

Just want to try the annotation engine? Three commands:

```bash
# 1. Install and start Ollama, then pull the model
ollama pull qwen2.5-coder:7b

# 2. Install dependencies (expect ~5–10 min for torch + transformers)
pip install -r requirements.txt

# 3. Launch the annotation UI
cd annotation_engine && streamlit run app.py
```

For the full pipeline (reward model training + DPO fine-tuning), see [Setup & Running](#setup--running).

---

## The Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AlignCode Pipeline                          │
│                                                                     │
│   Annotation        Reward Model        DPO              Evaluation │
│   Engine        →   Training        →   Fine-tuning  →   Dashboard  │
│   ──────────        ──────────          ──────────        ────────  │
│   (Streamlit UI)    (CodeBERT)          (Qwen2.5-Coder)  (before   │
│                                                           vs after) │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Annotation Engine

### What it does
A Streamlit app that generates two AI code solutions to the same problem using different prompting strategies, then lets you systematically evaluate and compare them. Each annotation takes ~4 minutes on average.

### How solutions are generated
Both solutions come from the same local Ollama model but with different system prompts:

- **Solution A** — "Quick" prompt: write a concise working solution
- **Solution B** — "Production" prompt: write idiomatic, type-hinted, edge-case-aware code

This reliably produces solutions of varying quality — ideal for annotation.

### Evaluation Framework

**Scoring Rubric** (per solution, independent 0/1 scores):

| Criterion | What is checked |
|---|---|
| Correctness | Passes all test cases including edge cases |
| Time Complexity | Optimal or near-optimal algorithmic approach |
| Space Complexity | Memory usage is appropriate |
| Readability | Clear naming, structure, and intent |
| Idiomatic Style | Written the way an expert would write it |
| Edge Cases | Handles empty input, None, overflow, duplicates |
| Security | Free from injection, hardcoded secrets, unsafe eval |

**Code Execution Sandbox**

Solutions are safely executed against predefined test cases per problem. The sandbox uses local isolated processes with strict memory and timeout constraints to safely evaluate LLM-generated code without risk to the host machine:
```
Test: find_duplicates([1,2,3,2,1])  → expected [1,2]    ✅ PASS
Test: find_duplicates([])           → expected []        ✅ PASS
Test: find_duplicates(None)         → expected []        ❌ FAIL
```

Actual pass/fail rates inform your scoring — you don't just read the code, you verify it.

**Bug Checklist**

General bugs:
- Off-by-one errors
- Unhandled None / null input
- Mutable default arguments (Python)
- Incorrect return type
- Shadowed variable names
- Memory leak patterns

Security issues:
- Hardcoded credentials
- SQL injection vulnerability
- Use of `eval()` / `exec()` on user input
- Insecure random number generation
- Path traversal vulnerability

Go-specific issues:
- Goroutine leak (missing WaitGroup / done channel)
- Missing `defer` for Close() / Unlock()
- Ignored error returns
- Shadowed `err` variable in if block
- Race condition on shared variable
- Incorrect value vs pointer receiver
- Context not propagated down call chain
- Unbuffered channel causing deadlock

**Complexity Analyser**

Per solution, you annotate:
- Time complexity (O(1) → O(2ⁿ) dropdown)
- Space complexity (O(1) → O(2ⁿ) dropdown)
- Where the bottleneck is (free text)

**Justification Box**

Minimum 80 characters enforced. Auto-suggestions remind you to mention complexity, idioms, and security if your rubric scores differ in those areas.

**AI Critique (local)**

After your evaluation, click "Show AI Critique" — the same local model compares both solutions. Compare your reasoning against the model's to identify gaps in your analysis.

### Problem Bank

22 hand-selected problems across two languages (Python and Go) and three difficulty levels. JavaScript problems are included in the problem bank for annotation but are not yet part of the published preference dataset.

**Python (10 problems)**
- Easy: find duplicates, palindrome check, word frequency, FizzBuzz, reverse string
- Medium: LRU cache, binary search, group anagrams, merge intervals, validate parentheses
- Hard: thread-safe singleton, async task queue with retries

**Go (8 problems)**
- Easy: string reversal, basic error handling
- Medium: concurrent worker pool, HTTP server with timeout, JSON API handler, file reader with error handling
- Hard: rate limiter with goroutines, context-aware pipeline

**JavaScript (4 problems — annotation only, not in published dataset)**
- Medium: debounce function, async fetch with retry, event emitter, promise chain

### Output — Preference Dataset

Each completed evaluation is saved as:

```json
{
  "id": "uuid",
  "timestamp": "2025-05-05T10:00:00",
  "problem": "Implement a concurrent worker pool in Go",
  "language": "go",
  "model": "qwen2.5-coder:7b",
  "solution_a": "func RunWorkers() {\n  go func() { ... }()\n}",
  "solution_b": "func NewPool(n int) *Pool {\n  p := &Pool{...}\n  ...\n}",
  "test_results": {
    "solution_a": { "passed": 2, "failed": 2, "total": 4 },
    "solution_b": { "passed": 4, "failed": 0, "total": 4 }
  },
  "scores": {
    "correctness":   [0, 1],
    "complexity":    [0, 1],
    "readability":   [0, 1],
    "edge_cases":    [0, 1],
    "idiomatic":     [0, 1],
    "security":      [1, 1]
  },
  "bugs_found": {
    "solution_a": ["goroutine_leak", "missing_error_handling"],
    "solution_b": []
  },
  "complexity": {
    "solution_a": { "time": "O(n)", "space": "O(1)", "bottleneck": "none" },
    "solution_b": { "time": "O(n)", "space": "O(n)", "bottleneck": "channel buffer" }
  },
  "winner": "B",
  "justification": "Solution B uses sync.WaitGroup correctly and properly closes the jobs channel. Solution A leaks goroutines — there is no mechanism to signal workers to stop. B also handles the context cancellation pattern, making it production-safe.",
  "ai_critique": "..."
}
```

This format is directly compatible with HuggingFace `datasets` for reward model training.

---

## Reward Model Training

### What it does
Trains a code-aware reward model on your collected preference pairs. The model learns a single thing: `score(chosen) > score(rejected)`.

### Architecture
- **Base model:** `microsoft/codebert-base` (encoder, 125M parameters)
- **Fine-tuning:** Bradley-Terry preference learning
- **Loss:** `−log(sigmoid(reward_chosen − reward_rejected))`
- **Output:** A scalar score 0–1 representing code quality

### Training
```
Dataset        : your evaluations.jsonl (80/20 train/val split)
Training time  : ~30 min on M4 MacBook (MPS backend)
Target metric  : >80% accuracy on held-out preference pairs
```

### What good reward model accuracy means
If your reward model achieves 80%+ accuracy, it means it has successfully learned the quality signals from your human annotations — it can now score new code solutions without you.

---

## DPO Fine-Tuning

### What it does
Fine-tunes a small code LLM using Direct Preference Optimization (DPO) — the modern alternative to PPO-based RLHF. DPO works directly on preference pairs, skipping the need for an explicit RL training loop.

### Why DPO over PPO
- Simpler to implement and more stable to train
- No separate reward model needed at fine-tuning time
- Increasingly adopted by industry (Mistral, Llama fine-tunes)
- Runs on consumer hardware

### Setup
```
Base model     : Qwen2.5-Coder 1.5B (fits M4 MacBook RAM)
Method         : LoRA (Low-Rank Adaptation) via peft
Trainer        : DPOTrainer from HuggingFace trl
Dataset        : your preference pairs from the Annotation Engine
Training time  : ~2 hours on M4 MacBook (MPS backend)
```

### What changes after fine-tuning
The fine-tuned model, when given the same coding problems, should:
- Add type hints and docstrings more consistently
- Handle edge cases (None, empty input) by default
- Write more idiomatic Python and Go patterns
- Produce code that scores higher on your reward model

---

## Evaluation Dashboard

### What it shows
A Streamlit dashboard that runs both the base and fine-tuned model on the same problem and compares them side by side — scored by your reward model.

```
Problem: Implement an LRU Cache in Python
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Base Model                  Fine-tuned Model
Reward Score: 0.41          Reward Score: 0.79  ↑93%

class LRUCache:             class LRUCache:
  def __init__(self, cap):    def __init__(self, capacity: int) -> None:
    self.cap = cap              """LRU cache with O(1) get and put."""
    self.cache = {}             self.capacity = capacity
                                self._cache: dict = {}
  def get(self, key):           self._order: list = []
    if key in self.cache:
      return self.cache[key]  def get(self, key: int) -> int:
    return -1                   if key not in self._cache:
                                  return -1
                                self._order.remove(key)
                                self._order.append(key)
                                return self._cache[key]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test Cases:  2/4 pass        Test Cases:  4/4 pass
```

### Aggregate Metrics (actual results)
```
Reward Model Accuracy          : 84% on held-out pairs
Average Reward Improvement     : +89% after DPO fine-tuning
Test Case Pass Rate            : 71% → 94%
Total Preference Pairs         : 127
Languages                      : Python (80) | Go (47)
Problems Covered               : Easy (40) | Medium (60) | Hard (27)
```

---

## Project Structure

```
AlignCode/
│
├── annotation_engine/
│   ├── app.py                      # Streamlit annotation UI
│   ├── services/
│   │   └── ollama_service.py       # Solution generation + AI critique
│   ├── utils/
│   │   ├── sandbox.py              # Safe code execution
│   │   ├── storage.py              # Save / load JSONL
│   │   └── export.py               # HuggingFace-compatible export
│   └── problems/
│       └── problem_bank.py         # 22 problems, Python + Go + JS
│
├── reward_model/
│   ├── train.py                    # Reward model training script
│   ├── evaluate.py                 # Accuracy on held-out pairs
│   └── model/                      # Saved reward model weights
│
├── dpo_finetuning/
│   ├── train_dpo.py                # DPO training with TRL
│   ├── lora_config.py              # LoRA hyperparameters
│   └── model/                      # Fine-tuned model weights
│
├── evaluation_dashboard/
│   └── dashboard.py                # Before vs after comparison UI
│
├── data/
│   ├── evaluations.jsonl           # Raw preference pairs
│   └── dataset_stats.md            # Coverage and quality metrics per language/difficulty
│
├── portfolio/
│   ├── sample_evaluations/         # 20 hand-picked annotated pairs (reviewable without running the pipeline)
│   │   ├── python_lru_cache.json
│   │   ├── go_worker_pool.json
│   │   └── python_rate_limiter.json
│   └── annotation_guidelines.md   # Personal evaluation rubric
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Annotation UI | Streamlit |
| LLM Backend | Ollama (qwen2.5-coder:7b) |
| Reward Model | CodeBERT + HuggingFace transformers |
| Fine-tuning | TRL (DPOTrainer) + PEFT (LoRA) |
| Hardware | Apple M4 MacBook Pro (MPS backend) |
| Storage | JSONL → HuggingFace datasets |
| Language | Python + Go problem domain |

**Key dependencies:** `torch`, `transformers`, `trl`, `peft`, `streamlit`, `datasets`, `ollama`

---

## Setup & Running

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running (see Quick Start)
- ~15GB free disk space (model weights + dependencies)

> **Hardware note (MacBook Pro M4 16GB):** Training scripts are optimized for Apple Silicon via PyTorch MPS, but are fully compatible with NVIDIA GPUs (CUDA).
>
> | Stage | RAM Usage | Time |
> |---|---|---|
> | Annotation (qwen2.5-coder:7b via Ollama) | ~5GB | real-time |
> | Reward Model Training | ~2–4GB | ~30 min |
> | DPO Fine-Tuning (Qwen2.5-Coder 1.5B) | ~8–12GB | ~2 hours |
>
> To verify MPS is available after installing dependencies:
> ```bash
> python -c "import torch; print(torch.backends.mps.is_available())"
> ```

### Installation

```bash
git clone https://github.com/your-username/aligncode
cd aligncode
python -m venv venv
source venv/bin/activate

# Note: first install takes 5–10 min due to torch + transformers
pip install -r requirements.txt
```

### Run the Annotation Engine
```bash
cd annotation_engine
streamlit run app.py
```

### Train the Reward Model
```bash
cd reward_model
python train.py --data ../data/evaluations.jsonl
```

### DPO Fine-Tuning
```bash
cd dpo_finetuning
python train_dpo.py --data ../data/evaluations.jsonl
```

### Run the Evaluation Dashboard
```bash
cd evaluation_dashboard
streamlit run dashboard.py
```

---

## Dataset

The preference dataset collected during this project is published on HuggingFace:

**[rajan/aligncode-preference-dataset](https://huggingface.co/datasets/rajan/aligncode-preference-dataset)**

- 127 preference pairs (Python: 80, Go: 47)
- Easy / Medium / Hard distribution
- Includes correctness, complexity, security, and idiom scores per pair
- Go-specific bug annotations (goroutine leaks, error handling, context propagation)

This is one of the few public preference datasets that includes Go — most existing datasets are Python-only.

Coverage and quality metrics are documented in [`data/dataset_stats.md`](data/dataset_stats.md).

---

## Annotation Guidelines

The personal rubric used for evaluating code pairs is documented in [`portfolio/annotation_guidelines.md`](portfolio/annotation_guidelines.md).

It covers:
- How to evaluate correctness vs readability tradeoffs
- When idiomatic style matters more than brevity
- Go-specific patterns to look for
- Security issues that are easy to miss
- How to write a strong justification

---

## Connection to RLHF

This project implements each stage of the RLHF pipeline that powers models like Claude and ChatGPT:

| RLHF Stage | AlignCode Implementation |
|---|---|
| Supervised Fine-Tuning (SFT) | Base Ollama model (already instruction-tuned) |
| Human Preference Collection | Annotation Engine |
| Reward Model Training | Reward Model Training — CodeBERT |
| Policy Optimization | DPO Fine-Tuning |
| Evaluation | Evaluation Dashboard — before vs after |

---

## Results

| Metric | Value |
|---|---|
| Reward Model Accuracy | 84% on held-out pairs |
| Average Reward Score Improvement | +89% after DPO |
| Test Case Pass Rate (base) | 71% |
| Test Case Pass Rate (fine-tuned) | 94% |
| Total Annotations | 127 pairs |
| Annotation Time | ~4 min per pair average |

---

## Troubleshooting

**`ollama: command not found`**
Ollama isn't installed or not on your PATH. Follow the [Ollama install guide](https://ollama.ai/) and restart your terminal.

**`torch.backends.mps.is_available()` returns `False`**
Ensure you're on macOS 12.3+ with Apple Silicon and have installed the MPS-enabled torch build (`torch >= 2.0`). On Intel Macs or Linux, the scripts fall back to CPU automatically.

**Reward model training is very slow**
Check that MPS or CUDA is being used — if torch falls back to CPU, training will be significantly slower. Run `python -c "import torch; print(torch.device('mps' if torch.backends.mps.is_available() else 'cpu'))"` to confirm.

**DPO training runs out of memory**
The 1.5B model is sized for 16GB unified memory. If you're on a smaller machine, try reducing `per_device_train_batch_size` to `1` and enabling `gradient_checkpointing=True` in `lora_config.py`.

**Streamlit app doesn't connect to Ollama**
Make sure Ollama is running in the background (`ollama serve`) before launching the Streamlit app. The annotation engine expects Ollama at `http://localhost:11434` by default.

---

## Roadmap & Future Work

| Priority | Item | Notes |
|---|---|---|
| High | Expand problem bank to 100+ problems | Increase dataset variance across more domains |
| High | Add JavaScript pairs to published dataset | Currently annotated but not exported |
| Medium | PPO vs. DPO comparison | Implement PPO loop to benchmark stability and quality against DPO |
| Medium | Multi-turn evaluation | Extend annotation engine to evaluate conversational code refinement |
| Low | Inter-annotator agreement tooling | Support multiple annotators and measure consistency |

---

## Author

**Rajan**
- GitHub: [github.com/rajan0860](https://github.com/rajan0860)
- HuggingFace: [huggingface.co/rajan0860](https://huggingface.co/rajan0860)
- LinkedIn: [linkedin.com/in/rajan0860](https://linkedin.com/in/rajan0860)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
