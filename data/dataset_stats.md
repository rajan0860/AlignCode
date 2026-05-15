# AlignCode — Dataset Statistics

Coverage and quality metrics for the AlignCode preference dataset.

---

## Problem Bank Coverage

| Language | Easy | Medium | Hard | Total |
|---|---|---|---|---|
| Python | 5 | 5 | 2 | 12 |
| Python (MBPP) | — | 80 | — | 80 |
| Go | 2 | 4 | 2 | 8 |
| JavaScript | — | 4 | — | 4 |
| **Total** | **7** | **93** | **4** | **104** |

---

## Annotation Targets

| Metric | Target | Notes |
|---|---|---|
| Total preference pairs | 127 | Python: 80, Go: 47 |
| Avg annotation time | ~4 min/pair | Includes test review + justification |
| Min justification length | 80 chars | Enforced by the UI |

---

## Scoring Distribution (Target)

| Criterion | Avg Score (Solution A) | Avg Score (Solution B) |
|---|---|---|
| Correctness | 0.55 | 0.82 |
| Time Complexity | 0.65 | 0.78 |
| Space Complexity | 0.70 | 0.75 |
| Readability | 0.50 | 0.85 |
| Idiomatic Style | 0.40 | 0.88 |
| Edge Cases | 0.45 | 0.80 |
| Security | 0.85 | 0.90 |

> **Note:** Solution B consistently scores higher because it is generated with the "production" prompt that emphasizes type hints, docstrings, and edge case handling. This intentional quality gap makes the dataset ideal for preference learning.

---

## Winner Distribution

| Winner | Count | Percentage |
|---|---|---|
| Solution B | ~100 | ~79% |
| Solution A | ~27 | ~21% |

> Solution A occasionally wins when the "quick" prompt produces a more elegant or efficient solution than the over-engineered "production" variant.

---

## Common Bugs Found

| Bug Type | Frequency | Notes |
|---|---|---|
| Unhandled None/null | High | Most common in Solution A |
| Off-by-one errors | Medium | Especially in binary search problems |
| Goroutine leak | Medium | Go-specific, Solution A |
| Missing defer | Medium | Go-specific |
| Mutable default args | Low | Python-specific |
| Security issues | Low | Rare in algorithmic problems |

---

## Language-Specific Notes

### Python
- 80 problems from the hand-selected bank + 80 MBPP-imported problems
- MBPP problems use assert-style test cases
- Strong signal on readability and idiomatic style differences

### Go
- 8 hand-selected problems covering concurrency, error handling, and HTTP patterns
- Go-specific bug annotations (goroutine leaks, context propagation) are a unique feature of this dataset
- No automated test execution (solutions reviewed manually)

### JavaScript
- 4 problems currently in annotation-only mode
- Sandbox execution via Node.js is ready
- Export to the published HuggingFace dataset is pending annotation collection

---

## Model & Hardware

| Parameter | Value |
|---|---|
| Generation model | qwen2.5-coder:7b via Ollama |
| Reward model base | microsoft/codebert-base (125M) |
| DPO fine-tuning base | Qwen2.5-Coder-1.5B-Instruct |
| Hardware | Apple M4 MacBook Pro, 16GB unified |
