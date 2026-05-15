# AlignCode — Annotation Guidelines

Personal evaluation rubric used for scoring AI-generated code pairs in the AlignCode annotation engine.

---

## General Principles

1. **Score independently.** Evaluate Solution A and Solution B on their own merits before deciding a winner. Don't let one solution's quality bias your perception of the other.
2. **Test results inform, not decide.** A solution that passes all tests but uses `eval()` on user input is still insecure. A solution that fails one edge case but has excellent structure may still be the better foundation.
3. **Context matters.** A "quick script" and a "production service" have different quality bars. Score based on what the problem asks for.

---

## Criterion-by-Criterion Guide

### Correctness (0/1)
- **Pass:** Solution produces correct output for all provided test cases, including edge cases.
- **Fail:** Any test case failure, incorrect return type, or runtime error.
- **Grey area:** If the solution is correct for all *provided* tests but you can identify an untested input that would break it, mark as 0 and note the case in justification.

### Time Complexity (0/1)
- **Pass:** Uses an optimal or near-optimal algorithm for the problem class.
- **Fail:** Uses a clearly suboptimal approach (e.g., O(n²) when O(n log n) is standard).
- **Grey area:** If the problem size is small enough that complexity doesn't matter practically, lean toward Pass but note it.

### Space Complexity (0/1)
- **Pass:** Memory usage is proportional to what the problem requires.
- **Fail:** Unnecessary data structure duplication, unbounded caching, or storing the entire input when streaming would suffice.

### Readability (0/1)
- **Pass:** Clear variable names, logical structure, consistent formatting. A new team member could understand it in under 2 minutes.
- **Fail:** Single-letter variables (except loop counters), deeply nested logic, no separation of concerns.

### Idiomatic Style (0/1)
- **Pass:** Written the way an experienced developer in that language would write it.
  - Python: list comprehensions where appropriate, context managers, type hints
  - Go: error-first returns, `defer` for cleanup, exported types with comments
- **Fail:** Transliterated from another language's idioms (e.g., Java-style getters in Python, C-style loops in Go).

### Edge Cases (0/1)
- **Pass:** Explicitly handles `None`/`nil`, empty input, single-element input, overflow, and duplicate values where relevant.
- **Fail:** Crashes or produces wrong output on boundary inputs.

### Security (0/1)
- **Pass:** No injection vulnerabilities, no hardcoded secrets, no unsafe deserialization, no `eval()`/`exec()` on user input.
- **Fail:** Any of the above present, even if the problem doesn't explicitly mention security.

---

## Bug Checklist

When reviewing each solution, actively check for these categories:

### General Bugs
- Off-by-one errors (especially in binary search, sliding window)
- Unhandled `None` / `null` input
- Mutable default arguments (Python `def f(x=[])`)
- Incorrect return type (returning `None` when `int` expected)
- Shadowed variable names
- Memory leak patterns (unclosed files, connections)

### Security Issues
- Hardcoded credentials or API keys
- SQL injection vulnerability
- Use of `eval()` / `exec()` on user input
- Insecure random number generation (`random` instead of `secrets`)
- Path traversal vulnerability

### Go-Specific Issues
- Goroutine leak (missing `WaitGroup` / done channel)
- Missing `defer` for `Close()` / `Unlock()`
- Ignored error returns (`val, _ := someFunc()` without justification)
- Shadowed `err` variable in nested `if` block
- Race condition on shared variable (missing mutex)
- Incorrect value vs pointer receiver
- Context not propagated down call chain
- Unbuffered channel causing deadlock

---

## Complexity Annotation

For each solution, record:
- **Time complexity:** Select from O(1) → O(2ⁿ)
- **Space complexity:** Select from O(1) → O(2ⁿ)
- **Bottleneck:** Free-text description of where the dominant cost comes from (e.g., "sorting step", "hash table construction", "recursive calls without memoization")

---

## Writing a Strong Justification

The justification field requires a minimum of 80 characters. A good justification:

1. **States the winner clearly** — "B is better because..."
2. **References specific criteria** — "B handles the None edge case (line 12) while A crashes"
3. **Acknowledges tradeoffs** — "A is more readable, but B's O(n) approach is significantly faster for large inputs"
4. **Mentions security if relevant** — "A uses `eval()` which is a security risk even in this context"

### Examples

**Weak:** "B is better overall."

**Strong:** "Solution B is the clear winner. It handles None input gracefully (returns empty list), uses a set-based approach for O(n) time vs A's nested loop O(n²), and includes type hints throughout. A's only advantage is brevity, but it crashes on empty input and shadows the built-in `list` name."

---

## When to Use AI Critique

After completing your own evaluation, click "Show AI Critique" to get the local model's analysis. Use this to:
- **Catch bugs you missed** — the model may notice edge cases you overlooked
- **Validate your complexity analysis** — compare your Big-O assessment against the model's
- **Identify bias** — if the model disagrees with your winner choice, reconsider your reasoning

The AI critique should *supplement* your evaluation, never *replace* it. Your human judgment on readability, idiomatic style, and real-world usability is what makes this dataset valuable.
