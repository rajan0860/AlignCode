"""
annotation_engine/problems/problem_bank.py

22 hand-selected coding problems across Python, Go, and JavaScript.
Each problem includes a statement, language, difficulty, and predefined test cases.

Structure:
    PROBLEMS: list[dict] — all problems
    get_problems_by_language(lang) — filter by "python" | "go" | "javascript"
    get_problems_by_difficulty(level) — filter by "easy" | "medium" | "hard"
"""

from typing import Any

PROBLEMS: list[dict[str, Any]] = [
    # ── Python — Easy ──────────────────────────────────────────────────────
    {
        "id": "py_find_duplicates",
        "language": "python",
        "difficulty": "easy",
        "title": "Find Duplicates",
        "statement": (
            "Write a function `find_duplicates(nums: list[int]) -> list[int]` "
            "that returns a sorted list of all duplicate values in the input list. "
            "Handle None and empty input gracefully."
        ),
        "test_cases": [
            {"call": "find_duplicates([1,2,3,2,1])", "expected": [1, 2]},
            {"call": "find_duplicates([])", "expected": []},
            {"call": "find_duplicates(None)", "expected": []},
            {"call": "find_duplicates([1,1,1])", "expected": [1]},
        ],
    },
    {
        "id": "py_palindrome",
        "language": "python",
        "difficulty": "easy",
        "title": "Palindrome Check",
        "statement": (
            "Write a function `is_palindrome(s: str) -> bool` that returns True "
            "if the string is a palindrome (ignoring case and non-alphanumeric characters)."
        ),
        "test_cases": [
            {"call": "is_palindrome('racecar')", "expected": True},
            {"call": "is_palindrome('A man a plan a canal Panama')", "expected": True},
            {"call": "is_palindrome('hello')", "expected": False},
            {"call": "is_palindrome('')", "expected": True},
        ],
    },
    {
        "id": "py_word_frequency",
        "language": "python",
        "difficulty": "easy",
        "title": "Word Frequency",
        "statement": (
            "Write a function `word_frequency(text: str) -> dict[str, int]` "
            "that returns a dictionary of word counts (case-insensitive). "
            "Handle None and empty string input."
        ),
        "test_cases": [
            {"call": "word_frequency('hello world hello')", "expected": {"hello": 2, "world": 1}},
            {"call": "word_frequency('')", "expected": {}},
            {"call": "word_frequency(None)", "expected": {}},
        ],
    },
    {
        "id": "py_fizzbuzz",
        "language": "python",
        "difficulty": "easy",
        "title": "FizzBuzz",
        "statement": (
            "Write a function `fizzbuzz(n: int) -> list[str]` that returns a list "
            "of strings from 1 to n, replacing multiples of 3 with 'Fizz', "
            "multiples of 5 with 'Buzz', and multiples of both with 'FizzBuzz'."
        ),
        "test_cases": [
            {"call": "fizzbuzz(5)", "expected": ["1", "2", "Fizz", "4", "Buzz"]},
            {"call": "fizzbuzz(15)[14]", "expected": "FizzBuzz"},
            {"call": "fizzbuzz(0)", "expected": []},
        ],
    },
    {
        "id": "py_reverse_string",
        "language": "python",
        "difficulty": "easy",
        "title": "Reverse String",
        "statement": (
            "Write a function `reverse_string(s: str) -> str` that returns the "
            "reversed string. Handle None input by returning an empty string."
        ),
        "test_cases": [
            {"call": "reverse_string('hello')", "expected": "olleh"},
            {"call": "reverse_string('')", "expected": ""},
            {"call": "reverse_string(None)", "expected": ""},
        ],
    },
    # ── Python — Medium ────────────────────────────────────────────────────
    {
        "id": "py_lru_cache",
        "language": "python",
        "difficulty": "medium",
        "title": "LRU Cache",
        "statement": (
            "Implement an LRU (Least Recently Used) cache class `LRUCache` with "
            "`__init__(self, capacity: int)`, `get(self, key: int) -> int` (return -1 if missing), "
            "and `put(self, key: int, value: int) -> None`. Both operations must be O(1)."
        ),
        "test_cases": [
            {
                "call": (
                    "(lambda c: (c.put(1,1), c.put(2,2), c.get(1), c.put(3,3), c.get(2))[-1])"
                    "(LRUCache(2))"
                ),
                "expected": -1,
            },
        ],
    },
    {
        "id": "py_binary_search",
        "language": "python",
        "difficulty": "medium",
        "title": "Binary Search",
        "statement": (
            "Write a function `binary_search(nums: list[int], target: int) -> int` "
            "that returns the index of target in a sorted list, or -1 if not found."
        ),
        "test_cases": [
            {"call": "binary_search([1,3,5,7,9], 5)", "expected": 2},
            {"call": "binary_search([1,3,5,7,9], 6)", "expected": -1},
            {"call": "binary_search([], 1)", "expected": -1},
        ],
    },
    {
        "id": "py_group_anagrams",
        "language": "python",
        "difficulty": "medium",
        "title": "Group Anagrams",
        "statement": (
            "Write a function `group_anagrams(words: list[str]) -> list[list[str]]` "
            "that groups anagrams together. Order of groups and words within groups does not matter."
        ),
        "test_cases": [
            {
                "call": "sorted([sorted(g) for g in group_anagrams(['eat','tea','tan','ate','nat','bat'])])",
                "expected": [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]],
            },
        ],
    },
    {
        "id": "py_merge_intervals",
        "language": "python",
        "difficulty": "medium",
        "title": "Merge Intervals",
        "statement": (
            "Write a function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` "
            "that merges all overlapping intervals and returns the result sorted."
        ),
        "test_cases": [
            {"call": "merge_intervals([[1,3],[2,6],[8,10],[15,18]])", "expected": [[1, 6], [8, 10], [15, 18]]},
            {"call": "merge_intervals([[1,4],[4,5]])", "expected": [[1, 5]]},
            {"call": "merge_intervals([])", "expected": []},
        ],
    },
    {
        "id": "py_validate_parentheses",
        "language": "python",
        "difficulty": "medium",
        "title": "Validate Parentheses",
        "statement": (
            "Write a function `is_valid(s: str) -> bool` that returns True if the "
            "string of brackets is valid (every open bracket has a matching close bracket "
            "in the correct order). Valid brackets: (), [], {}."
        ),
        "test_cases": [
            {"call": "is_valid('()[]{}')", "expected": True},
            {"call": "is_valid('(]')", "expected": False},
            {"call": "is_valid('')", "expected": True},
            {"call": "is_valid('([)]')", "expected": False},
        ],
    },
    # ── Python — Hard ──────────────────────────────────────────────────────
    {
        "id": "py_thread_safe_singleton",
        "language": "python",
        "difficulty": "hard",
        "title": "Thread-Safe Singleton",
        "statement": (
            "Implement a thread-safe Singleton class in Python. "
            "The class must ensure only one instance is created even under concurrent access. "
            "Use double-checked locking or an equivalent pattern."
        ),
        "test_cases": [],  # Verified manually — concurrency correctness requires runtime testing
    },
    {
        "id": "py_async_task_queue",
        "language": "python",
        "difficulty": "hard",
        "title": "Async Task Queue with Retries",
        "statement": (
            "Implement an async task queue `AsyncTaskQueue` that processes tasks concurrently "
            "with a configurable worker count. Failed tasks should be retried up to `max_retries` "
            "times with exponential backoff. Use asyncio."
        ),
        "test_cases": [],  # Verified manually — async runtime required
    },
    # ── Go — Easy ──────────────────────────────────────────────────────────
    {
        "id": "go_reverse_string",
        "language": "go",
        "difficulty": "easy",
        "title": "String Reversal",
        "statement": (
            "Write a Go function `ReverseString(s string) string` that returns the reversed string. "
            "Handle Unicode characters correctly (reverse by rune, not byte)."
        ),
        "test_cases": [],  # Go tests run via go test, not Python sandbox
    },
    {
        "id": "go_basic_error_handling",
        "language": "go",
        "difficulty": "easy",
        "title": "Basic Error Handling",
        "statement": (
            "Write a Go function `ParsePositiveInt(s string) (int, error)` that parses a string "
            "to a positive integer. Return a descriptive error for invalid input, negative numbers, "
            "and zero."
        ),
        "test_cases": [],
    },
    # ── Go — Medium ────────────────────────────────────────────────────────
    {
        "id": "go_worker_pool",
        "language": "go",
        "difficulty": "medium",
        "title": "Concurrent Worker Pool",
        "statement": (
            "Implement a concurrent worker pool in Go. `NewPool(workers int) *Pool` creates a pool. "
            "`Submit(job func())` adds a job. `Shutdown()` waits for all jobs to complete and stops workers. "
            "No goroutine leaks allowed."
        ),
        "test_cases": [],
    },
    {
        "id": "go_http_server",
        "language": "go",
        "difficulty": "medium",
        "title": "HTTP Server with Timeout",
        "statement": (
            "Implement a Go HTTP server with configurable read/write timeouts and graceful shutdown "
            "on SIGINT/SIGTERM. The server should propagate context cancellation to handlers."
        ),
        "test_cases": [],
    },
    {
        "id": "go_json_api_handler",
        "language": "go",
        "difficulty": "medium",
        "title": "JSON API Handler",
        "statement": (
            "Write a Go HTTP handler `CreateUserHandler` that decodes a JSON request body into a User struct, "
            "validates required fields (name, email), and returns appropriate HTTP status codes with JSON responses."
        ),
        "test_cases": [],
    },
    {
        "id": "go_file_reader",
        "language": "go",
        "difficulty": "medium",
        "title": "File Reader with Error Handling",
        "statement": (
            "Write a Go function `ReadLines(path string) ([]string, error)` that reads a file line by line. "
            "Use defer for cleanup, wrap errors with context, and handle empty files gracefully."
        ),
        "test_cases": [],
    },
    # ── Go — Hard ──────────────────────────────────────────────────────────
    {
        "id": "go_rate_limiter",
        "language": "go",
        "difficulty": "hard",
        "title": "Rate Limiter with Goroutines",
        "statement": (
            "Implement a token-bucket rate limiter in Go. `NewRateLimiter(rate int, burst int) *RateLimiter` "
            "creates the limiter. `Allow() bool` returns true if a request is permitted. "
            "Must be safe for concurrent use."
        ),
        "test_cases": [],
    },
    {
        "id": "go_context_pipeline",
        "language": "go",
        "difficulty": "hard",
        "title": "Context-Aware Pipeline",
        "statement": (
            "Implement a multi-stage data processing pipeline in Go using channels and context. "
            "Each stage should respect context cancellation and propagate it downstream. "
            "No goroutine leaks on cancellation."
        ),
        "test_cases": [],
    },
    # ── JavaScript — Medium ────────────────────────────────────────────────
    {
        "id": "js_debounce",
        "language": "javascript",
        "difficulty": "medium",
        "title": "Debounce Function",
        "statement": (
            "Implement a `debounce(fn, delay)` function in JavaScript that returns a debounced version "
            "of fn. The debounced function delays invoking fn until after delay milliseconds have elapsed "
            "since the last invocation."
        ),
        "test_cases": [],
    },
    {
        "id": "js_async_fetch_retry",
        "language": "javascript",
        "difficulty": "medium",
        "title": "Async Fetch with Retry",
        "statement": (
            "Implement `fetchWithRetry(url, options, maxRetries)` in JavaScript that retries a failed "
            "fetch request up to maxRetries times with exponential backoff. Reject after all retries exhausted."
        ),
        "test_cases": [],
    },
    {
        "id": "js_event_emitter",
        "language": "javascript",
        "difficulty": "medium",
        "title": "Event Emitter",
        "statement": (
            "Implement an `EventEmitter` class in JavaScript with `on(event, listener)`, "
            "`off(event, listener)`, `emit(event, ...args)`, and `once(event, listener)` methods."
        ),
        "test_cases": [],
    },
    {
        "id": "js_promise_chain",
        "language": "javascript",
        "difficulty": "medium",
        "title": "Promise Chain",
        "statement": (
            "Implement `runSequential(tasks)` in JavaScript where tasks is an array of functions "
            "that return Promises. Run them sequentially (not in parallel) and return a Promise "
            "that resolves with an array of all results."
        ),
        "test_cases": [],
    },
]


def get_problems_by_language(language: str) -> list[dict]:
    """Filter problems by language: 'python', 'go', or 'javascript'."""
    return [p for p in PROBLEMS if p["language"] == language.lower()]


def get_problems_by_difficulty(difficulty: str) -> list[dict]:
    """Filter problems by difficulty: 'easy', 'medium', or 'hard'."""
    return [p for p in PROBLEMS if p["difficulty"] == difficulty.lower()]


def get_problem_by_id(problem_id: str) -> dict | None:
    """Look up a problem by its ID."""
    return next((p for p in PROBLEMS if p["id"] == problem_id), None)
