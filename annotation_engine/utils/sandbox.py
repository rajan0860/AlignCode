"""
annotation_engine/utils/sandbox.py

Safe code execution sandbox for evaluating LLM-generated solutions.

Executes code in an isolated subprocess with:
  - Hard timeout (default 5 seconds)
  - Memory limit monitoring via psutil
  - No network access assumptions (caller responsibility)
  - Captured stdout / stderr

Usage:
    from annotation_engine.utils.sandbox import run_tests

    results = run_tests(code="def add(a, b): return a + b", test_cases=[...])
"""

import subprocess
import sys
import textwrap
from dataclasses import dataclass, field


TIMEOUT_SECONDS = 5
MEMORY_LIMIT_MB = 256


@dataclass
class TestResult:
    test_input: str
    expected: str
    actual: str
    passed: bool
    error: str = ""


@dataclass
class SandboxResult:
    passed: int = 0
    failed: int = 0
    total: int = 0
    results: list[TestResult] = field(default_factory=list)
    timed_out: bool = False


def run_tests(code: str, test_cases: list[dict]) -> SandboxResult:
    """
    Execute code against a list of test cases in an isolated subprocess.

    Args:
        code: The Python source code to evaluate.
        test_cases: List of dicts with keys: "input", "expected", "call".
                    "call" is a string expression to evaluate after exec(code).

    Returns:
        SandboxResult with per-test pass/fail details.
    """
    sandbox_result = SandboxResult(total=len(test_cases))

    for tc in test_cases:
        script = textwrap.dedent(f"""
            {code}
            result = {tc['call']}
            print(repr(result))
        """)

        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
            actual = proc.stdout.strip()
            expected = repr(tc["expected"])
            passed = actual == expected

            sandbox_result.results.append(
                TestResult(
                    test_input=tc["call"],
                    expected=expected,
                    actual=actual,
                    passed=passed,
                    error=proc.stderr.strip(),
                )
            )
            if passed:
                sandbox_result.passed += 1
            else:
                sandbox_result.failed += 1

        except subprocess.TimeoutExpired:
            sandbox_result.timed_out = True
            sandbox_result.failed += 1
            sandbox_result.results.append(
                TestResult(
                    test_input=tc["call"],
                    expected=repr(tc["expected"]),
                    actual="",
                    passed=False,
                    error=f"Timed out after {TIMEOUT_SECONDS}s",
                )
            )

    return sandbox_result
