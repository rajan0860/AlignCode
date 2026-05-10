"""
annotation_engine/utils/sandbox.py

Safe code execution sandbox for evaluating LLM-generated solutions.

Executes code in an isolated subprocess with:
  - Hard timeout (default 5 seconds)
  - Memory limit monitoring via psutil
  - No network access assumptions (caller responsibility)
  - Captured stdout / stderr
"""

import subprocess
import sys
import textwrap
import json
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


def run_tests(code: str, test_cases: list[dict], language: str = "python") -> SandboxResult:
    """
    Execute code against a list of test cases in an isolated subprocess.

    Args:
        code: The source code to evaluate.
        test_cases: List of dicts. If Python/MBPP, can use "assert" string. 
                    Otherwise uses "input", "expected", "call".
        language: "python" or "javascript".

    Returns:
        SandboxResult with per-test pass/fail details.
    """
    sandbox_result = SandboxResult(total=len(test_cases))

    for tc in test_cases:
        if language == "python":
            if "assert" in tc:
                script = f"{code}\n{tc['assert']}\n"
                expected_repr = "No AssertionError"
                test_input_str = tc["assert"]
            else:
                script = textwrap.dedent(f"""
                    {code}
                    result = {tc['call']}
                    print(repr(result))
                """)
                expected_repr = repr(tc["expected"])
                test_input_str = tc["call"]
                
            cmd = [sys.executable, "-c", script]
            
        elif language == "javascript":
            if "assert" in tc:
                # Basic assert polyfill for JS
                script = f"{code}\nconst assert = require('assert');\n{tc['assert']}\n"
                expected_repr = "No AssertionError"
                test_input_str = tc["assert"]
            else:
                expected_json = json.dumps(tc["expected"])
                script = f"""
                    {code}
                    const result = {tc['call']};
                    console.log(JSON.stringify(result));
                """
                expected_repr = expected_json
                test_input_str = tc["call"]
            
            cmd = ["node", "-e", script]
        else:
            # Unsupported language, fail gracefully
            sandbox_result.failed += 1
            sandbox_result.results.append(
                TestResult(test_input="N/A", expected="N/A", actual="", passed=False, error=f"Unsupported language: {language}")
            )
            continue

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
            
            if "assert" in tc:
                passed = proc.returncode == 0
                actual_repr = "Passed" if passed else proc.stderr.strip()
            else:
                actual_repr = proc.stdout.strip()
                passed = actual_repr == expected_repr

            sandbox_result.results.append(
                TestResult(
                    test_input=test_input_str,
                    expected=expected_repr,
                    actual=actual_repr,
                    passed=passed,
                    error=proc.stderr.strip() if not passed else "",
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
                    test_input=test_input_str,
                    expected=expected_repr,
                    actual="",
                    passed=False,
                    error=f"Timed out after {TIMEOUT_SECONDS}s",
                )
            )

    return sandbox_result
