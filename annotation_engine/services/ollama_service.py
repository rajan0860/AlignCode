"""
annotation_engine/services/ollama_service.py

Handles all communication with the local Ollama API.

Responsibilities:
  - Generate Solution A using the "quick" system prompt
  - Generate Solution B using the "production" system prompt
  - Request an AI critique comparing both solutions
  - Manage model selection and connection health checks

Ollama is expected to be running at http://localhost:11434.
Run `ollama serve` before launching the annotation engine.
"""

import ollama


QUICK_SYSTEM_PROMPT = (
    "You are a pragmatic developer. Write a concise, working solution. "
    "Focus on correctness. Keep it short."
)

PRODUCTION_SYSTEM_PROMPT = (
    "You are a senior engineer. Write idiomatic, production-quality code. "
    "Include type hints, docstrings, and handle all edge cases explicitly."
)


def generate_solution(problem: str, prompt_style: str, model: str = "qwen2.5-coder:7b") -> str:
    """
    Generate a code solution for the given problem.

    Args:
        problem: The problem statement.
        prompt_style: "quick" or "production".
        model: Ollama model name to use.

    Returns:
        Generated code as a string.
    """
    system_prompt = QUICK_SYSTEM_PROMPT if prompt_style == "quick" else PRODUCTION_SYSTEM_PROMPT
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ],
    )
    return response["message"]["content"]


def get_ai_critique(problem: str, solution_a: str, solution_b: str, model: str = "qwen2.5-coder:7b") -> str:
    """
    Ask the local model to compare both solutions and provide a critique.

    Args:
        problem: The original problem statement.
        solution_a: The quick solution.
        solution_b: The production solution.
        model: Ollama model name to use.

    Returns:
        AI critique as a string.
    """
    prompt = (
        f"Problem:\n{problem}\n\n"
        f"Solution A:\n{solution_a}\n\n"
        f"Solution B:\n{solution_b}\n\n"
        "Compare both solutions. Which is better and why? "
        "Consider correctness, complexity, idioms, edge cases, and security."
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def check_ollama_connection(model: str = "qwen2.5-coder:7b") -> bool:
    """Return True if Ollama is reachable and the model is available."""
    try:
        ollama.show(model)
        return True
    except Exception:
        return False


def refine_solution(
    problem: str,
    original_code: str,
    feedback: str,
    model: str = "qwen2.5-coder:7b",
) -> str:
    """
    Send a multi-turn conversation to refine an existing solution based on feedback.

    Args:
        problem: The original problem statement.
        original_code: The code the model generated in the first turn.
        feedback: The human annotator's feedback on what to fix/improve.
        model: Ollama model name to use.

    Returns:
        Refined code as a string.
    """
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": PRODUCTION_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": original_code},
            {"role": "user", "content": f"Please revise the solution based on this feedback:\n{feedback}"},
        ],
    )
    return response["message"]["content"]

