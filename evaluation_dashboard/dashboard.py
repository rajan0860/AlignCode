"""
evaluation_dashboard/dashboard.py

Streamlit dashboard for comparing base vs fine-tuned model output.

For each problem:
  1. Generate a solution from the base Ollama model
  2. Generate a solution from the fine-tuned model (loaded via transformers)
  3. Score both solutions using the trained reward model
  4. Run both solutions through the sandbox test cases
  5. Display side-by-side comparison with reward scores and test results

Usage:
    streamlit run dashboard.py

Requires:
  - Ollama running locally (ollama serve)
  - reward_model/model/reward_model.pt trained
  - dpo_finetuning/model/ fine-tuned weights saved
"""

import sys
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow imports from sibling packages
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation_engine.problems.problem_bank import PROBLEMS
from annotation_engine.utils.sandbox import run_tests
from reward_model.train import MODEL_NAME as REWARD_BASE_MODEL
from reward_model.train import RewardModel
from reward_model.evaluate import score_solution

REWARD_MODEL_PATH = Path(__file__).parents[1] / "reward_model" / "model" / "reward_model.pt"
FINETUNED_MODEL_PATH = Path(__file__).parents[1] / "dpo_finetuning" / "model"


@st.cache_resource
def load_reward_model():
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    from transformers import AutoTokenizer as HFTokenizer
    tokenizer = HFTokenizer.from_pretrained(REWARD_BASE_MODEL)
    model = RewardModel()
    model.load_state_dict(torch.load(REWARD_MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model, tokenizer, device


@st.cache_resource
def load_finetuned_model():
    tokenizer = AutoTokenizer.from_pretrained(str(FINETUNED_MODEL_PATH), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(FINETUNED_MODEL_PATH), trust_remote_code=True)
    return model, tokenizer


def generate_with_hf(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    st.set_page_config(page_title="AlignCode — Evaluation Dashboard", layout="wide")
    st.title("AlignCode — Evaluation Dashboard")
    st.caption("Base model vs fine-tuned model, scored by the reward model.")

    python_problems = [p for p in PROBLEMS if p["language"] == "python" and p["test_cases"]]
    problem_titles = [p["title"] for p in python_problems]
    selected_title = st.selectbox("Select a problem", problem_titles)
    problem = next(p for p in python_problems if p["title"] == selected_title)

    st.markdown(f"**Problem:** {problem['statement']}")

    if st.button("Run Comparison"):
        with st.spinner("Loading models..."):
            reward_model, reward_tokenizer, device = load_reward_model()
            ft_model, ft_tokenizer = load_finetuned_model()

        import ollama as ollama_client

        with st.spinner("Generating base model solution..."):
            base_response = ollama_client.chat(
                model="qwen2.5-coder:7b",
                messages=[{"role": "user", "content": problem["statement"]}],
            )
            base_code = base_response["message"]["content"]

        with st.spinner("Generating fine-tuned model solution..."):
            ft_code = generate_with_hf(ft_model, ft_tokenizer, problem["statement"])

        base_score = score_solution(reward_model, reward_tokenizer, base_code, device)
        ft_score = score_solution(reward_model, reward_tokenizer, ft_code, device)

        base_tests = run_tests(base_code, problem["test_cases"])
        ft_tests = run_tests(ft_code, problem["test_cases"])

        improvement = ((ft_score - base_score) / base_score * 100) if base_score > 0 else 0

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Base Model")
            st.metric("Reward Score", f"{base_score:.2f}")
            st.metric("Test Cases", f"{base_tests.passed}/{base_tests.total} pass")
            st.code(base_code, language="python")

        with col2:
            st.subheader("Fine-tuned Model")
            st.metric("Reward Score", f"{ft_score:.2f}", delta=f"{improvement:+.0f}%")
            st.metric("Test Cases", f"{ft_tests.passed}/{ft_tests.total} pass")
            st.code(ft_code, language="python")


if __name__ == "__main__":
    main()
