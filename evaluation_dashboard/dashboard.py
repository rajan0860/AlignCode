"""
evaluation_dashboard/dashboard.py

Streamlit dashboard for comparing base vs DPO vs PPO fine-tuned model output.

For each problem:
  1. Generate a solution from the base Ollama model
  2. Generate a solution from the DPO fine-tuned model
  3. Generate a solution from the PPO fine-tuned model (if available)
  4. Score all solutions using the trained reward model
  5. Run all solutions through the sandbox test cases
  6. Display side-by-side comparison with reward scores and test results

Usage:
    streamlit run dashboard.py

Requires:
  - Ollama running locally (ollama serve)
  - reward_model/model/reward_model.pt trained
  - dpo_finetuning/model/ and/or ppo_finetuning/model/ fine-tuned weights saved
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
DPO_MODEL_PATH = Path(__file__).parents[1] / "dpo_finetuning" / "model"
PPO_MODEL_PATH = Path(__file__).parents[1] / "ppo_finetuning" / "model"


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
def load_finetuned_model(model_path: str):
    """Load a fine-tuned HF model from a given path. Returns None if path doesn't exist."""
    p = Path(model_path)
    if not p.exists() or not any(p.iterdir()):
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(p), trust_remote_code=True)
    return model, tokenizer


def generate_with_hf(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def render_model_column(label: str, code: str, score: float, tests, base_score: float, lang: str):
    """Render a single model's results in a Streamlit column."""
    st.subheader(label)
    if base_score > 0 and label != "Base Model":
        improvement = ((score - base_score) / base_score * 100)
        st.metric("Reward Score", f"{score:.2f}", delta=f"{improvement:+.0f}%")
    else:
        st.metric("Reward Score", f"{score:.2f}")
    st.metric("Test Cases", f"{tests.passed}/{tests.total} pass")
    st.code(code, language=lang)


def main():
    st.set_page_config(page_title="AlignCode — Evaluation Dashboard", layout="wide")
    st.title("AlignCode — Evaluation Dashboard")
    st.caption("Base model vs DPO vs PPO, scored by the reward model.")

    # Check which fine-tuned models are available
    dpo_available = DPO_MODEL_PATH.exists() and any(DPO_MODEL_PATH.iterdir()) if DPO_MODEL_PATH.exists() else False
    ppo_available = PPO_MODEL_PATH.exists() and any(PPO_MODEL_PATH.iterdir()) if PPO_MODEL_PATH.exists() else False

    status_parts = ["🟢 Base (Ollama)"]
    status_parts.append("🟢 DPO" if dpo_available else "⚪ DPO (not trained)")
    status_parts.append("🟢 PPO" if ppo_available else "⚪ PPO (not trained)")
    st.sidebar.markdown("### Available Models")
    for s in status_parts:
        st.sidebar.markdown(s)

    python_problems = [p for p in PROBLEMS if p["language"] == "python" and p["test_cases"]]
    problem_titles = [p["title"] for p in python_problems]
    selected_title = st.selectbox("Select a problem", problem_titles)
    problem = next(p for p in python_problems if p["title"] == selected_title)

    st.markdown(f"**Problem:** {problem['statement']}")

    if st.button("Run Comparison"):
        with st.spinner("Loading reward model..."):
            reward_model, reward_tokenizer, device = load_reward_model()

        import ollama as ollama_client

        # 1. Base model
        with st.spinner("Generating base model solution..."):
            base_response = ollama_client.chat(
                model="qwen2.5-coder:7b",
                messages=[{"role": "user", "content": problem["statement"]}],
            )
            base_code = base_response["message"]["content"]

        base_score = score_solution(reward_model, reward_tokenizer, base_code, device)
        base_tests = run_tests(base_code, problem["test_cases"])

        # 2. DPO model
        dpo_code, dpo_score, dpo_tests = None, None, None
        if dpo_available:
            with st.spinner("Loading DPO model..."):
                dpo_model, dpo_tokenizer = load_finetuned_model(str(DPO_MODEL_PATH))
            if dpo_model:
                with st.spinner("Generating DPO solution..."):
                    dpo_code = generate_with_hf(dpo_model, dpo_tokenizer, problem["statement"])
                dpo_score = score_solution(reward_model, reward_tokenizer, dpo_code, device)
                dpo_tests = run_tests(dpo_code, problem["test_cases"])

        # 3. PPO model
        ppo_code, ppo_score, ppo_tests = None, None, None
        if ppo_available:
            with st.spinner("Loading PPO model..."):
                ppo_model, ppo_tokenizer = load_finetuned_model(str(PPO_MODEL_PATH))
            if ppo_model:
                with st.spinner("Generating PPO solution..."):
                    ppo_code = generate_with_hf(ppo_model, ppo_tokenizer, problem["statement"])
                ppo_score = score_solution(reward_model, reward_tokenizer, ppo_code, device)
                ppo_tests = run_tests(ppo_code, problem["test_cases"])

        # Determine column count
        num_cols = 1 + (1 if dpo_code else 0) + (1 if ppo_code else 0)
        cols = st.columns(num_cols)

        idx = 0
        with cols[idx]:
            render_model_column("Base Model", base_code, base_score, base_tests, base_score, "python")

        if dpo_code:
            idx += 1
            with cols[idx]:
                render_model_column("DPO Fine-tuned", dpo_code, dpo_score, dpo_tests, base_score, "python")

        if ppo_code:
            idx += 1
            with cols[idx]:
                render_model_column("PPO Fine-tuned", ppo_code, ppo_score, ppo_tests, base_score, "python")


if __name__ == "__main__":
    main()

