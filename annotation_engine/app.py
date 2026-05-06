"""
annotation_engine/app.py

Streamlit annotation UI — entry point for the AlignCode annotation engine.

Workflow:
  1. Select a problem from the problem bank
  2. Generate Solution A (quick prompt) and Solution B (production prompt) via Ollama
  3. Run both solutions through the sandbox against predefined test cases
  4. Score each solution using the evaluation rubric (correctness, complexity, etc.)
  5. Flag bugs, annotate complexity, write a justification
  6. Optionally request an AI critique from the local model
  7. Save the completed preference pair to data/evaluations.jsonl
"""

import streamlit as st


def main():
    st.set_page_config(page_title="AlignCode — Annotation Engine", layout="wide")
    st.title("AlignCode — Annotation Engine")
    st.info("Annotation engine coming soon. See README for pipeline overview.")


if __name__ == "__main__":
    main()
