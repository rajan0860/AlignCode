"""
annotation_engine/app.py

Streamlit annotation UI — entry point for the AlignCode annotation engine.
"""

import datetime
import uuid
import streamlit as st

from problems.problem_bank import PROBLEMS
from services.ollama_service import generate_solution, get_ai_critique, refine_solution
from utils.sandbox import run_tests
from utils.storage import save_evaluation

# Constants
CRITERIA = ["Correctness", "Time Complexity", "Space Complexity", "Readability", "Idiomatic Style", "Edge Cases", "Security"]
BUGS = [
    "Off-by-one errors", "Unhandled None/null", "Mutable default args", 
    "Incorrect return type", "Shadowed variables", "Memory leak", 
    "Hardcoded credentials", "SQL injection", "Insecure eval", 
    "Goroutine leak", "Missing defer", "Ignored error", "Race condition", "Deadlock"
]
COMPLEXITIES = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(2^n)"]


def init_session_state():
    if "solution_a" not in st.session_state:
        st.session_state.solution_a = None
        st.session_state.solution_b = None
        st.session_state.test_results_a = None
        st.session_state.test_results_b = None
        st.session_state.ai_critique = None
        st.session_state.generation_complete = False
        st.session_state.turns = []


def clear_session_state():
    st.session_state.solution_a = None
    st.session_state.solution_b = None
    st.session_state.test_results_a = None
    st.session_state.test_results_b = None
    st.session_state.ai_critique = None
    st.session_state.generation_complete = False
    st.session_state.turns = []


def main():
    st.set_page_config(page_title="AlignCode — Annotation Engine", layout="wide")
    st.title("AlignCode — Annotation Engine")
    init_session_state()

    # Sidebar: Annotator & Problem Selection
    st.sidebar.header("Annotator")
    annotator_id = st.sidebar.text_input("Your Name / ID", value="default")
    st.sidebar.divider()
    st.sidebar.header("Problem Selection")
    languages = sorted(list(set(p["language"] for p in PROBLEMS)))
    selected_lang = st.sidebar.selectbox("Language", languages, on_change=clear_session_state)
    
    difficulties = sorted(list(set(p["difficulty"] for p in PROBLEMS if p["language"] == selected_lang)))
    selected_diff = st.sidebar.selectbox("Difficulty", difficulties, on_change=clear_session_state)
    
    available_problems = [p for p in PROBLEMS if p["language"] == selected_lang and p["difficulty"] == selected_diff]
    selected_problem_title = st.sidebar.selectbox("Problem", [p["title"] for p in available_problems], on_change=clear_session_state)
    
    problem = next((p for p in available_problems if p["title"] == selected_problem_title), None)
    
    if not problem:
        st.warning("No problems found for this combination.")
        return

    st.markdown(f"**Problem Statement:**\n\n{problem['statement']}")

    if not st.session_state.generation_complete:
        if st.button("Generate Solutions"):
            with st.spinner("Generating Solution A (Quick)..."):
                st.session_state.solution_a = generate_solution(problem["statement"], "quick")
            with st.spinner("Generating Solution B (Production)..."):
                st.session_state.solution_b = generate_solution(problem["statement"], "production")
            
            with st.spinner("Running Tests..."):
                st.session_state.test_results_a = run_tests(st.session_state.solution_a, problem["test_cases"], language=problem["language"])
                st.session_state.test_results_b = run_tests(st.session_state.solution_b, problem["test_cases"], language=problem["language"])
            
            st.session_state.generation_complete = True

    if st.session_state.generation_complete:
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Solution A (Quick)")
            st.code(st.session_state.solution_a, language=problem["language"])
            if st.session_state.test_results_a:
                passed_a = st.session_state.test_results_a.passed
                total_a = st.session_state.test_results_a.total
                st.metric("Tests Passed", f"{passed_a}/{total_a}")
                
        with col2:
            st.subheader("Solution B (Production)")
            st.code(st.session_state.solution_b, language=problem["language"])
            if st.session_state.test_results_b:
                passed_b = st.session_state.test_results_b.passed
                total_b = st.session_state.test_results_b.total
                st.metric("Tests Passed", f"{passed_b}/{total_b}")

        st.divider()
        st.header("Evaluation Rubric")
        
        # Scoring Forms
        with st.form("evaluation_form"):
            st.subheader("Criteria Scores (0 = Fail, 1 = Pass)")
            scores = {}
            for criterion in CRITERIA:
                c1, c2 = st.columns(2)
                key = criterion.lower().replace(" ", "_")
                val_a = c1.radio(f"{criterion} (A)", [0, 1], key=f"{key}_a", horizontal=True)
                val_b = c2.radio(f"{criterion} (B)", [0, 1], key=f"{key}_b", horizontal=True)
                scores[key] = [val_a, val_b]
            
            st.subheader("Bugs Found")
            c1, c2 = st.columns(2)
            bugs_a = c1.multiselect("Bugs in Solution A", BUGS)
            bugs_b = c2.multiselect("Bugs in Solution B", BUGS)
            
            st.subheader("Complexity")
            c1, c2, c3 = st.columns(3)
            time_a = c1.selectbox("Time Complexity (A)", COMPLEXITIES, index=2)
            time_b = c1.selectbox("Time Complexity (B)", COMPLEXITIES, index=2)
            space_a = c2.selectbox("Space Complexity (A)", COMPLEXITIES, index=2)
            space_b = c2.selectbox("Space Complexity (B)", COMPLEXITIES, index=2)
            bottleneck_a = c3.text_input("Bottleneck (A)", "none")
            bottleneck_b = c3.text_input("Bottleneck (B)", "none")
            
            st.subheader("Final Decision")
            winner = st.radio("Which solution is better overall?", ["A", "B"], horizontal=True)
            justification = st.text_area("Justification (minimum 80 chars)", height=100)
            
            submitted = st.form_submit_button("Save Evaluation")
            
            if submitted:
                if len(justification) < 80:
                    st.error("Justification must be at least 80 characters.")
                else:
                    record = {
                        "id": str(uuid.uuid4()),
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                        "annotator_id": annotator_id,
                        "problem": problem["statement"],
                        "language": problem["language"],
                        "model": "qwen2.5-coder:7b",
                        "solution_a": st.session_state.solution_a,
                        "solution_b": st.session_state.solution_b,
                        "test_results": {
                            "solution_a": {
                                "passed": st.session_state.test_results_a.passed if st.session_state.test_results_a else 0,
                                "failed": st.session_state.test_results_a.failed if st.session_state.test_results_a else 0,
                                "total": st.session_state.test_results_a.total if st.session_state.test_results_a else 0
                            },
                            "solution_b": {
                                "passed": st.session_state.test_results_b.passed if st.session_state.test_results_b else 0,
                                "failed": st.session_state.test_results_b.failed if st.session_state.test_results_b else 0,
                                "total": st.session_state.test_results_b.total if st.session_state.test_results_b else 0
                            }
                        },
                        "scores": scores,
                        "bugs_found": {
                            "solution_a": bugs_a,
                            "solution_b": bugs_b
                        },
                        "complexity": {
                            "solution_a": {"time": time_a, "space": space_a, "bottleneck": bottleneck_a},
                            "solution_b": {"time": time_b, "space": space_b, "bottleneck": bottleneck_b}
                        },
                        "winner": winner,
                        "justification": justification,
                        "ai_critique": st.session_state.ai_critique or "",
                        "turns": st.session_state.turns,
                    }
                    save_evaluation(record)
                    st.success("Evaluation saved successfully!")
                    
        # AI Critique
        if st.button("Show AI Critique"):
            with st.spinner("Generating AI Critique..."):
                st.session_state.ai_critique = get_ai_critique(
                    problem["statement"],
                    st.session_state.solution_a,
                    st.session_state.solution_b
                )
        if st.session_state.ai_critique:
            st.info(st.session_state.ai_critique)

        # Multi-Turn Refinement
        st.divider()
        st.header("Refine Solution (Multi-Turn)")
        st.caption("Provide feedback to have the model improve one of its solutions.")

        refine_target = st.radio("Which solution to refine?", ["A", "B"], horizontal=True, key="refine_target")
        feedback_text = st.text_area("Your feedback (e.g., 'Handle the None edge case')", key="feedback_text")

        if st.button("Generate Refinement"):
            if not feedback_text.strip():
                st.error("Please provide feedback before requesting a refinement.")
            else:
                original = st.session_state.solution_a if refine_target == "A" else st.session_state.solution_b
                with st.spinner(f"Refining Solution {refine_target}..."):
                    refined_code = refine_solution(
                        problem["statement"],
                        original,
                        feedback_text,
                    )
                refined_tests = run_tests(refined_code, problem["test_cases"], language=problem["language"])

                turn_record = {
                    "turn": len(st.session_state.turns) + 1,
                    "target": refine_target,
                    "feedback": feedback_text,
                    "refined_code": refined_code,
                    "refined_test_results": {
                        "passed": refined_tests.passed,
                        "failed": refined_tests.failed,
                        "total": refined_tests.total,
                    },
                }
                st.session_state.turns.append(turn_record)

                st.subheader(f"Refined Solution {refine_target} (Turn {turn_record['turn']})")
                st.code(refined_code, language=problem["language"])
                st.metric("Tests Passed", f"{refined_tests.passed}/{refined_tests.total}")

        # Show refinement history
        if st.session_state.turns:
            with st.expander(f"Refinement History ({len(st.session_state.turns)} turn(s))"):
                for t in st.session_state.turns:
                    st.markdown(f"**Turn {t['turn']}** — Refined Solution {t['target']}")
                    st.markdown(f"*Feedback:* {t['feedback']}")
                    st.code(t["refined_code"], language=problem["language"])
                    tr = t["refined_test_results"]
                    st.markdown(f"Tests: {tr['passed']}/{tr['total']} passed")
                    st.divider()


if __name__ == "__main__":
    main()
