"""
evaluation_dashboard/agreement.py

Inter-annotator agreement dashboard.

Calculates Cohen's Kappa across multiple annotators who scored the same
problem/solution pairs. Helps measure annotation consistency and identify
criteria where human evaluators tend to disagree.

Usage:
    streamlit run agreement.py

Requires:
  - data/evaluations.jsonl with an "annotator_id" field in records
"""

import sys
from pathlib import Path
from collections import defaultdict

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation_engine.utils.storage import load_evaluations


def cohens_kappa(annotations_1: list[int], annotations_2: list[int]) -> float:
    """
    Calculate Cohen's Kappa for two lists of binary annotations.

    Args:
        annotations_1: List of 0/1 scores from annotator 1.
        annotations_2: List of 0/1 scores from annotator 2.

    Returns:
        Kappa score (-1 to 1). 1 = perfect agreement, 0 = chance, <0 = worse than chance.
    """
    if len(annotations_1) != len(annotations_2) or len(annotations_1) == 0:
        return 0.0

    n = len(annotations_1)
    # Observed agreement
    agree = sum(1 for a, b in zip(annotations_1, annotations_2) if a == b)
    p_o = agree / n

    # Expected agreement by chance
    p_yes_1 = sum(annotations_1) / n
    p_yes_2 = sum(annotations_2) / n
    p_e = (p_yes_1 * p_yes_2) + ((1 - p_yes_1) * (1 - p_yes_2))

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)


def interpret_kappa(kappa: float) -> str:
    """Return a human-readable interpretation of a Kappa score."""
    if kappa < 0:
        return "Poor (worse than chance)"
    elif kappa < 0.20:
        return "Slight agreement"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement"
    elif kappa < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def main():
    st.set_page_config(page_title="AlignCode — Inter-Annotator Agreement", layout="wide")
    st.title("AlignCode — Inter-Annotator Agreement")
    st.caption("Measure consistency across multiple annotators on the same problems.")

    records = load_evaluations()

    if not records:
        st.warning("No evaluations found. Run the annotation engine first.")
        return

    # Group records by problem statement
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_problem[r["problem"]].append(r)

    # Filter to problems with 2+ annotations
    multi_annotated = {k: v for k, v in by_problem.items() if len(v) >= 2}

    if not multi_annotated:
        st.info(
            "No problems have been annotated by multiple annotators yet. "
            "To use this tool, have two or more people annotate the same problems "
            "(each using a different Annotator ID in the sidebar)."
        )

        # Still show single-annotator stats
        st.subheader("Current Annotation Coverage")
        annotators = set()
        for r in records:
            annotators.add(r.get("annotator_id", "default"))

        st.metric("Total Evaluations", len(records))
        st.metric("Unique Annotators", len(annotators))
        st.metric("Unique Problems Annotated", len(by_problem))
        return

    st.success(f"Found {len(multi_annotated)} problem(s) with multiple annotations.")

    # Calculate per-criterion agreement
    criteria = [
        "correctness", "time_complexity", "space_complexity",
        "readability", "idiomatic_style", "edge_cases", "security"
    ]

    st.subheader("Per-Criterion Agreement (Cohen's Kappa)")

    results = []
    for criterion in criteria:
        all_a_scores = []
        all_b_scores = []

        for problem, evals in multi_annotated.items():
            # Compare first two annotators
            e1, e2 = evals[0], evals[1]
            scores_1 = e1.get("scores", {}).get(criterion, [0, 0])
            scores_2 = e2.get("scores", {}).get(criterion, [0, 0])

            # Use Solution B score for comparison (the "production" one)
            all_a_scores.append(scores_1[1] if len(scores_1) > 1 else 0)
            all_b_scores.append(scores_2[1] if len(scores_2) > 1 else 0)

        if all_a_scores and all_b_scores:
            kappa = cohens_kappa(all_a_scores, all_b_scores)
            results.append({
                "Criterion": criterion.replace("_", " ").title(),
                "Kappa": f"{kappa:.3f}",
                "Interpretation": interpret_kappa(kappa),
                "Pairs": len(all_a_scores),
            })

    if results:
        st.table(results)

    # Winner agreement
    st.subheader("Winner Agreement")
    winner_agree = 0
    winner_total = 0
    for problem, evals in multi_annotated.items():
        e1, e2 = evals[0], evals[1]
        winner_total += 1
        if e1.get("winner") == e2.get("winner"):
            winner_agree += 1

    if winner_total > 0:
        pct = winner_agree / winner_total * 100
        st.metric("Winner Agreement Rate", f"{pct:.0f}%", help=f"{winner_agree}/{winner_total} problems")

    # Disagreement details
    st.subheader("Disagreement Details")
    for problem, evals in multi_annotated.items():
        e1, e2 = evals[0], evals[1]
        if e1.get("winner") != e2.get("winner"):
            with st.expander(f"⚠️ {problem[:80]}..."):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Annotator:** {e1.get('annotator_id', 'unknown')}")
                    st.markdown(f"**Winner:** {e1.get('winner')}")
                    st.markdown(f"*{e1.get('justification', 'No justification')}*")
                with c2:
                    st.markdown(f"**Annotator:** {e2.get('annotator_id', 'unknown')}")
                    st.markdown(f"**Winner:** {e2.get('winner')}")
                    st.markdown(f"*{e2.get('justification', 'No justification')}*")


if __name__ == "__main__":
    main()
