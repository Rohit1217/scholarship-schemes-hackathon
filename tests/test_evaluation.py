"""Tests for live response evaluation helpers."""

from __future__ import annotations

import pandas as pd

from scholarship.evaluation import evaluate_live_response, evaluation_markdown


def _chunks_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scheme_id": "SCH001",
                "scheme_name": "First Scholarship",
                "text": "Context 1",
                "score": 0.91,
                "rank": 1,
            },
            {
                "scheme_id": "SCH002",
                "scheme_name": "Second Scholarship",
                "text": "Context 2",
                "score": 0.78,
                "rank": 2,
            },
        ]
    )


def test_evaluate_live_response_scores_grounded_complete_answer():
    assistant_en = (
        "1. First Scholarship (SCH001): You qualify based on category. Award: Rs 10,000. "
        "Deadline: October 31 annually.\n"
        "2. Second Scholarship (SCH002): You qualify based on state. Award: Rs 5,000. "
        "Deadline: Rolling."
    )
    evaluation = evaluate_live_response(
        assistant_en=assistant_en,
        reply_markdown=assistant_en,
        lang="en",
        chunks_df=_chunks_df(),
        citations_markdown="- First Scholarship\n- Second Scholarship",
    )

    metrics = evaluation["metrics"]
    assert metrics["answer_item_count"] == 2
    assert metrics["grounded_scheme_precision"] == 1.0
    assert metrics["award_coverage_score"] == 1.0
    assert metrics["deadline_coverage_score"] == 1.0
    assert metrics["bhashabench_proxy_score"] > 0.95

    markdown = evaluation_markdown(evaluation)
    assert "BhashaBench-Style Live Evaluation" in markdown
    assert "Groundedness" in markdown


def test_evaluate_live_response_penalises_missing_bilingual_render():
    assistant_en = "1. First Scholarship (SCH001): Award: Rs 10,000. Deadline: October 31."
    evaluation = evaluate_live_response(
        assistant_en=assistant_en,
        reply_markdown=assistant_en,
        lang="hi",
        chunks_df=_chunks_df(),
        citations_markdown="- First Scholarship",
    )

    metrics = evaluation["metrics"]
    assert metrics["language_render_score"] == 0.0
    assert metrics["bhashabench_proxy_score"] < 1.0
