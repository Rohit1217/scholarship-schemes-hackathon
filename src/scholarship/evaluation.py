"""Live response evaluation helpers for scholarship RAG answers.

The real BhashaBench datasets are offline, labelled benchmarks. This module
provides a BhashaBench-style live proxy scorecard for app results together with
quantitative grounding / completeness metrics that can be logged to MLflow.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

_NO_MATCH_PATTERNS = (
    "no matching scholarship schemes found",
    "no matching schemes found",
    "no eligible scholarship schemes found",
)
_DEADLINE_PATTERN = re.compile(
    r"\b(deadline|last date|apply by|closing|rolling|january|february|march|april|may|june|"
    r"july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_AWARD_PATTERN = re.compile(
    r"(\u20b9|\baward\b|\bstipend\b|\btuition\b|\bgrant\b|\bscholarship amount\b)",
    re.IGNORECASE,
)
_LIST_ITEM_PATTERN = re.compile(r"(?m)^\s*(?:\d+\.|[-*])\s+")
_NON_WORD_PATTERN = re.compile(r"[^a-z0-9]+")


def _normalise_key(value: str) -> str:
    return _NON_WORD_PATTERN.sub("", value.strip().lower())


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _list_items(text: str) -> list[str]:
    matches = list(_LIST_ITEM_PATTERN.finditer(text))
    if not matches:
        return []

    items: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        item = text[start:end].strip()
        if item:
            items.append(item)
    return items


def _mentions_scheme(answer: str, scheme_id: str, scheme_name: str) -> bool:
    answer_norm = _normalise_key(answer)
    for candidate in (scheme_id, scheme_name):
        norm = _normalise_key(str(candidate))
        if norm and norm in answer_norm:
            return True
    return False


def _retrieved_scheme_rows(chunks_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in chunks_df.iterrows():
        scheme_id = str(row.get("scheme_id") or "").strip()
        scheme_name = str(row.get("scheme_name") or "").strip()
        key = scheme_id or scheme_name
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append({
            "scheme_id": scheme_id,
            "scheme_name": scheme_name or scheme_id,
            "score": _safe_float(row.get("score")),
        })
    return rows


def evaluate_live_response(
    *,
    assistant_en: str,
    reply_markdown: str,
    lang: str,
    chunks_df: pd.DataFrame,
    citations_markdown: str,
) -> dict[str, Any]:
    retrieved_rows = _retrieved_scheme_rows(chunks_df)
    item_texts = _list_items(assistant_en)
    no_match = any(pattern in assistant_en.lower() for pattern in _NO_MATCH_PATTERNS)

    mentioned_scheme_count = 0
    for row in retrieved_rows:
        if _mentions_scheme(assistant_en, row["scheme_id"], row["scheme_name"]):
            mentioned_scheme_count += 1

    answer_item_count = len(item_texts)
    award_hits = sum(1 for item in item_texts if _AWARD_PATTERN.search(item))
    deadline_hits = sum(1 for item in item_texts if _DEADLINE_PATTERN.search(item))

    if no_match:
        grounded_scheme_precision = 1.0
        award_coverage_score = 1.0
        deadline_coverage_score = 1.0
        format_compliance_score = 1.0
    else:
        grounded_scheme_precision = mentioned_scheme_count / max(answer_item_count, 1)
        award_coverage_score = award_hits / max(answer_item_count, 1)
        deadline_coverage_score = deadline_hits / max(answer_item_count, 1)
        format_compliance_score = (
            0.5 * float(answer_item_count > 0)
            + 0.25 * float(award_hits > 0)
            + 0.25 * float(deadline_hits > 0)
        )

    completeness_score = (award_coverage_score + deadline_coverage_score) / 2
    bilingual_expected = lang != "en"
    language_render_score = 1.0
    if bilingual_expected:
        language_render_score = float("**English:**" in reply_markdown and reply_markdown.startswith("**"))

    retrieval_scores = [_safe_float(row["score"]) for row in retrieved_rows]
    retrieval_score_top1 = retrieval_scores[0] if retrieval_scores else 0.0
    retrieval_score_mean = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
    citation_line_count = len([line for line in citations_markdown.splitlines() if line.strip().startswith("- ")])
    retrieval_coverage_score = mentioned_scheme_count / max(len(retrieved_rows), 1) if retrieved_rows else 0.0
    bhashabench_proxy_score = (
        0.40 * grounded_scheme_precision
        + 0.25 * completeness_score
        + 0.20 * format_compliance_score
        + 0.15 * language_render_score
    )

    metrics = {
        "answer_item_count": float(answer_item_count),
        "retrieved_unique_scheme_count": float(len(retrieved_rows)),
        "mentioned_retrieved_scheme_count": float(mentioned_scheme_count),
        "grounded_scheme_precision": grounded_scheme_precision,
        "retrieval_coverage_score": retrieval_coverage_score,
        "award_coverage_score": award_coverage_score,
        "deadline_coverage_score": deadline_coverage_score,
        "format_compliance_score": format_compliance_score,
        "language_render_score": language_render_score,
        "retrieval_score_top1": retrieval_score_top1,
        "retrieval_score_mean": retrieval_score_mean,
        "citation_line_count": float(citation_line_count),
        "bhashabench_proxy_score": bhashabench_proxy_score,
        "bhashabench_groundedness_score": grounded_scheme_precision,
        "bhashabench_completeness_score": completeness_score,
        "bhashabench_format_score": format_compliance_score,
        "bhashabench_language_score": language_render_score,
        "no_match_response": float(no_match),
    }
    return {
        "metrics": metrics,
        "retrieved_rows": retrieved_rows,
        "item_texts": item_texts,
        "no_match": no_match,
    }


def evaluation_markdown(evaluation: dict[str, Any]) -> str:
    metrics = evaluation["metrics"]
    return (
        "### BhashaBench-Style Live Evaluation\n\n"
        "_Proxy scorecard for the current answer. This is not the official offline BhashaBench benchmark._\n\n"
        f"- Overall proxy score: `{metrics['bhashabench_proxy_score']:.2f}`\n"
        f"- Groundedness: `{metrics['bhashabench_groundedness_score']:.2f}`\n"
        f"- Completeness: `{metrics['bhashabench_completeness_score']:.2f}`\n"
        f"- Format: `{metrics['bhashabench_format_score']:.2f}`\n"
        f"- Language rendering: `{metrics['bhashabench_language_score']:.2f}`\n\n"
        "### Quantitative Metrics\n\n"
        f"- Retrieved unique schemes: `{int(metrics['retrieved_unique_scheme_count'])}`\n"
        f"- Answer list items: `{int(metrics['answer_item_count'])}`\n"
        f"- Grounded scheme precision: `{metrics['grounded_scheme_precision']:.2f}`\n"
        f"- Retrieval coverage: `{metrics['retrieval_coverage_score']:.2f}`\n"
        f"- Award coverage: `{metrics['award_coverage_score']:.2f}`\n"
        f"- Deadline coverage: `{metrics['deadline_coverage_score']:.2f}`\n"
        f"- Top retrieval score: `{metrics['retrieval_score_top1']:.2f}`\n"
        f"- Mean retrieval score: `{metrics['retrieval_score_mean']:.2f}`"
    )
