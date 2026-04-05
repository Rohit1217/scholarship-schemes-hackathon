"""Shared helpers for turning a saved profile into scholarship matches."""

from __future__ import annotations

from typing import Any

from scholarship.retriever import Retriever

REQUIRED_PROFILE_FIELDS = ("state", "category", "income", "gender", "age", "education")


def profile_to_query(
    state: str,
    category: str,
    income: float,
    gender: str,
    age: float,
    education: str,
    disability: bool,
    minority: bool,
) -> str:
    income_lakh = income / 100_000
    income_lakh_str = (
        f"{int(income_lakh)} lakh" if income_lakh == int(income_lakh)
        else f"{income_lakh:.2f} lakh"
    )
    parts = [
        f"I am a {int(age)}-year-old {gender} student from {state}.",
        f"My category is {category}.",
        f"My annual family income is \u20b9{income_lakh_str} (i.e. \u20b9{int(income):,}).",
        f"I am currently studying at {education} level.",
    ]
    if disability:
        parts.append("I have a disability.")
    if minority:
        parts.append("I belong to a minority community.")
    parts.append("Which scholarship schemes am I eligible for?")
    return " ".join(parts)


def saved_profile_complete(profile: dict[str, Any] | None) -> bool:
    profile = profile or {}
    return all(profile.get(field) not in ("", None) for field in REQUIRED_PROFILE_FIELDS)


def profile_query_from_saved_profile(profile: dict[str, Any]) -> str:
    return profile_to_query(
        str(profile.get("state") or ""),
        str(profile.get("category") or ""),
        float(profile.get("income") or 0),
        str(profile.get("gender") or ""),
        float(profile.get("age") or 0),
        str(profile.get("education") or ""),
        bool(profile.get("disability", False)),
        bool(profile.get("minority", False)),
    )


def matching_schemes_for_profile(
    profile: dict[str, Any],
    *,
    retriever: Retriever,
    k: int = 20,
) -> list[dict[str, str]]:
    query_en = profile_query_from_saved_profile(profile)
    chunks_df = retriever.search(query_en, k=k)

    matches: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, row in chunks_df.iterrows():
        scheme_id = str(row.get("scheme_id") or "").strip()
        scheme_name = str(row.get("scheme_name") or "").strip()
        key = scheme_id or scheme_name
        if not key or key in seen:
            continue
        seen.add(key)
        matches.append({
            "scheme_id": scheme_id,
            "scheme_name": scheme_name or scheme_id,
        })
    return matches
