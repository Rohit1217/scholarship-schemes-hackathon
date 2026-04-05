"""OpenAI-compatible chat completions for Databricks Playground / serving endpoints.

Typical env (from Playground **Get code** — do not commit secrets).

**AI Gateway** (OpenAI SDK ``base_url`` often ends with ``/mlflow/v1``)::

    LLM_OPENAI_BASE_URL=https://<workspace-id>.ai-gateway.cloud.databricks.com/mlflow/v1
    LLM_MODEL=databricks-meta-llama-3-1-70b-instruct
    DATABRICKS_TOKEN=dapi...

We POST to ``{LLM_OPENAI_BASE_URL}/chat/completions`` (same as the OpenAI client).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a scholarship eligibility assistant for India. "
    "Given a user's profile and retrieved scholarship scheme information, "
    "output ONLY the schemes the user is fully eligible for. "
    "Do NOT mention, list, or explain schemes the user is not eligible for — skip them entirely. "
    "If no schemes match, say: 'No matching scholarship schemes found for your profile.' "
    "For each eligible scheme output: scheme name, one-line reason they qualify, "
    "award amount, and application deadline. Format as a numbered list. "
    "Be factual and concise. Do not invent criteria not present in the context. "
    "INCOME RULE: 1 lakh = 100,000. Compare income numerically. "
    "If the user's income exceeds the scheme's maximum limit, skip that scheme. "
    "CATEGORY RULE: SC/ST/OBC/EWS schemes are NOT open to General category students "
    "unless the scheme text explicitly says so. "
    "ALL criteria (category, income, state, education, gender, disability, minority) "
    "must be satisfied — if even one criterion fails, skip that scheme entirely."
)


def _chat_url() -> str:
    full = os.environ.get("LLM_CHAT_COMPLETIONS_URL", "").strip()
    if full:
        return full
    base = os.environ.get("LLM_OPENAI_BASE_URL", "").strip().rstrip("/")
    if not base:
        raise RuntimeError(
            "Set LLM_CHAT_COMPLETIONS_URL (full POST URL) or LLM_OPENAI_BASE_URL "
            "(e.g. AI Gateway …/mlflow/v1 from Playground Get code)."
        )
    if base.endswith("/chat/completions"):
        return base
    # OpenAI SDK + AI Gateway Get code: base_url ends with …/mlflow/v1 (also ends with /v1).
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _bearer() -> str:
    token = (
        os.environ.get("DATABRICKS_TOKEN", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if token:
        return token
    return _sdk_oauth_token()


def _extract_bearer(obj) -> str:
    """Extract Bearer token from a dict-like object."""
    if isinstance(obj, dict):
        auth = obj.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
    return ""


def _sdk_oauth_token() -> str:
    """Get a fresh OAuth token from the Databricks SDK (service principal auth).

    On Databricks Apps, the service principal uses OAuth M2M — there is no
    static PAT.  ``WorkspaceClient().config.token`` is ``None`` in that case.
    """
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        logger.debug("SDK auth_type=%s, host=%s", w.config.auth_type, w.config.host)
        # Try static token first (PAT auth).
        if w.config.token:
            return w.config.token
        # OAuth M2M: config.authenticate() behaviour varies by SDK version.
        result = w.config.authenticate()
        # Pattern 1: authenticate() returns a dict of headers directly.
        token = _extract_bearer(result)
        if token:
            return token
        # Pattern 2: authenticate() returns a callable header factory.
        if callable(result):
            token = _extract_bearer(result())
            if token:
                return token
        logger.warning("Could not extract Bearer token from SDK (auth_type=%s)", w.config.auth_type)
    except Exception as exc:
        logger.warning("SDK OAuth token failed: %s", exc)
    return ""


def chat_completions(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """POST OpenAI-compatible chat/completions; returns parsed JSON body."""
    url = _chat_url()
    token = _bearer()
    if not token:
        raise RuntimeError(
            "Set DATABRICKS_TOKEN, LLM_API_KEY, or OPENAI_API_KEY for LLM calls."
        )
    model = (model or os.environ.get("LLM_MODEL", "")).strip()
    if not model:
        raise RuntimeError("Set LLM_MODEL (or pass model=).")
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def extract_assistant_text(response: dict[str, Any]) -> str:
    try:
        return response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected LLM response shape: {response!r}") from e


def rag_user_message(context_chunks: list[str], question: str) -> str:
    """Single user message with retrieved context (Playground-style)."""
    ctx = "\n\n".join(c.strip() for c in context_chunks if c and str(c).strip())
    return f"Context:\n{ctx}\n\nQuestion: {question}"
