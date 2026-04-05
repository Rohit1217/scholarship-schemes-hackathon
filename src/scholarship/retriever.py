"""Databricks Vector Search retriever for scholarship scheme RAG.

Queries the VS REST API directly — same auth path as the LLM client
(DATABRICKS_TOKEN locally, OAuth M2M on Databricks Apps). No separate
VectorSearchClient SDK auth needed.

Usage::

    from scholarship.retriever import get_retriever

    retriever = get_retriever()
    results_df = retriever.search("SC student from Maharashtra with income below 2 lakh", k=7)
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_DEFAULT_VS_ENDPOINT = "scholarship-vs-endpoint"
_DEFAULT_VS_INDEX    = "main.scholarships.scheme_vs_index"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Retriever(Protocol):
    """Uniform search interface for RAG retrieval backends."""

    def search(self, query: str, k: int = 7) -> pd.DataFrame:
        """Return top-k chunks as a DataFrame.

        Expected columns: scheme_id, scheme_name, text, score.
        """
        ...


# ---------------------------------------------------------------------------
# Auth helper (shared with llm_client)
# ---------------------------------------------------------------------------

def _get_bearer() -> str:
    """Resolve a Databricks bearer token.

    Priority: DATABRICKS_TOKEN env var → LLM_API_KEY env var → SDK OAuth M2M.
    Works for both local PAT auth and Databricks Apps service-principal auth.
    """
    token = (
        os.environ.get("DATABRICKS_TOKEN", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
    )
    if token:
        return token
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        if w.config.token:
            return w.config.token
        result = w.config.authenticate()
        if callable(result):
            result = result()
        if isinstance(result, dict):
            auth = result.get("Authorization", "")
            if auth.startswith("Bearer "):
                return auth[7:]
    except Exception as exc:
        logger.warning("Could not obtain bearer token: %s", exc)
    return ""


# ---------------------------------------------------------------------------
# Databricks Vector Search backend (REST API)
# ---------------------------------------------------------------------------

class DatabricksVSRetriever:
    """Queries a Databricks Vector Search index via the REST API.

    Uses the same auth path as the LLM client — no separate SDK credentials
    needed. Works on Databricks Apps (OAuth M2M) and locally (PAT).
    """

    def __init__(self, endpoint_name: str, index_name: str) -> None:
        self._endpoint_name = endpoint_name
        self._index_name    = index_name

    def search(self, query: str, k: int = 7) -> pd.DataFrame:
        host  = os.environ.get("DATABRICKS_HOST", "").strip().rstrip("/")
        if not host:
            raise RuntimeError("DATABRICKS_HOST is not set.")
        if not host.startswith("http"):
            host = f"https://{host}"

        token = _get_bearer()
        if not token:
            raise RuntimeError(
                "No Databricks bearer token available. "
                "Set DATABRICKS_TOKEN or ensure OAuth M2M is configured."
            )

        url = f"{host}/api/2.0/vector-search/indexes/{self._index_name}/query"
        payload = {
            "query_text": query.strip(),
            "columns": ["scheme_id", "scheme_name", "text"],
            "num_results": k,
        }
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        r.raise_for_status()

        data = r.json().get("result", {}).get("data_array", [])
        rows = []
        for rank, row in enumerate(data):
            rows.append({
                "scheme_id":   row[0] if len(row) > 0 else "",
                "scheme_name": row[1] if len(row) > 1 else "",
                "text":        row[2] if len(row) > 2 else "",
                "score":       float(row[3]) if len(row) > 3 else 0.0,
                "rank":        rank + 1,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_retriever() -> DatabricksVSRetriever:
    """Instantiate a DatabricksVSRetriever from env vars.

    Env vars (with defaults):
      VS_ENDPOINT_NAME  — name of the Vector Search endpoint
      VS_INDEX_NAME     — fully-qualified index name (catalog.schema.index)
    """
    endpoint = os.environ.get("VS_ENDPOINT_NAME", _DEFAULT_VS_ENDPOINT).strip()
    index    = os.environ.get("VS_INDEX_NAME",    _DEFAULT_VS_INDEX).strip()
    logger.info("DatabricksVSRetriever: endpoint=%s  index=%s", endpoint, index)
    return DatabricksVSRetriever(endpoint_name=endpoint, index_name=index)
