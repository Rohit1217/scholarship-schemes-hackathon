"""Tests for DatabricksVSRetriever and profile query builder."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vs_response(n: int = 5) -> dict:
    """Build a fake VS REST API query response."""
    data = [
        [f"SCH00{i}", f"Test Scheme {i}", f"Scheme text {i} for SC students in Maharashtra", 0.9 - i * 0.05]
        for i in range(n)
    ]
    return {"result": {"data_array": data, "row_count": n}}


def _make_mock_post(response_dict: dict):
    """Return a mock for requests.post that returns response_dict as JSON."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = response_dict
    mock_post = MagicMock(return_value=mock_resp)
    return mock_post, mock_resp


# ---------------------------------------------------------------------------
# Tests — DatabricksVSRetriever
# ---------------------------------------------------------------------------

class TestDatabricksVSRetriever:
    def _env(self):
        return {"DATABRICKS_HOST": "https://test.cloud.databricks.com",
                "DATABRICKS_TOKEN": "test-token"}

    def test_search_returns_dataframe(self):
        """search() returns a non-empty DataFrame."""
        from scholarship.retriever import DatabricksVSRetriever
        mock_post, _ = _make_mock_post(_make_vs_response(5))
        with patch.dict(os.environ, self._env()), \
             patch("scholarship.retriever.requests.post", mock_post):
            r = DatabricksVSRetriever("ep", "main.test.idx")
            results = r.search("scholarship for SC student", k=5)
        assert isinstance(results, pd.DataFrame)
        assert not results.empty

    def test_required_columns_present(self):
        """Returned DataFrame must have scheme_id, scheme_name, text, score."""
        from scholarship.retriever import DatabricksVSRetriever
        mock_post, _ = _make_mock_post(_make_vs_response(5))
        with patch.dict(os.environ, self._env()), \
             patch("scholarship.retriever.requests.post", mock_post):
            r = DatabricksVSRetriever("ep", "main.test.idx")
            results = r.search("disability scholarship", k=3)
        for col in ("scheme_id", "scheme_name", "text", "score"):
            assert col in results.columns, f"Missing column: {col}"

    def test_returns_at_most_k_rows(self):
        """search() returns at most k rows."""
        from scholarship.retriever import DatabricksVSRetriever
        mock_post, _ = _make_mock_post(_make_vs_response(3))
        with patch.dict(os.environ, self._env()), \
             patch("scholarship.retriever.requests.post", mock_post):
            r = DatabricksVSRetriever("ep", "main.test.idx")
            results = r.search("OBC student", k=3)
        assert len(results) <= 3

    def test_rank_column_starts_at_1(self):
        """rank column starts at 1 and is sequential."""
        from scholarship.retriever import DatabricksVSRetriever
        mock_post, _ = _make_mock_post(_make_vs_response(4))
        with patch.dict(os.environ, self._env()), \
             patch("scholarship.retriever.requests.post", mock_post):
            r = DatabricksVSRetriever("ep", "main.test.idx")
            results = r.search("any query", k=4)
        assert results["rank"].tolist() == list(range(1, len(results) + 1))

    def test_query_text_sent_in_payload(self):
        """search() sends query_text in the POST body."""
        from scholarship.retriever import DatabricksVSRetriever
        mock_post, _ = _make_mock_post(_make_vs_response(3))
        with patch.dict(os.environ, self._env()), \
             patch("scholarship.retriever.requests.post", mock_post):
            r = DatabricksVSRetriever("ep", "main.test.idx")
            r.search("SC girl Maharashtra scholarship", k=3)
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1].get("json") or call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {}
        assert "SC girl Maharashtra scholarship" in str(call_kwargs)

    def test_empty_vs_response_returns_empty_df(self):
        """Empty VS response returns empty DataFrame (not an error)."""
        from scholarship.retriever import DatabricksVSRetriever
        mock_post, _ = _make_mock_post({"result": {"data_array": [], "row_count": 0}})
        with patch.dict(os.environ, self._env()), \
             patch("scholarship.retriever.requests.post", mock_post):
            r = DatabricksVSRetriever("ep", "main.test.idx")
            results = r.search("any query", k=5)
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0

    def test_uses_databricks_host_for_url(self):
        """search() constructs URL from DATABRICKS_HOST."""
        from scholarship.retriever import DatabricksVSRetriever
        mock_post, _ = _make_mock_post(_make_vs_response(1))
        env = {"DATABRICKS_HOST": "https://my-workspace.cloud.databricks.com",
               "DATABRICKS_TOKEN": "dapi123"}
        with patch.dict(os.environ, env), \
             patch("scholarship.retriever.requests.post", mock_post):
            r = DatabricksVSRetriever("ep", "main.s.idx")
            r.search("test", k=1)
        called_url = mock_post.call_args[0][0]
        assert "my-workspace.cloud.databricks.com" in called_url
        assert "vector-search/indexes" in called_url

    def test_missing_host_raises(self):
        """search() raises RuntimeError when DATABRICKS_HOST is not set."""
        from scholarship.retriever import DatabricksVSRetriever
        env = {k: v for k, v in os.environ.items()
               if k not in ("DATABRICKS_HOST", "DATABRICKS_TOKEN")}
        with patch.dict(os.environ, env, clear=True):
            r = DatabricksVSRetriever("ep", "main.s.idx")
            with pytest.raises(RuntimeError, match="DATABRICKS_HOST"):
                r.search("test", k=1)


# ---------------------------------------------------------------------------
# Tests — get_retriever() factory
# ---------------------------------------------------------------------------

class TestGetRetriever:
    def test_constructs_without_raising(self):
        """get_retriever() constructs without error (defers VS connection to search())."""
        from scholarship.retriever import get_retriever

        with patch.dict(os.environ, {
            "VS_ENDPOINT_NAME": "test-endpoint",
            "VS_INDEX_NAME": "main.test.index",
        }):
            r = get_retriever()
            assert r is not None

    def test_uses_env_vars(self):
        """get_retriever() reads VS_ENDPOINT_NAME and VS_INDEX_NAME from env."""
        from scholarship.retriever import DatabricksVSRetriever, get_retriever

        with patch.dict(os.environ, {
            "VS_ENDPOINT_NAME": "my-endpoint",
            "VS_INDEX_NAME": "main.scholarships.my_index",
        }):
            r = get_retriever()
            assert isinstance(r, DatabricksVSRetriever)
            assert r._endpoint_name == "my-endpoint"
            assert r._index_name == "main.scholarships.my_index"

    def test_uses_defaults_when_env_not_set(self):
        """get_retriever() falls back to default endpoint/index names."""
        from scholarship.retriever import (
            _DEFAULT_VS_ENDPOINT, _DEFAULT_VS_INDEX, get_retriever,
        )

        env = {k: v for k, v in os.environ.items()
               if k not in ("VS_ENDPOINT_NAME", "VS_INDEX_NAME")}
        with patch.dict(os.environ, env, clear=True):
            r = get_retriever()
            assert r._endpoint_name == _DEFAULT_VS_ENDPOINT
            assert r._index_name == _DEFAULT_VS_INDEX


# ---------------------------------------------------------------------------
# Tests — profile_to_query()
# ---------------------------------------------------------------------------

class TestProfileToQuery:
    def test_produces_non_empty_string(self):
        from app.main import profile_to_query

        q = profile_to_query(
            state="Maharashtra", category="SC", income=80000,
            gender="Female", age=19, education="Class 12",
            disability=False, minority=False,
        )
        assert isinstance(q, str)
        assert len(q) > 0

    def test_includes_key_fields(self):
        from app.main import profile_to_query

        q = profile_to_query(
            state="Tamil Nadu", category="OBC", income=150000,
            gender="Male", age=22, education="Undergraduate",
            disability=True, minority=True,
        )
        assert "Tamil Nadu" in q
        assert "OBC" in q
        assert "disability" in q.lower()
        assert "minority" in q.lower()
        assert "Undergraduate" in q

    def test_no_disability_minority_when_false(self):
        from app.main import profile_to_query

        q = profile_to_query(
            state="Delhi", category="General", income=300000,
            gender="Other", age=25, education="Postgraduate",
            disability=False, minority=False,
        )
        assert "disability" not in q.lower()
        assert "minority" not in q.lower()

    def test_ends_with_question_mark(self):
        from app.main import profile_to_query

        q = profile_to_query(
            state="Gujarat", category="EWS", income=50000,
            gender="Female", age=16, education="Class 10",
            disability=False, minority=False,
        )
        assert q.strip().endswith("?")

    def test_income_in_lakh_format(self):
        from app.main import profile_to_query

        q = profile_to_query(
            state="Bihar", category="ST", income=200000,
            gender="Male", age=18, education="Class 12",
            disability=False, minority=False,
        )
        assert "lakh" in q.lower()
