"""Tests for safe MLflow logging wrapper."""

from __future__ import annotations

import os
from unittest.mock import patch

from scholarship.mlflow_tracking import log_search_experiment


class _RunContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeMlflow:
    def __init__(self):
        self.tracking_uri = ""
        self.experiment = ""
        self.run_name = ""
        self.params = {}
        self.tags = {}
        self.metrics = {}
        self.text_artifacts = {}
        self.dict_artifacts = {}

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, experiment):
        self.experiment = experiment

    def start_run(self, run_name=None):
        self.run_name = run_name or ""
        return _RunContext()

    def log_param(self, key, value):
        self.params[key] = value

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_metric(self, key, value):
        self.metrics[key] = value

    def log_text(self, payload, path):
        self.text_artifacts[path] = payload

    def log_dict(self, payload, path):
        self.dict_artifacts[path] = payload


def test_log_search_experiment_logs_params_metrics_and_artifacts():
    fake_mlflow = _FakeMlflow()

    with patch.dict(
        os.environ,
        {
            "ENABLE_MLFLOW_LOGGING": "1",
            "MLFLOW_TRACKING_URI": "databricks",
            "MLFLOW_EXPERIMENT_NAME": "/Shared/test-scholarship-app",
            "LLM_MODEL": "test-model",
            "VS_INDEX_NAME": "main.scholarships.scheme_vs_index",
        },
        clear=False,
    ), patch("scholarship.mlflow_tracking._get_mlflow_module", return_value=fake_mlflow):
        logged = log_search_experiment(
            login_id="student@example.com",
            lang="en",
            state="Maharashtra",
            category="OBC",
            education="Undergraduate",
            disability=False,
            minority=False,
            income=150000,
            query_en="Which scholarships am I eligible for?",
            answer_markdown="1. First Scholarship",
            citations_markdown="- First Scholarship",
            evaluation={"metrics": {"bhashabench_proxy_score": 0.88}},
            runtime_metrics={"retrieval_latency_ms": 12.0},
            retrieved_rows=[{"scheme_id": "SCH001", "scheme_name": "First Scholarship"}],
        )

    assert logged is True
    assert fake_mlflow.tracking_uri == "databricks"
    assert fake_mlflow.experiment == "/Shared/test-scholarship-app"
    assert fake_mlflow.params["lang"] == "en"
    assert fake_mlflow.params["login_id_hash"] != "student@example.com"
    assert fake_mlflow.metrics["bhashabench_proxy_score"] == 0.88
    assert fake_mlflow.metrics["retrieval_latency_ms"] == 12.0
    assert "artifacts/answer.md" in fake_mlflow.text_artifacts
    assert "artifacts/evaluation.json" in fake_mlflow.dict_artifacts


def test_log_search_experiment_returns_false_when_mlflow_missing():
    with patch.dict(os.environ, {"ENABLE_MLFLOW_LOGGING": "1"}, clear=False), \
         patch("scholarship.mlflow_tracking._get_mlflow_module", return_value=None):
        logged = log_search_experiment(
            login_id="student@example.com",
            lang="en",
            state="Maharashtra",
            category="OBC",
            education="Undergraduate",
            disability=False,
            minority=False,
            income=150000,
            query_en="Which scholarships am I eligible for?",
            answer_markdown="1. First Scholarship",
            citations_markdown="- First Scholarship",
            evaluation={"metrics": {"bhashabench_proxy_score": 0.88}},
            runtime_metrics={"retrieval_latency_ms": 12.0},
            retrieved_rows=[],
        )

    assert logged is False
