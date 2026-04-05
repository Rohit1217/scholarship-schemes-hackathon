"""Safe MLflow experiment logging for live scholarship searches."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_EXPERIMENT_NAME = "/Shared/scholarship-finder-app"


def _get_mlflow_module():
    try:
        return importlib.import_module("mlflow")
    except Exception:
        return None


def mlflow_enabled() -> bool:
    value = os.environ.get("ENABLE_MLFLOW_LOGGING", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _configure_mlflow(mlflow_module) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", _DEFAULT_EXPERIMENT_NAME).strip()
    if tracking_uri:
        mlflow_module.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow_module.set_experiment(experiment_name)


def _artifact_path(name: str, suffix: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    fd, temp_name = tempfile.mkstemp(prefix=f"{safe_name}-", suffix=suffix)
    os.close(fd)
    return Path(temp_name)


def _hash_identifier(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _safe_metric_value(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _log_json_artifact(mlflow_module, path: str, payload: Any) -> None:
    if hasattr(mlflow_module, "log_dict"):
        mlflow_module.log_dict(payload, path)
        return

    temp_path = _artifact_path(path.replace("/", "_"), ".json")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        mlflow_module.log_artifact(str(temp_path), artifact_path=str(Path(path).parent))
    finally:
        temp_path.unlink(missing_ok=True)


def _log_text_artifact(mlflow_module, path: str, payload: str) -> None:
    if hasattr(mlflow_module, "log_text"):
        mlflow_module.log_text(payload, path)
        return

    temp_path = _artifact_path(path.replace("/", "_"), ".txt")
    try:
        temp_path.write_text(payload, encoding="utf-8")
        mlflow_module.log_artifact(str(temp_path), artifact_path=str(Path(path).parent))
    finally:
        temp_path.unlink(missing_ok=True)


def log_search_experiment(
    *,
    login_id: str,
    lang: str,
    state: str,
    category: str,
    education: str,
    disability: bool,
    minority: bool,
    income: float,
    query_en: str,
    answer_markdown: str,
    citations_markdown: str,
    evaluation: dict[str, Any],
    runtime_metrics: dict[str, float],
    retrieved_rows: list[dict[str, Any]],
) -> bool:
    if not mlflow_enabled():
        return False

    mlflow_module = _get_mlflow_module()
    if mlflow_module is None:
        logger.info("MLflow is not installed; skipping experiment logging.")
        return False

    try:
        _configure_mlflow(mlflow_module)
        income_bucket_lakh = int(float(income) // 100_000) if income is not None else -1
        params = {
            "lang": lang,
            "state": state,
            "category": category,
            "education": education,
            "disability": str(bool(disability)).lower(),
            "minority": str(bool(minority)).lower(),
            "income_bucket_lakh": str(income_bucket_lakh),
            "vs_index_name": os.environ.get("VS_INDEX_NAME", "main.scholarships.scheme_vs_index").strip(),
            "llm_model": os.environ.get("LLM_MODEL", "").strip(),
            "login_id_hash": _hash_identifier(login_id),
        }
        tags = {
            "app": "scholarship-finder",
            "feature": "live-search-evaluation",
            "evaluation_mode": "bhashabench_style_proxy",
        }
        run_name = f"scholarship-search-{lang}"

        with mlflow_module.start_run(run_name=run_name):
            for key, value in params.items():
                if value != "":
                    mlflow_module.log_param(key, value)
            for key, value in tags.items():
                mlflow_module.set_tag(key, value)

            merged_metrics = dict(evaluation["metrics"])
            merged_metrics.update(runtime_metrics)
            for key, value in merged_metrics.items():
                numeric = _safe_metric_value(value)
                if numeric is not None:
                    mlflow_module.log_metric(key, numeric)

            _log_text_artifact(mlflow_module, "artifacts/query_en.txt", query_en)
            _log_text_artifact(mlflow_module, "artifacts/answer.md", answer_markdown)
            _log_text_artifact(mlflow_module, "artifacts/citations.md", citations_markdown)
            _log_json_artifact(mlflow_module, "artifacts/retrieved_rows.json", retrieved_rows)
            _log_json_artifact(mlflow_module, "artifacts/evaluation.json", evaluation)
        return True
    except Exception as exc:
        logger.warning("MLflow experiment logging failed: %s", exc)
        return False
