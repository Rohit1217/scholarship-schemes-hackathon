"""Tests for notification refresh + email orchestration."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

import pandas as pd
import pytest

from scholarship.notification_service import process_user_email_notifications
from scholarship.user_store import UserStore


@contextmanager
def _store_path():
    path = Path.cwd() / f"notification-store-{uuid4().hex}.json"
    try:
        yield path
    finally:
        if path.exists():
            path.unlink()


class _SequencedRetriever:
    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0

    def search(self, query: str, k: int = 7):
        if self._index >= len(self._responses):
            frame = self._responses[-1]
        else:
            frame = self._responses[self._index]
        self._index += 1
        return frame


def _profile_payload() -> dict[str, object]:
    return {
        "state": "Maharashtra",
        "category": "OBC",
        "income": 150000,
        "gender": "Female",
        "age": 19,
        "education": "Undergraduate",
        "disability": False,
        "minority": False,
    }


def _matches_frame(*rows: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scheme_id": scheme_id,
                "scheme_name": scheme_name,
                "text": f"{scheme_name} details",
                "score": 0.9,
            }
            for scheme_id, scheme_name in rows
        ]
    )


def test_process_user_email_notifications_sends_and_marks_alerts():
    with _store_path() as store_path:
        store = UserStore(store_path)
        store.register_user(
            "same-user",
            "secret123",
            full_name="Asha",
            email="student@example.com",
        )
        store.save_profile("same-user", _profile_payload())
        retriever = _SequencedRetriever([
            _matches_frame(("SCH001", "First Scholarship")),
            _matches_frame(("SCH001", "First Scholarship"), ("SCH002", "Second Scholarship")),
        ])
        sent = []

        def _email_sender(**kwargs):
            sent.append(kwargs)
            return {"message_id": "abc123"}

        with patch("scholarship.notification_service.email_configured", return_value=True):
            first = process_user_email_notifications(
                "same-user",
                user_store=store,
                retriever=retriever,
                email_sender=_email_sender,
            )
            second = process_user_email_notifications(
                "same-user",
                user_store=store,
                retriever=retriever,
                email_sender=_email_sender,
            )

        assert first["refresh"]["baseline_reset"] is True
        assert first["email"]["sent"] is False
        assert second["refresh"]["new_count"] == 1
        assert second["email"]["sent"] is True
        assert second["email"]["sent_count"] == 1
        assert len(sent) == 1
        assert sent[0]["to_email"] == "student@example.com"
        assert second["user"]["pending_email_notifications"] == 0
        assert second["user"]["notifications"][0]["email_sent_at"] != ""


def test_process_user_email_notifications_keeps_pending_alerts_on_send_failure():
    with _store_path() as store_path:
        store = UserStore(store_path)
        store.register_user(
            "same-user",
            "secret123",
            full_name="Asha",
            email="student@example.com",
        )
        store.save_profile("same-user", _profile_payload())
        retriever = _SequencedRetriever([
            _matches_frame(("SCH001", "First Scholarship")),
            _matches_frame(("SCH001", "First Scholarship"), ("SCH002", "Second Scholarship")),
        ])

        with patch("scholarship.notification_service.email_configured", return_value=True):
            process_user_email_notifications(
                "same-user",
                user_store=store,
                retriever=retriever,
                email_sender=lambda **kwargs: {"message_id": "baseline"},
            )
            with pytest.raises(RuntimeError, match="mail failed"):
                process_user_email_notifications(
                    "same-user",
                    user_store=store,
                    retriever=retriever,
                    email_sender=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("mail failed")),
                )

        user = store.get_user("same-user")
        pending = store.get_pending_email_notifications("same-user")
        assert user is not None
        assert user["pending_email_notifications"] == 1
        assert len(pending) == 1
        assert pending[0]["email_sent_at"] == ""
