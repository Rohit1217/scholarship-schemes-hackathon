"""Tests for login and saved-profile persistence."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest

from scholarship.user_store import UserStore, resolve_user_store_path


@contextmanager
def _store_path():
    path = Path.cwd() / f"user-store-{uuid4().hex}.json"
    try:
        yield path
    finally:
        if path.exists():
            path.unlink()


def test_register_and_authenticate_round_trip():
    with _store_path() as store_path:
        store = UserStore(store_path)

        created = store.register_user(
            "student@example.com",
            "secret123",
            full_name="Asha",
            email="student@example.com",
            phone="9999999999",
        )
        authenticated = store.authenticate_user("student@example.com", "secret123")

        assert created["login_id"] == "student@example.com"
        assert authenticated is not None
        assert authenticated["full_name"] == "Asha"
        assert "password_hash" not in authenticated


def test_duplicate_login_id_is_rejected():
    with _store_path() as store_path:
        store = UserStore(store_path)
        store.register_user("same-user", "secret123")

        with pytest.raises(ValueError, match="already exists"):
            store.register_user("same-user", "another-secret")


def test_wrong_password_returns_none():
    with _store_path() as store_path:
        store = UserStore(store_path)
        store.register_user("same-user", "secret123")

        assert store.authenticate_user("same-user", "wrong") is None


def test_save_profile_persists_details():
    with _store_path() as store_path:
        store = UserStore(store_path)
        store.register_user("same-user", "secret123")

        updated = store.save_profile(
            "same-user",
            {
                "state": "Maharashtra",
                "category": "OBC",
                "income": 150000,
                "gender": "Female",
                "age": 19,
                "education": "Undergraduate",
                "disability": True,
                "minority": False,
            },
            preferred_language="hi",
        )

        assert updated["preferred_language"] == "hi"
        assert updated["profile"]["state"] == "Maharashtra"
        assert updated["profile"]["income"] == 150000
        assert updated["profile"]["disability"] is True

        persisted = json.loads(store_path.read_text(encoding="utf-8"))
        saved_user = persisted["users"]["same-user"]
        assert saved_user["preferred_language"] == "hi"
        assert saved_user["profile"]["education"] == "Undergraduate"


def test_resolve_user_store_path_uses_explicit_path():
    explicit = Path.cwd() / f"custom-store-{uuid4().hex}.json"
    try:
        assert resolve_user_store_path(explicit) == explicit
    finally:
        if explicit.exists():
            explicit.unlink()


def test_refresh_notifications_sets_baseline_then_detects_new_matches():
    with _store_path() as store_path:
        store = UserStore(store_path)
        store.register_user("same-user", "secret123")
        store.save_profile(
            "same-user",
            {
                "state": "Maharashtra",
                "category": "OBC",
                "income": 150000,
                "gender": "Female",
                "age": 19,
                "education": "Undergraduate",
                "disability": False,
                "minority": False,
            },
        )

        first = store.refresh_notifications(
            "same-user",
            [{"scheme_id": "SCH001", "scheme_name": "First Scholarship"}],
        )
        second = store.refresh_notifications(
            "same-user",
            [
                {"scheme_id": "SCH001", "scheme_name": "First Scholarship"},
                {"scheme_id": "SCH002", "scheme_name": "Second Scholarship"},
            ],
        )

        assert first["baseline_reset"] is True
        assert first["new_count"] == 0
        assert second["baseline_reset"] is False
        assert second["new_count"] == 1
        assert second["user"]["notifications"][0]["scheme_id"] == "SCH002"
        assert second["user"]["notifications"][0]["notification_id"]
        assert second["user"]["notifications"][0]["email_sent_at"] == ""


def test_mark_notifications_emailed_clears_pending_count():
    with _store_path() as store_path:
        store = UserStore(store_path)
        store.register_user("same-user", "secret123", email="student@example.com")
        store.save_profile(
            "same-user",
            {
                "state": "Maharashtra",
                "category": "OBC",
                "income": 150000,
                "gender": "Female",
                "age": 19,
                "education": "Undergraduate",
                "disability": False,
                "minority": False,
            },
        )
        store.refresh_notifications(
            "same-user",
            [{"scheme_id": "SCH001", "scheme_name": "First Scholarship"}],
        )
        store.refresh_notifications(
            "same-user",
            [
                {"scheme_id": "SCH001", "scheme_name": "First Scholarship"},
                {"scheme_id": "SCH002", "scheme_name": "Second Scholarship"},
            ],
        )

        pending = store.get_pending_email_notifications("same-user")
        updated = store.mark_notifications_emailed(
            "same-user",
            [pending[0]["notification_id"]],
        )

        assert len(pending) == 1
        assert updated["pending_email_notifications"] == 0
        assert updated["notifications"][0]["email_sent_at"] != ""


def test_save_profile_resets_existing_notification_tracking():
    with _store_path() as store_path:
        store = UserStore(store_path)
        store.register_user("same-user", "secret123")
        store.save_profile(
            "same-user",
            {
                "state": "Maharashtra",
                "category": "OBC",
                "income": 150000,
                "gender": "Female",
                "age": 19,
                "education": "Undergraduate",
                "disability": False,
                "minority": False,
            },
        )
        store.refresh_notifications(
            "same-user",
            [{"scheme_id": "SCH001", "scheme_name": "First Scholarship"}],
        )
        store.refresh_notifications(
            "same-user",
            [
                {"scheme_id": "SCH001", "scheme_name": "First Scholarship"},
                {"scheme_id": "SCH002", "scheme_name": "Second Scholarship"},
            ],
        )

        updated = store.save_profile(
            "same-user",
            {
                "state": "Tamil Nadu",
                "category": "OBC",
                "income": 150000,
                "gender": "Female",
                "age": 19,
                "education": "Undergraduate",
                "disability": False,
                "minority": False,
            },
        )

        assert updated["notifications"] == []
        assert updated["notification_last_checked_at"] == ""
