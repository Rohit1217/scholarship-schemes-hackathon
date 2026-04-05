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
