"""File-backed login and profile persistence for the Gradio app.

The store is intentionally simple so it can run unchanged locally and on
Databricks Apps. When available, it prefers a Databricks Volume path for
durable storage; otherwise it falls back to the repository's data directory.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

_DEFAULT_STORE_FILENAME = "user_profiles.json"
_PBKDF2_ITERATIONS = 200_000
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_login_id(login_id: str) -> str:
    return login_id.strip().lower()


def _candidate_paths() -> list[Path]:
    env_path = os.environ.get("USER_DETAILS_STORE_PATH", "").strip()
    temp_root = (
        os.environ.get("TMPDIR", "").strip()
        or os.environ.get("TEMP", "").strip()
        or "/tmp"
    )

    candidates = []
    if env_path:
        candidates.append(Path(env_path))

    volume_root = Path("/Volumes/main/scholarships")
    if volume_root.exists():
        candidates.append(volume_root / "appdata" / _DEFAULT_STORE_FILENAME)

    candidates.append(_REPO_ROOT / "data" / _DEFAULT_STORE_FILENAME)
    candidates.append(Path(temp_root) / f"scholarship_{_DEFAULT_STORE_FILENAME}")

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _is_writable_store_path(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix="user-store-", suffix=".tmp")
        os.close(fd)
        os.unlink(temp_name)
        return True
    except OSError:
        return False


def resolve_user_store_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    candidates = _candidate_paths()
    for candidate in candidates:
        if _is_writable_store_path(candidate):
            return candidate
    return candidates[0]


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
    )
    return base64.b64encode(digest).decode("ascii")


def _verify_password(password: str, salt_b64: str, password_hash: str) -> bool:
    salt = base64.b64decode(salt_b64.encode("ascii"))
    expected = _hash_password(password, salt)
    return hmac.compare_digest(expected, password_hash)


def _normalise_profile(profile: dict[str, Any]) -> dict[str, Any]:
    def _to_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        return int(value)

    return {
        "state": str(profile.get("state") or "").strip(),
        "category": str(profile.get("category") or "").strip(),
        "income": _to_int(profile.get("income")),
        "gender": str(profile.get("gender") or "").strip(),
        "age": _to_int(profile.get("age")),
        "education": str(profile.get("education") or "").strip(),
        "disability": bool(profile.get("disability", False)),
        "minority": bool(profile.get("minority", False)),
    }


def _profile_signature(profile: dict[str, Any]) -> str:
    normalised = _normalise_profile(profile)
    return json.dumps(normalised, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _empty_notification_state() -> dict[str, Any]:
    return {
        "matched_scheme_ids": [],
        "items": [],
        "last_checked_at": "",
        "last_emailed_at": "",
        "profile_signature": "",
    }


def _normalise_matching_schemes(matching_schemes: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    normalised: list[dict[str, str]] = []
    for match in matching_schemes:
        scheme_id = str(match.get("scheme_id") or "").strip()
        scheme_name = str(match.get("scheme_name") or "").strip()
        key = scheme_id or scheme_name
        if not key or key in seen:
            continue
        seen.add(key)
        normalised.append({
            "scheme_id": scheme_id,
            "scheme_name": scheme_name or scheme_id,
        })
    return normalised


def _normalise_notification_items(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []

    normalised: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        notification_id = str(item.get("notification_id") or uuid4().hex)
        if notification_id in seen:
            continue
        seen.add(notification_id)
        normalised.append({
            "notification_id": notification_id,
            "scheme_id": str(item.get("scheme_id") or "").strip(),
            "scheme_name": str(item.get("scheme_name") or item.get("scheme_id") or "").strip(),
            "detected_at": str(item.get("detected_at") or "").strip(),
            "message": str(item.get("message") or "").strip(),
            "email_sent_at": str(item.get("email_sent_at") or "").strip(),
        })
    return normalised


def _normalise_notification_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return _empty_notification_state()

    return {
        "matched_scheme_ids": [
            str(value).strip()
            for value in state.get("matched_scheme_ids", [])
            if str(value).strip()
        ],
        "items": _normalise_notification_items(state.get("items", [])),
        "last_checked_at": str(state.get("last_checked_at") or "").strip(),
        "last_emailed_at": str(state.get("last_emailed_at") or "").strip(),
        "profile_signature": str(state.get("profile_signature") or "").strip(),
    }


def _email_notifications_enabled(record: dict[str, Any]) -> bool:
    value = record.get("email_notifications_enabled")
    if value is None:
        return bool(str(record.get("email") or "").strip())
    return bool(value)


class UserStore:
    """Small JSON-backed user store for login + saved search profiles."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = resolve_user_store_path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> str:
        return str(self._path)

    def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"users": {}}
        with self._path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"User store at {self._path} is not a JSON object.")
        users = data.get("users", {})
        if not isinstance(users, dict):
            raise ValueError(f"User store at {self._path} has an invalid users section.")
        return {"users": users}

    def _write(self, payload: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=self._path.parent,
            suffix=".tmp",
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.flush()
            temp_name = handle.name
        os.replace(temp_name, self._path)

    def _public_user(self, record: dict[str, Any]) -> dict[str, Any]:
        notification_state = _normalise_notification_state(record.get("notification_state"))
        pending_email_notifications = [
            item for item in notification_state["items"] if not item.get("email_sent_at")
        ]
        return {
            "login_id": record["login_id"],
            "full_name": record.get("full_name", ""),
            "email": record.get("email", ""),
            "phone": record.get("phone", ""),
            "preferred_language": record.get("preferred_language", "en"),
            "email_notifications_enabled": _email_notifications_enabled(record),
            "profile": record.get("profile", {}),
            "notifications": notification_state.get("items", []),
            "notification_last_checked_at": notification_state.get("last_checked_at", ""),
            "pending_email_notifications": len(pending_email_notifications),
            "email_notifications_last_sent_at": notification_state.get("last_emailed_at", ""),
            "created_at": record.get("created_at", ""),
            "updated_at": record.get("updated_at", ""),
        }

    def register_user(
        self,
        login_id: str,
        password: str,
        *,
        full_name: str = "",
        email: str = "",
        phone: str = "",
        preferred_language: str = "en",
        email_notifications_enabled: bool | None = None,
    ) -> dict[str, Any]:
        login_id = login_id.strip()
        if not login_id:
            raise ValueError("Login ID is required.")
        if not password:
            raise ValueError("Password is required.")

        key = _normalise_login_id(login_id)
        now = _utc_now()
        salt = os.urandom(16)
        record = {
            "login_id": login_id,
            "full_name": full_name.strip(),
            "email": email.strip(),
            "phone": phone.strip(),
            "preferred_language": preferred_language.strip() or "en",
            "email_notifications_enabled": (
                bool(email.strip()) if email_notifications_enabled is None
                else bool(email_notifications_enabled)
            ),
            "profile": {},
            "notification_state": _empty_notification_state(),
            "password_salt": base64.b64encode(salt).decode("ascii"),
            "password_hash": _hash_password(password, salt),
            "created_at": now,
            "updated_at": now,
        }

        with self._lock:
            payload = self._read()
            if key in payload["users"]:
                raise ValueError("A user with that Login ID already exists.")
            payload["users"][key] = record
            self._write(payload)

        return self._public_user(record)

    def authenticate_user(self, login_id: str, password: str) -> dict[str, Any] | None:
        key = _normalise_login_id(login_id)
        if not key or not password:
            return None

        with self._lock:
            payload = self._read()
            record = payload["users"].get(key)

        if not record:
            return None

        if not _verify_password(password, record["password_salt"], record["password_hash"]):
            return None
        return self._public_user(record)

    def get_user(self, login_id: str) -> dict[str, Any] | None:
        key = _normalise_login_id(login_id)
        if not key:
            return None
        with self._lock:
            payload = self._read()
            record = payload["users"].get(key)
        return self._public_user(record) if record else None

    def list_users(self) -> list[dict[str, Any]]:
        with self._lock:
            payload = self._read()
            records = list(payload["users"].values())
        return [self._public_user(record) for record in records]

    def save_profile(
        self,
        login_id: str,
        profile: dict[str, Any],
        *,
        preferred_language: str | None = None,
        full_name: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        email_notifications_enabled: bool | None = None,
    ) -> dict[str, Any]:
        key = _normalise_login_id(login_id)
        if not key:
            raise ValueError("Login ID is required.")

        with self._lock:
            payload = self._read()
            record = payload["users"].get(key)
            if not record:
                raise ValueError("User not found.")

            normalised_profile = _normalise_profile(profile)
            if record.get("profile", {}) != normalised_profile:
                # Reset tracking when the saved eligibility profile changes.
                record["notification_state"] = _empty_notification_state()
            record["profile"] = normalised_profile
            if preferred_language is not None:
                record["preferred_language"] = preferred_language.strip() or "en"
            if full_name is not None:
                record["full_name"] = full_name.strip()
            if email is not None:
                record["email"] = email.strip()
            if phone is not None:
                record["phone"] = phone.strip()
            if email_notifications_enabled is not None:
                record["email_notifications_enabled"] = bool(email_notifications_enabled)
            elif "email_notifications_enabled" not in record:
                record["email_notifications_enabled"] = bool(record.get("email", "").strip())
            record["updated_at"] = _utc_now()

            payload["users"][key] = record
            self._write(payload)

        return self._public_user(record)

    def get_pending_email_notifications(self, login_id: str) -> list[dict[str, Any]]:
        key = _normalise_login_id(login_id)
        if not key:
            raise ValueError("Login ID is required.")

        with self._lock:
            payload = self._read()
            record = payload["users"].get(key)
            if not record:
                raise ValueError("User not found.")

            notification_state = _normalise_notification_state(record.get("notification_state"))
            record["notification_state"] = notification_state
            payload["users"][key] = record
            self._write(payload)

        return [
            dict(item)
            for item in notification_state["items"]
            if not item.get("email_sent_at")
        ]

    def mark_notifications_emailed(
        self,
        login_id: str,
        notification_ids: list[str],
    ) -> dict[str, Any]:
        key = _normalise_login_id(login_id)
        if not key:
            raise ValueError("Login ID is required.")

        ids_to_mark = {str(notification_id).strip() for notification_id in notification_ids if str(notification_id).strip()}
        if not ids_to_mark:
            return self.get_user(login_id) or {}

        with self._lock:
            payload = self._read()
            record = payload["users"].get(key)
            if not record:
                raise ValueError("User not found.")

            notification_state = _normalise_notification_state(record.get("notification_state"))
            emailed_at = _utc_now()
            updated_any = False
            for item in notification_state["items"]:
                if item["notification_id"] in ids_to_mark and not item.get("email_sent_at"):
                    item["email_sent_at"] = emailed_at
                    updated_any = True
            if updated_any:
                notification_state["last_emailed_at"] = emailed_at
                record["notification_state"] = notification_state
                record["updated_at"] = emailed_at
                payload["users"][key] = record
                self._write(payload)

        return self.get_user(login_id) or {}

    def refresh_notifications(
        self,
        login_id: str,
        matching_schemes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        key = _normalise_login_id(login_id)
        if not key:
            raise ValueError("Login ID is required.")

        now = _utc_now()

        with self._lock:
            payload = self._read()
            record = payload["users"].get(key)
            if not record:
                raise ValueError("User not found.")

            normalised_profile = _normalise_profile(record.get("profile") or {})
            if not all(
                normalised_profile.get(field) not in ("", None)
                for field in ("state", "category", "income", "gender", "age", "education")
            ):
                raise ValueError("Save a complete profile before checking notifications.")

            notification_state = _normalise_notification_state(record.get("notification_state"))

            normalised_matches = _normalise_matching_schemes(matching_schemes)
            current_ids = [match["scheme_id"] or match["scheme_name"] for match in normalised_matches]
            current_signature = _profile_signature(normalised_profile)

            baseline_reset = (
                notification_state.get("profile_signature") != current_signature
                or not notification_state.get("last_checked_at")
            )
            new_notifications: list[dict[str, str]] = []

            if baseline_reset:
                notification_state = _empty_notification_state()
                notification_state["matched_scheme_ids"] = current_ids
                notification_state["profile_signature"] = current_signature
                notification_state["last_checked_at"] = now
            else:
                existing_ids = set(notification_state.get("matched_scheme_ids", []))
                existing_items = notification_state.get("items", [])
                if not isinstance(existing_items, list):
                    existing_items = []

                for match in normalised_matches:
                    match_id = match["scheme_id"] or match["scheme_name"]
                    if match_id in existing_ids:
                        continue
                    new_notifications.append({
                        "notification_id": uuid4().hex,
                        "scheme_id": match["scheme_id"],
                        "scheme_name": match["scheme_name"],
                        "detected_at": now,
                        "message": (
                            f"New matching scholarship found: {match['scheme_name']}"
                        ),
                        "email_sent_at": "",
                    })

                notification_state["items"] = (new_notifications + existing_items)[:50]
                notification_state["matched_scheme_ids"] = current_ids
                notification_state["profile_signature"] = current_signature
                notification_state["last_checked_at"] = now

            record["notification_state"] = notification_state
            record["updated_at"] = now
            payload["users"][key] = record
            self._write(payload)

        return {
            "user": self._public_user(record),
            "new_notifications": new_notifications,
            "new_count": len(new_notifications),
            "baseline_reset": baseline_reset,
        }
