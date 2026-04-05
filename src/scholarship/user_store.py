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
        return {
            "login_id": record["login_id"],
            "full_name": record.get("full_name", ""),
            "email": record.get("email", ""),
            "phone": record.get("phone", ""),
            "preferred_language": record.get("preferred_language", "en"),
            "profile": record.get("profile", {}),
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
            "profile": {},
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

    def save_profile(
        self,
        login_id: str,
        profile: dict[str, Any],
        *,
        preferred_language: str | None = None,
        full_name: str | None = None,
        email: str | None = None,
        phone: str | None = None,
    ) -> dict[str, Any]:
        key = _normalise_login_id(login_id)
        if not key:
            raise ValueError("Login ID is required.")

        with self._lock:
            payload = self._read()
            record = payload["users"].get(key)
            if not record:
                raise ValueError("User not found.")

            record["profile"] = _normalise_profile(profile)
            if preferred_language is not None:
                record["preferred_language"] = preferred_language.strip() or "en"
            if full_name is not None:
                record["full_name"] = full_name.strip()
            if email is not None:
                record["email"] = email.strip()
            if phone is not None:
                record["phone"] = phone.strip()
            record["updated_at"] = _utc_now()

            payload["users"][key] = record
            self._write(payload)

        return self._public_user(record)
