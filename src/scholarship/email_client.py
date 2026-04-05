"""Transactional email delivery helpers for scholarship alerts."""

from __future__ import annotations

import base64
import logging
import os
from html import escape
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BREVO_SEND_EMAIL_URL = "https://api.brevo.com/v3/smtp/email"


class EmailConfigurationError(RuntimeError):
    """Raised when required email delivery settings are missing."""


def _maybe_load_brevo_api_key() -> None:
    if os.environ.get("BREVO_API_KEY", "").strip():
        return
    try:
        from databricks.sdk import WorkspaceClient

        workspace = WorkspaceClient()
        secret = workspace.secrets.get_secret(scope="scholarship", key="brevo_api_key")
        if secret and secret.value:
            try:
                decoded = base64.b64decode(secret.value).decode("utf-8")
            except Exception:
                decoded = secret.value
            os.environ["BREVO_API_KEY"] = decoded
            logger.info("Loaded BREVO_API_KEY from Databricks secret scope scholarship/brevo_api_key.")
    except Exception as exc:
        logger.warning("Could not load BREVO_API_KEY from Databricks secret scope: %s", exc)


def email_configured() -> bool:
    _maybe_load_brevo_api_key()
    return bool(
        os.environ.get("BREVO_API_KEY", "").strip()
        and os.environ.get("EMAIL_FROM_ADDRESS", "").strip()
    )


def _require_email_config() -> dict[str, str]:
    _maybe_load_brevo_api_key()
    api_key = os.environ.get("BREVO_API_KEY", "").strip()
    from_address = os.environ.get("EMAIL_FROM_ADDRESS", "").strip()
    from_name = os.environ.get("EMAIL_FROM_NAME", "").strip() or "Scholarship Finder"
    if not api_key:
        raise EmailConfigurationError(
            "BREVO_API_KEY is not configured. Add it as an env var or Databricks secret."
        )
    if not from_address:
        raise EmailConfigurationError("EMAIL_FROM_ADDRESS is not configured.")
    return {
        "api_key": api_key,
        "from_address": from_address,
        "from_name": from_name,
    }


def _email_subject(notification_count: int) -> str:
    if notification_count == 1:
        return "1 new scholarship matched your saved profile"
    return f"{notification_count} new scholarships matched your saved profile"


def _render_text_body(
    recipient_name: str,
    notifications: list[dict[str, Any]],
) -> str:
    greeting_name = recipient_name.strip() or "there"
    lines = [
        f"Hello {greeting_name},",
        "",
        "We found new scholarships that match your saved profile:",
        "",
    ]
    for notification in notifications:
        scheme_name = str(notification.get("scheme_name") or notification.get("scheme_id") or "Scholarship")
        scheme_id = str(notification.get("scheme_id") or "")
        if scheme_id:
            lines.append(f"- {scheme_name} ({scheme_id})")
        else:
            lines.append(f"- {scheme_name}")
    lines.extend([
        "",
        "Sign in to the Scholarship Finder app to review the latest matches.",
        "",
        "Scholarship Finder",
    ])
    return "\n".join(lines)


def _render_html_body(
    recipient_name: str,
    notifications: list[dict[str, Any]],
) -> str:
    greeting_name = escape(recipient_name.strip() or "there")
    items = []
    for notification in notifications:
        scheme_name = escape(
            str(notification.get("scheme_name") or notification.get("scheme_id") or "Scholarship")
        )
        scheme_id = escape(str(notification.get("scheme_id") or ""))
        suffix = f" <span style='color:#567;'>(ID: {scheme_id})</span>" if scheme_id else ""
        items.append(f"<li><strong>{scheme_name}</strong>{suffix}</li>")
    list_html = "".join(items)
    return (
        "<html><body style='font-family:Segoe UI,Arial,sans-serif;color:#16324f;'>"
        f"<p>Hello {greeting_name},</p>"
        "<p>We found new scholarships that match your saved profile:</p>"
        f"<ul>{list_html}</ul>"
        "<p>Sign in to the Scholarship Finder app to review the latest matches.</p>"
        "<p>Scholarship Finder</p>"
        "</body></html>"
    )


def send_notification_email(
    *,
    to_email: str,
    to_name: str,
    notifications: list[dict[str, Any]],
) -> dict[str, Any]:
    if not to_email.strip():
        raise ValueError("Recipient email is required.")
    if not notifications:
        raise ValueError("At least one notification is required.")

    config = _require_email_config()
    response = requests.post(
        _BREVO_SEND_EMAIL_URL,
        headers={
            "accept": "application/json",
            "api-key": config["api_key"],
            "content-type": "application/json",
        },
        json={
            "sender": {
                "name": config["from_name"],
                "email": config["from_address"],
            },
            "to": [
                {
                    "name": to_name.strip() or to_email.strip(),
                    "email": to_email.strip(),
                }
            ],
            "subject": _email_subject(len(notifications)),
            "textContent": _render_text_body(to_name, notifications),
            "htmlContent": _render_html_body(to_name, notifications),
            "tags": ["scholarship-alert"],
        },
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json() if response.content else {}
    return {
        "message_id": payload.get("messageId", ""),
        "provider": "brevo",
    }
