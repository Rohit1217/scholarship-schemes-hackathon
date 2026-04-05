"""Tests for transactional email client."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from scholarship.email_client import email_configured, send_notification_email


def test_email_configured_requires_key_and_sender():
    env = {k: v for k, v in os.environ.items() if k not in ("BREVO_API_KEY", "EMAIL_FROM_ADDRESS")}
    with patch.dict(os.environ, env, clear=True), \
         patch("scholarship.email_client._maybe_load_brevo_api_key", return_value=None):
        assert email_configured() is False


def test_send_notification_email_posts_brevo_payload():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"messageId": "msg-123"}
    mock_response.content = b'{"messageId":"msg-123"}'

    with patch.dict(
        os.environ,
        {
            "BREVO_API_KEY": "key-123",
            "EMAIL_FROM_ADDRESS": "noreply@example.com",
            "EMAIL_FROM_NAME": "Scholarship Finder",
        },
        clear=False,
    ), patch("scholarship.email_client.requests.post", return_value=mock_response) as mock_post:
        result = send_notification_email(
            to_email="student@example.com",
            to_name="Asha",
            notifications=[
                {
                    "notification_id": "notif-1",
                    "scheme_id": "SCH001",
                    "scheme_name": "First Scholarship",
                }
            ],
        )

    assert result["provider"] == "brevo"
    assert result["message_id"] == "msg-123"
    assert mock_post.call_count == 1
    payload = mock_post.call_args.kwargs["json"]
    assert payload["to"][0]["email"] == "student@example.com"
    assert payload["sender"]["email"] == "noreply@example.com"
    assert "First Scholarship" in payload["textContent"]
