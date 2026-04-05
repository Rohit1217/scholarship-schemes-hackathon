"""Shared orchestration for scholarship alert refresh and email delivery."""

from __future__ import annotations

import logging
from typing import Any, Callable

from scholarship.email_client import email_configured, send_notification_email
from scholarship.profile_matching import matching_schemes_for_profile, saved_profile_complete
from scholarship.retriever import Retriever, get_retriever
from scholarship.user_store import UserStore

logger = logging.getLogger(__name__)

EmailSender = Callable[..., dict[str, Any]]


def refresh_user_notifications(
    login_id: str,
    *,
    user_store: UserStore | None = None,
    retriever: Retriever | None = None,
    k: int = 20,
) -> dict[str, Any]:
    store = user_store or UserStore()
    user = store.get_user(login_id)
    if not user:
        raise ValueError("User not found.")

    profile = user.get("profile") or {}
    if not saved_profile_complete(profile):
        raise ValueError("Save a complete profile before checking notifications.")

    active_retriever = retriever or get_retriever()
    matches = matching_schemes_for_profile(profile, retriever=active_retriever, k=k)
    return store.refresh_notifications(login_id, matches)


def _send_pending_notifications(
    user: dict[str, Any],
    pending_notifications: list[dict[str, Any]],
    *,
    email_sender: EmailSender,
) -> dict[str, Any]:
    if not pending_notifications:
        return {
            "attempted": False,
            "sent": False,
            "sent_count": 0,
            "message": "",
        }

    if not user.get("email"):
        return {
            "attempted": False,
            "sent": False,
            "sent_count": 0,
            "message": "No email address is saved on this account.",
        }

    if not user.get("email_notifications_enabled", False):
        return {
            "attempted": False,
            "sent": False,
            "sent_count": 0,
            "message": "Email notifications are disabled for this account.",
        }

    if not email_configured():
        return {
            "attempted": False,
            "sent": False,
            "sent_count": 0,
            "message": (
                "Email delivery is not configured yet. "
                "Set BREVO_API_KEY and EMAIL_FROM_ADDRESS to send alerts."
            ),
        }

    send_result = email_sender(
        to_email=str(user.get("email") or ""),
        to_name=str(user.get("full_name") or user.get("login_id") or ""),
        notifications=pending_notifications,
    )
    return {
        "attempted": True,
        "sent": True,
        "sent_count": len(pending_notifications),
        "message": "Email sent successfully.",
        "provider_result": send_result,
    }


def process_user_email_notifications(
    login_id: str,
    *,
    user_store: UserStore | None = None,
    retriever: Retriever | None = None,
    email_sender: EmailSender = send_notification_email,
    k: int = 20,
) -> dict[str, Any]:
    store = user_store or UserStore()
    refresh_result = refresh_user_notifications(
        login_id,
        user_store=store,
        retriever=retriever,
        k=k,
    )
    refreshed_user = refresh_result["user"]
    pending_notifications = store.get_pending_email_notifications(login_id)
    email_result = _send_pending_notifications(
        refreshed_user,
        pending_notifications,
        email_sender=email_sender,
    )

    if email_result["sent"]:
        sent_ids = [
            str(notification.get("notification_id") or "")
            for notification in pending_notifications
            if str(notification.get("notification_id") or "")
        ]
        refreshed_user = store.mark_notifications_emailed(login_id, sent_ids)

    return {
        "user": refreshed_user,
        "refresh": refresh_result,
        "pending_email_notifications": pending_notifications,
        "email": email_result,
    }


def process_all_user_email_notifications(
    *,
    user_store: UserStore | None = None,
    retriever: Retriever | None = None,
    email_sender: EmailSender = send_notification_email,
    k: int = 20,
) -> dict[str, Any]:
    store = user_store or UserStore()
    active_retriever = retriever or get_retriever()

    processed_users: list[dict[str, Any]] = []
    summary = {
        "processed": 0,
        "emailed": 0,
        "new_notifications": 0,
        "failures": 0,
        "skipped": 0,
        "users": processed_users,
    }

    for user in store.list_users():
        login_id = str(user.get("login_id") or "")
        if not login_id:
            continue

        if not saved_profile_complete(user.get("profile") or {}):
            processed_users.append({
                "login_id": login_id,
                "status": "skipped",
                "reason": "incomplete_profile",
            })
            summary["skipped"] += 1
            continue

        try:
            result = process_user_email_notifications(
                login_id,
                user_store=store,
                retriever=active_retriever,
                email_sender=email_sender,
                k=k,
            )
        except Exception as exc:
            logger.exception("Failed processing scholarship email notifications for %s", login_id)
            processed_users.append({
                "login_id": login_id,
                "status": "failed",
                "reason": str(exc),
            })
            summary["failures"] += 1
            continue

        processed_users.append({
            "login_id": login_id,
            "status": "processed",
            "new_notifications": result["refresh"]["new_count"],
            "email_sent": result["email"]["sent"],
            "email_sent_count": result["email"]["sent_count"],
            "email_message": result["email"]["message"],
        })
        summary["processed"] += 1
        summary["new_notifications"] += int(result["refresh"]["new_count"])
        summary["emailed"] += int(result["email"]["sent_count"])

    return summary
