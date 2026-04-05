"""Check all saved users for new scholarship matches and email pending alerts."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scholarship.notification_service import process_all_user_email_notifications

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    logger.info("Scholarship email notification job starting.")
    summary = process_all_user_email_notifications()
    logger.info(
        "Scholarship email notification job finished: processed=%s skipped=%s failures=%s new_notifications=%s emailed=%s",
        summary["processed"],
        summary["skipped"],
        summary["failures"],
        summary["new_notifications"],
        summary["emailed"],
    )
    for user_result in summary["users"]:
        logger.info("Notification job user result: %s", user_result)


if __name__ == "__main__":
    main()
