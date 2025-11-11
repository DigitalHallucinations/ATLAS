"""Notification dispatch helper used by job workflows."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence

import logging

logger = logging.getLogger(__name__)


from .utils import coerce_metadata, dedupe_strings


async def send_notification(
    *,
    channel: str,
    message: str,
    recipients: Optional[Sequence[str]] = None,
    urgency: str = "normal",
    metadata: Optional[Mapping[str, object]] = None,
) -> Mapping[str, object]:
    """Record a logical notification delivery."""

    if not isinstance(channel, str) or not channel.strip():
        raise ValueError("Notification channel must be provided.")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Notification message must be provided.")

    await asyncio.sleep(0)

    payload = {
        "channel": channel.strip(),
        "message": message.strip(),
        "recipients": dedupe_strings(recipients),
        "urgency": urgency.strip().lower() if isinstance(urgency, str) else "normal",
        "metadata": coerce_metadata(metadata),
        "dispatched_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "Notification dispatched via %s to %s", payload["channel"], payload["recipients"] or "(broadcast)"
    )

    return payload


__all__ = ["send_notification"]

