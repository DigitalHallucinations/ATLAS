"""Asynchronous email dispatch stub used for reporting workflows."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence

import logging
import uuid

logger = logging.getLogger(__name__)


from .utils import coerce_metadata, dedupe_strings


async def send_email(
    *,
    subject: str,
    body: str,
    to: Sequence[str],
    cc: Optional[Sequence[str]] = None,
    bcc: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Mapping[str, object]:
    if not subject.strip():
        raise ValueError("Email subject must be provided.")
    if not body.strip():
        raise ValueError("Email body must be provided.")
    if not to:
        raise ValueError("At least one recipient must be specified.")

    await asyncio.sleep(0)

    message_id = f"msg-{uuid.uuid4().hex}"
    payload = {
        "message_id": message_id,
        "subject": subject.strip(),
        "body": body.strip(),
        "to": dedupe_strings(to),
        "cc": dedupe_strings(cc),
        "bcc": dedupe_strings(bcc),
        "metadata": coerce_metadata(metadata),
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info("Dispatched email %s to %s", message_id, ", ".join(payload["to"]))

    return payload


__all__ = ["send_email"]

