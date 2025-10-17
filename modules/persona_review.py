"""Persona review scheduling helpers and policy utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from modules.logging.audit import (
    PersonaAuditEntry,
    PersonaAuditLogger,
    PersonaReviewLogger,
    PersonaReviewQueue,
    format_persona_timestamp,
    get_persona_audit_logger,
    get_persona_review_logger,
    get_persona_review_queue,
    parse_persona_timestamp,
)
from modules.logging.logger import setup_logger


REVIEW_INTERVAL_DAYS = 90

logger = setup_logger(__name__)


def _ensure_timezone(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def discover_persona_names(*, app_root: Optional[Path] = None) -> List[str]:
    """Return a list of persona names based on files in ``modules/Personas``."""

    if app_root is None:
        app_root = Path(__file__).resolve().parents[2]

    personas_root = app_root / "modules" / "Personas"
    if not personas_root.exists():
        return []

    names: List[str] = []
    try:
        for candidate in personas_root.iterdir():
            if not candidate.is_dir():
                continue
            persona_dir = candidate / "Persona"
            if not persona_dir.exists():
                continue
            for definition in persona_dir.glob("*.json"):
                names.append(definition.stem)
                break
    except OSError:
        logger.warning("Failed to enumerate personas for review scheduling", exc_info=True)

    return sorted({name for name in names if name})


def _latest_change(
    audit_logger: PersonaAuditLogger,
    persona_name: str,
) -> Optional[PersonaAuditEntry]:
    entries, _total = audit_logger.get_history(persona_name=persona_name, limit=1)
    if entries:
        return entries[0]
    return None


@dataclass(frozen=True)
class PersonaReviewStatus:
    persona_name: str
    last_change: Optional[str]
    last_review: Optional[str]
    reviewer: Optional[str]
    expires_at: Optional[str]
    overdue: bool
    pending_task: bool
    next_due: Optional[str]
    policy_days: int
    notes: Optional[str]


def compute_review_status(
    persona_name: str,
    *,
    audit_logger: Optional[PersonaAuditLogger] = None,
    review_logger: Optional[PersonaReviewLogger] = None,
    review_queue: Optional[PersonaReviewQueue] = None,
    now: Optional[datetime] = None,
    interval_days: int = REVIEW_INTERVAL_DAYS,
) -> PersonaReviewStatus:
    """Return the current review status for ``persona_name``."""

    audit_logger = audit_logger or get_persona_audit_logger()
    review_logger = review_logger or get_persona_review_logger()
    review_queue = review_queue or get_persona_review_queue()

    moment = _ensure_timezone(now or datetime.now(timezone.utc))

    latest_change = _latest_change(audit_logger, persona_name)
    change_ts = latest_change.timestamp if latest_change else None
    change_dt = parse_persona_timestamp(change_ts or "") if change_ts else None

    attestation = review_logger.get_latest_attestation(persona_name)
    attestation_ts = attestation.timestamp if attestation else None
    attestation_dt = parse_persona_timestamp(attestation_ts or "") if attestation_ts else None
    expires_dt = parse_persona_timestamp(attestation.expires_at) if attestation else None

    next_due_dt: Optional[datetime] = None
    overdue = False

    if expires_dt is not None:
        next_due_dt = expires_dt
        overdue = expires_dt <= moment
    else:
        reference = attestation_dt or change_dt
        if reference is not None:
            next_due_dt = reference + timedelta(days=interval_days)
            overdue = next_due_dt <= moment
        elif change_dt is not None:
            next_due_dt = change_dt + timedelta(days=interval_days)
            overdue = next_due_dt <= moment
        else:
            overdue = True

    pending = review_queue.is_pending(persona_name) if review_queue else False

    next_due = format_persona_timestamp(next_due_dt) if next_due_dt else None

    return PersonaReviewStatus(
        persona_name=persona_name,
        last_change=change_ts,
        last_review=attestation_ts,
        reviewer=(attestation.reviewer if attestation else None),
        expires_at=(attestation.expires_at if attestation else next_due),
        overdue=overdue,
        pending_task=pending,
        next_due=next_due,
        policy_days=interval_days,
        notes=(attestation.notes if attestation else None),
    )


class PersonaReviewScheduler:
    """Identify personas that require attention and enqueue review tasks."""

    def __init__(
        self,
        *,
        audit_logger: Optional[PersonaAuditLogger] = None,
        review_logger: Optional[PersonaReviewLogger] = None,
        review_queue: Optional[PersonaReviewQueue] = None,
        interval_days: int = REVIEW_INTERVAL_DAYS,
    ) -> None:
        self._audit_logger = audit_logger or get_persona_audit_logger()
        self._review_logger = review_logger or get_persona_review_logger()
        self._review_queue = review_queue or get_persona_review_queue()
        self._interval_days = max(1, int(interval_days))

    @property
    def interval(self) -> timedelta:
        return timedelta(days=self._interval_days)

    def scan_and_queue(
        self,
        persona_names: Sequence[str],
        *,
        now: Optional[datetime] = None,
    ) -> List[str]:
        """Scan ``persona_names`` and queue overdue reviews.

        Returns a list of persona names that resulted in new tasks being enqueued.
        """

        moment = _ensure_timezone(now or datetime.now(timezone.utc))
        queued: List[str] = []

        for persona_name in persona_names:
            status = compute_review_status(
                persona_name,
                audit_logger=self._audit_logger,
                review_logger=self._review_logger,
                review_queue=self._review_queue,
                now=moment,
                interval_days=self._interval_days,
            )

            if not status.overdue:
                continue

            due_dt = parse_persona_timestamp(status.next_due or "") or moment

            reason: str
            if status.last_review:
                reason = f"Review expired on {status.expires_at or status.next_due}"
            elif status.last_change:
                reason = (
                    "No review recorded since persona change on "
                    f"{status.last_change}"
                )
            else:
                reason = "Persona has never been reviewed"

            if status.pending_task:
                continue

            self._review_queue.enqueue(
                persona_name,
                due_at=due_dt,
                reason=reason,
                now=moment,
            )
            queued.append(persona_name)

        return queued


def queue_overdue_personas(
    *,
    persona_names: Optional[Iterable[str]] = None,
    audit_logger: Optional[PersonaAuditLogger] = None,
    review_logger: Optional[PersonaReviewLogger] = None,
    review_queue: Optional[PersonaReviewQueue] = None,
    now: Optional[datetime] = None,
    interval_days: int = REVIEW_INTERVAL_DAYS,
) -> List[str]:
    """High level helper to queue overdue personas using defaults."""

    scheduler = PersonaReviewScheduler(
        audit_logger=audit_logger,
        review_logger=review_logger,
        review_queue=review_queue,
        interval_days=interval_days,
    )

    if persona_names is None:
        persona_names = discover_persona_names()

    return scheduler.scan_and_queue(list(persona_names), now=now)


__all__ = [
    "PersonaReviewScheduler",
    "PersonaReviewStatus",
    "REVIEW_INTERVAL_DAYS",
    "compute_review_status",
    "discover_persona_names",
    "queue_overdue_personas",
]

