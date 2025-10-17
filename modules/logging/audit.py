"""Audit logging helpers for persona configuration changes."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from modules.logging.logger import setup_logger

try:  # Optional dependency used to resolve the active user.
    from modules.user_accounts.user_account_service import UserAccountService
except Exception:  # pragma: no cover - user services unavailable during some tests
    UserAccountService = None  # type: ignore[misc]


@dataclass(frozen=True)
class PersonaAuditEntry:
    """Describe a persona change event for administrative review."""

    timestamp: str
    persona_name: str
    username: str
    old_tools: List[str]
    new_tools: List[str]
    rationale: str


class PersonaAuditLogger:
    """Persist persona audit events in a JSON lines log file."""

    def __init__(
        self,
        *,
        log_path: Optional[Path] = None,
        clock: Optional[Callable[[], datetime]] = None,
        user_service_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        self._logger = setup_logger(__name__)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._user_service_factory = user_service_factory or self._default_user_service_factory
        self._user_service: Optional[object] = None
        self._lock = threading.RLock()

        if log_path is None:
            base = Path(__file__).resolve().parents[2]
            log_dir = base / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "persona_audit.jsonl"
        else:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        self._log_path = log_path

    @staticmethod
    def _default_user_service_factory() -> Optional[object]:
        if UserAccountService is None:  # pragma: no cover - dependency not available
            return None
        try:
            return UserAccountService()
        except Exception:  # pragma: no cover - defensive guard
            return None

    def clear(self) -> None:
        """Remove all audit entries from the backing log file."""

        with self._lock:
            try:
                if self._log_path.exists():
                    self._log_path.unlink()
            except OSError:
                self._logger.warning("Failed to clear persona audit log", exc_info=True)

    def record_change(
        self,
        persona_name: str,
        old_tools: Iterable[str],
        new_tools: Iterable[str],
        *,
        rationale: Optional[str] = None,
        username: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> PersonaAuditEntry:
        """Append a persona audit entry to the JSONL log."""

        normalised_old = self._normalise_tools(old_tools)
        normalised_new = self._normalise_tools(new_tools)
        moment = timestamp or self._clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        else:
            moment = moment.astimezone(timezone.utc)

        entry = PersonaAuditEntry(
            timestamp=moment.isoformat().replace("+00:00", "Z"),
            persona_name=str(persona_name or ""),
            username=self._resolve_username(username),
            old_tools=normalised_old,
            new_tools=normalised_new,
            rationale=(rationale or "").strip(),
        )

        payload = json.dumps(asdict(entry), ensure_ascii=False)

        with self._lock:
            try:
                with self._log_path.open("a", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.write("\n")
            except OSError:
                self._logger.warning("Failed to persist persona audit entry", exc_info=True)

        return entry

    def get_history(
        self,
        *,
        persona_name: Optional[str] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> Tuple[List[PersonaAuditEntry], int]:
        """Return paginated audit entries matching ``persona_name``."""

        entries = list(self._iter_entries())
        if persona_name:
            persona_lower = persona_name.lower()
            entries = [
                entry
                for entry in entries
                if entry.persona_name.lower() == persona_lower
            ]

        entries.sort(key=lambda item: item.timestamp, reverse=True)

        total = len(entries)
        if offset < 0:
            offset = 0
        if limit <= 0:
            return [], total

        paginated = entries[offset : offset + limit]
        return paginated, total

    def _iter_entries(self) -> Iterator[PersonaAuditEntry]:
        with self._lock:
            try:
                if not self._log_path.exists():
                    lines: List[str] = []
                else:
                    lines = self._log_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                self._logger.warning("Failed to read persona audit log", exc_info=True)
                lines = []

        def _deserialize(line: str) -> Optional[PersonaAuditEntry]:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self._logger.warning("Encountered invalid JSON in persona audit log")
                return None

            if not isinstance(payload, dict):
                return None

            timestamp = str(payload.get("timestamp") or "")
            persona = str(payload.get("persona_name") or "")
            username = str(payload.get("username") or "")

            def _as_list(value: object) -> List[str]:
                if isinstance(value, list):
                    return [str(item) for item in value]
                if isinstance(value, str):
                    return [value]
                return []

            return PersonaAuditEntry(
                timestamp=timestamp,
                persona_name=persona,
                username=username,
                old_tools=self._normalise_tools(_as_list(payload.get("old_tools"))),
                new_tools=self._normalise_tools(_as_list(payload.get("new_tools"))),
                rationale=str(payload.get("rationale") or ""),
            )

        for line in lines:
            entry = _deserialize(line)
            if entry is not None:
                yield entry

    def _resolve_username(self, explicit: Optional[str]) -> str:
        if explicit:
            return explicit
        try:
            if self._user_service is None and callable(self._user_service_factory):
                self._user_service = self._user_service_factory()
            service = self._user_service
            if service is None:
                return "unknown"
            getter = getattr(service, "get_active_user", None)
            if callable(getter):
                username = getter()
                if username:
                    return str(username)
        except Exception:  # pragma: no cover - defensive logging only
            self._logger.warning("Failed to resolve active user for persona audit", exc_info=True)
        return "unknown"

    @staticmethod
    def _normalise_tools(tools: Iterable[str]) -> List[str]:
        normalised: List[str] = []
        seen: set[str] = set()
        for item in tools:
            text = str(item or "").strip()
            if text and text not in seen:
                normalised.append(text)
                seen.add(text)
        return normalised


_default_logger: Optional[PersonaAuditLogger] = None


@dataclass(frozen=True)
class PersonaReviewAttestation:
    """Describe a persona review attestation event."""

    timestamp: str
    persona_name: str
    reviewer: str
    expires_at: str
    notes: str


@dataclass(frozen=True)
class PersonaReviewTask:
    """Queued review task awaiting completion."""

    persona_name: str
    queued_at: str
    due_at: str
    reason: str
    status: str = "pending"
    completed_at: Optional[str] = None


def _normalise_iso_timestamp(moment: datetime) -> str:
    """Return ``moment`` as an ISO string with UTC ``Z`` suffix."""

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    else:
        moment = moment.astimezone(timezone.utc)
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso_timestamp(text: str) -> Optional[datetime]:
    """Parse ``text`` into an aware ``datetime`` in UTC."""

    if not text:
        return None
    candidate = text.strip()
    if not candidate:
        return None
    try:
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


class PersonaReviewLogger:
    """Persist persona review attestations alongside change history."""

    def __init__(
        self,
        *,
        log_path: Optional[Path] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._logger = setup_logger(__name__)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()

        if log_path is None:
            base = Path(__file__).resolve().parents[2]
            log_dir = base / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "persona_reviews.jsonl"
        else:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        self._log_path = log_path

    def clear(self) -> None:
        with self._lock:
            try:
                if self._log_path.exists():
                    self._log_path.unlink()
            except OSError:
                self._logger.warning("Failed to clear persona review log", exc_info=True)

    def record_attestation(
        self,
        persona_name: str,
        *,
        reviewer: str,
        expires_at: Optional[datetime] = None,
        notes: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        validity: Optional[timedelta] = None,
    ) -> PersonaReviewAttestation:
        """Persist a review attestation entry for ``persona_name``."""

        moment = timestamp or self._clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        else:
            moment = moment.astimezone(timezone.utc)

        if expires_at is None and validity is not None:
            expires_at = moment + validity

        if expires_at is None:
            expires = moment
        else:
            if expires_at.tzinfo is None:
                expires = expires_at.replace(tzinfo=timezone.utc)
            else:
                expires = expires_at.astimezone(timezone.utc)

        entry = PersonaReviewAttestation(
            timestamp=_normalise_iso_timestamp(moment),
            persona_name=str(persona_name or ""),
            reviewer=str(reviewer or "unknown"),
            expires_at=_normalise_iso_timestamp(expires),
            notes=(notes or "").strip(),
        )

        payload = json.dumps(asdict(entry), ensure_ascii=False)

        with self._lock:
            try:
                with self._log_path.open("a", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.write("\n")
            except OSError:
                self._logger.warning("Failed to persist persona review attestation", exc_info=True)

        return entry

    def _iter_entries(self) -> Iterator[PersonaReviewAttestation]:
        with self._lock:
            try:
                if not self._log_path.exists():
                    lines: List[str] = []
                else:
                    lines = self._log_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                self._logger.warning("Failed to read persona review log", exc_info=True)
                lines = []

        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self._logger.warning("Encountered invalid JSON in persona review log")
                continue

            if not isinstance(payload, dict):
                continue

            timestamp = str(payload.get("timestamp") or "")
            persona = str(payload.get("persona_name") or "")
            reviewer = str(payload.get("reviewer") or "")
            expires_at = str(payload.get("expires_at") or "")
            notes = str(payload.get("notes") or "")

            yield PersonaReviewAttestation(
                timestamp=timestamp,
                persona_name=persona,
                reviewer=reviewer,
                expires_at=expires_at,
                notes=notes,
            )

    def get_attestations(
        self,
        *,
        persona_name: Optional[str] = None,
    ) -> List[PersonaReviewAttestation]:
        """Return attestation history, optionally filtered by persona."""

        entries = list(self._iter_entries())
        if persona_name:
            persona_lower = persona_name.lower()
            entries = [
                entry
                for entry in entries
                if entry.persona_name.lower() == persona_lower
            ]

        entries.sort(key=lambda item: item.timestamp, reverse=True)
        return entries

    def get_latest_attestation(
        self,
        persona_name: str,
    ) -> Optional[PersonaReviewAttestation]:
        """Return the most recent attestation for ``persona_name``."""

        for entry in self.get_attestations(persona_name=persona_name):
            return entry
        return None


class PersonaReviewQueue:
    """Persist a queue of persona review tasks to a JSON document."""

    def __init__(self, *, queue_path: Optional[Path] = None) -> None:
        self._logger = setup_logger(__name__)
        self._lock = threading.RLock()

        if queue_path is None:
            base = Path(__file__).resolve().parents[2]
            log_dir = base / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            queue_path = log_dir / "persona_review_queue.json"
        else:
            queue_path = Path(queue_path)
            queue_path.parent.mkdir(parents=True, exist_ok=True)

        self._queue_path = queue_path

    def _load_tasks_locked(self) -> List[PersonaReviewTask]:
        try:
            if not self._queue_path.exists():
                return []
            raw = self._queue_path.read_text(encoding="utf-8")
        except OSError:
            self._logger.warning("Failed to read persona review queue", exc_info=True)
            return []

        try:
            payload = json.loads(raw) if raw.strip() else []
        except json.JSONDecodeError:
            self._logger.warning("Encountered invalid JSON in persona review queue")
            return []

        tasks: List[PersonaReviewTask] = []
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                tasks.append(
                    PersonaReviewTask(
                        persona_name=str(item.get("persona_name") or ""),
                        queued_at=str(item.get("queued_at") or ""),
                        due_at=str(item.get("due_at") or ""),
                        reason=str(item.get("reason") or ""),
                        status=str(item.get("status") or "pending"),
                        completed_at=(
                            str(item["completed_at"])
                            if item.get("completed_at") not in (None, "")
                            else None
                        ),
                    )
                )
        return tasks

    def _write_tasks_locked(self, tasks: List[PersonaReviewTask]) -> None:
        serialised = [asdict(task) for task in tasks]
        try:
            self._queue_path.write_text(
                json.dumps(serialised, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            self._logger.warning("Failed to persist persona review queue", exc_info=True)

    def list_tasks(self, *, status: Optional[str] = None) -> List[PersonaReviewTask]:
        with self._lock:
            tasks = self._load_tasks_locked()
        if status is None:
            return tasks
        status_lower = status.lower()
        return [task for task in tasks if task.status.lower() == status_lower]

    def enqueue(
        self,
        persona_name: str,
        *,
        due_at: datetime,
        reason: str,
        now: Optional[datetime] = None,
    ) -> PersonaReviewTask:
        """Queue a persona review task if one is not already pending."""

        moment = now or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        else:
            moment = moment.astimezone(timezone.utc)

        due = due_at
        if due.tzinfo is None:
            due = due.replace(tzinfo=timezone.utc)
        else:
            due = due.astimezone(timezone.utc)

        with self._lock:
            tasks = self._load_tasks_locked()
            updated = False
            for index, task in enumerate(tasks):
                if task.persona_name.lower() != persona_name.lower():
                    continue
                if task.status.lower() != "pending":
                    continue
                existing_due = _parse_iso_timestamp(task.due_at) or due
                if due < existing_due:
                    tasks[index] = PersonaReviewTask(
                        persona_name=task.persona_name,
                        queued_at=task.queued_at,
                        due_at=_normalise_iso_timestamp(due),
                        reason=reason,
                        status=task.status,
                        completed_at=task.completed_at,
                    )
                else:
                    tasks[index] = PersonaReviewTask(
                        persona_name=task.persona_name,
                        queued_at=task.queued_at,
                        due_at=task.due_at,
                        reason=reason,
                        status=task.status,
                        completed_at=task.completed_at,
                    )
                updated = True
                queued_task = tasks[index]
                break
            else:
                queued_task = PersonaReviewTask(
                    persona_name=str(persona_name or ""),
                    queued_at=_normalise_iso_timestamp(moment),
                    due_at=_normalise_iso_timestamp(due),
                    reason=str(reason or ""),
                )
                tasks.append(queued_task)

            if not updated:
                tasks.sort(key=lambda item: item.persona_name.lower())

            self._write_tasks_locked(tasks)
            return queued_task

    def mark_completed(
        self,
        persona_name: str,
        *,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Mark pending tasks for ``persona_name`` as completed."""

        moment = timestamp or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        else:
            moment = moment.astimezone(timezone.utc)

        completed = _normalise_iso_timestamp(moment)

        with self._lock:
            tasks = self._load_tasks_locked()
            changed = False
            for index, task in enumerate(tasks):
                if task.persona_name.lower() != persona_name.lower():
                    continue
                if task.status.lower() != "pending":
                    continue
                tasks[index] = PersonaReviewTask(
                    persona_name=task.persona_name,
                    queued_at=task.queued_at,
                    due_at=task.due_at,
                    reason=task.reason,
                    status="completed",
                    completed_at=completed,
                )
                changed = True
            if changed:
                self._write_tasks_locked(tasks)

    def is_pending(self, persona_name: str) -> bool:
        return any(
            task
            for task in self.list_tasks(status="pending")
            if task.persona_name.lower() == persona_name.lower()
        )


def get_persona_audit_logger() -> PersonaAuditLogger:
    """Return the shared persona audit logger instance."""

    global _default_logger
    if _default_logger is None:
        _default_logger = PersonaAuditLogger()
    return _default_logger


def set_persona_audit_logger(logger: Optional[PersonaAuditLogger]) -> None:
    """Override the shared persona audit logger (primarily for tests)."""

    global _default_logger
    _default_logger = logger


_default_review_logger: Optional[PersonaReviewLogger] = None


def get_persona_review_logger() -> PersonaReviewLogger:
    """Return the shared persona review logger instance."""

    global _default_review_logger
    if _default_review_logger is None:
        _default_review_logger = PersonaReviewLogger()
    return _default_review_logger


def set_persona_review_logger(logger: Optional[PersonaReviewLogger]) -> None:
    """Override the shared persona review logger (primarily for tests)."""

    global _default_review_logger
    _default_review_logger = logger


_default_review_queue: Optional[PersonaReviewQueue] = None


def get_persona_review_queue() -> PersonaReviewQueue:
    """Return the shared persona review task queue."""

    global _default_review_queue
    if _default_review_queue is None:
        _default_review_queue = PersonaReviewQueue()
    return _default_review_queue


def set_persona_review_queue(queue: Optional[PersonaReviewQueue]) -> None:
    """Override the shared persona review queue (primarily for tests)."""

    global _default_review_queue
    _default_review_queue = queue


def format_persona_timestamp(moment: datetime) -> str:
    """Public helper to format timestamps for persona audit records."""

    return _normalise_iso_timestamp(moment)


def parse_persona_timestamp(value: str) -> Optional[datetime]:
    """Parse a persisted persona audit or review timestamp."""

    return _parse_iso_timestamp(value)


__all__ = [
    "PersonaAuditEntry",
    "PersonaAuditLogger",
    "PersonaReviewAttestation",
    "PersonaReviewLogger",
    "PersonaReviewQueue",
    "PersonaReviewTask",
    "format_persona_timestamp",
    "get_persona_audit_logger",
    "get_persona_review_logger",
    "get_persona_review_queue",
    "parse_persona_timestamp",
    "set_persona_audit_logger",
    "set_persona_review_logger",
    "set_persona_review_queue",
]
