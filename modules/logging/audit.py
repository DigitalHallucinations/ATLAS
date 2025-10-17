"""Audit logging helpers for persona configuration changes."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

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


__all__ = [
    "PersonaAuditEntry",
    "PersonaAuditLogger",
    "get_persona_audit_logger",
    "set_persona_audit_logger",
]
