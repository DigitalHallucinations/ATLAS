"""
Notification history service.

Provides persistent tracking of sent notifications, missed alerts,
and user interactions for the notification system.

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Protocol
from uuid import uuid4

from .types import (
    NotificationChannel,
    NotificationHistoryEntry,
    NotificationPriority,
    NotificationRequest,
    NotificationStatus,
)


logger = logging.getLogger(__name__)


class NotificationHistoryRepository(Protocol):
    """Repository interface for notification history persistence.
    
    Implementations should provide PostgreSQL, SQLite, or in-memory backends.
    """
    
    async def save(self, entry: NotificationHistoryEntry) -> NotificationHistoryEntry:
        """Save or update a history entry."""
        ...
    
    async def get(self, entry_id: str) -> Optional[NotificationHistoryEntry]:
        """Get a history entry by ID."""
        ...
    
    async def list_recent(
        self,
        limit: int = 100,
        since: Optional[datetime] = None,
        status: Optional[NotificationStatus] = None,
        source_type: Optional[str] = None,
    ) -> List[NotificationHistoryEntry]:
        """List recent notification history entries."""
        ...
    
    async def list_unread(self, limit: int = 50) -> List[NotificationHistoryEntry]:
        """List notifications that haven't been read/acknowledged."""
        ...
    
    async def list_pending(self) -> List[NotificationHistoryEntry]:
        """List notifications pending delivery (scheduled for future)."""
        ...
    
    async def mark_read(self, entry_id: str) -> bool:
        """Mark a notification as read."""
        ...
    
    async def mark_all_read(self) -> int:
        """Mark all notifications as read. Returns count updated."""
        ...
    
    async def delete_old(self, older_than: datetime) -> int:
        """Delete entries older than given date. Returns count deleted."""
        ...


class InMemoryHistoryRepository:
    """In-memory implementation of notification history repository.
    
    Suitable for testing and single-session use. Does not persist
    across application restarts.
    """
    
    def __init__(self, max_entries: int = 1000):
        self._entries: Dict[str, NotificationHistoryEntry] = {}
        self._max_entries = max_entries
    
    async def save(self, entry: NotificationHistoryEntry) -> NotificationHistoryEntry:
        """Save or update a history entry."""
        self._entries[entry.id] = entry
        
        # Trim old entries if over limit
        if len(self._entries) > self._max_entries:
            await self._trim_oldest()
        
        return entry
    
    async def get(self, entry_id: str) -> Optional[NotificationHistoryEntry]:
        """Get a history entry by ID."""
        return self._entries.get(entry_id)
    
    async def list_recent(
        self,
        limit: int = 100,
        since: Optional[datetime] = None,
        status: Optional[NotificationStatus] = None,
        source_type: Optional[str] = None,
    ) -> List[NotificationHistoryEntry]:
        """List recent notification history entries."""
        entries = list(self._entries.values())
        
        # Apply filters
        if since:
            entries = [e for e in entries if e.created_at >= since]
        if status:
            entries = [e for e in entries if e.status == status]
        if source_type:
            entries = [e for e in entries if e.source_type == source_type]
        
        # Sort by created_at descending
        entries.sort(key=lambda e: e.created_at, reverse=True)
        
        return entries[:limit]
    
    async def list_unread(self, limit: int = 50) -> List[NotificationHistoryEntry]:
        """List notifications that haven't been read/acknowledged."""
        entries = [
            e for e in self._entries.values()
            if e.read_at is None and e.status == NotificationStatus.DELIVERED
        ]
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries[:limit]
    
    async def list_pending(self) -> List[NotificationHistoryEntry]:
        """List notifications pending delivery."""
        return [
            e for e in self._entries.values()
            if e.status == NotificationStatus.PENDING
        ]
    
    async def mark_read(self, entry_id: str) -> bool:
        """Mark a notification as read."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.read_at = datetime.now(timezone.utc)
            return True
        return False
    
    async def mark_all_read(self) -> int:
        """Mark all notifications as read."""
        count = 0
        now = datetime.now(timezone.utc)
        for entry in self._entries.values():
            if entry.read_at is None:
                entry.read_at = now
                count += 1
        return count
    
    async def delete_old(self, older_than: datetime) -> int:
        """Delete entries older than given date."""
        to_delete = [
            entry_id for entry_id, entry in self._entries.items()
            if entry.created_at < older_than
        ]
        for entry_id in to_delete:
            del self._entries[entry_id]
        return len(to_delete)
    
    async def _trim_oldest(self) -> None:
        """Remove oldest entries to stay under max_entries limit."""
        if len(self._entries) <= self._max_entries:
            return
        
        entries = sorted(
            self._entries.items(),
            key=lambda x: x[1].created_at,
        )
        
        # Remove oldest entries
        to_remove = len(entries) - self._max_entries
        for entry_id, _ in entries[:to_remove]:
            del self._entries[entry_id]


class NotificationHistoryService:
    """Service for tracking notification history.
    
    Provides:
    - Recording of all sent notifications
    - Missed notification tracking
    - User interaction logging
    - History queries and cleanup
    """
    
    def __init__(
        self,
        repository: Optional[NotificationHistoryRepository] = None,
        retention_days: int = 30,
        logger: Optional[logging.Logger] = None,
    ):
        self._repository = repository or InMemoryHistoryRepository()
        self._retention_days = retention_days
        self._logger = logger or logging.getLogger(__name__)
        
        # Callbacks for notification events
        self._on_missed_callbacks: List[Callable[[List[NotificationHistoryEntry]], None]] = []
    
    async def record_sent(
        self,
        request: NotificationRequest,
        success: bool,
        notification_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> NotificationHistoryEntry:
        """Record that a notification was sent (or failed)."""
        entry = NotificationHistoryEntry(
            id=notification_id or str(uuid4()),
            title=request.title,
            message=request.message,
            channel=request.channel,
            priority=request.priority,
            source_id=request.source_id,
            source_type=request.source_type,
            status=NotificationStatus.DELIVERED if success else NotificationStatus.FAILED,
            error=error,
            created_at=datetime.now(timezone.utc),
            delivered_at=datetime.now(timezone.utc) if success else None,
            metadata=request.metadata,
        )
        
        return await self._repository.save(entry)
    
    async def record_suppressed(
        self,
        request: NotificationRequest,
        reason: str = "DND active",
        scheduled_for: Optional[datetime] = None,
    ) -> NotificationHistoryEntry:
        """Record that a notification was suppressed (DND/quiet hours)."""
        entry = NotificationHistoryEntry(
            title=request.title,
            message=request.message,
            channel=request.channel,
            priority=request.priority,
            source_id=request.source_id,
            source_type=request.source_type,
            status=NotificationStatus.SUPPRESSED if not scheduled_for else NotificationStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            original_time=datetime.now(timezone.utc),
            scheduled_for=scheduled_for,
            metadata={**request.metadata, "suppression_reason": reason},
        )
        
        return await self._repository.save(entry)
    
    async def record_interaction(
        self,
        entry_id: str,
        action: str,
        new_status: Optional[NotificationStatus] = None,
    ) -> bool:
        """Record user interaction with a notification.
        
        Args:
            entry_id: The notification ID
            action: Action taken (e.g., "clicked", "snooze_5m", "dismissed")
            new_status: Optional new status to set
        """
        entry = await self._repository.get(entry_id)
        if not entry:
            return False
        
        entry.action_taken = action
        entry.read_at = datetime.now(timezone.utc)
        
        if new_status:
            entry.status = new_status
        elif action == "clicked":
            entry.status = NotificationStatus.CLICKED
        elif action == "dismissed":
            entry.status = NotificationStatus.DISMISSED
        
        await self._repository.save(entry)
        return True
    
    async def get_missed_notifications(
        self,
        since: Optional[datetime] = None,
    ) -> List[NotificationHistoryEntry]:
        """Get notifications that were delivered but not read.
        
        Useful for showing "You missed N notifications while away" UI.
        """
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(hours=24)
        
        return await self._repository.list_unread(limit=100)
    
    async def get_pending_notifications(self) -> List[NotificationHistoryEntry]:
        """Get notifications scheduled for future delivery."""
        return await self._repository.list_pending()
    
    async def get_history(
        self,
        limit: int = 100,
        since: Optional[datetime] = None,
        source_type: Optional[str] = None,
    ) -> List[NotificationHistoryEntry]:
        """Get notification history."""
        return await self._repository.list_recent(
            limit=limit,
            since=since,
            source_type=source_type,
        )
    
    async def mark_read(self, entry_id: str) -> bool:
        """Mark a notification as read."""
        return await self._repository.mark_read(entry_id)
    
    async def mark_all_read(self) -> int:
        """Mark all notifications as read."""
        return await self._repository.mark_all_read()
    
    async def cleanup_old_entries(self) -> int:
        """Remove entries older than retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        deleted = await self._repository.delete_old(cutoff)
        
        if deleted > 0:
            self._logger.info(f"Cleaned up {deleted} old notification history entries")
        
        return deleted
    
    def on_missed_notifications(
        self,
        callback: Callable[[List[NotificationHistoryEntry]], None],
    ) -> None:
        """Register callback for when missed notifications are detected.
        
        Callback receives list of missed notification entries.
        """
        self._on_missed_callbacks.append(callback)
    
    async def check_missed_and_notify(self) -> List[NotificationHistoryEntry]:
        """Check for missed notifications and invoke callbacks.
        
        Call this when user returns from being away (e.g., session resume).
        """
        missed = await self.get_missed_notifications()
        
        if missed and self._on_missed_callbacks:
            for callback in self._on_missed_callbacks:
                try:
                    callback(missed)
                except Exception as e:
                    self._logger.error(f"Error in missed notification callback: {e}")
        
        return missed
    
    async def get_summary(self) -> Dict[str, Any]:
        """Get a summary of notification activity.
        
        Returns counts and stats useful for UI display.
        """
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_ago = now - timedelta(days=7)
        
        today_entries = await self._repository.list_recent(limit=1000, since=today)
        week_entries = await self._repository.list_recent(limit=1000, since=week_ago)
        unread = await self._repository.list_unread()
        pending = await self._repository.list_pending()
        
        def count_by_status(entries: List[NotificationHistoryEntry]) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for entry in entries:
                status = entry.status.value
                counts[status] = counts.get(status, 0) + 1
            return counts
        
        return {
            "unread_count": len(unread),
            "pending_count": len(pending),
            "today": {
                "total": len(today_entries),
                "by_status": count_by_status(today_entries),
            },
            "week": {
                "total": len(week_entries),
                "by_status": count_by_status(week_entries),
            },
        }
