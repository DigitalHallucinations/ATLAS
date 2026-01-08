"""
Smart notification scheduler.

Provides intelligent scheduling of notifications with:
- Do Not Disturb / Quiet Hours support
- Smart queueing for non-critical notifications
- Rate limiting and batching
- Automatic rescheduling when delivery window opens

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol

from .types import (
    DNDConfig,
    DNDMode,
    NotificationChannel,
    NotificationPriority,
    NotificationRequest,
    QuietHours,
    SchedulingConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class QueuedNotification:
    """A notification waiting to be delivered."""
    
    request: NotificationRequest
    queued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_for: Optional[datetime] = None
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    
    @property
    def age_seconds(self) -> float:
        """How long this notification has been queued."""
        return (datetime.now(timezone.utc) - self.queued_at).total_seconds()


class NotificationScheduler:
    """Smart notification scheduler with DND and queueing.
    
    Handles:
    - Do Not Disturb mode checking
    - Quiet hours enforcement
    - Smart queueing of non-critical notifications
    - Automatic delivery when window opens
    - Rate limiting
    """
    
    def __init__(
        self,
        dnd_config: Optional[DNDConfig] = None,
        scheduling_config: Optional[SchedulingConfig] = None,
        delivery_callback: Optional[Callable[[NotificationRequest], bool]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._dnd = dnd_config or DNDConfig()
        self._scheduling = scheduling_config or SchedulingConfig()
        self._deliver = delivery_callback
        self._logger = logger or logging.getLogger(__name__)
        
        # Notification queue for delayed delivery
        self._queue: Deque[QueuedNotification] = deque()
        
        # Rate limiting state
        self._recent_sends: Deque[datetime] = deque(maxlen=100)
        self._last_send: Optional[datetime] = None
        
        # Background task
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False
    
    # =========================================================================
    # DND Management
    # =========================================================================
    
    def enable_dnd(self) -> None:
        """Enable Do Not Disturb mode."""
        self._dnd.mode = DNDMode.ENABLED
        self._logger.info("Do Not Disturb enabled")
    
    def disable_dnd(self) -> None:
        """Disable Do Not Disturb mode."""
        self._dnd.mode = DNDMode.OFF
        self._dnd.focus_until = None
        self._dnd.focus_reason = None
        self._logger.info("Do Not Disturb disabled")
    
    def enable_focus_mode(
        self,
        duration_minutes: int = 60,
        reason: Optional[str] = None,
    ) -> datetime:
        """Enable focus mode for specified duration.
        
        Returns the time when focus mode will end.
        """
        self._dnd.mode = DNDMode.FOCUS
        self._dnd.focus_until = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        self._dnd.focus_reason = reason
        
        self._logger.info(
            f"Focus mode enabled until {self._dnd.focus_until}"
            f"{f' ({reason})' if reason else ''}"
        )
        
        return self._dnd.focus_until
    
    def enable_scheduled_dnd(self, quiet_hours: List[QuietHours]) -> None:
        """Enable scheduled DND with quiet hours rules."""
        self._dnd.mode = DNDMode.SCHEDULED
        self._dnd.quiet_hours = quiet_hours
        self._logger.info(f"Scheduled DND enabled with {len(quiet_hours)} quiet hours rules")
    
    def get_dnd_status(self) -> Dict[str, Any]:
        """Get current DND status."""
        return {
            "mode": self._dnd.mode.value,
            "is_active": self._is_dnd_active(),
            "focus_until": self._dnd.focus_until.isoformat() if self._dnd.focus_until else None,
            "focus_reason": self._dnd.focus_reason,
            "quiet_hours_count": len(self._dnd.quiet_hours),
            "bypass_channels": [c.value for c in self._dnd.bypass_channels],
        }
    
    def _is_dnd_active(self) -> bool:
        """Check if DND is currently active."""
        return self._dnd.should_suppress(
            NotificationPriority.NORMAL,
            NotificationChannel.DESKTOP,
        )
    
    # =========================================================================
    # Scheduling
    # =========================================================================
    
    async def schedule(
        self,
        request: NotificationRequest,
    ) -> tuple[bool, Optional[datetime]]:
        """Schedule a notification for delivery.
        
        Returns:
            Tuple of (delivered_immediately, scheduled_for_time)
            - (True, None) if delivered immediately
            - (False, datetime) if queued for later
            - (False, None) if suppressed entirely
        """
        # Critical notifications always go through immediately
        if request.priority == NotificationPriority.CRITICAL:
            success = await self._deliver_now(request)
            return (success, None)
        
        # Check DND
        if self._dnd.should_suppress(request.priority, request.channel):
            # Queue for later if scheduling is enabled
            if self._scheduling.enabled:
                scheduled_for = self._scheduling.next_delivery_time()
                self._queue.append(QueuedNotification(
                    request=request,
                    scheduled_for=scheduled_for,
                ))
                self._logger.debug(
                    f"Notification queued until {scheduled_for}: {request.title}"
                )
                return (False, scheduled_for)
            else:
                # Suppressed entirely
                self._logger.debug(f"Notification suppressed by DND: {request.title}")
                return (False, None)
        
        # Check delivery window
        if self._scheduling.enabled and not self._scheduling.is_delivery_window_open():
            scheduled_for = self._scheduling.next_delivery_time()
            self._queue.append(QueuedNotification(
                request=request,
                scheduled_for=scheduled_for,
            ))
            self._logger.debug(
                f"Notification queued (outside delivery window) until {scheduled_for}: {request.title}"
            )
            return (False, scheduled_for)
        
        # Check rate limiting
        if not self._can_send_now():
            # Queue for slight delay
            scheduled_for = datetime.now(timezone.utc) + timedelta(
                seconds=self._scheduling.cooldown_seconds
            )
            self._queue.append(QueuedNotification(
                request=request,
                scheduled_for=scheduled_for,
            ))
            return (False, scheduled_for)
        
        # Deliver immediately
        success = await self._deliver_now(request)
        return (success, None)
    
    async def _deliver_now(self, request: NotificationRequest) -> bool:
        """Deliver a notification immediately."""
        if not self._deliver:
            self._logger.warning("No delivery callback configured")
            return False
        
        try:
            # Record send time for rate limiting
            now = datetime.now(timezone.utc)
            self._recent_sends.append(now)
            self._last_send = now
            
            # Deliver
            if asyncio.iscoroutinefunction(self._deliver):
                return await self._deliver(request)
            else:
                return self._deliver(request)
                
        except Exception as e:
            self._logger.error(f"Failed to deliver notification: {e}")
            return False
    
    def _can_send_now(self) -> bool:
        """Check if we can send a notification (rate limiting)."""
        now = datetime.now(timezone.utc)
        
        # Check cooldown
        if self._last_send:
            elapsed = (now - self._last_send).total_seconds()
            if elapsed < self._scheduling.cooldown_seconds:
                return False
        
        # Check rate limit
        minute_ago = now - timedelta(minutes=1)
        recent_count = sum(1 for t in self._recent_sends if t > minute_ago)
        if recent_count >= self._scheduling.max_per_minute:
            return False
        
        return True
    
    # =========================================================================
    # Queue Processing
    # =========================================================================
    
    async def start_processor(self) -> None:
        """Start the background queue processor."""
        if self._running:
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_loop())
        self._logger.info("Notification scheduler started")
    
    async def stop_processor(self) -> None:
        """Stop the background queue processor."""
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None
        
        self._logger.info("Notification scheduler stopped")
    
    async def _process_loop(self) -> None:
        """Background loop to process queued notifications."""
        while self._running:
            try:
                await self._process_queue()
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in notification processor: {e}")
                await asyncio.sleep(30)
    
    async def _process_queue(self) -> None:
        """Process pending notifications in the queue."""
        if not self._queue:
            return
        
        now = datetime.now(timezone.utc)
        to_deliver: List[QueuedNotification] = []
        remaining: Deque[QueuedNotification] = deque()
        
        # Find notifications ready for delivery
        while self._queue:
            queued = self._queue.popleft()
            
            # Check if scheduled time has passed
            if queued.scheduled_for and queued.scheduled_for > now:
                remaining.append(queued)
                continue
            
            # Check if DND still blocks it
            if self._dnd.should_suppress(
                queued.request.priority,
                queued.request.channel,
            ):
                # Reschedule
                queued.scheduled_for = self._scheduling.next_delivery_time()
                remaining.append(queued)
                continue
            
            # Ready to deliver
            to_deliver.append(queued)
        
        # Put remaining back
        self._queue = remaining
        
        # Deliver ready notifications (with rate limiting)
        for queued in to_deliver:
            if not self._can_send_now():
                # Re-queue with short delay
                queued.scheduled_for = now + timedelta(
                    seconds=self._scheduling.cooldown_seconds
                )
                self._queue.append(queued)
                continue
            
            queued.attempts += 1
            queued.last_attempt = now
            
            success = await self._deliver_now(queued.request)
            
            if not success and queued.attempts < 3:
                # Retry later
                queued.scheduled_for = now + timedelta(minutes=1)
                self._queue.append(queued)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queue_length": len(self._queue),
            "oldest_queued_seconds": (
                max(q.age_seconds for q in self._queue) if self._queue else 0
            ),
            "is_running": self._running,
        }
    
    def flush_queue(self) -> int:
        """Clear all queued notifications. Returns count cleared."""
        count = len(self._queue)
        self._queue.clear()
        return count


def create_default_quiet_hours() -> List[QuietHours]:
    """Create sensible default quiet hours configuration.
    
    Default: 10 PM - 7 AM, all days
    """
    from datetime import time
    
    return [
        QuietHours(
            start_time=time(22, 0),  # 10 PM
            end_time=time(7, 0),     # 7 AM
            enabled=True,
        )
    ]


def create_weekday_quiet_hours() -> List[QuietHours]:
    """Create quiet hours for work week.
    
    Weekdays: 10 PM - 7 AM
    Weekends: 11 PM - 9 AM
    """
    from datetime import time
    
    return [
        # Weekday quiet hours (Mon-Fri: 0-4)
        QuietHours(
            start_time=time(22, 0),
            end_time=time(7, 0),
            enabled=True,
            days_of_week={0, 1, 2, 3, 4},
        ),
        # Weekend quiet hours (Sat-Sun: 5-6)
        QuietHours(
            start_time=time(23, 0),
            end_time=time(9, 0),
            enabled=True,
            days_of_week={5, 6},
        ),
    ]
