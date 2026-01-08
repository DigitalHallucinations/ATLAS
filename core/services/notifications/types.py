"""
Notification types and data structures.

Defines the domain types for notifications used by the notification services.

Author: ATLAS Team  
Date: Jan 8, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4


class NotificationChannel(str, Enum):
    """Notification delivery channels."""
    DESKTOP = "desktop"          # Desktop notification (libnotify/GNotification)
    POPUP = "popup"              # In-app popup dialog
    SOUND = "sound"              # Audio alert
    SPEECH = "speech"            # Text-to-speech
    EMAIL = "email"              # Email notification
    MOBILE_PUSH = "mobile_push"  # Mobile push (NTFY/Pushover/Gotify)
    WEBHOOK = "webhook"          # HTTP webhook
    LOG = "log"                  # Log only (for testing/debugging)


class NotificationPriority(str, Enum):
    """Notification urgency levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"  # Requires immediate attention


@dataclass
class NotificationRequest:
    """Request to send a notification."""
    
    title: str
    message: str
    channel: NotificationChannel = NotificationChannel.DESKTOP
    priority: NotificationPriority = NotificationPriority.NORMAL
    icon: Optional[str] = None  # Icon name or path
    actions: Optional[List[Dict[str, str]]] = None  # Action buttons
    timeout_ms: int = 5000  # Auto-dismiss timeout
    category: Optional[str] = None  # Notification category for grouping
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Context for tracking
    source_id: Optional[str] = None  # ID of source entity (e.g., reminder_id)
    source_type: Optional[str] = None  # Type of source (e.g., "reminder", "job")


@dataclass
class NotificationResult:
    """Result of a notification delivery attempt."""
    
    success: bool
    notification_id: Optional[str] = None  # Platform-specific ID
    channel: NotificationChannel = NotificationChannel.DESKTOP
    error: Optional[str] = None
    delivered_at: Optional[datetime] = None
    
    @classmethod
    def ok(
        cls,
        channel: NotificationChannel,
        notification_id: Optional[str] = None
    ) -> "NotificationResult":
        """Create a successful result."""
        return cls(
            success=True,
            notification_id=notification_id,
            channel=channel,
            delivered_at=datetime.now(timezone.utc),
        )
    
    @classmethod
    def failed(
        cls,
        channel: NotificationChannel,
        error: str
    ) -> "NotificationResult":
        """Create a failed result."""
        return cls(
            success=False,
            channel=channel,
            error=error,
        )


# =============================================================================
# Action Buttons (for interactive notifications)
# =============================================================================

@dataclass
class NotificationAction:
    """An action button that can appear on a notification."""
    
    action_id: str  # Unique ID for this action (e.g., "snooze_5m")
    label: str      # Display text (e.g., "Snooze 5 min")
    icon: Optional[str] = None  # Optional icon name
    
    # Callback invoked when user clicks this action
    # Signature: callback(notification_id: str, action_id: str, user_data: dict)
    callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None
    
    # Data passed to the callback
    user_data: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Do Not Disturb / Quiet Hours
# =============================================================================

class DNDMode(str, Enum):
    """Do Not Disturb mode states."""
    OFF = "off"                    # Normal operation
    ENABLED = "enabled"            # DND active (all non-critical suppressed)
    SCHEDULED = "scheduled"        # DND follows quiet hours schedule
    FOCUS = "focus"                # Focus mode (temporary DND)


@dataclass
class QuietHours:
    """Quiet hours configuration for a day or default."""
    
    start_time: time  # When quiet hours begin (e.g., 22:00)
    end_time: time    # When quiet hours end (e.g., 07:00)
    enabled: bool = True
    
    # Override for specific days (0=Monday, 6=Sunday)
    # If None, applies to all days
    days_of_week: Optional[Set[int]] = None
    
    def is_quiet_now(self, current_time: Optional[datetime] = None) -> bool:
        """Check if we're currently in quiet hours."""
        if not self.enabled:
            return False
        
        now = current_time or datetime.now()
        current = now.time()
        
        # Check day of week if specified
        if self.days_of_week is not None:
            if now.weekday() not in self.days_of_week:
                return False
        
        # Handle overnight quiet hours (e.g., 22:00 - 07:00)
        if self.start_time > self.end_time:
            return current >= self.start_time or current < self.end_time
        else:
            return self.start_time <= current < self.end_time


@dataclass
class DNDConfig:
    """Do Not Disturb configuration."""
    
    mode: DNDMode = DNDMode.OFF
    quiet_hours: List[QuietHours] = field(default_factory=list)
    
    # Channels that bypass DND (critical always bypasses)
    bypass_channels: Set[NotificationChannel] = field(default_factory=set)
    
    # Focus mode settings
    focus_until: Optional[datetime] = None  # When focus mode ends
    focus_reason: Optional[str] = None      # "Meeting", "Deep work", etc.
    
    def should_suppress(
        self,
        priority: NotificationPriority,
        channel: NotificationChannel,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """Check if a notification should be suppressed.
        
        Returns True if notification should be queued/suppressed.
        Critical priority notifications are NEVER suppressed.
        """
        # Critical notifications always go through
        if priority == NotificationPriority.CRITICAL:
            return False
        
        # Check bypass channels
        if channel in self.bypass_channels:
            return False
        
        now = current_time or datetime.now(timezone.utc)
        
        # Check mode
        if self.mode == DNDMode.OFF:
            return False
        
        if self.mode == DNDMode.ENABLED:
            return True
        
        if self.mode == DNDMode.FOCUS:
            if self.focus_until and now < self.focus_until:
                return True
            return False
        
        if self.mode == DNDMode.SCHEDULED:
            # Check all quiet hours rules
            for qh in self.quiet_hours:
                if qh.is_quiet_now(now):
                    return True
            return False
        
        return False


# =============================================================================
# Notification History
# =============================================================================

class NotificationStatus(str, Enum):
    """Status of a notification in the history."""
    PENDING = "pending"      # Queued but not yet sent
    DELIVERED = "delivered"  # Successfully delivered
    FAILED = "failed"        # Delivery failed
    SUPPRESSED = "suppressed"  # Suppressed by DND
    CLICKED = "clicked"      # User clicked/interacted
    DISMISSED = "dismissed"  # User dismissed
    EXPIRED = "expired"      # Auto-dismissed after timeout


@dataclass
class NotificationHistoryEntry:
    """A record of a sent or queued notification."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    
    # Notification content
    title: str = ""
    message: str = ""
    channel: NotificationChannel = NotificationChannel.DESKTOP
    priority: NotificationPriority = NotificationPriority.NORMAL
    
    # Source tracking
    source_id: Optional[str] = None
    source_type: Optional[str] = None
    
    # Status tracking
    status: NotificationStatus = NotificationStatus.PENDING
    error: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    
    # Smart scheduling
    scheduled_for: Optional[datetime] = None  # When to deliver (if queued)
    original_time: Optional[datetime] = None  # When it was originally meant to fire
    
    # User interaction
    action_taken: Optional[str] = None  # Which action button was clicked
    
    # Full request preserved for retry
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Smart Scheduling
# =============================================================================

@dataclass
class SchedulingConfig:
    """Configuration for smart notification scheduling."""
    
    enabled: bool = True
    
    # Default delivery window (when non-critical notifications can be sent)
    delivery_start: time = field(default_factory=lambda: time(7, 0))   # 7:00 AM
    delivery_end: time = field(default_factory=lambda: time(22, 0))    # 10:00 PM
    
    # Batching settings
    batch_interval_seconds: int = 300  # Batch notifications every 5 minutes
    max_batch_size: int = 10           # Max notifications per batch
    
    # Rate limiting
    max_per_minute: int = 10           # Max notifications per minute
    cooldown_seconds: int = 5          # Min time between notifications
    
    def is_delivery_window_open(
        self,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """Check if we're in the delivery window."""
        now = current_time or datetime.now()
        current = now.time()
        
        # Handle overnight window (e.g., if someone sets 22:00 - 07:00)
        if self.delivery_start > self.delivery_end:
            return current >= self.delivery_start or current < self.delivery_end
        else:
            return self.delivery_start <= current < self.delivery_end
    
    def next_delivery_time(
        self,
        current_time: Optional[datetime] = None,
    ) -> datetime:
        """Calculate when the next delivery window opens."""
        now = current_time or datetime.now()
        
        if self.is_delivery_window_open(now):
            return now
        
        # Calculate next delivery start
        today_start = datetime.combine(now.date(), self.delivery_start)
        
        if now.time() < self.delivery_start:
            # Before today's window - deliver at today's start
            return today_start
        else:
            # After today's window - deliver at tomorrow's start
            from datetime import timedelta
            return today_start + timedelta(days=1)


# =============================================================================
# Email & Mobile Push Configuration (Stubs)
# =============================================================================

@dataclass
class EmailConfig:
    """Configuration for email notifications.
    
    TODO: Integrate with core email service when available.
    """
    enabled: bool = False
    recipient: Optional[str] = None  # Default recipient email
    
    # SMTP settings (placeholder - will use core email service)
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    # Note: smtp_password should come from secure credential store
    use_tls: bool = True
    
    # Templates
    subject_template: str = "[ATLAS] {title}"
    body_template: str = "{message}"


@dataclass  
class MobilePushConfig:
    """Configuration for mobile push notifications.
    
    TODO: Implement providers when ready.
    Supported providers (planned):
    - NTFY (self-hosted, open source)
    - Pushover (commercial)
    - Gotify (self-hosted)
    """
    enabled: bool = False
    provider: Optional[str] = None  # "ntfy", "pushover", "gotify"
    
    # Provider-specific settings
    server_url: Optional[str] = None  # For NTFY/Gotify
    topic: Optional[str] = None       # NTFY topic or Gotify app token
    # Note: API keys should come from secure credential store
    
    # Priority mapping
    priority_map: Dict[str, int] = field(default_factory=lambda: {
        "low": 1,
        "normal": 3,
        "high": 4,
        "critical": 5,
    })
