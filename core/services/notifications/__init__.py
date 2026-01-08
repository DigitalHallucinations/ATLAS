"""
ATLAS Notification Services

Provides notification delivery across multiple channels including
desktop notifications, audio alerts, speech synthesis, email,
and mobile push with intelligent scheduling and DND support.

Author: Jeremy Shows Digital Hallucinations  
Date: Jan 8, 2026
"""

from .notification_service import (
    NotificationService,
    BaseNotificationService,
    DesktopNotificationService,
    DummyNotificationService,
    CompositeNotificationService,
    create_notification_service,
)
from .types import (
    # Channels and priorities
    NotificationChannel,
    NotificationPriority,
    NotificationRequest,
    NotificationResult,
    # Actions
    NotificationAction,
    # History
    NotificationStatus,
    NotificationHistoryEntry,
    # DND / Quiet Hours
    DNDMode,
    DNDConfig,
    QuietHours,
    # Scheduling
    SchedulingConfig,
    # Config types
    EmailConfig,
    MobilePushConfig,
)
from .history import (
    NotificationHistoryService,
    NotificationHistoryRepository,
    InMemoryHistoryRepository,
)
from .scheduler import (
    NotificationScheduler,
    QueuedNotification,
    create_default_quiet_hours,
    create_weekday_quiet_hours,
)
from .actions import (
    DBusActionHandler,
    create_reminder_actions,
    create_calendar_event_actions,
    create_job_alert_actions,
)
from .email_service import EmailNotificationService
from .mobile_push import MobilePushNotificationService, MultiProviderPushService

__all__ = [
    # Core Services
    "NotificationService",
    "BaseNotificationService",
    "DesktopNotificationService",
    "DummyNotificationService",
    "CompositeNotificationService",
    "create_notification_service",
    
    # History
    "NotificationHistoryService",
    "NotificationHistoryRepository",
    "InMemoryHistoryRepository",
    
    # Scheduler
    "NotificationScheduler",
    "QueuedNotification",
    "create_default_quiet_hours",
    "create_weekday_quiet_hours",
    
    # Actions
    "DBusActionHandler",
    "create_reminder_actions",
    "create_calendar_event_actions",
    "create_job_alert_actions",
    
    # Channel Services (stubs)
    "EmailNotificationService",
    "MobilePushNotificationService",
    "MultiProviderPushService",
    
    # Types - Channels & Priorities
    "NotificationChannel",
    "NotificationPriority",
    "NotificationRequest",
    "NotificationResult",
    
    # Types - Actions
    "NotificationAction",
    
    # Types - History
    "NotificationStatus",
    "NotificationHistoryEntry",
    
    # Types - DND / Quiet Hours
    "DNDMode",
    "DNDConfig",
    "QuietHours",
    
    # Types - Scheduling
    "SchedulingConfig",
    
    # Types - Config
    "EmailConfig",
    "MobilePushConfig",
]
