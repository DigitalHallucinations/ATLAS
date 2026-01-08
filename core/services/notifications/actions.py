"""
DBus notification action handling.

Provides interactive notification support with action buttons
(Snooze, Dismiss, Open, etc.) via DBus signals.

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from .types import NotificationAction


logger = logging.getLogger(__name__)


# Type for action callbacks
ActionCallback = Callable[[str, str, Dict[str, Any]], None]


@dataclass
class PendingNotification:
    """Tracks a notification waiting for user action."""
    
    notification_id: str
    dbus_id: Optional[int] = None  # The ID returned by DBus
    actions: List[NotificationAction] = field(default_factory=list)
    user_data: Dict[str, Any] = field(default_factory=dict)


class DBusActionHandler:
    """Handles DBus notification action signals.
    
    Connects to the org.freedesktop.Notifications interface to receive
    action callbacks when users click notification buttons.
    
    Example usage:
        handler = DBusActionHandler()
        await handler.connect()
        
        # Register actions for a notification
        handler.register_notification(
            notification_id="reminder-123",
            actions=[
                NotificationAction("snooze_5m", "Snooze 5 min"),
                NotificationAction("dismiss", "Dismiss"),
            ],
            callback=on_action,
        )
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(__name__)
        self._connected = False
        self._bus = None
        self._proxy = None
        
        # Track pending notifications and their callbacks
        self._pending: Dict[str, PendingNotification] = {}
        self._dbus_id_map: Dict[int, str] = {}  # DBus ID -> our notification ID
        
        # Global callback for all actions (optional)
        self._global_callback: Optional[ActionCallback] = None
        
        # Default action handlers
        self._default_handlers: Dict[str, ActionCallback] = {}
    
    async def connect(self) -> bool:
        """Connect to the DBus session bus and notifications interface.
        
        Returns True if connected successfully.
        """
        try:
            import gi
            gi.require_version("Gio", "2.0")
            from gi.repository import Gio, GLib  # type: ignore[import-untyped]
            
            self._Gio = Gio
            self._GLib = GLib
            
            # Get the session bus
            self._bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            
            # Create a proxy for the notifications service
            self._proxy = Gio.DBusProxy.new_sync(
                self._bus,
                Gio.DBusProxyFlags.NONE,
                None,
                "org.freedesktop.Notifications",
                "/org/freedesktop/Notifications",
                "org.freedesktop.Notifications",
                None,
            )
            
            # Subscribe to signals
            self._bus.signal_subscribe(
                "org.freedesktop.Notifications",
                "org.freedesktop.Notifications",
                "ActionInvoked",
                "/org/freedesktop/Notifications",
                None,
                Gio.DBusSignalFlags.NONE,
                self._on_action_invoked,
                None,
            )
            
            self._bus.signal_subscribe(
                "org.freedesktop.Notifications",
                "org.freedesktop.Notifications",
                "NotificationClosed",
                "/org/freedesktop/Notifications",
                None,
                Gio.DBusSignalFlags.NONE,
                self._on_notification_closed,
                None,
            )
            
            self._connected = True
            self._logger.info("Connected to DBus notifications service")
            return True
            
        except Exception as e:
            self._logger.warning(f"Failed to connect to DBus: {e}")
            self._connected = False
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to DBus."""
        return self._connected
    
    def register_notification(
        self,
        notification_id: str,
        dbus_id: Optional[int] = None,
        actions: Optional[List[NotificationAction]] = None,
        user_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a notification for action tracking.
        
        Args:
            notification_id: Our internal notification ID
            dbus_id: The ID returned by DBus Notify call
            actions: List of action buttons on this notification
            user_data: Data to pass to action callbacks
        """
        pending = PendingNotification(
            notification_id=notification_id,
            dbus_id=dbus_id,
            actions=actions or [],
            user_data=user_data or {},
        )
        
        self._pending[notification_id] = pending
        
        if dbus_id is not None:
            self._dbus_id_map[dbus_id] = notification_id
    
    def set_global_callback(self, callback: ActionCallback) -> None:
        """Set a global callback for all notification actions.
        
        Args:
            callback: Function(notification_id, action_id, user_data)
        """
        self._global_callback = callback
    
    def register_default_handler(
        self,
        action_id: str,
        handler: ActionCallback,
    ) -> None:
        """Register a default handler for a specific action type.
        
        Args:
            action_id: The action ID (e.g., "snooze_5m", "dismiss")
            handler: Function(notification_id, action_id, user_data)
        """
        self._default_handlers[action_id] = handler
    
    def _on_action_invoked(
        self,
        connection: Any,
        sender: str,
        path: str,
        interface: str,
        signal: str,
        params: Any,
        user_data: Any,
    ) -> None:
        """Handle DBus ActionInvoked signal."""
        try:
            dbus_id = params[0]
            action_key = params[1]
            
            # Look up our notification
            notification_id = self._dbus_id_map.get(dbus_id)
            if not notification_id:
                self._logger.debug(f"Unknown notification action: {dbus_id}")
                return
            
            pending = self._pending.get(notification_id)
            if not pending:
                return
            
            self._logger.info(
                f"Notification action: {notification_id} -> {action_key}"
            )
            
            # Find the action and its callback
            action = next(
                (a for a in pending.actions if a.action_id == action_key),
                None,
            )
            
            # Invoke callbacks
            callback_data = {**pending.user_data}
            if action and action.user_data:
                callback_data.update(action.user_data)
            
            # Action-specific callback
            if action and action.callback:
                try:
                    action.callback(notification_id, action_key, callback_data)
                except Exception as e:
                    self._logger.error(f"Action callback error: {e}")
            
            # Default handler for this action type
            if action_key in self._default_handlers:
                try:
                    self._default_handlers[action_key](
                        notification_id, action_key, callback_data
                    )
                except Exception as e:
                    self._logger.error(f"Default handler error: {e}")
            
            # Global callback
            if self._global_callback:
                try:
                    self._global_callback(notification_id, action_key, callback_data)
                except Exception as e:
                    self._logger.error(f"Global callback error: {e}")
            
            # Clean up
            self._cleanup_notification(notification_id)
            
        except Exception as e:
            self._logger.error(f"Error handling action signal: {e}")
    
    def _on_notification_closed(
        self,
        connection: Any,
        sender: str,
        path: str,
        interface: str,
        signal: str,
        params: Any,
        user_data: Any,
    ) -> None:
        """Handle DBus NotificationClosed signal."""
        try:
            dbus_id = params[0]
            reason = params[1]  # 1=expired, 2=dismissed, 3=closed, 4=undefined
            
            notification_id = self._dbus_id_map.get(dbus_id)
            if notification_id:
                reason_str = {
                    1: "expired",
                    2: "dismissed",
                    3: "closed",
                    4: "undefined",
                }.get(reason, "unknown")
                
                self._logger.debug(
                    f"Notification closed: {notification_id} ({reason_str})"
                )
                
                # Invoke global callback with dismiss action if dismissed by user
                if reason == 2 and self._global_callback:
                    pending = self._pending.get(notification_id)
                    try:
                        self._global_callback(
                            notification_id,
                            "dismissed",
                            pending.user_data if pending else {},
                        )
                    except Exception as e:
                        self._logger.error(f"Dismiss callback error: {e}")
                
                self._cleanup_notification(notification_id)
                
        except Exception as e:
            self._logger.error(f"Error handling closed signal: {e}")
    
    def _cleanup_notification(self, notification_id: str) -> None:
        """Clean up tracking for a notification."""
        pending = self._pending.pop(notification_id, None)
        if pending and pending.dbus_id is not None:
            self._dbus_id_map.pop(pending.dbus_id, None)
    
    async def send_with_actions(
        self,
        title: str,
        message: str,
        actions: List[NotificationAction],
        app_name: str = "ATLAS",
        icon: str = "",
        timeout_ms: int = -1,  # -1 = server default
        user_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Send a notification with action buttons via DBus.
        
        Returns the notification ID if successful, None otherwise.
        """
        if not self._connected:
            self._logger.warning("Not connected to DBus")
            return None
        
        try:
            # Build actions array for DBus: [id1, label1, id2, label2, ...]
            actions_array = []
            for action in actions:
                actions_array.append(action.action_id)
                actions_array.append(action.label)
            
            # Add default action (clicking notification body)
            if not any(a.action_id == "default" for a in actions):
                actions_array.insert(0, "default")
                actions_array.insert(1, "Open")
            
            if self._proxy is None:
                self._logger.error("DBus proxy not initialized")
                return None
            
            # Call Notify via DBus
            result = self._proxy.call_sync(
                "Notify",
                self._GLib.Variant(
                    "(susssasa{sv}i)",
                    (
                        app_name,           # app_name
                        0,                  # replaces_id (0 = new notification)
                        icon,               # app_icon
                        title,              # summary
                        message,            # body
                        actions_array,      # actions
                        {},                 # hints
                        timeout_ms,         # expire_timeout
                    ),
                ),
                self._Gio.DBusCallFlags.NONE,
                -1,
                None,
            )
            
            dbus_id = result.unpack()[0]
            notification_id = str(uuid4())
            
            # Register for action tracking
            self.register_notification(
                notification_id=notification_id,
                dbus_id=dbus_id,
                actions=actions,
                user_data=user_data,
            )
            
            self._logger.debug(
                f"Sent notification with actions: {notification_id} (dbus_id={dbus_id})"
            )
            
            return notification_id
            
        except Exception as e:
            self._logger.error(f"Failed to send notification with actions: {e}")
            return None


# =============================================================================
# Common action button presets
# =============================================================================

def create_reminder_actions(
    snooze_callback: Optional[ActionCallback] = None,
    dismiss_callback: Optional[ActionCallback] = None,
    open_callback: Optional[ActionCallback] = None,
) -> List[NotificationAction]:
    """Create standard reminder notification actions.
    
    Returns actions for: Snooze 5 min, Snooze 15 min, Dismiss
    """
    return [
        NotificationAction(
            action_id="snooze_5m",
            label="Snooze 5 min",
            callback=snooze_callback,
            user_data={"snooze_minutes": 5},
        ),
        NotificationAction(
            action_id="snooze_15m",
            label="Snooze 15 min",
            callback=snooze_callback,
            user_data={"snooze_minutes": 15},
        ),
        NotificationAction(
            action_id="dismiss",
            label="Dismiss",
            callback=dismiss_callback,
        ),
    ]


def create_calendar_event_actions(
    open_callback: Optional[ActionCallback] = None,
    snooze_callback: Optional[ActionCallback] = None,
) -> List[NotificationAction]:
    """Create calendar event notification actions.
    
    Returns actions for: Open Event, Snooze
    """
    return [
        NotificationAction(
            action_id="open_event",
            label="Open Event",
            callback=open_callback,
        ),
        NotificationAction(
            action_id="snooze_5m",
            label="Snooze",
            callback=snooze_callback,
            user_data={"snooze_minutes": 5},
        ),
    ]


def create_job_alert_actions(
    view_callback: Optional[ActionCallback] = None,
    acknowledge_callback: Optional[ActionCallback] = None,
) -> List[NotificationAction]:
    """Create job/task alert notification actions.
    
    Returns actions for: View Details, Acknowledge
    """
    return [
        NotificationAction(
            action_id="view_job",
            label="View Details",
            callback=view_callback,
        ),
        NotificationAction(
            action_id="acknowledge",
            label="Acknowledge",
            callback=acknowledge_callback,
        ),
    ]
