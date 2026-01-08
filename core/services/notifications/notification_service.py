"""
Notification service implementations.

Provides notification delivery through various channels including desktop
notifications (via GLib/Gio), audio alerts, and speech synthesis.

Author: ATLAS Team  
Date: Jan 8, 2026
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol
from uuid import uuid4

from .types import (
    NotificationChannel,
    NotificationPriority,
    NotificationRequest,
    NotificationResult,
)


logger = logging.getLogger(__name__)


class NotificationService(Protocol):
    """Protocol defining the notification service interface.
    
    This matches the protocol expected by ReminderService for
    delivering reminder notifications.
    """
    
    async def send_notification(
        self,
        method: str,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send a notification via the specified method.
        
        Args:
            method: Delivery method (notification, email, popup, sound, speech)
            title: Notification title
            message: Notification body
            metadata: Additional context for the notification
            
        Returns:
            True if notification was delivered successfully
        """
        ...


class BaseNotificationService(ABC):
    """Abstract base class for notification services."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    async def send_notification(
        self,
        method: str,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send a notification."""
        ...
    
    async def send(self, request: NotificationRequest) -> NotificationResult:
        """Send a notification using a structured request.
        
        Args:
            request: Notification request with all parameters
            
        Returns:
            NotificationResult with delivery status
        """
        success = await self.send_notification(
            method=request.channel.value,
            title=request.title,
            message=request.message,
            metadata={
                "priority": request.priority.value,
                "icon": request.icon,
                "timeout_ms": request.timeout_ms,
                "category": request.category,
                "source_id": request.source_id,
                "source_type": request.source_type,
                **request.metadata,
            },
        )
        
        if success:
            return NotificationResult.ok(
                channel=request.channel,
                notification_id=str(uuid4()),
            )
        else:
            return NotificationResult.failed(
                channel=request.channel,
                error="Notification delivery failed",
            )


class DummyNotificationService(BaseNotificationService):
    """Stub notification service that logs but doesn't deliver.
    
    Used for testing and environments without desktop notification support.
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        record_notifications: bool = False,
    ):
        super().__init__(logger)
        self._record = record_notifications
        self._notifications: List[Dict[str, Any]] = []
    
    async def send_notification(
        self,
        method: str,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Log the notification without actually delivering it."""
        notification = {
            "method": method,
            "title": title,
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        if self._record:
            self._notifications.append(notification)
        
        self._logger.info(
            f"[DummyNotification] {method}: {title} - {message}"
        )
        
        # Always succeed for testing purposes
        return True
    
    def get_notifications(self) -> List[Dict[str, Any]]:
        """Get recorded notifications (for testing)."""
        return self._notifications.copy()
    
    def clear_notifications(self) -> None:
        """Clear recorded notifications."""
        self._notifications.clear()


class DesktopNotificationService(BaseNotificationService):
    """Desktop notification service using GLib/Gio or notify-send fallback.
    
    Attempts to use GNotification via Gio for GTK apps, falling back to
    notify-send command if running headless or GTK is unavailable.
    """
    
    def __init__(
        self,
        app_id: str = "com.atlas.assistant",
        app_name: str = "ATLAS",
        logger: Optional[logging.Logger] = None,
        use_fallback: bool = True,
    ):
        super().__init__(logger)
        self._app_id = app_id
        self._app_name = app_name
        self._use_fallback = use_fallback
        self._gio_available = False
        self._application: Any = None
        
        # Try to initialize Gio
        self._init_gio()
    
    def _init_gio(self) -> None:
        """Initialize Gio for desktop notifications."""
        try:
            import gi
            gi.require_version("Gio", "2.0")
            from gi.repository import Gio, GLib  # type: ignore
            
            self._Gio = Gio
            self._GLib = GLib
            self._gio_available = True
            self._logger.debug("Gio notification backend initialized")
        except (ImportError, ValueError) as e:
            self._logger.debug(f"Gio not available, will use fallback: {e}")
            self._gio_available = False
    
    def set_application(self, application: Any) -> None:
        """Set the GTK application for sending notifications.
        
        Args:
            application: A Gtk.Application instance
        """
        self._application = application
    
    async def send_notification(
        self,
        method: str,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send a desktop notification.
        
        Args:
            method: Notification method (notification, popup, etc.)
            title: Notification title
            message: Notification body
            metadata: Additional notification options
            
        Returns:
            True if notification was sent successfully
        """
        metadata = metadata or {}
        priority = metadata.get("priority", "normal")
        icon = metadata.get("icon")
        timeout_ms = metadata.get("timeout_ms", 5000)
        
        # Map method to appropriate handler
        if method in ("notification", "desktop"):
            return await self._send_desktop_notification(
                title, message, priority, icon, timeout_ms
            )
        elif method == "popup":
            # Popups are handled by UI layer, just log for now
            self._logger.info(f"[Popup requested] {title}: {message}")
            return True
        elif method == "sound":
            return await self._play_sound(metadata.get("sound_file"))
        elif method == "speech":
            return await self._speak(message)
        else:
            self._logger.warning(f"Unknown notification method: {method}")
            return await self._send_desktop_notification(
                title, message, priority, icon, timeout_ms
            )
    
    async def _send_desktop_notification(
        self,
        title: str,
        message: str,
        priority: str = "normal",
        icon: Optional[str] = None,
        timeout_ms: int = 5000,
    ) -> bool:
        """Send a desktop notification via Gio or notify-send."""
        
        # Try Gio first if we have an application
        if self._gio_available and self._application is not None:
            try:
                return await self._send_via_gio(title, message, priority, icon)
            except Exception as e:
                self._logger.warning(f"Gio notification failed: {e}")
                if not self._use_fallback:
                    return False
        
        # Fallback to notify-send
        if self._use_fallback:
            return await self._send_via_notify_send(
                title, message, priority, icon, timeout_ms
            )
        
        return False
    
    async def _send_via_gio(
        self,
        title: str,
        message: str,
        priority: str = "normal",
        icon: Optional[str] = None,
    ) -> bool:
        """Send notification via GNotification."""
        try:
            notification = self._Gio.Notification.new(title)
            notification.set_body(message)
            
            if icon:
                try:
                    gicon = self._Gio.ThemedIcon.new(icon)
                    notification.set_icon(gicon)
                except Exception:
                    pass  # Icon is optional
            
            # Set priority
            priority_map = {
                "low": self._Gio.NotificationPriority.LOW,
                "normal": self._Gio.NotificationPriority.NORMAL,
                "high": self._Gio.NotificationPriority.HIGH,
                "critical": self._Gio.NotificationPriority.URGENT,
            }
            notification.set_priority(
                priority_map.get(priority, self._Gio.NotificationPriority.NORMAL)
            )
            
            # Send via application
            notification_id = str(uuid4())
            self._application.send_notification(notification_id, notification)
            
            self._logger.debug(f"Sent Gio notification: {title}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to send Gio notification: {e}")
            return False
    
    async def _send_via_notify_send(
        self,
        title: str,
        message: str,
        priority: str = "normal",
        icon: Optional[str] = None,
        timeout_ms: int = 5000,
    ) -> bool:
        """Send notification via notify-send command."""
        try:
            cmd = ["notify-send"]
            
            # Add urgency
            urgency_map = {
                "low": "low",
                "normal": "normal",
                "high": "critical",
                "critical": "critical",
            }
            cmd.extend(["-u", urgency_map.get(priority, "normal")])
            
            # Add timeout (convert to ms)
            cmd.extend(["-t", str(timeout_ms)])
            
            # Add app name
            cmd.extend(["-a", self._app_name])
            
            # Add icon if specified
            if icon:
                cmd.extend(["-i", icon])
            
            # Add title and message
            cmd.append(title)
            cmd.append(message)
            
            # Run asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            
            if process.returncode == 0:
                self._logger.debug(f"Sent notify-send notification: {title}")
                return True
            else:
                error = stderr.decode().strip() if stderr else "Unknown error"
                self._logger.warning(f"notify-send failed: {error}")
                return False
                
        except FileNotFoundError:
            self._logger.warning("notify-send not found, cannot send notification")
            return False
        except Exception as e:
            self._logger.error(f"Failed to send notify-send notification: {e}")
            return False
    
    async def _play_sound(self, sound_file: Optional[str] = None) -> bool:
        """Play an audio notification sound."""
        try:
            # Default system sound
            if not sound_file:
                sound_file = "/usr/share/sounds/freedesktop/stereo/message.oga"
            
            # Try paplay first (PulseAudio), then aplay (ALSA)
            for player in ["paplay", "aplay", "play"]:
                try:
                    process = await asyncio.create_subprocess_exec(
                        player, sound_file,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await process.communicate()
                    if process.returncode == 0:
                        return True
                except FileNotFoundError:
                    continue
            
            self._logger.warning("No audio player found for sound notification")
            return False
            
        except Exception as e:
            self._logger.error(f"Failed to play sound: {e}")
            return False
    
    async def _speak(self, message: str) -> bool:
        """Speak the message using text-to-speech."""
        try:
            # Try espeak-ng, then espeak, then spd-say
            for speaker in ["espeak-ng", "espeak", "spd-say"]:
                try:
                    process = await asyncio.create_subprocess_exec(
                        speaker, message,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await process.communicate()
                    if process.returncode == 0:
                        return True
                except FileNotFoundError:
                    continue
            
            self._logger.warning("No TTS engine found for speech notification")
            return False
            
        except Exception as e:
            self._logger.error(f"Failed to speak message: {e}")
            return False


class CompositeNotificationService(BaseNotificationService):
    """Notification service that delegates to multiple backends.
    
    Allows combining different notification methods (e.g., desktop + sound).
    """
    
    def __init__(
        self,
        services: Optional[Dict[str, BaseNotificationService]] = None,
        default_service: Optional[BaseNotificationService] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger)
        self._services = services or {}
        self._default = default_service or DummyNotificationService()
    
    def register(self, method: str, service: BaseNotificationService) -> None:
        """Register a service for a specific notification method."""
        self._services[method] = service
    
    async def send_notification(
        self,
        method: str,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send notification via the appropriate backend."""
        service = self._services.get(method, self._default)
        return await service.send_notification(method, title, message, metadata)
    
    async def send_multi(
        self,
        methods: List[str],
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        """Send notification via multiple methods simultaneously.
        
        Args:
            methods: List of notification methods to use
            title: Notification title
            message: Notification body
            metadata: Additional context
            
        Returns:
            Dict mapping method to success status
        """
        tasks = [
            self.send_notification(method, title, message, metadata)
            for method in methods
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            method: (
                result if isinstance(result, bool) else False
            )
            for method, result in zip(methods, results)
        }


def create_notification_service(
    use_desktop: bool = True,
    use_fallback: bool = True,
    app_name: str = "ATLAS",
) -> BaseNotificationService:
    """Factory function to create an appropriate notification service.
    
    Args:
        use_desktop: Whether to try desktop notifications
        use_fallback: Whether to fall back to notify-send
        app_name: Application name for notifications
        
    Returns:
        Configured notification service
    """
    if use_desktop:
        return DesktopNotificationService(
            app_name=app_name,
            use_fallback=use_fallback,
        )
    else:
        return DummyNotificationService()
