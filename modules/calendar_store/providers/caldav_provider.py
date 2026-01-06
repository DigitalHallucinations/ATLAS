"""CalDAV calendar sync provider.

Implements sync from CalDAV servers (Google Calendar, iCloud, Nextcloud, etc).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from ..sync_engine import CalendarSyncProvider, ExternalEvent

logger = logging.getLogger(__name__)


class CalDAVProvider(CalendarSyncProvider):
    """Calendar sync provider for CalDAV servers.
    
    Supports:
    - Standard CalDAV servers (Nextcloud, ownCloud, Radicale)
    - Google Calendar (via CalDAV)
    - iCloud Calendar
    - Fastmail
    """
    
    # Well-known CalDAV endpoints
    KNOWN_ENDPOINTS = {
        "google": "https://apidata.googleusercontent.com/caldav/v2",
        "icloud": "https://caldav.icloud.com",
        "fastmail": "https://caldav.fastmail.com/dav",
        "nextcloud": "/remote.php/dav/calendars",
    }
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._client: Any = None
        self._principal: Any = None
        self._calendars: Dict[str, Any] = {}
        self._connected = False
    
    @property
    def provider_type(self) -> str:
        return "caldav"
    
    @property
    def display_name(self) -> str:
        return "CalDAV Server"
    
    def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to CalDAV server.
        
        Config options:
        - url: CalDAV server URL
        - username: Username for authentication
        - password: Password or app-specific password
        - provider: Optional known provider name (google, icloud, etc)
        """
        try:
            import caldav
        except ImportError:
            logger.error("caldav library not installed. Run: pip install caldav")
            return False
        
        self._config = config
        
        # Determine URL
        url = config.get("url")
        provider = config.get("provider", "").lower()
        
        if not url and provider in self.KNOWN_ENDPOINTS:
            url = self.KNOWN_ENDPOINTS[provider]
        
        if not url:
            logger.error("CalDAV URL required")
            return False
        
        username = config.get("username")
        password = config.get("password")
        
        try:
            # Create CalDAV client
            self._client = caldav.DAVClient(
                url=url,
                username=username,
                password=password,
            )
            
            # Get principal (user's calendar home)
            self._principal = self._client.principal()
            
            # Enumerate calendars
            self._calendars.clear()
            for cal in self._principal.calendars():
                cal_id = self._get_calendar_id(cal)
                self._calendars[cal_id] = {
                    "id": cal_id,
                    "name": cal.name or cal_id,
                    "color": self._get_calendar_color(cal),
                    "readonly": False,
                    "_calendar": cal,  # Keep reference
                }
            
            self._connected = True
            logger.info("CalDAV connected with %d calendars", len(self._calendars))
            return True
            
        except Exception as e:
            logger.error("CalDAV connection failed: %s", e)
            return False
    
    def disconnect(self) -> None:
        """Disconnect from CalDAV server."""
        self._client = None
        self._principal = None
        self._calendars.clear()
        self._connected = False
    
    def list_calendars(self) -> List[Dict[str, Any]]:
        """List available calendars."""
        result = []
        for cal_id, cal_info in self._calendars.items():
            result.append({
                "id": cal_info["id"],
                "name": cal_info["name"],
                "color": cal_info.get("color"),
                "readonly": cal_info.get("readonly", False),
            })
        return result
    
    def fetch_events(
        self,
        calendar_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        sync_token: Optional[str] = None,
    ) -> Tuple[List[ExternalEvent], Optional[str]]:
        """Fetch events from a CalDAV calendar."""
        cal_info = self._calendars.get(calendar_id)
        if not cal_info:
            logger.warning("Calendar not found: %s", calendar_id)
            return [], None
        
        calendar = cal_info.get("_calendar")
        if not calendar:
            return [], None
        
        events = []
        
        try:
            # Default date range if not specified
            if not start:
                start = datetime.now(timezone.utc) - timedelta(days=90)
            if not end:
                end = datetime.now(timezone.utc) + timedelta(days=365)
            
            # Fetch events from CalDAV
            caldav_events = calendar.date_search(
                start=start,
                end=end,
                expand=True,  # Expand recurring events
            )
            
            for caldav_event in caldav_events:
                event = self._parse_caldav_event(caldav_event)
                if event:
                    events.append(event)
            
            # CalDAV sync tokens
            new_token = None
            try:
                new_token = calendar.get_ctag()
            except Exception:
                pass
            
            return events, new_token
            
        except Exception as e:
            logger.error("Failed to fetch CalDAV events: %s", e)
            return [], None
    
    def push_event(
        self,
        calendar_id: str,
        event: ExternalEvent,
    ) -> Optional[str]:
        """Push an event to CalDAV calendar."""
        cal_info = self._calendars.get(calendar_id)
        if not cal_info:
            return None
        
        calendar = cal_info.get("_calendar")
        if not calendar:
            return None
        
        try:
            ical_data = self._event_to_ical(event)
            
            caldav_event = calendar.save_event(ical_data)
            
            # Return the new event's UID/href
            return self._get_event_id(caldav_event)
            
        except Exception as e:
            logger.error("Failed to push event to CalDAV: %s", e)
            return None
    
    def delete_event(
        self,
        calendar_id: str,
        external_id: str,
    ) -> bool:
        """Delete an event from CalDAV calendar."""
        cal_info = self._calendars.get(calendar_id)
        if not cal_info:
            return False
        
        calendar = cal_info.get("_calendar")
        if not calendar:
            return False
        
        try:
            # Find the event by UID
            events = calendar.event_by_uid(external_id)
            if events:
                if hasattr(events, "delete"):
                    events.delete()
                elif isinstance(events, list) and events:
                    events[0].delete()
                return True
            return False
            
        except Exception as e:
            logger.error("Failed to delete CalDAV event: %s", e)
            return False
    
    def _get_calendar_id(self, calendar: Any) -> str:
        """Extract calendar ID from CalDAV calendar object."""
        # Use URL path as ID
        if hasattr(calendar, "url"):
            url = str(calendar.url)
            parsed = urlparse(url)
            return parsed.path.rstrip("/").split("/")[-1]
        return str(hash(str(calendar)))
    
    def _get_calendar_color(self, calendar: Any) -> Optional[str]:
        """Extract calendar color from CalDAV properties."""
        try:
            # Try to get calendar-color property
            props = calendar.get_properties(["{http://apple.com/ns/ical/}calendar-color"])
            if props:
                color = list(props.values())[0]
                if color:
                    return str(color)
        except Exception:
            pass
        return None
    
    def _get_event_id(self, event: Any) -> str:
        """Extract event ID from CalDAV event object."""
        try:
            # Try UID first
            if hasattr(event, "vobject_instance"):
                vevent = event.vobject_instance.vevent
                if hasattr(vevent, "uid"):
                    return str(vevent.uid.value)
            
            # Fall back to href
            if hasattr(event, "url"):
                url = str(event.url)
                return urlparse(url).path.rstrip("/").split("/")[-1]
        except Exception:
            pass
        
        return str(hash(str(event)))
    
    def _parse_caldav_event(self, caldav_event: Any) -> Optional[ExternalEvent]:
        """Parse a CalDAV event into an ExternalEvent."""
        try:
            # Get the iCal data
            if hasattr(caldav_event, "icalendar_component"):
                component = caldav_event.icalendar_component
            elif hasattr(caldav_event, "vobject_instance"):
                vobj = caldav_event.vobject_instance
                component = vobj.vevent
            else:
                return None
            
            # Extract UID
            uid = str(component.get("uid", ""))
            if not uid and hasattr(component, "uid"):
                uid = str(component.uid.value if hasattr(component.uid, "value") else component.uid)
            
            # Extract title
            title = str(component.get("summary", "Untitled"))
            if hasattr(component, "summary"):
                title = str(component.summary.value if hasattr(component.summary, "value") else component.summary)
            
            # Extract dates
            dtstart = component.get("dtstart")
            dtend = component.get("dtend")
            
            # Handle vobject format
            if hasattr(component, "dtstart"):
                dtstart = component.dtstart
            if hasattr(component, "dtend"):
                dtend = component.dtend
            
            if not dtstart:
                return None
            
            # Get datetime value
            start_dt = dtstart.dt if hasattr(dtstart, "dt") else (
                dtstart.value if hasattr(dtstart, "value") else dtstart
            )
            
            is_all_day = not isinstance(start_dt, datetime)
            
            if is_all_day:
                start_time = datetime.combine(start_dt, datetime.min.time())
                if dtend:
                    end_dt = dtend.dt if hasattr(dtend, "dt") else (
                        dtend.value if hasattr(dtend, "value") else dtend
                    )
                    end_time = datetime.combine(end_dt, datetime.min.time())
                else:
                    end_time = start_time + timedelta(days=1)
            else:
                start_time = start_dt
                if dtend:
                    end_dt = dtend.dt if hasattr(dtend, "dt") else (
                        dtend.value if hasattr(dtend, "value") else dtend
                    )
                    end_time = end_dt
                else:
                    end_time = start_time + timedelta(hours=1)
            
            # Ensure timezone awareness
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
            
            # Extract other properties
            description = None
            if hasattr(component, "description"):
                desc = component.description
                description = str(desc.value if hasattr(desc, "value") else desc)
            elif component.get("description"):
                description = str(component.get("description"))
            
            location = None
            if hasattr(component, "location"):
                loc = component.location
                location = str(loc.value if hasattr(loc, "value") else loc)
            elif component.get("location"):
                location = str(component.get("location"))
            
            # Recurrence rule
            rrule_str = None
            rrule = component.get("rrule")
            if rrule:
                if hasattr(rrule, "to_ical"):
                    rrule_str = rrule.to_ical().decode("utf-8")
                else:
                    rrule_str = str(rrule)
            
            # Timestamps
            created_at = None
            updated_at = None
            
            if component.get("created"):
                created = component.get("created")
                if hasattr(created, "dt"):
                    created_at = created.dt
            
            if component.get("last-modified"):
                modified = component.get("last-modified")
                if hasattr(modified, "dt"):
                    updated_at = modified.dt
            
            return ExternalEvent(
                external_id=uid,
                title=title,
                start_time=start_time,
                end_time=end_time,
                description=description,
                location=location,
                is_all_day=is_all_day,
                recurrence_rule=rrule_str,
                created_at=created_at,
                updated_at=updated_at,
            )
            
        except Exception as e:
            logger.warning("Failed to parse CalDAV event: %s", e)
            return None
    
    def _event_to_ical(self, event: ExternalEvent) -> str:
        """Convert an ExternalEvent to iCal format."""
        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//ATLAS//Calendar//EN",
            "BEGIN:VEVENT",
        ]
        
        # UID
        lines.append(f"UID:{event.external_id}")
        
        # Timestamps
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        lines.append(f"DTSTAMP:{now}")
        
        # Dates
        if event.is_all_day:
            start_str = event.start_time.strftime("%Y%m%d")
            end_str = event.end_time.strftime("%Y%m%d")
            lines.append(f"DTSTART;VALUE=DATE:{start_str}")
            lines.append(f"DTEND;VALUE=DATE:{end_str}")
        else:
            start_str = event.start_time.strftime("%Y%m%dT%H%M%SZ")
            end_str = event.end_time.strftime("%Y%m%dT%H%M%SZ")
            lines.append(f"DTSTART:{start_str}")
            lines.append(f"DTEND:{end_str}")
        
        # Title
        lines.append(f"SUMMARY:{self._escape_ical(event.title)}")
        
        # Optional fields
        if event.description:
            lines.append(f"DESCRIPTION:{self._escape_ical(event.description)}")
        
        if event.location:
            lines.append(f"LOCATION:{self._escape_ical(event.location)}")
        
        if event.recurrence_rule:
            lines.append(f"RRULE:{event.recurrence_rule}")
        
        lines.extend([
            "END:VEVENT",
            "END:VCALENDAR",
        ])
        
        return "\r\n".join(lines)
    
    def _escape_ical(self, text: str) -> str:
        """Escape special characters for iCal format."""
        text = text.replace("\\", "\\\\")
        text = text.replace(";", "\\;")
        text = text.replace(",", "\\,")
        text = text.replace("\n", "\\n")
        return text
