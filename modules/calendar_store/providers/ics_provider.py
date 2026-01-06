"""ICS file calendar sync provider.

Implements sync from ICS (iCalendar) files, both local and remote URLs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from ..sync_engine import CalendarSyncProvider, ExternalEvent

logger = logging.getLogger(__name__)


class ICSProvider(CalendarSyncProvider):
    """Calendar sync provider for ICS files.
    
    Supports:
    - Local .ics files
    - Remote ICS URLs (webcal://, https://)
    - Multiple calendars (one per file/URL)
    """
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._calendars: Dict[str, Dict[str, Any]] = {}
        self._connected = False
    
    @property
    def provider_type(self) -> str:
        return "ics"
    
    @property
    def display_name(self) -> str:
        return "ICS File / URL"
    
    def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to ICS source(s).
        
        Config options:
        - sources: List of ICS sources, each with:
            - id: Unique identifier
            - name: Display name
            - path: Local file path OR URL
            - color: Optional color hint
        """
        self._config = config
        self._calendars.clear()
        
        sources = config.get("sources", [])
        if not sources:
            # Single source mode
            path = config.get("path") or config.get("url")
            if path:
                sources = [{
                    "id": "default",
                    "name": config.get("name", "Calendar"),
                    "path": path,
                    "color": config.get("color"),
                }]
        
        for source in sources:
            source_id = source.get("id", str(len(self._calendars)))
            self._calendars[source_id] = {
                "id": source_id,
                "name": source.get("name", f"Calendar {source_id}"),
                "path": source.get("path"),
                "color": source.get("color"),
                "readonly": True,  # ICS is always read-only
            }
        
        self._connected = True
        logger.info("ICS provider connected with %d sources", len(self._calendars))
        return True
    
    def disconnect(self) -> None:
        """Disconnect from ICS source."""
        self._connected = False
        self._calendars.clear()
    
    def list_calendars(self) -> List[Dict[str, Any]]:
        """List configured ICS calendars."""
        return list(self._calendars.values())
    
    def fetch_events(
        self,
        calendar_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        sync_token: Optional[str] = None,
    ) -> Tuple[List[ExternalEvent], Optional[str]]:
        """Fetch events from an ICS source.
        
        Note: ICS doesn't support incremental sync, so sync_token is ignored.
        """
        calendar = self._calendars.get(calendar_id)
        if not calendar:
            logger.warning("Calendar not found: %s", calendar_id)
            return [], None
        
        path = calendar.get("path")
        if not path:
            return [], None
        
        try:
            ics_data = self._load_ics_data(path)
            events = self._parse_ics(ics_data, start, end)
            
            # Generate a simple "sync token" based on content hash
            import hashlib
            token = hashlib.md5(ics_data.encode()).hexdigest()
            
            return events, token
            
        except Exception as e:
            logger.error("Failed to fetch ICS events: %s", e)
            return [], None
    
    def _load_ics_data(self, path: str) -> str:
        """Load ICS data from file or URL."""
        parsed = urlparse(path)
        
        if parsed.scheme in ("http", "https", "webcal"):
            return self._fetch_remote_ics(path)
        else:
            return self._read_local_ics(path)
    
    def _fetch_remote_ics(self, url: str) -> str:
        """Fetch ICS data from a remote URL."""
        import urllib.request
        import ssl
        
        # Handle webcal:// scheme
        if url.startswith("webcal://"):
            url = "https://" + url[9:]
        
        # Create SSL context
        context = ssl.create_default_context()
        
        headers = {
            "User-Agent": "ATLAS Calendar Sync/1.0",
        }
        
        request = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(request, context=context, timeout=30) as response:
            return response.read().decode("utf-8")
    
    def _read_local_ics(self, path: str) -> str:
        """Read ICS data from a local file."""
        file_path = Path(path).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"ICS file not found: {path}")
        
        return file_path.read_text(encoding="utf-8")
    
    def _parse_ics(
        self,
        ics_data: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[ExternalEvent]:
        """Parse ICS data into ExternalEvent objects."""
        events = []
        
        try:
            # Try using icalendar library if available
            from icalendar import Calendar
            
            cal = Calendar.from_ical(ics_data)
            
            for component in cal.walk():
                if component.name == "VEVENT":
                    event = self._parse_vevent(component)
                    if event:
                        # Apply date filter
                        if start and event.end_time < start:
                            continue
                        if end and event.start_time > end:
                            continue
                        events.append(event)
            
        except ImportError:
            # Fallback to simple parsing
            logger.warning("icalendar library not available, using simple parser")
            events = self._simple_parse_ics(ics_data, start, end)
        
        return events
    
    def _parse_vevent(self, component: Any) -> Optional[ExternalEvent]:
        """Parse a VEVENT component into an ExternalEvent."""
        try:
            # Get UID
            uid = str(component.get("uid", ""))
            if not uid:
                import uuid
                uid = str(uuid.uuid4())
            
            # Get summary/title
            title = str(component.get("summary", "Untitled Event"))
            
            # Get dates
            dtstart = component.get("dtstart")
            dtend = component.get("dtend")
            
            if not dtstart:
                return None
            
            start_dt = dtstart.dt
            is_all_day = not isinstance(start_dt, datetime)
            
            if is_all_day:
                # Convert date to datetime
                start_time = datetime.combine(start_dt, datetime.min.time())
                if dtend:
                    end_time = datetime.combine(dtend.dt, datetime.min.time())
                else:
                    end_time = start_time + timedelta(days=1)
            else:
                start_time = start_dt
                if dtend:
                    end_time = dtend.dt
                else:
                    # Default 1 hour duration
                    end_time = start_time + timedelta(hours=1)
            
            # Ensure timezone awareness
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
            
            # Get other properties
            description = str(component.get("description", "")) or None
            location = str(component.get("location", "")) or None
            
            # Get recurrence rule
            rrule = component.get("rrule")
            rrule_str = None
            if rrule:
                rrule_str = rrule.to_ical().decode("utf-8")
            
            # Get timestamps
            created = component.get("created")
            modified = component.get("last-modified")
            
            created_at = None
            if created:
                created_at = created.dt if isinstance(created.dt, datetime) else None
            
            updated_at = None
            if modified:
                updated_at = modified.dt if isinstance(modified.dt, datetime) else None
            
            # Parse attendees
            attendees = []
            for attendee in component.get("attendee", []):
                attendees.append({
                    "email": str(attendee).replace("mailto:", ""),
                    "status": str(attendee.params.get("PARTSTAT", "NEEDS-ACTION")),
                })
            
            # Parse alarms/reminders
            reminders = []
            for alarm in component.walk("VALARM"):
                trigger = alarm.get("trigger")
                if trigger:
                    # Convert to minutes before
                    if hasattr(trigger.dt, "total_seconds"):
                        minutes = abs(int(trigger.dt.total_seconds() / 60))
                        reminders.append({"minutes_before": minutes})
            
            return ExternalEvent(
                external_id=uid,
                title=title,
                start_time=start_time,
                end_time=end_time,
                description=description,
                location=location,
                is_all_day=is_all_day,
                recurrence_rule=rrule_str,
                attendees=attendees,
                reminders=reminders,
                created_at=created_at,
                updated_at=updated_at,
            )
            
        except Exception as e:
            logger.warning("Failed to parse VEVENT: %s", e)
            return None
    
    def _simple_parse_ics(
        self,
        ics_data: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[ExternalEvent]:
        """Simple ICS parser without icalendar library."""
        events = []
        current_event: Dict[str, Any] = {}
        in_event = False
        
        for line in ics_data.split("\n"):
            line = line.strip()
            
            if line == "BEGIN:VEVENT":
                in_event = True
                current_event = {}
            elif line == "END:VEVENT":
                in_event = False
                event = self._simple_event_from_dict(current_event)
                if event:
                    if start and event.end_time < start:
                        continue
                    if end and event.start_time > end:
                        continue
                    events.append(event)
                current_event = {}
            elif in_event and ":" in line:
                # Handle property;params:value format
                if ";" in line.split(":")[0]:
                    key = line.split(";")[0]
                else:
                    key = line.split(":")[0]
                value = ":".join(line.split(":")[1:])
                current_event[key] = value
        
        return events
    
    def _simple_event_from_dict(self, data: Dict[str, str]) -> Optional[ExternalEvent]:
        """Convert simple parsed data to ExternalEvent."""
        try:
            uid = data.get("UID", str(hash(str(data))))
            title = data.get("SUMMARY", "Untitled")
            
            dtstart = data.get("DTSTART", "")
            dtend = data.get("DTEND", "")
            
            if not dtstart:
                return None
            
            # Simple date parsing
            is_all_day = len(dtstart) == 8  # YYYYMMDD format
            
            if is_all_day:
                start_time = datetime.strptime(dtstart, "%Y%m%d")
                if dtend:
                    end_time = datetime.strptime(dtend, "%Y%m%d")
                else:
                    end_time = start_time + timedelta(days=1)
            else:
                # Try common formats
                for fmt in ["%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"]:
                    try:
                        start_time = datetime.strptime(dtstart, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return None
                
                if dtend:
                    for fmt in ["%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"]:
                        try:
                            end_time = datetime.strptime(dtend, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        end_time = start_time + timedelta(hours=1)
                else:
                    end_time = start_time + timedelta(hours=1)
            
            # Ensure UTC
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
            
            return ExternalEvent(
                external_id=uid,
                title=title,
                start_time=start_time,
                end_time=end_time,
                description=data.get("DESCRIPTION"),
                location=data.get("LOCATION"),
                is_all_day=is_all_day,
                recurrence_rule=data.get("RRULE"),
            )
            
        except Exception as e:
            logger.debug("Failed to parse simple event: %s", e)
            return None
