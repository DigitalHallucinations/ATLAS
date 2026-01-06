"""Recurrence rule handling for ATLAS Calendar.

Implements RFC 5545 RRULE support for recurring events using the
dateutil and icalendar libraries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Iterator, List, Optional, Set, Union

from dateutil import rrule as dateutil_rrule
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)


class Frequency(str, Enum):
    """Recurrence frequency types (RFC 5545)."""

    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    YEARLY = "YEARLY"


class Weekday(str, Enum):
    """Days of the week for recurrence rules."""

    MONDAY = "MO"
    TUESDAY = "TU"
    WEDNESDAY = "WE"
    THURSDAY = "TH"
    FRIDAY = "FR"
    SATURDAY = "SA"
    SUNDAY = "SU"


# Mapping from Weekday enum to dateutil weekday
WEEKDAY_MAP = {
    Weekday.MONDAY: dateutil_rrule.MO,
    Weekday.TUESDAY: dateutil_rrule.TU,
    Weekday.WEDNESDAY: dateutil_rrule.WE,
    Weekday.THURSDAY: dateutil_rrule.TH,
    Weekday.FRIDAY: dateutil_rrule.FR,
    Weekday.SATURDAY: dateutil_rrule.SA,
    Weekday.SUNDAY: dateutil_rrule.SU,
}

# Mapping from frequency to dateutil frequency
FREQ_MAP = {
    Frequency.DAILY: dateutil_rrule.DAILY,
    Frequency.WEEKLY: dateutil_rrule.WEEKLY,
    Frequency.MONTHLY: dateutil_rrule.MONTHLY,
    Frequency.YEARLY: dateutil_rrule.YEARLY,
}

# String to Weekday mapping
STR_TO_WEEKDAY = {wd.value: wd for wd in Weekday}


@dataclass
class RecurrenceRule:
    """Represents an RFC 5545 RRULE.

    Attributes:
        frequency: How often the event recurs (daily, weekly, etc.)
        interval: Number of frequency units between occurrences
        count: Total number of occurrences (mutually exclusive with until)
        until: End date for recurrence (mutually exclusive with count)
        by_weekday: List of weekdays for weekly/monthly recurrence
        by_monthday: List of month days (1-31, negative counts from end)
        by_month: List of months (1-12)
        by_setpos: Position within the set of matching days
        week_start: First day of week (default Monday)
    """

    frequency: Frequency
    interval: int = 1
    count: Optional[int] = None
    until: Optional[datetime] = None
    by_weekday: Optional[List[Weekday]] = None
    by_monthday: Optional[List[int]] = None
    by_month: Optional[List[int]] = None
    by_setpos: Optional[List[int]] = None
    week_start: Weekday = Weekday.MONDAY

    def to_rrule_string(self) -> str:
        """Convert to RFC 5545 RRULE string format.

        Returns:
            RRULE string like "FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,WE,FR"
        """
        parts = [f"FREQ={self.frequency.value}"]

        if self.interval != 1:
            parts.append(f"INTERVAL={self.interval}")

        if self.count is not None:
            parts.append(f"COUNT={self.count}")

        if self.until is not None:
            # Format as UTC timestamp
            if isinstance(self.until, datetime):
                until_str = self.until.strftime("%Y%m%dT%H%M%SZ")
            else:
                until_str = self.until.strftime("%Y%m%d")
            parts.append(f"UNTIL={until_str}")

        if self.by_weekday:
            days = ",".join(wd.value for wd in self.by_weekday)
            parts.append(f"BYDAY={days}")

        if self.by_monthday:
            days = ",".join(str(d) for d in self.by_monthday)
            parts.append(f"BYMONTHDAY={days}")

        if self.by_month:
            months = ",".join(str(m) for m in self.by_month)
            parts.append(f"BYMONTH={months}")

        if self.by_setpos:
            positions = ",".join(str(p) for p in self.by_setpos)
            parts.append(f"BYSETPOS={positions}")

        if self.week_start != Weekday.MONDAY:
            parts.append(f"WKST={self.week_start.value}")

        return ";".join(parts)

    @classmethod
    def from_rrule_string(cls, rrule_str: str) -> "RecurrenceRule":
        """Parse an RFC 5545 RRULE string.

        Args:
            rrule_str: RRULE string (with or without "RRULE:" prefix)

        Returns:
            RecurrenceRule instance

        Raises:
            ValueError: If the RRULE string is invalid
        """
        # Strip prefix if present
        if rrule_str.upper().startswith("RRULE:"):
            rrule_str = rrule_str[6:]

        # Parse into dict
        params: dict = {}
        for part in rrule_str.split(";"):
            if "=" in part:
                key, value = part.split("=", 1)
                params[key.upper()] = value

        # Frequency is required
        if "FREQ" not in params:
            raise ValueError("RRULE must have FREQ parameter")

        try:
            frequency = Frequency(params["FREQ"].upper())
        except ValueError:
            raise ValueError(f"Invalid frequency: {params['FREQ']}")

        # Parse optional parameters
        interval = int(params.get("INTERVAL", 1))

        count = None
        if "COUNT" in params:
            count = int(params["COUNT"])

        until = None
        if "UNTIL" in params:
            until = _parse_datetime(params["UNTIL"])

        by_weekday = None
        if "BYDAY" in params:
            by_weekday = _parse_byday(params["BYDAY"])

        by_monthday = None
        if "BYMONTHDAY" in params:
            by_monthday = [int(d) for d in params["BYMONTHDAY"].split(",")]

        by_month = None
        if "BYMONTH" in params:
            by_month = [int(m) for m in params["BYMONTH"].split(",")]

        by_setpos = None
        if "BYSETPOS" in params:
            by_setpos = [int(p) for p in params["BYSETPOS"].split(",")]

        week_start = Weekday.MONDAY
        if "WKST" in params:
            week_start = STR_TO_WEEKDAY.get(params["WKST"].upper(), Weekday.MONDAY)

        return cls(
            frequency=frequency,
            interval=interval,
            count=count,
            until=until,
            by_weekday=by_weekday,
            by_monthday=by_monthday,
            by_month=by_month,
            by_setpos=by_setpos,
            week_start=week_start,
        )

    def to_dateutil_rrule(
        self, dtstart: datetime
    ) -> dateutil_rrule.rrule:
        """Convert to dateutil rrule object.

        Args:
            dtstart: Start datetime for the recurrence

        Returns:
            dateutil.rrule.rrule instance
        """
        kwargs: dict = {
            "freq": FREQ_MAP[self.frequency],
            "dtstart": dtstart,
            "interval": self.interval,
            "wkst": WEEKDAY_MAP[self.week_start],
        }

        if self.count is not None:
            kwargs["count"] = self.count

        if self.until is not None:
            kwargs["until"] = self.until

        if self.by_weekday:
            kwargs["byweekday"] = [WEEKDAY_MAP[wd] for wd in self.by_weekday]

        if self.by_monthday:
            kwargs["bymonthday"] = self.by_monthday

        if self.by_month:
            kwargs["bymonth"] = self.by_month

        if self.by_setpos:
            kwargs["bysetpos"] = self.by_setpos

        return dateutil_rrule.rrule(**kwargs)


def _parse_datetime(dt_str: str) -> datetime:
    """Parse an RFC 5545 datetime string.

    Supports formats:
    - 20261231T235959Z (UTC)
    - 20261231T235959 (local)
    - 20261231 (date only)
    """
    dt_str = dt_str.strip()

    if len(dt_str) == 8:
        # Date only: YYYYMMDD
        return datetime.strptime(dt_str, "%Y%m%d")
    elif dt_str.endswith("Z"):
        # UTC timestamp
        return datetime.strptime(dt_str, "%Y%m%dT%H%M%SZ")
    else:
        # Local timestamp
        return datetime.strptime(dt_str, "%Y%m%dT%H%M%S")


def _parse_byday(byday_str: str) -> List[Weekday]:
    """Parse BYDAY parameter value.

    Handles values like:
    - "MO,WE,FR" -> [MONDAY, WEDNESDAY, FRIDAY]
    - "1MO" -> [MONDAY] (first Monday - setpos handled elsewhere)
    - "-1FR" -> [FRIDAY] (last Friday)
    """
    weekdays = []
    for day_part in byday_str.split(","):
        # Strip numeric prefix if present (e.g., "1MO" -> "MO")
        day_code = day_part.lstrip("-0123456789")
        if day_code in STR_TO_WEEKDAY:
            weekdays.append(STR_TO_WEEKDAY[day_code])
    return weekdays


class RecurrenceExpander:
    """Expands recurrence rules into individual occurrences.

    Handles exceptions (EXDATE) and modifications (RDATE) as well.
    """

    def __init__(
        self,
        rrule: RecurrenceRule,
        dtstart: datetime,
        exdates: Optional[Set[datetime]] = None,
        rdates: Optional[Set[datetime]] = None,
    ):
        """Initialize the expander.

        Args:
            rrule: The recurrence rule
            dtstart: Start datetime of the recurring event
            exdates: Set of exception dates to exclude
            rdates: Set of additional dates to include
        """
        self.rrule = rrule
        self.dtstart = dtstart
        self.exdates = exdates or set()
        self.rdates = rdates or set()
        self._dateutil_rrule = rrule.to_dateutil_rrule(dtstart)

    def expand(
        self,
        range_start: datetime,
        range_end: datetime,
        max_occurrences: int = 1000,
    ) -> List[datetime]:
        """Expand occurrences within a date range.

        Args:
            range_start: Start of the range to expand
            range_end: End of the range to expand
            max_occurrences: Maximum occurrences to return (safety limit)

        Returns:
            List of occurrence datetimes within the range
        """
        occurrences: List[datetime] = []

        # Get occurrences from rrule
        for dt in self._dateutil_rrule:
            if dt > range_end:
                break
            if dt >= range_start and dt not in self.exdates:
                occurrences.append(dt)
            if len(occurrences) >= max_occurrences:
                logger.warning(
                    "Max occurrences (%d) reached for recurrence expansion",
                    max_occurrences,
                )
                break

        # Add rdates within range
        for rdate in self.rdates:
            if range_start <= rdate <= range_end and rdate not in occurrences:
                occurrences.append(rdate)

        # Sort and return
        return sorted(occurrences)

    def get_next_occurrence(
        self, after: Optional[datetime] = None
    ) -> Optional[datetime]:
        """Get the next occurrence after a given datetime.

        Args:
            after: Datetime to search after (default: now)

        Returns:
            Next occurrence datetime, or None if no more occurrences
        """
        if after is None:
            after = datetime.now()

        # Check rdates first
        for rdate in sorted(self.rdates):
            if rdate > after:
                return rdate

        # Use dateutil's after() method
        try:
            next_dt = self._dateutil_rrule.after(after, inc=False)
            while next_dt and next_dt in self.exdates:
                next_dt = self._dateutil_rrule.after(next_dt, inc=False)
            return next_dt
        except StopIteration:
            return None

    def __iter__(self) -> Iterator[datetime]:
        """Iterate through all occurrences (limited by rule or safety cap)."""
        count = 0
        max_iterations = 10000  # Safety cap

        for dt in self._dateutil_rrule:
            if dt not in self.exdates:
                yield dt
                count += 1
            if count >= max_iterations:
                break


# ============================================================================
# Convenience functions for common recurrence patterns
# ============================================================================


def daily(
    interval: int = 1,
    count: Optional[int] = None,
    until: Optional[datetime] = None,
) -> RecurrenceRule:
    """Create a daily recurrence rule.

    Args:
        interval: Number of days between occurrences
        count: Total number of occurrences
        until: End date for recurrence

    Returns:
        RecurrenceRule for daily recurrence
    """
    return RecurrenceRule(
        frequency=Frequency.DAILY,
        interval=interval,
        count=count,
        until=until,
    )


def weekly(
    interval: int = 1,
    weekdays: Optional[List[Weekday]] = None,
    count: Optional[int] = None,
    until: Optional[datetime] = None,
) -> RecurrenceRule:
    """Create a weekly recurrence rule.

    Args:
        interval: Number of weeks between occurrences
        weekdays: Days of the week to recur (default: same day as event)
        count: Total number of occurrences
        until: End date for recurrence

    Returns:
        RecurrenceRule for weekly recurrence
    """
    return RecurrenceRule(
        frequency=Frequency.WEEKLY,
        interval=interval,
        by_weekday=weekdays,
        count=count,
        until=until,
    )


def monthly_by_day(
    interval: int = 1,
    monthday: int = 1,
    count: Optional[int] = None,
    until: Optional[datetime] = None,
) -> RecurrenceRule:
    """Create a monthly recurrence on a specific day of month.

    Args:
        interval: Number of months between occurrences
        monthday: Day of month (1-31, negative counts from end)
        count: Total number of occurrences
        until: End date for recurrence

    Returns:
        RecurrenceRule for monthly recurrence
    """
    return RecurrenceRule(
        frequency=Frequency.MONTHLY,
        interval=interval,
        by_monthday=[monthday],
        count=count,
        until=until,
    )


def monthly_by_weekday(
    interval: int = 1,
    weekday: Weekday = Weekday.MONDAY,
    week_number: int = 1,
    count: Optional[int] = None,
    until: Optional[datetime] = None,
) -> RecurrenceRule:
    """Create a monthly recurrence on a specific weekday.

    Args:
        interval: Number of months between occurrences
        weekday: Day of week
        week_number: Which occurrence (1-4, or -1 for last)
        count: Total number of occurrences
        until: End date for recurrence

    Returns:
        RecurrenceRule for monthly recurrence (e.g., "2nd Tuesday")
    """
    return RecurrenceRule(
        frequency=Frequency.MONTHLY,
        interval=interval,
        by_weekday=[weekday],
        by_setpos=[week_number],
        count=count,
        until=until,
    )


def yearly(
    interval: int = 1,
    month: Optional[int] = None,
    monthday: Optional[int] = None,
    count: Optional[int] = None,
    until: Optional[datetime] = None,
) -> RecurrenceRule:
    """Create a yearly recurrence rule.

    Args:
        interval: Number of years between occurrences
        month: Month (1-12) to recur
        monthday: Day of month to recur
        count: Total number of occurrences
        until: End date for recurrence

    Returns:
        RecurrenceRule for yearly recurrence
    """
    return RecurrenceRule(
        frequency=Frequency.YEARLY,
        interval=interval,
        by_month=[month] if month else None,
        by_monthday=[monthday] if monthday else None,
        count=count,
        until=until,
    )


def workdays(
    interval: int = 1,
    count: Optional[int] = None,
    until: Optional[datetime] = None,
) -> RecurrenceRule:
    """Create a weekday-only (Mon-Fri) recurrence rule.

    Args:
        interval: Number of weeks between occurrences
        count: Total number of occurrences
        until: End date for recurrence

    Returns:
        RecurrenceRule for weekdays only
    """
    return RecurrenceRule(
        frequency=Frequency.WEEKLY,
        interval=interval,
        by_weekday=[
            Weekday.MONDAY,
            Weekday.TUESDAY,
            Weekday.WEDNESDAY,
            Weekday.THURSDAY,
            Weekday.FRIDAY,
        ],
        count=count,
        until=until,
    )


# ============================================================================
# Human-readable description generation
# ============================================================================


def describe_recurrence(rrule: RecurrenceRule) -> str:
    """Generate a human-readable description of a recurrence rule.

    Args:
        rrule: The recurrence rule to describe

    Returns:
        Human-readable string like "Every 2 weeks on Monday and Wednesday"
    """
    parts: List[str] = []

    # Frequency with interval
    if rrule.interval == 1:
        freq_text = {
            Frequency.DAILY: "Daily",
            Frequency.WEEKLY: "Weekly",
            Frequency.MONTHLY: "Monthly",
            Frequency.YEARLY: "Yearly",
        }[rrule.frequency]
    else:
        freq_text = {
            Frequency.DAILY: f"Every {rrule.interval} days",
            Frequency.WEEKLY: f"Every {rrule.interval} weeks",
            Frequency.MONTHLY: f"Every {rrule.interval} months",
            Frequency.YEARLY: f"Every {rrule.interval} years",
        }[rrule.frequency]
    parts.append(freq_text)

    # Weekdays
    if rrule.by_weekday:
        day_names = {
            Weekday.MONDAY: "Monday",
            Weekday.TUESDAY: "Tuesday",
            Weekday.WEDNESDAY: "Wednesday",
            Weekday.THURSDAY: "Thursday",
            Weekday.FRIDAY: "Friday",
            Weekday.SATURDAY: "Saturday",
            Weekday.SUNDAY: "Sunday",
        }
        days = [day_names[wd] for wd in rrule.by_weekday]
        if len(days) == 1:
            parts.append(f"on {days[0]}")
        elif len(days) == 5 and all(
            wd in rrule.by_weekday
            for wd in [
                Weekday.MONDAY,
                Weekday.TUESDAY,
                Weekday.WEDNESDAY,
                Weekday.THURSDAY,
                Weekday.FRIDAY,
            ]
        ):
            parts.append("on weekdays")
        else:
            parts.append(f"on {', '.join(days[:-1])} and {days[-1]}")

    # Month days
    if rrule.by_monthday:
        if len(rrule.by_monthday) == 1:
            day = rrule.by_monthday[0]
            if day == -1:
                parts.append("on the last day")
            else:
                parts.append(f"on day {day}")
        else:
            parts.append(f"on days {', '.join(str(d) for d in rrule.by_monthday)}")

    # Months
    if rrule.by_month:
        month_names = [
            "",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        months = [month_names[m] for m in rrule.by_month]
        if len(months) == 1:
            parts.append(f"in {months[0]}")
        else:
            parts.append(f"in {', '.join(months[:-1])} and {months[-1]}")

    # End condition
    if rrule.count:
        parts.append(f"({rrule.count} times)")
    elif rrule.until:
        parts.append(f"until {rrule.until.strftime('%b %d, %Y')}")

    return " ".join(parts)
