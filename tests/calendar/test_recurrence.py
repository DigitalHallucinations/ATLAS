"""Tests for calendar recurrence and reminder modules.

Tests RRULE parsing, occurrence expansion, and reminder scheduling.
"""

from __future__ import annotations

import pytest
from datetime import datetime, date, time, timedelta


class TestRecurrenceRule:
    """Tests for RecurrenceRule dataclass."""

    def test_daily_rule_to_string(self):
        """Test daily recurrence rule conversion to RRULE string."""
        from modules.calendar_store.recurrence import RecurrenceRule, Frequency

        rule = RecurrenceRule(frequency=Frequency.DAILY)
        assert rule.to_rrule_string() == "FREQ=DAILY"

    def test_daily_with_interval(self):
        """Test daily recurrence with interval."""
        from modules.calendar_store.recurrence import RecurrenceRule, Frequency

        rule = RecurrenceRule(frequency=Frequency.DAILY, interval=3)
        rrule_str = rule.to_rrule_string()
        assert "FREQ=DAILY" in rrule_str
        assert "INTERVAL=3" in rrule_str

    def test_daily_with_count(self):
        """Test daily recurrence with count limit."""
        from modules.calendar_store.recurrence import RecurrenceRule, Frequency

        rule = RecurrenceRule(frequency=Frequency.DAILY, count=10)
        rrule_str = rule.to_rrule_string()
        assert "COUNT=10" in rrule_str

    def test_daily_with_until(self):
        """Test daily recurrence with end date."""
        from modules.calendar_store.recurrence import RecurrenceRule, Frequency

        until = datetime(2026, 12, 31, 23, 59, 59)
        rule = RecurrenceRule(frequency=Frequency.DAILY, until=until)
        rrule_str = rule.to_rrule_string()
        assert "UNTIL=20261231T235959Z" in rrule_str

    def test_weekly_with_weekdays(self):
        """Test weekly recurrence with specific weekdays."""
        from modules.calendar_store.recurrence import RecurrenceRule, Frequency, Weekday

        rule = RecurrenceRule(
            frequency=Frequency.WEEKLY,
            by_weekday=[Weekday.MONDAY, Weekday.WEDNESDAY, Weekday.FRIDAY],
        )
        rrule_str = rule.to_rrule_string()
        assert "FREQ=WEEKLY" in rrule_str
        assert "BYDAY=MO,WE,FR" in rrule_str

    def test_monthly_by_day(self):
        """Test monthly recurrence by day of month."""
        from modules.calendar_store.recurrence import RecurrenceRule, Frequency

        rule = RecurrenceRule(
            frequency=Frequency.MONTHLY,
            by_monthday=[15],
        )
        rrule_str = rule.to_rrule_string()
        assert "FREQ=MONTHLY" in rrule_str
        assert "BYMONTHDAY=15" in rrule_str

    def test_yearly_with_month(self):
        """Test yearly recurrence with specific month."""
        from modules.calendar_store.recurrence import RecurrenceRule, Frequency

        rule = RecurrenceRule(
            frequency=Frequency.YEARLY,
            by_month=[12],
            by_monthday=[25],
        )
        rrule_str = rule.to_rrule_string()
        assert "FREQ=YEARLY" in rrule_str
        assert "BYMONTH=12" in rrule_str
        assert "BYMONTHDAY=25" in rrule_str


class TestRRULEParsing:
    """Tests for parsing RRULE strings."""

    def test_parse_simple_daily(self):
        """Test parsing simple daily rule."""
        from modules.calendar_store.recurrence import RecurrenceRule, Frequency

        rule = RecurrenceRule.from_rrule_string("FREQ=DAILY")
        assert rule.frequency == Frequency.DAILY
        assert rule.interval == 1

    def test_parse_with_prefix(self):
        """Test parsing with RRULE: prefix."""
        from modules.calendar_store.recurrence import RecurrenceRule, Frequency

        rule = RecurrenceRule.from_rrule_string("RRULE:FREQ=WEEKLY;INTERVAL=2")
        assert rule.frequency == Frequency.WEEKLY
        assert rule.interval == 2

    def test_parse_weekly_with_days(self):
        """Test parsing weekly rule with BYDAY."""
        from modules.calendar_store.recurrence import RecurrenceRule, Frequency, Weekday

        rule = RecurrenceRule.from_rrule_string("FREQ=WEEKLY;BYDAY=MO,WE,FR")
        assert rule.frequency == Frequency.WEEKLY
        assert rule.by_weekday == [Weekday.MONDAY, Weekday.WEDNESDAY, Weekday.FRIDAY]

    def test_parse_with_count(self):
        """Test parsing rule with COUNT."""
        from modules.calendar_store.recurrence import RecurrenceRule

        rule = RecurrenceRule.from_rrule_string("FREQ=DAILY;COUNT=5")
        assert rule.count == 5

    def test_parse_with_until_date(self):
        """Test parsing rule with UNTIL date."""
        from modules.calendar_store.recurrence import RecurrenceRule

        rule = RecurrenceRule.from_rrule_string("FREQ=DAILY;UNTIL=20261231")
        assert rule.until is not None
        assert rule.until.year == 2026
        assert rule.until.month == 12
        assert rule.until.day == 31

    def test_parse_with_until_datetime(self):
        """Test parsing rule with UNTIL datetime."""
        from modules.calendar_store.recurrence import RecurrenceRule

        rule = RecurrenceRule.from_rrule_string("FREQ=DAILY;UNTIL=20261231T235959Z")
        assert rule.until == datetime(2026, 12, 31, 23, 59, 59)

    def test_parse_invalid_frequency_raises(self):
        """Test that invalid frequency raises ValueError."""
        from modules.calendar_store.recurrence import RecurrenceRule

        with pytest.raises(ValueError):
            RecurrenceRule.from_rrule_string("FREQ=INVALID")

    def test_parse_missing_freq_raises(self):
        """Test that missing FREQ raises ValueError."""
        from modules.calendar_store.recurrence import RecurrenceRule

        with pytest.raises(ValueError):
            RecurrenceRule.from_rrule_string("INTERVAL=2")

    def test_roundtrip_complex_rule(self):
        """Test that parse -> to_string -> parse produces same result."""
        from modules.calendar_store.recurrence import RecurrenceRule, Frequency, Weekday

        original = RecurrenceRule(
            frequency=Frequency.WEEKLY,
            interval=2,
            by_weekday=[Weekday.TUESDAY, Weekday.THURSDAY],
            count=10,
        )
        rrule_str = original.to_rrule_string()
        parsed = RecurrenceRule.from_rrule_string(rrule_str)

        assert parsed.frequency == original.frequency
        assert parsed.interval == original.interval
        assert parsed.by_weekday == original.by_weekday
        assert parsed.count == original.count


class TestRecurrenceExpander:
    """Tests for RecurrenceExpander."""

    def test_expand_daily(self):
        """Test expanding daily recurrence."""
        from modules.calendar_store.recurrence import RecurrenceRule, RecurrenceExpander, Frequency

        rule = RecurrenceRule(frequency=Frequency.DAILY, count=5)
        start = datetime(2026, 1, 1, 10, 0, 0)
        expander = RecurrenceExpander(rule, start)

        occurrences = expander.expand(
            datetime(2026, 1, 1),
            datetime(2026, 1, 31),
        )

        assert len(occurrences) == 5
        assert occurrences[0] == start
        assert occurrences[1] == datetime(2026, 1, 2, 10, 0, 0)

    def test_expand_weekly(self):
        """Test expanding weekly recurrence."""
        from modules.calendar_store.recurrence import RecurrenceRule, RecurrenceExpander, Frequency

        rule = RecurrenceRule(frequency=Frequency.WEEKLY, count=4)
        start = datetime(2026, 1, 6, 14, 0, 0)  # Tuesday
        expander = RecurrenceExpander(rule, start)

        occurrences = expander.expand(
            datetime(2026, 1, 1),
            datetime(2026, 2, 28),
        )

        assert len(occurrences) == 4
        # Each occurrence should be 7 days apart
        for i in range(1, len(occurrences)):
            delta = occurrences[i] - occurrences[i - 1]
            assert delta.days == 7

    def test_expand_with_exdates(self):
        """Test expansion respects exception dates."""
        from modules.calendar_store.recurrence import RecurrenceRule, RecurrenceExpander, Frequency

        rule = RecurrenceRule(frequency=Frequency.DAILY, count=5)
        start = datetime(2026, 1, 1, 10, 0, 0)
        exdates = {datetime(2026, 1, 3, 10, 0, 0)}  # Exclude Jan 3

        expander = RecurrenceExpander(rule, start, exdates=exdates)
        occurrences = expander.expand(
            datetime(2026, 1, 1),
            datetime(2026, 1, 31),
        )

        # Should have 4 occurrences (5 - 1 excluded)
        assert len(occurrences) == 4
        assert datetime(2026, 1, 3, 10, 0, 0) not in occurrences

    def test_expand_range_filtering(self):
        """Test that expansion respects date range."""
        from modules.calendar_store.recurrence import RecurrenceRule, RecurrenceExpander, Frequency

        rule = RecurrenceRule(frequency=Frequency.DAILY, count=30)
        start = datetime(2026, 1, 1, 10, 0, 0)
        expander = RecurrenceExpander(rule, start)

        # Only get Jan 10-15 (end is exclusive at midnight, so 15th at 10:00 is excluded)
        occurrences = expander.expand(
            datetime(2026, 1, 10),
            datetime(2026, 1, 16),  # End exclusive
        )

        assert len(occurrences) == 6  # 10, 11, 12, 13, 14, 15
        assert all(10 <= o.day <= 15 for o in occurrences)

    def test_get_next_occurrence(self):
        """Test getting next occurrence after a date."""
        from modules.calendar_store.recurrence import RecurrenceRule, RecurrenceExpander, Frequency

        rule = RecurrenceRule(frequency=Frequency.WEEKLY)
        start = datetime(2026, 1, 6, 10, 0, 0)  # Tuesday
        expander = RecurrenceExpander(rule, start)

        next_occ = expander.get_next_occurrence(datetime(2026, 1, 10))
        assert next_occ == datetime(2026, 1, 13, 10, 0, 0)  # Next Tuesday


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_daily_helper(self):
        """Test daily() helper function."""
        from modules.calendar_store.recurrence import daily, Frequency

        rule = daily(interval=2, count=10)
        assert rule.frequency == Frequency.DAILY
        assert rule.interval == 2
        assert rule.count == 10

    def test_weekly_helper(self):
        """Test weekly() helper function."""
        from modules.calendar_store.recurrence import weekly, Frequency, Weekday

        rule = weekly(weekdays=[Weekday.MONDAY, Weekday.FRIDAY])
        assert rule.frequency == Frequency.WEEKLY
        assert rule.by_weekday == [Weekday.MONDAY, Weekday.FRIDAY]

    def test_workdays_helper(self):
        """Test workdays() helper function."""
        from modules.calendar_store.recurrence import workdays, Weekday

        rule = workdays()
        assert rule.by_weekday is not None
        assert len(rule.by_weekday) == 5
        assert Weekday.SATURDAY not in rule.by_weekday
        assert Weekday.SUNDAY not in rule.by_weekday

    def test_monthly_by_day_helper(self):
        """Test monthly_by_day() helper function."""
        from modules.calendar_store.recurrence import monthly_by_day, Frequency

        rule = monthly_by_day(monthday=15)
        assert rule.frequency == Frequency.MONTHLY
        assert rule.by_monthday == [15]

    def test_yearly_helper(self):
        """Test yearly() helper function."""
        from modules.calendar_store.recurrence import yearly, Frequency

        rule = yearly(month=12, monthday=25)
        assert rule.frequency == Frequency.YEARLY
        assert rule.by_month == [12]
        assert rule.by_monthday == [25]


class TestDescribeRecurrence:
    """Tests for human-readable recurrence descriptions."""

    def test_describe_daily(self):
        """Test description of daily recurrence."""
        from modules.calendar_store.recurrence import daily, describe_recurrence

        rule = daily()
        desc = describe_recurrence(rule)
        assert "Daily" in desc

    def test_describe_weekly_with_days(self):
        """Test description of weekly with specific days."""
        from modules.calendar_store.recurrence import weekly, describe_recurrence, Weekday

        rule = weekly(weekdays=[Weekday.MONDAY, Weekday.FRIDAY])
        desc = describe_recurrence(rule)
        assert "Weekly" in desc
        assert "Monday" in desc
        assert "Friday" in desc

    def test_describe_workdays(self):
        """Test description of workdays recurrence."""
        from modules.calendar_store.recurrence import workdays, describe_recurrence

        rule = workdays()
        desc = describe_recurrence(rule)
        assert "weekdays" in desc.lower()

    def test_describe_with_count(self):
        """Test description includes count."""
        from modules.calendar_store.recurrence import daily, describe_recurrence

        rule = daily(count=5)
        desc = describe_recurrence(rule)
        assert "5 times" in desc


class TestReminder:
    """Tests for Reminder dataclass."""

    def test_reminder_to_dict(self):
        """Test reminder serialization."""
        from modules.calendar_store.reminders import Reminder, ReminderMethod

        reminder = Reminder(
            minutes_before=15,
            method=ReminderMethod.NOTIFICATION,
            message="Test message",
        )
        data = reminder.to_dict()

        assert data["minutes_before"] == 15
        assert data["method"] == "notification"
        assert data["message"] == "Test message"

    def test_reminder_from_dict(self):
        """Test reminder deserialization."""
        from modules.calendar_store.reminders import Reminder, ReminderMethod

        data = {"minutes_before": 30, "method": "popup"}
        reminder = Reminder.from_dict(data)

        assert reminder.minutes_before == 30
        assert reminder.method == ReminderMethod.POPUP

    def test_reminder_get_trigger_time(self):
        """Test trigger time calculation."""
        from modules.calendar_store.reminders import Reminder

        reminder = Reminder(minutes_before=15)
        event_start = datetime(2026, 1, 15, 14, 0, 0)
        trigger = reminder.get_trigger_time(event_start)

        assert trigger == datetime(2026, 1, 15, 13, 45, 0)

    def test_reminder_is_due(self):
        """Test reminder due check."""
        from modules.calendar_store.reminders import Reminder

        reminder = Reminder(minutes_before=15)
        event_start = datetime(2026, 1, 15, 14, 0, 0)

        # Before trigger time
        assert not reminder.is_due(event_start, now=datetime(2026, 1, 15, 13, 30, 0))

        # At trigger time
        assert reminder.is_due(event_start, now=datetime(2026, 1, 15, 13, 45, 0))

        # After trigger time
        assert reminder.is_due(event_start, now=datetime(2026, 1, 15, 13, 50, 0))

    def test_reminder_describe_minutes(self):
        """Test human-readable description for minutes."""
        from modules.calendar_store.reminders import Reminder

        reminder = Reminder(minutes_before=15)
        assert reminder.describe() == "15 minutes before"

    def test_reminder_describe_hours(self):
        """Test human-readable description for hours."""
        from modules.calendar_store.reminders import Reminder

        reminder = Reminder(minutes_before=120)
        assert reminder.describe() == "2 hours before"

    def test_reminder_describe_days(self):
        """Test human-readable description for days."""
        from modules.calendar_store.reminders import Reminder

        reminder = Reminder(minutes_before=1440)
        assert reminder.describe() == "1 day before"


class TestReminderScheduler:
    """Tests for ReminderScheduler."""

    def test_schedule_event_reminders(self):
        """Test scheduling reminders for an event."""
        from modules.calendar_store.reminders import Reminder, ReminderScheduler

        scheduler = ReminderScheduler()
        event_start = datetime.now() + timedelta(hours=2)

        reminders = [
            Reminder(minutes_before=15),
            Reminder(minutes_before=60),
        ]

        scheduled = scheduler.schedule_event_reminders(
            event_id="test-event",
            event_title="Test Event",
            event_start=event_start,
            reminders=reminders,
        )

        assert len(scheduled) == 2

    def test_cancel_event_reminders(self):
        """Test canceling reminders for an event."""
        from modules.calendar_store.reminders import Reminder, ReminderScheduler

        scheduler = ReminderScheduler()
        event_start = datetime.now() + timedelta(hours=2)

        scheduler.schedule_event_reminders(
            event_id="test-event",
            event_title="Test Event",
            event_start=event_start,
            reminders=[Reminder(minutes_before=15)],
        )

        cancelled = scheduler.cancel_event_reminders("test-event")
        assert cancelled == 1

        # Second cancel should return 0
        assert scheduler.cancel_event_reminders("test-event") == 0

    def test_get_upcoming_reminders(self):
        """Test getting upcoming reminders."""
        from modules.calendar_store.reminders import Reminder, ReminderScheduler

        scheduler = ReminderScheduler()
        event_start = datetime.now() + timedelta(minutes=30)

        scheduler.schedule_event_reminders(
            event_id="soon-event",
            event_title="Soon Event",
            event_start=event_start,
            reminders=[Reminder(minutes_before=15)],
        )

        upcoming = scheduler.get_upcoming_reminders(within_minutes=60)
        assert len(upcoming) == 1

    def test_reminder_callback(self):
        """Test reminder callback is called when processing due reminders."""
        from modules.calendar_store.reminders import (
            Reminder, ReminderScheduler, ScheduledReminder, ReminderStatus
        )

        scheduler = ReminderScheduler()
        triggered_reminders = []

        def callback(reminder: ScheduledReminder):
            triggered_reminders.append(reminder)

        scheduler.register_callback(callback)

        # Create a ScheduledReminder that's already due by manually setting trigger_time
        event_start = datetime.now() + timedelta(minutes=30)
        sr = ScheduledReminder(
            event_id="test-event",
            event_title="Test Event",
            reminder=Reminder(minutes_before=15),
            trigger_time=datetime.now() - timedelta(seconds=1),  # Already past
            event_start=event_start,
            status=ReminderStatus.PENDING,
        )
        scheduler._scheduled["test-event"] = [sr]

        scheduler.process_due_reminders()
        assert len(triggered_reminders) == 1
        assert triggered_reminders[0].event_id == "test-event"


class TestCreateDefaultReminders:
    """Tests for default reminder creation."""

    def test_default_for_regular_event(self):
        """Test default reminders for regular events."""
        from modules.calendar_store.reminders import create_default_reminders

        reminders = create_default_reminders(event_duration_minutes=60)
        assert len(reminders) == 1
        assert reminders[0].minutes_before == 15

    def test_default_for_all_day_event(self):
        """Test default reminders for all-day events."""
        from modules.calendar_store.reminders import create_default_reminders

        reminders = create_default_reminders(is_all_day=True)
        assert len(reminders) == 1
        # Should remind 9 hours before (morning of)
        assert reminders[0].minutes_before == 540

    def test_default_for_multi_day_event(self):
        """Test default reminders for multi-day events."""
        from modules.calendar_store.reminders import create_default_reminders

        reminders = create_default_reminders(event_duration_minutes=2880)  # 2 days
        assert len(reminders) == 2
        assert any(r.minutes_before == 1440 for r in reminders)  # 1 day
        assert any(r.minutes_before == 60 for r in reminders)  # 1 hour


class TestParseNaturalLanguageReminder:
    """Tests for natural language reminder parsing."""

    def test_parse_minutes(self):
        """Test parsing minute-based reminders."""
        from modules.calendar_store.reminders import parse_reminder_from_natural_language

        reminder = parse_reminder_from_natural_language("15 minutes before")
        assert reminder is not None
        assert reminder.minutes_before == 15

    def test_parse_hours(self):
        """Test parsing hour-based reminders."""
        from modules.calendar_store.reminders import parse_reminder_from_natural_language

        reminder = parse_reminder_from_natural_language("2 hours before")
        assert reminder is not None
        assert reminder.minutes_before == 120

    def test_parse_days(self):
        """Test parsing day-based reminders."""
        from modules.calendar_store.reminders import parse_reminder_from_natural_language

        reminder = parse_reminder_from_natural_language("1 day before")
        assert reminder is not None
        assert reminder.minutes_before == 1440

    def test_parse_at_event_time(self):
        """Test parsing 'at event time'."""
        from modules.calendar_store.reminders import parse_reminder_from_natural_language

        reminder = parse_reminder_from_natural_language("at event time")
        assert reminder is not None
        assert reminder.minutes_before == 0

    def test_parse_invalid_returns_none(self):
        """Test that invalid strings return None."""
        from modules.calendar_store.reminders import parse_reminder_from_natural_language

        reminder = parse_reminder_from_natural_language("invalid text")
        assert reminder is None
