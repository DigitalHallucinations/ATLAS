"""Calendar Manager GTKUI package.

Provides GTK 4 interface for managing multiple calendar sources
and the ATLAS Master Calendar categories.
"""

from .calendar_agenda_view import CalendarAgendaView
from .calendar_day_view import CalendarDayView
from .calendar_dialog import CalendarDialog
from .calendar_list import CalendarListPanel
from .calendar_management import CalendarManagement
from .calendar_month_view import CalendarMonthView
from .calendar_settings import CalendarSettingsPanel
from .calendar_view_stack import CalendarViewStack
from .calendar_week_view import CalendarWeekView
from .category_dialog import CategoryDialog
from .category_panel import CategoryPanel
from .color_chooser import ColorChooser, DEFAULT_PALETTE
from .event_card import EventCard, EventCardList
from .event_dialog import EventDialog
from .import_mapping_panel import ImportMappingPanel, AddSourceDialog, SyncSourceRow
from .mini_calendar import MiniCalendar, MINI_CALENDAR_CSS
from .sync_status import SyncStatusPanel

__all__ = [
    "CalendarAgendaView",
    "CalendarDayView",
    "CalendarDialog",
    "CalendarListPanel",
    "CalendarManagement",
    "CalendarMonthView",
    "CalendarSettingsPanel",
    "CalendarViewStack",
    "CalendarWeekView",
    "CategoryDialog",
    "CategoryPanel",
    "ColorChooser",
    "DEFAULT_PALETTE",
    "EventCard",
    "EventCardList",
    "EventDialog",
    "ImportMappingPanel",
    "AddSourceDialog",
    "MiniCalendar",
    "MINI_CALENDAR_CSS",
    "SyncSourceRow",
    "SyncStatusPanel",
]
