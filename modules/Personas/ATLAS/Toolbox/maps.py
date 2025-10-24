# modules\Personas\ATLAS\Toolbox\maps.py

from modules.Tools.Base_Tools.time import get_current_info
from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Tools.Base_Tools.policy_reference import policy_reference
from modules.Tools.Base_Tools.context_tracker import context_tracker
from modules.Tools.Base_Tools.priority_queue import priority_queue
from modules.Tools.Base_Tools.debian12_calendar import (
    Debian12CalendarTool,
    debian12_calendar,
)
from modules.Tools.Base_Tools.browser import BrowserTool
from modules.Tools.Base_Tools.notebook import NotebookTool
from modules.Tools.Base_Tools.notification_service import send_notification
from modules.Tools.Base_Tools.dashboard_service import DashboardService
from modules.Tools.Base_Tools.spreadsheet import SpreadsheetTool
from modules.Tools.Base_Tools.atlas_dashboard import AtlasDashboardClient
from modules.Tools.Base_Tools.roadmap_service import RoadmapService
from modules.Tools.Base_Tools.ticketing_system import TicketingSystem
from modules.Tools.Base_Tools.labor_market_feed import fetch_labor_market_signals
from modules.Tools.Base_Tools.crm_service import CRMService
from modules.Tools.Base_Tools.email_service import send_email
from modules.Tools.Base_Tools.workspace_publisher import WorkspacePublisher

from .catalog import task_catalog_snapshot

# Create an instance of GoogleSearch
google_search_instance = GoogleSearch()
browser_tool = BrowserTool()
notebook_tool = NotebookTool()
dashboard_service_tool = DashboardService()
spreadsheet_tool = SpreadsheetTool()
atlas_dashboard_tool = AtlasDashboardClient()
roadmap_service_tool = RoadmapService()
ticketing_system_tool = TicketingSystem()
crm_service_tool = CRMService()
workspace_publisher_tool = WorkspacePublisher()


# Debian calendar helper mirrors shared tool map behaviour.
try:
    debian12_calendar_tool = Debian12CalendarTool()
except Exception:  # pragma: no cover - defensive runtime fallback
    debian12_calendar_tool = None


async def _debian12_calendar_dispatch(*args, **kwargs):
    if debian12_calendar_tool is not None:
        return await debian12_calendar_tool.run(*args, **kwargs)
    return await debian12_calendar(*args, **kwargs)


# A dictionary to map function names to actual function objects
function_map = {
    "get_current_info": get_current_info,
    "google_search": google_search_instance._search,
    "policy_reference": policy_reference,
    "context_tracker": context_tracker,
    "priority_queue": priority_queue,
    "debian12_calendar": _debian12_calendar_dispatch,
    "task_catalog_snapshot": task_catalog_snapshot,
    "browser": browser_tool.run,
    "notebook": notebook_tool.run,
    "notification_service": send_notification,
    "dashboard_service": dashboard_service_tool.run,
    "spreadsheet": spreadsheet_tool.run,
    "atlas_dashboard": atlas_dashboard_tool.run,
    "roadmap_service": roadmap_service_tool.run,
    "ticketing_system": ticketing_system_tool.run,
    "labor_market_feed": fetch_labor_market_signals,
    "crm_service": crm_service_tool.run,
    "email_service": send_email,
    "workspace_publisher": workspace_publisher_tool.run,
}
