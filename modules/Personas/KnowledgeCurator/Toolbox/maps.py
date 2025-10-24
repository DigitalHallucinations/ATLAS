from modules.Tools.Base_Tools.time import get_current_info
from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Tools.Base_Tools.context_tracker import context_tracker
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

from .knowledge_cards import knowledge_card_builder


_google_search = GoogleSearch()
_browser_tool = BrowserTool()
_notebook_tool = NotebookTool()
_dashboard_service_tool = DashboardService()
_spreadsheet_tool = SpreadsheetTool()
_atlas_dashboard_tool = AtlasDashboardClient()
_roadmap_service_tool = RoadmapService()
_ticketing_system_tool = TicketingSystem()
_crm_service_tool = CRMService()


function_map = {
    "get_current_info": get_current_info,
    "google_search": _google_search._search,
    "context_tracker": context_tracker,
    "knowledge_card_builder": knowledge_card_builder,
    "browser": _browser_tool.run,
    "notebook": _notebook_tool.run,
    "notification_service": send_notification,
    "dashboard_service": _dashboard_service_tool.run,
    "spreadsheet": _spreadsheet_tool.run,
    "atlas_dashboard": _atlas_dashboard_tool.run,
    "roadmap_service": _roadmap_service_tool.run,
    "ticketing_system": _ticketing_system_tool.run,
    "labor_market_feed": fetch_labor_market_signals,
    "crm_service": _crm_service_tool.run,
    "email_service": send_email,
}
