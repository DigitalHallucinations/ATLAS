# modules\Personas\SCOUT\Toolbox\maps.py

from functools import lru_cache

from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Personas.ResumeGenius.Toolbox.ats_scoring import ATSScoringService
from modules.Tools.Base_Tools.policy_reference import policy_reference
from modules.Tools.Base_Tools.time import get_current_info
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


@lru_cache(maxsize=1)
def _ats_scoring_service() -> ATSScoringService:
    """Lazily construct the ATS scoring client.

    The cache avoids repeated configuration resolution across calls while
    ensuring the service is only instantiated when the tool is invoked.
    """

    return ATSScoringService()


async def ats_scoring_service(**kwargs):
    service = _ats_scoring_service()
    return await service.score_resume(**kwargs)


# A dictionary to map function names to actual function objects
function_map = {
    "get_current_info": get_current_info,
    "google_search": google_search_instance._search,
    "policy_reference": policy_reference,
    "ats_scoring_service": ats_scoring_service,
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
}
