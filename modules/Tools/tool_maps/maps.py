# components/tools/tool_maps/maps.py

from functools import partial

from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Tools.Base_Tools.policy_reference import policy_reference
from modules.Tools.Base_Tools.time import get_current_info
from modules.Tools.Base_Tools.context_tracker import context_tracker
from modules.Tools.Base_Tools.priority_queue import priority_queue
from modules.Tools.Base_Tools.terminal_command import TerminalCommand
from modules.Tools.Base_Tools.webpage_fetch import WebpageFetcher
from modules.Tools.Base_Tools.current_location import get_current_location
from modules.Tools.Base_Tools.browser_lite import BrowserLite
from modules.Tools.Base_Tools.calculator import Calculator
from modules.Tools.Base_Tools.browser import BrowserTool
from modules.Tools.Base_Tools.notebook import NotebookTool
from modules.Tools.Base_Tools.notification_service import send_notification
from modules.Tools.Base_Tools.dashboard_service import DashboardService
from modules.Tools.Base_Tools.analytics_dashboard import AnalyticsDashboardClient
from modules.Tools.Base_Tools.spreadsheet import SpreadsheetTool
from modules.Tools.Base_Tools.atlas_dashboard import AtlasDashboardClient
from modules.Tools.Base_Tools.roadmap_service import RoadmapService
from modules.Tools.Base_Tools.ticketing_system import TicketingSystem
from modules.Tools.Base_Tools.labor_market_feed import fetch_labor_market_signals
from modules.Tools.Base_Tools.crm_service import CRMService
from modules.Tools.Base_Tools.email_service import send_email
from modules.Tools.Base_Tools.content_repository import ContentRepository
from modules.Tools.Base_Tools.workspace_publisher import WorkspacePublisher
from modules.Tools.Base_Tools.calendar_service import CalendarService
from modules.Tools.Base_Tools.memory_episodic import EpisodicMemoryTool
from modules.Tools.Base_Tools.memory_graph import MemoryGraphTool
from modules.Tools.Base_Tools.debian12_calendar import (
    Debian12CalendarTool,
    debian12_calendar,
)
from modules.Tools.Base_Tools.filesystem_io import read_file, write_file, list_dir
from modules.Tools.Base_Tools.structured_parser import StructuredParser
from modules.Tools.Base_Tools.kv_store import kv_delete, kv_get, kv_increment, kv_set
from modules.Tools.Base_Tools.vault_secrets import VaultSecretsTool
from modules.Tools.Base_Tools.budget_limiter import BudgetLimiterTool
from modules.Tools.Base_Tools.log_event import log_event
from modules.Tools.Base_Tools.hitl_approval import HITLApprovalTool
from modules.Tools.Base_Tools.trace_explain import trace_explain
from modules.Tools.Base_Tools.vector_store import (
    delete_namespace as _vector_delete_namespace,
    query_vectors as _vector_query_vectors,
    upsert_vectors as _vector_upsert_vectors,
)
from modules.Tools.Base_Tools.task_queue import (
    cancel_task as _task_queue_cancel,
    enqueue_task as _task_queue_enqueue,
    get_task_status as _task_queue_status,
    schedule_cron_task as _task_queue_schedule,
)
from ATLAS.config import ConfigManager
from modules.Tools.Code_Execution import JavaScriptExecutor, PythonInterpreter

_config_manager = ConfigManager()

# Create an instance of GoogleSearch
google_search_instance = GoogleSearch()
webpage_fetcher = WebpageFetcher()
python_interpreter = PythonInterpreter()
structured_parser = StructuredParser()
browser_tool = BrowserTool()
notebook_tool = NotebookTool()
dashboard_service_tool = DashboardService()
analytics_dashboard_tool = AnalyticsDashboardClient()
spreadsheet_tool = SpreadsheetTool()
atlas_dashboard_tool = AtlasDashboardClient()
roadmap_service_tool = RoadmapService()
ticketing_system_tool = TicketingSystem()
crm_service_tool = CRMService()
workspace_publisher_tool = WorkspacePublisher()
calendar_service_tool = CalendarService()
content_repository_tool = ContentRepository()
episodic_memory_tool = EpisodicMemoryTool(config_manager=_config_manager)
memory_graph_tool = MemoryGraphTool(config_manager=_config_manager)
javascript_executor = JavaScriptExecutor.from_config(
    _config_manager.get_javascript_executor_settings()
)

_tools_config = _config_manager.get_config("tools", {})
browser_lite_settings = (
    _tools_config.get("browser_lite") if isinstance(_tools_config, dict) else None
)
calculator_settings = (
    _tools_config.get("calculator") if isinstance(_tools_config, dict) else None
)

browser_lite = BrowserLite.from_config(
    browser_lite_settings,
    config_manager=_config_manager,
)
calculator_tool = Calculator.from_config(calculator_settings)
vault_secrets_tool = VaultSecretsTool(config_manager=_config_manager)
budget_limiter_tool = BudgetLimiterTool(config_manager=_config_manager)
hitl_approval_tool = HITLApprovalTool(config_manager=_config_manager)

vector_upsert = partial(_vector_upsert_vectors, config_manager=_config_manager)
vector_query = partial(_vector_query_vectors, config_manager=_config_manager)
vector_delete = partial(_vector_delete_namespace, config_manager=_config_manager)
task_queue_enqueue = partial(_task_queue_enqueue, config_manager=_config_manager)
task_queue_schedule = partial(_task_queue_schedule, config_manager=_config_manager)
task_queue_cancel = partial(_task_queue_cancel, config_manager=_config_manager)
task_queue_status = partial(_task_queue_status, config_manager=_config_manager)

try:
    debian12_calendar_tool = Debian12CalendarTool()
except Exception:
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
    "webpage_fetch": webpage_fetcher.fetch,
    "terminal_command": TerminalCommand,
    "execute_python": python_interpreter.run,
    "execute_javascript": javascript_executor.run,
    "get_current_location": get_current_location,
    "debian12_calendar": _debian12_calendar_dispatch,
    "filesystem_read": read_file,
    "filesystem_write": write_file,
    "filesystem_list": list_dir,
    "structured_parse": structured_parser.parse,
    "kv_get": kv_get,
    "kv_set": kv_set,
    "kv_delete": kv_delete,
    "kv_increment": kv_increment,
    "browser_lite": browser_lite.run,
    "calculator": calculator_tool.evaluate,
    "browser": browser_tool.run,
    "notebook": notebook_tool.run,
    "notification_service": send_notification,
    "dashboard_service": dashboard_service_tool.run,
    "analytics_dashboard": analytics_dashboard_tool.run,
    "spreadsheet": spreadsheet_tool.run,
    "atlas_dashboard": atlas_dashboard_tool.run,
    "roadmap_service": roadmap_service_tool.run,
    "ticketing_system": ticketing_system_tool.run,
    "labor_market_feed": fetch_labor_market_signals,
    "crm_service": crm_service_tool.run,
    "email_service": send_email,
    "workspace_publisher": workspace_publisher_tool.run,
    "calendar_service": calendar_service_tool.run,
    "content_repository": content_repository_tool.run,
    "memory_episodic_store": episodic_memory_tool.store,
    "memory_episodic_query": episodic_memory_tool.query,
    "memory_episodic_prune": episodic_memory_tool.prune,
    "memory.graph": memory_graph_tool.run,
    "vault.secrets": vault_secrets_tool.run,
    "budget.limiter": budget_limiter_tool.run,
    "upsert_vectors": vector_upsert,
    "query_vectors": vector_query,
    "delete_namespace": vector_delete,
    "task_queue_enqueue": task_queue_enqueue,
    "task_queue_schedule": task_queue_schedule,
    "task_queue_cancel": task_queue_cancel,
    "task_queue_status": task_queue_status,
    "log.event": log_event,
    "hitl.approval": hitl_approval_tool.run,
    "trace.explain": trace_explain,
}
