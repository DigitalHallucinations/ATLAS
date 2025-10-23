# components/tools/tool_maps/maps.py

from functools import lru_cache, partial

from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Tools.Base_Tools.ats_scoring import ATSScoringService
from modules.Tools.Base_Tools.policy_reference import policy_reference
from modules.Tools.Base_Tools.time import get_current_info
from modules.Tools.Base_Tools.context_tracker import context_tracker
from modules.Tools.Base_Tools.priority_queue import priority_queue
from modules.Tools.Base_Tools.terminal_command import TerminalCommand
from modules.Tools.Base_Tools.webpage_fetch import WebpageFetcher
from modules.Tools.Base_Tools.geocode import geocode_location
from modules.Tools.Base_Tools.current_location import get_current_location
from modules.Tools.Base_Tools.browser_lite import BrowserLite
from modules.Tools.Base_Tools.calculator import Calculator
from modules.Tools.Base_Tools.debian12_calendar import (
    Debian12CalendarTool,
    debian12_calendar,
)
from modules.Tools.Base_Tools.filesystem_io import read_file, write_file, list_dir
from modules.Tools.Base_Tools.structured_parser import StructuredParser
from modules.Tools.Base_Tools.kv_store import kv_delete, kv_get, kv_increment, kv_set
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

# Create an instance of GoogleSearch
google_search_instance = GoogleSearch()
webpage_fetcher = WebpageFetcher()
python_interpreter = PythonInterpreter()
structured_parser = StructuredParser()

_config_manager = ConfigManager()
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


@lru_cache(maxsize=1)
def _ats_scoring_client() -> ATSScoringService:
    return ATSScoringService()


async def _ats_scoring_dispatch(**kwargs):
    service = _ats_scoring_client()
    return await service.score_resume(**kwargs)

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
    "geocode_location": geocode_location,
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
    "ats_scoring_service": _ats_scoring_dispatch,
    "upsert_vectors": vector_upsert,
    "query_vectors": vector_query,
    "delete_namespace": vector_delete,
    "task_queue_enqueue": task_queue_enqueue,
    "task_queue_schedule": task_queue_schedule,
    "task_queue_cancel": task_queue_cancel,
    "task_queue_status": task_queue_status,
}
