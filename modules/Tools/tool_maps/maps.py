# components/tools/tool_maps/maps.py

from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Tools.Base_Tools.policy_reference import policy_reference
from modules.Tools.Base_Tools.time import get_current_info
from modules.Tools.Base_Tools.context_tracker import context_tracker
from modules.Tools.Base_Tools.priority_queue import priority_queue
from modules.Tools.Base_Tools.webpage_fetch import WebpageFetcher

# Create an instance of GoogleSearch
google_search_instance = GoogleSearch()
webpage_fetcher = WebpageFetcher()

# A dictionary to map function names to actual function objects
function_map = {
    "get_current_info": get_current_info,
    "google_search": google_search_instance._search,
    "policy_reference": policy_reference,
    "context_tracker": context_tracker,
    "priority_queue": priority_queue,
    "webpage_fetch": webpage_fetcher.fetch,
}
