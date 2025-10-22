from modules.Tools.Base_Tools.time import get_current_info
from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Tools.Base_Tools.context_tracker import context_tracker


_google_search = GoogleSearch()


function_map = {
    "get_current_info": get_current_info,
    "google_search": _google_search._search,
    "context_tracker": context_tracker,
}
