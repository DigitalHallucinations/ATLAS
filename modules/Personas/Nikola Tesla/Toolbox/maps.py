# modules\Personas\CBGenius\Toolbox\maps.py

from modules.Tools.Base_Tools.time import get_current_info
from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Tools.Base_Tools.policy_reference import policy_reference

from .innovation import wireless_power_brief

# Create an instance of GoogleSearch
google_search_instance = GoogleSearch()

# A dictionary to map function names to actual function objects
function_map = {
    "get_current_info": get_current_info,
    "google_search": google_search_instance._search,
    "policy_reference": policy_reference,
    "wireless_power_brief": wireless_power_brief,
}
