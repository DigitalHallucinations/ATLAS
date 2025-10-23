# # modules\Persona\MEDIC\Toolbox\maps.py

from modules.Tools.Base_Tools.time import get_current_info
from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Tools.Base_Tools.policy_reference import policy_reference
from modules.Tools.Medical_Tools import (
    fetch_pubmed_details,
    search_pmc,
    search_pubmed,
)


# Create an instance of GoogleSearch
google_search_instance = GoogleSearch()

# A dictionary to map function names to actual function objects
function_map = {
    "get_current_info": get_current_info,
    "google_search": google_search_instance._search,
    "search_pubmed": search_pubmed,
    "search_pmc": search_pmc,
    "fetch_pubmed_details": fetch_pubmed_details,
}
