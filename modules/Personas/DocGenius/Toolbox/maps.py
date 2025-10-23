# modules\Personas\CodeGenius\Toolbox\maps.py

from modules.Tools.Base_Tools.time import get_current_info
from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Tools.Base_Tools.policy_reference import policy_reference
from modules.Tools.Medical_Tools.PubMedCentral.ENTREZ_API import search_pubmed
from modules.Tools.Medical_Tools.PubMedCentral.PMC_API import search_pmc
from modules.Tools.Medical_Tools.PubMedCentral.efetch import fetch_pubmed_details

# Create an instance of GoogleSearch
google_search_instance = GoogleSearch()

# A dictionary to map function names to actual function objects
function_map = {
    "get_current_info": get_current_info,
    "google_search": google_search_instance._search,
    "policy_reference": policy_reference,
    "search_pubmed": search_pubmed,
    "search_pmc": search_pmc,
    "fetch_pubmed_details": fetch_pubmed_details,
}
