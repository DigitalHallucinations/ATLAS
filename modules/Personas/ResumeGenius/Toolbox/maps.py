# modules\Personas\SCOUT\Toolbox\maps.py

from functools import lru_cache

from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Tools.Base_Tools.ats_scoring import ATSScoringService
from modules.Tools.Base_Tools.policy_reference import policy_reference
from modules.Tools.Base_Tools.time import get_current_info

# Create an instance of GoogleSearch
google_search_instance = GoogleSearch()


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
}
