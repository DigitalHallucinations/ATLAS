"""Function map for the FitnessCoach persona."""

from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Tools.Base_Tools.policy_reference import policy_reference
from modules.Tools.Base_Tools.time import get_current_info

from .workouts import microcycle_plan

_google = GoogleSearch()

function_map = {
    "google_search": _google._search,
    "get_current_info": get_current_info,
    "policy_reference": policy_reference,
    "microcycle_plan": microcycle_plan,
}
