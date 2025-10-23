"""Function map for the MathTutor persona."""

from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Tools.Base_Tools.policy_reference import policy_reference
from modules.Tools.Base_Tools.time import get_current_info

from .problem_solver import stepwise_solution

_google = GoogleSearch()

function_map = {
    "google_search": _google._search,
    "get_current_info": get_current_info,
    "policy_reference": policy_reference,
    "stepwise_solution": stepwise_solution,
}
