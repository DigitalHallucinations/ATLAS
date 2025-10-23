# modules/OpenAI/maps.py

from modules.Tools.Base_Tools.current_location import get_current_location

function_map = {
    "get_current_location": get_current_location,
}

prefix_map = {
    "get_current_location": "Your current location is: ",
}
