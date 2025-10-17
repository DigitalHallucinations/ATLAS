# modules/OpenAI/maps.py

from modules.Tools.Base_Tools.current_location import get_current_location
from modules.Tools.Base_Tools.geocode import geocode_location

function_map = {      
        "get_current_location": get_current_location,  
        "geocode_location": geocode_location,
    }

prefix_map = {
        "get_current_location": "Your current location is: ",
        "geocode_location": "The geocoded coordinates are: ",
    }
