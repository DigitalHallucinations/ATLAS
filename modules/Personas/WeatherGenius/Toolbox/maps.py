# modules\Personas\Toolbox\WeatherGenius\maps.py

from modules.Tools.Base_Tools.time import get_current_info
from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Tools.Base_Tools.policy_reference import policy_reference
from modules.Personas.WeatherGenius.Toolbox.geocode import geocode_location
from modules.Personas.WeatherGenius.Toolbox.weather import get_current_weather
from modules.Personas.WeatherGenius.Toolbox.historical_weather import get_historical_weather
from modules.Personas.WeatherGenius.Toolbox.daily_summary import get_daily_weather_summary
from modules.Personas.WeatherGenius.Toolbox.alerts import weather_alert_feed


# Create an instance of GoogleSearch
google_search_instance = GoogleSearch()

# A dictionary to map function names to actual function objects
function_map = {
    "get_current_weather": get_current_weather,
    "get_historical_weather": get_historical_weather,
    "get_daily_weather_summary": get_daily_weather_summary,
    "get_current_info": get_current_info,
    "google_search": google_search_instance._search,
    "policy_reference": policy_reference,
    "weather_alert_feed": weather_alert_feed,
    "geocode_location": geocode_location,
}
