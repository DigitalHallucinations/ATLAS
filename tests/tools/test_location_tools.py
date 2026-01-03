import asyncio
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def stub_external_dependencies(monkeypatch):
    if "aiohttp" not in sys.modules:
        monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(ClientSession=object))
    if "yaml" not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            "yaml",
            SimpleNamespace(safe_load=lambda *_args, **_kwargs: {}, dump=lambda *_args, **_kwargs: None),
        )
    if "dotenv" not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            "dotenv",
            SimpleNamespace(
                load_dotenv=lambda *_args, **_kwargs: None,
                set_key=lambda *_args, **_kwargs: None,
                find_dotenv=lambda *_args, **_kwargs: "",
            ),
        )
    if "pytz" not in sys.modules:
        monkeypatch.setitem(sys.modules, "pytz", SimpleNamespace(timezone=lambda *_args, **_kwargs: None))


def test_geocode_location_success(monkeypatch):
    monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test-key")

    mock_geocode = AsyncMock(return_value={"lat": 10.0, "lon": 20.0, "city": "Paris"})
    monkeypatch.setattr(
        "modules.Tools.location_services.geocode.geocode_location",
        mock_geocode,
    )

    geocode_module = importlib.reload(
        importlib.import_module("modules.Personas.WeatherGenius.Toolbox.geocode")
    )

    result = asyncio.run(geocode_module.geocode_location("Paris, FR"))

    assert result == {"lat": 10.0, "lon": 20.0, "city": "Paris"}
    mock_geocode.assert_awaited_once_with("Paris, FR")


def test_geocode_location_requires_location(monkeypatch):
    monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test-key")
    geocode_module = importlib.reload(
        importlib.import_module("modules.Personas.WeatherGenius.Toolbox.geocode")
    )

    result = asyncio.run(geocode_module.geocode_location("  "))

    assert result["error"].startswith("A location value is required")


def test_geocode_location_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENWEATHERMAP_API_KEY", raising=False)

    mock_geocode = AsyncMock()
    monkeypatch.setattr(
        "modules.Tools.location_services.geocode.geocode_location",
        mock_geocode,
    )

    geocode_module = importlib.reload(
        importlib.import_module("modules.Personas.WeatherGenius.Toolbox.geocode")
    )

    result = asyncio.run(geocode_module.geocode_location("Berlin"))

    assert result["error"].startswith("OpenWeatherMap API key is required")
    mock_geocode.assert_not_called()


def test_geocode_location_provider_error(monkeypatch):
    monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test-key")

    mock_geocode = AsyncMock(return_value={"error": "No match"})
    monkeypatch.setattr(
        "modules.Tools.location_services.geocode.geocode_location",
        mock_geocode,
    )

    geocode_module = importlib.reload(
        importlib.import_module("modules.Personas.WeatherGenius.Toolbox.geocode")
    )

    result = asyncio.run(geocode_module.geocode_location("Atlantis"))

    assert result == {"error": "No match"}


def test_geocode_location_exception(monkeypatch):
    monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test-key")

    mock_geocode = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(
        "modules.Tools.location_services.geocode.geocode_location",
        mock_geocode,
    )

    geocode_module = importlib.reload(
        importlib.import_module("modules.Personas.WeatherGenius.Toolbox.geocode")
    )

    result = asyncio.run(geocode_module.geocode_location("Rome"))

    assert result["error"] == "Failed to retrieve geocoding information from the provider."


def test_get_current_location_success(monkeypatch):
    mock_current_location = AsyncMock(return_value={"city": "Chicago"})
    monkeypatch.setattr(
        "modules.Tools.location_services.ip_api.get_current_location",
        mock_current_location,
    )

    current_location_module = importlib.reload(
        importlib.import_module("modules.Tools.Base_Tools.current_location")
    )

    result = asyncio.run(current_location_module.get_current_location())

    assert result == {"city": "Chicago"}
    mock_current_location.assert_awaited_once()


def test_get_current_location_provider_error(monkeypatch):
    mock_current_location = AsyncMock(return_value={"error": "Denied"})
    monkeypatch.setattr(
        "modules.Tools.location_services.ip_api.get_current_location",
        mock_current_location,
    )

    current_location_module = importlib.reload(
        importlib.import_module("modules.Tools.Base_Tools.current_location")
    )

    result = asyncio.run(current_location_module.get_current_location())

    assert result == {"error": "Denied"}


def test_get_current_location_exception(monkeypatch):
    mock_current_location = AsyncMock(side_effect=RuntimeError("network"))
    monkeypatch.setattr(
        "modules.Tools.location_services.ip_api.get_current_location",
        mock_current_location,
    )

    current_location_module = importlib.reload(
        importlib.import_module("modules.Tools.Base_Tools.current_location")
    )

    result = asyncio.run(current_location_module.get_current_location())

    assert result["error"] == "Failed to retrieve current location from the provider."
