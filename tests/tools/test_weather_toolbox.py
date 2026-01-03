import asyncio
from typing import List

import pytest

from modules.Personas.WeatherGenius.Toolbox import daily_summary, historical_weather, weather, weather_history


@pytest.mark.parametrize(
    "module, func_name, call_kwargs, expected_phrase, expected_query_fragment",
    [
        (
            weather,
            "get_current_weather",
            {"lat": 1, "lon": 2, "units": "metric"},
            "fetch current weather data",
            "units=metric",
        ),
        (
            daily_summary,
            "get_daily_weather_summary",
            {"lat": 1, "lon": 2, "date": "2024-01-01"},
            "fetch the daily weather summary",
            "date=2024-01-01",
        ),
        (
            historical_weather,
            "get_historical_weather",
            {"lat": 1, "lon": 2, "dt": 1704067200},
            "fetch historical weather data",
            "dt=1704067200",
        ),
        (
            weather_history,
            "get_historical_weather",
            {"lat": 1, "lon": 2, "time": 1704067200},
            "fetch historical weather data",
            "dt=1704067200",
        ),
    ],
)
def test_weather_helpers_missing_api_key_returns_error(
    monkeypatch, module, func_name, call_kwargs, expected_phrase, expected_query_fragment
):
    monkeypatch.delenv("OPENWEATHERMAP_API_KEY", raising=False)

    async def _unexpected_call(url):  # pragma: no cover - defensive guard
        raise AssertionError("Network helper should not be called when API key is missing")

    monkeypatch.setattr(module, "get_weather_data", _unexpected_call, raising=False)

    func = getattr(module, func_name)
    result = asyncio.run(func(**call_kwargs))
    assert result == {"error": f"OPENWEATHERMAP_API_KEY is not set; cannot {expected_phrase}."}


@pytest.mark.parametrize(
    "module, func_name, call_kwargs, expected_query_fragment",
    [
        (
            weather,
            "get_current_weather",
            {"lat": 1, "lon": 2, "units": "metric"},
            "units=metric",
        ),
        (
            daily_summary,
            "get_daily_weather_summary",
            {"lat": 1, "lon": 2, "date": "2024-01-01"},
            "date=2024-01-01",
        ),
        (
            historical_weather,
            "get_historical_weather",
            {"lat": 1, "lon": 2, "dt": 1704067200},
            "dt=1704067200",
        ),
        (
            weather_history,
            "get_historical_weather",
            {"lat": 1, "lon": 2, "time": 1704067200},
            "dt=1704067200",
        ),
    ],
)
def test_weather_helpers_success(monkeypatch, module, func_name, call_kwargs, expected_query_fragment):
    monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "test-key")

    captured_urls: List[str] = []

    async def _fake_get_weather_data(url):
        captured_urls.append(url)
        return {"status": "ok"}

    monkeypatch.setattr(module, "get_weather_data", _fake_get_weather_data)

    func = getattr(module, func_name)
    result = asyncio.run(func(**call_kwargs))

    assert result == {"status": "ok"}
    assert captured_urls, "Expected weather helper to call get_weather_data"
    assert len(captured_urls) == 1
    captured_url = captured_urls[0]
    assert "appid=test-key" in captured_url
    assert expected_query_fragment in captured_url
