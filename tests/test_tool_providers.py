import asyncio
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import pytest

if "yaml" not in sys.modules:
    sys.modules["yaml"] = SimpleNamespace(
        safe_load=lambda *_args, **_kwargs: {},
        dump=lambda *_args, **_kwargs: None,
    )

if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = SimpleNamespace(
        load_dotenv=lambda *_args, **_kwargs: None,
        set_key=lambda *_args, **_kwargs: None,
        find_dotenv=lambda *_args, **_kwargs: "",
    )

from modules.Tools.Base_Tools import task_queue as task_queue_module
from modules.Tools.Base_Tools.webpage_fetch import WebpageFetchResult, WebpageFetcher
from modules.Tools.providers import debian12_local as debian12_provider
from modules.Tools.providers import internal_http_client as internal_http_provider
from modules.Tools.providers import ncbi_entrez as ncbi_entrez_provider
from modules.Tools.providers import openweathermap as openweathermap_provider
from modules.Tools.providers import task_queue_default as task_queue_provider
from modules.Tools.providers.base import ToolProvider, ToolProviderSpec
from modules.Tools.providers.registry import tool_provider_registry
from modules.Tools.providers.router import ToolProviderRouter


class _FailingProvider(ToolProvider):
    async def call(self, **kwargs: Any) -> Any:
        raise RuntimeError("simulated failure")


class _HealthyProvider(ToolProvider):
    async def call(self, **kwargs: Any) -> Any:
        return kwargs.get("value")


class _DelayedProvider(ToolProvider):
    def __init__(self, spec: ToolProviderSpec, *, tool_name: str, fallback_callable=None) -> None:
        super().__init__(spec, tool_name=tool_name, fallback_callable=fallback_callable)
        self.delay = 0.2

    async def call(self, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        start = loop.time()
        await asyncio.sleep(self.delay)
        end = loop.time()
        # Store the last interval for assertions if required by future tests.
        self.logger.debug("Call interval: %s -> %s", start, end)
        return kwargs.get("value")


class _HealthCheckProvider(ToolProvider):
    def __init__(self, spec: ToolProviderSpec, *, tool_name: str, fallback_callable=None) -> None:
        super().__init__(spec, tool_name=tool_name, fallback_callable=fallback_callable)
        self.called = False

    async def call(self, **kwargs: Any) -> Any:
        self.called = True
        return "should not run"

    async def health_check(self) -> bool:
        return False


def test_provider_router_falls_back_to_healthiest_provider():
    metrics: List[Dict[str, Any]] = []
    with tool_provider_registry.temporary_provider("failing", _FailingProvider), tool_provider_registry.temporary_provider(
        "healthy", _HealthyProvider
    ):
        router = ToolProviderRouter(
            tool_name="demo",
            provider_specs=[
                {"name": "failing", "priority": 0},
                {"name": "healthy", "priority": 1},
            ],
        )
        router.register_metrics_callback(lambda payload: metrics.append(payload))

        result = asyncio.run(router.call(value="ok"))
        assert result == "ok"

    assert metrics, "Metrics callback should be invoked"
    assert any(not entry["success"] and entry["selected"] == "failing" for entry in metrics)
    assert metrics[-1]["success"] is True
    assert metrics[-1]["selected"] == "healthy"
    assert metrics[-1]["providers"]["failing"]["failures"] >= 1


def test_router_records_metrics_with_default_callback(monkeypatch):
    recorded: List[Mapping[str, Any]] = []

    class _Recorder:
        def record_provider_metrics(
            self,
            *,
            persona: Optional[str],
            tool_name: str,
            summary: Mapping[str, Any],
        ) -> None:
            recorded.append(
                {
                    "persona": persona,
                    "tool": tool_name,
                    "summary": dict(summary),
                }
            )

    dummy_registry = _Recorder()
    monkeypatch.setattr(
        "modules.Tools.providers.router.get_capability_registry",
        lambda: dummy_registry,
    )

    with tool_provider_registry.temporary_provider("healthy", _HealthyProvider):
        router = ToolProviderRouter(
            tool_name="demo",
            provider_specs=[{"name": "healthy", "priority": 1}],
            persona="ops",
        )
        result = asyncio.run(router.call(value="ok"))
        assert result == "ok"

    assert recorded, "Registry callback should capture metrics"
    payload = recorded[-1]
    assert payload["persona"] == "ops"
    assert payload["tool"] == "demo"
    assert payload["summary"].get("selected") == "healthy"
    assert payload["summary"].get("latency_ms") is not None


def test_provider_router_skips_unhealthy_providers():
    captured: List[_HealthCheckProvider] = []

    def _factory(spec: ToolProviderSpec, tool_name: str, fallback_callable=None) -> _HealthCheckProvider:
        provider = _HealthCheckProvider(spec, tool_name=tool_name, fallback_callable=fallback_callable)
        captured.append(provider)
        return provider

    with tool_provider_registry.temporary_provider("unhealthy", _factory), tool_provider_registry.temporary_provider(
        "healthy", _HealthyProvider
    ):
        router = ToolProviderRouter(
            tool_name="demo",
            provider_specs=[
                {"name": "unhealthy", "priority": 0},
                {"name": "healthy", "priority": 1},
            ],
        )
        result = asyncio.run(router.call(value="ok"))
        assert result == "ok"

    assert captured, "Health check provider should be instantiated"
    assert captured[0].called is False, "Provider failing health check should not be invoked"


def test_provider_router_allows_concurrent_calls_and_updates_health():
    delay = 0.2

    def _factory(spec: ToolProviderSpec, tool_name: str, fallback_callable=None) -> _DelayedProvider:
        provider = _DelayedProvider(spec, tool_name=tool_name, fallback_callable=fallback_callable)
        provider.delay = delay
        return provider

    async def _run_concurrent_calls() -> Dict[str, Any]:
        router = ToolProviderRouter(
            tool_name="demo",
            provider_specs=[{"name": "delayed", "priority": 0}],
        )

        async def _invoke() -> Dict[str, float]:
            loop = asyncio.get_running_loop()
            start = loop.time()
            result = await router.call(value="ok")
            end = loop.time()
            return {"result": result, "start": start, "end": end}

        first, second = await asyncio.gather(_invoke(), _invoke())
        overlap = min(first["end"], second["end"]) - max(first["start"], second["start"])
        return {
            "first": first,
            "second": second,
            "overlap": overlap,
            "health": router._states[0].health.snapshot(),
        }

    with tool_provider_registry.temporary_provider("delayed", _factory):
        result = asyncio.run(_run_concurrent_calls())

    assert result["first"]["result"] == "ok"
    assert result["second"]["result"] == "ok"
    assert result["overlap"] > 0, "Provider calls should overlap in time"
    assert result["health"]["successes"] >= 2
    assert result["health"]["failures"] == 0


def test_atlas_runtime_provider_wraps_fallback_callable():
    captured: List[str] = []

    async def _fallback(**kwargs: Any) -> str:
        value = kwargs.get("value")
        captured.append(value)
        return str(value)

    router = ToolProviderRouter(
        tool_name="context_tracker",
        provider_specs=[{"name": "atlas_runtime", "priority": 0}],
        fallback_callable=_fallback,
    )

    result = asyncio.run(router.call(value="tracked"))

    assert result == "tracked"
    assert captured == ["tracked"]
    assert router._states and router._states[0].name == "atlas_runtime"
    assert router._states[0].health.successes >= 1


def test_internal_http_client_provider_uses_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    class _StubFetcher:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def fetch(self, url: str) -> WebpageFetchResult:
            calls.append(url)
            return WebpageFetchResult(url=url, title="Example", text="body", content_length=4)

    monkeypatch.setattr(internal_http_provider, "WebpageFetcher", _StubFetcher)

    async def _fallback(**_kwargs: Any) -> Any:  # pragma: no cover - should not execute
        raise AssertionError("Fallback should not execute")

    router = ToolProviderRouter(
        tool_name="webpage_fetch",
        provider_specs=[{"name": "internal_http_client", "priority": 0}],
        fallback_callable=_fallback,
    )

    result = asyncio.run(router.call(url="https://example.com"))

    assert result.title == "Example"
    assert calls == ["https://example.com"]
    assert router._states and router._states[0].name == "internal_http_client"
    assert router._states[0].health.successes >= 1


def test_location_providers_execute_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_geocode(location: str) -> Dict[str, Any]:
        return {"location": location}

    async def _fake_current_location() -> Dict[str, Any]:
        return {"city": "Testville"}

    monkeypatch.setattr(openweathermap_provider, "_geocode_location", _fake_geocode)
    monkeypatch.setattr(openweathermap_provider, "_require_api_key", lambda: "token")
    monkeypatch.setattr(
        openweathermap_provider, "os", SimpleNamespace(getenv=lambda *_args, **_kwargs: "token")
    )

    from modules.Tools.providers import ip_api as ip_api_provider

    monkeypatch.setattr(ip_api_provider, "get_current_location", _fake_current_location)

    async def _fallback(**_kwargs: Any) -> Any:  # pragma: no cover - should not execute
        raise AssertionError("Fallback should not execute")

    geo_router = ToolProviderRouter(
        tool_name="geocode_location",
        provider_specs=[{"name": "openweathermap", "priority": 0}],
        fallback_callable=_fallback,
    )
    location_result = asyncio.run(geo_router.call(location="Paris"))

    ip_router = ToolProviderRouter(
        tool_name="get_current_location",
        provider_specs=[{"name": "ip-api", "priority": 0}],
        fallback_callable=_fallback,
    )
    current_location = asyncio.run(ip_router.call())

    assert location_result == {"location": "Paris"}
    assert current_location == {"city": "Testville"}
    assert geo_router._states[0].name == "openweathermap"
    assert ip_router._states[0].name == "ip-api"
    assert geo_router._states[0].health.successes >= 1
    assert ip_router._states[0].health.successes >= 1


def test_service_providers_execute_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_pubmed(**kwargs: Any) -> Any:
        return 200, {"ids": ["PMID123"], "kwargs": kwargs}

    async def _fake_calendar(**kwargs: Any) -> Dict[str, Any]:
        return {"operation": kwargs.get("operation"), "payload": kwargs}

    def _fake_enqueue_task(**kwargs: Any) -> Dict[str, Any]:
        return {"job_id": "task-123", "payload": kwargs}

    monkeypatch.setitem(ncbi_entrez_provider._TOOL_DISPATCH, "search_pubmed", _fake_pubmed)
    monkeypatch.setitem(ncbi_entrez_provider._TOOL_DISPATCH, "search_pmc", _fake_pubmed)
    monkeypatch.setitem(task_queue_provider._CALL_DISPATCH, "task_queue_enqueue", _fake_enqueue_task)

    async def _healthy(self) -> bool:
        return True

    monkeypatch.setattr(task_queue_provider.TaskQueueDefaultProvider, "health_check", _healthy)
    monkeypatch.setattr(debian12_provider, "debian12_calendar", _fake_calendar)

    async def _fallback(**_kwargs: Any) -> Any:  # pragma: no cover - should not execute
        raise AssertionError("Fallback should not execute")

    pubmed_router = ToolProviderRouter(
        tool_name="search_pubmed",
        provider_specs=[{"name": "ncbi_entrez", "priority": 0}],
        fallback_callable=_fallback,
    )
    task_router = ToolProviderRouter(
        tool_name="task_queue_enqueue",
        provider_specs=[{"name": "task_queue_default", "priority": 0}],
        fallback_callable=_fallback,
    )
    calendar_router = ToolProviderRouter(
        tool_name="debian12_calendar",
        provider_specs=[{"name": "debian12_local", "priority": 0}],
        fallback_callable=_fallback,
    )

    status, payload = asyncio.run(pubmed_router.call(query="term"))
    task_result = asyncio.run(task_router.call(name="example", payload={}, idempotency_key="k"))
    calendar_result = asyncio.run(calendar_router.call(operation="list"))

    assert status == 200
    assert payload["ids"] == ["PMID123"]
    assert task_result["job_id"] == "task-123"
    assert calendar_result["operation"] == "list"
    assert pubmed_router._states[0].name == "ncbi_entrez"
    assert task_router._states[0].name == "task_queue_default"
    assert calendar_router._states[0].name == "debian12_local"
    assert pubmed_router._states[0].health.successes >= 1
    assert task_router._states[0].health.successes >= 1
    assert calendar_router._states[0].health.successes >= 1
