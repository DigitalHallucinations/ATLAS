import asyncio
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

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
