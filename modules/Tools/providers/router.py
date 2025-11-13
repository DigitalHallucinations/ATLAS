"""Provider routing for tool execution."""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

from modules.logging.logger import setup_logger
from modules.orchestration.capability_registry import get_capability_registry

from .base import ProviderHealth, ToolProvider, ToolProviderSpec
from .registry import tool_provider_registry


async def _invoke_callable(callable_obj: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    result = callable_obj(**kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


class _ProviderState:
    __slots__ = ("provider", "spec", "health", "lock")

    def __init__(self, provider: ToolProvider, spec: ToolProviderSpec) -> None:
        self.provider = provider
        self.spec = spec
        self.health: ProviderHealth = provider.health
        self.lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def priority(self) -> int:
        return self.provider.priority

    @property
    def health_interval(self) -> float:
        return self.provider.health_check_interval


class ToolProviderRouter:
    """Route tool invocations to the healthiest available provider."""

    def __init__(
        self,
        *,
        tool_name: str,
        provider_specs: Sequence[Mapping[str, Any]],
        fallback_callable: Optional[Callable[..., Any]] = None,
        persona: Optional[str] = None,
    ) -> None:
        self._tool_name = tool_name
        self._logger = setup_logger(f"{__name__}.{tool_name}")
        self._fallback_callable = fallback_callable
        self._states: List[_ProviderState] = []
        self._metrics_callbacks: List[Callable[[Mapping[str, Any]], None]] = []
        self._lock = asyncio.Lock()
        self._persona = persona

        for raw_spec in provider_specs:
            try:
                spec = ToolProviderSpec.from_mapping(raw_spec)
            except ValueError as exc:
                self._logger.warning("Skipping invalid provider specification for %s: %s", tool_name, exc)
                continue
            try:
                provider = tool_provider_registry.create(
                    spec,
                    tool_name=tool_name,
                    fallback_callable=fallback_callable,
                )
            except KeyError as exc:
                self._logger.warning("No provider registered for '%s': %s", spec.name, exc)
                continue
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.error(
                    "Failed to initialize provider '%s' for tool '%s': %s",
                    spec.name,
                    tool_name,
                    exc,
                    exc_info=True,
                )
                continue
            self._states.append(_ProviderState(provider, spec))

        self._states.sort(key=lambda state: (state.priority, state.name))
        self._register_default_metrics_callback()

    def register_metrics_callback(self, callback: Callable[[Mapping[str, Any]], None]) -> None:
        if callback is None:
            return
        self._metrics_callbacks.append(callback)

    async def call(self, **kwargs: Any) -> Any:
        errors: List[Exception] = []
        attempted: set[str] = set()

        while True:
            async with self._lock:
                state = await self._select_provider(exclude=attempted)
                if state is None:
                    break
                attempted.add(state.name)

            try:
                start = time.perf_counter()
                result = await self._invoke_provider(state, kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                errors.append(exc)
                async with self._lock:
                    await self._record_failure(state, exc)
                    self._emit_metrics(selected=state.name, success=False, latency_ms=elapsed_ms)
                continue
            else:
                async with self._lock:
                    await self._record_success(state)
                    self._emit_metrics(selected=state.name, success=True, latency_ms=elapsed_ms)
                return result

        if self._fallback_callable is not None:
            try:
                start = time.perf_counter()
                result = await _invoke_callable(self._fallback_callable, kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                errors.append(exc)
                async with self._lock:
                    self._emit_metrics(selected="fallback", success=False, latency_ms=elapsed_ms)
            else:
                self._logger.info(
                    "Executed fallback callable for tool '%s' after exhausting providers.",
                    self._tool_name,
                )
                async with self._lock:
                    self._emit_metrics(selected="fallback", success=True, latency_ms=elapsed_ms)
                return result

        async with self._lock:
            self._emit_metrics(selected=None, success=False, latency_ms=None)
        if errors:
            raise errors[-1]
        raise RuntimeError(f"No providers available for tool '{self._tool_name}'")

    async def _select_provider(self, *, exclude: Iterable[str]) -> Optional[_ProviderState]:
        now = time.time()
        exclude_set = {name.lower() for name in exclude}
        candidates: List[Tuple[float, int, str, _ProviderState]] = []

        for state in self._states:
            if state.name.lower() in exclude_set:
                continue

            await self._maybe_run_health_check(state, now)
            if not state.health.is_available(now):
                self._logger.debug(
                    "Provider '%s' for tool '%s' is backing off until %s",
                    state.name,
                    self._tool_name,
                    state.health.backoff_until,
                )
                continue

            score = state.health.score()
            candidates.append((score, state.priority, state.name, state))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][3]

    async def _maybe_run_health_check(self, state: _ProviderState, timestamp: float) -> None:
        if not state.health.should_run_health_check(state.health_interval, timestamp):
            return

        async with state.lock:
            if not state.health.should_run_health_check(state.health_interval, timestamp):
                return
            try:
                healthy = await _invoke_callable(state.provider.health_check, {})
            except Exception as exc:
                self._logger.warning(
                    "Health check failed for provider '%s' on tool '%s': %s",
                    state.name,
                    self._tool_name,
                    exc,
                )
                state.health.record_failure(timestamp)
            else:
                if healthy:
                    state.health.record_success(timestamp)
                else:
                    self._logger.debug(
                        "Health check reported unhealthy for provider '%s' on tool '%s'",
                        state.name,
                        self._tool_name,
                    )
                    state.health.record_failure(timestamp)

    async def _invoke_provider(self, state: _ProviderState, kwargs: Mapping[str, Any]) -> Any:
        return await _invoke_callable(state.provider.call, kwargs)

    async def _record_failure(self, state: _ProviderState, error: Exception) -> None:
        async with state.lock:
            delay = state.health.record_failure()
            failure_rate = state.health.failure_rate
        self._logger.warning(
            "Provider '%s' for tool '%s' failed (failure_rate=%.2f, backoff=%ss): %s",
            state.name,
            self._tool_name,
            failure_rate,
            f"{delay:.2f}",
            error,
        )

    async def _record_success(self, state: _ProviderState) -> None:
        async with state.lock:
            state.health.record_success()
            failure_rate = state.health.failure_rate
        self._logger.info(
            "Provider '%s' for tool '%s' succeeded (failure_rate=%.2f)",
            state.name,
            self._tool_name,
            failure_rate,
        )

    def _emit_metrics(self, *, selected: Optional[str], success: bool, latency_ms: Optional[float]) -> None:
        summary = {
            "tool": self._tool_name,
            "selected": selected,
            "success": success,
            "providers": {
                state.name: state.health.snapshot() for state in self._states
            },
        }
        if latency_ms is not None:
            try:
                summary["latency_ms"] = float(latency_ms)
            except (TypeError, ValueError):
                summary["latency_ms"] = None
        summary["timestamp"] = time.time()
        self._logger.debug("Provider health summary: %s", summary)
        for callback in list(self._metrics_callbacks):
            try:
                callback(summary)
            except Exception:  # pragma: no cover - defensive hook handling
                self._logger.warning(
                    "Metrics callback for tool '%s' raised an exception",
                    self._tool_name,
                    exc_info=True,
                )

    def _register_default_metrics_callback(self) -> None:
        def _forward(summary: Mapping[str, Any]) -> None:
            try:
                registry = get_capability_registry()
                registry.record_provider_metrics(
                    persona=self._persona,
                    tool_name=self._tool_name,
                    summary=summary,
                )
            except Exception:  # pragma: no cover - defensive guard for registry lookup
                self._logger.debug(
                    "Failed to record provider metrics for tool '%s'", self._tool_name, exc_info=True
                )

        self.register_metrics_callback(_forward)


__all__ = ["ToolProviderRouter"]
