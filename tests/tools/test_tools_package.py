import asyncio
import json
from typing import Any, Dict, Optional

import pytest

from core.tools import cache, manifests, streaming, execution


class _StubRegistry:
    def __init__(self) -> None:
        self.refresh_calls = 0
        self.payload_calls = 0
        self.revision = 1

    def refresh_if_stale(self) -> None:
        self.refresh_calls += 1

    def refresh(self, force: bool = False) -> None:
        self.refresh_calls += 1

    def persona_has_tool_manifest(self, persona: str) -> bool:
        return False

    def get_tool_manifest_payload(self, persona=None, allowed_names=None):
        self.payload_calls += 1
        return [{"name": "echo"}]

    def record_tool_execution(self, **kwargs) -> None:  # pragma: no cover - metrics stub
        pass


@pytest.fixture(autouse=True)
def reset_manifest_caches():
    with manifests._function_payload_cache_lock:
        manifests._function_payload_cache.clear()
    manifests._function_map_cache.clear()
    manifests._default_function_map_cache = None
    yield
    with manifests._function_payload_cache_lock:
        manifests._function_payload_cache.clear()
    manifests._function_map_cache.clear()
    manifests._default_function_map_cache = None


def test_manifests_load_functions_caches_results(monkeypatch):
    registry = _StubRegistry()
    monkeypatch.setattr(manifests, "get_capability_registry", lambda config_manager=None: registry)

    persona = {"name": "TestPersona"}
    first = manifests.load_functions_from_json(persona, config_manager=None)
    second = manifests.load_functions_from_json(persona, config_manager=None)

    assert first == second
    assert registry.payload_calls == 1


def test_streaming_iterator_collects_activity(monkeypatch):
    async def runner():
        async def fake_stream():
            yield {"content": "hello"}
            yield " world"

        monkeypatch.setattr(cache, "publish_bus_event", lambda *args, **kwargs: None)

        recorded_entries: list[Dict[str, Any]] = []

        def capture(entry: Dict[str, Any], replace: Optional[Dict[str, Any]] = None):
            if replace is not None:
                replace.update(entry)
                return replace
            recorded_entries.append(dict(entry))
            return dict(entry)

        monkeypatch.setattr(streaming, "record_tool_activity", capture)

        capture_result = await streaming.stream_tool_iterator(
            fake_stream(),
            log_entry={"tool_name": "echo", "status": "running"},
        )

        assert capture_result.items == [{"content": "hello"}, " world"]
        assert "world" in capture_result.text
        assert recorded_entries, "activity entries should be captured"

    asyncio.run(runner())


class _StubConversationHistory:
    def __init__(self) -> None:
        self.messages: list[Dict[str, Any]] = []
        self.responses: list[Any] = []

    def add_message(self, user, conversation_id, role, content, timestamp, **kwargs) -> None:
        entry = {"user": user, "conversation_id": conversation_id, "role": role, "content": content, "timestamp": timestamp}
        entry.update(kwargs)
        self.messages.append(entry)

    def get_history(self, user, conversation_id):
        return list(self.messages)

    def add_response(self, user, conversation_id, content, timestamp, **kwargs) -> None:
        self.responses.append(content)

    def add_tool_call(self, *args, **kwargs):  # pragma: no cover - compatibility shim
        pass


class _StubProvider:
    async def generate_response(self, **kwargs):
        return "assistant response"


def test_execution_use_tool_runs_callable(monkeypatch):
    async def runner():
        registry = _StubRegistry()
        monkeypatch.setattr(execution, "get_capability_registry", lambda config_manager=None: registry)
        monkeypatch.setattr(execution.budget_tracker, "resolve_conversation_budget_ms", lambda config_manager: None)
        monkeypatch.setattr(execution.budget_tracker, "get_consumed_runtime_ms", lambda conversation_id: 0)
        monkeypatch.setattr(execution, "record_persona_tool_event", lambda **kwargs: None)
        async def _fake_call_model(*args, **kwargs):
            return "assistant response"

        monkeypatch.setattr(execution, "call_model_with_new_prompt", _fake_call_model)
        monkeypatch.setattr(execution, "_record_tool_activity", lambda entry, replace=None: entry)
        monkeypatch.setattr(execution, "_record_tool_failure", lambda *args, **kwargs: None)
        monkeypatch.setattr(cache, "publish_bus_event", lambda *args, **kwargs: None)

        async def echo(text: str, context: Optional[dict] = None):
            return {"output_text": text.upper(), "context": context or {}}

        function_map = {"echo": {"callable": echo}}
        conversation_history = _StubConversationHistory()

        result = await execution.use_tool(
            user="user",
            conversation_id="conv",
            message={"tool_calls": [{"name": "echo", "arguments": json.dumps({"text": "hi"})}]},
            conversation_history=conversation_history,
            function_map=function_map,
            functions=[],
            current_persona={"name": "Persona"},
            temperature_var=0.5,
            top_p_var=1.0,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=None,
            provider_manager=_StubProvider(),
            config_manager=None,
            stream=False,
            generation_settings=None,
        )

        assert result == "assistant response"
        assert conversation_history.responses, "assistant reply should be recorded"

    asyncio.run(runner())
