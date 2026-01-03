import asyncio
import logging
from typing import Any, Dict, List, Mapping, Optional

from modules.Chat.chat_session import ChatSession


class _PersonaManagerStub:
    def __init__(self, prompt: str, collaboration: Mapping[str, Any]):
        self._prompt = prompt
        self._collaboration = dict(collaboration)

    def get_current_persona_prompt(self) -> str:
        return self._prompt

    def get_active_collaboration_profile(self) -> Dict[str, Any]:
        return dict(self._collaboration)

    def get_skill_collaboration_overrides(self) -> List[Dict[str, Any]]:
        return []


class _ProviderManagerStub:
    def __init__(self, responses: Dict[tuple[str, str], List[Any]], fallback: Any):
        self._responses = {key: list(value) for key, value in responses.items()}
        self._fallback = fallback
        self.calls: List[Dict[str, Any]] = []
        self.current_llm_provider = "primary"
        self._conversation_id: Optional[str] = None

    def set_current_conversation_id(self, conversation_id: str) -> None:
        self._conversation_id = conversation_id

    async def switch_llm_provider(self, provider: str) -> None:
        self.current_llm_provider = provider

    async def generate_response(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **_kwargs: Any,
    ) -> Any:
        record = {
            "provider": provider or self.current_llm_provider,
            "model": model,
            "messages": messages,
        }
        self.calls.append(record)

        active_provider = provider or self.current_llm_provider
        if provider:
            self.current_llm_provider = provider

        key = (active_provider, model)
        queue = self._responses.get(key)
        if queue:
            return queue.pop(0)
        return self._fallback

    def get_default_model_for_provider(self, provider: str) -> str:
        return f"model-{provider}"


class _AtlasStub:
    def __init__(self, collaboration: Mapping[str, Any], responses: Dict[tuple[str, str], List[Any]], fallback: Any):
        self._default_provider = "primary"
        self._default_model = "primary-model"
        self.persona_manager = _PersonaManagerStub("System prompt", collaboration)
        self.provider_manager = _ProviderManagerStub(responses, fallback)
        self.logger = logging.getLogger("ChatSessionNegotiationTest")
        self._tts_payloads: List[Any] = []

    def get_default_provider(self) -> str:
        return self._default_provider

    def get_default_model(self) -> str:
        return self._default_model

    async def maybe_text_to_speech(self, payload: Any) -> None:
        self._tts_payloads.append(payload)


def test_chat_session_uses_negotiation_success() -> None:
    collaboration = {
        "enabled": True,
        "protocol": "vote",
        "quorum": 0.5,
        "timeout": 0.2,
        "participants": [
            {"id": "alpha", "provider": "alpha", "model": "m1"},
            {"id": "beta", "provider": "beta", "model": "m2"},
        ],
    }

    responses = {
        ("alpha", "m1"): [{"text": "Alpha", "score": 0.4}],
        ("beta", "m2"): [{"text": "Beta", "score": 0.8}],
    }
    fallback = {"text": "Fallback", "score": 0.1}
    async def _run() -> tuple[Any, ChatSession, List[Dict[str, Any]]]:
        atlas = _AtlasStub(collaboration, responses, fallback)
        session = ChatSession(atlas)
        result = await session.send_message("Hello")
        return result, session, atlas.provider_manager.calls

    result, session, calls = asyncio.run(_run())

    assert result == {"text": "Beta", "score": 0.8}
    metadata = session.get_history()[-1].get("metadata") or {}
    negotiation_meta = metadata.get("negotiation") or {}
    assert negotiation_meta.get("status") == "success"
    assert session.negotiation_history
    assert calls[0]["provider"] == "alpha"
    assert calls[1]["provider"] == "beta"
    assert len(calls) == 2


def test_chat_session_falls_back_when_quorum_fails() -> None:
    collaboration = {
        "enabled": True,
        "protocol": "vote",
        "quorum": 1.0,
        "timeout": 0.01,
        "participants": [
            {"id": "alpha", "provider": "alpha", "model": "m1"},
            {"id": "beta", "provider": "beta", "model": "m2"},
        ],
    }

    responses: Dict[tuple[str, str], List[Any]] = {
        ("alpha", "m1"): [{"text": "Alpha", "score": 0.7}]
    }

    # Wrap coroutine responses to await inside provider stub
    class _AsyncProviderStub(_ProviderManagerStub):
        async def generate_response(self, *, provider: Optional[str] = None, model: Optional[str] = None, **kwargs: Any) -> Any:
            record = {
                "provider": provider or self.current_llm_provider,
                "model": model,
                "messages": kwargs.get("messages"),
            }
            self.calls.append(record)

            active_provider = provider or self.current_llm_provider
            if provider:
                self.current_llm_provider = provider

            key = (active_provider, model)
            queue = self._responses.get(key)
            if queue:
                payload = queue.pop(0)
                if asyncio.isfuture(payload) or asyncio.iscoroutine(payload):
                    return await payload
                return payload
            if key == ("beta", "m2"):
                await asyncio.sleep(0.1)
                return {"text": "Slow", "score": 0.5}
            return self._fallback

    async def _run() -> tuple[Any, ChatSession, List[Dict[str, Any]]]:
        fallback = {"text": "Fallback", "score": 0.2}
        atlas = _AtlasStub(collaboration, {}, fallback)
        atlas.provider_manager = _AsyncProviderStub(responses, fallback)
        session = ChatSession(atlas)
        result = await session.send_message("Needs fallback")
        return result, session, atlas.provider_manager.calls

    result, session, calls = asyncio.run(_run())

    assert result == {"text": "Fallback", "score": 0.2}
    negotiation_meta = (session.get_history()[-1].get("metadata") or {}).get("negotiation") or {}
    assert negotiation_meta.get("status") in {"timeout", "quorum_failed"}
    assert session.negotiation_history
    assert len(calls) == 3
