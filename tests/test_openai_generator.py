import asyncio
import base64
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

if "openai" not in sys.modules:
    class _StubAsyncOpenAI:  # pragma: no cover - basic import stub
        def __init__(self, *args, **kwargs):
            raise RuntimeError("AsyncOpenAI stub should be monkeypatched in tests")

    sys.modules["openai"] = SimpleNamespace(AsyncOpenAI=_StubAsyncOpenAI)

if "yaml" not in sys.modules:
    sys.modules["yaml"] = SimpleNamespace(safe_load=lambda *_args, **_kwargs: {}, dump=lambda *_a, **_k: None)

if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = SimpleNamespace(
        load_dotenv=lambda *_a, **_k: None,
        set_key=lambda *_a, **_k: None,
        find_dotenv=lambda *_a, **_k: "",
    )

if "tenacity" not in sys.modules:
    def _retry_stub(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    sys.modules["tenacity"] = SimpleNamespace(
        retry=_retry_stub,
        stop_after_attempt=lambda *_a, **_k: None,
        wait_exponential=lambda *_a, **_k: None,
    )

MODULE_PATH = Path(__file__).resolve().parents[1] / "modules" / "Providers" / "OpenAI" / "OA_gen_response.py"
_spec = importlib.util.spec_from_file_location("oa_module", MODULE_PATH)
oa_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oa_module)


class DummyConfig:
    def __init__(self, settings=None):
        self._settings = settings or {
            "model": "gpt-4o",
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 4000,
            "max_output_tokens": 333,
            "stream": False,
            "function_calling": True,
            "parallel_tool_calls": True,
            "tool_choice": None,
            "reasoning_effort": "high",
            "json_mode": False,
            "json_schema": None,
            "enable_code_interpreter": False,
            "enable_file_search": False,
            "audio_enabled": False,
            "audio_voice": "alloy",
            "audio_format": "wav",
        }

    def get_openai_api_key(self):
        return "test-key"

    def get_openai_llm_settings(self):
        return dict(self._settings)


class DummyModelManager:
    def __init__(self, _config):
        self.current_model = None

    def get_current_model(self):
        return self.current_model

    def set_model(self, model, _provider):
        self.current_model = model


class RecordingConversation:
    def __init__(self):
        self.records = []

    def add_message(
        self,
        user,
        conversation_id,
        role,
        content,
        timestamp=None,
        metadata=None,
        **extra,
    ):
        entry = {
            "user": user,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
        }
        if metadata is not None:
            entry["metadata"] = metadata
        entry.update({key: value for key, value in extra.items() if value is not None})
        self.records.append(entry)
        return entry

    def get_history(self, *_args, **_kwargs):  # pragma: no cover - helper for compatibility
        return list(self.records)


def _build_generator(config):
    model_manager = DummyModelManager(config)
    return oa_module.OpenAIGenerator(config, model_manager=model_manager)


def test_openai_generator_uses_supplied_model_manager(monkeypatch):
    config = DummyConfig()
    provided_manager = DummyModelManager(config)

    class _FailingModelManager(DummyModelManager):
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Fallback ModelManager should not be instantiated")

    monkeypatch.setattr(oa_module, "ModelManager", _FailingModelManager)

    class _DummyClient:
        def __init__(self, **_kwargs):
            self.responses = SimpleNamespace()
            self.chat = SimpleNamespace()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: _DummyClient(**kwargs))

    generator = oa_module.OpenAIGenerator(config, model_manager=provided_manager)

    assert generator.model_manager is provided_manager
    assert provided_manager.get_current_model() == config.get_openai_llm_settings()["model"]


def test_reasoning_model_routes_to_responses(monkeypatch):
    captured = {}

    class DummyResponses:
        async def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(output_text="done", output=[])

    class DummyChat:
        class _Completions:
            async def create(self, *args, **kwargs):
                raise AssertionError("chat.completions.create should not be called")

        completions = _Completions()

    class DummyClient:
        def __init__(self, **_):
            self.responses = DummyResponses()
            self.chat = DummyChat()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: DummyClient(**kwargs))

    config = DummyConfig()
    generator = _build_generator(config)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "system", "content": "Stay concise."}, {"role": "user", "content": "hi"}],
            model="o1-mini",
            stream=False,
            functions=[{"name": "test_tool"}],
        )

    result = asyncio.run(exercise())

    assert result == "done"
    kwargs = captured["kwargs"]
    assert kwargs["model"] == "o1-mini"
    assert kwargs["max_output_tokens"] == 333
    assert kwargs["reasoning"] == {"effort": "high"}
    assert kwargs["tools"][0]["function"]["name"] == "test_tool"
    assert kwargs["parallel_tool_calls"] is True
    assert "tool_choice" not in kwargs
    assert kwargs["input"][0]["role"] == "system"
    assert kwargs["input"][1]["content"][0]["text"] == "hi"


def test_responses_tool_call_invokes_tool(monkeypatch):
    recorded = {}
    conversation = RecordingConversation()

    async def fake_use_tool(*args, **kwargs):
        if "message" in kwargs:
            recorded["message"] = kwargs["message"]
        elif len(args) > 2:
            recorded["message"] = args[2]
        else:  # pragma: no cover - defensive fallback
            recorded["message"] = None
        assert len(conversation.records) == 1
        entry = conversation.records[0]
        assert entry["role"] == "assistant"
        return "tool-output"

    monkeypatch.setattr(oa_module, "use_tool", fake_use_tool)

    class DummyResponses:
        async def create(self, **_kwargs):
            return SimpleNamespace(
                output=[
                    {
                        "type": "tool_call",
                        "tool_calls": [
                            {
                                "id": "tool-call-1",
                                "type": "function",
                                "function": {"name": "my_tool", "arguments": {"value": 1}},
                            }
                        ],
                    }
                ],
                output_text="",
            )

    class DummyChat:
        class _Completions:
            async def create(self, *args, **kwargs):
                raise AssertionError("chat.completions.create should not be called")

        completions = _Completions()

    class DummyClient:
        def __init__(self, **_):
            self.responses = DummyResponses()
            self.chat = DummyChat()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: DummyClient(**kwargs))

    settings = DummyConfig({
        "model": "gpt-4o",
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 4000,
        "max_output_tokens": None,
        "stream": False,
        "function_calling": True,
        "parallel_tool_calls": True,
        "tool_choice": None,
        "reasoning_effort": "medium",
        "json_schema": None,
        "enable_code_interpreter": False,
        "enable_file_search": False,
    })
    generator = _build_generator(settings)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "call the tool"}],
            model="o1-preview",
            stream=False,
            functions=[{"name": "my_tool"}],
            conversation_manager=conversation,
            user="tester",
            conversation_id="conv-openai-resp",
        )

    result = asyncio.run(exercise())

    assert result == "tool-output"
    assert recorded["message"]["function_call"]["name"] == "my_tool"
    assert recorded["message"]["function_call"]["arguments"] == "{\"value\": 1}"
    assert conversation.records
    history_entry = conversation.records[0]
    assert history_entry["tool_calls"][0]["function"]["name"] == "my_tool"
    assert history_entry["tool_calls"][0]["id"] == "tool-call-1"


def test_non_reasoning_model_uses_chat_completions(monkeypatch):
    captured = {}

    class DummyResponses:
        async def create(self, **_kwargs):
            raise AssertionError("responses.create should not be called")

    class DummyChat:
        class _Completions:
            async def create(self, **kwargs):
                captured["kwargs"] = kwargs
                message = SimpleNamespace(content="moderated", function_call=None, tool_calls=None)
                choice = SimpleNamespace(message=message)
                return SimpleNamespace(choices=[choice])

        completions = _Completions()

    class DummyClient:
        def __init__(self, **_):
            self.responses = DummyResponses()
            self.chat = DummyChat()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: DummyClient(**kwargs))
    config = DummyConfig()
    generator = _build_generator(config)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "check"}],
            model="omni-moderation-latest",
            stream=False,
        )

    result = asyncio.run(exercise())

    assert result == "moderated"
    kwargs = captured["kwargs"]
    assert kwargs["model"] == "omni-moderation-latest"
    assert kwargs["messages"][0]["content"] == "check"


def test_reasoning_prefixes_can_be_extended_via_config(monkeypatch):
    captured = {}

    class DummyResponses:
        async def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(output_text="done", output=[])

    class DummyChat:
        class _Completions:
            async def create(self, *args, **kwargs):
                raise AssertionError("chat.completions.create should not be called")

        completions = _Completions()

    class DummyClient:
        def __init__(self, **_):
            self.responses = DummyResponses()
            self.chat = DummyChat()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: DummyClient(**kwargs))
    base_settings = DummyConfig().get_openai_llm_settings()
    base_settings["reasoning_model_prefix_allowlist"] = ["custom-"]
    generator = _build_generator(DummyConfig(base_settings))

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "go"}],
            model="custom-reasoner",
            stream=False,
        )

    result = asyncio.run(exercise())

    assert result == "done"
    assert captured["kwargs"]["model"] == "custom-reasoner"


def test_chat_completion_includes_response_format(monkeypatch):
    captured = {}

    class DummyChat:
        class _Completions:
            async def create(self, **kwargs):
                captured["kwargs"] = kwargs
                message = SimpleNamespace(content='{"ok": true}', function_call=None)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        completions = _Completions()

    class DummyClient:
        def __init__(self, **_):
            self.chat = DummyChat()
            self.responses = SimpleNamespace()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: DummyClient(**kwargs))

    settings = DummyConfig({
        "model": "gpt-4o",
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 4000,
        "max_output_tokens": None,
        "stream": False,
        "function_calling": True,
        "parallel_tool_calls": True,
        "tool_choice": None,
        "reasoning_effort": "medium",
        "json_mode": True,
        "json_schema": None,
        "enable_code_interpreter": False,
        "enable_file_search": False,
    })

    generator = _build_generator(settings)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "return json"}],
            model="gpt-4o",
            stream=False,
        )

    result = asyncio.run(exercise())

    assert result == '{"ok": true}'
    kwargs = captured["kwargs"]
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["stream"] is False
    assert "modalities" not in kwargs
    assert "audio" not in kwargs


def test_chat_completion_audio_combines_text_and_audio(monkeypatch):
    captured = {}
    audio_part1 = b"hello"
    audio_part2 = b"world"

    class DummyChat:
        class _Completions:
            async def create(self, **kwargs):
                captured["kwargs"] = kwargs
                nested_audio = {
                    "data": [
                        {"chunk": base64.b64encode(audio_part1).decode("ascii")},
                        base64.b64encode(audio_part2).decode("ascii"),
                    ],
                    "voice": "verse",
                    "format": "mp3",
                    "id": "audio-123",
                }
                message = SimpleNamespace(content="Response text", audio=nested_audio)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        completions = _Completions()

    class DummyClient:
        def __init__(self, **_):
            self.chat = DummyChat()
            self.responses = SimpleNamespace()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: DummyClient(**kwargs))

    settings = DummyConfig({
        "model": "gpt-4o",
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 4000,
        "max_output_tokens": None,
        "stream": False,
        "function_calling": True,
        "parallel_tool_calls": True,
        "tool_choice": None,
        "reasoning_effort": "medium",
        "json_mode": False,
        "json_schema": None,
        "enable_code_interpreter": False,
        "enable_file_search": False,
        "audio_enabled": True,
        "audio_voice": "verse",
        "audio_format": "mp3",
    })

    generator = _build_generator(settings)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )

    result = asyncio.run(exercise())

    assert result["text"] == "Response text"
    assert result["audio"] == audio_part1 + audio_part2
    assert result["audio_voice"] == "verse"
    assert result["audio_format"] == "mp3"
    assert result["audio_id"] == "audio-123"

    kwargs = captured["kwargs"]
    assert kwargs["modalities"] == ["text", "audio"]
    assert kwargs["audio"] == {"voice": "verse", "format": "mp3"}


def test_chat_completion_appends_builtin_tools(monkeypatch):
    captured = {}

    class DummyChat:
        class _Completions:
            async def create(self, **kwargs):
                captured["kwargs"] = kwargs
                message = SimpleNamespace(content="done", function_call=None, tool_calls=None)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        completions = _Completions()

    class DummyClient:
        def __init__(self, **_):
            self.chat = DummyChat()
            self.responses = SimpleNamespace()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: DummyClient(**kwargs))

    settings = DummyConfig({
        "model": "gpt-4o",
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 4000,
        "max_output_tokens": None,
        "stream": False,
        "function_calling": True,
        "parallel_tool_calls": True,
        "tool_choice": None,
        "reasoning_effort": "medium",
        "json_mode": False,
        "json_schema": None,
        "enable_code_interpreter": True,
        "enable_file_search": True,
    })

    generator = _build_generator(settings)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            functions=[{"name": "tool"}],
        )

    result = asyncio.run(exercise())

    assert result == "done"
    tools = captured["kwargs"].get("tools")
    assert tools is not None
    assert {"type": "code_interpreter"} in tools
    assert {"type": "file_search"} in tools
    function_entries = [entry for entry in tools if entry.get("type") == "function"]
    assert function_entries and function_entries[0]["function"]["name"] == "tool"
    kwargs = captured["kwargs"]
    assert "modalities" not in kwargs


def test_chat_completion_streaming_yields_final_audio(monkeypatch):
    captured = {}
    audio_chunk = base64.b64encode(b"stream-audio").decode("ascii")

    class DummyStream:
        def __init__(self):
            self._step = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._step == 0:
                self._step += 1
                delta = {"content": [{"text": "Hello"}]}
                choice = SimpleNamespace(delta=delta)
                return SimpleNamespace(choices=[choice])
            if self._step == 1:
                self._step += 1
                delta = {"audio": {"data": audio_chunk}}
                choice = SimpleNamespace(delta=delta)
                return SimpleNamespace(choices=[choice])
            raise StopAsyncIteration

    class DummyChat:
        class _Completions:
            async def create(self, **kwargs):
                captured["kwargs"] = kwargs
                return DummyStream()

        completions = _Completions()

    class DummyClient:
        def __init__(self, **_):
            self.chat = DummyChat()
            self.responses = SimpleNamespace()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: DummyClient(**kwargs))

    settings = DummyConfig({
        "model": "gpt-4o",
        "audio_enabled": True,
        "audio_voice": "assistant",
        "audio_format": "wav",
    })

    generator = _build_generator(settings)

    async def exercise():
        stream = await generator.generate_response(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        outputs = []
        async for chunk in stream:
            outputs.append(chunk)
        return outputs

    outputs = asyncio.run(exercise())

    assert outputs[0] == "Hello"
    final = outputs[-1]
    assert isinstance(final, dict)
    assert final["text"] == "Hello"
    assert final["audio"] == base64.b64decode(audio_chunk)
    assert final["audio_format"] == "wav"
    assert final["audio_voice"] == "assistant"

    kwargs = captured["kwargs"]
    assert kwargs["modalities"] == ["text", "audio"]
    assert kwargs["audio"] == {"voice": "assistant", "format": "wav"}


def test_chat_completion_uses_json_schema(monkeypatch):
    captured = {}

    class DummyChat:
        class _Completions:
            async def create(self, **kwargs):
                captured["kwargs"] = kwargs
                message = SimpleNamespace(content='{"ok": true}', function_call=None)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        completions = _Completions()

    class DummyClient:
        def __init__(self, **_):
            self.chat = DummyChat()
            self.responses = SimpleNamespace()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: DummyClient(**kwargs))

    schema_payload = {
        "name": "atlas_response",
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    settings = DummyConfig({
        "model": "gpt-4o",
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 4000,
        "max_output_tokens": None,
        "stream": False,
        "function_calling": True,
        "parallel_tool_calls": True,
        "tool_choice": None,
        "reasoning_effort": "medium",
        "json_mode": False,
        "json_schema": schema_payload,
    })

    generator = _build_generator(settings)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "return json"}],
            model="gpt-4o",
            stream=False,
        )

    result = asyncio.run(exercise())

    assert result == '{"ok": true}'
    kwargs = captured["kwargs"]
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": schema_payload,
    }


def test_responses_api_audio_request_and_payload(monkeypatch):
    captured = {}
    audio_bytes = b"reasoning-audio"
    encoded = base64.b64encode(audio_bytes).decode("ascii")

    class DummyResponses:
        async def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                output_text="Reasoned answer",
                output=[
                    {
                        "type": "output_audio",
                        "audio": {
                            "data": encoded,
                            "voice": "narrator",
                            "format": "ogg",
                            "id": "resp-audio",
                        },
                    }
                ],
            )

    class DummyChat:
        class _Completions:
            async def create(self, *args, **kwargs):
                raise AssertionError("chat.completions.create should not be called")

        completions = _Completions()

    class DummyClient:
        def __init__(self, **_):
            self.responses = DummyResponses()
            self.chat = DummyChat()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: DummyClient(**kwargs))

    settings = DummyConfig({
        "model": "o1-mini",
        "audio_enabled": True,
        "audio_voice": "verse",
        "audio_format": "wav",
    })

    generator = _build_generator(settings)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Explain"}],
            model="o1-mini",
            stream=False,
        )

    result = asyncio.run(exercise())

    assert result["text"] == "Reasoned answer"
    assert result["audio"] == audio_bytes
    assert result["audio_voice"] == "narrator"
    assert result["audio_format"] == "ogg"
    assert result["audio_id"] == "resp-audio"

    kwargs = captured["kwargs"]
    assert kwargs["modalities"] == ["text", "audio"]
    assert kwargs["audio"] == {"voice": "verse", "format": "wav"}


def test_responses_streaming_emits_final_audio_chunk(monkeypatch):
    captured = {}
    encoded = base64.b64encode(b"stream-response").decode("ascii")

    class DummyStream:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self._events = iter(
                [
                    {"type": "response.output_text.delta", "delta": "Hi"},
                    {
                        "type": "response.output_audio.delta",
                        "delta": {"data": encoded, "voice": "delta-voice", "format": "mp3"},
                    },
                    {"type": "response.output_text.delta", "delta": " there"},
                ]
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                event = next(self._events)
            except StopIteration:
                raise StopAsyncIteration
            return event

        async def get_final_response(self):
            return SimpleNamespace(
                output_text="Hi there",
                output=[
                    {
                        "type": "output_audio",
                        "audio": {"data": encoded, "voice": "delta-voice", "format": "mp3"},
                    }
                ],
            )

    class DummyResponses:
        def stream(self, **kwargs):
            return DummyStream(**kwargs)

    class DummyChat:
        class _Completions:
            async def create(self, *args, **kwargs):
                raise AssertionError("chat.completions.create should not be called")

        completions = _Completions()

    class DummyClient:
        def __init__(self, **_):
            self.responses = DummyResponses()
            self.chat = DummyChat()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: DummyClient(**kwargs))

    settings = DummyConfig({
        "model": "o1-preview",
        "audio_enabled": True,
        "audio_voice": "verse",
        "audio_format": "wav",
    })

    generator = _build_generator(settings)

    async def exercise():
        stream = await generator.generate_response(
            messages=[{"role": "user", "content": "Explain"}],
            model="o1-preview",
            stream=True,
        )
        outputs = []
        async for chunk in stream:
            outputs.append(chunk)
        return outputs

    outputs = asyncio.run(exercise())

    assert outputs[0] == "Hi"
    assert outputs[1] == " there"
    final = outputs[-1]
    assert isinstance(final, dict)
    assert final["text"] == "Hi there"
    decoded = base64.b64decode(encoded)
    assert final["audio"].startswith(decoded)
    assert len(final["audio"]) >= len(decoded)
    assert final["audio_voice"] == "delta-voice"
    assert final["audio_format"] == "mp3"

    kwargs = captured["kwargs"]
    assert kwargs["modalities"] == ["text", "audio"]
    assert kwargs["audio"] == {"voice": "verse", "format": "wav"}


def test_chat_completion_tool_calls_invokes_tool(monkeypatch):
    recorded = {}
    conversation = RecordingConversation()

    async def fake_use_tool(*args, **kwargs):
        if "message" in kwargs:
            recorded["message"] = kwargs["message"]
        elif len(args) > 2:
            recorded["message"] = args[2]
        else:  # pragma: no cover - defensive fallback
            recorded["message"] = None
        assert len(conversation.records) == 1
        entry = conversation.records[0]
        assert entry["role"] == "assistant"
        return "tool-response"

    monkeypatch.setattr(oa_module, "use_tool", fake_use_tool)

    message = SimpleNamespace(
        content=None,
        function_call=None,
        tool_calls=[
            {
                "id": "openai-call-1",
                "type": "function",
                "function": {"name": "my_tool", "arguments": "{\"value\": 2}"},
            }
        ],
    )

    class DummyChat:
        class _Completions:
            async def create(self, **kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        completions = _Completions()

    class DummyClient:
        def __init__(self, **_):
            self.chat = DummyChat()
            self.responses = SimpleNamespace()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: DummyClient(**kwargs))

    generator = _build_generator(DummyConfig())

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "call"}],
            model="gpt-4o",
            stream=False,
            functions=[{"name": "my_tool"}],
            conversation_manager=conversation,
            user="user-1",
            conversation_id="conv-openai-chat",
        )

    result = asyncio.run(exercise())

    assert result == "tool-response"
    assert recorded["message"]["function_call"]["name"] == "my_tool"
    assert recorded["message"]["function_call"]["arguments"] == "{\"value\": 2}"
    assert conversation.records
    history_entry = conversation.records[0]
    assert history_entry["tool_calls"][0]["function"]["name"] == "my_tool"
    assert history_entry["tool_calls"][0]["id"] == "openai-call-1"


def test_handle_function_call_formats_code_interpreter_output(monkeypatch):
    class DummyChat:
        class _Completions:
            async def create(self, **_kwargs):
                raise AssertionError("chat.completions.create should not be invoked")

        completions = _Completions()

    class DummyClient:
        def __init__(self, **_):
            self.chat = DummyChat()
            self.responses = SimpleNamespace()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: DummyClient(**kwargs))

    generator = _build_generator(DummyConfig())

    async def fake_use_tool(*_args, **_kwargs):
        raise AssertionError("use_tool should not be called for built-in tools")

    monkeypatch.setattr(oa_module, "use_tool", fake_use_tool)

    message = {
        "type": "code_interpreter",
        "code_interpreter": {
            "outputs": [
                {"type": "logs", "logs": "result: 42"},
            ]
        },
    }

    async def exercise():
        return await generator.handle_function_call(
            user="user",
            conversation_id="conv",
            message=message,
            conversation_manager=None,
            function_map=None,
            functions=None,
            current_persona=None,
            temperature=0.0,
            model="gpt-4o",
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

    result = asyncio.run(exercise())

    assert result == "result: 42"


def test_chat_completion_sends_tool_preferences(monkeypatch):
    captured = {}

    class DummyChat:
        class _Completions:
            async def create(self, **kwargs):
                captured["kwargs"] = kwargs
                message = SimpleNamespace(content="done", function_call=None, tool_calls=None)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        completions = _Completions()

    class DummyClient:
        def __init__(self, **_):
            self.chat = DummyChat()
            self.responses = SimpleNamespace()

    monkeypatch.setattr(oa_module, "AsyncOpenAI", lambda **kwargs: DummyClient(**kwargs))

    settings = DummyConfig({
        "model": "gpt-4o",
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 4000,
        "max_output_tokens": None,
        "stream": False,
        "function_calling": True,
        "parallel_tool_calls": False,
        "tool_choice": "required",
        "reasoning_effort": "medium",
        "json_schema": None,
    })

    generator = _build_generator(settings)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o",
            stream=False,
            functions=[{"name": "my_tool"}],
        )

    result = asyncio.run(exercise())

    assert result == "done"
    kwargs = captured["kwargs"]
    assert kwargs["parallel_tool_calls"] is False
    assert kwargs["tool_choice"] == "required"
    assert kwargs["function_call"] == "auto"
