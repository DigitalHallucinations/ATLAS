import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
            "reasoning_effort": "high",
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
    monkeypatch.setattr(oa_module, "ModelManager", DummyModelManager)

    generator = oa_module.OpenAIGenerator(DummyConfig())

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
    assert kwargs["input"][0]["role"] == "system"
    assert kwargs["input"][1]["content"][0]["text"] == "hi"


def test_responses_tool_call_invokes_tool(monkeypatch):
    recorded = {}

    async def fake_use_tool(*args, **kwargs):
        recorded["message"] = args[2]
        return "tool-output"

    monkeypatch.setattr(oa_module, "use_tool", fake_use_tool)
    monkeypatch.setattr(oa_module, "ModelManager", DummyModelManager)

    class DummyResponses:
        async def create(self, **_kwargs):
            return SimpleNamespace(
                output=[
                    {
                        "type": "tool_call",
                        "tool_calls": [
                            {"function": {"name": "my_tool", "arguments": {"value": 1}}}
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
        "reasoning_effort": "medium",
    })
    generator = oa_module.OpenAIGenerator(settings)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "call the tool"}],
            model="o1-preview",
            stream=False,
            functions=[{"name": "my_tool"}],
        )

    result = asyncio.run(exercise())

    assert result == "tool-output"
    assert recorded["message"]["function_call"]["name"] == "my_tool"
    assert recorded["message"]["function_call"]["arguments"] == "{\"value\": 1}"
