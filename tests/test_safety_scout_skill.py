import asyncio
import importlib
import os
import sys
import types
from pathlib import Path

import pytest

from modules.Skills import load_skill_metadata


def _ensure_yaml(monkeypatch):
    if "yaml" in sys.modules:
        return

    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.dump = lambda *_args, **_kwargs: ""
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)


def _ensure_dotenv(monkeypatch):
    if "dotenv" in sys.modules:
        return

    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *_args, **_kwargs: True
    dotenv_stub.set_key = lambda *_args, **_kwargs: None
    dotenv_stub.find_dotenv = lambda *_args, **_kwargs: ""
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)


def _ensure_pytz(monkeypatch):
    if "pytz" in sys.modules:
        return

    import datetime

    pytz_stub = types.SimpleNamespace(
        timezone=lambda *_args, **_kwargs: datetime.timezone.utc,
        utc=datetime.timezone.utc,
    )
    monkeypatch.setitem(sys.modules, "pytz", pytz_stub)


def _ensure_jsonschema(monkeypatch):
    if "jsonschema" in sys.modules:
        return

    class _DummyValidationError(Exception):
        def __init__(self, message, path=None):
            super().__init__(message)
            self.message = message
            self.path = tuple(path or [])
            self.absolute_path = self.path

    class _DummyValidator:
        def __init__(self, *_args, **_kwargs):
            pass

        def iter_errors(self, *_args, **_kwargs):
            return iter(())

        def validate(self, *_args, **_kwargs):
            return True

    jsonschema_stub = types.ModuleType("jsonschema")
    jsonschema_stub.ValidationError = _DummyValidationError
    jsonschema_stub.Draft7Validator = _DummyValidator
    monkeypatch.setitem(sys.modules, "jsonschema", jsonschema_stub)


@pytest.mark.parametrize("persona_name", ["ATLAS", "Cleverbot"])
def test_safety_scout_skill_executes_policy_reference(persona_name, monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    _ensure_jsonschema(monkeypatch)

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )

    config_manager = tool_manager.ConfigManager()
    skills = load_skill_metadata(config_manager=config_manager)
    safety_scout = next(entry for entry in skills if entry.name == "SafetyScout")

    from ATLAS.SkillManager import SkillExecutionContext, SkillRunResult, use_skill

    context = SkillExecutionContext(
        conversation_id="conv-safety",
        conversation_history=[],
        persona={"name": persona_name},
    )

    async def _run():
        return await use_skill(
            safety_scout,
            context=context,
            tool_inputs={
                "policy_reference": {
                    "query": "Plan to deploy an experimental feature to production",
                    "include_full_text": True,
                    "limit": 2,
                }
            },
            tool_manager=tool_manager,
            timeout_seconds=5,
        )

    result = asyncio.run(_run())

    assert isinstance(result, SkillRunResult)
    assert "policy_reference" in result.tool_results
    payload = result.tool_results["policy_reference"]
    assert isinstance(payload, dict)
    assert payload.get("results"), "expected policy results"
    top_result = payload["results"][0]
    assert "policy_id" in top_result
    assert top_result.get("guidance")
