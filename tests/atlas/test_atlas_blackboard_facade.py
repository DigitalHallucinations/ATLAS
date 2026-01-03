import sys
import types
from unittest.mock import Mock

import pytest

execution_module = sys.modules.get("ATLAS.tools.execution")
if execution_module is None:
    execution_module = types.ModuleType("ATLAS.tools.execution")
    execution_module.ToolPolicyDecision = type("ToolPolicyDecision", (), {})
    execution_module.SandboxedToolRunner = type("SandboxedToolRunner", (), {})

    def _stub_function(*_args, **_kwargs):
        return None

    for name in (
        "compute_tool_policy_snapshot",
        "use_tool",
        "call_model_with_new_prompt",
        "_freeze_generation_settings",
        "_extract_text_and_audio",
        "_store_assistant_message",
        "_proxy_streaming_response",
        "_resolve_provider_manager",
        "get_required_args",
        "_freeze_metadata",
        "_extract_persona_name",
        "_normalize_persona_allowlist",
        "_join_with_and",
        "_normalize_requires_flags",
        "_coerce_persona_flag_value",
        "_persona_flag_enabled",
        "_collect_missing_flag_requirements",
        "_format_operation_flag_reason",
        "_format_denied_operations_summary",
        "_build_persona_context_snapshot",
        "_has_tool_consent",
        "_request_tool_consent",
        "_evaluate_tool_policy",
        "_get_sandbox_runner",
        "_resolve_tool_timeout_seconds",
        "_generate_idempotency_key",
        "_is_tool_idempotent",
        "_apply_idempotent_retry_backoff",
        "_run_with_timeout",
        "_resolve_function_callable",
    ):
        setattr(execution_module, name, _stub_function)
    sys.modules["ATLAS.tools.execution"] = execution_module

from ATLAS.ATLAS import ATLAS


def _make_atlas_with_server() -> ATLAS:
    atlas = ATLAS.__new__(ATLAS)
    atlas.server = Mock()
    atlas.tenant_id = "tenant-alpha"
    atlas.logger = Mock()
    return atlas


def test_create_blackboard_entry_delegates_with_context() -> None:
    atlas = _make_atlas_with_server()
    atlas.server.create_blackboard_entry.return_value = {"success": True}

    result = atlas.create_blackboard_entry(
        "conversation-1",
        category="Hypothesis",
        title="Exploration",
        content="Initial findings",
        author="Researcher",
        tags=["science", "field"],
    )

    atlas.server.create_blackboard_entry.assert_called_once()
    args, kwargs = atlas.server.create_blackboard_entry.call_args
    assert args[0] == "conversation"
    assert args[1] == "conversation-1"
    payload = args[2]
    assert payload["category"] == "Hypothesis"
    assert payload["title"] == "Exploration"
    assert payload["content"] == "Initial findings"
    assert payload["author"] == "Researcher"
    assert payload["tags"] == ["science", "field"]
    assert payload["metadata"]["tenant_id"] == "tenant-alpha"
    assert kwargs["context"] == {"tenant_id": "tenant-alpha"}
    assert result == {"success": True}


def test_create_blackboard_entry_rejects_invalid_tags() -> None:
    atlas = _make_atlas_with_server()
    with pytest.raises(TypeError):
        atlas.create_blackboard_entry(
            "conversation-1",
            category="claim",
            title="Title",
            content="Body",
            tags="not-valid",
        )


def test_create_blackboard_entry_requires_fields() -> None:
    atlas = _make_atlas_with_server()
    with pytest.raises(ValueError):
        atlas.create_blackboard_entry("", category="claim", title="x", content="y")
    with pytest.raises(ValueError):
        atlas.create_blackboard_entry("scope", category="", title="x", content="y")
    with pytest.raises(ValueError):
        atlas.create_blackboard_entry("scope", category="claim", title="", content="y")
    with pytest.raises(ValueError):
        atlas.create_blackboard_entry("scope", category="claim", title="x", content="")


def test_update_blackboard_entry_provides_context() -> None:
    atlas = _make_atlas_with_server()
    atlas.server.update_blackboard_entry.return_value = {"success": True}

    atlas.update_blackboard_entry(
        "conversation-1",
        "entry-9",
        title="Updated",
        content="Revised",
        metadata={"source": "ui"},
    )

    atlas.server.update_blackboard_entry.assert_called_once()
    args, kwargs = atlas.server.update_blackboard_entry.call_args
    assert args[0] == "conversation"
    assert args[1] == "conversation-1"
    assert args[2] == "entry-9"
    payload = args[3]
    assert payload["title"] == "Updated"
    assert payload["content"] == "Revised"
    assert payload["metadata"]["source"] == "ui"
    assert payload["metadata"]["tenant_id"] == "tenant-alpha"
    assert kwargs["context"] == {"tenant_id": "tenant-alpha"}


def test_update_blackboard_entry_validation() -> None:
    atlas = _make_atlas_with_server()
    with pytest.raises(ValueError):
        atlas.update_blackboard_entry("", "entry", title="x")
    with pytest.raises(ValueError):
        atlas.update_blackboard_entry("scope", "", title="x")
    with pytest.raises(ValueError):
        atlas.update_blackboard_entry("scope", "entry")
    with pytest.raises(ValueError):
        atlas.update_blackboard_entry("scope", "entry", title=" ")
    with pytest.raises(ValueError):
        atlas.update_blackboard_entry("scope", "entry", content=" ")


def test_delete_blackboard_entry_forwards_context() -> None:
    atlas = _make_atlas_with_server()
    atlas.server.delete_blackboard_entry.return_value = {"success": True}

    result = atlas.delete_blackboard_entry("conversation-1", "entry-3")

    atlas.server.delete_blackboard_entry.assert_called_once()
    args, kwargs = atlas.server.delete_blackboard_entry.call_args
    assert args[0] == "conversation"
    assert args[1] == "conversation-1"
    assert args[2] == "entry-3"
    assert kwargs["context"] == {"tenant_id": "tenant-alpha"}
    assert result == {"success": True}


def test_delete_blackboard_entry_requires_identifiers() -> None:
    atlas = _make_atlas_with_server()
    with pytest.raises(ValueError):
        atlas.delete_blackboard_entry("", "entry")
    with pytest.raises(ValueError):
        atlas.delete_blackboard_entry("scope", "")
