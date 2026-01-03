import sys
import types
from types import SimpleNamespace

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

import tests.test_chat_async_helper as chat_helper  # noqa: F401 - ensure GTK stubs
from GTKUI.Chat.chat_page import ChatPage, Gtk
from tests.test_tool_management_ui import _ParentWindowStub


class _BlackboardAtlasStub(chat_helper._AtlasForChatPage):
    def __init__(self):
        super().__init__()
        self.chat_session = SimpleNamespace(get_conversation_id=lambda: "conversation-1")
        self.entries: list[dict[str, object]] = []
        self.blackboard_actions: list[tuple[str, object]] = []

    def get_blackboard_summary(self, scope_id, *, scope_type="conversation"):
        return {"scope_id": scope_id, "scope_type": scope_type, "counts": {}, "entries": list(self.entries)}

    def create_blackboard_entry(
        self,
        scope_id,
        *,
        scope_type="conversation",
        category,
        title,
        content,
        author=None,
        tags=None,
        metadata=None,
    ):
        entry_id = f"entry-{len(self.entries) + 1}"
        entry = {
            "id": entry_id,
            "scope_id": scope_id,
            "scope_type": scope_type,
            "category": category,
            "title": title,
            "content": content,
            "author": author,
        }
        self.entries.append(entry)
        self.blackboard_actions.append(("create", scope_id, category, title, content, author))
        return {"success": True, "entry": entry}

    def update_blackboard_entry(
        self,
        scope_id,
        entry_id,
        *,
        scope_type="conversation",
        title=None,
        content=None,
        tags=None,
        metadata=None,
    ):
        for entry in self.entries:
            if entry.get("id") == entry_id:
                if title is not None:
                    entry["title"] = title
                if content is not None:
                    entry["content"] = content
                break
        self.blackboard_actions.append(("update", scope_id, entry_id))
        return {"success": True}

    def delete_blackboard_entry(self, scope_id, entry_id, *, scope_type="conversation"):
        self.entries = [entry for entry in self.entries if entry.get("id") != entry_id]
        self.blackboard_actions.append(("delete", scope_id, entry_id))
        return {"success": True}


def _invoke_clicked(button: Gtk.Button) -> None:
    for signal, callback in getattr(button, "_callbacks", []):
        if signal == "clicked":
            callback(button)


def test_chat_page_blackboard_create_and_delete(monkeypatch):
    atlas = _BlackboardAtlasStub()
    parent = _ParentWindowStub()

    page = ChatPage.__new__(ChatPage)
    page.ATLAS = atlas
    page._parent_window = parent
    page.get_root = lambda: parent
    page.blackboard_tab_box = Gtk.Box()
    page.blackboard_summary_label = Gtk.Label()
    page.blackboard_tab_box.append(page.blackboard_summary_label)
    page._blackboard_category_options = ["hypothesis", "claim", "artifact"]
    page._blackboard_editing_entry_id = None
    page._blackboard_entries_snapshot = {}

    def _clear_list_stub(self, list_widget=None):
        if list_widget is None:
            list_widget = getattr(self, "blackboard_list", None)
        if list_widget is None:
            return
        list_widget.children = []

    monkeypatch.setattr(ChatPage, "_clear_blackboard_list", _clear_list_stub, raising=False)

    page._build_blackboard_authoring_controls()
    page.blackboard_list = Gtk.ListBox()

    category_combo = page.blackboard_category_combo
    assert category_combo is not None
    if hasattr(category_combo, "set_active"):
        category_combo.set_active(0)

    page.blackboard_title_entry.set_text("Observation")
    page.blackboard_content_buffer.set_text("Shared findings")
    page.blackboard_author_entry.set_text("Analyst")

    page._on_blackboard_submit(page.blackboard_submit_button)

    assert atlas.blackboard_actions[0][:3] == ("create", "conversation-1", "hypothesis")
    assert page.blackboard_list.children, "Entry row should be rendered"
    row = page.blackboard_list.children[0]
    header_box = row.children[0]
    heading = header_box.children[0]
    assert "Observation" in heading.get_text()

    actions_box = header_box.children[1]
    delete_button = actions_box.children[1]
    _invoke_clicked(delete_button)

    assert atlas.blackboard_actions[-1] == ("delete", "conversation-1", "entry-1")
    assert page.blackboard_list.children, "Placeholder should remain"
    placeholder = page.blackboard_list.children[0]
    assert "No shared posts" in placeholder.get_text()
