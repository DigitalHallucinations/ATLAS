import copy
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest

tools_execution_module = sys.modules.setdefault(
    "ATLAS.tools.execution", types.ModuleType("ATLAS.tools.execution")
)
tools_execution_module.ToolPolicyDecision = getattr(
    tools_execution_module,
    "ToolPolicyDecision",
    type("ToolPolicyDecision", (), {}),
)
tools_execution_module.SandboxedToolRunner = getattr(
    tools_execution_module,
    "SandboxedToolRunner",
    type("SandboxedToolRunner", (), {}),
)
tools_execution_module.compute_tool_policy_snapshot = getattr(
    tools_execution_module,
    "compute_tool_policy_snapshot",
    lambda *args, **kwargs: {},
)
tools_execution_module.use_tool = getattr(
    tools_execution_module, "use_tool", lambda *args, **kwargs: None
)
tools_execution_module.call_model_with_new_prompt = getattr(
    tools_execution_module,
    "call_model_with_new_prompt",
    lambda *args, **kwargs: None,
)
tools_execution_module._coerce_persona_flag_value = getattr(
    tools_execution_module,
    "_coerce_persona_flag_value",
    lambda *args, **kwargs: None,
)

if not hasattr(tools_execution_module, "__getattr__"):
    def _tools_execution_getattr(_name: str):  # pragma: no cover - simple stub fallback
        return lambda *args, **kwargs: None

    tools_execution_module.__getattr__ = _tools_execution_getattr

import tests.test_chat_async_helper  # noqa: F401 - ensure GTK stubs are loaded

from gi.repository import Gtk

if "ATLAS.ToolManager" not in sys.modules:
    tool_manager_stub = types.ModuleType("ATLAS.ToolManager")
    tool_manager_stub.load_function_map_from_current_persona = lambda *_args, **_kwargs: {}
    tool_manager_stub.load_functions_from_json = lambda *_args, **_kwargs: {}
    tool_manager_stub.compute_tool_policy_snapshot = lambda *_args, **_kwargs: {}
    tool_manager_stub._resolve_function_callable = (
        lambda *_args, **_kwargs: lambda *_inner_args, **_inner_kwargs: None
    )
    tool_manager_stub.get_tool_activity_log = lambda *_args, **_kwargs: []
    tool_manager_stub.ToolPolicyDecision = type("ToolPolicyDecision", (), {})
    tool_manager_stub.SandboxedToolRunner = type("SandboxedToolRunner", (), {})
    tool_manager_stub.use_tool = lambda *_args, **_kwargs: None
    tool_manager_stub.call_model_with_new_prompt = lambda *_args, **_kwargs: None
    sys.modules["ATLAS.ToolManager"] = tool_manager_stub

pango_module = sys.modules.setdefault("gi.repository.Pango", types.ModuleType("Pango"))
sys.modules.setdefault("Pango", pango_module)

align = getattr(Gtk, "Align", None)
if align is not None and not hasattr(align, "FILL"):
    setattr(align, "FILL", 0)

_chat_helper = sys.modules.get("tests.test_chat_async_helper")
if _chat_helper is not None:
    dummy_base = getattr(_chat_helper, "_DummyWidget", object)
else:  # pragma: no cover - fallback when helper is unavailable
    dummy_base = object

if not hasattr(Gtk, "Scale"):
    class _Scale(dummy_base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._value = 0.0
            self._range = (0.0, 1.0)
            self._digits = 0
            self._draw_value = False
            self._value_pos = None

        def set_value(self, value: float) -> None:
            self._value = float(value)

        def get_value(self) -> float:
            return float(self._value)

        def set_range(self, lower: float, upper: float) -> None:
            self._range = (float(lower), float(upper))

        def set_digits(self, digits: int) -> None:
            self._digits = int(digits)

        def set_draw_value(self, draw: bool) -> None:
            self._draw_value = bool(draw)

        def set_value_pos(self, position) -> None:
            self._value_pos = position

    Gtk.Scale = _Scale

if not hasattr(Gtk, "Separator"):
    Gtk.Separator = type("Separator", (dummy_base,), {})

if not hasattr(Gtk, "PolicyType"):
    Gtk.PolicyType = types.SimpleNamespace(NEVER=0, AUTOMATIC=1)

notebook_cls = getattr(Gtk, "Notebook", None)
if notebook_cls is not None and not hasattr(notebook_cls, "set_scrollable"):
    notebook_cls.set_scrollable = lambda self, *args, **kwargs: None
if notebook_cls is not None and not hasattr(notebook_cls, "set_tab_reorderable"):
    notebook_cls.set_tab_reorderable = lambda self, *args, **kwargs: None
if notebook_cls is not None and not hasattr(notebook_cls, "set_current_page"):
    notebook_cls.set_current_page = lambda self, *args, **kwargs: None
if notebook_cls is not None and not hasattr(notebook_cls, "append_page"):
    notebook_cls.append_page = lambda self, child, label: 0

if not hasattr(Gtk, "SelectionMode"):
    Gtk.SelectionMode = types.SimpleNamespace(NONE=0, SINGLE=1, MULTIPLE=2)

accessible_role = getattr(Gtk, "AccessibleRole", None)
if accessible_role is None:
    Gtk.AccessibleRole = types.SimpleNamespace(LIST=0, LIST_ITEM=1)
else:
    setattr(accessible_role, "LIST", getattr(accessible_role, "LIST", 0))
    setattr(accessible_role, "LIST_ITEM", getattr(accessible_role, "LIST_ITEM", 1))

if not hasattr(Gtk, "ListBoxRow"):
    Gtk.ListBoxRow = type("ListBoxRow", (dummy_base,), {})

user_account_module = types.ModuleType("modules.user_accounts.user_account_service")


class _AccountError(RuntimeError):
    pass


class _DummyRequirements:
    def __init__(self, *args, **kwargs):  # pragma: no cover - simple stub
        self.minimum_length = kwargs.get("minimum_length", 8)
        self.require_numbers = kwargs.get("require_numbers", False)
        self.require_symbols = kwargs.get("require_symbols", False)


user_account_module.AccountLockedError = type("AccountLockedError", (_AccountError,), {})
user_account_module.DuplicateUserError = type("DuplicateUserError", (_AccountError,), {})
user_account_module.InvalidCurrentPasswordError = type(
    "InvalidCurrentPasswordError", (_AccountError,), {}
)
user_account_module.PasswordRequirements = _DummyRequirements

sys.modules.setdefault(
    "modules.user_accounts.user_account_service", user_account_module
)

_user_accounts_pkg = sys.modules.get("modules.user_accounts")
if _user_accounts_pkg is None:
    _user_accounts_pkg = types.ModuleType("modules.user_accounts")
    sys.modules["modules.user_accounts"] = _user_accounts_pkg

setattr(_user_accounts_pkg, "user_account_service", user_account_module)

account_dialog_module = types.ModuleType("GTKUI.UserAccounts.account_dialog")
account_dialog_module.AccountDialog = type("AccountDialog", (dummy_base,), {})
sys.modules.setdefault("GTKUI.UserAccounts.account_dialog", account_dialog_module)


_VECTOR_TOOL_NAMES = {"upsert_vectors", "query_vectors", "delete_namespace"}
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FUNCTIONS_PATH = _REPO_ROOT / "modules" / "Tools" / "tool_maps" / "functions.json"


def _extract_auth_env_keys(auth_block: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(auth_block, dict):
        env_value = auth_block.get("env")
        if isinstance(env_value, str):
            token = env_value.strip()
            if token:
                keys.add(token)
        envs_value = auth_block.get("envs")
        if isinstance(envs_value, dict):
            for value in envs_value.values():
                if isinstance(value, str):
                    token = value.strip()
                    if token:
                        keys.add(token)
        elif isinstance(envs_value, (list, tuple, set)):
            for value in envs_value:
                if isinstance(value, str):
                    token = value.strip()
                    if token:
                        keys.add(token)
    return keys


def _load_vector_env_keys() -> set[str]:
    try:
        raw = _FUNCTIONS_PATH.read_text(encoding="utf-8")
    except OSError:
        return set()
    try:
        payload = json.loads(raw) if raw.strip() else []
    except json.JSONDecodeError:
        return set()
    keys: set[str] = set()
    for entry in payload:
        if isinstance(entry, dict) and entry.get("name") in _VECTOR_TOOL_NAMES:
            keys.update(_extract_auth_env_keys(entry.get("auth")))
    return keys


_VECTOR_ENV_KEYS = _load_vector_env_keys()


class _DummyPersonaManagement:
    def __init__(self, atlas, parent):
        self._widget = Gtk.Box()

    def get_embeddable_widget(self):
        return self._widget


class _DummyProviderManagement:
    def __init__(self, atlas, parent):
        self._widget = Gtk.Box()

    def get_embeddable_widget(self):
        return self._widget


class _ParentWindowStub:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.toasts: list[str] = []

    def show_error_dialog(self, message: str) -> None:
        self.errors.append(message)

    def show_success_toast(self, message: str) -> None:
        self.toasts.append(message)


class _PersonaManagerStub:
    def __init__(self) -> None:
        self.allowed = ["google_search"]
        self.saved_calls: list[Dict[str, Any]] = []
        self.get_calls = 0

    def get_persona(self, persona_name: str) -> Dict[str, Any]:
        self.get_calls += 1
        return {"name": persona_name, "allowed_tools": list(self.allowed)}

    def set_allowed_tools(self, persona_name: str, tools: list[str]) -> Dict[str, Any]:
        record = {"persona": persona_name, "tools": list(tools)}
        self.saved_calls.append(record)
        self.allowed = list(tools)
        return {"success": True, "persona": record}


class _AtlasStub:
    def __init__(self) -> None:
        self.tool_fetches = 0
        self.skill_fetches = 0
        self.task_fetches = 0
        self.task_catalog_fetches = 0
        self.tool_requests: list[Dict[str, Any]] = []
        self.skill_requests: list[Dict[str, Any]] = []
        self.task_requests: list[Dict[str, Any]] = []
        self.task_search_requests: list[Dict[str, Any]] = []
        self.task_transitions: list[Dict[str, Any]] = []
        self.task_contexts: list[Any] = []
        self.settings_updates: list[Dict[str, Any]] = []
        self.credential_updates: list[Dict[str, Any]] = []
        self.persona_manager = _PersonaManagerStub()
        self.task_catalog_requests: list[Dict[str, Any]] = []
        self._unsupported_filters: set[str] = set()
        self._tool_catalog: list[Dict[str, Any]] = [
            {
                "name": "google_search",
                "title": "Google Search",
                "summary": "Search the web for up-to-date information.",
                "capabilities": ["web_search", "news"],
                "persona": None,
                "auth": {"required": True, "provider": "Google", "status": "Linked"},
                "providers": [{"name": "Google"}],
                "version": "1.4.0",
                "safety_level": "standard",
                "health": {"tool": {"success_rate": 0.9}},
                "settings": {"enabled": True, "providers": ["primary"]},
                "credentials": {
                    "GOOGLE_API_KEY": {
                        "configured": False,
                        "hint": "MASKED",
                        "required": True,
                    },
                    "GOOGLE_CSE_ID": {
                        "configured": False,
                        "hint": "",
                        "required": True,
                    },
                    "SERPAPI_KEY": {
                        "configured": False,
                        "hint": "",
                        "optional": True,
                        "required": False,
                    },
                },
            },
            {
                "name": "terminal_command",
                "title": "Terminal",
                "summary": "Execute safe shell commands.",
                "capabilities": ["system"],
                "persona": "Atlas",
                "auth": {"required": False, "status": "Optional"},
                "providers": [{"name": "Localhost"}],
                "version": "2.0.0",
                "safety_level": "elevated",
                "health": {"tool": {"success_rate": 0.65}},
                "settings": {"enabled": True, "shell": "bash"},
                "credentials": {},
            },
            {
                "name": "atlas_curated_search",
                "title": "Atlas Curated Search",
                "summary": "Search curated internal resources.",
                "capabilities": ["curated_search"],
                "persona": None,
                "persona_allowlist": ["Atlas", "Researcher"],
                "auth": {"required": True, "provider": "Atlas", "status": "Linked"},
                "providers": [{"name": "Atlas"}],
                "version": "1.1.0",
                "safety_level": "standard",
                "health": {"tool": {"success_rate": 0.82}},
                "settings": {"enabled": False},
                "credentials": {},
            },
            {
                "name": "restricted_calculator",
                "title": "Restricted Calculator",
                "summary": "Persona restricted tool.",
                "capabilities": ["math"],
                "persona": None,
                "persona_allowlist": ["Researcher"],
                "auth": {"required": False},
                "providers": [{"name": "Localhost"}],
                "version": "0.9.0",
                "safety_level": "restricted",
                "health": {"tool": {"success_rate": 0.4}},
                "settings": {"enabled": False},
                "credentials": {"TOKEN": {"configured": True}},
            },
            {
                "name": "upsert_vectors",
                "title": "Vector Store Upsert",
                "summary": "Manage vector embeddings in connected stores.",
                "capabilities": ["vector_store"],
                "persona": "Shared",
                "auth": {
                    "required": True,
                    "provider": "Vector Store",
                    "status": "Linked",
                    "type": "api_key",
                },
                "providers": [{"name": "Vector Store"}],
                "version": "3.0.1",
                "safety_level": "standard",
                "health": {"tool": {"success_rate": 0.96}},
                "settings": {"enabled": True, "providers": ["in_memory"]},
                "credentials": {
                    key: {"configured": False, "hint": "", "source": "env"}
                    for key in sorted(_VECTOR_ENV_KEYS)
                },
            },
        ]
        self._task_templates: list[Dict[str, Any]] = [
            {
                "name": "MissionControlWeeklyBrief",
                "persona": "Atlas",
                "summary": "Weekly mission control report.",
                "description": "Aggregate updates and risks for mission control stakeholders.",
                "required_skills": ["AtlasReporter", "RiskPulse"],
                "required_tools": ["context_tracker", "priority_queue"],
                "acceptance_criteria": [
                    "Highlights completed, in-progress, and blocked items.",
                    "Surfaces critical risks with mitigation steps.",
                ],
                "escalation_policy": {
                    "level": "Mission Control Duty Lead",
                    "contact": "mission-control@atlas",
                    "timeframe": "Within 1 business day",
                },
                "tags": ["operations", "reporting"],
                "priority": "High",
                "source": "modules/Tasks/tasks.json",
            },
            {
                "name": "AutomationPolicyPrecheck",
                "persona": None,
                "summary": "Compliance review for automation changes.",
                "description": "Confirm automation updates meet governance requirements before deployment.",
                "required_skills": ["Sentinel"],
                "required_tools": ["policy_reference"],
                "acceptance_criteria": [
                    "Confirms applicable automation policies.",
                ],
                "escalation_policy": {
                    "level": "Automation Governance",
                    "contact": "automation-governance@atlas",
                },
                "tags": ["automation", "compliance"],
                "priority": "Medium",
                "source": "modules/Tasks/tasks.json",
            },
        ]
        self._task_catalog: list[Dict[str, Any]] = [
            {
                "id": "task-1",
                "title": "Draft proposal",
                "description": "Prepare an initial proposal for review.",
                "status": "draft",
                "priority": 1,
                "owner_id": "user-1",
                "session_id": "session-1",
                "conversation_id": "conv-1",
                "tenant_id": "default",
                "metadata": {
                    "persona": "Atlas",
                    "required_skills": ["analysis"],
                    "required_tools": ["google_search"],
                    "acceptance_criteria": ["Proposal outline ready"],
                    "dependencies": [
                        {"id": "support-1", "title": "Collect references", "status": "done"}
                    ],
                },
                "created_at": "2024-01-01T10:00:00+00:00",
                "updated_at": "2024-01-01T10:00:00+00:00",
                "due_at": "2024-01-05T12:00:00+00:00",
            },
            {
                "id": "task-2",
                "title": "Deep dive research",
                "description": "Collect supporting data for the proposal.",
                "status": "ready",
                "priority": 2,
                "owner_id": "user-1",
                "session_id": "session-1",
                "conversation_id": "conv-2",
                "tenant_id": "default",
                "metadata": {
                    "persona": "Researcher",
                    "required_skills": ["research"],
                    "required_tools": ["terminal_command"],
                    "acceptance_criteria": ["Data summarized for review"],
                    "dependencies": [
                        {"id": "task-1", "title": "Draft proposal", "status": "draft"}
                    ],
                },
                "created_at": "2024-01-02T09:00:00+00:00",
                "updated_at": "2024-01-02T09:00:00+00:00",
                "due_at": None,
            },
            {
                "id": "task-3",
                "title": "QA review",
                "description": "Review deliverables for completeness.",
                "status": "in_progress",
                "priority": 3,
                "owner_id": None,
                "session_id": None,
                "conversation_id": "conv-3",
                "tenant_id": "default",
                "metadata": {
                    "required_skills": ["quality assurance"],
                    "required_tools": [],
                    "acceptance_criteria": ["Checklist completed"],
                    "dependencies": [],
                },
                "created_at": "2024-01-03T08:00:00+00:00",
                "updated_at": "2024-01-03T08:30:00+00:00",
                "due_at": None,
            },
        ]
        self.server = types.SimpleNamespace(
            get_tools=self._get_tools,
            get_skills=self._get_skills,
            list_tasks=self._list_tasks,
            get_task=self._get_task,
            transition_task=self._transition_task,
            search_tasks=lambda payload, *, context=None: self.search_tasks(**payload),
        )

    def is_initialized(self) -> bool:
        return True

    def get_active_persona_name(self) -> str:
        return "Atlas"

    def list_tools(self) -> list[Dict[str, Any]]:
        self.tool_fetches += 1
        return [copy.deepcopy(entry) for entry in self._tool_catalog]

    def update_tool_settings(self, tool_name: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"tool": tool_name, "settings": copy.deepcopy(settings)}
        self.settings_updates.append(payload)
        for entry in self._tool_catalog:
            if entry.get("name") == tool_name:
                entry_settings = entry.setdefault("settings", {})
                entry_settings.update(copy.deepcopy(settings))
                break
        return {"success": True, "settings": copy.deepcopy(settings)}

    def update_tool_credentials(self, tool_name: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"tool": tool_name, "credentials": copy.deepcopy(credentials)}
        self.credential_updates.append(payload)
        for entry in self._tool_catalog:
            if entry.get("name") == tool_name:
                entry_credentials = entry.setdefault("credentials", {})
                for key, value in credentials.items():
                    block = entry_credentials.setdefault(key, {})
                    block["configured"] = True
                    block["hint"] = "MASKED"
                break
        return {"success": True}

    def _get_tools(self, **kwargs: Any) -> Dict[str, Any]:
        record = dict(kwargs)
        self.tool_requests.append(record)
        self.tool_fetches += 1

        for key in list(record.keys()):
            if key in self._unsupported_filters:
                raise TypeError(f"get_tools() got an unexpected keyword argument '{key}'")

        persona = record.get("persona")
        provider_filter = str(record.get("provider") or "").strip().lower() or None
        safety_filter = str(record.get("safety_level") or "").strip().lower() or None
        version_filter = str(record.get("version") or "").strip() or None

        success_filter = record.get("min_success_rate")
        min_success_rate: float | None
        if success_filter is None:
            min_success_rate = None
        else:
            try:
                min_success_rate = float(success_filter)
            except (TypeError, ValueError):
                min_success_rate = None
            else:
                if min_success_rate > 1.0 and min_success_rate <= 100.0:
                    min_success_rate /= 100.0
                if min_success_rate <= 0.0:
                    min_success_rate = None

        def _matches(entry: Dict[str, Any]) -> bool:
            if persona:
                scope = entry.get("persona")
                if scope:
                    if scope != persona:
                        return False
                else:
                    allowlist = entry.get("persona_allowlist") or []
                    if allowlist and persona not in allowlist:
                        return False

            if provider_filter:
                providers_payload = entry.get("providers") or []
                if isinstance(providers_payload, Mapping):
                    provider_iterable = providers_payload.values()
                elif isinstance(providers_payload, (list, tuple, set)):
                    provider_iterable = providers_payload
                else:
                    provider_iterable = []
                provider_names = set()
                for provider in provider_iterable:
                    if isinstance(provider, Mapping):
                        name_candidate = (
                            provider.get("name")
                            or provider.get("provider")
                            or provider.get("id")
                            or provider.get("label")
                        )
                    else:
                        name_candidate = provider
                    if isinstance(name_candidate, str):
                        token = name_candidate.strip().lower()
                        if token:
                            provider_names.add(token)
                if provider_filter not in provider_names:
                    return False

            if safety_filter:
                entry_safety = str(entry.get("safety_level") or "").strip().lower()
                if entry_safety != safety_filter:
                    return False

            if version_filter:
                entry_version = str(entry.get("version") or "")
                if version_filter.lower() not in entry_version.lower():
                    return False

            if min_success_rate is not None:
                health = entry.get("health") if isinstance(entry.get("health"), Mapping) else None
                tool_metrics = health.get("tool") if isinstance(health, Mapping) else None
                success_value = (
                    tool_metrics.get("success_rate")
                    if isinstance(tool_metrics, Mapping)
                    else None
                )
                rate = None
                if isinstance(success_value, (int, float)):
                    rate = float(success_value)
                else:
                    try:
                        rate = float(str(success_value))
                    except (TypeError, ValueError):
                        rate = None
                if rate is None or rate < min_success_rate:
                    return False

            return True

        tools = [copy.deepcopy(entry) for entry in self._tool_catalog if _matches(entry)]
        return {"count": len(tools), "tools": tools}

    def _get_skills(self, **kwargs: Any) -> Dict[str, Any]:
        self.skill_fetches += 1
        self.skill_requests.append(dict(kwargs))
        return {
            "count": 2,
            "skills": [
                {
                    "name": "ResearchBrief",
                    "summary": "Compose a short research summary using trusted sources.",
                    "version": "1.2.0",
                    "persona": None,
                    "category": "Research",
                    "required_tools": ["google_search"],
                    "required_capabilities": ["web_search"],
                    "capability_tags": ["analysis", "writing"],
                    "safety_notes": "Review generated content for accuracy.",
                    "source": "modules/Skills/skills.json",
                },
                {
                    "name": "FollowUpPlanner",
                    "summary": "Suggest follow-up questions based on the current chat context.",
                    "version": "0.4.1",
                    "persona": "Atlas",
                    "category": "Conversation",
                    "required_tools": [],
                    "required_capabilities": ["conversation"],
                    "capability_tags": ["conversation", "planning"],
                    "safety_notes": "No elevated permissions required.",
                    "source": "modules/Personas/Atlas/Skills/skills.json",
                },
            ],
        }

    def _list_tasks(self, params: Dict[str, Any] | None = None, *, context: Any | None = None) -> Dict[str, Any]:
        params = dict(params or {})
        self.task_requests.append(dict(params))
        if context is not None:
            self.task_contexts.append(context)
        self.task_fetches += 1
        status_filter = params.get("status")
        if isinstance(status_filter, str):
            statuses = {status_filter}
        elif isinstance(status_filter, (list, tuple, set)):
            statuses = {str(value) for value in status_filter}
        else:
            statuses = None
        items = []
        for entry in self._task_catalog:
            if statuses and entry.get("status") not in statuses:
                continue
            items.append(copy.deepcopy(entry))
        return {
            "items": items,
            "page": {"next_cursor": None, "page_size": len(items), "count": len(items)},
        }

    def search_tasks(self, **kwargs: Any) -> Dict[str, Any]:
        payload = {key: copy.deepcopy(value) for key, value in kwargs.items()}
        self.task_search_requests.append(payload)

        status_filter = payload.get("status")
        owner_filter = payload.get("owner_id")
        conversation_filter = payload.get("conversation_id")
        text_query = str(payload.get("text") or "").strip().lower()
        metadata_payload = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        limit = payload.get("limit")
        offset = int(payload.get("offset") or 0)

        def _matches(entry: Mapping[str, Any]) -> bool:
            if status_filter and entry.get("status") != status_filter:
                return False
            if owner_filter and entry.get("owner_id") != owner_filter:
                return False
            if conversation_filter and entry.get("conversation_id") != conversation_filter:
                return False
            if text_query:
                haystack = " ".join(
                    [
                        str(entry.get("title") or ""),
                        str(entry.get("description") or ""),
                    ]
                ).lower()
                if text_query not in haystack:
                    return False
            if metadata_payload:
                metadata = entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else {}
                for key, value in metadata_payload.items():
                    if metadata.get(key) != value:
                        return False
            return True

        filtered = [copy.deepcopy(entry) for entry in self._task_catalog if _matches(entry)]
        self.task_fetches += 1

        use_search_filters = bool(text_query or metadata_payload)
        if use_search_filters:
            if limit is None:
                limit = len(filtered) or 1
            size = max(0, int(limit))
            start = max(0, offset)
            slice_items = filtered[start : start + size]
            return {"items": slice_items, "count": len(filtered)}

        page_items = filtered
        if limit is not None:
            try:
                size = int(limit)
            except (TypeError, ValueError):
                size = len(filtered)
            else:
                if size > 0:
                    page_items = filtered[:size]
        return {
            "items": page_items,
            "page": {
                "next_cursor": None,
                "page_size": len(page_items),
                "count": len(page_items),
            },
        }

    def get_task_catalog(
        self,
        *,
        persona: Any | None = None,
        tags: Any | None = None,
        required_skills: Any | None = None,
        required_tools: Any | None = None,
    ) -> Dict[str, Any]:
        request: Dict[str, Any] = {}
        if persona is not None:
            request["persona"] = persona
        if tags is not None:
            request["tags"] = tags
        if required_skills is not None:
            request["required_skills"] = required_skills
        if required_tools is not None:
            request["required_tools"] = required_tools
        self.task_catalog_requests.append(request)
        self.task_catalog_fetches += 1

        if persona is None:
            filtered = list(self._task_templates)
        else:
            if isinstance(persona, (list, tuple, set)):
                tokens = {str(token).strip().lower() for token in persona}
                persona_token = None
                if tokens:
                    persona_token = next(iter(tokens))
            else:
                persona_token = str(persona).strip().lower()

            if persona_token in {None, "", "all"}:
                filtered = list(self._task_templates)
            elif persona_token == "shared":
                filtered = [entry for entry in self._task_templates if entry.get("persona") is None]
            else:
                filtered = [
                    entry
                    for entry in self._task_templates
                    if entry.get("persona") is None
                    or str(entry.get("persona") or "").strip().lower() == persona_token
                ]

        return {
            "count": len(filtered),
            "tasks": [copy.deepcopy(entry) for entry in filtered],
        }

    def _get_task(
        self,
        task_id: str,
        *,
        context: Any | None = None,
        include_events: bool = False,
    ) -> Dict[str, Any]:
        if context is not None:
            self.task_contexts.append(context)
        for entry in self._task_catalog:
            if entry.get("id") == task_id:
                payload = copy.deepcopy(entry)
                if include_events:
                    payload["events"] = []
                return payload
        return {"id": task_id, "status": "unknown", "metadata": {}}

    def _transition_task(
        self,
        task_id: str,
        target_status: str,
        *,
        context: Any | None = None,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        record = {
            "task_id": task_id,
            "target": str(target_status),
            "context": context,
            "expected": expected_updated_at,
        }
        self.task_transitions.append(record)
        for entry in self._task_catalog:
            if entry.get("id") == task_id:
                entry["status"] = str(target_status)
                entry["updated_at"] = f"2024-02-{len(self.task_transitions):02d}T12:00:00+00:00"
                return copy.deepcopy(entry)
        return {"id": task_id, "status": str(target_status)}


def _walk(widget: Any):
    yield widget
    children = getattr(widget, "children", []) or []
    for child in children:
        yield from _walk(child)


def test_sidebar_adds_tools_button_and_workspace(monkeypatch):
    persona_stub = types.ModuleType("GTKUI.Persona_manager.persona_management")
    persona_stub.PersonaManagement = _DummyPersonaManagement
    provider_stub = types.ModuleType("GTKUI.Provider_manager.provider_management")
    provider_stub.ProviderManagement = _DummyProviderManagement

    monkeypatch.setitem(sys.modules, "GTKUI.Persona_manager.persona_management", persona_stub)
    monkeypatch.setitem(sys.modules, "GTKUI.Provider_manager.provider_management", provider_stub)

    from GTKUI import sidebar

    monkeypatch.setattr(sidebar, "apply_css", lambda: None)

    atlas = _AtlasStub()
    window = sidebar.MainWindow(atlas)

    buttons = [
        widget
        for widget in _walk(window.sidebar)
        if getattr(widget, "_tooltip", "") == "Tools"
    ]
    assert buttons, "Sidebar should include a Tools navigation button"

    window.show_tools_menu()
    assert "tools" in window._pages, "Tools page should be registered after opening"
    assert atlas.tool_fetches >= 1, "Opening tools workspace should query the backend"
    tools_widget = window.tool_management.get_embeddable_widget()
    assert window._pages["tools"] is tools_widget


def test_sidebar_adds_skills_button_and_workspace(monkeypatch):
    persona_stub = types.ModuleType("GTKUI.Persona_manager.persona_management")
    persona_stub.PersonaManagement = _DummyPersonaManagement
    provider_stub = types.ModuleType("GTKUI.Provider_manager.provider_management")
    provider_stub.ProviderManagement = _DummyProviderManagement

    monkeypatch.setitem(sys.modules, "GTKUI.Persona_manager.persona_management", persona_stub)
    monkeypatch.setitem(sys.modules, "GTKUI.Provider_manager.provider_management", provider_stub)

    from GTKUI import sidebar

    monkeypatch.setattr(sidebar, "apply_css", lambda: None)

    atlas = _AtlasStub()
    window = sidebar.MainWindow(atlas)

    buttons = [
        widget
        for widget in _walk(window.sidebar)
        if getattr(widget, "_tooltip", "") == "Skills"
    ]
    assert buttons, "Sidebar should include a Skills navigation button"

    window.show_skills_menu()
    assert "skills" in window._pages, "Skills page should be registered after opening"
    assert atlas.skill_fetches >= 1, "Opening skills workspace should query the backend"
    skills_widget = window.skill_management.get_embeddable_widget()
    assert window._pages["skills"] is skills_widget


def test_tool_management_save_and_reset():
    from GTKUI.Tool_manager.tool_management import ToolManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = ToolManagement(atlas, parent)

    widget = manager.get_embeddable_widget()
    assert widget is not None
    assert atlas.tool_fetches == 1

    google_entry = manager._entry_lookup.get("google_search")
    assert google_entry is not None
    assert google_entry.raw_metadata["settings"] == {"enabled": True, "providers": ["primary"]}
    credentials = google_entry.raw_metadata["credentials"]
    assert credentials["GOOGLE_API_KEY"]["configured"] is False
    assert credentials["GOOGLE_CSE_ID"]["configured"] is False
    assert credentials["SERPAPI_KEY"]["configured"] is False

    manager._select_tool("terminal_command")
    manager._on_switch_state_set(manager._switch, True)
    manager._on_save_clicked(manager._save_button)

    assert atlas.persona_manager.saved_calls, "Save action should persist via the persona manager"
    record = atlas.persona_manager.saved_calls[-1]
    assert record["persona"] == "Atlas"
    assert "terminal_command" in record["tools"]

    # Simulate the persona being reverted on disk and ensure reset reloads state.
    atlas.persona_manager.allowed = ["google_search"]
    manager._on_reset_clicked(manager._reset_button)

    assert atlas.tool_fetches >= 2, "Reset should trigger a fresh backend fetch"
    assert manager._enabled_tools == {"google_search"}
    assert not parent.errors, "Successful actions should not surface errors"


def test_tool_management_settings_editor_updates_backend():
    from GTKUI.Tool_manager.tool_management import ToolManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = ToolManagement(atlas, parent)

    manager.get_embeddable_widget()
    manager._select_tool("google_search")

    field = manager._settings_inputs.get("enabled")
    assert field is not None
    field.set_text("false")

    apply_button = manager._settings_apply_button
    assert apply_button is not None
    manager._on_settings_apply_clicked(apply_button)

    assert atlas.settings_updates, "Settings updates should be sent to the backend stub"
    record = atlas.settings_updates[-1]
    assert record["tool"] == "google_search"
    assert record["settings"]["enabled"] is False
    assert atlas.tool_fetches >= 2, "Applying settings should refresh tool metadata"
    assert parent.toasts and parent.toasts[-1] == "Settings saved."


def test_tool_management_credentials_validation_and_refresh():
    from GTKUI.Tool_manager.tool_management import ToolManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = ToolManagement(atlas, parent)

    manager.get_embeddable_widget()
    manager._select_tool("google_search")

    assert set(manager._credential_inputs) >= {
        "GOOGLE_API_KEY",
        "GOOGLE_CSE_ID",
        "SERPAPI_KEY",
    }

    field = manager._credential_inputs.get("GOOGLE_API_KEY")
    assert field is not None
    placeholder = getattr(field, "placeholder", getattr(field, "_placeholder_text", ""))
    assert placeholder
    assert "MASKED" not in placeholder
    assert all(char == "•" for char in placeholder.strip()), "Credentials placeholder should be masked"

    apply_button = manager._credentials_apply_button
    assert apply_button is not None
    manager._on_credentials_apply_clicked(apply_button)

    assert not atlas.credential_updates, "Empty credentials should not be submitted when required"
    assert "error" in getattr(field, "_css_classes", set()), "Missing required credential should be highlighted"

    field.set_text("secret-token")
    cse_field = manager._credential_inputs.get("GOOGLE_CSE_ID")
    assert cse_field is not None
    cse_field.set_text("cse-id-token")
    manager._on_credentials_apply_clicked(apply_button)

    assert atlas.credential_updates, "Credential updates should reach the backend stub"
    cred_record = atlas.credential_updates[-1]
    assert cred_record["tool"] == "google_search"
    assert cred_record["credentials"]["GOOGLE_API_KEY"] == "secret-token"
    assert cred_record["credentials"]["GOOGLE_CSE_ID"] == "cse-id-token"
    assert "SERPAPI_KEY" not in cred_record["credentials"]
    assert atlas.tool_fetches >= 2
    assert parent.toasts and parent.toasts[-1] == "Credentials saved."
    assert "error" not in getattr(field, "_css_classes", set())


def test_tool_management_vector_credentials_rendered():
    from GTKUI.Tool_manager.tool_management import ToolManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = ToolManagement(atlas, parent)

    manager.get_embeddable_widget()

    scope_widget = manager._scope_selector
    assert scope_widget is not None
    scope_widget.set_active(1)
    manager._on_scope_changed(scope_widget)

    assert "upsert_vectors" in manager._entry_lookup

    manager._select_tool("upsert_vectors")

    env_keys = set(manager._credential_inputs.keys())
    assert _VECTOR_ENV_KEYS, "Vector env keys should be detected from the manifest"
    assert _VECTOR_ENV_KEYS.issubset(env_keys)


def test_tool_management_filter_modes():
    from GTKUI.Tool_manager.tool_management import ToolManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = ToolManagement(atlas, parent)

    widget = manager.get_embeddable_widget()
    assert widget is not None
    assert atlas.tool_fetches == 1

    persona_entries = {entry.name for entry in manager._entries}
    assert persona_entries == {"google_search", "terminal_command", "atlas_curated_search"}

    scope_widget = manager._scope_selector
    assert scope_widget is not None
    assert scope_widget.get_active_text() == "Persona tools"

    scope_widget.set_active(1)
    manager._on_scope_changed(scope_widget)

    assert manager._tool_scope == "all"
    assert atlas.tool_fetches >= 2
    assert scope_widget.get_active_text() == "All tools"
    assert manager._entries, "Entries should remain populated when showing all tools"
    assert not getattr(manager._switch, "_sensitive", True)
    assert not getattr(manager._settings_apply_button, "_sensitive", True)
    assert not getattr(manager._credentials_apply_button, "_sensitive", True)

    all_entries = {entry.name for entry in manager._entries}
    assert "restricted_calculator" in all_entries
    assert "upsert_vectors" in all_entries

    scope_widget.set_active(0)
    manager._on_scope_changed(scope_widget)

    assert manager._tool_scope == "persona"
    assert atlas.tool_fetches >= 3
    assert scope_widget.get_active_text() == "Persona tools"
    manager._select_tool("google_search")
    assert getattr(manager._switch, "_sensitive", False)
    assert getattr(manager._settings_apply_button, "_sensitive", False)
    assert getattr(manager._credentials_apply_button, "_sensitive", False)
    persona_entries_after = {entry.name for entry in manager._entries}
    assert persona_entries_after == {"google_search", "terminal_command", "atlas_curated_search"}


def test_tool_management_backend_filters_refresh_catalog():
    from GTKUI.Tool_manager.tool_management import ToolManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = ToolManagement(atlas, parent)

    manager.get_embeddable_widget()
    manager._tool_scope = "all"
    manager._refresh_state()

    assert manager._provider_selector is not None
    manager._provider_filter = "Vector Store"
    manager._persist_view_preferences()
    manager._refresh_state()

    assert atlas.tool_requests
    assert atlas.tool_requests[-1]["provider"] == "Vector Store"
    assert {entry.name for entry in manager._visible_entries} == {"upsert_vectors"}

    manager._provider_filter = None
    manager._persist_view_preferences()
    manager._refresh_state()

    assert manager._safety_selector is not None
    manager._safety_filter = "restricted"
    manager._persist_view_preferences()
    manager._refresh_state()

    assert atlas.tool_requests[-1]["safety_level"] == "restricted"
    assert {entry.name for entry in manager._visible_entries} == {"restricted_calculator"}

    manager._safety_filter = None
    manager._persist_view_preferences()
    manager._refresh_state()

    scale = manager._success_rate_scale
    assert scale is not None
    scale.set_value(90)
    manager._success_rate_filter = 0.9
    manager._persist_view_preferences()
    manager._refresh_state()

    assert atlas.tool_requests[-1]["min_success_rate"] == pytest.approx(0.9)
    assert {entry.name for entry in manager._visible_entries} == {"google_search", "upsert_vectors"}


def test_tool_management_filter_fallback_when_backend_lacks_support():
    from GTKUI.Tool_manager.tool_management import ToolManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    atlas._unsupported_filters.add("provider")
    manager = ToolManagement(atlas, parent)

    manager.get_embeddable_widget()
    manager._tool_scope = "all"
    manager._refresh_state()

    initial_requests = len(atlas.tool_requests)
    manager._provider_filter = "Vector Store"
    manager._persist_view_preferences()
    manager._refresh_state()

    assert len(atlas.tool_requests) >= initial_requests + 2
    assert atlas.tool_requests[-2]["provider"] == "Vector Store"
    assert "provider" not in atlas.tool_requests[-1]
    assert {entry.name for entry in manager._visible_entries} == {"upsert_vectors"}
    assert not parent.errors


def test_skill_management_renders_payloads():
    from GTKUI.Skill_manager.skill_management import SkillManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = SkillManagement(atlas, parent)

    widget = manager.get_embeddable_widget()
    assert widget is not None
    assert atlas.skill_fetches == 1
    assert manager._entries, "Skill entries should be populated from backend payload"
    assert manager._active_skill is not None
    assert not parent.errors


def test_skill_management_scope_modes():
    from GTKUI.Skill_manager.skill_management import SkillManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = SkillManagement(atlas, parent)

    widget = manager.get_embeddable_widget()
    assert widget is not None
    assert atlas.skill_requests
    assert atlas.skill_requests[-1] == {"persona": "Atlas"}

    scope_widget = manager._scope_selector
    assert scope_widget is not None
    assert scope_widget.get_active_text() == "Persona skills"

    initial_fetches = atlas.skill_fetches

    scope_widget.set_active(1)
    manager._on_scope_changed(scope_widget)

    assert manager._skill_scope == "all"
    assert atlas.skill_requests[-1] == {}
    assert atlas.skill_fetches > initial_fetches
    assert scope_widget.get_active_text() == "All skills"
    assert manager._entries, "Entries should remain populated when showing all skills"

    scope_widget.set_active(0)
    manager._on_scope_changed(scope_widget)

    assert manager._skill_scope == "persona"
    assert atlas.skill_requests[-1] == {"persona": "Atlas"}
    assert scope_widget.get_active_text() == "Persona skills"
