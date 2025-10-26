"""Comprehensive GTK first-run setup wizard for ATLAS."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from ATLAS.config import ConfigManager, _DEFAULT_CONVERSATION_STORE_DSN
from modules.conversation_store.bootstrap import bootstrap_conversation_store
from modules.conversation_store.repository import ConversationStoreRepository
from modules.user_accounts.user_account_service import UserAccountService


Callback = Callable[[], None]
ErrorCallback = Callable[[BaseException], None]


AssistantBase = getattr(Gtk, "Assistant", Gtk.Window)


def _ensure_widget(widget_name: str, fallback: type[Gtk.Widget]) -> type[Gtk.Widget]:
    widget = getattr(Gtk, widget_name, None)
    if widget is None:
        class _FallbackWidget(fallback):  # type: ignore[misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - GTK parity
                super().__init__()
                self._text = ""
                self._active = False

            # Gtk.Entry compatibility -------------------------------------------------
            def set_text(self, text: str) -> None:
                self._text = text

            def get_text(self) -> str:
                return self._text

            # Gtk.Switch compatibility -----------------------------------------------
            def set_active(self, active: bool) -> None:
                self._active = bool(active)

            def get_active(self) -> bool:
                return bool(self._active)

            # Gtk.ComboBoxText compatibility ----------------------------------------
            def remove_all(self) -> None:
                self._items: list[str] = []

            def append_text(self, text: str) -> None:
                if not hasattr(self, "_items"):
                    self._items = []
                self._items.append(text)

            def set_active_text(self, text: str | None) -> None:
                self._text = text or ""

            def get_active_text(self) -> str | None:
                value = getattr(self, "_text", "")
                return value or None

        return _FallbackWidget
    return widget


Entry = _ensure_widget("Entry", Gtk.Box)
Switch = _ensure_widget("Switch", Gtk.Box)
ComboBoxText = _ensure_widget("ComboBoxText", Gtk.Box)
Stack = getattr(Gtk, "Stack", None)


if Stack is None:
    class Stack(Gtk.Box):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            self._children: Dict[str, Gtk.Widget] = {}
            self._visible: Optional[str] = None

        def add_titled(self, widget: Gtk.Widget, name: str, _title: str) -> None:
            self._children[name] = widget
            if self._visible is None:
                self._visible = name

        def set_visible_child_name(self, name: str) -> None:
            if name in self._children:
                self._visible = name

        def get_visible_child_name(self) -> str:
            return self._visible or next(iter(self._children.keys()), "")

        def set_vexpand(self, _expand: bool) -> None:
            return None

        def get_visible_child(self) -> Gtk.Widget | None:
            name = self.get_visible_child_name()
            return self._children.get(name)


@dataclass
class DatabaseState:
    host: str = "localhost"
    port: int = 5432
    database: str = "atlas"
    user: str = "atlas"
    password: str = ""
    dsn: str = ""


@dataclass
class RetryPolicyState:
    max_attempts: int = 3
    backoff_seconds: float = 30.0
    jitter_seconds: float = 5.0
    backoff_multiplier: float = 2.0


@dataclass
class JobSchedulingState:
    enabled: bool = False
    job_store_url: Optional[str] = None
    max_workers: Optional[int] = None
    retry_policy: RetryPolicyState = field(default_factory=RetryPolicyState)
    timezone: Optional[str] = None
    queue_size: Optional[int] = None


@dataclass
class MessageBusState:
    backend: str = "in_memory"
    redis_url: Optional[str] = None
    stream_prefix: Optional[str] = None


@dataclass
class ProviderState:
    default_provider: Optional[str] = None
    default_model: Optional[str] = None
    api_keys: Dict[str, str] = field(default_factory=dict)


@dataclass
class SpeechState:
    tts_enabled: bool = False
    stt_enabled: bool = False
    default_tts_provider: Optional[str] = None
    default_stt_provider: Optional[str] = None
    elevenlabs_key: Optional[str] = None
    openai_key: Optional[str] = None
    google_credentials: Optional[str] = None


@dataclass
class UserState:
    username: str = ""
    email: str = ""
    password: str = ""
    display_name: str = ""


@dataclass
class OptionalState:
    tenant_id: Optional[str] = None
    retention_days: Optional[int] = None
    retention_history_limit: Optional[int] = None
    scheduler_timezone: Optional[str] = None
    scheduler_queue_size: Optional[int] = None
    http_auto_start: bool = False


@dataclass
class WizardState:
    database: DatabaseState = field(default_factory=DatabaseState)
    job_scheduling: JobSchedulingState = field(default_factory=JobSchedulingState)
    message_bus: MessageBusState = field(default_factory=MessageBusState)
    providers: ProviderState = field(default_factory=ProviderState)
    speech: SpeechState = field(default_factory=SpeechState)
    user: UserState = field(default_factory=UserState)
    optional: OptionalState = field(default_factory=OptionalState)


def _parse_default_dsn(dsn: str) -> DatabaseState:
    from urllib.parse import urlparse

    parsed = urlparse(dsn)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    database = parsed.path.lstrip("/") or "atlas"
    user = parsed.username or "atlas"
    password = parsed.password or ""
    return DatabaseState(host=host, port=port, database=database, user=user, password=password, dsn=dsn)


def _compose_dsn(state: DatabaseState) -> str:
    auth = state.user or ""
    if state.password:
        auth = f"{auth}:{state.password}"
    if auth:
        auth = f"{auth}@"
    return f"postgresql+psycopg://{auth}{state.host}:{state.port}/{state.database}"


class SetupWizardController:
    """Coordinate persistence logic for the setup wizard."""

    def __init__(
        self,
        *,
        config_manager: Optional[ConfigManager] = None,
        atlas: Any | None = None,
        bootstrap: Callable[[str], str] = bootstrap_conversation_store,
    ) -> None:
        self.config_manager = config_manager or ConfigManager()
        self.atlas = atlas
        self._bootstrap = bootstrap
        self.state = WizardState()
        self._load_defaults()

    # -- state loaders -----------------------------------------------------

    def _load_defaults(self) -> None:
        database_url = self.config_manager.get_conversation_database_config().get("url")
        if isinstance(database_url, str) and database_url.strip():
            self.state.database = _parse_default_dsn(database_url)
        else:
            self.state.database = _parse_default_dsn(_DEFAULT_CONVERSATION_STORE_DSN)

        job_settings = self.config_manager.get_job_scheduling_settings()
        retry = job_settings.get("retry_policy") or {}
        self.state.job_scheduling = JobSchedulingState(
            enabled=bool(job_settings.get("enabled")),
            job_store_url=job_settings.get("job_store_url"),
            max_workers=job_settings.get("max_workers"),
            retry_policy=RetryPolicyState(
                max_attempts=int(retry.get("max_attempts", 3) or 3),
                backoff_seconds=float(retry.get("backoff_seconds", 30.0) or 30.0),
                jitter_seconds=float(retry.get("jitter_seconds", 5.0) or 5.0),
                backoff_multiplier=float(retry.get("backoff_multiplier", 2.0) or 2.0),
            ),
            timezone=job_settings.get("timezone"),
            queue_size=job_settings.get("queue_size"),
        )

        messaging = self.config_manager.get_messaging_settings()
        backend = str(messaging.get("backend", "in_memory"))
        self.state.message_bus = MessageBusState(
            backend=backend,
            redis_url=messaging.get("redis_url"),
            stream_prefix=messaging.get("stream_prefix"),
        )

        provider_keys = self.config_manager._get_provider_env_keys()
        api_key_state: Dict[str, str] = {}
        for provider, env_key in provider_keys.items():
            value = self.config_manager.get_config(env_key)
            if isinstance(value, str) and value.strip():
                api_key_state[provider] = value.strip()
        self.state.providers = ProviderState(
            default_provider=self.config_manager.get_default_provider(),
            default_model=self.config_manager.get_default_model(),
            api_keys=api_key_state,
        )

        tts_enabled = bool(self.config_manager.get_tts_enabled()) if hasattr(self.config_manager, "get_tts_enabled") else False
        default_tts = self.config_manager.get_config("DEFAULT_TTS_PROVIDER")
        default_stt = self.config_manager.get_config("DEFAULT_STT_PROVIDER")
        speech_state = SpeechState(
            tts_enabled=tts_enabled,
            stt_enabled=bool(default_stt),
            default_tts_provider=default_tts,
            default_stt_provider=default_stt,
        )
        speech_state.elevenlabs_key = self.config_manager.get_config("XI_API_KEY")
        speech_state.openai_key = self.config_manager.get_config("OPENAI_API_KEY")
        speech_state.google_credentials = self.config_manager.get_config("GOOGLE_APPLICATION_CREDENTIALS")
        self.state.speech = speech_state

        tenant_value = self.config_manager.get_config("tenant_id") or self.config_manager.env_config.get("TENANT_ID")
        retention = self.config_manager.get_conversation_retention_policies()
        http_block = self.config_manager.get_config("http_server") or {}
        self.state.optional = OptionalState(
            tenant_id=str(tenant_value).strip() if tenant_value else None,
            retention_days=retention.get("days") or retention.get("max_days"),
            retention_history_limit=retention.get("history_message_limit"),
            scheduler_timezone=self.state.job_scheduling.timezone,
            scheduler_queue_size=self.state.job_scheduling.queue_size,
            http_auto_start=bool(http_block.get("auto_start")) if isinstance(http_block, Mapping) else False,
        )

    # -- database -----------------------------------------------------------

    def apply_database_settings(self, state: DatabaseState) -> str:
        dsn = _compose_dsn(state)
        ensured = self._bootstrap(dsn)
        self.config_manager._persist_conversation_database_url(ensured)
        self.config_manager._write_yaml_config()
        self.state.database = dataclasses.replace(state, dsn=ensured)
        return ensured

    # -- job scheduling ----------------------------------------------------

    def apply_job_scheduling(self, state: JobSchedulingState) -> Dict[str, Any]:
        policy = {
            "max_attempts": state.retry_policy.max_attempts,
            "backoff_seconds": state.retry_policy.backoff_seconds,
            "jitter_seconds": state.retry_policy.jitter_seconds,
            "backoff_multiplier": state.retry_policy.backoff_multiplier,
        }
        settings = self.config_manager.set_job_scheduling_settings(
            enabled=state.enabled,
            job_store_url=state.job_store_url or ConfigManager.UNSET,
            max_workers=state.max_workers if state.max_workers is not None else ConfigManager.UNSET,
            retry_policy=policy,
            timezone=state.timezone or ConfigManager.UNSET,
            queue_size=state.queue_size if state.queue_size is not None else ConfigManager.UNSET,
        )
        self.state.job_scheduling = dataclasses.replace(state)
        return settings

    # -- messaging ---------------------------------------------------------

    def apply_message_bus(self, state: MessageBusState) -> Dict[str, Any]:
        settings = self.config_manager.set_messaging_settings(
            backend=state.backend,
            redis_url=state.redis_url,
            stream_prefix=state.stream_prefix,
        )
        self.state.message_bus = dataclasses.replace(state)
        return settings

    # -- providers ---------------------------------------------------------

    def apply_provider_settings(self, state: ProviderState) -> ProviderState:
        provider_keys = self.config_manager._get_provider_env_keys()
        for provider, key in provider_keys.items():
            if provider in state.api_keys:
                self.config_manager.update_api_key(provider, state.api_keys[provider])

        if state.default_provider:
            self.config_manager.set_default_provider(state.default_provider)
        if state.default_model:
            self.config_manager.set_default_model(state.default_model)

        self.state.providers = dataclasses.replace(state, api_keys=dict(state.api_keys))
        return self.state.providers

    # -- speech ------------------------------------------------------------

    def apply_speech_settings(self, state: SpeechState) -> SpeechState:
        self.config_manager.set_tts_enabled(bool(state.tts_enabled))
        self.config_manager.set_default_speech_providers(
            tts_provider=state.default_tts_provider,
            stt_provider=state.default_stt_provider if state.stt_enabled else None,
        )
        if state.elevenlabs_key:
            self.config_manager.set_elevenlabs_api_key(state.elevenlabs_key)
        if state.openai_key:
            self.config_manager.set_openai_speech_config(api_key=state.openai_key)
        if state.google_credentials:
            self.config_manager.set_google_speech_settings(tts_voice=None, stt_language=None)
            self.config_manager.set_google_credentials(state.google_credentials)

        self.state.speech = dataclasses.replace(state)
        return self.state.speech

    # -- user --------------------------------------------------------------

    def register_user(self, state: UserState) -> Mapping[str, Any]:
        repository = self._get_conversation_repository()
        service = UserAccountService(
            config_manager=self.config_manager,
            conversation_repository=repository,
        )
        account = service.register_user(
            username=state.username,
            password=state.password,
            email=state.email,
            name=state.display_name or None,
        )
        self.config_manager.set_active_user(account.username)
        self.state.user = dataclasses.replace(state)
        return {
            "username": account.username,
            "email": account.email,
            "display_name": account.name,
        }

    def _get_conversation_repository(self) -> ConversationStoreRepository:
        if self.atlas and getattr(self.atlas, "conversation_repository", None) is not None:
            return self.atlas.conversation_repository

        factory = self.config_manager.get_conversation_store_session_factory()
        if factory is None:
            raise RuntimeError("Conversation store is not configured")

        retention = self.config_manager.get_conversation_retention_policies()
        repository = ConversationStoreRepository(factory, retention=retention)
        repository.create_schema()
        return repository

    # -- optional ----------------------------------------------------------

    def apply_optional_settings(self, state: OptionalState) -> OptionalState:
        self.config_manager.set_tenant_id(state.tenant_id)
        self.config_manager.set_conversation_retention(
            days=state.retention_days,
            history_limit=state.retention_history_limit,
        )
        if state.scheduler_timezone is not None or state.scheduler_queue_size is not None:
            self.config_manager.set_job_scheduling_settings(
                timezone=state.scheduler_timezone if state.scheduler_timezone is not None else ConfigManager.UNSET,
                queue_size=state.scheduler_queue_size if state.scheduler_queue_size is not None else ConfigManager.UNSET,
            )
        self.config_manager.set_http_server_autostart(state.http_auto_start)
        self.state.optional = dataclasses.replace(state)
        return self.state.optional

    # -- summary -----------------------------------------------------------

    def build_summary(self) -> Dict[str, Any]:
        return {
            "database": dataclasses.asdict(self.state.database),
            "job_scheduling": dataclasses.asdict(self.state.job_scheduling),
            "message_bus": dataclasses.asdict(self.state.message_bus),
            "providers": {
                "default_provider": self.state.providers.default_provider,
                "default_model": self.state.providers.default_model,
                "configured_providers": sorted(self.state.providers.api_keys.keys()),
            },
            "speech": dataclasses.asdict(self.state.speech),
            "user": dataclasses.asdict(self.state.user),
            "optional": dataclasses.asdict(self.state.optional),
        }


class SetupWizardWindow(AssistantBase):
    """GTK wizard guiding operators through first-run configuration."""

    def __init__(
        self,
        *,
        application: Gtk.Application,
        atlas: Any | None,
        on_success: Callback,
        on_error: ErrorCallback,
        error: BaseException | None = None,
    ) -> None:
        super().__init__()
        if hasattr(self, "set_title"):
            self.set_title("ATLAS Setup Wizard")
        self.set_application(application)
        self._on_success = on_success
        self._on_error = on_error
        self._error_label = Gtk.Label()
        self._error_label.set_wrap(True)
        self._error_label.set_visible(False)

        self.atlas = atlas
        self.controller = SetupWizardController(atlas=atlas)

        if hasattr(self, "set_default_size"):
            self.set_default_size(720, 520)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        root.set_margin_top(18)
        root.set_margin_bottom(18)
        root.set_margin_start(18)
        root.set_margin_end(18)
        self.set_child(root)

        header = Gtk.Label(label="Complete the guided setup to finish configuring ATLAS.")
        header.set_wrap(True)
        root.append(header)
        root.append(self._error_label)

        self._stack = Stack()
        self._stack.set_vexpand(True)
        root.append(self._stack)

        self._build_pages()

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        root.append(button_box)

        self._back_button = Gtk.Button(label="Back")
        self._back_button.connect("clicked", lambda *_: self._show_previous())
        button_box.append(self._back_button)

        self._next_button = Gtk.Button(label="Next")
        self._next_button.connect("clicked", lambda *_: self._show_next())
        button_box.append(self._next_button)

        self._apply_button = Gtk.Button(label="Apply")
        self._apply_button.connect("clicked", lambda *_: self._apply_changes())
        button_box.append(self._apply_button)

        self._update_navigation()

        if error is not None:
            self.display_error(error)

    # -- UI construction ---------------------------------------------------

    def _build_pages(self) -> None:
        builders: Iterable[tuple[str, Callable[[], Gtk.Widget]]] = [
            ("database", self._build_database_page),
            ("job_scheduling", self._build_job_page),
            ("message_bus", self._build_message_bus_page),
            ("providers", self._build_providers_page),
            ("speech", self._build_speech_page),
            ("user", self._build_user_page),
            ("optional", self._build_optional_page),
            ("summary", self._build_summary_page),
        ]

        self._page_order: list[str] = []
        for page_id, builder in builders:
            widget = builder()
            self._stack.add_titled(widget, page_id, page_id.replace("_", " ").title())
            self._page_order.append(page_id)
        self._stack.set_visible_child_name(self._page_order[0])

    def _make_page(self, title: str, description: str) -> Gtk.Box:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        title_label = Gtk.Label(label=f"<b>{title}</b>")
        if hasattr(title_label, "set_use_markup"):
            title_label.set_use_markup(True)
        if hasattr(title_label, "set_xalign"):
            title_label.set_xalign(0.0)
        desc_label = Gtk.Label(label=description)
        desc_label.set_wrap(True)
        if hasattr(desc_label, "set_xalign"):
            desc_label.set_xalign(0.0)
        box.append(title_label)
        box.append(desc_label)
        return box

    def _build_database_page(self) -> Gtk.Widget:
        page = self._make_page(
            "Conversation database",
            "Configure the PostgreSQL database used to store conversations.",
        )
        return page

    def _build_job_page(self) -> Gtk.Widget:
        page = self._make_page(
            "Job scheduling",
            "Enable durable task scheduling and configure worker defaults.",
        )
        return page

    def _build_message_bus_page(self) -> Gtk.Widget:
        page = self._make_page(
            "Message bus",
            "Select how ATLAS distributes events between components.",
        )
        return page

    def _build_providers_page(self) -> Gtk.Widget:
        page = self._make_page(
            "LLM providers",
            "Provide API credentials and choose the default provider/model.",
        )
        return page

    def _build_speech_page(self) -> Gtk.Widget:
        page = self._make_page(
            "Speech settings",
            "Configure text-to-speech (TTS) and speech-to-text (STT) options.",
        )
        return page

    def _build_user_page(self) -> Gtk.Widget:
        page = self._make_page(
            "Administrator account",
            "Create the initial administrator account for ATLAS.",
        )
        return page

    def _build_optional_page(self) -> Gtk.Widget:
        page = self._make_page(
            "Optional adjustments",
            "Fine-tune tenancy, retention, scheduler, and server defaults.",
        )
        return page

    def _build_summary_page(self) -> Gtk.Widget:
        page = self._make_page(
            "Summary",
            "Review the collected settings before applying them.",
        )
        self._summary_label = Gtk.Label()
        self._summary_label.set_wrap(True)
        self._summary_label.set_xalign(0.0)
        page.append(self._summary_label)
        return page

    # -- navigation --------------------------------------------------------

    def _current_index(self) -> int:
        return self._page_order.index(self._stack.get_visible_child_name())

    def _update_navigation(self) -> None:
        index = self._current_index()
        self._back_button.set_sensitive(index > 0) if hasattr(self._back_button, "set_sensitive") else None
        is_last = index >= len(self._page_order) - 1
        self._next_button.set_sensitive(not is_last) if hasattr(self._next_button, "set_sensitive") else None
        if hasattr(self._next_button, "set_visible"):
            self._next_button.set_visible(not is_last)
        if hasattr(self._apply_button, "set_visible"):
            self._apply_button.set_visible(is_last)
        if is_last:
            summary = self.controller.build_summary()
            self._summary_label.set_text(self._format_summary(summary))

    def _show_next(self) -> None:
        index = self._current_index()
        if index < len(self._page_order) - 1:
            next_name = self._page_order[index + 1]
            self._stack.set_visible_child_name(next_name)
            self._update_navigation()

    def _show_previous(self) -> None:
        index = self._current_index()
        if index > 0:
            prev_name = self._page_order[index - 1]
            self._stack.set_visible_child_name(prev_name)
            self._update_navigation()

    # -- summary helpers ---------------------------------------------------

    def _format_summary(self, summary: Mapping[str, Any]) -> str:
        lines: list[str] = []
        lines.append("Conversation database: %s" % summary["database"].get("dsn", "configured"))
        job = summary["job_scheduling"]
        lines.append(
            "Job scheduling: %s" % ("enabled" if job.get("enabled") else "disabled")
        )
        mb = summary["message_bus"]
        lines.append(f"Message bus backend: {mb.get('backend')}")
        providers = summary["providers"]
        lines.append(
            "Default provider: %s (%s)"
            % (
                providers.get("default_provider") or "unset",
                providers.get("default_model") or "unset",
            )
        )
        speech = summary["speech"]
        lines.append("TTS enabled: %s" % ("yes" if speech.get("tts_enabled") else "no"))
        user = summary["user"]
        lines.append("Admin user: %s" % (user.get("username") or "pending"))
        optional = summary["optional"]
        if optional.get("tenant_id"):
            lines.append(f"Tenant: {optional['tenant_id']}")
        lines.append(
            "HTTP server auto-start: %s" % ("yes" if optional.get("http_auto_start") else "no")
        )
        return "\n".join(lines)

    # -- applying ----------------------------------------------------------

    def _apply_changes(self) -> None:
        try:
            self.controller.apply_database_settings(self.controller.state.database)
            self.controller.apply_job_scheduling(self.controller.state.job_scheduling)
            self.controller.apply_message_bus(self.controller.state.message_bus)
            self.controller.apply_provider_settings(self.controller.state.providers)
            self.controller.apply_speech_settings(self.controller.state.speech)
            if self.controller.state.user.username:
                self.controller.register_user(self.controller.state.user)
            self.controller.apply_optional_settings(self.controller.state.optional)
        except Exception as exc:  # pragma: no cover - surfaced through display_error
            self.display_error(exc)
            self._on_error(exc)
            return

        self.display_error(None)
        self._on_success()

    # -- public API --------------------------------------------------------

    def display_error(self, error: BaseException | None) -> None:
        if error is None:
            self._error_label.set_visible(False)
            self._error_label.set_text("")
            return
        message = str(error) or error.__class__.__name__
        self._error_label.set_text(message)
        self._error_label.set_visible(True)

    # Convenience hooks for tests -----------------------------------------

    def set_database_state(self, **updates: Any) -> None:
        self.controller.state.database = dataclasses.replace(
            self.controller.state.database,
            **updates,
        )

    def set_job_scheduling_state(self, **updates: Any) -> None:
        self.controller.state.job_scheduling = dataclasses.replace(
            self.controller.state.job_scheduling,
            **updates,
        )

    def set_message_bus_state(self, **updates: Any) -> None:
        self.controller.state.message_bus = dataclasses.replace(
            self.controller.state.message_bus,
            **updates,
        )

    def set_provider_state(self, **updates: Any) -> None:
        self.controller.state.providers = dataclasses.replace(
            self.controller.state.providers,
            **updates,
        )

    def set_speech_state(self, **updates: Any) -> None:
        self.controller.state.speech = dataclasses.replace(
            self.controller.state.speech,
            **updates,
        )

    def set_user_state(self, **updates: Any) -> None:
        self.controller.state.user = dataclasses.replace(
            self.controller.state.user,
            **updates,
        )

    def set_optional_state(self, **updates: Any) -> None:
        self.controller.state.optional = dataclasses.replace(
            self.controller.state.optional,
            **updates,
        )


__all__ = ["SetupWizardWindow", "SetupWizardController"]

