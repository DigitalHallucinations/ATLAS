"""Core setup coordination utilities shared by the CLI and GTK front-ends."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from ATLAS.config import ConfigManager, _DEFAULT_CONVERSATION_STORE_DSN
from modules.conversation_store.bootstrap import BootstrapError, bootstrap_conversation_store
from modules.conversation_store.repository import ConversationStoreRepository
from modules.user_accounts.user_account_service import UserAccountService


__all__ = [
    "BootstrapError",
    "ConfigManager",
    "DatabaseState",
    "JobSchedulingState",
    "KvStoreState",
    "MessageBusState",
    "OptionalState",
    "ProviderState",
    "RetryPolicyState",
    "SetupWizardController",
    "SpeechState",
    "UserState",
]


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
class KvStoreState:
    reuse_conversation_store: bool = True
    url: Optional[str] = None


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
    kv_store: KvStoreState = field(default_factory=KvStoreState)
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
    """Coordinate persistence logic for the setup workflow."""

    def __init__(
        self,
        *,
        config_manager: Optional[ConfigManager] = None,
        atlas: Any | None = None,
        bootstrap: Callable[[str], str] = bootstrap_conversation_store,
        request_privileged_password: Callable[[], str | None] | None = None,
    ) -> None:
        self.config_manager = config_manager or ConfigManager()
        self.atlas = atlas
        self._bootstrap = bootstrap
        self._request_privileged_password = request_privileged_password
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

        kv_settings = self.config_manager.get_kv_store_settings()
        postgres_settings = kv_settings.get("adapters", {}).get("postgres", {})
        self.state.kv_store = KvStoreState(
            reuse_conversation_store=bool(postgres_settings.get("reuse_conversation_store", True)),
            url=postgres_settings.get("url"),
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

    def apply_database_settings(
        self,
        state: DatabaseState,
        *,
        privileged_credentials: tuple[str | None, str | None] | None = None,
    ) -> str:
        dsn = _compose_dsn(state)
        kwargs: dict[str, Any] = {}
        if self._request_privileged_password is not None:
            kwargs["request_privileged_password"] = self._request_privileged_password
        if privileged_credentials is not None:
            kwargs["privileged_credentials"] = privileged_credentials
        ensured = self._bootstrap(dsn, **kwargs)
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

    def apply_kv_store_settings(self, state: KvStoreState) -> Mapping[str, Any]:
        url_value = state.url if state.url else ConfigManager.UNSET
        settings = self.config_manager.set_kv_store_settings(
            url=url_value,
            reuse_conversation_store=state.reuse_conversation_store,
        )
        self.state.kv_store = dataclasses.replace(state)
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
