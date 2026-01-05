"""Core setup coordination utilities shared by the CLI and GTK front-ends."""

from __future__ import annotations

import dataclasses
import importlib
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url

from core.config import (
    ConfigManager,
    PerformanceMode,
    StorageArchitecture,
    get_default_conversation_store_backends,
    infer_conversation_store_backend,
)
from modules.conversation_store.bootstrap import BootstrapError, bootstrap_conversation_store
from modules.conversation_store.repository import ConversationStoreRepository
from modules.job_store import ensure_job_schema
from modules.user_accounts.user_account_service import UserAccountService
from modules.orchestration.policy import MessagePolicy

from core.setup.rag_capabilities import RAGCapabilities, RAGCapabilitiesDetector


logger = logging.getLogger(__name__)

__all__ = [
    "BootstrapError",
    "ConfigManager",
    "DatabaseState",
    "AdminProfile",
    "JobSchedulingState",
    "KvStoreState",
    "MessageBusState",
    "VectorStoreState",
    "OptionalState",
    "HardwareProfile",
    "RAGCapabilities",
    "StorageArchitecture",
    "PrivilegedCredentialState",
    "ProviderState",
    "RAGState",
    "RetryPolicyState",
    "SetupTypeState",
    "SetupUserEntry",
    "SetupUsersState",
    "SetupWizardController",
    "SpeechState",
    "UserState",
]

DEFAULT_ENTERPRISE_AUDIT_TEMPLATE = "siem_30d"


@dataclass
class DatabaseState:
    backend: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str = "atlas"
    user: str = "atlas"
    password: str = ""
    dsn: str = ""
    options: str = ""


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
    backend: str = "in_memory"  # NCB always used; this controls bridging mode
    redis_url: Optional[str] = None
    stream_prefix: Optional[str] = None
    initial_offset: str = "$"
    replay_start: str = "$"
    min_idle_ms: int = 60_000
    delete_on_ack: bool = True
    trim_maxlen: Optional[int] = None
    policy_tier: str = "standard"
    policy_retry_attempts: int = 3
    policy_retry_delay: float = 0.1
    policy_dlq_enabled: bool = True
    policy_dlq_template: Optional[str] = "dlq.{topic}"
    policy_retention_seconds: Optional[int] = None
    policy_idempotency_enabled: bool = False
    policy_idempotency_key_field: Optional[str] = None
    policy_idempotency_ttl_seconds: Optional[int] = None
    kafka_enabled: bool = False
    kafka_bootstrap_servers: Optional[str] = None
    kafka_topic_prefix: str = "atlas.bus"
    kafka_client_id: str = "atlas-message-bridge"
    kafka_driver: Optional[str] = None
    kafka_enable_idempotence: bool = True
    kafka_acks: str = "all"
    kafka_max_in_flight: int = 5
    kafka_delivery_timeout: float = 10.0
    kafka_bridge_enabled: bool = False
    kafka_bridge_topics: tuple[str, ...] = ()
    kafka_bridge_batch_size: int = 1
    kafka_bridge_max_attempts: int = 3
    kafka_bridge_backoff_seconds: float = 1.0
    kafka_bridge_dlq_topic: str = "atlas.bridge.dlq"
    # NCB settings (Neural Cognitive Bus)
    ncb_persistence_path: Optional[str] = None
    ncb_enable_prometheus: bool = False
    ncb_prometheus_port: int = 8000


@dataclass
class VectorStoreState:
    adapter: str = "in_memory"


@dataclass
class KvStoreState:
    reuse_conversation_store: bool = True
    url: Optional[str] = None


@dataclass
class ProviderState:
    default_provider: Optional[str] = None
    default_model: Optional[str] = None
    api_keys: Dict[str, str] = field(default_factory=dict)
    settings: Dict[str, Dict[str, str]] = field(default_factory=dict)


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
class PrivilegedCredentialState:
    sudo_username: str = ""
    sudo_password: str = ""


@dataclass
class UserState:
    username: str = ""
    email: str = ""
    password: str = ""
    admin_password: str = ""
    display_name: str = ""
    full_name: str = ""
    domain: str = ""
    date_of_birth: str = ""
    privileged_db_username: str = ""
    privileged_db_password: str = ""
    privileged_credentials: PrivilegedCredentialState = field(default_factory=PrivilegedCredentialState)


@dataclass
class AdminProfile:
    username: str = ""
    email: str = ""
    password: str = ""
    admin_password: str = ""
    display_name: str = ""
    full_name: str = ""
    domain: str = ""
    date_of_birth: str = ""
    sudo_username: str = ""
    sudo_password: str = ""
    privileged_db_username: str = ""
    privileged_db_password: str = ""


@dataclass
class SetupUserEntry:
    username: str
    full_name: str
    password: str
    email: str = ""
    requires_password_reset: bool = True


@dataclass
class SetupUsersState:
    entries: list[SetupUserEntry] = field(default_factory=list)
    initial_admin_username: str = ""


@dataclass
class OptionalState:
    tenant_id: Optional[str] = None
    company_name: Optional[str] = None
    company_domain: Optional[str] = None
    primary_contact: Optional[str] = None
    contact_email: Optional[str] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    phone_number: Optional[str] = None
    retention_days: Optional[int] = None
    retention_history_limit: Optional[int] = None
    scheduler_timezone: Optional[str] = None
    scheduler_queue_size: Optional[int] = None
    http_auto_start: bool = False
    audit_template: Optional[str] = None
    data_region: Optional[str] = None
    residency_requirement: Optional[str] = None


@dataclass
class SetupTypeState:
    mode: str = "custom"
    applied: bool = False
    local_only: bool = False
    developer_mode: bool = False


@dataclass
class HardwareProfile:
    cpu_cores: int = 0
    cpu_score: int = 0
    memory_gb: float = 0.0
    memory_score: int = 0
    disk_free_gb: float = 0.0
    disk_score: int = 0
    gpu_count: int = 0
    gpu_score: int = 0
    network_speed_mbps: float = 0.0
    network_score: int = 0
    total_score: int = 0
    tier: str = "unknown"


@dataclass
class RAGState:
    """RAG configuration state for setup wizard."""
    
    enabled: bool = False
    auto_retrieve: bool = True
    
    # Embedding configuration
    embedding_provider: str = "huggingface"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    
    # Chunking configuration
    chunking_enabled: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Retrieval configuration
    retrieval_enabled: bool = True
    top_k: int = 10
    similarity_threshold: float = 0.7
    max_context_chunks: int = 5
    
    # Reranking configuration
    reranking_enabled: bool = False
    reranker_provider: str = "cross_encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Knowledge store
    knowledge_store_enabled: bool = True
    index_type: str = "hnsw"
    
    # Ingestion
    ingestion_enabled: bool = True
    max_file_size_mb: float = 50.0


@dataclass
class WizardState:
    database: DatabaseState = field(default_factory=DatabaseState)
    job_scheduling: JobSchedulingState = field(default_factory=JobSchedulingState)
    message_bus: MessageBusState = field(default_factory=MessageBusState)
    vector_store: VectorStoreState = field(default_factory=VectorStoreState)
    kv_store: KvStoreState = field(default_factory=KvStoreState)
    storage_architecture: StorageArchitecture = field(
        default_factory=StorageArchitecture
    )
    providers: ProviderState = field(default_factory=ProviderState)
    speech: SpeechState = field(default_factory=SpeechState)
    users: SetupUsersState = field(default_factory=SetupUsersState)
    user: UserState = field(default_factory=UserState)
    optional: OptionalState = field(default_factory=OptionalState)
    setup_type: SetupTypeState = field(default_factory=SetupTypeState)
    hardware_profile: HardwareProfile = field(default_factory=HardwareProfile)
    rag: RAGState = field(default_factory=RAGState)
    rag_capabilities: Any = None  # RAGCapabilities from rag_capabilities module
    setup_recommended_mode: str | None = None


def _parse_default_dsn(dsn: str, *, backend: Optional[str] = None) -> DatabaseState:
    from urllib.parse import urlparse

    parsed = urlparse(dsn)
    scheme = (parsed.scheme or "").lower()
    inferred_backend = backend or infer_conversation_store_backend(scheme) or infer_conversation_store_backend(dsn)
    normalized_backend = (inferred_backend or "postgresql").strip().lower() or "postgresql"

    if normalized_backend == "sqlite":
        database = parsed.path.lstrip("/") or parsed.netloc
        if not database:
            database = str(Path.home() / "atlas.sqlite3")
        else:
            # Ensure the path is absolute
            database = str(Path(database).expanduser().resolve())
        return DatabaseState(
            backend="sqlite",
            host="",
            port=0,
            database=database,
            user="",
            password="",
            dsn=dsn,
            options="",
        )

    if normalized_backend == "mongodb":
        database = parsed.path.lstrip("/").split("?", 1)[0] or "atlas"
        host = parsed.hostname or "localhost"
        port = parsed.port or 27017
        user = parsed.username or ""
        password = parsed.password or ""
        options = parsed.query or ""
        return DatabaseState(
            backend="mongodb",
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            dsn=dsn,
            options=options,
        )

    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    database = parsed.path.lstrip("/") or "atlas"
    user = parsed.username or "atlas"
    password = parsed.password or ""
    return DatabaseState(
        backend="postgresql",
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        dsn=dsn,
        options="",
    )


def _compose_dsn(state: DatabaseState) -> str:
    backend = (state.backend or "postgresql").strip().lower()
    if backend == "sqlite":
        database = (state.database or state.dsn or str(Path.home() / "atlas.sqlite3"))
        if isinstance(database, str):
            candidate = database.strip()
        else:
            candidate = str(database)
        if candidate.startswith("sqlite:"):
            return candidate
        # Ensure the path is absolute to avoid incorrect relative path concatenation
        path = Path(candidate).expanduser().resolve()
        path_text = path.as_posix()
        return f"sqlite:///{path_text}"

    if backend == "mongodb":
        if state.dsn and state.dsn.strip().lower().startswith("mongodb"):
            return state.dsn.strip()

        from urllib.parse import quote_plus

        host = (state.host or "localhost").strip() or "localhost"
        port = int(state.port or 27017)
        database = (state.database or "atlas").strip() or "atlas"
        username = (state.user or "").strip()
        password = state.password or ""
        auth = ""
        if username:
            encoded_user = quote_plus(username)
            encoded_pass = quote_plus(password) if password else ""
            auth = encoded_user
            if encoded_pass:
                auth = f"{auth}:{encoded_pass}"
            auth = f"{auth}@"
        port_fragment = f":{port}" if port else ""
        options = (state.options or "").strip()
        if options and not options.startswith("?"):
            options = f"?{options}"
        return f"mongodb://{auth}{host}{port_fragment}/{database}{options}"

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
        config_manager_factory: Callable[[], ConfigManager] | None = None,
        atlas: Any | None = None,
        bootstrap: Callable[[str], str] = bootstrap_conversation_store,
        request_privileged_password: Callable[[], str | None] | None = None,
    ) -> None:
        if config_manager_factory is None:
            if config_manager is not None:
                config_manager_factory = config_manager.__class__  # type: ignore[assignment]
            else:
                config_manager_factory = ConfigManager
        self._config_manager_factory = config_manager_factory

        self.config_manager = config_manager or self._config_manager_factory()
        self.atlas = atlas
        self._bootstrap = bootstrap
        self._privileged_password_requester = self._wrap_privileged_password_requester(
            request_privileged_password
        )
        self.state = WizardState()
        self._staged_admin_profile: AdminProfile | None = None
        self._staged_privileged_credentials: tuple[str | None, str | None] | None = None
        self._load_defaults()

    # -- state loaders -----------------------------------------------------

    def export_config(self, path: str | Path) -> str:
        """Persist the active YAML configuration to ``path`` and refresh state."""

        destination = Path(path).expanduser()
        logger.info("Exporting setup configuration to %s", destination)
        self.config_manager._write_yaml_config()
        exported = self.config_manager.export_yaml_config(destination)
        self._refresh_from_disk()
        return exported

    def import_config(self, path: str | Path) -> str:
        """Load configuration from ``path`` into the managed config and refresh state."""

        source = Path(path).expanduser()
        logger.info("Importing setup configuration from %s", source)
        self.config_manager.import_yaml_config(source)
        self._refresh_from_disk()
        return str(source)

    def _refresh_from_disk(self) -> None:
        """Reload the configuration manager and wizard state from disk."""

        logger.info("Refreshing setup state from disk")
        try:
            self.config_manager = self._config_manager_factory()
        except Exception:  # pragma: no cover - fall back to default manager
            self.config_manager = ConfigManager()
        self.state = WizardState()
        self._staged_admin_profile = None
        self._staged_privileged_credentials = None
        self._load_defaults()

    def _resolve_queue_defaults(self, *, backend: str | None) -> tuple[int, int]:
        """Return scheduler sizing defaults tuned for the selected backend."""

        normalized = (backend or "in_memory").strip().lower() or "in_memory"
        if normalized == "redis":
            return 8, 500
        return 4, 100

    def _load_defaults(self) -> None:
        logger.info("Loading setup defaults from configuration manager")
        conversation_config = self.config_manager.get_conversation_database_config()
        database_url = conversation_config.get("url") if isinstance(conversation_config, Mapping) else None
        backend_value = conversation_config.get("backend") if isinstance(conversation_config, Mapping) else None
        persisted_backend = self.config_manager.get_conversation_backend()
        if backend_value is None and persisted_backend is not None:
            backend_value = persisted_backend
        if isinstance(database_url, str) and database_url.strip():
            self.state.database = _parse_default_dsn(database_url, backend=backend_value)
        else:
            options = get_default_conversation_store_backends()
            fallback_backend = (backend_value or options[0].name)
            option_map = {option.name: option for option in options}
            selected = option_map.get(fallback_backend, options[0])
            self.state.database = _parse_default_dsn(selected.dsn, backend=selected.name)
        logger.info(
            "Loaded default conversation backend '%s' with database '%s'",
            self.state.database.backend,
            self.state.database.database,
        )

        vector_settings = self.config_manager.get_vector_store_settings()
        adapter = vector_settings.get("default_adapter") if isinstance(vector_settings, Mapping) else None
        if isinstance(adapter, str) and adapter.strip():
            self.state.vector_store = VectorStoreState(adapter=adapter.strip().lower())
        else:
            self.state.vector_store = VectorStoreState()

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
        policy_block = messaging.get("policy") or {}
        default_policy = policy_block.get("default") or {}
        policy_defaults = MessagePolicy()
        kafka_block = messaging.get("kafka") or {}
        kafka_bridge = kafka_block.get("bridge") or {}
        bridge_topics = kafka_bridge.get("topics") or []
        if isinstance(bridge_topics, (list, tuple)):
            topics_tuple = tuple(str(topic).strip() for topic in bridge_topics if str(topic).strip())
        else:
            topics_tuple = ()
        self.state.message_bus = MessageBusState(
            backend=backend,
            redis_url=messaging.get("redis_url"),
            stream_prefix=messaging.get("stream_prefix"),
            initial_offset=messaging.get("initial_offset") or "$",
            replay_start=messaging.get("replay_start") or messaging.get("initial_offset") or "$",
            min_idle_ms=int(messaging.get("min_idle_ms", 60_000) or 60_000),
            delete_on_ack=bool(messaging.get("delete_on_ack", True)),
            trim_maxlen=messaging.get("trim_maxlen"),
            policy_tier=str(default_policy.get("tier") or policy_defaults.tier or "standard"),
            policy_retry_attempts=int(
                default_policy.get("retry_attempts", policy_defaults.retry_attempts) or policy_defaults.retry_attempts
            ),
            policy_retry_delay=float(
                default_policy.get("retry_delay", policy_defaults.retry_delay) or policy_defaults.retry_delay
            ),
            policy_dlq_enabled=bool(default_policy.get("dlq_topic_template") or policy_defaults.dlq_topic_template),
            policy_dlq_template=default_policy.get("dlq_topic_template") or policy_defaults.dlq_topic_template,
            policy_retention_seconds=default_policy.get("retention_seconds") or policy_defaults.retention_seconds,
            policy_idempotency_enabled=bool(
                default_policy.get("idempotency_key_field") or default_policy.get("idempotency_ttl_seconds")
            ),
            policy_idempotency_key_field=default_policy.get("idempotency_key_field") or policy_defaults.idempotency_key_field,
            policy_idempotency_ttl_seconds=default_policy.get("idempotency_ttl_seconds")
            or policy_defaults.idempotency_ttl_seconds,
            kafka_enabled=bool(kafka_block.get("enabled")),
            kafka_bootstrap_servers=kafka_block.get("bootstrap_servers"),
            kafka_topic_prefix=kafka_block.get("topic_prefix") or "atlas.bus",
            kafka_client_id=kafka_block.get("client_id") or "atlas-message-bridge",
            kafka_driver=(kafka_block.get("driver") or kafka_block.get("preferred_driver")),
            kafka_enable_idempotence=bool(kafka_block.get("enable_idempotence", True)),
            kafka_acks=kafka_block.get("acks") or "all",
            kafka_max_in_flight=int(kafka_block.get("max_in_flight", 5) or 5),
            kafka_delivery_timeout=float(kafka_block.get("delivery_timeout", 10.0) or 10.0),
            kafka_bridge_enabled=bool(kafka_bridge.get("enabled")),
            kafka_bridge_topics=topics_tuple,
            kafka_bridge_batch_size=int(kafka_bridge.get("batch_size", 1) or 1),
            kafka_bridge_max_attempts=int(kafka_bridge.get("max_attempts", 3) or 3),
            kafka_bridge_backoff_seconds=float(kafka_bridge.get("backoff_seconds", 1.0) or 1.0),
            kafka_bridge_dlq_topic=kafka_bridge.get("dlq_topic") or "atlas.bridge.dlq",
        )

        kv_settings = self.config_manager.get_kv_store_settings()
        postgres_settings = kv_settings.get("adapters", {}).get("postgres", {})
        self.state.kv_store = KvStoreState(
            reuse_conversation_store=bool(postgres_settings.get("reuse_conversation_store", True)),
            url=postgres_settings.get("url"),
        )

        storage_architecture = self.config_manager.get_storage_architecture()
        self.state.storage_architecture = dataclasses.replace(storage_architecture)

        provider_keys = self.config_manager._get_provider_env_keys()
        api_key_state: Dict[str, str] = {}
        for provider, env_key in provider_keys.items():
            value = self.config_manager.get_config(env_key)
            if isinstance(value, str) and value.strip():
                api_key_state[provider] = value.strip()
        provider_settings_state: Dict[str, Dict[str, str]] = {}
        saved_provider_settings = self.config_manager.get_config("PROVIDER_SETTINGS")
        if isinstance(saved_provider_settings, Mapping):
            for provider, settings in saved_provider_settings.items():
                if not isinstance(settings, Mapping):
                    continue
                provider_settings_state[provider] = {
                    key: str(value)
                    for key, value in settings.items()
                    if isinstance(key, str) and isinstance(value, str)
                }
        if (openai_base := self.config_manager.get_config("OPENAI_BASE_URL")):
            provider_settings_state.setdefault("OpenAI", {})["base_url"] = str(openai_base)
        if (openai_org := self.config_manager.get_config("OPENAI_ORGANIZATION")):
            provider_settings_state.setdefault("OpenAI", {})["organization"] = str(openai_org)
        if (mistral_base := self.config_manager.get_config("MISTRAL_BASE_URL")):
            provider_settings_state.setdefault("Mistral", {})["base_url"] = str(mistral_base)
        self.state.providers = ProviderState(
            default_provider=self.config_manager.get_default_provider(),
            default_model=self.config_manager.get_default_model(),
            api_keys=api_key_state,
            settings=provider_settings_state,
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
        residency = self.config_manager.get_data_residency_settings()
        company_identity = self.config_manager.get_company_identity()
        self.state.optional = OptionalState(
            tenant_id=str(tenant_value).strip() if tenant_value else None,
            company_name=company_identity.get("name"),
            company_domain=company_identity.get("domain"),
            primary_contact=company_identity.get("primary_contact"),
            contact_email=company_identity.get("contact_email"),
            address_line1=company_identity.get("address_line1"),
            address_line2=company_identity.get("address_line2"),
            city=company_identity.get("city"),
            state=company_identity.get("state"),
            postal_code=company_identity.get("postal_code"),
            country=company_identity.get("country"),
            phone_number=company_identity.get("phone_number"),
            retention_days=retention.get("days") or retention.get("max_days"),
            retention_history_limit=retention.get("history_message_limit"),
            scheduler_timezone=self.state.job_scheduling.timezone,
            scheduler_queue_size=self.state.job_scheduling.queue_size,
            http_auto_start=bool(http_block.get("auto_start")) if isinstance(http_block, Mapping) else False,
            audit_template=self.config_manager.get_audit_template(),
            data_region=residency.get("region"),
            residency_requirement=residency.get("residency_requirement"),
        )

    # -- preflight ----------------------------------------------------------

    def _score_metric(self, value: float, thresholds: list[tuple[float, int]]) -> int:
        for boundary, score in thresholds:
            if value >= boundary:
                return score
        return 1 if value > 0 else 0

    def _determine_tier(self, total_score: int) -> str:
        if total_score >= 18:
            return "accelerated"
        if total_score >= 12:
            return "balanced"
        if total_score > 0:
            return "baseline"
        return "unknown"

    def _recommend_mode_for_tier(self, tier: str) -> str:
        normalized = (tier or "").strip().lower()
        if normalized == "accelerated":
            return "performance"
        if normalized == "balanced":
            return "balanced"
        return "eco"

    def _get_psutil(self):
        spec = importlib.util.find_spec("psutil")
        if spec is None:
            return None
        return importlib.import_module("psutil")

    def _detect_gpu(self) -> tuple[int, list[str]]:
        spec = importlib.util.find_spec("torch")
        if spec is None:
            return 0, []
        torch = importlib.import_module("torch")
        try:
            count = int(torch.cuda.device_count())
        except Exception:
            return 0, []
        names: list[str] = []
        for index in range(count):
            try:
                names.append(str(torch.cuda.get_device_name(index)))
            except Exception:
                names.append(f"GPU {index}")
        return count, names

    def _detect_network_speed(self, psutil_module: Any | None = None) -> float:
        if psutil_module is None:
            psutil_module = self._get_psutil()
        if psutil_module is None:
            return 0.0
        try:
            stats = psutil_module.net_if_stats()
        except Exception:
            return 0.0
        max_speed = 0.0
        for entry in stats.values():
            try:
                speed = float(getattr(entry, "speed", 0.0) or 0.0)
            except Exception:
                speed = 0.0
            if speed > 0:
                max_speed = max(max_speed, speed)
        return max_speed

    def _collect_preflight_metrics(self) -> Dict[str, float]:
        psutil_module = self._get_psutil()
        cpu_cores = (
            psutil_module.cpu_count(logical=False) or psutil_module.cpu_count(logical=True)
            if psutil_module is not None
            else 0
        )
        memory_total = 0.0
        if psutil_module is not None:
            memory = psutil_module.virtual_memory()
            memory_total = float(getattr(memory, "total", 0.0) or 0.0)
        home = Path.home()
        try:
            disk = shutil.disk_usage(home)
        except Exception:
            disk = shutil.disk_usage("/")
        disk_free = float(getattr(disk, "free", 0.0) or 0.0)
        gpu_count, _ = self._detect_gpu()
        network_speed = self._detect_network_speed(psutil_module)
        return {
            "cpu_cores": float(cpu_cores or 0),
            "memory_total": memory_total,
            "disk_free": disk_free,
            "gpu_count": float(gpu_count or 0),
            "network_speed": float(network_speed or 0.0),
        }

    def run_preflight(self) -> HardwareProfile:
        """Score local hardware and recommend a performance mode."""

        metrics = self._collect_preflight_metrics()

        memory_gb = metrics["memory_total"] / (1024**3) if metrics["memory_total"] else 0.0
        disk_free_gb = metrics["disk_free"] / (1024**3) if metrics["disk_free"] else 0.0

        cpu_score = self._score_metric(
            metrics["cpu_cores"],
            [(16, 5), (12, 4), (8, 3), (4, 2)],
        )
        memory_score = self._score_metric(
            memory_gb,
            [(64, 5), (32, 4), (16, 3), (8, 2)],
        )
        disk_score = self._score_metric(
            disk_free_gb,
            [(200, 5), (100, 4), (50, 3), (20, 2)],
        )
        gpu_score = self._score_metric(metrics["gpu_count"], [(2, 5), (1, 4)])
        network_score = self._score_metric(
            metrics["network_speed"],
            [(1000, 5), (500, 4), (200, 3), (100, 2)],
        )

        total_score = cpu_score + memory_score + disk_score + gpu_score + network_score
        tier = self._determine_tier(total_score)
        recommended_mode = self._recommend_mode_for_tier(tier)

        profile = HardwareProfile(
            cpu_cores=int(metrics["cpu_cores"]),
            cpu_score=cpu_score,
            memory_gb=round(memory_gb, 1),
            memory_score=memory_score,
            disk_free_gb=round(disk_free_gb, 1),
            disk_score=disk_score,
            gpu_count=int(metrics["gpu_count"]),
            gpu_score=gpu_score,
            network_speed_mbps=round(metrics["network_speed"], 1),
            network_score=network_score,
            total_score=total_score,
            tier=tier,
        )

        logger.info(
            "Preflight complete: tier=%s, recommended_mode=%s, cpu=%s, memory_gb=%.1f, disk_gb=%.1f, gpu_count=%s, network=%.1fMbps",
            tier,
            recommended_mode,
            profile.cpu_cores,
            profile.memory_gb,
            profile.disk_free_gb,
            profile.gpu_count,
            profile.network_speed_mbps,
        )

        self.state.hardware_profile = profile
        self.state.setup_recommended_mode = recommended_mode
        return profile

    def detect_rag_capabilities(
        self,
        *,
        database_dsn: Optional[str] = None,
    ) -> RAGCapabilities:
        """Detect RAG capabilities based on available hardware and database.
        
        Args:
            database_dsn: Optional database connection string for pgvector detection.
                         If not provided, uses the database state's dsn if available.
        
        Returns:
            RAGCapabilities object with detection results and recommendations.
        """
        dsn = database_dsn
        if dsn is None and hasattr(self.state, "database") and self.state.database:
            dsn = getattr(self.state.database, "dsn", None)
        
        detector = RAGCapabilitiesDetector(database_dsn=dsn)
        capabilities = detector.detect()
        
        # Store in state for use by wizard pages
        self.state.rag_capabilities = capabilities
        
        # Pre-fill RAG defaults based on recommendations
        if capabilities.recommended_model:
            rec = capabilities.recommended_model
            self.state.rag.embedding_provider = rec.provider
            self.state.rag.embedding_model = rec.model_id
            self.state.rag.embedding_dimensions = rec.dimensions
        
        if capabilities.pgvector:
            self.state.rag.knowledge_store_enabled = capabilities.pgvector.available
        
        logger.info(
            "RAG capabilities detected: tier=%s, provider=%s, model=%s, pgvector=%s",
            capabilities.embedding_tier.value if capabilities.embedding_tier else "unknown",
            self.state.rag.embedding_provider,
            self.state.rag.embedding_model,
            capabilities.pgvector.available if capabilities.pgvector else False,
        )
        
        return capabilities

    # -- presets ------------------------------------------------------------

    def apply_setup_type(
        self,
        mode: str,
        *,
        local_only: Optional[bool] = None,
        developer_mode: Optional[bool] = None,
    ) -> SetupTypeState:
        normalized = (mode or "").strip().lower()
        valid_modes = {"student", "personal", "enthusiast", "enterprise", "regulatory"}
        if normalized not in valid_modes:
            fallback_mode = normalized or "custom"
            logger.info("Unknown setup mode '%s'; falling back to custom configuration", mode)
            setup_state = SetupTypeState(mode=fallback_mode, applied=False, local_only=False, developer_mode=False)
            self.state.setup_type = setup_state
            return setup_state

        current = self.state.setup_type
        if current.mode == normalized and current.applied:
            return current

        local_flag = current.local_only if local_only is None else bool(local_only)

        logger.info("Applying setup profile '%s'", normalized)
        if normalized == "personal":
            self.state.message_bus = dataclasses.replace(
                self.state.message_bus,
                backend="in_memory",
                redis_url=None,
                stream_prefix=None,
                initial_offset="$",
            )
            self.state.job_scheduling = dataclasses.replace(
                self.state.job_scheduling,
                enabled=False,
                job_store_url=None,
                max_workers=None,
                retry_policy=dataclasses.replace(self.state.job_scheduling.retry_policy),
                timezone=None,
                queue_size=None,
            )
            self.state.kv_store = dataclasses.replace(
                self.state.kv_store,
                reuse_conversation_store=True,
                url=None,
            )
            self.state.optional = dataclasses.replace(
                self.state.optional,
                retention_days=None,
                retention_history_limit=None,
                http_auto_start=True,
                audit_template=None,
            )

            if local_flag:
                default_sqlite_path = str(Path.home() / "atlas.sqlite3")
                sqlite_dsn = f"sqlite:///{default_sqlite_path}"
                self.state.database = DatabaseState(
                    backend="sqlite",
                    host="",
                    port=0,
                    database=default_sqlite_path,
                    user="",
                    password="",
                    dsn=sqlite_dsn,
                    options="",
                )
                self.state.vector_store = dataclasses.replace(
                    self.state.vector_store,
                    adapter="in_memory",
                )
            profile = self._load_setup_profile("personal")
            self._apply_profile_overrides(profile)

        elif normalized == "student":
            # Free tier for learners with simple defaults and usage limits
            self.state.message_bus = dataclasses.replace(
                self.state.message_bus,
                backend="in_memory",
                redis_url=None,
                stream_prefix=None,
                initial_offset="$",
            )
            self.state.job_scheduling = dataclasses.replace(
                self.state.job_scheduling,
                enabled=False,
                job_store_url=None,
                max_workers=None,
                retry_policy=dataclasses.replace(self.state.job_scheduling.retry_policy),
                timezone=None,
                queue_size=None,
            )
            self.state.kv_store = dataclasses.replace(
                self.state.kv_store,
                reuse_conversation_store=True,
                url=None,
            )
            self.state.optional = dataclasses.replace(
                self.state.optional,
                retention_days=7,
                retention_history_limit=100,
                http_auto_start=True,
                audit_template=None,
            )

            if local_flag:
                default_sqlite_path = str(Path.home() / "atlas.sqlite3")
                sqlite_dsn = f"sqlite:///{default_sqlite_path}"
                self.state.database = DatabaseState(
                    backend="sqlite",
                    host="",
                    port=0,
                    database=default_sqlite_path,
                    user="",
                    password="",
                    dsn=sqlite_dsn,
                    options="",
                )
                self.state.vector_store = dataclasses.replace(
                    self.state.vector_store,
                    adapter="in_memory",
                )
            profile = self._load_setup_profile("student")
            self._apply_profile_overrides(profile)

        elif normalized == "enthusiast":
            # Power user tier with all features unlocked
            redis_url = self.state.message_bus.redis_url or "redis://localhost:6379/0"
            stream_prefix = self.state.message_bus.stream_prefix or "atlas-power"
            self.state.message_bus = dataclasses.replace(
                self.state.message_bus,
                backend="redis",
                redis_url=redis_url,
                stream_prefix=stream_prefix,
                initial_offset=self.state.message_bus.initial_offset or "$",
            )
            default_workers, default_queue_size = self._resolve_queue_defaults(
                backend=self.state.message_bus.backend
            )
            job_store_url = self.state.job_scheduling.job_store_url or (
                "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_jobs"
            )
            self.state.database = dataclasses.replace(
                self.state.database,
                backend="postgresql",
                host="localhost",
                port=5432,
                database="atlas",
                user="atlas",
                password=self.state.database.password or "",
                dsn="",
                options=self.state.database.options,
            )
            self.state.job_scheduling = dataclasses.replace(
                self.state.job_scheduling,
                enabled=True,
                job_store_url=job_store_url,
                max_workers=self.state.job_scheduling.max_workers or default_workers,
                retry_policy=dataclasses.replace(self.state.job_scheduling.retry_policy),
                timezone=self.state.job_scheduling.timezone or "UTC",
                queue_size=self.state.job_scheduling.queue_size or default_queue_size,
            )
            self.state.kv_store = dataclasses.replace(
                self.state.kv_store,
                reuse_conversation_store=False,
                url=self.state.kv_store.url or "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_cache",
            )
            self.state.optional = dataclasses.replace(
                self.state.optional,
                retention_days=90,
                retention_history_limit=1000,
                http_auto_start=True,
                audit_template="detailed_90d",
            )
            architecture = StorageArchitecture(
                performance_mode=PerformanceMode.PERFORMANCE,
                conversation_backend="postgresql",
                kv_reuse_conversation_store=False,
                vector_store_adapter="in_memory",
            )
            self.state.vector_store = dataclasses.replace(
                self.state.vector_store,
                adapter=architecture.vector_store_adapter,
            )
            self.apply_storage_architecture(architecture)
            profile = self._load_setup_profile("enthusiast")
            self._apply_profile_overrides(profile)

        elif normalized == "enterprise":
            redis_url = self.state.message_bus.redis_url or "redis://localhost:6379/0"
            stream_prefix = self.state.message_bus.stream_prefix or "atlas"
            self.state.message_bus = dataclasses.replace(
                self.state.message_bus,
                backend="redis",
                redis_url=redis_url,
                stream_prefix=stream_prefix,
                initial_offset=self.state.message_bus.initial_offset or "$",
            )
            default_workers, default_queue_size = self._resolve_queue_defaults(
                backend=self.state.message_bus.backend
            )
            job_store_url = self.state.job_scheduling.job_store_url or (
                "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_jobs"
            )
            self.state.job_scheduling = dataclasses.replace(
                self.state.job_scheduling,
                enabled=True,
                job_store_url=job_store_url,
                max_workers=self.state.job_scheduling.max_workers
                or default_workers,
                retry_policy=dataclasses.replace(self.state.job_scheduling.retry_policy),
                timezone=self.state.job_scheduling.timezone or "UTC",
                queue_size=self.state.job_scheduling.queue_size or default_queue_size,
            )
            self.state.kv_store = dataclasses.replace(
                self.state.kv_store,
                reuse_conversation_store=False,
                url=self.state.kv_store.url or "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_cache",
            )
            self.state.optional = dataclasses.replace(
                self.state.optional,
                retention_days=30,
                retention_history_limit=500,
                http_auto_start=False,
                audit_template=(
                    self.state.optional.audit_template
                    or DEFAULT_ENTERPRISE_AUDIT_TEMPLATE
                ),
            )

            default_workers, default_queue_size = self._resolve_queue_defaults(
                backend=self.state.message_bus.backend
            )
            job_workers = self.state.job_scheduling.max_workers or default_workers
            queue_size = self.state.job_scheduling.queue_size or default_queue_size

            self.state.job_scheduling = dataclasses.replace(
                self.state.job_scheduling,
                max_workers=job_workers,
                queue_size=queue_size,
            )

            profile = self._load_setup_profile("enterprise")
            self._apply_profile_overrides(profile)

        else:  # regulatory preset
            redis_url = self.state.message_bus.redis_url or "redis://localhost:6379/0"
            stream_prefix = self.state.message_bus.stream_prefix or "atlas"
            self.state.message_bus = dataclasses.replace(
                self.state.message_bus,
                backend="redis",
                redis_url=redis_url,
                stream_prefix=stream_prefix,
                initial_offset=self.state.message_bus.initial_offset or "$",
            )
            default_workers, default_queue_size = self._resolve_queue_defaults(
                backend=self.state.message_bus.backend
            )
            job_store_url = self.state.job_scheduling.job_store_url or (
                "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_jobs"
            )
            self.state.job_scheduling = dataclasses.replace(
                self.state.job_scheduling,
                enabled=True,
                job_store_url=job_store_url,
                max_workers=self.state.job_scheduling.max_workers
                or default_workers,
                retry_policy=dataclasses.replace(self.state.job_scheduling.retry_policy),
                timezone=self.state.job_scheduling.timezone or "UTC",
                queue_size=self.state.job_scheduling.queue_size or default_queue_size,
            )
            self.state.kv_store = dataclasses.replace(
                self.state.kv_store,
                reuse_conversation_store=False,
                url=self.state.kv_store.url
                or "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_cache",
            )
            self.state.optional = dataclasses.replace(
                self.state.optional,
                retention_days=90,
                retention_history_limit=1000,
                http_auto_start=False,
                audit_template=(
                    self.state.optional.audit_template
                    or DEFAULT_ENTERPRISE_AUDIT_TEMPLATE
                ),
                residency_requirement=self.state.optional.residency_requirement or "regional",
            )

            default_workers, default_queue_size = self._resolve_queue_defaults(
                backend=self.state.message_bus.backend
            )
            job_workers = self.state.job_scheduling.max_workers or default_workers
            queue_size = self.state.job_scheduling.queue_size or default_queue_size

            self.state.job_scheduling = dataclasses.replace(
                self.state.job_scheduling,
                max_workers=job_workers,
                queue_size=queue_size,
            )

            profile = self._load_setup_profile("regulatory")
            self._apply_profile_overrides(profile)

        dev_flag = current.developer_mode if developer_mode is None else bool(developer_mode)
        setup_state = SetupTypeState(mode=normalized, applied=True, developer_mode=dev_flag)
        if normalized in {"personal", "student"}:
            setup_state = dataclasses.replace(setup_state, local_only=local_flag)
        elif local_only is not None:
            setup_state = dataclasses.replace(setup_state, local_only=False)
        self.state.setup_type = setup_state

        # Apply developer mode overlay if enabled
        if dev_flag:
            self._apply_developer_mode_overlay()

        return setup_state

    def _apply_developer_mode_overlay(self) -> None:
        """Apply developer-friendly defaults as an overlay on the current configuration.

        Developer mode enables local Redis streams and PostgreSQL job scheduling
        to mirror production behaviours while iterating. This can be toggled on
        any base tier (Student, Personal, Enthusiast, Enterprise, Regulatory).
        """
        logger.info("Applying developer mode overlay")

        # Enable Redis message bus for production-like messaging
        redis_url = self.state.message_bus.redis_url or "redis://localhost:6379/0"
        stream_prefix = self.state.message_bus.stream_prefix or "atlas-dev"
        self.state.message_bus = dataclasses.replace(
            self.state.message_bus,
            backend="redis",
            redis_url=redis_url,
            stream_prefix=stream_prefix,
            initial_offset=self.state.message_bus.initial_offset or "$",
        )

        # Enable job scheduling with local PostgreSQL
        default_workers, default_queue_size = self._resolve_queue_defaults(
            backend=self.state.message_bus.backend
        )
        job_store_url = self.state.job_scheduling.job_store_url or (
            "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_jobs"
        )
        self.state.job_scheduling = dataclasses.replace(
            self.state.job_scheduling,
            enabled=True,
            job_store_url=job_store_url,
            max_workers=self.state.job_scheduling.max_workers or default_workers,
            retry_policy=dataclasses.replace(self.state.job_scheduling.retry_policy),
            timezone=self.state.job_scheduling.timezone or "UTC",
            queue_size=self.state.job_scheduling.queue_size or default_queue_size,
        )

        # If database is SQLite, upgrade to PostgreSQL for developer mode
        if (self.state.database.backend or "").lower() == "sqlite":
            self.state.database = dataclasses.replace(
                self.state.database,
                backend="postgresql",
                host="localhost",
                port=5432,
                database="atlas",
                user="atlas",
                password=self.state.database.password or "",
                dsn="",
                options=self.state.database.options,
            )

    def set_developer_mode(self, enabled: bool) -> SetupTypeState:
        """Toggle developer mode on the current setup type."""
        current = self.state.setup_type
        if current.developer_mode == enabled:
            return current

        setup_state = dataclasses.replace(current, developer_mode=enabled)
        self.state.setup_type = setup_state

        if enabled:
            self._apply_developer_mode_overlay()

        return setup_state

    def _apply_profile_overrides(self, profile: Mapping[str, Any]) -> None:
        retention = profile.get("retention") if isinstance(profile, Mapping) else None
        auditing = profile.get("auditing") if isinstance(profile, Mapping) else None
        personas = profile.get("personas") if isinstance(profile, Mapping) else None
        providers = profile.get("providers") if isinstance(profile, Mapping) else None

        optional_state = self.state.optional
        if isinstance(retention, Mapping):
            optional_state = dataclasses.replace(
                optional_state,
                retention_days=self._coalesce(
                    retention.get("days"), retention.get("max_days"), optional_state.retention_days
                ),
                retention_history_limit=self._coalesce(
                    retention.get("history_limit"),
                    retention.get("history_message_limit"),
                    optional_state.retention_history_limit,
                ),
            )

        if isinstance(auditing, Mapping):
            optional_state = dataclasses.replace(
                optional_state,
                audit_template=self._coalesce(
                    auditing.get("template"), optional_state.audit_template
                ),
                data_region=self._coalesce(
                    auditing.get("data_region"), optional_state.data_region
                ),
                residency_requirement=self._coalesce(
                    auditing.get("residency_requirement"), optional_state.residency_requirement
                ),
            )

        if isinstance(personas, Mapping):
            tenant = personas.get("tenant") or personas.get("tenant_id")
            if tenant:
                optional_state = dataclasses.replace(optional_state, tenant_id=str(tenant))

        self.state.optional = optional_state

        provider_state = self.state.providers
        if isinstance(providers, Mapping):
            provider_state = dataclasses.replace(
                provider_state,
                default_provider=self._coalesce(
                    providers.get("default_provider"), provider_state.default_provider
                ),
                default_model=self._coalesce(
                    providers.get("default_model"), provider_state.default_model
                ),
            )
        self.state.providers = provider_state

    @staticmethod
    def _coalesce(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    def _load_setup_profile(self, mode: str) -> Mapping[str, Any]:
        safe_mode = (mode or "").strip().lower() or "personal"
        profile_dir = Path(__file__).resolve().parent.parent / "config" / "setup_presets"
        profile_path = profile_dir / f"{safe_mode}.yaml"
        logger.info("Loading setup preset '%s' from %s", safe_mode, profile_path)
        if not profile_path.exists():
            logger.info("Preset '%s' not found; skipping overrides", safe_mode)
            return {}
        try:
            with profile_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
                if isinstance(loaded, Mapping):
                    return loaded
        except OSError:
            logger.exception("Failed to load preset '%s' from %s", safe_mode, profile_path)
            return {}
        return {}

    def _wrap_privileged_password_requester(
        self, callback: Callable[[], str | None] | None
    ) -> Callable[[], str | None] | None:
        if callback is None:
            return None

        def _request_password() -> str | None:
            staged_password = (self.state.user.privileged_credentials.sudo_password or "").strip()
            if staged_password:
                return staged_password
            password = callback()
            if isinstance(password, str) and password.strip():
                credentials = dataclasses.replace(
                    self.state.user.privileged_credentials,
                    sudo_password=password,
                )
                self.state.user = dataclasses.replace(
                    self.state.user,
                    privileged_credentials=credentials,
                )
                return password
            return password

        return _request_password

    # -- database -----------------------------------------------------------

    def apply_database_settings(
        self,
        state: DatabaseState,
        *,
        privileged_credentials: tuple[str | None, str | None] | None = None,
    ) -> str:
        logger.info("Applying database settings for backend '%s'", state.backend)
        dsn = _compose_dsn(state)
        kwargs: dict[str, Any] = {}
        if self._privileged_password_requester is not None:
            kwargs["request_privileged_password"] = self._privileged_password_requester
        if privileged_credentials is not None:
            self.set_privileged_credentials(privileged_credentials)
        staged_privileged = self.get_privileged_credentials()
        if staged_privileged is not None:
            kwargs["privileged_credentials"] = staged_privileged
        try:
            ensured = self._bootstrap(dsn, **kwargs)
        except BootstrapError:
            logger.exception("Conversation store bootstrap failed for DSN %s", dsn)
            raise
        except Exception:
            logger.exception("Unexpected error bootstrapping conversation store for DSN %s", dsn)
            raise
        backend = (state.backend or self.state.database.backend or "postgresql")
        self.config_manager._persist_conversation_database_url(ensured, backend=backend)
        self.config_manager._write_yaml_config()
        self.state.database = dataclasses.replace(state, dsn=ensured)
        architecture = dataclasses.replace(
            self.state.storage_architecture,
            conversation_backend=(backend or "postgresql").strip().lower(),
        )
        self.apply_storage_architecture(architecture)
        logger.info("Persisted conversation store configuration for backend '%s'", backend)
        return ensured

    def apply_vector_store_settings(self, state: VectorStoreState) -> Dict[str, Any]:
        logger.info("Applying vector store settings for adapter '%s'", state.adapter)
        settings = self.config_manager.set_vector_store_settings(
            default_adapter=state.adapter,
        )
        self.state.vector_store = dataclasses.replace(state)
        architecture = dataclasses.replace(
            self.state.storage_architecture,
            vector_store_adapter=state.adapter,
        )
        self.apply_storage_architecture(architecture)
        return settings

    def apply_storage_architecture(
        self, architecture: StorageArchitecture | Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Persist storage architecture selections."""

        if isinstance(architecture, StorageArchitecture):
            normalized = architecture
        else:
            normalized = StorageArchitecture.from_mapping(architecture)

        settings = self.config_manager.set_storage_architecture(normalized)
        self.state.storage_architecture = dataclasses.replace(normalized)
        return settings

    # -- job scheduling ----------------------------------------------------

    def apply_job_scheduling(self, state: JobSchedulingState) -> Dict[str, Any]:
        ensured_job_store_url = state.job_store_url
        if state.job_store_url:
            kwargs: dict[str, Any] = {}
            if self._privileged_password_requester is not None:
                kwargs["request_privileged_password"] = self._privileged_password_requester
            staged_privileged = self.get_privileged_credentials()
            if staged_privileged is not None:
                kwargs["privileged_credentials"] = staged_privileged
            logger.info("Provisioning job store at %s", state.job_store_url)
            try:
                ensured_job_store_url = self._bootstrap(state.job_store_url, **kwargs)
            except BootstrapError:
                logger.exception("Job store bootstrap failed for URL %s", state.job_store_url)
                raise
            except Exception as exc:
                logger.exception("Unexpected error bootstrapping job store for URL %s", state.job_store_url)
                raise BootstrapError(f"Failed to provision job store: {exc}") from exc

            if not ensured_job_store_url.startswith(("mongodb://", "mongodb+srv://")):
                logger.info("Validating SQLAlchemy job store URL")
                try:
                    make_url(ensured_job_store_url)
                except Exception as exc:
                    raise BootstrapError(
                        f"Invalid job store URL after provisioning: {exc}"
                    ) from exc

                engine = create_engine(ensured_job_store_url, future=True)
                try:
                    logger.info("Ensuring job store schema is initialized")
                    ensure_job_schema(engine)
                except Exception as exc:
                    logger.exception("Failed to initialize job store schema for URL %s", ensured_job_store_url)
                    raise BootstrapError(f"Failed to initialize job store schema: {exc}") from exc
                finally:
                    engine.dispose()

        policy = {
            "max_attempts": state.retry_policy.max_attempts,
            "backoff_seconds": state.retry_policy.backoff_seconds,
            "jitter_seconds": state.retry_policy.jitter_seconds,
            "backoff_multiplier": state.retry_policy.backoff_multiplier,
        }
        settings = self.config_manager.set_job_scheduling_settings(
            enabled=state.enabled,
            job_store_url=ensured_job_store_url or ConfigManager.UNSET,
            max_workers=state.max_workers if state.max_workers is not None else ConfigManager.UNSET,
            retry_policy=policy,
            timezone=state.timezone or ConfigManager.UNSET,
            queue_size=state.queue_size if state.queue_size is not None else ConfigManager.UNSET,
        )
        self.state.job_scheduling = dataclasses.replace(state, job_store_url=ensured_job_store_url)
        logger.info("Persisted job scheduling settings (enabled=%s)", state.enabled)
        return settings

    # -- messaging ---------------------------------------------------------

    def apply_message_bus(self, state: MessageBusState) -> Dict[str, Any]:
        logger.info("Applying message bus settings for backend '%s'", state.backend)
        if state.backend != "redis":
            initial_offset = "$"
            replay_start = "$"
            redis_url = None
            stream_prefix = None
        else:
            initial_offset = state.initial_offset if state.initial_offset in {"$", "0-0"} else "$"
            replay_start = state.replay_start if state.replay_start in {"$", "0-0"} else initial_offset
            redis_url = state.redis_url
            stream_prefix = state.stream_prefix

        policy_default = {
            "tier": state.policy_tier or "standard",
            "retry_attempts": state.policy_retry_attempts,
            "retry_delay": state.policy_retry_delay,
            "dlq_topic_template": state.policy_dlq_template if state.policy_dlq_enabled else None,
            "replay_start": replay_start,
            "retention_seconds": state.policy_retention_seconds,
            "idempotency_key_field": (
                state.policy_idempotency_key_field if state.policy_idempotency_enabled else None
            ),
            "idempotency_ttl_seconds": (
                state.policy_idempotency_ttl_seconds if state.policy_idempotency_enabled else None
            ),
        }

        kafka_bridge_topics = list(state.kafka_bridge_topics)
        bridge_enabled = state.kafka_enabled and state.kafka_bridge_enabled
        kafka_bridge = {
            "enabled": bridge_enabled,
            "source_prefix": "redis_kafka",
            "topics": kafka_bridge_topics,
            "topic_map": {},
            "batch_size": state.kafka_bridge_batch_size,
            "max_attempts": state.kafka_bridge_max_attempts,
            "backoff_seconds": state.kafka_bridge_backoff_seconds,
            "dlq_topic": state.kafka_bridge_dlq_topic,
        }

        kafka_block = {
            "enabled": state.kafka_enabled,
            "bootstrap_servers": state.kafka_bootstrap_servers,
            "topic_prefix": state.kafka_topic_prefix,
            "client_id": state.kafka_client_id,
            "driver": state.kafka_driver,
            "enable_idempotence": state.kafka_enable_idempotence,
            "acks": state.kafka_acks,
            "max_in_flight": state.kafka_max_in_flight,
            "delivery_timeout": state.kafka_delivery_timeout,
            "bridge": kafka_bridge,
        }
        settings = self.config_manager.set_messaging_settings(
            backend=state.backend,
            redis_url=redis_url,
            stream_prefix=stream_prefix,
            initial_offset=initial_offset,
            replay_start=replay_start,
            min_idle_ms=state.min_idle_ms,
            delete_on_ack=state.delete_on_ack,
            trim_maxlen=state.trim_maxlen,
            policy={"default": policy_default},
            kafka=kafka_block,
        )
        self.state.message_bus = dataclasses.replace(state, initial_offset=initial_offset, replay_start=replay_start)
        return settings

    def apply_kv_store_settings(self, state: KvStoreState) -> Mapping[str, Any]:
        logger.info("Applying key-value store settings (reuse=%s)", state.reuse_conversation_store)
        url_value = state.url if state.url else ConfigManager.UNSET
        settings = self.config_manager.set_kv_store_settings(
            url=url_value,
            reuse_conversation_store=state.reuse_conversation_store,
        )
        self.state.kv_store = dataclasses.replace(state)
        return settings

    # -- providers ---------------------------------------------------------

    def apply_provider_settings(self, state: ProviderState) -> ProviderState:
        logger.info(
            "Applying provider settings (default=%s, model=%s)",
            state.default_provider,
            state.default_model,
        )
        provider_keys = self.config_manager._get_provider_env_keys()
        for provider, key in provider_keys.items():
            if provider in state.api_keys:
                self.config_manager.update_api_key(provider, state.api_keys[provider])

        sanitized_settings: Dict[str, Dict[str, str]] = {}
        for provider, settings in state.settings.items():
            if not isinstance(settings, Mapping):
                continue
            filtered = {
                key: value
                for key, value in settings.items()
                if isinstance(key, str) and isinstance(value, str) and value.strip()
            }
            if filtered:
                sanitized_settings[provider] = filtered

        if state.default_provider:
            self.config_manager.set_default_provider(state.default_provider)
        if state.default_model:
            self.config_manager.set_default_model(state.default_model)

        if sanitized_settings:
            self.config_manager.yaml_config["PROVIDER_SETTINGS"] = dict(sanitized_settings)
            self.config_manager.config["PROVIDER_SETTINGS"] = dict(sanitized_settings)
        else:
            self.config_manager.yaml_config.pop("PROVIDER_SETTINGS", None)
            self.config_manager.config.pop("PROVIDER_SETTINGS", None)
        self.config_manager._write_yaml_config()

        self.state.providers = dataclasses.replace(
            state, api_keys=dict(state.api_keys), settings=dict(sanitized_settings)
        )
        logger.info("Persisted %d provider configurations", len(self.state.providers.settings))
        return self.state.providers

    # -- speech ------------------------------------------------------------

    def apply_speech_settings(self, state: SpeechState) -> SpeechState:
        logger.info(
            "Applying speech settings (tts_enabled=%s, stt_enabled=%s)",
            state.tts_enabled,
            state.stt_enabled,
        )
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
        logger.info("Speech settings persisted")
        return self.state.speech

    # -- user --------------------------------------------------------------

    def register_user(self, state: UserState | None = None) -> Mapping[str, Any]:
        if state is not None:
            profile = self._state_to_profile(state)
        elif self._staged_admin_profile is not None:
            profile = self._staged_admin_profile
        else:
            profile = self._state_to_profile(self.state.user)

        self.set_user_profile(profile)

        logger.info("Registering initial user '%s'", profile.username)

        repository = self._get_conversation_repository()
        service = UserAccountService(
            config_manager=self.config_manager,
            conversation_repository=repository,
        )
        tenant_value: Optional[str] = None
        optional_state = getattr(self.state, "optional", None)
        if optional_state and optional_state.tenant_id:
            tenant_value = optional_state.tenant_id.strip() or None
        if not tenant_value:
            configured_tenant = self.config_manager.get_config("tenant_id")
            if configured_tenant:
                tenant_value = str(configured_tenant).strip() or None
        account = service.register_user(
            username=profile.username,
            password=profile.password,
            email=profile.email,
            name=profile.display_name or profile.full_name or None,
            dob=profile.date_of_birth or None,
            full_name=profile.full_name or None,
            domain=profile.domain or None,
            tenant_id=tenant_value,
        )
        self.config_manager.set_active_user(account.username)
        self._staged_admin_profile = dataclasses.replace(profile)
        logger.info("Registered initial user '%s'", account.username)
        return {
            "username": account.username,
            "email": account.email,
            "display_name": account.name,
            "full_name": profile.full_name or None,
            "domain": profile.domain or None,
            "date_of_birth": profile.date_of_birth or None,
        }

    def _get_conversation_repository(self) -> ConversationStoreRepository:
        if self.atlas and getattr(self.atlas, "conversation_repository", None) is not None:
            return self.atlas.conversation_repository

        factory = self.config_manager.get_conversation_store_session_factory()
        if factory is None:
            raise RuntimeError("Conversation store is not configured")

        retention = self.config_manager.get_conversation_retention_policies()
        repository = ConversationStoreRepository(factory, retention=retention)
        repository.create_schema(logger=logger)
        return repository

    # -- optional ----------------------------------------------------------

    def apply_optional_settings(self, state: OptionalState) -> OptionalState:
        logger.info("Applying optional settings (tenant=%s)", state.tenant_id)
        self.config_manager.set_tenant_id(state.tenant_id)
        self.config_manager.set_conversation_retention(
            days=state.retention_days,
            history_limit=state.retention_history_limit,
        )
        self.config_manager.apply_audit_template(
            state.audit_template,
            apply_retention=False,
            retention_days=state.retention_days,
            retention_history_limit=state.retention_history_limit,
        )
        if state.scheduler_timezone is not None or state.scheduler_queue_size is not None:
            self.config_manager.set_job_scheduling_settings(
                timezone=state.scheduler_timezone if state.scheduler_timezone is not None else ConfigManager.UNSET,
                queue_size=state.scheduler_queue_size if state.scheduler_queue_size is not None else ConfigManager.UNSET,
            )
        self.config_manager.set_http_server_autostart(state.http_auto_start)
        self.config_manager.set_data_residency(
            region=state.data_region,
            residency_requirement=state.residency_requirement,
        )
        self.state.optional = dataclasses.replace(
            self.state.optional,
            tenant_id=state.tenant_id,
            retention_days=state.retention_days,
            retention_history_limit=state.retention_history_limit,
            scheduler_timezone=state.scheduler_timezone,
            scheduler_queue_size=state.scheduler_queue_size,
            http_auto_start=state.http_auto_start,
            audit_template=state.audit_template,
            data_region=state.data_region,
            residency_requirement=state.residency_requirement,
        )
        logger.info("Optional settings persisted")
        return self.state.optional

    def apply_company_identity(self, state: OptionalState) -> OptionalState:
        logger.info("Applying company identity for %s", state.company_name)
        self.config_manager.set_company_identity(
            name=state.company_name,
            domain=state.company_domain,
            primary_contact=state.primary_contact,
            contact_email=state.contact_email,
            address_line1=state.address_line1,
            address_line2=state.address_line2,
            city=state.city,
            state=state.state,
            postal_code=state.postal_code,
            country=state.country,
            phone_number=state.phone_number,
        )
        self.state.optional = dataclasses.replace(
            self.state.optional,
            company_name=state.company_name,
            company_domain=state.company_domain,
            primary_contact=state.primary_contact,
            contact_email=state.contact_email,
            address_line1=state.address_line1,
            address_line2=state.address_line2,
            city=state.city,
            state=state.state,
            postal_code=state.postal_code,
            country=state.country,
            phone_number=state.phone_number,
        )
        logger.info("Company identity persisted")
        return self.state.optional

    # -- staging helpers ----------------------------------------------------

    def _state_to_profile(self, state: UserState) -> AdminProfile:
        staged_privileged = self.get_privileged_credentials()
        privileged_state = state.privileged_credentials
        db_username = (state.privileged_db_username or "").strip()
        db_password = state.privileged_db_password
        if staged_privileged is not None:
            username, password = staged_privileged
            db_username = (username or "").strip()
            db_password = password or ""
        return AdminProfile(
            username=state.username,
            email=state.email,
            password=state.password,
            admin_password=state.admin_password,
            display_name=state.display_name,
            full_name=state.full_name,
            domain=state.domain,
            date_of_birth=state.date_of_birth,
            sudo_username=privileged_state.sudo_username,
            sudo_password=privileged_state.sudo_password,
            privileged_db_username=db_username,
            privileged_db_password=db_password,
        )

    def set_user_profile(self, profile: AdminProfile) -> UserState:
        self._staged_admin_profile = dataclasses.replace(profile)
        privileged_state = PrivilegedCredentialState(
            sudo_username=profile.sudo_username or "",
            sudo_password=profile.sudo_password or "",
        )
        self.state.user = UserState(
            username=profile.username or "",
            email=profile.email or "",
            password=profile.password or "",
            admin_password=profile.admin_password or "",
            display_name=profile.display_name or "",
            full_name=profile.full_name or "",
            domain=profile.domain or "",
            date_of_birth=profile.date_of_birth or "",
            privileged_db_username=profile.privileged_db_username or "",
            privileged_db_password=profile.privileged_db_password or "",
            privileged_credentials=privileged_state,
        )
        db_username = (profile.privileged_db_username or "").strip()
        db_password = profile.privileged_db_password
        if db_username or (isinstance(db_password, str) and db_password.strip()):
            self.set_privileged_credentials((db_username, db_password))
        return self.state.user

    def set_users(self, users: list[SetupUserEntry], *, initial_admin: str | None = None) -> SetupUsersState:
        usernames: set[str] = set()
        unique_users: list[SetupUserEntry] = []
        for entry in users:
            username = entry.username.strip()
            if not username:
                continue
            if username in usernames:
                continue
            usernames.add(username)
            requires_password_reset = getattr(entry, "requires_password_reset", True)
            email = getattr(entry, "email", "")
            unique_users.append(
                SetupUserEntry(
                    username=username,
                    full_name=entry.full_name.strip(),
                    password=entry.password,
                    email=email.strip(),
                    requires_password_reset=bool(requires_password_reset),
                )
            )
        if unique_users:
            unique_users[0] = dataclasses.replace(unique_users[0], requires_password_reset=False)
        admin_username = (initial_admin or self.state.users.initial_admin_username).strip()
        if admin_username and admin_username not in usernames:
            admin_username = ""
        if not admin_username and unique_users:
            admin_username = unique_users[0].username
        self.state.users = SetupUsersState(entries=unique_users, initial_admin_username=admin_username)
        return self.state.users

    def add_user_entry(self, entry: SetupUserEntry) -> SetupUsersState:
        existing = [e for e in self.state.users.entries if e.username != entry.username]
        existing.append(entry)
        return self.set_users(existing, initial_admin=self.state.users.initial_admin_username)

    def remove_user(self, username: str) -> SetupUsersState:
        remaining = [e for e in self.state.users.entries if e.username != username]
        admin_username = self.state.users.initial_admin_username
        if admin_username == username:
            admin_username = remaining[0].username if remaining else ""
        return self.set_users(remaining, initial_admin=admin_username)

    def set_privileged_credentials(
        self, credentials: tuple[str | None, str | None] | None
    ) -> None:
        if credentials is None:
            self._staged_privileged_credentials = None
            self.state.user.privileged_db_username = ""
            self.state.user.privileged_db_password = ""
            return
        username, password = credentials
        cleaned_username = (username or "").strip()
        cleaned_password: str | None
        if isinstance(password, str):
            cleaned_password = password.strip() or None
        else:
            cleaned_password = password
        if not cleaned_username and cleaned_password is None:
            self._staged_privileged_credentials = None
            return
        self._staged_privileged_credentials = (
            cleaned_username or None,
            cleaned_password,
        )
        self.state.user.privileged_db_username = cleaned_username
        self.state.user.privileged_db_password = cleaned_password or ""

    def get_privileged_credentials(self) -> tuple[str | None, str | None] | None:
        return self._staged_privileged_credentials

    def get_conversation_backend_options(self):
        """Return available conversation store backend presets."""

        return self.config_manager.get_conversation_store_backends()

    # -- summary -----------------------------------------------------------

    def build_summary(self) -> Dict[str, Any]:
        user_state = self.state.user
        privileged_state = user_state.privileged_credentials
        staged_privileged = self.get_privileged_credentials()
        architecture_state = self.state.storage_architecture
        user_summary: Dict[str, Any] = {
            "username": user_state.username,
            "email": user_state.email,
            "display_name": user_state.display_name,
            "full_name": user_state.full_name,
            "domain": user_state.domain,
            "date_of_birth": user_state.date_of_birth,
            "has_password": bool(user_state.password),
            "has_admin_password": bool(getattr(user_state, "admin_password", "")),
            "privileged_credentials": {
                "sudo_username": privileged_state.sudo_username,
                "has_sudo_password": bool(privileged_state.sudo_password),
            },
        }
        if staged_privileged is not None:
            username, password = staged_privileged
            user_summary["database_privileged_username"] = username
            user_summary["has_database_privileged_password"] = bool(password)
        return {
            "storage_architecture": {
                "performance_mode": architecture_state.performance_mode.value,
                "main_db": self.state.database.backend
                or architecture_state.conversation_backend,
                "document_db": architecture_state.conversation_backend,
                "vector_db": self.state.vector_store.adapter
                or architecture_state.vector_store_adapter,
                "kv_store": {
                    "reuse_conversation_store": architecture_state.kv_reuse_conversation_store,
                    "url": self.state.kv_store.url,
                },
            },
            "database": dataclasses.asdict(self.state.database),
            "job_scheduling": dataclasses.asdict(self.state.job_scheduling),
            "message_bus": dataclasses.asdict(self.state.message_bus),
            "vector_store": dataclasses.asdict(self.state.vector_store),
            "kv_store": dataclasses.asdict(self.state.kv_store),
            "providers": {
                "default_provider": self.state.providers.default_provider,
                "default_model": self.state.providers.default_model,
                "configured_providers": sorted(self.state.providers.api_keys.keys()),
            },
            "speech": dataclasses.asdict(self.state.speech),
            "users": {
                "initial_admin": self.state.users.initial_admin_username,
                "entries": [
                    {
                        "username": entry.username,
                        "full_name": entry.full_name,
                        "has_password": bool(entry.password),
                        "requires_password_reset": bool(
                            getattr(entry, "requires_password_reset", False)
                        ),
                    }
                    for entry in self.state.users.entries
                ],
            },
            "user": user_summary,
            "hardware_profile": dataclasses.asdict(self.state.hardware_profile),
            "setup_recommended_mode": self.state.setup_recommended_mode,
            "optional": dataclasses.asdict(self.state.optional),
            "setup_type": dataclasses.asdict(self.state.setup_type),
        }
