"""Command-line setup utility for ATLAS."""

from __future__ import annotations

import dataclasses
import getpass
import os
import platform
import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

from core.setup.controller import (
    AdminProfile,
    DatabaseState,
    JobSchedulingState,
    KvStoreState,
    MessageBusState,
    OptionalState,
    ProviderState,
    SetupTypeState,
    SetupWizardController,
    SpeechState,
    UserState,
    _parse_default_dsn,
)
from core.setup_marker import write_setup_marker
from modules.conversation_store.bootstrap import BootstrapError


class SetupUtility:
    """Interactive helper that collects configuration and applies it."""

    def __init__(
        self,
        *,
        controller: SetupWizardController | None = None,
        input_func: Callable[[str], str] = input,
        getpass_func: Callable[[str], str] = getpass.getpass,
        print_func: Callable[[str], None] = print,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        platform_system: Callable[[], str] = platform.system,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.controller = controller or SetupWizardController()
        self._input = input_func
        self._getpass = getpass_func
        self._print = print_func
        self._run = run
        self._platform_system = platform_system
        self._env: Mapping[str, str] = dict(os.environ if env is None else env)
        self._non_interactive = False

    # -- public API -----------------------------------------------------

    def run(self, *, non_interactive: bool = False) -> Path:
        if non_interactive:
            return self._run_non_interactive()
        return self._run_interactive()

    def _run_interactive(self) -> Path:
        self._non_interactive = False
        """Execute the full setup workflow."""

        self.choose_setup_type()
        self.configure_user()

        self.install_postgresql()

        dsn = self.configure_database()
        self._print(f"Conversation store DSN saved: {dsn}")

        self.configure_vector_store()

        self.configure_kv_store()
        self.configure_job_scheduling()
        self.configure_message_bus()
        self.configure_providers()
        self.configure_speech()
        self.configure_optional_settings()

        self.controller.register_user()

        summary = self.controller.build_summary()
        marker_path = self.finalize(summary)
        self._print(f"Setup complete. Sentinel written to {marker_path}")
        return marker_path

    def _run_non_interactive(self) -> Path:
        """Execute the setup workflow without prompting."""

        self._non_interactive = True
        self._load_non_interactive_sources()
        database_state = self._synchronize_database_state()
        privileged = self.controller.get_privileged_credentials()
        dsn = self.controller.apply_database_settings(
            database_state,
            privileged_credentials=privileged,
        )
        self._print(f"Conversation store DSN saved: {dsn}")

        self.controller.apply_vector_store_settings(self.controller.state.vector_store)

        self.controller.apply_kv_store_settings(self.controller.state.kv_store)
        self.controller.apply_job_scheduling(self.controller.state.job_scheduling)
        self.controller.apply_message_bus(self.controller.state.message_bus)
        self.controller.apply_provider_settings(self.controller.state.providers)
        self.controller.apply_speech_settings(self.controller.state.speech)
        self.controller.apply_company_identity(self.controller.state.optional)
        self.controller.apply_optional_settings(self.controller.state.optional)

        self.controller.register_user()

        summary = self.controller.build_summary()
        marker_path = self.finalize(summary)
        self._print(f"Setup complete. Sentinel written to {marker_path}")
        return marker_path

    def _load_non_interactive_sources(self) -> None:
        config_path = self._env.get("ATLAS_SETUP_CONFIG")
        if config_path:
            self.controller.import_config(config_path)

        setup_choice = self._env.get("ATLAS_SETUP_TYPE")
        if setup_choice:
            chosen_mode = setup_choice
        else:
            chosen_mode = self.controller.state.setup_type.mode or "personal"
        self.controller.apply_setup_type(chosen_mode)

        profile = self._build_admin_profile_from_env()
        self.controller.set_user_profile(profile)
        self._apply_state_overrides_from_env()

    def choose_setup_type(self) -> SetupTypeState:
        """Prompt for and apply the desired setup preset."""

        while True:
            choice = self._ask("Setup type (personal/enterprise/regulatory)", "personal")
            normalized = choice.strip().lower() or "personal"
            if normalized in {"personal", "enterprise", "regulatory"}:
                return self.controller.apply_setup_type(normalized)
            self._print("Please choose 'personal', 'enterprise', or 'regulatory'.")

    def _apply_state_overrides_from_env(self) -> None:
        self._apply_provider_env_overrides()
        self._apply_optional_env_overrides()
        self._apply_job_env_overrides()
        self._apply_message_bus_env_overrides()
        self._apply_vector_store_env_overrides()
        self._apply_kv_env_overrides()
        self._apply_speech_env_overrides()

    def _queue_defaults(self) -> tuple[int, int]:
        resolver = getattr(self.controller, "_resolve_queue_defaults", None)
        backend = getattr(self.controller.state.message_bus, "backend", "in_memory")
        if callable(resolver):
            return resolver(backend=backend)  # type: ignore
        normalized = (backend or "in_memory").strip().lower()
        if normalized == "redis":
            return 8, 500
        return 4, 100

    def _build_admin_profile_from_env(self) -> AdminProfile:
        state = self.controller.state.user
        privileged_state = state.privileged_credentials
        staged_privileged = self.controller.get_privileged_credentials()
        staged_username = ""
        staged_password = ""
        if staged_privileged is not None:
            staged_username = staged_privileged[0] or ""
            staged_password = staged_privileged[1] or ""

        username = self._require_value("ATLAS_ADMIN_USERNAME", state.username)
        email = self._require_value("ATLAS_ADMIN_EMAIL", state.email)

        password_candidate = self._env_get("ATLAS_ADMIN_PASSWORD")
        if password_candidate is None:
            password_candidate = state.password.strip()
        if not password_candidate:
            raise RuntimeError("Missing required configuration: ATLAS_ADMIN_PASSWORD")

        admin_password_candidate = self._env_get("ATLAS_ADMIN_PRIVILEGED_PASSWORD")
        if admin_password_candidate is None:
            admin_password_candidate = getattr(state, "admin_password", "").strip()
        if not admin_password_candidate:
            admin_password_candidate = password_candidate

        full_name = self._env_get("ATLAS_ADMIN_FULL_NAME")
        if full_name is None:
            full_name = state.full_name

        display_name = self._env_get("ATLAS_ADMIN_DISPLAY_NAME")
        if display_name is None:
            display_name = state.display_name

        domain = self._env_get("ATLAS_ADMIN_DOMAIN")
        if domain is None:
            domain = state.domain

        dob_value = self._env_get("ATLAS_ADMIN_DOB")
        if dob_value:
            try:
                dob = self._normalize_date(dob_value)
            except ValueError as exc:  # pragma: no cover - surfaced to caller
                raise RuntimeError(str(exc)) from exc
        else:
            dob = state.date_of_birth

        sudo_username = self._env_get("ATLAS_SUDO_USERNAME")
        if sudo_username is None:
            sudo_username = privileged_state.sudo_username

        sudo_password = self._env_get("ATLAS_SUDO_PASSWORD")
        if sudo_password is None:
            sudo_password = privileged_state.sudo_password

        db_privileged_user = self._env_get("ATLAS_DATABASE_PRIVILEGED_USER")
        if db_privileged_user is None:
            db_privileged_user = staged_username

        db_privileged_password = self._env_get("ATLAS_DATABASE_PRIVILEGED_PASSWORD")
        if db_privileged_password is None:
            db_privileged_password = staged_password

        return AdminProfile(
            username=username,
            email=email,
            password=password_candidate,
            admin_password=admin_password_candidate,
            display_name=display_name or "",
            full_name=full_name or "",
            domain=domain or "",
            date_of_birth=dob or "",
            sudo_username=sudo_username or "",
            sudo_password=sudo_password or "",
            privileged_db_username=db_privileged_user or "",
            privileged_db_password=db_privileged_password or "",
        )

    def _synchronize_database_state(self) -> DatabaseState:
        state = self.controller.state.database
        host_override = self._env_get("ATLAS_DATABASE_HOST")
        user_override = self._env_get("ATLAS_DATABASE_USER")
        database_override = self._env_get("ATLAS_DATABASE_NAME")
        password_override = self._env_get("ATLAS_DATABASE_PASSWORD")
        port_override = self._env_int("ATLAS_DATABASE_PORT")

        new_state = dataclasses.replace(
            state,
            host=host_override or state.host,
            port=port_override if port_override is not None else state.port,
            database=database_override or state.database,
            user=user_override or state.user,
            password=state.password if password_override is None else password_override,
        )
        self.controller.state.database = new_state
        return new_state

    def _env_get(self, key: str) -> str | None:
        if key not in self._env:
            return None
        return self._env[key].strip()

    def _require_value(self, key: str, fallback: str) -> str:
        candidate = self._env_get(key)
        if candidate is None:
            candidate = (fallback or "").strip()
        if not candidate:
            raise RuntimeError(f"Missing required configuration: {key}")
        return candidate

    def _ensure_prompts_enabled(self) -> None:
        if self._non_interactive:
            raise RuntimeError(
                "Prompts are disabled in non-interactive mode. Provide configuration "
                "via environment variables or configuration files."
            )

    def _env_int(self, key: str) -> int | None:
        value = self._env_get(key)
        if value in (None, ""):
            return None
        try:
            return int(value)
        except ValueError as exc:
            raise RuntimeError(f"Invalid integer value for {key!r}: {value}") from exc

    def _env_float(self, key: str) -> float | None:
        value = self._env_get(key)
        if value in (None, ""):
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise RuntimeError(f"Invalid numeric value for {key!r}: {value}") from exc

    def _env_bool(self, key: str) -> bool | None:
        value = self._env_get(key)
        if value is None:
            return None
        if value == "":
            return None
        normalized = value.lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise RuntimeError(f"Invalid boolean value for {key!r}: {value}")

    def _apply_provider_env_overrides(self) -> None:
        state = self.controller.state.providers
        api_keys = dict(state.api_keys)
        prefix = "ATLAS_PROVIDER_KEY_"
        for key, value in self._env.items():
            if not key.startswith(prefix):
                continue
            provider = key[len(prefix) :].lower()
            cleaned = value.strip()
            if cleaned:
                api_keys[provider] = cleaned
            else:
                api_keys.pop(provider, None)

        default_provider = self._env_get("ATLAS_PROVIDER_DEFAULT")
        if default_provider is not None:
            default_provider = default_provider or None
        else:
            default_provider = state.default_provider

        default_model = self._env_get("ATLAS_PROVIDER_MODEL")
        if default_model is not None:
            default_model = default_model or None
        else:
            default_model = state.default_model

        self.controller.state.providers = dataclasses.replace(
            state,
            api_keys=api_keys,
            default_provider=default_provider,
            default_model=default_model,
        )

    def _apply_optional_env_overrides(self) -> None:
        state = self.controller.state.optional
        tenant = self._env_get("ATLAS_OPTIONAL_TENANT_ID")
        if tenant is not None:
            tenant_value = tenant or None
        else:
            tenant_value = state.tenant_id

        retention_days_override = self._env_int("ATLAS_OPTIONAL_RETENTION_DAYS")
        retention_days = (
            retention_days_override if retention_days_override is not None else state.retention_days
        )

        retention_history_override = self._env_int("ATLAS_OPTIONAL_RETENTION_HISTORY_LIMIT")
        retention_history = (
            retention_history_override
            if retention_history_override is not None
            else state.retention_history_limit
        )

        scheduler_timezone_override = self._env_get("ATLAS_OPTIONAL_SCHEDULER_TIMEZONE")
        if scheduler_timezone_override is not None:
            scheduler_timezone = scheduler_timezone_override or None
        else:
            scheduler_timezone = state.scheduler_timezone

        scheduler_queue_override = self._env_int("ATLAS_OPTIONAL_SCHEDULER_QUEUE_SIZE")
        scheduler_queue = (
            scheduler_queue_override
            if scheduler_queue_override is not None
            else state.scheduler_queue_size
        )

        http_auto_start_override = self._env_bool("ATLAS_OPTIONAL_HTTP_AUTO_START")
        http_auto_start = (
            http_auto_start_override if http_auto_start_override is not None else state.http_auto_start
        )

        self.controller.state.optional = dataclasses.replace(
            state,
            tenant_id=tenant_value,
            retention_days=retention_days,
            retention_history_limit=retention_history,
            scheduler_timezone=scheduler_timezone,
            scheduler_queue_size=scheduler_queue,
            http_auto_start=http_auto_start,
        )

    def _apply_job_env_overrides(self) -> None:
        state = self.controller.state.job_scheduling
        enabled_override = self._env_bool("ATLAS_JOB_ENABLED")
        enabled = enabled_override if enabled_override is not None else state.enabled

        job_store_override = self._env_get("ATLAS_JOB_STORE_URL")
        if job_store_override is not None:
            job_store_url = job_store_override or None
        else:
            job_store_url = state.job_store_url

        max_workers_override = self._env_int("ATLAS_JOB_MAX_WORKERS")
        max_workers = max_workers_override if max_workers_override is not None else state.max_workers

        timezone_override = self._env_get("ATLAS_JOB_TIMEZONE")
        if timezone_override is not None:
            timezone = timezone_override or None
        else:
            timezone = state.timezone

        queue_size_override = self._env_int("ATLAS_JOB_QUEUE_SIZE")
        queue_size = queue_size_override if queue_size_override is not None else state.queue_size

        retry = state.retry_policy
        max_attempts_override = self._env_int("ATLAS_JOB_RETRY_ATTEMPTS")
        backoff_seconds_override = self._env_float("ATLAS_JOB_RETRY_BACKOFF_SECONDS")
        jitter_seconds_override = self._env_float("ATLAS_JOB_RETRY_JITTER_SECONDS")
        backoff_multiplier_override = self._env_float("ATLAS_JOB_RETRY_BACKOFF_MULTIPLIER")

        retry_policy = dataclasses.replace(
            retry,
            max_attempts=max_attempts_override if max_attempts_override is not None else retry.max_attempts,
            backoff_seconds=(
                backoff_seconds_override if backoff_seconds_override is not None else retry.backoff_seconds
            ),
            jitter_seconds=(
                jitter_seconds_override if jitter_seconds_override is not None else retry.jitter_seconds
            ),
            backoff_multiplier=(
                backoff_multiplier_override
                if backoff_multiplier_override is not None
                else retry.backoff_multiplier
            ),
        )

        self.controller.state.job_scheduling = dataclasses.replace(
            state,
            enabled=enabled,
            job_store_url=job_store_url,
            max_workers=max_workers,
            timezone=timezone,
            queue_size=queue_size,
            retry_policy=retry_policy,
        )

    def _apply_message_bus_env_overrides(self) -> None:
        state = self.controller.state.message_bus
        backend_override = self._env_get("ATLAS_MESSAGE_BUS_BACKEND")
        backend = state.backend
        if backend_override:
            normalized = backend_override.lower()
            if normalized in {"in_memory", "redis", "ncb"}:
                backend = normalized

        redis_override = self._env_get("ATLAS_REDIS_URL")
        stream_override = self._env_get("ATLAS_STREAM_PREFIX")

        redis_url = redis_override if redis_override is not None else state.redis_url
        if redis_url == "":
            redis_url = None

        stream_prefix = stream_override if stream_override is not None else state.stream_prefix
        if stream_prefix == "":
            stream_prefix = None

        initial_offset = (state.initial_offset or "$").strip() or "$"
        replay_start = (state.replay_start or initial_offset).strip() or initial_offset

        offset_override = self._env_get("ATLAS_MESSAGING_INITIAL_OFFSET")
        if offset_override is not None:
            initial_offset = self._normalize_offset_choice(offset_override, fallback=initial_offset)

        replay_override = self._env_get("ATLAS_MESSAGING_REPLAY_START")
        if replay_override is not None:
            replay_start = self._normalize_offset_choice(replay_override, fallback=replay_start)

        if backend != "redis":
            redis_url = None
            stream_prefix = None
            initial_offset = "$"
            replay_start = "$"

        ncb_persistence_path = self._env_get("ATLAS_NCB_PERSISTENCE_PATH") or state.ncb_persistence_path
        ncb_enable_prometheus = self._env_bool("ATLAS_NCB_ENABLE_PROMETHEUS") or state.ncb_enable_prometheus
        ncb_prometheus_port = self._env_int("ATLAS_NCB_PROMETHEUS_PORT") or state.ncb_prometheus_port

        if backend != "ncb":
            ncb_persistence_path = None
            ncb_enable_prometheus = False
            ncb_prometheus_port = 8000

        policy_tier = self._env_get("ATLAS_MESSAGING_TIER") or state.policy_tier
        dlq_enabled_override = self._env_bool("ATLAS_MESSAGING_DLQ_ENABLED")
        policy_dlq_enabled = state.policy_dlq_enabled if dlq_enabled_override is None else dlq_enabled_override

        dlq_template_override = self._env_get("ATLAS_MESSAGING_DLQ_TEMPLATE")
        policy_dlq_template = (
            dlq_template_override if dlq_template_override is not None else state.policy_dlq_template
        )
        if policy_dlq_template == "":
            policy_dlq_template = None
        if not policy_dlq_enabled:
            policy_dlq_template = None

        retention_override = self._env_int("ATLAS_MESSAGING_RETENTION_SECONDS")
        policy_retention_seconds = (
            retention_override if retention_override is not None else state.policy_retention_seconds
        )

        trim_override = self._env_int("ATLAS_MESSAGING_TRIM_MAXLEN")
        trim_maxlen = trim_override if trim_override is not None else state.trim_maxlen

        idempotency_enabled_override = self._env_bool("ATLAS_MESSAGING_IDEMPOTENCY_ENABLED")
        policy_idempotency_enabled = (
            state.policy_idempotency_enabled if idempotency_enabled_override is None else idempotency_enabled_override
        )
        idempotency_key_override = self._env_get("ATLAS_MESSAGING_IDEMPOTENCY_KEY")
        policy_idempotency_key_field = (
            idempotency_key_override if idempotency_key_override is not None else state.policy_idempotency_key_field
        )
        if policy_idempotency_key_field == "":
            policy_idempotency_key_field = None
        idempotency_ttl_override = self._env_int("ATLAS_MESSAGING_IDEMPOTENCY_TTL")
        policy_idempotency_ttl_seconds = (
            idempotency_ttl_override
            if idempotency_ttl_override is not None
            else state.policy_idempotency_ttl_seconds
        )
        if not policy_idempotency_enabled:
            policy_idempotency_key_field = None
            policy_idempotency_ttl_seconds = None

        kafka_enabled_override = self._env_bool("ATLAS_KAFKA_ENABLED")
        kafka_enabled = state.kafka_enabled if kafka_enabled_override is None else kafka_enabled_override
        kafka_bootstrap_override = self._env_get("ATLAS_KAFKA_BOOTSTRAP")
        kafka_bootstrap_servers = (
            kafka_bootstrap_override
            if kafka_bootstrap_override is not None
            else state.kafka_bootstrap_servers
        )
        if kafka_bootstrap_servers == "":
            kafka_bootstrap_servers = None

        topic_prefix_override = self._env_get("ATLAS_KAFKA_TOPIC_PREFIX")
        kafka_topic_prefix = topic_prefix_override if topic_prefix_override is not None else state.kafka_topic_prefix

        kafka_client_id_override = self._env_get("ATLAS_KAFKA_CLIENT_ID")
        kafka_client_id = kafka_client_id_override if kafka_client_id_override is not None else state.kafka_client_id

        kafka_driver_override = self._env_get("ATLAS_KAFKA_DRIVER")
        kafka_driver = kafka_driver_override if kafka_driver_override is not None else state.kafka_driver
        if kafka_driver == "":
            kafka_driver = None

        kafka_idempotence_override = self._env_bool("ATLAS_KAFKA_IDEMPOTENCE")
        kafka_enable_idempotence = (
            state.kafka_enable_idempotence if kafka_idempotence_override is None else kafka_idempotence_override
        )

        kafka_acks_override = self._env_get("ATLAS_KAFKA_ACKS")
        kafka_acks = kafka_acks_override if kafka_acks_override is not None else state.kafka_acks

        kafka_max_in_flight_override = self._env_int("ATLAS_KAFKA_MAX_IN_FLIGHT")
        kafka_max_in_flight = (
            kafka_max_in_flight_override
            if kafka_max_in_flight_override is not None
            else state.kafka_max_in_flight
        )

        kafka_delivery_timeout_override = self._env_float("ATLAS_KAFKA_DELIVERY_TIMEOUT")
        kafka_delivery_timeout = (
            kafka_delivery_timeout_override
            if kafka_delivery_timeout_override is not None
            else state.kafka_delivery_timeout
        )

        bridge_enabled_override = self._env_bool("ATLAS_BRIDGE_ENABLED")
        kafka_bridge_enabled = (
            state.kafka_bridge_enabled if bridge_enabled_override is None else bridge_enabled_override
        )

        bridge_topics_override = self._env_get("ATLAS_BRIDGE_TOPICS")
        kafka_bridge_topics = state.kafka_bridge_topics
        if bridge_topics_override is not None:
            kafka_bridge_topics = tuple(
                topic.strip() for topic in bridge_topics_override.split(",") if topic.strip()
            )

        bridge_batch_override = self._env_int("ATLAS_BRIDGE_BATCH_SIZE")
        kafka_bridge_batch_size = (
            bridge_batch_override if bridge_batch_override is not None else state.kafka_bridge_batch_size
        )

        bridge_attempts_override = self._env_int("ATLAS_BRIDGE_MAX_ATTEMPTS")
        kafka_bridge_max_attempts = (
            bridge_attempts_override if bridge_attempts_override is not None else state.kafka_bridge_max_attempts
        )

        bridge_backoff_override = self._env_float("ATLAS_BRIDGE_BACKOFF_SECONDS")
        kafka_bridge_backoff_seconds = (
            bridge_backoff_override if bridge_backoff_override is not None else state.kafka_bridge_backoff_seconds
        )

        bridge_dlq_override = self._env_get("ATLAS_BRIDGE_DLQ_TOPIC")
        kafka_bridge_dlq_topic = bridge_dlq_override if bridge_dlq_override is not None else state.kafka_bridge_dlq_topic

        if not kafka_enabled:
            kafka_bridge_enabled = False
            kafka_bridge_topics = ()

        self.controller.state.message_bus = dataclasses.replace(
            state,
            backend=backend,
            redis_url=redis_url,
            stream_prefix=stream_prefix,
            initial_offset=initial_offset,
            replay_start=replay_start,
            trim_maxlen=trim_maxlen,
            policy_tier=policy_tier or "standard",
            policy_dlq_enabled=policy_dlq_enabled,
            policy_dlq_template=policy_dlq_template,
            policy_retention_seconds=policy_retention_seconds,
            policy_idempotency_enabled=policy_idempotency_enabled,
            policy_idempotency_key_field=policy_idempotency_key_field,
            policy_idempotency_ttl_seconds=policy_idempotency_ttl_seconds,
            kafka_enabled=kafka_enabled,
            kafka_bootstrap_servers=kafka_bootstrap_servers,
            kafka_topic_prefix=kafka_topic_prefix or "atlas.bus",
            kafka_client_id=kafka_client_id or "atlas-message-bridge",
            kafka_driver=kafka_driver,
            kafka_enable_idempotence=kafka_enable_idempotence,
            kafka_acks=kafka_acks or "all",
            kafka_max_in_flight=kafka_max_in_flight,
            kafka_delivery_timeout=kafka_delivery_timeout,
            kafka_bridge_enabled=kafka_bridge_enabled,
            kafka_bridge_topics=kafka_bridge_topics,
            kafka_bridge_batch_size=kafka_bridge_batch_size,
            kafka_bridge_max_attempts=kafka_bridge_max_attempts,
            kafka_bridge_backoff_seconds=kafka_bridge_backoff_seconds,
            kafka_bridge_dlq_topic=kafka_bridge_dlq_topic or "atlas.bridge.dlq",
        )

    def _apply_vector_store_env_overrides(self) -> None:
        adapter_override = self._env_get("ATLAS_VECTOR_STORE_ADAPTER")
        if not adapter_override:
            return

        normalized = adapter_override.strip().lower()
        state = self.controller.state.vector_store
        new_state = dataclasses.replace(state, adapter=normalized)
        self.controller.apply_vector_store_settings(new_state)

    def _apply_kv_env_overrides(self) -> None:
        state = self.controller.state.kv_store
        reuse_override = self._env_bool("ATLAS_KV_REUSE_CONVERSATION_STORE")
        reuse = reuse_override if reuse_override is not None else state.reuse_conversation_store

        url_override = self._env_get("ATLAS_KV_URL")
        if url_override is not None:
            url = url_override or None
        else:
            url = state.url

        self.controller.state.kv_store = dataclasses.replace(
            state,
            reuse_conversation_store=reuse,
            url=url,
        )

    def _apply_speech_env_overrides(self) -> None:
        state = self.controller.state.speech
        tts_override = self._env_bool("ATLAS_SPEECH_TTS_ENABLED")
        stt_override = self._env_bool("ATLAS_SPEECH_STT_ENABLED")
        default_tts_override = self._env_get("ATLAS_SPEECH_DEFAULT_TTS")
        default_stt_override = self._env_get("ATLAS_SPEECH_DEFAULT_STT")
        elevenlabs_override = self._env_get("ATLAS_SPEECH_ELEVENLABS_KEY")
        openai_override = self._env_get("ATLAS_SPEECH_OPENAI_KEY")
        google_override = self._env_get("ATLAS_SPEECH_GOOGLE_CREDENTIALS")

        self.controller.state.speech = dataclasses.replace(
            state,
            tts_enabled=tts_override if tts_override is not None else state.tts_enabled,
            stt_enabled=stt_override if stt_override is not None else state.stt_enabled,
            default_tts_provider=(
                default_tts_override or None if default_tts_override is not None else state.default_tts_provider
            ),
            default_stt_provider=(
                default_stt_override or None if default_stt_override is not None else state.default_stt_provider
            ),
            elevenlabs_key=(
                elevenlabs_override or None if elevenlabs_override is not None else state.elevenlabs_key
            ),
            openai_key=(openai_override or None if openai_override is not None else state.openai_key),
            google_credentials=(
                google_override or None if google_override is not None else state.google_credentials
            ),
        )

    # -- environment helpers -------------------------------------------

    def ensure_virtualenv(self, project_root: Path, python_executable: str | None = None) -> Path:
        """Create or update the project's virtual environment."""

        venv_path = project_root / ".venv"
        python_executable = python_executable or sys.executable

        if not venv_path.exists():
            self._print(f"Creating virtual environment at {venv_path}…")
            self._run([python_executable, "-m", "venv", str(venv_path)], check=True)

        pip = venv_path / ("Scripts" if os.name == "nt" else "bin") / "pip"
        requirements = project_root / "requirements.txt"
        if requirements.exists():
            self._print("Installing Python requirements…")
            self._run([str(pip), "install", "-r", str(requirements)], check=True)
        else:
            self._print("No requirements.txt found; skipping dependency installation.")
        return venv_path

    def install_postgresql(self) -> None:
        """Attempt to install PostgreSQL using platform-specific commands."""

        system = self._platform_system()
        commands = self._postgres_commands(system)
        if not commands:
            self._print(
                "Unsupported platform for automatic PostgreSQL installation. "
                "Install PostgreSQL manually before continuing."
            )
            return

        self._print("The setup utility can run the following commands to install PostgreSQL:")
        for command in commands:
            self._print("  " + " ".join(command))

        if not self._confirm("Proceed with PostgreSQL installation? [y/N]: ", default=False):
            self._print("Skipping PostgreSQL installation.")
            return

        sudo_state = self.controller.state.user.privileged_credentials
        sudo_password = sudo_state.sudo_password.strip() or None
        if any(cmd and cmd[0] == "sudo" for cmd in commands):
            if sudo_password is None:
                entered = self._getpass("Enter sudo password (leave blank to skip automation): ")
                sudo_password = entered.strip() or None

        for command in commands:
            kwargs: dict[str, object] = {"check": True}
            if sudo_password and command and command[0] == "sudo":
                kwargs["input"] = f"{sudo_password}\n"
                kwargs["text"] = True
            self._run(command, **kwargs)

    def _install_pgvector(self) -> None:
        """Install pgvector Python package for PostgreSQL vector support."""
        import importlib.util

        if importlib.util.find_spec("pgvector") is not None:
            return  # Already installed

        self._print("Installing pgvector for PostgreSQL vector support...")
        pip = sys.executable
        try:
            self._run([pip, "-m", "pip", "install", "pgvector"], check=True)
            self._print("pgvector installed successfully.")
        except subprocess.CalledProcessError as exc:
            self._print(
                f"Warning: Failed to install pgvector automatically: {exc}. "
                "You may need to install it manually with `pip install pgvector`."
            )

    # -- configuration collection --------------------------------------

    def configure_database(self) -> str:
        """Prompt for conversation store settings and persist them."""

        state = self.controller.state.database
        privileged_credentials = self.controller.get_privileged_credentials()
        user_state = self.controller.state.user
        backend_options = self.controller.get_conversation_backend_options()
        backend_names = [option.name for option in backend_options]
        backend_map = {option.name: option for option in backend_options}
        backend_default = state.backend or (backend_names[0] if backend_names else "postgresql")

        while True:
            choice_prompt = "Conversation store backend"
            if backend_names:
                choice_prompt += f" ({'/'.join(backend_names)})"
            backend_choice = self._ask(choice_prompt, backend_default)
            normalized_backend = (backend_choice or backend_default or "postgresql").strip().lower()
            if backend_names and normalized_backend not in backend_names:
                self._print(
                    "Please choose one of the supported backends: " + ", ".join(backend_names)
                )
                continue
            backend_default = normalized_backend
            break

        selected_backend = backend_default
        if state.backend != selected_backend:
            option = backend_map.get(selected_backend)
            if option is not None:
              state = _parse_default_dsn(option.dsn, backend=option.name)  
            else:
                state = dataclasses.replace(state, backend=selected_backend)
        else:
            state = dataclasses.replace(state, backend=selected_backend)

        # Install pgvector when PostgreSQL is selected
        if selected_backend == "postgresql":
            self._install_pgvector()

        while True:
            if state.backend == "sqlite":
                database_prompt = "SQLite database path"
                database_default = state.database or str(Path.home() / "atlas.sqlite3")
                database = self._ask(database_prompt, database_default) or database_default
                # Ensure the path is absolute
                database = str(Path(database).expanduser().resolve())
                new_state = dataclasses.replace(
                    state,
                    database=database,
                    host="",
                    port=0,
                    user="",
                    password="",
                    dsn="",
                    options="",
                )
            elif state.backend == "mongodb":
                existing_uri = state.dsn if isinstance(state.dsn, str) else ""
                if existing_uri and not existing_uri.strip().lower().startswith("mongodb"):
                    existing_uri = ""
                uri_prompt = "MongoDB connection URI (leave blank to build from details)"
                uri_value = self._ask(uri_prompt, existing_uri) or existing_uri
                uri_value = uri_value.strip()
                if uri_value:
                    new_state = dataclasses.replace(
                        state,
                        dsn=uri_value,
                        host="",
                        port=0,
                        database="",
                        user="",
                        password="",
                        options="",
                    )
                else:
                    host_default = state.host or "localhost"
                    host = self._ask("MongoDB host", host_default) or host_default
                    port_default = state.port if state.port not in (0, None) else 27017
                    port = self._ask_int("MongoDB port", port_default)
                    database_default = state.database or "atlas"
                    database = self._ask("MongoDB database", database_default) or database_default
                    user_default = state.user or ""
                    user = self._ask("MongoDB username", user_default) or user_default
                    password_prompt = "MongoDB password"
                    if state.password:
                        password_prompt += " (leave blank to keep existing, type !clear! to remove)"
                    password = self._ask_password(password_prompt, state.password)
                    options_prompt = "MongoDB connection options (e.g. authSource=admin)"
                    options_value = self._ask(options_prompt, state.options) or state.options
                    options = options_value.strip()
                    new_state = dataclasses.replace(
                        state,
                        host=host.strip() or "localhost",
                        port=port,
                        database=database,
                        user=user,
                        password=password,
                        dsn="",
                        options=options,
                    )
            else:
                host_default = state.host
                user_domain = user_state.domain.strip()
                if host_default in {"", "localhost"} and user_domain:
                    host_default = user_domain
                host = self._ask("PostgreSQL host", host_default) or host_default
                port = self._ask_int("PostgreSQL port", state.port or 5432)
                database = self._ask("Database name", state.database) or state.database
                user_default = state.user or user_state.username or state.user
                user = self._ask("Database user", user_default) or user_default
                password_prompt = "Database password"
                if state.password:
                    password_prompt += " (leave blank to keep existing, type !clear! to remove)"
                password = self._ask_password(password_prompt, state.password)
                new_state = dataclasses.replace(
                    state,
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password,
                    dsn="",
                    options="",
                )
            credentials_arg = privileged_credentials if state.backend == "postgresql" else None
            try:
                result = self.controller.apply_database_settings(
                    new_state,
                    privileged_credentials=credentials_arg,
                )
                if state.backend == "postgresql" and privileged_credentials is not None:
                    self.controller.set_privileged_credentials(privileged_credentials)
                return result
            except BootstrapError as exc:
                self._print(f"Failed to connect: {exc}")
                state = dataclasses.replace(new_state)
                if state.backend == "postgresql":
                    collected = self._maybe_collect_privileged_credentials(
                        existing=privileged_credentials
                    )
                    if collected is not None:
                        privileged_credentials = collected
                        self.controller.set_privileged_credentials(privileged_credentials)
                    if privileged_credentials is not None:
                        try:
                            result = self.controller.apply_database_settings(
                                new_state,
                                privileged_credentials=privileged_credentials,
                            )
                            self.controller.set_privileged_credentials(privileged_credentials)
                            return result
                        except BootstrapError as privileged_exc:
                            self._print(
                                "Failed to connect with privileged provisioning: "
                                f"{privileged_exc}"
                            )
                if not self._confirm("Try again? [Y/n]: ", default=True):
                    raise

    def configure_vector_store(self) -> Mapping[str, object]:
        state = self.controller.state.vector_store
        available: set[str] = set()
        try:
            from modules.Tools.Base_Tools import vector_store as vector_store_module
        except Exception:  # pragma: no cover - optional dependency resolution
            vector_store_module = None

        if vector_store_module is not None:
            try:
                available.update(vector_store_module.available_vector_store_adapters())
            except Exception:  # pragma: no cover - defensive guard
                pass

        settings = self.controller.config_manager.get_vector_store_settings()
        adapters_block = settings.get("adapters") if isinstance(settings, Mapping) else None
        if isinstance(adapters_block, Mapping):
            for name in adapters_block.keys():
                if isinstance(name, str) and name.strip():
                    available.add(name.strip().lower())

        ordered_adapters = sorted(available) or ["in_memory"]
        default_adapter = state.adapter or ordered_adapters[0]

        while True:
            prompt = "Vector store adapter"
            if ordered_adapters:
                prompt += f" ({'/'.join(ordered_adapters)})"
            choice = self._ask(prompt, default_adapter)
            normalized = (choice or default_adapter or "in_memory").strip().lower()
            if ordered_adapters and normalized not in ordered_adapters:
                self._print(
                    "Please choose one of the supported adapters: " + ", ".join(ordered_adapters)
                )
                continue
            break

        new_state = dataclasses.replace(state, adapter=normalized)
        settings = self.controller.apply_vector_store_settings(new_state)
        return settings

    def configure_kv_store(self) -> Mapping[str, object]:
        state = self.controller.state.kv_store
        reuse = self._confirm(
            "Reuse the conversation database for the key-value store? [Y/n]: ",
            default=state.reuse_conversation_store,
        )
        url: str | None = None
        if not reuse:
            url = self._ask("Key-value store DSN", state.url or "") or None
        new_state = dataclasses.replace(state, reuse_conversation_store=reuse, url=url)
        return self.controller.apply_kv_store_settings(new_state)

    def configure_job_scheduling(self) -> Mapping[str, object]:
        state = self.controller.state.job_scheduling
        enabled = self._confirm("Enable durable job scheduling? [y/N]: ", default=state.enabled)
        job_store_url = state.job_store_url
        max_workers = state.max_workers
        timezone = state.timezone
        queue_size = state.queue_size

        if enabled:
            job_store_url = self._ask("Job store DSN", job_store_url or "") or None
            default_workers, default_queue_size = self._queue_defaults()
            max_workers = self._ask_required_positive_int(
                "Max worker count", max_workers or default_workers
            )
            timezone = self._ask("Scheduler timezone", timezone or "") or None
            queue_size = self._ask_required_positive_int(
                "Queue size", queue_size or default_queue_size
            )

        retry = state.retry_policy
        max_attempts = self._ask_int("Retry attempts", retry.max_attempts)
        backoff_seconds = self._ask_float("Backoff seconds", retry.backoff_seconds)
        jitter_seconds = self._ask_float("Jitter seconds", retry.jitter_seconds)
        backoff_multiplier = self._ask_float("Backoff multiplier", retry.backoff_multiplier)

        new_state = dataclasses.replace(
            state,
            enabled=enabled,
            job_store_url=job_store_url,
            max_workers=max_workers,
            timezone=timezone,
            queue_size=queue_size,
            retry_policy=dataclasses.replace(
                retry,
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
                jitter_seconds=jitter_seconds,
                backoff_multiplier=backoff_multiplier,
            ),
        )
        return self.controller.apply_job_scheduling(new_state)

    def configure_message_bus(self) -> Mapping[str, object]:
        state = self.controller.state.message_bus
        backend = self._ask(
            "Message bus backend (in_memory/redis)",
            state.backend,
        ).lower() or "in_memory"
        redis_url = state.redis_url
        stream_prefix = state.stream_prefix
        initial_offset = state.initial_offset or "$"
        replay_start = state.replay_start or initial_offset
        trim_maxlen = state.trim_maxlen
        policy_tier = state.policy_tier or "standard"
        policy_dlq_enabled = state.policy_dlq_enabled
        policy_dlq_template = state.policy_dlq_template
        policy_retention_seconds = state.policy_retention_seconds
        policy_idempotency_enabled = state.policy_idempotency_enabled
        policy_idempotency_key_field = state.policy_idempotency_key_field
        policy_idempotency_ttl_seconds = state.policy_idempotency_ttl_seconds
        kafka_enabled = state.kafka_enabled
        kafka_bootstrap_servers = state.kafka_bootstrap_servers
        kafka_topic_prefix = state.kafka_topic_prefix
        kafka_client_id = state.kafka_client_id
        kafka_driver = state.kafka_driver
        kafka_enable_idempotence = state.kafka_enable_idempotence
        kafka_acks = state.kafka_acks
        kafka_max_in_flight = state.kafka_max_in_flight
        kafka_delivery_timeout = state.kafka_delivery_timeout
        kafka_bridge_enabled = state.kafka_bridge_enabled
        kafka_bridge_topics = state.kafka_bridge_topics
        kafka_bridge_batch_size = state.kafka_bridge_batch_size
        kafka_bridge_max_attempts = state.kafka_bridge_max_attempts
        kafka_bridge_backoff_seconds = state.kafka_bridge_backoff_seconds
        kafka_bridge_dlq_topic = state.kafka_bridge_dlq_topic

        if backend == "redis":
            redis_url = self._ask("Redis URL", redis_url or "") or None
            stream_prefix = self._ask("Stream prefix", stream_prefix or "") or None
            offset_choice = self._ask("Message replay behavior (tail/replay)", "tail")
            initial_offset = self._normalize_offset_choice(offset_choice, fallback="$")
            replay_default = replay_start or initial_offset
            if replay_default == "$" and initial_offset != "$":
                replay_default = initial_offset
            replay_choice = self._ask("Replay start (tail/replay)", replay_default)
            replay_start = self._normalize_offset_choice(replay_choice, fallback=initial_offset)
            trim_maxlen = self._ask_optional_int("Stream trim maxlen (blank to disable)", trim_maxlen)
        else:
            backend = "in_memory"
            redis_url = None
            stream_prefix = None
            initial_offset = "$"
            replay_start = "$"
            trim_maxlen = None

        policy_tier = self._ask("Default policy tier", policy_tier) or "standard"
        policy_dlq_enabled = self._confirm("Enable dead-letter queue? [Y/n]: ", default=policy_dlq_enabled)
        if policy_dlq_enabled:
            policy_dlq_template = self._ask("DLQ topic template", policy_dlq_template or "dlq.{topic}") or None
        else:
            policy_dlq_template = None
        policy_retention_seconds = self._ask_optional_int(
            "Retention seconds (blank for backend default)",
            policy_retention_seconds,
        )
        policy_idempotency_enabled = self._confirm(
            "Enable idempotency hints? [y/N]: ", default=policy_idempotency_enabled
        )
        if policy_idempotency_enabled:
            policy_idempotency_key_field = (
                self._ask("Idempotency key field", policy_idempotency_key_field or "") or None
            )
            policy_idempotency_ttl_seconds = self._ask_optional_int(
                "Idempotency TTL (seconds)", policy_idempotency_ttl_seconds
            )
        else:
            policy_idempotency_key_field = None
            policy_idempotency_ttl_seconds = None

        kafka_enabled = self._confirm("Enable Kafka sink? [y/N]: ", default=kafka_enabled)
        if kafka_enabled:
            kafka_bootstrap_servers = self._ask("Kafka bootstrap servers", kafka_bootstrap_servers or "") or None
            kafka_topic_prefix = self._ask("Kafka topic prefix", kafka_topic_prefix or "atlas.bus") or "atlas.bus"
            kafka_client_id = (
                self._ask("Kafka client ID", kafka_client_id or "atlas-message-bridge") or "atlas-message-bridge"
            )
            kafka_driver = self._ask(
                "Preferred Kafka driver (confluent/kafka_python/auto)",
                kafka_driver or "",
            ) or None
            kafka_enable_idempotence = self._confirm(
                "Enable Kafka idempotence? [Y/n]: ",
                default=kafka_enable_idempotence,
            )
            kafka_acks = self._ask("Kafka acknowledgements (all/1/0)", kafka_acks or "all") or "all"
            kafka_max_in_flight = self._ask_required_positive_int(
                "Kafka max in-flight requests", kafka_max_in_flight
            )
            kafka_delivery_timeout = self._ask_float(
                "Kafka delivery timeout (seconds)",
                kafka_delivery_timeout,
            )
            kafka_bridge_enabled = self._confirm(
                "Enable Kafka bridge from Redis streams? [y/N]: ",
                default=kafka_bridge_enabled,
            )
            if kafka_bridge_enabled:
                topics_input = self._ask(
                    "Kafka bridge topics (comma-separated)",
                    ", ".join(kafka_bridge_topics),
                )
                kafka_bridge_topics = tuple(topic.strip() for topic in topics_input.split(",") if topic.strip())
                kafka_bridge_batch_size = self._ask_required_positive_int(
                    "Bridge batch size",
                    kafka_bridge_batch_size,
                )
                kafka_bridge_max_attempts = self._ask_required_positive_int(
                    "Bridge max attempts",
                    kafka_bridge_max_attempts,
                )
                kafka_bridge_backoff_seconds = self._ask_float(
                    "Bridge backoff seconds",
                    kafka_bridge_backoff_seconds,
                )
                kafka_bridge_dlq_topic = (
                    self._ask("Bridge DLQ topic", kafka_bridge_dlq_topic or "atlas.bridge.dlq") or "atlas.bridge.dlq"
                )
            else:
                kafka_bridge_topics = ()
        else:
            kafka_bootstrap_servers = None
            kafka_bridge_enabled = False
            kafka_bridge_topics = ()
        new_state = dataclasses.replace(
            state,
            backend=backend,
            redis_url=redis_url,
            stream_prefix=stream_prefix,
            initial_offset=initial_offset,
            replay_start=replay_start,
            trim_maxlen=trim_maxlen,
            policy_tier=policy_tier or "standard",
            policy_dlq_enabled=policy_dlq_enabled,
            policy_dlq_template=policy_dlq_template,
            policy_retention_seconds=policy_retention_seconds,
            policy_idempotency_enabled=policy_idempotency_enabled,
            policy_idempotency_key_field=policy_idempotency_key_field,
            policy_idempotency_ttl_seconds=policy_idempotency_ttl_seconds,
            kafka_enabled=kafka_enabled,
            kafka_bootstrap_servers=kafka_bootstrap_servers,
            kafka_topic_prefix=kafka_topic_prefix or "atlas.bus",
            kafka_client_id=kafka_client_id or "atlas-message-bridge",
            kafka_driver=kafka_driver,
            kafka_enable_idempotence=kafka_enable_idempotence,
            kafka_acks=kafka_acks or "all",
            kafka_max_in_flight=kafka_max_in_flight,
            kafka_delivery_timeout=kafka_delivery_timeout,
            kafka_bridge_enabled=kafka_bridge_enabled,
            kafka_bridge_topics=kafka_bridge_topics,
            kafka_bridge_batch_size=kafka_bridge_batch_size,
            kafka_bridge_max_attempts=kafka_bridge_max_attempts,
            kafka_bridge_backoff_seconds=kafka_bridge_backoff_seconds,
            kafka_bridge_dlq_topic=kafka_bridge_dlq_topic or "atlas.bridge.dlq",
        )
        return self.controller.apply_message_bus(new_state)

    def configure_providers(self) -> ProviderState:
        state = self.controller.state.providers
        provider_keys = self.controller.config_manager._get_provider_env_keys()
        api_keys = dict(state.api_keys)
        for provider, env_key in provider_keys.items():
            prompt = f"API key for {provider} ({env_key})"
            api_keys[provider] = self._ask(prompt, api_keys.get(provider, "")) or ""
            if not api_keys[provider]:
                api_keys.pop(provider, None)
        default_provider = self._ask("Default provider", state.default_provider or "") or None
        default_model = self._ask("Default model", state.default_model or "") or None
        new_state = dataclasses.replace(
            state,
            default_provider=default_provider,
            default_model=default_model,
            api_keys=api_keys,
        )
        return self.controller.apply_provider_settings(new_state)

    def configure_speech(self) -> SpeechState:
        state = self.controller.state.speech
        tts_enabled = self._confirm("Enable text-to-speech? [y/N]: ", default=state.tts_enabled)
        stt_enabled = self._confirm("Enable speech-to-text? [y/N]: ", default=state.stt_enabled)
        default_tts = self._ask("Default TTS provider", state.default_tts_provider or "") or None
        default_stt = self._ask("Default STT provider", state.default_stt_provider or "") or None
        elevenlabs_key = self._ask("ElevenLabs API key", state.elevenlabs_key or "") or None
        openai_key = self._ask("OpenAI speech API key", state.openai_key or "") or None
        google_credentials = self._ask("Google speech credentials path", state.google_credentials or "") or None
        new_state = dataclasses.replace(
            state,
            tts_enabled=tts_enabled,
            stt_enabled=stt_enabled,
            default_tts_provider=default_tts,
            default_stt_provider=default_stt,
            elevenlabs_key=elevenlabs_key,
            openai_key=openai_key,
            google_credentials=google_credentials,
        )
        return self.controller.apply_speech_settings(new_state)

    def configure_user(self) -> Mapping[str, object]:
        state = self.controller.state.user
        privileged_state = state.privileged_credentials
        staged_privileged_db = self.controller.get_privileged_credentials()
        while True:
            full_name = self._ask("Administrator full name", state.full_name)
            username = self._ask_required("Administrator username", state.username)
            email = self._ask_required("Administrator email", state.email)
            domain = self._ask("Administrator domain", state.domain)
            dob_input = self._ask("Administrator date of birth (YYYY-MM-DD)", state.date_of_birth)
            try:
                dob = self._normalize_date(dob_input)
            except ValueError:
                self._print("Invalid date format. Please enter as YYYY-MM-DD or MM/DD/YYYY.")
                continue
            password_prompt = "Administrator login password"
            if state.password:
                password_prompt += " (leave blank to keep existing, type !clear! to remove)"
            password = self._ask_password(password_prompt, state.password)
            confirm = self._ask_password("Confirm login password", password)
            if password != confirm:
                self._print("Passwords do not match. Try again.")
                continue

            admin_password_prompt = "Administrative password"
            if getattr(state, "admin_password", ""):
                admin_password_prompt += " (leave blank to keep existing, type !clear! to remove)"
            admin_password = self._ask_password(admin_password_prompt, getattr(state, "admin_password", ""))
            admin_confirm = self._ask_password("Confirm administrative password", admin_password)
            if admin_password != admin_confirm:
                self._print("Admin passwords do not match. Try again.")
                continue
            display_name = self._ask("Display name", state.display_name)
            sudo_username = self._ask(
                "Privileged sudo username",
                privileged_state.sudo_username,
            )
            sudo_password_prompt = "Privileged sudo password"
            if privileged_state.sudo_password:
                sudo_password_prompt += " (leave blank to keep existing, type !clear! to remove)"
            sudo_password = self._ask_password(
                sudo_password_prompt,
                privileged_state.sudo_password,
            )
            sudo_confirm = self._ask_password("Confirm privileged sudo password", sudo_password)
            if sudo_password != sudo_confirm:
                self._print("Sudo passwords do not match. Try again.")
                continue

            db_username = ""
            db_password = ""
            if staged_privileged_db is not None:
                db_username = staged_privileged_db[0] or ""
                db_password = staged_privileged_db[1] or ""

            profile = AdminProfile(
                username=username,
                email=email,
                password=password,
                admin_password=admin_password,
                display_name=display_name,
                full_name=full_name,
                domain=domain,
                date_of_birth=dob,
                sudo_username=sudo_username,
                sudo_password=sudo_password,
                privileged_db_username=db_username,
                privileged_db_password=db_password,
            )
            staged_state = self.controller.set_user_profile(profile)
            return dataclasses.asdict(staged_state)

    def configure_optional_settings(self) -> OptionalState:
        state = self.controller.state.optional
        tenant_id = self._ask("Tenant ID", state.tenant_id or "") or None
        retention_days = self._ask_optional_int("Conversation retention days", state.retention_days)
        retention_history = self._ask_optional_int(
            "Conversation history limit", state.retention_history_limit
        )
        scheduler_timezone = self._ask("Scheduler timezone", state.scheduler_timezone or "") or None
        if self.controller.state.setup_type.mode == "enterprise":
            _, default_queue_size = self._queue_defaults()
            scheduler_queue_size = self._ask_required_positive_int(
                "Scheduler queue size", state.scheduler_queue_size or default_queue_size
            )
        else:
            scheduler_queue_size = self._ask_optional_int(
                "Scheduler queue size", state.scheduler_queue_size
            )
        http_auto_start = self._confirm("Auto-start HTTP server? [y/N]: ", default=state.http_auto_start)
        new_state = dataclasses.replace(
            state,
            tenant_id=tenant_id,
            retention_days=retention_days,
            retention_history_limit=retention_history,
            scheduler_timezone=scheduler_timezone,
            scheduler_queue_size=scheduler_queue_size,
            http_auto_start=http_auto_start,
        )
        self.controller.apply_company_identity(new_state)
        return self.controller.apply_optional_settings(new_state)

    def finalize(self, summary: Mapping[str, object]) -> Path:
        """Persist the setup sentinel with the collected summary."""

        payload = {"setup_complete": True, "summary": summary}
        return write_setup_marker(payload)

    # -- helper utilities ------------------------------------------------

    def _confirm(self, prompt: str, *, default: bool) -> bool:
        self._ensure_prompts_enabled()
        value = self._ask(prompt, "y" if default else "n")
        normalized = value.strip().lower()
        if not normalized:
            return default
        return normalized in {"y", "yes"}

    def _normalize_offset_choice(self, value: str | None, *, fallback: str) -> str:
        candidate = (value or "").strip()
        if not candidate:
            return fallback
        lowered = candidate.lower()
        if lowered in {"tail", "$"}:
            return "$"
        if lowered in {"replay", "from_start", "from-start", "all"}:
            return "0-0"
        if re.match(r"^\d+-\d+$", candidate):
            return candidate
        return fallback

    def _ask(self, prompt: str, default: str | int | float | None) -> str:
        self._ensure_prompts_enabled()
        if default not in (None, ""):
            text = f"{prompt} [{default}]: "
        else:
            text = f"{prompt}: "
        response = self._input(text)
        if response.strip():
            return response.strip()
        return str(default) if default not in (None, "") else ""

    def _ask_required(self, prompt: str, default: str = "") -> str:
        self._ensure_prompts_enabled()
        while True:
            value = self._ask(prompt, default)
            if value.strip():
                return value.strip()
            self._print("A value is required.")

    def _ask_password(self, prompt: str, default: str) -> str:
        self._ensure_prompts_enabled()
        value = self._getpass(f"{prompt}: ")
        normalized = value.strip()
        if not normalized:
            return default
        if normalized == "!clear!":
            return ""
        return normalized

    def _normalize_date(self, raw: str) -> str:
        if not raw:
            return ""
        value = raw.strip()
        if not value:
            return ""
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
            try:
                parsed = datetime.strptime(value, fmt).date()
            except ValueError:
                continue
            return parsed.isoformat()
        raise ValueError(f"Invalid date: {raw}")

    def _maybe_collect_privileged_credentials(
        self,
        *,
        existing: tuple[str | None, str | None] | None = None,
    ) -> tuple[str | None, str | None] | None:
        if existing is None:
            wants_credentials = self._confirm(
                "Provide privileged PostgreSQL credentials for automatic provisioning? [Y/n]: ",
                default=True,
            )
            if not wants_credentials:
                return None
            username_default = ""
            password_default = ""
        else:
            wants_credentials = self._confirm(
                "Update privileged PostgreSQL credentials? [y/N]: ",
                default=False,
            )
            if not wants_credentials:
                return None
            username_default = existing[0] or ""
            password_default = existing[1] or ""

        username = self._ask("Privileged PostgreSQL user", username_default).strip()
        if not username:
            self._print("Skipping privileged provisioning; no username provided.")
            return None

        password_prompt = "Privileged user password"
        if password_default:
            password_prompt += " (leave blank to keep existing, type !clear! to remove)"
        password = self._ask_password(password_prompt, password_default)
        return username, password

    def _ask_int(self, prompt: str, default: int) -> int:
        while True:
            value = self._ask(prompt, default)
            try:
                return int(value)
            except (TypeError, ValueError):
                self._print("Enter a valid integer.")

    def _ask_optional_int(self, prompt: str, default: int | None) -> int | None:
        value = self._ask(prompt, default if default is not None else "")
        if not value.strip():
            return None
        try:
            return int(value)
        except ValueError:
            self._print("Enter a valid integer or leave blank.")
            return self._ask_optional_int(prompt, default)

    def _ask_required_positive_int(self, prompt: str, default: int) -> int:
        while True:
            try:
                value = int(self._ask(prompt, default))
            except (TypeError, ValueError):
                self._print("Enter a valid integer.")
                continue
            if value <= 0:
                self._print("Enter a positive integer.")
                continue
            return value

    def _ask_float(self, prompt: str, default: float) -> float:
        while True:
            value = self._ask(prompt, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                self._print("Enter a valid number.")

    def _postgres_commands(self, system: str) -> list[list[str]]:
        if system == "Linux":
            return [["sudo", "apt-get", "update"], ["sudo", "apt-get", "install", "-y", "postgresql", "postgresql-contrib"]]
        if system == "Darwin":
            return [["brew", "update"], ["brew", "install", "postgresql"]]
        if system == "Windows":
            return [[
                "powershell",
                "-NoProfile",
                "-Command",
                "choco install postgresql --yes",
            ]]
        return []


__all__ = ["SetupUtility"]
