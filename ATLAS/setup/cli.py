"""Command-line setup utility for ATLAS."""

from __future__ import annotations

import dataclasses
import getpass
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

from ATLAS.setup.controller import (
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
)
from ATLAS.setup_marker import write_setup_marker
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

        self.controller.apply_kv_store_settings(self.controller.state.kv_store)
        self.controller.apply_job_scheduling(self.controller.state.job_scheduling)
        self.controller.apply_message_bus(self.controller.state.message_bus)
        self.controller.apply_provider_settings(self.controller.state.providers)
        self.controller.apply_speech_settings(self.controller.state.speech)
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
            choice = self._ask("Setup type (personal/enterprise)", "personal")
            normalized = choice.strip().lower() or "personal"
            if normalized in {"personal", "enterprise"}:
                return self.controller.apply_setup_type(normalized)
            self._print("Please choose either 'personal' or 'enterprise'.")

    def _apply_state_overrides_from_env(self) -> None:
        self._apply_provider_env_overrides()
        self._apply_optional_env_overrides()
        self._apply_job_env_overrides()
        self._apply_message_bus_env_overrides()
        self._apply_kv_env_overrides()
        self._apply_speech_env_overrides()

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
            if normalized in {"in_memory", "redis"}:
                backend = normalized

        redis_override = self._env_get("ATLAS_REDIS_URL")
        stream_override = self._env_get("ATLAS_STREAM_PREFIX")

        redis_url = redis_override if redis_override is not None else state.redis_url
        if redis_url == "":
            redis_url = None

        stream_prefix = stream_override if stream_override is not None else state.stream_prefix
        if stream_prefix == "":
            stream_prefix = None

        if backend != "redis":
            redis_url = None
            stream_prefix = None

        self.controller.state.message_bus = dataclasses.replace(
            state,
            backend=backend,
            redis_url=redis_url,
            stream_prefix=stream_prefix,
        )

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

    # -- configuration collection --------------------------------------

    def configure_database(self) -> str:
        """Prompt for PostgreSQL connection settings and persist them."""

        state = self.controller.state.database
        privileged_credentials = self.controller.get_privileged_credentials()
        user_state = self.controller.state.user
        default_host = state.host
        user_domain = user_state.domain.strip()
        if default_host in {"", "localhost"} and user_domain:
            default_host = user_domain
        default_user = state.user or user_state.username or state.user
        while True:
            host = self._ask("PostgreSQL host", default_host)
            port = self._ask_int("PostgreSQL port", state.port)
            database = self._ask("Database name", state.database)
            user = self._ask("Database user", default_user)
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
            )
            try:
                result = self.controller.apply_database_settings(
                    new_state,
                    privileged_credentials=privileged_credentials,
                )
                if privileged_credentials is not None:
                    self.controller.set_privileged_credentials(privileged_credentials)
                return result
            except BootstrapError as exc:
                self._print(f"Failed to connect: {exc}")
                state = dataclasses.replace(new_state)
                default_host = host
                default_user = user
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
                        self._print(f"Failed to connect with privileged provisioning: {privileged_exc}")
                if not self._confirm("Try again? [Y/n]: ", default=True):
                    raise

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
            max_workers = self._ask_optional_int("Max worker count", max_workers)
            timezone = self._ask("Scheduler timezone", timezone or "") or None
            queue_size = self._ask_optional_int("Queue size", queue_size)

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
        if backend == "redis":
            redis_url = self._ask("Redis URL", redis_url or "") or None
            stream_prefix = self._ask("Stream prefix", stream_prefix or "") or None
        else:
            backend = "in_memory"
            redis_url = None
            stream_prefix = None
        new_state = dataclasses.replace(state, backend=backend, redis_url=redis_url, stream_prefix=stream_prefix)
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
            password_prompt = "Administrator password"
            if state.password:
                password_prompt += " (leave blank to keep existing, type !clear! to remove)"
            password = self._ask_password(password_prompt, state.password)
            confirm = self._ask_password("Confirm password", password)
            if password != confirm:
                self._print("Passwords do not match. Try again.")
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
        scheduler_queue_size = self._ask_optional_int("Scheduler queue size", state.scheduler_queue_size)
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
