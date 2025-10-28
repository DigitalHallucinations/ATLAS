"""Command-line setup utility for ATLAS."""

from __future__ import annotations

import dataclasses
import getpass
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping

from ATLAS.setup.controller import (
    DatabaseState,
    JobSchedulingState,
    KvStoreState,
    MessageBusState,
    OptionalState,
    ProviderState,
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
    ) -> None:
        self.controller = controller or SetupWizardController()
        self._input = input_func
        self._getpass = getpass_func
        self._print = print_func
        self._run = run
        self._platform_system = platform_system

    # -- public API -----------------------------------------------------

    def run(self) -> Path:
        """Execute the full setup workflow."""

        self.install_postgresql()

        dsn = self.configure_database()
        self._print(f"Conversation store DSN saved: {dsn}")

        self.configure_kv_store()
        self.configure_job_scheduling()
        self.configure_message_bus()
        self.configure_providers()
        self.configure_speech()
        self.configure_user()
        self.configure_optional_settings()

        summary = self.controller.build_summary()
        marker_path = self.finalize(summary)
        self._print(f"Setup complete. Sentinel written to {marker_path}")
        return marker_path

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

        sudo_password: str | None = None
        if any(cmd and cmd[0] == "sudo" for cmd in commands):
            entered = self._getpass("Enter sudo password (leave blank to skip automation): ")
            sudo_password = entered if entered.strip() else None

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
        privileged_credentials: tuple[str | None, str | None] | None = None
        while True:
            host = self._ask("PostgreSQL host", state.host)
            port = self._ask_int("PostgreSQL port", state.port)
            database = self._ask("Database name", state.database)
            user = self._ask("Database user", state.user)
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
                return self.controller.apply_database_settings(
                    new_state,
                    privileged_credentials=privileged_credentials,
                )
            except BootstrapError as exc:
                self._print(f"Failed to connect: {exc}")
                collected = self._maybe_collect_privileged_credentials(
                    existing=privileged_credentials
                )
                if collected is not None:
                    privileged_credentials = collected
                if privileged_credentials is not None:
                    try:
                        return self.controller.apply_database_settings(
                            new_state,
                            privileged_credentials=privileged_credentials,
                        )
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
        while True:
            username = self._ask_required("Administrator username", state.username)
            email = self._ask_required("Administrator email", state.email)
            password = self._ask_password("Administrator password", "")
            confirm = self._ask_password("Confirm password", "")
            if password != confirm:
                self._print("Passwords do not match. Try again.")
                continue
            display_name = self._ask("Display name", state.display_name)
            new_state = dataclasses.replace(
                state,
                username=username,
                email=email,
                password=password,
                display_name=display_name,
            )
            return self.controller.register_user(new_state)

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
        value = self._ask(prompt, "y" if default else "n")
        normalized = value.strip().lower()
        if not normalized:
            return default
        return normalized in {"y", "yes"}

    def _ask(self, prompt: str, default: str | int | float | None) -> str:
        if default not in (None, ""):
            text = f"{prompt} [{default}]: "
        else:
            text = f"{prompt}: "
        response = self._input(text)
        if response.strip():
            return response.strip()
        return str(default) if default not in (None, "") else ""

    def _ask_required(self, prompt: str, default: str = "") -> str:
        while True:
            value = self._ask(prompt, default)
            if value.strip():
                return value.strip()
            self._print("A value is required.")

    def _ask_password(self, prompt: str, default: str) -> str:
        value = self._getpass(f"{prompt}: ")
        normalized = value.strip()
        if not normalized:
            return default
        if normalized == "!clear!":
            return ""
        return normalized

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
