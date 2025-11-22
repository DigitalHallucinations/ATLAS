"""Async system readiness checks for the GTK setup wizard."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import gi

gi.require_version("Gio", "2.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gio, GLib

from ATLAS.setup.controller import (
    ConfigManager,
    DatabaseState,
    MessageBusState,
    _compose_dsn,
)


PasswordProvider = Callable[[], str | None]
ResultCallback = Callable[["PreflightCheckResult"], None]
CompleteCallback = Callable[[list["PreflightCheckResult"]], None]

DATABASE_LOCAL_TIP = (
    "SQLite keeps data on this device and avoids service management on low-resource hosts."
)
DATABASE_LOCAL_PG_TIP = (
    "Local PostgreSQL stays fastest when you have CPU/RAM to spare and want everything on-box."
)
DATABASE_MANAGED_TIP = (
    "Managed Postgres or Atlas works well when collaborators join and cloud latency is acceptable."
)
VECTOR_HOSTING_TIP = (
    "Keep vector DBs local for trusted, offline work; lean on managed options when scaling ingestion."
)
MODEL_HOSTING_TIP = (
    "Run models locally for offline or low-latency paths when hardware allows; otherwise use hosted inference."
)


@dataclass(frozen=True)
class PreflightCheckDefinition:
    """Describe how to validate and optionally repair a dependency."""

    identifier: str
    label: str
    command: Sequence[str]
    success_message: str
    failure_hint: str
    fix_command: Sequence[str] | None = None
    fix_label: str | None = None
    fix_tooltip: str | None = None
    fix_available: bool = True
    requires_sudo: bool = False
    process_output: Callable[[bool, str, str, int], tuple[str, str | None]] | None = None


@dataclass
class PreflightCheckResult:
    """Outcome of executing a :class:`PreflightCheckDefinition`."""

    identifier: str
    label: str
    passed: bool
    message: str
    fix_label: str | None
    fix_available: bool = True
    fix_tooltip: str | None = None
    recommendation: str | None = None


def _read_mem_total() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().total)
    except Exception:
        pass

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    return float(parts[1]) * 1024
    except Exception:
        return None
    return None


def _resolve_path(value: str, *, app_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (app_root / candidate).resolve(strict=False)
    else:
        candidate = candidate.resolve(strict=False)
    return candidate


def _storage_paths_from_config() -> dict[str, str]:
    try:
        manager = ConfigManager()
    except Exception:
        return {}

    app_root = manager.get_config("APP_ROOT", ".")
    app_root_path = Path(app_root).expanduser()
    if not app_root_path.is_absolute():
        app_root_path = (Path.cwd() / app_root_path).resolve(strict=False)

    storage_paths: dict[str, str] = {}

    try:
        conversation_config = manager.get_conversation_database_config()
    except Exception:
        conversation_config = {}
    backend = (conversation_config.get("backend") or "").strip().lower()
    url = str(conversation_config.get("url") or "").strip()
    if backend == "sqlite" or url.startswith("sqlite"):
        db_path: str | None = None
        try:  # pragma: no cover - sqlalchemy import may fail on minimal installs
            from sqlalchemy.engine.url import make_url

            parsed = make_url(url)
            db_path = parsed.database
        except Exception:
            pass

        if not db_path and url.startswith("sqlite:///"):
            db_path = url.split("sqlite:///")[-1]

        if db_path:
            storage_paths["conversation_database"] = str(
                _resolve_path(db_path, app_root=app_root_path)
            )

    vector_settings = manager.get_vector_store_settings()
    adapter = (vector_settings.get("default_adapter") or "").strip().lower()
    adapters_block = vector_settings.get("adapters")
    adapter_config: Mapping[str, object] | None = None
    if isinstance(adapters_block, Mapping):
        candidate = adapters_block.get(adapter)
        if isinstance(candidate, Mapping):
            adapter_config = candidate

    vector_path = None
    if adapter_config is not None:
        if adapter == "chroma":
            vector_path = adapter_config.get("persist_directory")
        elif adapter == "faiss":
            vector_path = adapter_config.get("index_path")

    if vector_path:
        storage_paths["vector_store"] = str(
            _resolve_path(str(vector_path), app_root=app_root_path)
        )

    model_cache_dir = manager.get_config("MODEL_CACHE_DIR")
    if model_cache_dir:
        storage_paths["model_cache"] = str(
            _resolve_path(str(model_cache_dir), app_root=app_root_path)
        )

    return storage_paths


def _preferred_disk_free(paths: Mapping[str, str]) -> tuple[float | None, str | None]:
    lowest_free: float | None = None
    chosen_path: str | None = None

    for path in paths.values():
        expanded = Path(path).expanduser()
        free_bytes: float | None = None
        for candidate in (expanded, expanded.resolve(strict=False).parent):
            try:
                free_bytes = float(shutil.disk_usage(candidate).free)
                break
            except Exception:
                continue

        if free_bytes is None:
            continue

        resolved = str(expanded.resolve(strict=False))
        if lowest_free is None or free_bytes < lowest_free:
            lowest_free = free_bytes
            chosen_path = resolved

    return lowest_free, chosen_path


def _host_profile() -> dict[str, float | int | None]:
    cpu_count = os.cpu_count() or 0
    ram_bytes = _read_mem_total()
    storage_paths = _storage_paths_from_config()
    disk_bytes, disk_path = _preferred_disk_free(storage_paths)

    if disk_bytes is None:
        try:
            disk_bytes = float(shutil.disk_usage(".").free)
            disk_path = str(Path.cwd())
        except Exception:
            disk_bytes = None
            disk_path = None

    to_gb = lambda value: None if value is None else value / (1024 ** 3)
    return {
        "cpu_count": cpu_count,
        "ram_gb": to_gb(ram_bytes),
        "disk_gb": to_gb(disk_bytes),
        "disk_path": disk_path,
    }


def database_recommendation_for_state(
    state: DatabaseState | None, host_profile: dict[str, float | int | None] | None = None
) -> str:
    profile = host_profile or _host_profile()
    ram_gb = profile.get("ram_gb")
    disk_gb = profile.get("disk_gb")
    low_resources = bool(
        (ram_gb is not None and ram_gb < 8)
        or (disk_gb is not None and disk_gb < 40)
    )

    backend = (state.backend if state else "postgresql") or "postgresql"
    normalized = backend.strip().lower() or "postgresql"
    host = (state.host if state else "") or ""
    local_host = not host or host in {"localhost", "127.0.0.1"}

    if normalized == "sqlite":
        return DATABASE_LOCAL_TIP

    if normalized == "mongodb":
        if local_host and low_resources:
            return f"{DATABASE_LOCAL_TIP} {DATABASE_MANAGED_TIP}"
        return DATABASE_MANAGED_TIP

    if local_host and low_resources:
        return f"{DATABASE_LOCAL_TIP} {DATABASE_MANAGED_TIP}"
    if local_host:
        return DATABASE_LOCAL_PG_TIP
    return DATABASE_MANAGED_TIP


class PreflightHelper:
    """Run preflight dependency checks and optional repair commands."""

    _STDIO_FLAGS = Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_PIPE

    def __init__(
        self,
        *,
        request_password: PasswordProvider,
        checks: Iterable[PreflightCheckDefinition] | None = None,
        subprocess_factory: Callable[[Sequence[str], Gio.SubprocessFlags], Gio.Subprocess] | None = None,
    ) -> None:
        self._request_password = request_password
        self._database_state: DatabaseState = DatabaseState()
        self._message_bus_backend: str | None = None
        self._redis_url: str | None = None
        self._provided_checks = list(checks) if checks is not None else None
        self._checks: list[PreflightCheckDefinition] = (
            list(self._provided_checks)
            if self._provided_checks is not None
            else list(self._default_checks())
        )
        self._subprocess_factory = subprocess_factory or self._spawn_subprocess

        self._pending: list[PreflightCheckDefinition] = []
        self._results: dict[str, PreflightCheckResult] = {}
        self._running: bool = False
        self._fix_in_progress: str | None = None
        self._on_update: ResultCallback | None = None
        self._on_complete: CompleteCallback | None = None

    # -- public API -----------------------------------------------------

    def configure_database_target(self, state: DatabaseState | None) -> None:
        """Adjust the database check to match *state* for the next run."""

        if state is None:
            self._database_state = DatabaseState()
        else:
            self._database_state = dataclasses.replace(state)

    def configure_message_bus_target(self, state: MessageBusState | None) -> None:
        """Adjust the Redis check to match *state* for the next run."""

        if state is None:
            self._message_bus_backend = None
            self._redis_url = None
            return

        backend = (state.backend or "").strip().lower() or None
        self._message_bus_backend = backend
        if backend != "redis":
            self._redis_url = None
            return

        url = (state.redis_url or "").strip() or None
        self._redis_url = url

    def run_checks(
        self,
        *,
        on_update: ResultCallback | None = None,
        on_complete: CompleteCallback | None = None,
    ) -> None:
        """Execute all known checks sequentially."""

        if self._running:
            raise RuntimeError("Preflight checks are already running")

        self._on_update = on_update
        self._on_complete = on_complete
        self._results.clear()
        if self._provided_checks is not None:
            self._checks = list(self._provided_checks)
        else:
            self._checks = list(self._default_checks())
        self._pending = list(self._checks)
        self._running = True
        self._advance()

    def run_fix(self, identifier: str, callback: ResultCallback) -> None:
        """Attempt the fix command for *identifier* and re-run the check."""

        if self._running or self._fix_in_progress is not None:
            raise RuntimeError("Another preflight operation is already running")

        definition = self._definition_for(identifier)
        if (
            definition is None
            or definition.fix_command is None
            or not definition.fix_available
        ):
            result = self._results.get(identifier)
            if result is None:
                result = PreflightCheckResult(
                    identifier=identifier,
                    label=definition.label if definition else identifier,
                    passed=False,
                    message="No automated fix is available for this check.",
                    fix_label=definition.fix_label if definition else None,
                    fix_available=False,
                    fix_tooltip=definition.fix_tooltip if definition else None,
                    recommendation=None,
                )
            GLib.idle_add(lambda: callback(result))
            return

        password: str | None = None
        if definition.requires_sudo:
            password = (self._request_password() or "").strip()
            if not password:
                message = "Sudo password was not provided. Please retry with valid credentials."
                result = PreflightCheckResult(
                    identifier=identifier,
                    label=definition.label,
                    passed=False,
                    message=message,
                    fix_label=definition.fix_label,
                    fix_available=definition.fix_available and definition.fix_command is not None,
                    fix_tooltip=definition.fix_tooltip,
                    recommendation=None,
                )
                GLib.idle_add(lambda: callback(result))
                return

        self._fix_in_progress = identifier

        def _after_fix(result: PreflightCheckResult) -> None:
            self._fix_in_progress = None
            callback(result)

        def _after_fix_command(result: PreflightCheckResult) -> None:
            if not result.passed:
                self._emit_fix_failure(definition, result.message, _after_fix)
                return
            self._rerun_after_fix(definition, _after_fix)

        failure_hint = (
            f"{definition.fix_label or 'Fix command'} did not complete successfully."
        )

        self._run_command(
            definition,
            definition.fix_command,
            success_message="Fix command completed.",
            failure_hint=failure_hint,
            password=password,
            store_result=False,
            on_finished=_after_fix_command,
            on_error=lambda message: self._emit_fix_failure(definition, message, _after_fix),
        )

    # -- private helpers ------------------------------------------------

    def _definition_for(self, identifier: str) -> PreflightCheckDefinition | None:
        for check in self._checks:
            if check.identifier == identifier:
                return check
        return None

    def _advance(self) -> None:
        if not self._pending:
            self._running = False
            if self._on_complete is not None:
                ordered = [self._results.get(check.identifier) for check in self._checks]
                callback = self._on_complete
                self._on_complete = None
                callback([result for result in ordered if result is not None])
            return

        current = self._pending.pop(0)
        self._run_command(
            current,
            current.command,
            success_message=current.success_message,
            failure_hint=current.failure_hint,
            password=None,
            store_result=True,
            on_finished=lambda _result: self._advance(),
            on_error=lambda message: self._handle_error(current, message),
        )

    def _handle_error(self, definition: PreflightCheckDefinition, message: str) -> None:
        result = PreflightCheckResult(
            identifier=definition.identifier,
            label=definition.label,
            passed=False,
            message=message,
            fix_label=definition.fix_label,
            fix_available=definition.fix_available and definition.fix_command is not None,
            fix_tooltip=definition.fix_tooltip,
            recommendation=None,
        )
        self._store_and_emit(result)
        self._advance()

    def _rerun_after_fix(
        self,
        definition: PreflightCheckDefinition,
        callback: Callable[[PreflightCheckResult], None],
    ) -> None:
        self._run_command(
            definition,
            definition.command,
            success_message=definition.success_message,
            failure_hint=definition.failure_hint,
            password=None,
            store_result=True,
            on_finished=lambda result: callback(result),
            on_error=lambda message: callback(
                PreflightCheckResult(
                    identifier=definition.identifier,
                    label=definition.label,
                    passed=False,
                    message=message,
                    fix_label=definition.fix_label,
                    fix_available=definition.fix_available and definition.fix_command is not None,
                    fix_tooltip=definition.fix_tooltip,
                    recommendation=None,
                )
            ),
        )

    def _emit_fix_failure(
        self,
        definition: PreflightCheckDefinition,
        message: str,
        callback: Callable[[PreflightCheckResult], None],
    ) -> None:
        callback(
            PreflightCheckResult(
                identifier=definition.identifier,
                label=definition.label,
                passed=False,
                message=message,
                fix_label=definition.fix_label,
                fix_available=definition.fix_available and definition.fix_command is not None,
                fix_tooltip=definition.fix_tooltip,
                recommendation=None,
            )
        )

    def _run_command(
        self,
        definition: PreflightCheckDefinition,
        command: Sequence[str],
        *,
        success_message: str,
        failure_hint: str,
        password: str | None,
        store_result: bool,
        on_finished: Callable[[PreflightCheckResult], None],
        on_error: Callable[[str], None],
    ) -> None:
        try:
            subprocess = self._subprocess_factory(
                command,
                self._STDIO_FLAGS | (Gio.SubprocessFlags.STDIN_PIPE if password else 0),
            )
        except Exception as exc:  # pragma: no cover - defensive
            on_error(self._format_spawn_error(command, exc))
            return

        def _complete(_subprocess: Gio.Subprocess, task: Gio.AsyncResult) -> None:
            try:
                ok, stdout, stderr = subprocess.communicate_utf8_finish(task)
            except Exception as exc:  # pragma: no cover - defensive
                on_error(self._format_spawn_error(command, exc))
                return

            exit_status = subprocess.get_exit_status()
            passed = bool(ok and exit_status == 0)
            message = (
                success_message
                if passed
                else self._format_failure_message(failure_hint, exit_status, stdout, stderr)
            )
            recommendation: str | None = None
            if definition.process_output is not None:
                try:
                    message, recommendation = definition.process_output(
                        passed, stdout or "", stderr or "", exit_status
                    )
                except Exception:
                    pass

            result = PreflightCheckResult(
                identifier=definition.identifier,
                label=definition.label,
                passed=passed,
                message=message,
                fix_label=definition.fix_label,
                fix_available=definition.fix_available and definition.fix_command is not None,
                fix_tooltip=definition.fix_tooltip,
                recommendation=recommendation,
            )
            if store_result:
                self._store_and_emit(result)
            on_finished(result)

        stdin_data = None if password is None else f"{password}\n"
        try:
            subprocess.communicate_utf8_async(stdin_data, None, _complete)
        except Exception as exc:  # pragma: no cover - defensive
            on_error(self._format_spawn_error(command, exc))

    def _store_and_emit(self, result: PreflightCheckResult) -> None:
        self._results[result.identifier] = result
        if self._on_update is not None:
            self._on_update(result)

    def _spawn_subprocess(
        self, command: Sequence[str], flags: Gio.SubprocessFlags
    ) -> Gio.Subprocess:
        return Gio.Subprocess.new(list(command), flags)

    # -- message helpers ------------------------------------------------

    def _format_failure_message(
        self,
        hint: str,
        exit_status: int,
        stdout: str,
        stderr: str,
    ) -> str:
        details: list[str] = [hint.strip()]
        if stdout:
            details.append(stdout.strip())
        if stderr:
            details.append(stderr.strip())
        details.append(f"Exit status: {exit_status}")
        return "\n".join(part for part in details if part)

    def _format_spawn_error(self, command: Sequence[str], exc: Exception) -> str:
        return f"Unable to execute {' '.join(command)}: {exc}"

    # -- platform helpers -----------------------------------------------

    def _detect_service_manager(self) -> tuple[str | None, str | None]:
        """Identify the service manager for restart automation."""

        if sys.platform == "darwin":
            if shutil.which("brew"):
                return "brew", None
            return None, (
                "Automatic restarts require Homebrew. Run `brew services start <service>` manually."
            )

        if sys.platform.startswith("linux"):
            if shutil.which("systemctl") and os.path.isdir("/run/systemd/system"):
                return "systemctl", None
            if shutil.which("service"):
                return "service", "Using 'service' because systemd is unavailable."
            return None, "No supported init system detected; restart services manually."

        return None, "Automatic service management is unavailable on this platform."

    def _service_fix_details(
        self,
        *,
        display_name: str,
        base_hint: str,
        system_service: str,
        brew_service: str | None = None,
    ) -> tuple[Sequence[str] | None, str | None, bool, str | None, str, bool]:
        """Choose a service restart strategy for the host platform."""

        manager, platform_hint = self._detect_service_manager()
        requires_sudo = True
        fix_tooltip = None

        if manager == "systemctl":
            fix_command = [
                "/usr/bin/env",
                "sudo",
                "-S",
                "systemctl",
                "start",
                system_service,
            ]
            fix_label = f"Start {display_name} service"
        elif manager == "service":
            fix_command = [
                "/usr/bin/env",
                "sudo",
                "-S",
                "service",
                system_service,
                "start",
            ]
            fix_label = f"Start {display_name} service"
            fix_tooltip = platform_hint
        elif manager == "brew":
            fix_command = [
                "/usr/bin/env",
                "brew",
                "services",
                "start",
                brew_service or system_service,
            ]
            fix_label = f"Start {display_name} with brew services"
            requires_sudo = False
            fix_tooltip = "Using Homebrew services for restart automation."
        else:
            fix_command = None
            fix_label = f"Start {display_name} manually"
            fix_tooltip = platform_hint
            base_hint = f"{base_hint} {platform_hint}" if platform_hint else base_hint

        fix_available = bool(fix_command)
        return fix_command, fix_label, requires_sudo, fix_tooltip, base_hint, fix_available

    # -- check catalog --------------------------------------------------

    def _default_checks(self) -> Iterable[PreflightCheckDefinition]:
        checks: list[PreflightCheckDefinition] = []
        database_check = self._build_database_check()
        if database_check is not None:
            checks.append(database_check)
        redis_check = self._build_redis_check()
        if redis_check is not None:
            checks.append(redis_check)
        checks.append(self._build_hardware_check())
        return checks

    def _build_database_check(self) -> PreflightCheckDefinition | None:
        backend = (self._database_state.backend or "postgresql").strip().lower() or "postgresql"
        host_profile = _host_profile()
        if backend == "sqlite":
            return None
        if backend == "mongodb":
            uri = _compose_dsn(self._database_state).strip()
            if not uri:
                uri = "mongodb://localhost:27017/atlas"
            mongo_hint = (
                "MongoDB is unreachable. Verify the URI, credentials, and network access,"
                " including Atlas SRV records."
            )
            recommendation = database_recommendation_for_state(
                self._database_state, host_profile=host_profile
            )
            script = (
                "import sys\n"
                "uri = sys.argv[1]\n"
                "try:\n"
                "    from pymongo import MongoClient\n"
                "except Exception as exc:\n"
                "    print(f'pymongo import failed: {exc}', file=sys.stderr)\n"
                "    sys.exit(1)\n"
                "client = MongoClient(uri, serverSelectionTimeoutMS=5000)\n"
                "try:\n"
                "    client.admin.command('ping')\n"
                "except Exception as exc:\n"
                "    print(f'mongodb ping failed: {exc}', file=sys.stderr)\n"
                "    sys.exit(1)\n"
                "finally:\n"
                "    client.close()\n"
            )

            def _parse_mongodb(
                passed: bool, stdout: str, stderr: str, exit_status: int
            ) -> tuple[str, str | None]:
                message = (
                    "MongoDB connection succeeded."
                    if passed
                    else self._format_failure_message(mongo_hint, exit_status, stdout, stderr)
                )
                return message, recommendation

            return PreflightCheckDefinition(
                identifier="mongodb",
                label="MongoDB",
                command=["/usr/bin/env", "python3", "-c", script, uri],
                success_message="MongoDB connection succeeded.",
                failure_hint=mongo_hint,
                process_output=_parse_mongodb,
            )

        pg_hint = (
            "PostgreSQL is unreachable. Ensure the server is installed and that pg_isready"
            " can connect to the configured instance."
        )
        (
            fix_command,
            fix_label,
            requires_sudo,
            fix_tooltip,
            pg_hint,
            fix_available,
        ) = self._service_fix_details(
            display_name="PostgreSQL",
            base_hint=pg_hint,
            system_service="postgresql",
        )
        host = (self._database_state.host or "").strip()
        port = int(self._database_state.port or 0)
        database = (self._database_state.database or "").strip()
        recommendation = database_recommendation_for_state(
            self._database_state, host_profile=host_profile
        )
        command: list[str] = ["/usr/bin/env", "pg_isready", "-q"]
        if host:
            command.extend(["-h", host])
        if port:
            command.extend(["-p", str(port)])
        if database:
            command.extend(["-d", database])

        def _parse_postgres(
            passed: bool, stdout: str, stderr: str, exit_status: int
        ) -> tuple[str, str | None]:
            message = (
                "PostgreSQL is accepting connections."
                if passed
                else self._format_failure_message(pg_hint, exit_status, stdout, stderr)
            )
            return message, recommendation

        return PreflightCheckDefinition(
            identifier="postgresql",
            label="PostgreSQL",
            command=command,
            success_message="PostgreSQL is accepting connections.",
            failure_hint=pg_hint,
            fix_command=fix_command,
            fix_label=fix_label,
            fix_tooltip=fix_tooltip,
            fix_available=fix_available,
            requires_sudo=requires_sudo,
            process_output=_parse_postgres,
        )

    def _build_redis_check(self) -> PreflightCheckDefinition | None:
        if self._message_bus_backend != "redis":
            return None

        redis_hint = (
            "Redis did not respond to ping. Verify the redis-server service is installed"
            " and running."
        )
        command: list[str] = ["/usr/bin/env", "redis-cli"]
        if self._redis_url:
            command.extend(["-u", self._redis_url])
        command.append("ping")
        (
            fix_command,
            fix_label,
            requires_sudo,
            fix_tooltip,
            redis_hint,
            fix_available,
        ) = self._service_fix_details(
            display_name="Redis",
            base_hint=redis_hint,
            system_service="redis-server",
            brew_service="redis",
        )
        return PreflightCheckDefinition(
            identifier="redis",
            label="Redis",
            command=command,
            success_message="Redis responded to ping.",
            failure_hint=redis_hint,
            fix_command=fix_command,
            fix_label=fix_label,
            fix_tooltip=fix_tooltip,
            fix_available=fix_available,
            requires_sudo=requires_sudo,
        )

    def _build_hardware_check(self) -> PreflightCheckDefinition:
        storage_paths = _storage_paths_from_config()
        storage_paths_json = json.dumps(storage_paths)
        probe_script = textwrap.dedent(
            """
            import json, os, shutil, subprocess, sys

            def _read_mem_total():
                try:
                    import psutil

                    return float(psutil.virtual_memory().total)
                except Exception:
                    pass
                try:
                    with open('/proc/meminfo', 'r', encoding='utf-8') as fh:
                        for line in fh:
                            if line.startswith('MemTotal:'):
                                parts = line.split()
                                return float(parts[1]) * 1024
                except Exception:
                    return None

            STORAGE_PATHS = json.loads(%r)

            def _read_disk_free():
                lowest = None
                resolved_path = None
                for path in STORAGE_PATHS.values():
                    expanded = os.path.expanduser(path)
                    for candidate in (expanded, os.path.dirname(expanded)):
                        try:
                            free_bytes = float(shutil.disk_usage(candidate).free)
                        except Exception:
                            continue
                        location = os.path.realpath(expanded)
                        if lowest is None or free_bytes < lowest:
                            lowest = free_bytes
                            resolved_path = location
                        break

                if lowest is None:
                    try:
                        lowest = float(shutil.disk_usage('.').free)
                        resolved_path = os.path.realpath('.')
                    except Exception:
                        return None, None

                return lowest, resolved_path

            def _probe_gpus():
                names = []
                nvidia = shutil.which('nvidia-smi')
                if not nvidia:
                    return names
                try:
                    output = subprocess.check_output(
                        [nvidia, '--query-gpu=name', '--format=csv,noheader'], text=True
                    )
                    names = [line.strip() for line in output.splitlines() if line.strip()]
                except Exception:
                    names = ['GPU detected but details unavailable']
                return names

            cpu_count = os.cpu_count() or 0
            ram_bytes = _read_mem_total()
            disk_bytes, disk_path = _read_disk_free()
            gpus = _probe_gpus()

            def _format_gb(value):
                if value is None:
                    return 'unknown'
                return f'{value / (1024 ** 3):.1f} GB'

            ram_gb = (ram_bytes or 0) / (1024 ** 3)
            disk_gb = (disk_bytes or 0) / (1024 ** 3)
            gpu_summary = ', '.join(gpus) if gpus else 'none detected'

            disk_location = disk_path or 'unknown path'

            recommendations = []
            if cpu_count and cpu_count < 4:
                recommendations.append('Favor cloud-hosted services; limited CPU cores detected.')
            if ram_gb and ram_gb < 8:
                recommendations.append(
                    f'Consider cloud PostgreSQL/vector DB; only {ram_gb:.1f} GB RAM available.'
                )
            elif ram_gb and ram_gb < 16:
                recommendations.append(
                    'Plan smaller local models or use hosted inference for heavier workloads.'
                )
            if disk_gb and disk_gb < 40:
                recommendations.append(
                    f'Low free disk ({disk_gb:.1f} GB) at {disk_location}; '
                    'prefer cloud data stores and models.'
                )
            if not gpus:
                recommendations.append('No GPU detected; hosted model inference recommended for speed.')

            summary = (
                f"{cpu_count or 'Unknown'} CPU cores, "
                f"{_format_gb(ram_bytes)} RAM, "
                f"{_format_gb(disk_bytes)} free disk at {disk_location}. "
                f"GPU: {gpu_summary}."
            )
            payload = {
                'message': f'Hardware review completed: {summary}',
                'disk_path': disk_path,
                'recommendation': '\n'.join(recommendations)
                if recommendations
                else 'Local hosting looks sufficient for databases and moderate models.',
            }
            print(json.dumps(payload))
            """
        ) % storage_paths_json

        def _parse_probe(
            passed: bool, stdout: str, stderr: str, exit_status: int
        ) -> tuple[str, str | None]:
            recommendation = None
            message = (
                "Hardware review completed."
                if passed
                else self._format_failure_message(
                    "Hardware readiness check failed.", exit_status, stdout, stderr
                )
            )
            if not stdout:
                return message, recommendation

            try:
                payload = json.loads(stdout.splitlines()[-1])
            except Exception:
                return message, recommendation

            message = payload.get("message", message)
            recommendation = payload.get("recommendation")
            return message, recommendation

        return PreflightCheckDefinition(
            identifier="hardware",
            label="Hardware readiness",
            command=["/usr/bin/env", "python3", "-c", probe_script],
            success_message="Hardware review completed.",
            failure_hint="Hardware readiness check failed.",
            fix_command=None,
            fix_label=None,
            requires_sudo=False,
            process_output=_parse_probe,
        )


__all__ = [
    "PreflightCheckDefinition",
    "PreflightCheckResult",
    "PreflightHelper",
    "DATABASE_LOCAL_PG_TIP",
    "DATABASE_LOCAL_TIP",
    "DATABASE_MANAGED_TIP",
    "VECTOR_HOSTING_TIP",
    "MODEL_HOSTING_TIP",
    "database_recommendation_for_state",
]
