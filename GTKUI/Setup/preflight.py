"""Async system readiness checks for the GTK setup wizard."""

from __future__ import annotations

import dataclasses
import json
import textwrap
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import gi

gi.require_version("Gio", "2.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gio, GLib

from ATLAS.setup.controller import DatabaseState, MessageBusState, _compose_dsn


PasswordProvider = Callable[[], str | None]
ResultCallback = Callable[["PreflightCheckResult"], None]
CompleteCallback = Callable[[list["PreflightCheckResult"]], None]


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
    recommendation: str | None = None


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
        self._redis_url: str | None = None
        self._checks: list[PreflightCheckDefinition] = list(checks or self._default_checks())
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
            self._redis_url = None
            return

        if (state.backend or "").strip().lower() != "redis":
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
        self._checks = list(self._default_checks())
        self._pending = list(self._checks)
        self._running = True
        self._advance()

    def run_fix(self, identifier: str, callback: ResultCallback) -> None:
        """Attempt the fix command for *identifier* and re-run the check."""

        if self._running or self._fix_in_progress is not None:
            raise RuntimeError("Another preflight operation is already running")

        definition = self._definition_for(identifier)
        if definition is None or definition.fix_command is None:
            result = self._results.get(identifier)
            if result is None:
                result = PreflightCheckResult(
                    identifier=identifier,
                    label=definition.label if definition else identifier,
                    passed=False,
                    message="No automated fix is available for this check.",
                    fix_label=None,
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

    # -- check catalog --------------------------------------------------

    def _default_checks(self) -> Iterable[PreflightCheckDefinition]:
        checks: list[PreflightCheckDefinition] = []
        database_check = self._build_database_check()
        if database_check is not None:
            checks.append(database_check)
        checks.append(self._build_redis_check())
        checks.append(self._build_hardware_check())
        return checks

    def _build_database_check(self) -> PreflightCheckDefinition | None:
        backend = (self._database_state.backend or "postgresql").strip().lower() or "postgresql"
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
            return PreflightCheckDefinition(
                identifier="mongodb",
                label="MongoDB",
                command=["/usr/bin/env", "python3", "-c", script, uri],
                success_message="MongoDB connection succeeded.",
                failure_hint=mongo_hint,
            )

        pg_hint = (
            "PostgreSQL is unreachable. Ensure the server is installed and that pg_isready"
            " can connect to the configured instance."
        )
        host = (self._database_state.host or "").strip()
        port = int(self._database_state.port or 0)
        database = (self._database_state.database or "").strip()
        command: list[str] = ["/usr/bin/env", "pg_isready", "-q"]
        if host:
            command.extend(["-h", host])
        if port:
            command.extend(["-p", str(port)])
        if database:
            command.extend(["-d", database])
        return PreflightCheckDefinition(
            identifier="postgresql",
            label="PostgreSQL",
            command=command,
            success_message="PostgreSQL is accepting connections.",
            failure_hint=pg_hint,
            fix_command=[
                "/usr/bin/env",
                "sudo",
                "-S",
                "systemctl",
                "start",
                "postgresql",
            ],
            fix_label="Start PostgreSQL service",
            requires_sudo=True,
        )

    def _build_redis_check(self) -> PreflightCheckDefinition:
        redis_hint = (
            "Redis did not respond to ping. Verify the redis-server service is installed"
            " and running."
        )
        command: list[str] = ["/usr/bin/env", "redis-cli"]
        if self._redis_url:
            command.extend(["-u", self._redis_url])
        command.append("ping")
        return PreflightCheckDefinition(
            identifier="redis",
            label="Redis",
            command=command,
            success_message="Redis responded to ping.",
            failure_hint=redis_hint,
            fix_command=[
                "/usr/bin/env",
                "sudo",
                "-S",
                "systemctl",
                "start",
                "redis-server",
            ],
            fix_label="Start Redis service",
            requires_sudo=True,
        )

    def _build_hardware_check(self) -> PreflightCheckDefinition:
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

            def _read_disk_free():
                try:
                    return float(shutil.disk_usage('.').free)
                except Exception:
                    return None

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
            disk_bytes = _read_disk_free()
            gpus = _probe_gpus()

            def _format_gb(value):
                if value is None:
                    return 'unknown'
                return f'{value / (1024 ** 3):.1f} GB'

            ram_gb = (ram_bytes or 0) / (1024 ** 3)
            disk_gb = (disk_bytes or 0) / (1024 ** 3)
            gpu_summary = ', '.join(gpus) if gpus else 'none detected'

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
                    f'Low free disk ({disk_gb:.1f} GB); prefer cloud data stores and models.'
                )
            if not gpus:
                recommendations.append('No GPU detected; hosted model inference recommended for speed.')

            summary = (
                f"{cpu_count or 'Unknown'} CPU cores, "
                f"{_format_gb(ram_bytes)} RAM, "
                f"{_format_gb(disk_bytes)} free disk. GPU: {gpu_summary}."
            )
            payload = {
                'message': f'Hardware review completed: {summary}',
                'recommendation': '\n'.join(recommendations)
                if recommendations
                else 'Local hosting looks sufficient for databases and moderate models.',
            }
            print(json.dumps(payload))
            """
        )

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
]
