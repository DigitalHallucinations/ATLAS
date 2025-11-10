"""Async system readiness checks for the GTK setup wizard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import gi

gi.require_version("Gio", "2.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gio, GLib


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


@dataclass
class PreflightCheckResult:
    """Outcome of executing a :class:`PreflightCheckDefinition`."""

    identifier: str
    label: str
    passed: bool
    message: str
    fix_label: str | None


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
        self._checks: list[PreflightCheckDefinition] = list(checks or self._default_checks())
        self._subprocess_factory = subprocess_factory or self._spawn_subprocess

        self._pending: list[PreflightCheckDefinition] = []
        self._results: dict[str, PreflightCheckResult] = {}
        self._running: bool = False
        self._fix_in_progress: str | None = None
        self._on_update: ResultCallback | None = None
        self._on_complete: CompleteCallback | None = None

    # -- public API -----------------------------------------------------

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
            result = PreflightCheckResult(
                identifier=definition.identifier,
                label=definition.label,
                passed=passed,
                message=message,
                fix_label=definition.fix_label,
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
        pg_hint = (
            "PostgreSQL is unreachable. Ensure the server is installed and that pg_isready"
            " can connect to the configured instance."
        )
        redis_hint = (
            "Redis did not respond to ping. Verify the redis-server service is installed"
            " and running."
        )
        venv_hint = (
            "No project virtual environment detected. Create .venv before launching ATLAS."
        )
        python_check = (
            "import pathlib, sys; base = pathlib.Path('.venv');"
            "suffix = 'Scripts' if sys.platform.startswith('win') else 'bin';"
            "exe = base / suffix / ('python.exe' if sys.platform.startswith('win') else 'python');"
            "sys.exit(0 if exe.exists() else 1)"
        )
        return [
            PreflightCheckDefinition(
                identifier="postgresql",
                label="PostgreSQL",
                command=["/usr/bin/env", "pg_isready", "-q"],
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
            ),
            PreflightCheckDefinition(
                identifier="redis",
                label="Redis",
                command=["/usr/bin/env", "redis-cli", "ping"],
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
            ),
            PreflightCheckDefinition(
                identifier="virtualenv",
                label="Project virtualenv",
                command=["/usr/bin/env", "python3", "-c", python_check],
                success_message=".venv virtual environment is ready.",
                failure_hint=venv_hint,
                fix_command=["/usr/bin/env", "python3", "-m", "venv", ".venv"],
                fix_label="Create .venv virtualenv",
                requires_sudo=False,
            ),
        ]


__all__ = [
    "PreflightCheckDefinition",
    "PreflightCheckResult",
    "PreflightHelper",
]
