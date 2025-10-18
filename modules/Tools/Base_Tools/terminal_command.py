"""Secure terminal command execution tool."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from modules.Tools.tool_event_system import event_system

try:  # pragma: no cover - ``resource`` is unavailable on non-POSIX platforms
    import resource  # type: ignore
except ImportError:  # pragma: no cover - fallback for Windows environments
    resource = None  # type: ignore


logger = logging.getLogger(__name__)

__all__ = [
    "TerminalCommand",
    "TerminalCommandError",
    "CommandNotAllowedError",
    "WorkingDirectoryViolationError",
    "TerminalCommandTimeoutError",
    "DEFAULT_TERMINAL_JAIL",
]


_EVENT_NAME = "terminal_command_executed"
_ALLOWED_COMMANDS: tuple[str, ...] = (
    "cat",
    "echo",
    "grep",
    "head",
    "ls",
    "pwd",
    "sleep",
    "tail",
    "wc",
)
_DENIED_COMMANDS: tuple[str, ...] = (
    "apt",
    "apt-get",
    "bash",
    "chmod",
    "chown",
    "cp",
    "kill",
    "mv",
    "pip",
    "python",
    "python3",
    "rm",
    "sh",
    "shutdown",
    "sudo",
    "systemctl",
)
_DEFAULT_TIMEOUT_SECONDS = 5.0
_MAX_TIMEOUT_SECONDS = 30.0
_CPU_LIMIT_SECONDS = 2
_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
_MAX_OUTPUT_BYTES = 8 * 1024
_REDACTION_REPLACEMENT = "[REDACTED]"
_SENSITIVE_MARKERS = (
    "password",
    "secret",
    "token",
    "apikey",
    "key",
    "session",
    "bearer",
)
_SAFE_ENV_VARS = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM")
DEFAULT_TERMINAL_JAIL = Path(__file__).resolve().parents[3]


class TerminalCommandError(RuntimeError):
    """Base class for terminal command execution failures."""


class CommandNotAllowedError(TerminalCommandError):
    """Raised when attempting to run a command that violates policy."""


class WorkingDirectoryViolationError(TerminalCommandError):
    """Raised when the requested working directory violates the jail constraint."""


class TerminalCommandTimeoutError(TerminalCommandError):
    """Raised when a command exceeds the allotted execution time."""


@dataclass(frozen=True)
class TerminalCommandResult:
    """Structured result returned by :func:`TerminalCommand`."""

    command: tuple[str, ...]
    working_directory: str
    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    duration_ms: float

    def as_dict(self) -> Mapping[str, object]:
        """Return a JSON-serializable payload for downstream consumers."""

        return {
            "command": list(self.command),
            "working_directory": self.working_directory,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "duration_ms": self.duration_ms,
        }


def _current_jail_root() -> Path:
    candidate = os.environ.get("ATLAS_TERMINAL_JAIL")
    if candidate:
        try:
            resolved = Path(candidate).resolve()
        except (OSError, RuntimeError):
            logger.warning("Invalid ATLAS_TERMINAL_JAIL path '%s'. Falling back to default.", candidate)
        else:
            if resolved.is_dir():
                return resolved
    return DEFAULT_TERMINAL_JAIL


def _enforce_working_directory(cwd: str | None) -> Path:
    jail = _current_jail_root()
    if cwd is None:
        target = jail
    else:
        candidate = Path(cwd)
        target = candidate if candidate.is_absolute() else jail.joinpath(candidate)
        try:
            target = target.resolve(strict=True)
        except FileNotFoundError as exc:
            raise WorkingDirectoryViolationError(
                f"Requested working directory '{candidate}' does not exist within the sandbox."
            ) from exc
        except RuntimeError as exc:  # pragma: no cover - triggered by deep symlink loops
            raise WorkingDirectoryViolationError("Unable to resolve working directory path.") from exc

    try:
        target.relative_to(jail)
    except ValueError as exc:
        raise WorkingDirectoryViolationError(
            f"Working directory '{target}' escapes sandbox root '{jail}'."
        ) from exc

    if not target.is_dir():
        raise WorkingDirectoryViolationError(
            f"Working directory '{target}' is not a directory."
        )

    return target


def _normalize_command(command: object) -> str:
    if not isinstance(command, str):
        raise CommandNotAllowedError("Command name must be a string.")
    normalized = command.strip()
    if not normalized:
        raise CommandNotAllowedError("Command name cannot be empty.")
    if os.sep in normalized or (os.altsep and os.altsep in normalized):
        raise CommandNotAllowedError("Absolute or relative paths are not permitted.")
    lowered = normalized.lower()
    if lowered in _DENIED_COMMANDS:
        raise CommandNotAllowedError(f"Command '{normalized}' is explicitly denied.")
    if lowered not in _ALLOWED_COMMANDS:
        raise CommandNotAllowedError(f"Command '{normalized}' is not in the allowlist.")
    return normalized


def _normalize_arguments(arguments: Sequence[object] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    if not arguments:
        return tuple(normalized)

    for index, value in enumerate(arguments):
        if value is None:
            raise CommandNotAllowedError(f"Argument at position {index} cannot be None.")
        candidate = str(value)
        if len(candidate) > 2048:
            raise CommandNotAllowedError("Arguments longer than 2 KiB are not permitted.")
        normalized.append(candidate)
    return tuple(normalized)


def _sanitize_timeout(timeout: object | None) -> float:
    if timeout is None:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise TerminalCommandError("Timeout value must be numeric.") from exc
    if value <= 0:
        raise TerminalCommandError("Timeout must be greater than zero.")
    return min(value, _MAX_TIMEOUT_SECONDS)


def _build_environment(extra_env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    env: dict[str, str] = {}
    for key in _SAFE_ENV_VARS:
        if key == "PATH":
            env[key] = os.environ.get("PATH", "/usr/bin:/bin")
        else:
            value = os.environ.get(key)
            if value:
                env[key] = value
    if extra_env:
        for key, value in extra_env.items():
            if key in _SAFE_ENV_VARS:
                env[key] = str(value)
    return env


def _apply_resource_limits() -> None:  # pragma: no cover - exercised indirectly in tests
    if resource is None:
        return
    try:
        if _CPU_LIMIT_SECONDS:
            resource.setrlimit(resource.RLIMIT_CPU, (_CPU_LIMIT_SECONDS, _CPU_LIMIT_SECONDS))
    except (ValueError, OSError):
        logger.debug("Unable to apply RLIMIT_CPU restriction.")
    try:
        if _MEMORY_LIMIT_BYTES:
            resource.setrlimit(resource.RLIMIT_AS, (_MEMORY_LIMIT_BYTES, _MEMORY_LIMIT_BYTES))
    except (ValueError, OSError):
        logger.debug("Unable to apply RLIMIT_AS restriction.")


def _redact_command_tokens(tokens: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    redact_next_value = False
    sensitive_switches = {f"--{marker}" for marker in _SENSITIVE_MARKERS}

    for token in tokens:
        lowered = token.lower()
        if redact_next_value:
            redacted.append(_REDACTION_REPLACEMENT)
            redact_next_value = False
            continue
        if lowered in sensitive_switches:
            redacted.append(token)
            redact_next_value = True
            continue
        if "=" in token and any(marker in lowered for marker in _SENSITIVE_MARKERS):
            prefix, _, _ = token.partition("=")
            redacted.append(f"{prefix}={_REDACTION_REPLACEMENT}")
            continue
        if any(marker in lowered for marker in _SENSITIVE_MARKERS):
            redacted.append(_REDACTION_REPLACEMENT)
            continue
        redacted.append(token)

    if redact_next_value:
        redacted.append(_REDACTION_REPLACEMENT)

    return redacted


def _redact_text(value: str) -> str:
    lowered = value.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return _REDACTION_REPLACEMENT
    return value


async def _drain_stream(stream: asyncio.StreamReader | None) -> tuple[str, bool]:
    if stream is None:
        return "", False

    collected = bytearray()
    truncated = False

    while True:
        chunk = await stream.read(1024)
        if not chunk:
            break
        if not truncated:
            remaining = _MAX_OUTPUT_BYTES - len(collected)
            if remaining > 0:
                collected.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
        else:
            truncated = True

        if len(collected) >= _MAX_OUTPUT_BYTES:
            truncated = True

    text = collected.decode("utf-8", errors="replace")
    return text, truncated


async def TerminalCommand(
    *,
    command: str | Sequence[object],
    arguments: Sequence[object] | None = None,
    timeout: float | None = None,
    working_directory: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Execute a whitelisted terminal command inside the sandbox."""

    if isinstance(command, str):
        normalized_command = _normalize_command(command)
        command_tokens: tuple[str, ...] = ()
    elif isinstance(command, Sequence):
        if not command:
            raise CommandNotAllowedError("Command list cannot be empty.")

        command_items = list(command)
        normalized_command = _normalize_command(command_items[0])
        command_tokens = _normalize_arguments(command_items[1:])
    else:
        raise CommandNotAllowedError("Command must be a string or sequence of arguments.")

    explicit_args = _normalize_arguments(arguments)
    normalized_args = (*command_tokens, *explicit_args)
    resolved_cwd = _enforce_working_directory(working_directory)
    timeout_seconds = _sanitize_timeout(timeout)
    env = _build_environment(environment)

    command_line = (normalized_command, *normalized_args)
    redacted_command_tokens = _redact_command_tokens(command_line)
    sanitized_command = " ".join(redacted_command_tokens)
    logger.info("Executing terminal command: %s", sanitized_command)

    loop = asyncio.get_running_loop()
    start_time = loop.time()
    process = await asyncio.create_subprocess_exec(
        normalized_command,
        *normalized_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(resolved_cwd),
        env=dict(env),
        preexec_fn=_apply_resource_limits if resource is not None else None,
    )

    stdout_task = asyncio.create_task(_drain_stream(process.stdout))
    stderr_task = asyncio.create_task(_drain_stream(process.stderr))

    timed_out = False
    try:
        await asyncio.wait_for(process.wait(), timeout_seconds)
    except asyncio.TimeoutError:
        timed_out = True
        process.kill()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), 1)
    finally:
        stdout_text, stdout_truncated = await stdout_task
        stderr_text, stderr_truncated = await stderr_task

    duration_ms = (loop.time() - start_time) * 1000.0

    status: str
    exit_code: int | None
    if timed_out:
        status = "timeout"
        exit_code = None
    else:
        exit_code = process.returncode
        status = "success" if exit_code == 0 else "error"

    stdout_payload_text = stdout_text if not stdout_truncated else stdout_text[: _MAX_OUTPUT_BYTES]
    stderr_payload_text = stderr_text if not stderr_truncated else stderr_text[: _MAX_OUTPUT_BYTES]
    redacted_stdout = _redact_text(stdout_payload_text)
    redacted_stderr = _redact_text(stderr_payload_text)

    event_payload = {
        "tool": "terminal_command",
        "status": status,
        "command": redacted_command_tokens,
        "working_directory": str(resolved_cwd),
        "exit_code": exit_code,
        "stdout": redacted_stdout,
        "stderr": redacted_stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "duration_ms": duration_ms,
        "timeout_seconds": timeout_seconds,
    }

    event_system.publish(_EVENT_NAME, event_payload)

    if timed_out:
        raise TerminalCommandTimeoutError(
            f"Command '{normalized_command}' exceeded timeout of {timeout_seconds} seconds."
        )

    result = TerminalCommandResult(
        command=command_line,
        working_directory=str(resolved_cwd),
        exit_code=exit_code if exit_code is not None else 0,
        stdout=stdout_text,
        stderr=stderr_text,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        duration_ms=duration_ms,
    )

    payload = result.as_dict()
    exit_repr = exit_code if exit_code is not None else "[timeout]"
    logger.info(
        "Terminal command completed: %s -> exit code %s (duration %.2f ms). stdout=%s stderr=%s",
        sanitized_command,
        exit_repr,
        duration_ms,
        redacted_stdout,
        redacted_stderr,
    )
    return payload
