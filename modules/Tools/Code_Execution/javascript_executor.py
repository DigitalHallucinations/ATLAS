"""Sandboxed JavaScript execution powered by external runtimes."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import shlex
import shutil
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

__all__ = [
    "JavaScriptExecutor",
    "JavaScriptExecutorError",
    "JavaScriptExecutionTimeoutError",
    "JavaScriptSandboxError",
]


_DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024
_DEFAULT_MAX_FILE_BYTES = 128 * 1024
_DEFAULT_MAX_FILES = 32
_SANDBOX_VIOLATION_EXIT_CODES = {31, 64, 70}
_SANDBOX_VIOLATION_PATTERNS = (
    "permission denied",
    "operation not permitted",
    "sandbox violation",
    "restricted syscall",
)


class JavaScriptExecutorError(RuntimeError):
    """Raised when JavaScript execution fails for operational reasons."""


class JavaScriptExecutionTimeoutError(JavaScriptExecutorError):
    """Raised when a JavaScript program exceeds its allotted runtime."""


class JavaScriptSandboxError(JavaScriptExecutorError):
    """Raised when a JavaScript program violates sandbox restrictions."""


@dataclass(slots=True)
class _ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    duration_ms: float


def _truncate_output(value: bytes, *, limit: int) -> tuple[str, bool]:
    text = value.decode("utf-8", errors="replace")
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text, False
    truncated = encoded[:limit].decode("utf-8", errors="ignore")
    return truncated, True


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_directory(root: Path) -> dict[str, tuple[int, int, str]]:
    snapshot: dict[str, tuple[int, int, str]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        snapshot[rel] = (stat.st_mtime_ns, stat.st_size, _hash_file(path))
    return snapshot


def _sanitize_args(args: Sequence[str] | None) -> tuple[str, ...]:
    if not args:
        return ()
    sanitized: list[str] = []
    for value in args:
        if value is None:
            continue
        sanitized.append(str(value))
    return tuple(sanitized)


try:  # pragma: no cover - platform guard
    import resource
except ImportError:  # pragma: no cover - non-POSIX platforms
    resource = None  # type: ignore[assignment]


class JavaScriptExecutor:
    """Execute JavaScript via a sandboxed external runtime."""

    def __init__(
        self,
        *,
        executable: str | None = None,
        args: Sequence[str] | None = None,
        default_timeout: float = 5.0,
        cpu_time_limit: float | None = 2.0,
        memory_limit_bytes: int | None = 256 * 1024 * 1024,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
        max_files: int = _DEFAULT_MAX_FILES,
        environment: Mapping[str, str] | None = None,
        sandbox_violation_exit_codes: Iterable[int] | None = None,
        sandbox_violation_patterns: Iterable[str] | None = None,
    ) -> None:
        if default_timeout <= 0:
            raise ValueError("default_timeout must be greater than zero.")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive.")
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive.")
        if max_files <= 0:
            raise ValueError("max_files must be positive.")

        self._explicit_executable = executable
        self._engine_args = _sanitize_args(args)
        self._default_timeout = float(default_timeout)
        self._cpu_time_limit = cpu_time_limit if cpu_time_limit else None
        self._memory_limit = memory_limit_bytes if memory_limit_bytes else None
        self._max_output_bytes = int(max_output_bytes)
        self._max_file_bytes = int(max_file_bytes)
        self._max_files = int(max_files)
        self._base_env: dict[str, str] = {}
        if environment:
            for key, value in environment.items():
                if value is None:
                    continue
                self._base_env[str(key)] = str(value)

        exit_codes = set(_SANDBOX_VIOLATION_EXIT_CODES)
        if sandbox_violation_exit_codes:
            exit_codes.update(int(code) for code in sandbox_violation_exit_codes)
        self._sandbox_exit_codes = frozenset(exit_codes)

        patterns = list(_SANDBOX_VIOLATION_PATTERNS)
        if sandbox_violation_patterns:
            patterns.extend(str(pattern) for pattern in sandbox_violation_patterns)
        self._sandbox_patterns = tuple(pattern.lower() for pattern in patterns if pattern)

    @classmethod
    def from_config(cls, settings: Mapping[str, Any] | None) -> "JavaScriptExecutor":
        if not isinstance(settings, Mapping):
            return cls()

        config = dict(settings)
        executable = config.get("executable")
        args = config.get("args")
        default_timeout = config.get("default_timeout", 5.0)
        cpu_time_limit = config.get("cpu_time_limit", 2.0)
        memory_limit = config.get("memory_limit_bytes", 256 * 1024 * 1024)
        max_output_bytes = config.get("max_output_bytes", _DEFAULT_MAX_OUTPUT_BYTES)
        max_file_bytes = config.get("max_file_bytes", _DEFAULT_MAX_FILE_BYTES)
        max_files = config.get("max_files", _DEFAULT_MAX_FILES)
        environment = config.get("environment")
        sandbox_codes = config.get("sandbox_violation_exit_codes")
        sandbox_patterns = config.get("sandbox_violation_patterns")

        if isinstance(args, str):
            args = shlex.split(args)
        elif isinstance(args, Sequence):
            args = list(args)
        else:
            args = None

        return cls(
            executable=str(executable) if executable else None,
            args=args,
            default_timeout=float(default_timeout),
            cpu_time_limit=float(cpu_time_limit) if cpu_time_limit else None,
            memory_limit_bytes=int(memory_limit) if memory_limit else None,
            max_output_bytes=int(max_output_bytes),
            max_file_bytes=int(max_file_bytes),
            max_files=int(max_files),
            environment=environment if isinstance(environment, Mapping) else None,
            sandbox_violation_exit_codes=sandbox_codes
            if isinstance(sandbox_codes, Iterable)
            else None,
            sandbox_violation_patterns=sandbox_patterns
            if isinstance(sandbox_patterns, Iterable)
            else None,
        )

    async def run(
        self,
        *,
        command: str,
        timeout: float | None = None,
        files: Any = None,
        context: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        del context  # intentionally unused

        if not isinstance(command, str):
            raise JavaScriptExecutorError("The 'command' argument must be a string.")

        normalized = textwrap.dedent(command).strip()
        if not normalized:
            raise JavaScriptExecutorError("No JavaScript source code was provided.")

        executable = self._resolve_executable()
        timeout_seconds = self._sanitize_timeout(timeout)

        normalized_files = self._normalize_files(files)

        async with _SandboxDirectory(normalized_files) as sandbox:
            script_path = sandbox.write_script(normalized)
            before_snapshot = _snapshot_directory(sandbox.root)

            try:
                result = await asyncio.to_thread(
                    self._execute,
                    executable,
                    script_path,
                    timeout_seconds,
                    sandbox.root,
                    sandbox.environment(self._base_env),
                )
            except subprocess.TimeoutExpired as exc:
                raise JavaScriptExecutionTimeoutError(
                    "JavaScript execution timed out."
                ) from exc
            except FileNotFoundError as exc:
                raise JavaScriptExecutorError(
                    f"JavaScript runtime not found: {executable}"
                ) from exc

            if result.exit_code != 0:
                if self._is_sandbox_violation(result.exit_code, result.stderr):
                    raise JavaScriptSandboxError(result.stderr or "Sandbox violation detected.")
                raise JavaScriptExecutorError(
                    result.stderr or f"JavaScript runtime exited with status {result.exit_code}."
                )

            files_payload, files_truncated = sandbox.collect_generated_files(
                before_snapshot,
                max_files=self._max_files,
                max_bytes=self._max_file_bytes,
                exclude={script_path.name},
            )

        payload: dict[str, Any] = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "files": files_payload,
            "files_truncated": files_truncated,
        }
        return payload

    def _normalize_files(self, files: Any) -> Mapping[str, bytes]:
        if files is None:
            return {}

        if isinstance(files, Mapping):
            normalized: dict[str, bytes] = {}
            for key, value in files.items():
                if value is None:
                    continue
                normalized[str(key)] = (
                    value.encode("utf-8") if isinstance(value, str) else bytes(value)
                )
            return normalized

        if isinstance(files, Sequence) and not isinstance(files, (str, bytes)):
            normalized_list: dict[str, bytes] = {}
            for entry in files:
                if not isinstance(entry, Mapping):
                    continue
                path = entry.get("path")
                content = entry.get("content")
                if path is None or content is None:
                    continue
                encoding = str(entry.get("encoding", "utf-8") or "utf-8").lower()
                if isinstance(content, str) and encoding == "base64":
                    try:
                        data = base64.b64decode(content)
                    except Exception as exc:  # pragma: no cover - invalid base64
                        raise JavaScriptExecutorError(
                            "Invalid base64 file content provided."
                        ) from exc
                elif isinstance(content, str):
                    data = content.encode("utf-8")
                else:
                    data = bytes(content)
                normalized_list[str(path)] = data
            return normalized_list

        raise JavaScriptExecutorError(
            "Unsupported 'files' payload; expected mapping or sequence of descriptors."
        )

    def _resolve_executable(self) -> str:
        if self._explicit_executable:
            return self._explicit_executable

        for candidate in ("qjs", "quickjs", "deno", "node"):
            path = shutil.which(candidate)
            if path:
                return path
        raise JavaScriptExecutorError(
            "No JavaScript runtime was found. Configure 'executable' for the JavaScript executor."
        )

    def _sanitize_timeout(self, timeout: float | None) -> float:
        if timeout is None:
            return self._default_timeout
        try:
            value = float(timeout)
        except (TypeError, ValueError) as exc:
            raise JavaScriptExecutorError("Timeout must be numeric.") from exc
        if value <= 0:
            raise JavaScriptExecutorError("Timeout must be greater than zero.")
        return value

    def _execute(
        self,
        executable: str,
        script_path: Path,
        timeout_seconds: float,
        working_dir: Path,
        env: Mapping[str, str],
    ) -> _ExecutionResult:
        command = [executable, *self._engine_args, str(script_path)]
        started = time.perf_counter()

        preexec_fn = self._build_preexec_fn()

        proc = subprocess.Popen(
            command,
            cwd=str(working_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(env),
            preexec_fn=preexec_fn,
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise exc

        duration_ms = (time.perf_counter() - started) * 1000

        stdout_text, stdout_truncated = _truncate_output(
            stdout, limit=self._max_output_bytes
        )
        stderr_text, stderr_truncated = _truncate_output(
            stderr, limit=self._max_output_bytes
        )

        return _ExecutionResult(
            exit_code=proc.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            duration_ms=duration_ms,
        )

    def _build_preexec_fn(self):
        if os.name != "posix" or resource is None:
            return None

        cpu_limit = self._cpu_time_limit
        memory_limit = self._memory_limit

        def _set_limits():
            if cpu_limit:
                seconds = max(1, int(cpu_limit))
                resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
            if memory_limit:
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        return _set_limits

    def _is_sandbox_violation(self, exit_code: int, stderr: str) -> bool:
        if exit_code in self._sandbox_exit_codes:
            return True
        lowered = stderr.lower() if stderr else ""
        return any(pattern in lowered for pattern in self._sandbox_patterns)


class _SandboxDirectory:
    """Manage a transient sandboxed working directory."""

    def __init__(self, files: Mapping[str, bytes]) -> None:
        self._files = files
        self._tmp_dir: tempfile.TemporaryDirectory[str] | None = None

    async def __aenter__(self) -> "_SandboxDirectory":
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="atlas-js-")
        root = Path(self._tmp_dir.name)
        self._populate_files(root)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None

    @property
    def root(self) -> Path:
        if self._tmp_dir is None:
            raise RuntimeError("Sandbox has not been created yet.")
        return Path(self._tmp_dir.name)

    def write_script(self, source: str) -> Path:
        root = self.root
        script_path = root / "main.js"
        script_path.write_text(source, encoding="utf-8")
        return script_path

    def environment(self, base_env: Mapping[str, str]) -> MutableMapping[str, str]:
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(self.root)}
        for key, value in base_env.items():
            env[str(key)] = str(value)
        return env

    def collect_generated_files(
        self,
        before: Mapping[str, tuple[int, int, str]],
        *,
        max_files: int,
        max_bytes: int,
        exclude: Iterable[str],
    ) -> tuple[list[dict[str, Any]], bool]:
        root = self.root
        after = _snapshot_directory(root)
        exclude_set = {Path(name).as_posix() for name in exclude}

        files: list[dict[str, Any]] = []
        truncated = False

        for rel_path, (mtime, size, digest) in sorted(after.items()):
            if rel_path in exclude_set:
                continue
            previous = before.get(rel_path)
            if previous and previous[2] == digest:
                continue

            if len(files) >= max_files:
                truncated = True
                break

            payload = self._serialize_file(root / rel_path, rel_path, size, max_bytes)
            if payload["truncated"]:
                truncated = True
            files.append(payload)

        return files, truncated

    def _populate_files(self, root: Path) -> None:
        if not self._files:
            return
        for name, content in self._files.items():
            if content is None:
                continue
            relative = self._sanitize_relative_path(name)
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)

    @staticmethod
    def _sanitize_relative_path(name: str) -> Path:
        candidate = Path(name)
        if candidate.is_absolute():
            raise JavaScriptExecutorError("File paths must be relative to the sandbox root.")
        normalized = Path(os.path.normpath(candidate.as_posix()))
        if str(normalized).startswith(".."):
            raise JavaScriptExecutorError("File paths may not escape the sandbox root.")
        return normalized

    def _serialize_file(
        self,
        path: Path,
        rel_path: str,
        size: int,
        max_bytes: int,
    ) -> dict[str, Any]:
        raw = path.read_bytes()
        truncated = False
        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
            truncated = True
        encoded = base64.b64encode(raw).decode("ascii")
        return {
            "path": rel_path,
            "size": size,
            "content": encoded,
            "encoding": "base64",
            "truncated": truncated,
        }

