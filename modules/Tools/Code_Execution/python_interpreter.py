"""Sandboxed Python execution utilities."""

from __future__ import annotations

import ast
import asyncio
import builtins
import contextlib
import io
import math
import statistics
import textwrap
import time
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from types import CodeType, MappingProxyType
from typing import Any, Mapping, MutableMapping

__all__ = [
    "PythonInterpreter",
    "PythonInterpreterError",
    "SandboxViolationError",
    "PythonInterpreterTimeoutError",
]


_MAX_OUTPUT_BYTES = 8 * 1024
_REDACTION_REPLACEMENT = "[REDACTED]"
_SENSITIVE_MARKERS = (
    "password",
    "secret",
    "token",
    "apikey",
    "api_key",
    "bearer",
    "session",
)
_FORBIDDEN_NAMES = {
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
}
_FORBIDDEN_ATTRIBUTES = {
    "__subclasses__",
    "__mro__",
    "__globals__",
    "__code__",
    "__closure__",
    "__func__",
    "__self__",
    "__bases__",
    "mro",
    "system",
    "popen",
    "spawn",
    "fork",
    "remove",
    "unlink",
    "rmdir",
    "mkdir",
    "makedirs",
    "rmtree",
    "walk",
    "listdir",
    "chdir",
    "getcwd",
    "socket",
    "connect",
    "bind",
    "listen",
    "accept",
    "send",
    "recv",
    "urlopen",
}


class PythonInterpreterError(RuntimeError):
    """Base exception for Python interpreter failures."""


class SandboxViolationError(PythonInterpreterError):
    """Raised when code attempts to escape the sandbox."""


class PythonInterpreterTimeoutError(PythonInterpreterError):
    """Raised when execution exceeds the allotted timeout."""


class PythonInterpreterExecutionError(PythonInterpreterError):
    """Raised when execution fails for runtime reasons."""


@dataclass(slots=True)
class _ExecutionArtifacts:
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    result: str
    result_type: str
    duration_ms: float


def _redact_text(value: str) -> str:
    lowered = value.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return _REDACTION_REPLACEMENT
    return value


def _truncate_output(value: str) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= _MAX_OUTPUT_BYTES:
        return value, False
    truncated = encoded[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="ignore")
    return truncated, True


class _SandboxValidator(ast.NodeVisitor):
    """Validate that code adheres to sandbox rules."""

    def visit_Import(self, node: ast.Import) -> None:  # noqa: D401 - simple violation message
        raise SandboxViolationError("Import statements are not allowed in the sandbox.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: D401 - simple violation message
        raise SandboxViolationError("Import statements are not allowed in the sandbox.")

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _FORBIDDEN_NAMES:
            raise SandboxViolationError(f"Calling '{func.id}' is not permitted in the sandbox.")
        if isinstance(func, ast.Attribute):
            attr = func.attr
            if attr in _FORBIDDEN_ATTRIBUTES:
                raise SandboxViolationError(
                    f"Access to attribute '{attr}' is not permitted in the sandbox."
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES:
            raise SandboxViolationError(f"Name '{node.id}' is not permitted in the sandbox.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _FORBIDDEN_ATTRIBUTES:
            raise SandboxViolationError(
                f"Access to attribute '{node.attr}' is not permitted in the sandbox."
            )
        self.generic_visit(node)


def _default_builtins() -> Mapping[str, object]:
    allowed = {
        "abs": builtins.abs,
        "all": builtins.all,
        "any": builtins.any,
        "ascii": builtins.ascii,
        "bool": builtins.bool,
        "bytes": builtins.bytes,
        "chr": builtins.chr,
        "complex": builtins.complex,
        "dict": builtins.dict,
        "divmod": builtins.divmod,
        "enumerate": builtins.enumerate,
        "float": builtins.float,
        "format": builtins.format,
        "frozenset": builtins.frozenset,
        "hash": builtins.hash,
        "hex": builtins.hex,
        "int": builtins.int,
        "isinstance": builtins.isinstance,
        "issubclass": builtins.issubclass,
        "iter": builtins.iter,
        "len": builtins.len,
        "list": builtins.list,
        "map": builtins.map,
        "max": builtins.max,
        "min": builtins.min,
        "next": builtins.next,
        "pow": builtins.pow,
        "print": builtins.print,
        "range": builtins.range,
        "repr": builtins.repr,
        "reversed": builtins.reversed,
        "round": builtins.round,
        "set": builtins.set,
        "slice": builtins.slice,
        "sorted": builtins.sorted,
        "str": builtins.str,
        "sum": builtins.sum,
        "tuple": builtins.tuple,
        "zip": builtins.zip,
        "Exception": builtins.Exception,
        "ArithmeticError": builtins.ArithmeticError,
        "AssertionError": builtins.AssertionError,
        "AttributeError": builtins.AttributeError,
        "ValueError": builtins.ValueError,
        "TypeError": builtins.TypeError,
        "NameError": builtins.NameError,
        "ZeroDivisionError": builtins.ZeroDivisionError,
        "__build_class__": builtins.__build_class__,
        "object": builtins.object,
        "property": builtins.property,
        "staticmethod": builtins.staticmethod,
        "classmethod": builtins.classmethod,
    }
    return MappingProxyType(allowed)


def _default_globals() -> Mapping[str, object]:
    return MappingProxyType(
        {
            "math": math,
            "statistics": statistics,
            "Decimal": Decimal,
            "Fraction": Fraction,
        }
    )


def _stringify_result(value: Any) -> tuple[str, str]:
    if value is None:
        return "None", "NoneType"
    if isinstance(value, (str, int, float, bool)):
        return str(value), type(value).__name__
    return repr(value), type(value).__name__


class PythonInterpreter:
    """Execute Python code within a heavily sandboxed environment."""

    def __init__(
        self,
        *,
        default_timeout: float = 5.0,
        builtins_namespace: Mapping[str, object] | None = None,
        global_namespace: Mapping[str, object] | None = None,
    ) -> None:
        if default_timeout <= 0:
            raise ValueError("default_timeout must be greater than zero.")
        self._default_timeout = float(default_timeout)
        self._builtins = MappingProxyType(dict(builtins_namespace or _default_builtins()))
        self._globals = MappingProxyType(dict(global_namespace or _default_globals()))
        self._validator = _SandboxValidator()

    async def run(
        self,
        *,
        command: str,
        timeout: float | None = None,
        context: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        """Execute *command* inside the sandbox and return structured telemetry."""

        del context  # context is unused but accepted for compatibility

        if not isinstance(command, str):
            raise PythonInterpreterError("The 'command' argument must be a string.")

        normalized = textwrap.dedent(command).strip()
        if not normalized:
            raise PythonInterpreterError("No Python code was supplied for execution.")

        try:
            parsed = ast.parse(normalized, mode="exec")
        except SyntaxError as exc:  # pragma: no cover - syntax errors bubble to caller
            raise PythonInterpreterError(f"Syntax error while parsing code: {exc}") from exc

        self._validator.visit(parsed)

        exec_code, eval_code = self._compile(parsed)

        timeout_seconds = self._sanitize_timeout(timeout)

        try:
            artifacts = await asyncio.wait_for(
                asyncio.to_thread(self._execute, exec_code, eval_code),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise PythonInterpreterTimeoutError("Python execution timed out.") from exc

        payload: dict[str, Any] = {
            "stdout": _redact_text(artifacts.stdout),
            "stderr": _redact_text(artifacts.stderr),
            "stdout_truncated": artifacts.stdout_truncated,
            "stderr_truncated": artifacts.stderr_truncated,
            "result": _redact_text(artifacts.result),
            "result_type": artifacts.result_type,
            "duration_ms": artifacts.duration_ms,
        }
        return payload

    def _compile(self, parsed: ast.AST) -> tuple[CodeType, CodeType | None]:
        assert isinstance(parsed, ast.Module)
        body = list(parsed.body)
        eval_code = None

        if body and isinstance(body[-1], ast.Expr):
            expr = body.pop()
            exec_module = ast.Module(body=body, type_ignores=getattr(parsed, "type_ignores", []))
            ast.fix_missing_locations(exec_module)
            exec_code = compile(exec_module, "<sandbox>", "exec")
            expression = ast.Expression(expr.value)
            ast.fix_missing_locations(expression)
            eval_code = compile(expression, "<sandbox>", "eval")
        else:
            ast.fix_missing_locations(parsed)
            exec_code = compile(parsed, "<sandbox>", "exec")

        return exec_code, eval_code

    def _execute(self, exec_code: Any, eval_code: Any | None) -> _ExecutionArtifacts:
        globals_dict: MutableMapping[str, Any] = {**self._globals, "__builtins__": dict(self._builtins)}
        locals_dict: dict[str, Any] = {}

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        started = time.perf_counter()
        result_value: Any = None

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                exec(exec_code, globals_dict, locals_dict)
                if eval_code is not None:
                    result_value = eval(eval_code, globals_dict, locals_dict)
        except SandboxViolationError:
            raise
        except Exception as exc:  # pragma: no cover - bubbled to caller for clarity
            raise PythonInterpreterExecutionError(str(exc)) from exc
        finally:
            duration_ms = (time.perf_counter() - started) * 1000

        stdout_value, stdout_truncated = _truncate_output(stdout_buffer.getvalue())
        stderr_value, stderr_truncated = _truncate_output(stderr_buffer.getvalue())
        result_text, result_type = _stringify_result(result_value)

        return _ExecutionArtifacts(
            stdout=stdout_value,
            stderr=stderr_value,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            result=result_text,
            result_type=result_type,
            duration_ms=duration_ms,
        )

    def _sanitize_timeout(self, timeout: float | None) -> float:
        if timeout is None:
            return self._default_timeout
        try:
            value = float(timeout)
        except (TypeError, ValueError) as exc:
            raise PythonInterpreterError("Timeout must be numeric.") from exc
        if value <= 0:
            raise PythonInterpreterError("Timeout must be greater than zero.")
        return value
