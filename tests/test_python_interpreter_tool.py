import asyncio

import pytest

from modules.Tools.Code_Execution import (
    PythonInterpreter,
    PythonInterpreterTimeoutError,
    SandboxViolationError,
)


def test_python_interpreter_executes_simple_code():
    interpreter = PythonInterpreter()
    payload = asyncio.run(
        interpreter.run(
        command="""
print("hello")
value = sum(range(5))
value
"""
        )
    )

    assert payload["stdout"] == "hello\n"
    assert payload["stderr"] == ""
    assert payload["result"] == "10"
    assert payload["result_type"] == "int"
    assert not payload["stdout_truncated"]
    assert not payload["stderr_truncated"]
    assert payload["duration_ms"] >= 0


def test_python_interpreter_enforces_timeout():
    interpreter = PythonInterpreter()
    with pytest.raises(PythonInterpreterTimeoutError):
        asyncio.run(
            interpreter.run(
            command="""
result = 0
for _ in range(10**8):
    result += 1
""",
            timeout=0.01,
        )
        )


def test_python_interpreter_rejects_sandbox_violations():
    interpreter = PythonInterpreter()
    with pytest.raises(SandboxViolationError):
        asyncio.run(interpreter.run(command="import os"))
