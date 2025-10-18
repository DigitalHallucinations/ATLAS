from __future__ import annotations

import asyncio
import base64
import sys
import textwrap
from pathlib import Path

import pytest

from modules.Tools.Code_Execution import (
    JavaScriptExecutionTimeoutError,
    JavaScriptExecutor,
    JavaScriptSandboxError,
)


@pytest.fixture
def sandbox_runtime(tmp_path: Path) -> Path:
    runtime = tmp_path / "fake_js_runtime.py"
    runtime.write_text(
        textwrap.dedent(
            """
            #!/usr/bin/env python3
            import sys
            import time
            from pathlib import Path

            def _parse_directives(source: str):
                mode = "success"
                reads = []
                big_file = None
                for raw_line in source.splitlines():
                    line = raw_line.strip()
                    if line.startswith("// MODE:"):
                        mode = line.split(":", 1)[1].strip().lower()
                    elif line.startswith("// READ:"):
                        reads.append(line.split(":", 1)[1].strip())
                    elif line.startswith("// BIG_FILE:"):
                        try:
                            big_file = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            big_file = None
                return mode, reads, big_file

            def _render_console_logs(source: str):
                for raw_line in source.splitlines():
                    line = raw_line.strip()
                    if not line.startswith("console.log("):
                        continue
                    payload = line[len("console.log("):]
                    if payload.endswith(";"):
                        payload = payload[:-1]
                    if payload.endswith(")"):
                        payload = payload[:-1]
                    payload = payload.strip()
                    if payload.startswith("\"") and payload.endswith("\""):
                        payload = payload[1:-1]
                    elif payload.startswith("'") and payload.endswith("'"):
                        payload = payload[1:-1]
                    print(payload)

            def main():
                if len(sys.argv) < 2:
                    return 1

                script_path = Path(sys.argv[-1])
                source = script_path.read_text(encoding="utf-8")
                mode, reads, big_file = _parse_directives(source)

                if mode == "timeout":
                    time.sleep(5)
                    return 0

                if mode == "sandbox":
                    print("Sandbox violation: denied syscall", file=sys.stderr)
                    return 66

                if mode == "stderr":
                    print("stderr output " * 400, file=sys.stderr)
                    print("stdout output " * 400)
                else:
                    _render_console_logs(source)
                    for relative in reads:
                        target = script_path.parent / relative
                        if target.exists():
                            print(target.read_text(encoding="utf-8"))
                    print("debug info", file=sys.stderr)

                artifact = script_path.parent / "build" / "output.txt"
                artifact.parent.mkdir(exist_ok=True)
                if big_file:
                    artifact.write_text("X" * big_file, encoding="utf-8")
                else:
                    artifact.write_text("artifact-data", encoding="utf-8")

                return 0

            if __name__ == "__main__":
                sys.exit(main())
            """
        ),
        encoding="utf-8",
    )
    runtime.chmod(0o755)
    return runtime


def _create_executor(runtime: Path, **overrides) -> JavaScriptExecutor:
    kwargs = dict(
        executable=sys.executable,
        args=[str(runtime)],
        default_timeout=1.0,
        max_output_bytes=512,
        max_file_bytes=512,
        max_files=4,
        cpu_time_limit=None,
    )
    kwargs.update(overrides)
    return JavaScriptExecutor(**kwargs)


def test_execute_javascript_success(sandbox_runtime: Path) -> None:
    executor = _create_executor(sandbox_runtime)
    result = asyncio.run(
        executor.run(
            command="""
            console.log('hello world');
            // READ:input.txt
            """,
            files=[{"path": "input.txt", "content": "42"}],
        )
    )

    assert "hello world" in result["stdout"]
    assert "42" in result["stdout"]
    assert result["stderr"].strip() == "debug info"
    assert result["exit_code"] == 0
    assert not result["stdout_truncated"]
    assert not result["stderr_truncated"]
    assert not result["files_truncated"]

    files = result["files"]
    assert len(files) == 1
    artifact = files[0]
    assert artifact["path"] == "build/output.txt"
    decoded = base64.b64decode(artifact["content"]).decode("utf-8")
    assert decoded == "artifact-data"


def test_execute_javascript_timeout(sandbox_runtime: Path) -> None:
    executor = _create_executor(sandbox_runtime)
    with pytest.raises(JavaScriptExecutionTimeoutError):
        asyncio.run(executor.run(command="// MODE: timeout", timeout=0.2))


def test_execute_javascript_sandbox_violation(sandbox_runtime: Path) -> None:
    executor = _create_executor(sandbox_runtime)
    with pytest.raises(JavaScriptSandboxError):
        asyncio.run(executor.run(command="// MODE: sandbox"))


def test_execute_javascript_output_truncation(sandbox_runtime: Path) -> None:
    executor = _create_executor(
        sandbox_runtime,
        max_output_bytes=64,
        max_file_bytes=16,
        max_files=1,
    )
    result = asyncio.run(
        executor.run(
            command="""
            // MODE: stderr
            // BIG_FILE:128
            """,
        )
    )

    assert result["stdout_truncated"]
    assert result["stderr_truncated"]
    assert result["files_truncated"]
    assert result["files"], "Expected at least one captured artifact"
    artifact = result["files"][0]
    assert artifact["truncated"] is True
    decoded = base64.b64decode(artifact["content"]).decode("utf-8")
    assert set(decoded) == {"X"}
