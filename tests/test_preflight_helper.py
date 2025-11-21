import pytest

gi = pytest.importorskip("gi")

from GTKUI.Setup.preflight import (
    PreflightCheckDefinition,
    PreflightHelper,
)
from ATLAS.setup import MessageBusState


class _FakeProcess:
    def __init__(self, *, exit_status=0, stdout="", stderr="") -> None:
        self.exit_status = exit_status
        self.stdout = stdout
        self.stderr = stderr
        self.stdin_data = None

    def communicate_utf8_async(self, stdin_data, _cancellable, callback):
        self.stdin_data = stdin_data
        callback(self, object())

    def communicate_utf8_finish(self, _result):
        return (self.exit_status == 0, self.stdout, self.stderr)

    def get_exit_status(self) -> int:
        return self.exit_status


class _ProcessFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], int]] = []
        self._queue: list[tuple[tuple[str, ...], _FakeProcess]] = []

    def enqueue(self, command, process: _FakeProcess) -> None:
        self._queue.append((tuple(command), process))

    def __call__(self, command, flags):
        call = (tuple(command), int(flags))
        self.calls.append(call)
        if not self._queue:
            raise AssertionError("No subprocess queued for command")
        expected, process = self._queue.pop(0)
        assert expected == call[0]
        return process


def test_preflight_helper_runs_checks_and_reports_failures():
    checks = [
        PreflightCheckDefinition(
            identifier="db",
            label="Database",
            command=["check-db"],
            success_message="Database ready",
            failure_hint="Database is unavailable",
        ),
        PreflightCheckDefinition(
            identifier="redis",
            label="Redis",
            command=["check-redis"],
            success_message="Redis ready",
            failure_hint="Redis is unavailable",
        ),
    ]

    factory = _ProcessFactory()
    factory.enqueue(["check-db"], _FakeProcess(exit_status=0))
    factory.enqueue(
        ["check-redis"],
        _FakeProcess(exit_status=2, stdout="timeout", stderr="connection refused"),
    )

    helper = PreflightHelper(
        request_password=lambda: None,
        checks=checks,
        subprocess_factory=factory,
    )

    updates: list = []
    completed: list = []

    helper.run_checks(on_update=updates.append, on_complete=lambda results: completed.extend(results))

    assert len(updates) == 2
    assert len(completed) == 2
    assert completed[0].passed is True
    assert completed[0].message == "Database ready"

    redis_result = completed[1]
    assert redis_result.passed is False
    assert "Redis is unavailable" in redis_result.message
    assert "timeout" in redis_result.message
    assert "connection refused" in redis_result.message
    assert "Exit status" in redis_result.message


def test_preflight_helper_fix_rechecks_after_success():
    definition = PreflightCheckDefinition(
        identifier="db",
        label="Database",
        command=["check-db"],
        success_message="Database ready",
        failure_hint="Database is unavailable",
        fix_command=["fix-db"],
        fix_label="Start database",
        requires_sudo=True,
    )

    factory = _ProcessFactory()
    check_failure = _FakeProcess(exit_status=1, stderr="offline")
    fix_process = _FakeProcess(exit_status=0)
    check_success = _FakeProcess(exit_status=0)

    factory.enqueue(["check-db"], check_failure)
    factory.enqueue(["fix-db"], fix_process)
    factory.enqueue(["check-db"], check_success)

    helper = PreflightHelper(
        request_password=lambda: "hunter2",
        checks=[definition],
        subprocess_factory=factory,
    )

    completed: list = []
    helper.run_checks(on_complete=lambda results: completed.extend(results))
    assert completed[0].passed is False

    fix_results: list = []
    helper.run_fix("db", lambda result: fix_results.append(result))

    assert fix_process.stdin_data == "hunter2\n"
    assert len(fix_results) == 1
    assert fix_results[0].passed is True
    assert fix_results[0].message == "Database ready"


def test_preflight_helper_handles_spawn_errors():
    definition = PreflightCheckDefinition(
        identifier="redis",
        label="Redis",
        command=["check-redis"],
        success_message="Redis ready",
        failure_hint="Redis is unavailable",
    )

    def _failing_factory(command, _flags):
        raise RuntimeError("missing binary")

    helper = PreflightHelper(
        request_password=lambda: None,
        checks=[definition],
        subprocess_factory=_failing_factory,
    )

    completed: list = []
    helper.run_checks(on_complete=lambda results: completed.extend(results))

    assert len(completed) == 1
    assert completed[0].passed is False
    assert "Unable to execute" in completed[0].message


def test_preflight_helper_honors_redis_url():
    factory = _ProcessFactory()
    helper = PreflightHelper(
        request_password=lambda: None,
        subprocess_factory=factory,
    )

    helper.configure_message_bus_target(
        MessageBusState(backend="redis", redis_url="redis://cache:6379/1")
    )

    definitions = list(helper._default_checks())
    for definition in definitions:
        factory.enqueue(definition.command, _FakeProcess(exit_status=0))

    completed: list = []
    helper.run_checks(on_complete=lambda results: completed.extend(results))

    redis_commands = [call for call, _flags in factory.calls if "redis-cli" in call]
    assert any("-u" in command and "redis://cache:6379/1" in command for command in redis_commands)
    assert len(completed) == len(definitions)
