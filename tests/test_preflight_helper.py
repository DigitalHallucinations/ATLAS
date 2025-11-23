import json
import sys

import pytest

gi = pytest.importorskip("gi")

import subprocess

from GTKUI.Setup.preflight import (
    PreflightCheckDefinition,
    PreflightHelper,
    SQLITE_PATH_REMEDIATION,
)
from ATLAS.setup import DatabaseState, MessageBusState


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


def test_preflight_helper_runs_only_custom_checks(monkeypatch):
    checks = [
        PreflightCheckDefinition(
            identifier="custom-1",
            label="Custom One",
            command=["custom-one"],
            success_message="Custom one passed",
            failure_hint="Custom one failed",
        ),
        PreflightCheckDefinition(
            identifier="custom-2",
            label="Custom Two",
            command=["custom-two"],
            success_message="Custom two passed",
            failure_hint="Custom two failed",
        ),
    ]

    factory = _ProcessFactory()
    for definition in checks:
        factory.enqueue(definition.command, _FakeProcess(exit_status=0))

    helper = PreflightHelper(
        request_password=lambda: None,
        checks=checks,
        subprocess_factory=factory,
    )

    monkeypatch.setattr(
        helper, "_default_checks", lambda: (_ for _ in ()).throw(AssertionError("default checks used"))
    )

    completed: list = []
    helper.run_checks(on_complete=lambda results: completed.extend(results))

    assert [result.identifier for result in completed] == [definition.identifier for definition in checks]
    assert all(result.passed for result in completed)


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


def test_failed_fix_releases_lock_for_followup_operations():
    definition = PreflightCheckDefinition(
        identifier="db",
        label="Database",
        command=["check-db"],
        success_message="Database ready",
        failure_hint="Database is unavailable",
        fix_command=["fix-db"],
        fix_label="Start database",
    )

    factory = _ProcessFactory()
    factory.enqueue(["fix-db"], _FakeProcess(exit_status=1, stderr="still broken"))

    helper = PreflightHelper(
        request_password=lambda: None,
        checks=[definition],
        subprocess_factory=factory,
    )

    failed_fix_results: list = []
    helper.run_fix("db", failed_fix_results.append)

    assert failed_fix_results
    assert failed_fix_results[-1].passed is False
    assert helper._fix_in_progress is None

    followup_checks: list = []
    factory.enqueue(["check-db"], _FakeProcess(exit_status=0, stdout="ok"))
    helper.run_checks(on_complete=lambda results: followup_checks.extend(results))

    assert followup_checks
    assert all(result.passed for result in followup_checks)

    factory.enqueue(["fix-db"], _FakeProcess(exit_status=0, stdout="recovered"))
    factory.enqueue(["check-db"], _FakeProcess(exit_status=0, stdout="ready"))

    successful_fix_results: list = []
    helper.run_fix("db", successful_fix_results.append)

    assert successful_fix_results
    assert successful_fix_results[-1].passed is True


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

    definitions = list(helper._default_checks())
    assert all(definition.identifier != "redis" for definition in definitions)


def test_preflight_helper_builds_redis_check_when_configured():
    factory = _ProcessFactory()
    helper = PreflightHelper(
        request_password=lambda: None,
        subprocess_factory=factory,
    )
    helper.configure_message_bus_target(
        MessageBusState(backend="redis", redis_url="redis://localhost:6379/0")
    )

    definitions = list(helper._default_checks())
    redis_definition = next(
        (definition for definition in definitions if definition.identifier == "redis"), None
    )
    assert redis_definition is not None
    assert "redis://localhost:6379/0" in redis_definition.command

    completed: list = []
    for definition in definitions:
        if definition.identifier == "hardware":
            payload = {
                "message": "Hardware review completed: mock", 
                "recommendation": "Local hosting looks sufficient for databases and moderate models.",
            }
            stdout = json.dumps(payload)
        else:
            stdout = "ok"
        factory.enqueue(definition.command, _FakeProcess(exit_status=0, stdout=stdout))

    helper.run_checks(on_complete=lambda results: completed.extend(results))

    assert any(result.identifier == "redis" and result.passed for result in completed)


def test_sqlite_check_surfaces_recommendation(tmp_path):
    target = tmp_path / "data" / "atlas.sqlite3"
    dsn = f"sqlite:///{target}"

    helper = PreflightHelper(
        request_password=lambda: None,
        subprocess_factory=_ProcessFactory(),
    )
    helper.configure_database_target(DatabaseState(backend="sqlite", dsn=dsn))

    definitions = list(helper._default_checks())
    sqlite_definition = next(
        definition for definition in definitions if definition.identifier == "sqlite"
    )
    hardware_definition = next(
        definition for definition in definitions if definition.identifier == "hardware"
    )

    factory = _ProcessFactory()
    payload = {
        "ok": False,
        "message": "Parent directory missing",
        "recommendation": SQLITE_PATH_REMEDIATION,
    }
    factory.enqueue(sqlite_definition.command, _FakeProcess(exit_status=1, stdout=json.dumps(payload)))

    hardware_payload = {
        "message": "Hardware review completed: mock",
        "recommendation": "Local hosting looks sufficient for databases and moderate models.",
    }
    factory.enqueue(
        hardware_definition.command,
        _FakeProcess(exit_status=0, stdout=json.dumps(hardware_payload)),
    )

    helper = PreflightHelper(
        request_password=lambda: None,
        checks=definitions,
        subprocess_factory=factory,
    )

    completed: list = []
    helper.run_checks(on_complete=lambda results: completed.extend(results))

    sqlite_result = next(result for result in completed if result.identifier == "sqlite")
    assert sqlite_result.passed is False
    assert "Parent directory missing" in sqlite_result.message
    assert sqlite_result.recommendation == SQLITE_PATH_REMEDIATION

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


def test_hardware_check_adds_recommendations_and_replaces_virtualenv():
    factory = _ProcessFactory()
    helper = PreflightHelper(
        request_password=lambda: None,
        subprocess_factory=factory,
    )

    definitions = list(helper._default_checks())
    identifiers = [definition.identifier for definition in definitions]
    assert "virtualenv" not in identifiers
    assert "hardware" in identifiers

    payload = {
        "message": (
            "Hardware review completed: 2 CPU cores, 2.0 GB RAM, "
            "10.0 GB free disk at /var/lib/atlas."
        ),
        "recommendation": "Consider cloud PostgreSQL; only 2.0 GB RAM available.",
    }

    for definition in definitions:
        stdout = json.dumps(payload) if definition.identifier == "hardware" else ""
        factory.enqueue(definition.command, _FakeProcess(exit_status=0, stdout=stdout))

    completed: list = []
    helper.run_checks(on_complete=lambda results: completed.extend(results))

    hardware_result = next(result for result in completed if result.identifier == "hardware")
    assert hardware_result.passed is True
    assert payload["message"] in hardware_result.message
    assert hardware_result.recommendation == payload["recommendation"]


def test_service_fixes_adjust_to_platform(monkeypatch):
    helper = PreflightHelper(request_password=lambda: None)

    helper.configure_message_bus_target(MessageBusState(backend="redis"))

    monkeypatch.setattr(
        helper, "_detect_service_manager", lambda: (None, "Manual restart only.")
    )
    definitions = list(helper._default_checks())
    redis_definition = next(defn for defn in definitions if defn.identifier == "redis")

    assert redis_definition.fix_command is None
    assert redis_definition.fix_label is not None
    assert redis_definition.fix_tooltip == "Manual restart only."
    assert "Manual restart only." in redis_definition.failure_hint
    assert redis_definition.fix_available is False


def test_configure_message_bus_target_disables_redis_for_other_backends():
    helper = PreflightHelper(request_password=lambda: None)

    helper.configure_message_bus_target(MessageBusState(backend="redis"))
    redis_definitions = [
        definition
        for definition in helper._default_checks()
        if definition.identifier == "redis"
    ]
    assert redis_definitions

    helper.configure_message_bus_target(MessageBusState(backend="in_memory"))
    identifiers = [definition.identifier for definition in helper._default_checks()]
    assert "redis" not in identifiers


def test_mongodb_preflight_reports_missing_driver():
    helper = PreflightHelper(request_password=lambda: None)

    helper.configure_database_target(DatabaseState(backend="mongodb"))
    definition = helper._build_database_check()

    assert definition is not None
    assert definition.identifier == "mongodb"
    message, recommendation = definition.process_output(
        False,
        "",
        "pymongo import failed: No module named 'pymongo'",
        1,
    )

    assert "pip install pymongo" in message
    assert recommendation is not None and "pip install pymongo" in recommendation
    assert definition.fix_label == "Install PyMongo driver"
    assert definition.fix_tooltip is not None
    assert definition.fix_available is True
    assert definition.fix_command is not None
    assert sys.executable in definition.fix_command


def _run_sqlite_probe(helper: PreflightHelper, expected_path: str) -> tuple[bool, str, str | None]:
    definition = helper._build_database_check()
    assert definition is not None
    assert definition.identifier == "sqlite"

    completed = subprocess.run(definition.command, capture_output=True, text=True)
    passed = completed.returncode == 0
    message, recommendation = definition.process_output(
        passed, completed.stdout, completed.stderr, completed.returncode
    )
    assert expected_path in " ".join(definition.command)
    return passed, message, recommendation


def test_sqlite_preflight_reports_missing_parent(tmp_path):
    helper = PreflightHelper(request_password=lambda: None)
    missing_parent = tmp_path / "nope" / "atlas.sqlite3"
    helper.configure_database_target(DatabaseState(backend="sqlite", database=str(missing_parent)))

    passed, message, recommendation = _run_sqlite_probe(helper, str(missing_parent))

    assert passed is False
    assert "Parent directory" in message
    assert recommendation == SQLITE_PATH_REMEDIATION


def test_sqlite_preflight_passes_when_writable(tmp_path):
    helper = PreflightHelper(request_password=lambda: None)
    sqlite_path = tmp_path / "atlas.sqlite3"
    helper.configure_database_target(DatabaseState(backend="sqlite", database=str(sqlite_path)))

    passed, message, recommendation = _run_sqlite_probe(helper, str(sqlite_path))

    assert passed is True
    assert "SQLite path ready" in message
    assert recommendation is None
