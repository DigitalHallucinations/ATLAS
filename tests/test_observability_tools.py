from datetime import datetime, timezone
from importlib import util as importlib_util
from pathlib import Path
import sys
import types


def _load_tool_module(name: str):
    base = Path("modules/Tools/Base_Tools")
    if "modules" not in sys.modules:
        import modules  # type: ignore  # noqa: F401
    if "modules.Tools" not in sys.modules:
        import modules.Tools  # type: ignore  # noqa: F401
    base_pkg = sys.modules.get("modules.Tools.Base_Tools")
    if base_pkg is None:
        base_pkg = types.ModuleType("modules.Tools.Base_Tools")
        base_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["modules.Tools.Base_Tools"] = base_pkg
    spec = importlib_util.spec_from_file_location(f"tests.dynamic.{name}", base / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load tool module {name}")
    module = importlib_util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.modules[f"modules.Tools.Base_Tools.{name}"] = module
    spec.loader.exec_module(module)
    return module


LogParser = _load_tool_module("log_parser").LogParser
ThreatScanner = _load_tool_module("threat_scanner").ThreatScanner
AuditReporter = _load_tool_module("audit_reporter").AuditReporter
SysSnapshot = _load_tool_module("sys_snapshot").SysSnapshot
IncidentSummarizer = _load_tool_module("incident_summarizer").IncidentSummarizer
_load_tool_module("context_tracker")
_load_tool_module("time")
_load_tool_module("log_event")
_load_tool_module("policy_reference")


def _load_specter_maps():
    base = Path("modules/Personas/Specter/Toolbox")
    parent_name = "modules.Personas.Specter.Toolbox"
    for pkg_name in ["modules.Personas", "modules.Personas.Specter", parent_name]:
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = []  # type: ignore[attr-defined]
            sys.modules[pkg_name] = pkg
    spec = importlib_util.spec_from_file_location(
        "tests.dynamic.specter_maps", base / "maps.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load Specter toolbox map module")
    module = importlib_util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.modules[f"{parent_name}.maps"] = module
    spec.loader.exec_module(module)
    return module


def test_log_parser_filters_and_limits() -> None:
    parser = LogParser()
    now = datetime.now(timezone.utc).isoformat()
    log_payload = "\n".join(
        [
            f'{{"timestamp":"{now}","severity":"error","message":"Disk failure"}}',
            "level=info message=\"Heartbeat\"",
            "malformed-line",
        ]
    )

    result = parser.run(log_source=log_payload, severities=["error"], limit=1)

    assert result["total"] == 3
    assert result["matched"] == 1
    entry = result["entries"][0]
    assert entry["severity"] == "error"
    assert entry["message"] == "Disk failure"


def test_threat_scanner_ranks_by_score() -> None:
    scanner = ThreatScanner()
    events = [
        {"timestamp": "2024-01-01T00:00:00Z", "severity": "warning", "message": "login failure", "status": "failed"},
        {"timestamp": "2024-01-01T00:01:00Z", "severity": "info", "message": "heartbeat ok"},
    ]

    result = scanner.run(events=events, indicators=["login"], min_score=0.4)

    assert result["scanned"] == 2
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["score"] >= 0.5
    assert list(finding["indicator_hits"]) == ["login"]


def test_audit_reporter_summarizes_controls() -> None:
    reporter = AuditReporter()
    events = [
        {
            "severity": "error",
            "status": "failed",
            "controls": ["AC-1"],
            "rationale": "Access control failed",
        },
        {
            "severity": "info",
            "status": "passed",
            "controls": ["AC-1", "IR-1"],
        },
    ]

    report = reporter.run(events=events, controls=["AC-1"], include_passed=False)

    assert report["totals"]["events"] == 2
    assert report["by_severity"]["error"] == 1
    assert report["controls"]["AC-1"]["failed"] == 1
    assert report["notes"] == ["Access control failed"]


def test_sys_snapshot_normalizes_hosts() -> None:
    snapshot = SysSnapshot()
    payload = snapshot.run(
        hosts=[{"hostname": "db-1", "env": "prod"}, "web-1"],
        metrics={"cpu": {"avg": 0.4}},
        tags=["prod", "critical"],
        observations=["Load is stable"],
    )

    assert payload["summary"]["host_count"] == 2
    host_names = {host["name"] for host in payload["hosts"]}
    assert host_names == {"db-1", "web-1"}
    assert payload["metrics"]["cpu"]["avg"] == 0.4
    assert payload["tags"] == ["critical", "prod"]


def test_incident_summarizer_generates_recommendations() -> None:
    summarizer = IncidentSummarizer()
    timeline = [
        {"timestamp": "2024-01-01T00:00:00Z", "description": "Alert received"},
        {"timestamp": "2024-01-01T01:00:00Z", "description": "Containment ongoing"},
    ]
    impact = {"severity": "high", "affected_assets": ["db", "api"], "customer_visible": True}
    actions = [{"category": "eradication", "status": "pending"}]

    summary = summarizer.run(timeline=timeline, impact=impact, actions=actions)

    assert "headline" in summary
    assert any("containment" in rec.lower() for rec in summary["recommendations"])
    assert any("post-incident" in rec.lower() for rec in summary["recommendations"])
    assert any("customer" in rec.lower() for rec in summary["recommendations"])


def test_specter_toolbox_maps_to_observability_tools() -> None:
    module = _load_specter_maps()
    expected_tools = {
        "log_parser",
        "threat_scanner",
        "audit_reporter",
        "sys_snapshot",
        "incident_summarizer",
        "context_tracker",
        "get_current_info",
        "log.event",
        "policy_reference",
    }

    assert expected_tools.issubset(module.function_map.keys())

    parser = module.function_map["log_parser"]
    result = parser(log_source="severity=error message=\"test\"")
    assert result["matched"] == 1
