import asyncio
from pathlib import Path

import pytest

from modules.Tools.Base_Tools.api_connector import APIConnector, APIConnectorError
from modules.Tools.Base_Tools.file_ingest import FileIngestor, FileIngestError
from modules.Tools.Base_Tools.stream_monitor import StreamMonitor, StreamMonitorError
from modules.Tools.Base_Tools.schema_infer import SchemaInference, SchemaInferenceError
from modules.Tools.Base_Tools.data_bridge import DataBridge, DataBridgeError


def _run(coro):
    return asyncio.run(coro)


def test_api_connector_dry_run_includes_headers():
    connector = APIConnector()
    result = _run(
        connector.run(
            endpoint="https://example.com/v1/health",
            headers={"X-Request-ID": "abc123"},
        )
    )
    assert result["dry_run"] is True
    assert result["request"]["headers"]["X-Request-ID"] == "abc123"
    assert result["request"]["method"] == "GET"


def test_api_connector_domain_allowlist():
    connector = APIConnector(allowed_domains=["allowed.example"])
    with pytest.raises(APIConnectorError):
        _run(connector.run(endpoint="https://forbidden.example/path"))


def test_api_connector_executes_when_requested(monkeypatch):
    connector = APIConnector()

    class _StubResponse:
        status_code = 200
        headers = {"Content-Type": "application/json"}

        @property
        def content(self):
            return b"{\"ok\": true}"

    async def fake_to_thread(func, summary, body_payload):
        return func(summary, body_payload)

    def fake_request(method, url, params=None, headers=None, data=None, timeout=None):
        assert method == "POST"
        assert url == "https://example.com/ingest"
        return _StubResponse()

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(connector._session, "request", fake_request)

    result = _run(
        connector.run(
            endpoint="https://example.com/ingest",
            method="POST",
            body={"sample": True},
            dry_run=False,
        )
    )
    assert result["response"]["status_code"] == 200
    assert "body_preview" in result["response"]


def test_file_ingest_returns_text_preview(tmp_path: Path):
    target = tmp_path / "sample.txt"
    target.write_text("hello world", encoding="utf-8")

    ingestor = FileIngestor()
    payload = _run(ingestor.run(path=str(target)))
    assert payload["preview"]["encoding"] == "text"
    assert "hello" in payload["preview"]["content"]


def test_file_ingest_rejects_missing(tmp_path: Path):
    ingestor = FileIngestor()
    with pytest.raises(FileIngestError):
        _run(ingestor.run(path=str(tmp_path / "missing.txt")))


def test_stream_monitor_rolls_up_metrics():
    monitor = StreamMonitor()
    events = [
        {"status": "ok", "latency_ms": 12.5},
        {"status": "error", "latency_ms": 30, "details": "timeout"},
        {"status": "ok", "latency_ms": 10},
    ]
    summary = _run(monitor.run(stream_id="daily-feed", events=events))
    assert summary["status_counts"]["ok"] == 2
    assert summary["status_counts"]["error"] == 1
    assert summary["latency_ms"] > 0


def test_stream_monitor_requires_stream_id():
    monitor = StreamMonitor()
    with pytest.raises(StreamMonitorError):
        _run(monitor.run(stream_id="", events=[]))


def test_schema_infer_types_and_examples():
    inference = SchemaInference(max_examples=2)
    records = [
        {"id": 1, "name": "alpha", "score": 1.5},
        {"id": 2, "name": None, "score": 2.0},
    ]
    schema = _run(inference.run(records=records))
    field_names = {field["name"] for field in schema["fields"]}
    assert field_names == {"id", "name", "score"}
    id_field = next(field for field in schema["fields"] if field["name"] == "id")
    assert "integer" in id_field["types"]


def test_schema_infer_requires_records():
    inference = SchemaInference()
    with pytest.raises(SchemaInferenceError):
        _run(inference.run(records=[]))


def test_data_bridge_executes_pipeline(tmp_path: Path):
    bridge = DataBridge()
    sample_file = tmp_path / "payload.json"
    sample_file.write_text("{\"id\": 1}", encoding="utf-8")

    operations = [
        {"tool": "file_ingest", "params": {"path": str(sample_file)}},
        {"tool": "schema_infer", "params": {"records": [{"id": 1}, {"id": 2}]}}
    ]
    result = _run(bridge.run(source="crm-delta", operations=operations, dry_run=True))
    assert result["source"] == "crm-delta"
    assert len(result["steps"]) == 2
    assert result["steps"][0]["tool"] == "file_ingest"


def test_data_bridge_requires_operations():
    bridge = DataBridge()
    with pytest.raises(DataBridgeError):
        _run(bridge.run(source="example", operations=[], dry_run=True))
