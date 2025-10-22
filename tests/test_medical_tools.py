import asyncio
import sys
import types
from typing import Any, Dict, List

import pytest
import requests

# The logging helper depends on PyYAML; install a small stub for the test
# environment where the dependency might be unavailable.
yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda stream: {}
sys.modules.setdefault("yaml", yaml_stub)

dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *args, **kwargs: None
dotenv_stub.set_key = lambda *args, **kwargs: None
dotenv_stub.find_dotenv = lambda *args, **kwargs: ""
sys.modules.setdefault("dotenv", dotenv_stub)

sqlalchemy_stub = types.ModuleType("sqlalchemy")
sqlalchemy_stub.create_engine = lambda *args, **kwargs: None

sqlalchemy_engine_stub = types.ModuleType("sqlalchemy.engine")
sqlalchemy_engine_stub.Engine = object

sqlalchemy_orm_stub = types.ModuleType("sqlalchemy.orm")


class _SessionmakerStub:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):  # pragma: no cover - helper stub
        return None


sqlalchemy_orm_stub.sessionmaker = _SessionmakerStub

sqlalchemy_stub.engine = sqlalchemy_engine_stub
sqlalchemy_stub.orm = sqlalchemy_orm_stub

sys.modules.setdefault("sqlalchemy", sqlalchemy_stub)
sys.modules.setdefault("sqlalchemy.engine", sqlalchemy_engine_stub)
sys.modules.setdefault("sqlalchemy.orm", sqlalchemy_orm_stub)

task_store_stub = types.ModuleType("modules.task_store")
task_store_stub.TaskService = object
task_store_stub.TaskStoreRepository = object
sys.modules.setdefault("modules.task_store", task_store_stub)

from modules.Tools.Medical_Tools import fetch_pubmed_details, search_pmc, search_pubmed


class _FakeResponse:
    def __init__(self, status_code: int, payload: Dict[str, Any]):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            err = requests.HTTPError(f"status={self.status_code}")
            err.response = self
            raise err  # pragma: no cover - exercised

    def json(self) -> Dict[str, Any]:
        return self._payload


@pytest.fixture(autouse=True)
def _patch_rate_limit(monkeypatch):
    async def _noop_rate_limit():
        return None

    async def _direct_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(
        "modules.Tools.Medical_Tools._client.enforce_rate_limit",
        _noop_rate_limit,
    )
    monkeypatch.setattr(
        "modules.Tools.Medical_Tools._client.asyncio.to_thread",
        _direct_to_thread,
    )
    return monkeypatch


def test_search_pubmed_builds_expected_request(monkeypatch):
    captured: Dict[str, Any] = {}

    def _fake_get(url: str, *, params: Dict[str, Any], headers: Dict[str, str], timeout: float):
        captured.update({"url": url, "params": params, "headers": headers, "timeout": timeout})
        payload = {
            "esearchresult": {
                "count": "2",
                "idlist": ["1", "2"],
                "retstart": "0",
                "retmax": "2",
                "querytranslation": "test",
            }
        }
        return _FakeResponse(200, payload)

    monkeypatch.setattr(
        "modules.Tools.Medical_Tools._client.requests.get",
        lambda url, params, headers, timeout: _fake_get(url, params=params, headers=headers, timeout=timeout),
    )
    monkeypatch.setattr(
        "modules.Tools.Medical_Tools._client.get_auth_params",
        lambda: {"api_key": "secret"},
    )

    status, payload = asyncio.run(search_pubmed("test query", max_results=2, page_size=2))

    assert status == 200
    assert payload["ids"] == ["1", "2"]
    assert captured["url"].endswith("esearch.fcgi")
    assert captured["params"]["db"] == "pubmed"
    assert captured["params"]["term"] == "test query"
    assert captured["params"]["api_key"] == "secret"
    assert captured["params"]["retmax"] == 2
    assert captured["timeout"] == pytest.approx(15.0)


def test_search_pubmed_paginates(monkeypatch):
    call_params: List[Dict[str, Any]] = []

    def _fake_get(url: str, *, params: Dict[str, Any], headers: Dict[str, str], timeout: float):
        call_params.append(params)
        if params["retstart"] == 0:
            payload = {
                "esearchresult": {
                    "count": "5",
                    "idlist": ["1", "2", "3"],
                    "retstart": "0",
                    "retmax": "3",
                    "querytranslation": "foo",
                }
            }
        else:
            payload = {
                "esearchresult": {
                    "count": "5",
                    "idlist": ["4", "5"],
                    "retstart": "3",
                    "retmax": "2",
                    "querytranslation": "foo",
                }
            }
        return _FakeResponse(200, payload)

    monkeypatch.setattr(
        "modules.Tools.Medical_Tools._client.requests.get",
        lambda url, params, headers, timeout: _fake_get(url, params=params, headers=headers, timeout=timeout),
    )
    monkeypatch.setattr(
        "modules.Tools.Medical_Tools._client.get_auth_params",
        lambda: {},
    )

    status, payload = asyncio.run(search_pubmed("term", max_results=5, page_size=3))

    assert status == 200
    assert payload["ids"] == ["1", "2", "3", "4", "5"]
    assert len(call_params) == 2
    assert call_params[0]["retmax"] == 3
    assert call_params[1]["retmax"] == 2
    assert call_params[1]["retstart"] == 3


def test_search_pubmed_reports_http_error(monkeypatch):
    def _fake_get(url: str, *, params: Dict[str, Any], headers: Dict[str, str], timeout: float):
        return _FakeResponse(429, {"error": "rate limit"})

    monkeypatch.setattr(
        "modules.Tools.Medical_Tools._client.requests.get",
        lambda url, params, headers, timeout: _fake_get(url, params=params, headers=headers, timeout=timeout),
    )
    monkeypatch.setattr(
        "modules.Tools.Medical_Tools._client.get_auth_params",
        lambda: {},
    )

    status, payload = asyncio.run(search_pubmed("term"))

    assert status == 429
    assert payload["error"]


def test_search_pmc_applies_filters(monkeypatch):
    captured: Dict[str, Any] = {}

    def _fake_get(url: str, *, params: Dict[str, Any], headers: Dict[str, str], timeout: float):
        captured.update(params=params)
        payload = {
            "esearchresult": {
                "count": "1",
                "idlist": ["PMC123"],
                "retstart": "0",
                "retmax": "1",
                "querytranslation": "pmc",
            }
        }
        return _FakeResponse(200, payload)

    monkeypatch.setattr(
        "modules.Tools.Medical_Tools._client.requests.get",
        lambda url, params, headers, timeout: _fake_get(url, params=params, headers=headers, timeout=timeout),
    )
    monkeypatch.setattr(
        "modules.Tools.Medical_Tools._client.get_auth_params",
        lambda: {},
    )

    status, payload = asyncio.run(
        search_pmc("covid", article_type="clinicaltrial", has_abstract=True)
    )

    assert status == 200
    assert payload["ids"] == ["PMC123"]
    assert captured["params"]["db"] == "pmc"
    assert captured["params"]["articletype"] == "clinicaltrial"
    assert captured["params"]["hasabstract"] == "y"


def test_fetch_pubmed_details_batches_and_merges(monkeypatch):
    captured: List[Dict[str, Any]] = []

    def _fake_get(url: str, *, params: Dict[str, Any], headers: Dict[str, str], timeout: float):
        captured.append(params)
        pmids = params["id"].split(",")
        payload: Dict[str, Any] = {"result": {"uids": pmids}}
        for pmid in pmids:
            payload["result"][pmid] = {"title": f"Title {pmid}", "abstract": f"Abstract {pmid}"}
        return _FakeResponse(200, payload)

    monkeypatch.setattr(
        "modules.Tools.Medical_Tools._client.requests.get",
        lambda url, params, headers, timeout: _fake_get(url, params=params, headers=headers, timeout=timeout),
    )
    monkeypatch.setattr(
        "modules.Tools.Medical_Tools._client.get_auth_params",
        lambda: {},
    )

    status, payload = asyncio.run(
        fetch_pubmed_details(["100", "200", "300"], chunk_size=2, retmode="json", rettype="abstract")
    )

    assert status == 200
    assert len(captured) == 2
    assert captured[0]["id"] == "100,200"
    assert captured[1]["id"] == "300"
    assert payload["ids"] == ["100", "200", "300"]
    assert len(payload["records"]) == 3
    assert all(record["pmid"] in {"100", "200", "300"} for record in payload["records"])


def test_fetch_pubmed_details_requires_identifiers():
    status, payload = asyncio.run(fetch_pubmed_details([]))

    assert status == -1
    assert "error" in payload
