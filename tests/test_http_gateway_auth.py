import base64
import json
import sys
import types
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import pytest
from fastapi.testclient import TestClient


def _encode_basic(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8"))
    return token.decode("ascii")


class _AccountLockedError(RuntimeError):
    def __init__(
        self,
        username: str,
        *,
        retry_at: Any | None = None,
        retry_after: int | None = None,
    ) -> None:
        self.username = username
        self.retry_at = retry_at
        self.retry_after = retry_after
        message = "Too many failed login attempts. Please try again later."
        if retry_after is not None and retry_after > 0:
            plural = "s" if retry_after != 1 else ""
            message = f"Too many failed login attempts. Try again in {retry_after} second{plural}."
        super().__init__(message)


class _StubUserAccountService:
    def __init__(self, locked_error_cls: type[Exception]) -> None:
        self._users: Dict[str, Dict[str, Any]] = {}
        self._locked_error_cls = locked_error_cls

    def add_user(
        self,
        username: str,
        password: str,
        *,
        roles: Iterable[str] | None = None,
        locked: bool = False,
        **details: Any,
    ) -> None:
        normalized_roles = tuple(str(role) for role in roles) if roles else ()
        payload = dict(details)
        if normalized_roles:
            payload.setdefault("roles", list(normalized_roles))
        payload.setdefault("username", username)
        self._users[username] = {
            "password": password,
            "roles": normalized_roles,
            "locked": locked,
            "details": payload,
        }

    def authenticate_user(self, username: str, password: str) -> bool:
        record = self._users.get(username)
        if record is None:
            return False
        if record.get("locked"):
            raise self._locked_error_cls(username, retry_after=60)
        return record.get("password") == password

    def get_user_details(self, username: str) -> Optional[Mapping[str, Any]]:
        record = self._users.get(username)
        if record is None:
            return None
        return dict(record.get("details") or {})


class _StubFacade:
    def __init__(self, service: _StubUserAccountService) -> None:
        self._service = service

    def _get_user_account_service(self) -> _StubUserAccountService:
        return self._service


class _StubConversationRepository:
    def __init__(self) -> None:
        self._profiles: Dict[str, Dict[str, Any]] = {}

    def set_profile(
        self,
        username: str,
        *,
        profile: Mapping[str, Any] | None = None,
        documents: Mapping[str, Any] | None = None,
        display_name: Optional[str] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "profile": dict(profile or {}),
            "documents": dict(documents or {}),
        }
        if display_name:
            record["display_name"] = display_name
        self._profiles[username] = record

    def get_user_profile(
        self,
        username: str,
        *,
        tenant_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        record = self._profiles.get(username)
        if record is None:
            return None
        result: Dict[str, Any] = {
            "profile": dict(record.get("profile") or {}),
            "documents": dict(record.get("documents") or {}),
        }
        if "display_name" in record:
            result["display_name"] = record["display_name"]
        return result


class _StubAtlas:
    def __init__(
        self,
        service: _StubUserAccountService,
        repository: _StubConversationRepository,
        roles: Tuple[str, ...] = (),
    ) -> None:
        self.user_account_facade = _StubFacade(service)
        self.conversation_repository = repository
        self.config_manager = None
        self.message_bus = None
        self.job_service = None
        self.job_manager = None
        self.job_scheduler = None
        self.tenant_id = "tenant"
        self._initialized = False
        self._active_roles = roles

    async def initialize(self) -> None:
        self._initialized = True

    async def close(self) -> None:
        self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    def _resolve_active_user_roles(self) -> Tuple[str, ...]:
        return self._active_roles


class _StubServer:
    def __init__(self, **_: Any) -> None:
        self.calls: list[tuple[str, Any]] = []

    def _record(self, route: str, context: Any) -> Dict[str, Any]:
        self.calls.append((route, context))
        return {"route": route, "ok": True}

    def list_conversations(self, *, context: Any, params: Any = None) -> Dict[str, Any]:
        return self._record("list_conversations", context)

    def list_tasks(self, *, context: Any, params: Any = None) -> Dict[str, Any]:
        return self._record("list_tasks", context)

    def list_jobs(self, *, context: Any, params: Any = None) -> Dict[str, Any]:
        return self._record("list_jobs", context)


@pytest.fixture()
def http_client(monkeypatch: pytest.MonkeyPatch):
    module_name = "modules.user_accounts.user_account_service"
    original_module = sys.modules.get(module_name)
    stub_module = types.ModuleType(module_name)

    stub_module.AccountLockedError = _AccountLockedError
    sys.modules[module_name] = stub_module

    atlas_module_name = "ATLAS.ATLAS"
    original_atlas_module = sys.modules.get(atlas_module_name)
    atlas_stub_module = types.ModuleType(atlas_module_name)

    class _BaselineAtlas:
        def __init__(self) -> None:
            self.tenant_id = "tenant"
            self.config_manager = None
            self.conversation_repository = None
            self.message_bus = None
            self.job_service = None
            self.job_manager = None
            self.job_scheduler = None

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            return None

        def is_initialized(self) -> bool:
            return True

    atlas_stub_module.ATLAS = _BaselineAtlas
    sys.modules[atlas_module_name] = atlas_stub_module

    from server import http_gateway

    service = _StubUserAccountService(_AccountLockedError)
    repository = _StubConversationRepository()
    atlas_holder: Dict[str, _StubAtlas] = {}
    server_holder: Dict[str, _StubServer] = {}

    def atlas_factory() -> _StubAtlas:
        atlas = _StubAtlas(service, repository)
        atlas_holder["atlas"] = atlas
        return atlas

    def server_factory(**kwargs: Any) -> _StubServer:
        server = _StubServer(**kwargs)
        server_holder["server"] = server
        return server

    monkeypatch.setattr(http_gateway, "ATLAS", atlas_factory)
    monkeypatch.setattr(http_gateway, "AtlasServer", server_factory)

    try:
        with TestClient(http_gateway.app) as client:
            client.headers.update({http_gateway._HEADER_TENANT: "tenant"})
            server = server_holder.get("server")
            atlas = atlas_holder.get("atlas")
            assert server is not None
            assert atlas is not None
            yield (
                client,
                service,
                repository,
                server,
                atlas,
                http_gateway,
            )
    finally:
        if original_module is not None:
            sys.modules[module_name] = original_module
        else:
            sys.modules.pop(module_name, None)
        if original_atlas_module is not None:
            sys.modules[atlas_module_name] = original_atlas_module
        else:
            sys.modules.pop(atlas_module_name, None)


def test_conversations_require_authentication(http_client) -> None:
    client, _, __, server, _, http_gateway = http_client
    response = client.get("/conversations")
    assert response.status_code == 401
    assert server.calls == []


def test_tasks_require_authentication(http_client) -> None:
    client, _, __, server, _, http_gateway = http_client
    response = client.get("/tasks")
    assert response.status_code == 401
    assert server.calls == []


def test_basic_auth_success_populates_context(http_client) -> None:
    client, service, repository, server, atlas, http_gateway = http_client
    service.add_user(
        "alice",
        "wonderland",
        email="alice@example.com",
        name="Alice",
        roles=("admin",),
        display_name="Alice Liddell",
    )
    repository.set_profile(
        "alice",
        profile={"roles": ["admin", "analyst"], "team": "insights"},
        documents={"bio": "Explorer"},
        display_name="Alice Example",
    )

    headers = {
        "Authorization": f"Basic {_encode_basic('alice', 'wonderland')}",
        http_gateway._HEADER_METADATA: json.dumps({"source": "test"}),
    }

    response = client.get("/conversations", headers=headers)

    assert response.status_code == 200
    assert server.calls[-1][0] == "list_conversations"
    context = server.calls[-1][1]
    assert context.user_id == "alice"
    assert tuple(context.roles) == ("admin", "analyst")
    metadata = context.metadata or {}
    assert metadata["tenant_id"] == atlas.tenant_id
    assert metadata["user"]["email"] == "alice@example.com"
    assert metadata["profile"]["team"] == "insights"
    assert metadata["user_display_name"] == "Alice Liddell"
    assert metadata["source"] == "test"


def test_roles_header_is_ignored(http_client) -> None:
    client, service, _, server, _, http_gateway = http_client
    service.add_user("mallory", "pw", roles=("user",))

    headers = {
        "Authorization": f"Basic {_encode_basic('mallory', 'pw')}",
        http_gateway._HEADER_ROLES: "admin, auditor",
    }

    response = client.get("/conversations", headers=headers)

    assert response.status_code == 200
    context = server.calls[-1][1]
    assert context.user_id == "mallory"
    assert tuple(context.roles) == ("user",)


def test_invalid_credentials_rejected(http_client) -> None:
    client, service, _, server, _, http_gateway = http_client
    service.add_user("bob", "builder")

    headers = {"Authorization": f"Basic {_encode_basic('bob', 'wrong')}"}

    response = client.get("/conversations", headers=headers)

    assert response.status_code == 401
    assert server.calls == []


def test_bearer_token_uses_header_username(http_client) -> None:
    client, service, repository, server, _, http_gateway = http_client
    service.add_user("carol", "token123", name="Carol")
    repository.set_profile("carol", profile={"roles": ["user"]}, display_name="Carol Example")

    headers = {
        "Authorization": "Bearer token123",
        http_gateway._HEADER_USER: "carol",
    }

    response = client.get("/tasks", headers=headers)

    assert response.status_code == 200
    assert server.calls[-1][0] == "list_tasks"
    context = server.calls[-1][1]
    assert context.user_id == "carol"
    assert tuple(context.roles) == ("user",)


def test_mismatched_header_username_rejected(http_client) -> None:
    client, service, _, server, _, http_gateway = http_client
    service.add_user("dana", "secret")

    headers = {
        "Authorization": f"Basic {_encode_basic('dana', 'secret')}",
        http_gateway._HEADER_USER: "someone-else",
    }

    response = client.get("/conversations", headers=headers)

    assert response.status_code == 401
    assert server.calls == []


def test_locked_account_returns_403(http_client) -> None:
    client, service, _, server, _, http_gateway = http_client
    service.add_user("erin", "Password123", locked=True)

    headers = {"Authorization": f"Basic {_encode_basic('erin', 'Password123')}"}

    response = client.get("/conversations", headers=headers)

    assert response.status_code == 403
    assert server.calls == []


def test_missing_tenant_header_is_rejected(http_client) -> None:
    client, service, _, server, _, http_gateway = http_client
    service.add_user("tenantless", "pw")

    original_headers = dict(client.headers)
    client.headers.clear()

    headers = {"Authorization": f"Basic {_encode_basic('tenantless', 'pw')}"}

    response = client.get("/conversations", headers=headers)

    client.headers.update(original_headers)

    assert response.status_code == 403
    assert server.calls == []


def test_mismatched_tenant_header_is_rejected(http_client) -> None:
    client, service, _, server, _, http_gateway = http_client
    service.add_user("wrongtenant", "pw")

    headers = {
        "Authorization": f"Basic {_encode_basic('wrongtenant', 'pw')}",
        http_gateway._HEADER_TENANT: "other",
    }

    response = client.get("/tasks", headers=headers)

    assert response.status_code == 403
    assert server.calls == []


def test_tenant_header_required_across_routes(http_client) -> None:
    client, service, _, server, _, http_gateway = http_client
    service.add_user("tenantuser", "pw")

    headers = {
        "Authorization": f"Basic {_encode_basic('tenantuser', 'pw')}",
        http_gateway._HEADER_TENANT: "tenant",
    }

    response = client.get("/conversations", headers=headers)
    assert response.status_code == 200

    response = client.get("/tasks", headers=headers)
    assert response.status_code == 200

    response = client.get("/jobs", headers=headers)
    assert response.status_code == 200

    recorded_routes = [call[0] for call in server.calls[-3:]]
    assert recorded_routes == ["list_conversations", "list_tasks", "list_jobs"]


@pytest.fixture()
def http_client_with_api_key(http_client):
    client, service, repository, server, atlas, http_gateway = http_client
    http_gateway.app.state.api_key_config = http_gateway.ApiKeyConfig(
        enabled=True,
        valid_tokens=frozenset({"super-secret"}),
        public_paths=frozenset({"/healthz"}),
    )
    return client, service, repository, server, atlas, http_gateway


def test_api_key_required_when_enabled(http_client_with_api_key) -> None:
    client, _, __, server, _, __http_gateway = http_client_with_api_key

    response = client.get("/conversations")

    assert response.status_code == 401
    assert server.calls == []


def test_invalid_api_key_rejected(http_client_with_api_key) -> None:
    client, _, __, server, _, __http_gateway = http_client_with_api_key

    headers = {"X-API-Key": "not-correct"}

    response = client.get("/conversations", headers=headers)

    assert response.status_code == 403
    assert server.calls == []


def test_valid_api_key_allows_authenticated_request(http_client_with_api_key) -> None:
    client, service, repository, server, atlas, http_gateway = http_client_with_api_key
    service.add_user("api-user", "passw0rd", roles=("reader",))
    repository.set_profile("api-user", profile={"department": "it"}, display_name="API User")

    headers = {
        "Authorization": f"Basic {_encode_basic('api-user', 'passw0rd')}",
        "X-API-Key": "super-secret",
        http_gateway._HEADER_METADATA: json.dumps({"audit": True}),
    }

    response = client.get("/conversations", headers=headers)

    assert response.status_code == 200
    context = server.calls[-1][1]
    assert context.user_id == "api-user"
    assert atlas.tenant_id == context.tenant_id
