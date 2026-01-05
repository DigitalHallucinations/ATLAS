import asyncio
import inspect
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest

from core.config import ConfigManager
from modules.user_accounts import user_account_facade as user_account_facade_module
from modules.user_accounts import user_account_service as user_account_service_module
from modules.user_accounts.user_account_facade import (
    LEGACY_IDENTITY_OVERRIDE_FLAG,
    UserAccountFacade,
)


class _RepositoryStub:
    def __init__(self):
        self.profiles: dict[str, dict[str, object]] = {}

    def get_user_profile(self, username: str):
        return self.profiles.get(username)

    def list_user_profiles(self):
        return list(self.profiles.values())


class _ServiceStub:
    def __init__(self, *_args, **_kwargs):
        self.active_user: str | None = None
        self.users: list[dict[str, object]] = []
        self.authenticate_result = True
        self.delete_user_result = True
        self.password_reset_challenge = None
        self.verify_password_reset_token_result = True
        self.complete_password_reset_result = True
        self.set_active_calls: list[str | None] = []
        self.authenticate_calls: list[tuple[str, str, Optional[str]]] = []
        self.delete_calls: list[str] = []
        self.initiate_password_reset_calls: list[str] = []
        self.verify_password_reset_token_calls: list[tuple[str, str]] = []
        self.complete_password_reset_calls: list[tuple[str, str, str]] = []

    # synchronous helpers used by the facade
    def get_active_user(self):
        return self.active_user

    def list_users(self):
        return list(self.users)

    def set_active_user(self, username):
        self.set_active_calls.append(username)
        self.active_user = username

    def authenticate_user(self, username, password, *, tenant_id=None):
        self.authenticate_calls.append((username, password, tenant_id))
        return self.authenticate_result

    def delete_user(self, username):
        self.delete_calls.append(username)
        return self.delete_user_result

    def initiate_password_reset(self, identifier):
        self.initiate_password_reset_calls.append(identifier)
        challenge = self.password_reset_challenge
        if isinstance(challenge, Exception):
            raise challenge
        return challenge

    def verify_password_reset_token(self, username: str, token: str) -> bool:
        self.verify_password_reset_token_calls.append((username, token))
        result = self.verify_password_reset_token_result
        if isinstance(result, Exception):
            raise result
        return result

    def complete_password_reset(self, username: str, token: str, new_password: str) -> bool:
        self.complete_password_reset_calls.append((username, token, new_password))
        result = self.complete_password_reset_result
        if isinstance(result, Exception):
            raise result
        return result

    def register_user(self, username, password, email, name=None, dob=None):
        account = SimpleNamespace(
            id=len(self.users) + 1,
            username=username,
            email=email,
            name=name,
            dob=dob,
        )
        self.users.append({"username": username, "name": name, "email": email, "dob": dob})
        return account


@pytest.fixture(autouse=True)
def _patch_run_async(monkeypatch):
    async def immediate(func, *args, **kwargs):
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    monkeypatch.setattr(user_account_facade_module, "run_async_in_thread", immediate)


@pytest.fixture
def service_stub(monkeypatch) -> _ServiceStub:
    stub = _ServiceStub()

    def factory(*_args, **_kwargs):
        return stub

    monkeypatch.setattr(user_account_service_module, "UserAccountService", factory)
    return stub


@pytest.fixture
def facade(service_stub):
    config = Mock(spec=ConfigManager)
    config.get_config.return_value = False
    logger = Mock()
    repository = _RepositoryStub()
    return UserAccountFacade(
        config_manager=config,
        conversation_repository=repository,
        logger=logger,
    )


def test_refresh_identity_notifies_listeners(facade: UserAccountFacade, service_stub: _ServiceStub):
    repository = facade._conversation_repository
    assert isinstance(repository, _RepositoryStub)

    repository.profiles["alice"] = {"username": "alice", "display_name": "Alice Repo"}
    service_stub.active_user = "alice"
    service_stub.users = [{"username": "alice", "name": "Alice Service"}]

    calls: list[tuple[str, str]] = []

    def listener(username: str, display_name: str) -> None:
        calls.append((username, display_name))

    facade.add_active_user_change_listener(listener)
    initial_length = len(calls)

    repository.profiles["alice"]["display_name"] = "Alice Repo Updated"
    service_stub.users[0]["name"] = "Alice Service Updated"

    username, display_name = facade.refresh_active_user_identity()

    assert username == "alice"
    assert display_name == "Alice Repo Updated"
    assert facade.ensure_user_identity() == ("alice", "Alice Repo Updated")
    assert len(calls) == initial_length + 1
    assert calls[-1] == ("alice", "Alice Repo Updated")


def test_login_user_account_refreshes_identity(facade: UserAccountFacade, service_stub: _ServiceStub):
    service_stub.users = []

    calls: list[tuple[str, str]] = []

    def listener(username: str, display_name: str) -> None:
        calls.append((username, display_name))

    facade.add_active_user_change_listener(listener)
    initial_length = len(calls)

    service_stub.users = [{"username": "chris", "name": "Chris"}]

    success = asyncio.run(facade.login_user_account("chris", "pw123"))

    assert success is True
    assert service_stub.set_active_calls[-1] == "chris"
    assert facade.ensure_user_identity()[0] == "chris"
    assert len(calls) == initial_length + 1
    assert calls[-1][0] == "chris"


def test_delete_user_account_raises_for_unknown_user(facade: UserAccountFacade, service_stub: _ServiceStub):
    service_stub.delete_user_result = False

    with pytest.raises(ValueError):
        asyncio.run(facade.delete_user_account("missing"))

    assert service_stub.delete_calls == ["missing"]


def test_request_password_reset_returns_challenge_dict(
    monkeypatch, facade: UserAccountFacade, service_stub: _ServiceStub
):
    class Challenge:
        def __init__(self):
            self.username = "alice"
            self.token = "reset-token"
            self._expires_at = "2024-01-01T00:00:00Z"

        def expires_at_iso(self):
            return self._expires_at

    service_stub.password_reset_challenge = Challenge()
    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def capture(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(user_account_facade_module, "run_async_in_thread", capture)

    result = asyncio.run(facade.request_password_reset("alice"))

    assert result == {
        "username": "alice",
        "token": "reset-token",
        "expires_at": "2024-01-01T00:00:00Z",
    }
    assert calls == [
        (service_stub.initiate_password_reset, ("alice",), {}),
    ]


def test_verify_password_reset_token_propagates_result(
    monkeypatch, facade: UserAccountFacade, service_stub: _ServiceStub
):
    service_stub.verify_password_reset_token_result = False
    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def capture(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(user_account_facade_module, "run_async_in_thread", capture)

    result = asyncio.run(
        facade.verify_password_reset_token("alice", "verification-token")
    )

    assert result is False
    assert calls == [
        (
            service_stub.verify_password_reset_token,
            ("alice", "verification-token"),
            {},
        ),
    ]


def test_complete_password_reset_propagates_exceptions(
    monkeypatch, facade: UserAccountFacade, service_stub: _ServiceStub
):
    service_stub.complete_password_reset_result = RuntimeError("service failure")
    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def capture(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(user_account_facade_module, "run_async_in_thread", capture)

    with pytest.raises(RuntimeError, match="service failure"):
        asyncio.run(
            facade.complete_password_reset(
                "alice", "verification-token", "new-password"
            )
        )

    assert calls == [
        (
            service_stub.complete_password_reset,
            ("alice", "verification-token", "new-password"),
            {},
        ),
    ]


def test_override_user_identity_requires_feature_flag(facade: UserAccountFacade):
    with pytest.raises(RuntimeError, match="Manual user identity overrides are deprecated"):
        facade.override_user_identity("manual-user")


def test_override_user_identity_dispatches_when_enabled(facade: UserAccountFacade):
    facade._config_manager.get_config.return_value = True

    calls: list[tuple[str, str]] = []
    facade.add_active_user_change_listener(lambda username, display: calls.append((username, display)))
    calls.clear()

    facade.override_user_identity("manual-user")

    assert facade.ensure_user_identity() == ("manual-user", "manual-user")
    assert calls and calls[-1] == ("manual-user", "manual-user")
    facade._config_manager.get_config.assert_any_call(LEGACY_IDENTITY_OVERRIDE_FLAG, False)
