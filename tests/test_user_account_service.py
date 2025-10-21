import sys
import types
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda *_args, **_kwargs: {}
sys.modules.setdefault("yaml", yaml_stub)

dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *_args, **_kwargs: None
dotenv_stub.set_key = lambda *_args, **_kwargs: None
dotenv_stub.find_dotenv = lambda *_args, **_kwargs: ""
sys.modules.setdefault("dotenv", dotenv_stub)

from modules.background_tasks import run_async_in_thread
from modules.conversation_store import (
    Base,
    ConversationStoreRepository,
    User,
    UserCredential,
)
from modules.user_accounts import user_account_service


class _StubLogger:
    def __init__(self) -> None:
        self.infos: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.errors: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.warnings: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def info(self, *args, **kwargs):
        self.infos.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warnings.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.errors.append((args, kwargs))

    def debug(self, *args, **_kwargs):
        return None


class _StubConfigManager:
    def __init__(self, overrides: Optional[dict[str, object]] = None) -> None:
        self._active_user: Optional[str] = None
        self._overrides = dict(overrides or {})

    def get_active_user(self) -> Optional[str]:
        return self._active_user

    def set_active_user(self, username: Optional[str]) -> Optional[str]:
        self._active_user = username
        return username

    def get_config(self, key: str, default=None):
        return self._overrides.get(key, default)


class _SpyConversationRepository:
    def __init__(self, delegate: ConversationStoreRepository) -> None:
        self._delegate = delegate
        self.ensure_user_calls: list[tuple[str, Optional[str], dict[str, object]]] = []
        self.attach_credential_calls: list[tuple[str, str]] = []

    def ensure_user(
        self,
        external_id,
        *,
        display_name=None,
        metadata=None,
    ):
        record = dict(metadata or {})
        self.ensure_user_calls.append((external_id, display_name, record))
        return self._delegate.ensure_user(
            external_id,
            display_name=display_name,
            metadata=metadata,
        )

    def attach_credential(self, username, user_id):
        self.attach_credential_calls.append((str(username), str(user_id)))
        return self._delegate.attach_credential(username, user_id)

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


@pytest.fixture
def conversation_repository(postgresql):
    engine = create_engine(postgresql.dsn(), future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    repository = ConversationStoreRepository(factory)
    repository.create_schema()
    try:
        yield repository
    finally:
        engine.dispose()


def _create_service(
    monkeypatch,
    repository: ConversationStoreRepository,
    *,
    config_overrides: Optional[dict[str, object]] = None,
    clock=None,
):
    monkeypatch.setattr(
        user_account_service,
        "setup_logger",
        lambda *_args, **_kwargs: _StubLogger(),
    )
    config = _StubConfigManager(overrides=config_overrides)
    service = user_account_service.UserAccountService(
        config_manager=config,
        conversation_repository=repository,
        clock=clock,
    )
    return service, config


def test_register_user_persists_account(monkeypatch, conversation_repository):
    service, _ = _create_service(monkeypatch, conversation_repository)
    try:
        account = service.register_user(
            "alice",
            "Password123!",
            "alice@example.com",
            "Alice",
            "1999-01-01",
        )
        assert account.username == "alice"
        assert account.email == "alice@example.com"

        users = service.list_users()
        assert users[0]["username"] == "alice"
        assert users[0]["display_name"] == "Alice"
        assert users[0]["status_badge"] == "Never signed in"

        with pytest.raises(user_account_service.DuplicateUserError):
            service.register_user("alice", "Newpass1!@", "duplicate@example.com")

        with pytest.raises(user_account_service.DuplicateUserError):
            service.register_user("bob", "Password123!", "alice@example.com")

        session = conversation_repository._session_factory()
        credential_row = None
        user_row = None
        try:
            credential_row = session.execute(
                select(UserCredential).where(UserCredential.username == "alice")
            ).scalar_one()
            assert credential_row.user_id is not None
            user_row = session.execute(
                select(User).where(User.id == credential_row.user_id)
            ).scalar_one_or_none()
        finally:
            session.close()

        assert user_row is not None
        assert str(user_row.id) == str(credential_row.user_id)
    finally:
        service.close()


def test_update_user_validates_and_persists(monkeypatch, conversation_repository):
    service, _ = _create_service(monkeypatch, conversation_repository)
    try:
        service.register_user(
            "alice",
            "Password123!",
            "alice@example.com",
            "Alice",
            "1999-01-01",
        )

        updated = service.update_user(
            "alice",
            password="Newpass123!",
            current_password="Password123!",
            email="alice.new@example.com",
            name="Alice Updated",
            dob="2000-02-02",
        )

        assert updated.email == "alice.new@example.com"
        assert updated.name == "Alice Updated"
        assert updated.dob == "2000-02-02"
        assert service.authenticate_user("alice", "Newpass123!") is True

        details = service.get_user_details("alice")
        assert details is not None
        assert details["email"] == "alice.new@example.com"
        assert details["name"] == "Alice Updated"
    finally:
        service.close()


def test_conversation_store_synchronised(monkeypatch, conversation_repository):
    spy_repo = _SpyConversationRepository(conversation_repository)
    service, _ = _create_service(monkeypatch, spy_repo)
    try:
        service.register_user(
            "bob",
            "Password123!",
            "bob@example.com",
            "Bob",
            "1985-05-05",
        )
        assert spy_repo.ensure_user_calls
        external_id, display_name, metadata = spy_repo.ensure_user_calls[0]
        assert external_id == "bob"
        assert display_name == "Bob"
        assert metadata["email"] == "bob@example.com"
        assert metadata["name"] == "Bob"
        assert spy_repo.attach_credential_calls
        attach_username, attach_uuid = spy_repo.attach_credential_calls[0]
        assert attach_username == "bob"
        session = conversation_repository._session_factory()
        try:
            credential_row = session.execute(
                select(UserCredential).where(UserCredential.username == "bob")
            ).scalar_one()
            assert credential_row.user_id is not None
            assert str(credential_row.user_id) == attach_uuid
        finally:
            session.close()
    finally:
        service.close()


def test_authenticate_user_success_and_failure(monkeypatch, conversation_repository):
    service, _ = _create_service(monkeypatch, conversation_repository)
    try:
        service.register_user("carol", "Password123!", "carol@example.com")

        assert service.authenticate_user("carol", "Password123!") is True
        assert service.authenticate_user("carol", "wrong") is False

        attempts = service.get_login_attempts("carol")
        assert len(attempts) == 2
        success_flags = [attempt["successful"] for attempt in attempts]
        assert success_flags.count(True) == 1
        assert success_flags.count(False) == 1
    finally:
        service.close()


def test_authenticate_user_lockout(monkeypatch, conversation_repository):
    overrides = {
        "ACCOUNT_LOCKOUT_MAX_FAILURES": 2,
        "ACCOUNT_LOCKOUT_WINDOW_SECONDS": 60,
        "ACCOUNT_LOCKOUT_DURATION_SECONDS": 30,
    }
    now = datetime.now(timezone.utc)
    service, _ = _create_service(
        monkeypatch,
        conversation_repository,
        config_overrides=overrides,
        clock=lambda: now,
    )
    try:
        service.register_user("dave", "Password123!", "dave@example.com")

        assert service.authenticate_user("dave", "bad1") is False
        assert service.authenticate_user("dave", "bad2") is False

        with pytest.raises(user_account_service.AccountLockedError):
            service.authenticate_user("dave", "Password123!")

        now_plus = now + timedelta(seconds=31)
        service._clock = lambda: now_plus  # type: ignore[assignment]
        assert service.authenticate_user("dave", "Password123!") is True
    finally:
        service.close()


def test_password_reset_flow_success(monkeypatch, conversation_repository):
    service, _ = _create_service(monkeypatch, conversation_repository)
    try:
        service.register_user("erin", "Password123!", "erin@example.com")

        challenge = service.initiate_password_reset("erin")
        assert challenge is not None
        assert challenge.username == "erin"

        success = service.complete_password_reset("erin", challenge.token, "Newpass123!")
        assert success is True
        assert service.authenticate_user("erin", "Newpass123!") is True
    finally:
        service.close()


def test_lockout_state_persists(monkeypatch, conversation_repository):
    overrides = {
        "ACCOUNT_LOCKOUT_MAX_FAILURES": 1,
        "ACCOUNT_LOCKOUT_WINDOW_SECONDS": 300,
        "ACCOUNT_LOCKOUT_DURATION_SECONDS": 120,
    }
    now = datetime.now(timezone.utc)
    service, _ = _create_service(
        monkeypatch,
        conversation_repository,
        config_overrides=overrides,
        clock=lambda: now,
    )
    try:
        service.register_user("frank", "Password123!", "frank@example.com")
        assert service.authenticate_user("frank", "wrong") is False
        with pytest.raises(user_account_service.AccountLockedError):
            service.authenticate_user("frank", "Password123!")
    finally:
        service.close()

    restored, _ = _create_service(
        monkeypatch,
        conversation_repository,
        config_overrides=overrides,
        clock=lambda: now,
    )
    try:
        with pytest.raises(user_account_service.AccountLockedError):
            restored.authenticate_user("frank", "Password123!")

        now_later = now + timedelta(seconds=130)
        restored._clock = lambda: now_later  # type: ignore[assignment]
        assert restored.authenticate_user("frank", "Password123!") is True
    finally:
        restored.close()


def test_run_async_in_thread_integrates_with_service(monkeypatch, conversation_repository):
    service, _ = _create_service(monkeypatch, conversation_repository)
    try:
        service.register_user("gwen", "Password123!", "gwen@example.com")

        future = run_async_in_thread(service.list_users)
        users = future.result(timeout=5)
        assert users[0]["username"] == "gwen"
    finally:
        service.close()
