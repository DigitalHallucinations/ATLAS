import asyncio
import inspect
from types import SimpleNamespace

import pytest

sqlalchemy_mod = pytest.importorskip(
    "sqlalchemy",
    reason="SQLAlchemy is required for ATLAS active user tests",
)
if getattr(getattr(sqlalchemy_mod, "create_engine", None), "__module__", "").startswith(
    "tests.conftest"
):
    pytest.skip(
        "SQLAlchemy runtime is unavailable for ATLAS active user tests",
        allow_module_level=True,
    )

pytest.importorskip(
    "pytest_postgresql",
    reason="PostgreSQL fixture is required for ATLAS active user tests",
)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.conversation_store import ConversationStoreRepository

from ATLAS import ATLAS as atlas_module
from ATLAS.ATLAS import ATLAS


@pytest.fixture(autouse=True)
def configure_conversation_store(monkeypatch, postgresql):
    dsn = postgresql.dsn()
    monkeypatch.setenv("CONVERSATION_DATABASE_URL", dsn)

    engine = create_engine(dsn, future=True)
    try:
        factory = sessionmaker(bind=engine, future=True)
        ConversationStoreRepository(factory).create_schema()
    finally:
        engine.dispose()


@pytest.fixture(autouse=True)
def patch_run_async_in_thread(monkeypatch):
    async def immediate(func, *args, **kwargs):
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    monkeypatch.setattr(atlas_module, "run_async_in_thread", immediate)


class _AccountServiceStub:
    def __init__(self):
        self.users: dict[str, dict[str, object]] = {}
        self.active_user: str | None = None

    def register_user(self, username, password, email, name=None, dob=None):
        account = SimpleNamespace(
            id=len(self.users) + 1,
            username=username,
            email=email,
            name=name,
            dob=dob,
        )
        self.users[username] = {"account": account, "password": password}
        return account

    def authenticate_user(self, username, password):
        record = self.users.get(username)
        if not record:
            return False
        return record["password"] == password

    def set_active_user(self, username):
        self.active_user = username

    def get_active_user(self):
        return self.active_user

    def list_users(self):
        rows = []
        for username, payload in self.users.items():
            account = payload["account"]
            rows.append(
                {
                    "id": account.id,
                    "username": account.username,
                    "email": account.email,
                    "name": account.name,
                    "dob": account.dob,
                }
            )
        rows.sort(key=lambda row: row["username"].lower())
        return rows


class _PersonaManagerStub:
    def __init__(self):
        self.user_updates: list[str] = []

    def set_user(self, username):
        self.user_updates.append(username)


def test_active_user_listener_receives_register_and_login_updates():
    atlas = ATLAS()
    service = _AccountServiceStub()
    persona_manager = _PersonaManagerStub()
    atlas._user_account_service = service
    atlas.persona_manager = persona_manager

    events = []

    def listener(username, display_name):
        events.append((username, display_name))

    atlas.add_active_user_change_listener(listener)
    events.clear()

    asyncio.run(atlas.register_user_account("alice", "pw123", "alice@example.com", name="Alice"))
    assert events[-1] == ("alice", "Alice")
    assert persona_manager.user_updates[-1] == "alice"

    asyncio.run(atlas.logout_active_user())
    events.clear()
    asyncio.run(atlas.login_user_account("alice", "pw123"))
    assert events[-1] == ("alice", "Alice")
    assert persona_manager.user_updates[-1] == "alice"


def test_logout_notifies_listeners_with_generic_identity():
    atlas = ATLAS()
    service = _AccountServiceStub()
    persona_manager = _PersonaManagerStub()
    atlas._user_account_service = service
    atlas.persona_manager = persona_manager

    events = []

    atlas.add_active_user_change_listener(lambda username, display: events.append((username, display)))
    events.clear()

    asyncio.run(atlas.register_user_account("bob", "pw", "bob@example.com", name="Bob"))
    events.clear()

    asyncio.run(atlas.logout_active_user())
    assert events, "Expected logout to trigger an event"
    username, display = events[-1]
    assert username != "bob"
    assert persona_manager.user_updates[-1] != "bob"
