import asyncio
import pytest

sqlalchemy_mod = pytest.importorskip(
    "sqlalchemy",
    reason="SQLAlchemy is required for ATLAS user account tests",
)
if getattr(getattr(sqlalchemy_mod, "create_engine", None), "__module__", "").startswith(
    "tests.conftest"
):
    pytest.skip(
        "SQLAlchemy runtime is unavailable for ATLAS user account tests",
        allow_module_level=True,
    )

pytest.importorskip(
    "pytest_postgresql",
    reason="PostgreSQL fixture is required for ATLAS user account tests",
)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.conversation_store import ConversationStoreRepository


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


from ATLAS.ATLAS import ATLAS


class _StubService:
    def __init__(self, users):
        self._users = users

    def list_users(self):
        return list(self._users)


def _make_atlas_with_service(users):
    atlas = ATLAS()
    atlas.user_account_facade._user_account_service = _StubService(users)
    return atlas


def test_list_user_accounts_is_awaitable():
    atlas = _make_atlas_with_service([
        {"username": "alice", "name": "Alice"},
        {"username": "bob", "name": "Bob"},
    ])

    result = asyncio.run(atlas.list_user_accounts())

    assert result == [
        {"username": "alice", "name": "Alice"},
        {"username": "bob", "name": "Bob"},
    ]

