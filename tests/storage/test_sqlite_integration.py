from pathlib import Path

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
if "tests/conftest.py" in getattr(sqlalchemy, "__file__", ""):
    pytest.skip(
        "SQLAlchemy stubbed; SQLite integration tests require the real library",
        allow_module_level=True,
    )

from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker

from modules.conversation_store.bootstrap import bootstrap_conversation_store
from modules.conversation_store.repository import ConversationStoreRepository
from modules.job_store.models import ensure_job_schema
from modules.job_store.repository import JobStoreRepository
from modules.Tools.Base_Tools.kv_store import SQLiteKeyValueStoreAdapter
from modules.store_common.model_utils import generate_uuid


def _build_sqlite_engine(dsn: str):
    return create_engine(dsn, future=True)


def test_bootstrap_conversation_store_accepts_sqlite(tmp_path):
    db_path = tmp_path / "atlas_conversations.sqlite"
    result = bootstrap_conversation_store(f"sqlite:///{db_path}")
    url = make_url(result)
    assert Path(url.database).exists()
    assert url.database.endswith("atlas_conversations.sqlite")


def test_conversation_repository_operates_with_sqlite(tmp_path):
    dsn = f"sqlite:///{tmp_path / 'conversation_store.sqlite'}"
    engine = _build_sqlite_engine(dsn)
    Session = sessionmaker(bind=engine, future=True)
    repository = ConversationStoreRepository(Session)
    repository.create_schema()

    tenant_id = "tenant-alpha"
    user_id = repository.ensure_user("user-1", display_name="User", tenant_id=tenant_id)
    session_id = repository.ensure_session(user_id, "session-1", metadata={"origin": "test"})
    conversation_id = repository.ensure_conversation(
        generate_uuid(),
        tenant_id=tenant_id,
        session_id=session_id,
        metadata={"topic": "demo"},
    )

    stored = repository.add_message(
        conversation_id,
        tenant_id=tenant_id,
        role="user",
        content={"text": "Hello SQLite"},
        user_id=user_id,
        session_id=session_id,
    )
    assert stored["content"]["text"] == "Hello SQLite"

    recent = repository.load_recent_messages(conversation_id, tenant_id=tenant_id, limit=5)
    assert any(entry["content"]["text"] == "Hello SQLite" for entry in recent)


def test_job_repository_operates_with_sqlite(tmp_path):
    dsn = f"sqlite:///{tmp_path / 'job_store.sqlite'}"
    engine = _build_sqlite_engine(dsn)
    ensure_job_schema(engine)
    Session = sessionmaker(bind=engine, future=True)
    repository = JobStoreRepository(Session)

    job = repository.create_job(
        "demo-job",
        tenant_id="tenant-alpha",
        description="SQLite integration test",
    )
    fetched = repository.get_job(job["id"], tenant_id="tenant-alpha", with_events=True)
    assert fetched["id"] == job["id"]
    assert fetched["events"], "Job creation should record an event"


def test_sqlite_kv_adapter_supports_crud(tmp_path):
    dsn = f"sqlite:///{tmp_path / 'kv_store.sqlite'}"
    engine = _build_sqlite_engine(dsn)
    adapter = SQLiteKeyValueStoreAdapter(
        engine=engine,
        namespace_quota_bytes=1024,
        global_quota_bytes=None,
    )

    adapter.set("state", "token", {"value": 42}, ttl_seconds=None)
    retrieved = adapter.get("state", "token")
    assert retrieved.found is True
    assert retrieved.record is not None and retrieved.record.value == {"value": 42}

    incremented = adapter.increment(
        "state",
        "counter",
        delta=2,
        ttl_seconds=None,
        initial_value=1,
    )
    assert incremented.record.value == 3

    deleted = adapter.delete("state", "token")
    assert deleted.deleted is True

