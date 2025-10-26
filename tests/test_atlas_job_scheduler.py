import pytest

sqlalchemy_mod = pytest.importorskip(
    "sqlalchemy",
    reason="SQLAlchemy is required to initialize the job scheduler",
)
if getattr(getattr(sqlalchemy_mod, "create_engine", None), "__module__", "").startswith(
    "tests.conftest"
):
    pytest.skip(
        "SQLAlchemy runtime is unavailable for job scheduler tests",
        allow_module_level=True,
    )
pytest.importorskip(
    "sqlalchemy.exc",
    reason="SQLAlchemy exception helpers are required for job scheduler tests",
)
pytest.importorskip(
    "pytest_postgresql",
    reason="PostgreSQL fixture is required for job scheduler persistence tests",
)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.conversation_store import ConversationStoreRepository

from ATLAS.ATLAS import ATLAS


@pytest.mark.asyncio
async def test_initialize_registers_job_manifests(tmp_path, monkeypatch, postgresql):
    dsn = postgresql.dsn()
    monkeypatch.setenv("CONVERSATION_DATABASE_URL", dsn)

    engine = create_engine(dsn, future=True)
    try:
        factory = sessionmaker(bind=engine, future=True)
        ConversationStoreRepository(factory).create_schema()
    finally:
        engine.dispose()

    atlas = ATLAS()
    try:
        await atlas.initialize()
        assert atlas.job_scheduler is not None

        repository = atlas.config_manager.get_job_repository()
        assert repository is not None

        jobs = repository.list_jobs(tenant_id=atlas.tenant_id)
        assert jobs["items"], "Expected at least one scheduled job manifest"
    finally:
        await atlas.close()
