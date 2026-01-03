from pathlib import Path

import pytest

pytest.importorskip(
    "sqlalchemy",
    reason="SQLAlchemy is required to verify the conversation store configuration",
)
pytest.importorskip(
    "pytest_postgresql",
    reason="PostgreSQL fixture is required to verify the conversation store configuration",
)

from sqlalchemy import create_engine  # type: ignore[assignment]
from sqlalchemy.orm import sessionmaker  # type: ignore[assignment]

from ATLAS.config import ConfigManager
from modules.conversation_store import Base, ConversationStoreRepository


def _provision_conversation_store(dsn: str) -> None:
    engine = create_engine(dsn, future=True)
    try:
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, future=True)
        repository = ConversationStoreRepository(factory)
        repository.create_schema()
    finally:
        engine.dispose()


@pytest.fixture
def config_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ConfigManager:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DEFAULT_PROVIDER", "OpenAI")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-4o")
    monkeypatch.delenv("CONVERSATION_DATABASE_URL", raising=False)

    monkeypatch.setattr(
        ConfigManager,
        "_compute_yaml_path",
        lambda self: str(yaml_path),
        raising=False,
    )

    manager = ConfigManager()
    manager._yaml_path = str(yaml_path)
    return manager


@pytest.mark.postgresql
def test_ensure_postgres_conversation_store_verifies_existing_schema(
    config_manager: ConfigManager,
    postgresql,
) -> None:
    dsn = postgresql.dsn()
    _provision_conversation_store(dsn)

    config_manager.config.setdefault("conversation_database", {})["url"] = dsn
    config_manager.yaml_config.setdefault("conversation_database", {})["url"] = dsn

    result = config_manager.ensure_postgres_conversation_store()

    assert result == dsn
    assert config_manager.config["conversation_database"]["url"] == dsn
    assert config_manager.yaml_config["conversation_database"]["url"] == dsn
    assert config_manager.is_conversation_store_verified() is True


@pytest.mark.postgresql
def test_ensure_postgres_conversation_store_raises_when_schema_missing(
    config_manager: ConfigManager,
    postgresql,
) -> None:
    dsn = postgresql.dsn()

    config_manager.config.setdefault("conversation_database", {})["url"] = dsn
    config_manager.yaml_config.setdefault("conversation_database", {})["url"] = dsn

    with pytest.raises(RuntimeError) as excinfo:
        config_manager.ensure_postgres_conversation_store()

    message = str(excinfo.value)
    assert "Run the standalone setup utility" in message
    assert config_manager.is_conversation_store_verified() is False
