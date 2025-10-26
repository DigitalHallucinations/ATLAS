import logging
import pytest


pytest.importorskip("sqlalchemy")
pytest.importorskip(
    "pytest_postgresql",
    reason="PostgreSQL fixture is required for chat session persistence tests",
)

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
except ImportError as exc:  # pragma: no cover - skip when SQLAlchemy helpers missing
    pytest.skip(
        f"SQLAlchemy runtime helpers unavailable: {exc}", allow_module_level=True
    )

if getattr(create_engine, "__module__", "").startswith("tests.conftest"):
    pytest.skip(
        "SQLAlchemy runtime is unavailable for chat session persistence tests",
        allow_module_level=True,
    )

from modules.Chat.chat_session import ChatSession
from modules.conversation_store import Base, ConversationStoreRepository


@pytest.fixture
def repository(postgresql):
    engine = create_engine(postgresql.dsn(), future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    repo = ConversationStoreRepository(factory)
    try:
        repo.create_schema()
    except RuntimeError as exc:  # pragma: no cover - optional dependency guard
        engine.dispose()
        pytest.skip(f"Conversation store unavailable: {exc}")
    try:
        yield repo
    finally:
        engine.dispose()


class _ConfigManagerStub:
    def get_conversation_retention_policies(self):
        return {}


class _PersonaManagerStub:
    def get_current_persona_prompt(self):
        return None


class _ProviderManagerStub:
    def __init__(self):
        self._conversation_ids: list[str] = []

    def set_current_conversation_id(self, conversation_id: str) -> None:
        self._conversation_ids.append(conversation_id)

    def get_default_model_for_provider(self, _provider: str) -> str:
        return "dummy-model"


class _ChatSessionAtlasStub:
    def __init__(self, repository: ConversationStoreRepository) -> None:
        self.conversation_repository = repository
        self.logger = logging.getLogger("ChatSessionResetPersistenceTests")
        self.config_manager = _ConfigManagerStub()
        self.persona_manager = _PersonaManagerStub()
        self.provider_manager = _ProviderManagerStub()
        self._default_provider = "dummy-provider"
        self._default_model = "dummy-model"
        self.tenant_id = "tenant"
        self.notifications: list[tuple[str, str | None]] = []

    def get_default_provider(self) -> str:
        return self._default_provider

    def get_default_model(self) -> str:
        return self._default_model

    def notify_conversation_updated(self, conversation_id: str, *, reason: str | None = None):
        self.notifications.append((conversation_id, reason))


def test_reset_conversation_archives_existing_session_and_creates_new_one(repository):
    atlas = _ChatSessionAtlasStub(repository)
    session = ChatSession(atlas)

    original_id = session.conversation_id
    assert repository.get_conversation(original_id, tenant_id=atlas.tenant_id)

    session.reset_conversation()

    replacement_id = session.conversation_id
    assert replacement_id != original_id

    all_conversations = repository.list_conversations_for_tenant(atlas.tenant_id)
    identifiers = {item["id"] for item in all_conversations}
    assert {str(original_id), str(replacement_id)} <= identifiers

    archived_conversation = repository.get_conversation(
        original_id, tenant_id=atlas.tenant_id
    )
    assert archived_conversation is not None
    assert archived_conversation.get("archived_at") is not None
    assert archived_conversation.get("metadata", {}).get("archived_reason") == "reset"

    new_conversation = repository.get_conversation(
        replacement_id, tenant_id=atlas.tenant_id
    )
    assert new_conversation is not None
    assert new_conversation.get("archived_at") is None

    assert (str(original_id), "archived") in atlas.notifications
    assert (str(replacement_id), "created") in atlas.notifications

