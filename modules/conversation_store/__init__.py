"""Conversation store integration for ATLAS."""

from .models import (
    Base,
    Conversation,
    Message,
    EpisodicMemory,
    MessageAsset,
    MessageEvent,
    MessageVector,
    GraphNode,
    GraphEdge,
    PasswordResetToken,
    Session,
    User,
    UserCredential,
    UserLoginAttempt,
)
from ._shared import (
    MESSAGE_TYPE_TEXT,
    MESSAGE_TYPE_IMAGE,
    MESSAGE_TYPE_AUDIO,
    MESSAGE_TYPE_VIDEO,
    MESSAGE_TYPE_FILE,
    MESSAGE_TYPE_TOOL_CALL,
    MESSAGE_TYPE_TOOL_RESULT,
    ASSET_TYPE_IMAGE,
    ASSET_TYPE_AUDIO,
    ASSET_TYPE_VIDEO,
    ASSET_TYPE_FILE,
    ASSET_TYPE_ATTACHMENT,
)
from .mongo_repository import MongoConversationStoreRepository

try:
    from .repository import ConversationStoreRepository, create_conversation_engine
    from .vector_pipeline import ConversationVectorCatalog, ConversationVectorPipeline
except ImportError as exc:  # pragma: no cover - startup import guard
    raise ImportError(
        "Conversation store SQL backend requires SQLAlchemy with PostgreSQL dialect. "
        "For PostgreSQL vector support, also install pgvector: "
        "`pip install SQLAlchemy psycopg[binary] pgvector`."
    ) from exc

# Import task models so they share the conversation metadata during bootstrap.
from modules.task_store import models as task_models  # noqa: F401

__all__ = [
    "Base",
    "User",
    "Session",
    "Conversation",
    "Message",
    "MessageAsset",
    "EpisodicMemory",
    "MessageVector",
    "MessageEvent",
    "GraphNode",
    "GraphEdge",
    "UserCredential",
    "UserLoginAttempt",
    "PasswordResetToken",
    "ConversationStoreRepository",
    "MongoConversationStoreRepository",
    "create_conversation_engine",
    "ConversationVectorCatalog",
    "ConversationVectorPipeline",
    # Message type constants
    "MESSAGE_TYPE_TEXT",
    "MESSAGE_TYPE_IMAGE",
    "MESSAGE_TYPE_AUDIO",
    "MESSAGE_TYPE_VIDEO",
    "MESSAGE_TYPE_FILE",
    "MESSAGE_TYPE_TOOL_CALL",
    "MESSAGE_TYPE_TOOL_RESULT",
    # Asset type constants
    "ASSET_TYPE_IMAGE",
    "ASSET_TYPE_AUDIO",
    "ASSET_TYPE_VIDEO",
    "ASSET_TYPE_FILE",
    "ASSET_TYPE_ATTACHMENT",
]
