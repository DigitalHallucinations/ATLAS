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
from .mongo_repository import MongoConversationStoreRepository

try:
    from .repository import ConversationStoreRepository, create_conversation_engine
    from .vector_pipeline import ConversationVectorCatalog, ConversationVectorPipeline
except ImportError as exc:  # pragma: no cover - startup import guard
    raise ImportError(
        "Conversation store SQL backend requires SQLAlchemy with PostgreSQL dialect and "
        "pgvector installed. Install with `pip install SQLAlchemy pgvector psycopg[binary]`."
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
]
