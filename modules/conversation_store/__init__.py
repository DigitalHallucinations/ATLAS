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
from .repository import ConversationStoreRepository, create_conversation_engine
from .vector_pipeline import ConversationVectorCatalog, ConversationVectorPipeline

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
    "create_conversation_engine",
    "ConversationVectorCatalog",
    "ConversationVectorPipeline",
]
