"""ATLAS services layer.

High-level service facades that coordinate multiple subsystems:
- RAGService: Retrieval-Augmented Generation pipeline
- ConversationService: Conversation management (future)
- ProviderService: LLM provider orchestration (future)
"""

from core.services.rag import RAGService, RAGServiceStatus

__all__ = [
    "RAGService",
    "RAGServiceStatus",
]
