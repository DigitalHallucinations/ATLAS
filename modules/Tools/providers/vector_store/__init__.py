"""Vector store provider registrations."""

from importlib import import_module
import logging

logger = logging.getLogger(__name__)

_ADAPTER_MODULES = (
    "in_memory",
    "chroma",
    "faiss",
    "pinecone",
)

for _module in _ADAPTER_MODULES:
    try:
        import_module(f".{_module}", __name__)
    except Exception as exc:  # pragma: no cover - optional dependencies may be absent
        logger.debug("Skipping vector store adapter module %s: %s", _module, exc)

__all__ = list(_ADAPTER_MODULES)
