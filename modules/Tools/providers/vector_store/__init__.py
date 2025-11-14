"""Vector store provider registrations."""

from importlib import import_module, util
import logging

logger = logging.getLogger(__name__)

_ADAPTER_MODULES = (
    "in_memory",
    "chroma",
    "faiss",
    "pinecone",
    "mongodb",
)

_OPTIONAL_DEPENDENCIES = {
    "mongodb": "pymongo",
}

for _module in _ADAPTER_MODULES:
    dependency = _OPTIONAL_DEPENDENCIES.get(_module)
    if dependency and util.find_spec(dependency) is None:
        logger.debug(
            "Skipping vector store adapter module %s: optional dependency %s missing",
            _module,
            dependency,
        )
        continue
    try:
        import_module(f".{_module}", __name__)
    except Exception as exc:  # pragma: no cover - optional dependencies may be absent
        logger.debug("Skipping vector store adapter module %s: %s", _module, exc)

__all__ = list(_ADAPTER_MODULES)
