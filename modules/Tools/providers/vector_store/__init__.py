"""Vector store provider registrations."""

from . import in_memory  # noqa: F401 - ensure registration side-effect

__all__ = [
    "in_memory",
]

