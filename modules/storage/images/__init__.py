"""Image artifact storage for ATLAS.

Provides persistent storage for generated images with JSON metadata sidecars.
Integrates with the ATLAS StorageManager architecture.
"""

from .repository import ImageArtifactRepository

__all__ = ["ImageArtifactRepository"]
