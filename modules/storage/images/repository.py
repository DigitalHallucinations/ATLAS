"""Image artifact storage repository.

Provides persistent storage for generated images with JSON metadata sidecars.
Follows ATLAS storage patterns with XDG-compliant default paths.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from modules.Providers.Media.base import (
        ImageGenerationRequest,
        ImageGenerationResult,
    )
    from modules.storage.settings import StorageSettings

logger = setup_logger(__name__)


class ImageArtifactRepository:
    """Repository for storing generated images with metadata.

    Images are stored in a date-partitioned directory structure with
    accompanying JSON metadata sidecar files.

    Directory structure::

        {base_path}/
            2026/
                01/
                    {artifact_id}.png
                    {artifact_id}.json
                02/
                    ...

    Usage::

        repo = ImageArtifactRepository(settings)
        paths = await repo.save(result, request)
        artifact = await repo.get(artifact_id)
        artifacts = await repo.list_recent(limit=10)
    """

    def __init__(
        self,
        settings: Optional["StorageSettings"] = None,
        base_path: Optional[Path] = None,
    ):
        """Initialize the image artifact repository.

        Args:
            settings: Storage settings (for path configuration).
            base_path: Override base path for artifact storage.
        """
        self._settings = settings
        self._base_path = base_path or self._resolve_default_path(settings)
        self._base_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Image artifact repository initialized at %s", self._base_path)

    def _resolve_default_path(
        self, settings: Optional["StorageSettings"]
    ) -> Path:
        """Resolve the default artifact storage path.

        Uses settings if available, otherwise falls back to XDG_DATA_HOME.
        """
        if settings and hasattr(settings, "image_artifact_path"):
            custom_path = getattr(settings, "image_artifact_path", None)
            if custom_path:
                return Path(custom_path)

        # XDG-compliant default
        xdg_data = os.environ.get("XDG_DATA_HOME", "~/.local/share")
        return Path(xdg_data).expanduser() / "ATLAS" / "assets" / "generated"

    def _get_storage_dir(self, timestamp: Optional[datetime] = None) -> Path:
        """Get date-partitioned storage directory.

        Args:
            timestamp: Timestamp for partitioning (defaults to now).

        Returns:
            Path to the partition directory.
        """
        ts = timestamp or datetime.now(timezone.utc)
        dir_path = self._base_path / str(ts.year) / f"{ts.month:02d}"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _mime_to_ext(self, mime: str) -> str:
        """Convert MIME type to file extension."""
        mapping = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/svg+xml": ".svg",
        }
        return mapping.get(mime, ".png")

    async def save(
        self,
        result: "ImageGenerationResult",
        request: "ImageGenerationRequest",
    ) -> List[str]:
        """Save generated images and metadata.

        Args:
            result: Image generation result containing images.
            request: Original request for metadata context.

        Returns:
            List of saved image file paths.
        """
        from modules.Providers.Media.base import GeneratedImage

        storage_dir = self._get_storage_dir()
        saved_paths: List[str] = []
        now = datetime.now(timezone.utc)

        for image in result.images:
            # Generate or use existing artifact ID
            artifact_id = image.id or str(uuid.uuid4())
            ext = self._mime_to_ext(image.mime)
            image_filename = f"{artifact_id}{ext}"
            image_path = storage_dir / image_filename

            # Save image data
            saved = False
            if image.b64:
                try:
                    image_path.write_bytes(base64.b64decode(image.b64))
                    saved = True
                except Exception as exc:
                    logger.error("Failed to decode base64 image: %s", exc)

            elif image.path and Path(image.path).exists():
                try:
                    shutil.copy2(image.path, image_path)
                    saved = True
                except Exception as exc:
                    logger.error("Failed to copy image file: %s", exc)

            elif image.url:
                # Download from URL
                try:
                    import aiohttp

                    async with aiohttp.ClientSession() as session:
                        async with session.get(image.url) as resp:
                            if resp.status == 200:
                                image_path.write_bytes(await resp.read())
                                saved = True
                except Exception as exc:
                    logger.warning("Failed to download image from URL: %s", exc)

            if not saved:
                logger.warning("No image data available for artifact %s", artifact_id)
                continue

            # Update image path reference
            image.path = str(image_path)

            # Build metadata
            metadata = self._build_metadata(
                artifact_id=artifact_id,
                image=image,
                result=result,
                request=request,
                timestamp=now,
            )

            # Save metadata sidecar
            metadata_path = storage_dir / f"{artifact_id}.json"
            try:
                metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
            except Exception as exc:
                logger.error("Failed to save metadata: %s", exc)

            saved_paths.append(str(image_path))
            logger.debug("Saved image artifact: %s", image_path)

        return saved_paths

    def _build_metadata(
        self,
        artifact_id: str,
        image: Any,
        result: "ImageGenerationResult",
        request: "ImageGenerationRequest",
        timestamp: datetime,
    ) -> Dict[str, Any]:
        """Build metadata dictionary for an artifact."""
        return {
            "id": artifact_id,
            "prompt": request.prompt,
            "model": result.model,
            "provider": result.provider,
            "seed": image.seed_used,
            "size": request.size,
            "aspect_ratio": request.aspect_ratio,
            "style_preset": request.style_preset,
            "quality": request.quality,
            "revised_prompt": image.revised_prompt,
            "timing_ms": result.timing_ms,
            "cost_estimate": result.cost_estimate,
            "generated_at": timestamp.isoformat(),
            "request_metadata": request.metadata,
        }

    async def get(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve artifact metadata by ID.

        Args:
            artifact_id: Artifact identifier.

        Returns:
            Metadata dict if found, None otherwise.
        """
        # Search all partitions (could optimize with index)
        for year_dir in self._base_path.iterdir():
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue
                metadata_path = month_dir / f"{artifact_id}.json"
                if metadata_path.exists():
                    try:
                        return json.loads(metadata_path.read_text())
                    except Exception as exc:
                        logger.error("Failed to read metadata: %s", exc)
                        return None
        return None

    async def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent artifacts.

        Args:
            limit: Maximum number of artifacts to return.

        Returns:
            List of metadata dicts, newest first.
        """
        artifacts: List[Dict[str, Any]] = []

        # Collect from newest partitions first
        year_dirs = sorted(
            [d for d in self._base_path.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: d.name,
            reverse=True,
        )

        for year_dir in year_dirs:
            month_dirs = sorted(
                [d for d in year_dir.iterdir() if d.is_dir()],
                key=lambda d: d.name,
                reverse=True,
            )

            for month_dir in month_dirs:
                json_files = sorted(
                    month_dir.glob("*.json"),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )

                for json_file in json_files:
                    if len(artifacts) >= limit:
                        return artifacts

                    try:
                        metadata = json.loads(json_file.read_text())
                        # Add image path
                        image_path = json_file.with_suffix("")
                        for ext in [".png", ".jpg", ".webp", ".gif"]:
                            candidate = image_path.with_suffix(ext)
                            if candidate.exists():
                                metadata["path"] = str(candidate)
                                break
                        artifacts.append(metadata)
                    except Exception as exc:
                        logger.warning("Failed to read %s: %s", json_file, exc)

        return artifacts

    async def delete(self, artifact_id: str) -> bool:
        """Delete an artifact and its metadata.

        Args:
            artifact_id: Artifact identifier.

        Returns:
            True if deleted, False if not found.
        """
        for year_dir in self._base_path.iterdir():
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue

                metadata_path = month_dir / f"{artifact_id}.json"
                if metadata_path.exists():
                    # Delete metadata
                    metadata_path.unlink()

                    # Delete image file
                    for ext in [".png", ".jpg", ".webp", ".gif", ".svg"]:
                        image_path = month_dir / f"{artifact_id}{ext}"
                        if image_path.exists():
                            image_path.unlink()
                            break

                    logger.debug("Deleted artifact: %s", artifact_id)
                    return True

        return False

    async def cleanup_old(self, days: int = 30) -> int:
        """Remove artifacts older than specified days.

        Args:
            days: Age threshold in days.

        Returns:
            Number of artifacts deleted.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
        deleted = 0

        for year_dir in self._base_path.iterdir():
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue

                for json_file in month_dir.glob("*.json"):
                    if json_file.stat().st_mtime < cutoff:
                        artifact_id = json_file.stem
                        if await self.delete(artifact_id):
                            deleted += 1

        if deleted > 0:
            logger.info("Cleaned up %d old image artifacts", deleted)

        return deleted
