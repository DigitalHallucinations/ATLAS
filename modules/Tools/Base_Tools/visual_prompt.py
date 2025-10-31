"""Generate deterministic visual prompt scaffolds."""

from __future__ import annotations

import asyncio
from typing import Mapping, Optional, Sequence


_DEFAULT_STYLES = (
    "digital painting",
    "film still",
    "concept sketch",
    "mixed media collage",
)


class VisualPrompt:
    """Compose a grounded visual art prompt without external services."""

    async def run(
        self,
        *,
        subject: str,
        style: Optional[str] = None,
        color_palette: Optional[Sequence[str]] = None,
        lighting: Optional[str] = None,
        medium: Optional[str] = None,
        camera: Optional[str] = None,
    ) -> Mapping[str, object]:
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("VisualPrompt requires a non-empty subject.")

        await asyncio.sleep(0)

        normalized_subject = subject.strip()
        normalized_style = (style or _DEFAULT_STYLES[0]).strip()
        normalized_lighting = (lighting or "soft studio").strip()
        normalized_medium = (medium or "digital").strip()
        normalized_camera = (camera or "50mm prime").strip()

        palette = []
        if color_palette:
            for color in color_palette:
                if isinstance(color, str) and color.strip():
                    candidate = color.strip()
                    if candidate.lower() not in {entry.lower() for entry in palette}:
                        palette.append(candidate)
        if not palette:
            palette = ["copper", "deep teal", "opal highlights"]

        descriptors = [
            normalized_subject,
            f"rendered as {normalized_style}",
            f"medium: {normalized_medium}",
            f"lighting: {normalized_lighting}",
            f"camera: {normalized_camera}",
        ]

        prompt = ", ".join(descriptors) + "."

        return {
            "subject": normalized_subject,
            "style": normalized_style,
            "color_palette": palette,
            "lighting": normalized_lighting,
            "medium": normalized_medium,
            "camera": normalized_camera,
            "prompt": prompt,
            "keywords": [normalized_subject, normalized_style] + palette,
        }


__all__ = ["VisualPrompt"]
