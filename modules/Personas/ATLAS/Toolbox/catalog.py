"""ATLAS-specific catalog helpers."""

from __future__ import annotations

import asyncio
from typing import Mapping, Optional, Sequence

from modules.Server import atlas_server


async def task_catalog_snapshot(
    persona: Optional[Sequence[str] | str] = None,
    *,
    tags: Optional[Sequence[str] | str] = None,
    required_skills: Optional[Sequence[str] | str] = None,
    required_tools: Optional[Sequence[str] | str] = None,
) -> Mapping[str, object]:
    """Return a filtered snapshot of the task catalog for ATLAS.

    Parameters
    ----------
    persona:
        Persona filter(s) to apply when enumerating the catalog. Accepts a
        single string or a sequence of persona identifiers.
    tags:
        Optional tag filters limiting results to manifests labeled with the
        supplied tags.
    required_skills:
        Optional collection of skill identifiers that must be satisfied by the
        returned tasks.
    required_tools:
        Optional collection of tool identifiers that must be satisfied by the
        returned tasks.

    Returns
    -------
    Mapping[str, object]
        Canonical payload produced by :meth:`AtlasServer.get_task_catalog`
        containing ``count`` and ``tasks`` keys.

    Examples
    --------
    >>> await task_catalog_snapshot(persona="DocGenius", tags=["clinical"])
    {'count': 2, 'tasks': [...]}  # doctest: +SKIP
    """

    return await asyncio.to_thread(
        atlas_server.get_task_catalog,
        persona=persona,
        tags=tags,
        required_skills=required_skills,
        required_tools=required_tools,
    )
