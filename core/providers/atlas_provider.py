"""Utility helpers for constructing or retrieving an :class:`ATLAS` instance."""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

from core.ATLAS import ATLAS
from core.setup_marker import is_setup_complete

logger = logging.getLogger(__name__)

TAtlas = TypeVar("TAtlas", bound=ATLAS)


class AtlasProvider:
    """Provide a guarded, reusable :class:`ATLAS` singleton."""

    def __init__(
        self,
        *,
        atlas_cls: type[TAtlas] = ATLAS,
        setup_check: Callable[[], bool] = is_setup_complete,
    ) -> None:
        self._atlas_cls = atlas_cls
        self._setup_check = setup_check
        self._atlas_instance: TAtlas | None = None

    def get_atlas(self) -> TAtlas:
        """Return the shared :class:`ATLAS` instance, validating setup first."""

        if not self._setup_check():
            logger.error("ATLAS setup is incomplete. Launching the setup wizard.")
            raise RuntimeError("ATLAS setup is incomplete.")

        if self._atlas_instance is None:
            self._atlas_instance = self._atlas_cls()
        return self._atlas_instance
