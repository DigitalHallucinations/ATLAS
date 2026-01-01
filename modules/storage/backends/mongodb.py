"""MongoDB backend adapter for StorageManager."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set

from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


class MongoDBBackend:
    """MongoDB-specific backend operations.

    Handles MongoDB-specific collection management, indexes, and operations.
    """

    def __init__(self, client: Any, database: Any) -> None:
        self._client = client
        self._database = database

    @property
    def client(self) -> Any:
        """Get the underlying MongoClient."""
        return self._client

    @property
    def database(self) -> Any:
        """Get the database instance."""
        return self._database

    def get_collection(self, name: str) -> Any:
        """Get a collection by name."""
        return self._database[name]

    async def list_collections(self) -> List[str]:
        """List all collection names in the database."""

        def _list() -> List[str]:
            return self._database.list_collection_names()

        return await asyncio.to_thread(_list)

    async def ensure_collection(self, name: str) -> bool:
        """Ensure a collection exists (creates if not present)."""

        def _ensure() -> bool:
            existing = set(self._database.list_collection_names())
            if name not in existing:
                self._database.create_collection(name)
                logger.debug(f"Created MongoDB collection: {name}")
            return True

        return await asyncio.to_thread(_ensure)

    async def verify_collections(self, required: Set[str]) -> Dict[str, bool]:
        """Verify that required collections exist."""
        existing = set(await self.list_collections())
        return {coll: coll in existing for coll in required}

    async def create_index(
        self,
        collection_name: str,
        keys: List[tuple],
        *,
        unique: bool = False,
        sparse: bool = False,
        background: bool = True,
        name: Optional[str] = None,
    ) -> str:
        """Create an index on a collection.

        Args:
            collection_name: Name of the collection.
            keys: List of (field, direction) tuples.
            unique: Whether the index should enforce uniqueness.
            sparse: Whether the index should be sparse.
            background: Whether to build the index in the background.
            name: Optional name for the index.

        Returns:
            The name of the created index.
        """

        def _create() -> str:
            collection = self._database[collection_name]
            index_name = collection.create_index(
                keys,
                unique=unique,
                sparse=sparse,
                background=background,
                name=name,
            )
            logger.debug(
                f"Created index {index_name} on {collection_name}",
                extra={"keys": keys, "unique": unique},
            )
            return index_name

        return await asyncio.to_thread(_create)

    async def get_index_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about indexes on a collection."""

        def _get_info() -> Dict[str, Any]:
            collection = self._database[collection_name]
            return dict(collection.index_information())

        return await asyncio.to_thread(_get_info)

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection."""

        def _get_stats() -> Dict[str, Any]:
            return self._database.command("collStats", collection_name)

        return await asyncio.to_thread(_get_stats)

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics for the database."""

        def _get_stats() -> Dict[str, Any]:
            return self._database.command("dbStats")

        return await asyncio.to_thread(_get_stats)

    async def get_database_size(self) -> int:
        """Get the database size in bytes."""
        stats = await self.get_database_stats()
        return int(stats.get("dataSize", 0) + stats.get("indexSize", 0))

    async def drop_collection(self, name: str) -> bool:
        """Drop a collection."""

        def _drop() -> bool:
            self._database.drop_collection(name)
            logger.info(f"Dropped MongoDB collection: {name}")
            return True

        return await asyncio.to_thread(_drop)

    async def aggregate(
        self,
        collection_name: str,
        pipeline: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run an aggregation pipeline."""

        def _aggregate() -> List[Dict[str, Any]]:
            collection = self._database[collection_name]
            return list(collection.aggregate(pipeline, **kwargs))

        return await asyncio.to_thread(_aggregate)


__all__ = ["MongoDBBackend"]
