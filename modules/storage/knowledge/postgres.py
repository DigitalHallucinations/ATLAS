"""PostgreSQL knowledge store implementation.

Uses PostgreSQL with pgvector extension for storing and searching
knowledge base documents and chunks with vector embeddings.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from modules.logging.logger import setup_logger

from .base import (
    ChunkMetadata,
    ChunkNotFoundError,
    DocumentNotFoundError,
    DocumentStatus,
    DocumentType,
    DocumentVersion,
    IngestionError,
    KnowledgeBase,
    KnowledgeBaseNotFoundError,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeStore,
    KnowledgeStoreError,
    SearchQuery,
    SearchResult,
    register_knowledge_store,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import sessionmaker
    from modules.storage.embeddings import EmbeddingProvider
    from modules.storage.chunking import TextSplitter

logger = setup_logger(__name__)


def _content_hash(content: str) -> str:
    """Compute SHA-256 hash of content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@register_knowledge_store("postgres")
class PostgresKnowledgeStore(KnowledgeStore):
    """PostgreSQL-based knowledge store using pgvector.

    Stores knowledge bases, documents, and chunks in PostgreSQL tables
    with vector embeddings for semantic search.
    """

    # Table names
    TABLE_KNOWLEDGE_BASES = "knowledge_bases"
    TABLE_DOCUMENTS = "knowledge_documents"
    TABLE_CHUNKS = "knowledge_chunks"
    TABLE_DOCUMENT_VERSIONS = "knowledge_document_versions"

    def __init__(
        self,
        engine: "Engine",
        session_factory: "sessionmaker",
        *,
        embedding_provider: Optional["EmbeddingProvider"] = None,
        text_splitter: Optional["TextSplitter"] = None,
        schema: str = "public",
        index_type: str = "hnsw",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
    ) -> None:
        """Initialize the PostgreSQL knowledge store.

        Args:
            engine: SQLAlchemy engine connected to PostgreSQL.
            session_factory: Session factory for database operations.
            embedding_provider: Provider for generating embeddings.
            text_splitter: Splitter for chunking documents.
            schema: Database schema to use.
            index_type: Vector index type ("hnsw" or "ivfflat").
            hnsw_m: HNSW M parameter.
            hnsw_ef_construction: HNSW ef_construction parameter.
        """
        self._engine = engine
        self._session_factory = session_factory
        self._embedding_provider = embedding_provider
        self._text_splitter = text_splitter
        self._schema = schema
        self._index_type = index_type.lower()
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "postgres"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def set_embedding_provider(self, provider: "EmbeddingProvider") -> None:
        """Set the embedding provider."""
        self._embedding_provider = provider

    def set_text_splitter(self, splitter: "TextSplitter") -> None:
        """Set the text splitter."""
        self._text_splitter = splitter

    async def initialize(self) -> None:
        """Initialize the store and create required tables."""
        async with self._lock:
            if self._initialized:
                return

            await self._ensure_extension()
            await self._create_tables()
            self._initialized = True
            logger.info("PostgreSQL knowledge store initialized")

    async def _ensure_extension(self) -> None:
        """Ensure pgvector extension is installed."""

        def _ensure() -> None:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()

        await asyncio.to_thread(_ensure)

    async def _create_tables(self) -> None:
        """Create the knowledge store tables if they don't exist."""

        def _create() -> None:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                # Knowledge bases table
                conn.execute(
                    text(f"""
                        CREATE TABLE IF NOT EXISTS {self._schema}.{self.TABLE_KNOWLEDGE_BASES} (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            name TEXT NOT NULL,
                            description TEXT DEFAULT '',
                            embedding_model TEXT NOT NULL,
                            embedding_dimension INTEGER NOT NULL,
                            chunk_size INTEGER DEFAULT 512,
                            chunk_overlap INTEGER DEFAULT 50,
                            document_count INTEGER DEFAULT 0,
                            chunk_count INTEGER DEFAULT 0,
                            owner_id TEXT,
                            metadata JSONB DEFAULT '{{}}',
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                )

                # Knowledge documents table
                conn.execute(
                    text(f"""
                        CREATE TABLE IF NOT EXISTS {self._schema}.{self.TABLE_DOCUMENTS} (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            knowledge_base_id UUID NOT NULL REFERENCES {self._schema}.{self.TABLE_KNOWLEDGE_BASES}(id) ON DELETE CASCADE,
                            title TEXT NOT NULL,
                            content TEXT,
                            content_hash TEXT,
                            source_uri TEXT,
                            document_type TEXT DEFAULT 'text',
                            status TEXT DEFAULT 'pending',
                            chunk_count INTEGER DEFAULT 0,
                            token_count INTEGER DEFAULT 0,
                            metadata JSONB DEFAULT '{{}}',
                            error_message TEXT,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW(),
                            indexed_at TIMESTAMPTZ,
                            current_version INTEGER DEFAULT 1,
                            version_count INTEGER DEFAULT 1
                        )
                    """)
                )

                # Knowledge chunks table with vector column
                # We use a large dimension (3072) to support most embedding models
                # Actual dimension is stored in knowledge_base
                conn.execute(
                    text(f"""
                        CREATE TABLE IF NOT EXISTS {self._schema}.{self.TABLE_CHUNKS} (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            document_id UUID NOT NULL REFERENCES {self._schema}.{self.TABLE_DOCUMENTS}(id) ON DELETE CASCADE,
                            knowledge_base_id UUID NOT NULL REFERENCES {self._schema}.{self.TABLE_KNOWLEDGE_BASES}(id) ON DELETE CASCADE,
                            content TEXT NOT NULL,
                            embedding vector(3072),
                            chunk_index INTEGER NOT NULL,
                            token_count INTEGER DEFAULT 0,
                            start_char INTEGER DEFAULT 0,
                            end_char INTEGER DEFAULT 0,
                            start_line INTEGER,
                            end_line INTEGER,
                            section TEXT,
                            language TEXT,
                            metadata JSONB DEFAULT '{{}}',
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                )

                # Document versions table for version history
                conn.execute(
                    text(f"""
                        CREATE TABLE IF NOT EXISTS {self._schema}.{self.TABLE_DOCUMENT_VERSIONS} (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            document_id UUID NOT NULL REFERENCES {self._schema}.{self.TABLE_DOCUMENTS}(id) ON DELETE CASCADE,
                            version_number INTEGER NOT NULL,
                            title TEXT NOT NULL,
                            content TEXT NOT NULL,
                            content_hash TEXT,
                            change_summary TEXT,
                            created_by TEXT,
                            metadata JSONB DEFAULT '{{}}',
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            UNIQUE(document_id, version_number)
                        )
                    """)
                )

                # Create indexes
                conn.execute(
                    text(f"""
                        CREATE INDEX IF NOT EXISTS idx_kb_owner
                        ON {self._schema}.{self.TABLE_KNOWLEDGE_BASES}(owner_id)
                    """)
                )
                conn.execute(
                    text(f"""
                        CREATE INDEX IF NOT EXISTS idx_doc_kb
                        ON {self._schema}.{self.TABLE_DOCUMENTS}(knowledge_base_id)
                    """)
                )
                conn.execute(
                    text(f"""
                        CREATE INDEX IF NOT EXISTS idx_doc_status
                        ON {self._schema}.{self.TABLE_DOCUMENTS}(status)
                    """)
                )
                conn.execute(
                    text(f"""
                        CREATE INDEX IF NOT EXISTS idx_doc_hash
                        ON {self._schema}.{self.TABLE_DOCUMENTS}(content_hash)
                    """)
                )
                conn.execute(
                    text(f"""
                        CREATE INDEX IF NOT EXISTS idx_chunk_doc
                        ON {self._schema}.{self.TABLE_CHUNKS}(document_id)
                    """)
                )
                conn.execute(
                    text(f"""
                        CREATE INDEX IF NOT EXISTS idx_chunk_kb
                        ON {self._schema}.{self.TABLE_CHUNKS}(knowledge_base_id)
                    """)
                )
                conn.execute(
                    text(f"""
                        CREATE INDEX IF NOT EXISTS idx_version_doc
                        ON {self._schema}.{self.TABLE_DOCUMENT_VERSIONS}(document_id)
                    """)
                )

                # Vector similarity index
                if self._index_type == "hnsw":
                    conn.execute(
                        text(f"""
                            CREATE INDEX IF NOT EXISTS idx_chunk_embedding
                            ON {self._schema}.{self.TABLE_CHUNKS}
                            USING hnsw (embedding vector_cosine_ops)
                            WITH (m = {self._hnsw_m}, ef_construction = {self._hnsw_ef_construction})
                        """)
                    )
                else:
                    conn.execute(
                        text(f"""
                            CREATE INDEX IF NOT EXISTS idx_chunk_embedding
                            ON {self._schema}.{self.TABLE_CHUNKS}
                            USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100)
                        """)
                    )

                conn.commit()

        await asyncio.to_thread(_create)

    async def shutdown(self) -> None:
        """Shutdown the knowledge store."""
        async with self._lock:
            self._initialized = False
            logger.info("PostgreSQL knowledge store shutdown")

    async def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the store is healthy."""
        if not self._initialized:
            return False

        try:

            def _check() -> bool:
                from sqlalchemy import text

                with self._engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                    return result.scalar() == 1

            return await asyncio.wait_for(asyncio.to_thread(_check), timeout=timeout)
        except Exception as exc:
            logger.warning(f"Knowledge store health check failed: {exc}")
            return False

    # --- Knowledge Base Management ---

    async def create_knowledge_base(
        self,
        name: str,
        *,
        description: str = "",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        owner_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBase:
        """Create a new knowledge base."""
        kb_id = str(uuid.uuid4())

        def _create() -> Dict[str, Any]:
            from sqlalchemy import text
            import json

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"""
                        INSERT INTO {self._schema}.{self.TABLE_KNOWLEDGE_BASES}
                        (id, name, description, embedding_model, embedding_dimension,
                         chunk_size, chunk_overlap, owner_id, metadata)
                        VALUES (:id, :name, :description, :embedding_model, :embedding_dimension,
                                :chunk_size, :chunk_overlap, :owner_id, :metadata)
                        RETURNING id, created_at, updated_at
                    """),
                    {
                        "id": kb_id,
                        "name": name,
                        "description": description,
                        "embedding_model": embedding_model,
                        "embedding_dimension": embedding_dimension,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "owner_id": owner_id,
                        "metadata": json.dumps(metadata or {}),
                    },
                )
                row = result.fetchone()
                conn.commit()
                return {
                    "id": str(row[0]),
                    "created_at": row[1],
                    "updated_at": row[2],
                }

        result = await asyncio.to_thread(_create)
        logger.info(f"Created knowledge base: {name} ({result['id']})")

        return KnowledgeBase(
            id=result["id"],
            name=name,
            description=description,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            owner_id=owner_id,
            metadata=metadata or {},
            created_at=result["created_at"],
            updated_at=result["updated_at"],
        )

    async def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        """Get a knowledge base by ID."""

        def _get() -> Optional[Dict[str, Any]]:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"""
                        SELECT id, name, description, embedding_model, embedding_dimension,
                               chunk_size, chunk_overlap, document_count, chunk_count,
                               owner_id, metadata, created_at, updated_at
                        FROM {self._schema}.{self.TABLE_KNOWLEDGE_BASES}
                        WHERE id = :id
                    """),
                    {"id": kb_id},
                )
                row = result.fetchone()
                if not row:
                    return None
                return {
                    "id": str(row[0]),
                    "name": row[1],
                    "description": row[2],
                    "embedding_model": row[3],
                    "embedding_dimension": row[4],
                    "chunk_size": row[5],
                    "chunk_overlap": row[6],
                    "document_count": row[7],
                    "chunk_count": row[8],
                    "owner_id": row[9],
                    "metadata": row[10] or {},
                    "created_at": row[11],
                    "updated_at": row[12],
                }

        data = await asyncio.to_thread(_get)
        if not data:
            return None

        return KnowledgeBase(**data)

    async def list_knowledge_bases(
        self,
        *,
        owner_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[KnowledgeBase]:
        """List knowledge bases."""

        def _list() -> List[Dict[str, Any]]:
            from sqlalchemy import text

            query = f"""
                SELECT id, name, description, embedding_model, embedding_dimension,
                       chunk_size, chunk_overlap, document_count, chunk_count,
                       owner_id, metadata, created_at, updated_at
                FROM {self._schema}.{self.TABLE_KNOWLEDGE_BASES}
            """
            params: Dict[str, Any] = {"limit": limit, "offset": offset}

            if owner_id:
                query += " WHERE owner_id = :owner_id"
                params["owner_id"] = owner_id

            query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"

            with self._engine.connect() as conn:
                result = conn.execute(text(query), params)
                rows = result.fetchall()

            return [
                {
                    "id": str(row[0]),
                    "name": row[1],
                    "description": row[2],
                    "embedding_model": row[3],
                    "embedding_dimension": row[4],
                    "chunk_size": row[5],
                    "chunk_overlap": row[6],
                    "document_count": row[7],
                    "chunk_count": row[8],
                    "owner_id": row[9],
                    "metadata": row[10] or {},
                    "created_at": row[11],
                    "updated_at": row[12],
                }
                for row in rows
            ]

        data_list = await asyncio.to_thread(_list)
        return [KnowledgeBase(**data) for data in data_list]

    async def update_knowledge_base(
        self,
        kb_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[KnowledgeBase]:
        """Update a knowledge base."""
        if not any([name, description, metadata]):
            return await self.get_knowledge_base(kb_id)

        def _update() -> bool:
            from sqlalchemy import text
            import json

            updates = ["updated_at = NOW()"]
            params: Dict[str, Any] = {"id": kb_id}

            if name is not None:
                updates.append("name = :name")
                params["name"] = name
            if description is not None:
                updates.append("description = :description")
                params["description"] = description
            if metadata is not None:
                updates.append("metadata = metadata || :metadata")
                params["metadata"] = json.dumps(metadata)

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"""
                        UPDATE {self._schema}.{self.TABLE_KNOWLEDGE_BASES}
                        SET {", ".join(updates)}
                        WHERE id = :id
                    """),
                    params,
                )
                conn.commit()
                return result.rowcount > 0

        updated = await asyncio.to_thread(_update)
        if not updated:
            return None

        return await self.get_knowledge_base(kb_id)

    async def delete_knowledge_base(
        self,
        kb_id: str,
        *,
        delete_documents: bool = True,
    ) -> bool:
        """Delete a knowledge base."""

        def _delete() -> bool:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                # Cascade delete handles documents and chunks
                result = conn.execute(
                    text(f"""
                        DELETE FROM {self._schema}.{self.TABLE_KNOWLEDGE_BASES}
                        WHERE id = :id
                    """),
                    {"id": kb_id},
                )
                conn.commit()
                return result.rowcount > 0

        deleted = await asyncio.to_thread(_delete)
        if deleted:
            logger.info(f"Deleted knowledge base: {kb_id}")
        return deleted

    # --- Document Management ---

    async def add_document(
        self,
        kb_id: str,
        title: str,
        content: str,
        *,
        source_uri: Optional[str] = None,
        document_type: DocumentType = DocumentType.TEXT,
        metadata: Optional[Dict[str, Any]] = None,
        auto_chunk: bool = True,
        auto_embed: bool = True,
    ) -> KnowledgeDocument:
        """Add a document to a knowledge base."""
        # Verify knowledge base exists
        kb = await self.get_knowledge_base(kb_id)
        if not kb:
            raise KnowledgeBaseNotFoundError(f"Knowledge base not found: {kb_id}")

        doc_id = str(uuid.uuid4())
        content_hash_value = _content_hash(content)

        def _create_doc() -> Dict[str, Any]:
            from sqlalchemy import text
            import json

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"""
                        INSERT INTO {self._schema}.{self.TABLE_DOCUMENTS}
                        (id, knowledge_base_id, title, content, content_hash,
                         source_uri, document_type, status, metadata)
                        VALUES (:id, :kb_id, :title, :content, :content_hash,
                                :source_uri, :document_type, :status, :metadata)
                        RETURNING id, created_at, updated_at
                    """),
                    {
                        "id": doc_id,
                        "kb_id": kb_id,
                        "title": title,
                        "content": content,
                        "content_hash": content_hash_value,
                        "source_uri": source_uri,
                        "document_type": document_type.value,
                        "status": DocumentStatus.PROCESSING.value if auto_chunk else DocumentStatus.PENDING.value,
                        "metadata": json.dumps(metadata or {}),
                    },
                )
                row = result.fetchone()
                conn.commit()
                return {
                    "id": str(row[0]),
                    "created_at": row[1],
                    "updated_at": row[2],
                }

        result = await asyncio.to_thread(_create_doc)

        doc = KnowledgeDocument(
            id=result["id"],
            knowledge_base_id=kb_id,
            title=title,
            content=content,
            content_hash=content_hash_value,
            source_uri=source_uri,
            document_type=document_type,
            status=DocumentStatus.PROCESSING if auto_chunk else DocumentStatus.PENDING,
            metadata=metadata or {},
            created_at=result["created_at"],
            updated_at=result["updated_at"],
        )

        # Process chunks if auto_chunk is enabled
        if auto_chunk:
            try:
                await self._process_document_chunks(doc, kb, auto_embed=auto_embed)
            except Exception as exc:
                logger.error(f"Failed to process document {doc_id}: {exc}")
                await self._update_document_status(
                    doc_id,
                    DocumentStatus.FAILED,
                    error_message=str(exc),
                )
                doc.status = DocumentStatus.FAILED
                doc.error_message = str(exc)

        return doc

    async def _process_document_chunks(
        self,
        doc: KnowledgeDocument,
        kb: KnowledgeBase,
        *,
        auto_embed: bool = True,
    ) -> None:
        """Process a document into chunks and optionally embed them."""
        if not self._text_splitter:
            raise IngestionError("No text splitter configured")

        # Split document into chunks
        chunks = self._text_splitter.split_text(
            doc.content or "",
            chunk_size=kb.chunk_size,
            chunk_overlap=kb.chunk_overlap,
        )

        if not chunks:
            logger.warning(f"No chunks generated for document: {doc.id}")
            await self._update_document_status(doc.id, DocumentStatus.INDEXED)
            return

        # Generate embeddings if enabled
        embeddings: Optional[List[List[float]]] = None
        if auto_embed and self._embedding_provider:
            await self._embedding_provider.initialize()
            result = await self._embedding_provider.embed_batch(
                [c.content for c in chunks]
            )
            embeddings = result.embeddings

        # Store chunks
        chunk_records = []
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            embedding = embeddings[i] if embeddings else None
            chunk_records.append({
                "id": chunk_id,
                "document_id": doc.id,
                "knowledge_base_id": doc.knowledge_base_id,
                "content": chunk.content,
                "embedding": embedding,
                "chunk_index": i,
                "token_count": chunk.token_count,
                "start_char": chunk.metadata.start_char,
                "end_char": chunk.metadata.end_char,
                "start_line": chunk.metadata.start_line,
                "end_line": chunk.metadata.end_line,
                "section": chunk.metadata.section,
                "language": chunk.metadata.language,
                "metadata": chunk.metadata.extra,
            })

        await self._insert_chunks(chunk_records)

        # Update document status and counts
        await self._update_document_after_chunking(
            doc.id,
            doc.knowledge_base_id,
            len(chunks),
            sum(c.token_count for c in chunks),
        )

    async def _insert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Insert chunk records into the database."""

        def _insert() -> None:
            from sqlalchemy import text
            import json

            with self._engine.connect() as conn:
                for chunk in chunks:
                    embedding_str = (
                        f"[{','.join(str(v) for v in chunk['embedding'])}]"
                        if chunk["embedding"]
                        else None
                    )
                    conn.execute(
                        text(f"""
                            INSERT INTO {self._schema}.{self.TABLE_CHUNKS}
                            (id, document_id, knowledge_base_id, content, embedding,
                             chunk_index, token_count, start_char, end_char,
                             start_line, end_line, section, language, metadata)
                            VALUES (:id, :document_id, :knowledge_base_id, :content,
                                    {'$1::vector' if embedding_str else 'NULL'},
                                    :chunk_index, :token_count, :start_char, :end_char,
                                    :start_line, :end_line, :section, :language, :metadata)
                        """).bindparams(
                            **({"embedding": embedding_str} if embedding_str else {}),
                        ),
                        {
                            "id": chunk["id"],
                            "document_id": chunk["document_id"],
                            "knowledge_base_id": chunk["knowledge_base_id"],
                            "content": chunk["content"],
                            "chunk_index": chunk["chunk_index"],
                            "token_count": chunk["token_count"],
                            "start_char": chunk["start_char"],
                            "end_char": chunk["end_char"],
                            "start_line": chunk["start_line"],
                            "end_line": chunk["end_line"],
                            "section": chunk["section"],
                            "language": chunk["language"],
                            "metadata": json.dumps(chunk["metadata"] or {}),
                        },
                    )
                conn.commit()

        await asyncio.to_thread(_insert)

    async def _update_document_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """Update document status."""

        def _update() -> None:
            from sqlalchemy import text

            params: Dict[str, Any] = {
                "id": doc_id,
                "status": status.value,
            }
            query = f"""
                UPDATE {self._schema}.{self.TABLE_DOCUMENTS}
                SET status = :status, updated_at = NOW()
            """
            if error_message:
                query += ", error_message = :error_message"
                params["error_message"] = error_message
            if status == DocumentStatus.INDEXED:
                query += ", indexed_at = NOW()"
            query += " WHERE id = :id"

            with self._engine.connect() as conn:
                conn.execute(text(query), params)
                conn.commit()

        await asyncio.to_thread(_update)

    async def _update_document_after_chunking(
        self,
        doc_id: str,
        kb_id: str,
        chunk_count: int,
        token_count: int,
    ) -> None:
        """Update document and knowledge base after chunking."""

        def _update() -> None:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                # Update document
                conn.execute(
                    text(f"""
                        UPDATE {self._schema}.{self.TABLE_DOCUMENTS}
                        SET status = :status,
                            chunk_count = :chunk_count,
                            token_count = :token_count,
                            indexed_at = NOW(),
                            updated_at = NOW()
                        WHERE id = :id
                    """),
                    {
                        "id": doc_id,
                        "status": DocumentStatus.INDEXED.value,
                        "chunk_count": chunk_count,
                        "token_count": token_count,
                    },
                )

                # Update knowledge base counts
                conn.execute(
                    text(f"""
                        UPDATE {self._schema}.{self.TABLE_KNOWLEDGE_BASES}
                        SET document_count = document_count + 1,
                            chunk_count = chunk_count + :chunk_count,
                            updated_at = NOW()
                        WHERE id = :id
                    """),
                    {"id": kb_id, "chunk_count": chunk_count},
                )

                conn.commit()

        await asyncio.to_thread(_update)

    async def get_document(self, doc_id: str) -> Optional[KnowledgeDocument]:
        """Get a document by ID."""

        def _get() -> Optional[Dict[str, Any]]:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"""
                        SELECT id, knowledge_base_id, title, content, content_hash,
                               source_uri, document_type, status, chunk_count, token_count,
                               metadata, error_message, created_at, updated_at, indexed_at,
                               current_version, version_count
                        FROM {self._schema}.{self.TABLE_DOCUMENTS}
                        WHERE id = :id
                    """),
                    {"id": doc_id},
                )
                row = result.fetchone()
                if not row:
                    return None
                return {
                    "id": str(row[0]),
                    "knowledge_base_id": str(row[1]),
                    "title": row[2],
                    "content": row[3],
                    "content_hash": row[4],
                    "source_uri": row[5],
                    "document_type": DocumentType(row[6]) if row[6] else DocumentType.TEXT,
                    "status": DocumentStatus(row[7]) if row[7] else DocumentStatus.PENDING,
                    "chunk_count": row[8],
                    "token_count": row[9],
                    "metadata": row[10] or {},
                    "error_message": row[11],
                    "created_at": row[12],
                    "updated_at": row[13],
                    "indexed_at": row[14],
                    "current_version": row[15] if row[15] else 1,
                    "version_count": row[16] if row[16] else 1,
                }

        data = await asyncio.to_thread(_get)
        if not data:
            return None

        return KnowledgeDocument(**data)

    async def list_documents(
        self,
        kb_id: str,
        *,
        status: Optional[DocumentStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[KnowledgeDocument]:
        """List documents in a knowledge base."""

        def _list() -> List[Dict[str, Any]]:
            from sqlalchemy import text

            query = f"""
                SELECT id, knowledge_base_id, title, content_hash,
                       source_uri, document_type, status, chunk_count, token_count,
                       metadata, error_message, created_at, updated_at, indexed_at,
                       current_version, version_count
                FROM {self._schema}.{self.TABLE_DOCUMENTS}
                WHERE knowledge_base_id = :kb_id
            """
            params: Dict[str, Any] = {"kb_id": kb_id, "limit": limit, "offset": offset}

            if status:
                query += " AND status = :status"
                params["status"] = status.value

            query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"

            with self._engine.connect() as conn:
                result = conn.execute(text(query), params)
                rows = result.fetchall()

            return [
                {
                    "id": str(row[0]),
                    "knowledge_base_id": str(row[1]),
                    "title": row[2],
                    "content_hash": row[3],
                    "source_uri": row[4],
                    "document_type": DocumentType(row[5]) if row[5] else DocumentType.TEXT,
                    "status": DocumentStatus(row[6]) if row[6] else DocumentStatus.PENDING,
                    "chunk_count": row[7],
                    "token_count": row[8],
                    "metadata": row[9] or {},
                    "error_message": row[10],
                    "created_at": row[11],
                    "updated_at": row[12],
                    "indexed_at": row[13],
                    "current_version": row[14] if row[14] else 1,
                    "version_count": row[15] if row[15] else 1,
                }
                for row in rows
            ]

        data_list = await asyncio.to_thread(_list)
        return [KnowledgeDocument(**data) for data in data_list]

    async def find_duplicate(
        self,
        kb_id: str,
        content: str,
    ) -> Optional[KnowledgeDocument]:
        """Find a duplicate document by content hash."""
        content_hash_value = _content_hash(content)

        def _find() -> Optional[Dict[str, Any]]:
            from sqlalchemy import text

            query = f"""
                SELECT id, knowledge_base_id, title, content_hash,
                       source_uri, document_type, status, chunk_count, token_count,
                       metadata, error_message, created_at, updated_at, indexed_at,
                       current_version, version_count
                FROM {self._schema}.{self.TABLE_DOCUMENTS}
                WHERE knowledge_base_id = :kb_id AND content_hash = :content_hash
                LIMIT 1
            """

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(query),
                    {"kb_id": kb_id, "content_hash": content_hash_value}
                )
                row = result.fetchone()

            if not row:
                return None

            return {
                "id": str(row[0]),
                "knowledge_base_id": str(row[1]),
                "title": row[2],
                "content_hash": row[3],
                "source_uri": row[4],
                "document_type": DocumentType(row[5]) if row[5] else DocumentType.TEXT,
                "status": DocumentStatus(row[6]) if row[6] else DocumentStatus.PENDING,
                "chunk_count": row[7],
                "token_count": row[8],
                "metadata": row[9] or {},
                "error_message": row[10],
                "created_at": row[11],
                "updated_at": row[12],
                "indexed_at": row[13],
                "current_version": row[14] if row[14] else 1,
                "version_count": row[15] if row[15] else 1,
            }

        data = await asyncio.to_thread(_find)
        return KnowledgeDocument(**data) if data else None

    async def update_document(
        self,
        doc_id: str,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rechunk: bool = False,
        reembed: bool = False,
        create_version: bool = True,
        change_summary: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Optional[KnowledgeDocument]:
        """Update a document, optionally creating a version snapshot."""
        doc = await self.get_document(doc_id)
        if not doc:
            return None

        # Create version snapshot before updating (if content is changing)
        if create_version and content is not None and content != doc.content:
            await self._create_version_snapshot(
                doc=doc,
                change_summary=change_summary,
                created_by=created_by,
            )

        # Handle content update with rechunking
        if content and rechunk:
            # Delete existing chunks
            await self._delete_chunks_for_document(doc_id)

            # Update content and rechunk
            kb = await self.get_knowledge_base(doc.knowledge_base_id)
            if not kb:
                raise KnowledgeBaseNotFoundError(
                    f"Knowledge base not found: {doc.knowledge_base_id}"
                )

            doc.content = content
            doc.content_hash = _content_hash(content)
            await self._process_document_chunks(doc, kb, auto_embed=reembed)

        # Update document fields
        def _update() -> bool:
            from sqlalchemy import text
            import json

            updates = ["updated_at = NOW()"]
            params: Dict[str, Any] = {"id": doc_id}

            if title is not None:
                updates.append("title = :title")
                params["title"] = title
            if content is not None:
                updates.append("content = :content")
                updates.append("content_hash = :content_hash")
                params["content"] = content
                params["content_hash"] = _content_hash(content)
                # Increment version if content changed
                if create_version:
                    updates.append("current_version = current_version + 1")
                    updates.append("version_count = version_count + 1")
            if metadata is not None:
                updates.append("metadata = metadata || :metadata")
                params["metadata"] = json.dumps(metadata)

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"""
                        UPDATE {self._schema}.{self.TABLE_DOCUMENTS}
                        SET {", ".join(updates)}
                        WHERE id = :id
                    """),
                    params,
                )
                conn.commit()
                return result.rowcount > 0

        await asyncio.to_thread(_update)
        return await self.get_document(doc_id)

    async def _create_version_snapshot(
        self,
        doc: KnowledgeDocument,
        change_summary: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> None:
        """Create a version snapshot of the document's current state."""

        def _create() -> None:
            from sqlalchemy import text
            import json

            with self._engine.connect() as conn:
                conn.execute(
                    text(f"""
                        INSERT INTO {self._schema}.{self.TABLE_DOCUMENT_VERSIONS}
                        (document_id, version_number, title, content, content_hash,
                         change_summary, created_by, metadata)
                        VALUES (:doc_id, :version, :title, :content, :hash,
                                :summary, :created_by, :metadata)
                    """),
                    {
                        "doc_id": doc.id,
                        "version": doc.current_version,
                        "title": doc.title,
                        "content": doc.content or "",
                        "hash": doc.content_hash,
                        "summary": change_summary,
                        "created_by": created_by,
                        "metadata": json.dumps(doc.metadata),
                    },
                )
                conn.commit()

        await asyncio.to_thread(_create)

    async def _delete_chunks_for_document(self, doc_id: str) -> None:
        """Delete all chunks for a document."""

        def _delete() -> None:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                conn.execute(
                    text(f"""
                        DELETE FROM {self._schema}.{self.TABLE_CHUNKS}
                        WHERE document_id = :doc_id
                    """),
                    {"doc_id": doc_id},
                )
                conn.commit()

        await asyncio.to_thread(_delete)

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        doc = await self.get_document(doc_id)
        if not doc:
            return False

        def _delete() -> bool:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                # Cascade delete handles chunks
                result = conn.execute(
                    text(f"""
                        DELETE FROM {self._schema}.{self.TABLE_DOCUMENTS}
                        WHERE id = :id
                    """),
                    {"id": doc_id},
                )

                # Update knowledge base counts
                conn.execute(
                    text(f"""
                        UPDATE {self._schema}.{self.TABLE_KNOWLEDGE_BASES}
                        SET document_count = GREATEST(0, document_count - 1),
                            chunk_count = GREATEST(0, chunk_count - :chunk_count),
                            updated_at = NOW()
                        WHERE id = :kb_id
                    """),
                    {"kb_id": doc.knowledge_base_id, "chunk_count": doc.chunk_count},
                )

                conn.commit()
                return result.rowcount > 0

        deleted = await asyncio.to_thread(_delete)
        if deleted:
            logger.info(f"Deleted document: {doc_id}")
        return deleted

    # --- Chunk Management ---

    async def get_chunks(
        self,
        doc_id: str,
        *,
        include_embeddings: bool = False,
    ) -> List[KnowledgeChunk]:
        """Get all chunks for a document."""

        def _get() -> List[Dict[str, Any]]:
            from sqlalchemy import text

            columns = """
                id, document_id, knowledge_base_id, content, chunk_index,
                token_count, start_char, end_char, start_line, end_line,
                section, language, metadata, created_at
            """
            if include_embeddings:
                columns += ", embedding"

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"""
                        SELECT {columns}
                        FROM {self._schema}.{self.TABLE_CHUNKS}
                        WHERE document_id = :doc_id
                        ORDER BY chunk_index
                    """),
                    {"doc_id": doc_id},
                )
                rows = result.fetchall()

            chunks = []
            for row in rows:
                chunk_data = {
                    "id": str(row[0]),
                    "document_id": str(row[1]),
                    "knowledge_base_id": str(row[2]),
                    "content": row[3],
                    "chunk_index": row[4],
                    "token_count": row[5],
                    "metadata": ChunkMetadata(
                        start_char=row[6],
                        end_char=row[7],
                        start_line=row[8],
                        end_line=row[9],
                        section=row[10],
                        language=row[11],
                        extra=row[12] or {},
                    ),
                    "created_at": row[13],
                }
                if include_embeddings:
                    chunk_data["embedding"] = list(row[14]) if row[14] else None
                chunks.append(chunk_data)

            return chunks

        data_list = await asyncio.to_thread(_get)
        return [KnowledgeChunk(**data) for data in data_list]

    async def get_chunk(
        self,
        chunk_id: str,
        *,
        include_embedding: bool = False,
    ) -> Optional[KnowledgeChunk]:
        """Get a specific chunk by ID."""

        def _get() -> Optional[Dict[str, Any]]:
            from sqlalchemy import text

            columns = """
                id, document_id, knowledge_base_id, content, chunk_index,
                token_count, start_char, end_char, start_line, end_line,
                section, language, metadata, created_at
            """
            if include_embedding:
                columns += ", embedding"

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"""
                        SELECT {columns}
                        FROM {self._schema}.{self.TABLE_CHUNKS}
                        WHERE id = :id
                    """),
                    {"id": chunk_id},
                )
                row = result.fetchone()
                if not row:
                    return None

                chunk_data = {
                    "id": str(row[0]),
                    "document_id": str(row[1]),
                    "knowledge_base_id": str(row[2]),
                    "content": row[3],
                    "chunk_index": row[4],
                    "token_count": row[5],
                    "metadata": ChunkMetadata(
                        start_char=row[6],
                        end_char=row[7],
                        start_line=row[8],
                        end_line=row[9],
                        section=row[10],
                        language=row[11],
                        extra=row[12] or {},
                    ),
                    "created_at": row[13],
                }
                if include_embedding:
                    chunk_data["embedding"] = list(row[14]) if row[14] else None
                return chunk_data

        data = await asyncio.to_thread(_get)
        if not data:
            return None

        return KnowledgeChunk(**data)

    async def update_chunk(
        self,
        chunk_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reembed: bool = True,
    ) -> Optional[KnowledgeChunk]:
        """Update a chunk's content or metadata."""
        # Get existing chunk
        chunk = await self.get_chunk(chunk_id)
        if not chunk:
            return None

        # Prepare updates
        new_embedding = None
        new_token_count = None

        if content is not None and reembed and self._embedding_provider:
            # Generate new embedding for updated content
            await self._embedding_provider.initialize()
            result = await self._embedding_provider.embed_text(content)
            new_embedding = result.embedding
            new_token_count = result.token_count

        def _update() -> bool:
            from sqlalchemy import text
            import json

            updates = []
            params: Dict[str, Any] = {"id": chunk_id}

            if content is not None:
                updates.append("content = :content")
                params["content"] = content
                if new_token_count is not None:
                    updates.append("token_count = :token_count")
                    params["token_count"] = new_token_count

            if metadata is not None:
                updates.append("metadata = metadata || :metadata")
                params["metadata"] = json.dumps(metadata)

            if new_embedding is not None:
                updates.append("embedding = :embedding")
                params["embedding"] = new_embedding

            if not updates:
                return True  # Nothing to update

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"""
                        UPDATE {self._schema}.{self.TABLE_CHUNKS}
                        SET {", ".join(updates)}
                        WHERE id = :id
                    """),
                    params,
                )
                conn.commit()
                return result.rowcount > 0

        await asyncio.to_thread(_update)
        return await self.get_chunk(chunk_id)

    # --- Search ---

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for relevant chunks across knowledge bases."""
        # Get or compute query embedding
        query_embedding = query.query_embedding
        if not query_embedding and query.query_text:
            if not self._embedding_provider:
                raise KnowledgeStoreError("No embedding provider configured for search")
            await self._embedding_provider.initialize()
            result = await self._embedding_provider.embed_text(query.query_text)
            query_embedding = result.embedding

        if not query_embedding:
            raise KnowledgeStoreError("No query text or embedding provided")

        def _search() -> List[Dict[str, Any]]:
            from sqlalchemy import text
            import json

            # Build query
            embedding_str = f"[{','.join(str(v) for v in query_embedding)}]"

            where_clauses = ["c.embedding IS NOT NULL"]
            params: Dict[str, Any] = {
                "embedding": embedding_str,
                "top_k": query.top_k,
            }

            if query.knowledge_base_ids:
                placeholders = ", ".join(
                    f":kb_{i}" for i in range(len(query.knowledge_base_ids))
                )
                where_clauses.append(f"c.knowledge_base_id IN ({placeholders})")
                for i, kb_id in enumerate(query.knowledge_base_ids):
                    params[f"kb_{i}"] = kb_id

            if query.document_ids:
                placeholders = ", ".join(
                    f":doc_{i}" for i in range(len(query.document_ids))
                )
                where_clauses.append(f"c.document_id IN ({placeholders})")
                for i, doc_id in enumerate(query.document_ids):
                    params[f"doc_{i}"] = doc_id

            where_sql = " AND ".join(where_clauses)

            sql = f"""
                SELECT 
                    c.id,
                    c.document_id,
                    c.knowledge_base_id,
                    c.content,
                    c.chunk_index,
                    c.token_count,
                    c.start_char,
                    c.end_char,
                    c.section,
                    c.metadata,
                    c.created_at,
                    1 - (c.embedding <=> :embedding::vector) AS score,
                    c.embedding <=> :embedding::vector AS distance
                FROM {self._schema}.{self.TABLE_CHUNKS} c
                WHERE {where_sql}
                ORDER BY distance
                LIMIT :top_k
            """

            with self._engine.connect() as conn:
                result = conn.execute(text(sql), params)
                rows = result.fetchall()

            results = []
            for row in rows:
                score = float(row[11]) if row[11] else 0.0
                if score < query.min_score:
                    continue

                chunk = KnowledgeChunk(
                    id=str(row[0]),
                    document_id=str(row[1]),
                    knowledge_base_id=str(row[2]),
                    content=row[3] if query.include_content else "",
                    chunk_index=row[4],
                    token_count=row[5],
                    metadata=ChunkMetadata(
                        start_char=row[6],
                        end_char=row[7],
                        section=row[8],
                        extra=row[9] or {},
                    ),
                    created_at=row[10],
                )

                results.append({
                    "chunk": chunk,
                    "score": score,
                    "distance": float(row[12]) if row[12] else None,
                })

            return results

        search_results = await asyncio.to_thread(_search)

        # Optionally fetch document info
        final_results = []
        for result_data in search_results:
            doc = None
            if query.include_document:
                doc = await self.get_document(result_data["chunk"].document_id)

            final_results.append(
                SearchResult(
                    chunk=result_data["chunk"],
                    document=doc,
                    score=result_data["score"],
                    distance=result_data["distance"],
                )
            )

        return final_results

    # --- Version Management ---

    async def get_document_versions(
        self,
        doc_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> List[DocumentVersion]:
        """Get version history for a document."""

        def _get_versions() -> List[Dict[str, Any]]:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"""
                        SELECT id, document_id, version_number, title, content,
                               content_hash, change_summary, created_by, metadata, created_at
                        FROM {self._schema}.{self.TABLE_DOCUMENT_VERSIONS}
                        WHERE document_id = :doc_id
                        ORDER BY version_number DESC
                        LIMIT :limit OFFSET :offset
                    """),
                    {"doc_id": doc_id, "limit": limit, "offset": offset},
                )
                return [dict(row._mapping) for row in result.fetchall()]

        rows = await asyncio.to_thread(_get_versions)
        return [
            DocumentVersion(
                id=str(row["id"]),
                document_id=str(row["document_id"]),
                version_number=row["version_number"],
                title=row["title"],
                content=row["content"],
                content_hash=row["content_hash"],
                change_summary=row["change_summary"],
                created_by=row["created_by"],
                metadata=row["metadata"] or {},
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def get_document_version(
        self,
        doc_id: str,
        version_number: int,
    ) -> Optional[DocumentVersion]:
        """Get a specific version of a document."""

        def _get_version() -> Optional[Dict[str, Any]]:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"""
                        SELECT id, document_id, version_number, title, content,
                               content_hash, change_summary, created_by, metadata, created_at
                        FROM {self._schema}.{self.TABLE_DOCUMENT_VERSIONS}
                        WHERE document_id = :doc_id AND version_number = :version
                    """),
                    {"doc_id": doc_id, "version": version_number},
                )
                row = result.fetchone()
                return dict(row._mapping) if row else None

        row = await asyncio.to_thread(_get_version)
        if not row:
            return None

        return DocumentVersion(
            id=str(row["id"]),
            document_id=str(row["document_id"]),
            version_number=row["version_number"],
            title=row["title"],
            content=row["content"],
            content_hash=row["content_hash"],
            change_summary=row["change_summary"],
            created_by=row["created_by"],
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
        )

    async def restore_document_version(
        self,
        doc_id: str,
        version_number: int,
        *,
        rechunk: bool = True,
        reembed: bool = True,
        created_by: Optional[str] = None,
    ) -> Optional[KnowledgeDocument]:
        """Restore a document to a previous version."""
        # Get the version to restore
        version = await self.get_document_version(doc_id, version_number)
        if not version:
            return None

        # Update the document with the version's content
        return await self.update_document(
            doc_id,
            title=version.title,
            content=version.content,
            rechunk=rechunk,
            reembed=reembed,
            create_version=True,
            change_summary=f"Restored from version {version_number}",
            created_by=created_by,
        )

    async def compare_document_versions(
        self,
        doc_id: str,
        version_a: int,
        version_b: int,
    ) -> Optional[Dict[str, Any]]:
        """Compare two versions of a document."""
        version_a_data = await self.get_document_version(doc_id, version_a)
        version_b_data = await self.get_document_version(doc_id, version_b)

        if not version_a_data or not version_b_data:
            return None

        # Basic diff information
        import difflib

        diff_lines = list(
            difflib.unified_diff(
                (version_a_data.content or "").splitlines(keepends=True),
                (version_b_data.content or "").splitlines(keepends=True),
                fromfile=f"Version {version_a}",
                tofile=f"Version {version_b}",
                lineterm="",
            )
        )

        # Count changes
        additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

        return {
            "document_id": doc_id,
            "version_a": {
                "number": version_a,
                "title": version_a_data.title,
                "created_at": version_a_data.created_at.isoformat() if version_a_data.created_at else None,
                "content_length": len(version_a_data.content or ""),
            },
            "version_b": {
                "number": version_b,
                "title": version_b_data.title,
                "created_at": version_b_data.created_at.isoformat() if version_b_data.created_at else None,
                "content_length": len(version_b_data.content or ""),
            },
            "title_changed": version_a_data.title != version_b_data.title,
            "content_changed": version_a_data.content != version_b_data.content,
            "additions": additions,
            "deletions": deletions,
            "diff": "".join(diff_lines),
        }

    async def delete_document_version(
        self,
        doc_id: str,
        version_number: int,
    ) -> bool:
        """Delete a specific version from history."""
        # First check if this is the current version
        doc = await self.get_document(doc_id)
        if not doc:
            return False

        if doc.current_version == version_number:
            # Cannot delete the current version
            logger.warning(
                f"Cannot delete current version {version_number} of document {doc_id}"
            )
            return False

        def _delete() -> bool:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"""
                        DELETE FROM {self._schema}.{self.TABLE_DOCUMENT_VERSIONS}
                        WHERE document_id = :doc_id AND version_number = :version
                    """),
                    {"doc_id": doc_id, "version": version_number},
                )

                if result.rowcount > 0:
                    # Decrement version_count
                    conn.execute(
                        text(f"""
                            UPDATE {self._schema}.{self.TABLE_DOCUMENTS}
                            SET version_count = version_count - 1
                            WHERE id = :doc_id
                        """),
                        {"doc_id": doc_id},
                    )

                conn.commit()
                return result.rowcount > 0

        return await asyncio.to_thread(_delete)
