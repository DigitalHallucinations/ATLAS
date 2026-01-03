"""Integration tests for the RAG (Retrieval-Augmented Generation) system.

Tests cover end-to-end workflows including document ingestion,
chunking, embedding, and search operations.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.storage.knowledge import (
    ChunkMetadata,
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
)


# --- Mock Fixtures ---


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> List[float]:
        """Generate a deterministic mock embedding based on text hash."""
        hash_val = hash(text) % 1000
        return [float(hash_val + i) / 1000 for i in range(self._dimension)]

    async def embed_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """Generate mock embeddings for a batch of texts."""
        return [await self.embed_text(t) for t in texts]


class MockKnowledgeStore(KnowledgeStore):
    """In-memory mock knowledge store for testing."""

    def __init__(self):
        self._kbs: Dict[str, KnowledgeBase] = {}
        self._docs: Dict[str, KnowledgeDocument] = {}
        self._chunks: Dict[str, KnowledgeChunk] = {}
        self._versions: Dict[str, List[DocumentVersion]] = {}  # doc_id -> versions
        self._initialized: bool = False

    @property
    def name(self) -> str:
        """Return the store implementation name."""
        return "mock"

    @property
    def is_initialized(self) -> bool:
        """Check if the store has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def health_check(self, timeout: float = 5.0) -> bool:
        return self._initialized

    async def close(self) -> None:
        pass

    async def create_knowledge_base(
        self,
        name: str,
        description: str = "",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        owner_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBase:
        import uuid

        kb_id = str(uuid.uuid4())
        kb = KnowledgeBase(
            id=kb_id,
            name=name,
            description=description,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            owner_id=owner_id,
            metadata=metadata or {},
            document_count=0,
            chunk_count=0,
        )
        self._kbs[kb_id] = kb
        return kb

    async def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        return self._kbs.get(kb_id)

    async def list_knowledge_bases(
        self, owner_id: Optional[str] = None
    ) -> List[KnowledgeBase]:
        kbs = list(self._kbs.values())
        if owner_id:
            kbs = [kb for kb in kbs if kb.owner_id == owner_id]
        return kbs

    async def update_knowledge_base(
        self,
        kb_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBase:
        kb = self._kbs.get(kb_id)
        if not kb:
            raise KnowledgeBaseNotFoundError(f"KB not found: {kb_id}")
        if name:
            kb.name = name
        if description is not None:
            kb.description = description
        if metadata is not None:
            kb.metadata = metadata
        return kb

    async def delete_knowledge_base(
        self, kb_id: str, delete_documents: bool = True
    ) -> bool:
        if kb_id not in self._kbs:
            return False
        del self._kbs[kb_id]
        if delete_documents:
            # Delete associated documents and chunks
            doc_ids = [d.id for d in self._docs.values() if d.knowledge_base_id == kb_id]
            for doc_id in doc_ids:
                del self._docs[doc_id]
            chunk_ids = [
                c.id for c in self._chunks.values() if c.knowledge_base_id == kb_id
            ]
            for chunk_id in chunk_ids:
                del self._chunks[chunk_id]
        return True

    async def add_document(
        self,
        kb_id: str,
        title: str,
        content: str,
        source_uri: Optional[str] = None,
        document_type: DocumentType = DocumentType.TEXT,
        metadata: Optional[Dict[str, Any]] = None,
        auto_chunk: bool = True,
        auto_embed: bool = True,
    ) -> KnowledgeDocument:
        import uuid

        if kb_id not in self._kbs:
            raise KnowledgeBaseNotFoundError(f"KB not found: {kb_id}")

        doc_id = str(uuid.uuid4())
        doc = KnowledgeDocument(
            id=doc_id,
            knowledge_base_id=kb_id,
            title=title,
            content=content,
            source_uri=source_uri,
            document_type=document_type,
            status=DocumentStatus.INDEXED if auto_chunk else DocumentStatus.PENDING,
            metadata=metadata or {},
            chunk_count=0,
            token_count=len(content.split()),
        )
        self._docs[doc_id] = doc

        # Update KB document count
        self._kbs[kb_id].document_count += 1

        # Auto-chunk if requested
        if auto_chunk:
            chunk_size = self._kbs[kb_id].chunk_size
            words = content.split()
            chunk_count = 0
            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i : i + chunk_size])
                chunk_id = str(uuid.uuid4())
                chunk = KnowledgeChunk(
                    id=chunk_id,
                    document_id=doc_id,
                    knowledge_base_id=kb_id,
                    content=chunk_text,
                    chunk_index=chunk_count,
                    embedding=[0.0] * self._kbs[kb_id].embedding_dimension
                    if auto_embed
                    else None,
                    token_count=len(chunk_text.split()),
                )
                self._chunks[chunk_id] = chunk
                chunk_count += 1

            doc.chunk_count = chunk_count
            self._kbs[kb_id].chunk_count += chunk_count

        return doc

    async def get_document(self, doc_id: str) -> Optional[KnowledgeDocument]:
        return self._docs.get(doc_id)

    async def list_documents(
        self, kb_id: str, limit: int = 100, offset: int = 0
    ) -> List[KnowledgeDocument]:
        docs = [d for d in self._docs.values() if d.knowledge_base_id == kb_id]
        return docs[offset : offset + limit]

    async def update_document(
        self,
        doc_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rechunk: bool = False,
        reembed: bool = False,
        create_version: bool = True,
        change_summary: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> KnowledgeDocument:
        doc = self._docs.get(doc_id)
        if not doc:
            raise DocumentNotFoundError(f"Document not found: {doc_id}")

        # Create version snapshot before updating
        if create_version and content is not None and content != doc.content:
            import uuid
            import hashlib

            version = DocumentVersion(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                version_number=doc.current_version,
                title=doc.title,
                content=doc.content or "",
                content_hash=hashlib.sha256((doc.content or "").encode()).hexdigest(),
                change_summary=change_summary,
                created_by=created_by,
            )
            if doc_id not in self._versions:
                self._versions[doc_id] = []
            self._versions[doc_id].append(version)
            doc.current_version += 1
            doc.version_count += 1

        if title:
            doc.title = title
        if content is not None:
            doc.content = content
        if metadata is not None:
            doc.metadata = metadata
        return doc

    async def delete_document(self, doc_id: str) -> bool:
        doc = self._docs.get(doc_id)
        if not doc:
            return False
        kb_id = doc.knowledge_base_id
        del self._docs[doc_id]
        # Delete associated chunks
        chunk_ids = [c.id for c in self._chunks.values() if c.document_id == doc_id]
        for chunk_id in chunk_ids:
            del self._chunks[chunk_id]
        # Update counts
        if kb_id in self._kbs:
            self._kbs[kb_id].document_count -= 1
            self._kbs[kb_id].chunk_count -= len(chunk_ids)
        return True

    async def find_duplicate(
        self, kb_id: str, content: str
    ) -> Optional[KnowledgeDocument]:
        """Check for duplicate document by content."""
        import hashlib

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        for doc in self._docs.values():
            if doc.knowledge_base_id == kb_id and doc.content:
                doc_hash = hashlib.sha256(doc.content.encode()).hexdigest()
                if doc_hash == content_hash:
                    return doc
        return None

    async def get_chunks(
        self, doc_id: str, include_embeddings: bool = False
    ) -> List[KnowledgeChunk]:
        chunks = [c for c in self._chunks.values() if c.document_id == doc_id]
        if not include_embeddings:
            # Return copies without embeddings
            return [
                KnowledgeChunk(
                    id=c.id,
                    document_id=c.document_id,
                    knowledge_base_id=c.knowledge_base_id,
                    content=c.content,
                    chunk_index=c.chunk_index,
                    embedding=None,
                    token_count=c.token_count,
                )
                for c in chunks
            ]
        return sorted(chunks, key=lambda c: c.chunk_index)

    async def get_chunk(self, chunk_id: str) -> Optional[KnowledgeChunk]:
        return self._chunks.get(chunk_id)

    async def update_chunk(
        self,
        chunk_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[ChunkMetadata] = None,
    ) -> KnowledgeChunk:
        chunk = self._chunks.get(chunk_id)
        if not chunk:
            raise KnowledgeStoreError(f"Chunk not found: {chunk_id}")
        if content is not None:
            chunk.content = content
        if embedding is not None:
            chunk.embedding = embedding
        if metadata is not None:
            chunk.metadata = metadata
        return chunk

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Simple text-based search for testing."""
        results = []
        query_lower = query.query_text.lower()

        for chunk in self._chunks.values():
            # Filter by KB if specified
            if query.knowledge_base_ids:
                if chunk.knowledge_base_id not in query.knowledge_base_ids:
                    continue

            # Simple relevance scoring based on word overlap
            chunk_words = set(chunk.content.lower().split())
            query_words = set(query_lower.split())
            overlap = len(chunk_words & query_words)
            if overlap > 0:
                score = overlap / len(query_words)
                doc = self._docs.get(chunk.document_id)
                if doc:
                    results.append(
                        SearchResult(
                            chunk=chunk,
                            document=doc,
                            score=score,
                        )
                    )

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: query.top_k]

    # --- Version Management ---

    async def get_document_versions(
        self,
        doc_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> List[DocumentVersion]:
        versions = self._versions.get(doc_id, [])
        # Sort by version number descending
        sorted_versions = sorted(versions, key=lambda v: v.version_number, reverse=True)
        return sorted_versions[offset : offset + limit]

    async def get_document_version(
        self,
        doc_id: str,
        version_number: int,
    ) -> Optional[DocumentVersion]:
        versions = self._versions.get(doc_id, [])
        for v in versions:
            if v.version_number == version_number:
                return v
        return None

    async def restore_document_version(
        self,
        doc_id: str,
        version_number: int,
        *,
        rechunk: bool = True,
        reembed: bool = True,
        created_by: Optional[str] = None,
    ) -> Optional[KnowledgeDocument]:
        version = await self.get_document_version(doc_id, version_number)
        if not version:
            return None

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
        doc = self._docs.get(doc_id)
        if not doc:
            return None

        # Helper to get version content (current version = document content)
        def get_version_info(version_num: int) -> Optional[Dict[str, Any]]:
            if doc.current_version == version_num:
                return {"title": doc.title, "content": doc.content}
            for v in self._versions.get(doc_id, []):
                if v.version_number == version_num:
                    return {"title": v.title, "content": v.content}
            return None

        va_info = get_version_info(version_a)
        vb_info = get_version_info(version_b)
        if not va_info or not vb_info:
            return None

        return {
            "document_id": doc_id,
            "version_a": {"number": version_a, "title": va_info["title"]},
            "version_b": {"number": version_b, "title": vb_info["title"]},
            "content_changed": va_info["content"] != vb_info["content"],
        }

    async def delete_document_version(
        self,
        doc_id: str,
        version_number: int,
    ) -> bool:
        doc = self._docs.get(doc_id)
        if not doc or doc.current_version == version_number:
            return False

        versions = self._versions.get(doc_id, [])
        for i, v in enumerate(versions):
            if v.version_number == version_number:
                versions.pop(i)
                doc.version_count -= 1
                return True
        return False


@pytest.fixture
def mock_store() -> MockKnowledgeStore:
    """Create a mock knowledge store for testing."""
    return MockKnowledgeStore()


@pytest.fixture
def mock_embedding_provider() -> MockEmbeddingProvider:
    """Create a mock embedding provider for testing."""
    return MockEmbeddingProvider()


# --- Knowledge Base Tests ---


class TestKnowledgeBaseOperations:
    """Tests for knowledge base CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_knowledge_base(self, mock_store: MockKnowledgeStore):
        """Test creating a new knowledge base."""
        kb = await mock_store.create_knowledge_base(
            name="Test KB",
            description="A test knowledge base",
            embedding_model="text-embedding-3-small",
            embedding_dimension=1536,
        )

        assert kb.id is not None
        assert kb.name == "Test KB"
        assert kb.description == "A test knowledge base"
        assert kb.embedding_model == "text-embedding-3-small"
        assert kb.embedding_dimension == 1536
        assert kb.document_count == 0
        assert kb.chunk_count == 0

    @pytest.mark.asyncio
    async def test_get_knowledge_base(self, mock_store: MockKnowledgeStore):
        """Test retrieving a knowledge base by ID."""
        created = await mock_store.create_knowledge_base(name="Test KB")
        retrieved = await mock_store.get_knowledge_base(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == created.name

    @pytest.mark.asyncio
    async def test_get_nonexistent_knowledge_base(self, mock_store: MockKnowledgeStore):
        """Test retrieving a non-existent knowledge base."""
        result = await mock_store.get_knowledge_base("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_knowledge_bases(self, mock_store: MockKnowledgeStore):
        """Test listing knowledge bases."""
        await mock_store.create_knowledge_base(name="KB 1")
        await mock_store.create_knowledge_base(name="KB 2")
        await mock_store.create_knowledge_base(name="KB 3")

        kbs = await mock_store.list_knowledge_bases()
        assert len(kbs) == 3
        names = {kb.name for kb in kbs}
        assert names == {"KB 1", "KB 2", "KB 3"}

    @pytest.mark.asyncio
    async def test_list_knowledge_bases_by_owner(self, mock_store: MockKnowledgeStore):
        """Test listing knowledge bases filtered by owner."""
        await mock_store.create_knowledge_base(name="KB 1", owner_id="user-1")
        await mock_store.create_knowledge_base(name="KB 2", owner_id="user-1")
        await mock_store.create_knowledge_base(name="KB 3", owner_id="user-2")

        user1_kbs = await mock_store.list_knowledge_bases(owner_id="user-1")
        assert len(user1_kbs) == 2

        user2_kbs = await mock_store.list_knowledge_bases(owner_id="user-2")
        assert len(user2_kbs) == 1

    @pytest.mark.asyncio
    async def test_update_knowledge_base(self, mock_store: MockKnowledgeStore):
        """Test updating a knowledge base."""
        kb = await mock_store.create_knowledge_base(name="Original Name")

        updated = await mock_store.update_knowledge_base(
            kb.id,
            name="Updated Name",
            description="New description",
        )

        assert updated.name == "Updated Name"
        assert updated.description == "New description"

    @pytest.mark.asyncio
    async def test_delete_knowledge_base(self, mock_store: MockKnowledgeStore):
        """Test deleting a knowledge base."""
        kb = await mock_store.create_knowledge_base(name="To Delete")

        result = await mock_store.delete_knowledge_base(kb.id)
        assert result is True

        retrieved = await mock_store.get_knowledge_base(kb.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_kb_cascades_to_documents(self, mock_store: MockKnowledgeStore):
        """Test that deleting a KB also deletes its documents."""
        kb = await mock_store.create_knowledge_base(name="KB with docs")
        doc = await mock_store.add_document(
            kb.id, title="Test Doc", content="Test content"
        )

        await mock_store.delete_knowledge_base(kb.id, delete_documents=True)

        retrieved_doc = await mock_store.get_document(doc.id)
        assert retrieved_doc is None


# --- Document Tests ---


class TestDocumentOperations:
    """Tests for document CRUD operations."""

    @pytest.mark.asyncio
    async def test_add_document(self, mock_store: MockKnowledgeStore):
        """Test adding a document to a knowledge base."""
        kb = await mock_store.create_knowledge_base(name="Test KB")

        doc = await mock_store.add_document(
            kb.id,
            title="Test Document",
            content="This is the test document content.",
            document_type=DocumentType.TEXT,
        )

        assert doc.id is not None
        assert doc.title == "Test Document"
        assert doc.knowledge_base_id == kb.id
        assert doc.status == DocumentStatus.INDEXED

    @pytest.mark.asyncio
    async def test_add_document_auto_chunks(self, mock_store: MockKnowledgeStore):
        """Test that adding a document creates chunks."""
        kb = await mock_store.create_knowledge_base(
            name="Test KB", chunk_size=10  # Small chunks for testing
        )

        # Create content with more than 10 words to trigger multiple chunks
        content = " ".join([f"word{i}" for i in range(25)])
        doc = await mock_store.add_document(
            kb.id, title="Long Doc", content=content, auto_chunk=True
        )

        chunks = await mock_store.get_chunks(doc.id)
        assert len(chunks) > 1

    @pytest.mark.asyncio
    async def test_add_document_to_nonexistent_kb(self, mock_store: MockKnowledgeStore):
        """Test adding a document to a non-existent KB raises error."""
        with pytest.raises(KnowledgeBaseNotFoundError):
            await mock_store.add_document(
                "nonexistent-kb",
                title="Test",
                content="Content",
            )

    @pytest.mark.asyncio
    async def test_get_document(self, mock_store: MockKnowledgeStore):
        """Test retrieving a document by ID."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        created = await mock_store.add_document(
            kb.id, title="Test Doc", content="Content"
        )

        retrieved = await mock_store.get_document(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.title == created.title

    @pytest.mark.asyncio
    async def test_list_documents(self, mock_store: MockKnowledgeStore):
        """Test listing documents in a knowledge base."""
        kb = await mock_store.create_knowledge_base(name="Test KB")

        await mock_store.add_document(kb.id, title="Doc 1", content="Content 1")
        await mock_store.add_document(kb.id, title="Doc 2", content="Content 2")
        await mock_store.add_document(kb.id, title="Doc 3", content="Content 3")

        docs = await mock_store.list_documents(kb.id)
        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_update_document(self, mock_store: MockKnowledgeStore):
        """Test updating a document."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        doc = await mock_store.add_document(
            kb.id, title="Original Title", content="Original content"
        )

        updated = await mock_store.update_document(
            doc.id,
            title="New Title",
            content="Updated content",
        )

        assert updated.title == "New Title"
        assert updated.content == "Updated content"

    @pytest.mark.asyncio
    async def test_delete_document(self, mock_store: MockKnowledgeStore):
        """Test deleting a document."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        doc = await mock_store.add_document(
            kb.id, title="To Delete", content="Content"
        )

        result = await mock_store.delete_document(doc.id)
        assert result is True

        retrieved = await mock_store.get_document(doc.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_find_duplicate(self, mock_store: MockKnowledgeStore):
        """Test finding duplicate documents by content."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        content = "This is unique content for duplicate detection."

        # Add first document
        doc1 = await mock_store.add_document(
            kb.id, title="Original", content=content
        )

        # Try to find duplicate
        duplicate = await mock_store.find_duplicate(kb.id, content)
        assert duplicate is not None
        assert duplicate.id == doc1.id

    @pytest.mark.asyncio
    async def test_find_duplicate_not_found(self, mock_store: MockKnowledgeStore):
        """Test that non-duplicates return None."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        await mock_store.add_document(
            kb.id, title="Doc 1", content="Content one"
        )

        duplicate = await mock_store.find_duplicate(kb.id, "Completely different content")
        assert duplicate is None


# --- Chunk Tests ---


class TestChunkOperations:
    """Tests for chunk operations."""

    @pytest.mark.asyncio
    async def test_get_chunks_for_document(self, mock_store: MockKnowledgeStore):
        """Test retrieving chunks for a document."""
        kb = await mock_store.create_knowledge_base(name="Test KB", chunk_size=10)
        content = " ".join([f"word{i}" for i in range(25)])
        doc = await mock_store.add_document(kb.id, title="Doc", content=content)

        chunks = await mock_store.get_chunks(doc.id)
        assert len(chunks) > 0
        assert all(c.document_id == doc.id for c in chunks)

    @pytest.mark.asyncio
    async def test_get_chunks_with_embeddings(self, mock_store: MockKnowledgeStore):
        """Test retrieving chunks with embeddings included."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        doc = await mock_store.add_document(
            kb.id, title="Doc", content="Test content", auto_embed=True
        )

        chunks_with = await mock_store.get_chunks(doc.id, include_embeddings=True)
        chunks_without = await mock_store.get_chunks(doc.id, include_embeddings=False)

        assert all(c.embedding is not None for c in chunks_with)
        assert all(c.embedding is None for c in chunks_without)

    @pytest.mark.asyncio
    async def test_update_chunk(self, mock_store: MockKnowledgeStore):
        """Test updating a chunk's content."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        doc = await mock_store.add_document(
            kb.id, title="Doc", content="Original chunk content"
        )
        chunks = await mock_store.get_chunks(doc.id)
        assert len(chunks) > 0

        updated = await mock_store.update_chunk(
            chunks[0].id, content="Updated chunk content"
        )
        assert updated.content == "Updated chunk content"


# --- Search Tests ---


class TestSearchOperations:
    """Tests for search functionality."""

    @pytest.mark.asyncio
    async def test_basic_search(self, mock_store: MockKnowledgeStore):
        """Test basic text search."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        await mock_store.add_document(
            kb.id,
            title="Python Guide",
            content="Python is a programming language for data science",
        )
        await mock_store.add_document(
            kb.id,
            title="Java Guide",
            content="Java is a programming language for enterprise",
        )

        query = SearchQuery(
            query_text="Python programming",
            knowledge_base_ids=[kb.id],
            top_k=5,
        )
        results = await mock_store.search(query)

        assert len(results) > 0
        # Python doc should rank higher
        assert "python" in results[0].chunk.content.lower()

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, mock_store: MockKnowledgeStore):
        """Test that search respects top_k limit."""
        kb = await mock_store.create_knowledge_base(name="Test KB")

        # Add many documents
        for i in range(10):
            await mock_store.add_document(
                kb.id,
                title=f"Doc {i}",
                content=f"Document {i} about testing",
            )

        query = SearchQuery(
            query_text="testing",
            knowledge_base_ids=[kb.id],
            top_k=3,
        )
        results = await mock_store.search(query)

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_filters_by_kb(self, mock_store: MockKnowledgeStore):
        """Test that search filters by knowledge base."""
        kb1 = await mock_store.create_knowledge_base(name="KB 1")
        kb2 = await mock_store.create_knowledge_base(name="KB 2")

        await mock_store.add_document(kb1.id, title="KB1 Doc", content="apple banana")
        await mock_store.add_document(kb2.id, title="KB2 Doc", content="apple orange")

        query = SearchQuery(
            query_text="apple",
            knowledge_base_ids=[kb1.id],
            top_k=10,
        )
        results = await mock_store.search(query)

        assert len(results) == 1
        assert results[0].document.title == "KB1 Doc"

    @pytest.mark.asyncio
    async def test_search_multiple_kbs(self, mock_store: MockKnowledgeStore):
        """Test searching across multiple knowledge bases."""
        kb1 = await mock_store.create_knowledge_base(name="KB 1")
        kb2 = await mock_store.create_knowledge_base(name="KB 2")

        await mock_store.add_document(kb1.id, title="KB1 Doc", content="apple banana")
        await mock_store.add_document(kb2.id, title="KB2 Doc", content="apple orange")

        query = SearchQuery(
            query_text="apple",
            knowledge_base_ids=[kb1.id, kb2.id],
            top_k=10,
        )
        results = await mock_store.search(query)

        assert len(results) == 2


# --- Embedding Provider Tests ---


class TestEmbeddingProvider:
    """Tests for embedding provider operations."""

    @pytest.mark.asyncio
    async def test_embed_text(self, mock_embedding_provider: MockEmbeddingProvider):
        """Test generating embedding for text."""
        embedding = await mock_embedding_provider.embed_text("Hello world")

        assert len(embedding) == mock_embedding_provider.dimension
        assert all(isinstance(v, float) for v in embedding)

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_embedding_provider: MockEmbeddingProvider):
        """Test batch embedding generation."""
        texts = ["Text one", "Text two", "Text three"]
        embeddings = await mock_embedding_provider.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(e) == mock_embedding_provider.dimension for e in embeddings)

    @pytest.mark.asyncio
    async def test_embedding_determinism(
        self, mock_embedding_provider: MockEmbeddingProvider
    ):
        """Test that same text produces same embedding."""
        text = "Deterministic embedding test"
        embedding1 = await mock_embedding_provider.embed_text(text)
        embedding2 = await mock_embedding_provider.embed_text(text)

        assert embedding1 == embedding2


# --- Integration Workflow Tests ---


class TestRAGWorkflows:
    """End-to-end workflow tests."""

    @pytest.mark.asyncio
    async def test_full_ingestion_workflow(self, mock_store: MockKnowledgeStore):
        """Test complete document ingestion workflow."""
        # Create KB
        kb = await mock_store.create_knowledge_base(
            name="Project Docs",
            description="Project documentation",
            chunk_size=50,
        )

        # Add documents
        doc1 = await mock_store.add_document(
            kb.id,
            title="README",
            content="This project provides tools for natural language processing.",
        )
        doc2 = await mock_store.add_document(
            kb.id,
            title="Installation",
            content="Install using pip install our-package.",
        )

        # Verify KB state
        kb = await mock_store.get_knowledge_base(kb.id)
        assert kb.document_count == 2
        assert kb.chunk_count > 0

        # Search
        results = await mock_store.search(
            SearchQuery(
                query_text="natural language",
                knowledge_base_ids=[kb.id],
                top_k=5,
            )
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_document_update_workflow(self, mock_store: MockKnowledgeStore):
        """Test document update and re-search workflow."""
        kb = await mock_store.create_knowledge_base(name="Test KB")

        # Add initial document
        doc = await mock_store.add_document(
            kb.id,
            title="API Docs",
            content="The old API uses REST endpoints.",
        )

        # Search for old content
        results = await mock_store.search(
            SearchQuery(query_text="REST", knowledge_base_ids=[kb.id], top_k=5)
        )
        assert len(results) > 0

        # Update document
        await mock_store.update_document(
            doc.id,
            content="The new API uses GraphQL endpoints.",
        )

        # Verify update
        updated = await mock_store.get_document(doc.id)
        assert "GraphQL" in updated.content

    @pytest.mark.asyncio
    async def test_kb_export_import_simulation(self, mock_store: MockKnowledgeStore):
        """Test simulated export/import workflow."""
        # Create source KB with documents
        source_kb = await mock_store.create_knowledge_base(name="Source KB")
        await mock_store.add_document(
            source_kb.id, title="Doc 1", content="Content one"
        )
        await mock_store.add_document(
            source_kb.id, title="Doc 2", content="Content two"
        )

        # Get documents for "export"
        docs = await mock_store.list_documents(source_kb.id)
        export_data = [{"title": d.title, "content": d.content} for d in docs]

        # Create target KB and "import"
        target_kb = await mock_store.create_knowledge_base(name="Imported KB")
        for item in export_data:
            await mock_store.add_document(
                target_kb.id,
                title=item["title"],
                content=item["content"],
            )

        # Verify import
        imported_docs = await mock_store.list_documents(target_kb.id)
        assert len(imported_docs) == 2


# --- Error Handling Tests ---


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_update_nonexistent_kb(self, mock_store: MockKnowledgeStore):
        """Test updating a non-existent KB raises error."""
        with pytest.raises(KnowledgeBaseNotFoundError):
            await mock_store.update_knowledge_base(
                "nonexistent-id", name="New Name"
            )

    @pytest.mark.asyncio
    async def test_update_nonexistent_document(self, mock_store: MockKnowledgeStore):
        """Test updating a non-existent document raises error."""
        with pytest.raises(DocumentNotFoundError):
            await mock_store.update_document("nonexistent-id", title="New Title")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_kb(self, mock_store: MockKnowledgeStore):
        """Test deleting a non-existent KB returns False."""
        result = await mock_store.delete_knowledge_base("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, mock_store: MockKnowledgeStore):
        """Test deleting a non-existent document returns False."""
        result = await mock_store.delete_document("nonexistent-id")
        assert result is False


# --- Document Versioning Tests ---


class TestDocumentVersioning:
    """Tests for document version history functionality."""

    @pytest.mark.asyncio
    async def test_update_creates_version(self, mock_store: MockKnowledgeStore):
        """Test that updating document content creates a version."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        doc = await mock_store.add_document(
            kb.id, title="Doc", content="Original content"
        )

        # Initial state
        assert doc.current_version == 1
        assert doc.version_count == 1

        # Update content
        updated = await mock_store.update_document(
            doc.id, content="Updated content", create_version=True
        )

        assert updated.current_version == 2
        assert updated.version_count == 2

        # Check version history
        versions = await mock_store.get_document_versions(doc.id)
        assert len(versions) == 1
        assert versions[0].version_number == 1
        assert versions[0].content == "Original content"

    @pytest.mark.asyncio
    async def test_update_without_version(self, mock_store: MockKnowledgeStore):
        """Test updating without creating version."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        doc = await mock_store.add_document(
            kb.id, title="Doc", content="Original content"
        )

        # Update without version
        updated = await mock_store.update_document(
            doc.id, content="Updated content", create_version=False
        )

        assert updated.current_version == 1
        versions = await mock_store.get_document_versions(doc.id)
        assert len(versions) == 0

    @pytest.mark.asyncio
    async def test_get_specific_version(self, mock_store: MockKnowledgeStore):
        """Test retrieving a specific version."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        doc = await mock_store.add_document(
            kb.id, title="Doc", content="Version 1"
        )

        # Create multiple versions
        await mock_store.update_document(doc.id, content="Version 2")
        await mock_store.update_document(doc.id, content="Version 3")

        # Get specific version
        v1 = await mock_store.get_document_version(doc.id, 1)
        assert v1 is not None
        assert v1.content == "Version 1"

        v2 = await mock_store.get_document_version(doc.id, 2)
        assert v2 is not None
        assert v2.content == "Version 2"

    @pytest.mark.asyncio
    async def test_get_nonexistent_version(self, mock_store: MockKnowledgeStore):
        """Test getting a version that doesn't exist."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        doc = await mock_store.add_document(
            kb.id, title="Doc", content="Content"
        )

        version = await mock_store.get_document_version(doc.id, 999)
        assert version is None

    @pytest.mark.asyncio
    async def test_restore_version(self, mock_store: MockKnowledgeStore):
        """Test restoring a document to a previous version."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        doc = await mock_store.add_document(
            kb.id, title="Doc", content="Version 1 content"
        )

        # Create version 2
        await mock_store.update_document(doc.id, content="Version 2 content")

        # Restore to version 1
        restored = await mock_store.restore_document_version(doc.id, 1)

        assert restored is not None
        assert restored.content == "Version 1 content"
        assert restored.current_version == 3  # Restore creates a new version

    @pytest.mark.asyncio
    async def test_compare_versions(self, mock_store: MockKnowledgeStore):
        """Test comparing two versions."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        doc = await mock_store.add_document(
            kb.id, title="Doc", content="Original"
        )

        await mock_store.update_document(doc.id, content="Changed")

        comparison = await mock_store.compare_document_versions(doc.id, 1, 2)
        assert comparison is not None
        assert comparison["content_changed"] is True

    @pytest.mark.asyncio
    async def test_delete_version(self, mock_store: MockKnowledgeStore):
        """Test deleting a version from history."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        doc = await mock_store.add_document(
            kb.id, title="Doc", content="Version 1"
        )

        await mock_store.update_document(doc.id, content="Version 2")
        await mock_store.update_document(doc.id, content="Version 3")

        # Delete version 1
        result = await mock_store.delete_document_version(doc.id, 1)
        assert result is True

        # Version should be gone
        v1 = await mock_store.get_document_version(doc.id, 1)
        assert v1 is None

    @pytest.mark.asyncio
    async def test_cannot_delete_current_version(self, mock_store: MockKnowledgeStore):
        """Test that current version cannot be deleted."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        doc = await mock_store.add_document(
            kb.id, title="Doc", content="Content"
        )

        await mock_store.update_document(doc.id, content="Version 2")

        # Try to delete current version (2)
        result = await mock_store.delete_document_version(doc.id, 2)
        assert result is False

    @pytest.mark.asyncio
    async def test_version_with_change_summary(self, mock_store: MockKnowledgeStore):
        """Test version includes change summary."""
        kb = await mock_store.create_knowledge_base(name="Test KB")
        doc = await mock_store.add_document(
            kb.id, title="Doc", content="Original"
        )

        await mock_store.update_document(
            doc.id,
            content="Updated",
            change_summary="Fixed typos",
            created_by="user-123",
        )

        versions = await mock_store.get_document_versions(doc.id)
        assert len(versions) == 1
        assert versions[0].change_summary == "Fixed typos"
        assert versions[0].created_by == "user-123"