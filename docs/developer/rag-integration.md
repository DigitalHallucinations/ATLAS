# RAG Integration Developer Guide

This guide covers how to integrate with and extend the ATLAS RAG (Retrieval-Augmented Generation) system.

## Architecture Overview

The RAG system consists of several key components:

```Text
┌─────────────────────────────────────────────────────────┐
│                      RAGService                         │
│  (ATLAS/services/rag/rag_service.py)                   │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐  │
│  │ EmbedManager│ │KnowledgeStore│ │DocumentIngester │  │
│  └─────────────┘ └─────────────┘ └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
| --------- | -------------- |
| **RAGService** | Orchestration facade for all RAG operations |
| **EmbedManager** | Manages embedding provider connections |
| **KnowledgeStore** | Persistence layer for KBs, documents, chunks |
| **DocumentIngester** | Handles file parsing, chunking, embedding |

## Core Data Models

### KnowledgeBase

```python
@dataclass
class KnowledgeBase:
    id: str
    name: str
    description: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 50
    document_count: int = 0
    chunk_count: int = 0
    owner_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### KnowledgeDocument

```python
@dataclass
class KnowledgeDocument:
    id: str
    knowledge_base_id: str
    title: str
    content: Optional[str] = None
    content_hash: Optional[str] = None
    source_uri: Optional[str] = None
    document_type: DocumentType = DocumentType.TEXT
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### KnowledgeChunk

```python
@dataclass
class KnowledgeChunk:
    id: str
    document_id: str
    knowledge_base_id: str
    content: str
    index: int = 0
    embedding: Optional[List[float]] = None
    token_count: int = 0
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
```

## Using RAGService

### Initialization

```python
from core.services.rag import RAGService

# Initialize with dependencies
rag_service = RAGService(
    config_manager=config_manager,
    embedding_manager=embedding_manager,
    knowledge_store=knowledge_store,
)

# Start the service
await rag_service.start()
```

### Creating a Knowledge Base

```python
kb = await rag_service.create_knowledge_base(
    name="Project Docs",
    description="Internal project documentation",
    embedding_model="text-embedding-3-small",
)
```

### Ingesting Documents

```python
# Ingest a file
success = await rag_service.ingest_file(
    kb_id=kb.id,
    file_path=Path("/path/to/document.md"),
    title="Architecture Overview",
    metadata={"author": "team", "tags": ["architecture"]},
)

# Ingest text content directly
doc = await rag_service.ingest_text(
    kb_id=kb.id,
    title="API Reference",
    content="# API Reference\n\n...",
    source_uri="https://docs.example.com/api",
)
```

### Searching

```python
from modules.storage.knowledge.base import SearchQuery

query = SearchQuery(
    query_text="How do I configure authentication?",
    knowledge_base_ids=[kb.id],
    top_k=5,
)

results = await rag_service.search(query)

for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"Document: {result.document.title}")
    print(f"Content: {result.chunk.content[:200]}...")
```

### Retrieving Context for LLM

```python
# Get formatted context for a prompt
context = await rag_service.get_context_for_query(
    query="What are the security best practices?",
    kb_ids=[kb.id],
    max_chunks=5,
    max_tokens=2000,
)

# Use in LLM prompt
prompt = f"""Answer based on this context:

{context}

Question: What are the security best practices?
"""
```

## KnowledgeStore API

The `KnowledgeStore` is the persistence layer. Use the abstract base class for type hints.

### Knowledge Base Operations

```python
# Create
kb = await store.create_knowledge_base(
    name="My KB",
    description="Description",
    embedding_model="all-MiniLM-L6-v2",
)

# Get
kb = await store.get_knowledge_base(kb_id)

# List
kbs = await store.list_knowledge_bases(owner_id=user_id)

# Update
kb = await store.update_knowledge_base(
    kb_id,
    name="Updated Name",
    description="New description",
)

# Delete
await store.delete_knowledge_base(kb_id, delete_documents=True)
```

### Document Operations

```python
# Add document (auto-chunks and embeds)
doc = await store.add_document(
    kb_id=kb.id,
    title="Document Title",
    content="Full document content...",
    source_uri="https://source.url",
    document_type=DocumentType.MARKDOWN,
    auto_chunk=True,
    auto_embed=True,
)

# List documents
docs = await store.list_documents(kb_id, limit=100)

# Get single document
doc = await store.get_document(doc_id)

# Update document
doc = await store.update_document(
    doc_id,
    content="Updated content...",
    rechunk=True,
    reembed=True,
)

# Check for duplicates
existing = await store.find_duplicate(kb_id, content)

# Delete
await store.delete_document(doc_id)
```

### Chunk Operations

```python
# Get chunks for a document
chunks = await store.get_chunks(doc_id, include_embeddings=True)

# Get single chunk
chunk = await store.get_chunk(chunk_id)

# Update chunk content
chunk = await store.update_chunk(chunk_id, content="Updated text")

# Search across knowledge bases
results = await store.search(search_query)
```

## Implementing Custom Embedding Providers

Create a new embedding provider by extending the base class:

```python
from core.services.rag.embedding import EmbeddingProvider

class CustomEmbeddingProvider(EmbeddingProvider):
    """Custom embedding provider implementation."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._dimension = 768  # Your model's dimension
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        # Your embedding API call
        response = await self._call_api(text)
        return response["embedding"]
    
    async def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self._call_batch_api(batch)
            embeddings.extend(batch_embeddings)
        return embeddings
```

Register your provider:

```python
from core.services.rag.embedding import register_embedding_provider

register_embedding_provider("custom-model", CustomEmbeddingProvider)
```

## Implementing Custom Knowledge Stores

For alternative storage backends:

```python
from modules.storage.knowledge.base import KnowledgeStore

class CustomKnowledgeStore(KnowledgeStore):
    """Custom knowledge store implementation."""
    
    async def create_knowledge_base(self, **kwargs) -> KnowledgeBase:
        # Implementation
        pass
    
    async def add_document(self, **kwargs) -> KnowledgeDocument:
        # Implementation
        pass
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        # Implementation
        pass
    
    # ... implement all abstract methods
```

Register your store:

```python
from modules.storage.knowledge.base import register_knowledge_store

register_knowledge_store("custom", CustomKnowledgeStore)
```

## Event Hooks

Subscribe to RAG events for monitoring:

```python
@rag_service.on("document_indexed")
async def on_document_indexed(event):
    logger.info(f"Document indexed: {event.document_id}")
    logger.info(f"Chunks created: {event.chunk_count}")

@rag_service.on("search_completed")
async def on_search(event):
    logger.info(f"Search query: {event.query}")
    logger.info(f"Results: {len(event.results)}")
```

## Configuration

RAG settings in `config.yaml`:

```yaml
rag:
  enabled: true
  default_store: postgres
  
  embedding:
    default_provider: openai
    default_model: text-embedding-3-small
    batch_size: 32
    cache_enabled: true
    
  chunking:
    default_size: 512
    default_overlap: 50
    max_size: 2000
    
  search:
    default_top_k: 5
    max_top_k: 20
    min_score_threshold: 0.0
    
  postgres:
    schema: atlas_rag
    vector_index_type: ivfflat  # or hnsw
```

## Testing

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_knowledge_store():
    store = AsyncMock(spec=KnowledgeStore)
    store.search.return_value = [
        SearchResult(
            chunk=KnowledgeChunk(id="1", content="test"),
            score=0.95,
        )
    ]
    return store

async def test_search(mock_knowledge_store):
    rag_service = RAGService(knowledge_store=mock_knowledge_store)
    results = await rag_service.search(SearchQuery(query_text="test"))
    assert len(results) == 1
    assert results[0].score == 0.95
```

### Integration Tests

```python
@pytest.mark.integration
async def test_full_ingestion_pipeline(rag_service, tmp_path):
    # Create KB
    kb = await rag_service.create_knowledge_base(name="Test KB")
    
    # Create test file
    test_file = tmp_path / "test.md"
    test_file.write_text("# Test\n\nThis is test content.")
    
    # Ingest
    success = await rag_service.ingest_file(
        kb_id=kb.id,
        file_path=test_file,
    )
    assert success
    
    # Search
    results = await rag_service.search(
        SearchQuery(query_text="test content", knowledge_base_ids=[kb.id])
    )
    assert len(results) > 0
    assert "test content" in results[0].chunk.content.lower()
```

## Error Handling

```python
from modules.storage.knowledge.base import (
    KnowledgeStoreError,
    KnowledgeBaseNotFoundError,
    DocumentNotFoundError,
    IngestionError,
)

try:
    doc = await rag_service.ingest_file(kb_id, file_path)
except KnowledgeBaseNotFoundError:
    logger.error(f"KB not found: {kb_id}")
except IngestionError as e:
    logger.error(f"Ingestion failed: {e}")
except KnowledgeStoreError as e:
    logger.error(f"Storage error: {e}")
```

## Performance Considerations

### Batch Operations

Always use batch operations when processing multiple items:

```python
# Good - batch embedding
embeddings = await provider.embed_batch(texts)

# Bad - individual calls
for text in texts:
    embedding = await provider.embed_text(text)  # Slow!
```

### Caching

Enable embedding cache for repeated queries:

```python
rag_service = RAGService(
    embedding_cache_enabled=True,
    embedding_cache_ttl=3600,  # 1 hour
)
```

### Index Optimization

For PostgreSQL with pgvector, consider index types:

- **IVFFlat**: Faster indexing, good for frequent updates
- **HNSW**: Faster queries, higher memory usage

## Related Documentation

- [User Guide](../user/rag-guide.md) - End-user documentation
- [Configuration Reference](../configuration.md) - All settings
- [Architecture Overview](../architecture-overview.md) - System design
