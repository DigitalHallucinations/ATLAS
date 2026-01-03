# ATLAS RAG System - Technical Deep Dive

## Overview

The ATLAS RAG (Retrieval-Augmented Generation) system is a comprehensive, production-grade implementation that enhances LLM responses by retrieving relevant context from knowledge bases. The architecture follows a modular, layered design with clear separation of concerns.

## Architecture Components

```text
┌─────────────────────────────────────────────────────────────┐
│                       RAGService                            │
│              (ATLAS/services/rag.py)                        │
│         High-level facade orchestrating all RAG ops         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │  RAGRetriever   │  │KnowledgeStore│  │DocumentIngester│ │
│  │  (retrieval/)   │  │ (knowledge/) │  │  (ingestion/)  │ │
│  └────────┬────────┘  └──────┬───────┘  └───────┬────────┘ │
└───────────┼──────────────────┼──────────────────┼──────────┘
            │                  │                  │
    ┌───────▼───────┐   ┌──────▼──────┐    ┌──────▼──────┐
    │EmbeddingProvider│  │ PostgreSQL  │    │TextSplitters│
    │ (embeddings/)  │  │  pgvector   │    │ (chunking/) │
    └───────────────┘   └─────────────┘    └─────────────┘
```

---

## 1. Knowledge Store Layer

**Location**: `modules/storage/knowledge/`

### PostgreSQL + pgvector Implementation

The knowledge store uses **PostgreSQL with the pgvector extension** for vector similarity search.

**Key Tables**:

| Table                         | Purpose                                              |
| ----------------------------- | ---------------------------------------------------- |
| `knowledge_bases`             | Top-level KB containers with embedding configuration |
| `knowledge_documents`         | Document metadata, content hash, status tracking     |
| `knowledge_chunks`            | Text chunks with `vector(3072)` embeddings           |
| `knowledge_document_versions` | Version history for document updates                 |

**Vector Indexing**:

- **HNSW** (Hierarchical Navigable Small World): Default index type
  - Parameters: `m=16`, `ef_construction=64`
  - Provides approximate nearest neighbor search with excellent query performance
- **IVFFlat**: Alternative index with `lists=100`

```sql
-- HNSW index creation (from postgres.py)
CREATE INDEX idx_chunk_embedding
ON knowledge_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
```

**Search Query Structure**:

```python
@dataclass
class SearchQuery:
    query_text: str
    knowledge_base_ids: Optional[List[str]]
    top_k: int = 10
    min_score: float = 0.0
    include_content: bool = True
    include_document: bool = True
```

---

## 2. Embedding Providers

**Location**: `modules/storage/embeddings/`

### Supported Providers

| Provider                | Models                                         | Dimensions | Use Case                    |
| ----------------------- | ---------------------------------------------- | ---------- | --------------------------- |
| **HuggingFace** (local) | all-MiniLM-L6-v2, BGE, E5, GTE families        | 384-1024   | Privacy, no API costs       |
| **OpenAI**              | text-embedding-3-small, text-embedding-3-large | 256-3072   | High quality, production    |
| **Cohere**              | embed-english-v3.0                             | 1024       | Strong multilingual support |

### Asymmetric Embedding Support

The system supports **asymmetric embedding models** (BGE, E5) that use different prefixes for queries vs. documents:

```python
# From huggingface.py
LOCAL_MODELS = {
    "bge-small-en-v1.5": {
        "dimension": 384,
        "symmetric": False,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "e5-base-v2": {
        "dimension": 768,
        "symmetric": False,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
}
```

### Input Type Handling

```python
class EmbeddingInputType(str, Enum):
    DOCUMENT = "search_document"   # For indexing
    QUERY = "search_query"         # For retrieval
    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"
```

---

## 3. Text Chunking Strategies

**Location**: `modules/storage/chunking/`

### Three Chunking Strategies

#### 1. Recursive Text Splitter (Default)

```python
DEFAULT_SEPARATORS = [
    "\n\n",   # Paragraph breaks
    "\n",     # Line breaks
    ". ",     # Sentence endings
    "! ", "? ",
    "; ", ", ",
    " ",      # Words
    "",       # Characters (last resort)
]
```

- Tries separators in order of priority
- Preserves semantic boundaries (paragraphs → sentences → words)
- Configurable `chunk_size` (default 512) and `chunk_overlap` (default 50)

#### 2. Sentence Text Splitter

- Splits on sentence boundaries using regex patterns
- Enforces minimum sentence length
- Good for structured text

#### 3. Semantic Text Splitter

- **Uses embeddings** to group semantically related sentences
- Computes cosine similarity between sentence embeddings
- Creates topic-coherent chunks

```python
# From semantic.py
async def split_text_async(self, text: str) -> List[str]:
    sentences = self._get_sentences(text)
    embeddings = await self._get_sentence_embeddings(sentences)
    break_points = self._find_semantic_breaks(sentences, embeddings)
    return self._create_chunks_from_breaks(sentences, break_points)
```

**Similarity threshold**: 0.75 default (configurable)

---

## 4. Document Ingestion Pipeline

**Location**: `modules/storage/ingestion/ingester.py`

### Supported File Types

| Category     | Extensions                                 |
| ------------ | ------------------------------------------ |
| **Text**     | .txt                                       |
| **Markdown** | .md, .markdown                             |
| **HTML**     | .html, .htm                                |
| **PDF**      | .pdf                                       |
| **Code**     | .py, .js, .ts, .java, .cpp, .go, .rs, etc. |
| **Data**     | .json, .yaml, .yml, .csv, .xml             |

### Ingestion Workflow

```text
File Upload → Type Detection → Parsing → Chunking → Embedding → Storage
```

```python
@dataclass
class IngestionResult:
    document_id: str
    knowledge_base_id: str
    title: str
    chunk_count: int
    token_count: int
    file_type: FileType
    duration_seconds: float
    success: bool
```

### Deduplication

Uses SHA-256 content hashing:

```python
def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
```

---

## 5. Retrieval Pipeline

**Location**: `modules/storage/retrieval/retriever.py`

### Retrieval Flow

```text
Query → Embed Query → Vector Search → (Optional) Rerank → Filter → Format
```

### Two-Stage Retrieval with Reranking

```python
async def retrieve(self, query: str, ...) -> RetrievalResult:
    # Stage 1: Vector similarity search (over-fetch for reranking)
    search_query = SearchQuery(
        query_text=query,
        top_k=top_k * 2 if rerank else top_k,  # Over-fetch
        min_score=0.0,  # Filter after rerank
    )
    results = await self._knowledge_store.search(search_query)
    
    # Stage 2: Reranking (optional)
    if rerank and self._reranker:
        results = await self._reranker.rerank(query, results, top_n=self._top_n_rerank)
    
    # Post-filtering
    results = [r for r in results if r.score >= min_score][:top_k]
```

### Reranker Options

#### Cross-Encoder Reranker (Local)

- Uses `sentence-transformers` CrossEncoder
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Provides accurate relevance scoring by jointly encoding query+document

```python
class CrossEncoderReranker(Reranker):
    async def rerank(self, query: str, results: List[SearchResult], top_n: int):
        pairs = [(query, r.chunk.content) for r in results]
        scores = self._model.predict(pairs)  # Joint scoring
        # Re-sort by new scores
```

#### Cohere Reranker (API)

- Model: `rerank-english-v3.0`
- High-quality reranking via API call

```python
class CohereReranker(Reranker):
    async def rerank(self, query: str, results: List[SearchResult], top_n: int):
        response = self._client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[r.chunk.content for r in results],
            top_n=top_n,
        )
```

---

## 6. Context Assembly

The retriever formats results for LLM prompt injection:

### Output Formats

| Format     | Use Case                         |
| ---------- | -------------------------------- |
| `PLAIN`    | Simple numbered list             |
| `MARKDOWN` | Rich formatting for display      |
| `XML`      | Structured for XML-based prompts |
| `JSON`     | Programmatic access              |

### Context Assembly Example

```python
def assemble_context(self, results: RetrievalResult, ...) -> AssembledContext:
    chunks = [ContextChunk(
        content=r.chunk.content,
        source=r.document.title,
        score=r.score,
    ) for r in results.chunks]
    
    # Token estimation (~4 chars/token)
    token_count = len(text) // 4
    
    # Truncation if needed
    if max_tokens and token_count > max_tokens:
        text = text[:max_tokens * 4] + "\n[Context truncated...]"
```

---

## 7. LLM Context Integration

**Location**: `ATLAS/context/llm_context_manager.py`

The `LLMContextManager` automatically retrieves RAG context for queries:

```python
async def _get_rag_context(self, messages: List[Dict]) -> Optional[str]:
    # Check RAG enabled
    if not self._config_manager.is_rag_enabled():
        return None
    
    # Extract query from last user message
    query = self._extract_query_from_messages(messages)
    
    # Retrieve using RAGRetriever
    result = await self._rag_retriever.retrieve(query=query)
    
    # Assemble context
    assembled = self._rag_retriever.assemble_context(
        result,
        max_tokens=rag_settings.max_context_tokens,
        include_sources=True,
    )
    return assembled.text
```

---

## 8. Configuration System

**Location**: `ATLAS/config/rag.py`

### Master Settings Structure

```python
@dataclass
class RAGSettings:
    enabled: bool = False                    # Master switch
    auto_retrieve: bool = True               # Auto-retrieve on queries
    max_context_tokens: int = 4000           # Max tokens for context
    
    embeddings: EmbeddingSettings            # Provider config
    chunking: ChunkingSettings               # Splitter config
    retrieval: RetrievalSettings             # Search config
    reranking: RerankingSettings             # Reranker config
    knowledge_store: KnowledgeStoreSettings  # Storage config
    ingestion: IngestionSettings             # File handling config
```

### Granular Enable/Disable

Each subsystem has its own `enabled` flag:

- `embeddings.enabled`
- `chunking.enabled`
- `retrieval.enabled`
- `reranking.enabled`
- `knowledge_store.enabled`
- `ingestion.enabled`

---

## 9. Current System Characteristics

### Strengths

✅ **Multi-provider embedding support** (OpenAI, Cohere, HuggingFace)  
✅ **Two-stage retrieval** with optional reranking  
✅ **Multiple chunking strategies** including semantic chunking  
✅ **pgvector with HNSW indexing** for fast ANN search  
✅ **Document versioning and deduplication**  
✅ **Asymmetric embedding support** for optimized retrieval  
✅ **Flexible context formatting** (Plain, Markdown, XML, JSON)  
✅ **Comprehensive configuration system**

### Areas for SOTA Improvements

Based on current capabilities, here are potential enhancements for state-of-the-art performance:

| Area           | Current              | SOTA Enhancement                               |
| -------------- | -------------------- | ---------------------------------------------- |
| **Chunking**   | Recursive/Semantic   | Late chunking, parent-child hierarchies        |
| **Search**     | Dense vector only    | Hybrid search (BM25 + dense)                   |
| **Embeddings** | Single-vector        | ColBERT-style multi-vector                     |
| **Reranking**  | Cross-encoder        | FlashRank, ListWise rerankers                  |
| **Context**    | Simple concatenation | Lost-in-middle mitigation, context compression |
| **Query**      | Direct embedding     | Query expansion, HyDE                          |
| **Caching**    | None apparent        | Semantic caching for repeated queries          |
| **Evaluation** | None apparent        | RAG evaluation metrics (RAGAS)                 |

---

## File Reference Summary

| Component          | Primary File(s)                                                                  |
| ------------------ | -------------------------------------------------------------------------------- |
| RAG Service Facade | `ATLAS/services/rag.py`                                                          |
| Knowledge Store    | `modules/storage/knowledge/postgres.py`                                          |
| Embeddings         | `modules/storage/embeddings/base.py`, `huggingface.py`, `openai.py`, `cohere.py` |
| Chunking           | `modules/storage/chunking/recursive.py`, `semantic.py`, `sentence.py`            |
| Ingestion          | `modules/storage/ingestion/ingester.py`                                          |
| Retrieval          | `modules/storage/retrieval/retriever.py`                                         |
| Configuration      | `ATLAS/config/rag.py`                                                            |
| LLM Integration    | `ATLAS/context/llm_context_manager.py`                                           |

---

## 10. SOTA Upgrade Plan

The following upgrade plan fits the current architecture and keeps PostgreSQL as the backbone.

### 10.1 Hybrid Retrieval in Postgres

**Problem**: Current system is "dense-only," missing obvious keyword matches.

**Solution**: Add BM25-style lexical retrieval via Postgres full-text search and fuse with dense results.

#### Schema Changes

Add to `knowledge_chunks`:

```sql
-- Add tsvector column for full-text search
ALTER TABLE knowledge_chunks
  ADD COLUMN tsv tsvector;

-- Populate from existing content
UPDATE knowledge_chunks
SET tsv = to_tsvector('english', coalesce(content,''));

-- Create GIN index for fast text search
CREATE INDEX idx_chunks_tsv
ON knowledge_chunks
USING GIN (tsv);
```

Optional: Add `content_len int` for BM25 length normalization.

#### Ingestion Update

Set/refresh `tsv` on insert/update:

```python
# In DocumentIngester or KnowledgeStore
INSERT INTO knowledge_chunks (content, embedding, tsv, ...)
VALUES ($1, $2, to_tsvector('english', $1), ...);
```

#### Retrieval Change

In `KnowledgeStore.search()`, run two parallel queries:

1. **Dense**: Current HNSW cosine similarity
2. **Lexical**: `ts_rank_cd(tsv, plainto_tsquery(...))`

Then fuse using **Reciprocal Rank Fusion (RRF)**:

```python
from collections import defaultdict

def rrf_fuse(
    dense_ids_ranked: List[str],
    lexical_ids_ranked: List[str],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Fuse ranked lists using Reciprocal Rank Fusion.
    
    Args:
        dense_ids_ranked: Chunk IDs from dense retrieval, best first.
        lexical_ids_ranked: Chunk IDs from lexical retrieval, best first.
        k: RRF constant (default 60, per original paper).
        
    Returns:
        List of (chunk_id, fused_score) tuples, sorted by score descending.
    """
    scores = defaultdict(float)
    for rank, cid in enumerate(dense_ids_ranked, 1):
        scores[cid] += 1.0 / (k + rank)
    for rank, cid in enumerate(lexical_ids_ranked, 1):
        scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Why this matters**: Hybrid sharply reduces "missed obvious keyword matches" and improves robustness on code/log/error messages, names, IDs, acronyms—while staying very fast.

---

### 10.2 Query Router (Selective Reranking + Escalation)

**Problem**: Reranking everything is expensive and unnecessary.

**Solution**: Make reranking conditional based on query characteristics.

#### Pre-Retrieval Heuristics

| Query Pattern | Action |
| ------------- | ------ |
| Contains symbols/paths/stack traces/IDs | Bias lexical weight higher |
| Short + ambiguous ("what is this") | Increase `top_k`, consider clarification |
| Long/narrative query | Dense-first + rerank top 20 |

#### Post-Retrieval Heuristics

| Condition | Action |
| --------- | ------ |
| Top scores weak OR results disagree | Trigger evidence gating (see 10.3) |
| High confidence, results agree | Skip rerank (speed win) |

#### Implementation: RetrievalPolicy

```python
@dataclass
class RetrievalPolicy:
    """Dynamic retrieval policy based on query analysis."""
    
    use_hybrid: bool = True
    dense_weight: float = 0.5      # For weighted fusion alternative
    lexical_weight: float = 0.5
    
    dense_top_k: int = 20
    lexical_top_k: int = 20
    
    rerank: bool = True
    rerank_top_n: int = 10
    
    min_evidence_confidence: float = 0.4
    
    @classmethod
    def from_query(cls, query: str) -> "RetrievalPolicy":
        """Analyze query and return appropriate policy."""
        policy = cls()
        
        # Detect code/technical queries
        code_indicators = ['/', '\\', '::', '->', '()', '{}', 'error', 'exception']
        if any(ind in query.lower() for ind in code_indicators):
            policy.lexical_weight = 0.7
            policy.dense_weight = 0.3
        
        # Short ambiguous queries
        if len(query.split()) < 4:
            policy.dense_top_k = 30
            policy.lexical_top_k = 30
        
        # Long narrative queries benefit from reranking
        if len(query.split()) > 15:
            policy.rerank = True
            policy.rerank_top_n = 20
        
        return policy
```

---

### 10.3 Evidence Gating (Reliability Multiplier)

**Problem**: System gives "confident nonsense" when evidence is weak.

**Solution**: Compute evidence confidence score and gate responses accordingly.

#### Confidence Computation

```python
def compute_evidence_confidence(
    results: List[SearchResult],
    rerank_top_score: Optional[float] = None,
) -> float:
    """Compute confidence score from retrieval results.
    
    Returns:
        Confidence score between 0 and 1.
    """
    if not results:
        return 0.0
    
    # Component scores
    top1_score = min(max(results[0].score, 0), 1)
    
    margin = 0.0
    if len(results) >= 2:
        margin = min(max(results[0].score - results[1].score, 0), 1)
    
    # Count unique documents
    unique_docs = len(set(r.chunk.document_id for r in results[:5]))
    doc_diversity = min(unique_docs / 3, 1)
    
    # Reranker contribution
    rerank_component = min(max(rerank_top_score or 0, 0), 1)
    
    # Weighted blend
    confidence = (
        0.45 * top1_score +
        0.25 * margin +
        0.20 * doc_diversity +
        0.10 * rerank_component
    )
    
    return confidence
```

#### Gating Behavior

```python
@dataclass
class EvidenceGate:
    """Result of evidence gating check."""
    
    confidence: float
    action: Literal["proceed", "clarify", "abstain", "show_options"]
    message: Optional[str] = None

def gate_evidence(
    results: List[SearchResult],
    threshold: float = 0.4,
    contradiction_check: bool = True,
) -> EvidenceGate:
    """Check if evidence is sufficient to proceed."""
    
    conf = compute_evidence_confidence(results)
    
    if conf < threshold:
        return EvidenceGate(
            confidence=conf,
            action="abstain",
            message="I couldn't find strong supporting context in the knowledge base.",
        )
    
    # Optional: Check for contradictions
    if contradiction_check and _results_contradict(results[:3]):
        return EvidenceGate(
            confidence=conf,
            action="show_options",
            message="I found conflicting information. Here's what each source says:",
        )
    
    return EvidenceGate(confidence=conf, action="proceed")
```

This bolts into `RAGRetriever.retrieve()` right after reranking and filtering.

---

### 10.4 Parent–Child Chunking

**Problem**: Small chunks lose context; large chunks hurt retrieval precision.

**Solution**: Index small child chunks for retrieval, attach parent context at answer time.

#### Parent-Child Schema Changes

Add to `knowledge_chunks`:

```sql
ALTER TABLE knowledge_chunks
  ADD COLUMN parent_chunk_id UUID NULL
    REFERENCES knowledge_chunks(id) ON DELETE SET NULL,
  ADD COLUMN section_path TEXT;  -- "Doc > Heading > Subheading"
  
CREATE INDEX idx_chunk_parent ON knowledge_chunks(parent_chunk_id);
```

#### Chunk Hierarchy

| Level | Token Size | Purpose |
| ----- | ---------- | ------- |
| **Child** (indexed) | 256–512 | Precision retrieval |
| **Parent** (attached) | 1000–3000 | Context for LLM |

#### Ingestion Changes

```python
def create_hierarchical_chunks(
    document: str,
    metadata: Dict,
) -> List[HierarchicalChunk]:
    """Build parent-child chunk hierarchy from document structure."""
    
    # Parse document structure (Markdown headings, code blocks, etc.)
    sections = parse_sections(document)
    
    chunks = []
    for section in sections:
        # Create parent chunk (larger context window)
        parent = HierarchicalChunk(
            content=section.full_text[:3000],
            section_path=section.path,
            is_parent=True,
        )
        chunks.append(parent)
        
        # Create child chunks (indexed for retrieval)
        for child_text in split_into_children(section.full_text, size=400):
            child = HierarchicalChunk(
                content=child_text,
                parent_chunk_id=parent.id,
                section_path=section.path,
                is_parent=False,
            )
            chunks.append(child)
    
    return chunks
```

#### Retrieval Changes

```python
async def retrieve_with_parents(
    self,
    query: str,
    top_k: int = 5,
) -> RetrievalResultWithParents:
    """Retrieve children, then attach unique parents."""
    
    # Retrieve small children (fast, precise)
    children = await self.retrieve(query, top_k=top_k * 2)
    
    # Collect unique parent IDs
    parent_ids = set()
    for child in children.chunks:
        if child.parent_chunk_id:
            parent_ids.add(child.parent_chunk_id)
    
    # Fetch parents
    parents = await self._fetch_chunks_by_ids(list(parent_ids))
    
    return RetrievalResultWithParents(
        children=children.chunks[:top_k],
        parents=parents,
    )
```

**Benefits**: Model sees "surrounding truth" for faithfulness; retrieval stays fast on small vectors.

---

### 10.5 Context Compression

**Problem**: Raw chunk concatenation wastes tokens and causes "lost in middle."

**Solution**: Intelligent context assembly with deduplication and compression.

#### Chunk Deduplication

```python
def deduplicate_chunks(
    chunks: List[ContextChunk],
    similarity_threshold: float = 0.9,
) -> List[ContextChunk]:
    """Remove near-duplicate chunks based on content similarity."""
    
    unique = []
    seen_hashes = set()
    
    for chunk in chunks:
        # Quick hash check
        content_hash = hashlib.md5(chunk.content.encode()).hexdigest()[:16]
        if content_hash in seen_hashes:
            continue
        
        # Detailed similarity check against accepted chunks
        is_duplicate = False
        for accepted in unique:
            if _text_similarity(chunk.content, accepted.content) > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(chunk)
            seen_hashes.add(content_hash)
    
    return unique
```

#### Compression Strategies

```python
class ContextCompressor:
    """Compress context for more efficient LLM consumption."""
    
    def compress(
        self,
        chunks: List[ContextChunk],
        max_tokens: int,
        strategy: Literal["extractive", "abstractive", "hybrid"] = "extractive",
    ) -> CompressedContext:
        
        if strategy == "extractive":
            return self._extractive_compress(chunks, max_tokens)
        elif strategy == "abstractive":
            return self._abstractive_compress(chunks, max_tokens)
        else:
            return self._hybrid_compress(chunks, max_tokens)
    
    def _extractive_compress(
        self,
        chunks: List[ContextChunk],
        max_tokens: int,
    ) -> CompressedContext:
        """Extract key sentences, code blocks, and headings."""
        
        compressed_parts = []
        for chunk in chunks:
            # Extract high-signal elements
            code_blocks = extract_code_blocks(chunk.content)
            headings = extract_headings(chunk.content)
            key_sentences = extract_key_sentences(chunk.content, top_n=3)
            
            compressed_parts.append(CompressedChunk(
                source=chunk.source,
                code_blocks=code_blocks,
                headings=headings,
                key_points=key_sentences,
                original_chunk_id=chunk.metadata.get("chunk_id"),
            ))
        
        return CompressedContext(
            parts=compressed_parts,
            compression_ratio=self._calculate_ratio(chunks, compressed_parts),
        )
```

#### Lost-in-Middle Mitigation

```python
def reorder_for_attention(
    chunks: List[ContextChunk],
) -> List[ContextChunk]:
    """Reorder chunks to place most relevant at start and end.
    
    LLMs attend better to beginning and end of context.
    Places best chunks at positions 1, N, 2, N-1, 3, N-2, ...
    """
    if len(chunks) <= 2:
        return chunks
    
    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
    reordered = []
    
    left = 0
    right = len(sorted_chunks) - 1
    use_left = True
    
    while left <= right:
        if use_left:
            reordered.append(sorted_chunks[left])
            left += 1
        else:
            reordered.append(sorted_chunks[right])
            right -= 1
        use_left = not use_left
    
    return reordered
```

---

### 10.6 Multi-Level Caching

**Problem**: Repeated queries incur full retrieval cost.

**Solution**: Cache at multiple levels of the pipeline.

#### Cache Layers

| Cache | Key | Value | TTL |
| ----- | --- | ----- | --- |
| **Embedding** | `query_text` | `embedding vector` | 1 hour |
| **Retrieval** | `norm_query + kb_ids + settings_hash` | `chunk_ids + scores` | 15 min |
| **Rerank** | `query + chunk_ids` | `reranked_order` | 15 min |
| **Context** | `reranked_ids + format + max_tokens` | `assembled_text` | 15 min |

#### Implementation

```python
from functools import lru_cache
from hashlib import sha256
import time

class RAGCache:
    """Multi-level cache for RAG pipeline."""
    
    def __init__(
        self,
        embedding_ttl: int = 3600,
        retrieval_ttl: int = 900,
        max_size: int = 1000,
    ):
        self._embedding_cache: Dict[str, Tuple[List[float], float]] = {}
        self._retrieval_cache: Dict[str, Tuple[List[str], float]] = {}
        self._rerank_cache: Dict[str, Tuple[List[str], float]] = {}
        self._embedding_ttl = embedding_ttl
        self._retrieval_ttl = retrieval_ttl
        self._max_size = max_size
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for cache key."""
        import re
        normalized = query.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized
    
    def _make_retrieval_key(
        self,
        query: str,
        kb_ids: Optional[List[str]],
        settings: RetrievalSettings,
    ) -> str:
        """Create cache key for retrieval results."""
        parts = [
            self._normalize_query(query),
            ','.join(sorted(kb_ids or [])),
            str(settings.top_k),
            str(settings.similarity_threshold),
        ]
        return sha256('|'.join(parts).encode()).hexdigest()
    
    def get_embedding(self, query: str) -> Optional[List[float]]:
        """Get cached embedding if valid."""
        key = self._normalize_query(query)
        if key in self._embedding_cache:
            embedding, timestamp = self._embedding_cache[key]
            if time.time() - timestamp < self._embedding_ttl:
                return embedding
            del self._embedding_cache[key]
        return None
    
    def set_embedding(self, query: str, embedding: List[float]) -> None:
        """Cache an embedding."""
        self._evict_if_needed(self._embedding_cache)
        key = self._normalize_query(query)
        self._embedding_cache[key] = (embedding, time.time())
    
    def _evict_if_needed(self, cache: Dict) -> None:
        """LRU eviction when cache is full."""
        if len(cache) >= self._max_size:
            # Remove oldest 10%
            items = sorted(cache.items(), key=lambda x: x[1][1])
            for key, _ in items[:len(items) // 10]:
                del cache[key]
```

**Performance Impact**: Often the difference between 800ms and 80ms on repeated workflows.

---

### 10.7 Evaluation Harness

**Problem**: No way to detect retrieval quality regressions.

**Solution**: Add a lightweight RAG evaluation CLI and test suite.

#### Golden Test Set Structure

```yaml
# tests/rag_golden/test_cases.yaml
test_cases:
  - id: "auth-flow-001"
    query: "How does user authentication work?"
    expected_sources:
      - "docs/user-accounts.md"
      - "modules/user_accounts/"
    expected_chunks_contain:
      - "password"
      - "session"
    should_abstain: false
    
  - id: "nonexistent-002"
    query: "How do I configure the flux capacitor?"
    expected_sources: []
    should_abstain: true
```

#### Metrics

```python
@dataclass
class RAGEvalMetrics:
    """Evaluation metrics for RAG system."""
    
    # Retrieval quality
    hit_at_1: float      # Did we retrieve the right doc at position 1?
    hit_at_5: float      # Did we retrieve the right doc in top 5?
    mrr: float           # Mean Reciprocal Rank
    
    # Abstention quality
    abstain_precision: float  # When we abstained, were we right to?
    abstain_recall: float     # When we should abstain, did we?
    
    # Citation coverage (requires LLM eval)
    citation_coverage: float  # % of answer sentences supported by chunks
    
    # Performance
    latency_p50_ms: float
    latency_p95_ms: float
    
    # Counts
    total_queries: int
    successful_retrievals: int
    abstentions: int

def compute_hit_at_k(
    retrieved_doc_ids: List[str],
    expected_doc_ids: List[str],
    k: int,
) -> bool:
    """Check if any expected doc appears in top-k results."""
    return bool(set(retrieved_doc_ids[:k]) & set(expected_doc_ids))

def compute_mrr(
    retrieved_doc_ids: List[str],
    expected_doc_ids: List[str],
) -> float:
    """Compute Mean Reciprocal Rank."""
    for rank, doc_id in enumerate(retrieved_doc_ids, 1):
        if doc_id in expected_doc_ids:
            return 1.0 / rank
    return 0.0
```

#### CLI Command

```python
# scripts/rag_eval.py

@click.command()
@click.option('--test-file', default='tests/rag_golden/test_cases.yaml')
@click.option('--output', default='rag_eval_report.json')
@click.option('--verbose', is_flag=True)
def run_rag_eval(test_file: str, output: str, verbose: bool):
    """Run RAG evaluation suite."""
    
    test_cases = load_test_cases(test_file)
    results = []
    
    for case in test_cases:
        start = time.time()
        retrieval_result = await rag_service.retrieve(case.query)
        latency = (time.time() - start) * 1000
        
        result = evaluate_case(case, retrieval_result, latency)
        results.append(result)
        
        if verbose:
            print(f"{case.id}: hit@5={result.hit_at_5}, latency={latency:.0f}ms")
    
    metrics = aggregate_metrics(results)
    save_report(metrics, output)
    
    print(f"\n=== RAG Evaluation Summary ===")
    print(f"Hit@1: {metrics.hit_at_1:.1%}")
    print(f"Hit@5: {metrics.hit_at_5:.1%}")
    print(f"MRR: {metrics.mrr:.3f}")
    print(f"Abstain Precision: {metrics.abstain_precision:.1%}")
    print(f"P50 Latency: {metrics.latency_p50_ms:.0f}ms")
    print(f"P95 Latency: {metrics.latency_p95_ms:.0f}ms")
```

**Goal**: Not boil the ocean—just enough to stop accidental regressions when tweaking chunking/index/rerank thresholds.

---

## Implementation Priority

| Priority | Feature | Effort | Impact |
| -------- | ------- | ------ | ------ |
| **P0** | Hybrid Retrieval (10.1) | Medium | High - fixes keyword misses |
| **P0** | Evidence Gating (10.3) | Low | High - prevents confident nonsense |
| **P1** | Multi-Level Caching (10.6) | Low | Medium - major latency win |
| **P1** | Query Router (10.2) | Medium | Medium - smart resource use |
| **P2** | Parent-Child Chunking (10.4) | Medium | Medium - better context |
| **P2** | Context Compression (10.5) | Medium | Medium - token efficiency |
| **P3** | Evaluation Harness (10.7) | Low | High - prevents regressions |
