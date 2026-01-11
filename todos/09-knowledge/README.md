# Knowledge Base Service

> **Status**: 📋 Planning  
> **Priority**: Medium  
> **Complexity**: Medium  
> **Effort**: 3-4 days  
> **Created**: 2026-01-07

---

## Overview

Extract knowledge base management from `GTKUI/KnowledgeBase/` into `core/services/knowledge/`:

- Document collection management
- Semantic search integration with RAGService
- Indexing operations

---

## Phases

### Phase 1: Service Creation

- [ ] **1.1** Create `core/services/knowledge/` package
- [ ] **1.2** Implement KnowledgeBaseService:
  - `list_collections()` - Get document collections
  - `create_collection(actor, name, config)` - New collection
  - `delete_collection(actor, collection_id)` - Remove collection
  - `add_document(actor, collection_id, document)` - Add doc
  - `remove_document(actor, collection_id, doc_id)` - Remove doc
  - `update_document(actor, collection_id, doc_id, content)` - Update
  - `search(query, collections, top_k)` - Semantic search
  - `get_indexing_status(collection_id)` - Index progress
  - `reindex_collection(actor, collection_id)` - Rebuild index
- [ ] **1.3** Integrate with RAGService for retrieval
- [ ] **1.4** Add MessageBus events:
  - `knowledge.collection_created`
  - `knowledge.collection_deleted`
  - `knowledge.document_added`
  - `knowledge.document_removed`
  - `knowledge.indexing_started`
  - `knowledge.indexing_completed`
- [ ] **1.5** Write unit tests

> **UI Integration**: See [40-ui-integration](../40-ui-integration/)

---

## Service Methods

```python
class KnowledgeBaseService:
    # Collections
    def list_collections(self, actor: Actor) -> OperationResult[list[Collection]]: ...
    def create_collection(self, actor: Actor, name: str, config: CollectionConfig) -> OperationResult[Collection]: ...
    def get_collection(self, actor: Actor, collection_id: UUID) -> OperationResult[Collection]: ...
    def update_collection(self, actor: Actor, collection_id: UUID, updates: CollectionUpdate) -> OperationResult[Collection]: ...
    def delete_collection(self, actor: Actor, collection_id: UUID) -> OperationResult[None]: ...
    
    # Documents
    def add_document(self, actor: Actor, collection_id: UUID, document: DocumentCreate) -> OperationResult[Document]: ...
    def get_document(self, actor: Actor, collection_id: UUID, doc_id: UUID) -> OperationResult[Document]: ...
    def update_document(self, actor: Actor, collection_id: UUID, doc_id: UUID, content: str) -> OperationResult[Document]: ...
    def remove_document(self, actor: Actor, collection_id: UUID, doc_id: UUID) -> OperationResult[None]: ...
    def list_documents(self, actor: Actor, collection_id: UUID) -> OperationResult[list[Document]]: ...
    
    # Search
    def search(self, actor: Actor, query: str, collections: list[UUID] | None, top_k: int = 10) -> OperationResult[list[SearchResult]]: ...
    
    # Indexing
    def get_indexing_status(self, collection_id: UUID) -> OperationResult[IndexingStatus]: ...
    def reindex_collection(self, actor: Actor, collection_id: UUID) -> OperationResult[None]: ...
    def reindex_all(self, actor: Actor) -> OperationResult[None]: ...
```

---

## MessageBus Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `knowledge.collection_created` | `CollectionEvent` | KnowledgeBaseService |
| `knowledge.collection_deleted` | `CollectionEvent` | KnowledgeBaseService |
| `knowledge.document_added` | `DocumentEvent` | KnowledgeBaseService |
| `knowledge.document_removed` | `DocumentEvent` | KnowledgeBaseService |
| `knowledge.indexing_started` | `IndexingEvent` | KnowledgeBaseService |
| `knowledge.indexing_completed` | `IndexingEvent` | KnowledgeBaseService |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/knowledge/__init__.py` | Package exports |
| `core/services/knowledge/types.py` | Dataclasses, events |
| `core/services/knowledge/permissions.py` | KnowledgePermissionChecker |
| `core/services/knowledge/service.py` | KnowledgeBaseService |
| `tests/services/knowledge/` | Service tests |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `core/services/rag.py` - RAGService for search
- `modules/storage/` - Document storage
- `core/messaging/` - MessageBus for events

---

## Success Criteria

1. Knowledge base operations centralized
2. Search integrated with RAGService
3. Indexing status trackable
4. UI updated
5. >90% test coverage

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Should indexing be synchronous or background? | Sync for small / Background always | TBD |
| Document chunking strategy? | Fixed size / Semantic / Configurable | TBD |
| Collection visibility settings? | Per-collection / Inherit tenant | TBD |
| Supported document formats? | PDF, MD, TXT / Extensible | TBD |
