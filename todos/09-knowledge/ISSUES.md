# Knowledge & RAG Issues

> **Epic**: Knowledge Base Refactor
> **Parent**: [README.md](./README.md)

## 📋 Ready for Development

### Kno-001: Analyze Existing RAG Module

**Description**: Analyze `modules/storage` to separate "RAG/Vector" logic from generic "File" logic.
**Acceptance Criteria**:

- Report on dependencies in `modules/storage`.
- Plan for moving `modules/storage/vectors`, `embeddings`, `chunking` -> `core/services/rag/`.

### Kno-002: Scaffold Knowledge Service

**Description**: Create `core/services/knowledge`.
**Acceptance Criteria**:

- Service interface for `Collection` and `Document` management.

### Kno-003: Migrate GTKUI Logic

**Description**: Move business logic from `GTKUI/KnowledgeBase` to the new service.
**Acceptance Criteria**:

- UI controllers calling `KnowledgeService` instead of direct DB/Storage access.

### Kno-004: Implement Semantic Search

**Description**: Expose the underlying RAG capabilities via a clean Service API.
**Acceptance Criteria**:

- `search(query: str, filters: dict) -> list[Document]`.
