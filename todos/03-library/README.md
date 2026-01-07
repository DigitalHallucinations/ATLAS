# Library Feature

> **Status**: 📋 Planning  
> **Priority**: Medium  
> **Complexity**: Medium  
> **Effort**: 1 week  
> **Created**: 2026-01-07

---

## Overview

The Library is a persistent collection of finalized artifacts, providing:

1. **Unified view** of all saved content across conversations, jobs, tasks
2. **Organization** via collections (folders) and tags
3. **Provenance** linking back to source (blackboard, chat, job, task)
4. **Versioning** for edit history
5. **Sharing** across tenants/users (future)

The Library is **separate from Blackboard** - Blackboard remains the real-time collaboration surface for skills, while Library is the archive of finalized outputs.

---

## Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Relationship to Blackboard | Separate systems | Clear separation: working area vs archive |
| Promotion behavior | Copy with link | Artifact stays in blackboard, copy in Library with provenance link |
| Auto-promotion | Media from jobs only | Jobs themselves don't auto-promote; generated media does |
| Editing | Allowed | Library items can be modified after promotion |
| Versioning | Yes | Keep edit history for all Library items |
| Deletion | Both options | Soft delete (archive) and hard delete available |
| Organization | Flat + hierarchical | Filters AND collections/folders |

---

## Phases

### Phase 1: Core Library Service

- [ ] **1.1** Create `core/services/library/` package
- [ ] **1.2** Create `modules/library_store/` repository package
- [ ] **1.3** Implement LibraryService CRUD
- [ ] **1.4** Add database migrations
- [ ] **1.5** Write unit tests

### Phase 2: Versioning

- [ ] **2.1** Implement version management
- [ ] **2.2** Version storage strategy for files
- [ ] **2.3** Write version tests

### Phase 3: Collections & Tags

- [ ] **3.1** Implement collection management
- [ ] **3.2** Implement collection membership
- [ ] **3.3** Implement tag management
- [ ] **3.4** Write collection/tag tests

### Phase 4: Promotion from Sources

- [ ] **4.1** Blackboard promotion
- [ ] **4.2** Chat promotion
- [ ] **4.3** Job output auto-promotion
- [ ] **4.4** Task deliverable promotion
- [ ] **4.5** Direct upload

### Phase 5: GTK UI - Library View

- [ ] **5.1** Create `GTKUI/Library/` package
- [ ] **5.2** Implement main Library view
- [ ] **5.3** Implement collection sidebar
- [ ] **5.4** Implement item detail view
- [ ] **5.5** Add to sidebar navigation

### Phase 6: Chat & Blackboard Integration

- [ ] **6.1** Add "Save to Library" action in chat
- [ ] **6.2** Add "Save to Library" in Blackboard
- [ ] **6.3** Add Library links in provenance sources

### Phase 7: Sharing (Future)

- [ ] **7.1** Implement share management
- [ ] **7.2** Permission model
- [ ] **7.3** UI for sharing

### Phase 8: Advanced Features

- [ ] **8.1** Starred/favorites
- [ ] **8.2** Recent items
- [ ] **8.3** Duplicate detection
- [ ] **8.4** Bulk operations
- [ ] **8.5** Export

---

## Data Model

### LibraryItem

```python
@dataclass
class LibraryItem:
    id: UUID
    tenant_id: str
    title: str
    description: str | None
    content_type: ContentType  # document, image, audio, video, code, data, other
    mime_type: str
    file_path: str | None
    content_text: str | None
    file_size_bytes: int | None
    source_type: SourceType    # blackboard, chat, job, task, upload, generated
    source_id: UUID | None
    source_metadata: dict
    collection_ids: list[UUID]
    tags: list[str]
    status: ItemStatus         # active, archived
    version: int
    visibility: Visibility     # private, shared, public
    shared_with: list[str]
    created_at: datetime
    updated_at: datetime
```

---

## Auto-Promotion Rules

| Source | Auto-Promote? | What Gets Promoted |
|--------|---------------|-------------------|
| Blackboard artifact | ❌ No (manual) | - |
| Chat attachment | ❌ No (manual) | - |
| Job completion | ❌ No | - |
| Job-generated media | ✅ Yes | Audio, images, documents created by job |
| Task attachment | ❌ No (manual) | - |
| TTS audio output | ✅ Yes | Generated speech files |
| Image generation | ✅ Yes | Generated images |

---

## MessageBus Events

| Event | Payload | Emitted When |
|-------|---------|--------------|
| `library.item.created` | `LibraryItemEvent` | Item created |
| `library.item.updated` | `LibraryItemEvent` | Item updated |
| `library.item.deleted` | `LibraryItemEvent` | Item hard deleted |
| `library.item.archived` | `LibraryItemEvent` | Item archived |
| `library.item.promoted` | `LibraryItemPromoted` | Item promoted from source |
| `library.collection.created` | `LibraryCollectionEvent` | Collection created |
| `library.collection.item_added` | `LibraryCollectionItemEvent` | Item added to collection |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/library/__init__.py` | Package exports |
| `core/services/library/types.py` | Dataclasses, enums, events |
| `core/services/library/permissions.py` | LibraryPermissionChecker |
| `core/services/library/service.py` | LibraryService |
| `modules/library_store/__init__.py` | Store package exports |
| `modules/library_store/models.py` | SQLAlchemy models |
| `modules/library_store/repository.py` | LibraryRepository |
| `GTKUI/Library/*.py` | UI components |
| `tests/services/library/` | Service tests |
| `tests/library_store/` | Repository tests |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `core/messaging/` - MessageBus for events
- `modules/storage/` - File storage backend
- `modules/conversation_store/` - Chat attachment access
- Blackboard API - For promotion

---

## Success Criteria

1. All artifacts accessible from one place
2. Collections and tags work intuitively
3. Always know where an item came from (provenance)
4. Edit history preserved
5. List/search <100ms for 10k items
6. Cross-tenant sharing works (future phase)

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Should Library have its own file storage or reuse existing? | Shared storage / Dedicated storage | TBD |
| Max file size for Library items? | 100MB / 1GB / Configurable | TBD |
| Should version history be prunable? | Yes with retention policy / Keep all | TBD |
| Export format for collections? | ZIP with manifest / Platform-specific | TBD |

---

## UI Mockup

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  📚 Library                                              [Grid] [List] [+]  │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────────────────────────────────────────┐ │
│ │ Collections     │ │ 🔍 Search...        [Type ▼] [Date ▼] [Source ▼]   │ │
│ │ ───────────────│ ├─────────────────────────────────────────────────────┤ │
│ │ 📁 All Items    │ │                                                     │ │
│ │ ⭐ Starred      │ │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │ │
│ │ 🕐 Recent       │ │  │ 📄       │ │ 🖼️       │ │ 🔊       │ │ 📊     │ │ │
│ │ 📥 Uncategorized│ │  │ Report   │ │ Diagram  │ │ Summary  │ │ Data   │ │ │
│ │ 🗄️ Archived     │ │  │ .pdf     │ │ .png     │ │ .mp3     │ │ .csv   │ │ │
│ │ ───────────────│ │  │ Job #42  │ │ Chat     │ │ TTS      │ │ Task   │ │ │
│ │ 📂 Projects     │ │  └──────────┘ └──────────┘ └──────────┘ └────────┘ │ │
│ │   📁 Website    │ │                                                     │ │
│ │   📁 API Docs   │ │                          Showing 7 of 156 items     │ │
│ └─────────────────┘ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```
