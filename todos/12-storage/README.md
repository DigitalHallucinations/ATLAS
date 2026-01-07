# Storage Service

> **Status**: 📋 Planning  
> **Priority**: Low  
> **Complexity**: Low  
> **Effort**: 2-3 days  
> **Created**: 2026-01-07

---

## Overview

Consolidate `modules/storage/` into `core/services/storage/`:

- File storage operations
- Storage usage monitoring
- Orphan cleanup
- Integration with backup/restore

---

## Phases

### Phase 1: Service Creation

- [ ] **1.1** Create `core/services/storage/` package
- [ ] **1.2** Implement StorageService:
  - `store_file(actor, file, metadata)` - Upload file
  - `get_file(file_id)` - Retrieve file
  - `delete_file(actor, file_id)` - Remove file
  - `list_files(filters)` - List files
  - `get_file_url(file_id, expiry)` - Signed URL
  - `get_storage_usage()` - Disk usage stats
  - `cleanup_orphaned()` - Remove unreferenced files
- [ ] **1.3** Add MessageBus events:
  - `storage.file_uploaded`
  - `storage.file_deleted`
- [ ] **1.4** Integrate with backup/restore system
- [ ] **1.5** Write unit tests

---

## Service Methods

```python
class StorageService:
    # File operations
    def store_file(
        self,
        actor: Actor,
        file: bytes | Path,
        metadata: FileMetadata
    ) -> OperationResult[StoredFile]: ...
    
    def get_file(self, file_id: UUID) -> OperationResult[StoredFile]: ...
    def delete_file(self, actor: Actor, file_id: UUID) -> OperationResult[None]: ...
    def list_files(self, filters: FileFilters) -> OperationResult[list[StoredFile]]: ...
    
    # URLs
    def get_file_url(
        self,
        file_id: UUID,
        expiry: timedelta | None = None
    ) -> OperationResult[str]: ...
    
    # Usage
    def get_storage_usage(self) -> OperationResult[StorageUsage]: ...
    def get_usage_by_type(self) -> OperationResult[dict[str, int]]: ...
    
    # Maintenance
    def cleanup_orphaned(self, actor: Actor) -> OperationResult[CleanupResult]: ...
    def verify_integrity(self, actor: Actor) -> OperationResult[IntegrityReport]: ...
```

---

## MessageBus Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `storage.file_uploaded` | `FileEvent` | StorageService |
| `storage.file_deleted` | `FileEvent` | StorageService |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/storage/__init__.py` | Package exports |
| `core/services/storage/types.py` | Dataclasses, events |
| `core/services/storage/permissions.py` | StoragePermissionChecker |
| `core/services/storage/service.py` | StorageService |
| `tests/services/storage/` | Service tests |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `modules/storage/` - Storage backend (consolidate)
- `core/messaging/` - MessageBus for events

---

## Success Criteria

1. Storage operations centralized
2. Usage monitoring working
3. Orphan cleanup functional
4. Backup integration working
5. >90% test coverage

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Storage backend? | Local filesystem / S3-compatible / Both | TBD |
| File deduplication? | Hash-based / None | TBD |
| Max file size? | 100MB / 1GB / Configurable | TBD |
| Cleanup schedule? | Daily / Weekly / Manual | TBD |
