# Storage Service Issues

> **Epic**: Unified File Storage
> **Parent**: [README.md](./README.md)

## 📋 Ready for Development

### STO-001: Define Storage Interface

**Description**: Create a simplified abstract interface for file storage (blob).
**Acceptance Criteria**:

- `StorageProvider` protocol (`put`, `get`, `delete`, `list`).
- `LocalFileSystemProvider` implementation.

### STO-002: Scaffold Storage Service

**Description**: Create `core/services/storage` wrapper.
**Acceptance Criteria**:

- `StorageService` managing the providers.
- Separation of metadata (DB) from content (Filesystem/S3).

### STO-003: Migrate Orphan Cleanup

**Description**: Extract cleanup logic from `modules/storage` if it exists.
**Acceptance Criteria**:

- Periodic task to find files with no DB reference.
