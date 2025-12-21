---
audience: Tool users and operators
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/Tools/Base_Tools/filesystem_io.py; modules/Tools/tool_maps/functions.json
---

# Filesystem Sandbox Tools

The filesystem tool suite provides read, write, and directory listing helpers for
personas that need controlled access to the ATLAS runtime files. All operations
are constrained to a sandbox root and enforce strict quotas so tools cannot
escape the workspace or overwhelm the host.

## Available functions

Three asynchronous handlers are exposed from
`modules/Tools/Base_Tools/filesystem_io.py` and registered in the shared tool
map under the following identifiers:

- `filesystem_read`: return file contents with MIME metadata and truncation
  details.
- `filesystem_write`: create or overwrite a file after validating per-call and
  total storage quotas.
- `filesystem_list`: enumerate directory entries with lightweight metadata.

Each handler returns JSON-serialisable dictionaries that mirror the runtime
responses exposed through the tool router. The manifest entries in
`modules/Tools/tool_maps/functions.json` declare their side effects, idempotency
guidance, and quota hints for downstream consumers.

## Sandbox configuration

By default, the sandbox root resolves to the repository workspace. The
behaviour can be adjusted with environment variables that are read for every
operation:

| Variable | Purpose | Default |
| --- | --- | --- |
| `ATLAS_FILESYSTEM_SANDBOX` | Absolute path to the sandbox root. Must exist and remain within the allowed workspace. | Repository root (`modules/Tools/Base_Tools/filesystem_io.py` parent three levels up). |
| `ATLAS_FILESYSTEM_MAX_READ_BYTES` | Maximum bytes returned by a single read operation. | `262144` (256 KiB) |
| `ATLAS_FILESYSTEM_MAX_WRITE_BYTES` | Maximum payload accepted per write call. | `262144` (256 KiB) |
| `ATLAS_FILESYSTEM_MAX_LIST_ENTRIES` | Maximum number of entries returned from a directory listing. | `512` |
| `ATLAS_FILESYSTEM_OPERATION_TIMEOUT` | Timeout budget (seconds) enforced for each filesystem operation. | `2.0` |
| `ATLAS_FILESYSTEM_MAX_TOTAL_BYTES` | Aggregate storage quota for the sandbox tree. | `10485760` (10 MiB) |

If a variable is set to an invalid value (e.g., non-numeric quota) the call
fails with a `QuotaExceededError`. Missing directories or path traversal
attempts raise `SandboxViolationError`.

## MIME detection and payload encoding

File responses include the MIME type inferred from :mod:`mimetypes`. Text files
are returned as UTF-8 strings (with replacement for invalid sequences). Binary
payloads are base64 encoded and tagged with `content_encoding="base64"` so
callers can decode safely.

## Error handling

The helpers raise structured exceptions exported from the module:

- `FilesystemError`: base class for runtime issues.
- `SandboxViolationError`: triggered by missing paths, path traversal, or
  invalid sandbox roots.
- `QuotaExceededError`: raised when per-operation or aggregate quotas would be
  exceeded.
- `FilesystemTimeoutError`: raised if an operation exceeds the configured
  timeout budget.

These exceptions propagate through the tool router, allowing personas to surface
clear error messages or implement retries in accordance with the manifest
idempotency guidance.
