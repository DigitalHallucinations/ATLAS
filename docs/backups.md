---
audience: Operators and support engineers
status: in_review
last_verified: 2025-12-21
source_of_truth: AtlasServer backup endpoints (REST surface); GTK settings workspace
---

# Conversation and persona backups

ATLAS can now export or import a combined backup that includes conversation history and persona manifests.

## Exporting from the GTK shell
1. Open the **Settings** workspace in the GTK shell.
2. Use the **Browse…** control in the *Export* section to choose a target directory. This can be a removable drive mount point.
3. Click **Export conversations and personas**. When the export completes, a timestamped `atlas-backup-YYYYMMDD-HHMMSS.json` file is created in the selected directory.
4. Copy the generated file to the destination system if you are moving data between machines.

## Importing a backup
1. Open the **Settings** workspace in the GTK shell and navigate to the *Import* section.
2. Select the backup JSON file using **Browse…**.
3. Click **Import backup** to restore personas and conversation history into the active tenant.

The import and export buttons report status directly in the workspace so you can confirm completion.

## HTTP endpoints
For automation, the HTTP gateway exposes the same workflow:
- `POST /backups/export` returns a base64-encoded backup bundle.
- `POST /backups/import` accepts a `bundle` field containing the base64 payload.

Both endpoints respect the tenant and user context provided via request headers.
