---
audience: Operators and support engineers
status: in_review
last_verified: 2025-12-21
source_of_truth: server/http_gateway.py; modules/Server/routes.py; ATLAS/ATLAS.py; GTKUI/Settings/backup_settings.py
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

## Programmatic and headless flows
- **AtlasServer routes** – The server exposes `export_backup_bundle` and
  `import_backup_bundle` for automation that already has a `RequestContext`.
  Callers must supply at least a `tenant_id` or the request is rejected as
  unauthorized.【F:modules/Server/routes.py†L3143-L3196】【F:modules/Server/routes.py†L3198-L3226】
- **Headless CLI invocation** – The GTK settings workspace delegates to
  `ATLAS.export_user_backup`/`import_user_backup`, which wrap the server routes
  and write/read JSON files directly.【F:ATLAS/ATLAS.py†L1849-L1890】【F:GTKUI/Settings/backup_settings.py†L114-L170】
  For example, to export from a headless session:

  ```bash
  python - <<'PY'
  from pathlib import Path
  from ATLAS.ATLAS import ATLAS

  atlas = ATLAS()
  result = atlas.export_user_backup(Path("/tmp/atlas-backups"))
  print(result)
  PY
  ```

  The helper writes a timestamped JSON to the target directory and surfaces any
  server-side errors so CI or automation can fail fast.【F:ATLAS/ATLAS.py†L1849-L1890】
