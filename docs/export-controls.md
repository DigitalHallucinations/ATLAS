---
audience: Security and compliance reviewers
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/Server/dlp.py; modules/Server/routes.py
---

# Export controls and data residency

ATLAS supports deployments that must respect regional handling and export
control obligations. Enterprise tenants can stage residency expectations in the
setup wizard so that operators and downstream automation inherit a consistent
baseline.

## Regional handling

* **Primary region** – Sets the geographic home for persisted data. Align this
  value with where your PostgreSQL and cache services reside.
* **Residency requirement** – Signals whether records must stay in-region, may
  burst temporarily to other regions, or can be processed globally. Choose the
  strictest option your legal and compliance teams expect.

These selections are written to `data_residency` in `config.yaml` and are
available to scheduling, storage, and audit modules that need to understand the
preferred handling rules.【F:ATLAS/config/atlas_config.yaml†L100-L126】

## Export guidance

* **DLP and redaction** – The HTTP gateway redacts common secrets, PII, and
  credentials before they are written to the conversation store. Patterns are
  defined per tenant under the `dlp` block and can be tailored for regulated
  datasets.【F:modules/Server/dlp.py†L1-L78】【F:modules/Server/routes.py†L3177-L3240】
* **Tenant scoping** – All server routes require a tenant identifier so request
  pipelines can enforce per-tenant residency and export policies consistently.
* **Auditing** – Pair residency settings with audit templates so exports to SIEM
  or archive destinations carry the right retention windows and metadata.

## Suggested rollout steps

1. Confirm the target region and residency stance with legal/compliance.
2. Enter those selections in the **Company (Enterprise)** step of the setup
   wizard.
3. Review the default DLP patterns and adjust per-tenant entries in
   `config.yaml` if specific identifiers must be masked or blocked.
4. Connect audit sinks or SIEM exports that match your jurisdictional
   obligations before onboarding additional tenants.
