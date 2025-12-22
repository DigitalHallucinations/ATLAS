---
audience: Security reviewers and documentation maintainers
status: draft
last_verified: 2026-02-22
source_of_truth: AGENTS.md
---

# Security audit checklist

Use this checklist when reviewing security-facing documentation and configuration guidance. It centers on DLP, access controls, and transport security expectations for ATLAS deployments.

## How to use this checklist

- Align the review scope with the owning modules or configuration files before scoring any items.
- Capture outcomes in [`alignment-report.md`](./alignment-report.md) with links to issues or pull requests for every finding.
- Reflect any doc changes or risk status updates in [`inventory.md`](./inventory.md), including owners and next review dates.

## Data loss prevention (DLP)

- [ ] Document how sensitive inputs and outputs are classified, scrubbed, or redacted.
- [ ] Confirm export/residency controls cover all storage locations and backup paths.
- [ ] Verify logging guidance masks secrets, tokens, and user identifiers by default.
- [ ] Ensure data retention settings link to the correct configuration keys and defaults.

## Access controls

- [ ] List all authentication flows and the roles or permissions they enforce.
- [ ] Flag privileged endpoints, admin tooling, and service accounts with required scopes.
- [ ] Call out default credential policies (passwords, tokens, API keys) and rotation expectations.
- [ ] Note least-privilege defaults for background workers, schedulers, and cross-service calls.

## Transport security

- [ ] Specify TLS expectations for HTTP, websocket, and gRPC entry points (cert chains, ciphers).
- [ ] Document certificate management workflows, renewal intervals, and fallback behaviors.
- [ ] Include client/server timeout and retry expectations that prevent downgrade or plaintext retries.
- [ ] Capture requirements for intra-service encryption on message buses and data pipelines.

## Reporting and follow-up

- Record each checked item’s status (Aligned, Needs update, Blocked) in [`alignment-report.md`](./alignment-report.md) with owner and due date.
- Cross-reference findings with related code or configuration files using the sourcing rules in [`linking-and-sources.md`](./linking-and-sources.md).
- Update [`inventory.md`](./inventory.md) to reflect new risks, mitigation owners, and next review targets after each audit cycle.
