---
audience: Contributors
status: draft
last_verified: 2025-12-21
source_of_truth: Links to setup wizard UI/tasks
---

# Setup Wizard Enhancements Task List

This tracker captures engineering tasks that expand the GTK setup wizard with the
additional configuration fields requested during review. Each task references the
corresponding controller state so implementers can navigate the codebase quickly.

## Tasks

### 1. Durable Job Scheduling Controls
- Build a dedicated wizard page bound to `JobSchedulingState` to surface the
  enable toggle, job store DSN, retry policy, worker counts, queue size, and
  timezone selections.
- Validate numeric inputs inline and call
  `SetupWizardController.apply_job_scheduling_settings` before proceeding to the
  next step.
- Update `docs/setup-wizard.md` to describe the job scheduling UI and add unit
  coverage in `tests/test_setup_wizard.py`.

### 2. Message Bus Backend Selection
- Add a step that binds to `MessageBusState`, allowing operators to configure
  the AgentBus/NCB transport options, including optional Redis bridging DSN and
  Kafka producer settings that conditionally enable based on the selection.
- Invoke
  `SetupWizardController.apply_message_bus_settings` when the user advances,
  surfacing validation errors within the wizard.
- Extend test coverage to confirm bridging-specific fields are saved and documented
  in `docs/setup-wizard.md`.

### 3. Key-Value Store Configuration
- Provide UI that binds to `KvStoreState` so administrators can reuse the
  conversation database or supply an alternate SQLAlchemy DSN for cache/backplane
  workloads.
- Persist choices through `SetupWizardController.apply_kv_store_settings` and
  block navigation until validation succeeds.
- Document the new page in `docs/setup-wizard.md` and exercise the flow in
  `tests/test_setup_wizard.py`.

### 4. Speech Provider Integrations
- Create a wizard page for `SpeechState` with toggles for text-to-speech and
  speech-to-text, alongside fields for ElevenLabs, OpenAI, and Google API keys
  and default voice/model selections.
- Call `SetupWizardController.apply_speech_settings` on completion and surface
  any provider-specific validation messages inline.
- Refresh `docs/setup-wizard.md` and expand tests to cover enabling/disabling
  individual speech providers.

### 5. Operational Policy Settings
- Expose `OptionalState` controls for tenant ID, conversation retention window,
  scheduler overrides, and the HTTP auto-start toggle.
- Apply inputs with
  `SetupWizardController.apply_optional_settings` before allowing the wizard to
  finish.
- Update `docs/setup-wizard.md` with the new policy section and verify coverage
  through unit tests.
