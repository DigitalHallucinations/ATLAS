# GTKUI Agent Guidelines

## Scope
- UI Agent changes are limited to `GTKUI/` (and may reference assets in `Icons/` when necessary).
- Avoid modifying backend logic, persistence layers, or configuration outside the GTK shell wiring.

## Accessibility, contrast, and layout
- Default to WCAG AA contrast; use existing color tokens and verify text-to-background contrast before merging.
- Preserve keyboard navigability and focus visibility on new or updated controls.
- Respect GTK spacing scales and padding values already present in adjacent components; avoid one-off pixel values.
- Keep layouts fluid for 1280×720 through 1920×1080; ensure no truncation at 1366×768. If you add breakpoints, document them in the MR/PR notes.

## Screenshot and visual validation
- Capture before/after screenshots for any perceptible visual change. Use consistent window sizes (at least 1366×768) and note DPI scaling if non-default.
- Include captions describing the flow and state shown. If screenshots cannot be taken, note the limitation in the PR.
- Confirm high-contrast themes or dark/light modes remain legible; add comparative screenshots when contrast-sensitive elements change.

## UI flow checklist
- Primary navigation: open sidebars/drawers, traverse tabs, and verify focus order.
- Data entry: fill forms (including validation errors), submit, and confirm toasts/dialogs render.
- Empty/loading states: show skeletons or placeholders and ensure layout stability.
- Resizing: drag window between 1280×720 and 1920×1080 to confirm responsive behavior without clipped text or controls.

## Required checks before merge
- Run UI smoke verification via `python3 main.py` when feasible to confirm GTK shell still starts.
- Run focused UI tests when added under `tests/` (coordinate with a Testing Agent if test changes are needed).

## Traceability and audits
- Link UI changes to the `_audit` references: keep `docs/_audit/inventory.md` and `docs/_audit/alignment-report.md` in sync with new UI patterns or significant interactions.
- Follow the cadence in `_audit/architecture-alignment-report.md` for periodic alignment updates; note missed cadences in the PR description.

## Coordination
- Follow the **non-overlap principle**: keep GTKUI changes separate from backend/storage edits. For cross-cutting UI + backend work, split commits by scope or coordinate with the Backend or Data/DB Agent.
- Reference the shared workflow in [`../docs/contributing/agent-workflow.md`](../docs/contributing/agent-workflow.md) to capture intent, check guardrails, align designs, honor execution constraints, validate changes (including screenshots), and document handoffs.
