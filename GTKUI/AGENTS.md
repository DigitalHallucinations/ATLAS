# GTKUI Agent Guidelines

## Scope
- UI Agent changes are limited to `GTKUI/` (and may reference assets in `Icons/` when necessary).
- Avoid modifying backend logic, persistence layers, or configuration outside the GTK shell wiring.

## Required checks before merge
- Run UI smoke verification via `python3 main.py` when feasible to confirm GTK shell still starts.
- Run focused UI tests when added under `tests/` (coordinate with a Testing Agent if test changes are needed).

## Coordination
- Follow the **non-overlap principle**: keep GTKUI changes separate from backend/storage edits. For cross-cutting UI + backend work, split commits by scope or coordinate with the Backend or Data/DB Agent.
