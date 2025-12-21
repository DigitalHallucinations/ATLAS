---
audience: Persona authors
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/Tools/Base_Tools/tone_analyzer.py; modules/Tools/Base_Tools/reflective_prompt.py; modules/Tools/Base_Tools/memory_recall.py; modules/Tools/Base_Tools/conflict_resolver.py
---

# Mediation support tools

Echo introduces a focused set of mediation utilities that help mirror emotional
signals, prompt reflection, recall earlier agreements, and outline next steps.
Each tool is designed to keep tense conversations grounded in shared intent
without performing network or filesystem operations.

## `tone_analyzer`

* **Side effects:** none
* **Safety level:** Low
* **Typical call pattern:** Provide an ordered list of `{speaker, content}`
  messages (recent transcripts work best). Optionally include
  `focus_topics` so Echo can highlight how often specific concerns appear.
* **Returns:** The dominant tone (supportive, tense, concerned, or neutral), a
  breakdown of tone counts, per-message tone signals, keyword hit counts, and
  facilitator guidance for the next response.
* **Usage tips:** Run before you change topics or propose decisions. Echo uses
  the signals to restate emotions and choose between slowing down or moving the
  group toward commitments.

## `reflective_prompt`

* **Side effects:** none
* **Safety level:** Low
* **Typical call pattern:** Supply the mediation `topic` along with optional
  `tension_points`, a short `tone_observation`, and upcoming `next_review`
  deadlines.
* **Returns:** A deduplicated list of neutral prompts that invite participants
  to articulate needs, assumptions, and commitments.
* **Usage tips:** Trigger after `tone_analyzer` when the group needs to pause
  and reflect before selecting an option. The prompts are intentionally short so
  Echo can pair them with mirrored summaries.

## `memory_recall`

* **Side effects:** none
* **Safety level:** Low
* **Typical call pattern:** Provide the mediation `query`, a list of prior
  `memories` (each with `topic`, `summary`, optional `tags`, and `timestamp`),
  and an optional `limit` for the number of matches to return.
* **Returns:** Ranked memory matches with the original payload, the number of
  evaluated memories, and guidance when nothing matches.
* **Defaults and limits:** Caps results to `limit` (defaults to `5`) and requires
  at least one memory candidate; empty inputs raise a validation error.
* **Usage tips:** Run before proposing actions so Echo can remind participants
  of earlier commitments or risks. Pair with `context_tracker` so new agreements
  are stored for future sessions.

## `conflict_resolver`

* **Side effects:** none
* **Safety level:** Medium (because responses influence human decisions)
* **Typical call pattern:** Provide `positions` capturing each participant's
  statement and optional `non_negotiables`. Add any existing `shared_goals` and
  a `decision_horizon` if there is a deadline.
* **Returns:** Participant list, inferred shared focus areas, non-negotiable
  summaries, and sequenced next steps.
* **Usage tips:** Reserve for moments when the group needs structure. Echo uses
  the output to summarise shared intent, catalog constraints, and document the
  follow-up cadence.

## Operational notes

These mediation tools avoid external side effects (no network or filesystem
access) and do not rely on environment configuration. Inputs are validated to
require non-empty queries or message lists, and defaults (such as the five-item
memory recall cap) are applied inside the Base_Tools implementations.
