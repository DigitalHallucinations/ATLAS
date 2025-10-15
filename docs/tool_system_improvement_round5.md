# Tool System Improvement Opportunities (Round 5)

The following items highlight additional refactors and feature gaps observed in the tool execution and provider plumbing. Each entry includes a short rationale and proposed follow-up tasks.

## 1. Persist streaming follow-up messages in history
- **Observation:** When the post-tool model reply is streamed, `use_tool` returns the iterator immediately and never appends the assistant's final message to the conversation history, so transcripts omit the reply that fulfilled the tool call. 【F:ATLAS/ATLAS/ToolManager.py†L669-L739】
- **Why it matters:** Modern APIs (and UI consumers) expect the transcript to contain the assistant turn associated with each `tool_call_id`, even if the reply streamed to the client. Without that entry, later turns lose context and exporting conversations is incomplete.
- **Tasks:**
  1. Buffer streamed chunks (or register a callback) so that once streaming finishes, the assistant reply is materialized and recorded through `conversation_history.add_message` with the appropriate metadata.
  2. Extend the streaming tests to assert that the assistant message eventually appears in history while preserving incremental delivery to clients.

## 2. Preserve multi-part tool follow-up payloads
- **Observation:** `_extract_text_and_audio` only handles string fields and falls back to `str(payload)` for anything else, so list-based or structured `content` payloads from newer Responses/OpenAI models are flattened into unreadable strings before being logged to history. 【F:ATLAS/ATLAS/ToolManager.py†L65-L88】
- **Why it matters:** The Responses API can return mixed text, data, and audio parts; collapsing them prevents downstream renderers from showing tool-rich outputs and makes JSON responses unusable.
- **Tasks:**
  1. Teach `_extract_text_and_audio` (and the subsequent `add_message` call) to normalize list/dict payloads into the content-part schema that `ChatSession` already understands.
  2. Add regression coverage with a synthetic assistant reply containing multiple `output_text` and `output_json` parts to confirm the history preserves the structure.

## 3. Thread-safe caching for persona function metadata
- **Observation:** `_function_payload_cache` is mutated without any locking, unlike the default-map cache that uses `_default_function_map_lock`, so concurrent tool calls can race while reading/writing persona `functions.json` payloads. 【F:ATLAS/ATLAS/ToolManager.py†L22-L25】【F:ATLAS/ATLAS/ToolManager.py†L381-L420】
- **Why it matters:** Provider generators may invoke `use_tool` from multiple tasks simultaneously, leading to partially-populated cache entries or repeated JSON loads.
- **Tasks:**
  1. Introduce a dedicated lock (or reuse an existing threading primitive) around reads/writes to `_function_payload_cache`.
  2. Add a stress test that spins up parallel loads for the same persona and asserts the cached payload remains consistent.

## 4. Harmonize error returns with modern tool schema
- **Observation:** On execution failures, `use_tool` returns a tuple `(error_message, True)`, which bubbles back to providers as an opaque Python tuple instead of a structured `tool` message with the original `tool_call_id`. 【F:ATLAS/ATLAS/ToolManager.py†L640-L693】
- **Why it matters:** The current OpenAI/Mistral APIs expect tool failures to be surfaced either as raised exceptions or as `role="tool"` messages keyed to the initiating call ID. Returning a tuple forces downstream callers to special-case legacy behavior and breaks `submit_tool_outputs` integrations.
- **Tasks:**
  1. Replace the tuple return with an exception or a normalized error object that still records the `tool_call_id`, and ensure the conversation history captures the failure.
  2. Update provider tests to assert that tool errors propagate through the modern schema rather than a tuple sentinel.

## 5. Expand post-tool model invocation options
- **Observation:** `call_model_with_new_prompt` still issues a legacy `functions=` request using the provider's "current model" and omits newer knobs such as `tool_choice`, `parallel_tool_calls`, `json_mode`, or persona-specific model selection. 【F:ATLAS/ATLAS/ToolManager.py†L742-L795】
- **Why it matters:** After executing tools we should replay the exact generation settings from the initiating turn, otherwise capabilities like forced tool choices or JSON-mode completions are silently dropped.
- **Tasks:**
  1. Plumb the original generation parameters (e.g., tool choice, reasoning mode, json schema) into the follow-up call and respect persona/model overrides.
  2. Add integration tests in the OpenAI generator to confirm these settings survive a tool round-trip.
