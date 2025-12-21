---
audience: Persona authors and ops
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/Tools/Base_Tools/terminal_command.py
---

# Terminal Command Tool

The shared `terminal_command` tool executes carefully sandboxed commands inside the ATLAS runtime. It
is available to personas that explicitly allow it in their `allowed_tools` list and have the required
runtime flags enabled. The tool reports both `stdout` and `stderr`, includes safety metadata, and
adheres to the same execution limits as the standalone `TerminalCommand` helper in
`modules/Tools/Base_Tools/terminal_command.py`.

## Manifest entry

The shared manifest entry exposes a standard schema so personas can opt in by referencing the shared
identifier. The entry includes explicit consent, flag gating, and idempotency guidance:

```json
{
  "name": "terminal_command",
  "version": "1.0.0",
  "safety_level": "high",
  "requires_consent": true,
  "requires_flags": {
    "execute": [
      "type.developer.terminal_access"
    ]
  },
  "idempotency_key": {
    "required": true,
    "scope": "per-command",
    "guidance": "Combine the normalized command tokens and working directory to deduplicate retried executions."
  },
  "parameters": {
    "type": "object",
    "required": ["command"],
    "properties": {
      "command": {
        "type": ["string", "array"],
        "description": "Shell-free command (string) or tokenized argument list."
      },
      "timeout": {
        "type": "number",
        "default": 5,
        "description": "Maximum seconds to allow before forcing a timeout (capped at 30s)."
      },
      "working_directory": {
        "type": "string",
        "description": "Optional path relative to the sandbox root."
      }
    }
  }
}
```

## Example usage

```python
import asyncio

from modules.Tools.Base_Tools.terminal_command import TerminalCommand

result = asyncio.run(
    TerminalCommand(
        command=["ls", "-la"],
        timeout=5,
        working_directory="workspace/ATLAS",
    )
)
print(result.exit_code)
print(result.stdout)
```

When invoked through the tool manager, the metadata above ensures the call is only permitted when the
persona has the `type.developer.terminal_access` flag and the user has granted consent. The idempotency
instructions allow higher-level orchestration layers to safely retry read-only commands without
replaying stateful operations.

## Log redaction behavior

The tool emits structured audit logs before and after each invocation. Sensitive arguments such as
`--password=...`, bearer tokens, API keys, and similarly named fields are automatically redacted with
`[REDACTED]` in both the execution and completion log lines. The raw `stdout`/`stderr` text is also
sanitized when it contains those markers. Non-sensitive context—including the normalized command name,
safe arguments, working directory, status, and timing metadata—remains fully visible so operators can
audit activity without leaking credentials.
