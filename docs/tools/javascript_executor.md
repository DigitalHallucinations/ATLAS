# JavaScript Executor Tool

The JavaScript executor runs snippets inside a tightly controlled sandbox. It is
backed by an external runtime such as QuickJS, Deno, or a custom bridge and
enforces CPU, memory, and output quotas so that generated scripts remain
well-behaved.

## Configuration

The executor configuration lives under the `tools.javascript_executor` section
of `config.yaml` (environment variables with the prefix `JAVASCRIPT_EXECUTOR_*`
override the same fields). The most relevant options are:

| Key | Description |
| --- | --- |
| `executable` | Absolute path to the runtime binary. Auto-detected when omitted. |
| `args` | Optional list of command-line arguments passed before the script path. |
| `default_timeout` | Fallback timeout (seconds) when the caller does not specify one. |
| `cpu_time_limit` | POSIX CPU time quota applied via `RLIMIT_CPU`. Set to `null` to disable. |
| `memory_limit_bytes` | Address space limit enforced through `RLIMIT_AS`. |
| `max_output_bytes` | Maximum stdout/stderr bytes captured before truncation. |
| `max_file_bytes` | Maximum number of bytes preserved for each generated file. |
| `max_files` | Upper bound on how many artifacts are returned per invocation. |

### Example configuration

```yaml
tools:
  javascript_executor:
    executable: /usr/bin/qjs
    args: ["--std"]
    default_timeout: 5.0
    cpu_time_limit: 2.0
    memory_limit_bytes: 268435456
    max_output_bytes: 65536
    max_file_bytes: 131072
    max_files: 16
```

To switch the runtime via environment variables:

```bash
export JAVASCRIPT_EXECUTOR_BIN=/usr/local/bin/deno
export JAVASCRIPT_EXECUTOR_ARGS="run --allow-read"
```

## Usage

Invoke the tool through the `execute_javascript` function. The payload mirrors
the manifest entry:

```json
{
  "name": "execute_javascript",
  "arguments": {
    "command": "console.log('Hello from JS');",
    "timeout": 3.0,
    "files": [
      {"path": "input/data.txt", "content": "seed text"}
    ]
  }
}
```

The response includes stdout, stderr, truncation flags, and any new or modified
files (base64 encoded). Persona tool policies require the flag
`type.developer.javascript_execution_enabled` to be enabled before the tool is
callable.
