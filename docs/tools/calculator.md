---
audience: Persona authors
status: in_review
last_verified: 2025-12-21
source_of_truth: Calculator tool implementation (Base_Tools)
---

# Calculator Tool

The calculator tool evaluates mathematical expressions using Python's `decimal` module so that
results remain precise even for long chains of operations. It accepts a single expression string and
sanitises it using an abstract syntax tree (AST) walk before any evaluation occurs, blocking
potentially dangerous constructs such as attribute access or function definitions.

## Supported operations

- Standard arithmetic operators (`+`, `-`, `*`, `/`, `%`, `**`).
- Unary plus/minus, nested parentheses, and variable substitution.
- Mathematical functions such as `sin`, `cos`, `tan`, `log`, `sqrt`, `exp`, and hyperbolic variants.
- Configurable precision (defaults to 28 decimal places) with a hard ceiling to avoid runaway
  computations.
- Built-in constants (`pi`, `tau`, `e`) plus caller-supplied variables.
- Unit conversions for a curated set of distance, mass, volume, and time units.

## Usage examples

```json
{
  "expression": "sin(pi / 2) + 0.5",
  "precision": 16
}
```

```json
{
  "expression": "distance / time",
  "variables": {"distance": 5, "time": 0.25},
  "input_unit": "km",
  "output_unit": "m"
}
```

## Safety considerations

- Expressions are parsed in `eval` mode and each AST node is checked against an allowlist before
  evaluation to prevent arbitrary code execution.
- The evaluator runs with a bounded node budget, configurable per deployment, to keep workloads
  predictable.
- Unit conversions only accept known unit symbols. Requests referencing unsupported units fail with
  a descriptive error rather than attempting an implicit conversion.
