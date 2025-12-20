# JSON CLI API

The `cas_cli` supports a non-interactive JSON output mode for scripting, testing, and notebook integration (e.g., Google Colab). This API allows programmatic access to the CAS engine without flooding terminal output.

## Quick Start

```bash
# Evaluate an expression
cas_cli eval-json "x^2 + 1"

# With options
cas_cli eval-json "expand((x+1)^5)" --max-chars 500 --steps off
```

## Subcommands

### `eval-json`

Evaluates a single expression and returns structured JSON output.

```bash
cas_cli eval-json <EXPR> [OPTIONS]
```

**Arguments:**
- `<EXPR>` - Expression to evaluate (required)

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--max-chars <N>` | 2000 | Maximum characters for result output. Truncates if larger. |
| `--steps <MODE>` | off | Steps mode: `on`, `off`, `compact` |
| `--context <MODE>` | auto | Context mode: `auto`, `standard`, `solve`, `integrate` |
| `--branch <MODE>` | strict | Branch mode: `strict`, `principal` |
| `--complex <MODE>` | auto | Complex mode: `auto`, `on`, `off` |
| `--autoexpand <MODE>` | off | Expand policy: `off`, `auto` |
| `--threads <N>` | (system) | Number of Rayon threads for parallel processing |

**Examples:**

```bash
# Simple evaluation
cas_cli eval-json "sin(pi/4)"

# Polynomial GCD
cas_cli eval-json "poly_gcd_exact(x^2-1, x^2-2*x+1)"

# Expansion with truncation for large results
cas_cli eval-json "expand((x+1)^10)" --max-chars 200

# With specific options
cas_cli eval-json "atan(tan(x))" --branch principal --steps on
```

## JSON Output Format

### Success Response

```json
{
  "ok": true,
  "input": "x^2 + 1",
  "result": "1 + x^2",
  "result_truncated": false,
  "result_chars": 7,
  "steps_mode": "off",
  "steps_count": 0,
  "warnings": [],
  "stats": {
    "node_count": 5,
    "depth": 2
  },
  "hash": null,
  "timings_us": {
    "parse_us": 701,
    "simplify_us": 1229,
    "total_us": 4372
  },
  "options": {
    "context_mode": "auto",
    "branch_mode": "strict",
    "expand_policy": "off",
    "complex_mode": "auto",
    "steps_mode": "off"
  }
}
```

### Error Response

```json
{
  "ok": false,
  "error": "Parse error: unexpected token",
  "input": "x + + 1"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `ok` | bool | Whether evaluation succeeded |
| `input` | string | Original input expression |
| `result` | string | Pretty-printed result (may be truncated) |
| `result_truncated` | bool | True if result was truncated |
| `result_chars` | int | Original character count before truncation |
| `steps_mode` | string | Steps mode that was used |
| `steps_count` | int | Number of simplification steps |
| `warnings` | array | Domain assumption warnings |
| `stats.node_count` | int | Total AST nodes in result |
| `stats.depth` | int | Maximum depth of result AST |
| `hash` | string? | Expression hash (only when truncated) |
| `timings_us.parse_us` | int | Parsing time in microseconds |
| `timings_us.simplify_us` | int | Simplification time in microseconds |
| `timings_us.total_us` | int | Total time in microseconds |
| `options` | object | Options that were used |

### Warnings Format

When domain assumptions are made during simplification:

```json
{
  "warnings": [
    {
      "rule": "InverseTrigRule",
      "assumption": "Assuming x ∈ (-π/2, π/2) for atan(tan(x)) → x"
    }
  ]
}
```

## Usage from Python/Colab

```python
import json
import subprocess

# Simple evaluation
def eval_expr(expr, **options):
    cmd = ["./cas_cli", "eval-json", expr]
    for k, v in options.items():
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    result = subprocess.check_output(cmd, text=True)
    return json.loads(result)

# Examples
result = eval_expr("x^2 + 1")
print(result["result"])  # "1 + x^2"

result = eval_expr("poly_gcd_exact(x^2-1, x-1)")
print(result["result"])  # "x - 1"
print(result["timings_us"]["simplify_us"])  # e.g., 1500

# With truncation for large expressions
result = eval_expr("expand((x+1)^20)", max_chars=100)
if result["result_truncated"]:
    print(f"Result has {result['result_chars']} chars, hash: {result['hash']}")
```

## Truncation and Large Expressions

The `--max-chars` option prevents terminal flooding for large expressions:

1. When result exceeds `max_chars`, it's truncated with `" … <truncated>"` suffix
2. `result_truncated` is set to `true`
3. `result_chars` shows original length
4. `hash` is populated for identity comparison without printing

```bash
# Large polynomial - safely truncated
cas_cli eval-json "expand((x+1)^15)" --max-chars 100
```

```json
{
  "result": "1 + x^15 + 15·x + 15·x^14 + 105·x^2 + 105·x^13 + 455·x^3 + 455·x^12 + 1365·x^ … <truncated>",
  "result_truncated": true,
  "result_chars": 287,
  "hash": "a1b2c3d4e5f67890"
}
```

## Controlling Parallelism

For reproducible benchmarks or resource-constrained environments:

```bash
# Single-threaded execution
cas_cli eval-json "poly_gcd_exact(a, b)" --threads 1

# Limit to 4 threads
cas_cli eval-json "expand((x+1)^10)" --threads 4
```

## Backward Compatibility

The JSON API is additive - the interactive REPL continues to work as before:

```bash
# REPL mode (default, no subcommand)
cas_cli
cas_cli --no-pretty

# JSON mode (explicit subcommand)
cas_cli eval-json "..."
```

## Future Subcommands

Planned additions:

- **`script-json`**: Execute multi-line scripts from stdin
- **`mm-gcd-modp-json`**: Run mm_gcd benchmark with JSON output

## Files

- `crates/cas_cli/src/commands/eval_json.rs` - eval-json implementation
- `crates/cas_cli/src/json_types.rs` - JSON output structures
- `crates/cas_cli/src/format.rs` - Truncation and stats utilities
