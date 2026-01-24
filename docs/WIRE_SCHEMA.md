# Wire Schema Documentation

> **Version**: 1  
> **Status**: Stable (backwards compatible)

## Overview

The Wire format provides a unified, versioned messaging contract for all ExpliCAS output channels:

- CLI (`eval --format json`)
- Web API
- FFI (Android, future: PyO3)

## Schema

```json
{
  "wire": {
    "schema_version": 1,
    "messages": [
      {
        "kind": "output|info|warn|error|steps|debug",
        "text": "message content",
        "span": { "start": 0, "end": 5 }  // optional, for errors
      }
    ]
  }
}
```

## Message Kinds

| Kind | Description | Example |
|------|-------------|---------|
| `output` | Main result | `"Result: 4"` |
| `info` | Informational | `"ℹ️ Requires: x ≠ 0"` |
| `warn` | Warning | `"⚠ Division by zero assumed"` |
| `error` | Error | `"Parse error: unexpected token"` |
| `steps` | Step summary | `"5 simplification step(s)"` |
| `debug` | Debug output | (only in debug mode) |

## Message Order

Messages appear in display order:

1. `warn` — domain warnings
2. `info` — required conditions
3. `output` — main result
4. `steps` — step count (if steps_mode=on)

## Span (Error Localization)

For parse errors, `span` contains byte offsets:

```json
{
  "kind": "error",
  "text": "unexpected token",
  "span": { "start": 4, "end": 5 }
}
```

Use this to render carets:

```
x + * 3
    ^
```

## Versioning

- `schema_version: 1` is the current version
- Additions are backwards compatible
- Breaking changes increment version

## Examples

### Success

```bash
expli eval "2+2" --format json | jq '.wire'
```

```json
{
  "schema_version": 1,
  "messages": [
    { "kind": "output", "text": "Result: 4 [LaTeX: 4]" }
  ]
}
```

### With Warnings

```json
{
  "schema_version": 1,
  "messages": [
    { "kind": "warn", "text": "⚠ x ≠ 0 assumed (SimplifyFraction)" },
    { "kind": "output", "text": "Result: 1/x" }
  ]
}
```

### Parse Error

```json
{
  "schema_version": 1,
  "messages": [
    { 
      "kind": "error", 
      "text": "unexpected token",
      "span": { "start": 4, "end": 5 }
    }
  ]
}
```

## Implementation

- **Rust module**: `crates/cas_cli/src/repl/wire.rs`
- **Conversion**: `From<ReplReply> for WireReply`
- **JSON field**: `EvalJsonOutput.wire`
