# ExpliCAS JSON API Specification (Schema v1)

## Overview

This document specifies the JSON API contract for `CasNative.evalJson()`.  
Both CLI (`expli eval-json`) and FFI return this exact same schema.

**Schema Version**: `1` (stable)  
**ABI Version**: `2`

---

## Request

### Function Signature

```kotlin
external fun evalJson(expr: String, optsJson: String): String
```

### Options JSON

```json
{
  "budget": {
    "preset": "cli",
    "mode": "best-effort"
  },
  "steps": false,
  "pretty": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `budget.preset` | string | `"cli"` | `"small"`, `"cli"`, `"unlimited"` |
| `budget.mode` | string | `"best-effort"` | `"strict"` (fail on exceed) or `"best-effort"` (partial result) |
| `steps` | bool | `false` | Include simplification steps |
| `pretty` | bool | `false` | Pretty-print JSON output |

---

## Response Structure

```json
{
  "schema_version": 1,
  "ok": true,
  "result": "2·x",
  "budget": { "preset": "cli", "mode": "best-effort" },
  "steps": [],
  "warnings": []
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Always `1` |
| `ok` | bool | `true` if success, `false` if error |
| `budget` | object | Budget info (always present) |

### Optional Fields

| Field | Present When | Type |
|-------|--------------|------|
| `result` | `ok=true` | string |
| `error` | `ok=false` | object |
| `steps` | `steps=true` in opts | array |
| `warnings` | any warnings | array |

---

## Success Response

```json
{
  "schema_version": 1,
  "ok": true,
  "result": "x² + 2·x + 1",
  "budget": {
    "preset": "cli",
    "mode": "best-effort"
  }
}
```

---

## Error Response

```json
{
  "schema_version": 1,
  "ok": false,
  "error": {
    "kind": "ParseError",
    "code": "E_PARSE",
    "message": "unexpected end of input",
    "span": { "start": 0, "end": 1 },
    "details": null
  },
  "budget": {
    "preset": "cli",
    "mode": "best-effort"
  }
}
```

### Error Object

| Field | Type | Stability | Description |
|-------|------|-----------|-------------|
| `kind` | string | **STABLE** | Error category |
| `code` | string | **STABLE** | Error code (starts with `E_`) |
| `message` | string | may change | Human-readable message |
| `span` | object? | optional | Source location `{start, end}` |
| `details` | any | extensible | Additional structured data |

### Error Kinds (Stable)

| Kind | Description |
|------|-------------|
| `ParseError` | Input parsing failed |
| `DomainError` | Mathematical domain violation |
| `SolverError` | Equation solving failed |
| `BudgetExceeded` | Resource limit hit |
| `NotImplemented` | Feature not available |
| `InternalError` | Bug in the engine |
| `InvalidInput` | Invalid options JSON |

### Error Codes (Stable)

| Code | Kind | Meaning |
|------|------|---------|
| `E_PARSE` | ParseError | Syntax error |
| `E_DIV_ZERO` | DomainError | Division by zero |
| `E_VAR_NOT_FOUND` | DomainError | Unknown variable |
| `E_BUDGET` | BudgetExceeded | Budget exceeded |
| `E_NOT_IMPL` | NotImplemented | Not supported |
| `E_INTERNAL` | InternalError | Engine bug |
| `E_INVALID_INPUT` | InvalidInput | Bad opts JSON |

---

## Budget Object

```json
{
  "preset": "cli",
  "mode": "best-effort",
  "exceeded": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `preset` | string | Budget preset used |
| `mode` | string | `"strict"` or `"best-effort"` |
| `exceeded` | object? | Only in best-effort when limit hit |

### Budget Exceeded (best-effort mode)

When `mode="best-effort"` and limit is hit:

```json
{
  "schema_version": 1,
  "ok": true,
  "result": "(partial result)",
  "budget": {
    "preset": "small",
    "mode": "best-effort",
    "exceeded": {
      "op": "Expand",
      "metric": "TermsMaterialized",
      "used": 150,
      "limit": 100
    }
  }
}
```

**UI Handling**: Show result with warning "computation limit reached".

---

## Steps Array

When `steps=true`:

```json
{
  "schema_version": 1,
  "ok": true,
  "result": "2·x",
  "steps": [
    {
      "phase": "Simplify",
      "rule": "Combine Like Terms",
      "before": "x + x",
      "after": "2·x"
    }
  ],
  "budget": { "preset": "cli", "mode": "best-effort" }
}
```

### Step Object

| Field | Type | Description |
|-------|------|-------------|
| `phase` | string | Phase name |
| `rule` | string | Rule applied |
| `before` | string | Expression before |
| `after` | string | Expression after |

---

## Warnings Array

```json
{
  "warnings": [
    {
      "kind": "DomainAssumption",
      "message": "PowerRule: x ≠ 0"
    }
  ]
}
```

---

## Examples

### Simple evaluation

**Input**: `"x + x"`, `"{}"`

```json
{"schema_version":1,"ok":true,"result":"2·x","budget":{"preset":"cli","mode":"best-effort"}}
```

### Parse error

**Input**: `"("`, `"{}"`

```json
{"schema_version":1,"ok":false,"error":{"kind":"ParseError","code":"E_PARSE","message":"unexpected end of input"},"budget":{"preset":"cli","mode":"best-effort"}}
```

### Invalid options

**Input**: `"x"`, `"{bad"`

```json
{"schema_version":1,"ok":false,"error":{"kind":"InvalidInput","code":"E_INVALID_INPUT","message":"Invalid options JSON: ...","details":{"error":"..."}},"budget":{"preset":"unknown","mode":"strict"}}
```

### With steps

**Input**: `"x + x"`, `'{"steps":true}'`

```json
{
  "schema_version": 1,
  "ok": true,
  "result": "2·x",
  "steps": [
    {"phase":"Simplify","rule":"Combine Like Terms","before":"x + x","after":"2·x"}
  ],
  "budget": {"preset":"cli","mode":"best-effort"}
}
```

---

## Kotlin Data Classes

```kotlin
@Serializable
data class EngineResponse(
    @SerialName("schema_version") val schemaVersion: Int,
    val ok: Boolean,
    val result: String? = null,
    val error: EngineError? = null,
    val budget: BudgetInfo,
    val steps: List<EngineStep> = emptyList(),
    val warnings: List<EngineWarning> = emptyList()
)

@Serializable
data class EngineError(
    val kind: String,
    val code: String,
    val message: String,
    val span: Span? = null,
    val details: JsonElement? = null
)

@Serializable
data class BudgetInfo(
    val preset: String,
    val mode: String,
    val exceeded: BudgetExceeded? = null
)

@Serializable
data class BudgetExceeded(
    val op: String,
    val metric: String,
    val used: Long,
    val limit: Long
)

@Serializable
data class Span(val start: Int, val end: Int)

@Serializable
data class EngineStep(
    val phase: String,
    val rule: String,
    val before: String,
    val after: String
)

@Serializable
data class EngineWarning(
    val kind: String,
    val message: String
)
```

---

## Stability Contract

- `schema_version`, `ok`, `kind`, `code` are **STABLE**
- `message` is human-readable and may change
- `details` is extensible (new keys may be added)
- Result strings never contain `__hold` (internal marker)
- **Unknown fields are ignored** - use `ignoreUnknownKeys = true`

---

## UI Handling by Error Code

| Code | UI Action |
|------|-----------|
| `E_PARSE` | Show message. If `span` present, highlight range or move cursor. |
| `E_DIV_ZERO` | Show "División por cero detectada" |
| `E_VAR_NOT_FOUND` | Show "Variable no encontrada: X" |
| `E_BUDGET` | Show "Límite de cálculo. Intenta con un preset mayor (cli/unlimited)." |
| `E_NOT_IMPL` | Show "Función no soportada" |
| `E_INVALID_INPUT` | Log error (bug del cliente). No mostrar a usuario. |
| `E_INTERNAL` | Show "Error interno. Por favor reporta." Sin detalles sensibles. |

### Partial Results (best-effort)

```kotlin
if (resp.ok && resp.budget.exceeded != null) {
    showWarning("Resultado parcial - límite de cálculo alcanzado")
}
```

---

## Options JSON Defaults

### Minimal (uses all defaults)

```kotlin
CasNative.evalJson("x + x", "{}")
// preset="cli", mode="best-effort", steps=false, pretty=false
```

### Full example

```json
{
  "budget": { "preset": "small", "mode": "strict" },
  "steps": true,
  "pretty": false
}
```

### Forward Compatibility

Unknown fields are ignored. Future versions may add new options.

```kotlin
val json = Json { ignoreUnknownKeys = true }
```

---

## Kotlin Helpers

```kotlin
/** Get result or throw typed exception */
fun EngineResponse.resultOrThrow(): String {
    if (ok) return result ?: ""
    throw CasException(error ?: EngineError("InternalError", "E_INTERNAL", "Missing error"))
}

/** Check if result is partial (budget exceeded in best-effort) */
fun EngineResponse.isPartial(): Boolean = ok && budget.exceeded != null

/** Exception with structured error info */
class CasException(val error: EngineError) : Exception("${error.kind} (${error.code}): ${error.message}")
```

---

## Threading

> ⚠️ **JNI calls must NOT run on the UI thread.**

```kotlin
// Option A: Coroutine
withContext(Dispatchers.Default) {
    CasNative.evalJson(expr, opts)
}

// Option B: Executor
Executors.newSingleThreadExecutor().submit {
    CasNative.evalJson(expr, opts)
}
```

For production: use a dedicated computation dispatcher or worker thread pool.

