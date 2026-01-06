# Android Output Envelope V1

**Schema Version**: 1  
**Transport**: JSON (stdout/pipe) → UniFFI later

## Design Principles

- **Single envelope** for `eval` and `solve`
- **Dual rendering**: `display` (UI) + `canonical` (logic/tests)
- **Polymorphism by `kind`** (Kotlin sealed classes)
- **Forward-compatible**: unknown `kind` → ignore

---

## 1. Root Envelope

```json
{
  "schema_version": 1,
  "engine": { "name": "ExpliCAS", "version": "1.3.7" },
  "request": {
    "kind": "eval|solve",
    "input": "sqrt(x)^2",
    "solve_var": null,
    "options": {
      "domain_mode": "generic",
      "value_domain": "real",
      "hints": true,
      "explain": false
    }
  },
  "result": { /* EvalResultDto | SolveResultDto */ },
  "transparency": {
    "required_conditions": [],
    "assumptions_used": [],
    "blocked_hints": []
  },
  "steps": []
}
```

---

## 2. Common DTOs

### ExprDto
```json
{ "display": "√(x)", "canonical": "sqrt(x)" }
```

### ConditionDto (Requires)
```json
{
  "kind": "NonNegative|Positive|NonZero",
  "expr_display": "x",
  "expr_canonical": "x",
  "display": "x ≥ 0"
}
```

### AssumptionDto (Assumed)
```json
{
  "kind": "NonZero",
  "expr_display": "x - y",
  "expr_canonical": "x - y",
  "display": "x - y ≠ 0",
  "rule": "Cancel Common Factors"
}
```

### BlockedHintDto
```json
{
  "rule": "Exponential-Log Inverse",
  "requires": ["x > 0"],
  "tip": "use `domain assume`"
}
```

---

## 3. EvalResultDto

```json
{
  "kind": "eval_result",
  "value": { "display": "x", "canonical": "x" }
}
```

---

## 4. SolveResultDto

```json
{
  "kind": "solve_result",
  "solutions": { /* SolutionSetDto */ },
  "residual": null
}
```

### SolutionSetDto variants

| Kind | Schema |
|------|--------|
| `finite_set` | `{ "elements": [ExprDto] }` |
| `interval` | `{ "lower": BoundDto, "upper": BoundDto }` |
| `all_reals` | `{}` |
| `empty_set` | `{}` |
| `conditional` | `{ "cases": [CaseDto] }` |

### CaseDto (for conditional)
```json
{
  "when": { "predicates": [ConditionDto], "is_otherwise": false },
  "then": { "solutions": SolutionSetDto, "residual": null }
}
```

---

## 5. StepDto

```json
{
  "index": 1,
  "rule": "Power of a Power",
  "before": ExprDto,
  "after": ExprDto,
  "assumptions_used": [AssumptionDto],
  "required_conditions": [ConditionDto]
}
```

---

## 6. Examples

### A) `sqrt(x)^2` → Requires

```json
{
  "result": { "kind": "eval_result", "value": { "display": "x", "canonical": "x" } },
  "transparency": {
    "required_conditions": [
      { "kind": "NonNegative", "display": "x ≥ 0", "expr_canonical": "x" }
    ],
    "assumptions_used": [],
    "blocked_hints": []
  }
}
```

### B) `(x-y)/(sqrt(x)-sqrt(y))` → Assumed only

```json
{
  "transparency": {
    "required_conditions": [],
    "assumptions_used": [
      { "kind": "NonZero", "display": "x - y ≠ 0", "rule": "Cancel Common Factors" }
    ]
  }
}
```

---

## 7. Kotlin Binding

```kotlin
@Serializable
sealed class SolutionSetDto {
    @SerialName("finite_set")
    data class FiniteSet(val elements: List<ExprDto>) : SolutionSetDto()
    
    @SerialName("all_reals")
    object AllReals : SolutionSetDto()
    
    @SerialName("empty_set")
    object EmptySet : SolutionSetDto()
    
    @SerialName("conditional")
    data class Conditional(val cases: List<CaseDto>) : SolutionSetDto()
}

@Serializable
data class ConditionDto(
    val kind: String,
    val display: String,
    @SerialName("expr_canonical") val exprCanonical: String
)
```

---

## 8. Versioning Policy

- `schema_version` increments on **breaking changes**
- Unknown `kind` values → **ignore** (forward-compatible)
- New optional fields → **backwards-compatible**
