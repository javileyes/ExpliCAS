# JSON API Reference

API version: `schema_version: 1`

## Versioning Policy
- `schema_version` increments on breaking changes
- Unknown fields: **ignore** forward-compatible
- Unknown `kind` values: **ignore** (extensibility)

## required_conditions

Implicit domain constraints from input expression structure.

```json
"required_conditions": [
  {
    "kind": "NonNegative",
    "expr_display": "x",
    "expr_canonical": "x"
  }
]
```

### Fields
| Field | Type | Description |
|-------|------|-------------|
| `kind` | string | `NonNegative`, `Positive`, `NonZero` |
| `expr_display` | string | May vary with display transforms |
| `expr_canonical` | string | Stable canonical rendering |

### Contract
- `required_conditions` separate from `assumptions` (warnings)
- Witness survival: if expression is preserved in result, no requirement emitted
- Empty array if no requirements
