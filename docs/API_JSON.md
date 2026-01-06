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

---

## Known `kind` Values

| kind | Meaning |
|------|---------|
| `NonNegative` | x ≥ 0 |
| `Positive` | x > 0 |
| `NonZero` | x ≠ 0 |
| `Defined` | (reserved) |
| `EqZero` | x = 0 |
| `EqOne` | x = 1 |
| `Otherwise` | Default case |

> **Policy**: Unknown kinds must be **ignored** (forward-compatible).

---

## See Also

- [ANDROID_OUTPUT_ENVELOPE.md](./ANDROID_OUTPUT_ENVELOPE.md) — Full FFI envelope spec
- [Requires_vs_assumed.md](./Requires_vs_assumed.md) — Semantic contract
