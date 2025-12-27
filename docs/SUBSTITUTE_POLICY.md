# Substitute Policy V1

Pattern substitution without inventing algebra.

## Entry Point

```rust
pub fn substitute_power_aware(ctx, root, target, replacement, opts) -> ExprId
```

## Allowlist V1

| Case | Behavior |
|------|----------|
| **Exact match** | `s == target` → `replacement` |
| **Power multiple** | `Pow(u, n)` where `target=Pow(u, k)` and `n%k==0` → `Pow(replacement, n/k)` |
| **Pow of target** | `Pow(target, m)` → `Pow(replacement, m)` |

## Denylist

The substitute module must NOT:
- Call `simplify`, `expand`, `rationalize`
- Use `poly_*`, `gcd_*`, `multinomial`
- Apply `apply_rules_loop` or other engine pipelines
- Depend on `DomainMode`, `ValueDomain`, `BranchPolicy`

## No-Goals V1

- Converting products to powers (`x*x` is NOT `x^2`)
- Commutative/associative matching
- Domain assumptions

## Enforcement

```bash
bash scripts/lint_substitute_enforcement.sh
```

## Contract Tests

Located in `crates/cas_engine/tests/substitute_contract_tests.rs`:
- A1, A3: ExactOnly mode
- B1-B5: PowerPattern mode
- C1-C3: Robustness (no invented algebra)
- D1: No `__hold` leak in result
- D2: Deep expression substitution
- D3: Multiple matches replaced
- D4: No invention (`x*x` ≠ `x^2`)

---

## Steps Schema (v1)

Contract for output when `steps=true`.

### Stable Fields

| Field | Type | Stability | Description |
|-------|------|-----------|-------------|
| `rule` | `String` | **STABLE** | Rule name (do not change) |
| `before` | `String` | Stable | Render of expression before rewrite (no `__hold`) |
| `after` | `String` | Stable | Render of expression after rewrite (no `__hold`) |
| `note` | `Option<String>` | Unstable | Human-readable metadata (NOT for client logic) |

### Stable Rule Names

- `SubstituteExact` — Exact structural match
- `SubstitutePowerMultiple` — Power multiple: `x^n → y^(n/k)` when target is `x^k`
- `SubstitutePowOfTarget` — Power of target: `(x^k)^m → y^m`

### Invariants

1. `before != after` for all steps
2. Neither `before` nor `after` contains `__hold`
3. Steps are ordered by application sequence
4. Each step represents a **local rewrite**, not a global algebraic step

### JSON Example

```json
{
  "ok": true,
  "result": "1 + y + y^2",
  "steps": [
    { "rule": "SubstituteExact", "before": "x^2", "after": "y", "note": null },
    { "rule": "SubstitutePowerMultiple", "before": "x^4", "after": "y^2", "note": "n=4, k=2, m=2" }
  ],
  "request": {
    "target": "x^2",
    "with": "y",
    "mode": "power",
    "steps": true
  }
}
```

### Empty Steps

When `steps=false` or no substitutions occurred, the `steps` field is omitted (not empty array).

---

## Future: Metrics (Reserved)

```rust
pub struct SubstituteStats {
    pub nodes_visited: u64,
    pub rewrites_applied: u64,
    pub steps_emitted: u64,
}
```

Reserved for budget integration. Not yet exposed in JSON.

---

## JSON API (v1)

### Entry Point

```rust
pub fn substitute_str_to_json(
    expr_str: &str,
    target_str: &str,
    with_str: &str,
    opts_json: Option<&str>,
) -> String
```

**Canonical entry point** for CLI + FFI + Android. All implementations must use this function.

### Options

```json
{
  "mode": "power",    // "exact" or "power" (default: power)
  "steps": true,      // Include step details
  "pretty": false     // Pretty-print output
}
```

### Response Schema

```json
{
  "schema_version": 1,
  "ok": true,
  "result": "y^2 + y + 1",
  "request": { "expr": "x^4 + x^2 + 1", "target": "x^2", "with": "y" },
  "options": { "substitute": { "mode": "power", "steps": true } },
  "steps": [
    { "rule": "SubstituteExact", "before": "x^2", "after": "y", "note": null },
    { "rule": "SubstitutePowerMultiple", "before": "x^4", "after": "y^2", "note": "n=4, k=2, m=2" }
  ]
}
```

### Contract Tests

Located in `crates/cas_engine/tests/substitute_json_contract_tests.rs`:
- Schema version = 1
- Request echo present
- Options reflected
- Steps schema v1
- No `__hold` leak
- Error path structure


