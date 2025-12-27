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
