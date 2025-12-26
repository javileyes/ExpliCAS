# Limits Policy

> **V1.3** — Pre-simplification contract and enforcement

## 1. Purpose

The limit engine uses a **conservative policy**: never invent results.

To improve resolution rate without breaking this guarantee, V1.3 introduces a **pre-simplification layer** that is:

- **Optional**: `--presimplify off|safe` (default: off)
- **Bounded**: allowlist-only transforms
- **Auditable**: lint + contract tests

---

## 2. PreSimplify Safe Contract

### 2.1 Entry Point

```rust
// limits/presimplify.rs
pub fn presimplify_safe(ctx, expr, budget) -> Result<ExprId, CasError>
```

Single point of entry ensures auditability and test coverage concentration.

### 2.2 Allowlist (Transforms Permitted)

| Transform | Example | Notes |
|-----------|---------|-------|
| Add zero | `a + 0 → a`, `0 + a → a` | Identity |
| Mul one | `a * 1 → a`, `1 * a → a` | Identity |
| Mul zero | `0 * a → 0`, `a * 0 → 0` | Absorbing |
| Sub zero | `a - 0 → a` | Identity |
| Sub self | `a - a → 0` | Structural equality only |
| Add neg | `a + (-a) → 0` | Structural equality only |
| Double neg | `-(-a) → a` | Involution |

### 2.3 Denylist (Explicitly Forbidden)

| Category | Examples | Why Forbidden |
|----------|----------|---------------|
| Division cancel | `a/a → 1` | Domain assumption (`a ≠ 0`) |
| Power zero | `a^0 → 1` | `0^0` undefined |
| Rationalization | `1/(1+√2) → ...` | Non-conservative |
| Expansion | `(a+b)^n → ...` | Term explosion |
| Polynomial ops | `gcd_*`, `poly_*` | Out of scope |
| General simplify | `simplify()`, `apply_rules_loop` | Bypass |

### 2.4 Non-goals (What PreSimplify Safe Does NOT Do)

> [!IMPORTANT]
> PRs attempting to add these capabilities to `presimplify_safe` should be rejected.

- **No rationalization**: Does not conjugate or clear radicals in denominators
- **No domain assumptions**: Does not assume `x ≠ 0` for cancellations like `x/x → 1`
- **No new canonical forms**: Only eliminates structural noise (`+0`, `*1`, `-a+a`)
- **No term expansion**: Does not expand `(a+b)^n` or distribute multiplication
- **No general simplification**: Does not call into the main simplifier pipeline

---

## 3. Enforcement

### 3.1 Lint: `lint_limit_presimplify.sh`

**HARD FAIL** if `presimplify.rs` contains:

```bash
DENY_PATTERNS=(
  "rationalize"
  "expand_with_stats"
  "expand::"
  "multinomial"
  "gcd_"
  "poly_"
  "simplify_with_stats"
  "apply_rules_loop"
  "domain_assumption"
  "Simplifier"
  "::simplify("
)
```

**Also enforced**: no local `is_zero`/`is_one` (use `crate::helpers::*`).

### 3.2 Contract Tests

| ID | Test Case | Contract Verified |
|----|-----------|-------------------|
| T1 | `(x-x)/x` → 0 with safe | Improvement |
| T2 | `x/x` → 1 (by limit rule, not presimplify) | No domain assumption |
| T3 | `1/(1+sqrt(2))` → residual | No rationalization |
| T4 | `x^2/x^2` same result off/safe | Stability |
| T5 | `0*x/x` → 0 | Mul zero works |
| T6 | Standard polynomial limit | No regression |
| T7 | `sqrt(x)+1` → ∞ | No expand irrationals |
| T8 | `((x+0)+0)/x` → 1 | Nested transforms |

File: `crates/cas_cli/tests/presimplify_contract_tests.rs`

---

## 4. Usage

### CLI

```bash
# Default (off): maximum conservatism
expli limit "x^2/x" --to infinity

# With safe pre-simplification
expli limit "(x-x)/x" --to infinity --presimplify=safe
```

### REPL

```
> limit x^2/x
lim_{x→+∞} = ∞

> limit (x-x)/x, x, infinity, safe
lim_{x→+∞} = 0
```

---

## 5. Evolution Path

Future enhancements should follow the same pattern:

1. **Define contract** (allowlist + denylist)
2. **Single entry point** (auditable)
3. **Add lint** (prevent regression)
4. **Add contract tests** (freeze behavior)
5. **Document in POLICY** (this file)

Candidates for similar treatment:
- Limits V2 (x→a point limits)
- Rationalization layer
- Algebraic cancellation (`x/x`, `(x²-1)/(x-1)`)

---

## 6. PR Checklist

Before merging changes to `limits/presimplify.rs`:

- [ ] **Allowlist check**: Is the new transform in the allowlist table (§2.2)?
- [ ] **Domain neutral**: Does it assume any domain constraints (e.g., `x ≠ 0`)?
  - If yes → **REJECT** or move to a different mode
- [ ] **Contract test**: Is there a new contract test if behavior changes?
- [ ] **Lint passes**: Does `./scripts/lint_limit_presimplify.sh` pass?
- [ ] **Denylist untouched**: Was the lint denylist modified to "make it pass"?
  - If yes → **REJECT** (unless documented escalation)
- [ ] **LIMITS_POLICY.md updated**: Is this doc updated if allowlist/denylist changes?

