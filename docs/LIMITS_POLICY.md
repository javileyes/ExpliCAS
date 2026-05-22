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
// crates/cas_math/src/limits_support.rs
pub fn presimplify_safe_for_limit(ctx, expr) -> ExprId
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

**HARD FAIL** if the isolated `presimplify_safe_for_limit` region contains:

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

### 5.1 Relationship To Calculus Strategy

Limits are one vertical slice of the broader calculus engine strategy:

- [CALCULUS_ENGINE_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/CALCULUS_ENGINE_STRATEGY.md)

That broader strategy does not weaken this policy. It makes this policy the
model for calculus work:

- conservative first
- explicit domain and infinity assumptions
- single auditable entry points for risky pre-processing
- tests before promotion
- unsupported residuals instead of speculative answers

When an auto-improvement cycle chooses `calculus` and touches limits, it should
follow this file first, then use the calculus strategy to decide how the limit
work feeds simplification, equivalence, domains, and didactic traces.

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

### 5.2 Finite-Point Residual Boundary

Inline eval syntax may accept finite-point notation such as
`limit(ln(x), x, -1)`, but this is only a residual contract until Limits V2 is
defined and tested beyond the narrow exception below.

Current contract:

- accepted finite points render as `limit(expr, var, point)` when unresolved
- expressions that do not depend on the limit variable may evaluate directly,
  preserving their own implicit domain requirements
- the identity variable limit may evaluate directly to the finite point when
  that point does not depend on the limit variable
- polynomial expressions may evaluate at numeric rational finite points because
  real polynomials are total on the real line and require no continuity,
  branch, side, or domain assumption
- rational polynomial expressions may evaluate at numeric rational finite
  points only when the denominator evaluates explicitly to a nonzero value at
  that point; zero denominators, including removable-looking holes, remain
  residual until a dedicated finite-point continuity/cancellation policy exists
- `exp(p(x))`, `sin(p(x))`, `cos(p(x))`, `sinh(p(x))`, `cosh(p(x))`,
  `tanh(p(x))`, `atan(p(x))`/`arctan(p(x))`, `asinh(p(x))`, `cbrt(p(x))`,
  and `abs(p(x))` may evaluate at numeric rational finite points when `p(x)`
  is polynomial in the limit variable, because these functions are continuous
  and total on the real line; only exact special values such as a zero
  argument, exact rational cube roots, or exact rational absolute values may
  collapse to rational constants in this local rule
- `sqrt(p(x))`, `ln(p(x))`, `log2(p(x))`, and `log10(p(x))` may evaluate at
  numeric rational finite points only when `p(x)` is polynomial in the limit
  variable and `p(a)` is strictly positive; zero and negative argument values
  remain residual to avoid endpoint, side, branch, or domain-path assumptions
- arithmetic compositions of already-resolved safe finite sublimits may
  evaluate through `+`, `-`, `*`, unary negation, and division only when the
  computed denominator is either an explicit nonzero numeric value or is proven
  structurally positive by the existing sign prover; if any sublimit is
  unresolved or a denominator is not proven safe, the whole finite limit remains
  residual; after all operands are safe, this local rule may fold exact rational
  arithmetic and structural identities such as `g - g -> 0`, `0 + g -> g`,
  `1*g -> g`, and `g/g -> 1` only after the denominator has been proven
  nonzero
- total-real continuous unary compositions already in the finite allowlist
  (`exp`, `sin`, `cos`, `sinh`, `cosh`, `tanh`, `atan`/`arctan`, `asinh`,
  `cbrt`, and `abs`) may evaluate when their argument has already resolved to a
  safe finite sublimit; this does not add support for discontinuous functions
  (`sign`, `floor`, `ceil`) or domain-partial outer functions (`ln`, `sqrt`,
  `asin`/`acos`, `atanh`, `acosh`)
- `ln(g(x))`, `log2(g(x))`, `log10(g(x))`, and `sqrt(g(x))` may evaluate when
  `g(x)` has already resolved to a safe finite sublimit that is explicitly
  numeric positive or is proven strictly positive by the existing sign prover;
  zero, negative, or unproven positive sublimits remain residual, including
  endpoint-looking cases such as `sqrt(abs(x))` at `x -> 0`
- after those finite composition checks succeed, local exact presentation folds
  may reduce `ln(exp(g))` to `g`, `exp(ln(g))` to `g` only when `g` is
  explicitly or structurally proven strictly positive, and `abs(g)`/`abs(-g)`
  to `g` only when `g` is explicitly or structurally proven strictly positive;
  these are finite-limit result folds, not global simplification or
  pre-simplification rules
- binary `log(b(x), g(x))` may evaluate only when `b(x)` has already resolved
  to an explicit rational finite sublimit with `b > 0` and `b != 1`, and
  `g(x)` has already resolved to a safe strictly positive finite sublimit;
  invalid base sublimits, base sublimit `1`, non-rational or unresolved base
  sublimits, zero arguments, negative arguments, or unproven positive arguments
  remain residual
- integer powers `g(x)^n` may evaluate when `g(x)` has already resolved to a
  safe finite sublimit; positive integer exponents are total over real finite
  base sublimits, while zero and negative integer exponents require the base
  sublimit to be explicitly or structurally proven nonzero so the engine does
  not promote `0^0` or division by zero; exact rational square-root sublimits
  such as `sqrt(q)` with `q > 0` may present even integer powers as `q^k` or
  `1/q^k`, while odd powers and broader root algebra remain in the explicit
  root/power form; exact rational cube-root sublimits such as `cbrt(q)` may
  present integer powers that are multiples of three as `q^k` or `1/q^k`, with
  reciprocal forms allowed only when `q != 0`; non-multiple powers remain in
  explicit root/power form
- real one-third powers, written as `(p(x))^(1/3)`, follow the same finite
  cube-root rule as `cbrt(p(x))`; this does not promote arbitrary fractional
  powers or even roots over negative values
- unresolved finite-point limits carry a warning that finite point limits are
  not supported safely yet
- unresolved finite-point residuals must not surface literal-impossible public
  requirements such as `0 ≠ 0` from a collapsed denominator witness; suppressing
  that display artifact does not prove or promote the limit, and the residual
  result plus finite-limit warning remains the contract
- no finite-point answer is invented
- no side, path, continuity, branch, or domain assumption is inferred
- the CLI subcommand `limit --to` remains scoped to `infinity` and `-infinity`
  until it has a dedicated finite-point API contract

This boundary is intentional. A future Limits V2 implementation must still
define:

- point representation and side policy
- domain/path conditions at the approach point
- residual rendering for unresolved finite limits
- contract tests before any evaluation rule is promoted

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
