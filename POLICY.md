# Distribution Policy A+

## Contract Summary

| Function | Behavior |
|----------|----------|
| **simplify()** | Does NOT expand binomial×binomial. Only applies structural identity reductions (e.g., difference of squares). Avoids complexity explosion. |
| **expand()** | Aggressively distributes and expands. Uses budgets to avoid explosion. Goal is expanded polynomial form. |

## Examples

| Input | `simplify()` | `expand()` |
|-------|--------------|------------|
| `(x+1)*(x+2)` | `(x+1)*(x+2)` ❌ | `x² + 3x + 2` ✅ |
| `(x-1)*(x+1)` | `x² - 1` ✅ | `x² - 1` ✅ |
| `(x+y)*(x-y)` | `x² - y²` ✅ | `x² - y²` ✅ |
| `(x+1)^3` | `(x+1)^3` ❌ | `x³ + 3x² + 3x + 1` ✅ |
| `3*(x+2)` | `3x + 6` ✅ | `3x + 6` ✅ |

Legend: ✅ = reduces/expands, ❌ = preserves structure

## Implementation Details

### Rules

- **DistributeRule** (polynomial.rs): Guards `is_binomial(l) && is_binomial(r)` to prevent binomial×binomial expansion
- **DifferenceOfSquaresRule**: Detects `(a-b)(a+b)` conjugate pairs → `a² - b²` (structural reduction, applies in both)
- **ExpandRule** / **BinomialExpansionRule**: Only triggered by `expand()` wrapper

### Phase Masks

| Rule | Phase |
|------|-------|
| DistributeRule | TRANSFORM |
| ExpandRule | TRANSFORM |
| DifferenceOfSquaresRule | CORE |

## Auto-expand Mode (★ 2025-12)

Auto-expand provides **intelligent expansion** that detects cancellation contexts.

### Three-Tier Expansion System

| Mode | Behavior | Use Case |
|------|----------|----------|
| `simplify()` | Preserves `(x+1)^n` | Default, solver-friendly |
| `autoexpand on` | Expands only in **cancellation contexts** | Derivative limits, identities |
| `expand()` | Aggressively expands | Explicit polynomial form |

### Intelligent Context Detection

Auto-expand marks **context nodes** (Div/Sub) where expansion is likely to lead to cancellation:

```
Pattern 1: Div(Sub(Pow(Add(..), n), _), _) — Difference quotient
Pattern 2: Sub(Pow(Add(..), n), polynomial) — Sub cancellation
```

**Key insight**: Marking the **context** rather than individual `Pow` nodes is more robust against rewrites that change ExprIds.

| Expression | Standard | Auto |
|------------|----------|------|
| `(x+1)^3` | `(x+1)^3` ❌ | `(x+1)^3` ❌ (no context) |
| `((x+h)^3 - x^3)/h` | Unchanged | `3*x^2 + 3*h*x + h^2` ✅ |
| `(x+1)^2 - (x^2+2x+1)` | Unchanged | `0` ✅ (zero-shortcut) |

### Zero-Shortcut (Phase 2)

`AutoExpandSubCancelRule` proves cancellation via MultiPoly comparison:

1. Convert `Pow(Add(..), n)` → `MultiPoly` via repeated multiplication
2. Convert other side → `MultiPoly`
3. If `P - Q = 0` → return `0` immediately (NO AST expansion)

### Solve Mode Firewall

`ContextMode::Solve` blocks all auto-expansion for solver-friendly forms:
- Scanner skips marking contexts in Solve mode
- Rule guards check `context_mode != Solve`

### Budget Limits

Auto-expand only expands expressions that meet **all** criteria:

```
max_pow_exp: 4          # (x+1)^4 ok, (x+1)^5 not
max_base_terms: 4       # Binomials, trinomials, tetranomials
max_generated_terms: 300 # Prevents explosion
max_vars: 4             # Maximum variables in base
```

### CLI Usage

```text
> autoexpand
Auto-expand: off
  (use 'autoexpand on|off' to change)

> autoexpand on
Auto-expand: on
  Budget: max_exp=4, max_terms=4, max_result=300, max_vars=4
  Cheap polynomial powers will be expanded automatically.
```

### Library Usage

```rust
use cas_engine::options::EvalOptions;
use cas_engine::phase::ExpandPolicy;

// Create options with auto-expand enabled
let opts = EvalOptions {
    expand_policy: ExpandPolicy::Auto,
    ..Default::default()
};

// Use simplify_with_options to propagate expand_policy
let simplify_opts = opts.to_simplify_options();
let (result, steps) = simplifier.simplify_with_options(expr, simplify_opts);
```

### When to Use

| Scenario | Recommended |
|----------|-------------|
| Proving identities: `(1+i)^2 = 2i` | ✅ Auto-expand |
| Normalizing for comparison | ✅ Auto-expand |
| Factor preservation | ❌ Keep Off |
| Solve mode | ❌ Keep Off |
| GCD computations | ❌ Keep Off |

## Tests

See `crates/cas_cli/tests/policy_tests.rs` for specification tests:
- `test_simplify_preserves_binomial_product`
- `test_simplify_applies_difference_of_squares_*`
- `test_expand_expands_*`
- `test_*_idempotence`

## Clippy Allow Policy

### Crate-Level Allows
- **Prohibited**: `#![allow(...)]` at crate root requires explicit approval
- Currently: **0** crate-level allows in `cas_engine`

### Local Allows
- `#[allow(...)]` only on specific items (function/struct) with justifying comment
- Format: `#[allow(clippy::lint_name)] // why this is necessary`
- If technical debt: add `// TODO(#issue): refactor to eliminate`

### Current Exceptions (5 total)
| Lint | Location | Reason |
|------|----------|--------|
| `arc_with_non_send_sync` | profile_cache.rs | Arc for shared ownership, not threading |
| `too_many_arguments` ×4 | inverse_trig.rs, gcd_zippel_modp.rs, step.rs | Math algorithms with distinct params |

### Audit Target
Run `make lint-allowlist` to list current local allows.

## Hold Contract (`__hold()`)

### Purpose

`__hold(expr)` is an internal wrapper that blocks expansive/structural rules but is **transparent to basic algebra**.

### Semantics

| Behavior | Description |
|----------|-------------|
| **Blocks** | Autoexpand, distribute, factor-undo rules |
| **Transparent to** | AddView, MulView, cancellation, combine-like-terms |
| **MUST strip before** | User-facing output (Display, JSON, FFI) |

### Canonical Implementation

**Single source of truth**: `cas_ast::hold` module

```rust
// In cas_ast/src/hold.rs
pub fn is_hold(ctx: &Context, id: ExprId) -> bool;
pub fn unwrap_hold(ctx: &Context, id: ExprId) -> ExprId;
pub fn strip_all_holds(ctx: &mut Context, id: ExprId) -> ExprId;
```

**Do NOT duplicate** these functions elsewhere. Use `crate::strip_all_holds` (re-export in engine.rs).

### Contribution Rules

1. **Never duplicate strip_hold** - use `cas_ast::hold::strip_all_holds`
2. **Never return __hold to users** - all output boundaries must strip
3. **Views unwrap __hold** - AddView/MulView call `unwrap_hold` when collecting
4. **HoldAll functions skip simplification** - poly_gcd, pgcd, __hold

### Current Usage

| Location | Purpose |
|----------|---------|
| `multinomial_expand.rs` | Prevent O(n²) post-expand traversal |
| `factoring.rs::FactorRule` | Prevent DifferenceOfSquaresRule undo |
| `gcd_modp.rs` | Preserve GCD structure |
| `poly_arith_modp.rs` | Detect `__hold(P) - __hold(Q) = 0` |

