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

---

## N-ary Views Contract (`AddView`/`MulView`) (Added 2025-12)

### Purpose

`AddView` and `MulView` provide **shape-independent** iteration over Add/Mul chains.
They are the canonical API for flattening additive/multiplicative expressions.

### Canonical Implementation

**Location**: `crates/cas_engine/src/nary.rs`

```rust
// Shape-independent sum traversal with signed terms
pub fn add_terms_no_sign(ctx, root) -> SmallVec<[ExprId; 8]>;
pub fn add_terms_signed(ctx, root) -> SmallVec<[(ExprId, Sign); 8]>;

// Shape-independent product traversal
pub fn mul_factors(ctx, root) -> SmallVec<[ExprId; 8]>;
```

### Contribution Rules

1. **Never create local `fn flatten_add*`** - use `crate::nary::add_terms_*`
2. **Never create local `fn flatten_mul*`** - use `crate::nary::mul_factors`
3. **Wrappers must call canonical** - if you need a local function, it must delegate to nary
4. **AddView/MulView are __hold-transparent** - collectors call `unwrap_hold`

### Lint Enforcement

The CI lint `scripts/lint_no_duplicate_utils.sh` will **FAIL** if:
- A file defines `fn flatten_add*` without calling `crate::nary::`
- A file defines `fn flatten_mul*` without calling `crate::nary::`

---

## Predicates Contract (Added 2025-12)

### Purpose

Expression predicates (`is_zero`, `is_one`, `is_negative`, `get_integer`) provide
consistent value extraction and type checking across the engine.

### Canonical Implementation

**Location**: `crates/cas_engine/src/helpers.rs`

```rust
pub fn is_zero(ctx, expr) -> bool;      // Number(0) literal
pub fn is_one(ctx, expr) -> bool;       // Number(1) literal  
pub fn is_negative(ctx, expr) -> bool;  // Negative number OR Neg(_) wrapper
pub fn get_integer(ctx, expr) -> Option<i64>;      // i64 extraction
pub fn get_integer_exact(ctx, expr) -> Option<BigInt>;  // BigInt + Neg handling
```

### Contribution Rules

1. **For i64 integers**: Use `crate::helpers::get_integer`
2. **For BigInt integers**: Use `crate::helpers::get_integer_exact`
3. **Wrappers must call canonical** - if you need a local function, it must delegate to helpers
4. **Extended semantics need new names** - e.g., `is_known_negative` for Mul analysis

### Lint Enforcement

The CI lint `scripts/lint_no_duplicate_utils.sh` will **FAIL** if:
- A file defines `fn is_zero/is_one/is_negative/get_integer` without using `crate::helpers::`
- Exception: struct methods with `&self` signature are allowed (different scope)

---

## Builders Contract (Added 2025-12)

### Purpose

Product builders construct multiplication trees with consistent structure.
Two canonical builders serve different use cases.

### Canonical Implementations

| Builder | Location | Shape | Use Case |
|---------|----------|-------|----------|
| `MulBuilder` | `cas_ast::views` | Right-fold `a*(b*(c*d))` | Pattern matching, rules |
| `Context::build_balanced_mul` | `cas_ast::expression` | Balanced `(a*b)*(c*d)` | Expansion, long products |

### Contribution Rules

1. **For rules/transforms**: Use `MulBuilder::new_simple()` + `push_pow()`
2. **For expansion/collection**: Use `Context::build_balanced_mul()` or `nary::build_balanced_mul()`
3. **Wrappers must call canonical** - local functions must delegate to one of the above
4. **No new `build_mul_from_factors`** - use `MulBuilder` instead

### Lint Enforcement

The CI lint `scripts/lint_no_duplicate_utils.sh` will **FAIL** if:
- A file defines `fn build_mul_from_factors*` without using `MulBuilder`
- A file defines `fn build_balanced_mul` without delegating to canonical

---

## Traversal Contract (Added 2025-12)

### Purpose

Traversal utilities for counting nodes and computing metrics.
Stack-safe (iterative) implementations prevent stack overflow on deep trees.

### Canonical Implementations

| Function | Location | Use Case |
|----------|----------|----------|
| `count_all_nodes` | `cas_ast::traversal` | Simple node count |
| `count_nodes_matching` | `cas_ast::traversal` | Count with predicate |
| `count_nodes_and_max_depth` | `cas_ast::traversal` | Count + depth metrics |

### Contribution Rules

1. **For simple counting**: Use `count_all_nodes(ctx, root)`
2. **For filtered counting**: Use `count_nodes_matching(ctx, root, |e| predicate)`
3. **For complexity metrics**: Use `count_nodes_and_max_depth(ctx, root)`
4. **Wrappers must call canonical** - local functions must delegate

### Lint Enforcement

The CI lint `scripts/lint_no_duplicate_utils.sh` will **FAIL** if:
- A file defines `fn count_nodes*` without using `cas_ast::traversal::`

---

## Error API Stability Contract (Added 2025-12)

### Purpose

Unified error handling for engine, CLI, and FFI with stable codes for UI routing.

### Canonical Types

| Type | Location | Purpose |
|------|----------|---------|
| `Span` | `cas_ast::Span` | Source location (byte offsets) |
| `ParseError` | `cas_parser::ParseError` | Parse failures with span |
| `CasError` | `cas_engine::CasError` | Unified engine error |

### Stable API Methods

All `CasError` instances provide:

| Method | Returns | Stability |
|--------|---------|-----------|
| `kind()` | `&'static str` | **STABLE** - do not change |
| `code()` | `&'static str` | **STABLE** - do not change |
| `budget_details()` | `Option<&BudgetExceeded>` | Stable |

### Error Kinds (Stable)

| Kind | Description |
|------|-------------|
| `ParseError` | Input parsing failed |
| `DomainError` | Mathematical domain violation |
| `SolverError` | Equation solving failed |
| `BudgetExceeded` | Resource limit hit |
| `NotImplemented` | Feature not available |
| `InternalError` | Bug in the engine |

### Error Codes (Stable)

All codes start with `E_`:

| Code | Meaning |
|------|---------|
| `E_PARSE` | Parse error |
| `E_DIV_ZERO` | Division by zero |
| `E_VAR_NOT_FOUND` | Variable not found |
| `E_BUDGET` | Budget exceeded |
| `E_NOT_IMPL` | Not implemented |
| `E_INTERNAL` | Internal error |

### JSON Contract

When serializing errors to JSON:

```json
{
  "ok": false,
  "kind": "DomainError",
  "code": "E_DIV_ZERO",
  "message": "division by zero",
  "span": { "start": 5, "end": 10 },
  "details": null
}
```

- `kind` and `code` are stable and must not change
- `message` is human-readable and may change
- `details` is extensible (new keys may be added)
