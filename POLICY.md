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

Auto-expand provides a middle ground between conservative `simplify()` and aggressive `expand()`.

### Three-Tier Expansion System

| Mode | Behavior | Use Case |
|------|----------|----------|
| `simplify()` | Preserves `(x+1)^n` | Default, solver-friendly |
| `autoexpand on` | Expands within budget | Identity tests, normalization |
| `expand()` | Aggressively expands | Explicit polynomial form |

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
