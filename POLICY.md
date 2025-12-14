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

## Tests

See `crates/cas_cli/tests/policy_tests.rs` for specification tests:
- `test_simplify_preserves_binomial_product`
- `test_simplify_applies_difference_of_squares_*`
- `test_expand_expands_*`
- `test_*_idempotence`
