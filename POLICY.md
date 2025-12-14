# Distribution Policy A+

## Summary

| Operation | `simplify()` | `expand()` |
|-----------|--------------|------------|
| `(x+1)(x+2)` | ❌ Keep factored | ✅ → `x² + 3x + 2` |
| `(x-1)(x+1)` | ✅ → `x² - 1` (diff of squares) | ✅ → `x² - 1` |
| `a(b+c)` | ⚠️ Only if reduces complexity | ✅ → `ab + ac` |

## Rationale

1. **Factored form is often preferred** - More compact, reveals structure
2. **Expanding can explode complexity** - `(a+b)^10` → 11 terms
3. **User controls expansion** - Explicit `expand()` when needed

## Rules

### `simplify()` (default)
- Does NOT expand `(a+b)(c+d)` products
- DOES apply reducing patterns:
  - Difference of squares: `(a-b)(a+b)` → `a² - b²`
  - Perfect squares: `(a+b)²` → `a² + 2ab + b²` (educational, optional)
- DOES simplify `0*x → 0`, `1*x → x`, etc.

### `expand()` (explicit)
- Distributes all products: `a(b+c)` → `ab + ac`
- Expands powers: `(a+b)^n` using binomial theorem
- Always followed by `simplify()` for cleanup

## Implementation

- `ConservativeExpandRule`: Only expands if complexity doesn't increase
- `ExpandRule`: Responds to `expand()` function wrapper
- `DifferenceOfSquaresRule`: Detects conjugate pairs, applies reduction

## Tests
```rust
// These are verified in torture_tests.rs and integration tests
simplify((x+1)(x+2))    → (x+1)(x+2)     // NOT expanded
simplify((x-1)(x+1))    → x² - 1         // Diff of squares applied
expand((x+1)(x+2))      → x² + 3x + 2    // Explicit expansion
```
