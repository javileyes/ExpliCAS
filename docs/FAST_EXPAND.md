# Fast Multinomial Expansion

This document describes the optimized multinomial expansion algorithm implemented for the `expand()` function.

## Overview

The `expand()` function now includes a **fast-path for linear polynomials** raised to a power:
```
(c0 + c1*x1 + c2*x2 + ... + cn*xn)^exp
```

This algorithm runs in **O(output_terms)** instead of O(terms²) for repeated multiplication.

## Performance

| Input | Terms | Time (release) |
|-------|-------|----------------|
| `expand((1+x)^10)` | 11 | ~1ms |
| `expand((1+3*x1+5*x2+7*x3)^4)` | 35 | ~12ms |
| `expand((1+3*x1+...+15*x7)^7)` | 3432 | **0.5s** (was 3+ min) |

## Limits (Budget)

To prevent runaway computation, the fast-path has these limits:

```rust
MultinomialExpandBudget {
    max_exp: 100,            // Maximum exponent
    max_base_terms: 16,      // Maximum terms in base
    max_vars: 8,             // Maximum distinct variables
    max_output_terms: 100_000, // Maximum result terms (real constraint)
}
```

The real constraint is `max_output_terms`. For a binomial `(a+b)^100`, there are only 101 terms, so it expands instantly. The limit kicks in for multinomials with many terms in the base.

## Algorithm

1. **Pattern Detection**: Check if base is linear sum with integer exponent
2. **Estimate Output**: Calculate C(n+k-1, k-1) term count
3. **Precompute Tables**: Factorials and coefficient powers
4. **Enumerate Compositions**: All ways to partition `n` into `k` parts
5. **Accumulate Terms**: Use HashMap to combine like terms
6. **Emit Result**: Build balanced Add tree wrapped in `__hold()`

## The `__hold()` Barrier

Large expanded expressions are wrapped in `__hold(...)` to prevent the simplifier from traversing the AST:

```rust
let held = ctx.add(Expr::Function("__hold".to_string(), vec![expanded]));
```

`__hold` is **transparent**:
- **Display**: Invisible - `__hold(1+x)` displays as `1+x`
- **poly_gcd**: Stripped via `strip_hold()` before factor collection
- **poly_gcd_exact**: Stripped before converting to MultiPoly
- **eval boundary**: `unwrap_hold_top()` removes wrapper from final result

This allows `expand()` results to work seamlessly with other operations.


## Example

```bash
# Fast multinomial expansion
cargo run -p cas_cli --release -- eval-json "expand((1+x)^10)"
# → 1 + x^10 + 10·x + 10·x^9 + 45·x^2 + 45·x^8 + 120·x^3 + 120·x^7 + 210·x^4 + 210·x^6 + 252·x^5

# Exceeds budget (exp=20 > max_exp=12), falls back to no expansion
cargo run -p cas_cli --release -- eval-json "expand((x+1)^20)"
# → (1 + x)^20
```

To expand larger exponents, increase `max_exp` in `MultinomialExpandBudget::default()`.

---

## Default-Simplify: SmallMultinomialExpansionRule (v2.15.58)

Starting from v2.15.58, a **separate rule** auto-expands small multinomials during
**default simplification** (no `expand()` call needed):

```
(a + b + c)^2   →   a² + 2·a·b + 2·a·c + b² + 2·b·c + c²
(x + y + z)^3   →   x³ + 3·x²·y + 3·x·y² + ...  (10 terms)
```

> [!IMPORTANT]
> **UX asymmetry (deliberate):** Multinomials (k≥3) auto-expand in default mode,
> but binomials like `(a+b)^4` do **not** — `BinomialExpansionRule` requires
> `expand_mode`. Binomials have compact closed form; multinomials don't.
> If future UX feedback changes this, add an `n ≤ threshold` path to
> `BinomialExpansionRule` that fires in default mode.


### When Does It Fire?

`SmallMultinomialExpansionRule` fires in **default mode** when ALL guards pass:

| Guard | Value | Rationale |
|-------|-------|-----------|
| `k` (base terms) | ≥ 3 | k=2 stays on `BinomialExpansionRule` |
| `k` (base terms) | ≤ 6 | Prevents large-base blow-up |
| `n` (exponent) | ≤ 4 | Keeps output manageable |
| `pred_terms` | ≤ 35 | Pre-estimated output terms |
| `base_nodes` | ≤ 25 | Rejects complex bases (sin(x+y+z)^4) |
| `output_nodes` (post) | ≤ 350 | Final safety net after expansion |

### Opaque Atoms (`parse_linear_atom`)

The multinomial algorithm treats each term as `coefficient × atom`. Atoms include:

| Accepted | Example |
|----------|---------|
| Variables | `x`, `y`, `z` |
| Constants | `π`, `e` |
| Functions | `sin(x)`, `ln(y)` |
| Integer-exponent Pow | `x²`, `a^3` |

| Rejected | Example | Reason |
|----------|--------|--------|
| Fractional-exponent Pow | `2^(1/2)` = √2 | Interferes with rationalization |

### Performance (Measured)

| Input | k | n | pred_terms | output_nodes |
|-------|---|---|------------|--------------|
| `(a+b+c)^2` | 3 | 2 | 6 | 24 |
| `(a+b+c)^3` | 3 | 3 | 10 | 68 |
| `(a+b+c+d)^2` | 4 | 2 | 10 | 52 |
| `(a+b+c+d)^4` | 4 | 4 | 35 | 302 |

### Difference from `expand()` Fast-Path

| Aspect | `expand()` Fast-Path | `SmallMultinomialExpansionRule` |
|--------|---------------------|-------------------------------|
| Trigger | `expand((...)^n)` call | Default simplification |
| Max exponent | 100 | 4 |
| Max base terms | 16 | 6 |
| Output limit | 100,000 terms | 35 terms + 350 nodes |
| Budget mode | Uses global budget | `budget_exempt` (own guards) |

---

## Files

- [`multinomial_expand.rs`](../crates/cas_engine/src/multinomial_expand.rs) - Core algorithm + `parse_linear_atom`
- [`expansion.rs`](../crates/cas_engine/src/rules/polynomial/expansion.rs) - `SmallMultinomialExpansionRule`
- [`expand.rs`](../crates/cas_engine/src/expand.rs) - `expand()` integration point
