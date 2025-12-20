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

This is automatically unwrapped at the eval boundary, so users see clean output.

## Files

- [`multinomial_expand.rs`](../crates/cas_engine/src/multinomial_expand.rs) - Core algorithm
- [`expand.rs`](../crates/cas_engine/src/expand.rs) - Integration point

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
