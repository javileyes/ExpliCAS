# Polynomial GCD (Structural)

The `poly_gcd` function computes the **structural GCD** of two polynomial expressions directly in the REPL, without expanding them.

## Usage

```txt
poly_gcd(expr1, expr2)
pgcd(expr1, expr2)      # alias
```

## Example

```txt
cas> let g = (1 + 3*x1 + 5*x2 + 7*x3)^5 + 3
cas> let a = (2 + x1)^3 - 1
cas> let b = (3 + x2)^4 + 1
cas> poly_gcd(a*g, b*g)
Result: (1 + 3·x1 + 5·x2 + 7·x3)^5 + 3
```

The function detects `g` as a common factor in both `a*g` and `b*g` and returns it **without expanding** the polynomial (which could have thousands of terms).

## How It Works

### 1. HoldAll Semantics

`poly_gcd` has **HoldAll** semantics, meaning its arguments are **not simplified** before the function sees them. This preserves the multiplicative structure:

```
poly_gcd(a*g, b*g)
  → sees: Mul(a, g), Mul(b, g)
  → NOT the expanded forms
```

### 2. Factor Collection

The function collects multiplicative factors from each argument:
- `Mul(x, y)` → flatten to factors `[x, y]`
- `Pow(base, n)` with integer `n` → factor `(base, n)`
- Everything else → factor with exponent 1

### 3. AC-Canonical Key Comparison

To handle expressions that are mathematically equal but have different tree structures (e.g., different parenthesization or term order), the function uses an **AC-canonical key**:

- **A**ssociative: `(a + b) + c` = `a + (b + c)`
- **C**ommutative: `a + b` = `b + a`

The key is computed by:
1. Flattening `Add` and `Mul` chains
2. Sorting children by their hash
3. Producing a stable 64-bit hash

Two expressions with the same key are considered structurally equivalent.

### 4. Hold Protection

The result is wrapped in `hold()` internally during simplification to prevent subsequent rules (like Binomial Expansion) from expanding it. The wrapper is removed before displaying to the user.

## Comparison with Integer GCD

| Function | Operands | Result | Example |
|----------|----------|--------|---------|
| `gcd(a, b)` | Integers | Integer | `gcd(12, 18) = 6` |
| `poly_gcd(p, q)` | Expressions | Expression | `poly_gcd(x*g, y*g) = g` |

## When to Use

- **Factored polynomials**: When expressions are products of factors
- **Avoiding expansion**: When expanding would create too many terms
- **Symbolic GCD**: When you need the structural common factor

## Limitations

- **Structural only**: Detects factors that appear explicitly as multiplicands
- **Not algebraic GCD**: Does not factor polynomials to find hidden common factors
- For algebraic GCD of expanded polynomials, use the Zippel algorithm (see `ZIPPEL_GCD.md`)

## Automatic Cancellation

When you subtract `g` from `poly_gcd(a*g, b*g)`, the system will automatically simplify to `0`:

```txt
cas> let g = (1 + 3*x1 + 5*x2 + 7*x3)^7 + 3
cas> poly_gcd(a*g, b*g) - g
Result: 0
```

This works through the `AnnihilationRule` which detects that `__hold(g) - g = 0` even when the expressions have different structural representations.

### Binomial Expansion Trade-off

For binomials with small exponents (2, 3, or 4), the system may expand the expression outside the `__hold` wrapper, causing the cancellation to fail:

```txt
cas> let g = (x + y)^3 + 1
cas> poly_gcd(2*g, 3*g) - g
Result: __hold((x + y)³ + 1) - x³ - 3x²y - 3xy² - y³ - 1  # Not simplified to 0
```

For exponents > 4, the binomial is NOT expanded, and cancellation works:

```txt
cas> let g = (x + y)^5 + 1
cas> poly_gcd(2*g, 3*g) - g
Result: 0
```

## Implementation Files

- `crates/cas_engine/src/rules/algebra/poly_gcd.rs` - Rule implementation
- `crates/cas_engine/src/rules/polynomial.rs` - `AnnihilationRule` for `X - X = 0` detection
- `crates/cas_engine/src/engine.rs` - HoldAll semantics and unwrap logic
