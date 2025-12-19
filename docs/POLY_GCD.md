# Polynomial GCD Functions

## Quick Reference: When to Use Which?

| Function | Type | Use When | Speed | Example |
|----------|------|----------|-------|---------|
| `gcd(12, 18)` | Integer | Both arguments are integers | ⚡ Fast | `→ 6` |
| `poly_gcd(a*g, b*g)` | Structural | Expressions share **visible** factors | ⚡ Fast | `→ g` |
| `poly_gcd_exact(x²-1, x-1)` | Algebraic | Need **true** polynomial GCD over ℚ | 🐢 Slower | `→ x-1` |

---

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
- For algebraic GCD of expanded polynomials, use `poly_gcd_exact` (see below)

## Automatic Cancellation

When you subtract `g` from `poly_gcd(a*g, b*g)`, the system will automatically simplify to `0`:

```txt
cas> let g = (1 + 3*x1 + 5*x2 + 7*x3)^7 + 3
cas> poly_gcd(a*g, b*g) - g
Result: 0
```

This works through the `AnnihilationRule` which detects that `__hold(g) - g = 0` even when the expressions have different structural representations.

### Binomial Preservation

Binomials like `(x+1)^3` are **not expanded automatically** in Standard mode. This ensures consistent cancellation for any exponent:

```txt
cas> let g = (x + y)^3 + 1
cas> poly_gcd(2*g, 3*g) - g
Result: 0
```

To explicitly expand a binomial, use the `expand` command:

```txt
cas> expand (x+1)^3
Result: 1 + x³ + 3·x + 3·x²
```

---

# Polynomial GCD Exact (Algebraic)

The `poly_gcd_exact` function computes the **algebraic GCD** of two polynomials over ℚ[x₁,...,xₙ].

## Usage

```txt
poly_gcd_exact(expr1, expr2)
pgcdx(expr1, expr2)      # alias
```

## Examples

```txt
cas> poly_gcd_exact(x^2 - 1, x - 1)
Result: x - 1

cas> poly_gcd_exact(x^2 - 1, x^2 - 2*x + 1)
Result: x - 1

cas> poly_gcd_exact(2*x + 2*y, 4*x + 4*y)
Result: x + y

cas> poly_gcd_exact(6, 15)
Result: 1
```

## Difference from `poly_gcd`

| Function | Type | Description |
|----------|------|-------------|
| `poly_gcd(a, b)` | Structural | Finds common visible factors only |
| `poly_gcd_exact(a, b)` | Algebraic | Computes actual polynomial GCD over ℚ |

Example where they differ:

```txt
cas> poly_gcd(x^2 - 1, x - 1)
Result: 1                    # No visible common factor

cas> poly_gcd_exact(x^2 - 1, x - 1)
Result: x - 1                # Finds (x-1) as factor of (x-1)(x+1)
```

## Algorithm

1. Convert expressions to `MultiPoly` (sparse multivariate polynomial)
2. Try **univariate** Euclidean GCD if single variable
3. Try **Layer 1**: monomial + content GCD
4. Try **Layer 2**: heuristic seeds interpolation
5. Try **Layer 2.5**: tensor grid interpolation
6. Verify result with exact division
7. Normalize: primitive part + positive leading coefficient

## Normalization Contract

- **Primitive**: GCD of coefficients = 1
- **Positive leading coefficient** (in lex monomial order)
- `gcd(0, p) = normalize(p)`
- `gcd(c₁, c₂) = 1` for non-zero constants over ℚ

## When to Use

- **Expanded polynomials**: When expressions are sums, not products
- **Algebraic factorization**: When you need the true polynomial GCD
- **Fraction simplification**: Underlying algorithm for simplifying rational functions

## Budget Limits

To prevent runaway computation:
- Max 5 variables
- Max 500 terms per input
- Max 50 total degree

If limits exceeded, returns `1` with a warning.

---

## Implementation Files

- `crates/cas_engine/src/rules/algebra/poly_gcd.rs` - Structural GCD rule
- `crates/cas_engine/src/rules/algebra/gcd_exact.rs` - Algebraic GCD rule
- `crates/cas_engine/src/multipoly.rs` - MultiPoly representation and Layer 1/2/2.5
- `crates/cas_engine/src/rules/polynomial.rs` - `AnnihilationRule` and `BinomialExpansionRule`
- `crates/cas_engine/src/parent_context.rs` - `expand_mode` context for rules
