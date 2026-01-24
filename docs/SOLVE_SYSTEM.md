# Linear Systems Solver (`solve_system`)

> **Added in Build 40.48** (2x2), **Extended in Build 40.49** (3x3)

Specialized solver for 2×2 and 3×3 linear equation systems using Cramer's rule with exact rational arithmetic.

## Syntax

```
2x2: solve_system(eq1; eq2; x; y)
3x3: solve_system(eq1; eq2; eq3; x; y; z)
```

**Important**: Uses semicolons (`;`) as separators to avoid conflicts with expression parsing.

## Examples

### 2x2 Systems
```
> solve_system(x+y=3; x-y=1; x; y)
{ x = 2, y = 1 }

> solve_system(2*x+3*y=7; x-y=1; x; y)
{ x = 2, y = 1 }
```

### 3x3 Systems
```
> solve_system(x+y+z=6; x-y=0; y+z=4; x; y; z)
{ x = 2, y = 2, z = 2 }

> solve_system(x+y+z=1; 2*x+y=3; x+z=2; x; y; z)
{ x = 2, y = -1, z = 0 }
```

### Error Cases

**Non-linear system:**
```
> solve_system(x*y=1; x=2; x; y)
Error in equation 1: non-linear term: degree > 1 in the system
solve_system() only handles linear equations.
```

**Infinitely many solutions (dependent equations):**
```
> solve_system(x+y=2; 2*x+2*y=4; x; y)
System has infinitely many solutions.
The equations are dependent (same line).
```

**No solution (inconsistent equations):**
```
> solve_system(x+y=2; x+y=3; x; y)
System has no solution.
The equations are inconsistent (parallel lines).
```

**Inequality instead of equality:**
```
> solve_system(x+y<3; x-y=1; x; y)
solve_system(): only '=' equations supported
Inequalities (<, >, <=, >=, !=) are not supported.
```

## Algorithm

### 2x2 Systems
1. **Parse** each equation using `cas_parser`
2. **Normalize** to form `lhs - rhs = 0`
3. **Convert** to `MultiPoly` for coefficient extraction
4. **Validate** linearity (total degree ≤ 1)
5. **Extract** coefficients: `a₁x + b₁y + c₁ = 0` and `a₂x + b₂y + c₂ = 0`
6. **Compute determinant**: `det = a₁b₂ - a₂b₁`
7. **Apply Cramer's rule**:
   - `x = (d₁b₂ - b₁d₂) / det` where `d = -c`
   - `y = (a₁d₂ - d₁a₂) / det`

### 3x3 Systems
Same as 2x2, but with:
- Extract 3 coefficients per equation: `ax + by + cz + d = 0`
- Compute 3×3 determinant via cofactor expansion (Sarrus rule)
- Apply Cramer's rule with column substitution for each variable

## Implementation Details

### Files

| File | Purpose |
|------|---------|
| `crates/cas_cli/src/repl/commands_system.rs` | Core solver logic |
| `crates/cas_cli/src/repl/dispatch.rs` | Command routing |
| `crates/cas_cli/src/repl/init.rs` | Semicolon bypass |

### Arithmetic Precision

Uses `BigRational` from `num-rational` crate for exact rational arithmetic. No floating-point approximations.

### Linearity Check

A system is considered linear if:
- All terms have total exponent ≤ 1
- No products of variables (e.g., `x*y`)
- No powers > 1 (e.g., `x^2`)
- No transcendental functions (e.g., `sin(x)`)

## Limitations

1. **2×2 and 3×3 only**: No support for 4×4 or larger systems (yet)
2. **Rational coefficients**: Symbolic coefficients not supported
3. **Equalities only**: No inequality systems
4. **No parametric solutions**: Degenerate systems return error, not general solution

## Future Extensions

- 4×4+ systems via LU decomposition
- Parametric solutions for dependent systems
- Integration with standard `solve` for automatic system detection

## Related

- [SOLVER_SIMPLIFY_POLICY.md](SOLVER_SIMPLIFY_POLICY.md) - General solver policies
- [POLY_GCD.md](POLY_GCD.md) - Polynomial infrastructure used for coefficient extraction
