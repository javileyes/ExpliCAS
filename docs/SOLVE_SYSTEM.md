# Linear Systems Solver (`solve_system`)

> **Build 40.48** (2×2), **Build 40.49** (3×3), **Build 40.50** (n×n Gaussian)

General solver for n×n linear equation systems:
- 2×2, 3×3: Cramer's rule
- n≥4: Gaussian elimination with partial pivoting

Uses exact `BigRational` arithmetic — no floating-point approximations.

## Syntax

```
solve_system(eq1; eq2; ...; eqn; var1; var2; ...; varn)
```

**Important**: Uses semicolons (`;`) as separators.

## Examples

### 2×2 Systems
```
> solve_system(x+y=3; x-y=1; x; y)
{ x = 2, y = 1 }
```

### 3×3 Systems
```
> solve_system(x+y+z=6; x-y=0; y+z=4; x; y; z)
{ x = 2, y = 2, z = 2 }
```

### 4×4 Systems (Gaussian)
```
> solve_system(a+b+c+d=10; a=1; b=2; c=3; a; b; c; d)
{ a = 1, b = 2, c = 3, d = 4 }
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

### 3×3 Systems
Same as 2×2, but with:
- Extract 3 coefficients per equation: `ax + by + cz + d = 0`
- Compute 3×3 determinant via cofactor expansion (Sarrus rule)
- Apply Cramer's rule with column substitution for each variable

### n×n Systems (n ≥ 4)
Uses Gaussian elimination with partial pivoting:
1. **Build** augmented matrix `[A|b]` from coefficient extraction
2. **Forward elimination** with partial pivoting (swap largest pivot)
3. **Back-substitution** to recover solution vector
4. **Degeneracy detection**: rank-deficient → Infinite/No Solution classification

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

1. **Rational coefficients**: Symbolic coefficients not supported
2. **Equalities only**: No inequality systems
3. **No parametric solutions**: Degenerate systems return error, not general solution

## Future Extensions

- Parametric solutions for dependent systems
- Integration with standard `solve` for automatic system detection

## Related

- [SOLVER_SIMPLIFY_POLICY.md](SOLVER_SIMPLIFY_POLICY.md) - General solver policies
- [POLY_GCD.md](POLY_GCD.md) - Polynomial infrastructure used for coefficient extraction
