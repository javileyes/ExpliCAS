# Linear Systems Solver (`solve_system`)

> **Added in Build 40.48**

Specialized solver for 2×2 linear equation systems using Cramer's rule with exact rational arithmetic.

## Syntax

```
solve_system(eq1; eq2; var1; var2)
```

**Important**: Uses semicolons (`;`) as separators to avoid conflicts with expression parsing.

## Examples

### Unique Solution
```
> solve_system(x+y=3; x-y=1; x; y)
{ x = 2, y = 1 }

> solve_system(2*x+3*y=7; x-y=1; x; y)
{ x = 2, y = 1 }
```

### Variable Order
The output order follows the variable order you specify:
```
> solve_system(x+y=3; x-y=1; y; x)
{ y = 1, x = 2 }
```

### Error Cases

**Non-linear system:**
```
> solve_system(x*y=1; x=2; x; y)
Error in equation 1: non-linear term: degree > 1 in the system
solve_system() only handles linear equations.
```

**Degenerate system (infinite or no solutions):**
```
> solve_system(x+y=2; 2*x+2*y=4; x; y)
determinant is 0; system has no unique solution
The system may have infinitely many solutions or none.
```

**Inequality instead of equality:**
```
> solve_system(x+y<3; x-y=1; x; y)
solve_system(): only '=' equations supported
Inequalities (<, >, <=, >=, !=) are not supported.
```

## Algorithm

1. **Parse** each equation using `cas_parser`
2. **Normalize** to form `lhs - rhs = 0`
3. **Convert** to `MultiPoly` for coefficient extraction
4. **Validate** linearity (total degree ≤ 1)
5. **Extract** coefficients: `a₁x + b₁y + c₁ = 0` and `a₂x + b₂y + c₂ = 0`
6. **Compute determinant**: `det = a₁b₂ - a₂b₁`
7. **Apply Cramer's rule**:
   - `x = (d₁b₂ - b₁d₂) / det` where `d = -c`
   - `y = (a₁d₂ - d₁a₂) / det`

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

1. **2×2 only**: No support for 3×3 or larger systems (yet)
2. **Rational coefficients**: Symbolic coefficients not supported
3. **Equalities only**: No inequality systems
4. **No parametric solutions**: Degenerate systems return error, not general solution

## Future Extensions

- 3×3 systems via Gaussian elimination or extended Cramer's rule
- Parametric solutions for dependent systems
- Integration with standard `solve` for automatic system detection

## Related

- [SOLVER_SIMPLIFY_POLICY.md](SOLVER_SIMPLIFY_POLICY.md) - General solver policies
- [POLY_GCD.md](POLY_GCD.md) - Polynomial infrastructure used for coefficient extraction
