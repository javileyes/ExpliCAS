# Polynomial GCD Functions

## Quick Reference

| Command | Mode | Use When | Speed |
|---------|------|----------|-------|
| `poly_gcd(a, b)` | Structural | Expressions share **visible** factors | ⚡ Fast |
| `poly_gcd(a, b, auto)` | Auto | Let engine choose best method | ⚡→🐢 |
| `poly_gcd(a, b, exact)` | Algebraic | Need **true** GCD over ℚ | 🐢 Slower |
| `poly_gcd(a, b, modp)` | Modular | Large polynomials, verification | ⚡⚡ Fastest |

---

# Unified poly_gcd API

```txt
poly_gcd(a, b [, mode] [, preset])
pgcd(a, b [, mode] [, preset])    # alias
```

## Modes

| Mode | Aliases | Description |
|------|---------|-------------|
| (none) | — | Structural: visible factors only |
| `auto` | — | Structural → exact → modp (auto-select) |
| `exact` | `rational`, `algebraic`, `q` | Force exact GCD over ℚ[x] |
| `modp` | `fast`, `zippel`, `mod_p` | Force modular GCD (𝔽p) |

## Examples

### Structural (default)
Detects **visible** multiplicative factors without expansion:

```txt
cas> let g = (x+1)^5 + 3
cas> let a = (y+2)^3
cas> let b = (z+3)^4
cas> poly_gcd(a*g, b*g)
Result: (1 + x)^5 + 3      # g detected structurally
```

### Auto Mode
Tries structural first, then exact, falls back to modp if too large:

```txt
cas> poly_gcd(x^2-1, x-1, auto)
Result: x - 1              # exact used (small poly)

cas> poly_gcd(huge_a*g, huge_b*g, auto)
[poly_gcd:auto] Exact exceeded budget, falling back to modp
Result: ...                # modp used (large poly)
```

### Exact Mode
Algebraic GCD over rational coefficients:

```txt
cas> poly_gcd(x^2-1, x^2-2*x+1, exact)
Result: x - 1              # (x-1)(x+1) ∩ (x-1)² = x-1

cas> poly_gcd(6*x^2, 9*x, exact)
Result: x                  # content normalized
```

### Modp Mode
Fast modular GCD for large polynomials (probabilistic):

```txt
cas> let g = (1+3*x1+5*x2+7*x3)^7 + 3
cas> let a = (2+x1)^3 - 1
cas> let b = (3+x2)^4 + 1
cas> poly_gcd(a*g, b*g, modp)
[poly_gcd_modp] Zippel GCD: 800ms
Result: ...                # GCD computed mod p
```

---

# How Auto Mode Works

```
1. STRUCTURAL (HoldAll)
   → If visible factors found: return immediately ✅

2. EXACT (ℚ[x]) if within budget:
   - vars ≤ 5
   - terms ≤ 2000
   - degree ≤ 30
   → If succeeds: return exact result ✅

3. MODP (𝔽p) fallback
   → Warning: "probabilistic"
   → Return modular result
```

---

# Legacy Functions (Still Available)

| Function | Description |
|----------|-------------|
| `poly_gcd_exact(a, b)` | Force exact mode |
| `poly_gcd_modp(a, b [, main_var] [, preset])` | Force modp with full control |
| `poly_eq_modp(a, b)` | Fast equality check mod p |

---

# Using expand() with poly_gcd

When verifying polynomial identities with large polynomials, use `expand()` inside `poly_gcd`:

```txt
cas> let g = (1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 + 15*x7)^7 + 3
cas> let a = (1 + x1)^3 - 1
cas> let b = (1 + x2)^4 + 1
cas> poly_gcd(expand(a*g), expand(b*g), modp) - expand(g)
Result: 0
```

**How it works:**
1. `expand(...)` returns `__hold(polynomial)` to prevent simplifier explosion
2. `pre_evaluate_for_gcd()` evaluates `expand()` before passing to GCD algorithms
3. `PolySubModpRule` handles `__hold(P) - __hold(Q)` in polynomial domain

---

# Polynomial Arithmetic on __hold

The engine automatically handles arithmetic between `__hold`-wrapped polynomials:

```txt
__hold(P) - __hold(Q) → 0    (if equal mod p, up to scalar)
```

This enables verification patterns like:

```txt
cas> poly_gcd(expand(a*g), expand(b*g), modp) - expand(g)
Result: 0   # GCD matches g
```

**Key features:**
- Only activates when at least one operand is `__hold` (doesn't affect normal arithmetic)
- Normalizes polynomials to monic form before comparison
- Computes in `MultiPolyModP` domain (fast, mod-p arithmetic)

---

## Implementation Files

| File | Description |
|------|-------------|
| `rules/algebra/poly_gcd.rs` | Unified `poly_gcd` rule + structural algorithm |
| `rules/algebra/gcd_exact.rs` | Exact GCD over ℚ |
| `rules/algebra/gcd_modp.rs` | Modular GCD (Zippel) |
| `rules/algebra/poly_arith_modp.rs` | `PolySubModpRule` for __hold arithmetic |
| `gcd_zippel_modp.rs` | Zippel algorithm + ZippelPreset |
| `poly_modp_conv.rs` | Expr ↔ MultiPolyModP conversion |
| `expand.rs` | Full polynomial expansion |

---

# GCD Router Unification (V2.14.36) ★★★

> **CRÍTICO**: Arquitectura unificada para GCD que diferencia entre uso **interactivo** (REPL) y uso **interno** (simplificador de fracciones).

## Problema Original

Antes de la unificación, existían dos caminos de GCD:

1. **`poly_gcd(a, b)`** en REPL → podía usar Zippel/modp (probabilístico)
2. **`SimplifyFractionRule`** → duplicaba lógica sin soundness labels

**Riesgos**:
- El simplificador de fracciones podía devolver resultados probabilísticos sin advertencia
- Stack overflow en fracciones complejas como `((x+y)^10)/((x+y)^9)`
- Código duplicado entre `gcd()` y `poly_gcd()`

## Solución: GcdGoal Enum

```rust
/// Objetivo del cálculo GCD - determina qué métodos son seguros
pub enum GcdGoal {
    /// Usuario invoca gcd() en REPL - permite todo el pipeline
    UserPolyGcd,
    /// Uso interno para cancelar fracciones - solo métodos exactos
    CancelFraction,
}
```

### Pipeline por Goal

```
┌─────────────────────────────────────────────────────────────┐
│                      GCD Router                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  UserPolyGcd (REPL):                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                   │
│  │Structural│───▶│  Exact  │───▶│  Modp   │ ✅ Allowed       │
│  └─────────┘    └─────────┘    └─────────┘                   │
│                                                              │
│  CancelFraction (Internal):                                  │
│  ┌─────────┐    ┌─────────┐                                  │
│  │Structural│───▶│  Exact  │───▶ return gcd=1 ⛔ Modp blocked│
│  └─────────┘    └─────────┘                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Ejemplos de Uso

### 1. Comando `gcd()` en REPL (unificado)

```txt
cas> gcd(12, 18)
6                                       # Enteros: algoritmo Euclidiano

cas> gcd(x^2-1, x-1)
x - 1                                   # Polinomios: auto-dispatch a poly_gcd

cas> gcd(6*x^2 + 12*x, 9*x)
3·x                                     # Content GCD + variable factor
```

### 2. Simplificación automática de fracciones

```txt
cas> ((x+y)^10)/((x+y)^9)
Steps:
1. Cancel: P^10/P^9 → P^1  [Cancel Same-Base Powers]
Result: x + y

cas> ((x+y)^10)/((x+y)^10)
Steps:
1. Cancel: P^10/P^10 → 1  [Cancel Same-Base Powers]
Result: 1

cas> ((x+y)^9)/((x+y)^10)
Steps:
1. Cancel: P^9/P^10 → 1/P  [Cancel Same-Base Powers]
Result: 1/(x + y)
```

### 3. Exponentes negativos

```txt
cas> ((x+y)^(-2))/((x+y)^(-5))
Steps:
1. Cancel: P^-2/P^-5 → P^3  [Cancel Same-Base Powers]
Result: (x + y)³
```

---

## CancelPowersDivisionRule (Pre-Order Shallow)

Regla ligera que intercepta `P^m / P^n` **antes** de las reglas pesadas de fracciones.

### Características

| Aspecto | Detalle |
|---------|---------|
| **Comparación de bases** | `compare_expr` (estructural, no poly_relation) |
| **Casos manejados** | `m = n`, `m > n`, `m < n`, negativos |
| **Stack depth** | O(1) - no recursión |
| **Posición en pipeline** | PRE-ORDER (antes de `SimplifyFractionRule`) |

### Código Resumido

```rust
define_rule!(
    CancelPowersDivisionRule,
    "Cancel Same-Base Powers",
    |ctx, expr, parent_ctx| {
        // Match Div(Pow(base1, m), Pow(base2, n))
        let (num, den) = as_div(ctx, expr)?;
        let (base1, exp1) = as_pow(ctx, num)?;
        let (base2, exp2) = as_pow(ctx, den)?;
        
        // STRUCTURAL comparison (not poly_relation)
        if compare_expr(ctx, base1, base2) != Ordering::Equal {
            return None;
        }
        
        let m = as_i64(ctx, exp1)?;
        let n = as_i64(ctx, exp2)?;
        let diff = m - n;
        
        // Build result: 1, P, P^k, 1/P, or 1/P^k
        let result = match diff {
            0 => ctx.num(1),
            1 => base1,
            -1 => ctx.add(Expr::Div(ctx.num(1), base1)),
            d if d > 0 => ctx.add(Expr::Pow(base1, ctx.num(d))),
            d => {
                let pow = ctx.add(Expr::Pow(base1, ctx.num(-d)));
                ctx.add(Expr::Div(ctx.num(1), pow))
            }
        };
        
        Some(Rewrite::new(result)
            .desc(format!("Cancel: P^{}/P^{} → ...", m, n)))
    }
);
```

---

## API Interna: compute_poly_gcd_unified

Función central del router:

```rust
pub fn compute_poly_gcd_unified(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    goal: GcdGoal,        // ← Determina qué métodos son seguros
    mode: GcdMode,        // ← Structural, Exact, Modp, Auto
    preset: ZippelPreset,
) -> GcdResult
```

### Comportamiento por Goal

| Goal | Structural | Exact | Modp |
|------|------------|-------|------|
| `UserPolyGcd` | ✅ | ✅ | ✅ |
| `CancelFraction` | ✅ | ✅ | ⛔ → gcd=1 |

---

## Shallow GCD Utilities (disponibles para futuro uso)

Para casos donde se necesita cancelación ultra-rápida sin recursión:

```rust
/// GCD shallow para fracciones - O(1) stack depth
pub fn gcd_shallow_for_fraction(
    ctx: &mut Context, 
    num: ExprId, 
    den: ExprId
) -> (ExprId, String)

/// Comparación estructural 1-2 niveles
fn expr_equal_shallow(ctx: &Context, a: ExprId, b: ExprId) -> bool
```

---

## Identity Neutral Bug Fix (relacionado)

Durante la implementación se descubrió un bug donde `e + 0` activaba auto-expand pero `e` solo no.

**Root cause**: `auto_expand_scan.rs` contaba literales `0` como "otros términos".

**Fix**: Filtrar `Number(0)` de `other_terms`:

```rust
let other_terms: Vec<ExprId> = other_terms
    .into_iter()
    .filter(|t| !matches!(ctx.get(*t), Expr::Number(n) if n.is_zero()))
    .collect();
```

---

## Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| `anti_catastrophe_tests` | 21 | ✅ |
| `property_tests` | 19 | ✅ |
| `poly_gcd_unified_tests` | 16 | ✅ |

### Tests específicos añadidos

```rust
#[test] fn test_power_cancel_equal_exponents()     // P^n/P^n → 1
#[test] fn test_power_cancel_smaller_numerator()   // P^9/P^10 → 1/P
#[test] fn test_power_cancel_zero_numerator_exp()  // P^0/P^9 → 1/P^9
#[test] fn test_power_cancel_negative_exponents()  // P^-2/P^-5 → P^3
#[test] fn test_identity_neutral_add_zero()        // e+0 == e
```

---

## Soundness Labels (PR-3 - TODO)

Para futuras versiones, añadir etiquetas de soundness:

```rust
pub enum SoundnessLabel {
    Equivalence,  // Structural/Exact: transformación exacta
    Heuristic,    // Modp: probabilístico (correcto con alta probabilidad)
}
```
