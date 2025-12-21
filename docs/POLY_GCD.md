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

