# Rules Reference

Rule inventory and ownership for the CAS engine.

> **Legend**  
> - **Phase**: CORE (safe local), TRANSFORM (expansion/distribution), RATIONALIZE (surd cleanup), POST (final cleanup)  
> - **Churn Risk**: LOW (stable), MEDIUM (may undo other rules), HIGH (can loop)

---

## Distribution & Expansion Family

| Rule | File | Name String | PhaseMask | Intent | Churn Risk |
|------|------|-------------|-----------|--------|------------|
| `DistributeRule` | `polynomial/` | "Distributive Property" | CORE\|POST (default) | Complex distribution with guards for binomials, fraction GCD | LOW |
| `DistributeRule` | `algebra/distribution.rs` | "Distributive Property (Simple)" | TRANSFORM | Simple a*(b+c) distribution, no guards | MEDIUM |
| `ExpandRule` | `algebra/distribution.rs` | "Expand Polynomial" | TRANSFORM | Handles `expand(expr)` function call | LOW |
| `ConservativeExpandRule` | `algebra/distribution.rs` | "Conservative Expand" | TRANSFORM | Implicit expansion only if ≤ node count | LOW |
| `BinomialExpansionRule` | `polynomial/` | "Binomial Expansion" | CORE\|POST | (a+b)^n expansion via binomial theorem | MEDIUM |
| `DifferenceOfSquaresRule` | `algebra/factoring.rs` | "Difference of Squares (Product to Difference)" | CORE\|POST | (a-b)(a+b) → a²-b² | LOW |

### Semantic Duplicates

**Two DistributeRules exist** with different behaviors:

| Aspect | polynomial/ | algebra/distribution.rs |
|--------|---------------|-----------------|
| Name | "Distributive Property" | "Distributive Property (Simple)" |
| Phase | CORE\|POST | TRANSFORM only |
| Guards | Binomial detection, conjugate check, GCD check | None (simple pattern match) |
| Registered | `polynomial::register()` | NOT registered by default |

**Recommendation**: 
- The `polynomial/` version is the "source of truth" for simplify behavior
- The `algebra/distribution.rs` version is designed for explicit `expand()` phase
- Keep both, they serve different purposes

---

## Factorization Family

| Rule | File | Name String | PhaseMask | Intent | Churn Risk |
|------|------|-------------|-----------|--------|------------|
| `FactorRule` | `algebra/factoring.rs` | "Factor Polynomial" | CORE\|POST | Handles `factor(expr)` function | LOW |
| `FactorDifferenceSquaresRule` | `algebra/factoring.rs` | "Factor Difference of Squares" | **DISABLED** | a²-b² → (a-b)(a+b) | HIGH - loops with DistributeRule |
| `AutomaticFactorRule` | `algebra/factoring.rs` | "Automatic Factorization" | CORE\|POST | Auto-factor only if reduces size | MEDIUM |

---

## Rationalization Family

| Rule | File | Name String | PhaseMask | Intent | Churn Risk |
|------|------|-------------|-----------|--------|------------|
| `RationalizeDenominatorRule` | `algebra/fractions/rationalize.rs` | "Rationalize Denominator" | RATIONALIZE | General rationalization | LOW |
| `RationalizeSingleSurdRule` | `algebra/fractions/rationalize.rs` | "Rationalize Single Surd" | RATIONALIZE | Simple √n denominator | LOW |
| `RationalizeBinomialSurdRule` | `algebra/fractions/rationalize.rs` | "Rationalize Binomial Surd" | RATIONALIZE | Conjugate multiplication | LOW |
| `RationalizeProductDenominatorRule` | `algebra/fractions/rationalize.rs` | "Rationalize Product Denominator" | RATIONALIZE | k/√n cases | LOW |

---

## Phase Pipeline

```
CORE → TRANSFORM → RATIONALIZE → POST
  ↑                                 ↑
  └── Safe simplifications         └── Final cleanup
       (no expansion)                  (no expansion)
```

Rules with `TRANSFORM` only run when distribution/expansion is desired (e.g., explicit `expand()` call or expand mode).
