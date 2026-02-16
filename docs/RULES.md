# Rules Reference

Rule inventory and ownership for the CAS engine.

> **Legend**  
> - **Phase**: CORE (safe local), TRANSFORM (expansion/distribution), RATIONALIZE (surd cleanup), POST (final cleanup)  
> - **Churn Risk**: LOW (stable), MEDIUM (may undo other rules), HIGH (can loop)

---

## Canonicalization & Rational Arithmetic Family

| Rule | File | Name String | PhaseMask | Intent | Churn Risk |
|------|------|-------------|-----------|--------|------------|
| `CanonicalizeAddRule` | `canonicalization.rs` | "Canonicalize Addition" | CORE | Sort Add terms, flatten | LOW |
| `CanonicalizeMulRule` | `canonicalization.rs` | "Canonicalize Multiplication" | CORE | Sort Mul factors, flatten | LOW |
| `CanonicalizeNegationRule` | `canonicalization.rs` | "Canonicalize Negation" | CORE | Sub(a,b) → Add(a,Neg(b)) | LOW |
| `CanonicalizeRationalDivRule` | `rational_canonicalization.rs` | "Rational Division" | CORE | Div(p,q) → Number(p/q) | LOW |
| `CanonicalizeNestedPowRule` | `rational_canonicalization.rs` | "Fold Nested Powers" | CORE | Pow(Pow(b,k),r) → Pow(b,k*r), domain-safe | LOW |

### Equation-Level Cancel Primitive (NOT a simplifier rule)

| Primitive | File | Intent |
|-----------|------|--------|
| `cancel_common_additive_terms` | `cancel_common_terms.rs` | Cancel shared terms across LHS/RHS equation pair |

> **Architectural note**: Additive term cancellation operates on two expression trees
> (LHS, RHS) as a *relational* operation. It cannot be a single-expression simplifier
> rule because `CanonicalizeNegationRule` converts `Sub→Add(Neg)` before any
> Sub-targeting rule fires. Called from `solve_core.rs` pre-solve pipeline.

### Registration Order Contract

```
canonicalization → rational_canonicalization
```

> ⚠️ **"Sub is NOT stable"**: Any rule depending on Sub nodes must either:
> (a) fire before canonicalization, (b) also match Add(x, Neg(y)), or
> (c) be an equation-level operation in the solver pipeline.

---

## Distribution & Expansion Family

| Rule | File | Name String | PhaseMask | Intent | Churn Risk |
|------|------|-------------|-----------|--------|------------|
| `DistributeRule` | `polynomial/` | "Distributive Property" | CORE\|POST (default) | Complex distribution with guards for binomials, fraction GCD | LOW |
| `DistributeRule` | `algebra/distribution.rs` | "Distributive Property (Simple)" | TRANSFORM | Simple a*(b+c) distribution, no guards | MEDIUM |
| `ExpandRule` | `algebra/distribution.rs` | "Expand Polynomial" | TRANSFORM | Handles `expand(expr)` function call | LOW |
| `ConservativeExpandRule` | `algebra/distribution.rs` | "Conservative Expand" | TRANSFORM | Implicit expansion only if ≤ node count | LOW |
| `BinomialExpansionRule` | `polynomial/expansion.rs` | "Binomial Expansion" | CORE\|POST | (a+b)^n expansion via binomial theorem. **Requires expand_mode.** | MEDIUM |
| `SmallMultinomialExpansionRule` | `polynomial/expansion.rs` | "Small Multinomial Expansion" | CORE\|POST | (a+b+c+...)^n for k∈[3,6], n∈[2,4]. **Default mode.** `budget_exempt`. | LOW |
| `ExpandSmallBinomialPowRule` | `polynomial/expansion_normalize.rs` | "Expand Small Binomial Pow" | CORE\|POST | Atom-based binomial expansion. Opt-in via `autoexpand_binomials`. | MEDIUM |
| `DifferenceOfSquaresRule` | `algebra/factoring.rs` | "Difference of Squares (Product to Difference)" | CORE\|POST | (a-b)(a+b) → a²-b² | LOW |

### Registration Order Contract (Expansion)

```
BinomialExpansionRule → SmallMultinomialExpansionRule → ExpandSmallBinomialPowRule
```

> **Guard boundaries**: `BinomialExpansionRule` handles k=2 (expand_mode only).
> `SmallMultinomialExpansionRule` handles k≥3 (default mode, strict guards).
> `ExpandSmallBinomialPowRule` is opt-in via `autoexpand_binomials` flag.
> No overlap between the three.

### SmallMultinomialExpansionRule Guards

| Guard | Value | Stage |
|-------|-------|-------|
| `n` (exponent) | [2, 4] | Pre-expansion |
| `k` (base terms) | [3, 6] | Pre-expansion |
| `pred_terms` | ≤ 35 | Pre-expansion |
| `base_nodes` | ≤ 25 | Pre-expansion |
| `output_nodes` | ≤ 350 | Post-expansion |

> Uses `budget_exempt` because these guards are stricter than the global
> anti-worsen budget. See [BUDGET_POLICY.md](BUDGET_POLICY.md) for policy.

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
