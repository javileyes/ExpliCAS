# Solver Rule Safety Checklist

Quick reference for implementing `SolveSafety` labels on existing rules.

## Classification

- **A (Always Safe)**: Safe in solver pre-pass
- **B (Definability)**: Requires ≠0 conditions, use in tactics with validation
- **C (Analytic)**: Requires positivity/ranges, use only in Assume + validation

---

## Rules by Module

### `arithmetic.rs` ✅ All Type A
- `AddZeroRule` → A
- `MulOneRule` → A  
- `MulZeroRule` → A (but see note¹)
- `SubSameRule` → A
- `DivOneRule` → A
- `DivSameRule` → B (requires ≠0)
- `NegNegRule` → A
- `CombineConstantsRule` → A

### `canonicalization.rs` ✅ All Type A
- `CanonicalizeAddRule` → A
- `CanonicalizeMulRule` → A
- `NormalizeSignsRule` → A
- `CollectLikeTermsRule` → A

### `exponents.rs` ⚠️ Mixed
- `ProductPowerRule` → A (same base)
- `ProductSameExponentRule` → A
- `PowerPowerRule` → C (unsafe for non-integer exp)
- `EvaluatePowerRule` → A (literals only)
- `IdentityPowerRule`:
  - `x^1 → x` → A
  - `x^0 → 1` → B (requires x≠0)
  - `1^x → 1` → A
  - `0^x → 0` → C ⛔ (requires x>0) **ALREADY FIXED**
- `NegativeBasePowerRule` → A (integer exp only)
- `PowerProductRule` → A
- `PowerQuotientRule` → A

### `logarithms.rs` ⚠️ Mixed
- `LogOfOneRule` → A
- `LogOfBaseRule` → A
- `LogOfPowerRule` → C (requires base>0, arg>0)
- `LogExpRule` (`ln(exp(x))→x`) → A (RealOnly contract)
- `ExpLogRule` (`exp(ln(x))→x`) → C (requires x>0)
- `LogExpansionRule` (`ln(xy)→ln(x)+ln(y)`) → C (requires x,y>0)
- `LogContractionRule` → C (inverse of expansion)

### `algebra/fractions.rs` ⚠️ Mixed
- `CancelCommonFactorsRule` → B (requires factor≠0)
- `SimplifyFractionRule` → B (requires denom≠0)
- `AddFractionsRule` → A
- `DivZeroRule` (`0/d→0`) → B (requires d≠0)

### `trigonometry/` ⚠️ Mixed
- Basic identities (`sin²+cos²=1`) → A
- `TanToSinCosRule` → A
- Double/triple angle → A

### `inverse_trig.rs` ⛔ Type C
- `AsinSinRule` → C (requires range)
- `AcosCosRule` → C (requires range)
- `AtanTanRule` → C (requires range)
- All inverse compositions → C

### `functions.rs` ✅ Mostly Type A
- `AbsOfNegRule` → A
- `AbsOfAbsRule` → A
- `SqrtSquareRule` → A* (introduces |x|)

### `grouping.rs` ✅ Type A
- All → A (just restructuring)

---

## Implementation Quick Guide

### Option 1: Disable List (Fast)
Add to `ContextMode::Solve` in `engine.rs`:

```rust
ContextMode::Solve => {
    // Already disabled
    s.disabled_rules.insert("Simplify Square Root of Square".to_string());
    s.disabled_rules.insert("Simplify Odd Half-Integer Power".to_string());
    
    // Type B - Definability (add these)
    s.disabled_rules.insert("Cancel Common Factors".to_string());
    s.disabled_rules.insert("Simplify Fraction".to_string());
    
    // Type C - Analytic (add these)
    s.disabled_rules.insert("Log of Power".to_string());
    s.disabled_rules.insert("Exp of Log".to_string());
    s.disabled_rules.insert("Log Expansion".to_string());
    s.disabled_rules.insert("Log Contraction".to_string());
    // 0^x already handled in rule itself
}
```

### Option 2: SolveSafety Enum (Clean)
Add to rule trait:

```rust
pub enum SolveSafety {
    Always,                    // Type A
    NeedsDefinability,         // Type B  
    NeedsAnalytic,             // Type C
}

trait Rule {
    fn solve_safety(&self) -> SolveSafety { SolveSafety::Always }
    // ... existing methods
}
```

---

## Notes

¹ `MulZeroRule` (`0*x→0`): Technically safe, but can hide "x undefined" in equations. Consider B for strict mode.

² Rules that introduce `abs()` are Type A mathematically but can complicate isolation heuristics.
