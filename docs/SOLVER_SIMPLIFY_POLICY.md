# Solver Simplify Policy

This document defines the policy for expression simplification during equation solving. The key insight is that **solver simplification must preserve solution sets**, which is stricter than general evaluation simplification.

## Core Principle

> For `solve`, only simplify with transformations that are **global equivalences**. Leave conditional transformations as **controlled solver techniques** (with assumptions and validation).

## SimplifyPurpose Enum

```rust
pub enum SimplifyPurpose {
    /// Standard evaluation - can use Generic/Assume mode freely
    Eval,
    /// Solving equations - must preserve solution equivalence
    Solve,
}
```

## Rule Classification

### Type A: Global Equivalence (Safe for Solve)

Rewrites that preserve solutions for all domain assignments:
- Normalization: commutative/associative reordering
- `x + 0 → x`, `x * 1 → x`
- `x - 0 → x`, `x / 1 → x`
- Expand/factor (when reversible, no divisions introduced)
- Collecting like terms

✅ **Allowed in solver pre-pass**

### Type B: Conditional Equivalence (Needs Assumptions)

Rewrites valid under conditions:
- `x / x → 1` (requires `x ≠ 0`)
- `ln(exp(x)) → x` (requires appropriate domain)
- `sqrt(x²) → |x|` or `→ x` with assumptions

⚠️ **NOT as pre-simplify**. Only as controlled solver techniques with:
1. Registered assumption
2. Solution validation (ideally)

### Type C: Domain Pruning / Case Collapse (Dangerous)

Rewrites that fundamentally change solution structure:
- `0^x → 0` (loses `x > 0` as condition)
- `abs(x) → x` (loses `x ≥ 0`)
- `x/x → 1` applied silently (loses `x ≠ 0`)

🚫 **Prohibited in solver pre-pass** even in Generic/Assume mode

## Current Rule Classification

| Rule | Type | Solver Pre-pass |
|------|------|-----------------|
| `x + 0 → x` | A | ✅ Allow |
| `x * 1 → x` | A | ✅ Allow |
| `x^1 → x` | A | ✅ Allow |
| `x^0 → 1` | B | ⚠️ Only if x≠0 proven |
| `0^x → 0` | C | 🚫 Block |
| `1^x → 1` | A | ✅ Allow |
| `x/x → 1` | B | ⚠️ Only with assumption |
| `0 * x → 0` | C | 🚫 Block (loses x defined) |

## Implementation

### Phase 1: Minimal Fix (Current)
- Add `solver_safe: bool` flag to rules
- Block `0^x → 0` when called from solver context
- Propagate solver context through simplifier

### Phase 2: Full Architecture
- `SimplifyPurpose` enum in `SimplifyOptions`
- Rule trait method: `fn allowed_for(&self, purpose: SimplifyPurpose) -> bool`
- Solver creates temporary simplifier with `Solve` purpose

### Phase 3: Solution Validation
- Post-solve substitution check
- Verify solutions satisfy original equation
- Verify domain conditions (definedness)

## Related Policies

- `DISPLAY_POLICY.md` - Display transform rules
- `CONST_FOLD_POLICY.md` - Constant folding behavior
- `ASSUMPTIONS_POLICY.md` - Assumption handling
