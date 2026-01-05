# Solver Simplify Policy

> **Status**: V1.3.7 Implemented ✅

This document defines the `SolveSafety` contract for the ExpliCAS solver simplification pipeline.

## Core Contract

### SimplifyPurpose Enum

| Mode | When Used | Allowed Rules |
|------|-----------|---------------|
| `Eval` | Normal evaluation | All rules |
| `SolvePrepass` | Solver pre-simplification | Only `SolveSafety::Always` |
| `SolveTactic` | Solver tactics with conditions | `Always` + `NeedsCondition` per DomainMode |

### SolveSafety Classification

| Class | Description | Example |
|-------|-------------|---------|
| `Always` | Global equivalence, never changes solution set | `a + 0 → a` |
| `NeedsCondition(Definability)` | Requires ≠0 conditions | `(xy)/x → y` (x≠0) |
| `NeedsCondition(Analytic)` | Requires sign/range conditions | `ln(xy) → ln(x)+ln(y)` (x,y>0) |
| `Never` | Never safe in solver | (reserved) |

### Filtering Rules

```
SolvePrepass:
  rule.solve_safety().safe_for_prepass()  → only Always

SolveTactic:
  rule.solve_safety().safe_for_tactic(domain_mode)
    - Strict: only Always
    - Generic: Always + Definability
    - Assume: Always + Definability + Analytic
```

---

## Marked Rules (13 total)

### Definability (6)
| Rule | File | Reason |
|------|------|--------|
| `CancelCommonFactorsRule` | fractions.rs | factor≠0 |
| `SimplifyFractionRule` | fractions.rs | denom≠0 |
| `QuotientOfPowersRule` | fractions.rs | base≠0 |
| `IdentityPowerRule` | exponents.rs | x^0→1 needs x≠0 |
| `MulZeroRule` | arithmetic.rs | hides undefined |
| `DivZeroRule` | arithmetic.rs | 0/d→0 needs d≠0 |

### Analytic (7)
| Rule | File | Reason |
|------|------|--------|
| `LogExpansionRule` | logarithms.rs | x,y>0 |
| `ExponentialLogRule` | logarithms.rs | x>0 |
| `LogInversePowerRule` | logarithms.rs | range |
| `SplitLogExponentsRule` | logarithms.rs | x>0 |
| `HyperbolicCompositionRule` | hyperbolic.rs | range |
| `TrigInverseExpansionRule` | trig_inverse_expansion.rs | range |
| `PowerPowerRule` | exponents.rs | non-integer exp |

---

## API Usage

```rust
// Solver pre-pass: safe simplification
let simplified = simplifier.simplify_for_solve(expr);

// Options for tactics
let opts = SimplifyOptions::for_solve_tactic(DomainMode::Assume);
```

---

## Maintenance Contract

**When adding a new rule:**
1. If the rule can change solution sets, add `solve_safety:` to `define_rule!`
2. Use `Definability` for ≠0 conditions
3. Use `Analytic` for sign/range/branch conditions
4. Add to the contract test in `solve_safety_contract_tests.rs`

**Modules requiring explicit `solve_safety`:**
- `fractions.rs` - any cancellation
- `logarithms.rs` - any log manipulation
- `exponents.rs` - x^0, power-power
- `trig_inverse_*.rs` - all compositions
- `hyperbolic.rs` - all compositions

---

## V2.0 Roadmap: Conditional Solutions

> **Status**: Design Phase (Not Implemented)

### Vision

The solver should be **above** `strict/generic/assume` and produce **conditional/piecewise solutions** instead of blocking or assuming silently.

### Key Insight

`strict/generic/assume` become **exploration policies**, not limitations:

| Mode | Behavior with Unproven Conditions |
|------|-----------------------------------|
| `Strict` | Returns guarded solutions explicitly |
| `Generic` | May accept definability holes, returns guards for analytic |
| `Assume` | Accepts all guards, records assumptions |

### Proposed Extensions

#### 1. `SolutionSet::Conditional`

```rust
pub enum SolutionSet {
    // ... existing variants ...
    Conditional(Vec<Case>),
}

pub struct Case {
    when: ConditionSet,  // conjunction of conditions
    then: SolutionSet,
}
```

#### 2. Extended Condition Predicates

Current `AssumptionKey` covers:
- `NonZero`, `Positive`, `NonNegative`, `Defined`, `InvTrigPrincipalRange`

Needs:
- `Eq(expr, const)` — for case splits like `a = 0`
- `Ne(expr, const)` — for clean partitions

#### 3. Example: `a^x = a`

Ideal output:
```
Case when: (a = 1) → AllReals    (1^x = 1 ∀x)
Case when: (a = 0) → (0, ∞)      (0^x = 0 only for x > 0)
Case when: True    → {1}         (a^1 = a for a ≠ 0, a ≠ 1)
```

### Incremental Implementation Path

1. **Phase 1**: `SolutionSet::Conditional` without else-branch
   - If technique requires `cond`, return "solution under `cond`" + "residual"

2. **Phase 2**: Case splits only for "cheap" constants: `a=0`, `a=1`
   - Very educational, limited branching

3. **Phase 3**: Condition merge/simplification
   - Absorb subsets, eliminate redundant cases

### Non-Goals (V2.0)

- Full SAT-solver integration
- Arbitrary Boolean logic in conditions
- Automatic periodic trig solutions (infinite branches)
- Complex-domain branch tracking

