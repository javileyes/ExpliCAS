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
| `IntrinsicCondition(class)` | Condition inherited from input AST | `exp(ln(x)) → x` (x>0 from ln) |
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
    - Generic: Always + IntrinsicCondition + Definability
    - Assume: Always + IntrinsicCondition + Definability + Analytic
```

---

## Marked Rules (17 total)

### Definability (9)
| Rule | File | Reason |
|------|------|--------|
| `CancelPowersDivisionRule` | algebra/fractions/gcd_cancel.rs | base≠0 |
| `CancelIdenticalFractionRule` | algebra/fractions/gcd_cancel.rs | denom≠0 |
| `CancelPowerFractionRule` | algebra/fractions/gcd_cancel.rs | denom≠0 |
| `SimplifyFractionRule` | algebra/fractions/gcd_cancel.rs | denom≠0 |
| `CancelCommonFactorsRule` | algebra/fractions/cancel_rules_factor.rs | factor≠0 |
| `QuotientOfPowersRule` | algebra/fractions/rationalize.rs | base≠0 |
| `IdentityPowerRule` | exponents/simplification.rs | x^0→1 needs x≠0 |
| `MulZeroRule` | arithmetic.rs | hides undefined |
| `DivZeroRule` | arithmetic.rs | 0/d→0 needs d≠0 |

### Analytic (7)
| Rule | File | Reason |
|------|------|--------|
| `LogExpansionRule` | logarithms/properties.rs | x,y>0 |
| `SplitLogExponentsRule` | logarithms/inverse.rs | x>0 |
| `LogInversePowerRule` | logarithms/inverse.rs | range |
| `SqrtConjugateCollapseRule` | algebra/fractions/cancel_rules.rs | other ≥ 0 |
| `PowerPowerRule` | exponents/power_rules.rs | non-integer exp |
| `HyperbolicCompositionRule` | hyperbolic.rs | range |
| `TrigInverseExpansionRule` | trig_inverse_expansion.rs | range |

### IntrinsicCondition/Analytic (1)
| Rule | File | Reason |
|------|------|--------|
| `ExponentialLogRule` | logarithms/inverse.rs | x>0 inherited from ln(x) |

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
- `algebra/fractions/` - any cancellation
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
| `Strict` | Never auto-assume → returns `Conditional` or `Residual` |
| `Generic` | May auto-accept `Definability`, returns guards for `Analytic` |
| `Assume` | May collapse cases, but recommend showing all (more honest/educational) |

---

### Data Structures

#### 1. Unified `ConditionPredicate`

Extend `AssumptionKey` as a logical object, not just event:

```rust
pub enum ConditionPredicate {
    // Existing (from AssumptionKey)
    NonZero(ExprId),
    Positive(ExprId),
    NonNegative(ExprId),
    Defined(ExprId),
    InvTrigPrincipalRange(ExprId),
    
    // V2.0 Phase 2: specialized equality (cheap, no general algebra)
    EqZero(ExprId),
    NeZero(ExprId),
    EqOne(ExprId),
    NeOne(ExprId),
    
    // V2.0 Phase 3+: general equality (optional, later)
    // Eq(ExprId, ExprId),
    // Ne(ExprId, ExprId),
}
```

#### 2. `ConditionSet`

Conjunction of predicates with stable ordering for snapshots:

```rust
pub struct ConditionSet {
    predicates: Vec<ConditionPredicate>,  // sorted, deduplicated
}
```

#### 3. `SolutionSet::Conditional`

```rust
pub enum SolutionSet {
    // ... existing variants ...
    Conditional(Vec<Case>),
}

pub struct Case {
    when: ConditionSet,
    then: SolutionSet,
}
```

#### 4. `SolveResult` (Phase 1 approach)

```rust
pub struct SolveResult {
    pub solutions: SolutionSet,       // may include Conditional
    pub residual: Option<ExprId>,     // unsolved portion (solve(...) call)
}
```

---

### Invariants (extend V1.3.7)

- **No conditional branch without explicit guard record** — every `Case` must have traceable `when`
- **No branch explosion without budget** — limit case splits to prevent combinatorial explosion

---

### Incremental Implementation Path

#### Phase 1: Conditional without else-branch

When a technique requires `cond` and it's `Unknown`:
- `solutions = Conditional([Case{when:cond, then:sol}])`
- `residual = Some(original_solve_call)`

#### Phase 2: Case splits for cheap constants

Only split on `EqZero`, `EqOne`, `NeZero`, `NeOne`:
- `a^x = a` → 3 cases (a=0, a=1, otherwise)
- Very educational, controlled branching

#### Phase 3: Condition simplification (3 cheap rules)

1. **Contradictions**: `EqZero(a)` ∧ `NeZero(a)` → eliminate case
2. **Subsumption**: if `when = True` and more specific cases exist → keep both or prefer specific
3. **Proven/Disproven**: if `prove_nonzero(a) = Proven` → remove `NeZero(a)` from guards (redundant)

---

### Example: `a^x = a`

```
x ∈ ℝ solutions:
  • if a = 1: AllReals      (1^x = 1 ∀x)
  • if a = 0: (0, ∞)        (0^x = 0 only for x > 0)
  • otherwise: {1}          (a^1 = a for a ≠ 0, a ≠ 1)

Unresolved: none (all cases covered)
```

---

### Non-Goals (V2.0)

- Full SAT-solver integration
- Arbitrary Boolean logic in conditions
- Automatic periodic trig solutions (infinite branches)
- Complex-domain branch tracking
- `Eq(expr, expr)` general — start with `EqZero`/`EqOne` only
