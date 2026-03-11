# Assumptions Policy

Policy document for assumption collection, deduplication, and reporting in the CAS engine.

## Overview

The CAS engine makes **mathematical assumptions** during simplification (e.g., "x ≠ 0" when cancelling x/x). These assumptions must be:

1. **Collected** - Not lost, even when rules don't fire warnings
2. **Deduplicated** - Same assumption used 3 times → count=3, not 3 warnings
3. **Stable** - JSON schema immutable for clients
4. **Auditable** - User can see what was assumed and how often

## JSON Schema (v1)

```json
{
  "result": "...",
  "assumptions": [
    {
      "kind": "nonzero",
      "expr": "x",
      "message": "Assumed x ≠ 0",
      "count": 3
    },
    {
      "kind": "defined",
      "expr": "x + 1",
      "message": "Assumed expression is defined",
      "count": 1
    }
  ]
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `kind` | string | Assumption type: `nonzero`, `positive`, `defined`, `principal_range`, `principal_branch` |
| `expr` | string | Canonical display of the expression assumed about |
| `message` | string | Human-readable explanation |
| `count` | u32 | How many times this assumption was used (dedup counter) |

### Invariants

1. `assumptions` is **omitted** if empty or reporting is `off`
2. `assumptions` array has **stable order** (sorted by kind, then expr)
3. **No duplicates** - same (kind, expr) appears once with count ≥ 1
4. Assumptions **do not affect result** - only metadata

## Assumption Kinds

| Kind | Gate | Example |
|------|------|---------|
| `nonzero` | `prove_nonzero` returns Unknown | `x/x → 1` assumes x ≠ 0 |
| `positive` | `prove_positive` returns Unknown | `√x² → x` assumes x > 0 |
| `defined` | `has_undefined_risk` returns true | `a-a → 0` when a has division |
| `principal_range` | InvTrig composition | `arctan(tan(x)) → x` assumes x ∈ (-π/2, π/2) |
| `principal_branch` | Complex multi-valued | `√-1 → i` in ComplexEnabled |

## V2.12.13: AssumptionKind Classification System

The `AssumptionKind` enum classifies assumptions for **display filtering** and **semantic accuracy**.

> [!IMPORTANT]
> **Intrinsic vs Introduced (V2.16):** The key criterion for whether a condition is allowed
> in Generic mode is its **provenance**, not its strictness (≥ vs >).
> See [POLICY_TABLES.md](./POLICY_TABLES.md) for the full decision matrix.

### The 6 Categories

| Kind | Icon | Display? | Meaning |
|------|------|----------|---------|
| `DerivedFromRequires` | - | ❌ NO | Redundant with input domain |
| `RequiresIntroduced` | ℹ️ | ✅ YES | New constraint for equivalence |
| `BranchChoice` | 🔀 | ✅ YES | Multi-valued function branch |
| `DomainExtension` | 🧿 | ✅ YES | Domain change (ℝ→ℂ) |
| `HeuristicAssumption` | ⚠️ | ✅ YES | Simplification heuristic |

### Definitions

#### 1. Requires (input) — Intrinsic Conditions
Conditions **inferred from the original expression** that are necessary for it to be defined.
These are **intrinsic operator preconditions** — they come directly from operators already
present in the AST (e.g., `ln(x)` → `x > 0`, `sqrt(x)` → `x ≥ 0`, `1/(x-1)` → `x-1 ≠ 0`).

```
Input: 1/(x-1)      → Requires: x-1 ≠ 0  (intrinsic)
Input: exp(ln(x))   → Requires: x > 0    (intrinsic, from ln)
Input: sqrt(x)^2    → Requires: x ≥ 0    (intrinsic, from sqrt)
```

#### 2. DerivedFromRequires
An assumption emitted by a rule that is **already implied** by intrinsic Requires or
previously introduced requires. **NOT displayed** to avoid redundancy.

```
Input: (x²-1)/(x-1)
Requires (input): x-1 ≠ 0

Step: Cancel (x-1)
→ Assumption "x-1 ≠ 0" is DerivedFromRequires (not shown — already intrinsic)
```

#### 3. RequiresIntroduced
A **new constraint** introduced by a step that was **not deducible** from intrinsic input
conditions. This narrows the domain of validity. **Blocked in Generic mode** per Invariant A.

```
Input: log(a·b)
Step: log(a·b) → log(a) + log(b)
→ RequiresIntroduced: a > 0, b > 0
   (the input only required a·b > 0 — the new constraint is INTRODUCED)
```

> [!NOTE]
> The distinction between `DerivedFromRequires` (intrinsic) and `RequiresIntroduced` maps
> directly to `SolveSafety::IntrinsicCondition` vs `SolveSafety::NeedsCondition` in the engine.

#### 4. BranchChoice
The engine **chose one branch** of a multi-valued function. This is an explicit choice, not a logical necessity.

```
Input: √(x²)
Step: √(x²) → x  (instead of |x|)
→ BranchChoice: "Choosing x ≥ 0 branch"
```

> **Contract**: BranchChoice is **NEVER promoted** to RequiresIntroduced. It stays as a branch indicator.

#### 5. DomainExtension
The engine **extended the domain** (typically ℝ → ℂ) to continue simplification.

```
Input: (-1)^(1/2)
Step: → i
→ DomainExtension: ℝ → ℂ
```

#### 6. HeuristicAssumption
The engine applied a **simplification heuristic** that is convenient but not strictly required for validity.

```
Step: Combined terms assuming specific form
→ HeuristicAssumption: "Applied heuristic grouping"
```

### Mathematical Contracts

#### When NO Branch/Assume/Extension appears:

> Under `Requires_input ∪ Requires_introduced`, the result is **equivalent** to the input (same value where defined).

#### When BranchChoice appears:

> The result is valid **under the stated branch condition**. The general identity may differ (e.g., `|x|` vs `x`).

#### When HeuristicAssumption appears:

> The simplification is for **convenience/aesthetics**. Equivalence may not hold generally.

### Canonical Examples

| Input | Step | Category | Display | Provenance |
|-------|------|----------|---------|------------|
| `(x²-4)/(x-2)` | Cancel | DerivedFromRequires | *(hidden)* | Intrinsic |
| `exp(ln(x))` | → x | DerivedFromRequires | *(hidden, Requires: x>0)* | Intrinsic (from `ln`) |
| `log(a·b)` | Split | RequiresIntroduced | ℹ️ `a>0, b>0` | Introduced |
| `√(x²)` | Simplify to x | BranchChoice | 🔀 `x≥0` | Intrinsic |
| `(-1)^(1/2)` | → i | DomainExtension | 🧿 `ℝ→ℂ` | — |
| `sin(arcsin(x))` | → x | BranchChoice | 🔀 Principal range | — |
| `√x · √x` | → x | DerivedFromRequires | *(hidden)* | Intrinsic |

### Rule Authoring Checklist

When writing a new rule that emits assumptions:

1. **Is the condition already intrinsic to an operator in the AST?**
   - YES → Use `.requires(ImplicitCondition::...)` and set `SolveSafety::IntrinsicCondition(...)`
   - The condition is inherited, not introduced. Allowed in Generic mode.
   - Example: `exp(ln(x)) → x` inherits `x > 0` from `ln`

2. **Does the condition come from division in input?**
   - YES → Use `AssumptionEvent::nonzero()` (default: DerivedFromRequires)

3. **Does the rule introduce a NEW constraint not in input?**
   - YES → Use `positive()` with default RequiresIntroduced
   - Set `SolveSafety::NeedsCondition(...)`. Blocked in Generic mode (Invariant A).
   - Example: `log(a·b) → log(a)+log(b)` introduces `a>0, b>0`

4. **Does the rule choose a branch?**
   - YES → Use `inv_trig_principal_range()` or `complex_principal_branch()`
   - These default to BranchChoice

5. **Is this a heuristic simplification?**
   - YES → Manually set `kind: AssumptionKind::HeuristicAssumption`

6. **Does the rule extend domain?**
   - YES → Manually set `kind: AssumptionKind::DomainExtension`

### DomainContext and Classification

The `DomainContext` struct tracks:
- `global_requires`: Inferred from input expression
- `introduced_requires`: Accumulated from steps

The `classify_assumption()` function reclassifies events:
1. BranchChoice/Heuristic/DomainExtension → **Keep as-is**
2. Condition **implied** by known requires → `DerivedFromRequires`
3. Condition **not implied** → Promote to `RequiresIntroduced`

## SolveSafety Classification (Added Feb 2026)

Rules declare their safety level for solver contexts via `SolveSafety` in `solve_safety.rs`.
This classification interacts with assumption provenance:

| Classification | Provenance | Prepass | Tactic(Generic) | Tactic(Assume) | Tactic(Strict) |
|---|---|---|---|---|---|
| `Always` | — | ✅ | ✅ | ✅ | ✅ |
| `IntrinsicCondition(class)` | Intrinsic | ⛔ | ✅ | ✅ | ⛔ |
| `NeedsCondition(Definability)` | Introduced | ⛔ | ✅ | ✅ | ⛔ |
| `NeedsCondition(Analytic)` | Introduced | ⛔ | ⛔ | ✅ | ⛔ |
| `Never` | — | ⛔ | ⛔ | ⛔ | ⛔ |

The relationship to `Provenance` is machine-queryable via `RequirementDescriptor`:

```rust
// Bridge: SolveSafety → domain vocabulary
let desc = rule.solve_safety().requirement_descriptor();
// desc.class: ConditionClass (Definability / Analytic)
// desc.provenance: Provenance (Intrinsic / Introduced)
```

See [SEMANTICS_POLICY.md § Domain Oracle Architecture](./SEMANTICS_POLICY.md#domain-oracle-architecture-v151--feb-2026) for the complete architecture.

### Three Domain Invariants

1. **Invariant A — No introduced requires in Generic.**
   A rule in Generic mode cannot add Requires that aren't already backed by intrinsic
   operator preconditions present in the input AST.

2. **Invariant B — Requires must be preserved.**
   If a simplification eliminates a node that provided a precondition (e.g., removes `ln`),
   the `Requires` must be propagated to the result.

3. **Invariant C — Equivalence under current requires.**
   A rule only fires if it produces an equivalence under the accumulated Requires,
   without inventing new assumptions.

See [POLICY_TABLES.md](./POLICY_TABLES.md) for the full decision matrix by `DomainMode`.

## Solver Assumptions (Added Dec 2025)

When solving exponential equations like `2^x = y`, the solver may emit assumptions if `DomainMode = Assume`. These are collected via a **thread-local `SolveAssumptionsGuard`** RAII pattern.

### Solver Assumption Kinds

| Kind | Context | Message Example |
|------|---------|-----------------|
| `positive` | Log of base | `Assumed base > 0 for logarithm` |
| `positive` | Log of RHS | `Assumed y > 0 for logarithm` |

### Architecture

```
┌────────────────────────────────────────────────────────┐
│             classify_log_solve()                       │
│  Single source of truth for log domain decisions      │
├────────────────────────────────────────────────────────┤
│  strategies.rs ───┐                                    │
│                   ├──► OkWithAssumptions(vec)          │
│  isolation.rs ────┘           │                        │
│                               ▼                        │
│                  SolverAssumption::to_assumption_event │
│                               │                        │
│                               ▼                        │
│                     note_assumption()                  │
│                               │                        │
│                               ▼                        │
│               SOLVE_ASSUMPTIONS (thread-local)         │
│                               │                        │
│                               ▼                        │
│          SolveAssumptionsGuard.finish()                │
│                               │                        │
│                               ▼                        │
│            EvalOutput.solver_assumptions               │
└────────────────────────────────────────────────────────┘
```

### RAII Guard Pattern

The `SolveAssumptionsGuard` handles:
- **Leak prevention**: Always clears on drop
- **Nested solve isolation**: Saves/restores previous collector
- **Thread safety**: Each solve gets its own collector

```rust
pub struct SolveAssumptionsGuard {
    previous: Option<AssumptionCollector>,  // For nested solves
    enabled: bool,
}
```

### Contract Tests

- `assume_mode_emits_positive_rhs_assumption`: `2^x = y` → `positive(y)`
- `strict_mode_no_assumptions`: No assumptions in Strict
- `assumptions_are_deduplicated`: Same assumption counted, not repeated
- `nested_solve_guards_are_isolated`: Inner solve doesn't leak to outer

## Reporting Levels

```rust
pub enum AssumptionReporting {
    Off,     // No assumptions shown
    Summary, // Deduped list at end
    Trace,   // (Future) Include step locations
}
```

### Defaults by DomainMode

| DomainMode | Default Reporting | Rationale |
|------------|------------------|-----------|
| Strict | Off | Assumptions not used |
| Generic | Summary | Educational: show what's assumed |
| Assume | Summary | Explicit mode: show everything |

## Collector Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Engine.eval()                       │
├─────────────────────────────────────────────────────┤
│  1. Create AssumptionCollector                      │
│  2. Pass to Simplifier                              │
│  3. Rules call collector.note(key, expr, message)   │
│  4. collector.finish() → Vec<AssumptionRecord>      │
│  5. Add to EvalOutput.assumptions                   │
└─────────────────────────────────────────────────────┘

AssumptionKey (for dedup):
  - NonZero { expr_fingerprint: u64 }
  - Positive { expr_fingerprint: u64 }
  - Defined { expr_fingerprint: u64 }
  - PrincipalRange { func, arg_fingerprint }
  - PrincipalBranch { func, arg_fingerprint }

Fingerprint = hash(canonical_display(expr))
```

## Anti-Cascade Contract

The following MUST NOT produce 50 separate warnings:

```
x/x + 2x/2x + 3x/3x + ... + 50x/50x
```

**Expected output** (Summary mode):
```
Result: 50
⚠ Assumptions: nonzero(x) (×50)
```

## REPL Commands

```
semantics set reporting off     # No assumptions shown
semantics set reporting summary # Default, deduped list
semantics set reporting trace   # Future: include step refs
```

## Migration Path

### Phase 1 (PR-A1): Collector Infra ✅
- `AssumptionCollector` type with dedup
- `AssumptionReporting` enum (Off/Summary/Trace)
- Wired to `SimplifyOptions` and `EvalOptions`

### Phase 2 (PR-A2): REPL Command ✅
- `semantics set assumptions off|summary|trace`
- `semantics assumptions` displays current state
- Default: `Off` (conservative)

### Phase 3 (PR-A3): Engine Wiring ✅
- `Orchestrator.simplify_pipeline` collects from `Step.assumption_events`
- `PipelineStats.assumptions` field
- `EngineWireResponse.assumptions` field
- REPL summary line in `do_simplify`

### Phase 4 (PR-A4): Structured Emission ✅

**Status**: ✅ Complete (2025-12-28)

**Migration Completed**:
1. Added `assumption_events: SmallVec<[AssumptionEvent; 1]>` to `Rewrite` struct
2. Migrated all 18 hotspot rules to emit structured `AssumptionEvent`s
3. **Removed legacy `domain_assumption` field** from `Rewrite` and `Step` (316+ instances via codemod)
4. Updated engine propagation: `rewrite.assumption_events` → `step.assumption_events`
5. Updated `eval.rs`, `orchestrator.rs`, `timeline/` module to use only `assumption_events`

**Codemod Pattern Used**: Python dry-run script with brace-counting (see `scripts/remove_domain_assumption.py`)

### Phase 5 (Future): User-Declared Assumptions
- User constraints: `assume x > 0`
- Verification: "used NonZero(x), but user said Positive(x) ✓"

## Implementation Status

| Component | Status |
|-----------|--------|
| `assumptions.rs` module | ✅ Complete |
| `AssumptionCollector` with dedup | ✅ Complete |
| `AssumptionReporting` enum | ✅ Complete |
| REPL command | ✅ Complete |
| Engine wiring | ✅ Complete |
| JSON surface | ✅ Complete |
| Contract tests (12+) | ✅ Passing |
| Legacy string parsing | ✅ Complete |
| **Structured emission from rules** | **✅ Complete** |
| **`domain_assumption` field removed** | **✅ Zero-Debt** |

## Migrated Hotspots (Completed 2025-12-28)

All 18 rules that previously emitted `domain_assumption` strings have been migrated to structured `AssumptionEvent`s:

| Rule | File | Assumption Kind | Factory Used |
|------|------|-----------------|--------------|
| `DivZeroRule` | arithmetic.rs | NonZero | `AssumptionEvent::nonzero()` |
| `SimplifyFractionRule` (GCD) | algebra/fractions/ | NonZero | `AssumptionEvent::nonzero()` |
| `AddInverseRule` | arithmetic.rs | Defined | `AssumptionEvent::defined()` |
| `IdentityPowerRule` (x^0) | exponents.rs | NonZero | `AssumptionEvent::nonzero()` |
| `IdentityPowerRule` (0^x) | exponents.rs | Positive | `AssumptionEvent::positive()` |
| `EvaluateLogRule` | logarithms.rs | Positive | `AssumptionEvent::positive()` |
| `LogExpansionRule` | logarithms.rs | Positive | `AssumptionEvent::positive()` |
| `ExponentialLogRule` | logarithms.rs | Positive | `AssumptionEvent::positive()` |
| `LogExpInverseRule` | logarithms.rs | Positive | `AssumptionEvent::positive()` |
| `CosProductTelescopingRule` | integration.rs | NonZero | `AssumptionEvent::nonzero()` |
| `LiftConjugateToSqrtRule` | algebra/fractions/ | Defined | `AssumptionEvent::defined()` |
| `PrincipalBranchInverseTrigRule` | inverse_trig.rs | InvTrigPrincipalRange | `AssumptionEvent::inv_trig_principal_range()` |
| `InverseTrigCompositionRule` (sin∘arcsin) | inverse_trig.rs | Defined | `AssumptionEvent::defined()` |
| `InverseTrigCompositionRule` (cos∘arccos) | inverse_trig.rs | Defined | `AssumptionEvent::defined()` |

## Test Contracts

1. **Dedup**: `x/x + x/x` → assumptions.len() == 1, count == 2
2. **Strict empty**: DomainMode::Strict → assumptions empty
3. **Generic collects**: DomainMode::Generic → assumptions populated
4. **Off hides**: reporting=Off → assumptions not in JSON
5. **Stable order**: Same input → same assumption order
6. **Legacy parsing**: `from_legacy_string()` correctly infers kind
7. **Options propagation**: EvalOptions → SimplifyOptions → Orchestrator
