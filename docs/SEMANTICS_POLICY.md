# SEMANTICS POLICY

> **Version 1.3.3** | Last updated: 2026-01-01

This document defines the semantic configuration axes that control how ExpliCAS evaluates and simplifies expressions. Each axis is orthogonal and controls a specific aspect of mathematical semantics.

## Overview

ExpliCAS uses **5 orthogonal semantic axes**:

| Axis | Controls | Values |
|------|----------|--------|
| **DomainMode** | Variable assumptions (≠0, >0, etc.) | `Strict`, `Generic`, `Assume` |
| **AssumeScope** | What can be assumed in Assume mode | `Real`, `Wildcard` |
| **ValueDomain** | Universe of constants | `RealOnly`, `ComplexEnabled` |
| **BranchPolicy** | Multi-valued function branches (ℂ only) | `Principal` |
| **InverseTrigPolicy** | Inverse∘function compositions | `Strict`, `PrincipalValue` |


---

## Axis A: DomainMode ✅ (V1.3 - ConditionClass Contract)

Controls which transformations are allowed based on **typed side conditions**.

### ConditionClass Taxonomy

Conditions required by transformations are classified into two types:

| Class | Description | Examples |
|-------|-------------|----------|
| **Definability** | Small holes at isolated points | `x ≠ 0`, `a is defined` |
| **Analytic** | Big restrictions (half-lines, ranges) | `x > 0`, `x ≥ 0`, `x ∈ [-π/2, π/2]` |

### DomainMode Gate

| Mode | Definability | Analytic | Use Case |
|------|--------------|----------|----------|
| `Strict` | Only if proven | Only if proven | Formal proofs |
| `Generic` | ✅ Accept (with warning) | ❌ Block | Educational default |
| `Assume` | ✅ Accept (with warning) | ✅ Accept (with warning) | Research, exploration |

### Canonical Examples

| Expression | Condition | Class | Strict | Generic | Assume |
|------------|-----------|-------|--------|---------|--------|
| `x/x → 1` | NonZero(x) | Definability | ❌ | ✅ | ✅ |
| `0/x → 0` | NonZero(x) | Definability | ❌ | ✅ | ✅ |
| `ln(x*y) → ln(x)+ln(y)` | Positive(x), Positive(y) | Analytic | ❌ | ❌ | ✅ |
| `exp(ln(x)) → x` | Positive(x) | Analytic | ❌ | ❌ | ✅ |
| `sqrt(x)² → x` | NonNegative(x) | Analytic | ❌ | ❌ | ✅ |
| `2/2 → 1` | — (proven) | Definability | ✅ | ✅ | ✅ |

### Implementation Details

- **Gate Functions**: `can_cancel_factor()` (Definability), `can_apply_analytic()` (Analytic)
- **Central Logic**: `DomainMode::allows_unproven(ConditionClass)`
- **Condition Types**: `AssumptionKey::class()` returns `ConditionClass`

### Affected Rules

- `SimplifyFractionRule` — uses `can_cancel_factor()`
- `DivZeroRule`, `MulZeroRule` — Definability gate
- `LogExpansionRule` — uses `can_apply_analytic()`
- `ExponentialLogRule` — uses `can_apply_analytic_with_hint()` (V1.3.1)
- `IdentityPowerRule`, `CancelCommonFactorsRule`, `QuotientOfPowersRule`

### Blocked Hints (V1.3.1)

When Generic mode blocks a transformation due to an unproven Analytic condition, the engine emits **pedagogical hints** to guide the user:

```
> exp(ln(x))
Result: e^(ln(x))
ℹ️  Blocked in Generic: requires x > 0 [Exponential-Log Inverse]
   use `domain assume` to allow analytic assumptions
```

**Key behaviors:**

| Scenario | Hint Emitted? |
|----------|---------------|
| Generic + unproven Analytic | ✅ Yes |
| Strict + unproven Analytic | ❌ No (expected behavior) |
| Assume + unproven Analytic | ❌ No (allowed with warning) |
| Any mode + proven condition | ❌ No (simplification proceeds) |

**Implementation:**

- `can_apply_analytic_with_hint(mode, proof, key, expr_id, rule)` — Rich gate with hint emission
- `BlockedHint { key, expr_id, rule, suggestion }` — Structured hint data
- Thread-local collector: `register_blocked_hint()`, `take_blocked_hints()`
- Hints are deduplicated by `(rule, AssumptionKey)`

### Transparency Invariant (V1.3.3)

> **Invariant: No assumptions without a timeline record**

When a transformation is applied under an **unproven** side condition (i.e., the condition proof is `Unknown` and the current `DomainMode` allows it), the engine **must**:

1. Attach the corresponding `AssumptionKey` to the rewrite as an `AssumptionEvent`.
2. Propagate it to the produced `Step` so the timeline shows it under **Assumptions (assumed)**.

This guarantees that any result that depends on a condition is **always traceable** in the timeline.

**Implementation:**

- `CancelDecision.assumed_keys`: SmallVec storing assumed conditions
- `CancelDecision.allow_with_keys()`: Constructor for allowed-with-assumption decisions
- `CancelDecision.assumption_events()`: Converts keys to structured events for `Rewrite`
- Propagation: `Rewrite.assumption_events` → `Step.assumption_events` (automatic in engine)

**Contract tests:**

- `step_tracks_assumed_nonzero_in_generic` — Definability (x ≠ 0)
- `step_tracks_assumed_positive_in_assume` — Analytic (x > 0)

---

## Axis A': AssumeScope ✅ (Implemented Dec 2025)

Controls **what** can be assumed when `DomainMode = Assume`. Only active in Assume mode.

### Values

| Value | Behavior | Use Case |
|-------|----------|----------|
| `Real` | Only assume proven-safe operations (e.g., `positive(y)` for log) | Default, educational |
| `Wildcard` | Allow residual output for operations requiring complex domain | Research, permissive |

### Solver Decision Table (Exponential Equations)

| Scenario | `scope=Real` | `scope=Wildcard` |
|----------|--------------|------------------|
| Base>0, RHS Unknown | Solution + `positive(rhs)` | Solution + `positive(rhs)` |
| Base<0 (proven) | Error: NeedsComplex | **Residual** + warning |

### Example

```
> semantics set assume_scope wildcard
> solve (-2)^x = 5
⚠ Requires complex logarithm — returning as residual
Result: solve((-2)^x = 5, x)
```

### Implementation

- **Classifier**: `classify_log_solve()` in `domain_guards.rs`
- **Single Source of Truth**: Used by both `strategies.rs` and `isolation.rs`
- **RAII Guard**: `SolveAssumptionsGuard` for nested solve isolation

---

## Axis B: ValueDomain ✅ (Implemented Dec 2025)

Defines the **field of default values for symbols**.

> **Design Principle**:
> `ValueDomain` decides the **field by default of symbols** (ℝ vs ℂ).
> `DomainMode` decides if transformations requiring **additional conditions** (x>0, x≠0, inverse ranges) are allowed.

### Values

| Value | Description | Symbol Default | `i` Behavior |
|-------|-------------|----------------|--------------|
| `RealOnly` | All symbols are real by definition | x ∈ ℝ | `i` = symbol (not constant) |
| `ComplexEnabled` | Symbols may be complex | x ∈ ℂ | `i² = -1` |

### Critical Behaviors

#### The `i` Constant

| Operation | RealOnly | ComplexEnabled |
|-----------|----------|----------------|
| `i * i` | `i²` + warning | `-1` |
| `i^2` | `i²` + warning | `-1` |
| `10/(3+4i)` | unchanged | rationalized |

> In `RealOnly`, using `i` emits a pedagogical warning:
> `⚠ To use complex arithmetic (i² = -1), run: semantics set value complex`

#### Log-Exp Composition

| Expression | RealOnly+Strict | RealOnly+Generic | ComplexEnabled |
|------------|-----------------|------------------|----------------|
| `ln(e^x)` | ✅ `x` | ✅ `x` | ❌ unchanged |
| `e^(ln(x))` | ❌ unchanged | ✅ `x` + warning | ✅ `x` + warning |

**Rationale**: In RealOnly, x ∈ ℝ by contract, so e^x > 0 always. In ComplexEnabled, ln is multivalued.

#### Log Power Rule

| Expression | Even Exponent | Odd Exponent |
|------------|---------------|--------------|
| `ln(x²)` | `2·ln(\|x\|)` (no assumption) | - |
| `ln(x³)` | - | `3·ln(x)` + `⚠ x > 0` |
| `ln(x⁴)` | `4·ln(\|x\|)` (no assumption) | - |

**Rationale**: Similar to `sqrt(x²) = |x|`, even powers are non-negative.

#### Log with Symbolic Base

| Expression | Strict | Generic | Assume |
|------------|--------|---------|--------|
| `log(2, 2^x)` | ✅ `x` | ✅ `x` | ✅ `x` |
| `log(b, b^x)` | ❌ unchanged | ❌ unchanged | ✅ `x` + `⚠ b > 0` |

**Rationale**: Literal bases (2, e, π) are provably positive. Symbolic bases require assumption.

#### Square Root

| Expression | RealOnly+Strict | RealOnly+Generic |
|------------|-----------------|------------------|
| `sqrt(x²)` | `\|x\|` | `\|x\|` |
| `sqrt(-1)` | `undefined` | `undefined` |

### Default

`RealOnly` — symbols are real, `i` is a warning, no complex arithmetic.


---

## Axis C: BranchPolicy (Planned PR1)

Controls how multi-valued functions are resolved. **Only applies when `ValueDomain = ComplexEnabled`**.

### Values

| Value | Behavior |
|-------|----------|
| `Principal` | Use principal branch (e.g., `log(-1) = iπ`) |

### Future Values (Non-Goals for V1)

- `AllBranches` — return set of values
- `Symbolic` — preserve multi-valued structure

---

## Axis D: InverseTrigPolicy (Planned PR4)

Controls simplification of inverse∘function compositions like `arctan(tan(x))`.

> **Important**: This is NOT the same as BranchPolicy. InverseTrigPolicy applies to inverse trig functions in ℝ, not complex branch cuts.

### Values

| Value | `arctan(tan(x))` | Warning |
|-------|------------------|---------|
| `Strict` | `arctan(tan(x))` (no change) | None |
| `PrincipalValue` | `x` | "assumed x ∈ (-π/2, π/2)" |

### Current State

The existing REPL `mode principal/strict` command currently controls this behavior but is conflated with other settings. PR4 will extract it cleanly.

---

## Rule Dependency Table

| Rule | DomainMode | ValueDomain | BranchPolicy | InverseTrigPolicy |
|------|:----------:|:-----------:|:------------:|:-----------------:|
| `SimplifyFractionRule` | ✅ | - | - | - |
| `IdentityPowerRule` | ✅ | - | - | - |
| `CancelCommonFactorsRule` | ✅ | - | - | - |
| `QuotientOfPowersRule` | ✅ | - | - | - |
| `sqrt(negative literal)` | - | ✅ | ✅ | - |
| `log(negative literal)` | - | ✅ | ✅ | - |
| `ArcTanTanRule` | - | - | - | ✅ |
| `ArcSinSinRule` | - | - | - | ✅ |
| `ArcCosCosRule` | - | - | - | ✅ |
| `i * i → -1` | - | ✅ | - | - |

---

## Strict Principle (Formal Invariant)

> **"No rewrite shall reduce the set of points where an expression may be undefined, unless definedness is proven."**

This is the "mother rule" for `DomainMode::Strict`. It justifies all gates (`prove_nonzero`, `has_undefined_risk`, etc.) and ensures that:

1. **No information loss** — If `f(x)` is undefined at `x=a`, the rewritten form must also be undefined at `x=a`
2. **Provable soundness** — Simplifications only apply when the engine can *prove* the condition (literal evaluation, algebraic identity, etc.)
3. **Conservative failure** — When in doubt, preserve the original form

### Gate Functions

| Function | Proves | Returns |
|----------|--------|---------|
| `prove_nonzero(e)` | `e ≠ 0` | `Proven`, `Refuted`, `Unknown` |
| `prove_positive(e)` | `e > 0` | `Proven`, `Refuted`, `Unknown` |
| `has_undefined_risk(e)` | `e` may be undefined | `true`, `false` |

In `Strict` mode: `Unknown` → rule **does not fire**.
In `Assume` mode: `Unknown` → rule fires with `AssumptionEvent`.
In `Generic` mode: `Unknown` → rule fires silently.

---

## Definedness Hotspots (Exhaustive List)

These are the **only** rewrite patterns that can "erase" undefined points. Each must be gated in `Strict` mode:

### 1. Zero Numerator: `0/d → 0`

| Condition | Action |
|-----------|--------|
| `d` is proven nonzero | ✅ Collapse to `0` |
| `d` may be zero | ❌ Keep `0/d` |

**Gate**: `prove_nonzero(d)` in `DivZeroRule`

### 2. Additive Inverse: `t - t → 0`, `t + (-t) → 0`

| Condition | Action |
|-----------|--------|
| `t` has no undefined risk | ✅ Collapse to `0` |
| `t` contains division/function | ❌ Keep `t - t` |

**Gate**: `has_undefined_risk(t)` in `AddInverseRule`

### 3. Zero Annihilation: `0 * e → 0`

| Condition | Action |
|-----------|--------|
| `e` has no undefined risk | ✅ Collapse to `0` |
| `e` contains division/function | ❌ Keep `0 * e` |

**Gate**: `has_undefined_risk(e)` in `AnnihilationRule`

### 4. Factor Cancellation: `f*g / f*h → g/h`

| Condition | Action |
|-----------|--------|
| Common factor proven nonzero | ✅ Cancel |
| Common factor may be zero | ❌ Keep original |

**Gate**: `prove_nonzero(gcd)` in `SimplifyFractionRule`, `CancelCommonFactorsRule`

### 5. Zero Exponent: `x^0 → 1`

| Condition | Action |
|-----------|--------|
| `x` proven nonzero | ✅ Return `1` |
| `x` may be zero (`0^0` case) | ❌ Keep `x^0` |

**Gate**: `prove_nonzero(x)` in `IdentityPowerRule`

### 6. Inverse Composition: `f⁻¹(f(x)) → x`

| Condition | Action |
|-----------|--------|
| `x` in principal range | ✅ Return `x` |
| `x` outside range | ❌ Keep composed |

**Gate**: `InverseTrigPolicy` axis + range check

---

## Contract Tests for Definedness

The following behaviors MUST hold (see `strict_definedness_contract_tests.rs`):

| # | Expression | Strict | Assume | Generic |
|---|------------|--------|--------|---------|
| 1 | `0/(x+1)` | `0/(x+1)` | `0` + assumption | `0` |
| 2 | `t-t` where `t=x/(x+1)` | `t-t` | `0` + assumption | `0` |
| 3 | `0/2` | `0` | `0` | `0` |
| 4 | `0*(x/(x+1))` | `0*(x/(x+1))` | `0` + assumption | `0` |
| 5 | `x/x` | `x/x` | `1` + assumption | `1` |

---

## Hotspots (Audit List)

### DomainMode Hotspots

- Fraction cancellation: `SimplifyFractionRule`, `CancelCommonFactorsRule`
- Power identity: `x^0`, `0^x`, `0^0`
- Root simplification: `sqrt(x^2)` → `|x|` vs `x`
- Algebraic inverse: `f^{-1}(f(x))`

### ValueDomain / BranchPolicy Hotspots

- `sqrt(negative_literal)` — first case to implement
- `log(negative_literal)` — future
- `pow(negative_base, rational_exp)` — future
- Complex trig/hyperbolic — very future

### InverseTrigPolicy Hotspots

- `arctan(tan(x))`, `arcsin(sin(x))`, `arccos(cos(x))`
- Machin-like sums of arctan
- Parent context propagation bugs (historical)

---

## JSON Schema (V1 Compatible)

New fields are **optional** and do not break existing consumers:

```json
{
  "schema_version": 1,
  "ok": true,
  "result": "1",
  "domain": { "mode": "assume" },
  "semantics": {
    "domain_mode": "assume",
    "value_domain": "real",
    "branch": "principal",
    "inv_trig": "strict"
  },
  "warnings": [
    { "code": "W_DOMAIN_ASSUMPTION", "message": "cancelled factor assumed nonzero" }
  ]
}
```

---

## Defaults (Stability Contract)

| Axis | Default | Rationale |
|------|---------|-----------|
| `domain_mode` | `Generic` | Backward compatibility |
| `value_domain` | `RealOnly` | Safe, no complex |
| `branch` | `Principal` | Standard convention |
| `inv_trig` | `Strict` | Safe, no domain assumptions |
| `const_fold` | `Off` | Defer semantic decisions |

---

## Presets (V1.1)

Presets provide **explicit, auditable shortcuts** to configure multiple axes at once. They do not change defaults and are only applied when the user invokes them explicitly.

### Design Principles

1. **Explicit, not magic** — User must invoke `semantics preset <name>`
2. **Traceable** — Apply prints diff of changes
3. **Complete** — Each preset defines all 5 axes
4. **Few and clear** — Only 4 presets, each with distinct purpose

### Available Presets

| Preset | domain | value | branch | inv_trig | const_fold | Purpose |
|--------|--------|-------|--------|----------|------------|---------|
| `default` | generic | real | principal | strict | off | Reset to engine defaults |
| `strict` | strict | real | principal | strict | off | Conservative, no assumptions |
| `complex` | generic | complex | principal | strict | safe | Enable ℂ + materialization |
| `school` | generic | real | principal | principal | off | Classroom mode |

### REPL Commands

```
semantics preset              # List available presets
semantics preset <name>       # Apply preset (prints diff)
semantics preset help <name>  # Show preset configuration
```

### Example

```
> semantics preset complex
Applied preset: complex
Changes:
  value_domain: real → complex
  const_fold:   off → safe

> sqrt(-1)
Result: i
```

### Non-Goals

- Presets do not create aliases for `semantics set`
- Presets do not introduce "auto" or heuristic behavior
- Custom/user-defined presets are not supported in V1

---

## CLI Flags (Current and Planned)

| Flag | Current | Planned |
|------|:-------:|:-------:|
| `--domain strict\|generic\|assume` | ✅ | - |
| `--value-domain real\|complex` | - | PR3 |
| `--inv-trig strict\|principal` | - | PR4 |

---

## REPL Commands

| Command | Current | After PR4 |
|---------|---------|-----------|
| `mode principal` | Changes branch + inv_trig | Changes only `inv_trig` |
| `mode strict` | Changes branch + inv_trig | Changes only `inv_trig` |
| `domain strict` | ✅ Sets DomainMode | Same |
| `domain assume` | ✅ Sets DomainMode | Same |

---

## PR Checklist

For any PR touching semantic axes:

- [ ] Update this document if adding/changing behavior
- [ ] Add contract test(s) for new axis values
- [ ] Ensure axis is reflected in JSON (if user-facing)
- [ ] Update hotspots section if adding new affected rules
- [ ] Verify defaults don't change existing behavior
- [ ] Run `make ci` to ensure no regressions

---

## Breaking Changes (V1.2)

The following behavioral changes were introduced in V1.2:

### RealOnly Contract (Dec 2025)

**Before**: `ValueDomain::RealOnly` only affected literal evaluation.
**After**: `RealOnly` now means **"all symbols are real by default"**.

| Behavior | V1.1 | V1.2 |
|----------|------|------|
| `ln(e^x)` in Strict | unchanged | `x` |
| `ln(x²)` | `2·ln(x)` + warning | `2·ln(\|x\|)` |
| `prove_positive(e^x)` | Unknown | Proven (RealOnly) |

**Migration**: Tests expecting `ln(e^x)` to remain unchanged in Strict will fail. Update expectations or use `ComplexEnabled`.

---

## Non-Goals (V1)

The following are explicitly **not** supported in V1:

- General complex logarithm (`log(z)` for arbitrary z)
- Multi-valued symbolic expressions
- Automatic domain inference from context
- Riemann surface branch tracking
- Symbolic complex conjugate/modulus

These may be considered for future versions.

---

## Related: NormalFormGoal

The **NormalFormGoal** system is a separate, orthogonal mechanism that controls which *inverse* rules are allowed to apply. It prevents rules like `DistributeRule` or `LogContractionRule` from undoing explicit user transformations (`collect()`, `expand_log()`).

Unlike the semantic axes defined above (which affect *what* the engine considers valid), NormalFormGoal affects *which rules* are allowed to apply during a specific simplification pass.

See [docs/NORMAL_FORM_GOAL.md](NORMAL_FORM_GOAL.md) for details.
