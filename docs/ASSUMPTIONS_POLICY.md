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

### Phase 1 (PR-A1): Collector Infra
- Add `AssumptionCollector` type
- Collect from `Rewrite.domain_assumption`
- Dedup by fingerprint

### Phase 2 (PR-A2): Reporting Axis
- Add `AssumptionReporting` to semantics
- Wire to CLI/REPL

### Phase 3 (PR-A3): Structured Keys
- Replace `domain_assumption: Option<&str>` with `Assumption` struct
- All rules use structured keys

### Phase 4 (Future): Explicit Assumptions
- User-declared constraints: `assume x > 0`
- Verification: "used NonZero(x), but user said Positive(x) ✓"

## No-Goals (v1)

- No step/node IDs in assumptions (too much API surface)
- No user-declared assumptions yet (Phase 4)
- No intervals or numeric constraints (future)

## Hotspots

Rules that emit assumptions:
- `GcdCancelRule` (fractions.rs)
- `DivZeroRule` (arithmetic.rs)
- `AddInverseRule` (arithmetic.rs)
- `AnnihilationRule` (polynomial.rs)
- `CombineSameDenominatorFractionsRule` (fractions.rs)
- `CombineLikeTermsRule` (polynomial.rs)
- InvTrig composition rules (trigonometry.rs)

## Test Contracts

1. **Dedup**: `x/x + x/x` → assumptions.len() == 1, count == 2
2. **Strict empty**: DomainMode::Strict → assumptions empty
3. **Generic collects**: DomainMode::Generic → assumptions populated
4. **Off hides**: reporting=Off → assumptions not in JSON
5. **Stable order**: Same input → same assumption order
