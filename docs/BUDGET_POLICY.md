# Anti-Explosion Budget Policy

> **Status**: 📋 Planned (not yet implemented)  
> **Tracking**: See [MAINTENANCE.md](../MAINTENANCE.md) section 10

## Overview

This document describes the planned unified budget system to prevent computational explosion across all CAS operations.

## Current State (Fragmented)

| Budget Type | Location | What it limits |
|-------------|----------|----------------|
| `ExpandBudget` | phase.rs | Auto-expand pow/terms |
| `MultinomialExpandBudget` | multinomial_expand.rs | Multinomial term count |
| `PolyBudget` | multipoly.rs | Polynomial conversion |
| `ZippelBudget` | gcd_zippel_modp.rs | GCD interpolation |
| `PhaseBudgets` | phase.rs | Rewrite iterations |

**Problem**: Each measures different things with different enforcement.

## Target Architecture

```
┌────────────────────────────────────────────────┐
│              Unified Budget System             │
├────────────────────────────────────────────────┤
│  Operation enum: SimplifyCore, Expand, GCD...  │
│  Metric enum: NodesCreated, RewriteSteps...    │
│  BudgetConfig: limits per (Operation, Metric)  │
│  BudgetScope: RAII tracking of current op      │
└────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────┐
│           3-Layer Enforcement                  │
├────────────────────────────────────────────────┤
│ A. Central: NodesCreated in Context::add       │
│ B. Hotspot: Terms/PolyOps in specific modules  │
│ C. Pre-estimation: Fail fast before work       │
└────────────────────────────────────────────────┘
```

## Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Infrastructure (`budget.rs`, `ContextStats`) | ⬜ |
| 1 | Unify error types (`BudgetExceeded`) | ⬜ |
| 2 | Simplify pipeline integration | ⬜ |
| 3 | Expand / multinomial integration | ⬜ |
| 4 | Polynomial operations integration | ⬜ |
| 5 | Zippel GCD integration | ⬜ |
| 6 | CI lint enforcement | ⬜ |

## Key Design Decisions

### 1. Backward Compatibility
Old budget structs (`ExpandBudget`, `PolyBudget`) become "frontends" that convert to `BudgetConfig`. No API breakage.

### 2. Central Node Counting
`Context::add` always increments `nodes_created`. Even if a module forgets explicit charges, real growth is tracked.

### 3. Single Error Type
```rust
pub struct BudgetExceeded {
    pub op: Operation,
    pub metric: Metric,
    pub used: u64,
    pub limit: u64,
}
```

All modules convert their budget errors to this.

### 4. CI Audit
`scripts/lint_budget_enforcement.sh` checks that hotspot modules contain budget charges.

## References

- [Implementation plan](../MAINTENANCE.md) (section 10)
- [Zippel GCD](ZIPPEL_GCD.md) — Uses `ZippelBudget`
- [Auto-expand](../POLICY.md) — Uses `ExpandBudget`
