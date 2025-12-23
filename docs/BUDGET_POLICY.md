# Anti-Explosion Budget Policy

> **Status**: ✅ Implemented  
> **Tracking**: See [MAINTENANCE.md](../MAINTENANCE.md) section 10

## Overview

The unified budget system prevents computational explosion across all CAS operations by tracking resource consumption with consistent metrics and enforcement.

## Metrics

| Metric | What it measures | Used by |
|--------|-----------------|---------|
| `NodesCreated` | AST nodes added to Context | All operations (Layer A) |
| `RewriteSteps` | Rule applications | Simplify phases |
| `TermsMaterialized` | Terms generated during expansion | Expand, Multinomial, Poly ops |
| `PolyOps` | Expensive polynomial operations | Mul, Div, GCD |

## Operations

| Operation | Description |
|-----------|-------------|
| `SimplifyCore` | Core simplification (algebraic rules) |
| `SimplifyTransform` | Transform phase (distribution) |
| `Expand` | Explicit expand() calls |
| `MultinomialExpand` | Multinomial expansion |
| `PolyOps` | Polynomial multiplication, division |
| `GcdZippel` | Zippel GCD algorithm |

## Modes

### Strict Mode (Library default)
- Returns `Err(CasError::BudgetExceeded)` when limit exceeded
- Stops immediately, no partial result

### Best-Effort Mode (CLI default)
- Returns partial result without error
- Logs warning and stops processing

## Usage

```rust
use cas_engine::{Budget, Operation, Metric};

// Preset for different environments
let budget = Budget::preset_small();  // Conservative
let budget = Budget::preset_cli();    // Interactive use
let budget = Budget::preset_unlimited(); // No limits

// Custom limits
let mut budget = Budget::new();
budget.set_limit(Operation::Expand, Metric::TermsMaterialized, 10_000);
budget.set_limit(Operation::SimplifyCore, Metric::RewriteSteps, 5_000);
```

## Presets

| Preset | RewriteSteps | NodesCreated | TermsMaterialized | PolyOps | Mode |
|--------|-------------|--------------|-------------------|---------|------|
| `preset_small()` | 5,000 | 25,000 | 10,000 | 500 | Strict |
| `preset_cli()` | 50,000 | 250,000 | 100,000 | 5,000 | Best-effort |
| `preset_unlimited()` | 0 | 0 | 0 | 0 | Best-effort |

## Architecture

```
┌────────────────────────────────────────────────┐
│              Unified Budget System             │
├────────────────────────────────────────────────┤
│  budget.rs:  Budget, PassStats, BudgetExceeded │
│  Operation:  What is being done                │
│  Metric:     What is being measured            │
└────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────┐
│           3-Layer Enforcement                  │
├────────────────────────────────────────────────┤
│ A. Central: NodesCreated via ContextStats      │
│ B. Hotspot: _with_stats functions in modules   │
│ C. Pre-estimation: Fail fast before work       │
└────────────────────────────────────────────────┘
```

## Examples

### Explosive input protection

```rust
// (a+b)^200 → pre-estimation catches before expanding
// Returns BudgetExceeded { op: Expand, metric: TermsMaterialized }
```

### Simplify runaway prevention

```rust
// Deep recursive expression → stops after rewrite limit
// Returns BudgetExceeded { op: SimplifyCore, metric: RewriteSteps }
```

## Instrumented Functions

| Module | Function | PassStats fields |
|--------|----------|------------------|
| engine.rs | `apply_rules_loop*` | rewrite_count, nodes_delta |
| expand.rs | `expand_with_stats` | terms_materialized, nodes_delta |
| multipoly.rs | `mul_with_stats` | poly_ops, terms_materialized |
| multipoly.rs | `div_exact_with_stats` | poly_ops, terms_materialized |
| multipoly.rs | `gcd_multivar_layer2_with_stats` | poly_ops, terms_materialized |

## CI Enforcement

`scripts/lint_budget_enforcement.sh` verifies all hotspot modules contain budget instrumentation:

```bash
make lint-budget
```

## Implementation Phases (All Complete)

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Infrastructure (`budget.rs`, `ContextStats`) | ✅ |
| 1 | Unify error types (`CasError::BudgetExceeded`) | ✅ |
| 2 | Simplify pipeline (`PassStats`) | ✅ |
| 3 | Expand / multinomial | ✅ |
| 4 | Polynomial operations | ✅ |
| 5 | Zippel GCD | ✅ |
| 6 | CI lint enforcement | ✅ |
