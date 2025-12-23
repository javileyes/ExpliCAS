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

## CLI Usage

### Running the CLI

```bash
# Option 1: With cargo run (development)
cargo run -p cas_cli --release -- eval "expand((a+b)^5)" --budget small

# Option 2: Run compiled binary directly
./target/release/cas_cli eval "expand((a+b)^5)" --budget small

# Option 3: Create alias (add to ~/.zshrc or ~/.bashrc)
alias expli="$HOME/developer/math/target/release/cas_cli"
expli eval "expand((a+b)^5)" --budget small
```

### Budget Options

```bash
# Use conservative limits
cargo run -p cas_cli --release -- eval "expand((a+b)^200)" --budget small

# Use CLI defaults (larger limits)
cargo run -p cas_cli --release -- eval "expand((a+b)^10)" --budget cli

# No limits (use with caution)
cargo run -p cas_cli --release -- eval "expr" --budget unlimited

# Fail-fast mode (error on budget exceeded)
cargo run -p cas_cli --release -- eval "expand((a+b)^200)" --budget small --strict

# JSON output with budget info
cargo run -p cas_cli --release -- eval "x+1" --format json
```

### JSON Output

```json
{
  "schema_version": 1,
  "ok": true,
  "result": "1 + x",
  "budget": {
    "preset": "cli",
    "mode": "best-effort"
  }
}
```

## Flag Clarification

> [!IMPORTANT]
> `--strict` (budget) and `--branch strict` (domains) are **different flags**:

| Flag | Purpose | Options |
|------|---------|---------|
| `--strict` | Budget enforcement | Flag: fail on budget exceeded (vs best-effort) |
| `--branch` | Domain handling | `strict` (safe, shows warnings) / `principal` (assumes principal branch) |

### Examples

```bash
# Budget strict: Fail immediately if limits exceeded
cargo run -p cas_cli --release -- eval "expand((a+b)^200)" --budget small --strict

# Branch strict: Domain-safe, reports warnings for multi-valued functions
cargo run -p cas_cli --release -- eval "sqrt(x^2)" --branch strict

# Branch principal: Assumes principal branch (sqrt(x^2) → x)
cargo run -p cas_cli --release -- eval "sqrt(x^2)" --branch principal
```
