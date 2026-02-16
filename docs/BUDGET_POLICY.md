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

## Two Orthogonal Dimensions

The budget system has **two independent dimensions**:

### Dimension 1: Limits (via `--budget`)

Controls **numeric thresholds** for resource consumption:

| Preset | RewriteSteps | NodesCreated | TermsMaterialized | PolyOps | Use Case |
|--------|-------------|--------------|-------------------|---------|----------|
| `small` | 5,000 | 25,000 | 10,000 | 500 | Teaching, REPL |
| `standard` | 50,000 | 250,000 | 100,000 | 5,000 | Interactive use |
| `unlimited` | ∞ | ∞ | ∞ | ∞ | Scripts, CI |

### Dimension 2: Error Mode (via `--strict`)

Controls **what happens when limits are exceeded**:

| Flag | Behavior | Exit Code |
|------|----------|-----------|
| (default) | **Best-effort**: return partial/unexpanded result | 0 |
| `--strict` | **Fail**: return error, stop processing | non-zero |

> [!IMPORTANT]
> Presets only set limits, NOT the error mode.
> Use `--strict` separately to control error handling.

### Rust Presets (Library)

```rust
let budget = Budget::preset_small();     // Conservative limits
let budget = Budget::preset_standard();  // Standard limits (was preset_cli)
let budget = Budget::preset_unlimited(); // No limits
```

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

## Phase Rules for Auto-Expand

> [!IMPORTANT]
> Auto-expand rules like `AutoExpandPowSumRule` must include `RATIONALIZE` in their `allowed_phases()`.
> This ensures expressions like `(1+√2)²` created during rationalization are expanded to complete the simplification.

| Rule | Allowed Phases | Reason |
|------|----------------|--------|
| `AutoExpandPowSumRule` | CORE, TRANSFORM, **RATIONALIZE** | Close rationalizations that create Pow(Add, n) |
| `AutoExpandSubCancelRule` | TRANSFORM | Zero-shortcut for cancellation patterns |

## `budget_exempt` Rules

Some rules bypass the global anti-worsen budget because their **own guards** are
strictly tighter. Each must be on the CI allowlist with documented justification.

| Rule | File | Guards | Justification |
|------|------|--------|--------------|
| `SmallMultinomialExpansionRule` | `expansion.rs` | n≤4, k≤6, terms≤35, base_nodes≤25, output_nodes≤350 | Tight pre+post guards cap all dimensions |
| `InvTrigNAngleRule` | `inv_trig_n_angle.rs` | MAX_N=5, output/input bounded by formula | Finite, closed-form expansions |

### Allowlist Enforcement

The allowlist is enforced in `inv_trig_n_angle_tests.rs::budget_exempt_allowlist`:

```rust
const BUDGET_EXEMPT_ALLOWLIST: &[&str] = &[
    "inv_trig_n_angle.rs",  // MAX_N=5, closed-form
    "expansion.rs",         // n≤4, k≤6, output_nodes≤350
];
```

> [!IMPORTANT]
> Adding a new `budget_exempt` rule requires: (1) adding to the allowlist array,
> (2) documenting the guards in the entry comment, (3) adding a corresponding
> test to verify the guards are effective.

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

### Budget Presets (--budget)

Presets only set **numeric limits**, not the error mode:

| Preset | Rewrites | Nodes | Terms | Use Case |
|--------|----------|-------|-------|----------|
| `small` | 5,000 | 25,000 | 10,000 | Teaching, REPL |
| `standard` (default) | 50,000 | 250,000 | 100,000 | Interactive |
| `unlimited` | ∞ | ∞ | ∞ | Scripts, CI |

### Error Mode (--strict)

| Flag | Behavior on Budget Exceeded |
|------|------------------------------|
| (default) | **Best-effort**: return partial result, exit 0 |
| `--strict` | **Fail**: return error, non-zero exit code |

### Combining Preset and Mode

```bash
# Teaching mode: small limits, no errors
expli eval "expand((a+b)^10)" --budget small

# CI mode: standard limits, strict errors  
expli eval "simplify(huge_expr)" --budget standard --strict

# Interactive: large limits, best-effort
expli eval "expand((a+b)^50)" --budget unlimited
```

### Reading from stdin

```bash
# Use "-" as expression to read from stdin
echo "x+1" | expli eval "-" --format json

# Or omit expression entirely
echo "x+1" | expli eval --format json

# Pipe from file
cat expressions.txt | expli eval --format json
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
