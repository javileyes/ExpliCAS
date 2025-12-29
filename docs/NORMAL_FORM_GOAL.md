# NormalFormGoal Policy

This document describes the `NormalFormGoal` system that prevents inverse simplification rules from undoing explicit user transformations like `collect()` or `expand_log()`.

## Problem Statement

When a user explicitly calls a transformation function like `collect(a*x + b*x, x)`, they expect the result `(a+b)*x` to be preserved. However, the default simplification pipeline includes rules like `DistributeRule` that would immediately undo this transformation, returning `a*x + b*x`.

Similarly, `expand_log ln(x*y)` should return `ln(x) + ln(y)`, but `LogContractionRule` would undo this.

## Solution: Goal-Gated Rules

The `NormalFormGoal` enum communicates the user's intent through the simplification pipeline:

```rust
pub enum NormalFormGoal {
    Simplify,    // Default: all rules enabled
    Expanded,    // expand(): don't collect/factor
    Collected,   // collect(): don't distribute
    Factored,    // factor(): don't expand
    ExpandedLog, // expand_log(): don't contract logs
}
```

## Architecture

```
User Input: collect(a*x + b*x, x)
                 │
                 ▼
┌────────────────────────────────────────────────┐
│              TOOL DISPATCHER                   │
│  Engine::eval() detects Function("collect",_) │
│  Sets: SimplifyOptions.goal = Collected       │
└────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│           SIMPLIFICATION PIPELINE              │
│  SimplifyOptions.goal propagates through:      │
│  Orchestrator → Engine → ParentContext        │
└────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│              RULE GATING                       │
│  DistributeRule checks parent_ctx.goal()      │
│  If goal == Collected → rule returns None     │
└────────────────────────────────────────────────┘
                 │
                 ▼
Output: (a+b)*x  ← Preserved!
```

## Gated Rules

| Goal | Gated Rules | Purpose |
|------|-------------|---------|
| `Collected` | `DistributeRule` | Preserve grouped terms |
| `Factored` | `DistributeRule` | Preserve factored forms |
| `ExpandedLog` | `LogContractionRule` | Preserve expanded logarithms |

## Implementation Details

### 1. Tool Dispatcher

Located in `Engine::eval()` (`crates/cas_engine/src/eval.rs`):

```rust
if let Expr::Function(name, _args) = ctx.get(resolved) {
    match name.as_str() {
        "collect" => opts.goal = NormalFormGoal::Collected,
        "expand_log" => opts.goal = NormalFormGoal::ExpandedLog,
        _ => {}
    }
}
```

### 2. Goal Propagation

The goal flows through:
1. `SimplifyOptions.goal` (set by dispatcher)
2. `Orchestrator.run_phase()` passes goal to engine
3. `Simplifier.apply_rules_loop_with_phase_and_mode()` creates `ParentContext.with_goal()`
4. `Rule::apply()` receives `parent_ctx` with goal

### 3. Rule Gating Pattern

Rules that can undo transformations check the goal:

```rust
// In DistributeRule
match parent_ctx.goal() {
    NormalFormGoal::Collected | NormalFormGoal::Factored => return None,
    _ => {}
}
// ... proceed with distribution logic
```

## Design Principles

### Node Count Principle

This system is complementary to the [Node Count Principle](./POLICY.md):

- **Node-reducing** transforms (contraction) → automatic (default rules)
- **Node-increasing** transforms (expansion) → explicit commands

The `NormalFormGoal` adds a second dimension:

- **Explicit** transforms (`collect`, `expand_log`) → preserve result via goal gating

### Orthogonal Semantic Axes

`NormalFormGoal` is orthogonal to other semantic settings:

| Axis | Controls |
|------|----------|
| `DomainMode` | Strictness of domain checking |
| `ValueDomain` | Real vs complex arithmetic |
| `NormalFormGoal` | Which inverse rules to gate |

## Adding New Goals

To add a new goal (e.g., for a `factor()` command):

1. Add variant to `NormalFormGoal` enum in `semantics.rs`
2. Add tool detection in `Engine::eval()` dispatcher
3. Add gate check in inverse rules (e.g., expand rules)
4. Add tests verifying preservation

## Related Documentation

- [POLICY.md](./POLICY.md) - Node Count Principle
- [SEMANTICS_POLICY.md](./SEMANTICS_POLICY.md) - Domain and value modes
- [cas_logarithm_simplification_and_notation KI](../.gemini/antigravity/knowledge/cas_logarithm_simplification_and_notation/) - Log-specific details
