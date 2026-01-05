# Display Transform Policy

This document defines the rules and constraints for the display transform system.
Display transforms provide context-aware rendering without modifying canonical form.

## Core Principles

### 1. Display-Only Transforms

Transforms are **strictly display-only**. They:
- Never modify the AST or canonical representation
- Never affect mathematical semantics
- Never generate assumptions or side-effects
- Only affect the string output of expressions

```
Canonical: Pow(3, Div(1, 2))  →  Always stores as ^(1/2)
Display:   "√(3)" or "sqrt(3)" depending on scope
```

### 2. Scope-Gated Activation

All transforms must be gated by explicit scopes:
- `ScopeTag::Rule("QuadraticFormula")` - transformation rules
- `ScopeTag::Solver("isolate")` - solver strategies  
- `ScopeTag::Command("solve")` - REPL command context

Transforms should **never** apply globally without scope context.

### 3. Prefilter Efficiency Contract

The `DisplayTransformRegistry::active_for(scopes)` method filters transforms
**once per render session**, not per node. This is verified by CI test:

```rust
#[test]
fn test_prefilter_efficiency() {
    // Verifies applies() called once per ScopedRenderer::new(),
    // not per rendered node
}
```

### 4. ASCII Default for Test Stability

Test snapshots should use ASCII output (`sqrt(x)`) rather than Unicode (`√x`)
for cross-platform stability. The `is_pretty_output()` flag controls this.

### 5. New Transform Requirements

Every new transform must include:
1. **Scope gate**: Explicit `applies(&[ScopeTag])` implementation
2. **Unit test**: Verify scope activation/deactivation
3. **Integration test**: Verify with real expressions
4. **No global side-effects**: Pure function from AST → String

## Current Transforms

| Transform | Scope | Effect |
|-----------|-------|--------|
| `HalfPowerAsSqrt` | `Rule("QuadraticFormula")` | `x^(1/2)` → `sqrt(x)` |

## Future Transforms (Proposed)

| Transform | Scope | Effect |
|-----------|-------|--------|
| `LogBaseNotation` | `Solver("isolate")` | `log(a, x)` → `log_a(x)` |
| `PlusMinusNotation` | `Rule("QuadraticFormula")` | Combined `±` display |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    EvalOutput                         │
│  output_scopes: Vec<ScopeTag>                        │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│              DisplayTransformRegistry                 │
│  transforms: Vec<Box<dyn DisplayTransform>>          │
│  ──────────────────────────────────────              │
│  active_for(scopes) → Vec<&dyn DisplayTransform>    │
└────────────────────┬─────────────────────────────────┘
                     │ (called once)
                     ▼
┌──────────────────────────────────────────────────────┐
│                 ScopedRenderer                        │
│  active_transforms: Vec<&dyn DisplayTransform>       │
│  ──────────────────────────────────────              │
│  render(id) → String                                 │
│    └── tries each active transform                   │
│        └── fallback to DisplayExpr                   │
└──────────────────────────────────────────────────────┘
```

## Related Policies

- `CONST_FOLD_POLICY.md` - Constant folding behavior
- `ASSUMPTIONS_POLICY.md` - Domain assumption handling
