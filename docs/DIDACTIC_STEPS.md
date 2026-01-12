# Didactic Step Systems: ChainedRewrite vs SubSteps

ExpliCAS uses two complementary systems for presenting multi-step mathematical transformations to users.

## Overview

| System | Layer | Output | Purpose |
|--------|-------|--------|---------|
| **ChainedRewrite** | Engine (`rule.rs`) | Real `Step` objects with `ExprId` | Multi-step algebraic decomposition |
| **SubSteps** | Display (`didactic.rs`) | `SubStep` with LaTeX strings | Educational annotation within a Step |

---

## ChainedRewrite (Engine Layer)

### Data Structure

```rust
// rule.rs
pub struct ChainedRewrite {
    pub after: ExprId,                    // Real expression
    pub description: String,
    pub before_local: Option<ExprId>,     // Focus for "Rule:" line
    pub after_local: Option<ExprId>,
    pub required_conditions: Vec<ImplicitCondition>,
    pub assumption_events: SmallVec<[AssumptionEvent; 1]>,
    pub importance: Option<ImportanceLevel>,
}
```

### Usage Pattern

Rules use the fluent builder API to chain transformations:

```rust
// Example: SimplifyFractionRule in fractions.rs
let factor_rw = Rewrite::new(factored_form)
    .desc("Factor by GCD: x - 2")
    .local(expr, factored_form);

let cancel = ChainedRewrite::new(result)
    .desc("Cancel common factor")
    .local(factored_form, result);

return Some(factor_rw.chain(cancel));
```

### Engine Processing

The engine in `engine.rs` processes each chained rewrite sequentially:

```rust
// engine.rs (simplified)
for chain_rw in chained_rewrites {
    let chain_step = Step::with_snapshots(
        &chain_rw.description,
        rule.name(),
        current,        // Before = previous step's After
        chain_rw.after, // After = this step's result
        ...
    );
    chain_step.is_chained = true;  // Mark for didactic gating
    self.steps.push(chain_step);
    current = chain_rw.after;
}
```

### Output

ChainedRewrite produces **separate visible Steps** in the timeline:

```
1. Factor by GCD: x - 2  [Simplify Fraction]
   Before: (x² - 4) / (x + 2)
   Rule: x² - 4 -> (x - 2)(x + 2)
   After: (x - 2)(x + 2) / (x + 2)

2. Cancel common factor  [Simplify Fraction]
   Before: (x - 2)(x + 2) / (x + 2)
   Rule: (x - 2)(x + 2) / (x + 2) -> x - 2
   After: x - 2
```

---

## SubSteps (Display Layer)

### Data Structure

```rust
// didactic.rs
pub struct SubStep {
    pub description: String,    // "Denominador binomial con radical"
    pub before_latex: String,   // LaTeX string (display only)
    pub after_latex: String,    // LaTeX string (display only)
}

pub struct EnrichedStep {
    pub base_step: Step,
    pub sub_steps: Vec<SubStep>,
}
```

### Processing

`enrich_steps()` post-processes Steps after simplification:

```rust
pub fn enrich_steps(ctx: &Context, original_expr: ExprId, steps: Vec<Step>) -> Vec<EnrichedStep> {
    for step in steps {
        let mut sub_steps = Vec::new();
        
        // Pattern matching on description/rule_name
        if step.description.contains("Rationalize") {
            sub_steps.extend(generate_rationalization_substeps(ctx, &step));
        }
        
        enriched.push(EnrichedStep { base_step: step, sub_steps });
    }
}
```

### Use Cases

| Case | Generator Function |
|------|-------------------|
| Rationalization | `generate_rationalization_substeps()` |
| Fraction Arithmetic | `generate_fraction_sum_substeps()` |
| Nested Fractions | `generate_nested_fraction_substeps()` |
| Polynomial Identity | `generate_polynomial_identity_substeps()` |
| GCD Factorization | `generate_gcd_factorization_substeps()` (gated) |

### Output

SubSteps appear as **indented annotations** within a single Step:

```
1. Rationalize denominator  [Rationalize Denominator]
   Before: 1 / (√x - 1)
      → Denominador binomial con radical
        1/(√x - 1) → Conjugado: √x + 1
      → (a+b)(a-b) = a² - b²
        ...
   Rule: 1 / (√x - 1) -> (1 + √x) / (x - 1)
   After: (1 + √x) / (x - 1)
```

---

## Contract: Avoiding Duplication (V2.12.13)

> **Rule**: SubSteps MUST NOT duplicate decompositions that already exist as chained Steps via ChainedRewrite.

### Implementation

The `is_chained` flag on `Step` gates substep generation:

```rust
// didactic.rs
if step.description.starts_with("Simplified fraction by GCD") && !step.is_chained {
    sub_steps.extend(generate_gcd_factorization_substeps(ctx, step));
}
```

### Lifecycle of `is_chained`

| Event | Value |
|-------|-------|
| `Step::new()` / `Step::new_compact()` | `false` |
| ChainedRewrite processing in engine | `true` |
| Step coalescing in optimization | `false` |

---

## Decision Guide: When to Use Which

| Scenario | Use | Rationale |
|----------|-----|-----------|
| Multi-step algebraic decomposition (Factor → Cancel) | **ChainedRewrite** | Each step needs real `ExprId`, visible in timeline |
| Explaining a technique (find conjugate) | **SubSteps** | Educational annotation, no state change |
| Step needs its own `Requires`/`Assumes` | **ChainedRewrite** | SubSteps can't carry conditions |
| Micro-explanation within a rule | **SubSteps** | Doesn't warrant separate timeline row |

---

## Version History

| Version | Change |
|---------|--------|
| V2.12.11 | ChainedRewrite introduced |
| V2.12.13 | `is_chained` flag added, SubSteps gated for GCD |
