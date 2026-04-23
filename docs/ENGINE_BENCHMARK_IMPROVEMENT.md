# Engine Benchmark Improvement

## Goal

Make engine evaluation more honest, more actionable, and harder to misread.

The engine does not need one bigger benchmark.
It needs a better benchmark stack:

- one layer for mathematical power and completeness
- one layer for real engine runtime pressure
- one layer for robustness under contextual composition
- one layer for didactic output quality

This document complements:

- [ENGINE_IMPROVEMENT_AUTOMATION.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_IMPROVEMENT_AUTOMATION.md)
- [ORCHESTRATOR_OBSERVABILITY_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/ORCHESTRATOR_OBSERVABILITY_STRATEGY.md)
- [METAMORPHIC_TESTING.md](/Users/javiergimenezmoya/developer/math/docs/METAMORPHIC_TESTING.md)
- [DIDACTIC_STEP_QUALITY_AUDIT_PLAN.md](/Users/javiergimenezmoya/developer/math/docs/DIDACTIC_STEP_QUALITY_AUDIT_PLAN.md)
- [SOLVER_EVENT_OBSERVER.md](/Users/javiergimenezmoya/developer/math/docs/SOLVER_EVENT_OBSERVER.md)

## Why This Is Necessary

The current scorecard already does several valuable things well:

- catches semantic failures
- catches timeouts and stack-overflow style fragility
- protects embedded contextual quality
- tracks `derive` reachability and short-path quality

But it still has interpretation gaps.

The biggest one observed during engine-improvement loops is this:

- the unified metamorphic benchmark can be dominated by `proved-composed`
- that is a real semantic signal
- but it is not the same thing as direct engine closure cost on one raw path

So a fully green benchmark can still hide the fact that:

- the harness did much of the structural composition work
- one specific composition family is much slower than the aggregate table suggests
- a local “improvement” only moved benchmark shape, not real engine cost

That makes benchmark reading too easy to overfit.

## Current Benchmark Stack

Today the retained runnable stack is roughly:

- `fast`
  - quick sanity loop
- `guardrail`
  - `embedded_equivalence_context`
  - `derive_contract`
  - `simplify_strict`
- `pressure`
  - `simplify_zero_mixed`
  - `simplify_nf_first`

This is already useful, but each layer answers a different question.

The mistake is treating them as interchangeable evidence.

## Main Weaknesses Observed

### 1. `proved-composed` inflation in the unified benchmark

When `simplify_strict` is mostly `proved-composed`, the benchmark is telling us:

- semantic closure is good
- composition invariants still hold

It is **not** necessarily telling us:

- the raw engine path is cheap
- the engine found the result early
- the engine handled the final expression without harness-shaped help

This matters most for runtime-guided iterations.

For those, `simplify_strict` is a strong robustness/correctness guardrail, but a
weaker direct runtime proxy.

### 2. Aggregate totals hide composition hotspots

The mixed zero corpus is often a better runtime pressure signal than the unified
metamorphic benchmark because it runs through the canonical eval path on real
composed zero-target expressions.

But plain totals like:

- total passed
- total failed
- total elapsed

still hide where the cost actually sits.

In practice, a single composition bucket such as:

- `sum`
- `difference`
- `product`
- `shifted_quotient`

can dominate elapsed time or failure risk.

If the scorecard does not surface that split, the next profitable move stays
harder to see than it should be.

### 3. Pressure slices are not yet standardized enough

The repo already allows manual focused runs such as:

- `--composition sum`
- `--limit N`
- `--trace-from N`

But those are still mostly investigator tools, not first-class scorecard
artifacts.

That means a good local investigation can still be lost unless someone writes
it down manually in the iteration log or ledger.

### 4. Runtime and didactic quality are still too separate

`derive_contract` already tracks:

- reachability
- supported equivalence rate
- mean step count
- long path rate

And didactic audits already exist for step quality.

But the engine scorecard still does not provide a compact “educational output
health” view for simplify/derive together.

So the current evaluation stack is stronger on semantic truth than on
step-by-step teaching quality.

### 5. We still lack a clean “direct engine work vs harness help” split

The benchmark stack needs a more explicit distinction between:

- closure produced directly by the engine on the tested expression
- closure that is mathematically valid but benchmark-shaped by composition logic
- closure that only survives through a late fallback path

Without that split, “power”, “robustness”, and “cost” are too easy to blur.

## Interpretation Policy

Use these metrics for different jobs.

### Mathematical power and completeness

Primary evidence:

- `simplify_strict`
- `embedded_equivalence_context`
- `derive_contract`

Interpretation:

- strong semantic/coverage evidence
- not enough by itself for runtime claims

### Real engine runtime pressure

Primary evidence:

- `embedded_equivalence_context` elapsed and baseline delta
- `simplify_zero_mixed` elapsed by composition
- orchestrator profile slices when the hotspot is still ambiguous

Interpretation:

- better proxy for real engine work
- should lead `runtime` iterations more often than raw `proved-composed` counts

### Robustness

Primary evidence:

- failures
- timeouts
- stack overflows
- nontermination or pathologically slow pockets

Interpretation:

- a green semantic summary is not enough if one promoted lane is still brittle

### Educational output quality

Primary evidence:

- [DIDACTIC_STEP_QUALITY_AUDIT_PLAN.md](/Users/javiergimenezmoya/developer/math/docs/DIDACTIC_STEP_QUALITY_AUDIT_PLAN.md)
- derive didactic audits
- step-wire / timeline parity tests

Interpretation:

- this is a separate quality axis
- it should not be inferred from semantic correctness alone

## Retained Immediate Improvements

This document keeps the following immediate changes as part of the benchmark
policy:

### A. Surface `simplify_zero_mixed` cost by composition

The mixed zero corpus runner should report, per composition:

- total
- passed
- failed
- elapsed
- average milliseconds per case

This turns “the corpus feels slow” into a stable, inspectable hotspot signal.

### B. Surface proof-shape mix in the scorecard

The scorecard should explicitly show when the unified simplification benchmark
is dominated by `proved-composed`.

That is not a failure.

It is an interpretation warning:

- strong semantic signal
- weaker direct runtime signal

### C. Prefer composition-level pressure for runtime ROI selection

When choosing the next `runtime` iteration, prefer evidence from:

- `embedded`
- `simplify_zero_mixed`
- orchestrator profile slices

over a raw reading of global `proved-composed` totals.

## Recommended Next Improvements

### 1. Add a first-class “proof origin” split

The next scorecard generation should try to separate:

- direct closure
- composed closure
- fallback closure

The target is not perfect philosophical purity.
The target is enough signal to stop reading all symbolic proofs as equal runtime
evidence.

### 2. Promote stable pressure slices to named artifacts

Useful candidates:

- `simplify_zero_mixed.sum`
- `simplify_zero_mixed.difference`
- `simplify_zero_mixed.product`
- `simplify_zero_mixed.shifted_quotient`

Those should remain pressure tools, not necessarily merge guardrails.
But they should be stable enough to compare before/after iterations.

### 3. Add a simplify didactic scorecard slice

The engine scorecard should eventually include a compact educational lane for
simplify traces, not only `derive`.

Good first metrics:

- cases with no explanatory substeps where a magical jump remains
- generic-template substep rate
- redundant single-substep rate
- real-intermediate rate

### 4. Add capped slow-example output for pressure suites

For pressure corpora, the runner should eventually surface a very small sampled
set of the slowest examples.

That would help answer:

- what is slow?
- not just which bucket is slow

without turning every iteration into a manual tracing session.

### 5. Make benchmark roles explicit in review language

When describing results, avoid saying only “the benchmark is green”.

Prefer:

- semantic guardrail green
- runtime pressure improved
- composition hotspot unchanged
- didactic audit unchanged

That wording is more honest and leads to better iteration choices.

## Non-Goals

This document does not recommend:

- replacing metamorphic testing with corpus testing
- collapsing all metrics into one scalar
- treating didactic quality as a runtime concern
- inflating benchmark suites just to produce larger numbers

The engine needs sharper evidence, not just more evidence.

## Practical Rule

Do not use one benchmark as the answer to all of these at once:

- Is the math right?
- Is the engine robust?
- Is the engine fast on real traffic?
- Is the trace educational?

Use different lanes for different claims, and write the claim explicitly.
