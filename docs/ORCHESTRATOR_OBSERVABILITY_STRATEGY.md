# Orchestrator Observability Strategy

## Goal

Improve the engine by making
[orchestrator.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs)
more observable, more locally understandable, and safer to evolve.

This document is a derived strategy under
[ENGINE_IMPROVEMENT_AUTOMATION.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_IMPROVEMENT_AUTOMATION.md).

For the broader structural-cohesion policy that also covers
[arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs),
see
[ENGINE_COHESION_REFACTORING_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_COHESION_REFACTORING_STRATEGY.md).

For the current calculus-generalization campaign, the same observability
principle applies to calculus route families when repeated local helpers make
detection, domain reasoning, residual verification, post-calculus presentation,
or didactic step ownership unclear. Use the cohesion strategy and
[CALCULUS_ENGINE_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/CALCULUS_ENGINE_STRATEGY.md)
for those non-orchestrator cases.

It should lead an iteration only when the ROI selector chooses
`observability` or when orchestrator observability is the cheapest way to
unlock a higher-confidence `runtime`, `coverage`, or `combination` change.

The target is not cosmetic refactoring.

The target is to reduce three real risks:

- hot-path runtime regressions from broad no-match overhead
- accidental changes in shortcut priority or control flow
- difficulty reasoning about why a local improvement hurts global traffic

## Why This Matters

`orchestrator.rs` is large enough that it now hides several classes of debt:

- repeated local matching patterns
- duplicated anchor/partner routing logic
- implicit priority encoded only by source order
- broad shortcuts with expensive no-match behavior
- difficult-to-see interactions between embedded traffic and narrow `nf-first`
  wins

That combination makes engine work slower and less trustworthy.

When the orchestrator is hard to observe, the team loses leverage:

- small fixes become risky
- runtime regressions become hard to localize
- refactors become guesswork
- adding new mathematical families becomes more expensive than it should be

## Non-Goal

This strategy is not:

- a plan to rewrite the orchestrator wholesale
- a license to reorder shortcuts for aesthetic reasons
- a generic cleanup pass that changes many families at once

Large undirected refactors are especially dangerous in a symbolic engine because
they often change:

- rule priority
- normal-form orientation
- shortcut re-entry behavior
- timeout and stack-overflow characteristics

## Core Principle

Refactor only after increasing observability.

In practice that means:

- map the orchestrator before moving logic
- measure hot paths before generalizing a matcher
- separate cheap detection from expensive rewrite work
- preserve ordering semantics unless there is measured evidence to change them

This is not the primary strategy for the engine campaign.

It is the way the campaign should operate when:

- recent local wins fail the global guardrail and we still do not know why
- a broad orchestrator family is clearly hot, but the internal winner/loser
  routes are still opaque
- we need to decide whether the next best ROI is `runtime` or `combination`

## What Good Looks Like

The orchestrator should gradually move toward a shape where we can answer these
questions cheaply:

- which shortcut family handled this expression?
- which shortcut families were tested and rejected?
- which matchers dominate `embedded` runtime?
- which shortcuts are duplicated in structure but not yet unified?
- which blocks are broad and expensive even when they do not match?

That is the level of control needed to improve engine health safely.

## Recommended Work Order

### Phase 1: Build A Map

Before refactoring, produce a technical map of the current orchestrator:

- execution order of shortcut families
- rough grouping by domain:
  - trig / inverse trig
  - hyperbolic
  - algebraic factorization
  - fraction/telescoping
  - zero-difference / product-pair detection
  - embedded/contextual routing
- repeated helper shapes:
  - anchor detection
  - partner canonicalization
  - combined-factor rebuilding
  - isolated simplify gating

This map should be descriptive first, not prescriptive.

### Phase 2: Add Observability

Add lightweight instrumentation or profiling hooks that help answer:

- which shortcut names dominate wall time in `embedded`
- which shortcut names have high no-match counts
- which matcher families rebuild expressions repeatedly before failing
- which families are broad and expensive relative to their actual hit rate

Useful signals:

- hit count per shortcut
- no-match count per shortcut
- cumulative wall time per shortcut
- average wall time per call
- expressions or shape buckets associated with worst hotspots
- embedded scorecard runtime deltas against a recent baseline

This does not have to become user-facing product output.
It is an engine-maintenance tool.

Observability work should also record *failed-but-informative* runtime ideas.

When a change produces a strong local win but regresses `embedded`, do not just
discard the idea silently.

Record it in
[ENGINE_COMBINATION_LEDGER.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_COMBINATION_LEDGER.md)
with:

- the hotspot labels
- the local delta
- the embedded delta
- the likely cause of the regression
- the kind of complementary change that might make it safe later

That keeps the team from rediscovering the same local win repeatedly, and makes
future combination work evidence-driven instead of anecdotal.

### Phase 3: Extract By Family, Not By Layer

Do not start with “split the file into N modules”.

Start with families that are already coherent and measurable, for example:

- direct-pair anchor products
- special-angle exact value products
- collapsed fraction / telescoping products
- embedded trig product-to-sum routing

Each extraction should:

- preserve call order
- preserve behavior
- reduce duplication or clarify ownership
- come with the same benchmark checks before and after

When the same pattern also lives in `arithmetic.rs`, apply the broader cohesion
policy:

- first extract the family boundary without changing behavior
- then compare the extracted families for genuinely shared helper shapes
- only then introduce a bounded shared algorithm

Do not move a helper out of `orchestrator.rs` merely because it looks similar to
an arithmetic helper. Shared ownership is justified only when the domain policy,
candidate gate, step behavior, and embedded-runtime profile all match.

### Phase 4: Separate Cheap Match From Expensive Rewrite

This is likely the highest-ROI architectural improvement.

Many runtime regressions in large orchestrators come from shortcuts that:

- flatten or rebuild too much too early
- call rewrite helpers before proving the shape is plausible
- scan many factor combinations before a cheap guard would have ruled them out

A healthy pattern is:

1. cheap syntactic prefilter
2. narrow structural match
3. expensive canonicalization or rewrite
4. isolated simplify only after a real candidate exists

If a shortcut cannot be written in that shape, it should be treated as high-risk.

### Phase 5: Make Priority Explicit

Right now, much of orchestrator priority is encoded by source order.

That is acceptable short-term, but dangerous long-term.

The medium-term goal should be to make shortcut families legible as ordered
groups, for example:

- functional/robustness guards
- exact-zero collapses
- embedded/contextual local equivalence
- narrow anchor-partner products
- more general factor/canonicalization routes

This does not require changing behavior immediately.
It requires making the existing intent visible.

## Refactoring Rules

Any orchestrator refactor should satisfy all of these:

- no broad reorderings without measurement
- no family-wide generalization without a cheap prefilter
- no extraction that makes shortcut ownership less obvious
- no “cleanup” that merges distinct behaviors just because the code looks similar

Each retained refactor should ideally improve at least one of:

- local readability
- duplication reduction
- hotspot observability
- ability to benchmark a family in isolation

Each rejected-but-locally-strong refactor should ideally leave behind:

- one new observability cut, or
- one ledger entry in
  [ENGINE_COMBINATION_LEDGER.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_COMBINATION_LEDGER.md)

Otherwise the team loses the reasoning that justified the experiment.

## Validation Policy

Every orchestrator refactor must be treated as engine work, not formatting work.

Minimum validation:

- touched unit tests
- relevant metamorphic lane
- `embedded_equivalence_context_corpus`

If the refactor changes broad routing or hot-path matcher placement, then
`embedded` runtime is a required metric, not optional context.

That metric should be read from the scorecard with a baseline comparison, not
only from ad hoc shell notes after the run.

That means a refactor can fail even if:

- semantics still pass
- local targeted tests improve

if it causes a meaningful unforced slowdown in `embedded`.

When that happens, the default action is:

1. revert the runtime change
2. keep any safe observability that helped localize the issue
3. log the case in
   [ENGINE_COMBINATION_LEDGER.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_COMBINATION_LEDGER.md)
4. only revisit it later with a concrete complementary hypothesis such as:
   - a cheap gate
   - a call-site restriction
   - signature reuse or cache
   - a second patch that removes the new broad traffic cost

## Benchmark Roles For Orchestrator Work

Orchestrator work should be read against three benchmark roles, not one moving
number:

- `frozen`
  - a small representative suite snapshot that does not normally change
  - measures baseline overhead tax on already-known traffic
- `live`
  - the current representative guardrail workload
  - measures performance on the traffic we actually care about now
- `stress`
  - larger or more combinatorial expressions
  - measures scaling risk and asymptotic mistakes

### Why This Split Matters

Without this split, a slowdown is ambiguous:

- maybe the orchestrator got worse
- maybe the corpus simply got broader
- maybe the engine got more complete and the extra cost is justified

With the split:

- `frozen` tells us how much general tax we added
- `live` tells us whether that tax buys value on current workload
- `stress` tells us whether the change creates a scaling cliff

### Validation Use

For broad routing changes, the preferred reading order is:

1. `frozen` overhead
2. `live` workload delta
3. `stress` scaling delta

When discussing runtime, always say which role produced the number:

- frozen baseline
- live workload
- stress scaling

## How This Connects To Engine Improvement Strategy

This strategy exists to improve future engine work, not just maintain the
orchestrator file itself.

A more observable orchestrator should make it easier to:

- add new mathematical families safely
- understand why a family lands in `proved-symbolic` instead of `NF-convergent`
- localize runtime cliffs earlier
- reject narrow wins that degrade global contextual traffic
- recover performance after robustness fixes

In other words:

- better orchestrator observability improves engine health
- better engine health improves the chance that future mathematical work remains
  affordable

## Immediate Next Steps

The highest-value next actions are:

1. write a current-family map of `orchestrator.rs`
2. identify the hottest shortcut groups under `embedded`
3. isolate the top no-match cost centers
4. refactor one coherent family with before/after metrics

That is the right order.

Trying to “clean up the file” before that would likely produce more risk than
value.
