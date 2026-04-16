# Engine Improvement Automation

## Goal
Build one repeatable loop that improves three things together without guessing:

- simplification coverage
- equivalence proving power
- `derive` reachability and didactic quality

The core idea is simple: every engine improvement campaign should be driven by an
explicit scorecard, not by isolated anecdotes.

This improvement loop is deliberately broader than “add one more rule”.

A real engine improvement may come from:

- a new simplification rule
- a better root/orchestrator shortcut
- a better derive target classifier or planner route
- a robustness fix that prevents stack overflow / timeout on expressions the
  engine already handled semantically

The system should reward all of those, but only if they preserve the global
guardrails.

## Current Automation Base

Use [engine_improvement_scorecard.py](/Users/javiergimenezmoya/developer/math/scripts/engine_improvement_scorecard.py) as the unified runner.

It groups the existing embedded and metamorphic checks into profiles:

- `fast`
  - `metatest_csv_combinations_small`
  - `metatest_csv_contextual_pairs_strict`
- `guardrail`
  - embedded equivalence context corpus
  - derive contract corpus
  - unified simplification benchmark in `strict`
- `pressure`
  - simplify-zero mixed corpus
  - unified simplification benchmark in `nf-first`
- `full`
  - `guardrail + pressure`

Outputs:

- JSON scorecard at [docs/generated/engine_improvement_scorecard.json](/Users/javiergimenezmoya/developer/math/docs/generated/engine_improvement_scorecard.json)
- Markdown summary at [docs/generated/engine_improvement_scorecard.md](/Users/javiergimenezmoya/developer/math/docs/generated/engine_improvement_scorecard.md)

## Why These Lanes Exist

`strict` and `nf-first` are not the same measurement.

There should also be a deliberately cheap iteration lane.

- `fast` is the iteration lane:
  - small enough to run often
  - intended to be used together with area unit tests
  - not sufficient for merge decisions by itself

- `strict` is the regression lane:
  - stable
  - fast enough to rerun often
  - catches semantic failures and timeouts
- `nf-first` is the pressure lane:
  - expensive
  - closer to raw normalization power
  - useful when we want to know whether the engine itself is actually converging

The embedded corpora matter because they are harder to game than a single benchmark:

- [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  verifies contextual equivalence under real wrappers
- [simplify_zero_mixed_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/simplify_zero_mixed_corpus.csv)
  verifies composed simplify-to-zero behavior across heterogeneous identities
- [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  measures whether `derive` can actually bridge source to target and how long the path is

For work centered on
[orchestrator.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs),
follow the dedicated architecture guidance in
[ORCHESTRATOR_OBSERVABILITY_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/ORCHESTRATOR_OBSERVABILITY_STRATEGY.md).

## Embedded Equivalence Runtime Is A Guardrail Metric

The runtime of
[embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
is not a cosmetic benchmark number.

It is a guardrail metric for engine quality because it exercises broad,
contextual traffic through the real simplification/orchestration pipeline.

That makes it a good detector of changes that:

- add expensive matchers too high in the pipeline
- introduce broad no-match overhead
- improve a narrow `nf-first` slice while taxing common contextual traffic
- trade local wins for global slowdown

So the automation should treat embedded runtime as a first-class scorecard
dimension, not just pass/fail.

### Policy

A change that makes embedded runtime materially worse should usually be rejected,
even when:

- a narrow pressure lane improves
- a small family gains `NF-convergent`

The exception is when the regression is clearly justified by a larger win in:

- functional correctness
- mathematical coverage
- robustness against timeout / overflow / nontermination

Even then, the burden of proof is on the change:

- the new functionality must be real and reusable
- the runtime regression must be measured explicitly
- the slowdown should be reduced as much as possible before the change is retained

## Why The Embedded Context Corpus Must Keep Growing

`embedded_equivalence_context_corpus.csv` is one of the highest-value guardrails
for mathematical completeness.

It tests something narrower benchmarks often miss:

- not just whether `expr1 ~ expr2`
- but whether that equivalence survives inside realistic algebraic wrappers

That matters because many real engine failures are contextual:

- the naked identity works
- the same identity fails when embedded in an additive, multiplicative, or quotient wrapper
- the engine proves equivalence only through a late fallback instead of using the identity locally

Expanding the embedded context corpus is therefore not optional maintenance.
It is how we increase contextual mathematical coverage without fooling ourselves
with isolated examples.

## Embedded Corpus Growth Policy

The embedded context corpus should grow by structural family, not by anecdote.

Good additions are families that introduce at least one of:

- a new mathematical identity family
- a new wrapper shape
- a new composition pattern between already-known families
- a known fragile path that has already regressed in real engine work

Examples of high-value family buckets:

- trig identities
- inverse trig compositions
- hyperbolic identities
- polynomial factorization families
- radicals and rationalization
- log/exp expansion and contraction
- special-angle exact constants
- telescoping and fraction decomposition
- sum-of-squares and product identities

Within each family, prefer a curated pattern:

- one root equivalence pair
- several contextual wrappers
- one or two composed variants that reflect real engine traffic

Avoid this anti-pattern:

- adding many near-duplicate examples that all exercise the same matcher path

The goal is not raw corpus size.
The goal is broader contextual mathematical coverage per added case.

## What The Embedded Corpus Should Not Become

The embedded corpus should not be used as:

- a dumping ground for every failing anecdote
- a replacement for narrow metamorphic pressure slices
- a benchmark that rewards redundant variants of the same local shape

If a candidate case only duplicates an already-covered family with the same
wrapper behavior, it should usually stay out.

If a case exposes a new structural interaction, it belongs in.

## Promotion Rule For New Embedded Cases

A family should be promoted into `embedded_equivalence_context_corpus.csv` when:

- the family is mathematically important or repeatedly seen in pressure lanes
- the engine behavior is stable enough that the case is now a guardrail, not a moving target
- the family covers a new wrapper or composition axis
- the case is likely to catch future regressions that unit tests would miss

A family should stay out of `embedded` for now when:

- the engine still has no stable common representation for it
- the only current value is synthetic benchmark pressure
- the case is useful for exploration but not yet mature enough to become a guardrail

Those cases belong first in:

- metamorphic slices
- localized regression tests
- corpus backlog notes

and only later in `embedded`.

## What Counts As A Valid Improvement

An engine change is only a real improvement if it moves at least one of these in
the right direction without reopening the others:

- more corpus cases pass
- more cases converge by NF
- fewer cases need `proved-symbolic`
- fewer `numeric-only`
- fewer timeouts
- fewer stack overflows
- lower derive step count or better derive reachability

That means the user intuition is correct:

- completeness matters
- robustness matters
- planner/orchestrator quality matters
- runtime budget matters

If a change makes the engine “smarter” but causes previously fast or stable
traffic to explode in runtime, it is not a clean win.

That principle applies especially to embedded runtime:

- `embedded` can get slower a little by accident
- but a large regression should only be accepted if it closes a high-value
  functional or robustness gap that the previous engine genuinely could not handle
- and even then, the follow-up task should be to recover the lost runtime

This is especially relevant for orchestrator work:

- broad shortcut changes in the orchestrator can improve a narrow family while
  taxing the whole engine
- so orchestrator refactors and new shortcut families should be treated as
  observability-first work, not just feature work
- see
  [ORCHESTRATOR_OBSERVABILITY_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/ORCHESTRATOR_OBSERVABILITY_STRATEGY.md)
  for the recommended workflow

## Recommended Improvement Loop

1. Add or isolate a failing metamorphic family.
2. Reproduce it first in a narrow lane:
   - corpus slice
   - one structural substitution lane
   - one contextual family
   - one `derive` family
3. During local iteration, run:
   - touched unit tests
   - `fast`
4. Fix the engine or planner narrowly.
5. Every few iterations, rerun `guardrail`.
6. Rerun the relevant `pressure` lane when the change touches normalization,
   orchestration, or deep composed traffic.
7. If the change touches broad orchestration or hot-path matching, rerun
   `embedded` and compare elapsed time, not just pass/fail.
8. Promote the new family into an embedded corpus once the behavior is stable.

This loop is intentionally not the same as running full CI every time.

## Validation Cadence: When To Run `make ci`

`make ci` is a closure step, not the default inner-loop step for engine work.

Running it on every small engine iteration is usually too expensive and slows
down the campaign without improving decision quality.

Use this cadence instead:

### Short loop: every local iteration

Run only the cheapest validations that match the change:

- touched unit tests
- the relevant metamorphic slice
- the relevant embedded corpus if the change is broad enough

This is the default iteration loop.

### Medium loop: every few retained iterations

Run a broader validation pass after roughly `3-5` retained iterations, or
earlier if the change is more structural.

Typical choices:

- `make engine-scorecard`
- `make engine-scorecard-pressure` when normalization or orchestration changed
- `make ci` when the campaign has accumulated enough retained changes

### Full closure loop

Run `make ci` when:

- a batch of changes is ready to be considered stable
- the work touched shared orchestration, core routing, or broad engine behavior
- you are about to close the campaign, commit, or hand off the result

## Why This Cadence Is Correct

The purpose of the cadence is to preserve fast iteration without losing global
safety.

- the short loop keeps development fast
- the medium loop catches campaign-level drift early
- the full loop ensures the global repository contract still holds

This avoids two bad extremes:

- running `make ci` every turn and barely iterating
- never running `make ci` and discovering integration breakage too late

## Practical Rule

For engine campaigns:

- do not run `make ci` on every micro-iteration
- do run it periodically during the campaign
- always run it before treating the work as truly closed

This matters because not every engine fix deserves promotion into the heaviest benchmark immediately.

When the change affects derive routing, the loop should also include the derive
contract corpus, not only generic simplify/equivalence suites.

## Metrics We Should Treat As First-Class

For simplification/equivalence:

- `failed`
- `timeouts`
- `numeric_only`
- `inconclusive`
- `nf_convergent`
- `proved_symbolic`

For `derive`:

- `derived`
- `unsupported`
- `not_equivalent`
- `mean_step_count`
- `long_path_rate`

For operational robustness:

- `stack overflow`
- benchmark elapsed time by lane
- hotspot slices that regress sharply even if they still pass

The policy should be:

- `failed` must stay at `0` in all promoted suites
- `timeouts` must trend down in `strict`
- `numeric_only` should trend down or stay justified
- `unsupported` in `derive` should only grow if we intentionally add frontier cases
- `mean_step_count` should not drift upward accidentally
- stack overflows are immediate blockers in promoted guardrail lanes
- runtime blowups on previously cheap slices must be treated as regressions, not
  as acceptable collateral

## Why `NF-convergent` Matters

`NF-convergent` is not just a benchmark vanity metric.

When two equivalent expressions converge to the same normal form through the
main simplify pipeline, the engine gains real capabilities:

- more deterministic output
- more stable downstream matching
- fewer expensive `difference -> simplify_to_zero` fallback proofs
- less reliance on `proved-symbolic` as the primary path
- better reuse in solver, derive, factoring, and contextual wrappers
- better odds that equivalent traffic hits the same cache / canonical route

In practice this means the engine is not merely proving equivalence after the
fact; it is learning to represent equivalent math the same way.

That improves user-facing quality too:

- fewer “same meaning, different shape” surprises
- fewer branchy orchestrator routes
- less benchmark traffic that only passes because a late symbolic proof bails it out

## Why `NF-convergent` Is Not The Top-Level Goal

Maximizing `NF-convergent` blindly is a mistake.

`proved-symbolic` is still valuable. It is the engine's safety net when:

- a shared normal form is not yet stable
- a family is semantically solved but not canonically aligned
- forcing a common form would introduce brittle special-casing

The wrong optimization pattern is:

- add a broad shortcut
- move a few cases from `proved-symbolic` to `NF-convergent`
- silently degrade runtime, reopen recursion, or fragment other families

That is not a real improvement.

The right interpretation is:

- `NF-convergent` is a quality multiplier for reusable structural families
- `proved-symbolic` is an acceptable holding state when the shared normal form is
  not mature enough
- some families should remain symbolic until a stable common representation exists

## Automation Priority Order

The automation should optimize engine value in this order:

1. keep `failed` at `0`
2. eliminate stack overflows and visible timeouts
3. preserve or improve runtime on promoted lanes
4. increase `NF-convergent` on reusable structural families
5. reduce `proved-symbolic` when that does not harm the first four goals
6. only then optimize local aesthetics of the output

This is intentionally lexicographic, not additive.

For example:

- a change that removes 1 timeout and loses 2 NF cases can still be a win
- a change that gains 5 NF cases but reopens one stack overflow is a regression
- a change that gains NF only for one narrow anecdote but slows a whole slice is
  a regression

## Intelligent Triage For The Automation Loop

Before writing code, the automation should classify each hotspot into one of
these buckets:

### 1. Functional gap

Symptoms:

- `failed`
- `timeout`
- `stack overflow`
- `numeric-only` where symbolic should be possible

Preferred actions:

- engine rule
- orchestrator/root shortcut
- robustness guard
- harness hint only if the engine already has the correct semantics and the
  bottleneck is classification overhead

This bucket has the highest priority.

### 2. Normal-form gap with a stable common representation

Symptoms:

- `proved-symbolic`
- both sides are already expressible through one reusable canonical shape
- direct `cas_cli --release` checks show that shared shape is stable

Preferred actions:

- narrow canonicalization
- anchor-partner shortcut
- shared target builder for a small structural family

This is where `NF-convergent` work is worth the effort.

### 3. Normal-form gap without a stable common representation

Symptoms:

- `proved-symbolic`
- every attempted common form decomposes differently on the two sides
- the “fix” requires ad hoc rewrites or broad shortcuts

Preferred actions:

- keep as `proved-symbolic` for now
- document the family
- revisit only when a genuine common representation appears

This bucket should not dominate automation time.

### 4. Harness-only classification gap

Symptoms:

- direct engine or CLI already resolves the family cheaply
- metamorphic child still spends the budget and times out

Preferred actions:

- cheap child matcher
- textual anchor matcher
- narrow metamorphic hint

This is valid work, but it should be clearly labeled as harness improvement, not
core-engine normalization progress.

## Retention Criteria For A Candidate Change

The automation should keep a candidate only if all of these hold:

- the touched unit regressions pass
- the relevant scorecard lane improves or at least does not regress materially
- promoted corpora stay green
- the change benefits a reusable family, not just a single anecdotal case
- the resulting path is understandable enough to maintain

The automation should reject a candidate if any of these happen:

- `embedded` or guardrail lanes regress
- runtime on a previously cheap slice grows sharply
- a broad shortcut reopens recursion or ping-pong behavior
- the improvement only changes printed order or one isolated benchmark anecdote
- the change pushes the system toward benchmark-specific overfitting

## Practical Policy For Engine Campaigns

A good campaign should usually look like this:

1. remove hard blockers first:
   - failures
   - stack overflows
   - timeouts
2. then attack high-volume `proved-symbolic` clusters that share a reusable
   normal form
3. stop when the remaining mismatches are mostly “semantic proof is fine, common
   NF is not clean yet”
4. move to the next family instead of forcing brittle canonicalization

That is the key strategic point:

- `NF-convergent` should be maximized where it increases determinism and reuse
- it should not be maximized at any cost
- the automation should prefer robust, composable canonical families over local
  benchmark gaming

## How To Extend The System

Next expansions that make sense:

- add an equation metamorphic lane from [metamorphic_equation_tests.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs)
- split `derive` scorecard into:
  - reachability
  - equivalence floor
  - didactic path quality
- add baseline comparison in CI so we fail on true regressions, not just raw test failures
- promote recurring hotspot slices into named corpora rather than keeping them as ad hoc notes
- add explicit runtime-budget alerts for suites that remain semantically green
  but become materially slower

## Commands

```bash
make engine-fast
make engine-scorecard
make engine-scorecard-pressure
make engine-scorecard-full
```

Or directly:

```bash
python3 scripts/engine_improvement_scorecard.py --profile fast
python3 scripts/engine_improvement_scorecard.py --profile guardrail
python3 scripts/engine_improvement_scorecard.py --profile pressure
python3 scripts/engine_improvement_scorecard.py --profile full
```
