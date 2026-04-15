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
7. Promote the new family into an embedded corpus once the behavior is stable.

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
