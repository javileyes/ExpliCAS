# Derive Next-Level Roadmap

This document is the operational roadmap for turning `derive` into a strong,
planner-driven educational feature.

It complements, and partly supersedes in practical detail:

- `/Users/javiergimenezmoya/developer/math/docs/DERIVE_ROADMAP.md`
- `/Users/javiergimenezmoya/developer/math/docs/DERIVE_DIDACTIC_AUDIT.md`

## Short Answer

Yes: `derive` can become a powerful feature.

But not by trying to become a universal theorem prover or by doing blind graph
search over all rewrites.

The realistic target is:

- strong target recognition
- reusable algebraic transition families
- bounded multi-step planning
- didactic path selection
- strict quality and budget controls

That is achievable with the current architecture, because the codebase already
has:

- a target classifier
- a strategy registry
- many target-aware family rewrites
- derive-specific metrics and corpus tests
- a didactic audit layer

The main missing piece is not algebraic power. It is a planner that can chain
more than one or two safe intermediate states on purpose.

## Current State

Current implementation entrypoints:

- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs`
- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/strategy.rs`
- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/target_classifier.rs`

Current strengths:

- target classification exists
- strategy ordering exists
- many families already expose target-aware rewrites
- derive-specific corpus and shape-budget tests exist
- didactic audit exists
- output already distinguishes `derived`, `equivalent but unsupported`, and
  `not equivalent`

Current core limitation:

- `derive` is still mostly a shallow planner
- it tries a small ordered set of strategies
- each strategy usually produces one stage
- some strategies produce a fixed two-stage chain
- it does not yet explore short alternative paths through intermediate states

So today `derive` is strong at:

- direct family hits
- short curated chains
- pedagogically obvious one-step rewrites

And weaker at:

- exact user targets that need 3-5 intentional moves
- choosing between multiple plausible routes
- recovering from a locally-correct but globally-unhelpful first move

## Goal

Given two equivalent expressions, `derive` should be able to:

1. Prove the target is semantically reachable.
2. Find a short path of mathematically meaningful intermediate states.
3. Prefer the most teachable path, not merely the first valid path.
4. Render a clean step-by-step explanation without engine noise.

## Non-Goals

This roadmap explicitly rejects:

- blind BFS over all rewrites
- unbounded expression graph search
- fake derive-only transformations that do not correspond to real engine rules
- maximizing step count for its own sake
- replacing the existing strategy/family model with a generic prover

## Core Thesis

The next leap for `derive` is a bounded path planner over trusted transition
families.

In other words:

- not "try one strategy and stop"
- not "search everything"
- but "search a small number of good next moves, score them, and keep only the
  best short paths"

## Architectural Direction

### 1. Keep The Target Classifier

The target classifier remains the front door for planning.

Responsibilities:

- classify target family
- extract family parameters when relevant
- restrict which transitions are allowed
- help order the search

This remains centered in:

- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/target_classifier.rs`

### 2. Turn Strategies Into Transition Providers

Today many `run_*_stage(...)` functions act like "attempt one rewrite and return
one answer".

The planner needs a more general abstraction:

```text
transition provider:
- input expression
- optional target profile
- returns 0..n candidate transitions
```

Each transition should carry:

- resulting expression
- visible steps for that move
- strategy/family label
- cost metadata
- safety metadata

Suggested shape:

- `derive/transitions.rs`
- `derive/providers/*`

Possible data structure:

```rust
struct DeriveTransition {
    expr: ExprId,
    steps: Vec<crate::Step>,
    strategy: DeriveStrategy,
    local_cost: u32,
    semantic_distance_hint: u32,
}
```

### 3. Introduce A Bounded Planner

Add a planner module:

- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/planner.rs`

The planner should be:

- bounded
- deterministic
- family-aware
- budget-aware
- didactic-aware

Recommended search style:

- beam search or best-first search
- small depth cap, for example `max_depth = 4` initially
- small frontier cap, for example `beam_width = 8`
- hard timeout / node-expansion budget

Node shape:

```rust
struct DerivePlanNode {
    expr: ExprId,
    path: Vec<DeriveStageOutput>,
    total_cost: u32,
    depth: u8,
}
```

### 4. Add State Deduplication

Without visited-state control, a bounded planner will still waste budget.

The planner should deduplicate by a robust state signature:

- canonicalized expression surface where safe
- semantic equivalence hints where cheap
- target-family-sensitive normalization

Important:

- deduplication should be conservative
- it must never collapse states that are pedagogically distinct if that would
  hide a useful path

### 5. Add Path Scoring

The planner must choose the best path, not only any valid path.

The scoring function should prefer:

- fewer visible steps
- fewer generic `simplify` hops
- rules with explicit mathematical names
- transitions that reduce semantic distance to the target
- stable paths with low rewrite noise

The scoring function should penalize:

- loops
- `expand -> factor -> expand` churn
- repeated cosmetic rewrites
- paths that only differ by canonicalization noise
- long chains of low-information steps

### 6. Keep Direct Fast Paths

Do not route everything through the planner.

The execution order should be:

1. exact presentational match
2. current direct planner
3. fixed two-stage chains
4. bounded multi-step planner
5. equivalent-but-unsupported fallback

This preserves current speed and stability for the easy cases.

## Workstreams

### Workstream A. Planner Infrastructure

Deliverables:

- transition abstraction
- planner node type
- visited-state signature
- scoring function
- execution budget struct

Definition of done:

- planner can chain existing transitions without changing rule semantics
- planner can return a valid 3-step path on synthetic tests
- planner remains deterministic across repeated runs

### Workstream B. Transition Extraction

Refactor current stage runners so they can act both as:

- direct derive fast paths
- candidate generators for the planner

Priority families:

- simplify
- collect
- factor
- expand
- trig expand / contract
- log expand / contract
- fraction combine / expand / decompose

Definition of done:

- at least 80 percent of current direct strategies can be invoked through the
  shared transition interface

### Workstream C. Search Gating

The planner must never become the default source of complexity for trivial
cases.

Rules:

- only activate search when direct strategies fail
- only allow families relevant to the classified target
- keep low depth caps initially
- stop on first sufficiently good path, not exhaustive optimality

Definition of done:

- planner improves coverage on real unsupported cases without regressing median
  latency for direct hits

### Workstream D. Didactic Path Quality

Reaching the target is not enough.

The planner must integrate with the didactic layer so that:

- step labels remain accurate
- substeps are only kept when they explain a non-obvious jump
- repeated structural noise is suppressed
- final paths read as intentional derivations, not replayed engine traces

Definition of done:

- derive didactic audit remains green
- no new family introduces generic filler substeps

### Workstream E. Testing And Metrics

`derive_pairs.csv` remains representative and small.

Family generality must continue to live in:

- local tabular tests
- family-specific unit tests
- derive didactic audit

New metrics to add:

- planner activation rate
- planner success rate
- planner fallback rate
- planner mean explored nodes
- planner timeout rate
- unsupported-equivalent count by family
- average visible steps for planner-produced paths

## Delivery Phases

### Phase 0. Baseline And Guardrails

Status:

- mostly in place

Tasks:

- keep `derive_pairs.csv` representative
- keep shape-cluster guardrails
- keep didactic audit active
- add a curated set of real multi-step unsupported cases

Exit criteria:

- there is a stable baseline of unsupported-but-equivalent examples requiring
  3+ moves

### Phase 1. Transition Layer

Tasks:

- define `DeriveTransition`
- define provider interface
- adapt `simplify`, `collect`, `factor`, `expand`
- reuse existing `DeriveStageOutput`

Exit criteria:

- direct derive and transition generation share the same family logic

### Phase 2. Planner MVP

Tasks:

- implement bounded planner
- run it only after direct planner failure
- support depth up to 3 or 4
- restrict search to algebraically central families first

Target families for MVP:

- simplify
- collect
- factor
- expand
- fraction combine / expand

Exit criteria:

- at least a first batch of currently unsupported equivalent pairs becomes
  derived

### Phase 3. Family Expansion

Tasks:

- enable planner chaining for trig
- enable planner chaining for logs
- enable planner chaining for hyperbolic identities where useful
- add family-specific path scoring tweaks only when justified

Exit criteria:

- planner covers mixed-family chains such as `simplify -> expand trig -> collect`

### Phase 4. Didactic Ranking

Tasks:

- compare alternative successful paths
- choose the path with best pedagogical score
- suppress noisier valid paths

Exit criteria:

- cases with multiple valid routes consistently choose the cleaner one

### Phase 5. Performance Hardening

Tasks:

- add budgets and node caps
- add profiling hooks
- add timeout-safe fallback behavior
- track planner-only metrics in CI output

Exit criteria:

- planner stays within agreed latency budgets

## Coverage Snapshot 2026-04-04

This section records the current practical audit of `derive`, not only the
intended architecture.

Evidence used:

- corpus and metrics:
  - [/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)
  - [/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_contract_tests.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_contract_tests.rs)
- implementation shape:
  - [/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/strategy.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/strategy.rs)
  - [/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/target_classifier.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/target_classifier.rs)
  - [/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs)
- spot checks outside the curated corpus

Current corpus summary:

- `derived=197`
- `unsupported=0`
- `not_equivalent=1`
- `mean_step_count=1.00`
- `long_path_rate=0.000`

Important interpretation:

- this is strong evidence that the current curated perimeter is healthy
- it is not evidence that `derive` is complete outside that perimeter
- the system is still family-driven, not a general theorem-proving derivation
  engine

### Families Well Covered

| Family | Evidence | Current assessment |
|---|---|---|
| Trig expand / contract | 31 `expand trig`, 27 `contract trig` corpus cases | Strong for double-angle, multiple-angle, many product-to-sum and reciprocal identities |
| Hyperbolic rewrites | 20 `rewrite hyperbolics` corpus cases | Strong and in some areas more complete than the trig analogue |
| Fraction algebra | 11 `expand fraction`, 10 `combine fraction`, 5 `cancel fraction`, 3 `nested fraction`, 3 `rationalize` | Strong for school-style algebraic transformations and exact target matches |
| Logs and exponentials | 7 `expand_log`, 5 `contract logs`, 4 `rewrite exponentials` | Good support when the target belongs to a recognized rewrite family |
| Finite telescoping / prep families | 5 `finite sums/products`, 4 `integrate prep`, 5 `solve prep` | Good family-local coverage with concise educational traces |

Representative successful chains:

- `derive 2*cos(2*x)*sin(x), 4*cos(x)^2*sin(x)-2*sin(x)`
  - 2 clean steps: product-to-sum, then triple-angle expansion
- `derive 2*sinh(2*x)*sinh(x)+a, 4*cosh(x)^3-4*cosh(x)+a`
  - 2 clean steps: hyperbolic product-to-sum, then hyperbolic triple-angle
- `derive 1/(sqrt(x)-1) - (sqrt(x)+1)/(x-1), 0`
  - 2 clean steps: rationalize, then subtract equal expressions

### Families Missing Or Clearly Incomplete

| Gap | Representative example | Assessment |
|---|---|---|
| Trig sum-to-product | `derive cos(x)-cos(y), -2*sin((x+y)/2)*sin((x-y)/2)` | Real coverage gap; currently fails |
| Trig sum-to-product | `derive sin(x)+sin(y), 2*sin((x+y)/2)*cos((x-y)/2)` | Real coverage gap; currently fails |
| Trig sum-to-product | `derive cos(x)+cos(y), 2*cos((x+y)/2)*cos((x-y)/2)` | Real coverage gap; currently fails |
| Inverse trig with branch-sensitive composition | `derive arctan(a)+arctan(b), arctan((a+b)/(1-a*b))` | Not just a missing rewrite; needs branch/domain modeling |

Important contrast:

- the hyperbolic analogue `derive sinh(x)+sinh(y), 2*sinh((x+y)/2)*cosh((x-y)/2)` does work
- this suggests a concrete missing trig family, not a generic planner failure

### Families That Reach The Target But Teach Poorly

| Problem type | Representative example | Why it is bad |
|---|---|---|
| Oscillation / loopiness | `derive (sin(x)+cos(x))^2, 1+sin(2*x)` | Reaches the target, but emits a 17-step trace with repeated double-angle expand/contract churn |
| Opaque `simplify` jump | `derive sin(x)^4, (3-4*cos(2*x)+cos(4*x))/8` | Correct result, but only a generic `Simplify` step instead of a reduction-of-powers derivation |
| Cosmetic no-op step | `derive sinh(x)+sinh(y), 2*sinh((x+y)/2)*cosh((x-y)/2)` | Path includes a redundant `Canonicalize Multiplication` step after the meaningful rewrite |

These are not correctness failures. They are didactic-quality failures, and
they matter because `derive` is explicitly an educational feature.

### Priority Ranking

Recommended development order from highest ROI to lowest:

1. Add the missing trig sum-to-product family in
   [/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/trig.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/trig.rs).
   Reason:
   elementary identities, clear parity with existing hyperbolic support, high
   user value, low conceptual risk.
2. Add loop / oscillation suppression in
   [/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs).
   Reason:
   current planner quality is good on many cases, but traces can still become
   pedagogically unacceptable.
3. Promote reduction-of-powers identities out of generic `simplify`.
   Candidate family:
   `sin^4`, `cos^4`, `sin^2*cos^2`, and nearby half-angle chains.
4. Treat branch-sensitive inverse-trig composition as a separate workstream.
   Reason:
   this is a deeper semantic/domain task and should not block the simpler trig
   family gap.

### Operational Rule

For the next iterations, improvements should be classified into exactly one of
these buckets:

- reachability gap
- didactic-quality gap
- domain/branch modeling gap

This prevents mixing easy high-ROI family additions with much more expensive
semantic work such as inverse-trig branch correctness.

## Immediate Next Steps

This is the recommended implementation order for the next iterations:

1. Add a small set of real unsupported equivalent pairs that need 3-4 moves.
2. Extract a transition interface from the existing stage functions.
3. Implement a planner MVP behind a narrow feature gate inside `derive`.
4. Start with `simplify`, `collect`, `factor`, `expand`, and fractions only.
5. Add planner metrics to the derive contract test output.
6. Extend to trig and log families after the MVP is stable.

## Acceptance Criteria For "Derive Is Strong"

We should only claim `derive` is a strong feature when all of these are true:

- direct targets still resolve quickly
- multi-step supported-equivalent coverage is materially higher than today
- didactic quality does not regress
- the corpus remains representative, not inflated
- the planner does not rely on ad hoc one-off hacks
- unsupported cases become concentrated in genuinely hard families, not in
  obvious 3-step chains

## Risks

Main risks:

- over-searching and blowing budgets
- adding planner complexity without improving didactic quality
- duplicating direct-strategy logic instead of reusing it
- hiding regressions behind semantic equivalence while path quality worsens

Mitigations:

- bounded planner only
- keep direct fast path first
- planner metrics in tests
- didactic audit as a release gate
- family-local tabular tests instead of inflating the corpus

## Working Rule

From this point on, every derive improvement should answer these questions:

1. Is this a new transition family, or only a new surface variant?
2. Should this live in the corpus, or in a family-local table test?
3. Does this improve reachability, didactic quality, or both?
4. Can this be expressed as a reusable transition instead of a one-off hack?
5. Does this move us toward bounded multi-step planning?

If the answer to the last question is "no", the change is probably not on the
critical path for making `derive` truly powerful.
