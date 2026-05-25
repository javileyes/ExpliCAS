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

## Coupling With Engine Improvement

`derive` should not evolve as a separate feature track that only receives
manual examples after the engine is done.

The strongest strategy is bidirectional:

- engine -> derive:
  every retained mathematical engine/corpus improvement should be checked for a
  minimal `source -> target` derive shadow case
- derive -> engine:
  every unsupported-but-equivalent derive case should be classified to decide
  whether the gap belongs in planner/strategy/didactic code or in a reusable
  engine transition

This prevents two failure modes:

- the engine learns to prove or simplify an identity, but `derive` still renders
  it as a magical one-step jump or cannot target the desired form
- `derive` grows one-off routes that do not correspond to real engine
  transformations and therefore do not help simplification, equivalence, or
  contextual wrappers

Operationally, any new stable engine family should answer:

1. Is there a natural expanded -> contracted derive direction?
2. Is there a natural contracted -> expanded derive direction?
3. Does the route need an intentional 2-4 step path rather than one generic
   simplify step?
4. Is the case small enough to live in `derive_pairs.csv`, or should broader
   variants stay in family-local tests?

Any derive miss should answer:

1. Is the target family not classified?
2. Is the family classified but missing a strategy or transition provider?
3. Does the engine already prove equivalence but lack a reusable visible
   transition?
4. Is the only blocker branch/domain semantics, in which case it should be
   tracked separately and not patched with a fake derivation?

The best derive improvements should therefore either consume a real engine
capability or create pressure for one. They should not merely inflate the derive
corpus with easy one-step variants.

### Coupling With Calculus

`derive` is the educational derivation command. It is not the derivative
calculus command.

Even so, calculus work should inherit the same didactic standards:

- a derivative, limit, or integral result should not hide a meaningful
  transformation behind a generic one-step simplification
- product rule, quotient rule, chain rule, supported table integration, and
  conservative limit pre-simplification should be visible when they are the
  mathematical reason the result changes
- simplification after a calculus rule should reuse engine steps where possible
  instead of inventing calculus-only presentation shortcuts
- post-calculus presentation may improve the final public form of a correct
  derivative, limit, or integral, but it should be treated as display-facing
  calculus cleanup, not as a hidden derive route or a substitute for explaining
  the mathematical transformation
- broken highlights or magical substeps in calculus output should be treated as
  real didactic defects, not cosmetic noise

The calculus campaign has moved from isolated vertical slices toward
matrix-driven generalization. That changes the derive bridge criterion:
derive shadow cases should be selected from reusable calculus matrix cells, not
from every successful derivative, limit, or integral example.

The bridge also works the other way:

- calculus output often exposes algebraic target forms that `derive` should
  eventually explain
- post-calculus presentation output can create a derive shadow case only when
  the displayed form reveals a reusable algebraic transition, not merely because
  two strings look different
- unsupported `derive` cases can expose simplification gaps that block clean
  derivative or integral results
- a calculus family should add a derive shadow case only when it reveals a
  reusable algebraic transformation a user would plausibly ask `derive` to
  explain

Do not turn calculus into a second derive planner, and do not turn derive into a
calculus engine. Share route quality, target-awareness, and didactic pressure;
keep command semantics separate.

### Equivalent-But-Not-Derived Pressure

The current `derive_contract` can look green while still under-testing real
target-form reachability.

The scorecard should grow a diagnostic pressure lane that samples identities
already proved by simplify/metamorphic/embedded coverage and asks:

- can `derive` bridge the same pair?
- is the result unsupported even though equivalence is known?
- does it succeed only through a generic or magical one-step simplification?
- does the visible path stay within a small, teachable step budget?

This lane should initially be diagnostic, not a hard gate. Its purpose is to
expose families where the engine knows the algebra but `derive` does not yet
teach the transformation.

Current implementation status:

- the scorecard now includes `derive_shadow_pressure` in the `guardrail` and
  `full` profiles
- the lane samples representative rows from `identity_pairs.csv` plus selected
  minimal root pairs from `embedded_equivalence_context_corpus.csv`
- it reports `sampled`, `derived`, `unsupported`, `not_equivalent`,
  `single_step_successes`, and `multi_step_successes`
- nonzero misses are diagnostic pressure for future cycles, not an immediate
  support gate

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
- equivalent-but-not-derived count by family
- magical-one-step count for transformations that should have visible substeps
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
| Inverse trig with branch-sensitive composition | `derive arctan(a)+arctan(b), arctan((a+b)/(1-a*b))` | Not just a missing rewrite; needs branch/domain modeling |

Recent change:

- the trig sum-to-product family for arbitrary arguments is now covered directly in
  `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/trig.rs`
- examples like `derive cos(x)-cos(y), -2*sin((x+y)/2)*sin((x-y)/2)` now resolve in
  one `expand trig` step
- common trig reduction-of-powers targets now resolve directly in
  `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/trig.rs`
- examples like `derive sin(x)^4, (3-4*cos(2*x)+cos(4*x))/8`,
  `derive cos(x)^4, (3+4*cos(2*x)+cos(4*x))/8`, and
  `derive sin(x)^2*cos(x)^2, (1-cos(4*x))/8` now resolve in one
  `expand trig` step with `Power Reduction Identity`
- trig binomial-square identities now resolve directly in
  `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/trig.rs`
- examples like `derive (sin(x)+cos(x))^2, 1+sin(2*x)` and
  `derive (sin(x)-cos(x))^2, 1-sin(2*x)` now resolve in one
  `expand trig` step with `Trig Square Identity`
- sixth-power trig reductions now resolve directly in
  `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/trig.rs`
- examples like `derive sin(x)^6, (10-15*cos(2*x)+6*cos(4*x)-cos(6*x))/32` and
  `derive cos(x)^6, (10+15*cos(2*x)+6*cos(4*x)+cos(6*x))/32` now resolve in one
  `expand trig` step with `Power Reduction Identity`
- eighth-power trig reductions now resolve directly in
  `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/trig.rs`
- examples like `derive sin(x)^8, (35-56*cos(2*x)+28*cos(4*x)-8*cos(6*x)+cos(8*x))/128` and
  `derive cos(x)^8, (35+56*cos(2*x)+28*cos(4*x)+8*cos(6*x)+cos(8*x))/128` now resolve in one
  `expand trig` step with `Power Reduction Identity`
- tenth-power trig reductions now resolve directly in
  `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/trig.rs`
- examples like `derive sin(x)^10, (126-210*cos(2*x)+120*cos(4*x)-45*cos(6*x)+10*cos(8*x)-cos(10*x))/512` and
  `derive cos(x)^10, (126+210*cos(2*x)+120*cos(4*x)+45*cos(6*x)+10*cos(8*x)+cos(10*x))/512` now resolve in one
  `expand trig` step with `Power Reduction Identity`
- higher even-power trig reductions now resolve directly in
  `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/trig.rs`
- representative examples like
  `derive sin(x)^24, (1352078-2496144*cos(2*x)+1961256*cos(4*x)-1307504*cos(6*x)+735471*cos(8*x)-346104*cos(10*x)+134596*cos(12*x)-42504*cos(14*x)+10626*cos(16*x)-2024*cos(18*x)+276*cos(20*x)-24*cos(22*x)+cos(24*x))/8388608` and
  `derive cos(x)^24, (1352078+2496144*cos(2*x)+1961256*cos(4*x)+1307504*cos(6*x)+735471*cos(8*x)+346104*cos(10*x)+134596*cos(12*x)+42504*cos(14*x)+10626*cos(16*x)+2024*cos(18*x)+276*cos(20*x)+24*cos(22*x)+cos(24*x))/8388608` now resolve in one
  `expand trig` step with `Power Reduction Identity`
- direct hyperbolic angle-sum/difference expansions now stay on the `expand`
  path instead of being hijacked by the exponential bridge matcher
- examples like `derive sinh(x+y), sinh(x)*cosh(y)+cosh(x)*sinh(y)` now resolve
  in one `expand` step with `Hyperbolic Angle Sum/Difference Identity`
- trig `product-to-sum -> triple-angle` chains now stay inside `expand trig`
  instead of falling back to `planner`
- examples like `derive 2*sin(2*x)*sin(x), 4*cos(x)-4*cos(x)^3`,
  `derive 2*cos(2*x)*cos(x), 4*cos(x)^3-2*cos(x)`, and
  `derive 2*cos(2*x)*sin(x), 4*cos(x)^2*sin(x)-2*sin(x)` now resolve in two
  `expand trig` steps: `Product-to-Sum Identity` + `Triple Angle Expansion`
- the same `bridge -> expand trig` chain now also works with additive
  passthrough terms
- examples like `derive 2*sin(2*x)*sin(x)+a, 4*cos(x)-4*cos(x)^3+a` and
  `derive 2*cos(2*x)*sin(x)+a, 4*cos(x)^2*sin(x)-2*sin(x)+a` now resolve in
  two `expand trig` steps instead of falling back to noisy `simplify`
- the same cleanup now applies to exact hyperbolic `product-to-sum` inside
  additive passthrough terms
- hyperbolic `product-to-sum -> triple-angle` chains now stay inside `expand`
  instead of collapsing into a one-step semantic jump
- examples like `derive 2*sinh(2*x)*cosh(x), 4*sinh(x)+4*sinh(x)^3`,
  `derive 2*sinh(2*x)*sinh(x), 4*cosh(x)^3-4*cosh(x)`, and
  `derive 2*sinh(2*x)*sinh(x)+a, 4*cosh(x)^3-4*cosh(x)+a` now resolve in two
  `expand` steps: `Hyperbolic Product-to-Sum Identity` +
  `Hyperbolic Triple-Angle Identity`
- the bounded planner now explores `contract logs` as a trusted transition,
  which closes a real mixed-family gap for grouped-power log targets
- examples like `derive ln(x^2)+ln(y^2), ln((x*y)^2)`,
  `derive 2*ln(abs(x))+2*ln(abs(y)), 2*ln(abs(x*y))`, and
  `derive 2*log(b,x)+2*log(b,y), log(b,(x*y)^2)` now resolve via `planner`
  with a single visible step `Contraer logaritmos`
- equal-weight sine/cosine phase-shift identities now resolve directly
  instead of falling back to `unsupported` or noisy expansion/planner paths
- examples like `derive sin(x)+cos(x), sqrt(2)*sin(x+pi/4)`,
  `derive sin(x)-cos(x), sqrt(2)*sin(x-pi/4)`,
  `derive sqrt(2)*sin(x+pi/4), sin(x)+cos(x)`, and
  `derive sqrt(2)*cos(x-pi/4), sin(x)+cos(x)` now resolve in one
  `contract trig` or `expand trig` step with `Phase Shift Identity`
- the same phase-shift family now also handles a shared multiplicative factor
- examples like `derive 2*sin(x)+2*cos(x), 2*sqrt(2)*sin(x+pi/4)` and
  `derive 2*sqrt(2)*sin(x+pi/4), 2*sin(x)+2*cos(x)` now resolve in one
  direct trig step instead of falling back to `unsupported` or a 5-step
  expansion
- the same phase-shift family now also works through additive passthrough terms
- examples like `derive sin(x)+cos(x)+a, sqrt(2)*sin(x+pi/4)+a` and
  `derive 2*sqrt(2)*sin(x+pi/4)+a, 2*sin(x)+2*cos(x)+a` now resolve in one
  direct trig step instead of falling back to `unsupported`
- the bounded planner now explores additive local trig bridges as reusable
  intermediate transitions instead of only when they already hit the final
  target
- examples like `derive sin(x)+cos(x)+sin(y)+cos(y),
  sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)` now resolve directly via
  `expand trig` in two `Phase Shift Identity` steps, and the reverse
  `derive sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4),
  sin(x)+cos(x)+sin(y)+cos(y)` is handled directly by `expand trig` with the
  same two nominal steps
- exact unequal-weight phase-shift identities at special angles are now direct
- examples like `derive 2*sin(x)+2*sqrt(3)*cos(x), 4*sin(x+pi/3)`,
  `derive 4*sin(x+pi/3), 2*sin(x)+2*sqrt(3)*cos(x)`,
  `derive sqrt(3)*sin(x)+cos(x), 2*sin(x+pi/6)`, and
  `derive 2*sin(x+pi/6), sqrt(3)*sin(x)+cos(x)` now resolve in one direct
  `contract trig` / `expand trig` step with `Phase Shift Identity`
- the fully general exact unequal-weight phase shift is now direct too
- examples like `derive 3*sin(x)+4*cos(x), 5*sin(x+arctan(4/3))` and
  `derive 5*sin(x+arctan(4/3)), 3*sin(x)+4*cos(x)` now resolve in one direct
  `contract trig` / `expand trig` step with `Phase Shift Identity`
- the same exact phase shift with additive passthrough is now direct too
- examples like `derive 3*sin(x)+4*cos(x)+a, 5*sin(x+arctan(4/3))+a` and
  `derive 5*sin(x+arctan(4/3))+a, 3*sin(x)+4*cos(x)+a` now resolve in one
  direct `contract trig` / `expand trig` step with `Phase Shift Identity`
- the symbolic unequal-weight phase shift now resolves directly too, including
  simple passthrough and repeated-pair planner cases
- direct normalization between equivalent exact shifted trig terms is now
  direct too
- examples like `derive sqrt(2)*sin(x+pi/4), sqrt(2)*cos(x-pi/4)` and
  `derive sqrt(2)*sin(x+pi/4)+a, sqrt(2)*cos(x-pi/4)+a` now resolve in one
  direct `contract trig` step with `Phase Shift Identity`
- the same direct normalization now also covers the general unequal-weight
  shifted-term case, including additive passthrough
- examples like `derive 5*sin(x+arctan(4/3)), 5*cos(x-arctan(3/4))` and
  `derive 5*sin(x+arctan(4/3))+a, 5*cos(x-arctan(3/4))+a` now resolve directly
  in one `contract trig` step instead of falling back to `planner`
- the next ROI after this exact subfamily is no longer more linear phase shift;
  the best remaining candidates are fresh unsupported-equivalent families
  outside phase shift, or cosmetic tail cleanup in direct routes that still
  teach more steps than necessary
- grouped `contract logs` targets that were previously planner-only are now
  direct too
- examples like `derive ln(x^2)+ln(y^2), ln((x*y)^2)`,
  `derive 2*ln(abs(x))+2*ln(abs(y)), 2*ln(abs(x*y))`, and
  `derive 2*log(b,x)+2*log(b,y), log(b,(x*y)^2)` now resolve with
  `strategy: "contract logs"` in one direct `Contraer logaritmos` step
- the grouped `expand_log` reverse direction now resolves directly too instead
  of failing equivalence proof
- examples like `derive ln((x*y)^2), ln(x^2)+ln(y^2)`,
  `derive 2*ln(abs(x*y)), 2*ln(abs(x))+2*ln(abs(y))`, and
  `derive log(b,(x*y)^2), 2*log(b,x)+2*log(b,y)` now resolve with
  `strategy: "expand_log"` in one direct `Expandir logaritmos` step
- the same grouped log family now also works through additive passthrough terms
- examples like `derive ln((x*y)^2)+a, ln(x^2)+ln(y^2)+a`,
  `derive 2*ln(abs(x))+2*ln(abs(y))+a, 2*ln(abs(x*y))+a`, and
  `derive 2*log(b,x)+2*log(b,y)+a, log(b,(x*y)^2)+a` now resolve directly in
  one `expand_log` / `contract logs` step instead of failing equivalence proof
- the perfect-square radical rewrite now also works through additive
  passthrough terms
- examples like `derive sqrt(a^2 + 2*a*b + b^2)+c, abs(a+b)+c` now resolve
  directly with `strategy: "rewrite radicals"` instead of falling back to
  generic `simplify`
- trig identities that collapse to `1` now also work through additive
  passthrough terms
- examples like `derive tan(x)*cot(x)+a, 1+a` now resolve directly with
  `strategy: "rewrite trigs"` in one step instead of falling back to generic
  `simplify`
- the consecutive factorial ratio rewrite now also works through additive
  passthrough terms
- examples like `derive (n+1)!/n!+a, n+1+a` now resolve directly with
  `strategy: "rewrite factorials"` in one step instead of falling back to
  generic `simplify`
- `expand odd half power` now also works through additive passthrough terms
- examples like `derive sqrt(x^3)+a, abs(x)*sqrt(x)+a` now resolve directly
  with `strategy: "expand odd half power"` instead of falling back to generic
  `simplify`
- the same odd-half-power family now drops redundant display guards like
  `x^3 ≥ 0` when `x ≥ 0` is already present, so the wire teaches the minimal
  domain requirement for this route
- exact `cancel fraction` rewrites now also work through additive passthrough
  terms instead of falling back to generic `simplify`
- examples like `derive (a^2-b^2)/(a-b)+c, a+b+c` and
  `derive (a^3-b^3)/(a-b)+c, a^2+a*b+b^2+c` now resolve directly with
  `strategy: "cancel fraction"` in one named step

### Families That Reach The Target But Teach Poorly

| Problem type | Representative example | Why it is bad |
|---|---|---|
| Very high even-power reduction still bounded | exponents whose coefficients or denominators exceed current small-integer emission limits | A generic fallback now covers higher even powers beyond degree 22, but extremely large exponents may later want BigInt-backed coefficient emission |

These are not correctness failures. They are didactic-quality failures, and
they matter because `derive` is explicitly an educational feature.

### Priority Ranking

Recommended development order from highest ROI to lowest:

1. Continue cleaning remaining cosmetic tails after meaningful rewrites.
   Candidate family:
   other canonicalization/no-op steps outside the already cleaned direct trig
   and hyperbolic exact/passthrough families.
2. Treat branch-sensitive inverse-trig composition as a separate workstream.
   Reason:
   this is a deeper semantic/domain task and should not block the remaining
   planner and didactic gaps.

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
6. Does this consume a real engine capability, or expose a reusable engine gap?
7. If the corresponding engine family was just improved, is this the minimal
   derive shadow case for that family?

If the answer to the last question is "no", the change is probably not on the
critical path for making `derive` truly powerful.
