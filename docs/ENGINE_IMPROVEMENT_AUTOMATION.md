# Engine Improvement Automation

## Goal
Build one repeatable loop that improves four things together without guessing:

- simplification coverage
- equivalence proving power
- `derive` reachability and didactic quality
- serious real-domain calculus capability that reuses and hardens the
  pre-calculus core

The core idea is simple: every engine improvement campaign should be driven by an
explicit scorecard, not by isolated anecdotes.

This improvement loop is deliberately broader than “add one more rule”.

A real engine improvement may come from:

- a new simplification rule
- a better root/orchestrator shortcut
- a better derive target classifier or planner route
- a generalized calculus capability in differentiation, limits, or integration
- a calculus architecture boundary that separates detection, domain reasoning,
  transformation, verification, presentation, or didactic step construction
- a robustness fix that prevents stack overflow / timeout on expressions the
  engine already handled semantically

The system should reward all of those, but only if they preserve the global
guardrails.

## Primary Strategy: ROI-Directed Iteration Selection

This is the main strategy for automatic engine improvement.

The rest of the campaign policies should derive from it:

- orchestrator observability is the strategy for `observability` iterations
- corpus growth is the strategy for `coverage` and part of `robustness`
- calculus growth is governed by
  [CALCULUS_ENGINE_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/CALCULUS_ENGINE_STRATEGY.md)
- the combination ledger is the strategy for `combination` iterations
- cohesion refactoring is the strategy for structural work on large engine
  files when the goal is safer future iteration rather than immediate
  mathematical coverage:
  [ENGINE_COHESION_REFACTORING_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_COHESION_REFACTORING_STRATEGY.md)

Each iteration should choose one primary investment class before touching code:

- `runtime`
  - reduce hot-path cost or broad no-match traffic
- `coverage`
  - add or unlock mathematically useful families
- `robustness`
  - reduce timeout, loop, overflow, or brittle routing risk
- `observability`
  - improve signal when the next profitable move is still ambiguous
- `calculus`
  - generalize public `diff`, `limit`, or `integrate` behavior while reusing
    the pre-calculus core, preserving real-domain safety, and keeping didactic
    traces explainable
- `combination`
  - revisit a documented `local win / global fail` only when a complementary
    hypothesis is now available

### Expected ROI

The selector should not optimize for local speedup alone.

It should choose the next iteration by expected retained value:

`expected_roi ~= impact * breadth * confidence * retention_probability / implementation_cost`

For work that touches mathematical families, the selector must also score
cross-system value:

```text
cross_system_roi ~= engine_value
                  + derive_bridge_value
                  + calculus_value
                  + precalculus_reuse_value
                  + domain_safety_value
                  + architecture_pressure_value
                  + corpus_reuse_value
                  + didactic_value
                  - runtime_risk
                  - unsoundness_risk
                  - complexity_cost
```

This does not replace the main ROI formula. It prevents the loop from treating
`derive` as a separate afterthought or calculus as a bolt-on feature when the
same engine capability could also become a teachable `source -> target` bridge
or a public `diff`, `limit`, or `integrate` slice.

Interpretation:

- `impact`
  - expected gain in runtime, coverage, or robustness
- `breadth`
  - how much real traffic or mathematical surface area it affects
- `confidence`
  - how clear the causal hypothesis is
- `retention_probability`
  - probability that the change survives the global guardrails
- `implementation_cost`
  - code complexity, risk, and validation cost
- `derive_bridge_value`
  - whether the change exposes a reusable target-form transition for `derive`
- `calculus_value`
  - whether the change improves public `diff`, `limit`, or `integrate`
    behavior in a generalized, policy-backed, testable way
- `precalculus_reuse_value`
  - whether the calculus work reuses or hardens simplification, equivalence,
    domain, or didactic infrastructure instead of bypassing it
- `domain_safety_value`
  - whether assumptions around domain, branch, infinity, or integration
    constants are preserved explicitly
- `architecture_pressure_value`
  - whether the change reduces repeated local calculus shortcuts, clarifies
    ownership of detection/domain/verification/presentation/steps, or makes the
    next family generalization safer without broad behavior churn
- `corpus_reuse_value`
  - whether one minimal case can serve both engine coverage and derive
    reachability/didactic checks
- `didactic_value`
  - whether the change turns a magical or opaque transformation into an
    intentional step path
- `unsoundness_risk`
  - whether the candidate could introduce hidden domain, branch, infinity, or
    integration-constant assumptions

This matters because the campaign repeatedly encounters ideas with:

- high local impact
- low retention probability
- poor actual ROI

The ROI estimate should explicitly include dimensional coverage value, not just
local hotspot movement.

At minimum, each candidate should be evaluated across these lenses:

- `runtime ROI`
  - how much real cost it removes from retained traffic
- `correctness / robustness ROI`
  - how much stable mathematical or operational failure it removes
- `dimensional coverage ROI`
  - whether it expands an under-covered axis such as wrapper spread,
    `noise_budget`, `shell_depth`, `cross_family_composition`, semantic regime,
    or a derive-specific dimension
- `derive bridge ROI`
  - whether a simplification/equivalence improvement can become a stable
    `derive` transition or a minimal derive contract row
- `engine feedback ROI`
  - whether a `derive` miss reveals a missing reusable engine transition rather
    than only a planner or wording issue
- `calculus ROI`
  - whether calculus work turns existing pre-calculus strength and already
    retained calculus slices into a coherent public capability, and exposes
    reusable pressure back onto simplification, equivalence, domains, didactic
    traces, or final result presentation
- `promotion ROI`
  - whether the result is stable enough to become a lasting guardrail or corpus
    promotion

These lenses do not replace the main ROI formula.
They make it harder for the selector to overfit to a narrow hotspot that sits
inside an already overrepresented slice of the scorecard.

### Pre-Calculus And Calculus Strategy

The pre-calculus engine is now strong enough to serve as the foundation for
calculus work. It is not complete in a universal-CAS sense, and the loop should
not pretend otherwise.

The strategy is no longer merely to start calculus through disciplined vertical
slices. The existing scorecard already carries substantial green coverage for
`diff`, `limit`, and `integrate`. The next default should be to generalize those
verticals into a serious real-domain calculus surface:

- differentiation first, because it is high-value, broad, and stresses
  simplification in controlled ways
- limits under explicit policy matrices for finite points, infinity,
  one-sided behavior, continuity, and residual boundaries
- integration through table, substitution, by-parts, and verification families,
  not through broad search
- shared calculus support matrices that make supported, residual, and rejected
  regimes visible
- series/asymptotics later, after limits and differentiation have stronger
  domain and trace support

Interpret green calculus scorecards as retention evidence, not as proof of
general maturity. The remaining work is mostly systematic:

- make differentiation uniform across elementary real-domain families,
  preserving domain requirements and readable post-calculus forms
- keep limits conservative while expanding policy-backed finite, infinity,
  endpoint, one-sided, discontinuity, removable-singularity, and residual
  regimes
- grow integration through verified families with explicit domain and constant
  policy, not through unbounded integration search
- make didactic traces explain nontrivial calculus moves instead of returning
  answer-only shortcuts
- reduce architecture pressure when repeated local helpers obscure ownership of
  detection, domain reasoning, transformation, verification, presentation, or
  steps

For planning purposes, treat a serious educational calculus surface as a
dozens-of-retained-cycles target and a mature elementary real-domain calculus
surface as a substantially longer campaign.

The long-term integration direction is hybrid, not "make the current table
integrator search harder":

- keep the educational real-domain `integrate` route conservative, verified,
  and explainable
- create a separate algorithmic integration backend only behind an explicit
  boundary
- require structured method tags, assumptions, verification status, residual
  reasons, domain/constant policy, and runtime isolation before promoting broad
  integration behavior

A universal-style integration backend is therefore allowed as a future campaign
track, but not as an unbounded shortcut inside ordinary calculus ROI cycles.
Until the backend boundary exists, broad integration search remains a deferral
signal rather than an implementation target.

Post-calculus presentation is part of this strategy, but it is not a license to
make the global simplifier prettier. The selector should treat presentation as a
calculus concern only when a supported `diff`, `limit`, or `integrate` result is
already mathematically correct and the retained value is a safer, clearer public
form. Internal canonical forms may remain optimized for matching, equivalence,
and runtime.

This does not reduce the importance of pre-calculus. It gives pre-calculus a
new source of useful pressure:

- derivative outputs expose missing simplification, factoring, cancellation, and
  step-quality gaps
- derivative outputs expose awkward but correct final forms that may deserve a
  post-calculus presentation pass instead of a broad canonical rewrite
- limit cases expose safe pre-simplification and domain assumptions
- integration cases expose inverse-differentiation, substitution, and constant
  policy gaps

The selector should prefer calculus candidates when all of these are true:

- the user-visible calculus improvement generalizes an existing supported
  family, closes a matrix gap, or replaces repeated narrow cases with shared
  capability
- the implementation reuses the existing symbolic engine rather than bypassing
  it with an isolated shortcut
- the domain/branch/constant assumptions are explicit
- any presentation-only change is local to calculus output, preserves required
  conditions, and does not alter the internal canonical route
- the case can be validated with a narrow unit or contract test plus the needed
  pre-calculus guardrails
- the outcome improves a whole rule family, policy regime, presentation regime,
  or didactic trace class rather than only one string-shaped example

For calculus presentation and derivative-family work, the selector should also
run a bounded sibling sweep before editing. This is a cheap probe pass, not
unbounded search. It asks whether the same rule shape appears in sibling
families, orientations, and scale regimes. For example, a root-presentation
candidate should probe at least the direct/dual inverse pair, sign and external
scale variants, exact-square root content, and unit or negative-unit scales.

If retained cycles expose the same pattern in sibling families, the next
candidate should default to consolidation: extract or reuse a shared helper,
write the shared policy into the calculus matrix, or promote one minimal
representative row. Continue with another local variant only when consolidation
would broaden route matching unsafely, alter domain/branch behavior, or
materially risk runtime/guardrail regression.

The selector should defer calculus candidates when they require:

- general integration search inside the current educational route
- hidden domain assumptions
- broad simplifier changes with unclear embedded runtime cost
- answer-only behavior that would be a magical step in the UI
- pretty-printing that changes semantics, hides conditions, or turns into a
  global simplification preference without a calculus contract
- speculative branch behavior not covered by policy
- adding another isolated calculus example inside a family that already has
  enough representatives, unless it exposes a new domain, trace, presentation,
  runtime, or verification axis

The selector may choose early hybrid-backend work only when the retained output
is bounded and explicit:

- an integration method result boundary
- observability over backend candidates, verification outcomes, and residual
  reasons
- behavior-preserving routing of an already verified family through the new
  boundary
- discovery-only probes for algorithmic methods that cannot affect public
  results until verified

Classify such work by retained value: usually `observability` or `robustness`;
use `calculus` only when one verified algorithmic family becomes public behavior.

If a calculus probe reveals that the real blocker is a reusable algebraic,
normalization, or domain gap, the next retained work should usually be
classified as `coverage`, `robustness`, or `observability`, not as a calculus
workaround.

### Calculus Architecture Pressure

The calculus campaign has reached the point where architecture is a first-class
retention risk. Several useful local changes now fail or become fragile because
the pipeline boundary is unclear, not because the mathematical formula is
unknown.

Treat these as priority signals:

- post-calculus presentation patches change stable matrix rows outside the
  intended family
- antiderivative or derivative residual verification falls into deep generic
  simplification, `depth_overflow`, `cycle_detected`, or timeouts
- domain-condition compaction is rebuilt separately for sibling families
- repeated helpers accumulate in large calculus, integration-support, or
  didactic-step files
- a proposed calculus case needs a broad result-shape check instead of a
  source-side route predicate

When those signals appear, prefer an `observability` or `robustness` iteration
that advances calculus block 11 before adding another local calculus row. The
retained output should be small and testable:

- extract one coherent route family while preserving call order and behavior
- isolate domain-condition construction behind the family or policy that proves
  the condition
- isolate post-calculus presentation from internal canonical matching
- add a bounded residual verifier for a verified primitive family before
  broadening the integrator
- add a support-matrix or profiling classification when ownership is still
  ambiguous
- for hybrid integration, add the backend boundary before adding algorithmic
  search

This is not permission for a broad refactor. Architecture work has high ROI
only when it removes a demonstrated blocker to safe calculus generalization and
keeps the guardrails green.

For calculus work, the selector should also use the block-based maturity plan in
[CALCULUS_ENGINE_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/CALCULUS_ENGINE_STRATEGY.md).
Each calculus iteration should name the active maturity block and the block gate
it advances. This keeps the loop from treating a long sequence of green
near-duplicates as real generalization.

### Engine/Derive Cross-Pollination

Every mathematical improvement should be read in both directions:

- engine -> derive:
  when the engine gains or stabilizes a family, ask whether there is a minimal
  `source -> target` pair that should be added as a derive shadow case
- derive -> engine:
  when `derive` cannot bridge an equivalent pair, classify whether the gap is a
  missing target classifier, missing strategy, didactic-quality problem, or a
  missing reusable engine transition

The loop should prefer candidates that improve both sides without increasing
runtime risk:

- a new simplification family that also provides a real transition provider is
  higher ROI than the same family used only as a root shortcut
- a derive planner improvement that exposes reusable engine normalization is
  higher ROI than a one-off derive-only route
- a corpus promotion that can serve as both contextual engine coverage and a
  derive bridge is higher ROI than an isolated anecdote

Do not automatically add derive rows for every engine case. Promote only the
smallest target-form representative when it adds one of:

- new target family reachability
- multi-step route coverage
- didactic-path quality pressure
- a known magical one-step transformation that should become explainable

If several automatic iterations pass without any retained derive work while
`derive_contract` remains trivially green, the selector should raise the weight
of `derive_bridge_value` and run a small equivalent-but-not-derived discovery
pass before choosing the next non-runtime coverage candidate.

### ROI Tie-Breakers

When two candidate iterations are similar in local runtime value, prefer the
one that:

- expands a weakly covered dimension
- reduces concentration in an already dominant wrapper or family pocket
- improves both a hotspot and a corpus promotion path
- improves both engine behavior and a plausible derive bridge
- increases the chance of promoting a stable new guardrail

Be more skeptical of candidates that:

- improve a hotspot inside an already saturated wrapper axis
- add complexity without changing scorecard dimensions
- add engine-only coverage for a target-form family that is absent from derive
  without at least recording why no derive shadow case is appropriate
- only win on a narrow local profiler slice with no plausible promotion path

### Default Selection Rules

Prefer:

- `runtime`
  - when `frozen` or `live` shows a broad hotspot with real traffic
- `coverage`
  - when a reusable family fails repeatedly and has good mathematical value
- `robustness`
  - when there is timeout, nontermination, overflow, or path fragility
- `observability`
  - when recent iterations produce `local win / global fail` or the hotspot is
    still ambiguous
- `calculus`
  - when a `diff`, `limit`, or `integrate` change generalizes an existing
    family, fills a documented calculus support-matrix gap, improves a shared
    domain/presentation/didactic policy, or replaces repeated narrow cases with
    reusable capability
- `combination`
  - only when the ledger already contains a compatible pair:
    - expensive win + cheap gate
    - repeated extraction + reuse/cache
    - local win in shared helper + move to exact call-site
    - hotspot shift + follow-up patch for the shifted hotspot

Avoid selecting `combination` just because multiple ideas looked good locally.

Also allow dimensional coverage to steer the class choice:

- prefer `coverage` when a stable new wrapper, shell-depth level, or
  composition axis can be promoted cheaply
- prefer `coverage` or `derive`-oriented work when a stable engine family lacks
  any representative derive bridge and the case is not branch/domain-hard
- prefer `calculus` when existing simplification/equivalence strength and
  already-green calculus slices can be generalized into policy-backed public
  capability without broad search
- prefer `observability` when the current scorecard shows an under-covered
  dimension but the profitable next promotion is still unclear
- prefer `runtime` only if the hotspot is not merely the byproduct of severe
  concentration in an under-grown dimension that should be expanded instead

Also allow structural cohesion to steer the class choice:

- prefer `observability` when a large file or route family is too opaque to
  safely continue adding shortcuts
- prefer `robustness` when a god-file pattern creates stack, timeout, or
  harness fragility risk
- prefer `observability` or `robustness` when calculus generalization keeps
  adding local helpers for domain conditions, residual verification,
  post-calculus presentation, or didactic step construction
- prefer `runtime` only when a structural boundary removes measured hot-path
  work
- do not classify refactoring as a separate goal; classify the retained value it
  creates
- for this track, follow
  [ENGINE_COHESION_REFACTORING_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_COHESION_REFACTORING_STRATEGY.md)

### Per-Iteration Loop

Each automatic iteration should do this in order:

1. read the latest scorecard and profiler evidence
2. classify candidate work into `runtime`, `coverage`, `robustness`,
   `observability`, `calculus`, or `combination`
3. estimate expected ROI for the top candidates
4. annotate each serious candidate with:
   - primary dimension affected
   - secondary dimension affected
   - whether it is a hotspot move, a dimensional coverage move, or both
   - whether it has a realistic promotion path into corpus or guardrail
   - whether it creates or consumes a derive bridge
   - whether it generalizes a calculus capability, fills a calculus support
     matrix gap, or exposes a pre-calculus blocker
5. choose one primary class for the iteration
6. write down the success condition before editing code
7. validate against the relevant benchmark roles
8. if the change fails global retention, revert the runtime change and preserve
   the learning in observability and/or the combination ledger

For cohesion refactors, prefer extraction-only iterations before algorithmic
generalization. A behavior-preserving extraction can be retained as
`observability` or `robustness` only if the same scorecard guardrails remain
green and the new module boundary makes ownership clearer.

### Per-Iteration Capture Template

The ROI formula is only useful if each iteration writes down the same minimum
evidence in a comparable shape.

Use a short capture like this before and after the edit:

```text
investment_class:
success_condition:
primary_dimension:
secondary_dimension:
hypothesis:
relevant_lanes:
promotion_target:
derive_bridge_check:
engine_feedback_check:
calculus_scope:
calculus_maturity_block:
calculus_matrix_cell:
block_gate_check:
generalization_axis:
precalculus_dependency:
domain_safety_check:
presentation_check:
sibling_sweep_check:
consolidation_check:
architecture_pressure_check:
extraction_or_consolidation_check:
retain_if:
reject_if:
retain_or_reject_reason:
```

Minimum expectations:

- `investment_class`
  - one of `runtime`, `coverage`, `robustness`, `observability`, `calculus`,
    `combination`
- `success_condition`
  - concrete and measurable, not narrative
- `relevant_lanes`
  - the exact profiles/suites to rerun
- `promotion_target`
  - `none`, `unit`, `live`, `frozen`, or `stress`
- `derive_bridge_check`
  - for engine/corpus work: whether a minimal derive shadow case exists, was
    promoted, deferred, or rejected with reason
- `engine_feedback_check`
  - for derive work: whether the miss is planner-only, didactic-only, or points
    to a reusable engine capability gap
- `calculus_scope`
  - for calculus work: `diff`, `limit`, `integrate`, or the shared calculus
    command/surface, matrix row, or policy regime being changed
- `calculus_maturity_block`
  - for calculus work: the active block from the block-based maturity plan,
    such as differentiation, limits, base integration, generalized
    substitution, rational integration, trig/hyperbolic integration,
    radical/inverse families, residual policy, didactic trace maturity, or
    calculus architecture/observability
- `calculus_matrix_cell`
  - for calculus work: the command/family/argument/domain/trace/presentation
    cell being filled, generalized, or deliberately deferred
- `block_gate_check`
  - for calculus work: the concrete maturity gate improved by this iteration:
    correct public behavior, explicit real-domain conditions, residual safety,
    didactic trace, matrix representative, verification, or guardrail retention
- `generalization_axis`
  - for calculus work: the reusable axis being expanded, such as orientation,
    sign, argument shape, domain regime, verification policy, or presentation
- `precalculus_dependency`
  - for calculus work: which simplification, equivalence, domain, or didactic
    capability the change reuses or hardens
- `domain_safety_check`
  - for calculus work: how domain, branch, infinity, or integration-constant
    assumptions are represented or explicitly avoided
- `presentation_check`
  - for calculus work: whether the final public form is unchanged, improved by
    a local post-calculus presentation pass, or deliberately deferred because
    the internal canonical form is preferable for now
- `sibling_sweep_check`
  - for calculus work: the bounded sibling probes used to detect repeated
    family, orientation, sign, scale, or presentation gaps
- `consolidation_check`
  - for calculus work: whether this should remain local or become a shared
    helper, policy, or support-matrix representative before more variants are
    added
- `architecture_pressure_check`
  - whether the candidate is adding to an already crowded route/presentation/
    verifier/domain area, and whether the safer ROI is block 11 architecture
    work instead of another local case
- `extraction_or_consolidation_check`
  - when architecture pressure is present: the smallest behavior-preserving
    extraction, support-matrix clarification, or bounded shared helper that
    would reduce future risk
- `retain_or_reject_reason`
  - the shortest defensible explanation after validation

## Generated High-Temperature Discovery

The automation loop should also include a deliberate `generated discovery`
feeder, not only manually curated corpus growth.

This feeder should generate new expressions by composing already validated
equivalences into larger contexts with controlled structural aggression.

The goal is not random novelty for its own sake.
The goal is to expose engine weaknesses faster by forcing known identities to
survive:

- broader wrapper spread
- higher `noise_budget`
- greater `shell_depth`
- cross-family composition
- reordered or negated orientations
- more entangled additive or multiplicative context

This is a `coverage + robustness` accelerator.

It is especially valuable when manual corpus growth is finding real issues but
too slowly.

### Temperature Should Mean Structured Composition

`temperature` should not mean unconstrained syntactic randomness.

It should mean controlled pressure along dimensions that already matter in the
scorecard:

- wrapper axis
- `noise_budget`
- `shell_depth`
- cross-family composition
- sign/orientation variation
- semantic regime

Higher temperature should therefore mean:

- more families combined in one expression
- more realistic wrappers around known local identities
- more residual traffic around the useful core
- more chances for the engine to miss a local rewrite, route late, or pay broad
  no-match cost

### Generated Discovery Workflow

For each generated candidate, the default workflow should be:

1. generate from trusted equivalence seeds or already-retained identity families
2. run the expression through the cheap iteration lane; for one-off
   `embedded_equivalence_context` rows, prefer
   `scripts/engine_embedded_candidate_smoke.py --row '<csv row>'` so pass,
   fail, and timeout candidates are classified before promotion
3. if it fails semantically, panics, loops, overflows, or degrades didactic
   quality badly, classify the failure by structural signature
4. fix the family-level weakness, not only the anecdotal string instance
5. validate against the relevant guardrails, especially `embedded` elapsed time
6. choose the smallest durable promotion target

Generated candidates should be clustered by:

- family mix
- wrapper mix
- shell-depth level
- noise budget
- dominant failing route or fallback tier

That clustering matters because several hot generated expressions often expose
the same missing abstraction.

### Discovery Failures Worth Preserving

Not every failed generated candidate deserves a durable note. A failed smoke
case should normally be discarded without ceremony when it is malformed, a
near-duplicate, a parser artifact, or an isolated anecdote with no reusable
signature.

Do preserve the discovery when cheap isolating probes show a structural pattern:

- the root identity passes but a wrapper or composition fails
- a two-core composition passes but a three-core composition fails
- removing one syntactic companion makes the candidate pass
- reordering does not remove the failure
- the failure repeats across related companions or wrappers

In those cases, record the smallest useful signature in
[ENGINE_COMBINATION_LEDGER.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_COMBINATION_LEDGER.md)
with status `observe-only discovery`. The note should state what passed, what
failed, and the current suspected family-level weakness. This keeps generated
discovery useful without turning the live corpus or the ledger into a backlog
of random rejected strings.

### Promotion Policy For Generated Cases

Generated high-temperature cases should not go straight into the `live` corpus
by default.

Preferred promotion path:

1. keep the raw discovery case in a generated or local pressure lane while the
   family is still moving
2. promote to a focused unit or regression test when the bug is narrow and the
   shape is too specific for a long-lived corpus row
3. promote a small representative contextual variant to `live` when it adds a
   real new family, wrapper, composition, or shell-depth guardrail
4. promote only a stable minimal subset to `frozen`
5. send larger or hotter variants to `stress`

This preserves the main value of high-temperature generation:

- discover aggressively
- promote conservatively

### Didactic Review Should Follow Semantic Retention

Generated discovery should also improve the didactic layer, but only after the
semantic fix and runtime retention look good enough.

After a generated case is semantically retained, the follow-up review should
check:

- step quality
- sub-step quality
- whether a step is too magical for the underlying transformation
- `before/after` highlight correctness
- whether the explanation is generalizable across the family

If the didactic improvement is too case-specific, prefer a narrow regression
test instead of broad narration changes.

## Profiles Implemented Today

Use [engine_improvement_scorecard.py](/Users/javiergimenezmoya/developer/math/scripts/engine_improvement_scorecard.py) as the unified runner.

Benchmark interpretation caveats and the next-level evaluation roadmap live in
[ENGINE_BENCHMARK_IMPROVEMENT.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_BENCHMARK_IMPROVEMENT.md).

These are the runnable profiles exposed by the current scorecard runner:

- `fast`
  - `metatest_csv_combinations_small`
  - `metatest_csv_contextual_pairs_strict`
  - `metatest_csv_contextual_radical_pairs`
  - `diff_step_contract_tests`
- `fast_embedded`
  - `fast`
  - `embedded_equivalence_context`
  - use when the change touches orchestration, contextual equivalence, or
    broad simplify routing and `fast` alone would be too weak
- `guardrail`
  - embedded equivalence context corpus
  - derive contract corpus
  - derive shadow pressure over representative engine identity rows
  - public differentiation, limit, safe limit pre-simplification, and
    integration contract lanes
  - unified simplification benchmark in `strict`
- `pressure`
  - simplify-zero mixed corpus
  - unified simplification benchmark in `nf-first`
- `full`
  - `guardrail + pressure`

When the runner is executed, it writes:

- JSON scorecard at [docs/generated/engine_improvement_scorecard.json](/Users/javiergimenezmoya/developer/math/docs/generated/engine_improvement_scorecard.json)
- Markdown summary at [docs/generated/engine_improvement_scorecard.md](/Users/javiergimenezmoya/developer/math/docs/generated/engine_improvement_scorecard.md)

Those generated files are on-demand artifacts.
They may be absent in a clean checkout until the scorecard has been run.

When a previous scorecard is passed through `--baseline`, the runner should also
surface:

- suite elapsed deltas
- explicit `embedded` runtime assessment vs baseline
- a warning state when `embedded` regresses materially in elapsed time

The scorecard should also grow generated-discovery visibility over time.

At minimum, it should make room for:

- generated candidates explored this cycle
- how many exposed real engine failures
- how many collapsed into an already-known failure cluster
- how many were promoted to `unit`, `live`, `frozen`, or `stress`
- whether the retained fix changed `embedded` elapsed time materially

The scorecard should also grow calculus visibility over time. Initially this can
be diagnostic rather than gating:

- representative `diff`, `limit`, safe limit pre-simplification, and
  `integrate` contract counts
- calculus support matrices by family, argument regime, domain regime, trace
  regime, and presentation regime
- unsupported/residual counts by calculus family
- calculus cases whose final result is correct but depends on a weak
  pre-calculus simplification route
- calculus cases whose final result is correct but presentation-poor, especially
  reciprocal fractional powers, avoidable denominator expansion, and root forms
- didactic or highlight defects in calculus traces
- which calculus probes were reclassified as pre-calculus coverage,
  robustness, or observability work
- whether a cycle added reusable calculus capability or only another isolated
  example; isolated examples should need an explicit new-axis justification

## Target Benchmark Role Model

The next section describes the benchmark-role model the campaign should reason
about.

Important distinction:

- `fast / guardrail / pressure / full`
  - executable scorecard profiles that exist today
- `frozen / live / stress`
  - benchmark roles the campaign should reason about, but which are not yet
    fully exposed as first-class scorecard profile names

Do not treat those two vocabularies as interchangeable.

## Benchmark Stratification

The automation loop should reason about three benchmark roles:

- `frozen`
  - a small representative suite snapshot
  - intentionally stable across normal engine work
  - measures baseline engine overhead on known traffic
- `live`
  - the current representative guardrail workload
  - grows as new families become real product traffic
  - measures current usefulness
- `stress`
  - larger, deeper, or more combinatorial expressions
  - measures scaling behavior and complexity cliffs

### Why The Split Is Required

If we only keep one growing benchmark, elapsed time becomes ambiguous:

- maybe the engine got slower
- maybe the suite simply got larger
- maybe the engine got more complete and the cost is acceptable

The split removes that ambiguity:

- `frozen` measures tax
- `live` measures current value
- `stress` measures scaling

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
- `derive_shadow_pressure` samples representative rows from
  [identity_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/identity_pairs.csv)
  as a diagnostic engine-to-derive bridge lane. It is not a support gate; it
  reports which known engine identities are derived, unsupported, or not yet
  provable as explicit `derive` targets.

These are not interchangeable metrics.

- `embedded_equivalence_context_corpus.csv` is a contextual simplify/equivalence metric:
  it tells us whether the engine can preserve an equivalence once that identity is
  embedded inside realistic wrappers
- `derive_pairs.csv` is a bridgeability/path metric:
  it tells us whether the planner can find a valid `source -> target` route and how
  expensive that route is

Interpret them separately:

- `embedded` passes and `derive` fails:
  the algebra is present but the derive planner or strategy selection is weak
- `derive` passes and `embedded` fails:
  the planner can bridge the pair, but contextual simplification under wrappers is weak
- both fail:
  the gap is probably algebraic, normalization-related, or otherwise more fundamental

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

That should be visible in the scorecard itself, not only in manual notes after a
benchmark run.

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

Operationally, the default comparison loop should be:

1. keep a recent scorecard JSON as baseline
2. rerun the relevant profile or suite
3. inspect the embedded runtime delta in the scorecard output itself
4. reject or justify the change before retaining it

When possible, that comparison should explicitly name the benchmark role:

- frozen delta
- live delta
- stress delta

Immediate campaign policy:

1. run the narrow local slice
2. compare against the latest `live` baseline
3. check `frozen` if the change affects broad routing or common hot paths
4. check `stress` if the change could alter scaling on larger embedded terms

When the answer is “reject”, the work should still produce a durable artifact if
the local win was real.

Use
[ENGINE_COMBINATION_LEDGER.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_COMBINATION_LEDGER.md)
to record:

- the local profiler win
- the global guardrail loss
- the suspected reason the idea did not scale
- the specific complementary change that could make it viable later

This prevents the campaign from forgetting good local hypotheses that only need
better scoping.

## Calculus Runtime Budgets Are Block-Level Guardrails

`embedded_equivalence_context` remains the broadest runtime proxy for common
engine traffic, but it is not enough once `diff`, `limit`, and `integrate`
become first-class public surfaces.

Calculus lanes should also be interpreted as runtime guardrails, not only as
semantic green/red tests. A calculus scorecard that gains support while becoming
slower may be acceptable; a calculus scorecard that becomes slower without
retained coverage, domain safety, verification value, or didactic value is
usually a regression.

The useful calculus runtime numbers are:

- total elapsed time for the lane
- `avg_case_ms` normalized by matrix size or contract count
- p95/max case time when available
- per-family and per-regime hot spots
- whether slow cases are supported, residual, or undefined
- whether antiderivative/differentiation verification cost is the driver

When the scorecard does not yet expose one of those numbers, prefer adding that
observability over guessing from wall-clock time.

### Retention Heuristic

Use runtime deltas relative to a recent baseline, and evaluate them against the
capability retained by the change.

Accept by default when:

- `failed = 0`, no timeouts, no stack overflow, no nontermination
- `avg_case_ms` is flat or improves
- the change increases supported cases, domain safety, verification, didactic
  quality, or removes residuals

Accept with explicit justification when:

- a broad calculus lane slows by roughly 10-20%
- the slowdown is localized to the family being generalized
- the change adds a reusable public capability or a materially stronger domain
  policy
- a follow-up runtime candidate is clear if the cost remains high

Reject or defer when:

- a broad lane slows by more than roughly 20-30% without a major reusable win
- embedded runtime regresses materially as collateral damage
- the only gain is a near-duplicate matrix row or presentation preference
- a residual/unsupported path becomes expensive without improving user-facing
  behavior
- the change hides route churn behind a green result

Phrase the decision as:

```text
runtime_delta / retained_capability_delta
```

For example, a small `integrate_command_matrix` slowdown may be acceptable if it
adds a verified integration family with real domain conditions. The same
slowdown is not acceptable for one syntactic variant of an already-covered
family.

### Calculus Runtime Observability Targets

The scorecard should grow toward explicit runtime reporting for:

- `diff_command_matrix`
- `limit_command_matrix`
- `integrate_command_matrix`
- public contract lanes for `diff`, `limit`, and `integrate`

Preferred output:

- elapsed seconds
- case count
- `avg_case_ms`
- slow case count
- timeout count
- top slow case IDs
- family/regime breakdown for the slowest cases

This makes calculus runtime comparable to embedded runtime and prevents broad
generality work from silently turning into route churn.

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

Generated high-temperature expressions are a valid source of new embedded cases,
but they should feed curation rather than bypass it.

But family count alone is not enough.

Wrapper coverage is a first-class axis of embedded quality.

A corpus with many identity families but only a few wrappers can still miss the
real contextual failures that matter in the engine:

- the naked identity works
- the same identity fails under a square, quotient, reciprocal, or additive shell
- the engine succeeds only through a late expensive fallback when wrapped

So `embedded` coverage should be thought of as at least two-dimensional:

- family axis:
  what mathematical identities are present
- wrapper axis:
  what contextual shells and composition environments those identities survive in

Operationally, this means the scorecard and runner should not report only total
cases and family spread. They should also expose wrapper spread and wrapper
concentration, so we can notice when we have accidentally built a corpus with
broad family labels but narrow contextual shape coverage.

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

When choosing between:

- more depth inside already-common wrappers
- and a stable new wrapper axis

prefer the new wrapper axis unless the old wrapper is still clearly under-covered.

Avoid this anti-pattern:

- adding many near-duplicate examples that all exercise the same matcher path

The goal is not raw corpus size.
The goal is broader contextual mathematical coverage per added case.

### Generated Composition Should Feed Curation

When a generated expression exposes a new weakness, do not promote the whole
large composed expression automatically.

Instead:

1. identify the reusable structural lesson
2. decide whether the best durable guard is:
   - a unit test
   - a single curated `live` row
   - a hotter `stress` row
3. keep only the minimal representative case in `embedded`

This is the main safeguard against turning `embedded` into a bag of noisy
anecdotes.

Large composed expressions are useful for discovery precisely because they are
hard.
That does not mean they are always the right long-lived guardrail artifact.

## Coverage Dimensions To Grow Deliberately

`embedded` should be grown along explicit dimensions, not just by adding more
rows whenever a new anecdote appears.

The dimensions that matter most are:

- wrapper axis:
  which contextual shells are represented at all
- noise or width axis:
  how much unrelated algebraic traffic surrounds the core identity
- shell depth:
  how many meaningful contextual layers the identity survives through
- cross-family composition:
  whether one family still works when embedded inside another family's shape
- semantic regime:
  exact, symbolic, general, branch-sensitive, and domain-sensitive traffic
- strategy tier:
  whether the case resolves by a local rewrite, a structural equivalence, or only
  by a late expensive fallback

These dimensions are often more valuable than adding yet another example in an
already-covered family/wrapper pocket.

### Depth Should Mean Shell Depth, Not Raw AST Depth

Raw AST depth is a weak guardrail dimension by itself.

It is too easy to inflate syntactically without teaching us anything about the
real contextual strength of the engine.

Prefer `shell depth` or `context depth` instead:

- depth 0:
  naked equivalence
- depth 1:
  one wrapper
- depth 2:
  one wrapper plus passthrough or residual context
- depth 3:
  two nested wrappers
- depth 4:
  nested wrappers plus cross-family composition

This gives a much more meaningful progression for continuous improvement.

If a family is stable at one shell depth, the next promotion target should
usually be the next shell depth, not arbitrary extra syntax.

### Noise Or Width Is A First-Class Dimension

Many real slowdowns and failures come less from depth than from width:

- extra additive noise
- extra multiplicative noise
- extra residual terms that create candidate traffic and dead compares

That means `noise budget` is a better guardrail dimension than raw tree depth in
many campaigns.

When choosing the next contextual promotion, strongly consider:

- same family, same wrapper, higher noise budget
- before
- same family, much deeper but syntactically artificial nesting

### Cross-Family Composition Should Be Treated As Its Own Axis

We should explicitly value cases where a family is embedded inside another
family's environment, for example:

- trig inside quotient normalization
- factorization inside square wrappers
- telescoping inside passthrough shells
- log/exp identities inside algebraic wrappers

These are often more representative of real engine traffic than isolated family
benchmarks.

High-temperature generation is one of the best ways to grow this axis on
purpose, because it can deliberately compose already-known families instead of
waiting for those mixtures to appear only by anecdote.

### Strategy Tier Must Stay Visible

A case that still passes can still represent a regression if it has drifted from:

- local direct rewrite
- to structural equivalence
- to default simplify
- to proved-symbolic or numeric-only

So corpus growth and scorecard interpretation should care not only about
pass/fail, but also about how the engine is winning.

## Recommended Next Axes For Embedded Growth

When the current wrapper spread is healthy enough, the default next dimensions
to add are:

1. `noise_budget`
2. `shell_depth`
3. `cross_family_composition`

That ordering is intentional.

It reflects how real contextual traffic tends to hurt the engine:

- broad noisy traffic often appears before truly deep nesting
- moderate contextual nesting is common and important
- composed families are high-value once the simpler axes are stable

In practice, a good progression looks like:

- naked identity
- one wrapper
- one wrapper plus noise
- two wrappers
- two wrappers plus noise
- composed family under wrappers

This is a better long-term growth ladder than “just make the expression deeper”.

### Operational Complexity Levels For Embedded

This policy should be implemented in the embedded corpus runner, not kept as a
manual intuition.

For each embedded row, the runner should compute:

- `shell_depth`
  - `expression_depth - max(source_depth, target_depth)`
- `wrapper_overhead_nodes`
  - `expression_nodes - max(source_nodes, target_nodes)`

The baseline uses the larger naked side of the pair so that the contextual delta
tracks wrapper/noise/composition overhead, not the intrinsic asymmetry of one
orientation of the identity.

Those raw metrics should then be collapsed into a small number of stable
complexity levels:

- `l0_root_pair`
  - `shell_depth == 0` and `wrapper_overhead_nodes <= 2`
- `l1_single_wrapper`
  - `shell_depth <= 1` and `wrapper_overhead_nodes <= 6`
- `l2_wrapper_plus_noise`
  - `shell_depth <= 2` and `wrapper_overhead_nodes <= 14`
- `l3_nested_or_composed`
  - everything else

These levels are intentionally not “number of wrappers only”.

A case can move up a level through:

- one more contextual wrapper
- the same wrapper plus passthrough/noise overhead
- nested wrappers
- cross-family composition

That is the right behavior for the automation loop, because the engine often
pays for contextual complexity long before it pays for extreme raw AST depth.

Operationally, the runner should report at least:

- family spread
- wrapper spread
- shell-depth spread
- complexity-level spread
- `wrapper x complexity-level` concentration

The selector should then use this reading:

- if wrapper spread is weak, prefer a new wrapper axis first
- if wrapper spread is healthy but most traffic is still `l0/l1`, prefer the
  next retained complexity level inside an important wrapper
- if `wrapper x complexity-level` concentration is high, be skeptical of adding
  more examples in the dominant bucket

Promotion guidance should also use this stratification:

- do not promote a new row just because it is “more complex” syntactically
- do promote a row when it raises wrapper coverage, shell depth, or
  `wrapper x level` coverage for an important family
- a row that stays in the same family, same wrapper, and same complexity level
  should usually stay out unless it adds a new strategy tier or catches a known
  fragile routing path

This makes “add more wrappers” a controlled lever inside the scorecard, instead
of an ad-hoc intuition.

## Derive Needs Different Dimensions

`derive` should not inherit the same score interpretation as `embedded`.

For `derive`, the important dimensions are:

- reachability:
  can the planner bridge `source -> target` at all
- step count:
  how long the successful route is
- long-path rate:
  how often derive only succeeds through a bloated path
- strategy diversity:
  whether different families rely on a narrow brittle planner route
- planner-miss-with-algebra-present:
  cases where the algebra exists in simplify/equivalence but derive still fails

So when evolving the scorecard or corpus policy, keep two separate mental models:

- `embedded` asks:
  "does the engine preserve equivalence under realistic contextual shells?"
- `derive` asks:
  "can the planner intentionally find the bridge, and how expensive is that bridge?"

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

A candidate addition that does not increase mathematical family coverage can
still be a strong promotion if it adds a genuinely new wrapper dimension for an
already important family.

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

If step 7 fails but step 2 or the local profiler clearly improved, add a ledger
entry instead of treating the iteration as wasted work:

- [ENGINE_COMBINATION_LEDGER.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_COMBINATION_LEDGER.md)

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
- `make engine-scorecard-pressure` when normalization or orchestration changed,
  using bounded mixed composed traffic
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

## Slow CI Test Triage

Slow tests observed during `make ci` are useful signals, but they must be
classified before they influence engine runtime work.

Treat every slow-test report as belonging to one of these buckets:

- `runner noise`
  - compile time, package cache lock, build directory lock, or unrelated test
    binary startup cost
- `engine runtime pathology`
  - the test is slow because the engine or planner itself takes too long on the
    semantic workload being tested
- `test verification pathology`
  - the product behavior is already correct, but the assertion route used by the
    test is much more expensive than the real product path

This distinction is mandatory.

Without it, the campaign will eventually add engine shortcuts or broad gates
just to accelerate a pathological assertion, which is exactly how we end up
taxing the embedded corpus for no product benefit.

### Default Workflow For A Slow Test

When a test appears slow inside `make ci`, do this in order:

1. reproduce it outside `make ci` with the narrowest exact command possible
2. measure the test body itself, not the surrounding compile or lock time
3. classify it as `runner noise`, `engine runtime pathology`, or
   `test verification pathology`
4. record the command, timing, and current hypothesis in the slow-test ledger:
   - [SLOW_CI_TEST_LEDGER.md](/Users/javiergimenezmoya/developer/math/docs/SLOW_CI_TEST_LEDGER.md)
5. choose the smallest correct fix:
   - test-side fix for verification pathology
   - engine/planner fix for real runtime pathology
6. if the retained fix touches runtime behavior, rerun the embedded equivalence
   guardrail and compare elapsed time explicitly

### Fix Selection Rules

Prefer a test-side fix when:

- the engine already reaches the correct result quickly in CLI or normal
  simplify/equivalence flow
- the slowdown comes from `isolated_simplify_rewrites_to_zero`,
  didactic expansion, or another assertion-only path
- the test is checking semantic equivalence through a route broader than the
  product contract it is supposed to guard

Prefer an engine/runtime fix when:

- the product path itself is slow or nonterminating
- the same structural pocket affects real corpus traffic
- the root cause is reusable and not tied to one test harness shape

Do not count these as slow-test targets:

- compile time before the specific test starts
- cargo package-cache or build-directory lock waits
- broad suite startup tax that disappears when the test is rerun narrowly

### Relationship To Embedded Runtime

Any retained engine-side fix motivated by a slow test must still obey the same
global rule as ordinary engine work:

- a local test-speed win is not sufficient
- embedded equivalence runtime remains the main broad guardrail

So the correct closure for a real engine-side slow-test fix is:

1. reproduce and fix the specific slow test
2. rerun the exact narrow test
3. rerun the relevant neighboring tests
4. rerun the embedded equivalence corpus and compare elapsed time

If the narrow fix wins locally but loses globally, do not retain it.
Instead, capture the attempt in the appropriate ledger:

- [ENGINE_COMBINATION_LEDGER.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_COMBINATION_LEDGER.md)
  for local runtime wins that fail global retention
- [SLOW_CI_TEST_LEDGER.md](/Users/javiergimenezmoya/developer/math/docs/SLOW_CI_TEST_LEDGER.md)
  for the reproducible slow test itself and its current classification

### Worktree Rule

Use worktrees only when the slow-test investigation raises a real runtime
regression hypothesis across recent commits.

Do not use worktrees for every slow-test report by default.
They are justified when we need to answer one of these questions:

- did this slowdown exist `1..N` commits ago?
- is the regression coming from local dirty changes or from retained history?
- which file family is responsible for the broad runtime delta?

When the answer is instead “the test harness is pathological but the engine is
already fine”, stay in the current tree and fix the test narrowly.

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

Read those decisions as a tradeoff table, not a single elapsed number:

- `frozen`
  - baseline overhead tax
- `live`
  - current workload value
- `stress`
  - scaling risk
- correctness / robustness
  - new passes, fewer timeouts, fewer overflows, better derive reachability

`frozen` must remain frozen.

Do not casually add new cases to it during routine engine work.

By contrast:

- `live` is expected to grow
- `stress` is expected to grow in size and difficulty

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

- formalize explicit `frozen / live / stress` profiles in the scorecard runner
- add an equation metamorphic lane from [metamorphic_equation_tests.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs)
- split `derive` scorecard into:
  - reachability
  - equivalence floor
  - didactic path quality
- add baseline comparison in CI so we fail on true regressions, not just raw test failures
- promote recurring hotspot slices into named corpora rather than keeping them as ad hoc notes
- add explicit runtime-budget alerts for suites that remain semantically green
  but become materially slower
- add normalized calculus runtime metrics (`avg_case_ms`, p95/max case time,
  top slow case IDs, family/regime breakdown) for `diff`, `limit`, and
  `integrate` matrices and public contracts

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

`pressure` is the routine bounded campaign lane. `full` is the closure lane and
includes the full `nf-first` metamorphic benchmark under a suite-level timeout
so nontermination becomes an explicit scorecard failure instead of a hung run.
