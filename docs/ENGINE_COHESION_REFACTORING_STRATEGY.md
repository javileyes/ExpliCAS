# Engine Cohesion Refactoring Strategy

This document is a derived strategy under
[ENGINE_IMPROVEMENT_AUTOMATION.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_IMPROVEMENT_AUTOMATION.md).

It exists because some engine files have become large enough that continued
feature growth now carries structural risk, especially (line counts as of
2026-06-10):

- [orchestrator.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs) (41.7k lines)
- [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs) (38.8k lines)
- [symbolic_integration_support.rs](/Users/javiergimenezmoya/developer/math/crates/cas_math/src/symbolic_integration_support.rs) (23.3k lines)
- [focused_rule_substeps.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs) (19.7k lines)
- [calculus_residual_support/mod.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/calculus_residual_support/mod.rs)
  (12.4k lines, partially extracted; split into module directory 2026-06-10)

The goal is not cosmetic cleanup.

The goal is to make future engine improvements safer, more measurable, and less
dependent on adding one more local shortcut to an already crowded file.

## Strategic Position

Cohesion work is valid engine-improvement work when it improves at least one of:

- observability of routing or matcher cost
- robustness against timeout, stack overflow, or brittle route ordering
- ability to validate one mathematical family in isolation
- ability to avoid duplicate broad matchers
- ability to make future `runtime`, `coverage`, or `combination` iterations
  cheaper and safer

It should usually be classified as:

- `observability`
  - when the main output is clearer routing boundaries, instrumentation, or a
    family map
- `robustness`
  - when the main output removes a brittle test path, stack-risky call path, or
    duplicated route hazard
- `runtime`
  - only when the change measurably reduces hot-path work
- `combination`
  - only when the ledger already shows that two local ideas need a structural
    boundary to become globally retainable

Do not add a new top-level investment class for refactoring.
Refactoring is a means, not a scorecard objective.

## Current High-Priority Weakness: Calculus Pipeline Accretion

The calculus campaign has reached a point where continued local growth can
create more risk than value. The repeated pattern is not simply "large files are
large"; it is that the same calculus pipeline concerns are being solved locally
inside adjacent families:

- family detection and argument extraction
- real-domain condition construction and display compaction
- transformation or primitive construction
- derivative/antiderivative residual verification
- post-calculus presentation
- didactic step/substep construction

When these concerns stay interleaved, a narrow fix can accidentally change a
stable matrix row, widen matcher traffic, reorder public conditions, or send
verification into deep generic simplification.

Current priority:

- treat calculus architecture pressure as a valid `observability` or
  `robustness` iteration even when the immediate symptom is a calculus feature
  request
- extract one route family or one pipeline boundary at a time
- preserve behavior first; only generalize after two retained boundaries prove
  a shared shape
- keep source-side predicates preferred over broad result-side cleanup
- keep domain-condition ownership close to the policy that proves the condition
- keep post-calculus presentation local to calculus output, not the global
  simplifier

Good first extraction targets:

- scaled-root inverse-family detection and domain-condition construction
- integration derivative-cofactor recognizers versus primitive presentation
- bounded residual-verification routes used to prove antiderivatives
  (started 2026-06-10, see Retained extractions)
- limit residual result/step presentation cleanup
- focused calculus step/substep builders once the route policy is stable

Bad first extraction targets:

- a generic calculus matcher registry that hides route priority
- a shared inverse-family helper that merges different branch or domain
  semantics
- a presentation abstraction that operates only on final result shape
- moving code only to reduce line count without making ownership or validation
  clearer

## Core Principle

Extract before abstracting.

The first safe move is usually to move a coherent family into its own module
while preserving behavior exactly:

- same call order
- same guards
- same step names
- same domain policy
- same fallback behavior
- same tests and scorecard outcome

Only after repeated extracted families reveal the same shape should a shared
algorithm be introduced.

This matters because many engine functions look similar but differ in important
ways:

- mathematical domain assumptions
- sign and orientation policy
- when isolated simplify is allowed
- whether a shortcut is allowed in collect-steps mode
- whether a route is intended as a cheap gate or an expensive fallback
- whether a local win is safe in embedded traffic

A premature generic abstraction can silently widen traffic and regress the
global guardrails.

## Non-Goals

This strategy is not:

- a big-bang rewrite of the orchestrator or arithmetic rules
- a mandate to make every matcher generic
- permission to reorder shortcut families for readability alone
- a license to merge two identities because their code looks similar
- a substitute for mathematical coverage work
- a reason to accept embedded runtime degradation without measured value

If a refactor changes behavior, it must be treated as engine behavior work, not
as mechanical cleanup.

## Recommended Work Order

### Phase 0: Map Before Moving

Before extracting a large block, record enough context to avoid changing
semantics accidentally:

- owning family or route group
- current call-site order
- cheap candidate gates
- expensive helpers called after a match
- collect-steps behavior
- domain-mode assumptions
- relevant unit tests and corpus rows
- embedded/runtime lane likely to observe regressions

This map can live in a short implementation note, a code comment near the
extracted module boundary, or a generated investigation document.

Do not spend a cycle building a perfect architecture map if a smaller local map
is enough to make the extraction safe.

### Phase 1: Extract Coherent Families

Prefer extraction units that already have a natural owner:

- exact-zero additive composition
- direct pair equivalence matchers
- trig product-to-sum routes
- hyperbolic angle/cubic routes
- log expansion/contraction direct identities
- fraction and telescoping shortcuts
- arithmetic cancellation helpers
- step construction helpers shared by one route family
- calculus route families where detection, domain, verification, presentation,
  and steps can be separated without changing behavior
- post-calculus presentation helpers that are already family-owned and covered
  by command matrices
- bounded residual-verification helpers that avoid broad simplification for a
  verified derivative or antiderivative family

The extraction should be behavior-preserving.

Good retained outcome:

- god-file line count decreases
- module ownership becomes clearer
- tests and scorecard are unchanged
- no new broad helper has been introduced

Bad retained outcome:

- a large file shrinks, but call-order semantics become harder to see
- a helper moves to a generic module before it has multiple real users
- unrelated identities are now coupled through a shared abstraction

Retained extractions under this phase:

- 2026-06-10: `cas_math/src/general_integration_backend.rs` (11,368 lines,
  fastest-growing god file) split into a module directory with explicit
  ownership (`probe_runner`, `model`, `verification`,
  `verification_normalization`, `methods`, `tests`), behavior-preserving,
  public API re-exported unchanged. See the combination ledger entry of the
  same date for the seams it surfaced.
- 2026-06-10: `cas_engine/src/calculus_residual_support.rs` (13,590 lines,
  fastest-accreting calculus file) converted to a module directory and the
  affine trig power residual-verification family (1,139 lines of moved body,
  17 fan-in-1 entries) extracted to `affine_trig_power.rs` (1,145 lines
  including the module header). Family zones in the rest of
  the file are interleaved, so further extractions should go family-by-family
  after a seam check (exp, hyperbolic, arctan_sqrt are candidates), not as a
  whole-file split.

### Phase 2: Consolidate Repeated Helper Shapes

Only consolidate after at least two extracted families prove that the helper is
truly shared.

Good candidates:

- signed n-ary term collection
- bounded partition over small zero groups
- cheap shape prefilters
- local exact-pair comparison
- shared `Rewrite` finishing utilities
- step metadata construction with identical soundness policy
- route-level profile labels

Risky candidates:

- helpers that call full simplify as part of matching
- helpers that flatten large expressions before a cheap gate
- helpers that mix domain-sensitive and domain-free identities
- helpers that hide shortcut priority behind a generic registry too early

### Phase 3: Generalize Algorithms Narrowly

A shared algorithm is retainable only when it has:

- a cheap prefilter
- bounded search or no search
- explicit domain policy
- visible route labels or tests at each call site
- proof that embedded runtime is not materially worse

The abstraction should remove real complexity, not just move it.

### Phase 4: Make Priority More Explicit

Once families are extracted and stable, consider making ordered groups more
legible.

Do this conservatively:

- preserve source-order priority by default
- add names to groups before changing order
- measure before and after
- avoid a global registry until the route taxonomy is stable

The desired end state is not a universal rewrite engine.
It is a set of clear, measured, family-owned routes that can evolve safely.

## Per-Iteration Rules

Each cohesion iteration should be one small retainable move.

Before editing, capture:

```text
investment_class:
success_condition:
primary_dimension:
secondary_dimension:
hypothesis:
relevant_lanes:
promotion_target:
retain_if:
reject_if:
cohesion_scope:
behavior_change_expected:
```

Use `behavior_change_expected: no` for extraction-only work.

If behavior changes unexpectedly, stop treating the iteration as mechanical.
Classify the behavior change and validate it like any other engine change.

## Validation Policy

Minimum validation for extraction-only work:

- touched unit tests
- `make engine-fast`
- `cargo fmt --check`
- `git diff --check`

Also run `make engine-scorecard` when:

- call order changes
- a root/orchestrator route moves
- `arithmetic.rs` helper ownership changes
- the extraction touches domain policy, isolated simplify, or shortcut finishing

Run `make engine-scorecard-pressure` when:

- normalizer behavior changes
- a broad matcher is generalized
- deep composed traffic is touched
- a helper is shared across multiple route families

Note: nf-first runs only in the full profile; for normalizer changes run
`make engine-scorecard-pressure` plus the full profile (or `simplify_nf_first`
directly) for nf-first coverage.

Run `make ci` only as a closure step for a retained batch or when the structural
change touches broad repository contracts.

## Retention Policy

Retain a cohesion refactor only if:

- `failed=0` remains true in relevant lanes
- timeouts do not increase
- embedded runtime is not materially worse
- route ownership is clearer than before
- the diff is reviewable and reversible
- no unrelated mathematical behavior changed accidentally

Reject or split the refactor if:

- a general helper widens traffic without measured value
- embedded runtime regresses without a strong reusable gain
- the extraction requires broad reorderings to compile
- the new module boundary hides priority or domain policy
- the change is too large to review against the scorecard evidence

If a local cleanup wins in one slice but loses globally, revert the runtime part
and record the learning in
[ENGINE_COMBINATION_LEDGER.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_COMBINATION_LEDGER.md).

## Metrics To Track

Structural metrics are secondary to engine behavior, but they help steer the
campaign:

- line count of god files
- number of extracted family modules
- number of duplicated route/helper shapes removed
- embedded elapsed time
- strict failed/timeouts
- pressure failed/timeouts for touched families
- number of ignored or pathological harness tests retired

Do not optimize line count alone.

A smaller file that makes route priority harder to reason about is not an
improvement.

## Practical Recommendation

Use cohesion work periodically, not continuously.

A good cadence is:

- continue normal ROI-directed engine iterations
- when a family has accumulated several retained shortcuts, extract that family
  before adding more
- when two extracted families show the same helper shape, consider a bounded
  shared helper
- after several retained mathematical iterations in the same file, spend one
  iteration reducing structural risk
- during the current calculus campaign, bias earlier toward this structural
  iteration when repeated local fixes touch domain conditions, residual
  verification, post-calculus presentation, or didactic step construction

This prevents the engine from growing only by accretion while preserving the
guardrails that make the improvement campaign trustworthy.
