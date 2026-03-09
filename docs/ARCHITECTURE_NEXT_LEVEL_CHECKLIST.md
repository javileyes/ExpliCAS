# Architecture Next-Level Checklist

## Purpose

This checklist converts

- `/Users/javiergimenezmoya/developer/math/docs/ARCHITECTURE_NEXT_LEVEL_PLAN.md`
- `/Users/javiergimenezmoya/developer/math/docs/SOLVER_EVENT_OBSERVER.md`

into an execution list that can be used PR by PR.

It is intentionally strict:

- one architectural goal per PR
- no mixed tracks
- no speculative refactors
- validation gates on every step

## Priority Legend

- `P0`: do now, high value, low ambiguity
- `P1`: do next if `P0` is stable
- `P2`: only after reevaluation
- `R&D`: isolated experiment, not part of the main migration stream

## Global Rules

Before starting any item:

- [ ] Confirm the item removes real coupling, not just visual disorder.
- [ ] Confirm the item does not require a wide rename-only churn.
- [ ] Confirm the item can be validated independently.

After completing any item:

- [ ] `cargo fmt --all`
- [ ] `cargo check -p cas_solver_core -p cas_engine -p cas_solver -p cas_session -p cas_cli -p cas_didactic`
- [ ] `make ci` if the change is broad enough to justify full validation
- [ ] If behavior changed in simplify/solve: rerun the relevant metamorphic check

## Track 1: High-Value Boundary Cleanup

### P0.1 Review remaining transport-specific naming

Goal:
- identify internal modules whose names still encode transport (`json`) when
  they really build typed Rust models

Checklist:
- [x] Audit `cas_solver/src/json/*`
- [x] Audit `cas_didactic/src/eval_json_*`
- [x] Mark each case as:
  - `transport boundary`
  - `typed model builder`
  - `mixed`
- [x] Rename only the clearly mixed cases
- [x] Do not rename boundary modules that truly own wire/DTO assembly

Done when:
- internal naming no longer implies JSON when the module is not actually doing
  JSON transport work

Current progress:
- audited `cas_solver/src/json/*` as transport boundary
  - keep:
    - stateless JSON entry points
    - envelope assembly
    - API mappers
    - substitute/eval command JSON wrappers
- audited `cas_didactic/src/eval_json_*` as mixed typed-model builders
  - renamed to:
    - `step_payloads`
    - `step_payload_render`
  - rationale:
    - they build `StepJson` DTOs and highlighted latex snippets
    - they do not serialize JSON
    - they are internal didactic presentation helpers, not outer transport
- left `cas_solver/src/eval_json_*` open for narrower follow-up review
  - many of those modules are still tightly tied to the `eval-json` command
    boundary and should not be bulk-renamed without splitting by responsibility
- narrowed the remaining `cas_solver` mixed area to presentation-only helpers
  - renamed:
    - `eval_json_presentation*` -> `eval_output_presentation*`
  - rationale:
    - they format strings/latex and build typed output DTO fragments
    - they do not own request parsing, option decoding, wire assembly or
      serialization
- also renamed output metadata helpers:
  - `eval_json_stats*` -> `eval_output_stats*`
  - rationale:
    - they compute typed `ExprStatsJson`, truncation metadata and stable hashes
    - they do not perform transport serialization
- keep as `eval-json` boundary modules for now:
  - `eval_json_command_runtime*`
  - `eval_json_input*`
  - `eval_json_options*`
  - `eval_json_finalize*`
  - `eval_json_request_runtime`

### P0.2 Review `cas_session_core`

Goal:
- decide whether `cas_session_core` should remain or merge back into
  `cas_session`

Checklist:
- [x] Map current responsibilities of `cas_session_core`
- [x] Check whether merging reintroduces dependency cycles
- [x] Check whether merge reduces indirection
- [x] If no, explicitly document why it stays

Done when:
- `cas_session_core` is either justified as permanent or removed

Decision:
- `cas_session_core` stays.

Why:
- `cas_engine` uses its stateless/session-agnostic eval contracts.
- `cas_solver` uses its low-level store, snapshot and entry-id types.
- `cas_session` already depends on both `cas_engine` and `cas_solver`.

Merging it back into `cas_session` would worsen the graph, not simplify it.

### P0.3 Review remaining `cas_math` leaks

Goal:
- only continue if a remaining candidate is clearly not math-pure

Checklist:
- [x] Audit only modules still suspected to be rule-facing or user-facing
- [x] For each candidate, answer:
  - is this deterministic reusable math?
  - does it mainly exist for one engine rule?
  - does it contain `desc`/policy/runtime packaging?
- [x] Move or clean up only the clearly engine-facing candidates
- [x] Stop as soon as remaining candidates become debatable

Done when:
- no obvious engine-facing helper remains in `cas_math`

Current progress:
- moved shared runtime `step_*` helpers (`step_rules`, `step_optimize`,
  `step_absorption`, `step_productivity`, `step_semantic`) out of `cas_math`
  into `cas_solver_core`
- rationale:
  - they operate on step/rule heuristics, not reusable algebra
  - they already served runtime orchestration in `cas_engine` /
    `cas_solver_core`
  - leaving them in `cas_math` blurred the boundary between pure math and
    didactic/runtime cleanup
- moved `rationalize_policy` out of `cas_math` into `cas_solver_core`
  - rationale:
    - it is shared simplification configuration and outcome reporting
    - it is consumed by `cas_solver_core`, `cas_engine` and `cas_solver`
    - it does not implement algebraic transformation logic itself
- moved `undefined_risk_policy_support` out of `cas_math` into `cas_solver_core`
  - rationale:
    - it is domain-policy gating for cancellation rewrites
    - it does not perform symbolic algebra
    - it was only serving runtime rule decisions in `cas_engine`
- spot-checked the remaining suspicious families and stopped there:
  - `fraction_univar_gcd_support` remains structural polynomial reduction with
    domain policy intentionally delegated upward
  - `trig_dyadic_policy_support` remains a math/domain gate, not runtime
    packaging
  - `poly_gcd_dispatch` remains a shared algorithm/mode selector with injected
    callbacks, not solver-session wiring
  - `undefined_risk_support` remains pure structural detection with caller
    provided proof oracle
- remaining suspects in `cas_math` are now the more debatable
  `*_support` families with mixed policy/description concerns; review should
  stop again before broad churn

## Track 2: Solve Observer/Event Phase 1

### P1.1 Add minimal `SolverEvent` model

Goal:
- create the smallest event model that is equation-aware

Checklist:
- [x] Add `cas_solver_core/src/solver_events.rs`
- [x] Add `SolverEvent<Equation, Importance>`
- [x] Add `SolveEventListener`
- [x] Add `cas_solver_core/src/solver_event_collector.rs`
- [x] Register both in `cas_solver_core/src/lib.rs`

Done when:
- solve events exist as a stable core abstraction with no solver runtime
  rewiring yet

### P1.2 Add `SolveStep <-> SolverEvent` adapters

Goal:
- prove the event model is lossless before changing any consumer

Checklist:
- [x] Add adapter module in `cas_solver`
- [x] Convert `Vec<SolveStep>` to `Vec<SolverEvent>`
- [x] Convert `Vec<SolverEvent>` back to `Vec<SolveStep>`
- [x] Add roundtrip tests

Done when:
- `SolveStep -> events -> SolveStep` preserves:
  - description
  - equation
  - importance
  - substeps

### P1.3 Integrate one consumer: `timeline_solve`

Goal:
- test whether solve events simplify a real consumer

Checklist:
- [x] Integrate event stream into `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/timeline_solve_eval.rs`
- [x] Keep public API unchanged
- [x] Keep old path available while verifying parity
- [x] Validate HTML/output parity in `cas_didactic`

Done when:
- `timeline_solve` can consume solve events without changing user-visible output

Current status:
- implemented as a roundtrip `DisplaySolveSteps -> SolverEvent -> DisplaySolveSteps`
- guarded by shape-preserving fallback to the original `DisplaySolveSteps`
- HTML + CLI parity coverage added in `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/solve_timeline_parity_tests.rs`
- validated with `cargo test -p cas_didactic --test solve_timeline_parity_tests`

### P1.4 Stop-and-evaluate checkpoint

Checklist:
- [x] Confirm the event consumer did not become meaningfully simpler
- [x] Confirm the coupling reduction is only an adapter seam, not a new boundary
- [x] Confirm the experiment avoided fragile solve runtime paths

Decision:
- [ ] Open a design step for native emission
- [x] Stop here and keep the adapter layer only

Current decision:
- keep the derived `SolveStep <-> SolverEvent` adapter layer
- do not open native solve event emission yet
- revisit only if a concrete consumer shows clear payoff beyond `timeline_solve`

Why:
- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/timeline_solve_eval.rs`
  still solves into `DisplaySolveSteps` first and only then roundtrips through
  `SolverEvent`
- `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/timeline/render_api.rs`
  still renders `&[SolveStep]`, so the didactic boundary did not get narrower
- the event model is useful as a stable core contract and test seam, but not
  enough to justify native solve-runtime emission work

## Track 3: Optional Solve Observer/Event Phase 2

### P2.1 Native emission design only

Goal:
- design, not implement blindly

Checklist:
- [ ] Identify stable emission boundaries in:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver_core/src/strategy_kernels.rs`
- [ ] Define which events are emitted:
  - main step
  - substep
- [ ] Explicitly exclude:
  - failed attempts
  - branch-internal noise
  - diagnostics deltas

Done when:
- there is a short design note for native solve event emission

### P2.2 Native emission implementation

Goal:
- only if P2.1 proves it is worth it

Checklist:
- [ ] Emit events at strategy boundaries
- [ ] Keep `SolveStep` compatibility
- [ ] Validate parity against current `timeline_solve`

Done when:
- native events can replace derived ones without regressions

## Track 4: Measured Memory Work

### R&D.1 Symbol storage benchmark

Goal:
- measure whether string storage is worth changing

Checklist:
- [ ] Choose benchmark workloads
- [ ] Measure current allocation behavior
- [ ] Prototype `smol_str` or equivalent
- [ ] Compare memory + runtime

Done when:
- there is benchmark data and a go/no-go recommendation

### R&D.2 AST sharing experiment

Goal:
- test `Rc`/arena ideas without contaminating mainline work

Checklist:
- [ ] Isolate prototype branch
- [ ] Measure clone-heavy workloads
- [ ] Compare ergonomics and regressions

Done when:
- there is a data-backed recommendation, not an intuition

## Track 5: E-Graph Exploration

### R&D.3 `egg` prototype

Goal:
- assess whether one rule family benefits from equality saturation

Checklist:
- [ ] Choose one narrow target:
  - trig identities
  - fraction normalization
  - rational canonicalization
- [ ] Prototype outside the main migration stream
- [ ] Define extraction cost function
- [ ] Compare:
  - correctness
  - complexity
  - maintainability

Done when:
- there is a prototype note and a decision whether to proceed

## Suggested PR Order

### PR 1
- [x] `P0.1` transport naming audit/fixes, if any clear wins remain

### PR 2
- [x] `P0.2` `cas_session_core` review and decision

### PR 3
- [x] `P0.3` final `cas_math` leak review, only for clearly engine-facing cases

### PR 4
- [x] `P1.1` add `SolverEvent` core model

### PR 5
- [x] `P1.2` add `SolveStep <-> SolverEvent` adapters

### PR 6
- [x] `P1.3` integrate `timeline_solve` and validate renderer parity

### PR 7
- [x] `P1.4` stop-and-evaluate checkpoint

Only after PR 7:
- [x] decide whether to open `P2`
- [ ] or shift to memory / `egg` exploration

## Explicit Do-Not-Do List

- [ ] Do not rename `cas_solver`/`cas_solver_core` in the same stream as event work.
- [ ] Do not adopt `egg` in the mainline without a prototype branch.
- [ ] Do not introduce `Rc<Expr>` or arena ownership changes without benchmarks.
- [ ] Do not keep moving tiny helpers one by one once the next candidate is debatable.
- [ ] Do not expand the solve event model beyond step/substep in Phase 1.

## Exit Criteria

We stop this next-level architecture effort when:

- [x] no obvious domain leaks remain
- [x] solve-event Phase 1 has a clear yes/no outcome
- [x] any further work would require speculative redesign rather than bounded improvement

At that point, the next step is not "more migration".
It is either:

- a dedicated performance program
- a dedicated solver-event architecture program
- or no further architecture change at all
