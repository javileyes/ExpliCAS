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
- [ ] Audit `cas_solver/src/json/*`
- [ ] Audit `cas_didactic/src/eval_json_*`
- [ ] Mark each case as:
  - `transport boundary`
  - `typed model builder`
  - `mixed`
- [ ] Rename only the clearly mixed cases
- [ ] Do not rename boundary modules that truly own wire/DTO assembly

Done when:
- internal naming no longer implies JSON when the module is not actually doing
  JSON transport work

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
- [ ] Audit only modules still suspected to be rule-facing or user-facing
- [ ] For each candidate, answer:
  - is this deterministic reusable math?
  - does it mainly exist for one engine rule?
  - does it contain `desc`/policy/runtime packaging?
- [ ] Move or clean up only the clearly engine-facing candidates
- [ ] Stop as soon as remaining candidates become debatable

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
- [ ] Validate HTML/output parity in `cas_didactic`

Done when:
- `timeline_solve` can consume solve events without changing user-visible output

Current status:
- implemented as a roundtrip `DisplaySolveSteps -> SolverEvent -> DisplaySolveSteps`
- guarded by shape-preserving fallback to the original `DisplaySolveSteps`
- suitable for architectural validation
- not yet strong enough to claim full renderer parity

### P1.4 Stop-and-evaluate checkpoint

Checklist:
- [ ] Did event consumers become simpler?
- [ ] Did the model reduce coupling in a meaningful way?
- [x] Did it avoid touching fragile solve runtime paths?

Decision:
- [ ] If yes, open a design step for native emission
- [x] If no, stop here and keep the adapter layer only

Current decision:
- keep the derived `SolveStep <-> SolverEvent` adapter layer
- do not open native solve event emission yet
- revisit only if a concrete consumer shows clear payoff beyond `timeline_solve`

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
- [ ] `P0.1` transport naming audit/fixes, if any clear wins remain

### PR 2
- [ ] `P0.2` `cas_session_core` review and decision

### PR 3
- [ ] `P0.3` final `cas_math` leak review, only for clearly engine-facing cases

### PR 4
- [x] `P1.1` add `SolverEvent` core model

### PR 5
- [x] `P1.2` add `SolveStep <-> SolverEvent` adapters

### PR 6
- [x] `P1.3` integrate `timeline_solve` (minimal adapter path; renderer parity still pending)

### PR 7
- [ ] `P1.4` stop-and-evaluate checkpoint

Only after PR 7:
- [ ] decide whether to open `P2`
- [ ] or shift to memory / `egg` exploration

## Explicit Do-Not-Do List

- [ ] Do not rename `cas_solver`/`cas_solver_core` in the same stream as event work.
- [ ] Do not adopt `egg` in the mainline without a prototype branch.
- [ ] Do not introduce `Rc<Expr>` or arena ownership changes without benchmarks.
- [ ] Do not keep moving tiny helpers one by one once the next candidate is debatable.
- [ ] Do not expand the solve event model beyond step/substep in Phase 1.

## Exit Criteria

We stop this next-level architecture effort when:

- [ ] no obvious domain leaks remain
- [ ] solve-event Phase 1 has a clear yes/no outcome
- [ ] any further work would require speculative redesign rather than bounded improvement

At that point, the next step is not "more migration".
It is either:

- a dedicated performance program
- a dedicated solver-event architecture program
- or no further architecture change at all
