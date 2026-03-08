# Architecture Next-Level Plan

## Purpose

This document turns the post-migration feedback into an execution plan.

It is intentionally pragmatic:

- preserve the value already achieved by the migration
- avoid large churn with weak payoff
- separate "must fix now" from "worth exploring later"
- require validation gates at each step

This is **not** a mandate to keep refactoring indefinitely. It is a decision
framework and phased plan for the next architectural improvements.

## Current Baseline

The pragmatic migration is already in a strong state:

- Android is out of the Rust workspace path.
- `cas_ast`, `cas_parser`, `cas_formatter`, `cas_api_models` are split.
- `cas_math` contains the heavy algebraic machinery.
- `cas_session` owns session/state concerns.
- `cas_didactic` owns most educational/timeline concerns.
- `cas_cli` is much thinner than before.
- `cas_solver_core` already acts as a low-level shared semantic layer.

This means the next phase is **not** about bulk extraction.
It is about sharpening boundaries where architectural value is still high.

## Guiding Principles

1. Do not rename or move large subsystems unless the move eliminates real
   coupling, not just naming discomfort.
2. Do not optimize memory or algorithms before measuring.
3. Do not adopt a new symbolic paradigm (`egg`, arenas, interning, `Rc`) inside
   the migration stream without an isolated benchmark track.
4. One architectural goal per PR.
5. Every step must keep:
   - `make ci` green
   - metamorphic regressions monitored
   - public API churn minimized

## Executive Summary

### High-priority work

1. Clarify the application layer around `cas_solver` vs orchestration naming.
2. Finish purifying `cas_math` so only deterministic/pure math stays there.
3. Remove transport-specific naming and responsibilities from internal crates.
4. Decide whether `cas_session_core` still earns its existence.
5. Complete the solve observer/event design only if the model proves useful.

### Medium-priority work

6. Introduce measured AST-memory improvements.
7. Introduce zero-cost listener plumbing in hot paths where it matters.

### Exploratory work

8. Evaluate an E-graph track in isolation, behind a feature flag or prototype.

## Reality Check on the Feedback

The feedback is directionally correct, but not every proposal has the same
priority or ROI.

### Agreed

1. `cas_solver` currently behaves more like an application/orchestration layer
   than a pure solver.
2. `cas_math` still contains some residual helpers that are not truly pure.
3. Internal crates should not be named around a transport format when they are
   really building typed Rust outputs.
4. `_core` crates should be treated as tactical tools, not permanent clutter.

### Partially agreed

1. Renaming `cas_solver` -> `cas_orchestrator` and `cas_solver_core` ->
   `cas_solver` is architecturally coherent, but it is also high-churn and
   low-functional-value.
2. `cas_session_core` may be mergable, but only if that removes complexity
   without reintroducing cycles.

### Deliberately deferred

1. `Rc<Expr>`, arenas, or AST-sharing changes.
2. E-graphs / `egg`.
3. Broad crate renaming.

Those are valid ideas, but they belong to dedicated tracks with benchmarks and
success criteria.

## Phase A: Seal Remaining Domain Leaks

### Goal

Finish the high-value boundary cleanup without reopening large migration fronts.

### A1. Reclassify `cas_solver`

#### Problem

`cas_solver` still contains a mix of:

- command evaluation
- JSON-oriented outputs
- REPL/runtime orchestration
- true solve logic

#### Target

Treat the current `cas_solver` as the application/solver facade, and document
the distinction clearly.

#### Plan

1. Short term:
   - keep crate names stable
   - document the roles explicitly
   - finish moving pure DTOs/contracts into `cas_solver_core`
2. Medium term:
   - only rename crates if the team explicitly decides that naming confusion is
     an operational problem

#### Decision

Do **not** rename crates immediately.

Reason:
- high churn
- little functional payoff
- likely to create wide PR noise

Instead:
- add explicit role documentation
- treat rename as a separate, optional cleanup track

### A2. Purify `cas_math`

#### Goal

Keep only deterministic, reusable mathematical logic in `cas_math`.

#### What should stay

- multivariate polynomial machinery
- mod-p arithmetic
- exact gcd algorithms
- matrix algebra
- canonical lowering passes when they are purely algebraic
- deterministic AST-to-math transforms

#### What should not stay

- rule narration
- user-facing `desc`
- engine-specific default policies
- convenience wrappers that only package `Rewrite`
- didactic/support code tied to one engine rule

#### Status

A large amount of this cleanup is already done.

#### Remaining rule

Any candidate move must pass this test:

> If the module exists mainly to help one engine rule decide or narrate a
> rewrite, it does not belong in `cas_math`.

### A3. Remove transport-specific semantics from internal layers

#### Goal

Internal layers should produce typed Rust models, not "JSON" concepts.

#### Plan

1. Audit names like:
   - `eval_json_*`
   - `json/*`
   - `*_json_*`
2. Split into:
   - typed model building
   - actual serialization boundary

#### Rules

- `serde_json::to_string(...)` belongs only in true outer boundaries.
- Internal crates may still use DTO structs from `cas_api_models`.
- Internal code should talk about "presentation", "output", "payload",
  "envelope", "steps", not "json", unless it is truly transport code.

#### Priority

Medium.

This is worth doing only where naming still causes real confusion or transport
coupling.

### A4. Review `_core` crates

#### Goal

Decide which `_core` crates are tactical and which are actually good permanent
abstractions.

#### Recommendation

1. `cas_solver_core`
   - likely worth keeping
   - it already acts as the shared semantic layer for:
     - solve runtime types
     - engine events
     - command DTOs
     - shared contracts

2. `cas_session_core`
   - reviewed
   - keep as a permanent low-level crate
   - do not merge back into `cas_session`

#### Decision rule

Do not merge a `_core` crate just because its name looks temporary.
Merge only if the graph becomes simpler.

#### Current decision

`cas_session_core` stays.

Reason:

- `cas_engine` depends on session-eval contracts (`EvalSession`, `EvalStore`,
  prepared dispatch helpers, diagnostics accumulation helpers).
- `cas_solver` depends on low-level store/entry/snapshot/reference types and
  stateless eval adapters.
- `cas_session` depends on both `cas_engine` and `cas_solver`.

Merging `cas_session_core` back into `cas_session` would either:

- reintroduce dependency cycles, or
- force `cas_engine`/`cas_solver` to depend on the full stateful session crate.

That is worse than the current graph.

So `cas_session_core` should be treated as a **permanent shared kernel crate**
for:

- store and snapshot primitives
- ref resolution helpers
- stateless/session-agnostic eval contracts
- small shared DTOs needed below `cas_session`

## Phase B: Observer/Event Completion

### Goal

Finish only the valuable part of the observer pattern.

### Current state

Already working:

- engine expression events
- `eval-json` fallback from `EngineEvent`
- `timeline_simplify` fallback from `EngineEvent`
- `full_simplify` fallback from `EngineEvent`

### Open architectural question

`solve` still lacks a true equation-aware event model.

### Plan

1. Keep the design in:
   - `/Users/javiergimenezmoya/developer/math/docs/SOLVER_EVENT_OBSERVER.md`
2. Implement only Phase 1 there:
   - minimal `SolverEvent`
   - collector
   - `SolveStep <-> SolverEvent` adapters
3. Integrate one consumer:
   - `timeline_solve`
4. Reevaluate before native emission

### Stop condition

If event consumers do not simplify materially after the adapter phase, stop.
Do not force native emission.

## Phase C: Measured Memory Improvements

### Goal

Improve allocator pressure and clone behavior with evidence, not ideology.

### C1. Symbol intern/storage benchmark

#### Candidates

- `smol_str`
- `ustr`
- a lightweight intern table if needed

#### Preconditions

Measure:

- allocation count
- parse-heavy workloads
- simplify-heavy workloads
- solve-heavy workloads

#### Success criteria

- lower allocation pressure
- no semantic churn
- no large ergonomics cost

### C2. AST sharing experiment

#### Candidates

- `Rc<Expr>`
- `Arc<Expr>`
- arena-style storage

#### Important note

This is a separate track.
It should not be mixed into the main architecture cleanup PR stream.

#### Why

It changes ownership semantics everywhere and is very easy to over-engineer.

### C3. Zero-cost listeners

#### Goal

Where event collection stays in hot paths, prefer generics when the API permits.

#### Plan

1. Identify hot event-enabled paths
2. Benchmark current trait-object overhead
3. Only then decide whether a generic `NoOpListener` path is worth it

#### Decision

Do not blanket-convert listener APIs to generics without profiling.

## Phase D: E-Graph Exploration

### Goal

Evaluate whether equality saturation can replace selected rule clusters.

### Important

This is not a migration step.
This is an R&D track.

### Why not now

`egg` is powerful, but adopting it well requires:

- choosing the target rule family
- building extraction cost functions
- validating determinism and didactic compatibility
- understanding where equation solving still needs imperative control

### Recommended scope

Run it as a prototype on a contained domain:

- trig identities
- rational canonicalization
- fraction normalization

### Deliverable

A benchmark-backed design note, not a direct merge.

## Validation Matrix

Every phase should keep this matrix green:

1. `make ci`
2. `cargo bench --no-run`
3. `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
4. equation identity metamorphic checks when solve-related code changes

When a phase is performance-oriented, add:

5. a before/after benchmark snapshot

## Recommended Execution Order

### Immediate

1. Finish the few remaining high-value purity leaks.
2. Review remaining transport naming that still causes confusion.
3. Decide whether `cas_session_core` should stay or merge.

### Next

4. Implement `SolverEvent` Phase 1 only.

### After that

5. Pause and evaluate whether solve events gave real value.

### Separate track

6. Run AST memory experiments.

### Separate exploratory track

7. Prototype `egg` on one isolated rule family.

## PR Strategy

Each PR should have exactly one of these goals:

1. move one pure DTO/contract set down one layer
2. move one rule-facing helper out of `cas_math`
3. complete one bounded event-system phase
4. run one measured performance experiment

Avoid PRs that combine:

- naming cleanup
- performance changes
- event architecture
- algorithm changes

## Done Criteria

The next-level architecture effort is done when:

1. `cas_math` contains only math-pure or clearly justified low-level support.
2. `cas_session` depends on `cas_solver_core` for shared DTOs/contracts wherever
   practical.
3. transport-specific naming is limited to true transport boundaries.
4. the solve event question has a concrete answer:
   - implemented and useful
   - or intentionally stopped after adapter phase
5. performance experiments are documented with data, not assumptions.

## Recommendation

Proceed, but in this order:

1. high-value boundary cleanup
2. solve-event Phase 1
3. stop-and-evaluate
4. only then performance and E-graph exploration

That gives the project a real next level without turning a successful migration
into endless architectural churn.
