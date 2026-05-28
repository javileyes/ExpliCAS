# Architecture Follow-Up Backlog

## Purpose

This file starts where
[`ARCHITECTURE_BOUNDARY_CLEANUP_BACKLOG.md`](/Users/javiergimenezmoya/developer/math/docs/ARCHITECTURE_BOUNDARY_CLEANUP_BACKLOG.md)
stops.

It exists to keep the next architecture work concrete and bounded now that:

- the high-value boundary cleanup is complete
- `make ci` is green again
- the remaining candidates are no longer low-risk move-only changes

## Closed Track

The previous boundary-cleanup track is complete for its intended scope:

- no crate renames were needed
- `cas_solver_core` no longer owns the obvious history/health/DTO leaks
- owned command DTOs now live in `cas_api_models`
- borrowed REPL parse types now live in `cas_solver`
- `cas_didactic` no longer keeps large static HTML/CSS payloads in Rust source

That means the next work should not keep pretending the same backlog is still
open.

## Current Active Track: Calculus Architecture Pressure

Status:
- active

Why it matters:
- the calculus strategy has shifted from isolated verticals to generalized
  real-domain capability
- recent retained/rejected cycles show that the risky part is increasingly the
  pipeline boundary, not the formula itself
- repeated local helpers for domain conditions, post-calculus presentation,
  residual verification, and didactic steps can make each new `diff`, `limit`,
  or `integrate` family more fragile

Retained direction:
- treat this as localized architecture work under
  [ENGINE_COHESION_REFACTORING_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_COHESION_REFACTORING_STRATEGY.md)
  and
  [CALCULUS_ENGINE_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/CALCULUS_ENGINE_STRATEGY.md)
- classify each cycle by retained value: usually `observability` or
  `robustness`, sometimes `calculus` when the public capability itself changes
- extract before abstracting
- preserve route order and public behavior by default

Open bounded candidates:

1. Map one crowded calculus route family before moving code.
   - candidates: scaled-root inverse-family differentiation, reciprocal
     trig/hyperbolic integration products, or residual limit presentation
   - output: route owner, domain owner, presentation owner, verifier owner,
     relevant matrix rows, and guardrails

2. Extract one family-owned domain-condition builder.
   - goal: prevent late display cleanup from owning conditions it did not prove
   - retain only if matrix display remains stable and required conditions do
     not become broader or reordered accidentally

3. Extract one bounded antiderivative verification/residual route.
   - goal: keep verified primitives out of deep generic simplification
   - retain only if direct and nested residual probes stay bounded and
     antiderivative verification remains green

4. Extract one post-calculus presentation boundary.
   - goal: separate public result readability from internal canonical matching
   - retain only if existing matrix rows keep stable public presentation outside
     the intended family

5. Extract one didactic step/substep builder after the route policy is stable.
   - goal: remove repeated wording/highlight construction without changing
     mathematical routing
   - retain only if didactic audits and command step checks stay green

Stop or defer when:
- the candidate requires broad route reordering
- the shared helper would merge different domain or branch semantics
- the only measurable outcome is lower line count
- embedded or pressure lanes regress without a reusable robustness gain

## Next Candidates

### 1. Solver Event Phase 2 Design Only

Status:
- done

Why it still matters:
- Phase 1 is already done
- native emission is the only remaining open question in the solver-event track

Retained decision:
- stop at Phase 1 for now
- keep the adapter layer
- do not open native emission

Decision note:
- `/Users/javiergimenezmoya/developer/math/docs/SOLVER_EVENT_PHASE2_DECISION.md`

Reopen only if:
- a concrete consumer becomes meaningfully awkward behind the current
  `SolveStep <-> SolverEvent` adapter boundary
- or solver-runtime boundaries become stable enough that native emission is
  obviously cheap and semantically useful

### 2. Session Persistence Strategy Decision

Status:
- done

Why it matters:
- the persistence subtrack is already measured
- the real tradeoff is now product policy, not micro-optimization

Retained decision:
- keep the default as `atomic replace`
- do not expose a fast/non-atomic mode now
- do not expose a stronger-durability `sync` mode now

Decision note:
- `/Users/javiergimenezmoya/developer/math/docs/SESSION_PERSISTENCE_DECISION.md`

Known state:
- current snapshot save path is `write + flush + rename`
- overwrite/resave cost is materially above first-write cost
- stronger durability (`sync`) costs milliseconds, not microseconds

Reopen only if:
- session save latency becomes a proven product bottleneck
- or stronger durability becomes an explicit product requirement

### 3. Didactic Rendering Phase 2

Status:
- done

Why it matters:
- static payload extraction is done
- if further work happens here, it should be about renderer structure, not
  moving more strings around

Retained result:
- added tiny template helpers/macros in `timeline/render_template.rs`
- cleaned the repetitive adapter layer in `page_shell`, `solve_render`,
  `solve_timeline_render`, `simplify_step_html`, `simplify_substeps`, and
  `simplify_summary`
- kept the current static-asset plus typed-Rust-adapter model

Decision / execution note:
- `/Users/javiergimenezmoya/developer/math/docs/DIDACTIC_RENDERING_PHASE2_PLAN.md`

Do not do:
- broad frontend redesign inside Rust without a concrete UX goal

### 4. Measured Runtime R&D Only

Status:
- done for current scope (no active experiment)

Candidates:
- listener overhead in hot paths
- AST sharing / clone-pressure experiments
- isolated `egg` prototype

Rule:
- no mainline architecture churn without a benchmark-backed reason

Current execution note:
- `/Users/javiergimenezmoya/developer/math/docs/MEASURED_RUNTIME_RD_BACKLOG.md`

Current state:
- the listener-overhead experiment is closed and rejected for retention
- no second runtime experiment should be opened without a fresh measured
  hotspot hypothesis

## Suggested Execution Order

1. If calculus improvement cycles continue in crowded route families, choose the
   smallest open item from "Calculus Architecture Pressure" before adding more
   local variants.
2. If architecture follow-up outside calculus continues, open a fresh
   measured-runtime R&D experiment only with a new benchmark-backed hypothesis.
3. Reuse
   `/Users/javiergimenezmoya/developer/math/docs/MEASURED_RUNTIME_RD_BACKLOG.md`
   as the execution template and experiment log.

## Stop Condition

Stop architecture work in this stream when:

- the persistence strategy has an explicit decision
- the solver-event question has an explicit stop/go decision
- didactic rendering Phase 2 has either been completed or explicitly declined
- calculus route families no longer require repeated local helpers for the same
  detection/domain/verification/presentation/step pipeline shape
- any remaining ideas require speculative redesign rather than bounded payoff
