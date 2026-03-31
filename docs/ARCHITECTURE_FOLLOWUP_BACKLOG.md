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
- optional, benchmark-gated

Candidates:
- listener overhead in hot paths
- AST sharing / clone-pressure experiments
- isolated `egg` prototype

Rule:
- no mainline architecture churn without a benchmark-backed reason

## Suggested Execution Order

1. Optional didactic rendering Phase 2 cleanup
2. Any runtime R&D behind its own benchmark/prototype track

## Stop Condition

Stop architecture work in this stream when:

- the persistence strategy has an explicit decision
- the solver-event question has an explicit stop/go decision
- didactic rendering Phase 2 has either been completed or explicitly declined
- any remaining ideas require speculative redesign rather than bounded payoff
