# Solver Event Phase 2 Decision

## Decision

Stop at `SolverEvent` Phase 1 for now.

Keep:

- the core `SolverEvent` model
- the `SolveStep <-> SolverEvent` adapters
- the current `timeline_solve` roundtrip integration

Do **not** open native solve-event emission at this time.

## Why

Phase 1 already delivered the only clearly justified value:

- a stable solve-event contract in `cas_solver_core`
- a lossless adapter seam for tests and consumers
- parity-checked integration for `timeline_solve`

But it did **not** prove the stronger claim that native emission would simplify
the architecture enough to justify touching solver runtime internals.

The current reality is still:

1. `timeline_solve` solves into `DisplaySolveSteps` first.
2. The event stream is derived from that stable step model.
3. `cas_didactic` still fundamentally consumes solve-step-shaped data.

So native emission would currently add runtime plumbing before it removes any
meaningful consumer complexity.

## Retained Policy

1. Keep `SolverEvent` as a stable semantic contract and adapter seam.
2. Do not thread solve-event listeners through strategy/runtime internals yet.
3. Do not expand the event model beyond step/substep in this track.

## Revisit Conditions

Reopen Phase 2 only if at least one of these becomes true:

1. a new consumer can use `SolverEvent` directly and would clearly avoid
   `SolveStep` materialization
2. `timeline_solve` or another didactic consumer becomes awkward specifically
   because native solve-event emission is missing
3. a solver-runtime refactor already introduces stable strategy boundaries
   where event emission would be cheap and semantically obvious

## Practical Outcome

The solve-event question now has an explicit answer:

- `Phase 1`: retained
- `Phase 2`: no-go for now

That means the next architecture work should move to other fronts unless one of
the revisit conditions becomes real.
