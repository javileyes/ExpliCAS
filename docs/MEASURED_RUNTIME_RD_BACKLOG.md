# Measured Runtime R&D Backlog

## Purpose

This file starts after the bounded architecture follow-up work is done.

It exists to keep runtime experimentation disciplined:

- one experiment at a time
- benchmark-gated
- no architecture churn by intuition
- no mainline retention without a clear measured win

## Current Policy

This is **not** a refactor backlog.

Any experiment here must:

1. start from an explicit hotspot hypothesis
2. use existing benchmarks where possible
3. add dedicated measurement only if the current harness is insufficient
4. define success and rejection criteria before implementation
5. leave the mainline unchanged if the result is flat or worse

## Active Experiment

### 1. Listener Overhead In Event-Enabled Hot Paths

Status:
- done (rejected for retention)

Why this experiment:

- the session-eval path already proved that avoiding unnecessary collector
  installation on `steps = off` can be materially worthwhile
- the remaining open listener question is narrower and more honest:
  what does the current event-listener plumbing cost **when a listener is
  actually attached**, and can that overhead be reduced without widening
  architecture churn?

Relevant existing context:

- `cas_engine` still exposes listener plumbing through
  `Option<Box<dyn StepListener>>`
- listener-aware execution is threaded through:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/simplifier.rs`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/orchestration.rs`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/step_recording.rs`
- `steps = off` collector installation was already narrowed in:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/eval_command_runtime/prepare.rs`

### Hypothesis

The current attached-listener path has measurable overhead from:

- boxed trait-object dispatch
- listener install/replace plumbing
- event sink cloning/allocation patterns

and a narrower listener path can improve listener-enabled benches without
changing rendering, event semantics, or solver architecture.

### Allowed scope

Allowed:

- a narrow benchmark-driven prototype in the listener path
- adding one dedicated listener-enabled benchmark if existing benches are too
  indirect
- local plumbing changes in `cas_engine` listener handling

Not allowed:

- reopening `SolverEvent` Phase 2
- native solve-event emission
- broad generic conversion across the whole engine
- semantic changes to `Step`, `EngineEvent`, or didactic outputs

### Baseline measurements

Use the existing benches first:

1. `cargo bench -p cas_engine --bench repl_end_to_end 'steps_mode_comparison' -- --noplot`
2. `cargo bench -p cas_engine --bench profile_cache 'solve_prepass_inherited_steps_cached/steps_on_batch' -- --noplot`
3. `cargo bench -p cas_engine --bench profile_cache 'solver_verification_inherited_steps' -- --noplot`

If those are too indirect for attached-listener cost, add one dedicated bench
that compares:

- same workload with no listener
- same workload with a capturing listener attached

That dedicated bench should stay small and live next to the existing engine
bench harnesses.

### Candidate implementation shapes

Try only one of these at a time:

1. a narrower install/restore path that avoids unnecessary boxing churn
2. a cheaper listener handoff inside the transformer scope
3. a benchmark-only prototype for a generic or borrowed listener path

Do not try multiple shapes in the same retention step.

### Success criteria

Retain the experiment only if all of these are true:

1. there is a reproducible win of at least `5%` on at least one
   listener-enabled primary benchmark
2. there is no regression worse than `2%` on the neighboring non-listener
   benchmarks that share the same path
3. event semantics and step parity stay unchanged
4. `make ci` stays green

If a dedicated listener benchmark is added, its primary target becomes the
experiment's success gate.

### Outcome

Dedicated benchmark retained in:

- `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`
  as `listener_overhead_solve_prepass`

Measured baseline on the dedicated benchmark:

- `steps_off/no_listener`: about `260-262 us`
- `steps_off/counting_listener`: about `293-295 us`
- `steps_off/collector_listener`: about `293-295 us`
- `steps_on/no_listener`: about `260-262 us`
- `steps_on/counting_listener`: about `294-296 us`
- `steps_on/collector_listener`: about `293-295 us`

Interpretation:

- the dominant attached-listener overhead is already present in the base event
  emission path
- `EngineEventCollector` is not materially worse than a minimal counting
  listener on this workload
- this makes collector-specific churn a weak next move without a new hotspot
  hypothesis

Prototype tried:

- `EngineEvent.rule_name` using a small-string representation instead of
  `String`

Result:

- listener-enabled cases improved only around `~1-2%`
- no-listener neighbors stayed flat
- the change did **not** hit the required `>=5%` win

Retention decision:

- reject the production change
- keep the dedicated benchmark harness
- do not open a second runtime experiment without a fresh measured hypothesis

### Validation

Minimum validation before retaining any change:

- `cargo test -p cas_engine events_tests --lib -- --nocapture`
- `cargo test -p cas_didactic --test solve_timeline_parity_tests -- --nocapture`
- the chosen benchmark compare against a named baseline
- `make ci`

### Rejection criteria

Reject and revert the experiment if:

- the win is within noise
- the improvement exists only on a synthetic microbench and not on the
  listener-enabled workload
- it requires broad API genericization to keep
- or it complicates the engine/runtime boundaries more than it helps

## Stop Condition

Stop this backlog when:

- the listener experiment is either retained or explicitly rejected
- and no second runtime experiment is opened without a fresh benchmark-backed
  reason
