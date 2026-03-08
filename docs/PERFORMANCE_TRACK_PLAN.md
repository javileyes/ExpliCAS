# Performance Track Plan

## Goal

Shift the project from architecture migration to measurable performance work.

This track exists to answer one question:

- where do we get the highest runtime payoff without reopening large architectural
  changes?

It should be treated as a separate workstream from migration.

## Current Baseline

Known baseline at the end of the pragmatic migration:

- `make ci`: green
- `cargo bench`: working
- simplification metamorphic benchmark:
  - command:
    - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
  - current result:
    - `numeric-only = 185`

This baseline is the reference point for future optimization work.

## Non-Goals

Do not mix this track with:

- additional crate splitting
- more `desc` migration from `cas_math`
- solver event architecture changes
- AST ownership redesign
- E-graph experiments

Those are separate tracks and should only be reopened with explicit justification.

## Priority Order

### P0. Keep current baseline stable

Before any optimization PR:

- `cargo fmt --all`
- `cargo check -p cas_solver_core -p cas_engine -p cas_solver -p cas_session -p cas_cli -p cas_didactic`
- `make ci`

If any of those regress, stop and fix them first.

### P1. Simplification quality and benchmark stability

Primary metric:

- metamorphic simplification `numeric-only`

Commands:

```bash
cargo test --release -p cas_engine --test metamorphic_simplification_tests \
  metatest_unified_benchmark -- --ignored --nocapture
```

Optional targeted diagnostics:

```bash
METATEST_DIAG=1 cargo test -p cas_engine --test metamorphic_simplification_tests \
  metatest_individual -- --ignored --nocapture
```

Success criteria:

- `numeric-only` does not regress
- any optimization PR should ideally improve or hold the current `185`

### P2. Solver correctness/identity guardrail

Command:

```bash
METATEST_VERBOSE=1 cargo test --release -p cas_engine --test metamorphic_equation_tests \
  metatest_equation_identity_transforms -- --ignored --nocapture
```

Success criteria:

- `incomplete: 0`
- `mismatch: 0`

### P3. Cache/build-path performance

Relevant benchmark:

- `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`

Command:

```bash
cargo bench -p cas_engine --bench profile_cache
```

Success criteria:

- cached path is not slower than baseline
- uncached path does not regress significantly

### P4. End-to-end REPL cost

Relevant benchmark:

- `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/repl_end_to_end.rs`

Command:

```bash
cargo bench -p cas_engine --bench repl_end_to_end
```

Watch:

- `cached/batch_11_inputs`
- `uncached/batch_11_inputs`
- `steps_on`
- `steps_compact`
- `steps_off`

Success criteria:

- no large regressions in cached mode
- steps-off remains the fastest mode

## Candidate Optimization Areas

These are the only areas that currently have enough payoff to justify work.

### 1. Profile cache hit path

Files:

- `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/profile_cache.rs`
- `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`

Why:

- clear benchmark coverage
- architecture already stable
- low risk compared to symbolic rule changes

### 2. Simplification orchestration overhead

Files:

- `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/orchestration.rs`
- `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/mod.rs`
- `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/step_recording.rs`

Why:

- affects every simplify path
- likely cheaper to optimize than rethinking rewrite semantics

### 3. High-frequency canonicalization and trig paths

Files:

- `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/canonicalization.rs`
- `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/trigonometry/`

Why:

- these paths dominate metamorphic traffic
- improvements here can lower `numeric-only` or runtime noise

Constraint:

- only optimize after measuring
- do not refactor semantics blindly

## Workflow Per Optimization PR

Each performance PR should follow this exact loop:

1. Capture baseline
2. Apply one optimization hypothesis
3. Re-run:
   - `make ci`
   - simplification metamorphic benchmark
   - equation identity metamorphic benchmark if solver-facing
   - the relevant `cargo bench`
4. Record before/after numbers in the PR

Do not batch unrelated optimizations together.

## Stop Conditions

Stop an optimization branch when any of these happens:

- runtime win is within noise
- `numeric-only` regresses
- equation identity regresses
- code complexity increase is larger than the measured win

## Deferred R&D

These ideas stay deferred until there is benchmark evidence that current paths
cannot deliver enough payoff:

- `Rc<Expr>` or arena ownership
- symbol interning changes
- E-graphs / `egg`
- full solver-native event emission

Those are valid projects, but not part of this performance track.
