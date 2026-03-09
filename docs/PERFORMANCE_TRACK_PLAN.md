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
    - `numeric-only = 168`

This baseline is the reference point for future optimization work.

## Measured Notes

Recent validated win:

- `Simplifier::from_profile_with_context(...)` removes the throwaway `Context::new()`
  in cached-profile construction.
- It improved `profile_cache` microbenchmarks on the cached path and held the
  simplification metamorphic benchmark at `numeric-only = 168`.
- `ParentContext` hot-path construction now reuses shared `PatternMarks` via
  `Rc` and rebuilds ancestor-derived fields in one pass instead of cloning the
  marks or extending ancestor context incrementally per node.
- In fast local runs, this improved `repl_end_to_end`:
  - `repl_full_eval/cached/batch_11_inputs`: about `2.28 ms` -> `2.23-2.26 ms`
  - `steps_mode_comparison/batch_11/steps_on`: Criterion reported about
    `5.5-6.5%` improvement, landing at `2.20-2.23 ms`
  while the simplification metamorphic benchmark still held
  `numeric-only = 168`.
- Cached `RuleProfile` instances now also carry rule buckets prefiltered by
  simplification phase, so `from_profile_with_context(...)` can skip the
  `allowed_phases()` check inside the per-node hot loop.
- In fast local runs, this improved cached execution paths:
  - `repl_full_eval/cached/batch_11_inputs`: `2.220-2.247 ms`
    with Criterion reporting about `5.0-7.3%` improvement
  - `steps_mode_comparison/batch_11/steps_on`: `2.216-2.248 ms`
    with Criterion reporting about `5.0-7.2%` improvement
  - `profile_cache/simplify_cached_vs_uncached/cached/light/x_plus_1`:
    `46.7-48.1 µs` with Criterion reporting about `16-18%` improvement
  while the simplification metamorphic benchmark still held
  `numeric-only = 168`.
- `Simplifier::from_profile_with_context(...)` now keeps cached-profile rule
  buckets borrowed from `RuleProfile` instead of cloning `rules` and
  `global_rules` up front. The simplifier only materializes owned rule maps
  again if a caller mutates the profile-backed instance through
  `disable_rule`, `enable_rule`, or `add_rule`.
- Combined with the phase-prefiltered buckets, this lets the cached hot path
  skip both the rule-clone cost at construction time and the per-rule
  disabled-check in the node loop.
- In fast local runs, this improved cached execution paths further:
  - `repl_full_eval/cached/batch_11_inputs`: `2.115-2.139 ms`
    with Criterion reporting about `2.8-4.7%` improvement
  - `profile_cache/simplify_cached_vs_uncached/cached/light/x_plus_1`:
    `44.1-44.9 µs` with Criterion reporting about `5.1-7.6%` improvement
  while `cargo test -p cas_engine profile_cache_tests --lib` stayed green.
- `ParentContext` now stores runtime ancestors in `SmallVec<[ExprId; 8]>`
  instead of `Vec<ExprId>`, so the common shallow recursion path avoids heap
  allocation while keeping the same slice-based API for rules.
- In fast local runs, this improved cached execution paths further:
  - `repl_full_eval/cached/batch_11_inputs`: `1.954-1.971 ms`
    with Criterion reporting about `7.3-9.1%` improvement
  - `profile_cache/simplify_cached_vs_uncached/cached/light/x_plus_1`:
    `42.6-43.4 µs` with Criterion reporting about `2.6-4.9%` improvement
  while both `parent_context::tests` and the simplification metamorphic
  benchmark still held green with `numeric-only = 168`.
- The transformer's own `ancestor_stack` now also uses `SmallVec<[ExprId; 8]>`,
  removing another shallow-recursion allocation from the main rewrite walk.
- In fast local runs, this improved:
  - `repl_full_eval/cached/batch_11_inputs`: `1.918-1.938 ms`
    with Criterion reporting about `1.3-2.5%` improvement
  - `profile_cache/simplify_cached_vs_uncached/cached/light/x_plus_1` stayed
    statistically flat (`42.3-43.1 µs`, no significant change)
  while the simplification metamorphic benchmark still held
  `numeric-only = 168`.
- The step-trace hot path now also keeps the transformer's `current_path` and
  the stored `StepMeta.path` in `SmallVec<[PathStep; 8]>`, avoiding extra heap
  traffic in the common shallow-path case while preserving the slice-based API.
- In fast local runs, this improved steps-heavy cached execution paths further:
  - `repl_full_eval/cached/batch_11_inputs`: `1.876-1.914 ms`
    with Criterion reporting about `1.5-3.4%` improvement
  - `steps_mode_comparison/batch_11/steps_on`: `1.893-1.924 ms`
    with Criterion reporting about `13.2-14.9%` improvement
  while the simplification metamorphic benchmark still held
  `numeric-only = 168`.
- `reconstruct_at_path(...)` now fast-paths root-level rewrites: when the
  current path is empty, it updates `root_expr` directly instead of rebuilding
  an empty path and re-entering the generic AST rewrite helper.
- In fast local runs with the named baselines `steps_on_pre` /
  `steps_compact_pre`, this held the step-heavy path flat while improving the
  broader cached end-to-end flow:
  - `repl_full_eval/cached/batch_11_inputs`: `1.988-2.057 ms`
    with Criterion reporting about `1.8-4.7%` improvement
  - `steps_mode_comparison/batch_11/steps_on`: `1.996-2.036 ms`
    stayed within noise against the named baseline
  - `steps_mode_comparison/batch_11/steps_compact`: `1.951-1.971 ms`
    also stayed within noise against the named baseline
  while the simplification metamorphic benchmark still held
  `numeric-only = 168`.
- `Simplifier::from_profile_with_context(...)` no longer clones
  `disabled_rules` up front on the cached-profile fast path. It now borrows
  that set from `RuleProfile` until a caller mutates the simplifier and forces
  materialization.
- Coverage now includes a fast-path contract check in `profile_cache_tests`
  to ensure `from_profile(...)` exposes the profile-disabled set before
  materialization and still materializes correctly on later mutation.
- In fast local runs, this improved the cached micro path:
  - `profile_cache/simplify_cached_vs_uncached/cached/light/x_plus_1`:
    `39.683-40.051 µs` with Criterion reporting about `11-29%` improvement
  - `repl_full_eval/cached/batch_11_inputs`: one run reported
    `1.962-2.011 ms` with about `4.8-7.2%` improvement, and a follow-up rerun
    landed at `1.936-1.979 ms` (not statistically significant, but still below
    the pre-change absolute range)
  - `steps_mode_comparison/batch_11/steps_on`: `1.951-1.999 ms`, flat against
    the named baseline
  while the simplification metamorphic benchmark still held
  `numeric-only = 168`.

Recent rejected hypotheses:

- reducing `StepsMode::Compact` payloads inside `step_recording` did not produce
  a meaningful `repl_end_to_end` win; `steps_compact` stayed effectively tied
  to `steps_on`
- a second pass on `StepsMode::Compact` with named Criterion baselines also
  came back flat-to-worse and regressed the broader cached flow, so it was
  reverted again
- replacing `pathsteps_to_expr_path(...)` with a local `SmallVec<[u8; 8]>`
  inside `reconstruct_at_path(...)` regressed all measured step-heavy benches,
  so it was reverted
- sharing rule/profile collections via extra `Arc` layers regressed both
  `profile_cache` and `repl_end_to_end` in measured runs
- collapsing `BestSoFar::score_expr` into a single traversal looked plausible
  but regressed the fast `repl_end_to_end` benchmark, so it was reverted
- reusing scratch `Vec`s for `pattern_scanner` / `auto_expand_scan` regressed
  `repl_end_to_end` and was reverted
- skipping `auto_expand_scan` when `expand_policy = Off` looked attractive, but
  the marks are still consumed by context-aware expansion/log rules in standard
  mode; measured runs regressed and the change was reverted
- replacing `pathsteps_to_expr_path(...)` with a generic mapped-path rewrite
  helper in `cas_math::expr_path_rewrite` stayed flat-to-worse against the
  named baseline `pathmap_pre`, so it was reverted
- making `Step::after_str` lazy in the core `Step` constructor also stayed
  within noise on `steps_on` / cached end-to-end runs, so it was reverted
- sharing `ParentContext` implicit-domain state with `Rc` reduced clone cost in
  theory, but measured runs stayed within noise and the change was reverted
- pre-reserving a fixed `Vec<Step>` capacity in transformer construction also
  stayed within noise and was reverted

Decision:

- keep the simpler pre-change implementations for those rejected paths
- do not reopen them without a stronger benchmark-driven reason

Fast local loop:

- set `CAS_BENCH_FAST=1` to run `cas_engine` Criterion benches with shorter
  warmup/measurement windows
- this mode is for local iteration only; keep the default configuration for
  before/after numbers that will be used as a real benchmark reference
- when testing speculative micro-opts, prefer a named baseline first, e.g.
  `--save-baseline pathmap_pre`, and compare follow-up runs with
  `--baseline pathmap_pre` so reverted experiments do not contaminate the
  reference

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
- any optimization PR should ideally improve or hold the current `168`

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
