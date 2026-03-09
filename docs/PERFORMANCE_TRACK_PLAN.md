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
- `apply_rules()` now decides once per node whether cached eval buckets still
  need runtime filtering. In the common `phase_prefiltered + Eval` fast path,
  both specific and global rule loops skip the per-rule `should_skip_rule(...)`
  call entirely because phase/disabled filtering has already been handled by
  the cached profile.
- That bypass is intentionally limited to `Eval`: cached `SolvePrepass` and
  `SolveTactic` flows still keep the runtime `solve_safety` gate for specific
  rules, now covered by `profile_cache_tests`.
- Validated with full Criterion runs against the named baseline `skip_eval_pre`:
  - `repl_full_eval/cached/batch_11_inputs`: `2.016-2.048 ms`
    with Criterion reporting about `4.9-7.1%` improvement
  - `steps_mode_comparison/batch_11/steps_on`: `1.982-2.001 ms`
    with Criterion reporting about `6.3-8.5%` improvement
  - `profile_cache/simplify_cached_vs_uncached/cached/light/x_plus_1`:
    `41.971-42.449 µs` with Criterion reporting about `2.9-5.4%` improvement
  while `solve_safety_contract_tests` stayed green and the simplification
  metamorphic benchmark still held `numeric-only = 168`.
- The new per-input solve benchmark diagnostics showed that
  `solve_modes_cached` in `SolvePrepass` / `SolveTactic(Strict)` was spending
  its visible work on a single cosmetic path:
  `(x^2 - y^2)/(x - y)` was being rewritten through `Sub -> Add(Neg)` and then
  a sign-cancel canonicalization, even though strict/prepass intentionally
  block the real fraction cancellation behind `x != y`.
- `Canonicalize Negation` and `Cancel Fraction Signs` now skip that purely
  cosmetic double-implicit-negation path only in `SolvePrepass` and
  `SolveTactic(Strict)`, while still allowing explicit `(-A)/(-B)` cleanup and
  keeping the generic/assume tactic behavior unchanged.
- In fast local runs this improved the dedicated solve batches substantially:
  - `solve_modes_cached/solve_tactic_strict_batch`: `457.79-469.56 µs`
    with Criterion reporting about `15-18%` improvement
  - `solve_modes_cached/solve_prepass_batch`: `457.24-469.97 µs`
    with Criterion reporting about `15-18%` improvement
  - `repl_full_eval/cached/batch_11_inputs` stayed statistically flat
    (`2.177-2.269 ms`)
  - `profile_cache/simplify_cached_vs_uncached/cached/light/x_plus_1` also
    stayed statistically flat (`44.7-46.6 µs`)
  while `solve_safety_contract_tests` and `canonicalization_tests` stayed green,
  and the simplification metamorphic benchmark still held `numeric-only = 168`.
- For `SolveTactic(Generic/Assume)`, the detailed profile then isolated a
  different single hotspot: `(2*x + 2*y)/(4*x + 4*y)` was reaching `1/2` through
  two full `Simplify Nested Fraction` applications because the multivariate
  Layer-1 GCD path extracted only numeric content on the first pass.
- `fraction_multivar_gcd` now collapses the remaining scalar-multiple case
  immediately after Layer-1 reduction: when the reduced numerator and
  denominator share the same primitive multivariate polynomial, the planner
  folds that primitive into `gcd_expr` and returns the scalar ratio directly.
- In fast local runs this improved the cached solve batches materially:
  - `solve_modes_cached/solve_tactic_generic_batch`: `405.28-409.98 µs`
    with Criterion reporting about `22-24%` improvement
  - `solve_modes_cached/solve_tactic_assume_batch`: `406.54-437.99 µs`
    with Criterion reporting about `21-25%` improvement
  - `repl_full_eval/cached/batch_11_inputs`: `2.030-2.085 ms`
    with Criterion reporting about `6-9%` improvement
  - `profile_cache/simplify_cached_vs_uncached/cached/light/x_plus_1`:
    `44.14-44.45 µs` with Criterion reporting about `1.4-3.8%` improvement
  while the simplification metamorphic benchmark still held `numeric-only = 168`.
- The detailed solve diagnostics now show only one `Simplify Nested Fraction`
  hit for that generic/assume input instead of two.

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
- prefiltering cached `RuleProfile` buckets by solve-safety mode
  (`SolvePrepass` / `SolveTactic`) regressed the new `solve_modes_cached`
  benchmarks, especially `solve_tactic_assume_batch`, so it was reverted
- caching `SolveSafety` sidecars alongside phase-filtered rule buckets stayed
  flat on `solve_tactic_strict_batch` and regressed `solve_prepass_batch`
  heavily, so it was reverted too
- making the `Simplify Nested Fraction` factor-by-GCD description lazy under
  `steps_off` also regressed `solve_modes_cached/*`, so it was reverted
- replacing repeated `safe_for_*` calls with a precomputed transformer-local
  solve-safety enum also came back flat-to-worse on `solve_modes_cached`, so it
  was reverted
- moving canonicalization rules into target-kind buckets also regressed the real
  cached eval path; the extra specificity looked attractive, but it changed
  rule ordering versus the global bucket and lost more in rewrite churn than it
  saved in dispatch cost, so it was reverted

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
- use `profile_cache`'s `solve_modes_cached/*` benches when evaluating
  `SolvePrepass` / `SolveTactic` ideas; eval-only benches were not a reliable
  proxy for that path
- use `profile_cache`'s `solve_hotspots_cached/*` benches when one input is
  dominating the solve batch and the aggregate `solve_modes_cached/*` signal is
  too noisy to trust
- set `CAS_SOLVE_BENCH_PROFILE=1` to print top applied rules per phase before
  running `solve_modes_cached/*`; optionally narrow it with
  `CAS_SOLVE_BENCH_PROFILE_MODE=prepass|strict|generic|assume`
- set `CAS_SOLVE_BENCH_PROFILE_DETAIL=1` together with
  `CAS_SOLVE_BENCH_PROFILE=1` to print the per-input output and top applied
  rules, which is more useful than aggregate counts when one expression is
  dominating the solve batch
- the current strict/prepass diagnostic is now clean for the blocked
  difference-of-squares fraction case: the output stays as
  `"(x^2 - y^2) / (x - y)"` with no visible rule hits
- the current generic/assume diagnostic for `(2*x + 2*y)/(4*x + 4*y)` now
  reaches `1/2` with a single visible `Simplify Nested Fraction` hit
- current first diagnostic signal: `Simplify Nested Fraction` dominates the
  visible rewrite mix in `solve_modes_cached/*`, with `Exponential-Log Inverse`
  showing up in `generic` / `assume`
- the dedicated `solve_hotspots_cached/*` bench now makes that split explicit:
  `generic/x_over_x` and `generic/exp_ln_x` sit around `20-22 us`, while
  `generic/scalar_multiple_fraction` is still around `151-153 us`; the next
  meaningful solve ROI is therefore still in nested-fraction/polynomial work,
  not in the smaller cancellation/log-inverse cases
- `fraction_gcd_plan_support` now has an earlier structural fast path for the
  scalar-multiple-additive case: when numerator and denominator are sums with
  the same term bodies and only differ by a constant coefficient ratio, the
  planner reduces directly to that scalar ratio without paying the full
  `MultiPoly` conversion/GCD path
- that fast path preserved the visible behavior of
  `(2*x + 2*y)/(4*x + 4*y)` (`2` solve steps, same required condition
  `4*x + 4*y != 0`) while moving the dedicated hotspot to about
  `130-132 us` and the full `solve_tactic_generic_batch` to about
  `375-382 us` in fast mode
- retained two exact micro-fast-paths for that generic/assume batch:
  non-strict `Exponential-Log Inverse` / generic `Log Power Base` now avoid
  full positivity proof when policy only needs a cheap discharge, and
  fraction-cancel rules skip the non-zero prover for bare variables because
  `NonZero(x)` is guaranteed to stay `Unknown` outside `Strict`
- measured outcome is modest but acceptable: `solve_tactic_generic_batch`
  now lands around `391-398 us` in fast mode versus the prior `~405-410 us`
  range, while `solve_tactic_assume_batch` stayed essentially flat/noisy at
  `~404-426 us`

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
