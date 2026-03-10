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
- adding cheap syntactic preguards to the specialized `Div` rules
  (trig/root/hyperbolic/abs) also regressed the dedicated solve hotspots and
  the generic solve batch, so it was reverted too
- a targeted surface precheck for `Half-Angle Tangent Identity` also regressed
  the generic solve batch and was reverted; the per-rule probe made it clear
  that `Weierstrass Half-Angle Contraction` was the better next candidate
- three micro-opts inside the structural scalar-multiple fraction fast path
  were benchmarked and reverted because they regressed or stayed flat on the
  dedicated solve hotspot and the generic solve batch:
  `SmallVec` for the denominator term scratch buffer, a special-case matcher
  for the 2-term additive case, and skipping the unused intermediate
  `forms.result` construction in `build_fraction_cancel_forms(...)`
- a broader no-trace reorder in `SimplifyFractionRule` that tried the full
  GCD planner before the didactic matcher stack was also rejected: it helped
  `difference_of_squares_fraction` / `power_quotient_fraction`, but it drove
  `binomial_square_fraction` to an `~85-89%` regression in both
  `solve_hotspots_cached/*` and `solve_eval_hotspots_cached/*`
- making didactic fraction descriptions borrowed/lazy in
  `didactic_factor_support` was also rejected after sequential re-runs:
  despite looking like a free allocation win, it regressed
  `difference_of_squares_fraction` / `power_quotient_fraction`
- conditionally dropping `.local(...)` in the simple fraction-cancel rules
  (`P/P`, `P^n/P`, same-base powers, nested fraction) was likewise rejected:
  the extra `trace_payloads_enabled()` lookup cost more than the skipped local
  payload on the dedicated `x/x` hotspot
- skipping the non-zero prover for `x^0` outside `Strict` was also rejected:
  once re-measured with the new serial Criterion flow, `generic/x_pow_0` still
  drifted noisy-to-worse, so the old implementation was restored
- restricting `Convert Mixed Trig Fraction to sin/cos` to `Core` only was also
  rejected: the positive case stayed correct, but `a_pow_x_over_a` only moved
  within noise, so the extra phase churn was not worth keeping

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
- for tiny hotspot deltas, do not run Criterion compares in parallel: one bad
  parallel run was enough to fabricate regressions on `x^0`; rerunning the
  same compares in series brought the signal back to neutral / slight-improve
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
- `solve_hotspots_cached/*` now also covers the remaining generic/assume inputs
  from `solve_tactic_generic_batch`, including `(a^x)/a`, `x^0`, and the
  scalar-multiple fraction under `Assume`
- `solve_hotspots_cached/*` now also supports the same diagnostic flags as the
  batch benches via `CAS_SOLVE_BENCH_PROFILE_MODE=hotspots-generic|hotspots-assume`
  and optional `CAS_SOLVE_BENCH_PROFILE_DETAIL=1`
- detailed hotspot mode now prints `"(no rule hits)"` for inputs that stay as-is;
  the current relevant finding is that `(a^x)/a` is a no-op hotspot in
  `generic`, so its `~33 us` cost is dispatch/matching overhead rather than a
  successful rewrite
- when `CAS_SOLVE_BENCH_PROFILE_DETAIL=1` is active, the no-hit hotspot dump
  now prints the full candidate bucket per phase together with its count,
  instead of truncating to the first 8 rule names; that makes the next
  `Div`-bucket investigation for `(a^x)/a` actionable without adding more
  runtime instrumentation
- set `CAS_SOLVE_BENCH_PROFILE_PROBE=1` to benchmark the no-hit candidate rules
  individually before the timed run; optional `CAS_SOLVE_BENCH_PROFILE_PROBE_ITERS`
  controls the per-rule loop count (default `2000`)
- current full candidate dump for `(a^x)/a` in `hotspots-generic` is:
  - `Core/Div` count `10`: `Angle Sum Fraction to Tan`,
    `Half-Angle Tangent Identity`, `Division by Infinity`,
    `Infinity Divided by Finite`, `Merge Sqrt Quotient`,
    `Canonicalize Reciprocal Sqrt`, `Weierstrass Half-Angle Contraction`,
    `sinh(x)/cosh(x) = tanh(x)`, `Convert Mixed Trig Fraction to sin/cos`,
    `Abs Quotient`
  - `PostCleanup/Div` count `10`: same family, but with `Trig Quotient`
    replacing `Abs Quotient`
- retained runtime win: `try_rewrite_weierstrass_contraction_div_expr(...)`
  now has a constant-time surface gate, requiring the denominator to be an
  `Add` and the numerator to be `Mul/Add/Sub` before running the full
  Weierstrass matchers
- validated fast-mode signal for that change, measured in serial:
  - `solve_hotspots_cached/generic/a_pow_x_over_a`: `34.217-34.497 us`
  - `solve_modes_cached/solve_tactic_generic_batch`: `300.04-302.08 us`, with
    Criterion reporting about `4.4-6.2%` improvement
- the new probe confirms the effect on the no-op `Div` bucket: after the
  Weierstrass surface gate, `Weierstrass Half-Angle Contraction` drops to the
  bottom of the candidate list at roughly `0.003 us/apply`, and the next
  dominant no-hit candidates become `Convert Mixed Trig Fraction to sin/cos`
  (`~0.011 us/apply`) followed by `Half-Angle Tangent Identity` /
  `Angle Sum Fraction to Tan` (`~0.008-0.009 us/apply`)
- retained hotspot win: `is_mixed_trig_fraction(...)` in
  `trig_canonicalization_support` now uses an allocation-free trig bitmask scan
  instead of `HashSet<String>` collection, plus a cheap early return when both
  sides of the fraction are structurally incapable of containing trig calls
- validated fast-mode signal for that change:
  - `solve_hotspots_cached/generic/x_over_x`: `18.472-18.658 us`, with
    Criterion reporting about `9-20%` improvement
  - `solve_modes_cached/solve_tactic_generic_batch`: `322.13-325.27 us`, with
    Criterion reporting about `1.6-3.7%` improvement
  - `solve_hotspots_cached/generic/a_pow_x_over_a`: `36.061-36.513 us`, no
    statistically significant change
- retained follow-up win: the same `is_mixed_trig_fraction(...)` gate no longer
  treats every `Add/Sub/Mul/Div` subtree as an automatic trig candidate before
  the bitmask scan; it now does an exact `Function`-presence walk first, so
  algebraic fractions like `(2*x + 2*y)/(4*x + 4*y)` and `(a^x)/a` skip the
  expensive trig scan entirely
- validated fast-mode signal for that refinement against named baseline
  `mixed_trig_gate_pre`:
  - `solve_hotspots_cached/generic/scalar_multiple_fraction`: `93.192-94.731 us`,
    with Criterion reporting about `3.8-6.8%` improvement
  - `solve_hotspots_cached/generic/a_pow_x_over_a`: `33.292-33.640 us`, with
    Criterion reporting about `5.3-7.5%` improvement
  - `solve_modes_cached/solve_tactic_generic_batch`: `300.34-304.39 us`, with
    Criterion reporting about `5.3-7.1%` improvement
- semantic spot-check after that refinement:
  `(sin(x)+cos(x))/tan(x)` still rewrites in `solve generic` to
  `(cos(x)^2 + sin(x)·cos(x)) / sin(x)`, so the positive mixed-trig path stays
  intact while function-free fractions now bypass the rule cheaply
- `profile_cache` now also includes `solve_eval_hotspots_cached/*` for the
  direct `eval --context solve` path, which is distinct from `SolveTactic`
  and is the path affected by the retained strict-domain scalar-multiple
  simplification change
- `solve_eval_hotspots_cached/*` also supports the existing solve diagnostic
  flags, using `CAS_SOLVE_BENCH_PROFILE_MODE=eval-strict|eval-generic|eval-assume`
  and optional `CAS_SOLVE_BENCH_PROFILE_DETAIL=1` to print the direct eval-path
  output and top applied rules for that single hotspot input
- important measurement fix: `solve_modes_cached/*`, `solve_hotspots_cached/*`,
  and `solve_eval_hotspots_cached/*` now explicitly set the simplifier
  `steps_mode` from the source `EvalOptions` before calling
  `simplify_with_options(...)`
- before that fix, those benches were accidentally running with `steps` on
  because `Simplifier::from_profile_with_context(...)` defaults to
  `StepsMode::On` and the engine overwrites `SimplifyOptions.collect_steps`
  from the simplifier state at runtime; older `solve` numbers from this track
  are therefore not directly comparable to the corrected baselines below
- with `CAS_SOLVE_BENCH_PROFILE_DETAIL=1`, `solve_eval_hotspots_cached/*`
  now also prints the ordered `EngineEvent::RuleApplied` trace for that input;
  this is diagnostic only because detail mode installs a temporary listener and
  therefore exercises the traced path rather than the plain `steps off` hot path
- current traced order for the scalar-multiple hotspot in `eval-strict` and
  `eval-generic` is:
  - main `Simplify Nested Fraction` to
    `1 * (2 * x + 2 * y) / (2 * (2 * x + 2 * y))`
  - chained `Simplify Nested Fraction` to `1 / 2`
  - `Combine Constants` to `1/2`
- that trace confirms the residual fixed post-cancel work is real and happens
  after the chained fraction rewrite, rather than inside the GCD planner itself
- on the corrected plain `steps off` path, the profile for the scalar-multiple
  eval hotspot now collapses to a single visible `Simplify Nested Fraction`
  hit; the trailing `Combine Constants` was only present in the traced/listener
  path or in the previously misconfigured benches
- retained runtime change behind that result: `build_fraction_cancel_forms()`
  now materializes pure numeric reduced fractions directly as `Number` when
  `include_factored_form = false`, i.e. on the plain runtime path with no
  didactic trace payloads
- retained plain-runtime dispatch change: on the no-trace path,
  `SimplifyFractionRule` now probes only the exact structural scalar-multiple
  planner before falling back to the didactic matcher stack, so the known
  hotspot avoids paying the full didactic cancellation pipeline when the exact
  ratio escape hatch applies
- current fast-mode snapshot after the retained solve wins:
  - `solve_eval_hotspots_cached/strict/scalar_multiple_fraction`:
    about `93.8-97.3 us`
  - `solve_eval_hotspots_cached/generic/scalar_multiple_fraction`:
    about `90.9-94.7 us`
  - `solve_hotspots_cached/generic/scalar_multiple_fraction`:
    about `97.5-101.7 us`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    about `317.9-326.1 us`
  - `solve_modes_cached/solve_prepass_batch`:
    about `424.8-442.8 us`
- the only follow-up matcher experiment that remains explicitly rejected is the
  aligned-term shortcut inside the structural scalar-multiple matcher; that
  version still made `solve_eval_hotspots_cached/*` worse instead of better
- local Criterion hygiene is now codified in `Makefile`:
  `make bench-clean`, `make bench-engine-fast`, `make bench-engine-fast-save`,
  and `make bench-engine-fast-compare` keep named baselines separate from the
  mutable default `base` results under `target/criterion`
- `Makefile` now also exposes serial benchmark helpers:
  `make bench-engine-fast-save-seq`, `make bench-engine-fast-compare-seq`,
  `make bench-engine-solve-hotspots-save`, and
  `make bench-engine-solve-hotspots-compare`
- use those serial targets for tiny hotspot deltas; they prevent the
  measurement contamination that showed up when `x^0` compares were run in
  parallel
- `Makefile` also exposes a focused solve-diagnostics wrapper:
  `make bench-engine-solve-profile MODE=hotspots-generic FILTER=... [DETAIL=1] [PROBE=1] [PROBE_ITERS=...]`
- this wrapper codifies the current `profile_cache` investigation flow for
  `solve` hotspots and no-hit buckets without having to remember the
  `CAS_SOLVE_BENCH_PROFILE*` environment variables by hand
- validation guardrail restored: `cargo test -p cas_math --lib` is green again
  (`1092 passed`, `1 ignored`) after updating stale unit tests that were still
  asserting removed/internal representation details in
  `trig_core_identity_support`, `abs_support`, and `trig_phase_shift_support`
- that leaves the next ROI unchanged: the fraction/nested-fraction path is
  still roughly `3.5-6x` heavier than the other remaining generic/assume
  hotspot cases, so further work should stay there rather than moving to the
  exponent fast paths
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
- `SimplifyFractionRule` now separates `steps_enabled` from
  `trace_payloads_enabled`: when both steps and listeners are absent, the GCD
  path skips building the didactic factored intermediate entirely, while
  `events_tests` now covers that `steps off` listeners still receive chained
  fraction events
- `SimplifyFractionRule` now also treats the exact
  `FractionGcdRoute::StructuralScalarMultiple` case as intrinsically safe once
  the original `den != 0` condition is retained, because that route proves
  `den = c * gcd` with `c != 0`
- that change is primarily semantic, not a large measured win: it now lets
  `eval --context solve --domain strict` simplify
  `(2*x + 2*y)/(4*x + 4*y)` to `1/2` with the same required condition
  `4*x + 4*y != 0`, while the dedicated tactic hotspots stayed roughly flat /
  noisy at about `129-133 us` (`generic`) and `128-133 us` (`assume`)
- the measurable solve win came from moving the scalar-multiple structural
  fast path ahead of `collect_variables` in `fraction_gcd_plan_support`, so
  the dedicated hotspot no longer pays a full variable scan before taking its
  exact-match escape hatch
- that preserved the visible solve behavior of `(2*x + 2*y)/(4*x + 4*y)`
  (`2` steps, same required condition `4*x + 4*y != 0`) while improving:
  - `solve_hotspots_cached/generic/scalar_multiple_fraction` to about
    `129.1-130.3 us`, with Criterion reporting about `1.1-3.2%` improvement
  - `solve_modes_cached/solve_tactic_generic_batch` to about
    `369.0-371.3 us`, with Criterion reporting about `2.1-4.4%` improvement
- `profile_cache` now also tracks three more direct fraction-cancel hotspots
  in both `solve_hotspots_cached/*` and `solve_eval_hotspots_cached/*`:
  `(x^2 - y^2)/(x - y)`, `x^4/x^2`, and
  `(x^2 + 2*x*y + y^2)/(x + y)^2`
- retained follow-up win: when `SimplifyFractionRule` takes a didactic cancel
  plan but `trace_payloads_enabled = false`, it now drops the unused
  `.local(...)` metadata and keeps only the global rewrite/result; the
  step/listener path is unchanged because it still preserves the local payload
- rationale: the local didactic payload is only consumed by step rendering /
  listeners, so on the plain `steps off` path it was pure hot-path overhead
- validated against named baseline `didactic_bypass_pre`:
  - `solve_hotspots_cached/generic/difference_of_squares_fraction`:
    `68.267-70.102 us`, about `4.3-7.1%` better
  - `solve_hotspots_cached/generic/power_quotient_fraction`:
    `45.467-46.057 us`, about `6.3-8.1%` better
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `144.18-147.18 us`, about `6.8-9.2%` better
  - `solve_eval_hotspots_cached/generic/difference_of_squares_fraction`:
    `66.638-68.640 us`, about `3.9-7.0%` better
  - `solve_eval_hotspots_cached/generic/power_quotient_fraction`:
    `42.630-43.109 us`, about `6.7-9.4%` better
  - `solve_eval_hotspots_cached/generic/binomial_square_fraction`:
    `139.89-142.62 us`, about `7.8-10.6%` better
- retained follow-up win in `didactic_factor_support`: several planners now
  short-circuit `poly_eq(...)` when the rebuilt canonical form is already the
  exact same `ExprId`
- applied to:
  `(a^2 + 2ab + b^2)/(a+b)^2`, `(a^2 - 2ab + b^2)/(a-b)`, sum/difference of
  cubes denominator matching, and `P^m / P^n`
- rationale: on the current solve hotspots, `Canonicalize Multiplication`
  already normalizes the numerator into the exact tree that the didactic
  planner reconstructs, so the algebraic comparison was redundant hot-path work
- validated against named baseline `didactic_poly_eq_pre`:
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `135.50-137.90 us`, about `11.5-14.4%` better
  - `solve_eval_hotspots_cached/generic/binomial_square_fraction`:
    `128.25-129.39 us`, about `12.1-15.7%` better
  - `solve_hotspots_cached/generic/power_quotient_fraction`:
    `47.657-49.986 us`, no statistically significant change
  - `solve_eval_hotspots_cached/generic/power_quotient_fraction`:
    `44.442-44.955 us`, about `1.3-3.4%` better
- one unbaselined rerun of `solve_tactic_generic_batch` landed around
  `322.17-325.40 us`, but that number was not used as an acceptance criterion;
  keep relying on named-baseline compares for these small deltas
- retained follow-up win in `fraction_power_cancel_support`: the
  `P^m / P^n` and `P^n / (-P)` planners now fast-path exact AST matches before
  falling back to `compare_expr(...)` / `poly_relation(...)`
- scope kept intentionally narrow: the analogous fast path for plain `P/P` was
  tried and then removed because it regressed the dedicated `x/x` hotspot
- validated against named baseline `fraction_power_exact_pre`:
  - `solve_hotspots_cached/generic/x_over_x`:
    `18.239-18.526 us`, change within noise threshold
  - `solve_hotspots_cached/generic/power_quotient_fraction`:
    `48.048-48.377 us`, about `2.5-6.5%` better
  - `solve_eval_hotspots_cached/generic/power_quotient_fraction`:
    `45.226-45.968 us`, small improvement but still within Criterion's noise threshold
- retained follow-up win in `difference_of_squares_support`: the
  pre-order `(A^2 - B^2)/(A ± B)` planner now avoids duplicate
  `multipoly_from_expr(...)` work by reusing the already-built `MultiPoly`
  forms for `A-B` and `A+B`, and it short-circuits exact denominator matches
  before the polynomial comparison path
- this keeps the visible solve output unchanged for the canonical case
  `(x^2 - y^2)/(x - y)` (`2` steps, same required condition `x - y != 0`)
- validated against named baseline `diffsq_plan_pre`:
  - `solve_hotspots_cached/generic/difference_of_squares_fraction`:
    `65.760-66.510 us`, about `6.5-8.4%` better
  - `solve_eval_hotspots_cached/generic/difference_of_squares_fraction`:
    `62.940-64.650 us`, about `9.1-11.6%` better
- retained follow-up win in `SimplifyFractionRule` for the exact
  `(a^2 + 2ab + b^2)/(a+b)^2` didactic path when `steps off` and no listener is
  attached: if the planner already produced an exact `P/P` intermediate, the
  runtime now returns `1` directly with the inherited `requires` instead of
  paying a second rule pass through `Cancel Identical Numerator/Denominator`
- scope is deliberately narrow:
  `steps on` keeps the old two-phase didactic behavior, and unrelated hotspots
  (`x/x`, scalar-multiple fractions) were checked against the same baseline
- validated against named baseline `binomial_cancel_pre`:
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `104.70-105.39 us`, about `18.2-20.2%` better
  - `solve_eval_hotspots_cached/generic/binomial_square_fraction`:
    `101.35-103.90 us`, about `15.9-19.0%` better
  - `solve_hotspots_cached/generic/x_over_x`:
    no statistically significant change
  - `solve_eval_hotspots_cached/generic/scalar_multiple_fraction`:
    change stayed within Criterion's noise threshold
- retained follow-up win in `Exponential-Log Inverse`: the engine-side rule now
  resolves the domain policy inline instead of constructing
  `LogExpInversePolicyMode` and calling the shared planner on every hit; `Strict`
  still blocks unless positivity is proven, and non-strict modes still attach
  the same `Positive(...)` require when the subject is not already provably
  positive
- validated with sequential Criterion reruns against the current local baseline:
  - `solve_hotspots_cached/generic/exp_ln_x`:
    `17.958-18.304 us`, about `5.6-7.2%` better
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `304.91-307.83 us`, about `4.5-6.0%` better
- tried an exact-term-order fast path in the structural scalar-multiple planner
  and removed it again: `solve_hotspots_cached/generic/scalar_multiple_fraction`
  only moved to `105.44-106.73 us` and
  `solve_eval_hotspots_cached/generic/scalar_multiple_fraction` stayed within
  noise at `102.53-103.61 us`, so the extra matching complexity was not kept
- `solve_hotspots_cached/*` now also covers `generic/log_power_base` and
  `assume/log_power_base` with `log(x^2, x^6)` so the `Log Power Base` path has
  a dedicated micro-bench instead of hiding inside the batch
- current baseline after adding that hotspot:
  - `solve_hotspots_cached/generic/log_power_base`:
    `33.909-35.021 us`
  - `solve_hotspots_cached/assume/log_power_base`:
    `34.005-35.135 us`
- used the existing `CAS_SOLVE_BENCH_PROFILE_PROBE=1` path on
  `solve_hotspots_cached/generic/a_pow_x_over_a`; it showed the `Div` no-op
  bucket is not dominated by one obviously expensive rule
  (`Convert Mixed Trig Fraction to sin/cos` was the largest single reject at
  only about `0.012 us`, with the rest between `0.003-0.008 us`)
- tried inlining `Log Power Base` policy in the engine rule, but removed it
  again because the dedicated hotspot stayed flat in `generic` and regressed in
  `assume`; keep the bench, not the runtime churn
- retained follow-up win in orchestration: `Orchestrator` now caches
  `PatternMarks` by current `ExprId` and reuses them across phases until the
  expression actually changes, instead of rescanning the same no-op tree once
  per phase
- validated signal:
  - `solve_hotspots_cached/generic/a_pow_x_over_a`:
    `31.676-31.917 us`, about `2.1-3.4%` better
  - `solve_modes_cached/solve_prepass_batch`:
    `408.74-412.72 us`, about `3.9-7.0%` better
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `332.12-336.17 us`, improvement direction but still within Criterion's
    noise threshold
  - `solve_modes_cached/solve_tactic_assume_batch`:
    no statistically significant change
- safety check after the cache change:
  `metatest_unified_benchmark` stayed at `numeric-only = 168`
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

## Latest Retained Win

- fixed a real solver-path contract bug in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/orchestration.rs`:
  `simplify_for_solve()` now forces `StepsMode::Off` for the hidden solve
  prepass and restores the caller's exact mode afterward
- reason: the pipeline currently derives `collect_steps` from the simplifier's
  `steps_mode`, so the old implementation could still pay step/tracing overhead
  during an "invisible" prepass whenever the user had `steps on` or `compact`
- added dedicated bench coverage in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`:
  `solve_prepass_inherited_steps_cached/steps_on_batch`
- measured against named baseline `prepass_steps_inherited_pre`:
  - `solve_prepass_inherited_steps_cached/steps_on_batch`:
    `365.37-379.11 us`, improvement about `2.3-4.9%`
- guardrails:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/solve_safety_contract_tests.rs`
    now checks that `simplify_for_solve()` restores `StepsMode::Compact`
  - `metatest_unified_benchmark` stayed at `numeric-only = 168`
- added a new solver-internal verification micro-bench in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`:
  `solver_verification_inherited_steps/quadratic_two_roots_steps_on`
- current saved baseline for that path is roughly `113.23-114.38 us`
- also added a contract guard in
  `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/solver_core_contract_tests.rs`
  to ensure `verify_solution_set(...)` preserves the caller's exact
  `StepsMode`
- tried two runtime optimizations on top of that bench and removed both:
  - forcing `StepsMode::Off` inside the runtime adapter for
    `runtime_simplify_with_options_expr(...)`
  - removing the `RefCell` indirection in `verification_flow`
- both variants measured worse on the dedicated verification bench, so only the
  benchmark coverage and contract guard were kept
- also re-tried two fraction-path micro-opts and removed both:
  - exact-`ExprId` fast path inside the structural scalar-multiple matcher in
    `fraction_gcd_plan_support`
  - skipping `desc/local` metadata in the plain `steps off` return path of
    `SimplifyFractionRule`
- both were worse on named-baseline compares for
  `generic/scalar_multiple_fraction` and `solve_tactic_generic_batch`, so they
  were intentionally reverted
- retained a new `steps off` fast path for didactic fraction plans in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra/fractions/gcd_cancel.rs`
  plus plan metadata in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra/fractions/didactic_factor_support.rs`
- idea: when the didactic planner already knows the fully cancelled result for
  perfect-square-minus and sum/difference-of-cubes fractions, return that final
  expression directly in plain mode instead of emitting an intermediate
  factored fraction and paying a second cancellation pass
- added dedicated hotspots in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`
  for:
  - `generic/perfect_square_minus_fraction`
  - `generic/difference_of_cubes_fraction`
  - `generic/sum_of_cubes_fraction`
  - and their `solve_eval_hotspots_cached/*` counterparts
- measured against named baseline `didactic_plain_final_pre`:
  - `solve_hotspots_cached/generic/perfect_square_minus_fraction`:
    `179.46-181.09 us`, improvement about `2.1-4.1%`
  - `solve_eval_hotspots_cached/generic/perfect_square_minus_fraction`:
    `175.33-176.64 us`, improvement about `1.2-3.9%`
  - `solve_hotspots_cached/generic/difference_of_cubes_fraction`:
    `258.24-261.16 us`, improvement about `50.5-51.3%`
  - `solve_eval_hotspots_cached/generic/difference_of_cubes_fraction`:
    `250.56-253.68 us`, improvement about `50.6-51.9%`
  - `solve_hotspots_cached/generic/sum_of_cubes_fraction`:
    `204.39-206.31 us`, improvement about `40.4-43.7%`
  - `solve_eval_hotspots_cached/generic/sum_of_cubes_fraction`:
    `193.99-198.35 us`, improvement about `44.2-46.1%`
- guardrails added in
  `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/multivar_gcd_tests.rs`
  for `steps off` solve-context outputs of the perfect-square-minus and
  sum/difference-of-cubes fractions
- `steps on` remains didactic: spot checks still show the intermediate
  factorization/cancellation sequence in CLI JSON output
- `metatest_unified_benchmark` stayed at `numeric-only = 168`
- retained a follow-up optimization in the same didactic fraction planner family:
  `try_plan_fraction_didactic_cancel(...)` now receives
  `didactic_payloads_enabled`, and the sub-planners stop constructing unused
  intermediate factored forms / `Div(...)` nodes when the plain `steps off`
  path will return a final result directly anyway
- for binomial-square cancellation, the plain path now returns `1` directly
  while still preserving both existing `requires`:
  `(x + y)^2 != 0` and `x^2 + y^2 + 2*x*y != 0`
- measured against named baseline `binomial_early_return_pre`:
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `98.475-100.05 us`, improvement about `4.9-12.5%`
  - `solve_eval_hotspots_cached/generic/binomial_square_fraction`:
    `96.544-97.953 us`, point improvement but below significance threshold
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `294.13-299.55 us`, improvement about `3.8-5.7%`
- guardrail added in
  `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/multivar_gcd_tests.rs`
  to ensure the binomial-square `steps off` path still exposes both required
  conditions through `required_conditions`
- also re-measured the retained `didactic_payloads_enabled` change against the
  older baseline `didactic_plain_final_pre`; the follow-up pruning of unused
  didactic payloads improved the same family further:
  - `solve_hotspots_cached/generic/perfect_square_minus_fraction`:
    `169.05-172.07 us`, improvement about `7.9-9.7%`
  - `solve_eval_hotspots_cached/generic/perfect_square_minus_fraction`:
    `162.53-167.25 us`, improvement about `7.2-10.6%`
  - `solve_hotspots_cached/generic/difference_of_cubes_fraction`:
    `242.61-246.71 us`, improvement about `53.6-54.3%`
  - `solve_eval_hotspots_cached/generic/difference_of_cubes_fraction`:
    `231.97-234.11 us`, improvement about `54.4-55.6%`
  - `solve_hotspots_cached/generic/sum_of_cubes_fraction`:
    `194.54-195.95 us`, improvement about `45.1-46.7%`
  - `solve_eval_hotspots_cached/generic/sum_of_cubes_fraction`:
    `185.94-187.54 us`, improvement about `47.4-48.8%`
- rejected two follow-up hypotheses after dedicated named-baseline compares:
  - exact no-payload trinomial matcher for binomial / perfect-square planners:
    regressed `solve_eval_hotspots_cached/generic/binomial_square_fraction`
    and `solve_hotspots_cached/generic/perfect_square_minus_fraction`
  - lazy cycle `HashSet` state (`None/One/Many`) in
    `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`:
    no stable win on `a_pow_x_over_a`, `log_power_base`, or the solve batches
- rejected another follow-up hypothesis after an explicit A/B baseline:
  replacing static didactic fraction descriptions with borrowed
  `Cow<'static, str>` regressed
  `perfect_square_minus_fraction` and `difference_of_cubes_fraction`, so the
  planner keeps the original `String` fields
- retained a larger follow-up win on the same family instead:
  `transform_div(...)` now has a pre-order fast path for
  `(a^2 - 2ab + b^2) / (a - b)` when no engine listener is attached, mirroring
  the existing difference-of-squares shortcut and bypassing the later
  `Canonicalize* + Simplify Nested Fraction + Cancel Power Fraction` pipeline
- measured against named baseline `didactic_desc_pre`:
  - `solve_hotspots_cached/generic/perfect_square_minus_fraction`:
    `91.068-95.137 us`, improvement about `41.8-45.8%`
  - `solve_eval_hotspots_cached/generic/perfect_square_minus_fraction`:
    `85.176-85.884 us`, improvement about `47.0-48.0%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `297.89-300.47 us`, no statistically significant change
- added solver guardrails for the same expression with `steps on` in both
  `generic` and `strict` domains so the new pre-order path is covered beyond
  the existing `steps off` tests
- explored the same pre-order idea for
  `(a^3 - b^3)/(a-b)` and `(a^3 + b^3)/(a+b)` in the plain `steps off`
  runtime, but rejected it after named-baseline compares:
  hotspot benches improved strongly, yet
  `solve_modes_cached/solve_tactic_generic_batch` regressed about `1.9-4.7%`
  even after adding a shallow cube-shape prefilter on `Div`, so the runtime
  change was reverted
- retained only the stricter coverage added during that probe:
  `multivar_gcd_tests` now also locks `difference_of_cubes` and `sum_of_cubes`
  in `solve` + `strict` + `steps off`
- rejected a root-shape gate in
  `/Users/javiergimenezmoya/developer/math/crates/cas_math/src/fraction_gcd_plan_support.rs`
  that tried to skip the structural `scalar_multiple` planner for non-additive
  fractions before building `AddView`
- measured against named baseline `scalar_root_gate_pre`, it regressed all the
  relevant probes:
  - `solve_eval_hotspots_cached/generic/scalar_multiple_fraction` by about
    `2.2-5.1%`
  - `solve_hotspots_cached/generic/x_over_x` by about `1.7-3.9%`
  - `solve_hotspots_cached/generic/a_pow_x_over_a` by about `3.8-5.3%`
  - `solve_modes_cached/solve_tactic_generic_batch` by about `2.2-3.7%`
- rejected another follow-up in the same planner:
  a dedicated two-term fast path for the structural `scalar_multiple` case
  looked plausible for `(2*x + 2*y)/(4*x + 4*y)` but regressed the broader
  probes against `scalar_root_gate_pre`
- measured regressions were roughly:
  - `solve_eval_hotspots_cached/generic/scalar_multiple_fraction`:
    `1.6-4.2%`
  - `solve_hotspots_cached/generic/x_over_x`: `2.7-4.8%`
  - `solve_hotspots_cached/generic/a_pow_x_over_a`: `2.8-4.1%`
  - `solve_modes_cached/solve_tactic_generic_batch`: `3.8-5.1%`
- added a direct planner bench in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`
  under `fraction_gcd_planner_direct/*` to separate `try_plan_fraction_gcd_rewrite(...)`
  cost from full engine/runtime cost
- first measurements in fast mode:
  - `plain/scalar_multiple_fraction`: `6.61-6.93 us`
  - `trace/scalar_multiple_fraction`: `6.19-6.41 us`
  - `plain/x_over_x`: `13.46-13.71 us`
  - `plain/a_pow_x_over_a`: `3.47-3.68 us`
  - `plain/difference_of_squares_fraction`: `81.69-82.75 us`
- takeaway: the scalar-multiple planner itself is not the dominant hotspot in
  the `~95-100 us` solve/eval path, so the next optimization pass should focus
  on downstream engine work after planning rather than on more `AddView` /
  planner micro-opts
- added a second direct bench layer in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`
  under `fraction_rule_direct/*` to split:
  - `RationalFnView::from(...)`
  - `SimplifyFractionRule::apply(...)`
  - a minimal single-rule engine run with `steps off`
- first measurements in fast mode:
  - `rational_fn_view/scalar_multiple_fraction`: `2.967-3.116 us`
  - `apply/generic/scalar_multiple_fraction`: `14.475-14.822 us`
  - `apply/generic/x_over_x`: `13.120-13.356 us`
  - `apply/generic/a_pow_x_over_a`: `3.465-3.611 us`
  - `apply/generic/difference_of_squares_fraction`: `3.797-3.913 us`
  - `single_rule_engine/generic/scalar_multiple_fraction`: `24.387-24.628 us`
  - `single_rule_engine/generic/x_over_x`: `17.569-17.802 us`
  - `single_rule_engine/generic/a_pow_x_over_a`: `8.975-9.119 us`
- takeaway: for `scalar_multiple_fraction`, the planner (`~6.6 us`) plus
  `RationalFnView` (`~3.0 us`) plus direct rule body (`~14.7 us`) still sit far
  below the retained full hotspot (`~90-100 us`), so the next ROI is not inside
  `fraction_gcd_plan_support` but in downstream orchestration / extra passes /
  surrounding rule work on the solve path
- upgraded the solve benchmark diagnostics in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`
  so `CAS_SOLVE_BENCH_PROFILE_PROBE=1` can print bucket probes even when the
  input does hit rules (not only on no-op cases)
- that probe showed the remaining `Div` bucket rejects for
  `generic/scalar_multiple_fraction` are all sub-microsecond
  (`~0.003-0.118 us` in the top entries), so the missing cost was not in a
  single expensive rejected rule
- added `solve_phase_subset_cached/*` in the same bench file to isolate the
  cost of late pipeline phases for `(2*x + 2*y)/(4*x + 4*y)`:
  - before the runtime change:
    - `generic/full`: `96.092-97.283 us`
    - `generic/no_transform`: `96.815-97.979 us`
    - `generic/no_transform_no_rationalize`: `94.415-97.020 us`
- takeaway from that subset bench: `Transform`/`Rationalize` were not the
  dominant fixed cost; the remaining waste sat in the generic late pipeline
  after `Core`
- retained a narrow runtime fast path in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`:
  after the `Core` phase, if `steps` are off and the result is already terminal
  (`Number`, atom, or exact numeric `Div`), the pipeline now skips
  `BestSoFar`, `Transform`, `Rationalize`, `PostCleanup`, and final collection
  noise and returns immediately
- measured after the change:
  - `solve_phase_subset_cached/generic/full`:
    `93.967-96.446 us`, improvement about `1.8-4.3%`
  - `solve_phase_subset_cached/generic/no_transform`:
    `89.973-91.921 us`, improvement about `4.3-6.5%`
  - `solve_phase_subset_cached/generic/no_transform_no_rationalize`:
    `89.772-91.464 us`, improvement about `3.7-7.6%`
  - `solve_eval_hotspots_cached/generic/scalar_multiple_fraction`:
    `87.877-89.008 us`, improvement about `4.2-5.8%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `267.27-269.78 us`, improvement about `6.9-7.6%`
  - `solve_hotspots_cached/generic/x_over_x`:
    `11.350-11.561 us`, improvement about `37.7-39.1%`
- guardrails/validation after the runtime change:
  - `multivar_gcd_tests` generic + strict `steps off` for the scalar-multiple
    fraction still pass
  - `domain_contract_tests::test_generic_x_div_x_simplifies_to_1` still passes
  - `profile_cache_tests` still pass
  - `metatest_unified_benchmark` stayed at `numeric-only = 168`
- retained a follow-up orchestration optimization in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`:
  `BestSoFar` is now initialized lazily from the post-Core baseline instead of
  unconditionally; if no later phase changes the expression, the pipeline no
  longer pays the baseline scoring / `consider(...)` overhead
- rationale: after the new terminal-after-Core fast path, the next visible
  no-op/near-no-op solve cases were still spending fixed time in
  `BestSoFar::new(...)` and final rollback bookkeeping even when
  `Transform`/`Rationalize`/`PostCleanup` never changed the expression
- measured after the change:
  - `solve_hotspots_cached/generic/a_pow_x_over_a`:
    `26.864-27.048 us`, improvement about `5.1-6.2%`
  - `solve_hotspots_cached/generic/difference_of_squares_fraction`:
    `57.470-58.300 us`, improvement about `1.0-3.1%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `270.24-271.39 us`, no statistically significant change
- validation after the change:
  - `domain_contract_tests::test_generic_x_div_x_simplifies_to_1`
  - `multivar_gcd_tests::test_content_gcd_multivar_in_solve_generic_context_steps_off`
  - `profile_cache_tests`
  - `metatest_unified_benchmark` stayed at `numeric-only = 168`
- rejected two more orchestration follow-ups after re-measuring and reverting:
  - skipping final `collect_with_semantics(...)` when the root was not `Add/Sub`
    looked semantically safe, but stayed within noise on
    `solve_hotspots_cached/generic/a_pow_x_over_a`
  - short-circuiting the final rollback block when `best_expr == current`
    also stayed within noise on
    `solve_hotspots_cached/generic/difference_of_squares_fraction` and on the
    generic solve batch
- takeaway: the next ROI is no longer in the epilogue (`final collect` /
  rollback bookkeeping) but earlier in the no-op `Div` traversal itself
- retained a targeted Rationalize-phase gate in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`:
  the phase now runs only when the pattern pre-scan proves there is at least
  one root-like form somewhere inside a denominator subtree
- implementation details:
  - `PatternMarks` now carries a global `has_root_in_denominator` bit in
    `/Users/javiergimenezmoya/developer/math/crates/cas_math/src/pattern_marks.rs`
  - the existing linear pattern scan in
    `/Users/javiergimenezmoya/developer/math/crates/cas_math/src/pattern_scanner.rs`
    sets that bit while traversing denominator branches, so the runtime gate
    does not add a second tree walk
- retained measurements against baseline `rationalize_den_roots_pre`:
  - `solve_phase_subset_cached/a_pow_x_over_a/generic/full`:
    `23.097-23.537 us`, improvement about `10.3-12.2%`
  - `solve_hotspots_cached/generic/a_pow_x_over_a`:
    `23.561-24.418 us`, improvement about `10.1-12.5%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `265.40-267.22 us`, no statistically significant change
- interpretation: skipping `Rationalize` was the right local fix for no-op
  `Div` trees like `(a^x)/a`, but it is not a new batch-moving optimization
- guardrails added:
  - pattern-scan tests now cover positive / negative detection of denominator
    roots
  - `profile_cache_tests::test_from_profile_solve_tactic_skips_rationalize_without_denominator_roots`
    fixes the contract that cached solve generic should keep
    `stats.rationalize.iters_used == 0` for `(a^x)/a`
- retained a second, narrower solve fast path in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`:
  after `Core`, if the plain solve path reaches the exact no-op shape
  `atom^symbol / atom` (same atom on both sides), with no denominator roots and
  no auto-expand contexts, the pipeline now returns immediately instead of
  paying `Transform`, `Rationalize`, `PostCleanup`, final collect, and
  `BestSoFar` setup
- scope is intentionally narrow:
  - solve context only
  - `steps off` only
  - only `Div(Pow(atom, symbol), atom)` where `atom` is a variable/constant and
    the exponent is also a variable/constant
  - this avoids turning the previous finding into a broad phase skip with shaky
    algebraic assumptions
- retained measurements:
  - versus the immediately previous retained state:
    - `solve_hotspots_cached/generic/a_pow_x_over_a` moved from
      `23.561-24.418 us` to `12.046-12.212 us`
    - `solve_phase_subset_cached/a_pow_x_over_a/generic/full` moved from
      `23.097-23.537 us` to `12.401-12.775 us`
    - `solve_modes_cached/solve_tactic_generic_batch` moved from
      `265.40-267.22 us` to `247.25-249.39 us`
  - against the older baseline `rationalize_den_roots_pre`, Criterion reported:
    - `solve_phase_subset_cached/a_pow_x_over_a/generic/full`:
      improvement about `51.7-53.3%`
    - `solve_hotspots_cached/generic/a_pow_x_over_a`:
      improvement about `54.3-55.2%`
    - `solve_modes_cached/solve_tactic_generic_batch`:
      improvement about `5.7-7.2%`
- guardrails added:
  - `profile_cache_tests::test_from_profile_solve_tactic_skips_late_phases_for_symbolic_power_over_same_atom`
    fixes the contract that cached solve generic now keeps
    `transform/rationalize/post_cleanup` at `0` iterations for `(a^x)/a`
- retained a new pre-order fast path for exact additive scalar-multiple
  fractions in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/transform_helpers.rs`
  using the structural planner exposed from
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra/mod.rs`
- rationale:
  - the direct planner and direct rule benches had already shown that
    `(2*x + 2*y)/(4*x + 4*y)` was no longer bottlenecked by the GCD planner
    itself
  - the remaining cost was the full bottom-up `Div` traversal of both children
    before `Simplify Nested Fraction` could fire at the root
  - the new fast path uses the exact structural planner up front and returns the
    fully normalized scalar result immediately on the hidden path
- scope is intentionally narrow:
  - solve context only
  - `steps off` only
  - no listener attached
  - `Core` phase only
  - exact additive scalar-multiple fractions only, via
    `try_plan_structural_scalar_multiple_fraction_rewrite(...)`
- retained measurements against named baseline `scalar_preorder_pre`:
  - `solve_hotspots_cached/generic/scalar_multiple_fraction`:
    `13.224-13.440 us`, improvement about `85.7-86.3%`
  - `solve_eval_hotspots_cached/generic/scalar_multiple_fraction`:
    `12.700-12.940 us`, improvement about `85.7-86.1%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `168.43-175.37 us`, improvement about `30.6-32.5%`
- interpretation:
  - this confirms the remaining cost of the scalar-multiple hotspot was not in
    `try_plan_fraction_gcd_rewrite(...)`, but in reaching the root rule through
    the generic recursive traversal
  - the pre-order path removes that traversal cost without widening the public
    didactic/event surface
- guardrails added:
  - `profile_cache_tests::test_from_profile_solve_tactic_scalar_multiple_fraction_uses_preorder_fast_path`
    fixes the contract that cached solve generic keeps the result `1/2`,
    leaves late phases at `0`, and bypasses the normal rule-loop rewrite count
  - `multivar_gcd_tests::test_content_gcd_multivar_in_solve_generic_context_steps_off_keeps_requires`
    fixes the contract that the plain solve path still reports the denominator
    requirement for the scalar-multiple fraction
- validation after the change:
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests
    test_content_gcd_multivar_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo test -p cas_solver --test multivar_gcd_tests
    test_content_gcd_multivar_in_solve_generic_context_steps_off -- --exact`
  - `cargo test -p cas_solver --test multivar_gcd_tests
    test_content_gcd_multivar_in_solve_strict_context_steps_off -- --exact`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - CLI spot-check: `eval --context solve --domain generic --steps on --format json '(2*x + 2*y)/(4*x + 4*y)'`
    still returns `2` steps and required display `4·x + 4·y ≠ 0`
  - `metatest_unified_benchmark` stayed at `numeric-only = 168`
- retained a follow-up hidden-path optimization in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/transform_helpers.rs`
  for the existing pre-order fraction shortcuts:
  when `difference_of_squares` or `perfect_square_minus` already produce a
  plain symbolic binomial result (`x + y`, `x - y`, etc.) on the solve
  `steps off` path, the transformer now returns that result directly instead of
  recursively re-simplifying it
- rationale:
  - the pre-order planners were already matching these patterns early, but the
    hidden path still paid a second traversal of the already-final result
  - on the real hotspots, that second traversal was pure fixed-cost overhead
    because the result was just a symbolic `Add/Sub`
- scope is intentionally narrow:
  - solve context only
  - `steps off` only
  - no listener attached
  - `Core` phase only
  - result shape restricted to a symbolic `Add/Sub` (or negated one) whose
    leaves are variables/constants
- retained measurements against named baseline `div_plain_pre`:
  - `solve_hotspots_cached/generic/difference_of_squares_fraction`:
    `46.708-48.235 us`, improvement about `14.3-16.5%`
  - `solve_eval_hotspots_cached/generic/difference_of_squares_fraction`:
    `43.617-44.426 us`, improvement about `16.1-18.2%`
  - `solve_hotspots_cached/generic/perfect_square_minus_fraction`:
    `76.991-79.310 us`, improvement about `5.8-8.0%`
  - `solve_eval_hotspots_cached/generic/perfect_square_minus_fraction`:
    `71.788-72.456 us`, improvement about `6.6-8.8%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `157.25-158.90 us`, improvement about `4.9-8.2%`
- interpretation:
  - after the scalar-multiple pre-order shortcut, `difference_of_squares` had
    become the next real batch-moving fraction hotspot
  - this change confirms that a meaningful part of its remaining cost was not
    the planner itself, but the extra post-plan recursion through an already
    final `x ± y`
- validation after the change:
  - `cargo test -p cas_solver --test multivar_gcd_tests test_layer2_difference_of_squares -- --exact`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_perfect_square_minus_cancel_in_solve_generic_context_steps_off -- --exact`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `metatest_unified_benchmark` stayed at `numeric-only = 168`
- retained a follow-up orchestration cut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`:
  after `Core`, the plain solve path now exits early when the result is already
  a symbolic sum of atoms like `x + y`
- rationale:
  - after the previous hidden-path pre-order optimization,
    `difference_of_squares_fraction` still paid `Transform`, `Rationalize`,
    `PostCleanup`, and the `BestSoFar` epilogue even though `Core` had already
    finished at `x + y`
  - that made `difference_of_squares` the next real batch-moving hotspot
- scope is intentionally narrow:
  - solve context only
  - `steps off` only
  - no denominator roots
  - no auto-expand contexts
  - result shape restricted to a plain symbolic `Add` of variables/constants
  - deliberately does **not** include `Sub`, because the equivalent `x - y`
    path (`perfect_square_minus`) did not show a stable improvement
- retained measurements against named baseline `postcore_binomial_pre`:
  - `solve_hotspots_cached/generic/difference_of_squares_fraction`:
    `31.826-32.119 us`, improvement about `27.2-29.7%`
  - `solve_eval_hotspots_cached/generic/difference_of_squares_fraction`:
    `30.923-31.086 us`, improvement about `27.2-29.5%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `146.38-147.60 us`, improvement about `4.1-5.0%`
- follow-up spot-check on the untouched subtraction case:
  - `solve_hotspots_cached/generic/perfect_square_minus_fraction` rerun landed
    in Criterion's noise band (`+0.5%..+2.6%`)
  - `solve_eval_hotspots_cached/generic/perfect_square_minus_fraction` rerun
    also stayed within noise (`+0.9%..+5.7%`)
- guardrails added:
  - `profile_cache_tests::test_from_profile_solve_tactic_plain_binomial_result_skips_late_phases`
    fixes the contract that cached solve generic now keeps
    `transform/rationalize/post_cleanup` at `0` for `(x^2 - y^2)/(x - y)`
  - `multivar_gcd_tests::test_layer2_difference_of_squares_in_solve_generic_context_steps_off_keeps_requires`
    fixes the contract that the plain solve path still reports the original
    denominator requirement
- validation after the change:
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests
    test_layer2_difference_of_squares_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- retained another hidden-path pre-order cut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/transform_helpers.rs`
  for the exact `atom / atom` case used by `x/x`
- rationale:
  - after the batch-moving wins on scalar-multiple fractions and
    `difference_of_squares`, `x/x` was still paying the full `Div` child walk
    before `Cancel Identical Numerator/Denominator` could fire at the root
  - for solve generic/assume, `x/x` is a narrow, stable definability rewrite
    with a cheap structural predicate, so it is a good fit for the hidden
    pre-order path
- scope is intentionally narrow:
  - solve context only
  - `steps off` only
  - no listener attached
  - `Core` phase only
  - non-strict domain modes only
  - only exact `atom / atom` where both sides are the same variable/constant
- retained measurements against named baseline `xoverx_preorder_pre`:
  - `solve_hotspots_cached/generic/x_over_x`:
    `8.4982-8.6397 us`, improvement about `26.6-28.9%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `137.91-139.09 us`, improvement about `2.0-3.0%`
- validation after the change:
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test domain_contract_tests test_strict_x_div_x_stays_unchanged -- --exact`
  - CLI spot-check:
    `eval --context solve --domain generic --steps off --format json 'x/x'`
    still returns `1` with required display `x ≠ 0`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- retained a matching hidden-path pre-order cut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/transform_helpers.rs`
  for the `exp(ln(x)) -> x` shape
- rationale:
  - after the `x/x` shortcut, `exp(ln(x))` was still another small but steady
    solve-generic hotspot paying the full `Pow` child traversal before
    `Exponential-Log Inverse` could fire at the root
  - unlike the more general `b^(c*log(b,x)) -> x^c` family, the plain
    `exp(ln(x)) -> x` case collapses directly to an atom, so it fits the same
    hidden pre-order strategy with limited surface area
- scope is intentionally narrow:
  - solve context only
  - `steps off` only
  - no listener attached
  - `Core` phase only
  - non-strict domain modes only
  - only when the existing matcher rewrites directly to a symbolic atom
- retained measurements against named baseline `expln_preorder_pre`:
  - `solve_hotspots_cached/generic/exp_ln_x`:
    `7.6321-7.7409 us`, improvement about `42.6-44.4%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `133.34-134.93 us`, improvement about `4.2-8.4%`
- guardrails/validation after the change:
  - `profile_cache_tests::test_from_profile_solve_tactic_exp_ln_atom_uses_preorder_fast_path`
    fixes the contract that cached solve generic reaches `x` without late phases
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test domain_contract_tests exp_ln_x_generic_emits_positive_require -- --exact`
  - `cargo test -p cas_solver --test domain_contract_tests test_strict_exp_ln_x_stays_unchanged -- --exact`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- retained the same hidden-path pre-order strategy in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/transform_helpers.rs`
  for the exact `atom^0 -> 1` case used by `x^0`
- rationale:
  - after the `x/x` and `exp(ln(x))` shortcuts, `x^0` was still a steady
    solve-generic/assume hotspot paying the full `Pow` child walk before
    `Identity Power` could fire at the root
  - in solve generic/assume with `steps off`, the observable result is already
    the bare `1` with no required display, so the same hidden pre-order pattern
    applies cleanly
- scope is intentionally narrow:
  - solve context only
  - `steps off` only
  - no listener attached
  - `Core` phase only
  - non-strict domain modes only
  - only exact `atom^0` where the base is a variable/constant
- retained measurements against named baseline `xpow0_preorder_pre`:
  - `solve_hotspots_cached/generic/x_pow_0`:
    `9.0594-9.2470 us`, improvement about `32.4-35.0%`
  - `solve_hotspots_cached/assume/x_pow_0`:
    `8.9446-9.0989 us`, improvement about `33.8-35.5%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `126.05-127.80 us`, improvement about `0.9-3.3%`, still within
    Criterion's noise threshold
- guardrails/validation after the change:
  - `profile_cache_tests::test_from_profile_solve_tactic_pow_zero_atom_uses_preorder_fast_path`
    fixes the contract that cached solve generic reaches `1` without root-rule
    rewrites or late phases
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test domain_contract_tests test_strict_x_pow_0_stays_unchanged -- --exact`
  - `cargo test -p cas_solver --test domain_assume_warnings_contract_tests assume_x_pow_0_simplifies_with_assumption -- --exact`
  - `cargo test -p cas_solver --test assumption_key_contract_tests nonzero_emitted_for_zero_exponent -- --exact`
  - CLI spot-check:
    `eval --context solve --domain generic --steps off --format json 'x^0'`
    still returns `1`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
    keeps `numeric-only = 168`
- retained an exact-shape fast path in
  `/Users/javiergimenezmoya/developer/math/crates/cas_math/src/difference_of_squares_support.rs`
  for the denominator cases that are already syntactically `A-B` or `A+B`
- rationale:
  - the cached solve hotspot `(x^2 - y^2)/(x - y)` was already bypassing the
    rule loop, so the remaining cost was in the planner path itself
  - the planner was still converting `A-B` and `A+B` to `MultiPoly` even when
    the denominator already matched the raw factor exactly
  - for the exact hidden hotspot, that polynomial work is unnecessary; the
    final result and didactic intermediates are already known structurally
- scope is intentionally narrow:
  - only exact raw denominator matches `A-B` or `A+B`
  - all reordered / sign-flipped / algebraically equivalent denominators still
    go through the previous polynomial path unchanged
- retained measurements against named baseline `diffsq_exact_pre`:
  - `solve_hotspots_cached/generic/difference_of_squares_fraction`:
    `25.859-26.924 us`, improvement about `12.9-15.3%`
  - `solve_eval_hotspots_cached/generic/difference_of_squares_fraction`:
    `24.708-25.153 us`, improvement about `13.8-15.7%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `123.14-124.46 us`, no statistically significant change
  - note:
    the batch benchmark here is diluted by parse/setup work that the dedicated
    hotspot benches do not include, so this change is retained as a real local
    win rather than a batch-moving one
- validation after the change:
  - `cargo test -p cas_math difference_of_squares_support --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_layer2_difference_of_squares -- --exact`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_layer2_difference_of_squares_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
    keeps `numeric-only = 168`
- retained another hidden-path pre-order cut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/transform_helpers.rs`
  for the exact no-op shape `atom^atom / atom`
- rationale:
  - `(a^x)/a` was already skipping late phases after `Core`, but it was still
    paying the full `Div` child walk inside `Core` even though the result stays
    unchanged and no rule ever fires
  - because both the base and exponent are already symbolic atoms in the hot
    benchmark case, there is no profitable child simplification to do
- scope is intentionally narrow:
  - solve context only
  - `steps off` only
  - no listener attached
  - `Core` phase only
  - non-strict domain modes only
  - only exact `atom^atom / atom` where numerator base and denominator match
- retained measurements against named baseline `pow_over_same_atom_pre`:
  - `solve_hotspots_cached/generic/a_pow_x_over_a`:
    `6.8061-6.9345 us`, improvement about `43.9-45.6%`
  - `solve_phase_subset_cached/a_pow_x_over_a/generic/full`:
    `6.7557-6.8690 us`, improvement about `42.8-44.9%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `113.73-114.38 us`, improvement about `3.8-4.7%`
- guardrails/validation after the change:
  - `profile_cache_tests::test_from_profile_solve_tactic_skips_late_phases_for_symbolic_power_over_same_atom`
    now also fixes `stats.core.rewrites_used == 0`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
    keeps `numeric-only = 168`
- retained a small exact-match shortcut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra/fractions/didactic_factor_support.rs`
  for the expanded-denominator planner used by
  `(x^2 + 2*x*y + y^2)/(x + y)^2`
- rationale:
  - the planner already constructs the exact expanded form of `(a+b)^2`
  - when the numerator is already exactly that expanded expression, the
    polynomial matcher is unnecessary overhead
- retained measurements against named baseline `binomial_expand_exact_pre`:
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `97.470-99.248 us`, improvement about `1.3-4.4%`
  - `solve_eval_hotspots_cached/generic/binomial_square_fraction`:
    `94.878-96.370 us`, improved in absolute time but still within
    Criterion noise on rerun
- validation after the change:
  - `cargo test -p cas_engine didactic_factor_support --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_binomial_square_cancel_in_solve_generic_context_steps_off -- --exact`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
    keeps `numeric-only = 168`
- rejected a hidden-path pre-order shortcut for
  `(x^3 - y^3)/(x - y)` and `(x^3 + y^3)/(x + y)`
- rationale:
  - the dedicated cube hotspots improved strongly in isolation
  - but the representative `solve_modes_cached/solve_tactic_generic_batch`
    regressed against named baseline `cubes_exact_den_pre`, so the extra
    `Div` guard cost did not pay for itself on the real batch
- discarded measurements:
  - `solve_hotspots_cached/generic/difference_of_cubes_fraction` improved
    locally into roughly `190.84-192.75 us`
  - `solve_hotspots_cached/generic/sum_of_cubes_fraction` stayed roughly flat
    around `179.60-182.47 us`
  - `solve_modes_cached/solve_tactic_generic_batch` regressed to
    `124.86-125.80 us`, about `+6.0-7.7%`
- tried and removed an exact raw-denominator shortcut inside
  `try_plan_sum_diff_of_cubes_in_num(...)`
- rationale:
  - the `difference_of_squares` planner had benefited from skipping expensive
    equivalence checks when the denominator was already exactly `A-B` / `A+B`
  - the same idea looked plausible for `(a^3 - b^3)/(a-b)` and
    `(a^3 + b^3)/(a+b)`, because the benchmark inputs already use exact raw
    factors
- discarded measurements against named baseline `cubes_exact_den_planner_pre`:
  - `solve_hotspots_cached/generic/difference_of_cubes_fraction` stayed within
    noise around `284.09-287.12 us`
  - `solve_hotspots_cached/generic/sum_of_cubes_fraction` regressed to
    `192.63-195.39 us`
  - `solve_modes_cached/solve_tactic_generic_batch` stayed flat around
    `118.77-119.70 us`
- retained new cube instrumentation in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`
  under:
  - `fraction_rule_direct/apply/generic/difference_of_cubes_fraction`
  - `fraction_rule_direct/apply/generic/sum_of_cubes_fraction`
  - `fraction_rule_direct/single_rule_engine/generic/difference_of_cubes_fraction`
  - `fraction_rule_direct/single_rule_engine/generic/sum_of_cubes_fraction`
  - `solve_phase_subset_cached/difference_of_cubes_fraction/*`
  - `solve_phase_subset_cached/sum_of_cubes_fraction/*`
- current measurements from that instrumentation:
  - `fraction_rule_direct/apply/generic/difference_of_cubes_fraction`:
    `23.665-23.969 us`
  - `fraction_rule_direct/apply/generic/sum_of_cubes_fraction`:
    `5.4774-5.6330 us`
  - `fraction_rule_direct/single_rule_engine/generic/difference_of_cubes_fraction`:
    `64.419-64.922 us`
  - `fraction_rule_direct/single_rule_engine/generic/sum_of_cubes_fraction`:
    `29.063-29.573 us`
  - `solve_phase_subset_cached/difference_of_cubes_fraction/generic/full`:
    `286.54-290.42 us`
  - `solve_phase_subset_cached/difference_of_cubes_fraction/generic/no_transform`:
    `276.87-286.37 us`
  - `solve_phase_subset_cached/difference_of_cubes_fraction/generic/no_transform_no_rationalize`:
    `276.19-279.73 us`
  - `solve_phase_subset_cached/sum_of_cubes_fraction/generic/full`:
    `189.69-192.54 us`
  - `solve_phase_subset_cached/sum_of_cubes_fraction/generic/no_transform`:
    `182.33-188.76 us`
  - `solve_phase_subset_cached/sum_of_cubes_fraction/generic/no_transform_no_rationalize`:
    `180.94-182.65 us`
- takeaway:
  - the cube hotspots are not dominated by the didactic planner itself
  - `SimplifyFractionRule::apply(...)` is only a small slice of the total cost,
    especially for `sum_of_cubes`
  - removing `Transform` and `Rationalize` trims only a modest tail, so the
    remaining cost is still mostly inside the core solve loop / repeated rule
    traversal rather than in the planner helper
- retained a dedicated profiling mode
  `CAS_SOLVE_BENCH_PROFILE_MODE=hotspots-cubes`
- rationale:
  - the generic hotspot profile had become dominated by smaller no-op cases and
    did not include the cube fraction inputs
  - a dedicated mode makes it possible to inspect event traces and top-applied
    rules for `(x^3 - y^3)/(x - y)` and `(x^3 + y^3)/(x + y)` without mixing
    them with unrelated hotspots
- current detail snapshot from `hotspots-cubes`:
  - `(x^3 - y^3)/(x - y)` still pays:
    - `Canonicalize Negation=4`
    - `Simplify Nested Fraction=2`
    - `Cancel Fraction Signs=1`
    - `Canonicalize Division=1`
    - `Flip binomial under negative coefficient=1`
  - `(x^3 + y^3)/(x + y)` stays simpler:
    - `Simplify Nested Fraction=2`
    - `Canonicalize Addition=1`
  - no-hit `Div` bucket probes are cheap and not the dominant source of cost
- tried and removed a local reorder in `try_plan_sum_diff_of_cubes_in_num(...)`
  to build the sum case as `a² + b² - ab` in a more canonical order
- discarded measurements against named baseline `sum_cube_canonical_pre`:
  - `fraction_rule_direct/apply/generic/sum_of_cubes_fraction` stayed flat
  - `fraction_rule_direct/single_rule_engine/generic/sum_of_cubes_fraction`
    regressed to `28.985-29.439 us`
  - `solve_hotspots_cached/generic/sum_of_cubes_fraction` stayed flat
  - `solve_phase_subset_cached/sum_of_cubes_fraction/generic/full` regressed to
    `192.76-194.61 us`
  - `solve_modes_cached/solve_tactic_generic_batch` regressed to
    `122.01-123.14 us`
- retained a very narrow hidden-path pre-order shortcut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/transform_helpers.rs`
  backed by
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra/mod.rs`
  for the exact raw shapes:
  - `(a^3 - b^3)/(a-b)`
  - `(a^3 + b^3)/(a+b)`
- rationale:
  - the cube hotspots were still dominated by child-recursion and sign /
    canonicalization churn before `SimplifyFractionRule` could act at the root
  - the earlier cube pre-order experiment regressed the batch because it used a
    broader planner path; this version only matches the exact raw shape with a
    cheap structural guard and returns the closed-form result directly
- scope is intentionally narrow:
  - solve context hidden path only
  - `steps off` only
  - no listener attached
  - `Core` phase only
  - exact denominator shapes only; no `poly_eq` or generalized algebraic
    equivalence checks
- retained measurements against named baseline `cubes_exact_den_planner_pre`:
  - `solve_hotspots_cached/generic/difference_of_cubes_fraction`:
    `140.62-141.72 us`, improvement about `50.1-50.9%`
  - `solve_hotspots_cached/generic/sum_of_cubes_fraction`:
    `128.13-129.28 us`, improvement about `31.4-33.0%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `120.91-124.73 us`, change stayed within Criterion noise
- guardrails/validation after the change:
  - `profile_cache_tests::test_from_profile_solve_tactic_difference_of_cubes_uses_exact_preorder_fast_path`
  - `profile_cache_tests::test_from_profile_solve_tactic_sum_of_cubes_uses_exact_preorder_fast_path`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests
    test_difference_of_cubes_cancel_in_solve_generic_context_steps_off -- --exact`
  - `cargo test -p cas_solver --test multivar_gcd_tests
    test_sum_of_cubes_cancel_in_solve_generic_context_steps_off -- --exact`
  - `cargo test -p cas_solver --test multivar_gcd_tests
    test_difference_of_cubes_cancel_in_solve_strict_context_steps_off -- --exact`
  - `cargo test -p cas_solver --test multivar_gcd_tests
    test_sum_of_cubes_cancel_in_solve_strict_context_steps_off -- --exact`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests
    metatest_unified_benchmark -- --ignored --nocapture`
    keeps `numeric-only = 168`
- retained a second hidden solve fast path in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
  for the plain post-Core trinomial outputs of those cube rewrites
- exact shape accepted:
  - `a^2 + b^2 + a*b`
  - `a^2 + b^2 - a*b`
- rationale:
  - after the exact raw cube pre-order, the remaining time was no longer in the
    planner but in the late no-op phases of the pipeline
  - these trinomials are already in their final plain form on the hidden solve
    path, so `Transform`, `Rationalize`, `PostCleanup` and `BestSoFar` were just
    fixed-cost overhead
  - the guard is still narrow: only symbolic squares plus one symbolic cross
    term, solve context, `steps off`, no listener, and no denominator-root /
    auto-expand marks
- retained measurements:
  - compared against named baseline `cubes_exact_den_planner_pre`:
    - `solve_hotspots_cached/generic/difference_of_cubes_fraction`:
      `87.610-88.485 us`, improvement about `68.5-69.2%`
    - `solve_hotspots_cached/generic/sum_of_cubes_fraction`:
      `71.019-72.261 us`, improvement about `61.4-62.5%`
  - current absolute measurements on the end-to-end eval path:
    - `solve_eval_hotspots_cached/generic/difference_of_cubes_fraction`:
      `87.147-89.438 us`
    - `solve_eval_hotspots_cached/generic/sum_of_cubes_fraction`:
      `70.325-72.858 us`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `124.70-130.05 us`, Criterion reported no statistically significant change
- guardrails tightened after the cut:
  - `profile_cache_tests::test_from_profile_solve_tactic_difference_of_cubes_uses_exact_preorder_fast_path`
    now expects `Transform/Rationalize/PostCleanup = 0`
  - `profile_cache_tests::test_from_profile_solve_tactic_sum_of_cubes_uses_exact_preorder_fast_path`
    now expects `Transform/Rationalize/PostCleanup = 0`
- investigated the next hotspot `binomial_square_fraction` and kept only better
  instrumentation, not a runtime change
- added direct coverage in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`
  for:
  - `fraction_rule_direct/apply/generic/binomial_square_fraction`
  - `fraction_rule_direct/single_rule_engine/generic/binomial_square_fraction`
  - `solve_phase_subset_cached/binomial_square_fraction/generic/full`
  - `solve_phase_subset_cached/binomial_square_fraction/generic/no_transform`
  - `solve_phase_subset_cached/binomial_square_fraction/generic/no_transform_no_rationalize`
- current measurements:
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `101.99-103.71 us`
  - `solve_eval_hotspots_cached/generic/binomial_square_fraction`:
    `97.441-98.834 us`
  - `fraction_rule_direct/apply/generic/binomial_square_fraction`:
    `23.693-24.113 us`
  - `fraction_rule_direct/single_rule_engine/generic/binomial_square_fraction`:
    `37.070-38.265 us`
  - `solve_phase_subset_cached/binomial_square_fraction/generic/full`:
    `101.57-102.45 us`
  - `solve_phase_subset_cached/binomial_square_fraction/generic/no_transform`:
    `100.56-102.01 us`
  - `solve_phase_subset_cached/binomial_square_fraction/generic/no_transform_no_rationalize`:
    `101.94-103.28 us`
- takeaway:
  - `binomial_square_fraction` is not dominated by `Transform` or `Rationalize`
  - the root rule itself is only a minority of the total cost
  - the remaining time is living in the core solve loop / traversal around the
    rule, not in the late pipeline tail
- tried and removed a hidden-path root no-op preorder for the exact shape
  `(a^2 + 2ab + b^2)/(a+b)^2`
- discarded measurements against baseline `binomial_expand_exact_pre`:
  - `solve_hotspots_cached/generic/binomial_square_fraction` regressed to
    `103.30-104.00 us`
  - `solve_modes_cached/solve_tactic_generic_batch` regressed to
    `122.00-123.08 us`
- conclusion:
  - do not pursue a generic or root-preorder shortcut for this case
  - if we come back to it, the next hypothesis should target the core traversal
    around `SimplifyFractionRule`, not the planner helper and not late phases
- retained a solve-hidden early exit inside `Core` phase in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only `Core`
  - only `solve` context
  - only `steps off`
  - only after a pass that already changed the expression into:
    - a terminal exact value
    - a plain symbolic binomial
    - a plain symbolic cube trinomial (`a^2 + b^2 ± a*b`)
- rationale:
  - several hot solve cases were already paying a second full `Core` pass only to
    discover the fixed point after collapsing to their final closed form
  - this cut removes that extra pass without changing later pipeline decisions
- current absolute measurements after the change:
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `98.705-100.64 us`
  - `solve_hotspots_cached/generic/x_over_x`:
    `6.4083-6.6122 us`
  - `solve_hotspots_cached/generic/difference_of_cubes_fraction`:
    `42.438-43.140 us`
- practical takeaway:
  - the remaining good ROI is still in the `Core` solve loop, not in late
    phases or in more planner-only tweaks
