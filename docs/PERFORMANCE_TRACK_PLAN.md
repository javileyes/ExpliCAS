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
  - `profile_cache_tests::test_from_profile_solve_tactic_scalar_multiple_fraction_uses_root_fast_path`
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
- reranked `solve_hotspots_cached/generic/*` after the retained hidden-`Core`
  cuts to avoid optimizing stale leaders
- current ordering:
  - `binomial_square_fraction`: `96.066-96.723 us`
  - `perfect_square_minus_fraction`: `78.893-81.936 us`
  - `difference_of_cubes_fraction`: `40.897-41.288 us`
  - `power_quotient_fraction`: `40.952-42.095 us`
  - `log_power_base`: `24.238-24.549 us`
  - `sum_of_cubes_fraction`: `22.640-23.052 us`
  - `difference_of_squares_fraction`: `14.743-15.013 us`
  - `scalar_multiple_fraction`: `10.696-10.991 us`
  - `a_pow_x_over_a`: `7.1479-7.3147 us`
  - `x_pow_0`: `6.7079-6.8364 us`
  - `x_over_x`: `6.2308-6.3617 us`
  - `exp_ln_x`: `5.8491-6.0340 us`
- added a focused solve profile mode `hotspots-binomials` in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`
  to inspect the remaining top family with rule-hit detail
- diagnostic takeaway:
  - `(x^2 + 2*x*y + y^2)/(x + y)^2` still hits `Canonicalize Multiplication`
    before `Simplify Nested Fraction` and `Cancel Identical Numerator/Denominator`
  - `(x^2 - 2*x*y + y^2)/(x - y)` also pays `Canonicalize Multiplication`, then
    several sign/canonicalization rewrites before collapsing
- retained a small-chain fast path in
  `/Users/javiergimenezmoya/developer/math/crates/cas_math/src/expr_nary.rs`
  for `try_rewrite_canonicalize_mul_expr(...)`
- scope:
  - only 3-factor top-level `Mul` chains
  - only commutative factors
  - sorts or right-associates in-place without falling through `mul_leaves(...)`
- validation:
  - added directed coverage:
    - `expr_nary::tests::test_canonicalize_mul_small_chain_sorts_three_factors`
    - `expr_nary::tests::test_canonicalize_mul_small_chain_right_associates_sorted_chain`
  - `cargo test -p cas_math expr_nary --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- retained measurements:
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `97.148-98.783 us` (slight regression)
  - `solve_hotspots_cached/generic/perfect_square_minus_fraction`:
    `75.987-78.507 us`, improvement `~2.9-5.4%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `91.048-94.230 us`, improvement `~6.6-8.6%`
- decision:
  - keep the change even though the isolated binomial hotspot regresses slightly,
    because the representative batch improves clearly and the fast path is now
    covered directly
- retained a smaller win in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra/fractions/didactic_factor_support.rs`
  by checking structural equality via `compare_expr(...) == Equal` before
  falling back to `poly_eq(...)` in `expr_matches_poly(...)`
- rationale:
  - the remaining top hotspot family (`binomial_square_fraction`,
    `perfect_square_minus_fraction`) was still paying polynomial comparison even
    when the compared expressions were already structurally equal after earlier
    canonicalization
  - this narrows the cost inside the didactic fraction planner without adding a
    new global gate in the transformer
- retained measurements:
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `88.680-90.631 us`, improvement `~1.7-4.5%`
  - `solve_hotspots_cached/generic/perfect_square_minus_fraction`:
    `74.464-78.198 us`, no significant change
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `87.104-87.668 us`, improvement `~3.4-5.2%`
- validation:
  - `cargo test -p cas_engine didactic_factor_support --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_binomial_square_cancel_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- re-tried a more explicit exact trinomial matcher inside
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra/fractions/didactic_factor_support.rs`
  and dropped it again:
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `89.954-90.860 us`, no statistically significant change
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `86.998-90.817 us`, no statistically significant change
- retained a broader comparison-cost reduction instead:
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/ordering.rs`
  now uses `SmallVec<[(ExprId, ExprId); 8]>` inside `compare_expr(...)`
  instead of allocating a heap `Vec` per comparison
- rationale:
  - `compare_expr(...)` is on the hot path of several retained optimizations:
    didactic fraction planners, factor/power shortcuts, and the new
    `canonicalize_mul_small_chain` fast path
  - most comparisons are shallow, so keeping the work stack inline removes a
    fixed allocator cost from a broad cross-section of solve hotspots
- retained measurements:
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `86.942-87.422 us`, improvement `~2.6-4.8%`
  - `solve_hotspots_cached/generic/difference_of_cubes_fraction`:
    `36.262-36.643 us`, improvement `~5.7-7.2%`
  - `solve_hotspots_cached/generic/perfect_square_minus_fraction`:
    `72.768-74.692 us`, no statistically significant change
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `86.441-87.118 us`, change within Criterion noise threshold but still lower
    in absolute terms than the prior retained baseline
- validation:
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_binomial_square_cancel_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- retained another solve-specific cut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra/fractions/gcd_cancel.rs`:
  `Cancel Identical Numerator/Denominator`, `Cancel Power Fraction`, and
  `Cancel Same-Base Powers` now skip the oracle entirely whenever the current
  domain mode already allows unproven definability. In those modes the rule
  emits the existing `NonZero(expr)` assumption key directly and only `Strict`
  still falls through to the prover-backed oracle.
- rationale:
  - this was already the effective policy for plain variables via the old
    `fast_variable_nonzero_decision(...)`
  - the hidden solve hotspots were still paying oracle/prover overhead for
    non-atomic denominators such as `(x + y)^2` and `(x - y)`, even though
    `Generic/Assume` were always going to accept them with a condition
- retained measurements:
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `86.295-87.498 us`, change within Criterion noise but lower in absolute terms
  - `solve_hotspots_cached/generic/perfect_square_minus_fraction`:
    `70.244-71.235 us`, change within Criterion noise but lower in absolute terms
  - `solve_eval_hotspots_cached/generic/binomial_square_fraction`:
    `81.423-85.783 us`, improvement `~15.5-18.1%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `84.821-88.509 us`, change within Criterion noise but lower in absolute terms
- validation:
  - `cargo test -p cas_solver --test multivar_gcd_tests test_binomial_square_cancel_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo test -p cas_solver --test domain_contract_tests test_strict_x_div_x_stays_unchanged -- --exact`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- retained a narrow exact preorder for raw binomial-square fractions in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/transform_helpers.rs`
- scope:
  - only in the hidden solve path
  - only with `steps off`
  - only with no event listener
  - only in `Core`
  - only outside `Strict`
  - only for the exact raw shape `(a^2 + 2ab + b^2) / (a + b)^2`
- rationale:
  - after the broader cuts, `binomial_square_fraction` was still the top direct
    solve hotspot
  - the remaining work was mostly traversal and rule dispatch around a case that
    can be recognized exactly and collapsed to `1` without needing the full
    rule/orchestration path
  - keeping the gate exact avoids introducing cost to unrelated `Div` nodes or
    weakening the strict-domain contract
- retained measurements:
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `9.1353-9.3533 us`, improvement `~89%`
  - `solve_eval_hotspots_cached/generic/binomial_square_fraction`:
    `8.7842-8.9835 us`, improvement `~89%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `84.811-85.409 us`, no statistically significant change
- validation:
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_binomial_square_cancel_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_binomial_square_cancel_in_solve_strict_context_steps_off -- --exact`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- retained a matching exact preorder for raw perfect-square-minus fractions in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/transform_helpers.rs`
- scope:
  - only in the hidden solve path
  - only with `steps off`
  - only with no event listener
  - only in `Core`
  - only outside `Strict`
  - only for the exact raw shape `(a^2 - 2ab + b^2) / (a - b)`
- rationale:
  - after the binomial-square cut, `perfect_square_minus_fraction` became the
    next clear direct hotspot in the solve rerank
  - the generic preorder still paid `expr_matches_poly(...)` and then left a
    cheap trailing pass in the pipeline; the raw exact shape can be recognized
    directly and collapsed to `a - b` before entering that path
  - keeping the gate exact preserves the existing `Strict` behavior and avoids
    adding a new fixed cost to unrelated `Div` nodes
- retained measurements:
  - `solve_hotspots_cached/generic/perfect_square_minus_fraction`:
    `50.530-50.933 us`, improvement `~28-30%`
  - `solve_eval_hotspots_cached/generic/perfect_square_minus_fraction`:
    `48.091-48.494 us`, improvement `~69-70%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `87.375-88.654 us`, change within Criterion noise threshold
- validation:
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_perfect_square_minus_cancel_in_solve_generic_context_steps_off -- --exact`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_perfect_square_minus_cancel_in_solve_strict_context_steps_off -- --exact`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- retained a standard-context root no-op shortcut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only in `ContextMode::Standard`
  - only in `ValueDomain::RealOnly`
  - only with no step listener attached
  - only for exact roots that are guaranteed no-op while complex rewrites are
    disabled:
    - `i^n` with integer `n`
    - Gaussian rational division `(a+bi)/(c+di)`
- rationale:
  - the standard REPL path was still paying the full simplify pipeline for
    expressions that are intentionally left untouched in real mode and only
    produce the later `Imaginary Usage Warning`
  - the narrow root gate keeps complex-enabled semantics unchanged and avoids
    touching mixed expressions where real-only simplification can still do useful
    work in other subtrees
- retained measurements:
  - `simplify_cached_vs_uncached/cached/complex/i_power`:
    `20.326-20.514 us`, materially lower in absolute terms than the prior
    `~41 us` band and stable in reruns
  - `simplify_cached_vs_uncached/cached/complex/gaussian_div`:
    `135.02-136.18 us`, effectively flat in microbench reruns
  - `repl_full_eval/cached/batch_11_inputs`:
    `1.1808-1.1910 ms`, improvement `~3.4-6.0%`
- validation:
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_i_power_real_domain_uses_root_noop_fast_path --lib`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_gaussian_div_real_domain_uses_root_noop_fast_path --lib`
  - `cargo test -p cas_solver --test complex_number_tests`
  - `cargo run -q -p cas_cli -- eval 'i^5' --steps on --format json`
  - `cargo run -q -p cas_cli -- eval '(3 + 4*i)/(1 + 2*i)' --steps on --format json`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'simplify_cached_vs_uncached/cached/(complex/gaussian_div|complex/i_power)' -- --noplot`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
- discarded a standard-context root shortcut for exact affine `sqrt(square)` in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- result:
  - reverted
- rationale:
  - although the target case `((5*x + 8/3)*(5*x + 8/3))^(1/2)` looked like a
    good candidate for an early exit, the narrower root shortcut still regressed
    the real workload around it instead of helping
- measured outcome before revert:
  - `simplify_cached_vs_uncached/cached/heavy/abs_square`:
    `187.20-188.47 us`, no significant win
  - `simplify_cached_vs_uncached/cached/heavy/nested_root`:
    `134.56-137.49 us`, regression `~1.2-4.1%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `1.2177-1.2482 ms`, regression `~1.5-4.3%`
- retained an exact `Mul/Mul` common-factor preorder for the standard path in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra/mod.rs`
  and
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/transform_helpers.rs`
- scope:
  - exact raw shape only: `(F*A)/(F*B)`, `(A*F)/(F*B)`, `(F*A)/(B*F)`, `(A*F)/(B*F)`
  - only outside `Strict`
  - only with no listener attached
  - integrated in the existing pre-order `Div` transform path, so it helps
    standard `eval` / REPL with `steps on`
- rationale:
  - reranking `repl_individual/cached/*` showed
    `((x+y)*(a+b))/((x+y)*(c+d))` as the clear dominant cached REPL input at
    about `~650 us`
  - that case already exposes the shared factor structurally, so paying the
    full fraction-cancellation pipeline was unnecessary
  - the narrow preorder removes the traversal cost while preserving the visible
    behavior: one step, same result, and the same required conditions
- retained measurements:
  - `repl_individual/cached/08_((x+y)*(a+b))/((x+y)`:
    `158.04-159.47 us`, improvement `~75.4-75.9%`
  - `simplify_cached_vs_uncached/cached/gcd/layer25_multiparam`:
    `152.78-154.81 us`, improvement `~76.0-76.7%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `1.2446-1.2791 ms`, improvement `~23.9-25.7%`
- validation:
  - `cargo test -p cas_engine algebra::tests::test_exact_common_factor_mul_fraction_preorder --lib`
  - `cargo test -p cas_solver --test anti_catastrophe_tests test_cancel_vs_gcd_structural_cancel_multiparam -- --exact`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo run -q -p cas_cli -- eval '((x+y)*(a+b))/((x+y)*(c+d))' --steps on --format json`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- tried a narrower root dispatcher split for denominator-`Pow` cases in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`,
  but reverted it after a broader rerank
- scope:
  - only in the hidden `solve` root-shortcut path
  - only for `Div(_, Pow(_, _))`
  - if the numerator is `Pow`, try only the `power_quotient` shortcut
  - if the numerator is `Add`, try only the exact `binomial_square` shortcut
  - otherwise, skip both matchers
- rationale:
  - after the root fast paths landed, the dispatcher was still probing
    irrelevant matchers on denominator-`Pow` expressions
  - most notably, `x^4/x^2` still paid the `binomial_square` probe, and exact
    binomial-square inputs still went through the power-quotient branch shape
    checks
  - tightening that branch by numerator shape removes dead matcher attempts
    without changing any accepted shortcut shape
- initial focused measurements:
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `60.230-60.581 us`, improvement `~1.0-3.0%`
  - `solve_hotspots_cached/generic/power_quotient_fraction`:
    `3.8677-3.9762 us`, no statistically significant change
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `4.2923-4.5306 us`, no statistically significant change
- broader rerank before keeping the change:
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `61.243-61.831 us`, regression `~1.9-3.3%`
  - direct hotspots stayed inside noise
- decision:
  - reverted
  - the focused win did not survive the wider rerank, so the current fast
    benchmark frontier is too noisy to justify this extra branch split
- validation:
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_power_quotient_uses_root_fast_path --lib`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_binomial_square_uses_root_fast_path --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/solve_tactic_generic_batch|solve_hotspots_cached/generic/(power_quotient_fraction|binomial_square_fraction)' -- --noplot`
- retained a local fast equality check for the exact cubes preorder in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra/mod.rs`
- scope:
  - only inside `try_exact_sum_diff_of_cubes_preorder(...)`
  - only for the raw denominator atom match in `(a^3 - b^3)/(a-b)` and
    `(a^3 + b^3)/(a+b)`
  - direct `ExprId` match first, then cheap `Variable`/`Constant`/`Number`
    equality, and only then `compare_expr(...)`
- rationale:
  - the exact cubes root shortcut was already retained, but it still paid full
    structural comparison for atom-like denominator matching
  - on the current rerank, `difference_of_cubes_fraction` and
    `sum_of_cubes_fraction` are still frequent enough inside
    `solve_tactic_generic_batch` that shaving that fixed comparison cost moves
    the representative batch even when the isolated hotspots stay within noise
- retained measurements:
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `60.700-61.051 us`, improvement `~1.2-2.4%`
  - `solve_hotspots_cached/generic/difference_of_cubes_fraction`:
    `5.3401-5.5146 us`, lower in absolute terms but still within Criterion noise
  - `solve_hotspots_cached/generic/sum_of_cubes_fraction`:
    `5.2609-5.4558 us`, lower in absolute terms but still within Criterion noise
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_difference_of_cubes_uses_exact_preorder_fast_path --lib`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_sum_of_cubes_uses_exact_preorder_fast_path --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests difference_of_cubes -- --nocapture`
  - `cargo test -p cas_solver --test multivar_gcd_tests sum_of_cubes -- --nocapture`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_hotspots_cached/generic/(difference_of_cubes_fraction|sum_of_cubes_fraction)|solve_modes_cached/solve_tactic_generic_batch' -- --noplot`
- retained a pipeline setup cut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - delay `set_sticky_implicit_domain(...)` until after the hidden solve root
    shortcuts
  - keep it enabled for the full phase pipeline exactly as before
  - rely on final `diagnostics` to derive `required_conditions` from input/result
    on the shortcut path, which it already does
- rationale:
  - the representative `solve` batches now spend most of their time in root
    shortcuts (`x/x`, `x^0`, `exp(ln(x))`, `a^x/a`, scalar multiple fraction,
    difference of squares)
  - all of those cases were still paying an eager `infer_implicit_domain(...)`
    through `set_sticky_implicit_domain(...)` even though they return before any
    phase that consumes the sticky domain
  - moving that setup behind the shortcuts removes a fixed-cost prelude from the
    actual hot path without changing the final domain diagnostics
- retained measurements:
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `59.850-61.425 us`, improvement `~1.5-3.6%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `59.747-60.138 us`, improvement `~1.7-3.1%`
  - direct shortcut hotspots (`scalar_multiple_fraction`, `difference_of_squares`,
    `x/x`, `exp(ln(x))`, `a^x/a`, `x^0`) all moved down in absolute terms, with
    individual reruns staying within expected Criterion noise
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo run -q -p cas_cli -- eval --context solve --domain generic --steps off --format json 'x/x'`
  - `cargo run -q -p cas_cli -- eval --context solve --domain generic --steps off --format json 'exp(ln(x))'`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/solve_tactic_generic_batch|solve_modes_cached/solve_tactic_assume_batch|solve_hotspots_cached/generic/(scalar_multiple_fraction|difference_of_squares_fraction|x_over_x|exp_ln_x|a_pow_x_over_a|x_pow_0)|solve_hotspots_cached/assume/(x_over_x|x_pow_0)' -- --noplot`
- retained a narrower root-dispatch ordering for exact cubes in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only on hidden solve root shortcuts for `Div`
  - for `den = Add` or `den = Sub`, try the exact raw
    `try_exact_sum_diff_of_cubes_preorder(...)` path before the unrelated
    scalar-multiple / difference-of-squares / perfect-square-minus shortcuts
- rationale:
  - after the earlier exact cube shortcuts, the remaining cost for
    `sum_of_cubes_fraction` and `difference_of_cubes_fraction` was mostly the
    dispatcher paying several cheap-but-useless failed matchers before reaching
    the cube handler
  - the exact cube matcher is already narrow and raw-shape-only, so moving it
    ahead of the unrelated branches cuts that dead work without widening the
    root fast path surface
- retained measurements:
  - `solve_hotspots_cached/generic/difference_of_cubes_fraction`:
    `5.2322-5.3896 us`, lower in absolute terms than the immediate pre-change
    rerank (`5.3980-5.5630 us`)
  - `solve_hotspots_cached/generic/sum_of_cubes_fraction`:
    `5.0886-5.3141 us`, improvement `~10.5-16.5%`
  - `solve_hotspots_cached/generic/scalar_multiple_fraction`:
    `5.4282-5.5695 us`, no statistically significant change
  - `solve_hotspots_cached/generic/difference_of_squares_fraction`:
    `3.9495-4.1054 us`, no statistically significant change
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `60.046-60.697 us`, lower in absolute terms than the immediate pre-change
    rerank (`60.794-61.729 us`) with Criterion still reporting noise-threshold
    change
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_difference_of_cubes_uses_exact_preorder_fast_path --lib`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_sum_of_cubes_uses_exact_preorder_fast_path --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests difference_of_cubes -- --nocapture`
  - `cargo test -p cas_solver --test multivar_gcd_tests sum_of_cubes -- --nocapture`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo bench -p cas_engine --bench profile_cache "solve_hotspots_cached/generic/scalar_multiple_fraction|solve_hotspots_cached/generic/difference_of_squares_fraction|solve_hotspots_cached/generic/difference_of_cubes_fraction|solve_hotspots_cached/generic/sum_of_cubes_fraction|solve_modes_cached/solve_tactic_generic_batch" -- --noplot`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- retained a denominator-shape dispatch cut for hidden solve root shortcuts in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only on the existing hidden solve root shortcut path for `Div`
  - dispatches by denominator shape (`atom`, `Pow`, `Add`, `Sub`) before trying
    shortcut families, instead of probing unrelated shortcuts in sequence
- rationale:
  - after the earlier root fast paths, the remaining batch cost was largely
    dispatch overhead on just six solve inputs
  - `difference_of_squares`, scalar-multiple fractions, `x/x`, `(a^x)/a`,
    `x^0`, and `exp(ln(x))` do not need the same shortcut probes
  - narrowing by denominator shape removes dead checks from the hot path
    without changing semantics or widening any shortcut gate
- retained measurements:
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `58.363-58.780 us`, improvement `~1.4-2.3%`
  - `solve_hotspots_cached/generic/difference_of_squares_fraction`:
    `3.8041-3.9199 us`, no significant change on rerun
  - `solve_hotspots_cached/generic/x_over_x`:
    `3.2905-3.4177 us`, no significant change on rerun
- validation:
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/solve_tactic_generic_batch|solve_hotspots_cached/generic/difference_of_squares_fraction|solve_hotspots_cached/generic/x_over_x' -- --noplot`
- retained a structural-domain win for `power_quotient` in
  `/Users/javiergimenezmoya/developer/math/crates/cas_solver_core/src/domain_inference.rs`
  and `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - `infer_implicit_domain(...)` now propagates `NonZero(base)` from input
    denominators of the form `base^n` when `n` is a positive integer
  - the hidden root shortcut for `P^m/P^n` in solve mode now returns directly,
    without building a hidden `Step`
- rationale:
  - before this change, `x^4/x^2 -> x^2` still needed a hidden step to keep the
    extra `x ≠ 0` require, because structural diagnostics only inferred
    `x^2 ≠ 0`
  - once input-domain inference learns that `x^2 ≠ 0` implies `x ≠ 0`, the
    shortcut can behave like `x/x`: no hidden step, same visible requires
- retained measurements:
  - `solve_hotspots_cached/generic/power_quotient_fraction`:
    `4.0082-4.1520 us`, improvement `~4.8-13.7%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `58.738-59.178 us`, improvement `~1.7-3.5%`
- validation:
  - `cargo test -p cas_solver_core domain_inference --lib`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_power_quotient_uses_root_fast_path --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_power_quotient_cancel_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo run -q -p cas_cli -- eval --context solve --domain generic --steps off --format json 'x^4/x^2'`
  - `cargo run -q -p cas_cli -- eval --context solve --domain assume --steps off --format json 'x^4/x^2'`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- retained a raw-node construction fast path for exact cubes in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra/mod.rs`
- scope:
  - only the exact preorder planner for `(a^3 - b^3)/(a-b)` and `(a^3 + b^3)/(a+b)`
  - keeps the same rewritten forms as before
  - replaces smart builders in the exact-result path with direct `add_raw(...)`
    node construction
- rationale:
  - after the root shortcut removed most pipeline overhead, the remaining cost in
    `difference_of_cubes_fraction` and `sum_of_cubes_fraction` still included
    avoidable AST construction churn inside the exact cube result builder
  - these results are already canonical for the exact fast path, so using
    `add_raw(...)` avoids extra helper work without widening the rewrite surface
- retained measurements against baseline `cubes_exact_id_pre`:
  - `solve_hotspots_cached/generic/difference_of_cubes_fraction`:
    `5.9418-6.3011 us`, improvement `~4.5-11.1%`
  - `solve_hotspots_cached/generic/sum_of_cubes_fraction`:
    `5.9373-6.1831 us`, improvement `~8.4-14.0%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `61.093-61.485 us`, lower in absolute terms but still within Criterion noise
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_difference_of_cubes_uses_exact_preorder_fast_path --lib`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_sum_of_cubes_uses_exact_preorder_fast_path --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests difference_of_cubes -- --nocapture`
  - `cargo test -p cas_solver --test multivar_gcd_tests sum_of_cubes -- --nocapture`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- retained a root-kind dispatch cut for hidden solve shortcuts in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only in the hidden solve path (`solve`, `steps off`, no listener)
  - keeps the same shortcut set and the same strict/non-strict guards
  - routes by top-level `Expr` kind (`Pow`, `Function`, `Div`) before trying
    specialized root shortcuts
- rationale:
  - after adding many exact root shortcuts, the hot path was paying a fixed
    chain of irrelevant pattern checks on every expression, plus a repeated
    `Div` unpack in the second shortcut block
  - dispatching by root kind removes that fixed overhead without widening
    shortcut semantics; the retained win shows up in the representative batch,
    not in any single hotspot
- retained measurements:
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `61.491-62.352 us`, improvement `~0.9-2.8%`
  - `solve_hotspots_cached/generic/scalar_multiple_fraction`:
    `5.591-5.832 us`, within noise
  - `solve_hotspots_cached/generic/difference_of_cubes_fraction`:
    `6.011-6.237 us`, within noise
  - `solve_hotspots_cached/generic/sum_of_cubes_fraction`:
    `6.051-6.229 us`, within noise
  - `solve_hotspots_cached/generic/log_power_base`:
    `5.422-5.590 us`, within noise
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- retained a hidden-step description simplification for `assume/log_power_base`
  in `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only the hidden solve root shortcut for numeric `log(a^m, a^n)`
  - only the branch that still needs a hidden `Step` (for example `assume`,
    where `Positive(base_core)` must still be preserved)
  - replaces the formatted per-input narration with a static compact
    description because the hidden step is not rendered in `steps off`
- rationale:
  - the shortcut still needs the hidden `Step` to preserve `x > 0` in
    `assume`, so removing it entirely is not sound
  - but formatting `"log(a^m, a^n) = n/m"` per input was pure overhead in the
    hot path because that description is not surfaced when `steps` are off
- retained measurements:
  - `solve_hotspots_cached/assume/log_power_base`:
    `5.7111-5.9032 us`, improvement `~3.5-8.9%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `61.167-61.553 us`, lower in absolute terms, with Criterion marking the
    change inside noise
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_log_power_base_uses_root_fast_path --lib`
  - `cargo test -p cas_solver --test root_log_parity_tests log_power_base::log_x2_x6_gives_3 -- --exact`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo run -q -p cas_cli -- eval --context solve --domain assume --steps off --format json 'log(x^2, x^6)'`
- retained a root hidden-solve shortcut for the exact binomial-square fraction
  `(a^2 + 2ab + b^2)/(a+b)^2` in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only with `steps off`
  - only in solve mode
  - only with no step listener attached
  - only outside `Strict`
  - only for the exact raw root shape, with no hidden-step metadata
- rationale:
  - after reranking, `binomial_square_fraction` was still the hottest direct
    generic solve case at about `10.125-10.410 us`
  - an earlier attempt to preserve its `requires` through a hidden-required
    channel regressed badly; the better version is to return `1` at the root
    and let `eval::diagnostics` recover the denominator requirements from the
    original input and output implicit domains
  - that keeps the generic `requires` contract (`(x+y)^2 != 0` and
    `x^2 + y^2 + 2*x*y != 0`) without paying prepasses or any phase-loop work
- retained measurements:
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `4.6276-4.7328 us`, improvement `~54.0-55.4%`
  - `solve_eval_hotspots_cached/generic/binomial_square_fraction`:
    `4.5868-4.7841 us`, improvement `~53.1-55.3%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `70.062-71.882 us`, improvement `~9.5-12.2%`
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_binomial_square_uses_root_fast_path --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_binomial_square_cancel_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_binomial_square_cancel_in_solve_strict_context_steps_off -- --exact`
  - `cargo run -q -p cas_cli -- eval --context solve --domain generic --steps off --format json '(x^2 + 2*x*y + y^2)/(x + y)^2'`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- discarded a hidden-solve root shortcut for the exact binomial-square fraction
  `(a^2 + 2ab + b^2)/(a+b)^2`
- rationale:
  - unlike `difference_of_squares`, this path must preserve two explicit
    `requires` in `generic + steps off`: the original denominator and the
    expanded numerator
  - a root shortcut therefore needed to synthesize a hidden `Step` with both
    `required_conditions`, and that metadata cost outweighed the saved pipeline
    work
- measured result against baseline `binomial_root_pre`:
  - `solve_hotspots_cached/generic/binomial_square_fraction` regressed to
    `23.753-24.164 us`
  - `solve_eval_hotspots_cached/generic/binomial_square_fraction` regressed to
    `23.574-23.718 us`
  - `solve_modes_cached/solve_tactic_generic_batch` stayed effectively flat at
    `78.372-78.760 us`
- takeaway:
  - for this case, the existing exact preorder inside the transformer is better
    than a root-level shortcut if the latter has to manufacture hidden step
    metadata just to preserve `requires`
  - future work here should target a cheaper way to carry hidden
    `required_conditions`, not another root bypass
- retained a narrow hidden-solve root shortcut for the exact no-op
  `(a^x)/a` case in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only with `steps off`
  - only in solve mode
  - only with no step listener
  - only outside `Strict`
  - only when the root expression is exactly `atom^symbolic_atom / atom`
- rationale:
  - this case is a true no-op in the result domain, already recognized later by
    the transform layer, but it was still paying eager prepasses and the fixed
    solve pipeline overhead before discovering that nothing changes
  - unlike the discarded binomial root shortcut, this bypass does not need to
    build hidden `Step` metadata, so the saved pipeline work translates into a
    real end-to-end win
- retained measurements against baseline `pow_same_atom_root_pre`:
  - `solve_hotspots_cached/generic/a_pow_x_over_a`:
    `3.2558-3.4323 us`, improvement `~52.7-56.4%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `72.822-73.287 us`, improvement `~5.5-6.8%`
- validation:
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_skips_late_phases_for_symbolic_power_over_same_atom --lib`
  - `cargo run -q -p cas_cli -- eval --context solve --domain generic --steps off --format json '(a^x)/a'`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
    - `numeric-only = 168`
- retained a narrow hidden-solve root shortcut for the exact symbolic `x^0`
  case in `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only with `steps off`
  - only in solve mode
  - only with no step listener
  - only outside `Strict`
  - only when the root expression is exactly `symbolic_atom^0`
- rationale:
  - in `generic` and `assume`, `x^0` already returns `1` without public
    `required_conditions`, so this is another case where the hidden solve path
    was paying eager prepasses and the fixed pipeline tail just to rediscover a
    trivial result
  - unlike the discarded binomial root shortcut, this bypass does not need to
    synthesize hidden step metadata or extra `requires`, so the saved pipeline
    work survives in the benchmark signal
- retained measurements against baseline `xpow0_root_pre`:
  - `solve_hotspots_cached/generic/x_pow_0`:
    `3.4772-3.6151 us`, improvement `~46.9-50.6%`
  - `solve_hotspots_cached/assume/x_pow_0`:
    `3.4655-3.6373 us`, improvement `~47.3-50.9%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `69.437-70.331 us`, small gain but within Criterion noise threshold
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `69.336-69.941 us`, improvement `~2.6-4.4%`
- validation:
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_x_pow_0_uses_root_fast_path --lib`
  - `cargo test -p cas_solver --test domain_contract_tests test_strict_x_pow_0_stays_unchanged -- --exact`
  - `cargo test -p cas_solver --test domain_assume_warnings_contract_tests assume_x_pow_0_simplifies_with_assumption -- --exact`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
    - `numeric-only = 168`
- retained a narrow hidden-solve root shortcut for the exact symbolic `x/x`
  case in `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only with `steps off`
  - only in solve mode
  - only with no step listener
  - only outside `Strict`
  - only when the root expression is exactly `symbolic_atom / same_symbolic_atom`
  - only under the same definability-policy gate already used by the hidden
    scalar-multiple and power-quotient shortcuts
- rationale:
  - `x/x` already had a preorder fast path inside the transformer, but the
    hidden solve route was still paying eager prepasses and the root pipeline
    before reaching it
  - unlike the discarded binomial root shortcut, this path needs only one
    `NonZero(x)` requirement, so the compact hidden-step metadata stays cheaper
    than the saved solve-pipeline overhead
- retained measurements against baseline `xoverx_root_pre`:
  - `solve_hotspots_cached/generic/x_over_x`:
    `3.6091-3.7691 us`, improvement `~40.6-44.8%`
  - `solve_hotspots_cached/assume/x_over_x`:
    `3.6259-3.7653 us`, improvement `~40.6-44.5%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `65.976-67.956 us`, small gain but within Criterion noise threshold
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `66.238-66.657 us`, improvement `~3.6-5.1%`
- validation:
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_identical_atom_fraction_uses_root_fast_path --lib`
  - `cargo run -q -p cas_cli -- eval --context solve --domain generic --steps off --format json 'x/x'`
  - `cargo run -q -p cas_cli -- eval --context solve --domain assume --steps off --format json 'x/x'`
  - `cargo test -p cas_solver --test domain_contract_tests test_strict_x_div_x_stays_unchanged -- --exact`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
    - `numeric-only = 168`
- retained a follow-up simplification of that `x/x` root shortcut:
  removed the hidden compact `Step` and relied on structural diagnostics to keep
  the visible `x ≠ 0` requirement
- rationale:
  - once the root shortcut existed, the single hidden `NonZero(x)` step was
    redundant because `build_eval_diagnostics(...)` already derives the same
    condition from the input implicit domain of `x/x`
  - removing the hidden step cuts a little more overhead without changing the
    user-visible contract in `generic` or `assume`
- retained measurements against baseline `xoverx_hidden_step_pre`:
  - `solve_hotspots_cached/generic/x_over_x`:
    `3.3866-3.5506 us`, improvement `~0.7-9.1%` over the previous root shortcut
  - `solve_hotspots_cached/assume/x_over_x`:
    `3.3581-3.5200 us`, improvement `~1.1-10.3%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `65.175-67.370 us`, no statistically significant change
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `65.088-65.663 us`, change within Criterion noise threshold but lower in
    absolute time
- validation:
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_identical_atom_fraction_uses_root_fast_path --lib`
  - `cargo run -q -p cas_cli -- eval --context solve --domain generic --steps off --format json 'x/x'`
  - `cargo run -q -p cas_cli -- eval --context solve --domain assume --steps off --format json 'x/x'`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
- retained a tiny transversal fast path in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/transform_helpers.rs`
  for the exact-shape matchers:
  `multiset_matches_exact(...)` and the binomial-square term classifier now test
  direct `ExprId` equality before falling back to `compare_expr(...)`
- rationale:
  - several root/preorder shortcuts normalize expressions into exactly the same
    node ids they later compare structurally, so paying full `compare_expr(...)`
    on every candidate was wasted work in the common exact-hit case
  - the change is low risk because it only short-circuits the positive branch;
    the structural fallback remains unchanged
- retained measurements against baseline `exact_id_short_pre`:
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `10.212-10.486 us`, improvement `~2.0-5.9%`
  - `solve_hotspots_cached/generic/perfect_square_minus_fraction`:
    no statistically significant change
  - `solve_hotspots_cached/generic/difference_of_squares_fraction`:
    no statistically significant change
  - `solve_hotspots_cached/generic/difference_of_cubes_fraction`:
    no statistically significant change
  - `solve_hotspots_cached/generic/sum_of_cubes_fraction`:
    no statistically significant change
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `65.848-66.399 us`, lower in absolute terms but still within Criterion noise
- validation:
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_binomial_square_uses_exact_preorder_fast_path --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_binomial_square_cancel_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
    - `numeric-only = 168`
- retained a top-level hidden-solve root shortcut for exact same-base power
  fractions in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only with `steps off`
  - only in solve mode
  - only outside `Strict`
  - only with no step listener attached
  - gated by the same definability/solve-safety policy already used by the
    transformer path
  - only for exact raw same-base power fractions handled by
    `try_rewrite_cancel_same_base_powers_div_expr(...)`
- rationale:
  - `power_quotient_fraction` remained one of the largest direct hotspots after
    the root cuts for scalar-multiple, perfect-square, cubes and
    difference-of-squares
  - a pure root bypass had been avoided because `x^4/x^2 -> x^2` carries
    visible `requires`; synthesizing a single compact hidden step at the root
    preserves those `requires` while still skipping eager pre-passes and the
    whole phase pipeline
  - this keeps the visible contract (`x ≠ 0`, `x^2 ≠ 0` in generic; `x^2 ≠ 0`
    in strict because strict still uses the old path) and removes the fixed
    hidden solve overhead around the rule
- retained measurements against baseline `required_smallvec_pre`:
  - `solve_hotspots_cached/generic/power_quotient_fraction`:
    `4.2916-4.4442 us`, improvement `~80.6-81.8%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `77.431-80.493 us`, no statistically significant change
  - `solve_eval_hotspots_cached/generic/power_quotient_fraction` baseline was
    not present in that named snapshot; the absolute post-change run stays at
    `x^2` with the same visible requires in CLI validation
- validation:
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_power_quotient_uses_root_fast_path --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_power_quotient_cancel_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- retained a top-level hidden-solve root shortcut for exact structural
  scalar-multiple fractions in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only with `steps off`
  - only in solve mode
  - only with no step listener attached
  - gated by the same definability/solve-safety policy already used by the
    transform-layer scalar-multiple preorder
  - only before eager pre-passes and phase orchestration
  - uses the existing exact structural planner
    `try_structural_scalar_multiple_preorder(...)`
- rationale:
  - after the exact root cuts for perfect-square, cubes and
    difference-of-squares, reranking put `scalar_multiple_fraction` back among
    the most meaningful direct hotspots
  - the retained transform-layer preorder already proved the planner itself was
    safe and profitable, but the hidden solve pipeline still paid eager
    pre-passes and the fixed phase setup before reaching that rule
  - moving the same exact structural detection to the root of the hidden solve
    path removes that fixed overhead while preserving the same normalized
    result (`1/2`) and denominator requirement
- retained measurements against baseline `top_level_scalar_pre`:
  - `solve_hotspots_cached/generic/scalar_multiple_fraction`:
    `6.8378-7.0891 us`, improvement `~36.3-38.7%`
  - `solve_eval_hotspots_cached/generic/scalar_multiple_fraction`:
    `6.8642-7.1335 us`, improvement `~34.7-37.3%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `78.783-79.957 us`, improvement `~2.1-3.7%`
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_scalar_multiple_fraction_uses_root_fast_path --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_content_gcd_multivar_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_content_gcd_multivar_in_solve_strict_context_steps_off -- --exact`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- retained a top-level hidden-solve root shortcut for exact
  `perfect_square_minus` in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only with `steps off`
  - only in solve mode
  - only outside `Strict`
  - only with no step listener attached
  - only for the exact raw root shape `(a^2 - 2ab + b^2) / (a - b)`
  - only before eager pre-passes and phase orchestration
- rationale:
  - the retained exact preorder already collapsed this hotspot cheaply once it
    reached `Core`, but the layered benches showed the residual time was mostly
    fixed overhead before and around the rule loop
  - moving the same exact recognition to the root of the hidden solve pipeline
    avoids `expand`, `poly_gcd_modp`, `poly_lower`, the full `Core` pass, and
    all late phases for this one exact raw input class
  - the gate stays intentionally narrow and listener-free so it does not widen
    event gaps or add new fixed cost to unrelated `Div` inputs
- retained measurements against baseline `top_level_perfect_square_pre`:
  - `solve_hotspots_cached/generic/perfect_square_minus_fraction`:
    `4.1405-4.3049 us`, improvement `~92.6-93.0%`
  - `solve_eval_hotspots_cached/generic/perfect_square_minus_fraction`:
    `4.1383-4.3260 us`, improvement `~92.3-92.8%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `94.119-95.080 us`, change within noise threshold
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_perfect_square_minus_uses_exact_preorder_fast_path --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests perfect_square_minus -- --nocapture`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- retained matching top-level hidden-solve root shortcuts for exact raw cube
  fractions in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only with `steps off`
  - only in solve mode
  - only with no step listener attached
  - only for the exact raw root shapes `(a^3 - b^3)/(a - b)` and
    `(a^3 + b^3)/(a + b)`
  - only before eager pre-passes and phase orchestration
- rationale:
  - after reranking on the new baseline, `difference_of_cubes_fraction`
    became the clear next direct hotspot at about `~41 us`
  - the retained exact cube preorder already made the `Core` phase cheap, but
    the layered measurements still showed the residual time living mostly in
    fixed solve-path overhead before and around that pass
  - reusing the exact no-steps cube matcher at the root of the hidden solve
    pipeline removes `expand`, `poly_gcd_modp`, `poly_lower`, the full `Core`
    loop, and the late phases for those exact raw inputs, while keeping the
    guard narrow enough to avoid adding fixed cost to unrelated `Div` nodes
- retained measurements against baseline `top_level_cubes_pre`:
  - `solve_hotspots_cached/generic/difference_of_cubes_fraction`:
    `6.1946-6.3797 us`, improvement `~85.2-86.0%`
  - `solve_hotspots_cached/generic/sum_of_cubes_fraction`:
    `5.9704-6.1946 us`, improvement `~72.7-74.1%`
  - `solve_eval_hotspots_cached/generic/difference_of_cubes_fraction`:
    `6.0271-6.2036 us`, improvement `~85.3-86.0%`
  - `solve_eval_hotspots_cached/generic/sum_of_cubes_fraction`:
    `5.9798-6.2408 us`, improvement `~72.3-74.0%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `96.770-97.594 us`, no statistically significant change
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_difference_of_cubes_uses_exact_preorder_fast_path --lib`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_sum_of_cubes_uses_exact_preorder_fast_path --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests difference_of_cubes -- --nocapture`
  - `cargo test -p cas_solver --test multivar_gcd_tests sum_of_cubes -- --nocapture`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- discarded a narrow exact `Pow/Pow` fast path inside
  `/Users/javiergimenezmoya/developer/math/crates/cas_math/src/logarithm_inverse_support.rs`
  for `try_rewrite_log_power_base_numeric_expr(...)`
- rationale:
  - the idea was to short-circuit the dominant exact shape `log(a^m, a^n)` and
    avoid the generic `normalize_to_power(...)` path used for reciprocals and
    mixed forms
  - measured against a named baseline, it regressed both the direct hotspot and
    the representative `generic/assume` solve batches, so it is not retained
- measured result against baseline `log_power_exact_pre`:
  - `solve_hotspots_cached/generic/log_power_base` regressed to
    `24.435-24.915 us`
  - `solve_hotspots_cached/assume/log_power_base` regressed to
    `24.794-25.049 us`
  - `solve_modes_cached/solve_tactic_generic_batch` regressed to
    `96.002-98.137 us`
  - `solve_modes_cached/solve_tactic_assume_batch` regressed to
    `95.810-96.356 us`
- takeaway:
  - the hot path is not dominated by the generic normalize-to-power helper, so
    future work on `log_power_base` should focus on rule/orchestration overhead
    or domain-policy checks, not another local structural split
- retained a top-level hidden-solve root shortcut for exact raw
  `difference_of_squares` fractions in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only with `steps off`
  - only in solve mode
  - only outside `Strict`
  - only with no step listener attached
  - only for the exact raw root shapes `(a^2 - b^2)/(a - b)` and
    `(a^2 - b^2)/(a + b)`
  - only before eager pre-passes and phase orchestration
- rationale:
  - after the cube shortcuts landed, reranking put
    `difference_of_squares_fraction` back near the top at about `~15 us`
  - the retained pre-order rule already made `Core` itself cheap, but the
    residual time still sat in the fixed hidden solve pipeline before and
    around that phase
  - moving the exact raw-shape match to the root of the hidden solve pipeline
    removes `expand`, `poly_gcd_modp`, `poly_lower`, the full `Core` loop, and
    the late phases for this one structurally exact input class
- retained measurements against baseline `top_level_diffsq_pre`:
  - `solve_hotspots_cached/generic/difference_of_squares_fraction`:
    `4.0875-4.2679 us`, improvement `~73.2-75.1%`
  - `solve_eval_hotspots_cached/generic/difference_of_squares_fraction`:
    `4.0930-4.2961 us`, improvement `~72.9-74.5%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `83.404-85.456 us`, improvement `~12.3-14.8%`
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_difference_of_squares_uses_root_fast_path --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_layer2_difference_of_squares_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- retained a narrow post-`Core` hidden-solve cut for power-quotient outputs in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only with `steps off`
  - only in solve mode
  - only outside `Strict`
  - only when the original input root is a fraction
  - only when `Core` has already collapsed that fraction to a plain symbolic
    power `atom^n`
  - only when there are no denominator-root or auto-expand marks
- rationale:
  - the earlier broad “plain symbolic power” post-`Core` cut regressed the
    wider solve workload because it fired on too many native power expressions
  - `power_quotient_fraction` remained one of the direct hotspots, and its
    layered benches showed the rule body was already cheap while the remaining
    cost sat in fixed post-`Core` engine work
  - restricting the cut to fraction-origin power results keeps the useful case
    (`x^4/x^2 -> x^2`) without reopening the broader regression
- retained measurements:
  - `solve_hotspots_cached/generic/power_quotient_fraction`:
    `21.988-22.503 us`, improvement `~46.7-47.9%`
  - `solve_eval_hotspots_cached/generic/power_quotient_fraction`:
    `20.630-20.947 us`, improvement `~21.6-22.9%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `95.575-96.407 us`, improvement `~2.3-5.1%`
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_power_quotient_result_skips_late_phases --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_power_quotient_cancel_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- discarded a structural rewrite of the exact binomial/perfect-square hidden
  matcher in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/transform_helpers.rs`
- rationale:
  - the idea was to avoid interning `a^2` / `b^2` and replace the tiny
    multiset helper with direct structural classifiers (`square-of-a`,
    `square-of-b`, `2ab`)
  - the retained hidden fast path is already narrow; if this change did not
    move the direct hotspot or the batch, the extra code shape was not worth it
- measured result:
  - `solve_hotspots_cached/generic/perfect_square_minus_fraction` stayed within
    noise at `56.457-57.176 us`
  - `solve_hotspots_cached/generic/binomial_square_fraction` stayed within noise
    at `10.089-10.257 us`
  - `solve_modes_cached/solve_tactic_generic_batch` stayed within noise at
    `95.340-96.887 us`
- takeaway:
  - the remaining cost for these hidden binomial paths is not in that tiny
    `a^2`/`b^2` matcher shape, so future work should stay on broader core-loop
    overhead or higher-level pipeline cuts
- discarded two follow-up experiments after measurement:
  - combined pre-scan for `expand` / `poly_gcd_modp` / `poly_result` before the
    three eager pre-passes in
    `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
    - regressed `solve_hotspots_cached/generic/perfect_square_minus_fraction`
      to `57.813-58.331 us`
    - regressed `solve_eval_hotspots_cached/generic/perfect_square_minus_fraction`
      to `53.491-54.394 us`
    - regressed `solve_hotspots_cached/generic/power_quotient_fraction`
      to `42.777-44.109 us`
    - regressed `solve_modes_cached/solve_tactic_generic_batch`
      to `95.217-95.725 us`
    - takeaway: one generic scan is more expensive than the mostly-no-op eager
      passes on the current solve workload
  - widening the exact `perfect_square_minus` preorder to accept raw
    `a + (-b)` denominators and normalize them to `a - b`
    - left the direct hotspot within noise (`57.865-59.678 us`,
      `54.210-54.862 us`)
    - regressed `solve_modes_cached/solve_tactic_generic_batch`
      to `99.704-102.34 us`
    - takeaway: the extra branch work on every candidate `Div` costs more than
      it saves on the current batch
- extended `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`
  so `power_quotient_fraction` is now covered in both `fraction_rule_direct/*`
  and `solve_phase_subset_cached/*`
- rationale:
  - `power_quotient_fraction` became one of the next direct solve hotspots, but
    it lacked the same layered visibility already available for scalar-multiple,
    cubes, and binomial cases
  - before opening another runtime fast path, the missing piece was knowing
    whether the cost sat in `SimplifyFractionRule::apply(...)`, in the minimal
    single-rule engine, or in the surrounding solve pipeline
- retained measurements:
  - `fraction_rule_direct/apply/generic/power_quotient_fraction`:
    `3.6583-3.8001 us`
  - `fraction_rule_direct/single_rule_engine/generic/power_quotient_fraction`:
    `12.618-12.806 us`
  - `solve_phase_subset_cached/power_quotient_fraction/generic/full`:
    `38.914-39.296 us`
  - `solve_phase_subset_cached/power_quotient_fraction/generic/no_transform`:
    `36.161-36.829 us`
  - `solve_phase_subset_cached/power_quotient_fraction/generic/no_transform_no_rationalize`:
    `35.699-36.185 us`
- takeaway:
  - the direct rule body is already cheap; the remaining cost is broader engine
    work around the rule, not just the power-cancel planner itself
  - a follow-up post-`Core` cut for plain symbolic powers was tried and
    discarded because it regressed the broader hotspot set, so it is not
    retained
- validation:
  - `cargo check -p cas_engine --benches`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_power_quotient_cancel_in_solve_generic_context_steps_off_keeps_requires -- --exact`
- extended `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`
  so `perfect_square_minus_fraction` is now covered in both
  `fraction_rule_direct/*` and `solve_phase_subset_cached/*`
- rationale:
  - after the exact preorder and the narrow post-`Transform` cut, the direct
    hotspot still sat around `~50-56 us`, but it was no longer clear whether
    the remaining cost lived in `SimplifyFractionRule`, in a minimal one-rule
    engine, or in the surrounding solve pipeline
  - adding the same layered visibility used for `power_quotient_fraction`
    closes that gap and makes the next optimization target much clearer
- retained measurements:
  - `fraction_rule_direct/apply/generic/perfect_square_minus_fraction`:
    `24.721-25.063 us`
  - `fraction_rule_direct/single_rule_engine/generic/perfect_square_minus_fraction`:
    `11.523-11.800 us`
  - `solve_phase_subset_cached/perfect_square_minus_fraction/generic/full`:
    `56.141-56.880 us`
  - `solve_phase_subset_cached/perfect_square_minus_fraction/generic/no_transform`:
    `54.285-54.754 us`
  - `solve_phase_subset_cached/perfect_square_minus_fraction/generic/no_transform_no_rationalize`:
    `53.665-54.121 us`
- takeaway:
  - the residual hotspot is not dominated by the rule body itself, and
    `Transform`/`Rationalize` only account for a very small tail
  - the next ROI is therefore not another exact matcher for this case, but
    lower-level core-loop/traversal overhead or broader solve-path fixed costs
- validation:
  - `cargo fmt --all`
  - `cargo check -p cas_engine --benches`
- discarded an exact `Pow/Pow` fast path inside `LogPowerBaseRule.apply(...)`
  in `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/logarithms/inverse.rs`
- rationale:
  - the idea was to bypass `normalize_to_power(...)` only for the hot exact
    case `log(x^2, x^6)` while preserving the same domain-policy and `requires`
    handling inside the rule
  - measured against a named baseline, the narrower in-rule split still
    regressed the generic hotspot and did not move the representative solve
    batches enough to justify the extra branch
- measured result against baseline `log_power_apply_exact_pre`:
  - `solve_hotspots_cached/generic/log_power_base` regressed to
    `24.815-25.779 us`
  - `solve_hotspots_cached/assume/log_power_base` stayed effectively flat at
    `24.623-25.083 us`
  - `solve_modes_cached/solve_tactic_generic_batch` moved to
    `78.183-78.677 us` (noise / slightly worse absolute)
  - `solve_modes_cached/solve_tactic_assume_batch` stayed within noise at
    `78.123-79.949 us`
- takeaway:
  - `log_power_base` is not paying enough in `normalize_to_power(...)` alone to
    justify another local structural split in the rule body
  - future work here should target broader orchestration or policy/proof cost,
    not another exact-shape micro-fast-path
- discarded changing `required_conditions` from `Vec` to `SmallVec<[...; 2]>`
  in `Rewrite`, `ChainedRewrite` and `StepMeta`
- rationale:
  - most rewrites emit `0-2` required conditions, so replacing tiny heap-backed
    vectors with inline storage looked like a plausible transversal win across
    `power_quotient_fraction`, `log_power_base` and the fraction planners
  - measured against a named baseline, the effect stayed inside noise or mixed
    directions depending on the hotspot, so it does not justify widening that
    type change through the shared runtime model
- measured result against baseline `required_smallvec_pre`:
  - `solve_hotspots_cached/generic/power_quotient_fraction`:
    `22.145-22.512 us` (no statistically significant change)
  - `solve_hotspots_cached/generic/log_power_base`:
    `24.521-24.854 us` (no statistically significant change)
  - `solve_hotspots_cached/generic/binomial_square_fraction`:
    `10.329-10.628 us` (no statistically significant change)
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `78.981-79.879 us` (no statistically significant change)
- takeaway:
  - if there is any real gain here, it is smaller than the current benchmark
    noise floor
  - future transversal work should target costs with clearer signatures than
    tiny `required_conditions` storage
- retained a root hidden-solve shortcut for the exact `exp(ln(x)) -> x` case in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only with `steps off`
  - only in solve mode
  - only with no step listener attached
  - only outside `Strict`
  - only when the existing matcher already rewrites directly to a symbolic atom
- rationale:
  - after the latest reranking, `generic/exp_ln_x` was still a stable direct
    hotspot at about `5.985-6.217 us`
  - the hidden shortcut already existed inside the `Core` traversal, but that
    still paid prepasses and pipeline setup before reaching the root `Pow`
  - moving the same narrow rewrite to the pipeline root keeps the `x > 0`
    contract through structural diagnostics and skips all hidden pipeline work
- retained measurements:
  - `solve_hotspots_cached/generic/exp_ln_x`:
    `3.3241-3.4920 us`, improvement `~43.2-47.3%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `64.022-64.602 us`, improvement `~7.6-9.2%`
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_exp_ln_atom_uses_root_fast_path --lib`
  - `cargo test -p cas_solver --test domain_contract_tests exp_ln_x_generic_emits_positive_require -- --exact`
  - `cargo test -p cas_solver --test domain_contract_tests test_strict_exp_ln_x_stays_unchanged -- --exact`
  - `cargo run -q -p cas_cli -- eval --context solve --domain generic --steps off --format json 'exp(ln(x))'`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- retained a narrower root hidden-solve shortcut for the exact two-term scalar
  multiple fraction shape in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only with `steps off`
  - only in solve mode
  - only with no step listener attached
  - only on the exact raw `Add/Add` two-term shape, before the existing
    structural scalar-multiple planner
  - keeps falling back to the previous generic root planner for all other
    additive scalar-multiple cases
- rationale:
  - after the latest reranking, `scalar_multiple_fraction` was back on top at
    about `6.901-7.156 us`
  - the existing root shortcut still paid the full structural scalar-multiple
    planner, and earlier attempts to speed up that planner globally had already
    regressed unrelated `Div` paths
  - a cheaper exact raw matcher for just the hot two-term case avoids widening
    that regression surface while preserving the same generic `requires`
    contract (`4*x + 4*y != 0`)
- retained measurements:
  - `solve_hotspots_cached/generic/scalar_multiple_fraction`:
    `5.6568-5.8489 us`, about `16-20%` better vs the immediate pre-change rerank
    (`6.9010-7.1560 us`)
  - `solve_eval_hotspots_cached/generic/scalar_multiple_fraction`:
    `5.7247-5.9199 us`, about `17-19%` better vs the immediate pre-change rerank
    (`6.8642-7.1335 us`)
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `63.758-64.926 us`, lower in absolute terms but still within Criterion
    noise
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_scalar_multiple_fraction_uses_root_fast_path --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_content_gcd_multivar_in_solve_generic_context_steps_off_keeps_requires -- --exact`
  - `cargo run -q -p cas_cli -- eval --context solve --domain generic --steps off --format json '(2*x + 2*y)/(4*x + 4*y)'`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- retained a narrower hidden-solve root shortcut path for `log_power_base` in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only on the existing root shortcut path for `log(a^m, a^n)`
  - if the policy still needs `Positive(base)`, keep the old hidden-step path
  - if the policy only needs the structural `base - 1 != 0` guard, return the
    rewritten result directly and let `diagnostics` infer that require from the
    input shape
- rationale:
  - for the generic hot case `log(x^2, x^6)`, the hidden step had become
    redundant: the output already keeps `x^2 - 1 != 0` through structural
    diagnostics, so building a hidden `Step` and `Vec` was unnecessary runtime
    churn
  - this keeps the `assume` path unchanged, because it still needs the explicit
    `x > 0` requirement and therefore still goes through the old hidden-step
    branch
- retained measurements:
  - `solve_hotspots_cached/generic/log_power_base`:
    `5.4733-5.6699 us`, lower in absolute terms than the immediate pre-change
    rerank (`5.6729-5.8476 us`) but still within Criterion noise
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `63.109-63.754 us`, also lower in absolute terms but still within noise
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_log_power_base_uses_root_fast_path --lib`
  - `cargo test -p cas_solver --test root_log_parity_tests log_power_base::log_x2_x6_gives_3 -- --exact`
  - `cargo run -q -p cas_cli -- eval --context solve --domain generic --steps off --format json 'log(x^2, x^6)'`
  - `cargo run -q -p cas_cli -- eval --context solve --domain assume --steps off --format json 'log(x^2, x^6)'`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- retained a smaller hidden-step payload for the `assume/log_power_base` root
  shortcut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only the hidden solve root shortcut for `log(a^m, a^n)` in `Assume`
  - when a step is still needed to preserve `Positive(base_core)`, stop also
    attaching `NonZero(base_expr - 1)` there
- rationale:
  - `x^2 - 1 != 0` is already preserved by diagnostics on this path; carrying it
    again in the hidden `Step` was redundant metadata churn
  - the shortcut still needs `x > 0` in `Assume`, so this is a narrow cleanup,
    not a semantic change
- retained measurements:
  - `solve_hotspots_cached/assume/log_power_base`:
    `5.5463-5.7169 us`, lower in absolute terms than the earlier rerank
    (`5.7233-5.8579 us`) but still within Criterion noise
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `60.714-61.068 us`, lower in absolute terms than the regressive run and back
    inside the prior band
- validation:
  - `cargo fmt --all`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_log_power_base_uses_root_fast_path --lib`
  - `cargo test -p cas_solver --test root_log_parity_tests log_power_base -- --nocapture`
  - `cargo run -q -p cas_cli -- eval --context solve --domain assume --steps off --format json 'log(x^2, x^6)'`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- retained a narrow post-`Transform` hidden-solve cut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- scope:
  - only with `steps off`
  - only in solve mode
  - only when `Transform` actually changed the expression
  - only when the new result is a plain symbolic binomial
  - only when the pre-scan still proves there are no denominator roots and no
    auto-expand contexts
- rationale:
  - the exact perfect-square-minus preorder removed the heavy matcher cost, but
    the end-to-end `solve_eval` path still paid a small fixed late-phase tail
  - a broader post-`Transform` cut regressed other paths; keeping the gate to
    “transform changed into a plain binomial” preserves the useful part without
    reintroducing that fixed overhead everywhere
- retained measurements:
  - `solve_hotspots_cached/generic/perfect_square_minus_fraction`:
    `52.608-55.013 us`, no statistically significant change
  - `solve_eval_hotspots_cached/generic/perfect_square_minus_fraction`:
    `48.626-49.429 us`, improvement `~2.6-5.8%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `88.809-90.665 us`, change within Criterion noise threshold
- validation:
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_perfect_square_minus_cancel_in_solve_generic_context_steps_off -- --exact`
  - `cargo test -p cas_solver --test multivar_gcd_tests test_perfect_square_minus_cancel_in_solve_strict_context_steps_off -- --exact`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math`
- retained a stage-breakdown REPL benchmark in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/repl_end_to_end.rs`
  plus a convenience target in `/Users/javiergimenezmoya/developer/math/Makefile`
- scope:
  - new `repl_stage_breakdown/*` group
  - splits representative REPL cases into `parse`, `simplify`, and `format`
  - cases tracked:
    - `heavy/nested_root`
    - `heavy/abs_square`
    - `complex/gaussian_div`
  - make target:
    - `make bench-engine-repl-breakdown`
- rationale:
  - the recent REPL experiments were still too blind: `repl_full_eval` mixed parse,
    simplify, and format, so it was easy to overfit a local shortcut without
    knowing where the remaining latency actually lived
  - the split makes the next ROI obvious and avoids more no-op work on parse/render
- retained measurements:
  - `repl_stage_breakdown/parse/heavy/nested_root`:
    `5.7482-5.8977 us`
  - `repl_stage_breakdown/simplify/heavy/nested_root`:
    `126.94-128.62 us`
  - `repl_stage_breakdown/format/heavy/nested_root`:
    `906.37-915.24 ns`
  - `repl_stage_breakdown/parse/heavy/abs_square`:
    `14.286-14.547 us`
  - `repl_stage_breakdown/simplify/heavy/abs_square`:
    `180.32-181.62 us`
  - `repl_stage_breakdown/format/heavy/abs_square`:
    `825.15-862.88 ns`
  - `repl_stage_breakdown/parse/complex/gaussian_div`:
    `7.1727-7.3482 us`
  - `repl_stage_breakdown/simplify/complex/gaussian_div`:
    `138.60-140.62 us`
  - `repl_stage_breakdown/format/complex/gaussian_div`:
    `1.1641-1.1979 us`
- conclusion:
  - parse and format are negligible on the current REPL hotspots
  - the remaining work is overwhelmingly in `simplify`, especially
    `abs_square`, `gaussian_div`, and `nested_root`
- validation:
  - `cargo fmt --all`
  - `cargo check -p cas_engine --benches`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_stage_breakdown/(parse|simplify|format)/(heavy/nested_root|heavy/abs_square|complex/gaussian_div)' -- --noplot`
- corrected a benchmark wiring bug in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/repl_end_to_end.rs`
- scope:
  - `full_eval(...)`, `full_eval_with_mode(...)`, `simplify_only(...)`, and
    `formatted_result_from_eval(...)` now call
    `simplify_with_options(opts.to_simplify_options())`
  - previously they built the profile from `EvalOptions`, but then called bare
    `simplify(...)`, so parts of the REPL bench were not actually respecting the
    intended `ContextMode` / semantic configuration
- rationale:
  - this was inflating or distorting several REPL numbers, especially the
    standard real-domain complex no-op cases
  - the retained standard shortcut for `i^n` / Gaussian division was already
    correct in runtime, but the bench was not measuring it honestly
- retained measurements after the fix:
  - `repl_stage_breakdown/simplify/heavy/nested_root`:
    `128.25-129.28 us`
  - `repl_stage_breakdown/simplify/heavy/abs_square`:
    `179.66-181.56 us`
  - `repl_stage_breakdown/simplify/complex/gaussian_div`:
    `3.2787-3.4919 us`
  - `repl_full_eval/cached/batch_11_inputs`:
    `1.0386-1.0717 ms`, improvement `~15-18%`
  - `repl_individual/cached/05_i^5`:
    `7.5635-7.6846 us`, improvement `~70%`
- conclusion:
  - the previous `~140 us` reading for standard `gaussian_div` was a bench bug,
    not a real runtime regression
  - after the fix, the dominant remaining REPL hotspots are `abs_square` and
    `nested_root`
- validation:
  - `cargo fmt --all`
  - `cargo check -p cas_engine --benches`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_stage_breakdown/simplify/(heavy/nested_root|heavy/abs_square|complex/gaussian_div)' -- --noplot`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
- retained a standard phase-subset benchmark in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`
  plus a convenience target in `/Users/javiergimenezmoya/developer/math/Makefile`
- scope:
  - new `standard_phase_subset_cached/*` group
  - tracks `standard/full`, `standard/no_transform`, and
    `standard/no_transform_no_rationalize` for:
    - `heavy/nested_root`
    - `heavy/abs_square`
    - `complex/gaussian_div`
    - `complex/i_power`
  - make target:
    - `make bench-engine-standard-phase-subset`
- retained measurements:
  - `heavy/nested_root/standard/full`:
    `128.98-129.84 us`
  - `heavy/nested_root/standard/no_transform`:
    `118.70-120.39 us`
  - `heavy/nested_root/standard/no_transform_no_rationalize`:
    `119.78-121.37 us`
  - `heavy/abs_square/standard/full`:
    `180.64-182.01 us`
  - `heavy/abs_square/standard/no_transform`:
    `169.40-171.74 us`
  - `heavy/abs_square/standard/no_transform_no_rationalize`:
    `174.05-175.66 us`
  - `complex/gaussian_div/standard/full`:
    `3.2325-3.3878 us`
  - `complex/gaussian_div/standard/no_transform`:
    `3.2473-3.4315 us`
  - `complex/gaussian_div/standard/no_transform_no_rationalize`:
    `3.2963-3.4704 us`
- conclusion:
  - `gaussian_div` is no longer a meaningful standard-path target
  - `nested_root` and `abs_square` still have a real late-phase tail after
    `Core`, so the next REPL ROI is more likely a narrow post-`Core` cut than
    another root matcher
- validation:
  - `cargo fmt --all`
  - `cargo check -p cas_engine --benches`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'standard_phase_subset_cached/(heavy/nested_root|heavy/abs_square|complex/gaussian_div)/(standard/full|standard/no_transform|standard/no_transform_no_rationalize)' -- --noplot`
- discarded a cheaper boolean matcher plus a `simplify_with_stats(...)` early exit
  for standard real-domain Gaussian no-op roots
  in `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
  and `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/orchestration.rs`
- rationale:
  - the cheaper Gaussian matcher alone did not move `gaussian_div` or the REPL batch
    materially
  - lifting the shortcut above `Orchestrator` looked promising in theory, but the
    measurements stayed flat or worse in the real REPL path
- measured outcome before revert:
  - `repl_stage_breakdown/simplify/complex/gaussian_div`:
    `136.87-139.80 us` (matcher-only), then `140.90-142.67 us` (lifted shortcut)
  - `repl_full_eval/cached/batch_11_inputs`:
    `1.2104-1.2372 ms` (matcher-only), then `1.2608-1.2815 ms` (lifted shortcut)
- discarded a narrow standard post-`Core` cut for affine
  `sqrt(square) -> abs(...)` in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- rationale:
  - the hidden standard path still needed one `Transform` iteration for the retained
    `Root Power Cancel`, so the post-`Core` gate did not actually fire where intended
  - the measured path regressed even in the best-case target `abs_square`, so there
    was no case for keeping extra shape logic in `Orchestrator`
- measured outcome before revert:
  - `standard_phase_subset_cached/heavy/abs_square/standard/full`:
    `198.13-203.97 us`, regression `~9.5-13.2%`
  - `standard_phase_subset_cached/heavy/abs_square/standard/no_transform`:
    `180.97-182.82 us`, regression `~5.1-8.2%`
  - `repl_stage_breakdown/simplify/heavy/abs_square`:
    `193.69-198.28 us`, regression `~7.0-9.9%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `1.0886-1.1172 ms`, regression `~2.6-7.4%`
- discarded a standard post-`Core` cut for `numeric * sqrt(...)` outputs after
  `Extract Perfect Square from Radicand` in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- rationale:
  - the shape looked like a clean `nested_root` candidate, but the retained `Core`
    rewrite still left enough bookkeeping that the extra gate cost more than the
    late-phase tail it was trying to remove
  - the regression showed up both in the focused `nested_root` simplify bench and
    in the phase-subset breakdown, so there was no reason to keep the shortcut
- measured outcome before revert:
  - `standard_phase_subset_cached/heavy/nested_root/standard/full`:
    `136.09-138.32 us`, regression `~2.7-7.2%`
  - `standard_phase_subset_cached/heavy/nested_root/standard/no_transform`:
    `125.59-128.30 us`, regression `~2.1-5.0%`
  - `repl_stage_breakdown/simplify/heavy/nested_root`:
    `134.56-135.94 us`, regression `~4.2-6.5%`
- discarded an early return inside the `sqrt(square)` preorder in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/transform_helpers.rs`
  that skipped recursive transform when the generated `abs(...)` already looked canonical
- rationale:
  - the local `abs_square` subset looked roughly flat, but the full REPL batch regressed
    clearly, so the shortcut was not robust enough to keep
  - `abs(e^x)` remained semantically correct, but that was not enough to justify the
    batch regression
- measured outcome before revert:
  - `standard_phase_subset_cached/heavy/abs_square/standard/full`:
    `194.59-198.73 us`, only noise-level improvement
  - `repl_stage_breakdown/simplify/heavy/abs_square`:
    `196.47-200.66 us`, no meaningful improvement
  - `repl_full_eval/cached/batch_11_inputs`:
    `1.0928-1.1191 ms`, regression `~7.5-10.8%`
- retained tooling:
  - new Make target:
    - `make bench-engine-repl-individual`
  - purpose:
    - rerank the actual 11 cached standard REPL inputs with one command before
      attempting another micro-optimization
- current rerank snapshot from `repl_individual/cached/*`:
    - `08_((x+y)*(a+b))/((x+y)*(c+d))`: `157.21-161.11 us`
    - `04_((5*x + 8)^2)^(1/2)`: `153.84-155.45 us`
    - `03_sqrt(12*x^3)`: `137.86-139.61 us`
    - `06_(2*x + 2*y)/(4*x + 4*y)`: `127.45-128.94 us`
    - `10_sin(2*x + 1)^2 + cos(1 + 2*x)^2`: `80.374-81.163 us`
- retained tooling:
  - new Make target:
    - `make bench-engine-repl-hotspots`
  - purpose:
    - rerun just the current top-5 cached standard REPL hotspots instead of the full
      11-input rerank when checking whether a candidate change is worth a deeper run
- discarded a canonical-form early return after
  `try_exact_common_factor_mul_fraction_preorder(...)` in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine/transform/transform_helpers.rs`
- rationale:
  - the explicit common-factor fraction looked like the strongest standard-path
    candidate, but skipping the recursive transform on the canonical `(a+b)/(c+d)`
    result regressed the direct hotspot and did not move the batch
  - the visible contract stayed correct (`1` step, same `required_conditions`), so
    the regression was pure performance overhead from the extra canonical check
- measured outcome before revert:
  - `repl_individual/cached/08_((x+y)*(a+b))/((x+y)*(c+d))`:
    `162.55-165.80 us`, regression `~2.3-4.6%`
  - `simplify_cached_vs_uncached/cached/gcd/layer25_multiparam`:
    `150.57-154.24 us`, no meaningful change
  - `repl_full_eval/cached/batch_11_inputs`:
    `1.0734-1.1029 ms`, no meaningful improvement
- retained a standard root shortcut for exact common-factor `Mul/Mul` fractions in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- rationale:
  - the standard REPL leader `((x+y)*(a+b))/((x+y)*(c+d))` was still paying the full
    pipeline even though the common factor was already explicit at the root
  - reusing `try_exact_common_factor_mul_fraction_preorder(...)` at the root keeps
    the same visible contract (`1` step, same `required_conditions`) while removing
    almost all orchestration cost from that path
  - the gain shows up where it matters: the real REPL batch, not just a synthetic
    cached gcd microbench
- retained validation:
  - focused checks:
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_common_factor_fraction_uses_root_fast_path --lib`
    - `cargo test -p cas_solver --test anti_catastrophe_tests test_cancel_vs_gcd_structural_cancel_multiparam -- --exact`
  - smoke CLI:
    - `cargo run -q -p cas_cli -- eval '((x+y)*(a+b))/((x+y)*(c+d))' --steps on --format json`
    - output stays `(a + b) / (c + d)` with `1` step (`Pre-order Common Factor Cancel`)
      and the same `required_conditions`
- measured outcome:
  - `repl_individual/cached/08_((x+y)*(a+b))/((x+y)*(c+d))`:
    `15.119-15.281 us`, improvement `~2.7-4.8%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `862.02-887.60 us`, improvement `~4.4-7.8%`
  - `simplify_cached_vs_uncached/cached/gcd/layer25_multiparam`:
    `150.31-152.84 us`, no meaningful change
- retained a standard root shortcut for exact additive scalar-multiple fractions in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`,
  backed by a new preorder helper in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra/mod.rs`
- rationale:
  - the standard REPL hotspot `(2*x + 2*y)/(4*x + 4*y)` was still paying the full
    pipeline even though the structural scalar-multiple planner already knew the
    final answer at the root
  - the retained version keeps the visible standard-path contract intact:
    result still `1/2`, still `2` `Simplify Nested Fraction` steps, and the same
    `required_conditions` in both `generic` and `strict`
  - the win is large enough to move the real cached REPL batch materially
- retained validation:
  - focused check:
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_scalar_multiple_fraction_uses_root_fast_path --lib`
  - smoke CLI:
    - `cargo run -q -p cas_cli -- eval '(2*x + 2*y)/(4*x + 4*y)' --steps on --format json`
    - `cargo run -q -p cas_cli -- eval '(2*x + 2*y)/(4*x + 4*y)' --steps on --format json --domain strict`
    - outputs stay `1/2` with `2` steps and `4·x + 4·y ≠ 0`
- measured outcome:
  - `repl_individual/cached/06_(2*x + 2*y)/(4*x + 4*y)`:
    `21.473-22.287 us`, improvement `~83.2-83.6%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `780.80-812.50 us`, improvement `~9.4-12.6%`
- retained a standard root shortcut for the exact chain identity
  `sin²(t) + cos²(t) -> 1` in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- rationale:
  - the standard REPL hotspot `sin(2*x + 1)^2 + cos(1 + 2*x)^2` still paid the full
    pipeline even though the root already matched `Pythagorean Chain Identity`
  - lifting the existing `try_rewrite_pythagorean_chain_add_expr(...)` helper to the
    root preserves the visible contract (`1` step, same rule name, no new requires)
    while removing almost all orchestration cost from that path
  - unlike the discarded `sqrt(square)` post-`Core` cuts, this shortcut is exact and
    already returns the final canonical result directly
- retained validation:
  - focused check:
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_pythagorean_chain_uses_root_fast_path --lib`
  - smoke CLI:
    - `cargo run -q -p cas_cli -- eval 'sin(2*x + 1)^2 + cos(1 + 2*x)^2' --steps on --format json`
    - output stays `1` with `1` step (`Pythagorean Chain Identity`) and no `requires`
- measured outcome:
  - `repl_individual/cached/10_sin(2*x + 1)^2 + cos(1 + 2*x)^2`:
    `14.714-14.872 us`, improvement `~81.3-81.6%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `611.61-616.76 us`, improvement `~20.2-22.9%`
- rejected a standard root shortcut for `sqrt(12*x^3)` in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- rationale:
  - the shape looked like a good fit for reusing
    `try_rewrite_extract_perfect_power_from_radicand_expr(...)` at the root,
    limited to `steps off` and radicands of the form `Mul(Number, rest)`
  - in measurement, the shortcut did skip the phase pipeline, but it still lost
    against the retained tree on both the focused `nested_root` bench and the
    full cached REPL batch, so it was reverted
- measured outcome before revert:
  - `repl_individual/cached/03_sqrt(12*x^3)`:
    `146.97-148.49 us`, regression `~5.2-7.0%`
  - `repl_stage_breakdown/simplify/heavy/nested_root`:
    `139.32-141.62 us`, regression `~1.5-3.5%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `665.40-676.92 us`, regression `~8.0-10.6%`
- retained tooling:
  - `make bench-engine-repl-hotspots-save BASELINE=good`
  - `make bench-engine-repl-hotspots-compare BASELINE=good`
  - `make bench-engine-repl-individual-save BASELINE=good`
  - `make bench-engine-repl-individual-compare BASELINE=good`
  - these run the current top-5 cached standard REPL hotspots under one named
    Criterion baseline, or the full 11-input rerank under one named baseline,
    so we can accept or reject future REPL shortcuts against a stable reference
    instead of noisy reruns
- rejected a micro-optimization in
  `/Users/javiergimenezmoya/developer/math/crates/cas_math/src/pow_preorder_support.rs`
- rationale:
  - swapping `ctx.call(\"abs\", ...)` for `ctx.call_builtin(BuiltinFn::Abs, ...)`
    looked like a free win in the `sqrt(square)` planner, but the measured path
    regressed both the focused `abs_square` input and the full cached REPL batch
  - this confirms again that `04_((5*x + 8)^2)^(1/2)` is not responding well to
    tiny local tweaks inside the planner; the next win there will need a more
    structural hypothesis
- measured outcome before revert:
  - `repl_individual/cached/04_((5*x + 8)^2)^(1/2)`:
    `161.89-166.00 us`, regression `~1.4-3.9%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `647.41-672.65 us`, regression `~1.5-4.8%`
- rejected a shape-split of the standard root dispatcher in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- rationale:
  - routing standard `Div` roots by denominator shape (`Mul` -> common-factor,
    `Add` -> scalar-multiple) looked like a safe cross-cutting way to skip dead
    matcher calls
  - in measurement the effect was not robust: the focused compare regressed
    `04_((5*x + 8)^2)^(1/2)` and left the rest of the REPL hotspot suite flat,
    so the change was reverted
- measured outcome before revert:
  - `repl_individual/cached/04_((5*x + 8)^2)^(1/2)`:
    `157.87-161.35 us`, regression `~3.0-6.0%`
  - `repl_individual/cached/03_sqrt(12*x^3)`:
    within noise
  - `repl_individual/cached/06_(2*x + 2*y)/(4*x + 4*y)`:
    within noise
  - `repl_individual/cached/08_((x+y)*(a+b))/((x+y)*(c+d))`:
    within noise
- retained tooling:
  - `make bench-engine-root-direct`
  - new `root_rule_direct/*` benches in
    `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`
    to isolate direct rule cost for the two remaining heavy standard root cases
- rationale:
  - `04_((5*x + 8)^2)^(1/2)` is handled by `Root Power Cancel`
  - `03_sqrt(12*x^3)` is handled by `Extract Perfect Square from Radicand`
  - after several failed shortcut attempts, the missing information was whether
    the expensive part was the rule body itself or the engine/traversal around it
  - the new direct benches separate `apply(rule)` from a `single_rule_engine`
    simplifier with only that rule enabled, which is enough to tell whether the
    next win should be inside the rule or in shared orchestration
- measured outcome:
  - `root_rule_direct/apply/standard/root_power_cancel_abs_square`:
    `3.3281-3.5202 us`
  - `root_rule_direct/single_rule_engine/standard/root_power_cancel_abs_square`:
    `16.387-16.582 us`
  - `root_rule_direct/apply/standard/extract_perfect_square_nested_root`:
    `3.2607-3.3582 us`
  - `root_rule_direct/single_rule_engine/standard/extract_perfect_square_nested_root`:
    `9.9511-10.107 us`
  - compared to the current standard path subset:
    - `nested_root standard/full`: `129.58-134.38 us`
    - `abs_square standard/full`: `181.21-187.50 us`
- conclusion:
  - both hotspots are dominated by engine/traversal cost, not by the root rule
    body itself
  - that makes more local rule-level tweaks a low-ROI path for `03/04`; the next
    profitable work should target shared overhead in the standard pipeline, not
    another micro-optimization inside `Root Power Cancel` or radicand extraction
- retained a standard root shortcut for exact square-power cancellation in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- rationale:
  - the real REPL leader `04_((5*x + 8)^2)^(1/2)` is a raw `Pow` root, not the
    older affine-product variant used in the phase subset bench
  - calling `RootPowCancelRule` directly at the root in standard mode is cheap
    enough to bypass the full pipeline, while preserving the visible step
    contract (`Root Power Cancel`) and final result `|5*x + 8|`
  - the parallel attempt to do the same for `sqrt(12*x^3)` was removed again;
    the retained win is only the `RootPowCancelRule` shortcut
- retained validation:
  - focused test:
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_root_power_cancel_uses_root_fast_path --lib`
  - smoke CLI:
    - `cargo run -q -p cas_cli -- eval '((5*x + 8)^2)^(1/2)' --steps on --format json`
- measured outcome:
  - `repl_individual/cached/04_((5*x + 8)^2)^(1/2)`:
    `15.511-15.641 us`, improvement `~89.8-90.1%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `476.63-500.89 us`, improvement `~19.6-23.3%`
- retained a standard root shortcut for exact square-root extraction in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- rationale:
  - the remaining REPL leader `03_sqrt(12*x^3)` stays as a root `Function(sqrt, ...)`
    long enough that the existing `Pow`-based rule path still pays the standard
    engine loop around an otherwise cheap rewrite
  - the retained shortcut normalizes `sqrt(...)` internally just enough to reuse
    `try_rewrite_extract_perfect_power_from_radicand_expr(...)`, but still emits
    exactly the visible contract the CLI already had: one step
    `Extract Perfect Square from Radicand` and the same `requires`
  - unlike the earlier rejected nested-root shortcut, this path is tightly scoped
    to the exact `sqrt` root case and preserves the user-facing trace directly
- retained validation:
  - focused test:
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_extract_perfect_square_uses_root_fast_path --lib`
  - smoke CLI:
    - `cargo run -q -p cas_cli -- eval 'sqrt(12*x^3)' --steps on --format json`
- measured outcome:
  - `repl_individual/cached/03_sqrt(12*x^3)`:
    `14.121-14.221 us`, improvement `~90.2-90.3%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `334.94-340.14 us`, improvement `~28.7-31.2%`
- retained a transversal step-model optimization in
  `/Users/javiergimenezmoya/developer/math/crates/cas_solver_core/src/step_model.rs`
- rationale:
  - `Step::new(...)` was still materializing `after_str` eagerly through
    `DisplayExpr`, even though the current repo barely consumes that cache
    directly and the standard REPL bench ignores the step list entirely
  - this was pure overhead on every `steps on` path, especially in the standard
    cached REPL benchmark where many retained fast paths now spend more time
    constructing `Step` metadata than rewriting the expression itself
  - keeping the API stable and simply leaving `after_str` empty by default is
    enough to remove that formatting cost without changing the visible
    expression/result contracts
- retained validation:
  - `cargo test -p cas_engine step_tests --lib`
  - `cargo test -p cas_didactic --test step_wire_tests`
  - `cargo test -p cas_solver --test wire_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
- measured outcome:
  - `repl_full_eval/cached/batch_11_inputs`:
    first rerun was noisy, but a second isolated rerun settled at
    `265.56-266.82 us`, improvement `~2.0-3.3%`
  - `repl_individual/cached/03_sqrt(12*x^3)`:
    `13.993-14.180 us`, improvement `~4.3-5.2%`
  - `repl_individual/cached/04_((5*x + 8)^2)^(1/2)`:
    `14.751-14.893 us`, improvement `~4.0-4.9%`
  - `repl_individual/cached/06_(2*x + 2*y)/(4*x + 4*y)`:
    `18.545-18.735 us`, improvement `~10.8-11.8%`
  - `repl_individual/cached/07_(x^2 - y^2)/(x - y)`:
    `12.404-12.509 us`, improvement `~2.8-3.7%`
  - `repl_individual/cached/08_((x+y)*(a+b))/((x+y)*(c+d))`:
    `14.732-14.800 us`, improvement `~2.2-2.9%`
- updated rerank snapshot:
  - current standard cached REPL leaders are no longer the algebraic shortcuts
    that were previously dominating
  - top absolute inputs are now:
    - `00_x + 1`: `42.216-42.467 us`
    - `01_2 * 3 + 4`: `38.767-39.013 us`
    - `06_(2*x + 2*y)/(4*x + 4*y)`: `18.545-18.735 us`
    - `04_((5*x + 8)^2)^(1/2)`: `14.751-14.893 us`
    - `08_((x+y)*(a+b))/((x+y)*(c+d))`: `14.732-14.800 us`
- conclusion:
  - the next REPL-standard ROI is no longer another local algebraic shortcut
  - the dominant cost has shifted toward fixed overhead on very light inputs,
    so the next profitable work should be a transversal reduction in the
    standard path (dispatcher / phase setup / parse-facing overhead), not
    another case-by-case rewrite
- retained a standard root no-op shortcut for simple symbolic additive forms in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- rationale:
  - after the previous rerank, `repl_individual/cached/00_x_+_1` had become the
    top standard REPL input by a wide margin, but it still produced zero visible
    steps and no semantic change
  - the retained guard is intentionally narrow: exact root `Add` with one
    symbolic atom and one non-zero numeric literal, only in standard mode and
    without listeners
  - this lets the pipeline return immediately for cases like `x + 1` while
    preserving the current visible contract (`x + 1`, `steps_count = 0`)
- retained validation:
  - focused test:
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_symbol_plus_literal_uses_root_noop_fast_path --lib`
  - smoke CLI:
    - `cargo run -q -p cas_cli -- eval 'x + 1' --steps on --format json`
- measured outcome:
  - `repl_individual/cached/00_x_+_1`:
    `8.2179-8.2495 us`, improvement `~80.5-80.8%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `226.55-227.42 us`, improvement `~14.1-15.2%`
- updated rerank implication:
  - `x + 1` is no longer the dominant standard REPL cost
  - the next obvious standard leader is now `01_2 * 3 + 4`, followed by the
    remaining already-optimized algebraic inputs
- retained a standard root shortcut for pure numeric add chains in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- rationale:
  - after removing the `x + 1` overhead, `repl_individual/cached/01_2_*_3_+_4`
    became the next clear leader of the standard REPL batch
  - the current pipeline was paying two full iterations just to do
    `2 * 3 -> 6` and then `6 + 4 -> 10`, even though both rewrites are pure
    `Combine Constants`
  - the retained shortcut is narrow: exact root `Add` with one numeric side and
    the other side reducible by `try_rewrite_combine_constants_expr(...)` to a
    number; it synthesizes the same visible two-step `Combine Constants`
    sequence and exits before the standard phase pipeline
- retained validation:
  - focused test:
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_numeric_add_chain_uses_root_fast_path --lib`
  - smoke CLI:
    - `cargo run -q -p cas_cli -- eval '2 * 3 + 4' --steps on --format json`
- measured outcome:
  - `repl_individual/cached/01_2_*_3_+_4`:
    `12.923-12.973 us`, improvement `~67.0-68.6%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `197.29-199.44 us`, improvement `~12.6-13.8%`
- updated rerank implication:
  - the standard REPL batch is now down below `200 us`
  - the next obvious standard leaders are the already-optimized algebraic cases
    around `~12-19 us`, with no single outlier remotely comparable to the old
    `x + 1` / `2 * 3 + 4` overhead
- retained a cheaper exact two-term scalar-multiple root shortcut for the
  standard REPL path in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- rationale:
  - after the light-input shortcuts, `06_(2*x + 2*y)/(4*x + 4*y)` had become the
    top remaining leader of the standard cached REPL batch
  - the existing standard root shortcut still paid the structural GCD planner in
    order to reconstruct the two visible `Simplify Nested Fraction` steps
  - the retained replacement handles only the exact two-term additive shape and
    computes the scalar ratio directly; it still synthesizes the same visible
    two-step trace and preserves the same `required_conditions`
- retained validation:
  - focused test:
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_scalar_multiple_fraction_uses_root_fast_path --lib`
  - smoke CLI:
    - `cargo run -q -p cas_cli -- eval '(2*x + 2*y)/(4*x + 4*y)' --steps on --format json`
- measured outcome:
  - `repl_individual/cached/06_(2*x + 2*y)/(4*x + 4*y)`:
    `17.051-17.116 us`, improvement `~7.6-8.2%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `191.93-192.74 us`, improvement `~2.2-3.3%`
- updated rerank implication:
  - the standard REPL batch is now around `~192 us`
  - there is no longer a single dominant standard input; the remaining leaders
    are clustered in the low-to-mid teens, so the next profitable work is more
    likely a transversal dispatcher/setup reduction than another case-specific
    algebraic shortcut
- retained a standard `Add`-root dispatcher reorder in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- rationale:
  - even after the light-input shortcuts, the standard `Add` root path still
    tried the pythagorean matcher before the cheap exact cases `x + 1` and
    `2 * 3 + 4`
  - those non-trig cases can never match the pythagorean identity, so the old
    order was paying dead work on the new light leaders of the REPL batch
  - reordering the exact no-op/numeric add-chain shortcuts before the trig
    matcher is semantically neutral and cheap
- retained validation:
  - focused tests:
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_symbol_plus_literal_uses_root_noop_fast_path --lib`
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_numeric_add_chain_uses_root_fast_path --lib`
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_pythagorean_chain_uses_root_fast_path --lib`
- measured outcome:
  - `repl_individual/cached/00_x_+_1`:
    `7.7775-7.8302 us`, improvement `~5.8-6.6%`
  - `repl_individual/cached/01_2_*_3_+_4`:
    `11.131-11.179 us`, improvement `~12.8-13.7%`
  - `repl_individual/cached/10_sin(2*x + 1)^2 + cos(1 + 2*x)^2`:
    no statistically significant change
  - `repl_full_eval/cached/batch_11_inputs`:
    `191.54-193.09 us`, effectively flat / within noise
- conclusion:
  - the reorder is worth keeping because it is a clean no-regression cut on the
    new light leaders
  - but it also confirms that the remaining REPL-standard ROI is now highly
    diluted: future wins are more likely to come from broader dispatcher/setup
    reductions than from another local algebraic shortcut
- retained a transversal pipeline-start reduction in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- rationale:
  - `simplify_pipeline()` was clearing the thread-local cycle-event registry
    before every run, even when the expression exited through one of the many
    hidden root shortcuts and never entered the heavy phase pipeline
  - delaying `clear_cycle_events()` until just before the real phase pipeline
    keeps the semantics intact for full runs while removing fixed overhead from
    hot early-return paths in both the standard REPL and solve cached batches
- retained validation:
  - focused tests:
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_symbol_plus_literal_uses_root_noop_fast_path --lib`
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_numeric_add_chain_uses_root_fast_path --lib`
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_identical_atom_fraction_uses_root_fast_path --lib`
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_exp_ln_atom_uses_root_fast_path --lib`
  - full guardrail:
    - `cargo test -p cas_engine profile_cache_tests --lib`
- measured outcome:
  - `repl_full_eval/cached/batch_11_inputs`:
    `190.97-191.71 us`, slightly better in absolute terms and within noise
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `60.050-60.379 us`, improvement `~2.1-3.5%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `60.065-60.353 us`, improvement `~1.2-2.5%`
- conclusion:
  - this is the kind of cross-cutting win that still pays off after most
    algebraic hotspots have already been shortcut
  - the remaining ROI is now more likely in similar setup/dispatcher costs than
    in another narrowly scoped rewrite
- discarded a deeper pipeline-start deferral in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
- rationale:
  - after the successful `clear_cycle_events()` delay, I also tried delaying
    `self.pattern_marks_expr = None` until the heavy pipeline entry point
  - that looked plausible on paper because the hidden root shortcuts also avoid
    pattern scans, but unlike `clear_cycle_events()` it regressed the real
    `assume` solve batch and gave no compensating REPL win
- retained validation:
  - focused tests:
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_numeric_add_chain_uses_root_fast_path --lib`
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_solve_tactic_identical_atom_fraction_uses_root_fast_path --lib`
  - measured before revert:
    - `repl_full_eval/cached/batch_11_inputs`:
      `193.87-197.44 us`, slightly worse and within noise
    - `solve_modes_cached/solve_tactic_generic_batch`:
      `60.356-61.874 us`, flat / within noise
    - `solve_modes_cached/solve_tactic_assume_batch`:
      `60.932-61.664 us`, regression `~1.8-3.3%`
- conclusion:
  - keep `clear_cycle_events()` delayed, but keep `pattern_marks_expr = None`
    at the top of `simplify_pipeline()`
- retained a low-risk step-allocation reduction across standard root shortcuts
  plus compact small strings for `Step.description`/`Step.rule_name`
- files:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver_core/src/step_model.rs`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra/mod.rs`
- rationale:
  - the standard REPL batch is now dominated by cheap root shortcuts with
    `steps on`, so a lot of remaining overhead is no longer algebraic work but
    constructing tiny `Step`s
  - `Step.description` / `Step.rule_name` now use `SmolStr`, avoiding heap for
    the short static labels that dominate these paths
  - root shortcuts with empty path and no extra metadata now use
    `Step::new_compact(...)` instead of allocating a boxed `StepMeta`
- retained validation:
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_numeric_add_chain_uses_root_fast_path --lib`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_pythagorean_chain_uses_root_fast_path --lib`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_scalar_multiple_fraction_uses_root_fast_path --lib`
  - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_common_factor_fraction_uses_root_fast_path --lib`
  - `cargo test -p cas_didactic --test step_wire_tests`
  - `cargo test -p cas_solver --test wire_contract_tests`
- measured outcome:
  - `repl_full_eval/cached/batch_11_inputs`:
    `192.31-194.36 us`, slight improvement in absolute terms and within noise
  - `repl_individual/cached/10_sin(2*x + 1)^2 + cos(1 + 2*x)^2`:
    `14.488-14.598 us`, improvement `~1.0-2.4%`
  - `repl_individual/cached/06_(2*x + 2*y)/(4*x + 4*y)`:
    `17.107-17.213 us`, slightly better in absolute terms and within noise
- conclusion:
  - this is worth keeping because it is safe, cross-cutting, and helps the
    steps-on hot path without changing visible contracts
  - but the signal also confirms that the remaining REPL-standard ROI is now in
    very small fixed costs, not in another large local algebraic rewrite
- retained another standard `steps on` overhead reduction in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs`
  and
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra/mod.rs`
- rationale:
  - many of the remaining standard REPL leaders now exit through root shortcuts
    with `steps on`
  - for those root steps, `global_before/global_after` were redundant because
    `before/after` already describe the full-root transition and the renderers
    fall back to them when snapshots are absent
  - removing those redundant snapshots cuts a small but real fixed cost from
    the standard shortcuts without changing visible output
- retained validation:
  - focused tests:
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_symbol_plus_literal_uses_root_noop_fast_path --lib`
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_numeric_add_chain_uses_root_fast_path --lib`
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_pythagorean_chain_uses_root_fast_path --lib`
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_scalar_multiple_fraction_uses_root_fast_path --lib`
    - `cargo test -p cas_engine profile_cache_tests::tests::test_from_profile_standard_common_factor_fraction_uses_root_fast_path --lib`
  - contract coverage:
    - `cargo test -p cas_didactic --test step_wire_tests`
    - `cargo test -p cas_solver --test wire_contract_tests`
  - regression check:
    - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/solve_tactic_generic_batch' -- --noplot`
- measured outcome:
  - `repl_full_eval/cached/batch_11_inputs`:
    `189.35-190.71 us`, better in absolute terms and near significance
  - `steps_mode_comparison/batch_11/steps_on`:
    `189.68-191.24 us`, improvement `~1.8-3.5%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `60.238-60.544 us`, no regression
  - focused standard inputs also moved down in absolute terms:
    - `00_x_+_1`: `7.6577-7.7215 us`
    - `06_(2*x + 2*y)/(4*x + 4*y)`: `16.810-16.967 us`
    - `08_((x+y)*(a+b))/((x+y)*(c+d))`: `14.517-14.624 us`
- conclusion:
  - worth keeping: it is semantically neutral, keeps wire/timeline output
    stable, and improves the steps-on batch that now dominates the standard
    REPL profile
- retained tooling improvement for the standard REPL breakdown:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/repl_end_to_end.rs`
    now profiles the current real leaders instead of the older placeholder set
  - `make bench-engine-repl-breakdown` now covers:
    - `light/symbol_plus_literal`
    - `light/numeric_add_chain`
    - `heavy/nested_root`
    - `heavy/abs_square`
    - `gcd/scalar_multiple_fraction`
    - `gcd/common_factor_fraction`
    - `complex/gaussian_div`
    - `trig/pythagorean_chain`
- rationale:
  - the old breakdown was still useful, but it no longer matched the actual
    top inputs after the successive root shortcuts
  - with the batch already down near `~190 us`, further work needs per-stage
    visibility on the real remaining leaders rather than on historical cases
- retained a cross-cutting REPL/solve improvement in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/expression.rs`
  and
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/symbol.rs`
- rationale:
  - the updated `repl_stage_breakdown/*` showed that the remaining REPL
    leaders were no longer dominated by individual algebraic rules
  - for several standard inputs (`x + 1`, `2 * 3 + 4`, scalar/common-factor
    fractions, the trig shortcut) `parse` was already as large as or larger
    than `simplify`
  - `Context::new()` was still paying cold allocations for `nodes`,
    `interner`, and the builtin-filled `SymbolTable` on every eval
  - pre-reserving small capacities (`nodes`, `interner`) plus
    `SymbolTable::with_capacity(BuiltinFn::COUNT + 8)` removes that fixed
    allocation churn without changing any rule/runtime behavior
- retained validation:
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `make bench-engine-repl-individual`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/(solve_tactic_generic_batch|solve_tactic_assume_batch)' -- --noplot`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- measured outcome:
  - `repl_full_eval/cached/batch_11_inputs`:
    `175.94-176.67 us`, improvement `~9.3-11.1%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `52.314-52.740 us`, improvement `~12.0-13.3%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `52.349-52.801 us`, improvement `~11.4-12.5%`
  - full `repl_individual/cached/*` rerank improved across the board, for
    example:
    - `00_x_+_1`: `6.6547-6.7187 us`
    - `01_2_*_3_+_4`: `9.6477-9.7333 us`
    - `03_sqrt(12*x^3)`: `11.603-11.684 us`
    - `06_(2*x + 2*y)/(4*x + 4*y)`: `15.618-15.695 us`
    - `08_((x+y)*(a+b))/((x+y)*(c+d))`: `13.362-13.466 us`
  - metamorphic release stayed green with total `numeric-only = 164`
- conclusion:
  - this is the right kind of remaining optimization for the current state of
    the engine: it cuts fixed startup cost instead of adding another brittle
    shortcut
  - after this win, the next ROI should be another transversal reduction of
    eval setup/dispatch, or a stricter rerank before touching more local rules
- retained another cross-cutting startup reduction in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/symbol.rs`
  and
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/expression.rs`
- rationale:
  - after pre-reserving `Context`, the updated REPL breakdown still showed that
    parse/setup was dominating or tying simplify for many of the remaining real
    leaders
  - `SymbolTable` was still paying owned `String` allocations and larger hash
    map keys for every builtin and user symbol interned in each fresh context
  - switching the symbol table to `SmolStr` cuts those tiny heap allocations
    and shrinks key/value churn while preserving the same external API
- retained change:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_ast/Cargo.toml`:
    add `smol_str`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/symbol.rs`:
    `Vec<String>`/`HashMap<String, SymbolId>` →
    `Vec<SmolStr>`/`HashMap<SmolStr, SymbolId>`
- retained validation:
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `make bench-engine-repl-individual`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/(solve_tactic_generic_batch|solve_tactic_assume_batch)' -- --noplot`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- measured outcome:
  - `repl_full_eval/cached/batch_11_inputs`:
    `133.25-133.69 us`, improvement `~23.7-24.9%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `32.931-33.181 us`, improvement `~36.9-37.9%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `32.906-33.151 us`, improvement `~36.8-37.4%`
  - full `repl_individual/cached/*` rerank improved sharply again, for
    example:
    - `00_x_+_1`: `3.4251-3.5117 us`
    - `01_2_*_3_+_4`: `6.4373-6.4924 us`
    - `03_sqrt(12*x^3)`: `8.4703-8.7189 us`
    - `06_(2*x + 2*y)/(4*x + 4*y)`: `12.059-12.139 us`
    - `08_((x+y)*(a+b))/((x+y)*(c+d))`: `9.6599-9.8305 us`
  - metamorphic release stayed green with total `numeric-only = 164`
- conclusion:
  - this is a real baseline shift, not another local shortcut win
  - after this change, the remaining ROI is even more likely to be in
    dispatcher/setup costs or parser-side work, not in individual algebraic
    rules
- retained a parser-side reduction in
  `/Users/javiergimenezmoya/developer/math/crates/cas_parser/src/parser.rs`
  and
  `/Users/javiergimenezmoya/developer/math/crates/cas_parser/Cargo.toml`
- rationale:
  - after the `SmolStr` move in `SymbolTable`, parse/setup remained the next
    best transversal place to probe
  - the textual parser was still allocating owned `String`s for
    `ParseNode::{Variable, Function}` and always lowering function names via
    `ctx.call(...)`, even for builtins whose `SymbolId` is already cached
  - switching those temporary names to `SmolStr` and lowering known builtins
    through `ctx.call_builtin(...)` trims parser-side churn without changing the
    public AST model
- retained change:
  - `ParseNode::Variable(String)` / `ParseNode::Function(String, ...)` →
    `SmolStr`
  - builtin names in the parser lower through `ctx.call_builtin(...)`
  - added parser regression coverage for short and long inverse-trig names:
    `asin(x)` and `arcsin(x)`
- retained validation:
  - `cargo test -p cas_parser --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `make bench-engine-repl-individual`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/(solve_tactic_generic_batch|solve_tactic_assume_batch)' -- --noplot`
  - smoke CLI for `asin(x)` and `arcsin(x)`
- measured outcome:
  - `repl_full_eval/cached/batch_11_inputs`:
    `131.77-132.83 us`, better in absolute terms but statistically within noise
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `31.104-31.273 us`, improvement `~4.7-5.9%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `31.122-31.267 us`, improvement `~5.2-6.2%`
  - several parse-heavy REPL inputs also improved clearly:
    - `02_sin(x)^2 + cos(x)^2`: `5.6460-5.7036 us`
    - `03_sqrt(12*x^3)`: `8.1961-8.2793 us`
    - `06_(2*x + 2*y)/(4*x + 4*y)`: `11.685-11.742 us`
    - `08_((x+y)*(a+b))/((x+y)*(c+d))`: `8.1797-8.2219 us`
- conclusion:
  - worth keeping: it is semantically safe, improves the parser path in several
    real inputs, and still moves the solve batch in the right direction
  - but unlike the `SymbolTable` change, this is not a major baseline shift;
    the next ROI is still likely to be another transversal setup/dispatcher cut
- retained another AST startup/runtime reduction in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/builtin.rs`
- rationale:
  - after the parser-side cleanup, the next fixed cost still paid in every
    fresh `Context` was `BuiltinIds`
  - the cache was building a per-context reverse `HashMap<SymbolId, BuiltinFn>`
    even though the normal `Context::new()` path interns builtins first and in
    enum order
  - replacing that reverse map with:
    - an O(1) identity-layout fast path (`id < COUNT && ids[id] == id`)
    - a tiny fallback scan for any unusual non-identity layout
    removes the map allocation and insert churn without losing correctness
- retained change:
  - `BuiltinIds` now stores only the fixed `ids` array plus `initialized`
  - `lookup(id)` resolves directly from `ALL_BUILTINS[id]` in the common case,
    with a fallback linear scan over the 43 builtin ids
  - added regression coverage for non-identity fallback lookup
- retained validation:
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/(solve_tactic_generic_batch|solve_tactic_assume_batch)' -- --noplot`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- measured outcome:
  - `repl_full_eval/cached/batch_11_inputs`:
    `117.19-118.44 us`, improvement `~10.7-12.1%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `24.775-25.124 us`, improvement `~18.8-20.7%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `24.699-24.935 us`, improvement `~19.8-20.7%`
  - metamorphic release stayed green with total `numeric-only = 164`
- conclusion:
  - this is another real baseline shift from trimming startup/runtime metadata,
    not from adding a brittle algebraic shortcut
  - the next ROI is still likely to be in transversal setup/dispatch costs,
    but the remaining floor is now substantially lower
- retained another AST startup reduction in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/symbol.rs`,
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/builtin.rs`, and
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/expression.rs`
- rationale:
  - after removing the reverse `BuiltinIds` map, `Context::new()` was still
    paying per-context startup churn by interning all builtin names one by one
  - the normal layout is already fixed: builtins are always the prefix of a
    fresh symbol table and their `SymbolId`s match `BuiltinFn` discriminants
  - seeding that prefix directly lets `Context::new()` skip the whole
    `intern_symbol(...)` loop and start with an already initialized builtin
    cache
- retained change:
  - `SymbolTable::with_static_prefix(...)` preloads a fixed symbol prefix with
    reserved extra capacity for user symbols
  - `BuiltinIds::initialized_identity()` creates the standard identity-layout
    cache without runtime setup work
  - `Context::new()` now seeds builtin symbols and builtin ids directly instead
    of calling `init_builtins()`
  - added regression coverage for:
    - `SymbolTable::with_static_prefix(...)`
    - `BuiltinIds::initialized_identity()`
    - `Context::new()` preserving the builtin identity layout
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/(solve_tactic_generic_batch|solve_tactic_assume_batch)' -- --noplot`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- measured outcome:
  - `repl_full_eval/cached/batch_11_inputs`:
    `116.72-117.59 us`, better in absolute terms but statistically within noise
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `23.586-24.076 us`, improvement `~4.5-7.2%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `23.388-23.583 us`, improvement `~4.7-6.1%`
  - metamorphic release stayed green with total `numeric-only = 164`
- conclusion:
  - worth keeping: it is a semantically boring startup cut that materially
    improves the solve hot path without any contract drift
  - the next ROI still points at transversal setup/dispatcher work, not at
    another local algebraic shortcut
- retained a stronger symbol-table startup reduction in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/symbol.rs`,
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/builtin.rs`,
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/expression.rs`,
  and
  `/Users/javiergimenezmoya/developer/math/crates/cas_parser/src/parser.rs`
- rationale:
  - even after seeding builtin names directly, every fresh `Context` was still
    paying to own a dynamic symbol-table prefix that is logically fixed
  - builtin names do not need to live in `SymbolTable` storage at all:
    they already have stable enum discriminants and a canonical static name
  - moving builtins to a logical prefix means:
    - `SymbolTable` stores only user-defined symbols
    - builtin `SymbolId`s stay as `0..BuiltinFn::COUNT`
    - `Context::new()` no longer allocates or fills builtin string/hash entries
      of any kind
  - centralizing builtin textual lookup in `BuiltinFn::from_name(...)` also
    lets the parser reuse the same table instead of carrying its own duplicate
    matcher
- retained change:
  - `SymbolTable::intern/get_id/resolve/len` now treat builtins as a fixed
    logical prefix rather than dynamic entries
  - `BuiltinFn::from_name(...)` is the canonical builtin text lookup
  - `Context::new()` now starts with an empty dynamic symbol table plus
    `BuiltinIds::initialized_identity()`
  - the parser lowers known builtins through `BuiltinFn::from_name(...)`
    instead of a duplicated local lookup helper
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_parser --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/(solve_tactic_generic_batch|solve_tactic_assume_batch)' -- --noplot`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- measured outcome:
  - `repl_full_eval/cached/batch_11_inputs`:
    `105.66-106.31 us`, improvement `~8.5-10.0%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `18.022-18.186 us`, improvement `~22.4-24.1%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `18.048-18.139 us`, improvement `~22.6-23.7%`
  - metamorphic release stayed green with total `numeric-only = 164`
- conclusion:
  - this is another real baseline shift from eliminating fixed startup state,
    not from another brittle shortcut
  - after this change the remaining easy wins are even more likely to sit in
    generic setup/dispatch and parser/runtime scaffolding, not in local algebra
- retained another transversal startup reduction in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/symbol.rs`,
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/expression.rs`,
  and `/Users/javiergimenezmoya/developer/math/crates/cas_ast/Cargo.toml`
- rationale:
  - once builtin symbols stopped living in the dynamic symbol table, the next
    fixed cost still paid on every fresh context was standard `HashMap`
    hashing in the two hottest startup structures:
    - `SymbolTable.lookup`
    - `Context.interner`
  - both maps are hot, short-lived, and internal-only, so switching them to
    `FxHashMap` is a good fit: lower hashing overhead with no semantic drift
- retained change:
  - add `rustc-hash` to `cas_ast`
  - `SymbolTable.lookup` now uses `FxHashMap<SmolStr, SymbolId>`
  - `Context.interner` now uses `FxHashMap<u64, Vec<ExprId>>`
  - capacity reservation is preserved for both maps
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/(solve_tactic_generic_batch|solve_tactic_assume_batch)' -- --noplot`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- measured outcome:
  - `repl_full_eval/cached/batch_11_inputs`:
    `102.62-104.07 us`, improvement `~1.0-3.2%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `17.047-17.186 us`, improvement `~5.7-7.5%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `16.788-17.097 us`, improvement `~7.2-8.8%`
  - metamorphic release stayed green with total `numeric-only = 164`
- conclusion:
  - worth keeping: this keeps the current direction of removing setup/hash
    overhead, and the solve path still responds strongly to it
  - the next ROI still looks transversal, but the marginal returns are now
    clearly smaller than the previous builtin/storage cuts
- retained another interner-side setup reduction in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/expression.rs`
- rationale:
  - after moving the interner storage to `FxHashMap`, the next fixed cost still
    inside `Context::add/add_raw` was the expression hash itself
  - those hashes are only used for bucket partitioning before an exact
    structural equality check, so the engine does not need SipHash-grade
    protection there
  - switching the internal expression hash from `DefaultHasher` to `FxHasher`
    trims that per-node cost without changing interning correctness
- retained change:
  - `Context::expr_hash(...)` now computes the interner key with `FxHasher`
  - both `add(...)` and `add_raw(...)` reuse that helper
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/(solve_tactic_generic_batch|solve_tactic_assume_batch)' -- --noplot`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- measured outcome:
  - `repl_full_eval/cached/batch_11_inputs`:
    `99.917-100.65 us`, improvement `~2.5-4.7%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `16.403-16.919 us`, improvement `~2.3-4.6%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `16.440-16.540 us`, improvement `~1.8-3.1%`
  - metamorphic release stayed green with total `numeric-only = 164`
- conclusion:
  - worth keeping, but this is already a smaller step than the previous
    builtins/`FxHashMap` cuts
  - the remaining easy wins are now almost certainly in other transversal setup
    costs, not in the interner path itself
- retained another interner-side allocation cut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/expression.rs`
- rationale:
  - after lowering hash costs, the next fixed cost in the interner buckets was
    the bucket container itself
  - almost every interner bucket holds exactly one `ExprId`, but `Vec<ExprId>`
    still pays a heap allocation in that common case
  - switching buckets to `SmallVec<[ExprId; 1]>` removes that allocation while
    keeping the same collision behavior for the rare multi-entry bucket
- retained change:
  - `Context.interner` now stores `FxHashMap<u64, SmallVec<[ExprId; 1]>>`
  - all existing bucket logic (`get`, scan, `or_default().push(id)`) stays the
    same semantically
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/(solve_tactic_generic_batch|solve_tactic_assume_batch)' -- --noplot`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- measured outcome:
  - `repl_full_eval/cached/batch_11_inputs`:
    `96.830-97.748 us`, improvement `~2.3-3.7%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `15.237-15.535 us`, improvement `~6.6-9.0%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `15.318-15.407 us`, improvement `~6.3-7.5%`
  - metamorphic release stayed green with total `numeric-only = 164`
- conclusion:
  - worth keeping: even at this stage the interner still had one last easy
    allocation to remove
  - after this, the remaining setup wins are likely to be even more diffuse and
    need tighter measurement discipline than the last few cuts
- discarded two follow-up hypotheses after measurement:
  - moving builtin recognition into `cas_parser` `ParseNode::Function` regressed
    both `repl_full_eval/cached/batch_11_inputs` and
    `solve_modes_cached/solve_tactic_generic_batch`, so it was reverted
  - shrinking `Context::new()` startup capacities (`nodes`, `interner`,
    `symbols`) regressed `solve_tactic_assume_batch`, so it was reverted
- retained another transversal allocation cut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/expression.rs`
- rationale:
  - after the interner bucket work, the next fixed cost still visible in both
    `solve_tactic_*` and the standard REPL batch was temporary heap traffic
    inside `Context::add(...)`
  - the common case for canonicalization is still a very small number of `Add`
    terms / `Mul` factors, but `Vec` was being allocated for:
    - `terms` / `factors`
    - the DFS stack in `collect_add_terms(...)` / `collect_mul_factors(...)`
    - the `current` / `next` buffers in balanced rebuilds
  - switching those temporaries to `SmallVec<[ExprId; 8]>` removes that heap
    churn in the common small-expression case without changing canonicalization
    semantics
- retained change:
  - `Context::add(...)` now uses `SmallVec<[ExprId; 8]>` for additive terms and
    multiplicative factors
  - `collect_add_terms(...)` / `collect_mul_factors(...)` now use
    `SmallVec<[ExprId; 8]>` stacks
  - `build_balanced_add(...)` / `build_balanced_mul(...)` now use
    `SmallVec<[ExprId; 8]>` for their iterative `current` / `next` buffers
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/(solve_tactic_generic_batch|solve_tactic_assume_batch)' -- --noplot`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- measured outcome:
  - `repl_full_eval/cached/batch_11_inputs`:
    `96.048-97.616 us`, improvement `~2.5-5.5%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `14.819-14.995 us`, improvement `~3.8-5.4%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `14.698-14.863 us`, improvement `~4.2-5.4%`
  - metamorphic release stayed green with total `numeric-only = 164`
- conclusion:
  - worth keeping: this is a real transversal win, not another local shortcut
  - the remaining gains are likely to come from similar fixed-cost reductions
    in setup / canonicalization scaffolding, not from more rule-specific work
- discarded follow-up micro-opts after measurement:
  - collapsing `interner.get(...)` + `interner.entry(...)` into a single
    `entry(...)` path in `Context::add/add_raw` did not improve the real
    batches and slightly worsened `solve_tactic_generic_batch`, so it was
    reverted
  - rewriting `parse_identifier(...)` to scan bytes instead of chars also
    stayed in noise on the real REPL batch and was reverted
- retained another transversal setup reduction in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/expression.rs`
- rationale:
  - after the `SmallVec` canonicalization cut, the next fixed cost still hit on
    every new node was the interner hash itself
  - `expr_hash(...)` was still delegating to the derived `Hash` impl for `Expr`,
    which is generic and pays extra overhead per variant
  - the interner only needs a good bucket key before an exact structural
    equality check, so a hand-specialized `FxHasher` walk over `Expr` fields is
    safe and cheaper
- retained change:
  - `Context::expr_hash(...)` is now specialized for `Expr`
  - the hash writes a compact tag per variant plus the relevant ids / scalar
    payloads directly:
    - `ExprId` payloads as raw `u32`
    - `SymbolId` / shape fields directly
    - `BigRational` as `numer()` + `denom()`
    - `Function` / `Matrix` arguments as direct id sequences
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/(solve_tactic_generic_batch|solve_tactic_assume_batch)' -- --noplot`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- measured outcome:
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `14.550-14.728 us`, improvement `~2.4-4.3%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `14.444-14.513 us`, improvement `~2.7-3.7%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `95.776-97.404 us`, better in absolute terms but within Criterion noise
  - metamorphic release stayed green with total `numeric-only = 164`
- conclusion:
  - worth keeping: the signal is strong on the two solver batches and neutral
    on the standard REPL path
  - from here, the remaining easy wins still look transversal, but they are now
    clearly in the low-single-digit range
- retained another transversal canonicalization cut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/expression.rs`
- rationale:
  - after the `expr_hash(...)` specialization, the next setup cost still paid
    on every commutativity check was the stack allocation inside
    `mul_commutativity(...)`
  - that traversal runs from `Context::add(...)` on many standard `Mul`
    constructions, even when the subtree is tiny and purely scalar
  - the common case is still a handful of ids, so `SmallVec<[ExprId; 8]>`
    removes another fixed heap allocation without changing the traversal logic
- retained change:
  - `mul_commutativity(...)` now uses `SmallVec<[ExprId; 8]>` for its DFS stack
    instead of `Vec<ExprId>`
  - the traversal order and the function-scalar short-circuit stay unchanged
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'solve_modes_cached/(solve_tactic_generic_batch|solve_tactic_assume_batch)' -- --noplot`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- measured outcome:
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `13.851-13.987 us`, improvement `~5.8-7.2%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `13.824-13.896 us`, improvement `~1.7-4.3%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `90.942-93.974 us`, improvement `~4.5-6.9%`
  - metamorphic release stayed green with total `numeric-only = 164`
- conclusion:
  - worth keeping: this is another real fixed-cost win in setup/canonicalization
    rather than a benchmark-local shortcut
  - after this cut, the remaining opportunities in the AST still look
    transversal, but they are becoming small enough that every attempt needs
    batch-first validation
- discarded follow-up hypothesis after measurement:
  - adding a dedicated fast path cache for `Expr::Variable` /
    `Expr::Constant` in `Context::add()` looked plausible on paper, but it left
    `repl_full_eval/cached/batch_11_inputs` effectively flat and pushed
    `solve_tactic_generic_batch` slightly worse (`14.737-14.968 us`), so it was
    reverted
- retained tooling improvement in `/Users/javiergimenezmoya/developer/math/Makefile`
- rationale:
  - at the current baseline, `solve_modes_cached/(solve_tactic_generic_batch|
    solve_tactic_assume_batch)` is the main guardrail for transversal setup cuts
  - running those filters by hand every time was repetitive and made it easier
    to compare the wrong thing or skip the named-baseline discipline
- retained change:
  - added `make bench-engine-solve-batches`
  - added `make bench-engine-solve-batches-save BASELINE=...`
  - added `make bench-engine-solve-batches-compare BASELINE=...`
- conclusion:
  - worth keeping: it shortens the loop for the exact pair of benches that now
    decides whether a transversal optimization is real enough to retain
- retained another transversal setup cut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/expression.rs`
- rationale:
  - `ctx.builtin_of(...)` is still on a very hot path across trig, logarithm and
    canonicalization helpers in both `cas_math` and `cas_engine`
  - after the builtin-prefix work in the symbol table, runtime `Context`
    instances always use the identity layout for builtin `SymbolId`s, so the
    extra `BuiltinIds` indirection in `builtin_id()` / `builtin_of()` was pure
    compatibility overhead
- retained change:
  - `Context::builtin_id(...)` now returns the builtin discriminant directly as
    `SymbolId`
  - `Context::builtin_of(...)` now does a direct prefix check
    (`fn_id < BuiltinFn::COUNT`) and indexes `ALL_BUILTINS` directly
  - the now-dead `builtins` field was removed from `Context` to keep the tree
    warning-free
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `make bench-engine-solve-batches`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
- measured outcome:
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `13.887-14.239 us`, improvement `~0.6-2.7%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `13.497-13.683 us`, improvement `~2.3-4.2%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `91.521-95.055 us`, improvement `~1.7-4.8%`
- discarded follow-up hypotheses after measurement:
  - logical fixed-prefix ids for ASCII single-letter variables (`x`, `y`, `a`,
    `b`, etc.) looked plausible, but they left `repl_full_eval` flat and
    nudged `solve_tactic_generic_batch` slightly the wrong way, so they were
    reverted
  - switching `SymbolTable::intern()` to a raw-entry single lookup was also
    discarded immediately because the current toolchain does not expose the API
    on `FxHashMap` without another dependency/refactor
- retained another transversal canonicalization cut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/expression.rs`
- rationale:
  - after the setup/interner cuts, a visible fixed cost still remained in
    `Context::add()` for the common case of binary `Add` / `Mul` nodes that are
    already flat
  - those nodes were still going through `SmallVec + collect + sort + rebuild`,
    even when the canonicalization decision was only “keep” or “swap” two ids
  - many parse-heavy and simplify-heavy paths hit exactly that shape, so a
    narrow fast path there is more valuable than another rule-local shortcut
- retained change:
  - `Expr::Add(l, r)` now takes a direct fast path when neither side is another
    `Add`, using `compare_add_terms(...)` once and constructing the canonical
    binary form directly
  - `Expr::Mul(l, r)` now takes a direct fast path when neither side is another
    `Mul`, using the existing commutativity guard and a single comparison in the
    commutative case
  - added focused regression tests for binary `Add` ordering and positive-first
    behavior
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `make bench-engine-solve-batches`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- measured outcome:
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `13.711-13.805 us`, improvement `~1.3-3.5%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `13.556-13.880 us`, no statistically significant change
  - `repl_full_eval/cached/batch_11_inputs`:
    `90.406-93.107 us`, no statistically significant change, but better in
    absolute terms
  - metamorphic release stayed green with total `numeric-only = 164`
- conclusion:
  - worth keeping: the change improves the main generic solve batch, stays
    neutral elsewhere, and is semantically covered by both focused AST tests and
    the metamorphic release suite
- retained a small transversal canonicalization cut in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/expression.rs`
- rationale:
  - after the binary `Add` / `Mul` fast path, the remaining fixed cost in
    `Context::add()` still includes sorting of flattened term/factor lists
  - those sorts do not require stability: equal keys are already identical
    interned expressions, so `sort_unstable_by(...)` is semantically enough and
    can shave a bit of overhead from the hot path
- retained change:
  - switched additive term sorting from `sort_by(...)` to
    `sort_unstable_by(...)`
  - switched commutative multiplicative factor sorting from `sort_by(...)` to
    `sort_unstable_by(...)`
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `cargo check -p cas_engine --benches -p cas_solver -p cas_session -p cas_didactic -p cas_math -p cas_api_models -p cas_cli`
  - `make bench-engine-solve-batches`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
  - `cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture`
- measured outcome:
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `13.697-14.176 us`, no statistically significant change
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `13.323-13.465 us`, improvement `~0.4-2.1%` and within Criterion noise
    threshold
  - `repl_full_eval/cached/batch_11_inputs`:
    `89.112-90.161 us`, improvement `~1.6-3.5%`
  - metamorphic release stayed green with total `numeric-only = 164`
- conclusion:
  - worth keeping: the win is small, but it is on a very hot transversal path,
    the main REPL batch improves cleanly, and the change is semantically low
    risk
- discarded follow-up hypotheses after measurement:
  - specializing `build_balanced_add(...)` / `build_balanced_mul(...)` for
    arities `3` and `4` looked plausible, but it left
    `solve_tactic_generic_batch` effectively flat and nudged both
    `solve_tactic_assume_batch` and `repl_full_eval/cached/batch_11_inputs`
    slightly the wrong way, so it was reverted
  - adding a broad shallow fast path to `compare_expr(...)` regressed both
    `solve_tactic_generic_batch` and `repl_full_eval/cached/batch_11_inputs`,
    so it was reverted immediately
  - adding only an atom-vs-atom fast path to `compare_expr(...)` also failed to
    move the guardrail batches enough to justify the extra branching, so it was
    reverted
  - rewriting `Context::add()` / `add_raw()` to use a single `get_mut` lookup
    on the interner bucket compiled cleanly, but left `solve_tactic_generic`
    flat and made `solve_tactic_assume` drift worse, so it was reverted
  - marking `Context::get()` as `#[inline]` was effectively flat on both solve
    batches and on the REPL batch, so it was reverted to keep the tree minimal
  - preloading numeric literals `0`, `1`, and `-1` inside `Context::new()`
    regressed both solve guardrails and the REPL batch
    (`solve_tactic_generic_batch` to `15.744-15.897 us`,
    `solve_tactic_assume_batch` to `15.480-16.004 us`,
    `repl_full_eval/cached/batch_11_inputs` to `96.685-98.304 us`), so it was
    reverted even though the focused stats test passed
- retained a small-number hash fast path in
  `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/expression.rs`
- rationale:
  - after the recent setup/interner wins, the remaining hot path still hashes a
    large volume of tiny `BigRational` literals (`0`, `1`, `-1`, small
    integers, and short rationals) during AST interning
  - the generic `Hash` path for `BigInt`/`BigRational` is correct but heavier
    than needed for the overwhelmingly common case where both numerator and
    denominator fit in `i64`
- retained change:
  - `Context::expr_hash(...)` now hashes `Expr::Number` through a direct
    `(i64, i64)` fast path when both `numer()` and `denom()` fit in `i64`,
    falling back to the old `Hash` implementation for large rationals
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_ast --lib`
  - `cargo test -p cas_engine profile_cache_tests --lib`
  - `cargo test -p cas_solver --test solve_safety_contract_tests`
  - `make bench-engine-solve-batches`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
- measured outcome:
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `14.395-14.614 us`, improvement `~6.4-8.2%`
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `13.999-14.226 us`, improvement `~8.2-10.9%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `94.405-96.215 us`, improvement `~1.9-4.4%`
- conclusion:
  - worth keeping: the change is tiny, semantically safe because interning
    still confirms by exact structural equality, and it gives a clean win on
    the two solve guardrails plus a smaller but real win on the REPL batch
- discarded follow-up hypothesis after measurement:
  - packing binary node ids into a single `u64` write inside `expr_hash(...)`
    left both solve guardrails and the REPL batch within noise
    (`solve_tactic_generic_batch` `14.193-14.377 us`,
    `solve_tactic_assume_batch` `14.052-14.152 us`,
    `repl_full_eval/cached/batch_11_inputs` `94.029-95.881 us`), so it was
    reverted
  - switching local `MulBuilder` / `FractionParts` buffers in
    `/Users/javiergimenezmoya/developer/math/crates/cas_ast/src/views.rs`
    from `Vec` to `SmallVec` looked promising on paper, but it regressed the
    REPL batch (`repl_full_eval/cached/batch_11_inputs` to
    `98.018-99.242 us`) while leaving both solve guardrails flat, so it was
    reverted
- opened a cleaner frontend benchmark track in
  `/Users/javiergimenezmoya/developer/math/crates/cas_parser/benches/frontend_parse.rs`
- rationale:
  - the current `solve_tactic_*` and REPL batches are now fast enough that many
    small AST/setup hypotheses bounce in and out of noise
  - we needed a benchmark that isolates frontend setup and parse/lowering cost
    without the engine pipeline mixed in, so we can decide whether the next ROI
    is in `Context::new()`, textual parsing, or statement/equation lowering
- retained tooling:
  - new Criterion bench `frontend_parse` in `cas_parser` with:
    - `frontend_parse/context/new`
    - `frontend_parse/expr_batch/standard_8`
    - `frontend_parse/statement_batch/solve_5`
    - individual expression and statement parse cases
  - new Make targets:
    - `make bench-parser-frontend`
    - `make bench-parser-frontend-save BASELINE=...`
    - `make bench-parser-frontend-compare BASELINE=...`
- retained validation:
  - `cargo fmt --all`
  - `cargo check -p cas_parser --benches`
  - `cargo test -p cas_parser --lib`
  - `make bench-parser-frontend`
- initial snapshot:
  - `frontend_parse/context/new`: `89.474-91.456 ns`
  - `frontend_parse/expr_batch/standard_8`: `26.387-26.500 us`
  - `frontend_parse/statement_batch/solve_5`: `8.980-9.081 us`
  - direct expression parse leaders:
    - `heavy/abs_square`: `5.606-5.657 us`
    - `trig/pythagorean_chain`: `5.064-5.149 us`
    - `gcd/scalar_multiple_fraction`: `4.547-4.598 us`
    - `gcd/common_factor_fraction`: `3.997-4.127 us`
    - `complex/gaussian_div`: `3.707-3.764 us`
  - direct statement parse leaders:
    - `solve/fraction_eq`: `3.512-3.603 us`
    - `solve/trig_eq`: `2.415-2.476 us`
    - `solve/linear_eq`: `1.641-1.685 us`
    - `solve/quadratic_eq`: `1.566-1.617 us`
    - `relation/strict_less`: `0.674-0.706 us`
- conclusion:
  - this is a cleaner next front than the current solve batch guardrails:
    `Context::new()` is now tiny, so the next likely ROI in the frontend is
    parser/lowering work on heavy expression forms (`abs_square`,
    `pythagorean_chain`, fractional gcd shapes), not more AST context setup
- retained a unified identifier parser path in
  `/Users/javiergimenezmoya/developer/math/crates/cas_parser/src/parser.rs`
- rationale:
  - the first focused parser bench showed a clean hotspot on
    `statement/solve/fraction_eq`
  - the old `parse_atom(...)` path re-scanned the same identifier up to three
    times through `parse_function`, `parse_constant`, and `parse_variable`
    before deciding what the token actually was
- retained change:
  - introduced `parse_identifier_atom(...)` so identifier-starting tokens are
    classified in one pass as function call, constant, or variable
  - kept the existing lowering contract and builtin identity tests unchanged
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_parser --lib`
  - `make bench-parser-frontend`
  - `make bench-engine-solve-batches`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
- measured outcome:
  - `frontend_parse/statement/solve/fraction_eq`:
    `3.5407-3.6401 us`, improvement `~2.0-6.5%`
  - `solve_modes_cached/solve_tactic_generic_batch`:
    `14.379-14.577 us`, effectively flat to slightly better in absolute terms
  - `solve_modes_cached/solve_tactic_assume_batch`:
    `13.992-14.117 us`, no regression
  - `repl_full_eval/cached/batch_11_inputs`:
    `95.437-96.984 us`, improvement `~1.8-3.7%`
- conclusion:
  - worth keeping: this removes duplicated frontend work at very low semantic
    risk and improves the clean parser benchmark while still helping the real
    REPL batch
- retained a first-character dispatch in
  `/Users/javiergimenezmoya/developer/math/crates/cas_parser/src/parser.rs`
- rationale:
  - after collapsing identifier parsing, `parse_atom(...)` was still paying a
    generic `alt(...)` chain even though the first byte already determines the
    only plausible parser in the common ASCII cases
- retained change:
  - `parse_atom(...)` now dispatches directly by leading character to
    `parse_number`, `parse_identifier_atom`, `parse_matrix`, `parse_parens`,
    `parse_abs`, `parse_session_ref`, or `parse_unicode_root`
  - reverted an intermediate manual `parse_relop(...)` experiment; only the
    atom dispatch remains
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_parser --lib`
  - `make bench-parser-frontend`
  - `make bench-engine-solve-batches`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs' -- --noplot`
- measured outcome:
  - `frontend_parse/expr_batch/standard_8`:
    `25.832-26.365 us`, improvement `~1.9-4.2%`
  - `frontend_parse/statement_batch/solve_5`:
    `8.5848-8.6507 us`, improvement `~2.5-3.9%`
  - direct parser wins:
    - `gcd/common_factor_fraction`: `3.6664-3.7252 us`, improvement `~2.7-6.6%`
    - `heavy/nested_root`: `2.4535-2.5307 us`, improvement `~1.5-8.6%`
    - `heavy/abs_square`: `5.4822-5.5851 us`, improvement `~1.7-4.4%`
    - `trig/pythagorean_chain`: `5.0360-5.1548 us`, improvement `~1.5-6.8%`
  - engine guardrails:
    - `solve_modes_cached/solve_tactic_generic_batch`:
      `13.887-13.986 us`, improvement `~3.1-4.6%`
    - `solve_modes_cached/solve_tactic_assume_batch`:
      `13.751-13.871 us`, improvement `~1.4-2.7%`
    - `repl_full_eval/cached/batch_11_inputs`:
      `92.577-94.157 us`, improvement `~1.7-3.8%`
- conclusion:
  - worth keeping: this is the cleanest parser-side win of the new benchmark
    track because it improves both isolated parser cost and the two engine
    guardrails without relying on any domain-specific shortcut
- discarded follow-up hypotheses after measurement:
  - splitting parse nodes into `Function(...)` versus
    `BuiltinFunction(BuiltinFn, ...)` regressed
    `frontend_parse/statement/solve/fraction_eq` to `3.6737-3.7478 us` and did
    not help the heavy expression leaders enough, so it was reverted
  - rewriting `parse_number(...)` to a manual ASCII scanner regressed both
    `frontend_parse/expr_batch/standard_8` (`26.680-27.104 us`) and
    `frontend_parse/statement_batch/solve_5` (`9.0027-9.1193 us`), so it was
    reverted
  - rewriting `parse_relop(...)` as a manual prefix matcher did not produce a
    robust enough win once the full parser bench reran, and it also nudged
    unrelated microbenches the wrong way, so it was reverted
- front status:
  - this parser/lowering benchmark track is now reasonably resolved for the
    current pass: the two clean wins are retained, the next hypotheses already
    fall into mixed/noisy territory, and the right next move would be a
    different benchmark front rather than more parser micro-tweaks
- opened a formatter/frontend benchmark track in
  `/Users/javiergimenezmoya/developer/math/crates/cas_formatter/benches/frontend_render.rs`
  with matching Make targets:
  - `make bench-formatter-frontend`
  - `make bench-formatter-frontend-save BASELINE=...`
  - `make bench-formatter-frontend-compare BASELINE=...`
- initial snapshot from that track showed the next clean renderer-side ROI:
  - `frontend_render/display_expr_batch/standard_8`: `8.1027-8.2536 us`
  - `frontend_render/styled_clean_batch/standard_8`: `18.574-18.755 us`
  - `frontend_render/clean_only_batch/standard_8`: `11.369-11.489 us`
  - on small/no-op cases, `clean_display_string(...)` dominated formatting:
    - `light/x_plus_1`: `clean_only = 1.2457-1.2595 us`
    - `light/numeric_add_chain`: `clean_only = 1.2619-1.2757 us`
- retained a structural rewrite of
  `/Users/javiergimenezmoya/developer/math/crates/cas_formatter/src/display_clean.rs`
  instead of more regex/clone-heavy cleanup:
  - fast-return when the string contains no unit, hold, or sign-cleanup
    markers
  - unit cleanup now only runs targeted replacements when the corresponding
    substring exists, instead of cloning and `replace(...)`-ing blindly
  - sign cleanup no longer uses regex and now runs as a single manual scan only
    when a `+ -`, `- -`, `+-`, or `--` pattern is actually present
- retained validation:
  - `cargo fmt --all`
  - `cargo test -p cas_formatter --lib`
  - `cargo check -p cas_formatter --benches`
  - `make bench-formatter-frontend`
  - `CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_full_eval/cached/batch_11_inputs|repl_stage_breakdown/format/(light/symbol_plus_literal|light/numeric_add_chain|heavy/nested_root|heavy/abs_square|gcd/scalar_multiple_fraction|gcd/common_factor_fraction|complex/gaussian_div|trig/pythagorean_chain)' -- --noplot`
- measured outcome:
  - `frontend_render/styled_clean_batch/standard_8`:
    `8.3628-8.4727 us`, improvement `~54%`
  - `frontend_render/clean_only_batch/standard_8`:
    `1.7734-1.7863 us`, improvement `~84%`
  - direct `clean_only` wins:
    - `light/x_plus_1`: `116.87-120.42 ns`, improvement `~90%`
    - `light/numeric_add_chain`: `169.90-175.12 ns`, improvement `~86%`
    - `gcd/scalar_multiple_fraction`: `239.65-248.44 ns`, improvement `~83%`
    - `gcd/common_factor_fraction`: `253.84-260.08 ns`, improvement `~82%`
    - `heavy/nested_root`: `186.98-191.53 ns`, improvement `~85%`
    - `heavy/abs_square`: `194.77-198.86 ns`, improvement `~84%`
    - `complex/gaussian_div`: `209.99-214.08 ns`, improvement `~83%`
    - `trig/pythagorean_chain`: `249.13-254.73 ns`, improvement `~83%`
  - `repl_full_eval/cached/batch_11_inputs`:
    `89.426-90.219 us`, improvement `~2.4-4.2%`
  - `repl_stage_breakdown/format/*` stayed mostly within noise, so the win is
    real but isolated to the formatter/frontend microbench and not strong
    enough to justify a longer renderer-specific push right now
- front status:
  - this formatter/render benchmark track is now reasonably resolved for the
    current pass: the dominant cleanup cost was removed, the remaining broader
    REPL formatting guardrail is already close to noise, and the next high-ROI
    move should again be a different benchmark front rather than more
    formatter-specific micro-tweaks
