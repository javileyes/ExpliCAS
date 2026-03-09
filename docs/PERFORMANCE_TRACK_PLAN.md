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
