# Engine Embedded Corpus Performance Investigation

Status: active  
Scope: explain why the full embedded equivalence corpus moved from about `4.9s` to about `13s`, separate clean-branch regression from local dirty-tree overhead, and keep the improvements that are still paying for themselves.

## Canonical Question

The user-visible symptom is:

- `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
- Full corpus `docs/embedded_equivalence_context_corpus.csv`
- Current local run: about `13s`
- Historical reference from a few commits earlier: about `4.9s`

The investigation goal is to answer three different questions, not just one:

1. What is the clean regression on committed code?
2. What extra slowdown is coming from the current uncommitted local tree?
3. Which changes improved specific families enough that they should be preserved while the regression is reduced elsewhere?

## Method

All measurements below were taken with the same command family on the same machine:

```sh
cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus
```

For family slices:

```sh
cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family <family>
```

To avoid mixing local edits with committed history, the comparison used:

- current dirty workspace in `/Users/javiergimenezmoya/developer/math`
- clean temporary worktrees under `/tmp` for historical commits

All full-corpus runs below stayed green at `1125/1125`.

## Current Findings

As of `2026-04-21`, the evidence is:

- The observed `~13s` is not clean `main`; it is the current dirty workspace.
- Clean `HEAD` is around `9.2s`, not `13s`.
- The major clean regression happened between `5454eb18` and `f6993ae6`.
- The last few clean commits after `f6993ae6` do not add much more runtime on this corpus.
- The original dirty workspace added another `~3.8s` on top of clean `HEAD`.
- The clean regression is driven mainly by `trig_expand` and `simplify`.
- Several other families actually improved, so a blind rollback would throw away real wins.
- The extra local slowdown was then isolated to a single pocket inside `integrate_prep`.
- After removing that local pocket, the current workspace dropped to `7.95s` with `1125/1125` still green.

## Full Corpus Benchmark Table

| State | Commit | Result |
| --- | --- | ---: |
| current dirty workspace | local uncommitted tree | `12.99s` |
| clean `HEAD` | `ccc56c77` | `9.22s` |
| clean `HEAD~1` | `2e92826a` | `9.22s` |
| clean `HEAD~2` | `35e60615` | `9.10s` |
| clean `HEAD~3` | `f6993ae6` | `9.15s` |
| clean `HEAD~4` | `5454eb18` | `4.88s` |

Immediate read:

- Clean regression: `4.88s -> 9.15s` between `5454eb18` and `f6993ae6`
- Extra local overhead today: `9.22s -> 12.99s`

## Regression Window

The clean jump appears in a single commit window:

- from `5454eb18`
- to `f6993ae6` (`Mejorando`)

High-level diffstat for the main suspect files in that window:

```text
crates/cas_engine/src/eval/simplify_action.rs |  752 ++++++
crates/cas_engine/src/orchestrator.rs         | 3182 +++++++++++++++++++++----
crates/cas_engine/src/rules/arithmetic.rs     | 1952 +++++++++++++--
3 files changed, 5171 insertions(+), 715 deletions(-)
```

This already narrows the clean-regression search space:

- `crates/cas_engine/src/orchestrator.rs`
- `crates/cas_engine/src/rules/arithmetic.rs`
- `crates/cas_engine/src/eval/simplify_action.rs`

## Family Breakdown: Clean Fast Commit vs Clean Regressed Commit

Comparison baseline:

- fast clean commit: `5454eb18`
- regressed clean commit: `f6993ae6`

| Family | `5454eb18` | `f6993ae6` | Delta | Read |
| --- | ---: | ---: | ---: | --- |
| `trig_expand` | `2.18s` | `5.73s` | `+3.55s` | dominant regression |
| `simplify` | `410.70ms` | `1.33s` | `+919.30ms` | second major regression |
| `solve_prep` | `192.54ms` | `203.63ms` | `+11.09ms` | not a global driver |
| `collect` | `10.25ms` | `16.12ms` | `+5.87ms` | small regression |
| `log_contract` | `68.48ms` | `71.75ms` | `+3.27ms` | small regression |
| `power_merge` | `13.39ms` | `15.01ms` | `+1.62ms` | small regression |
| `radical_power` | `7.84ms` | `8.86ms` | `+1.02ms` | small regression |
| `integrate_prep` | `9.54ms` | `10.17ms` | `+0.63ms` | negligible |
| `conditional_factor` | `35.13ms` | `31.97ms` | `-3.16ms` | improved |
| `factor` | `28.99ms` | `24.00ms` | `-4.99ms` | improved |
| `expand` | `94.23ms` | `89.00ms` | `-5.23ms` | improved |
| `fraction_combine` | `79.59ms` | `74.35ms` | `-5.24ms` | improved |
| `nested_fraction` | `73.08ms` | `66.68ms` | `-6.40ms` | improved |
| `fraction_decompose` | `95.28ms` | `88.30ms` | `-6.98ms` | improved |
| `telescoping_fraction` | `89.87ms` | `86.86ms` | `-3.01ms` | improved |
| `log_expand` | `104.41ms` | `93.39ms` | `-11.02ms` | improved |
| `polynomial_product` | `28.84ms` | `15.42ms` | `-13.42ms` | improved |
| `finite_telescoping` | `130.97ms` | `100.79ms` | `-30.18ms` | improved |
| `trig_contract` | `444.14ms` | `401.58ms` | `-42.56ms` | improved |
| `fraction_expand` | `670.33ms` | `596.57ms` | `-73.76ms` | improved |
| `rationalize` | `225.71ms` | `121.41ms` | `-104.30ms` | improved |

Primary conclusion from the clean family split:

- The clean runtime regression is not explained by `solve_prep`.
- The clean runtime regression is mostly `trig_expand` and `simplify`.
- There are real retained wins in `rationalize`, `fraction_expand`, `trig_contract`, `finite_telescoping`, and several fraction/log families.

So the right strategy is not "undo everything". The right strategy is "recover `trig_expand` and `simplify` without discarding the families that got better".

## Dirty Workspace Delta

The current local workspace is slower than clean `HEAD` by about:

- `12.99s - 9.22s = 3.77s`

Spot-check family measurements on the current dirty tree show something important:

| Family | current dirty tree | Reading |
| --- | ---: | --- |
| `simplify` | `104.75ms` | much faster than clean `HEAD`; not the local culprit |
| `trig_expand` | `5.71s` | same band as clean `HEAD`; not the local culprit |
| `fraction_expand` | `590.68ms` | same band as clean `HEAD` |
| `trig_contract` | `403.47ms` | same or slightly better |
| `rationalize` | `117.80ms` | same or slightly better |

This means the extra local `+3.77s` does not appear to come from the obvious pockets already measured.

Current hypothesis for the dirty-tree overhead:

- it likely lives in other families not yet reswept on the dirty tree
- or in interaction cost introduced by the current uncommitted changes in:
  - `crates/cas_engine/src/orchestrator.rs`
  - `crates/cas_engine/src/orchestrator_shortcut_profiler.rs`
  - `crates/cas_engine/src/rules/arithmetic.rs`

Important note on the current runner diff:

- `crates/cas_solver/examples/run_embedded_equivalence_context_corpus.rs` currently switches to a larger-stack worker thread only when orchestrator profiling is enabled
- the normal non-profiled full-corpus command does not go through that path
- so that runner diff is not the first suspect for the normal `12.99s` run

## Dirty Workspace Family Sweep

The next pass compared every family on:

- current dirty workspace before the local fix
- clean `HEAD` at `ccc56c77`

This was the decisive result:

| Family | dirty workspace | clean `HEAD` | Delta |
| --- | ---: | ---: | ---: |
| `integrate_prep` | `5.11s` | `10.32ms` | `+5099.68ms` |
| `fraction_decompose` | `94.40ms` | `85.12ms` | `+9.28ms` |
| `collect` | `16.82ms` | `15.17ms` | `+1.65ms` |
| `power_merge` | `16.18ms` | `15.13ms` | `+1.05ms` |
| `log_contract` | `72.30ms` | `71.27ms` | `+1.03ms` |
| `radical_power` | `10.19ms` | `9.16ms` | `+1.03ms` |
| `simplify` | `97.65ms` | `1.31s` | `-1212.35ms` |
| `solve_prep` | `30.87ms` | `200.74ms` | `-169.87ms` |
| `trig_expand` | `5.70s` | `5.85s` | `-150.00ms` |
| `trig_contract` | `390.89ms` | `418.68ms` | `-27.79ms` |
| `fraction_expand` | `579.97ms` | `601.49ms` | `-21.52ms` |
| `rationalize` | `109.90ms` | `125.12ms` | `-15.22ms` |

Interpretation:

- The extra local `+3.77s` was not broad noise.
- It was overwhelmingly one family: `integrate_prep`.
- Several local changes were still beneficial at the same time, especially `simplify` and `solve_prep`.

This changed the local investigation from "search the whole tree" to "open `integrate_prep` first".

## Local Culprit Isolation

The `integrate_prep` family contains only `16` cases:

- Morrie forward / reverse
- Dirichlet forward / reverse
- wrapped with `additive_passthrough_zero`
- wrapped with `scaled_difference_zero`
- wrapped with `common_denominator_zero`
- wrapped with `shifted_quotient_one`

The dirty-vs-clean comparison on that family alone was:

- dirty workspace before the fix: `5.11s`
- clean `HEAD`: `8.91ms`

The route-level profiler then showed the hot pocket:

- `root.div.03.shifted_quotient_nested_zero_core`: `5734.860ms`
- `root.div.03g2.nested_zero.residual_difference_isolated_zero_fallback`: `5734.558ms`
- `root.div.03g2c3a.nested_zero.residual_difference_trig_double_angle_cos_variant`: `5734.476ms`

That pocket also triggered massive no-match traffic:

- `rule.direct_identity.try.zero_scope_exact_trig_equivalence`: `146988` misses
- `rule.direct_identity.try.expand_trig_sum_to_product`: `20276` misses
- many other `rule.direct_identity.try.*` probes with zero hits

This was consistent with a bad gate, not with a new useful match.

## Causality Experiment

To avoid disturbing the main workspace during diagnosis, a sandbox copy of clean `HEAD` was created and then only these local files were copied over:

- `crates/cas_engine/src/orchestrator.rs`
- `crates/cas_engine/src/orchestrator_shortcut_profiler.rs`
- `crates/cas_engine/src/rules/arithmetic.rs`

That sandbox reproduced the local regression immediately:

- `integrate_prep`: `5.19s`

Then a single experimental patch was applied in the sandbox:

- disable `root.div.03g2c3a.nested_zero.residual_difference_trig_double_angle_cos_variant`

Result:

- `integrate_prep`: `5.19s -> 10.84ms`
- full corpus: `1125/1125`, `7.58s`

This was strong causal evidence that the local slowdown was dominated by that one route.

## Retained Local Fix

After the sandbox confirmation, the same route was removed from the current workspace:

- `crates/cas_engine/src/orchestrator.rs`
- `root.div.03g2c3a.nested_zero.residual_difference_trig_double_angle_cos_variant`

Validation on the current workspace:

- `cargo test -p cas_engine trig_double_angle_cos_variant -- --nocapture`
- `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family integrate_prep`
- `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`

Current retained results:

- `integrate_prep`: `8.77ms`
- full corpus: `1125/1125`, `7.95s`

Important trade-off result:

- the targeted `trig_double_angle_cos_variant` tests still pass after removing this nested-zero route
- this means the expensive route was redundant for the currently retained regression coverage

## Clean Regression Split By File Group

After the local regression was isolated, the clean regression window `5454eb18 -> f6993ae6` was split by file groups on top of the fast base.

Important note:

- `orchestrator.rs` does not compile by itself on top of `5454eb18`
- it already depends on new APIs added in `arithmetic.rs`
- so the smallest comparable orchestrator slice is `orchestrator + arithmetic`

Measured variants:

| Variant on top of `5454eb18` | Full corpus | `simplify` | `trig_expand` |
| --- | ---: | ---: | ---: |
| fast base | `4.99s` | `431.89ms` | `2.24s` |
| clean regressed base | `9.17s` | `1.32s` | `5.90s` |
| `simplify_action` only | `11.10s` | `432.69ms` | `8.66s` |
| `arithmetic` only | `5.43s` | `1.38s` | `1.86s` |
| `simplify_action + arithmetic` | `9.85s` | `1.33s` | `6.15s` |
| `orchestrator + arithmetic` | `4.96s` | `1.30s` | `1.66s` |
| `all three` | `9.45s` | `1.36s` | `5.99s` |

Interpretation:

- `arithmetic.rs` is the main driver of the clean `simplify` regression.
- `simplify_action.rs` is the main driver of the clean `trig_expand` regression.
- `simplify_action + arithmetic` is already enough to reproduce almost the whole clean regression.
- `orchestrator + arithmetic` does **not** reproduce the global slowdown.
- So the clean regression is not primarily an orchestrator problem.

This was the key attribution result:

- `simplify_action.rs` and `arithmetic.rs` together explain almost everything that matters in the clean regression.
- `orchestrator.rs` is coupled to `arithmetic.rs`, but it is not the dominant runtime culprit in this benchmark.

## SimplifyAction Micro-Probe

Once the file-group split identified `simplify_action.rs` as the `trig_expand` driver, the runtime diff inside that file was inspected.

The important observation was that the runtime delta is tiny; most of the file growth is tests. The suspicious runtime line is:

```rust
ctx_simplifier.set_steps_mode(effective_opts.steps_mode);
```

### Probe 1: `simplify_action`-only sandbox

On the `simplify_action`-only sandbox:

- before removing that line:
  - full corpus: `11.10s`
  - `trig_expand`: `8.24s`
- after removing only that line:
  - full corpus: `4.87s`
  - `trig_expand`: `2.24s`

That is almost a perfect return to the fast baseline.

### Probe 2: sandbox mirroring the current local engine state

The same one-line experiment was then applied to the sandbox that mirrors the current local engine state after the retained local fix.

Result:

- full corpus: `7.58-7.95s` down to `3.53s`
- `1125/1125` still green on the corpus

However, the targeted `steps_off` regression suite exposed a likely trade-off:

- `cargo test -p cas_engine eval_simplify_steps_off -- --nocapture`
- many `eval_simplify_steps_off_*` tests passed
- but `eval_simplify_steps_off_handles_hyperbolic_sum_against_telescoping_sum_regression` kept running for over `60s` and did not finish within the observation window

Conclusion:

- `ctx_simplifier.set_steps_mode(effective_opts.steps_mode)` is a major runtime knob
- removing it blindly is not safe enough to retain yet
- the next useful move is a narrower fix around how `steps_mode` is applied, not deleting it globally

## SimplifyAction Narrow Retained Fix

The next probe kept the `steps_mode` line, but narrowed its application instead of deleting it.

Change tested:

- when `effective_opts.steps_mode == Off`
- keep `Off` only if the input expression contains hyperbolic builtins
- otherwise use `Compact` at runtime inside `eval_simplify`

Implementation shape:

- add a cheap AST scan in `crates/cas_engine/src/eval/simplify_action.rs`
- detect `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- map `Off -> Compact` only for non-hyperbolic expressions

Why this shape:

- the long-running regression case mixed a hyperbolic angle-sum identity with a telescoping rational residual
- the broad `Off -> Compact` experiment reopened that pocket
- the big corpus win came mostly from non-hyperbolic `trig_expand`

Sandbox validation:

- targeted regression:
  - `cargo test -p cas_engine eval_simplify_steps_off_handles_hyperbolic_sum_against_telescoping_sum_regression -- --nocapture`
  - passed quickly again
- full `steps_off` regression slice:
  - `cargo test -p cas_engine eval_simplify_steps_off -- --nocapture`
  - `14/14` passed
- family hotspot:
  - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_expand`
  - `1.54s`, `260/260`
- full corpus:
  - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
  - `3.47s`, `1125/1125`

Retained workspace validation:

- `cargo test -p cas_engine eval_simplify_steps_off -- --nocapture`
- `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_expand`
- `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`

Retained result in the real workspace:

- `steps_off` regression slice: `14/14` passed
- `trig_expand`: `1.54s`, `260/260`
- full embedded corpus: `3.53s`, `1125/1125`

Conclusion:

- the issue was not `steps off` in the abstract
- the expensive behavior came from applying `Off` too broadly inside `eval_simplify`
- keeping `Off` only on hyperbolic inputs retains the safety pocket and recovers most of the lost `trig_expand` runtime
- this fix is worth keeping

## Post-Fix Sweep Against `5454eb18`

After retaining the orchestrator route removal and the narrow `simplify_action.rs` gate, the next check was a full family sweep against the old fast baseline commit `5454eb18`.

Full corpus:

- current retained workspace: `3.53s`, `1125/1125`
- `5454eb18`: `4.87s`, `1125/1125`

Selected family comparisons:

- `trig_expand`: `1.55s` vs `2.23s`
- `simplify`: `91.13ms` vs `426.40ms`
- `solve_prep`: `26.79ms` vs `189.06ms`
- `rationalize`: `113.58ms` vs `216.41ms`
- `fraction_expand`: `602.50ms` vs `679.30ms`
- `trig_contract`: `393.56ms` vs `442.97ms`

Families still slightly worse than `5454eb18`:

- `collect`: `14.94ms` vs `9.69ms` (`+5.25ms`)
- `log_contract`: `73.33ms` vs `71.76ms` (`+1.57ms`)
- `nested_fraction`: `66.90ms` vs `65.72ms` (`+1.18ms`)
- `radical_power`: `8.79ms` vs `8.37ms` (`+0.42ms`)

Warm-run check on the largest residual pocket:

- current `collect`: `14.69-16.26ms`
- `5454eb18` `collect`: `8.89-9.51ms`

So `collect` is a real residual gap, but still a very small one in absolute terms.

Follow-up isolation:

- removing the hyperbolic builtin scan in a sandbox while still forcing `Off -> Compact` did **not** improve `collect`
- sandbox `collect` stayed at `14.94-15.93ms`

So the retained `simplify_action.rs` gate is not the source of the `collect` residual.

Current profiling signal on `collect` points elsewhere, but it needs to be read carefully:

- unprofiled `collect` currently stays in the same small band: `14.77ms`
- the current retained full corpus is still healthy: `1125/1125`, `3.33s`
- a fully profiled full-corpus run with the narrow shifted-quotient filters overflowed the stack, so the useful profiler view for this pocket is the guarded `--limit 480` slice, not the whole corpus
- on that `480` slice, the dominant sections are:
  - `root.div.04.shifted_quotient_exact_one_late`: `68.701ms`
  - `root.div.02.shifted_quotient_exact_one.rule_apply`: `58.886ms`
  - `root.div.02.shifted_quotient_exact_one`: `15.230ms`
- an experimental gate around the late `root.div.04` call removed profiled work but did not move the unprofiled `collect` family in a meaningful way, so it was reverted
- with additional `sq1.rule_apply.family.*` detail labels, the `default_simplify` sub-pocket now splits as:
  - `log = 7`
  - `quotient_cancel = 7`
  - `abs_sqrt = 1`
  - `inverse_trig = 1`
  - `hyperbolic = 1`
  - `other_non_hyperbolic = 1`
- with additional `rule.shifted_quotient.exact_one.route.*.family.*` labels, the log traffic also split by route:
  - additive log forms land in `exact_zero_direct_residual`
  - non-additive log forms land in `direct_core_equivalence`
- the `symbolic_scale_sum_rhs` pocket also narrowed further:
  - `power_reciprocal_tail = 2`
  - `linear_reciprocal_tail = 1`
- an experimental early `direct_core_equivalence` fast-path for non-additive log pairs improved the profiled `480` slice locally, but it did not retain unprofiled ROI:
  - full corpus stayed in noise (`3.30-3.32s`)
  - `collect` regressed to `26.86ms`
  - the runtime fast-path was reverted, while the route-level observability was kept
- after reverting that fast-path, the retained state returned to the previous band:
  - full corpus: `1125/1125`, `3.36s`
  - `collect`: `14.25ms`
  - profiled `480` slice: `742.11ms`
  - dominant profiled sections stayed the same:
    - `root.div.04.shifted_quotient_exact_one_late = 112.531ms`
    - `root.div.02.shifted_quotient_exact_one.rule_apply = 98.679ms`
    - `root.div.02.shifted_quotient_exact_one = 16.498ms`
- a narrower retained runtime fix did hold: add a direct `log_expansion` route inside `try_build_direct_core_equivalence_rewrite`, reusing the existing log product/power cancellation matching instead of falling through to `default_simplify`
  - full corpus stayed healthy: `1125/1125`, `3.30-3.31s`
  - `collect` stayed in-band: `13.73ms`
  - profiled `480` slice improved:
    - `742.11ms -> 735.49ms`
    - `root.div.04.shifted_quotient_exact_one_late: 112.531ms -> 99.824ms`
    - `root.div.02.shifted_quotient_exact_one.rule_apply: 98.679ms -> 85.787ms`
    - `rule.shifted_quotient.exact_one.try.direct_core_equivalence: 3.276ms -> 2.661ms`
  - the log traffic split became much cleaner:
    - `sq1.rule_apply.family.log_expansion = 8`
    - `sq1.rule_apply.family.default_simplify.detail.log = 1`
    - `rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.log_expansion = 6`
    - `rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.log_expansion = 2`

So the remaining `collect` residual is still real, but the next useful target is no longer "the late root.div.04 call" in general. It is the narrower `sq1.rule_apply` mix, especially `default_simplify.{log, quotient_cancel}` plus the residual routes already visible in `rule.shifted_quotient.exact_one.try.*`.

Practical interpretation:

- this is a wrapper-specific micro-pocket in shifted-quotient/division handling
- it is not evidence that the retained `steps_mode` fix should be rolled back or narrowed further

Conclusion:

- the old global regression is no longer present
- the retained workspace now beats the old fast baseline on the full corpus and on every large family that previously mattered
- the remaining losses versus `5454eb18` are micro-pockets only
- there is no longer evidence for an urgent top-level runtime regression in the embedded corpus

## What Looks Worth Keeping

The evidence already says some changes are probably valuable and should not be thrown away casually:

- `rationalize` improved by about `104ms`
- `fraction_expand` improved by about `74ms`
- `trig_contract` improved by about `43ms`
- `finite_telescoping` improved by about `30ms`
- several smaller improvements accumulated across fraction/log/polynomial families

This matters because the current tuning work on `solve_prep` can still be valid for its local pocket even though it does not explain the top-level regression.

## Residual Gaps

The original two-bucket regression diagnosis is now mostly historical context.

What remains after the retained fixes:

- no open full-corpus regression against `5454eb18`
- no open `steps_off` regression in the targeted `eval_simplify_steps_off` slice
- only a few sub-`6ms` family regressions remain in:
  - `collect`
  - `log_contract`
  - `nested_fraction`
  - `radical_power`

Those are not urgent performance risks relative to the previous state of the engine.

## Recommended Next Steps

1. Treat the embedded full-corpus regression as recovered unless a new measurement reopens it.
2. Land or preserve the retained fixes together:
   - `crates/cas_engine/src/orchestrator.rs`
   - `crates/cas_engine/src/eval/simplify_action.rs`
3. If more tuning is still wanted, target only the residual micro-pockets in priority order:
   - `collect`
   - `log_contract`
   - `nested_fraction`
   - `radical_power`
4. For the `collect` pocket specifically, do not reopen broad `root.div.04` gating first. The next narrower targets are:
   - the lone residual `sq1.rule_apply.family.default_simplify.detail.log`
   - `sq1.rule_apply.family.default_simplify.detail.quotient_cancel`
   - `rule.shifted_quotient.exact_one.try.shared_passthrough_residual`
   - `rule.shifted_quotient.exact_one.try.exact_zero_direct_residual`
5. Keep using this document as the canonical log so future measurements distinguish clearly between:
   - historical regression analysis
   - retained fixes
   - optional micro-optimizations

## Commands Already Used

Full corpus:

```sh
cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus
```

Family slices:

```sh
cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family simplify
cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_expand
cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_contract
cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family fraction_expand
cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family rationalize
```

Historical clean worktrees:

```sh
git worktree add /tmp/math-bench-head ccc56c77
git worktree add /tmp/math-bench-h1 2e92826a
git worktree add /tmp/math-bench-h2 35e60615
git worktree add /tmp/math-bench-h3 f6993ae6
git worktree add /tmp/math-bench-h4 5454eb18
```

## Update Log

### 2026-04-21

- Created this document as the canonical tracker for the embedded full-corpus regression investigation.
- Confirmed that the user-visible `~13s` comes from the dirty workspace, not from clean `main`.
- Isolated the clean regression window to `5454eb18 -> f6993ae6`.
- Identified `trig_expand` and `simplify` as the dominant clean-regression families.
- Confirmed that several other families improved enough that selective retention matters.
- Ran a complete dirty-vs-clean family sweep and isolated the extra local slowdown to `integrate_prep`.
- Used orchestrator profiling to isolate the hotspot to `root.div.03g2c3a.nested_zero.residual_difference_trig_double_angle_cos_variant`.
- Reproduced the regression in a sandbox by copying only local engine files over clean `HEAD`.
- Confirmed causality by removing only that route in the sandbox: `integrate_prep 5.19s -> 10.84ms`, full corpus `7.58s`.
- Retained the same runtime fix in the current workspace and validated `1125/1125` at `7.95s`.
- Split the clean regression by file group and showed that `simplify_action.rs + arithmetic.rs` reproduce almost the whole clean slowdown, while `orchestrator.rs` is not the dominant culprit.
- Isolated the `simplify_action.rs` runtime suspect to `ctx_simplifier.set_steps_mode(effective_opts.steps_mode)`.
- Confirmed that removing only that line almost restores the fast baseline in sandboxes, but did not retain it because a long-running `steps_off` regression test suggests a real trade-off.
- Retained a narrower `simplify_action.rs` fix: keep `StepsMode::Off` only for expressions containing hyperbolic builtins, map non-hyperbolic `steps_off` evals to `Compact`.
- Validated the retained fix on the real workspace: `eval_simplify_steps_off` `14/14` green, `trig_expand` `1.54s`, full corpus `1125/1125` in `3.53s`.
- Ran a full family sweep against `5454eb18` after the retained fixes and confirmed that the embedded full-corpus regression is no longer present.
- Confirmed that the retained workspace now beats `5454eb18` on the full corpus (`3.53s` vs `4.87s`) and only leaves four micro regressions smaller than `6ms`.
- Re-ran the top residual pocket (`collect`) five times on both sides and confirmed that its `+~5ms` gap is real, but still too small to justify urgent work.
- Isolated the `collect` residual away from the retained `simplify_action.rs` gate: removing the hyperbolic scan in sandbox did not move `collect`, and profiling showed the pocket is dominated by `shifted_quotient_exact_one` on wrapped division cases.
- Tried a late `root.div.04.shifted_quotient_exact_one_late` candidate gate, confirmed that it removed profiled late work but did not improve the unprofiled `collect` family, and reverted it.
- Confirmed that the retained runtime state after reverting that experiment stays healthy: `collect = 14.77ms`, full corpus `1125/1125` in `3.33s`.
- Added narrower `sq1.rule_apply.family.*` observability for the shifted-quotient residual pocket and split the previous `default_simplify` bucket into actionable subgroups, with `log` and `quotient_cancel` dominating the remaining `default_simplify` traffic.
- Added route-level `rule.shifted_quotient.exact_one.route.*.family.*` observability and confirmed that additive log pairs mainly land in `exact_zero_direct_residual`, while non-additive log pairs land in `direct_core_equivalence`.
- Tried an early `direct_core_equivalence` fast-path for the non-additive log sub-pocket, confirmed that it helped only the profiled slice, and reverted it because the unprofiled `collect` family regressed to `26.86ms` without a meaningful full-corpus win.
- Revalidated the retained post-revert state: `shifted_quotient` tests green, `collect = 14.25ms`, full corpus `1125/1125` in `3.36s`, and the narrowed `480` profiler slice still points to `sq1.rule_apply.family.default_simplify.detail.{log, quotient_cancel}` plus the residual shared/direct routes.
- Retained a narrower direct log-expansion route in `try_build_direct_core_equivalence_rewrite`, reusing the existing log product/power matching logic. This kept `collect` healthy (`13.73ms`) and the full corpus at `3.30-3.31s`, while moving most of the old `default_simplify.log` traffic into the explicit `log_expansion` bucket (`8` hits, leaving only `1` residual `default_simplify.log` hit in the profiled `480` slice).
- Tried a narrower `direct_core` additive-collect shortcut for the residual `collect` pocket and reverted it in the same iteration. The experiment matched some full-pair additive collect cases in isolation, but it did not fire on the real residual path that matters in runtime, because the live pocket still flows through `shared_passthrough` on the stripped residual cores, not through a useful early `direct_core` collect route. Revalidation after the revert kept the retained baseline healthy: `shifted_quotient` tests green, `collect = 13.77ms`, full corpus `1125/1125` in `3.22s`.
- Added retained observability on the stripped `shared_passthrough -> tail_direct_core_equivalence` path. This resolved the last ambiguity in the `collect` micro-pocket: the stripped residuals do **not** fall through a hidden `default_simplify` branch. In the `collect` family, the three shifted-quotient cases split as `2 x direct_match` and `1 x symbolic_scale_sum_rhs`, with the surviving sample shape `add(add, mul) || add(mul, mul)`. The coarser `sq1.rule_apply.family.default_simplify.other` bucket on the original wrapped cores is therefore misleading for this pocket; the live stripped residual is already a `symbolic_scale_sum_rhs` case after passthrough extraction. Revalidation kept the tree healthy: `shifted_quotient` tests green, `collect = 26.33ms` cold / `13.77ms` warm, and the full corpus `1125/1125` in `3.12s`.
- Split the stripped `shared_passthrough -> tail_direct_core_equivalence -> symbolic_scale_sum_*` bucket by sub-detail as well. That ruled out the next obvious hypothesis: the live residual is **not** a simple one-pass symbolic distribution case, but `grouped_multi_scale` (`rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_rhs.grouped_multi_scale`) on the same sample shape `add(add, mul) || add(mul, mul)`. In other words, the remaining `collect` micro-pocket is already the grouped residual path, not `single_scale_plain` or a reciprocal-tail subcase. Revalidation stayed healthy: `shifted_quotient` tests green, profiled `collect` still `12/12`, warm `35.13ms` under profiling overhead, and the full corpus `1125/1125` in `3.24s`. A full profiled `1125` run with this filter still overflows the worker stack, so the retained conclusion comes from the `collect` slice and the unprofiled full corpus.
- Retained a very narrow runtime fast-path for that grouped residual pocket. Instead of building the whole distributed grouped expression first, `direct_core_equivalence` now has a structural matcher for `grouped_multi_scale` symbolic-scale sums and uses it before the generic rewrite builder. The retained coverage is the direct grouped pair test plus the full `shifted_quotient` suite (`92` green). The runtime signal is intentionally modest: in the profiled `collect` slice, `rule.shared_passthrough.try.tail_direct_core_equivalence` dropped from roughly `0.361ms` to `0.328ms`, warm unprofiled `collect` stayed in the same healthy band (`14.90ms`), and the full corpus remained `1125/1125` while landing at `3.21s` and `3.14s` on revalidation passes. This is a micro-optimization for the last grouped collect pocket, not a new broad strategy.
- Shifted the next micro-investigation from `collect` to `log_contract`, because the retained `collect` residual was already sub-millisecond while `log_contract` still carried a measurable direct-identity pocket. A cheap gate around `rule.direct_identity.try.expand_log_product_power` did reduce attempts (`312 -> 138`) but did **not** retain ROI in unprofiled family runs, so it was reverted. The retained fix instead reuses the existing `cas_math::try_rewrite_log_chain_product_expr` helper inside `try_build_direct_core_equivalence_rewrite`, adding an explicit `log_chain_product` route for telescoping chain products like `log(b,a) * log(a,c) = log(b,c)`. Revalidation stayed clean: targeted `direct_core_equivalence_rewrite_matches_log_chain_product_pair` coverage plus the full `shifted_quotient` suite green, full corpus `1125/1125` in `3.18s`, and warm `log_contract` back in the good band at `61.41ms`. The profiler confirms the intended movement: the old shifted-quotient sample `mul(function, function) || log(variable, variable)` no longer falls through `rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair`, and now lands in the explicit `rule.direct_core_equivalence.route.log_chain_product` / `sq1.rule_apply.family.log_chain_product` bucket with only `~0.021ms` on the live `direct_core_equivalence` attempt.
- Tried the next obvious extension on top of that: reuse `try_rewrite_log_contraction_expr` as an explicit `direct_core_equivalence` route, with and without a cheap additive-shape gate. The profiler looked superficially better: the residual `default_simplify.family.other.non_hyperbolic.log_pair` bucket dropped from `42` misses / `6.763ms` to `30` misses / `~5.35ms`, and three explicit contraction hits moved into `sq1.rule_apply.family.log_contraction` (`add(function, function) || ln(mul)`, `add(function, neg) || ln(div)`, `add(function, function) || ln(pow)`). But the unprofiled numbers did **not** retain ROI. With the route live, `log_contract` drifted up into the `79-109ms` band and the full corpus rose to `3.28-3.31s`, despite `1125/1125` staying green. I reverted that route entirely and revalidated the retained baseline at `log_contract = 67.69ms`, full corpus `1125/1125` in `3.24s`, and `shifted_quotient` still green. Conclusion: reducing this profiled miss bucket by routing more log pairs through `direct_core_equivalence` is not sufficient; the remaining residual `ln(function) || neg(function)` / `log(variable, variable) || neg(function)` pocket needs a narrower reject or a more specific identity, not a broader contraction route.
- Retained finer observability inside `rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair`, without touching runtime routing. The coarse residual bucket now splits cleanly into two surviving sub-pockets in `log_contract`: `negated_ln.other` (`28` profiled misses, sample `ln(function) || neg(function)`) and `negated_general_base.other` (`14` profiled misses, sample `log(variable, variable) || neg(function)`). That is useful because it rules out a mixed or additive residual: after the retained `log_chain_product` route, what remains is specifically a non-additive `log` vs negated `log` pocket. Revalidation stayed on the retained baseline: `shifted_quotient` green, warm `log_contract = 67.70ms`, full corpus `1125/1125` in `3.21s`. The next step should therefore be a narrow reject or identity for those negated-log pairs, not another broad contraction path.
- Isolated the residual shifted-quotient `pair_id`s down to the concrete corpus rows instead of the coarse profiler bucket. The surviving `negated_ln.other` traffic is exactly `contract_log_grouped_power`, `contract_even_abs_logs_to_scaled_abs_product`, and their passthrough variants; the surviving `negated_general_base.other` traffic is exactly `contract_general_base_logs_to_grouped_power` and its passthrough variant. The other shifted-quotient `log_contract` rows (`contract_log_sum`, `contract_log_difference`, powered-denominator factors, change-of-base chain) do **not** contribute to this pocket. That matters because it narrows the target from “generic negated log pairs” to grouped-power / grouped-abs-product contraction shapes only.
- Retained a narrower follow-up on top of that isolation, but only for the `ln`/`abs` half of the pocket. `try_build_direct_log_expansion_equivalence_rewrite` now does two extra things: it compares the distributed `scale * (log(lhs) +/- log(rhs))` form against additive targets, and it allows an extra `exprs_match_after_default_simplify(...)` step only when both sides use natural logs (`ln`) rather than general-base `log`. Coverage added: grouped `ln((x*y)^2)` vs `ln(x^2)+ln(y^2)`, scaled `2*ln(abs(x*y))` vs `2*ln(abs(x))+2*ln(abs(y))`, plus a guard that the general-base grouped-power pair still does **not** become a global `direct_core_equivalence` rewrite. The full `shifted_quotient` suite stayed green (`92` tests).
- The retained runtime trade-off is acceptable, but the stable benefit is smaller than the first hot run suggested. On isolated shifted-quotient rows, `contract_log_grouped_power` and `contract_even_abs_logs_to_scaled_abs_product` now classify as `sq1.rule_apply.family.log_expansion` / `route.exact_zero_direct_residual.family.log_expansion`, while `contract_general_base_logs_to_grouped_power` stays in `sq1.rule_apply.family.other` with the same `negated_general_base.other` residual. The best early revalidation passes landed around `3.03-3.07s`, but after a full rebuild and a failed follow-up experiment/revert (see next bullet), the retained state revalidated more conservatively at `log_contract = 65.28ms` warm and the full corpus `1125/1125` in `3.18s`. That still keeps the patch worth retaining: it is correct, keeps `shifted_quotient` green, and leaves the next pocket cleaner (`negated_general_base.other`, not the grouped `ln` / `abs` cases).
- Tried the obvious wrapper-specific follow-up on that cleaner residual: a `shifted_quotient_exact_one`-only matcher for the grouped general-base pair `log(b,(x*y)^2)` vs `2*log(b,x)+2*log(b,y)`, with passthrough support. The experiment was intentionally narrow and did hit exactly the intended sample (`rule.shifted_quotient.exact_one.route.grouped_general_base_log_power` on `add(mul, mul) || log(variable, pow)`), but it did **not** retain ROI. Across the whole `log_contract` family it produced only `1` hit and `7` misses, `log_contract` stayed worse than the retained baseline (`77.46ms`), and the full corpus drifted to `3.16s`. I reverted that route and its tests, then revalidated the retained state above. Conclusion: the remaining `negated_general_base.other` pocket is not fixed by a small wrapper-local grouped-power matcher; the next step still needs a different narrow idea, or the residual is small enough to leave as backlog.
