# Engine Improvement Scorecard

- Generated: 2026-05-14T14:37:35.237845+00:00
- Git branch: main
- Git commit: `b438ff94a7a825d5087cc1163675cde420a2d8bb`
- Profile: `guardrail`

## Embedded Runtime Guardrail

- Dimension: contextual simplify/equivalence under real wrappers.
- Interpretation: strong for simplify/orchestration quality; not a derive-path metric.
- Elapsed: 7.78s
- Per-case runtime: 5.232ms/case
- Coverage axes: 7 wrappers across 27 families
- Context axes: 4 complexity levels across 6 shell depths
- Largest wrapper share: 19.8%
- Largest wrapper x complexity share: 16.9%
- Combined additive composition: total=177 families=27 collapse_rows=159 collapse_families=27/27 depth4_rows=39 depth4_families=27/27 orientation_rows=48 orientation_families=27/27 multi_core_rows=45 multi_core_families=27/27 min_family_case_count=6 target_family_case_count=6 families_at_min=21/27 families_under_target=0/27 low_family_counts=calculus_diff:6, conditional_factor:6, expand:6, factor:6, finite_telescoping:6, fraction_combine:6, fraction_decompose:6, fraction_expand:6, log_contract:6, log_exp_inverse:6, log_expand:6, log_inverse_power:6, nested_fraction:6, polynomial_product:6, power_merge:6, radical_power:6, rationalize:6, solve_prep:6, telescoping_fraction:6, trig_contract:6, trig_expand:6 above_min_family_counts=calculus_integrate:8, collect:7, finite_aggregate:7, integrate_prep:7, number_theory:7, simplify:15
- Embedded coverage saturation: balanced (5/5 checks); defer embedded corpus padding unless a new reproducible failure appears; prefer calculus, robustness, or presentation candidates.
- Wrappers: additive_passthrough_zero, combined_additive_zero, common_denominator_zero, reciprocal_shifted_difference_zero, scaled_difference_zero, shifted_quotient_one, squared_passthrough_zero
- Sparse wrappers: reciprocal_shifted_difference_zero total=60 failed=0
- Structural depth: max_shell_depth=5 max_expression_depth=18 avg_wrapper_overhead_nodes=13.49
- Complexity mix: l2_wrapper_plus_noise total=1016 failed=0 avg_shell_depth=1.90, l3_nested_or_composed total=456 failed=0 avg_shell_depth=2.62, l1_single_wrapper total=10 failed=0 avg_shell_depth=1.00
- Shell-depth mix: depth 0 total=5 failed=0, depth 1 total=141 failed=0, depth 2 total=1125 failed=0, depth 3 total=124 failed=0, depth 4 total=88 failed=0, depth 5 total=4 failed=0
- Sparse wrapper x shell-depth buckets: reciprocal_shifted_difference_zero x depth 2 total=3 failed=0, reciprocal_shifted_difference_zero x depth 3 total=30 failed=0, reciprocal_shifted_difference_zero x depth 4 total=27 failed=0
- Sparse wrapper x shell-depth family buckets: reciprocal_shifted_difference_zero x depth 2 x calculus_integrate total=1 failed=0, reciprocal_shifted_difference_zero x depth 2 x collect total=1 failed=0, reciprocal_shifted_difference_zero x depth 2 x fraction_decompose total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x calculus_integrate total=5 failed=0, reciprocal_shifted_difference_zero x depth 3 x calculus_diff total=2 failed=0, reciprocal_shifted_difference_zero x depth 3 x conditional_factor total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x expand total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x factor total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x finite_aggregate total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x finite_telescoping total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x fraction_combine total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x fraction_expand total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x integrate_prep total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x log_contract total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x log_exp_inverse total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x log_expand total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x log_inverse_power total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x nested_fraction total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x number_theory total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x polynomial_product total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x power_merge total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x radical_power total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x rationalize total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x simplify total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x solve_prep total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x telescoping_fraction total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x trig_contract total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x trig_expand total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x calculus_diff total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x calculus_integrate total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x collect total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x conditional_factor total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x expand total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x factor total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x finite_aggregate total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x finite_telescoping total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x fraction_combine total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x fraction_decompose total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x fraction_expand total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x integrate_prep total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x log_contract total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x log_exp_inverse total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x log_expand total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x log_inverse_power total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x nested_fraction total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x number_theory total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x polynomial_product total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x power_merge total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x radical_power total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x rationalize total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x simplify total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x solve_prep total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x telescoping_fraction total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x trig_contract total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x trig_expand total=1 failed=0
- Sparse wrapper depth 4 family breadth: reciprocal_shifted_difference_zero depth4_families=27/27 missing=0 cases=27
- Sparse wrapper noise budgets: reciprocal_shifted_difference_zero total=60 failed=0 avg_overhead=20.15 max_overhead=36
- Sparse wrapper x complexity buckets: reciprocal_shifted_difference_zero x l3_nested_or_composed total=60 failed=0 avg_shell_depth=3.40
- Dominant wrapper x complexity buckets: common_denominator_zero x l2_wrapper_plus_noise total=252 failed=0 avg_shell_depth=2.00, scaled_difference_zero x l2_wrapper_plus_noise total=252 failed=0 avg_shell_depth=1.87, shifted_quotient_one x l2_wrapper_plus_noise total=252 failed=0 avg_shell_depth=1.88
- Sparse wrapper l3 family breadth: reciprocal_shifted_difference_zero l3_families=27/27 missing=0 cases=60
- Sparse wrapper family breadth: reciprocal_shifted_difference_zero families=27/27 cases=60
- Sparse wrapper family gaps: reciprocal_shifted_difference_zero missing_families=0/27 covered=27 cases=60
- Sparse wrapper x family buckets: reciprocal_shifted_difference_zero x calculus_integrate total=7 failed=0, reciprocal_shifted_difference_zero x calculus_diff total=3 failed=0, reciprocal_shifted_difference_zero x collect total=2 failed=0, reciprocal_shifted_difference_zero x conditional_factor total=2 failed=0, reciprocal_shifted_difference_zero x expand total=2 failed=0, reciprocal_shifted_difference_zero x factor total=2 failed=0, reciprocal_shifted_difference_zero x finite_aggregate total=2 failed=0, reciprocal_shifted_difference_zero x finite_telescoping total=2 failed=0, reciprocal_shifted_difference_zero x fraction_combine total=2 failed=0, reciprocal_shifted_difference_zero x fraction_decompose total=2 failed=0, reciprocal_shifted_difference_zero x fraction_expand total=2 failed=0, reciprocal_shifted_difference_zero x integrate_prep total=2 failed=0, reciprocal_shifted_difference_zero x log_contract total=2 failed=0, reciprocal_shifted_difference_zero x log_exp_inverse total=2 failed=0, reciprocal_shifted_difference_zero x log_expand total=2 failed=0, reciprocal_shifted_difference_zero x log_inverse_power total=2 failed=0, reciprocal_shifted_difference_zero x nested_fraction total=2 failed=0, reciprocal_shifted_difference_zero x number_theory total=2 failed=0, reciprocal_shifted_difference_zero x polynomial_product total=2 failed=0, reciprocal_shifted_difference_zero x power_merge total=2 failed=0, reciprocal_shifted_difference_zero x radical_power total=2 failed=0, reciprocal_shifted_difference_zero x rationalize total=2 failed=0, reciprocal_shifted_difference_zero x simplify total=2 failed=0, reciprocal_shifted_difference_zero x solve_prep total=2 failed=0, reciprocal_shifted_difference_zero x telescoping_fraction total=2 failed=0, reciprocal_shifted_difference_zero x trig_contract total=2 failed=0, reciprocal_shifted_difference_zero x trig_expand total=2 failed=0

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Derive Reachability Guardrail

- Dimension: source-to-target bridgeability and path quality.
- Interpretation: measures planner/strategy reachability; not contextual wrapper strength.
- Outcomes: derived=390 unsupported=0 not_equivalent=1
- Path quality: mean_step_count=1.05 long_path_rate=0.00
- Strategy specificity: generic_simplify_expected=0 distinct_expected_strategies=28

## Derive Shadow Pressure

- Dimension: diagnostic engine-to-derive bridgeability over representative engine equivalence rows.
- Interpretation: exposes where known engine/metamorphic identities are not yet reachable or provable as derive targets; diagnostic, not a support gate.
- Outcomes: sampled=66 derived=66 unsupported=0 not_equivalent=0
- Path signal: mean_step_count=1.11 single_step_successes=61 multi_step_successes=5
- Strategy specificity: generic_simplify_strategy_successes=0 distinct_actual_strategies=28
- Embedded family coverage: sampled_families=26 total_families=26 missing=none
- Multi-step shadow IDs: identity_log_product_root_cleanup:2, identity_nested_radical_denesting:2, identity_log_inverse_power_tower:2, embedded_calculus_diff_bounded_arcsin_residual:5, embedded_log_inverse_power_unary_natural_alias:2
- Generic simplify shadow IDs: none
- Actual strategy counts: finite sums/products:11, expand trig:4, number theory:4, expand:3, expand fraction:3, rationalize:3, rewrite exponentials:3, rewrite hyperbolics:3, rewrite radicals:3, combine fraction:2, combine powers:2, contract logs:2, +16 more
- Derived family counts: Fundamental exp decomposition:3, Constant sum:2, Pythagorean Identities:2, log_exp_inverse:2, Addition of fractions:1, Binomial Expansion:1, Binomial coefficients (choose function):1, Change of base:1, Completing the square patterns:1, Difference of Squares:1, Exponential of logs:1, Factor-cancel patterns:1, +49 more
- Problem family counts: unsupported=none not_equivalent=none

## Derive Didactic Trace Audit

- Dimension: educational quality of derive target-driven traces and web substeps.
- Interpretation: diagnostic trace-quality lane; not a semantic correctness or runtime metric.
- Outcomes: cases=472 flagged=0 flagged_rate=0.0%
- Substeps: total_web_substeps=482 mean_step_count=1.06
- Flags: no_web_substeps=0 no_web_steps=0

## Simplify Didactic Trace Audit

- Dimension: educational quality of simplify step traces and web substeps.
- Interpretation: diagnostic trace-quality lane; not a semantic correctness or runtime metric.
- Outcomes: cases=14 flagged=0 flagged_rate=0.0%
- Substeps: total_wire_substeps=24 mean_step_count=2.07
- Flags: no_wire_substeps=0 single_step_no_substeps=0 missing_math_sides=0

## Calculus Contract Signal

- Dimension: public calculus behavior, result simplification, domain conditions, and step noise.
- Interpretation: small executable calculus vertical slices; failures should be classified before broadening pre-calculus rules.
- `diff`: passed=169 failed=0 ignored=1 filtered_out=0
- `limit`: passed=67 failed=0 ignored=0 filtered_out=0
- `limit_presimplify_safe`: passed=8 failed=0 ignored=0 filtered_out=0
- `integrate`: passed=230 failed=0 ignored=1 filtered_out=0

## Simplify Closure Signal

- Dimension: normal-form convergence vs symbolic residual proof in the unified simplification benchmark.
- Closure: symbolic=16519/16519 (100.0%), NF=0 (0.0%), proved-only=16519 (100.0%).
- Residual outcomes: numeric_only=0 (0.0%), inconclusive=0 (0.0%), timeouts=0.
- NF gap: 16519 cases are proved symbolically but do not converge to the same normal form.

## Simplify Benchmark Interpretation

- Dimension: broad semantic closure under metamorphic composition.
- Proof-shape mix: quotient=3498 diff=0 composed=13021 (non-composed 21.2%, composed 78.8%).
- Caveat: high `proved-composed` is a strong semantic/robustness signal, but a weaker direct runtime proxy because part of the closure is shaped by the benchmark harness rather than by one raw engine path.
- Runtime interpretation: use `embedded_equivalence_context`, `simplify_zero_mixed`, and orchestrator profiling to localize real engine cost.

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `embedded_equivalence_context` | `pass` | 7.78s | passed=1487 failed=0 total=1487 wrappers=7 families=27 avg_case=5.232ms |
| `derive_contract` | `pass` | 5.23s | derived=390 unsupported=0 not_equivalent=1 mean_step_count=1.05 generic_simplify_expected=0 |
| `derive_shadow_pressure` | `pass` | 0.31s | sampled=66 derived=66 unsupported=0 not_equivalent=0 mean_step_count=1.11 generic_simplify_strategy_successes=0 single_step=61 multi_step_ids=5 embedded_families=26/26 |
| `derive_didactic_audit` | `pass` | 37.45s | cases=472 flagged=0 no_web_substeps=0 no_web_steps=0 |
| `simplify_didactic_audit` | `pass` | 0.98s | cases=14 flagged=0 no_wire_substeps=0 missing_math_sides=0 |
| `simplify_strict` | `pass` | 20.09s | closure=100.0% nf=0 (0.0%) proved=16519 (100.0%) numeric=0 inconclusive=0 timeouts=0 |
| `calculus_diff_contract` | `pass` | 5.09s | passed=169 failed=0 |
| `calculus_limit_contract` | `pass` | 0.94s | passed=67 failed=0 |
| `calculus_limit_presimplify_contract` | `pass` | 0.66s | passed=8 failed=0 |
| `calculus_integrate_contract` | `pass` | 6.03s | passed=230 failed=0 |
