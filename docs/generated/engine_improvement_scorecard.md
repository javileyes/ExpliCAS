# Engine Improvement Scorecard

- Generated: 2026-05-24T16:27:33.343408+00:00
- Git branch: main
- Git commit: `58df39ec2e0b26038f155f6d4da6c0fbac215cae`
- Profile: `guardrail`

## Embedded Runtime Guardrail

- Dimension: contextual simplify/equivalence under real wrappers.
- Interpretation: strong for simplify/orchestration quality; not a derive-path metric.
- Elapsed: 13.38s
- Per-case runtime: 8.992ms/case
- Coverage axes: 7 wrappers across 27 families
- Context axes: 4 complexity levels across 6 shell depths
- Largest wrapper share: 19.8%
- Largest wrapper x complexity share: 16.9%
- Combined additive composition: total=177 families=27 collapse_rows=159 collapse_families=27/27 depth4_rows=39 depth4_families=27/27 orientation_rows=48 orientation_families=27/27 multi_core_rows=45 multi_core_families=27/27 min_family_case_count=6 target_family_case_count=6 families_at_min=21/27 families_under_target=0/27 low_family_counts=calculus_diff:6, conditional_factor:6, expand:6, factor:6, finite_telescoping:6, fraction_combine:6, fraction_decompose:6, fraction_expand:6, log_contract:6, log_exp_inverse:6, log_expand:6, log_inverse_power:6, nested_fraction:6, polynomial_product:6, power_merge:6, radical_power:6, rationalize:6, solve_prep:6, telescoping_fraction:6, trig_contract:6, trig_expand:6 above_min_family_counts=calculus_integrate:8, collect:7, finite_aggregate:7, integrate_prep:7, number_theory:7, simplify:15
- Embedded coverage saturation: balanced (5/5 checks); defer embedded corpus padding unless a new reproducible failure appears; prefer calculus, robustness, or presentation candidates.
- Wrappers: additive_passthrough_zero, combined_additive_zero, common_denominator_zero, reciprocal_shifted_difference_zero, scaled_difference_zero, shifted_quotient_one, squared_passthrough_zero
- Sparse wrappers: reciprocal_shifted_difference_zero total=60 failed=0
- Structural depth: max_shell_depth=5 max_expression_depth=18 avg_wrapper_overhead_nodes=13.49
- Complexity mix: l2_wrapper_plus_noise total=1016 failed=0 avg_shell_depth=1.90, l3_nested_or_composed total=457 failed=0 avg_shell_depth=2.62, l1_single_wrapper total=10 failed=0 avg_shell_depth=1.00
- Shell-depth mix: depth 0 total=5 failed=0, depth 1 total=141 failed=0, depth 2 total=1126 failed=0, depth 3 total=124 failed=0, depth 4 total=88 failed=0, depth 5 total=4 failed=0
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
- Observe-only discoveries: total=1
- By area: orchestrator / exact-zero additive composition:1
- Recent 1: `orchestrator / exact-zero additive composition` - 2026-05-24 - Discovery observe-only: hyperbolic angle-sum plus telescoping residual remains slow

## Derive Reachability Guardrail

- Dimension: source-to-target bridgeability and path quality.
- Interpretation: measures planner/strategy reachability; not contextual wrapper strength.
- Expected-status breakdown: derived=390 unsupported=0 not_equivalent=1
- Path quality: mean_step_count=1.05 long_path_rate=0.00 single_step_successes=371 multi_step_successes=19 max_step_count=3
- Strategy specificity: generic_simplify_expected=0 distinct_expected_strategies=28
- Expected strategy counts: expand trig:94, contract trig:48, rewrite hyperbolics:35, expand:31, finite sums/products:22, expand_log:17, contract logs:14, factor:13, rewrite exponentials:13, rewrite trigs:13, expand fraction:11, combine fraction:10, +16 more
- Non-derived expected families: unsupported=none not_equivalent=negative:1

## Derive Shadow Pressure

- Dimension: diagnostic engine-to-derive bridgeability over representative engine equivalence rows.
- Interpretation: exposes where known engine/metamorphic identities are not yet reachable or provable as derive targets; diagnostic, not a support gate.
- Outcomes: sampled=67 derived=67 unsupported=0 not_equivalent=0
- Path signal: mean_step_count=1.09 single_step_successes=62 multi_step_successes=5 max_step_count=4
- Strategy specificity: generic_simplify_strategy_successes=0 distinct_actual_strategies=29
- Embedded family coverage: sampled_families=26 total_families=26 missing=none
- Multi-step shadow IDs: identity_log_product_root_cleanup:2, identity_nested_radical_denesting:2, identity_log_inverse_power_tower:2, embedded_calculus_diff_arctan_sqrt_compact:4, embedded_log_inverse_power_unary_natural_alias:2
- Generic simplify shadow IDs: none
- Actual strategy counts: finite sums/products:11, expand trig:4, number theory:4, expand:3, expand fraction:3, rationalize:3, rewrite exponentials:3, rewrite hyperbolics:3, calculus diff:2, combine fraction:2, combine powers:2, contract logs:2, +17 more
- Derived family counts: Fundamental exp decomposition:3, Constant sum:2, Pythagorean Identities:2, calculus_diff:2, log_exp_inverse:2, Addition of fractions:1, Binomial Expansion:1, Binomial coefficients (choose function):1, Change of base:1, Completing the square patterns:1, Difference of Squares:1, Exponential of logs:1, +49 more
- Problem family counts: unsupported=none not_equivalent=none

## Derive Didactic Trace Audit

- Dimension: educational quality of derive target-driven traces and web substeps.
- Interpretation: diagnostic trace-quality lane; not a semantic correctness or runtime metric.
- Outcomes: cases=472 flagged=0 flagged_rate=0.0%
- Substeps: total_web_substeps=485 mean_step_count=1.06
- Flags: no_web_substeps=0 no_web_steps=0
- Runtime split: artifacts=1.04s cli=0.99s report=0.00s total=2.04s workers=4
- Runtime family hotspots: artifacts=expand:1.525,rationalize:0.612,solve_prep:0.238,finite_telescoping:0.227,trig_contract:0.209,factor:0.198 cli=expand:1.498,rationalize:0.603,trig_contract:0.220,solve_prep:0.215,finite_telescoping:0.198,factor:0.192
- Runtime case hotspots: artifacts=expand_hyperbolic_product_to_sum_to_cosh_cubic_polynomial_with_passthrough:0.677,expand_hyperbolic_product_to_sum_to_cosh_cubic_polynomial:0.642,rationalize_symbolic_linear_root_alt_var:0.251,rationalize_symbolic_linear_root:0.251,finite_aggregate_sum_of_cubes_symbolic_lower_bound:0.107,solve_prep_complete_square_negative_symbolic_leading_coeff:0.076 cli=expand_hyperbolic_product_to_sum_to_cosh_cubic_polynomial_with_passthrough:0.666,expand_hyperbolic_product_to_sum_to_cosh_cubic_polynomial:0.644,rationalize_symbolic_linear_root_alt_var:0.252,rationalize_symbolic_linear_root:0.247,finite_aggregate_sum_of_cubes_symbolic_lower_bound:0.100,solve_prep_complete_square_negative_symbolic_leading_coeff:0.077

## Simplify Didactic Trace Audit

- Dimension: educational quality of simplify step traces and web substeps.
- Interpretation: diagnostic trace-quality lane; not a semantic correctness or runtime metric.
- Outcomes: cases=14 flagged=0 flagged_rate=0.0%
- Substeps: total_wire_substeps=24 mean_step_count=2.07
- Flags: no_wire_substeps=0 single_step_no_substeps=0 missing_math_sides=0

## Calculus Contract Signal

- Dimension: public calculus behavior, result simplification, domain conditions, and step noise.
- Interpretation: small executable calculus vertical slices; failures should be classified before broadening pre-calculus rules.
- `diff`: passed=257 failed=0 ignored=1 filtered_out=0
- `diff` ignored tests: `inverse_reciprocal_trig_diff_evaluates_with_explicit_domain_conditions_exhaustive` (exhaustive inverse reciprocal trig diff contract is debug-slow; CI keeps representative structural smoke)
- `limit`: passed=143 failed=0 ignored=0 filtered_out=0
- `limit_presimplify_safe`: passed=8 failed=0 ignored=0 filtered_out=0
- `residual_matrix`: passed=706 failed=0 total=706 slow=0 timeouts=0 total_bases=79 wrapped_bases=57 standalone_bases=22 wrappers=12 wrapped_cases=684 standalone_cases=22 conditioned_cases=699 distinct_conditions=20
- `residual_matrix` sparse expected conditions: -1 < x < 1=12, 1 - x^2=12, 4 - (x + 1)^2=12, cos(1 - 2·x)=12, sin(sqrt(3 - 2·x))=12
- `residual_matrix` domain expected conditions: -1 < x < 1=12, x < 3/2=12, x > 0=24, x > -1/3=52
- `residual_matrix` wrapper coverage: expected_wrapped_cases=684 missing_wrapped_pairs=0 full_wrapper_bases=57 partial_wrapper_bases=0 largest_gap=0
- `integrate`: passed=337 failed=0 ignored=1 filtered_out=0
- `integrate` ignored tests: `integrate_contract_supported_antiderivatives_verify_by_differentiation_exhaustive` (exhaustive debug verification is intentionally slower; CI runs the representative smoke test)

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
| `embedded_equivalence_context` | `pass` | 13.38s | passed=1488 failed=0 total=1488 wrappers=7 families=27 avg_case=8.992ms |
| `derive_contract` | `pass` | 2.41s | derived=390 unsupported=0 expected_not_equivalent=1 mean_step_count=1.05 generic_simplify_expected=0 single_step=371 multi_step_ids=19 max_step=3 |
| `derive_shadow_pressure` | `pass` | 0.32s | sampled=67 derived=67 unsupported=0 not_equivalent=0 mean_step_count=1.09 generic_simplify_strategy_successes=0 single_step=62 multi_step_ids=5 max_step=4 embedded_families=26/26 |
| `derive_didactic_audit` | `pass` | 33.48s | cases=472 flagged=0 no_web_substeps=0 no_web_steps=0 artifacts=1.04s cli=0.99s workers=4 top_artifact_family=expand:1.525 top_cli_family=expand:1.498 |
| `simplify_didactic_audit` | `pass` | 1.29s | cases=14 flagged=0 no_wire_substeps=0 missing_math_sides=0 |
| `simplify_strict` | `pass` | 26.45s | closure=100.0% nf=0 (0.0%) proved=16519 (100.0%) numeric=0 inconclusive=0 timeouts=0 |
| `calculus_diff_contract` | `pass` | 5.95s | passed=257 failed=0 ignored=1 |
| `calculus_limit_contract` | `pass` | 0.72s | passed=143 failed=0 |
| `calculus_limit_presimplify_contract` | `pass` | 0.48s | passed=8 failed=0 |
| `calculus_residual_matrix_smoke` | `pass` | 12.07s | passed=706 failed=0 total=706 conditioned=699 conditions=20 total_bases=79 wrapped_bases=57 standalone_bases=22 wrappers=12 missing_wrapped_pairs=0 partial_wrapper_bases=0 |
| `calculus_integrate_contract` | `pass` | 8.61s | passed=337 failed=0 ignored=1 |
