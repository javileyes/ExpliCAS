# Engine Improvement Scorecard

- Generated: 2026-05-25T17:29:49.821353+00:00
- Git branch: main
- Git commit: `0e0f8b17d110820555956b4c650b7ef98a3260ed`
- Profile: `guardrail`

## Embedded Runtime Guardrail

- Dimension: contextual simplify/equivalence under real wrappers.
- Interpretation: strong for simplify/orchestration quality; not a derive-path metric.
- Elapsed: 10.86s
- Per-case runtime: 7.298ms/case
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
- Observe-only discoveries: total=4
- By area: calculus / limit:2, calculus / differentiation:1, orchestrator / exact-zero additive composition:1
- Recent 1: `calculus / differentiation` - 2026-05-25 - Observe-only discovery: inverse-tangent symbolic denominator shortcut leaves condition and external-scale gaps
- Recent 2: `calculus / limit` - 2026-05-24 - Observe-only discovery: limit command residuals cannot enter algebraic wrapper matrix
- Recent 3: `orchestrator / exact-zero additive composition` - 2026-05-24 - Discovery observe-only: hyperbolic angle-sum plus telescoping residual remains slow

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
- Runtime split: artifacts=0.89s cli=0.82s report=0.00s total=1.71s workers=4
- Runtime family hotspots: artifacts=expand:1.263,rationalize:0.521,solve_prep:0.183,finite_telescoping:0.181,trig_contract:0.172,factor:0.166 cli=expand:1.251,rationalize:0.507,trig_contract:0.166,factor:0.163,solve_prep:0.163,finite_telescoping:0.156
- Runtime case hotspots: artifacts=expand_hyperbolic_product_to_sum_to_cosh_cubic_polynomial_with_passthrough:0.565,expand_hyperbolic_product_to_sum_to_cosh_cubic_polynomial:0.538,rationalize_symbolic_linear_root_alt_var:0.214,rationalize_symbolic_linear_root:0.211,finite_aggregate_sum_of_cubes_symbolic_lower_bound:0.088,expand_log_general_base_powered_two_denominator_factors_with_powered_denominator:0.063 cli=expand_hyperbolic_product_to_sum_to_cosh_cubic_polynomial_with_passthrough:0.558,expand_hyperbolic_product_to_sum_to_cosh_cubic_polynomial:0.537,rationalize_symbolic_linear_root:0.209,rationalize_symbolic_linear_root_alt_var:0.207,finite_aggregate_sum_of_cubes_symbolic_lower_bound:0.087,factor_geometric_difference_power_6:0.063

## Simplify Didactic Trace Audit

- Dimension: educational quality of simplify step traces and web substeps.
- Interpretation: diagnostic trace-quality lane; not a semantic correctness or runtime metric.
- Outcomes: cases=14 flagged=0 flagged_rate=0.0%
- Substeps: total_wire_substeps=24 mean_step_count=2.07
- Flags: no_wire_substeps=0 single_step_no_substeps=0 missing_math_sides=0

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff`: passed=259 failed=0 ignored=1 filtered_out=0
- `diff` ignored tests: `inverse_reciprocal_trig_diff_evaluates_with_explicit_domain_conditions_exhaustive` (exhaustive inverse reciprocal trig diff contract is debug-slow; CI keeps representative structural smoke)
- `diff_command_matrix`: passed=40 failed=0 total=40 slow=0 timeouts=0 supported_cases=29 residual_cases=5 warning_expected=0 required_display=24 step_checked=37 unchecked_supported_steps=0 families=20
- `diff_command_matrix` argument regimes: affine_argument=3, affine_argument_with_trig_pole=1, affine_exponent=3, constant=1, dual_symbolic_denominator_scaled_nested_root=1, external_scale_symbolic_rational_denominator_affine_radicand=1, negated_nested_root=1, negative_affine_argument=1
- `diff_command_matrix` domain regimes: discontinuous_residual=1, empty_open_interval_domain=4, empty_positive_argument_domain=2, empty_positive_exponent_domain=1, interval_required=1, invalid_log_base_domain=1, negative_base_undefined=1, nonfinite_undefined=1
- `diff_command_matrix` required displays: 1 - (a·(x + 1)^(1/2))^2 > 0=1, a > 0=1, a ≠ 0=6, cos(2·x + 1) ≠ 0=1, cos(sqrt(x)) ≠ 0=1, x < 1=1, x < 1/2=1, x > -1=5
- `diff_command_matrix` outcomes: residual=5, supported=29, undefined=6
- `diff_command_matrix` trace regimes: chain_rule=6, constant_base_exponential_chain_rule=1, constant_multiple_chain_rule=1, constant_multiple_parameter_denominator_scaled_affine_radicand_chain_rule=1, dual_parameter_scaled_chain_rule=1, empty_open_interval_policy=4, exponential_chain_rule=1, log_abs_trig_chain_rule=1
- `diff_command_matrix` presentation regimes: canonical=5, compact_quotient=2, constant_zero=1, external_scale_symbolic_rational_denominator_affine_radicand_compact=1, factored=2, open_interval_boundary_deduped_compact=1, positive_gap_domain_condition_compact=1, post_calculus_compact=1
- `limit`: passed=151 failed=0 ignored=0 filtered_out=0
- `limit_presimplify_safe`: passed=8 failed=0 ignored=0 filtered_out=0
- `limit_command_matrix`: passed=20 failed=0 total=20 slow=0 timeouts=0 supported_cases=13 residual_cases=6 warning_expected=6 required_display=11 step_checked=20 unchecked_supported_steps=0 families=13
- `limit_command_matrix` point regimes: finite=15, infinity=5
- `limit_command_matrix` domain regimes: argument_zero_boundary_residual=1, discontinuous_residual=1, domain_path_conflict=1, empty_real_domain=1, endpoint_residual=2, removable_hole=1, required_condition=3, root_interior_interval_required=1
- `limit_command_matrix` required displays: 2·x^2 - 3 ≠ 0=1, x > -1=1, x > -2=1, x > -3=1, x > 0=2, x ≠ -1=1, x ≠ -2=1, x ≠ 0=1
- `limit_command_matrix` outcomes: residual=6, supported=13, undefined=1
- `limit_command_matrix` trace regimes: binary_log_finite_substitution=1, bounded_over_divergent_policy=1, exponential_tail_policy=1, finite_residual_policy=5, finite_special_angle_table=1, infinity_residual_policy=1, inverse_trig_root_finite_substitution=1, nested_positive_domain_substitution=1
- `limit_command_matrix` presentation regimes: binary_log=1, canonical=7, canonical_no_required_conditions=2, exact_radical=1, infinity=1, residual=5, residual_presentation_cleanup=1, special_angle=1
- `integrate_command_matrix`: passed=39 failed=0 total=39 slow=0 timeouts=0 supported_cases=28 residual_cases=9 warning_expected=0 required_display=25 step_checked=39 unchecked_supported_steps=0 antiderivative_verified=28 families=31
- `integrate_command_matrix` argument regimes: additive_interval_radical_residual=1, additive_nonlinear_trig_residual=1, affine_argument=2, affine_bounded_rational=1, affine_hyperbolic=1, affine_product=1, affine_radical=1, bounded_rational_expression=1
- `integrate_command_matrix` domain regimes: empty_real_domain=1, explicit_denominator_source_condition=1, explicit_hyperbolic_tangent_denominator_verified_substitution=1, explicit_hyperbolic_tangent_presimplified_condition_dedupe=1, explicit_tangent_denominator_source_condition=1, explicit_tangent_denominator_verified_substitution=1, nonfinite_undefined=1, nonzero_required=3
- `integrate_command_matrix` required displays: -1 < x < 1=3, -3 < x < 1=1, -3/2 < x < 1/2=1, cos(sqrt(x)) ≠ 0=2, cos(x^2) ≠ 0=3, sin(x^2 + 0) ≠ 0=1, sin(x^2) ≠ 0=3, sinh(sqrt(x)) ≠ 0=1
- `integrate_command_matrix` outcomes: residual=9, supported=28, undefined=2
- `integrate_command_matrix` verification regimes: residual_not_verified=9, undefined_not_verified=2, verified_by_diff=28
- `integrate_command_matrix` trace regimes: arctan_scaled_sqrt_reciprocal_substitution=1, arctan_sqrt_reciprocal_substitution=1, by_parts_affine_log=1, by_parts_log=1, constant_multiple_linear_substitution=1, exp_affine_substitution=1, hyperbolic_substitution=1, invalid_integrand_domain_policy=1
- `integrate_command_matrix` presentation regimes: abs_log=1, abs_log_exact_derivative=1, abs_log_hyperbolic_condition_dedupe=1, abs_log_hyperbolic_source_condition=1, abs_log_source_condition=1, abs_log_sqrt_chain=1, abs_log_sqrt_chain_hyperbolic_condition_dedupe=1, affine_arcsin=1
- `residual_matrix`: passed=730 failed=0 total=730 slow=0 timeouts=0 total_bases=81 wrapped_bases=59 standalone_bases=22 wrappers=12 wrapped_cases=708 standalone_cases=22 conditioned_cases=723 distinct_conditions=20
- `residual_matrix` sparse expected conditions: -1 < x < 1=12, 1 - x^2=12, 4 - (x + 1)^2=12, cos(1 - 2·x)=12, sin(sqrt(3 - 2·x))=12
- `residual_matrix` domain expected conditions: -1 < x < 1=12, x < 3/2=12, x > 0=36, x > -1/3=52
- `residual_matrix` wrapper coverage: expected_wrapped_cases=708 missing_wrapped_pairs=0 full_wrapper_bases=59 partial_wrapper_bases=0 largest_gap=0
- `integrate`: passed=340 failed=0 ignored=1 filtered_out=0
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
| `embedded_equivalence_context` | `pass` | 10.86s | passed=1488 failed=0 total=1488 wrappers=7 families=27 avg_case=7.298ms |
| `derive_contract` | `pass` | 2.08s | derived=390 unsupported=0 expected_not_equivalent=1 mean_step_count=1.05 generic_simplify_expected=0 single_step=371 multi_step_ids=19 max_step=3 |
| `derive_shadow_pressure` | `pass` | 0.27s | sampled=67 derived=67 unsupported=0 not_equivalent=0 mean_step_count=1.09 generic_simplify_strategy_successes=0 single_step=62 multi_step_ids=5 max_step=4 embedded_families=26/26 |
| `derive_didactic_audit` | `pass` | 26.98s | cases=472 flagged=0 no_web_substeps=0 no_web_steps=0 artifacts=0.89s cli=0.82s workers=4 top_artifact_family=expand:1.263 top_cli_family=expand:1.251 |
| `simplify_didactic_audit` | `pass` | 1.01s | cases=14 flagged=0 no_wire_substeps=0 missing_math_sides=0 |
| `simplify_strict` | `pass` | 20.57s | closure=100.0% nf=0 (0.0%) proved=16519 (100.0%) numeric=0 inconclusive=0 timeouts=0 |
| `calculus_diff_contract` | `pass` | 3.09s | passed=259 failed=0 ignored=1 |
| `calculus_diff_command_matrix_smoke` | `pass` | 0.48s | passed=40 failed=0 total=40 supported=29 residual=5 warning_expected=0 required_display=24 step_checked=37 unchecked_supported_steps=0 families=20 |
| `calculus_limit_contract` | `pass` | 0.51s | passed=151 failed=0 |
| `calculus_limit_presimplify_contract` | `pass` | 0.36s | passed=8 failed=0 |
| `calculus_limit_command_matrix_smoke` | `pass` | 0.18s | passed=20 failed=0 total=20 supported=13 residual=6 warning_expected=6 required_display=11 step_checked=20 unchecked_supported_steps=0 families=13 |
| `calculus_integrate_command_matrix_smoke` | `pass` | 0.61s | passed=39 failed=0 total=39 supported=28 residual=9 warning_expected=0 required_display=25 step_checked=39 unchecked_supported_steps=0 antiderivative_verified=28 families=31 |
| `calculus_residual_matrix_smoke` | `pass` | 7.74s | passed=730 failed=0 total=730 conditioned=723 conditions=20 total_bases=81 wrapped_bases=59 standalone_bases=22 wrappers=12 missing_wrapped_pairs=0 partial_wrapper_bases=0 |
| `calculus_integrate_contract` | `pass` | 4.54s | passed=340 failed=0 ignored=1 |
