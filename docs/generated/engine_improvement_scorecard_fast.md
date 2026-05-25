# Engine Improvement Scorecard

- Generated: 2026-05-25T17:28:04.254102+00:00
- Git branch: main
- Git commit: `0e0f8b17d110820555956b4c650b7ef98a3260ed`
- Profile: `fast`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=4
- By area: calculus / limit:2, calculus / differentiation:1, orchestrator / exact-zero additive composition:1
- Recent 1: `calculus / differentiation` - 2026-05-25 - Observe-only discovery: inverse-tangent symbolic denominator shortcut leaves condition and external-scale gaps
- Recent 2: `calculus / limit` - 2026-05-24 - Observe-only discovery: limit command residuals cannot enter algebraic wrapper matrix
- Recent 3: `orchestrator / exact-zero additive composition` - 2026-05-24 - Discovery observe-only: hyperbolic angle-sum plus telescoping residual remains slow

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
- `limit_compact`: passed=1 failed=0 ignored=0 filtered_out=150
- `limit_presimplify_safe`: passed=8 failed=0 ignored=0 filtered_out=0
- `limit_command_matrix`: passed=20 failed=0 total=20 slow=0 timeouts=0 supported_cases=13 residual_cases=6 warning_expected=6 required_display=11 step_checked=20 unchecked_supported_steps=0 families=13
- `limit_command_matrix` point regimes: finite=15, infinity=5
- `limit_command_matrix` domain regimes: argument_zero_boundary_residual=1, discontinuous_residual=1, domain_path_conflict=1, empty_real_domain=1, endpoint_residual=2, removable_hole=1, required_condition=3, root_interior_interval_required=1
- `limit_command_matrix` required displays: 2·x^2 - 3 ≠ 0=1, x > -1=1, x > -2=1, x > -3=1, x > 0=2, x ≠ -1=1, x ≠ -2=1, x ≠ 0=1
- `limit_command_matrix` outcomes: residual=6, supported=13, undefined=1
- `limit_command_matrix` trace regimes: binary_log_finite_substitution=1, bounded_over_divergent_policy=1, exponential_tail_policy=1, finite_residual_policy=5, finite_special_angle_table=1, infinity_residual_policy=1, inverse_trig_root_finite_substitution=1, nested_positive_domain_substitution=1
- `limit_command_matrix` presentation regimes: binary_log=1, canonical=7, canonical_no_required_conditions=2, exact_radical=1, infinity=1, residual=5, residual_presentation_cleanup=1, special_angle=1
- `integrate_compact`: passed=1 failed=0 ignored=0 filtered_out=340
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

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_add_small` | `pass` | 4.42s | passed=1 failed=0 closure=100.0% nf=0 (0.0%) proved=435 (100.0%) timeouts=0 |
| `contextual_strict_fast` | `pass` | 34.88s | passed=1 failed=0 closure=100.0% nf=0 (0.0%) proved=64 (100.0%) timeouts=0 |
| `contextual_radical_fast` | `pass` | 0.15s | passed=1 failed=0 closure=100.0% nf=0 (0.0%) proved=15 (100.0%) timeouts=0 |
| `calculus_diff_contract` | `pass` | 15.65s | passed=259 failed=0 ignored=1 |
| `calculus_diff_command_matrix_smoke` | `pass` | 0.49s | passed=40 failed=0 total=40 supported=29 residual=5 warning_expected=0 required_display=24 step_checked=37 unchecked_supported_steps=0 families=20 |
| `calculus_limit_compact_contract` | `pass` | 24.54s | passed=1 failed=0 |
| `calculus_limit_presimplify_contract` | `pass` | 0.69s | passed=8 failed=0 |
| `calculus_limit_command_matrix_smoke` | `pass` | 0.20s | passed=20 failed=0 total=20 supported=13 residual=6 warning_expected=6 required_display=11 step_checked=20 unchecked_supported_steps=0 families=13 |
| `calculus_integrate_compact_contract` | `pass` | 8.57s | passed=1 failed=0 |
| `calculus_integrate_command_matrix_smoke` | `pass` | 0.61s | passed=39 failed=0 total=39 supported=28 residual=9 warning_expected=0 required_display=25 step_checked=39 unchecked_supported_steps=0 antiderivative_verified=28 families=31 |
| `calculus_residual_matrix_smoke` | `pass` | 7.83s | passed=730 failed=0 total=730 conditioned=723 conditions=20 total_bases=81 wrapped_bases=59 standalone_bases=22 wrappers=12 missing_wrapped_pairs=0 partial_wrapper_bases=0 |
