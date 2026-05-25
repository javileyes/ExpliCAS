# Engine Improvement Scorecard

- Generated: 2026-05-25T06:24:55.924571+00:00
- Git branch: main
- Git commit: `d8054a8f2291f09fb0ccb7cd5ac856107acc8069`
- Profile: `fast`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=2
- By area: calculus / limit:1, orchestrator / exact-zero additive composition:1
- Recent 1: `calculus / limit` - 2026-05-24 - Observe-only discovery: limit command residuals cannot enter algebraic wrapper matrix
- Recent 2: `orchestrator / exact-zero additive composition` - 2026-05-24 - Discovery observe-only: hyperbolic angle-sum plus telescoping residual remains slow

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff`: passed=259 failed=0 ignored=1 filtered_out=0
- `diff` ignored tests: `inverse_reciprocal_trig_diff_evaluates_with_explicit_domain_conditions_exhaustive` (exhaustive inverse reciprocal trig diff contract is debug-slow; CI keeps representative structural smoke)
- `diff_command_matrix`: passed=30 failed=0 total=30 slow=0 timeouts=0 supported_cases=19 residual_cases=5 warning_expected=0 required_display=14 step_checked=26 unchecked_supported_steps=0 families=18
- `diff_command_matrix` argument regimes: affine_argument=3, affine_argument_with_trig_pole=1, affine_exponent=3, constant=1, negated_nested_root=1, negative_affine_argument=1, nested_bounded_root=1, nested_root=1
- `diff_command_matrix` domain regimes: discontinuous_residual=1, empty_open_interval_domain=4, empty_positive_argument_domain=2, empty_positive_exponent_domain=1, interval_required=1, invalid_log_base_domain=1, negative_base_undefined=1, nonfinite_undefined=1
- `diff_command_matrix` required displays: a > 0=1, cos(2·x + 1) ≠ 0=1, x < 1=1, x < 1/2=1, x > -1/2=2, x > 0=7, x ≠ -1=1, x ≠ 0=1
- `diff_command_matrix` outcomes: residual=5, supported=19, undefined=6
- `diff_command_matrix` trace regimes: chain_rule=4, constant_base_exponential_chain_rule=1, constant_multiple_chain_rule=1, empty_open_interval_policy=4, exponential_chain_rule=1, log_chain_rule=2, log_empty_domain_policy=1, log_invalid_base_policy=1
- `diff_command_matrix` presentation regimes: canonical=5, compact_quotient=2, constant_zero=1, factored=2, post_calculus_compact=1, quotient_abs=1, reciprocal_root=1, reciprocal_trig_power=1
- `limit_compact`: passed=1 failed=0 ignored=0 filtered_out=149
- `limit_presimplify_safe`: passed=8 failed=0 ignored=0 filtered_out=0
- `limit_command_matrix`: passed=12 failed=0 total=12 slow=0 timeouts=0 supported_cases=8 residual_cases=3 warning_expected=3 required_display=6 step_checked=12 unchecked_supported_steps=0 families=8
- `limit_command_matrix` point regimes: finite=7, infinity=5
- `limit_command_matrix` domain regimes: discontinuous_residual=1, domain_path_conflict=1, empty_real_domain=1, endpoint_residual=1, removable_hole=1, required_condition=3, unconditional=4
- `limit_command_matrix` required displays: 2·x^2 - 3 ≠ 0=1, x > -3=1, x > 0=1, x ≠ -2=1, x ≠ 1=1, x ≥ 0=1
- `limit_command_matrix` outcomes: residual=3, supported=8, undefined=1
- `limit_command_matrix` trace regimes: bounded_over_divergent_policy=1, exponential_tail_policy=1, finite_residual_policy=2, infinity_residual_policy=1, polynomial_leading_term_policy=1, positive_domain_substitution=1, rational_degree_policy=1, rational_multiplicity_policy=1
- `limit_command_matrix` presentation regimes: canonical=7, infinity=1, residual=3, undefined=1
- `integrate_compact`: passed=1 failed=0 ignored=0 filtered_out=338
- `integrate_command_matrix`: passed=26 failed=0 total=26 slow=0 timeouts=0 supported_cases=23 residual_cases=1 warning_expected=0 required_display=13 step_checked=26 unchecked_supported_steps=0 antiderivative_verified=23 families=23
- `integrate_command_matrix` argument regimes: affine_argument=2, affine_bounded_rational=1, affine_hyperbolic=1, affine_product=1, affine_radical=1, bounded_rational_expression=1, bounded_variable_radical=1, constant=1
- `integrate_command_matrix` domain regimes: empty_real_domain=1, nonfinite_undefined=1, nonzero_required=3, positive_required=4, radical_interval=2, rational_interval=2, sqrt_chain_nonzero_positive=2, structurally_positive_log_argument=1
- `integrate_command_matrix` required displays: -1 < x < 1=2, -3 < x < 1=1, -3/2 < x < 1/2=1, cos(sqrt(x)) ≠ 0=2, x > -1/2=1, x > 0=5, x ≠ -1/2=1, x ≠ 1/2=2
- `integrate_command_matrix` outcomes: residual=1, supported=23, undefined=2
- `integrate_command_matrix` verification regimes: residual_not_verified=1, undefined_not_verified=2, verified_by_diff=23
- `integrate_command_matrix` trace regimes: arctan_scaled_sqrt_reciprocal_substitution=1, arctan_sqrt_reciprocal_substitution=1, by_parts_affine_log=1, by_parts_log=1, exp_affine_substitution=1, hyperbolic_substitution=1, invalid_integrand_domain_policy=1, inverse_hyperbolic_rational_affine_table=1
- `integrate_command_matrix` presentation regimes: abs_log=1, abs_log_exact_derivative=1, abs_log_sqrt_chain=1, affine_arcsin=1, affine_atanh=1, compact_power=1, exponential=2, factored_by_parts=2
- `residual_matrix`: passed=730 failed=0 total=730 slow=0 timeouts=0 total_bases=81 wrapped_bases=59 standalone_bases=22 wrappers=12 wrapped_cases=708 standalone_cases=22 conditioned_cases=723 distinct_conditions=20
- `residual_matrix` sparse expected conditions: -1 < x < 1=12, 1 - x^2=12, 4 - (x + 1)^2=12, cos(1 - 2·x)=12, sin(sqrt(3 - 2·x))=12
- `residual_matrix` domain expected conditions: -1 < x < 1=12, x < 3/2=12, x > 0=36, x > -1/3=52
- `residual_matrix` wrapper coverage: expected_wrapped_cases=708 missing_wrapped_pairs=0 full_wrapper_bases=59 partial_wrapper_bases=0 largest_gap=0

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_add_small` | `pass` | 0.63s | passed=1 failed=0 closure=100.0% nf=0 (0.0%) proved=435 (100.0%) timeouts=0 |
| `contextual_strict_fast` | `pass` | 5.71s | passed=1 failed=0 closure=100.0% nf=0 (0.0%) proved=64 (100.0%) timeouts=0 |
| `contextual_radical_fast` | `pass` | 0.14s | passed=1 failed=0 closure=100.0% nf=0 (0.0%) proved=15 (100.0%) timeouts=0 |
| `calculus_diff_contract` | `pass` | 3.08s | passed=259 failed=0 ignored=1 |
| `calculus_diff_command_matrix_smoke` | `pass` | 0.27s | passed=30 failed=0 total=30 supported=19 residual=5 warning_expected=0 required_display=14 step_checked=26 unchecked_supported_steps=0 families=18 |
| `calculus_limit_compact_contract` | `pass` | 2.14s | passed=1 failed=0 |
| `calculus_limit_presimplify_contract` | `pass` | 0.67s | passed=8 failed=0 |
| `calculus_limit_command_matrix_smoke` | `pass` | 0.14s | passed=12 failed=0 total=12 supported=8 residual=3 warning_expected=3 required_display=6 step_checked=12 unchecked_supported_steps=0 families=8 |
| `calculus_integrate_compact_contract` | `pass` | 1.09s | passed=1 failed=0 |
| `calculus_integrate_command_matrix_smoke` | `pass` | 0.45s | passed=26 failed=0 total=26 supported=23 residual=1 warning_expected=0 required_display=13 step_checked=26 unchecked_supported_steps=0 antiderivative_verified=23 families=23 |
| `calculus_residual_matrix_smoke` | `pass` | 7.67s | passed=730 failed=0 total=730 conditioned=723 conditions=20 total_bases=81 wrapped_bases=59 standalone_bases=22 wrappers=12 missing_wrapped_pairs=0 partial_wrapper_bases=0 |
