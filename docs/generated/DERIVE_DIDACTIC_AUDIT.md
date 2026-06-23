# Derive Didactic Audit

Generated from [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv).

Command: `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`

## Summary

- Derived cases audited: `472`
- Mean top-level step count: `1.05`
- Total web substeps: `484`

## Flag Summary

- Cases with flags: `0`
- Cases flagged as no web substeps emitted: `0`

No audit flags emitted.

| family | cases | flagged | no-substeps flag | web substeps |
| --- | ---: | ---: | ---: | ---: |
| `collect` | 6 | 0 | 0 | 0 |
| `conditional_factor` | 6 | 0 | 0 | 12 |
| `expand` | 34 | 0 | 0 | 46 |
| `factor` | 17 | 0 | 0 | 21 |
| `finite_aggregate` | 17 | 0 | 0 | 34 |
| `finite_telescoping` | 9 | 0 | 0 | 21 |
| `fraction_combine` | 13 | 0 | 0 | 0 |
| `fraction_decompose` | 5 | 0 | 0 | 10 |
| `fraction_expand` | 10 | 0 | 0 | 8 |
| `integrate_prep` | 12 | 0 | 0 | 12 |
| `log_contract` | 19 | 0 | 0 | 2 |
| `log_exp_inverse` | 5 | 0 | 0 | 6 |
| `log_expand` | 17 | 0 | 0 | 2 |
| `log_inverse_power` | 2 | 0 | 0 | 4 |
| `nested_fraction` | 12 | 0 | 0 | 21 |
| `number_theory` | 3 | 0 | 0 | 5 |
| `polynomial_product` | 11 | 0 | 0 | 24 |
| `power_merge` | 10 | 0 | 0 | 2 |
| `radical_power` | 6 | 0 | 0 | 12 |
| `rationalize` | 9 | 0 | 0 | 25 |
| `simplify` | 90 | 0 | 0 | 82 |
| `solve_prep` | 8 | 0 | 0 | 21 |
| `telescoping_fraction` | 15 | 0 | 0 | 30 |
| `trig_contract` | 49 | 0 | 0 | 25 |
| `trig_expand` | 87 | 0 | 0 | 59 |

| id | family | web steps | web substeps | flags |
| --- | --- | ---: | ---: | --- |
| `arcsin_sin_arctan_safe_composition` | `simplify` | 1 | 3 | none |
| `cancel_fraction_common_factor_numeric` | `simplify` | 1 | 2 | none |
| `cancel_fraction_difference_cubes` | `simplify` | 1 | 3 | none |
| `cancel_fraction_difference_cubes_with_passthrough` | `simplify` | 1 | 3 | none |
| `cancel_fraction_difference_squares` | `simplify` | 1 | 2 | none |
| `cancel_fraction_difference_squares_with_passthrough` | `simplify` | 1 | 2 | none |
| `cancel_fraction_monomial_common_factor` | `simplify` | 1 | 2 | none |
| `cancel_fraction_perfect_square_minus_symbolic` | `simplify` | 1 | 2 | none |
| `cancel_fraction_sum_cubes` | `simplify` | 1 | 3 | none |
| `choose_numeric_binomial_coefficient` | `number_theory` | 1 | 2 | none |
| `choose_numeric_pascal_identity` | `number_theory` | 1 | 1 | none |
| `choose_numeric_symmetry` | `number_theory` | 1 | 2 | none |
| `collapse_exponential_log_product` | `simplify` | 1 | 2 | none |
| `collapse_exponential_scaled_log_product` | `simplify` | 1 | 2 | none |
| `collect_common_symbolic_coefficients` | `collect` | 1 | 0 | none |
| `collect_composite_monomial_factor` | `collect` | 1 | 0 | none |
| `collect_linear` | `collect` | 1 | 0 | none |
| `collect_linear_alt_variable` | `collect` | 1 | 0 | none |
| `collect_multiple_power_groups` | `collect` | 1 | 0 | none |
| `collect_two_composite_factor_groups` | `collect` | 1 | 0 | none |
| `combine_fraction_part_with_same_denominator` | `fraction_combine` | 1 | 0 | none |
| `combine_fraction_part_with_same_denominator_three_terms` | `fraction_combine` | 1 | 0 | none |
| `combine_general_fraction_difference` | `fraction_combine` | 1 | 0 | none |
| `combine_general_fraction_sum` | `fraction_combine` | 1 | 0 | none |
| `combine_like_terms` | `simplify` | 1 | 1 | none |
| `combine_negative_scaled_symbolic_whole_plus_remainder_into_fraction` | `fraction_combine` | 1 | 0 | none |
| `combine_same_denominator_fraction_difference` | `fraction_combine` | 1 | 0 | none |
| `combine_same_denominator_fraction_sum` | `fraction_combine` | 1 | 0 | none |
| `combine_scaled_symbolic_whole_plus_remainder_into_fraction` | `fraction_combine` | 1 | 0 | none |
| `combine_symbolic_same_denominator_fraction_subset_with_passthrough` | `fraction_combine` | 1 | 0 | none |
| `combine_symbolic_whole_plus_remainder_into_fraction` | `fraction_combine` | 1 | 0 | none |
| `combine_telescoping_fraction_affine_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_affine_symbolic_shift_gap` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_consecutive` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_difference_squares_unfactored` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_negative_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_shifted_quadratic_unfactored` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_symbolic_difference_squares_unfactored` | `telescoping_fraction` | 1 | 2 | none |
| `combine_term_and_fraction_subtraction` | `fraction_combine` | 1 | 0 | none |
| `combine_three_same_denominator_fractions` | `fraction_combine` | 1 | 0 | none |
| `combine_whole_plus_remainder_into_fraction` | `fraction_combine` | 1 | 0 | none |
| `consecutive_factorial_ratio` | `simplify` | 1 | 2 | none |
| `consecutive_factorial_ratio_gap_two` | `simplify` | 1 | 2 | none |
| `consecutive_factorial_ratio_with_passthrough` | `simplify` | 1 | 2 | none |
| `contract_even_abs_logs_to_scaled_abs_product` | `log_contract` | 1 | 0 | none |
| `contract_even_abs_logs_to_scaled_abs_product_with_passthrough` | `log_contract` | 1 | 0 | none |
| `contract_exponential_difference` | `simplify` | 1 | 1 | none |
| `contract_exponential_power` | `simplify` | 1 | 1 | none |
| `contract_exponential_reciprocal` | `simplify` | 1 | 1 | none |
| `contract_exponential_sum` | `simplify` | 1 | 1 | none |
| `contract_general_base_logs_to_grouped_power` | `log_contract` | 1 | 0 | none |
| `contract_general_base_logs_to_grouped_power_with_passthrough` | `log_contract` | 1 | 0 | none |
| `contract_hyperbolic_cosh_difference` | `simplify` | 1 | 1 | none |
| `contract_hyperbolic_cosh_sum` | `simplify` | 1 | 1 | none |
| `contract_hyperbolic_sinh_difference` | `simplify` | 1 | 1 | none |
| `contract_hyperbolic_sinh_sum` | `simplify` | 1 | 1 | none |
| `contract_hyperbolic_tanh_difference` | `simplify` | 1 | 1 | none |
| `contract_hyperbolic_tanh_sum` | `simplify` | 1 | 1 | none |
| `contract_log_change_of_base_chain` | `log_contract` | 1 | 0 | none |
| `contract_log_change_of_base_direct` | `log_contract` | 1 | 2 | none |
| `contract_log_difference` | `log_contract` | 1 | 0 | none |
| `contract_log_difference_with_scaled_powers` | `log_contract` | 1 | 0 | none |
| `contract_log_even_power_abs` | `log_contract` | 1 | 0 | none |
| `contract_log_general_base_difference` | `log_contract` | 1 | 0 | none |
| `contract_log_general_base_difference_with_scaled_powers` | `log_contract` | 1 | 0 | none |
| `contract_log_general_base_power` | `log_contract` | 1 | 0 | none |
| `contract_log_general_base_powered_two_denominator_factors_with_powered_denominator` | `log_contract` | 1 | 0 | none |
| `contract_log_grouped_power` | `log_contract` | 1 | 0 | none |
| `contract_log_grouped_power_with_passthrough` | `log_contract` | 1 | 0 | none |
| `contract_log_powered_two_denominator_factors` | `log_contract` | 1 | 0 | none |
| `contract_log_product_over_quotient` | `log_contract` | 1 | 0 | none |
| `contract_log_sum` | `log_contract` | 1 | 0 | none |
| `contract_log_sum_with_scaled_powers` | `log_contract` | 1 | 0 | none |
| `contract_trig_angle_diff_cosine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_angle_diff_sine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_angle_sum_cosine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_angle_sum_sine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_cos_diff_sin_diff_quotient` | `trig_contract` | 1 | 0 | none |
| `contract_trig_cot_quotient` | `trig_contract` | 1 | 0 | none |
| `contract_trig_csc_reciprocal` | `trig_contract` | 1 | 0 | none |
| `contract_trig_csc_squared` | `trig_contract` | 1 | 0 | none |
| `contract_trig_double_cos_from_one_minus_sin_sq` | `trig_contract` | 1 | 0 | none |
| `contract_trig_double_cos_from_two_cos_sq_minus_one` | `trig_contract` | 1 | 0 | none |
| `contract_trig_double_sin` | `trig_contract` | 1 | 0 | none |
| `contract_trig_double_tangent` | `trig_contract` | 1 | 0 | none |
| `contract_trig_half_angle_cos_squared` | `trig_contract` | 1 | 0 | none |
| `contract_trig_half_angle_sin_squared` | `trig_contract` | 1 | 0 | none |
| `contract_trig_half_angle_tangent` | `trig_contract` | 1 | 0 | none |
| `contract_trig_half_angle_tangent_alt` | `trig_contract` | 1 | 0 | none |
| `contract_trig_half_scaled_double_sin` | `trig_contract` | 1 | 0 | none |
| `contract_trig_negative_double_cos_from_square_diff` | `trig_contract` | 1 | 0 | none |
| `contract_trig_negative_double_sin` | `trig_contract` | 1 | 0 | none |
| `contract_trig_phase_shift_difference_to_shifted_sine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_phase_shift_exact_sixth_sum_to_shifted_sine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_phase_shift_exact_third_scaled_sum_to_shifted_sine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_phase_shift_general_shifted_sine_to_shifted_cosine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_phase_shift_general_shifted_terms_with_passthrough` | `trig_contract` | 1 | 1 | none |
| `contract_trig_phase_shift_general_sum_to_shifted_sine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_phase_shift_general_sum_to_shifted_sine_with_passthrough` | `trig_contract` | 1 | 1 | none |
| `contract_trig_phase_shift_scaled_sum_to_shifted_sine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_phase_shift_shifted_sine_to_shifted_cosine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_phase_shift_shifted_terms_with_passthrough` | `trig_contract` | 1 | 1 | none |
| `contract_trig_phase_shift_sum_to_shifted_sine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_phase_shift_sum_to_shifted_sine_with_passthrough` | `trig_contract` | 1 | 1 | none |
| `contract_trig_quadruple_angle_sine_expanded_product` | `trig_contract` | 1 | 1 | none |
| `contract_trig_quintuple_angle_cosine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_quintuple_angle_sine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_recursive_six_cosine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_recursive_six_sine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_sec_reciprocal` | `trig_contract` | 1 | 0 | none |
| `contract_trig_sec_squared` | `trig_contract` | 1 | 0 | none |
| `contract_trig_sin_diff_special` | `trig_contract` | 1 | 0 | none |
| `contract_trig_square_double_angle_sine_cosine_product` | `trig_contract` | 1 | 1 | none |
| `contract_trig_tan_quotient` | `trig_contract` | 1 | 0 | none |
| `contract_trig_tan_quotient_after_arg_simplify` | `trig_contract` | 1 | 0 | none |
| `contract_trig_tan_quotient_with_additive_passthrough` | `trig_contract` | 1 | 0 | none |
| `contract_trig_tan_quotient_with_cofactor` | `trig_contract` | 1 | 0 | none |
| `contract_trig_tangent_angle_difference` | `trig_contract` | 1 | 0 | none |
| `contract_trig_tangent_angle_sum` | `trig_contract` | 1 | 0 | none |
| `contract_trig_triple_angle_cosine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_triple_angle_sine` | `trig_contract` | 1 | 1 | none |
| `contract_trig_triple_angle_tangent` | `trig_contract` | 1 | 1 | none |
| `cos_arcsin_complement_projection` | `simplify` | 1 | 2 | none |
| `cos_arctan_right_triangle_projection` | `simplify` | 1 | 2 | none |
| `csc_cot_pythagorean_to_one` | `simplify` | 1 | 0 | none |
| `expand_binomial` | `expand` | 1 | 0 | none |
| `expand_common_factor_difference` | `expand` | 1 | 2 | none |
| `expand_common_factor_difference_three_terms` | `expand` | 2 | 4 | none |
| `expand_common_factor_sum` | `expand` | 1 | 2 | none |
| `expand_common_factor_sum_three_terms` | `expand` | 2 | 4 | none |
| `expand_cube_difference_product` | `polynomial_product` | 1 | 2 | none |
| `expand_cube_sum_product` | `polynomial_product` | 1 | 2 | none |
| `expand_difference_cubes` | `expand` | 1 | 2 | none |
| `expand_difference_of_squares_quadratic_product` | `polynomial_product` | 1 | 2 | none |
| `expand_eighth_power_minus_multifactor_product` | `polynomial_product` | 1 | 2 | none |
| `expand_exponential_power` | `simplify` | 1 | 1 | none |
| `expand_exponential_reciprocal` | `simplify` | 1 | 1 | none |
| `expand_exponential_sum` | `simplify` | 1 | 1 | none |
| `expand_fraction_exact_division_term_plus_remainder` | `fraction_expand` | 1 | 1 | none |
| `expand_fraction_mixed_variable_term_cancellation` | `fraction_expand` | 1 | 1 | none |
| `expand_fraction_part_with_same_denominator_three_terms` | `fraction_expand` | 1 | 0 | none |
| `expand_fraction_simple` | `fraction_expand` | 1 | 0 | none |
| `expand_fraction_three_factor_cross_cancellation_plus_remainder` | `fraction_expand` | 1 | 1 | none |
| `expand_fraction_three_factor_full_cancellation` | `fraction_expand` | 1 | 1 | none |
| `expand_fraction_three_factor_three_cancellations_to_constant` | `fraction_expand` | 1 | 1 | none |
| `expand_fraction_two_cancellations_plus_remainder` | `fraction_expand` | 1 | 1 | none |
| `expand_fraction_with_common_scalar_factor_in_denominator` | `fraction_expand` | 1 | 1 | none |
| `expand_fraction_with_term_cancellation` | `fraction_expand` | 1 | 1 | none |
| `expand_fractional_binomial_square` | `expand` | 1 | 0 | none |
| `expand_higher_odd_half_power` | `radical_power` | 1 | 2 | none |
| `expand_higher_odd_half_power_after_simplify` | `radical_power` | 1 | 2 | none |
| `expand_higher_odd_half_power_alt_var` | `radical_power` | 1 | 2 | none |
| `expand_hyperbolic_cosh_difference` | `expand` | 1 | 1 | none |
| `expand_hyperbolic_cosh_difference_to_product_exact` | `expand` | 1 | 1 | none |
| `expand_hyperbolic_cosh_sum` | `expand` | 1 | 1 | none |
| `expand_hyperbolic_cosh_sum_to_product_exact` | `expand` | 1 | 1 | none |
| `expand_hyperbolic_product_to_sum_to_cosh_cubic_polynomial` | `expand` | 1 | 1 | none |
| `expand_hyperbolic_product_to_sum_to_cosh_cubic_polynomial_with_passthrough` | `expand` | 2 | 2 | none |
| `expand_hyperbolic_product_to_sum_to_sinh_cubic_polynomial` | `expand` | 1 | 1 | none |
| `expand_hyperbolic_sinh_difference` | `expand` | 1 | 1 | none |
| `expand_hyperbolic_sinh_sum` | `expand` | 1 | 1 | none |
| `expand_hyperbolic_sinh_sum_to_product_exact` | `expand` | 1 | 1 | none |
| `expand_hyperbolic_tanh_difference` | `simplify` | 1 | 1 | none |
| `expand_hyperbolic_tanh_sum` | `simplify` | 1 | 1 | none |
| `expand_hyperbolic_tanh_triple_angle` | `simplify` | 1 | 0 | none |
| `expand_log_change_of_base_chain` | `log_expand` | 1 | 0 | none |
| `expand_log_change_of_base_direct` | `log_expand` | 1 | 2 | none |
| `expand_log_even_power_abs` | `log_expand` | 1 | 0 | none |
| `expand_log_general_base_power` | `log_expand` | 1 | 0 | none |
| `expand_log_general_base_powered_two_denominator_factors_with_powered_denominator` | `log_expand` | 2 | 0 | none |
| `expand_log_general_base_product_over_quotient` | `log_expand` | 1 | 0 | none |
| `expand_log_grouped_abs_product` | `log_expand` | 1 | 0 | none |
| `expand_log_grouped_abs_product_with_passthrough` | `log_expand` | 1 | 0 | none |
| `expand_log_grouped_general_base_power` | `log_expand` | 1 | 0 | none |
| `expand_log_grouped_general_base_power_with_passthrough` | `log_expand` | 1 | 0 | none |
| `expand_log_grouped_power` | `log_expand` | 1 | 0 | none |
| `expand_log_grouped_power_with_passthrough` | `log_expand` | 1 | 0 | none |
| `expand_log_powered_two_denominator_factors` | `log_expand` | 2 | 0 | none |
| `expand_log_product` | `log_expand` | 1 | 0 | none |
| `expand_log_product_over_quotient` | `log_expand` | 1 | 0 | none |
| `expand_log_product_with_root_cleanup` | `log_expand` | 2 | 0 | none |
| `expand_log_quotient` | `log_expand` | 1 | 0 | none |
| `expand_ninth_power_plus_product` | `polynomial_product` | 1 | 2 | none |
| `expand_odd_half_power` | `radical_power` | 1 | 2 | none |
| `expand_odd_half_power_after_simplify` | `radical_power` | 1 | 2 | none |
| `expand_odd_half_power_after_simplify_with_passthrough` | `radical_power` | 1 | 2 | none |
| `expand_recursive_hyperbolic_cosh_sum` | `expand` | 1 | 1 | none |
| `expand_recursive_hyperbolic_sinh_sum` | `expand` | 1 | 1 | none |
| `expand_sixth_power_minus_product` | `polynomial_product` | 1 | 3 | none |
| `expand_sixth_power_plus_product` | `polynomial_product` | 1 | 3 | none |
| `expand_sophie_germain` | `expand` | 1 | 2 | none |
| `expand_sum_cubes` | `expand` | 1 | 2 | none |
| `expand_symbolic_binomial` | `expand` | 1 | 0 | none |
| `expand_symbolic_binomial_cube` | `expand` | 1 | 0 | none |
| `expand_symbolic_binomial_cube_minus` | `expand` | 2 | 2 | none |
| `expand_symbolic_binomial_minus` | `expand` | 2 | 1 | none |
| `expand_symbolic_cube_difference_product` | `polynomial_product` | 1 | 2 | none |
| `expand_symbolic_cube_sum_product` | `polynomial_product` | 1 | 2 | none |
| `expand_symbolic_signed_trinomial_square` | `expand` | 3 | 0 | none |
| `expand_symbolic_sixth_power_minus_product` | `polynomial_product` | 1 | 2 | none |
| `expand_symbolic_sixth_power_plus_product` | `polynomial_product` | 1 | 2 | none |
| `expand_symbolic_trinomial_cube` | `expand` | 1 | 0 | none |
| `expand_symbolic_trinomial_square` | `expand` | 1 | 0 | none |
| `expand_then_cancel_to_square` | `expand` | 2 | 2 | none |
| `expand_trig_after_simplify` | `trig_expand` | 1 | 0 | none |
| `expand_trig_angle_diff_cosine` | `trig_expand` | 1 | 1 | none |
| `expand_trig_angle_diff_sine` | `trig_expand` | 1 | 1 | none |
| `expand_trig_angle_sum_cosine` | `trig_expand` | 1 | 1 | none |
| `expand_trig_angle_sum_sine` | `trig_expand` | 1 | 1 | none |
| `expand_trig_cofunction_cosine_minus` | `trig_expand` | 1 | 0 | none |
| `expand_trig_cofunction_sine_minus` | `trig_expand` | 1 | 0 | none |
| `expand_trig_cosine_eighteenth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_cosine_eighth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_cosine_fourteenth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_cosine_fourth_power_reduction` | `simplify` | 1 | 1 | none |
| `expand_trig_cosine_sixteenth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_cosine_sixth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_cosine_tenth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_cosine_twelfth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_cosine_twentieth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_cosine_twenty_fourth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_cosine_twenty_second_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_cot_quotient` | `trig_expand` | 1 | 0 | none |
| `expand_trig_csc_reciprocal` | `trig_expand` | 1 | 0 | none |
| `expand_trig_csc_squared` | `trig_expand` | 1 | 0 | none |
| `expand_trig_double_cos_as_one_minus_sin_sq` | `trig_expand` | 1 | 0 | none |
| `expand_trig_double_cos_as_two_cos_sq_minus_one` | `trig_expand` | 1 | 0 | none |
| `expand_trig_double_cos_inverse_arccos` | `trig_expand` | 1 | 2 | none |
| `expand_trig_double_cos_inverse_arcsin` | `trig_expand` | 1 | 2 | none |
| `expand_trig_double_sin` | `trig_expand` | 1 | 0 | none |
| `expand_trig_double_sin_arctan_projection` | `trig_expand` | 1 | 2 | none |
| `expand_trig_double_sin_inverse_arccos` | `trig_expand` | 1 | 2 | none |
| `expand_trig_double_sin_inverse_arcsin` | `trig_expand` | 1 | 2 | none |
| `expand_trig_double_tangent` | `trig_expand` | 1 | 0 | none |
| `expand_trig_half_angle_cos_squared` | `trig_expand` | 1 | 0 | none |
| `expand_trig_half_angle_sin_squared` | `trig_expand` | 1 | 0 | none |
| `expand_trig_half_angle_tangent` | `trig_expand` | 1 | 0 | none |
| `expand_trig_half_angle_tangent_alt` | `trig_expand` | 1 | 0 | none |
| `expand_trig_half_angle_tangent_one_minus_cos_over_sin` | `trig_expand` | 1 | 0 | none |
| `expand_trig_half_angle_tangent_sin_over_one_plus_cos` | `trig_expand` | 1 | 0 | none |
| `expand_trig_negative_double_cos_as_square_diff` | `trig_expand` | 1 | 0 | none |
| `expand_trig_negative_double_sin` | `trig_expand` | 1 | 0 | none |
| `expand_trig_negative_tangent_parity` | `trig_expand` | 1 | 1 | none |
| `expand_trig_phase_shift_exact_sixth_shifted_sine_to_sum` | `trig_expand` | 1 | 1 | none |
| `expand_trig_phase_shift_exact_third_scaled_shifted_sine_to_sum` | `trig_expand` | 1 | 1 | none |
| `expand_trig_phase_shift_general_shifted_sine_to_sum` | `trig_expand` | 1 | 1 | none |
| `expand_trig_phase_shift_general_shifted_sine_to_sum_with_passthrough` | `trig_expand` | 1 | 1 | none |
| `expand_trig_phase_shift_pair_sum_to_shifted_sine_pair` | `trig_expand` | 2 | 1 | none |
| `expand_trig_phase_shift_scaled_shifted_sine_to_sum` | `trig_expand` | 1 | 1 | none |
| `expand_trig_phase_shift_scaled_shifted_sine_to_sum_with_passthrough` | `trig_expand` | 1 | 1 | none |
| `expand_trig_phase_shift_shifted_cosine_to_sum` | `trig_expand` | 1 | 1 | none |
| `expand_trig_phase_shift_shifted_sine_pair_to_sum_pair` | `trig_expand` | 2 | 2 | none |
| `expand_trig_phase_shift_shifted_sine_to_sum` | `trig_expand` | 1 | 1 | none |
| `expand_trig_product_to_sum_cos_cos` | `trig_expand` | 1 | 1 | none |
| `expand_trig_product_to_sum_cos_sin` | `trig_expand` | 1 | 1 | none |
| `expand_trig_product_to_sum_sin_cos` | `trig_expand` | 1 | 1 | none |
| `expand_trig_product_to_sum_sin_sin` | `trig_expand` | 1 | 1 | none |
| `expand_trig_product_to_sum_to_cosine_difference_polynomial` | `expand` | 2 | 2 | none |
| `expand_trig_product_to_sum_to_cosine_difference_polynomial_with_passthrough` | `expand` | 2 | 2 | none |
| `expand_trig_product_to_sum_to_cosine_sum_polynomial` | `expand` | 2 | 2 | none |
| `expand_trig_product_to_sum_to_sine_difference_mixed_polynomial` | `expand` | 2 | 2 | none |
| `expand_trig_product_to_sum_to_sine_difference_mixed_polynomial_with_passthrough` | `expand` | 2 | 2 | none |
| `expand_trig_quadruple_angle_cosine` | `trig_expand` | 1 | 1 | none |
| `expand_trig_quadruple_angle_sine_expanded_product` | `trig_expand` | 1 | 1 | none |
| `expand_trig_quintuple_angle_cosine` | `trig_expand` | 1 | 1 | none |
| `expand_trig_quintuple_angle_sine` | `trig_expand` | 1 | 1 | none |
| `expand_trig_recursive_six_cosine` | `trig_expand` | 1 | 1 | none |
| `expand_trig_recursive_six_sine` | `trig_expand` | 1 | 1 | none |
| `expand_trig_scaled_half_angle_sine_square_to_shifted_cosine` | `trig_expand` | 1 | 0 | none |
| `expand_trig_sec_reciprocal` | `trig_expand` | 1 | 0 | none |
| `expand_trig_sec_squared` | `trig_expand` | 1 | 0 | none |
| `expand_trig_sin_cos_square_diff` | `trig_expand` | 1 | 0 | none |
| `expand_trig_sin_cos_square_sum` | `trig_expand` | 1 | 0 | none |
| `expand_trig_sine_cosine_square_product_reduction` | `simplify` | 1 | 1 | none |
| `expand_trig_sine_eighteenth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sine_eighth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sine_fourteenth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sine_fourth_power_reduction` | `simplify` | 1 | 1 | none |
| `expand_trig_sine_sixteenth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sine_sixth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sine_tenth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sine_twelfth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sine_twentieth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sine_twenty_fourth_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sine_twenty_second_power_reduction` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sum_to_product_cos_diff_general` | `trig_expand` | 1 | 0 | none |
| `expand_trig_sum_to_product_cos_diff_xy` | `trig_expand` | 1 | 0 | none |
| `expand_trig_sum_to_product_cos_sum_general` | `trig_expand` | 1 | 0 | none |
| `expand_trig_sum_to_product_cos_sum_xy` | `trig_expand` | 1 | 0 | none |
| `expand_trig_sum_to_product_sin_diff_general` | `trig_expand` | 1 | 0 | none |
| `expand_trig_sum_to_product_sin_sum_general` | `trig_expand` | 1 | 0 | none |
| `expand_trig_sum_to_product_sin_sum_xy` | `trig_expand` | 1 | 0 | none |
| `expand_trig_tan_to_sin_cos` | `trig_expand` | 1 | 0 | none |
| `expand_trig_tangent_angle_difference` | `trig_expand` | 1 | 0 | none |
| `expand_trig_tangent_angle_sum` | `trig_expand` | 1 | 0 | none |
| `expand_trig_tangent_half_angle_substitution_sine` | `trig_expand` | 1 | 0 | none |
| `expand_trig_triple_angle_cosine` | `trig_expand` | 1 | 1 | none |
| `expand_trig_triple_angle_sine` | `trig_expand` | 1 | 1 | none |
| `expand_trig_triple_angle_tangent` | `trig_expand` | 1 | 1 | none |
| `factor_alternating_cubic_vandermonde` | `factor` | 1 | 4 | none |
| `factor_common_factor_sum` | `factor` | 1 | 1 | none |
| `factor_common_factor_sum_three_terms` | `factor` | 1 | 1 | none |
| `factor_difference_cubes` | `factor` | 1 | 1 | none |
| `factor_difference_squares` | `factor` | 1 | 1 | none |
| `factor_difference_squares_with_passthrough` | `factor` | 1 | 1 | none |
| `factor_full_cyclotomic_sixth_power_difference` | `factor` | 1 | 1 | none |
| `factor_geometric_difference_power_6` | `factor` | 1 | 1 | none |
| `factor_out_cube_with_division_septic` | `conditional_factor` | 1 | 2 | none |
| `factor_out_square_with_division_quartic` | `conditional_factor` | 1 | 2 | none |
| `factor_out_with_division` | `conditional_factor` | 1 | 2 | none |
| `factor_out_with_division_mixed_septic` | `conditional_factor` | 1 | 2 | none |
| `factor_out_with_division_quadratic` | `conditional_factor` | 1 | 2 | none |
| `factor_out_with_division_sparse_quintic` | `conditional_factor` | 1 | 2 | none |
| `factor_perfect_square_trinomial` | `factor` | 1 | 1 | none |
| `factor_perfect_square_trinomial_minus` | `factor` | 1 | 1 | none |
| `factor_perfect_square_trinomial_symbolic` | `factor` | 1 | 1 | none |
| `factor_sophie_germain` | `factor` | 1 | 2 | none |
| `factor_sum_cubes` | `factor` | 1 | 1 | none |
| `factor_symbolic_binomial_cube` | `factor` | 1 | 1 | none |
| `factor_symbolic_binomial_cube_minus` | `factor` | 1 | 1 | none |
| `factor_symbolic_sixth_power_difference` | `factor` | 1 | 1 | none |
| `factor_symbolic_sixth_power_sum` | `factor` | 1 | 1 | none |
| `finite_aggregate_product_constant_symbolic` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_product_constant_symbolic_lower_bound` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_product_first_integers_symbolic` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_product_first_integers_symbolic_lower_bound` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_product_of_cubes_symbolic_lower_bound` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_product_of_squares_symbolic` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_product_of_squares_symbolic_lower_bound` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_sum_constant_symbolic` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_sum_constant_symbolic_lower_bound` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_sum_first_integers_symbolic` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_sum_first_integers_symbolic_lower_bound` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_sum_geometric_power_base_two_symbolic` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_sum_geometric_power_base_two_symbolic_lower_bound` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_sum_of_cubes_symbolic` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_sum_of_cubes_symbolic_lower_bound` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_sum_of_squares_symbolic` | `finite_aggregate` | 1 | 2 | none |
| `finite_aggregate_sum_of_squares_symbolic_lower_bound` | `finite_aggregate` | 1 | 2 | none |
| `finite_telescoping_product_affine_symbolic_shift_symbolic_lower` | `finite_telescoping` | 1 | 2 | none |
| `finite_telescoping_product_basic` | `finite_telescoping` | 1 | 3 | none |
| `finite_telescoping_product_factorized_square_shifted_base_numeric_symbolic_lower` | `finite_telescoping` | 1 | 3 | none |
| `finite_telescoping_product_factorized_square_shifted_base_symbolic_symbolic_lower` | `finite_telescoping` | 1 | 3 | none |
| `finite_telescoping_product_symbolic_shift_symbolic_lower` | `finite_telescoping` | 1 | 2 | none |
| `finite_telescoping_sum_affine_symbolic_arbitrary_shift_symbolic_lower` | `finite_telescoping` | 1 | 2 | none |
| `finite_telescoping_sum_affine_symbolic_shift_symbolic_lower` | `finite_telescoping` | 1 | 2 | none |
| `finite_telescoping_sum_basic` | `finite_telescoping` | 1 | 2 | none |
| `finite_telescoping_sum_symbolic_shift_symbolic_lower` | `finite_telescoping` | 1 | 2 | none |
| `hyperbolic_composition_sinh_asinh` | `simplify` | 1 | 2 | none |
| `hyperbolic_contract_cosh_triple_angle` | `simplify` | 1 | 0 | none |
| `hyperbolic_contract_exp_decomposition` | `simplify` | 1 | 0 | none |
| `hyperbolic_contract_negated_negative_exp_decomposition` | `simplify` | 1 | 0 | none |
| `hyperbolic_contract_negative_exp_decomposition` | `simplify` | 1 | 0 | none |
| `hyperbolic_contract_sinh_double_angle` | `simplify` | 1 | 0 | none |
| `hyperbolic_contract_sinh_double_angle_with_passthrough` | `simplify` | 1 | 0 | none |
| `hyperbolic_contract_sinh_triple_angle` | `simplify` | 1 | 0 | none |
| `hyperbolic_contract_tanh_double_angle` | `simplify` | 1 | 0 | none |
| `hyperbolic_contract_tanh_quotient` | `simplify` | 1 | 0 | none |
| `hyperbolic_expand_cosh_double_angle_cosh_sq` | `simplify` | 1 | 0 | none |
| `hyperbolic_expand_cosh_double_angle_sum` | `simplify` | 1 | 0 | none |
| `hyperbolic_expand_cosh_to_exp_definition` | `simplify` | 1 | 0 | none |
| `hyperbolic_expand_exp_to_sum` | `simplify` | 1 | 0 | none |
| `hyperbolic_expand_sinh_to_exp_definition` | `simplify` | 1 | 0 | none |
| `hyperbolic_expand_tanh_to_exp_definition` | `simplify` | 1 | 0 | none |
| `hyperbolic_half_angle_cosh_forward` | `simplify` | 1 | 1 | none |
| `hyperbolic_half_angle_sinh_forward` | `simplify` | 1 | 1 | none |
| `hyperbolic_negative_tanh_parity` | `simplify` | 1 | 1 | none |
| `hyperbolic_pythagorean_identity` | `simplify` | 1 | 0 | none |
| `hyperbolic_pythagorean_identity_with_passthrough` | `simplify` | 1 | 0 | none |
| `hyperbolic_pythagorean_shifted_forward` | `simplify` | 1 | 0 | none |
| `hyperbolic_special_value_sinh_zero` | `simplify` | 1 | 0 | none |
| `hyperbolic_tanh_pythagorean_forward` | `simplify` | 1 | 0 | none |
| `hyperbolic_tanh_pythagorean_reverse` | `simplify` | 1 | 0 | none |
| `integrate_prep_dirichlet_basic` | `integrate_prep` | 1 | 1 | none |
| `integrate_prep_dirichlet_longer` | `integrate_prep` | 1 | 1 | none |
| `integrate_prep_dirichlet_reverse_basic` | `integrate_prep` | 1 | 1 | none |
| `integrate_prep_dirichlet_reverse_symbolic_scale_longer` | `integrate_prep` | 1 | 1 | none |
| `integrate_prep_dirichlet_symbolic_argument` | `integrate_prep` | 1 | 1 | none |
| `integrate_prep_dirichlet_symbolic_scale` | `integrate_prep` | 1 | 1 | none |
| `integrate_prep_dirichlet_symbolic_scale_longer` | `integrate_prep` | 1 | 1 | none |
| `integrate_prep_morrie_basic` | `integrate_prep` | 1 | 1 | none |
| `integrate_prep_morrie_reverse_basic` | `integrate_prep` | 1 | 1 | none |
| `integrate_prep_morrie_reverse_symbolic_scale_longer` | `integrate_prep` | 1 | 1 | none |
| `integrate_prep_morrie_symbolic_argument` | `integrate_prep` | 1 | 1 | none |
| `integrate_prep_morrie_symbolic_scale` | `integrate_prep` | 1 | 1 | none |
| `inverse_hyperbolic_atanh_square_ratio_log` | `simplify` | 1 | 1 | none |
| `inverse_tan_identity` | `simplify` | 1 | 0 | none |
| `inverse_trig_arcsin_arccos_complement_sum` | `simplify` | 1 | 1 | none |
| `inverse_trig_composition_sin_arcsin` | `simplify` | 1 | 2 | none |
| `inverse_trig_special_value_arctan_sqrt_three` | `simplify` | 1 | 0 | none |
| `log_exp_inverse_ln_exp` | `log_exp_inverse` | 1 | 0 | none |
| `log_exp_inverse_ln_exp_power` | `log_exp_inverse` | 2 | 1 | none |
| `log_exp_inverse_ln_exp_product` | `log_exp_inverse` | 1 | 1 | none |
| `log_exp_inverse_log10_power_alias` | `log_exp_inverse` | 1 | 2 | none |
| `log_exp_inverse_natural_log_power_alias` | `log_exp_inverse` | 1 | 2 | none |
| `log_inverse_power_tower` | `log_inverse_power` | 2 | 2 | none |
| `log_inverse_power_unary_natural_alias` | `log_inverse_power` | 2 | 2 | none |
| `log_sum_difference_cancels_to_zero` | `simplify` | 1 | 0 | none |
| `merge_four_same_base_symbolic_powers` | `power_merge` | 1 | 0 | none |
| `merge_mixed_root_and_fractional_power_five_sixths` | `power_merge` | 1 | 0 | none |
| `merge_mixed_root_and_fractional_powers_to_integer_with_passthrough` | `power_merge` | 1 | 0 | none |
| `merge_mixed_root_and_symbolic_power` | `power_merge` | 1 | 0 | none |
| `merge_same_base_fractional_powers` | `power_merge` | 1 | 0 | none |
| `merge_same_base_fractional_powers_to_integer` | `power_merge` | 1 | 0 | none |
| `merge_same_base_integer_and_fractional_power` | `power_merge` | 1 | 0 | none |
| `merge_same_base_integer_and_symbolic_power` | `power_merge` | 1 | 0 | none |
| `merge_same_base_symbolic_powers` | `power_merge` | 1 | 0 | none |
| `merge_same_base_symbolic_quotient_powers` | `power_merge` | 1 | 2 | none |
| `merge_sqrt_product_requires_nonnegative` | `simplify` | 1 | 0 | none |
| `nested_fraction_fraction_over_sum_with_fraction_general` | `nested_fraction` | 1 | 2 | none |
| `nested_fraction_fraction_over_sum_with_fraction_general_reverse` | `nested_fraction` | 1 | 1 | none |
| `nested_fraction_one_over_sum` | `nested_fraction` | 1 | 2 | none |
| `nested_fraction_one_over_sum_with_fraction` | `nested_fraction` | 1 | 2 | none |
| `nested_fraction_one_over_sum_with_fraction_reverse` | `nested_fraction` | 1 | 1 | none |
| `nested_fraction_one_over_sum_with_passthrough` | `nested_fraction` | 1 | 2 | none |
| `nested_fraction_one_over_sum_with_subtractive_passthrough` | `nested_fraction` | 1 | 2 | none |
| `nested_fraction_one_over_three_reciprocals` | `nested_fraction` | 1 | 2 | none |
| `nested_fraction_reciprocal_inverse` | `nested_fraction` | 1 | 2 | none |
| `nested_fraction_sum_over_reciprocal` | `nested_fraction` | 1 | 2 | none |
| `nested_fraction_sum_with_fraction_over_scalar_general` | `nested_fraction` | 1 | 2 | none |
| `nested_fraction_sum_with_fraction_over_scalar_general_reverse` | `nested_fraction` | 1 | 1 | none |
| `nested_radical_denesting` | `simplify` | 2 | 2 | none |
| `perfect_square_root_direct_power_to_abs` | `simplify` | 1 | 2 | none |
| `perfect_square_root_to_abs` | `simplify` | 1 | 2 | none |
| `perfect_square_root_to_abs_with_passthrough` | `simplify` | 1 | 2 | none |
| `pythagorean_factor_form_from_sin_sq` | `simplify` | 1 | 0 | none |
| `pythagorean_factor_form_to_cos_sq` | `simplify` | 1 | 0 | none |
| `pythagorean_identity` | `simplify` | 1 | 0 | none |
| `radical_notable_quotient` | `rationalize` | 1 | 0 | none |
| `rationalize_cube_root_sum_denominator` | `rationalize` | 1 | 2 | none |
| `rationalize_linear_root` | `rationalize` | 1 | 3 | none |
| `rationalize_linear_root_plus` | `rationalize` | 1 | 3 | none |
| `rationalize_shifted_linear_root` | `rationalize` | 1 | 3 | none |
| `rationalize_symbolic_linear_root` | `rationalize` | 2 | 4 | none |
| `rationalize_symbolic_linear_root_alt_var` | `rationalize` | 2 | 4 | none |
| `rationalize_symbolic_linear_root_plus` | `rationalize` | 1 | 3 | none |
| `rationalize_then_cancel_to_zero` | `rationalize` | 2 | 3 | none |
| `reciprocal_trig_cos_sec_product_to_one` | `simplify` | 1 | 0 | none |
| `reciprocal_trig_product_to_one` | `simplify` | 1 | 0 | none |
| `reciprocal_trig_product_to_one_with_passthrough` | `simplify` | 1 | 0 | none |
| `reciprocal_trig_sin_csc_product_to_one` | `simplify` | 1 | 0 | none |
| `reciprocal_trig_special_value_sec_pi_fourth` | `simplify` | 1 | 0 | none |
| `sec_tan_pythagorean_to_one` | `simplify` | 1 | 0 | none |
| `simplify_sqrt_arithmetic_difference` | `simplify` | 1 | 1 | none |
| `simplify_sqrt_arithmetic_sum` | `simplify` | 1 | 1 | none |
| `sin_arccos_complement_projection` | `simplify` | 1 | 2 | none |
| `sin_arctan_right_triangle_projection` | `simplify` | 1 | 2 | none |
| `solve_prep_complete_square_alt_variable_symbolic_leading_coeff` | `solve_prep` | 1 | 3 | none |
| `solve_prep_complete_square_fractional_monic_numeric` | `solve_prep` | 1 | 2 | none |
| `solve_prep_complete_square_fractional_symbolic_leading_coeff` | `solve_prep` | 1 | 3 | none |
| `solve_prep_complete_square_monic_numeric` | `solve_prep` | 1 | 2 | none |
| `solve_prep_complete_square_negative_symbolic_leading_coeff` | `solve_prep` | 1 | 3 | none |
| `solve_prep_complete_square_symbolic_leading_coeff` | `solve_prep` | 1 | 3 | none |
| `solve_prep_complete_square_symbolic_monic_parametric` | `solve_prep` | 1 | 2 | none |
| `solve_prep_complete_square_symbolic_negative_linear_coeff` | `solve_prep` | 1 | 3 | none |
| `split_fraction_into_whole_plus_remainder` | `fraction_decompose` | 1 | 2 | none |
| `split_fraction_linear_over_scaled_linear` | `fraction_decompose` | 1 | 2 | none |
| `split_fraction_symbolic_over_general_shift` | `fraction_decompose` | 1 | 2 | none |
| `split_fraction_symbolic_over_negative_scaled_general_linear` | `fraction_decompose` | 1 | 2 | none |
| `split_fraction_symbolic_over_scaled_general_linear` | `fraction_decompose` | 1 | 2 | none |
| `split_telescoping_fraction_affine_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_affine_symbolic_shift_gap` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_consecutive` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_difference_squares_unfactored` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_negative_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_symbolic_difference_squares_unfactored` | `telescoping_fraction` | 1 | 2 | none |
| `square_of_square_root_requires_nonnegative` | `simplify` | 1 | 2 | none |
| `tan_arcsin_tangent_projection` | `simplify` | 1 | 2 | none |
| `trig_special_value_cos_two_pi_thirds_negative_half` | `simplify` | 1 | 0 | none |
| `trig_special_value_sin_zero` | `simplify` | 1 | 0 | none |

## arcsin_sin_arctan_safe_composition (simplify)

- Source: `asin(x/sqrt(x^2 + 1))`
- Target: `atan(x)`
- Result: `atan(x)`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: asin(x / sqrt(x^2 + 1))
Target: atan(x)
Strategy: rewrite inverse trigs
Steps (Aggressive Mode):
1. arcsin(x/sqrt(1+x^2)) = arctan(x)  [Aplicar composición trigonométrica inversa]
   Before: asin(x / sqrt(x^(2) + 1))
   Subpasos:
     1.1 Reconocer el argumento como seno de una arctangente
         x / sqrt(x^2 + 1) = sin(arctan(x))
     1.2 Sustituir ese seno dentro de arcsin
         asin(x / sqrt(x^2 + 1)) -> arcsin(sin(arctan(x)))
     1.3 Cancelar arcsin(sin(u)) en el rango principal
         arcsin(sin(arctan(x))) -> arctan(x)
   Cambio local: asin(x / sqrt(x^(2) + 1)) -> atan(x)
   After: atan(x)
Result: atan(x)
```

### Web / JSON Steps

1. `Aplicar composición trigonométrica inversa`
   - before: `arcsin(x/sqrt(x^2 + 1))`
   - after: `arctan(x)`
   - substeps:
     1. `Reconocer x/sqrt(1+x^2) como sin(arctan(x))`
     2. `Sustituir dentro de arcsin`
     3. `Usar asin(sin(u)) = u en el rango principal`

## cancel_fraction_common_factor_numeric (simplify)

- Source: `(2*x)/(4*x)`
- Target: `1/2`
- Result: `1 / 2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 * x / (4 * x)
Target: 1 / 2
Strategy: cancel fraction
Steps (Aggressive Mode):
1. Cancel common factor  [Cancelar un factor común]
   Before: 2 * x / (4 * x)
   Cambio local: 2 * x / (4 * x) -> 1 / 2
   After: 1 / 2
Result: 1 / 2
ℹ️ Requires:
  • x ≠ 0
```

### Web / JSON Steps

1. `Cancelar un factor común`
   - before: `(2 · x)/(4 · x)`
   - after: `1/2`
   - substeps:
     1. `Cancelar el factor común x`
     2. `Reducir la fracción que queda`

## cancel_fraction_difference_cubes (simplify)

- Source: `(a^3-b^3)/(a-b)`
- Target: `a^2+a*b+b^2`
- Result: `a^2 + b^2 + a * b`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: (a^3 - b^3) / (a - b)
Target: a^2 + b^2 + a * b
Strategy: cancel fraction
Steps (Aggressive Mode):
1. Factor numerator as a sum or difference of cubes and cancel the common factor  [Factorizar cubos y cancelar]
   Before: (a^(3) - b^(3)) / (a - b)
   Cambio local: (a^(3) - b^(3)) / (a - b) -> a^(2) + b^(2) + a * b
   After: a^2 + b^2 + a * b
Result: a^(2) + b^(2) + a * b
ℹ️ Requires:
  • a - b ≠ 0
```

### Web / JSON Steps

1. `Factorizar cubos y cancelar`
   - before: `(a^3 - b^3)/(a - b)`
   - after: `a^2 + b^2 + a · b`
   - substeps:
     1. `Factorizar el numerador como suma o diferencia de cubos`
     2. `Ahora se cancela el factor (a - b)`
     3. `Reemplazar ese bloque en la expresión`

## cancel_fraction_difference_cubes_with_passthrough (simplify)

- Source: `(a^3-b^3)/(a-b)+c`
- Target: `a^2+a*b+b^2+c`
- Result: `a^2 + b^2 + a * b + c`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: (a^3 - b^3) / (a - b) + c
Target: a^2 + b^2 + a * b + c
Strategy: cancel fraction
Steps (Aggressive Mode):
1. Factor numerator as a sum or difference of cubes and cancel the common factor  [Factorizar cubos y cancelar]
   Before: (a^(3) - b^(3)) / (a - b) + c
   Cambio local: (a^(3) - b^(3)) / (a - b) -> a^(2) + b^(2) + a * b
   After: a^2 + b^2 + a * b + c
Result: a^(2) + b^(2) + a * b + c
ℹ️ Requires:
  • a - b ≠ 0
```

### Web / JSON Steps

1. `Factorizar cubos y cancelar`
   - before: `(a^3 - b^3)/(a - b) + c`
   - after: `a^2 + b^2 + a · b + c`
   - substeps:
     1. `Factorizar el numerador como suma o diferencia de cubos`
     2. `Ahora se cancela el factor (a - b)`
     3. `Reemplazar ese bloque en la expresión`

## cancel_fraction_difference_squares (simplify)

- Source: `(a^2-b^2)/(a-b)`
- Target: `a+b`
- Result: `a + b`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a^2 - b^2) / (a - b)
Target: a + b
Strategy: cancel fraction
Steps (Aggressive Mode):
1. Cancel common factor  [Factorizar una diferencia de cuadrados y cancelar]
   Before: (a^(2) - b^(2)) / (a - b)
   Cambio local: (a^(2) - b^(2)) / (a - b) -> a + b
   After: a + b
Result: a + b
ℹ️ Requires:
  • a - b ≠ 0
```

### Web / JSON Steps

1. `Factorizar una diferencia de cuadrados y cancelar`
   - before: `(a^2 - b^2)/(a - b)`
   - after: `a + b`
   - substeps:
     1. `Factorizar el numerador como diferencia de cuadrados`
     2. `Ahora se cancela el factor a - b`

## cancel_fraction_difference_squares_with_passthrough (simplify)

- Source: `(a^2-b^2)/(a-b)+c`
- Target: `a+b+c`
- Result: `a + b + c`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a^2 - b^2) / (a - b) + c
Target: a + b + c
Strategy: cancel fraction
Steps (Aggressive Mode):
1. Cancel common factor  [Factorizar una diferencia de cuadrados y cancelar]
   Before: (a^(2) - b^(2)) / (a - b) + c
   Cambio local: (a^(2) - b^(2)) / (a - b) -> a + b
   After: a + b + c
Result: a + b + c
ℹ️ Requires:
  • a - b ≠ 0
```

### Web / JSON Steps

1. `Factorizar una diferencia de cuadrados y cancelar`
   - before: `(a^2 - b^2)/(a - b) + c`
   - after: `a + b + c`
   - substeps:
     1. `Factorizar el numerador como diferencia de cuadrados`
     2. `Ahora se cancela el factor a - b`

## cancel_fraction_monomial_common_factor (simplify)

- Source: `(a*x^2)/(b*x)`
- Target: `(a*x)/b`
- Result: `a * x / b`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * x^2 / (b * x)
Target: a * x / b
Strategy: cancel fraction
Steps (Aggressive Mode):
1. Cancel common factor  [Cancelar un factor común]
   Before: a * x^(2) / (b * x)
   Cambio local: a * x^(2) / (b * x) -> a * x / b
   After: a * x / b
Result: a * x / b
ℹ️ Requires:
  • b ≠ 0
  • x ≠ 0
```

### Web / JSON Steps

1. `Cancelar un factor común`
   - before: `(a · x^2)/(b · x)`
   - after: `(a · x)/b`
   - substeps:
     1. `Descomponer x^2 para exponer el factor común x`
     2. `Cancelar el factor común x`

## cancel_fraction_perfect_square_minus_symbolic (simplify)

- Source: `(a^2-2*a*b+b^2)/(a-b)`
- Target: `a-b`
- Result: `a - b`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a^2 + b^2 - 2 * a * b) / (a - b)
Target: a - b
Strategy: cancel fraction
Steps (Aggressive Mode):
1. Cancel common factor  [Cancelar un cuadrado perfecto con el mismo binomio]
   Before: (a^(2) + b^(2) - 2 * a * b) / (a - b)
   Cambio local: (a^(2) + b^(2) - 2 * a * b) / (a - b) -> (a - b)^(2) / (a - b)
   After: a - b
Result: a - b
ℹ️ Requires:
  • a - b ≠ 0
```

### Web / JSON Steps

1. `Cancelar un cuadrado perfecto con el mismo binomio`
   - before: `(a^2 - 2 · a · b + b^2)/(a - b)`
   - after: `a - b`
   - substeps:
     1. `Reconocer que el numerador es un cuadrado perfecto`
     2. `Si (a - b)^2 está dividido entre a - b, queda una sola copia`

## cancel_fraction_sum_cubes (simplify)

- Source: `(a^3+b^3)/(a+b)`
- Target: `a^2-a*b+b^2`
- Result: `a^2 + b^2 - a * b`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: (a^3 + b^3) / (a + b)
Target: a^2 + b^2 - a * b
Strategy: cancel fraction
Steps (Aggressive Mode):
1. Factor numerator as a sum or difference of cubes and cancel the common factor  [Factorizar cubos y cancelar]
   Before: (a^(3) + b^(3)) / (a + b)
   Cambio local: (a^(3) + b^(3)) / (a + b) -> a^(2) + b^(2) - a * b
   After: a^2 + b^2 - a * b
Result: a^(2) + b^(2) - a * b
ℹ️ Requires:
  • a + b ≠ 0
```

### Web / JSON Steps

1. `Factorizar cubos y cancelar`
   - before: `(a^3 + b^3)/(a + b)`
   - after: `a^2 - a · b + b^2`
   - substeps:
     1. `Factorizar el numerador como suma o diferencia de cubos`
     2. `Ahora se cancela el factor (a + b)`
     3. `Reemplazar ese bloque en la expresión`

## choose_numeric_binomial_coefficient (number_theory)

- Source: `choose(5,2)`
- Target: `10`
- Result: `10`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: choose(5, 2)
Target: 10
Strategy: number theory
Steps (Aggressive Mode):
1. choose(5, 2)  [Calcular coeficiente binomial]
   Before: choose(5, 2)
   Cambio local: choose(5, 2) -> 10
   After: 10
Result: 10
```

### Web / JSON Steps

1. `Calcular coeficiente binomial`
   - before: `choose(5, 2)`
   - after: `10`
   - substeps:
     1. `Usar C(5,2) = 5! / (2! · 3!)`
     2. `Calcular 5! / (2! · 3!) = 10`

## choose_numeric_pascal_identity (number_theory)

- Source: `choose(4,1) + choose(4,2)`
- Target: `choose(5,2)`
- Result: `choose(5, 2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: choose(4, 1) + choose(4, 2)
Target: choose(5, 2)
Strategy: number theory
Steps (Aggressive Mode):
1. Apply Pascal's identity for binomial coefficients  [Aplicar identidad de Pascal]
   Before: choose(4, 1) + choose(4, 2)
   Cambio local: choose(4, 1) + choose(4, 2) -> choose(5, 2)
   After: choose(5, 2)
Result: choose(5, 2)
```

### Web / JSON Steps

1. `Aplicar identidad de Pascal`
   - before: `choose(4, 1) + choose(4, 2)`
   - after: `choose(5, 2)`
   - substeps:
     1. `Usar C(4,1) + C(4,2) = C(5,2)`

## choose_numeric_symmetry (number_theory)

- Source: `choose(6,1)`
- Target: `choose(6,5)`
- Result: `choose(6, 5)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: choose(6, 1)
Target: choose(6, 5)
Strategy: number theory
Steps (Aggressive Mode):
1. Apply binomial coefficient symmetry  [Aplicar simetría del coeficiente binomial]
   Before: choose(6, 1)
   Cambio local: choose(6, 1) -> choose(6, 5)
   After: choose(6, 5)
Result: choose(6, 5)
```

### Web / JSON Steps

1. `Aplicar simetría del coeficiente binomial`
   - before: `choose(6, 1)`
   - after: `choose(6, 5)`
   - substeps:
     1. `Usar C(6,1) = C(6,6-1)`
     2. `Calcular 6-1 = 5`

## collapse_exponential_log_product (simplify)

- Source: `exp(ln(x)+ln(y))`
- Target: `x*y`
- Result: `x * y`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: e^(ln(x) + ln(y))
Target: x * y
Strategy: rewrite exponentials
Steps (Aggressive Mode):
1. Expand exp(u ± v ± ...) into products/quotients of exponentials  [Expandir exponencial de suma o diferencia]
   Before: e^(ln(x) + ln(y))
   Cambio local: e^(ln(x) + ln(y)) -> x * y
   After: x * y
Result: x * y
ℹ️ Requires:
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Reescribir exponenciales`
   - before: `e^(ln(x) + ln(y))`
   - after: `x · y`
   - substeps:
     1. `Separar la suma o resta del exponente en productos de exponenciales`
     2. `Cancelar e^(k·ln(u)) como potencia en cada factor`

## collapse_exponential_scaled_log_product (simplify)

- Source: `exp(2*ln(x)+3*ln(y))`
- Target: `x^2*y^3`
- Result: `x^2 * y^3`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: e^(2 * ln(x) + 3 * ln(y))
Target: x^2 * y^3
Strategy: rewrite exponentials
Steps (Aggressive Mode):
1. Expand exp(u ± v ± ...) into products/quotients of exponentials  [Expandir exponencial de suma o diferencia]
   Before: e^(2 * ln(x) + 3 * ln(y))
   Cambio local: e^(2 * ln(x) + 3 * ln(y)) -> x^(2) * y^(3)
   After: x^2 * y^3
Result: x^(2) * y^(3)
ℹ️ Requires:
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Reescribir exponenciales`
   - before: `e^(2 · ln(x) + 3 · ln(y))`
   - after: `x^2 · y^3`
   - substeps:
     1. `Separar la suma o resta del exponente en productos de exponenciales`
     2. `Cancelar e^(k·ln(u)) como potencia en cada factor`

## collect_common_symbolic_coefficients (collect)

- Source: `x*y + x*z + w`
- Target: `x*(y + z) + w`
- Result: `x * (y + z) + w`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: x * y + x * z + w
Target: x * (y + z) + w
Strategy: collect
Steps (Aggressive Mode):
1. Collect terms by x  [Agrupar términos por variable]
   Before: x * y + x * z + w
   Cambio local: x * y + x * z + w -> x * (y + z) + w
   After: x * (y + z) + w
Result: x * (y + z) + w
```

### Web / JSON Steps

1. `Agrupar términos por variable`
   - before: `x · y + x · z + w`
   - after: `x · (y + z) + w`
   - substeps: none

## collect_composite_monomial_factor (collect)

- Source: `a*x*y + b*x*y + c`
- Target: `(a + b)*x*y + c`
- Result: `x * y * (a + b) + c`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a * x * y + b * x * y + c
Target: x * y * (a + b) + c
Strategy: collect
Steps (Aggressive Mode):
1. Collect terms by x * y  [Agrupar términos por factor común]
   Before: a * x * y + b * x * y + c
   Cambio local: a * x * y + b * x * y + c -> x * y * (a + b) + c
   After: x * y * (a + b) + c
Result: x * y * (a + b) + c
```

### Web / JSON Steps

1. `Agrupar términos por factor común`
   - before: `a · x · y + b · x · y + c`
   - after: `x · y · (a + b) + c`
   - substeps: none

## collect_linear (collect)

- Source: `a*x + b*x + c`
- Target: `(a + b)*x + c`
- Result: `x * (a + b) + c`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a * x + b * x + c
Target: x * (a + b) + c
Strategy: collect
Steps (Aggressive Mode):
1. Collect terms by x  [Agrupar términos por variable]
   Before: a * x + b * x + c
   Cambio local: a * x + b * x + c -> x * (a + b) + c
   After: x * (a + b) + c
Result: x * (a + b) + c
```

### Web / JSON Steps

1. `Agrupar términos por variable`
   - before: `a · x + b · x + c`
   - after: `x · (a + b) + c`
   - substeps: none

## collect_linear_alt_variable (collect)

- Source: `a*y + b*y + c`
- Target: `(a + b)*y + c`
- Result: `y * (a + b) + c`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a * y + b * y + c
Target: y * (a + b) + c
Strategy: collect
Steps (Aggressive Mode):
1. Collect terms by y  [Agrupar términos por variable]
   Before: a * y + b * y + c
   Cambio local: a * y + b * y + c -> y * (a + b) + c
   After: y * (a + b) + c
Result: y * (a + b) + c
```

### Web / JSON Steps

1. `Agrupar términos por variable`
   - before: `a · y + b · y + c`
   - after: `y · (a + b) + c`
   - substeps: none

## collect_multiple_power_groups (collect)

- Source: `a*x^2 + b*x + c*x^2 + d*x + e*x^2 + f`
- Target: `(a + c + e)*x^2 + (b + d)*x + f`
- Result: `x * (b + d) + x^2 * (a + c + e) + f`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a * x^2 + c * x^2 + e * x^2 + b * x + d * x + f
Target: x * (b + d) + x^2 * (a + c + e) + f
Strategy: collect
Steps (Aggressive Mode):
1. Collect terms by x  [Agrupar términos por variable]
   Before: a * x^(2) + c * x^(2) + e * x^(2) + b * x + d * x + f
   Cambio local: a * x^(2) + c * x^(2) + e * x^(2) + b * x + d * x + f -> x * (b + d) + x^(2) * (a + c + e) + f
   After: x * (b + d) + x^2 * (a + c + e) + f
Result: x * (b + d) + x^(2) * (a + c + e) + f
```

### Web / JSON Steps

1. `Agrupar términos por variable`
   - before: `a · x^2 + c · x^2 + e · x^2 + b · x + d · x + f`
   - after: `x · (b + d) + x^2 · (a + c + e) + f`
   - substeps: none

## collect_two_composite_factor_groups (collect)

- Source: `a*x*y + b*x*y + c*x*z + d*x*z + e`
- Target: `(a + b)*x*y + (c + d)*x*z + e`
- Result: `x * y * (a + b) + x * z * (c + d) + e`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a * x * y + b * x * y + c * x * z + d * x * z + e
Target: x * y * (a + b) + x * z * (c + d) + e
Strategy: collect
Steps (Aggressive Mode):
1. Collect terms by x * y  [Agrupar términos por factor común]
   Before: a * x * y + b * x * y + c * x * z + d * x * z + e
   Cambio local: a * x * y + b * x * y + c * x * z + d * x * z + e -> x * y * (a + b) + x * z * (c + d) + e
   After: x * y * (a + b) + x * z * (c + d) + e
Result: x * y * (a + b) + x * z * (c + d) + e
```

### Web / JSON Steps

1. `Agrupar términos por factor común`
   - before: `a · x · y + b · x · y + c · x · z + d · x · z + e`
   - after: `x · y · (a + b) + x · z · (c + d) + e`
   - substeps: none

## combine_fraction_part_with_same_denominator (fraction_combine)

- Source: `1 + a/d + b/d`
- Target: `1 + (a+b)/d`
- Result: `(a + b) / d + 1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a / d + b / d + 1
Target: (a + b) / d + 1
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine fractions that already share the same denominator  [Combinar fracciones con el mismo denominador]
   Before: a / d + b / d + 1
   Cambio local: a / d + b / d -> (a + b) / d
   After: (a + b) / d + 1
Result: (a + b) / d + 1
ℹ️ Requires:
  • d ≠ 0
```

### Web / JSON Steps

1. `Sumar fracciones con mismo denominador`
   - before: `a/d + b/d + 1`
   - after: `(a + b)/d + 1`
   - substeps: none

## combine_fraction_part_with_same_denominator_three_terms (fraction_combine)

- Source: `1 + a/d + b/d + c/d`
- Target: `1 + (a+b+c)/d`
- Result: `(a + b + c) / d + 1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a / d + b / d + c / d + 1
Target: (a + b + c) / d + 1
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine fractions that already share the same denominator  [Combinar fracciones con el mismo denominador]
   Before: a / d + b / d + c / d + 1
   Cambio local: a / d + b / d + c / d -> (a + b + c) / d
   After: (a + b + c) / d + 1
Result: (a + b + c) / d + 1
ℹ️ Requires:
  • d ≠ 0
```

### Web / JSON Steps

1. `Sumar fracciones con mismo denominador`
   - before: `a/d + b/d + c/d + 1`
   - after: `(a + b + c)/d + 1`
   - substeps: none

## combine_general_fraction_difference (fraction_combine)

- Source: `a/x - b/y`
- Target: `(a*y-b*x)/(x*y)`
- Result: `(a * y - b * x) / (x * y)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a / x - b / y
Target: (a * y - b * x) / (x * y)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Subtract two fractions into a single denominator  [Restar fracciones en un solo denominador]
   Before: a / x - b / y
   Cambio local: a / x - b / y -> (a * y - b * x) / (x * y)
   After: (a * y - b * x) / (x * y)
Result: (a * y - b * x) / (x * y)
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Restar fracciones`
   - before: `a/x - b/y`
   - after: `(a · y - b · x)/(x · y)`
   - substeps: none

## combine_general_fraction_sum (fraction_combine)

- Source: `a/x + b/y`
- Target: `(a*y+b*x)/(x*y)`
- Result: `(a * y + b * x) / (x * y)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a / x + b / y
Target: (a * y + b * x) / (x * y)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine two fractions into a single denominator  [Sumar fracciones en un solo denominador]
   Before: a / x + b / y
   Cambio local: a / x + b / y -> (a * y + b * x) / (x * y)
   After: (a * y + b * x) / (x * y)
Result: (a * y + b * x) / (x * y)
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Sumar fracciones`
   - before: `a/x + b/y`
   - after: `(a · y + b · x)/(x · y)`
   - substeps: none

## combine_like_terms (simplify)

- Source: `x + x`
- Target: `2*x`
- Result: `2 * x`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: x + x
Target: 2 * x
Strategy: combine like terms
Steps (Aggressive Mode):
1. Combine like terms  [Agrupar términos semejantes]
   Before: x + x
   Cambio local: x + x -> 2 * x
   After: 2 * x
Result: 2 * x
```

### Web / JSON Steps

1. `Agrupar términos semejantes`
   - before: `x + x`
   - after: `2 · x`
   - substeps:
     1. `Sumar los coeficientes que acompañan a x`

## combine_negative_scaled_symbolic_whole_plus_remainder_into_fraction (fraction_combine)

- Source: `-a/c + (b+a*d/c)/(d-c*x)`
- Target: `(a*x+b)/(d-c*x)`
- Result: `(a * x + b) / (d - c * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (a * d / c + b) / (d - c * x) - a / c
Target: (a * x + b) / (d - c * x)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine the whole part with the remaining fraction  [Combinar parte entera y fracción]
   Before: (a * d / c + b) / (d - c * x) + a / c
   After: (a * x + b) / (d - c * x)
Result: (a * x + b) / (d - c * x)
ℹ️ Requires:
  • c ≠ 0
  • c * x - d ≠ 0
```

### Web / JSON Steps

1. `Unir parte entera y fracción`
   - before: `((a · d)/c + b)/(d - c · x) - a/c`
   - after: `(a · x + b)/(d - c · x)`
   - substeps: none

## combine_same_denominator_fraction_difference (fraction_combine)

- Source: `a/d - b/d`
- Target: `(a-b)/d`
- Result: `(a - b) / d`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a / d - b / d
Target: (a - b) / d
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine fractions with the same denominator into one subtraction  [Combinar resta de fracciones con el mismo denominador]
   Before: a / d - b / d
   Cambio local: a / d - b / d -> (a - b) / d
   After: (a - b) / d
Result: (a - b) / d
ℹ️ Requires:
  • d ≠ 0
```

### Web / JSON Steps

1. `Restar fracciones con mismo denominador`
   - before: `a/d - b/d`
   - after: `(a - b)/d`
   - substeps: none

## combine_same_denominator_fraction_sum (fraction_combine)

- Source: `a/d + b/d`
- Target: `(a+b)/d`
- Result: `(a + b) / d`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a / d + b / d
Target: (a + b) / d
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine fractions that already share the same denominator  [Combinar fracciones con el mismo denominador]
   Before: a / d + b / d
   Cambio local: a / d + b / d -> (a + b) / d
   After: (a + b) / d
Result: (a + b) / d
ℹ️ Requires:
  • d ≠ 0
```

### Web / JSON Steps

1. `Sumar fracciones con mismo denominador`
   - before: `a/d + b/d`
   - after: `(a + b)/d`
   - substeps: none

## combine_scaled_symbolic_whole_plus_remainder_into_fraction (fraction_combine)

- Source: `a/c + (b-a*d/c)/(c*x+d)`
- Target: `(a*x+b)/(c*x+d)`
- Result: `(a * x + b) / (c * x + d)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a / c + (b - a * d / c) / (c * x + d)
Target: (a * x + b) / (c * x + d)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine the whole part with the remaining fraction  [Combinar parte entera y fracción]
   Before: a / c + (b - a * d / c) / (c * x + d)
   After: (a * x + b) / (c * x + d)
Result: (a * x + b) / (c * x + d)
ℹ️ Requires:
  • c * x + d ≠ 0
  • c ≠ 0
```

### Web / JSON Steps

1. `Unir parte entera y fracción`
   - before: `a/c + (b - a · d/c)/(c · x + d)`
   - after: `(a · x + b)/(c · x + d)`
   - substeps: none

## combine_symbolic_same_denominator_fraction_subset_with_passthrough (fraction_combine)

- Source: `a/(x+y) + b/(x+y) + c`
- Target: `(a+b)/(x+y) + c`
- Result: `(a + b) / (x + y) + c`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a / (x + y) + b / (x + y) + c
Target: (a + b) / (x + y) + c
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine fractions that already share the same denominator  [Combinar fracciones con el mismo denominador]
   Before: a / (x + y) + b / (x + y) + c
   Cambio local: a / (x + y) + b / (x + y) -> (a + b) / (x + y)
   After: (a + b) / (x + y) + c
Result: (a + b) / (x + y) + c
ℹ️ Requires:
  • x + y ≠ 0
```

### Web / JSON Steps

1. `Sumar fracciones con mismo denominador`
   - before: `a/(x + y) + b/(x + y) + c`
   - after: `(a + b)/(x + y) + c`
   - substeps: none

## combine_symbolic_whole_plus_remainder_into_fraction (fraction_combine)

- Source: `a + (b-a*c)/(x+c)`
- Target: `(a*x+b)/(x+c)`
- Result: `(a * x + b) / (c + x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (b - a * c) / (c + x) + a
Target: (a * x + b) / (c + x)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine the whole part with the remaining fraction  [Combinar parte entera y fracción]
   Before: (b - a * c) / (c + x) + a
   After: (a * x + b) / (c + x)
Result: (a * x + b) / (c + x)
ℹ️ Requires:
  • c + x ≠ 0
```

### Web / JSON Steps

1. `Unir parte entera y fracción`
   - before: `(b - a · c)/(c + x) + a`
   - after: `(a · x + b)/(c + x)`
   - substeps: none

## combine_telescoping_fraction_affine_gap_two (telescoping_fraction)

- Source: `1/2*(1/(2*n+1) - 1/(2*n+3))`
- Target: `1/((2*n+1)*(2*n+3))`
- Result: `1 / ((2 * n + 1) * (2 * n + 3))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ((1 / (2 * n + 1) - 1 / (2 * n + 3)) * 1)/2
Target: 1 / ((2 * n + 1) * (2 * n + 3))
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Recomponer fracciones parciales telescópicas]
   Before: 1 / 2 * (1 / (2 * n + 1) - 1 / (2 * n + 3))
   Cambio local: 1 / 2 * (1 / (2 * n + 1) - 1 / (2 * n + 3)) -> 1 / ((2 * n + 1) * (2 * n + 3))
   After: 1 / ((2 * n + 1) * (2 * n + 3))
Result: 1 / ((2 * n + 1) * (2 * n + 3))
ℹ️ Requires:
  • n ≠ -1/2
  • n ≠ -3/2
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/2 · (1/(2 · n + 1) - 1/(2 · n + 3))`
   - after: `1/((2 · n + 1) · (2 · n + 3))`
   - substeps:
     1. `Llevar las fracciones al denominador común`
     2. `Simplificar el numerador telescópico`

## combine_telescoping_fraction_affine_symbolic_shift_gap (telescoping_fraction)

- Source: `1/(c-b)*(1/(a*n+b) - 1/(a*n+c))`
- Target: `1/((a*n+b)*(a*n+c))`
- Result: `1 / ((a * n + b) * (a * n + c))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ((1 / (a * n + b) - 1 / (a * n + c)) * 1)/(c - b)
Target: 1 / ((a * n + b) * (a * n + c))
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Recomponer fracciones parciales telescópicas]
   Before: 1 / (c - b) * (1 / (a * n + b) - 1 / (a * n + c))
   Cambio local: 1 / (c - b) * (1 / (a * n + b) - 1 / (a * n + c)) -> 1 / ((a * n + b) * (a * n + c))
   After: 1 / ((a * n + b) * (a * n + c))
Result: 1 / ((a * n + b) * (a * n + c))
ℹ️ Requires:
  • a * n + b ≠ 0
  • a * n + c ≠ 0
  • b - c ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/(c - b) · (1/(a · n + b) - 1/(a · n + c))`
   - after: `1/((a · n + b) · (a · n + c))`
   - substeps:
     1. `Llevar las fracciones al denominador común`
     2. `Simplificar el numerador telescópico`

## combine_telescoping_fraction_consecutive (telescoping_fraction)

- Source: `1/n - 1/(n+1)`
- Target: `1/(n*(n+1))`
- Result: `1 / (n * (n + 1))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / n - 1 / (n + 1)
Target: 1 / (n * (n + 1))
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Recomponer fracciones parciales telescópicas]
   Before: 1 / n - 1 / (n + 1)
   Subpasos:
     1.1 Llevar las fracciones al denominador común
         1 / n - 1 / (n + 1) -> (n + 1 - n) / (n * (n + 1))
     1.2 Simplificar el numerador telescópico
         (n + 1 - n) / (n * (n + 1)) -> 1 / (n * (n + 1))
   Cambio local: 1 / n - 1 / (n + 1) -> 1 / (n * (n + 1))
   After: 1 / (n * (n + 1))
Result: 1 / (n * (n + 1))
ℹ️ Requires:
  • n ≠ 0
  • n ≠ -1
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/n - 1/(n + 1)`
   - after: `1/(n · (n + 1))`
   - substeps:
     1. `Llevar las fracciones al denominador común`
     2. `Simplificar el numerador telescópico`

## combine_telescoping_fraction_difference_squares_unfactored (telescoping_fraction)

- Source: `1/2*(1/(x-1) - 1/(x+1))`
- Target: `1/(x^2-1)`
- Result: `1 / (x^2 - 1)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ((1 / (x - 1) - 1 / (x + 1)) * 1)/2
Target: 1 / (x^2 - 1)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Recomponer fracciones parciales telescópicas]
   Before: 1 / 2 * (1 / (x - 1) - 1 / (x + 1))
   Cambio local: 1 / 2 * (1 / (x - 1) - 1 / (x + 1)) -> 1 / (x^(2) - 1)
   After: 1 / (x^2 - 1)
Result: 1 / (x^(2) - 1)
ℹ️ Requires:
  • x ≠ -1
  • x ≠ 1
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/2 · (1/(x - 1) - 1/(x + 1))`
   - after: `1/(x^2 - 1)`
   - substeps:
     1. `Llevar las fracciones al denominador común`
     2. `Simplificar el numerador telescópico`

## combine_telescoping_fraction_gap_two (telescoping_fraction)

- Source: `1/2*(1/n - 1/(n+2))`
- Target: `1/(n*(n+2))`
- Result: `1 / (n * (n + 2))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ((1 / n - 1 / (n + 2)) * 1)/2
Target: 1 / (n * (n + 2))
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Recomponer fracciones parciales telescópicas]
   Before: 1 / 2 * (1 / n - 1 / (n + 2))
   Cambio local: 1 / 2 * (1 / n - 1 / (n + 2)) -> 1 / (n * (n + 2))
   After: 1 / (n * (n + 2))
Result: 1 / (n * (n + 2))
ℹ️ Requires:
  • n ≠ 0
  • n ≠ -2
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/2 · (1/n - 1/(n + 2))`
   - after: `1/(n · (n + 2))`
   - substeps:
     1. `Llevar las fracciones al denominador común`
     2. `Simplificar el numerador telescópico`

## combine_telescoping_fraction_negative_gap_two (telescoping_fraction)

- Source: `1/2*(1/(n-2) - 1/n)`
- Target: `1/(n*(n-2))`
- Result: `1 / (n * (n - 2))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ((1 / (n - 2) - 1 / n) * 1)/2
Target: 1 / (n * (n - 2))
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Recomponer fracciones parciales telescópicas]
   Before: 1 / 2 * (1 / (n - 2) - 1 / n)
   Cambio local: 1 / 2 * (1 / (n - 2) - 1 / n) -> 1 / (n * (n - 2))
   After: 1 / (n * (n - 2))
Result: 1 / (n * (n - 2))
ℹ️ Requires:
  • n ≠ 0
  • n ≠ 2
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/2 · (1/(n - 2) - 1/n)`
   - after: `1/(n · (n - 2))`
   - substeps:
     1. `Llevar las fracciones al denominador común`
     2. `Simplificar el numerador telescópico`

## combine_telescoping_fraction_shifted_quadratic_unfactored (telescoping_fraction)

- Source: `1/(x+b) - 1/(x+c)`
- Target: `(c-b)/(x^2+(b+c)*x+b*c)`
- Result: `(c - b) / (x * (b + c) + x^2 + b * c)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (b + x) - 1 / (c + x)
Target: (c - b) / (x * (b + c) + x^2 + b * c)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Subtract two fractions into a single denominator  [Restar fracciones en un solo denominador]
   Before: 1 / (b + x) - 1 / (c + x)
   Cambio local: 1 / (b + x) - 1 / (c + x) -> (c - b) / (x * (b + c) + x^(2) + b * c)
   After: (c - b) / (x * (b + c) + x^2 + b * c)
Result: (c - b) / (x * (b + c) + x^(2) + b * c)
ℹ️ Requires:
  • b + x ≠ 0
  • c + x ≠ 0
```

### Web / JSON Steps

1. `Restar fracciones`
   - before: `1/(b + x) - 1/(c + x)`
   - after: `(c - b)/(x · (b + c) + x^2 + b · c)`
   - substeps:
     1. `Llevar a denominador común`
     2. `Simplificar el numerador y el denominador`

## combine_telescoping_fraction_symbolic_difference_squares_unfactored (telescoping_fraction)

- Source: `1/(2*a)*(1/(x-a) - 1/(x+a))`
- Target: `1/(x^2-a^2)`
- Result: `1 / (x^2 - a^2)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ((1 / (x - a) - 1 / (a + x)) * 1)/(2 * a)
Target: 1 / (x^2 - a^2)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Recomponer fracciones parciales telescópicas]
   Before: 1 / (2 * a) * (1 / (x - a) - 1 / (a + x))
   Cambio local: 1 / (2 * a) * (1 / (x - a) - 1 / (a + x)) -> 1 / (x^(2) - a^(2))
   After: 1 / (x^2 - a^2)
Result: 1 / (x^(2) - a^(2))
ℹ️ Requires:
  • a + x ≠ 0
  • a ≠ 0
  • a - x ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/(2 · a) · (1/(x - a) - 1/(a + x))`
   - after: `1/(x^2 - a^2)`
   - substeps:
     1. `Llevar las fracciones al denominador común`
     2. `Simplificar el numerador telescópico`

## combine_term_and_fraction_subtraction (fraction_combine)

- Source: `a - b/a`
- Target: `(a^2-b)/a`
- Result: `(a^2 - b) / a`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a - b / a
Target: (a^2 - b) / a
Strategy: combine fraction
Steps (Aggressive Mode):
1. Put the term and the fraction over the same denominator  [Combinar resta de fracciones con el mismo denominador]
   Before: a - b / a
   Cambio local: a - b / a -> (a^(2) - b) / a
   After: (a^2 - b) / a
Result: (a^(2) - b) / a
ℹ️ Requires:
  • a ≠ 0
```

### Web / JSON Steps

1. `Restar fracciones con mismo denominador`
   - before: `a - b/a`
   - after: `(a^2 - b)/a`
   - substeps: none

## combine_three_same_denominator_fractions (fraction_combine)

- Source: `a/d + b/d + c/d`
- Target: `(a+b+c)/d`
- Result: `(a + b + c) / d`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a / d + b / d + c / d
Target: (a + b + c) / d
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine fractions that already share the same denominator  [Combinar fracciones con el mismo denominador]
   Before: a / d + b / d + c / d
   Cambio local: a / d + b / d + c / d -> (a + b + c) / d
   After: (a + b + c) / d
Result: (a + b + c) / d
ℹ️ Requires:
  • d ≠ 0
```

### Web / JSON Steps

1. `Sumar fracciones con mismo denominador`
   - before: `a/d + b/d + c/d`
   - after: `(a + b + c)/d`
   - substeps: none

## combine_whole_plus_remainder_into_fraction (fraction_combine)

- Source: `1 + 2/(x-1)`
- Target: `(x+1)/(x-1)`
- Result: `(x + 1) / (x - 1)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 / (x - 1) + 1
Target: (x + 1) / (x - 1)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine the whole part with the remaining fraction  [Combinar parte entera y fracción]
   Before: 2 / (x - 1) + 1
   After: (x + 1) / (x - 1)
Result: (x + 1) / (x - 1)
ℹ️ Requires:
  • x ≠ 1
```

### Web / JSON Steps

1. `Unir parte entera y fracción`
   - before: `2/(x - 1) + 1`
   - after: `(x + 1)/(x - 1)`
   - substeps: none

## consecutive_factorial_ratio (simplify)

- Source: `(n+1)!/n!`
- Target: `n+1`
- Result: `n + 1`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (n + 1)! / n!
Target: n + 1
Strategy: rewrite factorials
Steps (Aggressive Mode):
1. Cancel consecutive factorials  [Cancelar factoriales consecutivos]
   Before: (n + 1)! / n!
   Cambio local: (n + 1)! / n! -> n + 1
   After: n + 1
Result: n + 1
ℹ️ Requires:
  • n! ≠ 0
```

### Web / JSON Steps

1. `Cancelar factoriales consecutivos`
   - before: `(n + 1)!/n!`
   - after: `n + 1`
   - substeps:
     1. `Escribir el factorial superior como el siguiente número por el factorial anterior`
     2. `Cancelar el factorial común`

## consecutive_factorial_ratio_gap_two (simplify)

- Source: `(n+1)!/(n-1)!`
- Target: `n*(n+1)`
- Result: `n * (n + 1)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (n + 1)! / (n - 1)!
Target: n * (n + 1)
Strategy: rewrite factorials
Steps (Aggressive Mode):
1. Cancel consecutive factorials  [Cancelar factoriales consecutivos]
   Before: (n + 1)! / (n - 1)!
   Cambio local: (n + 1)! / (n - 1)! -> n * (n + 1)
   After: n * (n + 1)
Result: n * (n + 1)
ℹ️ Requires:
  • (n - 1)! ≠ 0
```

### Web / JSON Steps

1. `Cancelar factoriales consecutivos`
   - before: `(n + 1)!/(n - 1)!`
   - after: `n · (n + 1)`
   - substeps:
     1. `Expandir el factorial superior hasta llegar al factorial inferior`
     2. `Cancelar el factorial común`

## consecutive_factorial_ratio_with_passthrough (simplify)

- Source: `(n+1)!/n!+a`
- Target: `n+1+a`
- Result: `a + n + 1`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (n + 1)! / n! + a
Target: a + n + 1
Strategy: rewrite factorials
Steps (Aggressive Mode):
1. Cancel consecutive factorials  [Cancelar factoriales consecutivos]
   Before: (n + 1)! / n! + a
   Cambio local: (n + 1)! / n! + a -> a + n + 1
   After: a + n + 1
Result: a + n + 1
ℹ️ Requires:
  • n! ≠ 0
```

### Web / JSON Steps

1. `Cancelar factoriales consecutivos`
   - before: `(n + 1)!/n! + a`
   - after: `a + n + 1`
   - substeps:
     1. `Escribir el factorial superior como el siguiente número por el factorial anterior`
     2. `Cancelar el factorial común`

## contract_even_abs_logs_to_scaled_abs_product (log_contract)

- Source: `2*ln(abs(x))+2*ln(abs(y))`
- Target: `2*ln(abs(x*y))`
- Result: `2 * ln(|x * y|)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * ln(|x|) + 2 * ln(|y|)
Target: 2 * ln(|x * y|)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: 2 * ln(|x|) + 2 * ln(|y|)
   Cambio local: 2 * ln(|x|) + 2 * ln(|y|) -> 2 * ln(|x * y|)
   After: 2 * ln(|x * y|)
Result: 2 * ln(|x * y|)
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `2 · ln(|x|) + 2 · ln(|y|)`
   - after: `2 · ln(|x · y|)`
   - substeps: none

## contract_even_abs_logs_to_scaled_abs_product_with_passthrough (log_contract)

- Source: `2*ln(abs(x))+2*ln(abs(y))+a`
- Target: `2*ln(abs(x*y))+a`
- Result: `2 * ln(|x * y|) + a`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * ln(|x|) + 2 * ln(|y|) + a
Target: 2 * ln(|x * y|) + a
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: 2 * ln(|x|) + 2 * ln(|y|) + a
   Cambio local: 2 * ln(|x|) + 2 * ln(|y|) + a -> 2 * ln(|x * y|) + a
   After: 2 * ln(|x * y|) + a
Result: 2 * ln(|x * y|) + a
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `2 · ln(|x|) + 2 · ln(|y|) + a`
   - after: `2 · ln(|x · y|) + a`
   - substeps: none

## contract_exponential_difference (simplify)

- Source: `exp(x)/exp(y)`
- Target: `exp(x-y)`
- Result: `e^(x - y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: e^x / e^y
Target: e^(x - y)
Strategy: rewrite exponentials
Steps (Aggressive Mode):
1. Contract products/quotients of exponentials into exp(u ± v ± ...)  [Contraer productos exponenciales]
   Before: e^(x) / e^(y)
   Cambio local: e^(x) / e^(y) -> e^(x - y)
   After: e^(x - y)
Result: e^(x - y)
```

### Web / JSON Steps

1. `Reescribir exponenciales`
   - before: `e^x/e^y`
   - after: `e^(x - y)`
   - substeps:
     1. `Usar e^A / e^B = e^(A-B)`

## contract_exponential_power (simplify)

- Source: `exp(x)^3`
- Target: `exp(3*x)`
- Result: `e^(3 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: e^x^3
Target: e^(3 * x)
Strategy: rewrite exponentials
Steps (Aggressive Mode):
1. Recognize exp(u)^n as exp(n·u)  [Aplicar potencia de una exponencial]
   Before: e^(x)^(3)
   Cambio local: e^(x)^(3) -> e^(3 * x)
   After: e^(3 * x)
Result: e^(3 * x)
```

### Web / JSON Steps

1. `Reescribir potencia exponencial`
   - before: `e^x^3`
   - after: `e^(3 · x)`
   - substeps:
     1. `Usar (e^A)^n = e^(n·A)`

## contract_exponential_reciprocal (simplify)

- Source: `1/exp(x)`
- Target: `exp(-x)`
- Result: `e^(-x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 1 / e^x
Target: e^(-x)
Strategy: rewrite exponentials
Steps (Aggressive Mode):
1. Recognize 1 / exp(u) as exp(-u)  [Reescribir recíproco exponencial]
   Before: 1 / e^(x)
   Cambio local: 1 / e^(x) -> e^(-x)
   After: e^(-x)
Result: e^(-x)
```

### Web / JSON Steps

1. `Reescribir recíproco exponencial`
   - before: `1/e^x`
   - after: `e^(-x)`
   - substeps:
     1. `Usar 1/e^A = e^(-A)`

## contract_exponential_sum (simplify)

- Source: `exp(x)*exp(y)`
- Target: `exp(x+y)`
- Result: `e^(x + y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: e^x * e^y
Target: e^(x + y)
Strategy: rewrite exponentials
Steps (Aggressive Mode):
1. Contract products/quotients of exponentials into exp(u ± v ± ...)  [Contraer productos exponenciales]
   Before: e^(x) * e^(y)
   Cambio local: e^(x) * e^(y) -> e^(x + y)
   After: e^(x + y)
Result: e^(x + y)
```

### Web / JSON Steps

1. `Reescribir exponenciales`
   - before: `e^x · e^y`
   - after: `e^(x + y)`
   - substeps:
     1. `Usar e^A · e^B = e^(A+B)`

## contract_general_base_logs_to_grouped_power (log_contract)

- Source: `2*log(b,x)+2*log(b,y)`
- Target: `log(b,(x*y)^2)`
- Result: `log(b, (x * y)^2)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * log(b, x) + 2 * log(b, y)
Target: log(b, (x * y)^2)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: 2 * log(b, x) + 2 * log(b, y)
   Cambio local: 2 * log(b, x) + 2 * log(b, y) -> log(b, (x * y)^(2))
   After: log(b, (x * y)^2)
Result: log(b, (x * y)^(2))
ℹ️ Requires:
  • b > 0
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `2 · log_b(x) + 2 · log_b(y)`
   - after: `log_b((x · y)^2)`
   - substeps: none

## contract_general_base_logs_to_grouped_power_with_passthrough (log_contract)

- Source: `2*log(b,x)+2*log(b,y)+a`
- Target: `log(b,(x*y)^2)+a`
- Result: `log(b, (x * y)^2) + a`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * log(b, x) + 2 * log(b, y) + a
Target: log(b, (x * y)^2) + a
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: 2 * log(b, x) + 2 * log(b, y) + a
   Cambio local: 2 * log(b, x) + 2 * log(b, y) + a -> log(b, (x * y)^(2)) + a
   After: log(b, (x * y)^2) + a
Result: log(b, (x * y)^(2)) + a
ℹ️ Requires:
  • b > 0
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `2 · log_b(x) + 2 · log_b(y) + a`
   - after: `log_b((x · y)^2) + a`
   - substeps: none

## contract_hyperbolic_cosh_difference (simplify)

- Source: `cosh(x)*cosh(y)-sinh(x)*sinh(y)`
- Target: `cosh(x-y)`
- Result: `cosh(x - y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cosh(x) * cosh(y) - sinh(x) * sinh(y)
Target: cosh(x - y)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize cosh(u)·cosh(v) ± sinh(u)·sinh(v) as cosh(u ± v)  [Aplicar identidad hiperbólica de suma/diferencia de ángulos]
   Before: cosh(x) * cosh(y) - sinh(x) * sinh(y)
   Cambio local: cosh(x) * cosh(y) - sinh(x) * sinh(y) -> cosh(x - y)
   After: cosh(x - y)
Result: cosh(x - y)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de suma/diferencia de ángulos`
   - before: `cosh(x) · cosh(y) - sinh(x) · sinh(y)`
   - after: `cosh(x - y)`
   - substeps:
     1. `Usar cosh(A-B) = cosh(A) · cosh(B) - sinh(A) · sinh(B)`

## contract_hyperbolic_cosh_sum (simplify)

- Source: `cosh(x)*cosh(y)+sinh(x)*sinh(y)`
- Target: `cosh(x+y)`
- Result: `cosh(x + y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sinh(x) * sinh(y) + cosh(x) * cosh(y)
Target: cosh(x + y)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize cosh(u)·cosh(v) ± sinh(u)·sinh(v) as cosh(u ± v)  [Aplicar identidad hiperbólica de suma/diferencia de ángulos]
   Before: sinh(x) * sinh(y) + cosh(x) * cosh(y)
   Cambio local: sinh(x) * sinh(y) + cosh(x) * cosh(y) -> cosh(x + y)
   After: cosh(x + y)
Result: cosh(x + y)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de suma/diferencia de ángulos`
   - before: `sinh(x) · sinh(y) + cosh(x) · cosh(y)`
   - after: `cosh(x + y)`
   - substeps:
     1. `Usar cosh(A+B) = cosh(A) · cosh(B) + sinh(A) · sinh(B)`

## contract_hyperbolic_sinh_difference (simplify)

- Source: `sinh(x)*cosh(y)-cosh(x)*sinh(y)`
- Target: `sinh(x-y)`
- Result: `sinh(x - y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sinh(x) * cosh(y) - sinh(y) * cosh(x)
Target: sinh(x - y)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize sinh(u)·cosh(v) ± cosh(u)·sinh(v) as sinh(u ± v)  [Aplicar identidad hiperbólica de suma/diferencia de ángulos]
   Before: sinh(x) * cosh(y) - sinh(y) * cosh(x)
   Cambio local: sinh(x) * cosh(y) - sinh(y) * cosh(x) -> sinh(x - y)
   After: sinh(x - y)
Result: sinh(x - y)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de suma/diferencia de ángulos`
   - before: `sinh(x) · cosh(y) - sinh(y) · cosh(x)`
   - after: `sinh(x - y)`
   - substeps:
     1. `Usar sinh(A-B) = sinh(A) · cosh(B) - cosh(A) · sinh(B)`

## contract_hyperbolic_sinh_sum (simplify)

- Source: `sinh(x)*cosh(y)+cosh(x)*sinh(y)`
- Target: `sinh(x+y)`
- Result: `sinh(x + y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sinh(x) * cosh(y) + sinh(y) * cosh(x)
Target: sinh(x + y)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize sinh(u)·cosh(v) ± cosh(u)·sinh(v) as sinh(u ± v)  [Aplicar identidad hiperbólica de suma/diferencia de ángulos]
   Before: sinh(x) * cosh(y) + sinh(y) * cosh(x)
   Cambio local: sinh(x) * cosh(y) + sinh(y) * cosh(x) -> sinh(x + y)
   After: sinh(x + y)
Result: sinh(x + y)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de suma/diferencia de ángulos`
   - before: `sinh(x) · cosh(y) + sinh(y) · cosh(x)`
   - after: `sinh(x + y)`
   - substeps:
     1. `Usar sinh(A+B) = sinh(A) · cosh(B) + cosh(A) · sinh(B)`

## contract_hyperbolic_tanh_difference (simplify)

- Source: `(tanh(x)-tanh(y))/(1-tanh(x)*tanh(y))`
- Target: `tanh(x-y)`
- Result: `tanh(x - y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (tanh(x) - tanh(y)) / (1 - tanh(x) * tanh(y))
Target: tanh(x - y)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize (tanh(u) ± tanh(v)) / (1 ± tanh(u)·tanh(v)) as tanh(u ± v)  [Aplicar identidad hiperbólica de suma/diferencia de ángulos]
   Before: (tanh(x) - tanh(y)) / (1 - tanh(x) * tanh(y))
   Cambio local: (tanh(x) - tanh(y)) / (1 - tanh(x) * tanh(y)) -> tanh(x - y)
   After: tanh(x - y)
Result: tanh(x - y)
ℹ️ Requires:
  • 1 - tanh(x) * tanh(y) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de suma/diferencia de ángulos`
   - before: `(tanh(x) - tanh(y))/(1 - tanh(x) · tanh(y))`
   - after: `tanh(x - y)`
   - substeps:
     1. `Usar tanh(A-B) = (tanh(A) - tanh(B)) / (1 - tanh(A)·tanh(B))`

## contract_hyperbolic_tanh_sum (simplify)

- Source: `(tanh(x)+tanh(y))/(1+tanh(x)*tanh(y))`
- Target: `tanh(x+y)`
- Result: `tanh(x + y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (tanh(x) + tanh(y)) / (tanh(x) * tanh(y) + 1)
Target: tanh(x + y)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize (tanh(u) ± tanh(v)) / (1 ± tanh(u)·tanh(v)) as tanh(u ± v)  [Aplicar identidad hiperbólica de suma/diferencia de ángulos]
   Before: (tanh(x) + tanh(y)) / (tanh(x) * tanh(y) + 1)
   Cambio local: (tanh(x) + tanh(y)) / (tanh(x) * tanh(y) + 1) -> tanh(x + y)
   After: tanh(x + y)
Result: tanh(x + y)
ℹ️ Requires:
  • tanh(x) * tanh(y) + 1 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de suma/diferencia de ángulos`
   - before: `(tanh(x) + tanh(y))/(tanh(x) · tanh(y) + 1)`
   - after: `tanh(x + y)`
   - substeps:
     1. `Usar tanh(A+B) = (tanh(A) + tanh(B)) / (1 + tanh(A)·tanh(B))`

## contract_log_change_of_base_chain (log_contract)

- Source: `log(b,a)*log(a,c)`
- Target: `log(b,c)`
- Result: `log(b, c)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: log(a, c) * log(b, a)
Target: log(b, c)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: log(a, c) * log(b, a)
   Cambio local: log(a, c) * log(b, a) -> log(b, c)
   After: log(b, c)
Result: log(b, c)
ℹ️ Requires:
  • a > 0
  • b > 0
  • c > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `log_a(c) · log_b(a)`
   - after: `log_b(c)`
   - substeps: none

## contract_log_change_of_base_direct (log_contract)

- Source: `ln(x)/ln(2)`
- Target: `log(2, x)`
- Result: `log(2, x)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ln(x) / ln(2)
Target: log(2, x)
Strategy: contract logs
Steps (Aggressive Mode):
1. Recognize a logarithm written by change of base  [Aplicar cambio de base]
   Before: ln(x) / ln(2)
   Subpasos:
     1.1 Leer el argumento desde el numerador
         ln(x) -> argumento x
     1.2 Leer la base desde el denominador
         ln(2) -> base 2
     1.3 Reconstruir el logaritmo de base indicada
         ln(x) / ln(2) -> log(2, x)
   Cambio local: ln(x) / ln(2) -> log(2, x)
   After: log(2, x)
Result: log(2, x)
ℹ️ Requires:
  • x > 0
```

### Web / JSON Steps

1. `Aplicar cambio de base`
   - before: `ln(x)/ln(2)`
   - after: `log_2(x)`
   - substeps:
     1. `Leer el argumento desde el numerador`
     2. `Leer la base desde el denominador`

## contract_log_difference (log_contract)

- Source: `ln(x) - ln(y)`
- Target: `ln(x/y)`
- Result: `ln(x / y)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(x) - ln(y)
Target: ln(x / y)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: ln(x) - ln(y)
   Cambio local: ln(x) - ln(y) -> ln(x / y)
   After: ln(x / y)
Result: ln(x / y)
ℹ️ Requires:
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `ln(x) - ln(y)`
   - after: `ln(x/y)`
   - substeps: none

## contract_log_difference_with_scaled_powers (log_contract)

- Source: `3*ln(x) - 2*ln(y)`
- Target: `ln(x^3/y^2)`
- Result: `ln(x^3 / y^2)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 3 * ln(x) - 2 * ln(y)
Target: ln(x^3 / y^2)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: 3 * ln(x) - 2 * ln(y)
   Cambio local: 3 * ln(x) - 2 * ln(y) -> ln(x^(3) / y^(2))
   After: ln(x^3 / y^2)
Result: ln(x^(3) / y^(2))
ℹ️ Requires:
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `3 · ln(x) - 2 · ln(y)`
   - after: `ln(x^3/y^2)`
   - substeps: none

## contract_log_even_power_abs (log_contract)

- Source: `2*ln(abs(x))`
- Target: `ln(x^2)`
- Result: `ln(x^2)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * ln(|x|)
Target: ln(x^2)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: 2 * ln(|x|)
   Cambio local: 2 * ln(|x|) -> ln(x^(2))
   After: ln(x^2)
Result: ln(x^(2))
ℹ️ Requires:
  • x ≠ 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `2 · ln(|x|)`
   - after: `ln(x^2)`
   - substeps: none

## contract_log_general_base_difference (log_contract)

- Source: `log(2, x) - log(2, y)`
- Target: `log(2, x/y)`
- Result: `log(2, x / y)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: log(2, x) - log(2, y)
Target: log(2, x / y)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: log(2, x) - log(2, y)
   Cambio local: log(2, x) - log(2, y) -> log(2, x / y)
   After: log(2, x / y)
Result: log(2, x / y)
ℹ️ Requires:
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `log_2(x) - log_2(y)`
   - after: `log_2(x/y)`
   - substeps: none

## contract_log_general_base_difference_with_scaled_powers (log_contract)

- Source: `3*log(2, x) - 2*log(2, y)`
- Target: `log(2, x^3/y^2)`
- Result: `log(2, x^3 / y^2)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 3 * log(2, x) - 2 * log(2, y)
Target: log(2, x^3 / y^2)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: 3 * log(2, x) - 2 * log(2, y)
   Cambio local: 3 * log(2, x) - 2 * log(2, y) -> log(2, x^(3) / y^(2))
   After: log(2, x^3 / y^2)
Result: log(2, x^(3) / y^(2))
ℹ️ Requires:
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `3 · log_2(x) - 2 · log_2(y)`
   - after: `log_2(x^3/y^2)`
   - substeps: none

## contract_log_general_base_power (log_contract)

- Source: `3*log(2, x)`
- Target: `log(2, x^3)`
- Result: `log(2, x^3)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 3 * log(2, x)
Target: log(2, x^3)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: 3 * log(2, x)
   Cambio local: 3 * log(2, x) -> log(2, x^(3))
   After: log(2, x^3)
Result: log(2, x^(3))
ℹ️ Requires:
  • x > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `3 · log_2(x)`
   - after: `log_2(x^3)`
   - substeps: none

## contract_log_general_base_powered_two_denominator_factors_with_powered_denominator (log_contract)

- Source: `2*log(b, x) + 3*log(b, y) - 2*log(b, z) - log(b, t)`
- Target: `log(b, (x^2*y^3)/(z^2*t))`
- Result: `log(b, x^2 * y^3 / (t * z^2))`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t)
Target: log(b, x^2 * y^3 / (t * z^2))
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: 2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t)
   Cambio local: 2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t) -> log(b, x^(2) * y^(3) / (t * z^(2)))
   After: log(b, x^2 * y^3 / (t * z^2))
Result: log(b, x^(2) * y^(3) / (t * z^(2)))
ℹ️ Requires:
  • b > 0
  • t > 0
  • x > 0
  • y > 0
  • z > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `2 · log_b(x) + 3 · log_b(y) - 2 · log_b(z) - log_b(t)`
   - after: `log_b((x^2 · y^3)/(t · z^2))`
   - substeps: none

## contract_log_grouped_power (log_contract)

- Source: `ln(x^2)+ln(y^2)`
- Target: `ln((x*y)^2)`
- Result: `ln((x * y)^2)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(x^2) + ln(y^2)
Target: ln((x * y)^2)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: ln(x^(2)) + ln(y^(2))
   Cambio local: ln(x^(2)) + ln(y^(2)) -> ln((x * y)^(2))
   After: ln((x * y)^2)
Result: ln((x * y)^(2))
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `ln(x^2) + ln(y^2)`
   - after: `ln((x · y)^2)`
   - substeps: none

## contract_log_grouped_power_with_passthrough (log_contract)

- Source: `ln(x^2)+ln(y^2)+a`
- Target: `ln((x*y)^2)+a`
- Result: `ln((x * y)^2) + a`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(x^2) + ln(y^2) + a
Target: ln((x * y)^2) + a
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: ln(x^(2)) + ln(y^(2)) + a
   Cambio local: ln(x^(2)) + ln(y^(2)) + a -> ln((x * y)^(2)) + a
   After: ln((x * y)^2) + a
Result: ln((x * y)^(2)) + a
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `ln(x^2) + ln(y^2) + a`
   - after: `ln((x · y)^2) + a`
   - substeps: none

## contract_log_powered_two_denominator_factors (log_contract)

- Source: `2*ln(abs(x)) + ln(y) - ln(z) - ln(t)`
- Target: `ln((x^2*y)/(z*t))`
- Result: `ln(y * x^2 / (t * z))`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(y) + 2 * ln(|x|) - ln(z) - ln(t)
Target: ln(y * x^2 / (t * z))
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: ln(y) + 2 * ln(|x|) - ln(z) - ln(t)
   Cambio local: ln(y) + 2 * ln(|x|) - ln(z) - ln(t) -> ln(y * x^(2) / (t * z))
   After: ln(y * x^2 / (t * z))
Result: ln(y * x^(2) / (t * z))
ℹ️ Requires:
  • t > 0
  • x ≠ 0
  • y > 0
  • z > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `ln(y) + 2 · ln(|x|) - ln(z) - ln(t)`
   - after: `ln((y · x^2)/(t · z))`
   - substeps: none

## contract_log_product_over_quotient (log_contract)

- Source: `ln(x) + ln(y) - ln(z)`
- Target: `ln((x*y)/z)`
- Result: `ln(x * y / z)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(x) + ln(y) - ln(z)
Target: ln(x * y / z)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: ln(x) + ln(y) - ln(z)
   Cambio local: ln(x) + ln(y) - ln(z) -> ln(x * y / z)
   After: ln(x * y / z)
Result: ln(x * y / z)
ℹ️ Requires:
  • x > 0
  • y > 0
  • z > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `ln(x) + ln(y) - ln(z)`
   - after: `ln((x · y)/z)`
   - substeps: none

## contract_log_sum (log_contract)

- Source: `ln(x) + ln(y)`
- Target: `ln(x*y)`
- Result: `ln(x * y)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(x) + ln(y)
Target: ln(x * y)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: ln(x) + ln(y)
   Cambio local: ln(x) + ln(y) -> ln(x * y)
   After: ln(x * y)
Result: ln(x * y)
ℹ️ Requires:
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `ln(x) + ln(y)`
   - after: `ln(x · y)`
   - substeps: none

## contract_log_sum_with_scaled_powers (log_contract)

- Source: `3*ln(x) + 2*ln(abs(y))`
- Target: `ln(x^3*y^2)`
- Result: `ln(x^3 * y^2)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * ln(|y|) + 3 * ln(x)
Target: ln(x^3 * y^2)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Contraer logaritmos]
   Before: 2 * ln(|y|) + 3 * ln(x)
   Cambio local: 2 * ln(|y|) + 3 * ln(x) -> ln(x^(3) * y^(2))
   After: ln(x^3 * y^2)
Result: ln(x^(3) * y^(2))
ℹ️ Requires:
  • x > 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `2 · ln(|y|) + 3 · ln(x)`
   - after: `ln(x^3 · y^2)`
   - substeps: none

## contract_trig_angle_diff_cosine (trig_contract)

- Source: `cos(x)*cos(y)+sin(x)*sin(y)`
- Target: `cos(x-y)`
- Result: `cos(x - y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x) * sin(y) + cos(x) * cos(y)
Target: cos(x - y)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand or contract an angle sum/difference trig identity  [Aplicar suma/diferencia de ángulos]
   Before: sin(x) * sin(y) + cos(x) * cos(y)
   Cambio local: sin(x) * sin(y) + cos(x) * cos(y) -> cos(x - y)
   After: cos(x - y)
Result: cos(x - y)
```

### Web / JSON Steps

1. `Aplicar suma/diferencia de ángulos`
   - before: `sin(x) · sin(y) + cos(x) · cos(y)`
   - after: `cos(x - y)`
   - substeps:
     1. `Usar cos(A-B) = cos(A) · cos(B) + sin(A) · sin(B)`

## contract_trig_angle_diff_sine (trig_contract)

- Source: `sin(x)*cos(y)-cos(x)*sin(y)`
- Target: `sin(x-y)`
- Result: `sin(x - y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x) * cos(y) - sin(y) * cos(x)
Target: sin(x - y)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand or contract an angle sum/difference trig identity  [Aplicar suma/diferencia de ángulos]
   Before: sin(x) * cos(y) - sin(y) * cos(x)
   Cambio local: sin(x) * cos(y) - sin(y) * cos(x) -> sin(x - y)
   After: sin(x - y)
Result: sin(x - y)
```

### Web / JSON Steps

1. `Aplicar suma/diferencia de ángulos`
   - before: `sin(x) · cos(y) - sin(y) · cos(x)`
   - after: `sin(x - y)`
   - substeps:
     1. `Usar sin(A-B) = sin(A) · cos(B) - cos(A) · sin(B)`

## contract_trig_angle_sum_cosine (trig_contract)

- Source: `cos(x)*cos(y)-sin(x)*sin(y)`
- Target: `cos(x+y)`
- Result: `cos(x + y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x) * cos(y) - sin(x) * sin(y)
Target: cos(x + y)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand or contract an angle sum/difference trig identity  [Aplicar suma/diferencia de ángulos]
   Before: cos(x) * cos(y) - sin(x) * sin(y)
   Cambio local: cos(x) * cos(y) - sin(x) * sin(y) -> cos(x + y)
   After: cos(x + y)
Result: cos(x + y)
```

### Web / JSON Steps

1. `Aplicar suma/diferencia de ángulos`
   - before: `cos(x) · cos(y) - sin(x) · sin(y)`
   - after: `cos(x + y)`
   - substeps:
     1. `Usar cos(A+B) = cos(A) · cos(B) - sin(A) · sin(B)`

## contract_trig_angle_sum_sine (trig_contract)

- Source: `sin(x)*cos(y)+cos(x)*sin(y)`
- Target: `sin(x+y)`
- Result: `sin(x + y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x) * cos(y) + sin(y) * cos(x)
Target: sin(x + y)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand or contract an angle sum/difference trig identity  [Aplicar suma/diferencia de ángulos]
   Before: sin(x) * cos(y) + sin(y) * cos(x)
   Cambio local: sin(x) * cos(y) + sin(y) * cos(x) -> sin(x + y)
   After: sin(x + y)
Result: sin(x + y)
```

### Web / JSON Steps

1. `Aplicar suma/diferencia de ángulos`
   - before: `sin(x) · cos(y) + sin(y) · cos(x)`
   - after: `sin(x + y)`
   - substeps:
     1. `Usar sin(A+B) = sin(A) · cos(B) + cos(A) · sin(B)`

## contract_trig_cos_diff_sin_diff_quotient (trig_contract)

- Source: `(cos(x)-cos(3*x))/(sin(3*x)-sin(x))`
- Target: `tan(2*x)`
- Result: `tan(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (cos(x) - cos(3 * x)) / (sin(3 * x) - sin(x))
Target: tan(2 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize a cosine/sine difference quotient as tan((A+B)/2)  [Convertir un cociente trigonométrico en tangente]
   Before: (cos(x) - cos(3 * x)) / (sin(3 * x) - sin(x))
   Cambio local: (cos(x) - cos(3 * x)) / (sin(3 * x) - sin(x)) -> tan(2 * x)
   After: tan(2 * x)
Result: tan(2 * x)
ℹ️ Requires:
  • sin(3 * x) - sin(x) ≠ 0
```

### Web / JSON Steps

1. `Convertir un cociente trigonométrico en tangente`
   - before: `(cos(x) - cos(3 · x))/(sin(3 · x) - sin(x))`
   - after: `tan(2 · x)`
   - substeps: none

## contract_trig_cot_quotient (trig_contract)

- Source: `cos(x)/sin(x)`
- Target: `cot(x)`
- Result: `cot(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cos(x) / sin(x)
Target: cot(x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize cos(u) / sin(u) as cot(u)  [Aplicar identidad trigonométrica recíproca]
   Before: cos(x) / sin(x)
   Cambio local: cos(x) / sin(x) -> cot(x)
   After: cot(x)
Result: cot(x)
ℹ️ Requires:
  • sin(x) ≠ 0
```

### Web / JSON Steps

1. `Reconocer cotangente desde un cociente`
   - before: `cos(x)/sin(x)`
   - after: `cot(x)`
   - substeps: none

## contract_trig_csc_reciprocal (trig_contract)

- Source: `1/sin(x)`
- Target: `csc(x)`
- Result: `csc(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 1 / sin(x)
Target: csc(x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize 1 / sin(u) as csc(u)  [Aplicar identidad trigonométrica recíproca]
   Before: 1 / sin(x)
   Cambio local: 1 / sin(x) -> csc(x)
   After: csc(x)
Result: csc(x)
ℹ️ Requires:
  • sin(x) ≠ 0
```

### Web / JSON Steps

1. `Reconocer cosecante desde un recíproco`
   - before: `1/sin(x)`
   - after: `csc(x)`
   - substeps: none

## contract_trig_csc_squared (trig_contract)

- Source: `1 + cot(x)^2`
- Target: `csc(x)^2`
- Result: `csc(x)^2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cot(x)^2 + 1
Target: csc(x)^2
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize 1 + cot²(u) as csc²(u)  [Reconocer cosecante cuadrada]
   Before: cot(x)^(2) + 1
   Cambio local: cot(x)^(2) + 1 -> csc(x)^(2)
   After: csc(x)^2
Result: csc(x)^(2)
```

### Web / JSON Steps

1. `Reconocer cosecante cuadrada`
   - before: `cot(x)^2 + 1`
   - after: `csc(x)^2`
   - substeps: none

## contract_trig_double_cos_from_one_minus_sin_sq (trig_contract)

- Source: `1 - 2*sin(x)^2`
- Target: `cos(2*x)`
- Result: `cos(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 1 - 2 * sin(x)^2
Target: cos(2 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand cosine double-angle as 1 - 2·sin(u)^2  [Expandir ángulo doble]
   Before: 1 - 2 * sin(x)^(2)
   Cambio local: 1 - 2 * sin(x)^(2) -> cos(2 * x)
   After: cos(2 * x)
Result: cos(2 * x)
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `1 - 2 · sin(x)^2`
   - after: `cos(2 · x)`
   - substeps: none

## contract_trig_double_cos_from_two_cos_sq_minus_one (trig_contract)

- Source: `2*cos(x)^2 - 1`
- Target: `cos(2*x)`
- Result: `cos(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * cos(x)^2 - 1
Target: cos(2 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand cosine double-angle as 2·cos(u)^2 - 1  [Expandir ángulo doble]
   Before: 2 * cos(x)^(2) - 1
   Cambio local: 2 * cos(x)^(2) - 1 -> cos(2 * x)
   After: cos(2 * x)
Result: cos(2 * x)
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `2 · cos(x)^2 - 1`
   - after: `cos(2 · x)`
   - substeps: none

## contract_trig_double_sin (trig_contract)

- Source: `2*sin(x)*cos(x)`
- Target: `sin(2*x)`
- Result: `sin(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * sin(x) * cos(x)
Target: sin(2 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand double-angle sine  [Expandir ángulo doble]
   Before: 2 * sin(x) * cos(x)
   Cambio local: 2 * sin(x) * cos(x) -> sin(2 * x)
   After: sin(2 * x)
Result: sin(2 * x)
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `2 · sin(x) · cos(x)`
   - after: `sin(2 · x)`
   - substeps: none

## contract_trig_double_tangent (trig_contract)

- Source: `2*tan(x)/(1-tan(x)^2)`
- Target: `tan(2*x)`
- Result: `tan(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * tan(x) / (1 - tan(x)^2)
Target: tan(2 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize tangent double-angle form  [Aplicar identidad de tangente de ángulo doble]
   Before: 2 * tan(x) / (1 - tan(x)^(2))
   Cambio local: 2 * tan(x) / (1 - tan(x)^(2)) -> tan(2 * x)
   After: tan(2 * x)
Result: tan(2 * x)
ℹ️ Requires:
  • 1 - tan(x) ≠ 0
  • tan(x) + 1 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de tangente de ángulo doble`
   - before: `(2 · tan(x))/(1 - tan(x)^2)`
   - after: `tan(2 · x)`
   - substeps: none

## contract_trig_half_angle_cos_squared (trig_contract)

- Source: `(1+cos(2*x))/2`
- Target: `cos(x)^2`
- Result: `cos(x)^2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (cos(2 * x) + 1) / 2
Target: cos(x)^2
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize (1 + cos(2u))/2 as cos²(u)  [Aplicar identidad de ángulo mitad]
   Before: (cos(2 * x) + 1) / 2
   Cambio local: (cos(2 * x) + 1) / 2 -> cos(x)^(2)
   After: cos(x)^2
Result: cos(x)^(2)
```

### Web / JSON Steps

1. `Aplicar identidad de ángulo mitad`
   - before: `(cos(2 · x) + 1)/2`
   - after: `cos(x)^2`
   - substeps: none

## contract_trig_half_angle_sin_squared (trig_contract)

- Source: `(1-cos(2*x))/2`
- Target: `sin(x)^2`
- Result: `sin(x)^2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (1 - cos(2 * x)) / 2
Target: sin(x)^2
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize (1 - cos(2u))/2 as sin²(u)  [Aplicar identidad de ángulo mitad]
   Before: (1 - cos(2 * x)) / 2
   Cambio local: (1 - cos(2 * x)) / 2 -> sin(x)^(2)
   After: sin(x)^2
Result: sin(x)^(2)
```

### Web / JSON Steps

1. `Aplicar identidad de ángulo mitad`
   - before: `(1 - cos(2 · x))/2`
   - after: `sin(x)^2`
   - substeps: none

## contract_trig_half_angle_tangent (trig_contract)

- Source: `(1-cos(2*x))/sin(2*x)`
- Target: `tan(x)`
- Result: `tan(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (1 - cos(2 * x)) / sin(2 * x)
Target: tan(x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Contract half-angle tangent quotient  [Aplicar identidad de tangente de ángulo mitad]
   Before: (1 - cos(2 * x)) / sin(2 * x)
   Cambio local: (1 - cos(2 * x)) / sin(2 * x) -> tan(x)
   After: tan(x)
Result: tan(x)
ℹ️ Requires:
  • sin(2 * x) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de tangente de ángulo mitad`
   - before: `(1 - cos(2 · x))/sin(2 · x)`
   - after: `tan(x)`
   - substeps: none

## contract_trig_half_angle_tangent_alt (trig_contract)

- Source: `sin(2*x)/(1+cos(2*x))`
- Target: `tan(x)`
- Result: `tan(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(2 * x) / (cos(2 * x) + 1)
Target: tan(x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Contract half-angle tangent quotient  [Aplicar identidad de tangente de ángulo mitad]
   Before: sin(2 * x) / (cos(2 * x) + 1)
   Cambio local: sin(2 * x) / (cos(2 * x) + 1) -> tan(x)
   After: tan(x)
Result: tan(x)
ℹ️ Requires:
  • cos(2 * x) + 1 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de tangente de ángulo mitad`
   - before: `sin(2 · x)/(cos(2 · x) + 1)`
   - after: `tan(x)`
   - substeps: none

## contract_trig_half_scaled_double_sin (trig_contract)

- Source: `sin(x)*cos(x)`
- Target: `sin(2*x)/2`
- Result: `sin(2 * x) / 2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(x) * cos(x)
Target: sin(2 * x) / 2
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand double-angle sine  [Expandir ángulo doble]
   Before: sin(x) * cos(x)
   Cambio local: sin(x) * cos(x) -> sin(2 * x) / 2
   After: sin(2 * x) / 2
Result: sin(2 * x) / 2
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `sin(x) · cos(x)`
   - after: `sin(2 · x)/2`
   - substeps: none

## contract_trig_negative_double_cos_from_square_diff (trig_contract)

- Source: `sin(x)^2 - cos(x)^2`
- Target: `-cos(2*x)`
- Result: `-cos(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(x)^2 - cos(x)^2
Target: -cos(2 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize a negated cosine double-angle form  [Expandir ángulo doble]
   Before: sin(x)^(2) - cos(x)^(2)
   Cambio local: sin(x)^(2) - cos(x)^(2) -> -cos(2 * x)
   After: -cos(2 * x)
Result: -cos(2 * x)
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `sin(x)^2 - cos(x)^2`
   - after: `-cos(2 · x)`
   - substeps: none

## contract_trig_negative_double_sin (trig_contract)

- Source: `-2*sin(x)*cos(x)`
- Target: `-sin(2*x)`
- Result: `-sin(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: -2 * sin(x) * cos(x)
Target: -sin(2 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize a negated sine double-angle form  [Expandir ángulo doble]
   Before: -2 * sin(x) * cos(x)
   Cambio local: -2 * sin(x) * cos(x) -> -sin(2 * x)
   After: -sin(2 * x)
Result: -sin(2 * x)
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `-2 · sin(x) · cos(x)`
   - after: `-sin(2 · x)`
   - substeps: none

## contract_trig_phase_shift_difference_to_shifted_sine (trig_contract)

- Source: `sin(x)-cos(x)`
- Target: `sqrt(2)*sin(x-pi/4)`
- Result: `sin(x - pi / 4) * sqrt(2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x) - cos(x)
Target: sin(x - pi / 4) * sqrt(2)
Strategy: contract trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: sin(x) - cos(x)
   Cambio local: sin(x) - cos(x) -> sin(x - pi / 4) * sqrt(2)
   After: sin(x - pi / 4) * sqrt(2)
Result: sin(x - pi / 4) * sqrt(2)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `sin(x) - cos(x)`
   - after: `sin(x - pi/4) · sqrt(2)`
   - substeps:
     1. `Usar a·sin(u) + b·cos(u) = R·sin(u + φ)`

## contract_trig_phase_shift_exact_sixth_sum_to_shifted_sine (trig_contract)

- Source: `sqrt(3)*sin(x)+cos(x)`
- Target: `2*sin(x+pi/6)`
- Result: `2 * sin(pi / 6 + x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x) + sin(x) * sqrt(3)
Target: 2 * sin(pi / 6 + x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: cos(x) + sin(x) * sqrt(3)
   Cambio local: cos(x) + sin(x) * sqrt(3) -> 2 * sin(pi / 6 + x)
   After: 2 * sin(pi / 6 + x)
Result: 2 * sin(pi / 6 + x)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `cos(x) + sin(x) · sqrt(3)`
   - after: `2 · sin(pi/6 + x)`
   - substeps:
     1. `Usar a·sin(u) + b·cos(u) = R·sin(u + φ)`

## contract_trig_phase_shift_exact_third_scaled_sum_to_shifted_sine (trig_contract)

- Source: `2*sin(x)+2*sqrt(3)*cos(x)`
- Target: `4*sin(x+pi/3)`
- Result: `4 * sin(pi / 3 + x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * sin(x) + 2 * cos(x) * sqrt(3)
Target: 4 * sin(pi / 3 + x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: 2 * sin(x) + 2 * cos(x) * sqrt(3)
   Cambio local: 2 * sin(x) + 2 * cos(x) * sqrt(3) -> 4 * sin(pi / 3 + x)
   After: 4 * sin(pi / 3 + x)
Result: 4 * sin(pi / 3 + x)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `2 · sin(x) + 2 · cos(x) · sqrt(3)`
   - after: `4 · sin(pi/3 + x)`
   - substeps:
     1. `Usar a·sin(u) + b·cos(u) = R·sin(u + φ)`

## contract_trig_phase_shift_general_shifted_sine_to_shifted_cosine (trig_contract)

- Source: `5*sin(x+arctan(4/3))`
- Target: `5*cos(x-arctan(3/4))`
- Result: `5 * cos(x - arctan(3 / 4))`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 5 * sin(arctan(4 / 3) + x)
Target: 5 * cos(x - arctan(3 / 4))
Strategy: contract trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: 5 * sin(arctan(4 / 3) + x)
   Cambio local: 5 * sin(arctan(4 / 3) + x) -> 5 * cos(x - arctan(3 / 4))
   After: 5 * cos(x - arctan(3 / 4))
Result: 5 * cos(x - arctan(3 / 4))
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `5 · sin(arctan(4/3) + x)`
   - after: `5 · cos(x - arctan(3/4))`
   - substeps:
     1. `Usar sin(u + φ) = cos(u - (π/2 - φ))`

## contract_trig_phase_shift_general_shifted_terms_with_passthrough (trig_contract)

- Source: `5*sin(x+arctan(4/3))+a`
- Target: `5*cos(x-arctan(3/4))+a`
- Result: `5 * cos(x - arctan(3 / 4)) + a`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 5 * sin(arctan(4 / 3) + x) + a
Target: 5 * cos(x - arctan(3 / 4)) + a
Strategy: contract trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: 5 * sin(arctan(4 / 3) + x) + a
   Cambio local: 5 * sin(arctan(4 / 3) + x) + a -> 5 * cos(x - arctan(3 / 4)) + a
   After: 5 * cos(x - arctan(3 / 4)) + a
Result: 5 * cos(x - arctan(3 / 4)) + a
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `5 · sin(arctan(4/3) + x) + a`
   - after: `5 · cos(x - arctan(3/4)) + a`
   - substeps:
     1. `Aplicar la identidad de desfase al bloque que cambia`

## contract_trig_phase_shift_general_sum_to_shifted_sine (trig_contract)

- Source: `3*sin(x)+4*cos(x)`
- Target: `5*sin(x+arctan(4/3))`
- Result: `5 * sin(arctan(4 / 3) + x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 3 * sin(x) + 4 * cos(x)
Target: 5 * sin(arctan(4 / 3) + x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: 3 * sin(x) + 4 * cos(x)
   Cambio local: 3 * sin(x) + 4 * cos(x) -> 5 * sin(arctan(4 / 3) + x)
   After: 5 * sin(arctan(4 / 3) + x)
Result: 5 * sin(arctan(4 / 3) + x)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `3 · sin(x) + 4 · cos(x)`
   - after: `5 · sin(arctan(4/3) + x)`
   - substeps:
     1. `Usar a·sin(u) + b·cos(u) = R·sin(u + φ)`

## contract_trig_phase_shift_general_sum_to_shifted_sine_with_passthrough (trig_contract)

- Source: `3*sin(x)+4*cos(x)+a`
- Target: `5*sin(x+arctan(4/3))+a`
- Result: `5 * sin(arctan(4 / 3) + x) + a`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 3 * sin(x) + 4 * cos(x) + a
Target: 5 * sin(arctan(4 / 3) + x) + a
Strategy: contract trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: 3 * sin(x) + 4 * cos(x) + a
   Cambio local: 3 * sin(x) + 4 * cos(x) + a -> 5 * sin(arctan(4 / 3) + x) + a
   After: 5 * sin(arctan(4 / 3) + x) + a
Result: 5 * sin(arctan(4 / 3) + x) + a
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `3 · sin(x) + 4 · cos(x) + a`
   - after: `5 · sin(arctan(4/3) + x) + a`
   - substeps:
     1. `Aplicar la identidad de desfase al bloque que cambia`

## contract_trig_phase_shift_scaled_sum_to_shifted_sine (trig_contract)

- Source: `2*sin(x)+2*cos(x)`
- Target: `2*sqrt(2)*sin(x+pi/4)`
- Result: `2 * sin(pi / 4 + x) * sqrt(2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * sin(x) + 2 * cos(x)
Target: 2 * sin(pi / 4 + x) * sqrt(2)
Strategy: contract trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: 2 * sin(x) + 2 * cos(x)
   Cambio local: 2 * sin(x) + 2 * cos(x) -> 2 * sin(pi / 4 + x) * sqrt(2)
   After: 2 * sin(pi / 4 + x) * sqrt(2)
Result: 2 * sin(pi / 4 + x) * sqrt(2)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `2 · sin(x) + 2 · cos(x)`
   - after: `2 · sin(pi/4 + x) · sqrt(2)`
   - substeps:
     1. `Usar a·sin(u) + b·cos(u) = R·sin(u + φ)`

## contract_trig_phase_shift_shifted_sine_to_shifted_cosine (trig_contract)

- Source: `sqrt(2)*sin(x+pi/4)`
- Target: `sqrt(2)*cos(x-pi/4)`
- Result: `cos(x - pi / 4) * sqrt(2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(pi / 4 + x) * sqrt(2)
Target: cos(x - pi / 4) * sqrt(2)
Strategy: contract trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: sin(pi / 4 + x) * sqrt(2)
   Cambio local: sin(pi / 4 + x) * sqrt(2) -> cos(x - pi / 4) * sqrt(2)
   After: cos(x - pi / 4) * sqrt(2)
Result: cos(x - pi / 4) * sqrt(2)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `sin(pi/4 + x) · sqrt(2)`
   - after: `cos(x - pi/4) · sqrt(2)`
   - substeps:
     1. `Usar sin(u + φ) = cos(u - (π/2 - φ))`

## contract_trig_phase_shift_shifted_terms_with_passthrough (trig_contract)

- Source: `sqrt(2)*sin(x+pi/4)+a`
- Target: `sqrt(2)*cos(x-pi/4)+a`
- Result: `cos(x - pi / 4) * sqrt(2) + a`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(pi / 4 + x) * sqrt(2) + a
Target: cos(x - pi / 4) * sqrt(2) + a
Strategy: contract trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: sin(pi / 4 + x) * sqrt(2) + a
   Cambio local: sin(pi / 4 + x) * sqrt(2) + a -> cos(x - pi / 4) * sqrt(2) + a
   After: cos(x - pi / 4) * sqrt(2) + a
Result: cos(x - pi / 4) * sqrt(2) + a
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `sin(pi/4 + x) · sqrt(2) + a`
   - after: `cos(x - pi/4) · sqrt(2) + a`
   - substeps:
     1. `Aplicar la identidad de desfase al bloque que cambia`

## contract_trig_phase_shift_sum_to_shifted_sine (trig_contract)

- Source: `sin(x)+cos(x)`
- Target: `sqrt(2)*sin(x+pi/4)`
- Result: `sin(pi / 4 + x) * sqrt(2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x) + cos(x)
Target: sin(pi / 4 + x) * sqrt(2)
Strategy: contract trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: sin(x) + cos(x)
   Cambio local: sin(x) + cos(x) -> sin(pi / 4 + x) * sqrt(2)
   After: sin(pi / 4 + x) * sqrt(2)
Result: sin(pi / 4 + x) * sqrt(2)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `sin(x) + cos(x)`
   - after: `sin(pi/4 + x) · sqrt(2)`
   - substeps:
     1. `Usar a·sin(u) + b·cos(u) = R·sin(u + φ)`

## contract_trig_phase_shift_sum_to_shifted_sine_with_passthrough (trig_contract)

- Source: `sin(x)+cos(x)+a`
- Target: `sqrt(2)*sin(x+pi/4)+a`
- Result: `sin(pi / 4 + x) * sqrt(2) + a`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x) + cos(x) + a
Target: sin(pi / 4 + x) * sqrt(2) + a
Strategy: contract trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: sin(x) + cos(x) + a
   Cambio local: sin(x) + cos(x) + a -> sin(pi / 4 + x) * sqrt(2) + a
   After: sin(pi / 4 + x) * sqrt(2) + a
Result: sin(pi / 4 + x) * sqrt(2) + a
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `sin(x) + cos(x) + a`
   - after: `sin(pi/4 + x) · sqrt(2) + a`
   - substeps:
     1. `Aplicar la identidad de desfase al bloque que cambia`

## contract_trig_quadruple_angle_sine_expanded_product (trig_contract)

- Source: `4*sin(x)*cos(x)^3-4*sin(x)^3*cos(x)`
- Target: `sin(4*x)`
- Result: `sin(4 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 4 * sin(x) * cos(x)^3 - 4 * cos(x) * sin(x)^3
Target: sin(4 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand or contract sine quadruple-angle form  [Reescribir ángulo cuádruple]
   Before: 4 * sin(x) * cos(x)^(3) - 4 * cos(x) * sin(x)^(3)
   Cambio local: 4 * sin(x) * cos(x)^(3) - 4 * cos(x) * sin(x)^(3) -> sin(4 * x)
   After: sin(4 * x)
Result: sin(4 * x)
```

### Web / JSON Steps

1. `Reescribir ángulo cuádruple`
   - before: `4 · sin(x) · cos(x)^3 - 4 · cos(x) · sin(x)^3`
   - after: `sin(4 · x)`
   - substeps:
     1. `Usar sin(4u) = 4 · sin(u) · cos(u)^3 - 4 · sin(u)^3 · cos(u), con u = x`

## contract_trig_quintuple_angle_cosine (trig_contract)

- Source: `16*cos(x)^5-20*cos(x)^3+5*cos(x)`
- Target: `cos(5*x)`
- Result: `cos(5 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 5 * cos(x) + 16 * cos(x)^5 - 20 * cos(x)^3
Target: cos(5 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand or contract cosine quintuple-angle form  [Reescribir ángulo quíntuple]
   Before: 5 * cos(x) + 16 * cos(x)^(5) - 20 * cos(x)^(3)
   Cambio local: 5 * cos(x) + 16 * cos(x)^(5) - 20 * cos(x)^(3) -> cos(5 * x)
   After: cos(5 * x)
Result: cos(5 * x)
```

### Web / JSON Steps

1. `Reescribir ángulo quíntuple`
   - before: `5 · cos(x) + 16 · cos(x)^5 - 20 · cos(x)^3`
   - after: `cos(5 · x)`
   - substeps:
     1. `Usar cos(5u) = 16 · cos(u)^5 - 20 · cos(u)^3 + 5 · cos(u), con u = x`

## contract_trig_quintuple_angle_sine (trig_contract)

- Source: `5*sin(x)-20*sin(x)^3+16*sin(x)^5`
- Target: `sin(5*x)`
- Result: `sin(5 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 5 * sin(x) + 16 * sin(x)^5 - 20 * sin(x)^3
Target: sin(5 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand or contract sine quintuple-angle form  [Reescribir ángulo quíntuple]
   Before: 5 * sin(x) + 16 * sin(x)^(5) - 20 * sin(x)^(3)
   Cambio local: 5 * sin(x) + 16 * sin(x)^(5) - 20 * sin(x)^(3) -> sin(5 * x)
   After: sin(5 * x)
Result: sin(5 * x)
```

### Web / JSON Steps

1. `Reescribir ángulo quíntuple`
   - before: `16 · sin(x)^5 + 5 · sin(x) - 20 · sin(x)^3`
   - after: `sin(5 · x)`
   - substeps:
     1. `Usar sin(5u) = 5 · sin(u) - 20 · sin(u)^3 + 16 · sin(u)^5, con u = x`

## contract_trig_recursive_six_cosine (trig_contract)

- Source: `cos(5*x)*cos(x)-sin(5*x)*sin(x)`
- Target: `cos(6*x)`
- Result: `cos(6 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x) * cos(5 * x) - sin(x) * sin(5 * x)
Target: cos(6 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand or contract an angle sum/difference trig identity  [Aplicar suma/diferencia de ángulos]
   Before: cos(x) * cos(5 * x) - sin(x) * sin(5 * x)
   Cambio local: cos(x) * cos(5 * x) - sin(x) * sin(5 * x) -> cos(6 * x)
   After: cos(6 * x)
Result: cos(6 * x)
```

### Web / JSON Steps

1. `Aplicar suma/diferencia de ángulos`
   - before: `cos(x) · cos(5 · x) - sin(x) · sin(5 · x)`
   - after: `cos(6 · x)`
   - substeps:
     1. `Usar cos(5u+u) = cos(5u) · cos(u) - sin(5u) · sin(u), con u = x`

## contract_trig_recursive_six_sine (trig_contract)

- Source: `sin(5*x)*cos(x)+cos(5*x)*sin(x)`
- Target: `sin(6*x)`
- Result: `sin(6 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x) * cos(5 * x) + sin(5 * x) * cos(x)
Target: sin(6 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand or contract an angle sum/difference trig identity  [Aplicar suma/diferencia de ángulos]
   Before: sin(x) * cos(5 * x) + sin(5 * x) * cos(x)
   Cambio local: sin(x) * cos(5 * x) + sin(5 * x) * cos(x) -> sin(6 * x)
   After: sin(6 * x)
Result: sin(6 * x)
```

### Web / JSON Steps

1. `Aplicar suma/diferencia de ángulos`
   - before: `sin(x) · cos(5 · x) + sin(5 · x) · cos(x)`
   - after: `sin(6 · x)`
   - substeps:
     1. `Usar sin(5u+u) = sin(5u) · cos(u) + cos(5u) · sin(u), con u = x`

## contract_trig_sec_reciprocal (trig_contract)

- Source: `1/cos(x)`
- Target: `sec(x)`
- Result: `sec(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 1 / cos(x)
Target: sec(x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize 1 / cos(u) as sec(u)  [Aplicar identidad trigonométrica recíproca]
   Before: 1 / cos(x)
   Cambio local: 1 / cos(x) -> sec(x)
   After: sec(x)
Result: sec(x)
ℹ️ Requires:
  • cos(x) ≠ 0
```

### Web / JSON Steps

1. `Reconocer secante desde un recíproco`
   - before: `1/cos(x)`
   - after: `sec(x)`
   - substeps: none

## contract_trig_sec_squared (trig_contract)

- Source: `1 + tan(x)^2`
- Target: `sec(x)^2`
- Result: `sec(x)^2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: tan(x)^2 + 1
Target: sec(x)^2
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize 1 + tan²(u) as sec²(u)  [Reconocer secante cuadrada]
   Before: tan(x)^(2) + 1
   Cambio local: tan(x)^(2) + 1 -> sec(x)^(2)
   After: sec(x)^2
Result: sec(x)^(2)
```

### Web / JSON Steps

1. `Reconocer secante cuadrada`
   - before: `tan(x)^2 + 1`
   - after: `sec(x)^2`
   - substeps: none

## contract_trig_sin_diff_special (trig_contract)

- Source: `sin(3*x)-sin(x)`
- Target: `2*cos(2*x)*sin(x)`
- Result: `2 * sin(x) * cos(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(3 * x) - sin(x)
Target: 2 * sin(x) * cos(2 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand sine difference to product  [Aplicar suma a producto]
   Before: sin(3 * x) - sin(x)
   Cambio local: sin(3 * x) - sin(x) -> 2 * sin(x) * cos(2 * x)
   After: 2 * sin(x) * cos(2 * x)
Result: 2 * sin(x) * cos(2 * x)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `sin(3 · x) - sin(x)`
   - after: `2 · sin(x) · cos(2 · x)`
   - substeps: none

## contract_trig_square_double_angle_sine_cosine_product (trig_contract)

- Source: `sin(x)^2*cos(x)^2`
- Target: `sin(2*x)^2/4`
- Result: `sin(2 * x)^2 / 4`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x)^2 * cos(x)^2
Target: sin(2 * x)^2 / 4
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize sin²(u)·cos²(u) as sin²(2u)/4  [Contraer cuadrado de ángulo doble]
   Before: sin(x)^(2) * cos(x)^(2)
   Cambio local: sin(x)^(2) * cos(x)^(2) -> sin(2 * x)^(2) / 4
   After: sin(2 * x)^2 / 4
Result: sin(2 * x)^(2) / 4
```

### Web / JSON Steps

1. `Contraer cuadrado de ángulo doble`
   - before: `sin(x)^2 · cos(x)^2`
   - after: `(sin(2 · x))^2/4`
   - substeps:
     1. `Usar sin²(u)·cos²(u) = sin²(2u) / 4, con u = x`

## contract_trig_tan_quotient (trig_contract)

- Source: `(sin(2*x))/(cos(2*x))`
- Target: `tan(2*x)`
- Result: `tan(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(2 * x) / cos(2 * x)
Target: tan(2 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize sin(u) / cos(u) as tan(u)  [Convertir un cociente trigonométrico en tangente]
   Before: sin(2 * x) / cos(2 * x)
   Cambio local: sin(2 * x) / cos(2 * x) -> tan(2 * x)
   After: tan(2 * x)
Result: tan(2 * x)
ℹ️ Requires:
  • cos(2 * x) ≠ 0
```

### Web / JSON Steps

1. `Convertir un cociente trigonométrico en tangente`
   - before: `sin(2 · x)/cos(2 · x)`
   - after: `tan(2 · x)`
   - substeps: none

## contract_trig_tan_quotient_after_arg_simplify (trig_contract)

- Source: `sin(2*x)/cos(x+x)`
- Target: `tan(2*x)`
- Result: `tan(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(2 * x) / cos(x + x)
Target: tan(2 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize sin(u) / cos(u) as tan(u)  [Convertir un cociente trigonométrico en tangente]
   Before: sin(2 * x) / cos(x + x)
   Cambio local: sin(2 * x) / cos(x + x) -> tan(2 * x)
   After: tan(2 * x)
Result: tan(2 * x)
ℹ️ Requires:
  • cos(x + x) ≠ 0
```

### Web / JSON Steps

1. `Convertir un cociente trigonométrico en tangente`
   - before: `sin(2 · x)/cos(x + x)`
   - after: `tan(2 · x)`
   - substeps: none

## contract_trig_tan_quotient_with_additive_passthrough (trig_contract)

- Source: `1+x*sin(x^2)/cos(x^2)`
- Target: `1+x*tan(x^2)`
- Result: `x * tan(x^2) + 1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: x * sin(x^2) / cos(x^2) + 1
Target: x * tan(x^2) + 1
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize sin(u) / cos(u) as tan(u)  [Convertir un cociente trigonométrico en tangente]
   Before: x * sin(x^(2)) / cos(x^(2)) + 1
   Cambio local: x * sin(x^(2)) / cos(x^(2)) + 1 -> x * tan(x^(2)) + 1
   After: x * tan(x^2) + 1
Result: x * tan(x^(2)) + 1
ℹ️ Requires:
  • cos(x^2) ≠ 0
```

### Web / JSON Steps

1. `Convertir un cociente trigonométrico en tangente`
   - before: `(x · sin(x^2))/cos(x^2) + 1`
   - after: `x · tan(x^2) + 1`
   - substeps: none

## contract_trig_tan_quotient_with_cofactor (trig_contract)

- Source: `x*sin(x^2)/cos(x^2)`
- Target: `x*tan(x^2)`
- Result: `x * tan(x^2)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: x * sin(x^2) / cos(x^2)
Target: x * tan(x^2)
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize sin(u) / cos(u) as tan(u)  [Convertir un cociente trigonométrico en tangente]
   Before: x * sin(x^(2)) / cos(x^(2))
   Cambio local: x * sin(x^(2)) / cos(x^(2)) -> x * tan(x^(2))
   After: x * tan(x^2)
Result: x * tan(x^(2))
ℹ️ Requires:
  • cos(x^2) ≠ 0
```

### Web / JSON Steps

1. `Convertir un cociente trigonométrico en tangente`
   - before: `(x · sin(x^2))/cos(x^2)`
   - after: `x · tan(x^2)`
   - substeps: none

## contract_trig_tangent_angle_difference (trig_contract)

- Source: `(tan(x)-tan(y))/(1+tan(x)*tan(y))`
- Target: `tan(x-y)`
- Result: `tan(x - y)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (tan(x) - tan(y)) / (tan(x) * tan(y) + 1)
Target: tan(x - y)
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize tangent angle sum/difference form  [Aplicar identidad de tangente de suma/diferencia de ángulos]
   Before: (tan(x) - tan(y)) / (tan(x) * tan(y) + 1)
   Cambio local: (tan(x) - tan(y)) / (tan(x) * tan(y) + 1) -> tan(x - y)
   After: tan(x - y)
Result: tan(x - y)
ℹ️ Requires:
  • tan(x) * tan(y) + 1 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de tangente de suma/diferencia de ángulos`
   - before: `(tan(x) - tan(y))/(tan(x) · tan(y) + 1)`
   - after: `tan(x - y)`
   - substeps: none

## contract_trig_tangent_angle_sum (trig_contract)

- Source: `(tan(x)+tan(y))/(1-tan(x)*tan(y))`
- Target: `tan(x+y)`
- Result: `tan(x + y)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (tan(x) + tan(y)) / (1 - tan(x) * tan(y))
Target: tan(x + y)
Strategy: contract trig
Steps (Aggressive Mode):
1. Recognize tangent angle sum/difference form  [Aplicar identidad de tangente de suma/diferencia de ángulos]
   Before: (tan(x) + tan(y)) / (1 - tan(x) * tan(y))
   Cambio local: (tan(x) + tan(y)) / (1 - tan(x) * tan(y)) -> tan(x + y)
   After: tan(x + y)
Result: tan(x + y)
ℹ️ Requires:
  • 1 - tan(x) * tan(y) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de tangente de suma/diferencia de ángulos`
   - before: `(tan(x) + tan(y))/(1 - tan(x) · tan(y))`
   - after: `tan(x + y)`
   - substeps: none

## contract_trig_triple_angle_cosine (trig_contract)

- Source: `4*cos(x)^3-3*cos(x)`
- Target: `cos(3*x)`
- Result: `cos(3 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 4 * cos(x)^3 - 3 * cos(x)
Target: cos(3 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand or contract cosine triple-angle form  [Reescribir ángulo triple]
   Before: 4 * cos(x)^(3) - 3 * cos(x)
   Cambio local: 4 * cos(x)^(3) - 3 * cos(x) -> cos(3 * x)
   After: cos(3 * x)
Result: cos(3 * x)
```

### Web / JSON Steps

1. `Reescribir ángulo triple`
   - before: `4 · cos(x)^3 - 3 · cos(x)`
   - after: `cos(3 · x)`
   - substeps:
     1. `Usar cos(3u) = 4 · cos(u)^3 - 3 · cos(u), con u = x`

## contract_trig_triple_angle_sine (trig_contract)

- Source: `3*sin(x)-4*sin(x)^3`
- Target: `sin(3*x)`
- Result: `sin(3 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 3 * sin(x) - 4 * sin(x)^3
Target: sin(3 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand or contract sine triple-angle form  [Reescribir ángulo triple]
   Before: 3 * sin(x) - 4 * sin(x)^(3)
   Cambio local: 3 * sin(x) - 4 * sin(x)^(3) -> sin(3 * x)
   After: sin(3 * x)
Result: sin(3 * x)
```

### Web / JSON Steps

1. `Reescribir ángulo triple`
   - before: `3 · sin(x) - 4 · sin(x)^3`
   - after: `sin(3 · x)`
   - substeps:
     1. `Usar sin(3u) = 3 · sin(u) - 4 · sin(u)^3, con u = x`

## contract_trig_triple_angle_tangent (trig_contract)

- Source: `(3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)`
- Target: `tan(3*x)`
- Result: `tan(3 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (3 * tan(x) - tan(x)^3) / (1 - 3 * tan(x)^2)
Target: tan(3 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. Expand or contract tangent triple-angle form  [Reescribir ángulo triple]
   Before: (3 * tan(x) - tan(x)^(3)) / (1 - 3 * tan(x)^(2))
   Cambio local: (3 * tan(x) - tan(x)^(3)) / (1 - 3 * tan(x)^(2)) -> tan(3 * x)
   After: tan(3 * x)
Result: tan(3 * x)
ℹ️ Requires:
  • 1 - 3 * tan(x)^2 ≠ 0
```

### Web / JSON Steps

1. `Reescribir ángulo triple`
   - before: `(3 · tan(x) - tan(x)^3)/(1 - 3 · tan(x)^2)`
   - after: `tan(3 · x)`
   - substeps:
     1. `Usar tan(3u) = (3 · tan(u) - tan(u)^3) / (1 - 3 · tan(u)^2), con u = x`

## cos_arcsin_complement_projection (simplify)

- Source: `cos(arcsin(x))`
- Target: `sqrt(1-x^2)`
- Result: `sqrt(1 - x^2)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: cos(arcsin(x))
Target: sqrt(1 - x^2)
Strategy: rewrite inverse trigs
Steps (Aggressive Mode):
1. cos(arcsin(x)) = sqrt(1-x^2)  [Aplicar composición trigonométrica inversa]
   Before: cos(arcsin(x))
   Cambio local: cos(arcsin(x)) -> sqrt(1 - x^(2))
   After: sqrt(1 - x^2)
Result: sqrt(1 - x^(2))
ℹ️ Requires:
  • -1 ≤ x ≤ 1
```

### Web / JSON Steps

1. `Aplicar composición trigonométrica inversa`
   - before: `cos(arcsin(x))`
   - after: `sqrt(1 - x^2)`
   - substeps:
     1. `Calcular el cateto restante del triángulo asociado a arcsin(x)`
     2. `Leer el coseno desde ese triángulo`

## cos_arctan_right_triangle_projection (simplify)

- Source: `cos(arctan(x))`
- Target: `1/sqrt(1+x^2)`
- Result: `1 / sqrt(x^2 + 1)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: cos(arctan(x))
Target: 1 / sqrt(x^2 + 1)
Strategy: rewrite inverse trigs
Steps (Aggressive Mode):
1. cos(arctan(x)) = 1/sqrt(1+x^2)  [Aplicar composición trigonométrica inversa]
   Before: cos(arctan(x))
   Cambio local: cos(arctan(x)) -> 1 / sqrt(x^(2) + 1)
   After: 1 / sqrt(x^2 + 1)
Result: 1 / sqrt(x^(2) + 1)
```

### Web / JSON Steps

1. `Aplicar composición trigonométrica inversa`
   - before: `cos(arctan(x))`
   - after: `1/sqrt(x^2 + 1)`
   - substeps:
     1. `Calcular la hipotenusa del triángulo asociado a arctan(x)`
     2. `Leer el coseno desde ese triángulo`

## csc_cot_pythagorean_to_one (simplify)

- Source: `csc(x)^2 - cot(x)^2`
- Target: `1`
- Result: `1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: csc(x)^2 - cot(x)^2
Target: 1
Strategy: rewrite trigs
Steps (Aggressive Mode):
1. Recognize csc²(u) - cot²(u) = 1  [Aplicar identidad pitagórica recíproca]
   Before: csc(x)^(2) - cot(x)^(2)
   Cambio local: csc(x)^(2) - cot(x)^(2) -> 1
   After: 1
Result: 1
```

### Web / JSON Steps

1. `Aplicar identidad pitagórica recíproca`
   - before: `csc(x)^2 - cot(x)^2`
   - after: `1`
   - substeps: none

## expand_binomial (expand)

- Source: `(x + 1)^2`
- Target: `x^2 + 2*x + 1`
- Result: `x^2 + 2 * x + 1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (x + 1)^2
Target: x^2 + 2 * x + 1
Strategy: expand
Steps (Aggressive Mode):
1. Expand binomial power ^2  [Expandir binomio]
   Before: (x + 1)^(2)
   Cambio local: (x + 1)^(2) -> 1^(2) + x^(2) + 2 * x
   After: x^2 + 2 * x + 1
Result: x^(2) + 2 * x + 1
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `(x + 1)^2`
   - after: `x^2 + 2 · x + 1`
   - substeps: none

## expand_common_factor_difference (expand)

- Source: `a*(b-c)`
- Target: `a*b - a*c`
- Result: `a * b - a * c`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * (b - c)
Target: a * b - a * c
Strategy: expand
Steps (Aggressive Mode):
1. Distribute  [Expandir la expresión]
   Before: a * (b - c)
   Cambio local: a * (b - c) -> a * b - a * c
   After: a * b - a * c
Result: a * b - a * c
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `a · (b - c)`
   - after: `a · b - a · c`
   - substeps:
     1. `Identificar los productos que genera la distributiva`
     2. `Escribir los productos con los signos originales`

## expand_common_factor_difference_three_terms (expand)

- Source: `x*(a-b-c)`
- Target: `a*x - b*x - c*x`
- Result: `a * x - b * x - c * x`
- Web step count: `2`
- Web substep count: `4`
- Flags: none

### CLI

```text
Parsed: x * (a - b - c)
Target: a * x - b * x - c * x
Strategy: expand
Steps (Aggressive Mode):
1. Distribute  [Expandir la expresión]
   Before: x * (a - b - c)
   Cambio local: x * (a - b - c) -> x * (a - b) - x * c
   After: x * (a - b) - x * c
2. Distribute  [Expandir la expresión]
   Before: x * (a - b) - x * c
   Cambio local: x * (a - b) -> x * a - x * b
   After: a * x - b * x - c * x
Result: a * x - b * x - c * x
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `x · (a - b - c)`
   - after: `x · (a - b) - x · c`
   - substeps:
     1. `Identificar los productos que genera la distributiva`
     2. `Escribir los productos con los signos originales`
2. `Expandir la expresión`
   - before: `x · (a - b) - x · c`
   - after: `a · x - b · x - c · x`
   - substeps:
     1. `Identificar los productos que genera la distributiva`
     2. `Escribir los productos con los signos originales`

## expand_common_factor_sum (expand)

- Source: `a*(b+c)`
- Target: `a*b + a*c`
- Result: `a * b + a * c`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * (b + c)
Target: a * b + a * c
Strategy: expand
Steps (Aggressive Mode):
1. Distribute  [Expandir la expresión]
   Before: a * (b + c)
   Cambio local: a * (b + c) -> a * b + a * c
   After: a * b + a * c
Result: a * b + a * c
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `a · (b + c)`
   - after: `a · b + a · c`
   - substeps:
     1. `Identificar los productos que genera la distributiva`
     2. `Escribir los productos con los signos originales`

## expand_common_factor_sum_three_terms (expand)

- Source: `x*(a+b+c)`
- Target: `a*x + b*x + c*x`
- Result: `a * x + b * x + c * x`
- Web step count: `2`
- Web substep count: `4`
- Flags: none

### CLI

```text
Parsed: x * (a + b + c)
Target: a * x + b * x + c * x
Strategy: expand
Steps (Aggressive Mode):
1. Distribute  [Expandir la expresión]
   Before: x * (a + b + c)
   Cambio local: x * (a + b + c) -> x * (a + b) + x * c
   After: x * (a + b) + x * c
2. Distribute  [Expandir la expresión]
   Before: x * (a + b) + x * c
   Cambio local: x * (a + b) -> x * a + x * b
   After: a * x + b * x + c * x
Result: a * x + b * x + c * x
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `x · (a + b + c)`
   - after: `x · (a + b) + c · x`
   - substeps:
     1. `Identificar los productos que genera la distributiva`
     2. `Escribir los productos con los signos originales`
2. `Expandir la expresión`
   - before: `x · (a + b) + c · x`
   - after: `a · x + b · x + c · x`
   - substeps:
     1. `Identificar los productos que genera la distributiva`
     2. `Escribir los productos con los signos originales`

## expand_cube_difference_product (polynomial_product)

- Source: `(x-1)*(x^2+x+1)`
- Target: `x^3-1`
- Result: `x^3 - 1`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (x^2 + x + 1) * (x - 1)
Target: x^3 - 1
Strategy: expand
Steps (Aggressive Mode):
1. Sum/Difference of cubes  [Expandir la expresión]
   Before: (x - 1) * (x^(2) + x + 1)
   Cambio local: (x - 1) * (x^(2) + x + 1) -> x^(3) - 1
   After: x^3 - 1
Result: x^(3) - 1
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(x - 1) · (x^2 + x + 1)`
   - after: `x^3 - 1`
   - substeps:
     1. `Reconocer el patrón (a - b)(a^2 + ab + b^2)`
     2. `Aplicar (a - b)(a^2 + ab + b^2) = a^3 - b^3`

## expand_cube_sum_product (polynomial_product)

- Source: `(x+1)*(x^2-x+1)`
- Target: `x^3+1`
- Result: `x^3 + 1`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (x + 1) * (x^2 - x + 1)
Target: x^3 + 1
Strategy: expand
Steps (Aggressive Mode):
1. Sum/Difference of cubes  [Expandir la expresión]
   Before: (x + 1) * (x^(2) + 1 - x)
   Cambio local: (x + 1) * (x^(2) + 1 - x) -> x^(3) + 1
   After: x^3 + 1
Result: x^(3) + 1
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(x + 1) · (x^2 + 1 - x)`
   - after: `x^3 + 1`
   - substeps:
     1. `Reconocer el patrón (a + b)(a^2 - ab + b^2)`
     2. `Aplicar (a + b)(a^2 - ab + b^2) = a^3 + b^3`

## expand_difference_cubes (expand)

- Source: `(a-b)*(a^2+a*b+b^2)`
- Target: `a^3-b^3`
- Result: `a^3 - b^3`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a^2 + b^2 + a * b) * (a - b)
Target: a^3 - b^3
Strategy: expand
Steps (Aggressive Mode):
1. Expand the expression distributively  [Expandir la expresión]
   Before: (a^(2) + b^(2) + a * b) * (a - b)
   Cambio local: (a^(2) + b^(2) + a * b) * (a - b) -> a^(3) - b^(3)
   After: a^3 - b^3
Result: a^(3) - b^(3)
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(a^2 + b^2 + a · b) · (a - b)`
   - after: `a^3 - b^3`
   - substeps:
     1. `Reconocer el patrón (a - b)(a^2 + ab + b^2)`
     2. `Aplicar (a - b)(a^2 + ab + b^2) = a^3 - b^3`

## expand_difference_of_squares_quadratic_product (polynomial_product)

- Source: `(x^2+a^2)*(x^2-a^2)`
- Target: `x^4-a^4`
- Result: `x^4 - a^4`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a^2 + x^2) * (x^2 - a^2)
Target: x^4 - a^4
Strategy: expand
Steps (Aggressive Mode):
Result: x^(4) - a^(4)
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(a^2 + x^2) · (x^2 - a^2)`
   - after: `x^4 - a^4`
   - substeps:
     1. `Aplicar el producto de conjugados`
     2. `Simplificar las potencias`

## expand_eighth_power_minus_multifactor_product (polynomial_product)

- Source: `(x-a)*(x+a)*(x^2+a^2)*(x^4+a^4)`
- Target: `x^8-a^8`
- Result: `x^8 - a^8`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a + x) * (a^2 + x^2) * (a^4 + x^4) * (x - a)
Target: x^8 - a^8
Strategy: expand
Steps (Aggressive Mode):
1. (a-b)(a+b) = a² - b²  [Expandir la expresión]
   Before: (a^(4) + x^(4)) * (x^(4) - a^(4))
   Cambio local: (a^(4) + x^(4)) * (x^(4) - a^(4)) -> x^(8) - a^(8)
   After: x^8 - a^8
Result: x^(8) - a^(8)
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(a^4 + x^4) · (x^4 - a^4)`
   - after: `x^8 - a^8`
   - substeps:
     1. `Aplicar el producto de conjugados`
     2. `Simplificar las potencias`

## expand_exponential_power (simplify)

- Source: `exp(3*x)`
- Target: `exp(x)^3`
- Result: `e^x^3`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: e^(3 * x)
Target: e^x^3
Strategy: rewrite exponentials
Steps (Aggressive Mode):
1. Expand exp(n·u) as exp(u)^n  [Expandir potencia exponencial]
   Before: e^(3 * x)
   Cambio local: e^(3 * x) -> e^(x)^(3)
   After: e^x^3
Result: e^(x)^(3)
```

### Web / JSON Steps

1. `Reescribir potencia exponencial`
   - before: `e^(3 · x)`
   - after: `e^x^3`
   - substeps:
     1. `Usar e^(n·A) = (e^A)^n`

## expand_exponential_reciprocal (simplify)

- Source: `exp(-x)`
- Target: `1/exp(x)`
- Result: `1 / e^x`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: e^(-x)
Target: 1 / e^x
Strategy: rewrite exponentials
Steps (Aggressive Mode):
1. Expand exp(-u) as 1 / exp(u)  [Expandir como recíproco exponencial]
   Before: e^(-x)
   Cambio local: e^(-x) -> 1 / e^(x)
   After: 1 / e^x
Result: 1 / e^(x)
```

### Web / JSON Steps

1. `Reescribir recíproco exponencial`
   - before: `e^(-x)`
   - after: `1/e^x`
   - substeps:
     1. `Usar e^(-A) = 1/e^A`

## expand_exponential_sum (simplify)

- Source: `exp(x+y)`
- Target: `exp(x)*exp(y)`
- Result: `e^x * e^y`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: e^(x + y)
Target: e^x * e^y
Strategy: rewrite exponentials
Steps (Aggressive Mode):
1. Expand exp(u ± v ± ...) into products/quotients of exponentials  [Expandir exponencial de suma o diferencia]
   Before: e^(x + y)
   Cambio local: e^(x + y) -> e^(x) * e^(y)
   After: e^x * e^y
Result: e^(x) * e^(y)
```

### Web / JSON Steps

1. `Reescribir exponenciales`
   - before: `e^(x + y)`
   - after: `e^x · e^y`
   - substeps:
     1. `Usar e^(A+B) = e^A · e^B`

## expand_fraction_exact_division_term_plus_remainder (fraction_expand)

- Source: `(a*d+b)/d`
- Target: `a + b/d`
- Result: `b / d + a`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a * d + b) / d
Target: b / d + a
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Repartir el denominador común]
   Before: (a * d + b) / d
   After: b / d + a
Result: b / d + a
ℹ️ Requires:
  • d ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(a · d + b)/d`
   - after: `b/d + a`
   - substeps:
     1. `Cancelar los factores comunes en la fracción que queda`

## expand_fraction_mixed_variable_term_cancellation (fraction_expand)

- Source: `(a*x+b*y)/(x*y)`
- Target: `a/y + b/x`
- Result: `a / y + b / x`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a * x + b * y) / (x * y)
Target: a / y + b / x
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Repartir el denominador común]
   Before: (a * x + b * y) / (x * y)
   After: a / y + b / x
Result: a / y + b / x
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(a · x + b · y)/(x · y)`
   - after: `a/y + b/x`
   - substeps:
     1. `Cancelar los factores comunes en las fracciones resultantes`

## expand_fraction_part_with_same_denominator_three_terms (fraction_expand)

- Source: `1 + (a+b+c)/d`
- Target: `1 + a/d + b/d + c/d`
- Result: `a / d + b / d + c / d + 1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (a + b + c) / d + 1
Target: a / d + b / d + c / d + 1
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Repartir el denominador común]
   Before: (a + b + c) / d + 1
   After: a / d + b / d + c / d + 1
Result: a / d + b / d + c / d + 1
ℹ️ Requires:
  • d ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(a + b + c)/d + 1`
   - after: `a/d + b/d + c/d + 1`
   - substeps: none

## expand_fraction_simple (fraction_expand)

- Source: `(a+b)/d`
- Target: `a/d + b/d`
- Result: `a / d + b / d`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (a + b) / d
Target: a / d + b / d
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Repartir el denominador común]
   Before: (a + b) / d
   After: a / d + b / d
Result: a / d + b / d
ℹ️ Requires:
  • d ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(a + b)/d`
   - after: `a/d + b/d`
   - substeps: none

## expand_fraction_three_factor_cross_cancellation_plus_remainder (fraction_expand)

- Source: `(a*x*y+b*y*z+c*x*z+d)/(x*y*z)`
- Target: `a/z + b/x + c/y + d/(x*y*z)`
- Result: `a / z + b / x + c / y + d / (x * y * z)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a * x * y + b * y * z + c * x * z + d) / (x * y * z)
Target: a / z + b / x + c / y + d / (x * y * z)
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Repartir el denominador común]
   Before: (a * x * y + b * y * z + c * x * z + d) / (x * y * z)
   After: a / z + b / x + c / y + d / (x * y * z)
Result: a / z + b / x + c / y + d / (x * y * z)
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
  • z ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(a · x · y + b · y · z + c · x · z + d)/(x · y · z)`
   - after: `a/z + b/x + c/y + d/(x · y · z)`
   - substeps:
     1. `Cancelar los factores comunes en las fracciones resultantes`

## expand_fraction_three_factor_full_cancellation (fraction_expand)

- Source: `(a*x+b*y+c*z)/(x*y*z)`
- Target: `a/(y*z) + b/(x*z) + c/(x*y)`
- Result: `a / (y * z) + b / (x * z) + c / (x * y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a * x + b * y + c * z) / (x * y * z)
Target: a / (y * z) + b / (x * z) + c / (x * y)
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Repartir el denominador común]
   Before: (a * x + b * y + c * z) / (x * y * z)
   After: a / (y * z) + b / (x * z) + c / (x * y)
Result: a / (y * z) + b / (x * z) + c / (x * y)
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
  • z ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(a · x + b · y + c · z)/(x · y · z)`
   - after: `a/(y · z) + b/(x · z) + c/(x · y)`
   - substeps:
     1. `Cancelar los factores comunes en las fracciones resultantes`

## expand_fraction_three_factor_three_cancellations_to_constant (fraction_expand)

- Source: `(a*x*y+b*x*z+c*y*z+d*x*y*z)/(x*y*z)`
- Target: `a/z + b/y + c/x + d`
- Result: `a / z + b / y + c / x + d`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (d * x * y * z + a * x * y + b * x * z + c * y * z) / (x * y * z)
Target: a / z + b / y + c / x + d
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Repartir el denominador común]
   Before: (d * x * y * z + a * x * y + b * x * z + c * y * z) / (x * y * z)
   After: a / z + b / y + c / x + d
Result: a / z + b / y + c / x + d
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
  • z ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(d · x · y · z + a · x · y + b · x · z + c · y · z)/(x · y · z)`
   - after: `a/z + b/y + c/x + d`
   - substeps:
     1. `Cancelar los factores comunes en las fracciones resultantes`

## expand_fraction_two_cancellations_plus_remainder (fraction_expand)

- Source: `(a*x+b*y+c)/(x*y)`
- Target: `a/y + b/x + c/(x*y)`
- Result: `a / y + b / x + c / (x * y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a * x + b * y + c) / (x * y)
Target: a / y + b / x + c / (x * y)
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Repartir el denominador común]
   Before: (a * x + b * y + c) / (x * y)
   After: a / y + b / x + c / (x * y)
Result: a / y + b / x + c / (x * y)
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(a · x + b · y + c)/(x · y)`
   - after: `a/y + b/x + c/(x · y)`
   - substeps:
     1. `Cancelar los factores comunes en las fracciones resultantes`

## expand_fraction_with_common_scalar_factor_in_denominator (fraction_expand)

- Source: `(a*x+b)/(c*x)`
- Target: `a/c + b/(c*x)`
- Result: `a / c + b / (c * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a * x + b) / (c * x)
Target: a / c + b / (c * x)
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Repartir el denominador común]
   Before: (a * x + b) / (c * x)
   After: a / c + b / (c * x)
Result: a / c + b / (c * x)
ℹ️ Requires:
  • c ≠ 0
  • x ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(a · x + b)/(c · x)`
   - after: `a/c + b/(c · x)`
   - substeps:
     1. `Cancelar los factores comunes en la fracción que queda`

## expand_fraction_with_term_cancellation (fraction_expand)

- Source: `(a*y+b*x)/(x*y)`
- Target: `a/x + b/y`
- Result: `a / x + b / y`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a * y + b * x) / (x * y)
Target: a / x + b / y
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Repartir el denominador común]
   Before: (a * y + b * x) / (x * y)
   After: a / x + b / y
Result: a / x + b / y
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(a · y + b · x)/(x · y)`
   - after: `a/x + b/y`
   - substeps:
     1. `Cancelar los factores comunes en las fracciones resultantes`

## expand_fractional_binomial_square (expand)

- Source: `(x+1/2)^2`
- Target: `x^2 + x + 1/4`
- Result: `1 / 4 + x^2 + x`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (1 / 2 + x)^2
Target: 1 / 4 + x^2 + x
Strategy: expand
Steps (Aggressive Mode):
1. Expand binomial power ^2  [Expandir binomio]
   Before: (x + 1/2)^(2)
   Cambio local: (x + 1/2)^(2) -> 1/2^(2) + x^(2) + 2 * x/2
   After: 1 / 4 + x^2 + x
Result: 1 / 4 + x^(2) + x
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `(x + 1/2)^2`
   - after: `(1/2)^2 + x^2 + 2/2 · x`
   - substeps: none

## expand_higher_odd_half_power (radical_power)

- Source: `x^(5/2)`
- Target: `abs(x)^2*sqrt(x)`
- Result: `sqrt(x) * |x|^2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: x^(5 / 2)
Target: sqrt(x) * |x|^2
Strategy: expand odd half power
Steps (Aggressive Mode):
1. Rewrite an odd half-integer power using a square root  [Reescribir potencia semientera impar]
   Before: x^(5 / 2)
   After: sqrt(x) * |x|^2
Result: sqrt(x) * |x|^(2)
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Extraer potencia par de la raíz`
   - before: `sqrt(x^5)`
   - after: `sqrt(x) · |x|^2`
   - substeps:
     1. `Separar el radicando en una potencia par y un factor`
     2. `Como x ≥ 0, sacar x^2 fuera de la raíz`

## expand_higher_odd_half_power_after_simplify (radical_power)

- Source: `sqrt(x^5)`
- Target: `abs(x)^2*sqrt(x)`
- Result: `sqrt(x) * |x|^2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sqrt(x^5)
Target: sqrt(x) * |x|^2
Strategy: expand odd half power
Steps (Aggressive Mode):
1. Rewrite an odd half-integer power using a square root  [Reescribir potencia semientera impar]
   Before: sqrt(x^(5))
   After: sqrt(x) * |x|^2
Result: sqrt(x) * |x|^(2)
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Extraer potencia par de la raíz`
   - before: `sqrt(x^5)`
   - after: `sqrt(x) · |x|^2`
   - substeps:
     1. `Separar el radicando en una potencia par y un factor`
     2. `Como x ≥ 0, sacar x^2 fuera de la raíz`

## expand_higher_odd_half_power_alt_var (radical_power)

- Source: `y^(7/2)`
- Target: `abs(y)^3*sqrt(y)`
- Result: `sqrt(y) * |y|^3`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: y^(7 / 2)
Target: sqrt(y) * |y|^3
Strategy: expand odd half power
Steps (Aggressive Mode):
1. Rewrite an odd half-integer power using a square root  [Reescribir potencia semientera impar]
   Before: y^(7 / 2)
   After: sqrt(y) * |y|^3
Result: sqrt(y) * |y|^(3)
ℹ️ Requires:
  • y ≥ 0
```

### Web / JSON Steps

1. `Extraer potencia par de la raíz`
   - before: `sqrt(y^7)`
   - after: `sqrt(y) · |y|^3`
   - substeps:
     1. `Separar el radicando en una potencia par y un factor`
     2. `Como y ≥ 0, sacar y^3 fuera de la raíz`

## expand_hyperbolic_cosh_difference (expand)

- Source: `cosh(x-y)`
- Target: `cosh(x)*cosh(y) - sinh(x)*sinh(y)`
- Result: `cosh(x) * cosh(y) - sinh(x) * sinh(y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cosh(x - y)
Target: cosh(x) * cosh(y) - sinh(x) * sinh(y)
Strategy: expand
Steps (Aggressive Mode):
1. Recognize cosh(u)·cosh(v) ± sinh(u)·sinh(v) as cosh(u ± v)  [Aplicar identidad hiperbólica de suma/diferencia de ángulos]
   Before: cosh(x - y)
   Cambio local: cosh(x - y) -> cosh(x) * cosh(y) - sinh(x) * sinh(y)
   After: cosh(x) * cosh(y) - sinh(x) * sinh(y)
Result: cosh(x) * cosh(y) - sinh(x) * sinh(y)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de suma/diferencia de ángulos`
   - before: `cosh(x - y)`
   - after: `cosh(x) · cosh(y) - sinh(x) · sinh(y)`
   - substeps:
     1. `Usar cosh(A-B) = cosh(A) · cosh(B) - sinh(A) · sinh(B)`

## expand_hyperbolic_cosh_difference_to_product_exact (expand)

- Source: `cosh(x)-cosh(y)`
- Target: `2*sinh((x+y)/2)*sinh((x-y)/2)`
- Result: `2 * sinh((x + y) / 2) * sinh((x - y) / 2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cosh(x) - cosh(y)
Target: 2 * sinh((x + y) / 2) * sinh((x - y) / 2)
Strategy: expand
Steps (Aggressive Mode):
1. Contract cosh(u) - cosh(v) into 2·sinh((u + v)/2)·sinh((u - v)/2)  [Aplicar identidad hiperbólica de producto a suma]
   Before: cosh(x) - cosh(y)
   Cambio local: cosh(x) - cosh(y) -> 2 * sinh((x + y) / 2) * sinh((x - y) / 2)
   After: 2 * sinh((x + y) / 2) * sinh((x - y) / 2)
Result: 2 * sinh((x + y) / 2) * sinh((x - y) / 2)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de producto a suma`
   - before: `cosh(x) - cosh(y)`
   - after: `2 · sinh((x + y)/2) · sinh((x - y)/2)`
   - substeps:
     1. `Usar cosh(A)-cosh(B) = 2·sinh((A+B)/2)·sinh((A-B)/2)`

## expand_hyperbolic_cosh_sum (expand)

- Source: `cosh(x+y)`
- Target: `cosh(x)*cosh(y) + sinh(x)*sinh(y)`
- Result: `sinh(x) * sinh(y) + cosh(x) * cosh(y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cosh(x + y)
Target: sinh(x) * sinh(y) + cosh(x) * cosh(y)
Strategy: expand
Steps (Aggressive Mode):
1. Recognize cosh(u)·cosh(v) ± sinh(u)·sinh(v) as cosh(u ± v)  [Aplicar identidad hiperbólica de suma/diferencia de ángulos]
   Before: cosh(x + y)
   Cambio local: cosh(x + y) -> sinh(x) * sinh(y) + cosh(x) * cosh(y)
   After: sinh(x) * sinh(y) + cosh(x) * cosh(y)
Result: sinh(x) * sinh(y) + cosh(x) * cosh(y)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de suma/diferencia de ángulos`
   - before: `cosh(x + y)`
   - after: `sinh(x) · sinh(y) + cosh(x) · cosh(y)`
   - substeps:
     1. `Usar cosh(A+B) = cosh(A) · cosh(B) + sinh(A) · sinh(B)`

## expand_hyperbolic_cosh_sum_to_product_exact (expand)

- Source: `cosh(x)+cosh(y)`
- Target: `2*cosh((x+y)/2)*cosh((x-y)/2)`
- Result: `2 * cosh((x + y) / 2) * cosh((x - y) / 2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cosh(x) + cosh(y)
Target: 2 * cosh((x + y) / 2) * cosh((x - y) / 2)
Strategy: expand
Steps (Aggressive Mode):
1. Contract cosh(u) + cosh(v) into 2·cosh((u + v)/2)·cosh((u - v)/2)  [Aplicar identidad hiperbólica de producto a suma]
   Before: cosh(x) + cosh(y)
   Cambio local: cosh(x) + cosh(y) -> 2 * cosh((x + y) / 2) * cosh((x - y) / 2)
   After: 2 * cosh((x + y) / 2) * cosh((x - y) / 2)
Result: 2 * cosh((x + y) / 2) * cosh((x - y) / 2)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de producto a suma`
   - before: `cosh(x) + cosh(y)`
   - after: `2 · cosh((x + y)/2) · cosh((x - y)/2)`
   - substeps:
     1. `Usar cosh(A)+cosh(B) = 2·cosh((A+B)/2)·cosh((A-B)/2)`

## expand_hyperbolic_product_to_sum_to_cosh_cubic_polynomial (expand)

- Source: `2*sinh(2*x)*sinh(x)`
- Target: `4*cosh(x)^3-4*cosh(x)`
- Result: `4 * cosh(x)^3 - 4 * cosh(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * sinh(x) * sinh(2 * x)
Target: 4 * cosh(x)^3 - 4 * cosh(x)
Strategy: expand
Steps (Aggressive Mode):
1. Expand 2·sinh(u)·sinh(v) as cosh(u + v) - cosh(u - v)  [Aplicar identidad hiperbólica de producto a suma]
   Before: 2 * sinh(x) * sinh(2 * x)
   Cambio local: 2 * sinh(x) * sinh(2 * x) -> 4 * cosh(x)^(3) - 4 * cosh(x)
   After: 4 * cosh(x)^3 - 4 * cosh(x)
Result: 4 * cosh(x)^(3) - 4 * cosh(x)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de producto a suma`
   - before: `2 · sinh(x) · sinh(2 · x)`
   - after: `4 · cosh(x)^3 - 4 · cosh(x)`
   - substeps:
     1. `Usar 2·sinh(A)·sinh(B) = cosh(A+B) - cosh(A-B)`

## expand_hyperbolic_product_to_sum_to_cosh_cubic_polynomial_with_passthrough (expand)

- Source: `2*sinh(2*x)*sinh(x)+a`
- Target: `4*cosh(x)^3-4*cosh(x)+a`
- Result: `-4 * cosh(x) + 4 * cosh(x)^3 + a`
- Web step count: `2`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 * sinh(x) * sinh(2 * x) + a
Target: -4 * cosh(x) + 4 * cosh(x)^3 + a
Strategy: expand
Steps (Aggressive Mode):
1. Apply a hyperbolic product-to-sum or sum-to-product identity  [Aplicar identidad hiperbólica de producto a suma]
   Before: 2 * sinh(x) * sinh(2 * x) + a
   Cambio local: 2 * sinh(x) * sinh(2 * x) + a -> -cosh(x) + cosh(3 * x) + a
   After: -cosh(x) + cosh(3 * x) + a
2. Combine cosh(u) ± cosh(3u) using the hyperbolic triple-angle identity  [Aplicar identidad hiperbólica de ángulo triple]
   Before: -cosh(x) + cosh(3 * x) + a
   Cambio local: -cosh(x) + cosh(3 * x) + a -> -4 * cosh(x) + 4 * cosh(x)^(3) + a
   After: -4 * cosh(x) + 4 * cosh(x)^3 + a
Result: -4 * cosh(x) + 4 * cosh(x)^(3) + a
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de producto a suma`
   - before: `2 · sinh(x) · sinh(2 · x) + a`
   - after: `cosh(3 · x) - cosh(x) + a`
   - substeps:
     1. `Usar 2·sinh(A)·sinh(B) = cosh(A+B) - cosh(A-B)`
2. `Aplicar identidad hiperbólica de ángulo triple`
   - before: `cosh(3 · x) - cosh(x) + a`
   - after: `4 · cosh(x)^3 - 4 · cosh(x) + a`
   - substeps:
     1. `Usar cosh(3·x) = 4·cosh(x)^3 - 3·cosh(x)`

## expand_hyperbolic_product_to_sum_to_sinh_cubic_polynomial (expand)

- Source: `2*sinh(2*x)*cosh(x)`
- Target: `4*sinh(x)+4*sinh(x)^3`
- Result: `4 * sinh(x) + 4 * sinh(x)^3`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * sinh(2 * x) * cosh(x)
Target: 4 * sinh(x) + 4 * sinh(x)^3
Strategy: expand
Steps (Aggressive Mode):
1. Expand 2·sinh(u)·cosh(v) as sinh(u + v) + sinh(u - v)  [Aplicar identidad hiperbólica de producto a suma]
   Before: 2 * sinh(2 * x) * cosh(x)
   Cambio local: 2 * sinh(2 * x) * cosh(x) -> 4 * sinh(x) + 4 * sinh(x)^(3)
   After: 4 * sinh(x) + 4 * sinh(x)^3
Result: 4 * sinh(x) + 4 * sinh(x)^(3)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de producto a suma`
   - before: `2 · sinh(2 · x) · cosh(x)`
   - after: `4 · sinh(x) + 4 · sinh(x)^3`
   - substeps:
     1. `Usar 2·sinh(A)·cosh(B) = sinh(A+B) + sinh(A-B)`

## expand_hyperbolic_sinh_difference (expand)

- Source: `sinh(x-y)`
- Target: `sinh(x)*cosh(y) - cosh(x)*sinh(y)`
- Result: `sinh(x) * cosh(y) - sinh(y) * cosh(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sinh(x - y)
Target: sinh(x) * cosh(y) - sinh(y) * cosh(x)
Strategy: expand
Steps (Aggressive Mode):
1. Recognize sinh(u)·cosh(v) ± cosh(u)·sinh(v) as sinh(u ± v)  [Aplicar identidad hiperbólica de suma/diferencia de ángulos]
   Before: sinh(x - y)
   Cambio local: sinh(x - y) -> sinh(x) * cosh(y) - sinh(y) * cosh(x)
   After: sinh(x) * cosh(y) - sinh(y) * cosh(x)
Result: sinh(x) * cosh(y) - sinh(y) * cosh(x)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de suma/diferencia de ángulos`
   - before: `sinh(x - y)`
   - after: `sinh(x) · cosh(y) - sinh(y) · cosh(x)`
   - substeps:
     1. `Usar sinh(A-B) = sinh(A) · cosh(B) - cosh(A) · sinh(B)`

## expand_hyperbolic_sinh_sum (expand)

- Source: `sinh(x+y)`
- Target: `sinh(x)*cosh(y) + cosh(x)*sinh(y)`
- Result: `sinh(x) * cosh(y) + sinh(y) * cosh(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sinh(x + y)
Target: sinh(x) * cosh(y) + sinh(y) * cosh(x)
Strategy: expand
Steps (Aggressive Mode):
1. Recognize sinh(u)·cosh(v) ± cosh(u)·sinh(v) as sinh(u ± v)  [Aplicar identidad hiperbólica de suma/diferencia de ángulos]
   Before: sinh(x + y)
   Cambio local: sinh(x + y) -> sinh(x) * cosh(y) + sinh(y) * cosh(x)
   After: sinh(x) * cosh(y) + sinh(y) * cosh(x)
Result: sinh(x) * cosh(y) + sinh(y) * cosh(x)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de suma/diferencia de ángulos`
   - before: `sinh(x + y)`
   - after: `sinh(x) · cosh(y) + sinh(y) · cosh(x)`
   - substeps:
     1. `Usar sinh(A+B) = sinh(A) · cosh(B) + cosh(A) · sinh(B)`

## expand_hyperbolic_sinh_sum_to_product_exact (expand)

- Source: `sinh(x)+sinh(y)`
- Target: `2*sinh((x+y)/2)*cosh((x-y)/2)`
- Result: `2 * sinh((x + y) / 2) * cosh((x - y) / 2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sinh(x) + sinh(y)
Target: 2 * sinh((x + y) / 2) * cosh((x - y) / 2)
Strategy: expand
Steps (Aggressive Mode):
1. Contract sinh(u) ± sinh(v) into 2·cosh((u + v)/2)·sinh((u - v)/2) or 2·sinh((u + v)/2)·cosh((u - v)/2)  [Aplicar identidad hiperbólica de producto a suma]
   Before: sinh(x) + sinh(y)
   Cambio local: sinh(x) + sinh(y) -> 2 * sinh((x + y) / 2) * cosh((x - y) / 2)
   After: 2 * sinh((x + y) / 2) * cosh((x - y) / 2)
Result: 2 * sinh((x + y) / 2) * cosh((x - y) / 2)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de producto a suma`
   - before: `sinh(x) + sinh(y)`
   - after: `2 · sinh((x + y)/2) · cosh((x - y)/2)`
   - substeps:
     1. `Usar sinh(A)+sinh(B) = 2·sinh((A+B)/2)·cosh((A-B)/2)`

## expand_hyperbolic_tanh_difference (simplify)

- Source: `tanh(x-y)`
- Target: `(tanh(x)-tanh(y))/(1-tanh(x)*tanh(y))`
- Result: `(tanh(x) - tanh(y)) / (1 - tanh(x) * tanh(y))`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: tanh(x - y)
Target: (tanh(x) - tanh(y)) / (1 - tanh(x) * tanh(y))
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Expand tanh(u ± v) as (tanh(u) ± tanh(v)) / (1 ± tanh(u)·tanh(v))  [Aplicar identidad hiperbólica de suma/diferencia de ángulos]
   Before: tanh(x - y)
   Cambio local: tanh(x - y) -> (tanh(x) - tanh(y)) / (1 - tanh(x) * tanh(y))
   After: (tanh(x) - tanh(y)) / (1 - tanh(x) * tanh(y))
Result: (tanh(x) - tanh(y)) / (1 - tanh(x) * tanh(y))
ℹ️ Requires:
  • 1 - tanh(x) * tanh(y) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de suma/diferencia de ángulos`
   - before: `tanh(x - y)`
   - after: `(tanh(x) - tanh(y))/(1 - tanh(x) · tanh(y))`
   - substeps:
     1. `Usar tanh(A-B) = (tanh(A) - tanh(B)) / (1 - tanh(A)·tanh(B))`

## expand_hyperbolic_tanh_sum (simplify)

- Source: `tanh(x+y)`
- Target: `(tanh(x)+tanh(y))/(1+tanh(x)*tanh(y))`
- Result: `(tanh(x) + tanh(y)) / (tanh(x) * tanh(y) + 1)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: tanh(x + y)
Target: (tanh(x) + tanh(y)) / (tanh(x) * tanh(y) + 1)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Expand tanh(u ± v) as (tanh(u) ± tanh(v)) / (1 ± tanh(u)·tanh(v))  [Aplicar identidad hiperbólica de suma/diferencia de ángulos]
   Before: tanh(x + y)
   Cambio local: tanh(x + y) -> (tanh(x) + tanh(y)) / (tanh(x) * tanh(y) + 1)
   After: (tanh(x) + tanh(y)) / (tanh(x) * tanh(y) + 1)
Result: (tanh(x) + tanh(y)) / (tanh(x) * tanh(y) + 1)
ℹ️ Requires:
  • tanh(x) * tanh(y) + 1 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de suma/diferencia de ángulos`
   - before: `tanh(x + y)`
   - after: `(tanh(x) + tanh(y))/(tanh(x) · tanh(y) + 1)`
   - substeps:
     1. `Usar tanh(A+B) = (tanh(A) + tanh(B)) / (1 + tanh(A)·tanh(B))`

## expand_hyperbolic_tanh_triple_angle (simplify)

- Source: `tanh(3*x)`
- Target: `(3*tanh(x)+tanh(x)^3)/(1+3*tanh(x)^2)`
- Result: `(tanh(x)^3 + 3 * tanh(x)) / (3 * tanh(x)^2 + 1)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: tanh(3 * x)
Target: (tanh(x)^3 + 3 * tanh(x)) / (3 * tanh(x)^2 + 1)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Expand tanh(3u) as (3·tanh(u) + tanh(u)^3) / (1 + 3·tanh(u)^2)  [Aplicar identidad hiperbólica de ángulo triple]
   Before: tanh(3 * x)
   Cambio local: tanh(3 * x) -> (tanh(x)^(3) + 3 * tanh(x)) / (3 * tanh(x)^(2) + 1)
   After: (tanh(x)^3 + 3 * tanh(x)) / (3 * tanh(x)^2 + 1)
Result: (tanh(x)^(3) + 3 * tanh(x)) / (3 * tanh(x)^(2) + 1)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de ángulo triple`
   - before: `tanh(3 · x)`
   - after: `(tanh(x)^3 + 3 · tanh(x))/(3 · tanh(x)^2 + 1)`
   - substeps: none

## expand_log_change_of_base_chain (log_expand)

- Source: `log(b,c)`
- Target: `log(b,a)*log(a,c)`
- Result: `log(a, c) * log(b, a)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: log(b, c)
Target: log(a, c) * log(b, a)
Strategy: expand_log
Steps (Aggressive Mode):
1. Expand the logarithm using a change-of-base chain  [Aplicar cambio de base]
   Before: log(b, c)
   Cambio local: log(b, c) -> log(a, c) * log(b, a)
   After: log(a, c) * log(b, a)
Result: log(a, c) * log(b, a)
ℹ️ Requires:
  • a > 0
  • b > 0
  • c > 0
```

### Web / JSON Steps

1. `Expandir cambio de base`
   - before: `log_b(c)`
   - after: `log_a(c) · log_b(a)`
   - substeps: none

## expand_log_change_of_base_direct (log_expand)

- Source: `log(2, x)`
- Target: `ln(x)/ln(2)`
- Result: `ln(x) / ln(2)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: log(2, x)
Target: ln(x) / ln(2)
Strategy: expand_log
Steps (Aggressive Mode):
1. Rewrite the logarithm using the change-of-base formula  [Aplicar cambio de base]
   Before: log(2, x)
   Subpasos:
     1.1 Poner el argumento en el numerador
         x -> ln(x)
     1.2 Poner la base en el denominador
         2 -> ln(2)
     1.3 Formar el cociente de cambio de base
         log(2, x) -> ln(x) / ln(2)
   Cambio local: log(2, x) -> ln(x) / ln(2)
   After: ln(x) / ln(2)
Result: ln(x) / ln(2)
ℹ️ Requires:
  • x > 0
```

### Web / JSON Steps

1. `Aplicar cambio de base`
   - before: `log_2(x)`
   - after: `ln(x)/ln(2)`
   - substeps:
     1. `Poner el argumento en el numerador`
     2. `Poner la base en el denominador`

## expand_log_even_power_abs (log_expand)

- Source: `ln(x^2)`
- Target: `2*ln(abs(x))`
- Result: `2 * ln(|x|)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(x^2)
Target: 2 * ln(|x|)
Strategy: expand_log
Steps (Aggressive Mode):
1. Recognize an even power inside the logarithm  [Sacar un exponente fuera del logaritmo]
   Before: ln(x^(2))
   Cambio local: ln(x^(2)) -> 2 * ln(|x|)
   After: 2 * ln(|x|)
Result: 2 * ln(|x|)
ℹ️ Requires:
  • x ≠ 0
```

### Web / JSON Steps

1. `Sacar un exponente fuera del logaritmo`
   - before: `ln(x^2)`
   - after: `2 · ln(|x|)`
   - substeps: none

## expand_log_general_base_power (log_expand)

- Source: `log(b, x^3)`
- Target: `3*log(b, x)`
- Result: `3 * log(b, x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: log(b, x^3)
Target: 3 * log(b, x)
Strategy: expand_log
Steps (Aggressive Mode):
1. log(b, x^y) = y * log(b, x)  [Evaluar logaritmos]
   Before: log(b, x^(3))
   Cambio local: log(b, x^(3)) -> 3 * log(b, x)
   After: 3 * log(b, x)
Result: 3 * log(b, x)
ℹ️ Requires:
  • b > 0
  • x > 0
```

### Web / JSON Steps

1. `Sacar un exponente fuera del logaritmo`
   - before: `log_b(x^3)`
   - after: `3 · log_b(x)`
   - substeps: none

## expand_log_general_base_powered_two_denominator_factors_with_powered_denominator (log_expand)

- Source: `log(b, (x^2*y^3)/(z^2*t))`
- Target: `2*log(b, x) + 3*log(b, y) - 2*log(b, z) - log(b, t)`
- Result: `2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t)`
- Web step count: `2`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: log(b, x^2 * y^3 / (t * z^2))
Target: 2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t)
Strategy: expand_log
Steps (Aggressive Mode):
1. Log expansion  [Expandir logaritmos]
   Before: log(b, x^(2) * y^(3) / (t * z^(2)))
   Cambio local: log(b, x^(2) * y^(3) / (t * z^(2))) -> log(b, x^(2)) + log(b, y^(3)) - (log(b, t) + log(b, z^(2)))
   After: log(b, x^(2)) + log(b, y^(3)) - (log(b, t) + log(b, z^(2)))
   ℹ️ Requires: x^2 * y^3 > 0
   ℹ️ Requires: t * z^2 > 0
   ℹ️ Requires: x^2 > 0
   ℹ️ Requires: y^3 > 0
   ℹ️ Requires: t > 0
   ℹ️ Requires: z^2 > 0
2. log(b, x^y) = y * log(b, x)  [Evaluar logaritmos]
   Before: log(b, x^(2)) + log(b, y^(3)) - (log(b, t) + log(b, z^(2)))
   Cambio local: log(b, x^(2)) + log(b, y^(3)) - (log(b, t) + log(b, z^(2))) -> 2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t)
   After: 2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t)
Result: 2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t)
ℹ️ Requires:
  • b > 0
  • t > 0
  • x > 0
  • y > 0
  • z > 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `log_b((x^2 · y^3)/(t · z^2))`
   - after: `log_b(x^2) + log_b(y^3) - (log_b(t) + log_b(z^2))`
   - substeps: none
2. `Sacar un exponente fuera del logaritmo`
   - before: `log_b(x^2) + log_b(y^3) - (log_b(t) + log_b(z^2))`
   - after: `2 · log_b(x) + 3 · log_b(y) - 2 · log_b(z) - log_b(t)`
   - substeps: none

## expand_log_general_base_product_over_quotient (log_expand)

- Source: `log(b, (x*y)/z)`
- Target: `log(b, x) + log(b, y) - log(b, z)`
- Result: `log(b, x) + log(b, y) - log(b, z)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: log(b, x * y / z)
Target: log(b, x) + log(b, y) - log(b, z)
Strategy: expand_log
Steps (Aggressive Mode):
1. Log expansion  [Expandir logaritmos]
   Before: log(b, x * y / z)
   Cambio local: log(b, x * y / z) -> log(b, x) + log(b, y) - log(b, z)
   After: log(b, x) + log(b, y) - log(b, z)
   ℹ️ Requires: x * y > 0
   ℹ️ Requires: z > 0
   ℹ️ Requires: x > 0
   ℹ️ Requires: y > 0
Result: log(b, x) + log(b, y) - log(b, z)
ℹ️ Requires:
  • b > 0
  • x > 0
  • y > 0
  • z > 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `log_b((x · y)/z)`
   - after: `log_b(x) + log_b(y) - log_b(z)`
   - substeps: none

## expand_log_grouped_abs_product (log_expand)

- Source: `2*ln(abs(x*y))`
- Target: `2*ln(abs(x))+2*ln(abs(y))`
- Result: `2 * ln(|x|) + 2 * ln(|y|)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * ln(|x * y|)
Target: 2 * ln(|x|) + 2 * ln(|y|)
Strategy: expand_log
Steps (Aggressive Mode):
1. Expand a logarithm into a sum of logarithms  [Expandir logaritmos]
   Before: 2 * ln(|x * y|)
   Cambio local: 2 * ln(|x * y|) -> 2 * ln(|x|) + 2 * ln(|y|)
   After: 2 * ln(|x|) + 2 * ln(|y|)
Result: 2 * ln(|x|) + 2 * ln(|y|)
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `2 · ln(|x · y|)`
   - after: `2 · ln(|x|) + 2 · ln(|y|)`
   - substeps: none

## expand_log_grouped_abs_product_with_passthrough (log_expand)

- Source: `2*ln(abs(x*y))+a`
- Target: `2*ln(abs(x))+2*ln(abs(y))+a`
- Result: `2 * ln(|x|) + 2 * ln(|y|) + a`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * ln(|x * y|) + a
Target: 2 * ln(|x|) + 2 * ln(|y|) + a
Strategy: expand_log
Steps (Aggressive Mode):
1. Expand a logarithm into a sum of logarithms  [Expandir logaritmos]
   Before: 2 * ln(|x * y|) + a
   Cambio local: 2 * ln(|x * y|) + a -> 2 * ln(|x|) + 2 * ln(|y|) + a
   After: 2 * ln(|x|) + 2 * ln(|y|) + a
Result: 2 * ln(|x|) + 2 * ln(|y|) + a
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `2 · ln(|x · y|) + a`
   - after: `2 · ln(|x|) + 2 · ln(|y|) + a`
   - substeps: none

## expand_log_grouped_general_base_power (log_expand)

- Source: `log(b,(x*y)^2)`
- Target: `2*log(b,x)+2*log(b,y)`
- Result: `2 * log(b, x) + 2 * log(b, y)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: log(b, (x * y)^2)
Target: 2 * log(b, x) + 2 * log(b, y)
Strategy: expand_log
Steps (Aggressive Mode):
1. Expand a logarithm into a sum of logarithms  [Expandir logaritmos]
   Before: log(b, (x * y)^(2))
   Cambio local: log(b, (x * y)^(2)) -> 2 * log(b, x) + 2 * log(b, y)
   After: 2 * log(b, x) + 2 * log(b, y)
Result: 2 * log(b, x) + 2 * log(b, y)
ℹ️ Requires:
  • b > 0
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `log_b((x · y)^2)`
   - after: `2 · log_b(x) + 2 · log_b(y)`
   - substeps: none

## expand_log_grouped_general_base_power_with_passthrough (log_expand)

- Source: `log(b,(x*y)^2)+a`
- Target: `2*log(b,x)+2*log(b,y)+a`
- Result: `2 * log(b, x) + 2 * log(b, y) + a`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: log(b, (x * y)^2) + a
Target: 2 * log(b, x) + 2 * log(b, y) + a
Strategy: expand_log
Steps (Aggressive Mode):
1. Expand a logarithm into a sum of logarithms  [Expandir logaritmos]
   Before: log(b, (x * y)^(2)) + a
   Cambio local: log(b, (x * y)^(2)) + a -> 2 * log(b, x) + 2 * log(b, y) + a
   After: 2 * log(b, x) + 2 * log(b, y) + a
Result: 2 * log(b, x) + 2 * log(b, y) + a
ℹ️ Requires:
  • b > 0
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `log_b((x · y)^2) + a`
   - after: `2 · log_b(x) + 2 · log_b(y) + a`
   - substeps: none

## expand_log_grouped_power (log_expand)

- Source: `ln((x*y)^2)`
- Target: `ln(x^2)+ln(y^2)`
- Result: `ln(x^2) + ln(y^2)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln((x * y)^2)
Target: ln(x^2) + ln(y^2)
Strategy: expand_log
Steps (Aggressive Mode):
1. Expand a logarithm into a sum of logarithms  [Expandir logaritmos]
   Before: ln((x * y)^(2))
   Cambio local: ln((x * y)^(2)) -> ln(x^(2)) + ln(y^(2))
   After: ln(x^2) + ln(y^2)
Result: ln(x^(2)) + ln(y^(2))
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln((x · y)^2)`
   - after: `ln(x^2) + ln(y^2)`
   - substeps: none

## expand_log_grouped_power_with_passthrough (log_expand)

- Source: `ln((x*y)^2)+a`
- Target: `ln(x^2)+ln(y^2)+a`
- Result: `ln(x^2) + ln(y^2) + a`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln((x * y)^2) + a
Target: ln(x^2) + ln(y^2) + a
Strategy: expand_log
Steps (Aggressive Mode):
1. Expand a logarithm into a sum of logarithms  [Expandir logaritmos]
   Before: ln((x * y)^(2)) + a
   Cambio local: ln((x * y)^(2)) + a -> ln(x^(2)) + ln(y^(2)) + a
   After: ln(x^2) + ln(y^2) + a
Result: ln(x^(2)) + ln(y^(2)) + a
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln((x · y)^2) + a`
   - after: `ln(x^2) + ln(y^2) + a`
   - substeps: none

## expand_log_powered_two_denominator_factors (log_expand)

- Source: `ln((x^2*y)/(z*t))`
- Target: `2*ln(abs(x)) + ln(y) - ln(z) - ln(t)`
- Result: `ln(y) + 2 * ln(|x|) - ln(z) - ln(t)`
- Web step count: `2`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(y * x^2 / (t * z))
Target: ln(y) + 2 * ln(|x|) - ln(z) - ln(t)
Strategy: expand_log
Steps (Aggressive Mode):
1. Log expansion  [Expandir logaritmos]
   Before: ln(y * x^(2) / (t * z))
   Cambio local: ln(y * x^(2) / (t * z)) -> ln(y) + ln(x^(2)) - (ln(t) + ln(z))
   After: ln(y) + ln(x^(2)) - (ln(t) + ln(z))
   ℹ️ Requires: y * x^2 > 0
   ℹ️ Requires: t * z > 0
   ℹ️ Requires: y > 0
   ℹ️ Requires: x^2 > 0
   ℹ️ Requires: t > 0
   ℹ️ Requires: z > 0
2. log(b, x^y) = y * log(b, x)  [Evaluar logaritmos]
   Before: ln(y) + ln(x^(2)) - (ln(t) + ln(z))
   Cambio local: ln(y) + ln(x^(2)) - (ln(t) + ln(z)) -> ln(y) + 2 * ln(|x|) - ln(z) - ln(t)
   After: ln(y) + 2 * ln(|x|) - ln(z) - ln(t)
Result: ln(y) + 2 * ln(|x|) - ln(z) - ln(t)
ℹ️ Requires:
  • t > 0
  • x ≠ 0
  • y > 0
  • z > 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln((y · x^2)/(t · z))`
   - after: `ln(y) + ln(x^2) - (ln(t) + ln(z))`
   - substeps: none
2. `Sacar un exponente fuera del logaritmo`
   - before: `ln(y) + ln(x^2) - (ln(t) + ln(z))`
   - after: `ln(y) + 2 · ln(|x|) - ln(z) - ln(t)`
   - substeps: none

## expand_log_product (log_expand)

- Source: `ln(x*y)`
- Target: `ln(x) + ln(y)`
- Result: `ln(x) + ln(y)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(x * y)
Target: ln(x) + ln(y)
Strategy: expand_log
Steps (Aggressive Mode):
1. Log expansion  [Expandir logaritmos]
   Before: ln(x * y)
   Cambio local: ln(x * y) -> ln(x) + ln(y)
   After: ln(x) + ln(y)
   ℹ️ Requires: x > 0
   ℹ️ Requires: y > 0
Result: ln(x) + ln(y)
ℹ️ Requires:
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln(x · y)`
   - after: `ln(x) + ln(y)`
   - substeps: none

## expand_log_product_over_quotient (log_expand)

- Source: `ln((x*y)/z)`
- Target: `ln(x) + ln(y) - ln(z)`
- Result: `ln(x) + ln(y) - ln(z)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(x * y / z)
Target: ln(x) + ln(y) - ln(z)
Strategy: expand_log
Steps (Aggressive Mode):
1. Log expansion  [Expandir logaritmos]
   Before: ln(x * y / z)
   Cambio local: ln(x * y / z) -> ln(x) + ln(y) - ln(z)
   After: ln(x) + ln(y) - ln(z)
   ℹ️ Requires: x * y > 0
   ℹ️ Requires: z > 0
   ℹ️ Requires: x > 0
   ℹ️ Requires: y > 0
Result: ln(x) + ln(y) - ln(z)
ℹ️ Requires:
  • x > 0
  • y > 0
  • z > 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln((x · y)/z)`
   - after: `ln(x) + ln(y) - ln(z)`
   - substeps: none

## expand_log_product_with_root_cleanup (log_expand)

- Source: `ln(sqrt(x)*y)`
- Target: `ln(x)/2 + ln(y)`
- Result: `ln(y) + ln(x) / 2`
- Web step count: `2`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(y * sqrt(x))
Target: ln(y) + ln(x) / 2
Strategy: expand_log
Steps (Aggressive Mode):
1. Log expansion  [Expandir logaritmos]
   Before: ln(y * sqrt(x))
   Cambio local: ln(y * sqrt(x)) -> ln(y) + ln(sqrt(x))
   After: ln(y) + ln(sqrt(x))
   ℹ️ Requires: y > 0
   ℹ️ Requires: sqrt(x) > 0
2. log(b, x^y) = y * log(b, x)  [Evaluar logaritmos]
   Before: ln(y) + ln(sqrt(x))
   Cambio local: ln(y) + ln(sqrt(x)) -> ln(y) + ln(x) / 2
   After: ln(y) + ln(x) / 2
Result: ln(y) + ln(x) / 2
ℹ️ Requires:
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln(y · sqrt(x))`
   - after: `ln(y) + ln(sqrt(x))`
   - substeps: none
2. `Sacar un exponente fuera del logaritmo`
   - before: `ln(y) + ln(sqrt(x))`
   - after: `ln(y) + ln(x)/2`
   - substeps: none

## expand_log_quotient (log_expand)

- Source: `ln(x/y)`
- Target: `ln(x) - ln(y)`
- Result: `ln(x) - ln(y)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(x / y)
Target: ln(x) - ln(y)
Strategy: expand_log
Steps (Aggressive Mode):
1. Log expansion  [Expandir logaritmos]
   Before: ln(x / y)
   Cambio local: ln(x / y) -> ln(x) - ln(y)
   After: ln(x) - ln(y)
   ℹ️ Requires: x > 0
   ℹ️ Requires: y > 0
Result: ln(x) - ln(y)
ℹ️ Requires:
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln(x/y)`
   - after: `ln(x) - ln(y)`
   - substeps: none

## expand_ninth_power_plus_product (polynomial_product)

- Source: `(x^3+a^3)*(x^6-a^3*x^3+a^6)`
- Target: `x^9+a^9`
- Result: `a^9 + x^9`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a^3 + x^3) * (a^6 + x^6 - a^3 * x^3)
Target: a^9 + x^9
Strategy: expand
Steps (Aggressive Mode):
1. Expand the expression distributively  [Expandir la expresión]
   Before: (a^(3) + x^(3)) * (a^(6) + x^(6) - a^(3) * x^(3))
   Cambio local: (a^(3) + x^(3)) * (a^(6) + x^(6) - a^(3) * x^(3)) -> a^(9) + x^(9)
   After: a^9 + x^9
Result: a^(9) + x^(9)
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(a^3 + x^3) · (x^6 - a^3 · x^3 + a^6)`
   - after: `a^9 + x^9`
   - substeps:
     1. `Reconocer el patrón (a + b)(a^2 - ab + b^2)`
     2. `Aplicar (a + b)(a^2 - ab + b^2) = a^3 + b^3`

## expand_odd_half_power (radical_power)

- Source: `x^(3/2)`
- Target: `abs(x)*sqrt(x)`
- Result: `sqrt(x) * |x|`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: x^(3 / 2)
Target: sqrt(x) * |x|
Strategy: expand odd half power
Steps (Aggressive Mode):
1. Rewrite an odd half-integer power using a square root  [Reescribir potencia semientera impar]
   Before: x^(3 / 2)
   After: sqrt(x) * |x|
Result: sqrt(x) * |x|
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Extraer potencia par de la raíz`
   - before: `sqrt(x^3)`
   - after: `sqrt(x) · |x|`
   - substeps:
     1. `Separar el radicando en una potencia par y un factor`
     2. `Como x ≥ 0, sacar x fuera de la raíz`

## expand_odd_half_power_after_simplify (radical_power)

- Source: `sqrt(x^3)`
- Target: `abs(x)*sqrt(x)`
- Result: `sqrt(x) * |x|`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sqrt(x^3)
Target: sqrt(x) * |x|
Strategy: expand odd half power
Steps (Aggressive Mode):
1. Rewrite an odd half-integer power using a square root  [Reescribir potencia semientera impar]
   Before: sqrt(x^(3))
   After: sqrt(x) * |x|
Result: sqrt(x) * |x|
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Extraer potencia par de la raíz`
   - before: `sqrt(x^3)`
   - after: `sqrt(x) · |x|`
   - substeps:
     1. `Separar el radicando en una potencia par y un factor`
     2. `Como x ≥ 0, sacar x fuera de la raíz`

## expand_odd_half_power_after_simplify_with_passthrough (radical_power)

- Source: `sqrt(x^3)+a`
- Target: `abs(x)*sqrt(x)+a`
- Result: `sqrt(x) * |x| + a`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sqrt(x^3) + a
Target: sqrt(x) * |x| + a
Strategy: expand odd half power
Steps (Aggressive Mode):
1. Rewrite an odd half-integer power using a square root  [Reescribir potencia semientera impar]
   Before: sqrt(x^(3)) + a
   After: sqrt(x) * |x| + a
Result: sqrt(x) * |x| + a
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Extraer potencia par de la raíz`
   - before: `sqrt(x^3) + a`
   - after: `sqrt(x) · |x| + a`
   - substeps:
     1. `Separar el radicando en una potencia par y un factor`
     2. `Como x ≥ 0, sacar x fuera de la raíz`

## expand_recursive_hyperbolic_cosh_sum (expand)

- Source: `cosh(6*x)`
- Target: `cosh(5*x)*cosh(x)+sinh(5*x)*sinh(x)`
- Result: `sinh(x) * sinh(5 * x) + cosh(x) * cosh(5 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cosh(6 * x)
Target: sinh(x) * sinh(5 * x) + cosh(x) * cosh(5 * x)
Strategy: expand
Steps (Aggressive Mode):
1. Recognize cosh(u)·cosh(v) ± sinh(u)·sinh(v) as cosh(u ± v)  [Aplicar identidad hiperbólica de suma/diferencia de ángulos]
   Before: cosh(6 * x)
   Cambio local: cosh(6 * x) -> sinh(x) * sinh(5 * x) + cosh(x) * cosh(5 * x)
   After: sinh(x) * sinh(5 * x) + cosh(x) * cosh(5 * x)
Result: sinh(x) * sinh(5 * x) + cosh(x) * cosh(5 * x)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de suma/diferencia de ángulos`
   - before: `cosh(6 · x)`
   - after: `sinh(x) · sinh(5 · x) + cosh(x) · cosh(5 · x)`
   - substeps:
     1. `Usar cosh(5u+u) = cosh(5u) · cosh(u) + sinh(5u) · sinh(u), con u = x`

## expand_recursive_hyperbolic_sinh_sum (expand)

- Source: `sinh(6*x)`
- Target: `sinh(5*x)*cosh(x)+cosh(5*x)*sinh(x)`
- Result: `sinh(x) * cosh(5 * x) + sinh(5 * x) * cosh(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sinh(6 * x)
Target: sinh(x) * cosh(5 * x) + sinh(5 * x) * cosh(x)
Strategy: expand
Steps (Aggressive Mode):
1. Recognize sinh(u)·cosh(v) ± cosh(u)·sinh(v) as sinh(u ± v)  [Aplicar identidad hiperbólica de suma/diferencia de ángulos]
   Before: sinh(6 * x)
   Cambio local: sinh(6 * x) -> sinh(x) * cosh(5 * x) + sinh(5 * x) * cosh(x)
   After: sinh(x) * cosh(5 * x) + sinh(5 * x) * cosh(x)
Result: sinh(x) * cosh(5 * x) + sinh(5 * x) * cosh(x)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de suma/diferencia de ángulos`
   - before: `sinh(6 · x)`
   - after: `sinh(x) · cosh(5 · x) + sinh(5 · x) · cosh(x)`
   - substeps:
     1. `Usar sinh(5u+u) = sinh(5u) · cosh(u) + cosh(5u) · sinh(u), con u = x`

## expand_sixth_power_minus_product (polynomial_product)

- Source: `(x^2-1)*(x^4+x^2+1)`
- Target: `x^6-1`
- Result: `x^6 - 1`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: (x^4 + x^2 + 1) * (x^2 - 1)
Target: x^6 - 1
Strategy: expand
Steps (Aggressive Mode):
1. Expand and combine polynomial product  [Expandir y reagrupar un producto polinómico]
   Before: (x^(2) - 1) * (x^(4) + x^(2) + 1)
   After: x^6 - 1
Result: x^(6) - 1
```

### Web / JSON Steps

1. `Expandir y reagrupar un producto polinómico`
   - before: `(x^2 - 1) · (x^4 + x^2 + 1)`
   - after: `x^6 - 1`
   - substeps:
     1. `Distribuir cada término del producto`
     2. `Agrupar los términos del mismo grado`
     3. `Los términos intermedios se cancelan por parejas`

## expand_sixth_power_plus_product (polynomial_product)

- Source: `(x^2+1)*(x^4-x^2+1)`
- Target: `x^6+1`
- Result: `x^6 + 1`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: (x^2 + 1) * (x^4 - x^2 + 1)
Target: x^6 + 1
Strategy: expand
Steps (Aggressive Mode):
1. Expand and combine polynomial product  [Expandir y reagrupar un producto polinómico]
   Before: (x^(2) + 1) * (x^(4) + 1 - x^(2))
   After: x^6 + 1
Result: x^(6) + 1
```

### Web / JSON Steps

1. `Expandir y reagrupar un producto polinómico`
   - before: `(x^2 + 1) · (x^4 + 1 - x^2)`
   - after: `x^6 + 1`
   - substeps:
     1. `Distribuir cada término del producto`
     2. `Agrupar los términos del mismo grado`
     3. `Los términos intermedios se cancelan por parejas`

## expand_sophie_germain (expand)

- Source: `(x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2)`
- Target: `x^4 + 4*y^4`
- Result: `x^4 + 4 * y^4`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (x^2 + 2 * y^2 - 2 * x * y) * (x^2 + 2 * y^2 + 2 * x * y)
Target: x^4 + 4 * y^4
Strategy: expand
Steps (Aggressive Mode):
1. Expand the Sophie Germain identity  [Expandir la expresión]
   Before: (x^(2) + 2 * y^(2) - 2 * x * y) * (x^(2) + 2 * y^(2) + 2 * x * y)
   Cambio local: (x^(2) + 2 * y^(2) - 2 * x * y) * (x^(2) + 2 * y^(2) + 2 * x * y) -> x^(4) + 4 * y^(4)
   After: x^4 + 4 * y^4
Result: x^(4) + 4 * y^(4)
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(x^2 - 2 · x · y + 2 · y^2) · (x^2 + 2 · y^2 + 2 · x · y)`
   - after: `x^4 + 4 · y^4`
   - substeps:
     1. `Reconocer el patrón de Sophie Germain`
     2. `Aplicar la identidad de Sophie Germain`

## expand_sum_cubes (expand)

- Source: `(a+b)*(a^2-a*b+b^2)`
- Target: `a^3+b^3`
- Result: `a^3 + b^3`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a + b) * (a^2 + b^2 - a * b)
Target: a^3 + b^3
Strategy: expand
Steps (Aggressive Mode):
1. Expand the expression distributively  [Expandir la expresión]
   Before: (a + b) * (a^(2) + b^(2) - a * b)
   Cambio local: (a + b) * (a^(2) + b^(2) - a * b) -> a^(3) + b^(3)
   After: a^3 + b^3
Result: a^(3) + b^(3)
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(a + b) · (a^2 - a · b + b^2)`
   - after: `a^3 + b^3`
   - substeps:
     1. `Reconocer el patrón (a + b)(a^2 - ab + b^2)`
     2. `Aplicar (a + b)(a^2 - ab + b^2) = a^3 + b^3`

## expand_symbolic_binomial (expand)

- Source: `(a + b)^2`
- Target: `a^2 + 2*a*b + b^2`
- Result: `a^2 + b^2 + 2 * a * b`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (a + b)^2
Target: a^2 + b^2 + 2 * a * b
Strategy: expand
Steps (Aggressive Mode):
1. Expand binomial power ^2  [Expandir binomio]
   Before: (a + b)^(2)
   Cambio local: (a + b)^(2) -> b^(2) + 2 * a * b + a^(2)
   After: a^2 + b^2 + 2 * a * b
Result: a^(2) + b^(2) + 2 * a * b
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `(a + b)^2`
   - after: `a^2 + b^2 + 2 · a · b`
   - substeps: none

## expand_symbolic_binomial_cube (expand)

- Source: `(a + b)^3`
- Target: `a^3 + 3*a^2*b + 3*a*b^2 + b^3`
- Result: `a^3 + b^3 + 3 * a * b^2 + 3 * b * a^2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (a + b)^3
Target: a^3 + b^3 + 3 * a * b^2 + 3 * b * a^2
Strategy: expand
Steps (Aggressive Mode):
1. Expand binomial power ^3  [Expandir binomio]
   Before: (a + b)^(3)
   Cambio local: (a + b)^(3) -> b^(3) + 3 * a * b^(2) + 3 * a^(2) * b + a^(3)
   After: a^3 + b^3 + 3 * a * b^2 + 3 * b * a^2
Result: a^(3) + b^(3) + 3 * a * b^(2) + 3 * b * a^(2)
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `(a + b)^3`
   - after: `a^3 + b^3 + 3 · a · b^2 + 3 · b · a^2`
   - substeps: none

## expand_symbolic_binomial_cube_minus (expand)

- Source: `(a - b)^3`
- Target: `a^3 - 3*a^2*b + 3*a*b^2 - b^3`
- Result: `a^3 + 3 * a * b^2 - 3 * b * a^2 - b^3`
- Web step count: `2`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a - b)^3
Target: a^3 + 3 * a * b^2 - 3 * b * a^2 - b^3
Strategy: expand
Steps (Aggressive Mode):
1. Expand binomial power ^3  [Expandir binomio]
   Before: (a - b)^(3)
   Cambio local: (a - b)^(3) -> (-b)^(3) + 3 * a * (-b)^(2) + 3 * -a^(2) * b + a^(3)
   After: (-b)^(3) + 3 * a * (-b)^(2) + 3 * -a^(2) * b + a^(3)
2. (-x)^odd -> -(x^odd)  [Simplificar potencia con base negativa]
   Before: (-b)^(3) + 3 * a * (-b)^(2) + 3 * -a^(2) * b + a^(3)
   Cambio local: (-b)^(3) -> -b^(3)
   After: 3 * a * (-b)^(2) + 3 * -a^(2) * b + a^(3) - b^(3)
3. (-x)^even -> x^even  [Simplificar potencia con base negativa]
   Before: 3 * a * (-b)^(2) + 3 * -a^(2) * b + a^(3) - b^(3)
   Cambio local: (-b)^(2) -> b^(2)
   After: a^3 + 3 * a * b^2 - 3 * b * a^2 - b^3
Result: a^(3) + 3 * a * b^(2) - 3 * b * a^(2) - b^(3)
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `(a - b)^3`
   - after: `(-b)^3 + 3 · a · (-b)^2 + a^3 - 3 · b · a^2`
   - substeps: none
2. `Simplificar potencia con base negativa`
   - before: `(-b)^3 + 3 · a · (-b)^2 + a^3 - 3 · b · a^2`
   - after: `a^3 - 3 · b · a^2 + 3 · a · b^2 - b^3`
   - substeps:
     1. `Usar que una potencia impar conserva el signo negativo`
     2. `Usar que una potencia par elimina el signo`

## expand_symbolic_binomial_minus (expand)

- Source: `(a - b)^2`
- Target: `a^2 - 2*a*b + b^2`
- Result: `a^2 + b^2 - 2 * a * b`
- Web step count: `2`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a - b)^2
Target: a^2 + b^2 - 2 * a * b
Strategy: expand
Steps (Aggressive Mode):
1. Expand binomial power ^2  [Expandir binomio]
   Before: (a - b)^(2)
   Cambio local: (a - b)^(2) -> (-b)^(2) + 2 * -a * b + a^(2)
   After: (-b)^(2) + 2 * -a * b + a^(2)
2. (-x)^even -> x^even  [Simplificar potencia con base negativa]
   Before: (-b)^(2) + 2 * -a * b + a^(2)
   Cambio local: (-b)^(2) -> b^(2)
   After: a^2 + b^2 - 2 * a * b
Result: a^(2) + b^(2) - 2 * a * b
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `(a - b)^2`
   - after: `(-b)^2 + a^2 - 2 · a · b`
   - substeps: none
2. `Simplificar potencia con base negativa`
   - before: `(-b)^2 + a^2 - 2 · a · b`
   - after: `a^2 - 2 · a · b + b^2`
   - substeps:
     1. `Usar que una potencia par elimina el signo`

## expand_symbolic_cube_difference_product (polynomial_product)

- Source: `(x-a)*(x^2+a*x+a^2)`
- Target: `x^3-a^3`
- Result: `x^3 - a^3`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a^2 + x^2 + a * x) * (x - a)
Target: x^3 - a^3
Strategy: expand
Steps (Aggressive Mode):
1. Expand the expression distributively  [Expandir la expresión]
   Before: (a^(2) + x^(2) + a * x) * (x - a)
   Cambio local: (a^(2) + x^(2) + a * x) * (x - a) -> x^(3) - a^(3)
   After: x^3 - a^3
Result: x^(3) - a^(3)
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(a^2 + x^2 + a · x) · (x - a)`
   - after: `x^3 - a^3`
   - substeps:
     1. `Reconocer el patrón (a - b)(a^2 + ab + b^2)`
     2. `Aplicar (a - b)(a^2 + ab + b^2) = a^3 - b^3`

## expand_symbolic_cube_sum_product (polynomial_product)

- Source: `(x+a)*(x^2-a*x+a^2)`
- Target: `x^3+a^3`
- Result: `a^3 + x^3`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a + x) * (a^2 + x^2 - a * x)
Target: a^3 + x^3
Strategy: expand
Steps (Aggressive Mode):
1. Expand the expression distributively  [Expandir la expresión]
   Before: (a + x) * (a^(2) + x^(2) - a * x)
   Cambio local: (a + x) * (a^(2) + x^(2) - a * x) -> a^(3) + x^(3)
   After: a^3 + x^3
Result: a^(3) + x^(3)
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(a + x) · (x^2 - a · x + a^2)`
   - after: `a^3 + x^3`
   - substeps:
     1. `Reconocer el patrón (a + b)(a^2 - ab + b^2)`
     2. `Aplicar (a + b)(a^2 - ab + b^2) = a^3 + b^3`

## expand_symbolic_signed_trinomial_square (expand)

- Source: `(a - b + c)^2`
- Target: `a^2 + b^2 + c^2 - 2*a*b + 2*a*c - 2*b*c`
- Result: `a^2 + b^2 + c^2 - 2 * a * b + 2 * a * c - 2 * b * c`
- Web step count: `3`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (a - b + c)^2
Target: a^2 + b^2 + c^2 - 2 * a * b + 2 * a * c - 2 * b * c
Strategy: expand
Steps (Aggressive Mode):
1. For even exponent: (a-b)² = (b-a)², normalize for cancellation  [Invertir una resta dentro de una potencia par]
   Before: (a + c - b)^(2)
   Cambio local: a + c - b -> b - (a + c)
   After: (b - (a + c))^(2)
2. For even exponent: (a-b)² = (b-a)², normalize for cancellation  [Invertir una resta dentro de una potencia par]
   Before: (b - a - c)^(2)
   Cambio local: b - a - c -> c - (b - a)
   After: (c - (b - a))^(2)
3. Expand (3-term sum)^2  [Expandir binomio]
   Before: (a + c - b)^(2)
   Cambio local: (a + c - b)^(2) -> a^(2) + b^(2) + c^(2) - 2 * a * b - 2 * b * c + 2 * a * c
   After: a^2 + b^2 + c^2 - 2 * a * b + 2 * a * c - 2 * b * c
Result: a^(2) + b^(2) + c^(2) - 2 * a * b + 2 * a * c - 2 * b * c
```

### Web / JSON Steps

1. `Invertir una resta dentro de una potencia par`
   - before: `(a + c - b)^2`
   - after: `(b - (a + c))^2`
   - substeps: none
2. `Invertir una resta dentro de una potencia par`
   - before: `(b - a - c)^2`
   - after: `(c - (b - a))^2`
   - substeps: none
3. `Expandir binomio`
   - before: `(a + c - b)^2`
   - after: `a^2 + b^2 + c^2 - 2 · a · b + 2 · a · c - 2 · b · c`
   - substeps: none

## expand_symbolic_sixth_power_minus_product (polynomial_product)

- Source: `(x^2-a^2)*(x^4+a^2*x^2+a^4)`
- Target: `x^6-a^6`
- Result: `x^6 - a^6`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a^4 + x^4 + a^2 * x^2) * (x^2 - a^2)
Target: x^6 - a^6
Strategy: expand
Steps (Aggressive Mode):
1. Expand the expression distributively  [Expandir la expresión]
   Before: (a^(4) + x^(4) + a^(2) * x^(2)) * (x^(2) - a^(2))
   Cambio local: (a^(4) + x^(4) + a^(2) * x^(2)) * (x^(2) - a^(2)) -> x^(6) - a^(6)
   After: x^6 - a^6
Result: x^(6) - a^(6)
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(a^4 + x^4 + a^2 · x^2) · (x^2 - a^2)`
   - after: `x^6 - a^6`
   - substeps:
     1. `Reconocer el patrón (a^2 - b^2)(a^4 + a^2b^2 + b^4)`
     2. `Aplicar (a^2 - b^2)(a^4 + a^2b^2 + b^4) = a^6 - b^6`

## expand_symbolic_sixth_power_plus_product (polynomial_product)

- Source: `(x^2+a^2)*(x^4-a^2*x^2+a^4)`
- Target: `x^6+a^6`
- Result: `a^6 + x^6`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a^2 + x^2) * (a^4 + x^4 - a^2 * x^2)
Target: a^6 + x^6
Strategy: expand
Steps (Aggressive Mode):
1. Expand the expression distributively  [Expandir la expresión]
   Before: (a^(2) + x^(2)) * (a^(4) + x^(4) - a^(2) * x^(2))
   Cambio local: (a^(2) + x^(2)) * (a^(4) + x^(4) - a^(2) * x^(2)) -> a^(6) + x^(6)
   After: a^6 + x^6
Result: a^(6) + x^(6)
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(a^2 + x^2) · (x^4 - a^2 · x^2 + a^4)`
   - after: `a^6 + x^6`
   - substeps:
     1. `Reconocer el patrón (a^2 + b^2)(a^4 - a^2b^2 + b^4)`
     2. `Aplicar (a^2 + b^2)(a^4 - a^2b^2 + b^4) = a^6 + b^6`

## expand_symbolic_trinomial_cube (expand)

- Source: `(a + b + c)^3`
- Target: `a^3 + b^3 + c^3 + 3*a^2*b + 3*a^2*c + 3*a*b^2 + 6*a*b*c + 3*a*c^2 + 3*b^2*c + 3*b*c^2`
- Result: `a^3 + b^3 + c^3 + 3 * a * b^2 + 3 * a * c^2 + 3 * b * a^2 + 3 * b * c^2 + 3 * c * a^2 + 3 * c * b^2 + 6 * a * b * c`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (a + b + c)^3
Target: a^3 + b^3 + c^3 + 3 * a * b^2 + 3 * a * c^2 + 3 * b * a^2 + 3 * b * c^2 + 3 * c * a^2 + 3 * c * b^2 + 6 * a * b * c
Strategy: expand
Steps (Aggressive Mode):
1. Expand (3-term sum)^3  [Expandir binomio]
   Before: (a + b + c)^(3)
   Cambio local: (a + b + c)^(3) -> a^(3) + b^(3) + c^(3) + 3 * a * b^(2) + 3 * a * c^(2) + 3 * b * a^(2) + 3 * b * c^(2) + 3 * c * a^(2) + 3 * c * b^(2) + 6 * a * b * c
   After: a^3 + b^3 + c^3 + 3 * a * b^2 + 3 * a * c^2 + 3 * b * a^2 + 3 * b * c^2 + 3 * c * a^2 + 3 * c * b^2 + 6 * a * b * c
Result: a^(3) + b^(3) + c^(3) + 3 * a * b^(2) + 3 * a * c^(2) + 3 * b * a^(2) + 3 * b * c^(2) + 3 * c * a^(2) + 3 * c * b^(2) + 6 * a * b * c
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `(a + b + c)^3`
   - after: `a^3 + b^3 + c^3 + 3 · a · b^2 + 3 · a · c^2 + 3 · b · a^2 + 3 · b · c^2 + 3 · c · a^2 + 3 · c · b^2 + 6 · a · b · c`
   - substeps: none

## expand_symbolic_trinomial_square (expand)

- Source: `(a + b + c)^2`
- Target: `a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c`
- Result: `a^2 + b^2 + c^2 + 2 * a * b + 2 * a * c + 2 * b * c`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (a + b + c)^2
Target: a^2 + b^2 + c^2 + 2 * a * b + 2 * a * c + 2 * b * c
Strategy: expand
Steps (Aggressive Mode):
1. Expand (3-term sum)^2  [Expandir binomio]
   Before: (a + b + c)^(2)
   Cambio local: (a + b + c)^(2) -> a^(2) + b^(2) + c^(2) + 2 * a * b + 2 * a * c + 2 * b * c
   After: a^2 + b^2 + c^2 + 2 * a * b + 2 * a * c + 2 * b * c
Result: a^(2) + b^(2) + c^(2) + 2 * a * b + 2 * a * c + 2 * b * c
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `(a + b + c)^2`
   - after: `a^2 + b^2 + c^2 + 2 · a · b + 2 · a · c + 2 · b · c`
   - substeps: none

## expand_then_cancel_to_square (expand)

- Source: `(a+b)^2 - a^2 - 2*a*b`
- Target: `b^2`
- Result: `b^2`
- Web step count: `2`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a + b)^2 - a^2 - 2 * a * b
Target: b^2
Strategy: expand
Steps (Aggressive Mode):
1. Expand binomial power ^2  [Expandir binomio]
   Before: (a + b)^(2) - a^(2) - 2 * a * b
   Cambio local: (a + b)^(2) -> b^(2) + 2 * a * b + a^(2)
   After: b^(2) + 2 * a * b + a^(2) - a^(2) - 2 * a * b
2. Cancel exact additive pairs  [Cancelar términos opuestos]
   Before: b^(2) + 2 * a * b + a^(2) - a^(2) - 2 * a * b
   Cambio local: a^(2) + b^(2) + 2 * a * b - a^(2) -> b^(2) + 2 * a * b
   After: b^(2) + 2 * a * b - 2 * a * b
3. Cancel exact additive pairs  [Cancelar términos opuestos]
   Before: b^(2) + 2 * a * b - 2 * a * b
   Cambio local: b^(2) + 2 * a * b - 2 * a * b -> b^(2)
   After: b^2
Result: b^(2)
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `(a + b)^2 - a^2 - 2 · a · b`
   - after: `a^2 + b^2 + 2 · a · b - a^2 - 2 · a · b`
   - substeps: none
2. `Cancelar términos opuestos`
   - before: `a^2 + b^2 + 2 · a · b - a^2 - 2 · a · b`
   - after: `b^2`
   - substeps:
     1. `Cancelar términos opuestos exactos`
     2. `Cancelar términos opuestos exactos`

## expand_trig_after_simplify (trig_expand)

- Source: `sin(x + x)`
- Target: `2*sin(x)*cos(x)`
- Result: `2 * sin(x) * cos(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(x + x)
Target: 2 * sin(x) * cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand double-angle sine  [Expandir ángulo doble]
   Before: sin(x + x)
   Cambio local: sin(x + x) -> 2 * sin(x) * cos(x)
   After: 2 * sin(x) * cos(x)
Result: 2 * sin(x) * cos(x)
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `sin(x + x)`
   - after: `2 · sin(x) · cos(x)`
   - substeps: none

## expand_trig_angle_diff_cosine (trig_expand)

- Source: `cos(x-y)`
- Target: `cos(x)*cos(y)+sin(x)*sin(y)`
- Result: `sin(x) * sin(y) + cos(x) * cos(y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x - y)
Target: sin(x) * sin(y) + cos(x) * cos(y)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand or contract an angle sum/difference trig identity  [Aplicar suma/diferencia de ángulos]
   Before: cos(x - y)
   Cambio local: cos(x - y) -> sin(x) * sin(y) + cos(x) * cos(y)
   After: sin(x) * sin(y) + cos(x) * cos(y)
Result: sin(x) * sin(y) + cos(x) * cos(y)
```

### Web / JSON Steps

1. `Aplicar suma/diferencia de ángulos`
   - before: `cos(x - y)`
   - after: `sin(x) · sin(y) + cos(x) · cos(y)`
   - substeps:
     1. `Usar cos(A-B) = cos(A) · cos(B) + sin(A) · sin(B)`

## expand_trig_angle_diff_sine (trig_expand)

- Source: `sin(x-y)`
- Target: `sin(x)*cos(y)-cos(x)*sin(y)`
- Result: `sin(x) * cos(y) - sin(y) * cos(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x - y)
Target: sin(x) * cos(y) - sin(y) * cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand or contract an angle sum/difference trig identity  [Aplicar suma/diferencia de ángulos]
   Before: sin(x - y)
   Cambio local: sin(x - y) -> sin(x) * cos(y) - sin(y) * cos(x)
   After: sin(x) * cos(y) - sin(y) * cos(x)
Result: sin(x) * cos(y) - sin(y) * cos(x)
```

### Web / JSON Steps

1. `Aplicar suma/diferencia de ángulos`
   - before: `sin(x - y)`
   - after: `sin(x) · cos(y) - sin(y) · cos(x)`
   - substeps:
     1. `Usar sin(A-B) = sin(A) · cos(B) - cos(A) · sin(B)`

## expand_trig_angle_sum_cosine (trig_expand)

- Source: `cos(x+y)`
- Target: `cos(x)*cos(y)-sin(x)*sin(y)`
- Result: `cos(x) * cos(y) - sin(x) * sin(y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x + y)
Target: cos(x) * cos(y) - sin(x) * sin(y)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand or contract an angle sum/difference trig identity  [Aplicar suma/diferencia de ángulos]
   Before: cos(x + y)
   Cambio local: cos(x + y) -> cos(x) * cos(y) - sin(x) * sin(y)
   After: cos(x) * cos(y) - sin(x) * sin(y)
Result: cos(x) * cos(y) - sin(x) * sin(y)
```

### Web / JSON Steps

1. `Aplicar suma/diferencia de ángulos`
   - before: `cos(x + y)`
   - after: `cos(x) · cos(y) - sin(x) · sin(y)`
   - substeps:
     1. `Usar cos(A+B) = cos(A) · cos(B) - sin(A) · sin(B)`

## expand_trig_angle_sum_sine (trig_expand)

- Source: `sin(x+y)`
- Target: `sin(x)*cos(y)+cos(x)*sin(y)`
- Result: `sin(x) * cos(y) + sin(y) * cos(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x + y)
Target: sin(x) * cos(y) + sin(y) * cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand or contract an angle sum/difference trig identity  [Aplicar suma/diferencia de ángulos]
   Before: sin(x + y)
   Cambio local: sin(x + y) -> sin(x) * cos(y) + sin(y) * cos(x)
   After: sin(x) * cos(y) + sin(y) * cos(x)
Result: sin(x) * cos(y) + sin(y) * cos(x)
```

### Web / JSON Steps

1. `Aplicar suma/diferencia de ángulos`
   - before: `sin(x + y)`
   - after: `sin(x) · cos(y) + sin(y) · cos(x)`
   - substeps:
     1. `Usar sin(A+B) = sin(A) · cos(B) + cos(A) · sin(B)`

## expand_trig_cofunction_cosine_minus (trig_expand)

- Source: `cos(pi/2 - x)`
- Target: `sin(x)`
- Result: `sin(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cos(pi / 2 - x)
Target: sin(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Apply a sine/cosine cofunction identity  [Aplicar identidad de cofunción]
   Before: cos(pi / 2 - x)
   Cambio local: cos(pi / 2 - x) -> sin(x)
   After: sin(x)
Result: sin(x)
```

### Web / JSON Steps

1. `Aplicar identidad de cofunción`
   - before: `cos(pi/2 - x)`
   - after: `sin(x)`
   - substeps: none

## expand_trig_cofunction_sine_minus (trig_expand)

- Source: `sin(pi/2 - x)`
- Target: `cos(x)`
- Result: `cos(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(pi / 2 - x)
Target: cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Apply a sine/cosine cofunction identity  [Aplicar identidad de cofunción]
   Before: sin(pi / 2 - x)
   Cambio local: sin(pi / 2 - x) -> cos(x)
   After: cos(x)
Result: cos(x)
```

### Web / JSON Steps

1. `Aplicar identidad de cofunción`
   - before: `sin(pi/2 - x)`
   - after: `cos(x)`
   - substeps: none

## expand_trig_cosine_eighteenth_power_reduction (trig_expand)

- Source: `cos(x)^18`
- Target: `(24310+43758*cos(2*x)+31824*cos(4*x)+18564*cos(6*x)+8568*cos(8*x)+3060*cos(10*x)+816*cos(12*x)+153*cos(14*x)+18*cos(16*x)+cos(18*x))/131072`
- Result: `(cos(18 * x) + 18 * cos(16 * x) + 153 * cos(14 * x) + 816 * cos(12 * x) + 3060 * cos(10 * x) + 8568 * cos(8 * x) + 18564 * cos(6 * x) + 31824 * cos(4 * x) + 43758 * cos(2 * x) + 24310) / 131072`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x)^18
Target: (cos(18 * x) + 18 * cos(16 * x) + 153 * cos(14 * x) + 816 * cos(12 * x) + 3060 * cos(10 * x) + 8568 * cos(8 * x) + 18564 * cos(6 * x) + 31824 * cos(4 * x) + 43758 * cos(2 * x) + 24310) / 131072
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce cos¹⁸(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: cos(x)^(18)
   Cambio local: cos(x)^(18) -> (cos(18 * x) + 18 * cos(16 * x) + 153 * cos(14 * x) + 816 * cos(12 * x) + 3060 * cos(10 * x) + 8568 * cos(8 * x) + 18564 * cos(6 * x) + 31824 * cos(4 * x) + 43758 * cos(2 * x) + 24310) / 131072
   After: (cos(18 * x) + 18 * cos(16 * x) + 153 * cos(14 * x) + 816 * cos(12 * x) + 3060 * cos(10 * x) + 8568 * cos(8 * x) + 18564 * cos(6 * x) + 31824 * cos(4 * x) + 43758 * cos(2 * x) + 24310) / 131072
Result: (cos(18 * x) + 18 * cos(16 * x) + 153 * cos(14 * x) + 816 * cos(12 * x) + 3060 * cos(10 * x) + 8568 * cos(8 * x) + 18564 * cos(6 * x) + 31824 * cos(4 * x) + 43758 * cos(2 * x) + 24310) / 131072
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `cos(x)^18`
   - after: `(cos(18 · x) + 18 · cos(16 · x) + 153 · cos(14 · x) + 816 · cos(12 · x) + 3060 · cos(10 · x) + 8568 · cos(8 · x) + 18564 · cos(6 · x) + 31824 · cos(4 · x) + 43758 · cos(2 · x) + 24310)/131072`
   - substeps:
     1. `Usar cos²(u) = (1 + cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_cosine_eighth_power_reduction (trig_expand)

- Source: `cos(x)^8`
- Target: `(35+56*cos(2*x)+28*cos(4*x)+8*cos(6*x)+cos(8*x))/128`
- Result: `(cos(8 * x) + 8 * cos(6 * x) + 28 * cos(4 * x) + 56 * cos(2 * x) + 35) / 128`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x)^8
Target: (cos(8 * x) + 8 * cos(6 * x) + 28 * cos(4 * x) + 56 * cos(2 * x) + 35) / 128
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce cos⁸(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: cos(x)^(8)
   Cambio local: cos(x)^(8) -> (cos(8 * x) + 8 * cos(6 * x) + 28 * cos(4 * x) + 56 * cos(2 * x) + 35) / 128
   After: (cos(8 * x) + 8 * cos(6 * x) + 28 * cos(4 * x) + 56 * cos(2 * x) + 35) / 128
Result: (cos(8 * x) + 8 * cos(6 * x) + 28 * cos(4 * x) + 56 * cos(2 * x) + 35) / 128
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `cos(x)^8`
   - after: `(cos(8 · x) + 8 · cos(6 · x) + 28 · cos(4 · x) + 56 · cos(2 · x) + 35)/128`
   - substeps:
     1. `Usar cos²(u) = (1 + cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_cosine_fourteenth_power_reduction (trig_expand)

- Source: `cos(x)^14`
- Target: `(1716+3003*cos(2*x)+2002*cos(4*x)+1001*cos(6*x)+364*cos(8*x)+91*cos(10*x)+14*cos(12*x)+cos(14*x))/8192`
- Result: `(cos(14 * x) + 14 * cos(12 * x) + 91 * cos(10 * x) + 364 * cos(8 * x) + 1001 * cos(6 * x) + 2002 * cos(4 * x) + 3003 * cos(2 * x) + 1716) / 8192`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x)^14
Target: (cos(14 * x) + 14 * cos(12 * x) + 91 * cos(10 * x) + 364 * cos(8 * x) + 1001 * cos(6 * x) + 2002 * cos(4 * x) + 3003 * cos(2 * x) + 1716) / 8192
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce cos¹⁴(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: cos(x)^(14)
   Cambio local: cos(x)^(14) -> (cos(14 * x) + 14 * cos(12 * x) + 91 * cos(10 * x) + 364 * cos(8 * x) + 1001 * cos(6 * x) + 2002 * cos(4 * x) + 3003 * cos(2 * x) + 1716) / 8192
   After: (cos(14 * x) + 14 * cos(12 * x) + 91 * cos(10 * x) + 364 * cos(8 * x) + 1001 * cos(6 * x) + 2002 * cos(4 * x) + 3003 * cos(2 * x) + 1716) / 8192
Result: (cos(14 * x) + 14 * cos(12 * x) + 91 * cos(10 * x) + 364 * cos(8 * x) + 1001 * cos(6 * x) + 2002 * cos(4 * x) + 3003 * cos(2 * x) + 1716) / 8192
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `cos(x)^14`
   - after: `(cos(14 · x) + 14 · cos(12 · x) + 91 · cos(10 · x) + 364 · cos(8 · x) + 1001 · cos(6 · x) + 2002 · cos(4 · x) + 3003 · cos(2 · x) + 1716)/8192`
   - substeps:
     1. `Usar cos²(u) = (1 + cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_cosine_fourth_power_reduction (simplify)

- Source: `cos(x)^4`
- Target: `(3+4*cos(2*x)+cos(4*x))/8`
- Result: `(cos(4 * x) + 4 * cos(2 * x) + 3) / 8`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x)^4
Target: (cos(4 * x) + 4 * cos(2 * x) + 3) / 8
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce cos⁴(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: cos(x)^(4)
   Cambio local: cos(x)^(4) -> (cos(4 * x) + 4 * cos(2 * x) + 3) / 8
   After: (cos(4 * x) + 4 * cos(2 * x) + 3) / 8
Result: (cos(4 * x) + 4 * cos(2 * x) + 3) / 8
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `cos(x)^4`
   - after: `(cos(4 · x) + 4 · cos(2 · x) + 3)/8`
   - substeps:
     1. `Usar cos²(u) = (1 + cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_cosine_sixteenth_power_reduction (trig_expand)

- Source: `cos(x)^16`
- Target: `(6435+11440*cos(2*x)+8008*cos(4*x)+4368*cos(6*x)+1820*cos(8*x)+560*cos(10*x)+120*cos(12*x)+16*cos(14*x)+cos(16*x))/32768`
- Result: `(cos(16 * x) + 16 * cos(14 * x) + 120 * cos(12 * x) + 560 * cos(10 * x) + 1820 * cos(8 * x) + 4368 * cos(6 * x) + 8008 * cos(4 * x) + 11440 * cos(2 * x) + 6435) / 32768`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x)^16
Target: (cos(16 * x) + 16 * cos(14 * x) + 120 * cos(12 * x) + 560 * cos(10 * x) + 1820 * cos(8 * x) + 4368 * cos(6 * x) + 8008 * cos(4 * x) + 11440 * cos(2 * x) + 6435) / 32768
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce cos¹⁶(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: cos(x)^(16)
   Cambio local: cos(x)^(16) -> (cos(16 * x) + 16 * cos(14 * x) + 120 * cos(12 * x) + 560 * cos(10 * x) + 1820 * cos(8 * x) + 4368 * cos(6 * x) + 8008 * cos(4 * x) + 11440 * cos(2 * x) + 6435) / 32768
   After: (cos(16 * x) + 16 * cos(14 * x) + 120 * cos(12 * x) + 560 * cos(10 * x) + 1820 * cos(8 * x) + 4368 * cos(6 * x) + 8008 * cos(4 * x) + 11440 * cos(2 * x) + 6435) / 32768
Result: (cos(16 * x) + 16 * cos(14 * x) + 120 * cos(12 * x) + 560 * cos(10 * x) + 1820 * cos(8 * x) + 4368 * cos(6 * x) + 8008 * cos(4 * x) + 11440 * cos(2 * x) + 6435) / 32768
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `cos(x)^16`
   - after: `(cos(16 · x) + 16 · cos(14 · x) + 120 · cos(12 · x) + 560 · cos(10 · x) + 1820 · cos(8 · x) + 4368 · cos(6 · x) + 8008 · cos(4 · x) + 11440 · cos(2 · x) + 6435)/32768`
   - substeps:
     1. `Usar cos²(u) = (1 + cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_cosine_sixth_power_reduction (trig_expand)

- Source: `cos(x)^6`
- Target: `(10+15*cos(2*x)+6*cos(4*x)+cos(6*x))/32`
- Result: `(cos(6 * x) + 6 * cos(4 * x) + 15 * cos(2 * x) + 10) / 32`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x)^6
Target: (cos(6 * x) + 6 * cos(4 * x) + 15 * cos(2 * x) + 10) / 32
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce cos⁶(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: cos(x)^(6)
   Cambio local: cos(x)^(6) -> (cos(6 * x) + 6 * cos(4 * x) + 15 * cos(2 * x) + 10) / 32
   After: (cos(6 * x) + 6 * cos(4 * x) + 15 * cos(2 * x) + 10) / 32
Result: (cos(6 * x) + 6 * cos(4 * x) + 15 * cos(2 * x) + 10) / 32
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `cos(x)^6`
   - after: `(cos(6 · x) + 6 · cos(4 · x) + 15 · cos(2 · x) + 10)/32`
   - substeps:
     1. `Usar cos²(u) = (1 + cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_cosine_tenth_power_reduction (trig_expand)

- Source: `cos(x)^10`
- Target: `(126+210*cos(2*x)+120*cos(4*x)+45*cos(6*x)+10*cos(8*x)+cos(10*x))/512`
- Result: `(cos(10 * x) + 10 * cos(8 * x) + 45 * cos(6 * x) + 120 * cos(4 * x) + 210 * cos(2 * x) + 126) / 512`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x)^10
Target: (cos(10 * x) + 10 * cos(8 * x) + 45 * cos(6 * x) + 120 * cos(4 * x) + 210 * cos(2 * x) + 126) / 512
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce cos¹⁰(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: cos(x)^(10)
   Cambio local: cos(x)^(10) -> (cos(10 * x) + 10 * cos(8 * x) + 45 * cos(6 * x) + 120 * cos(4 * x) + 210 * cos(2 * x) + 126) / 512
   After: (cos(10 * x) + 10 * cos(8 * x) + 45 * cos(6 * x) + 120 * cos(4 * x) + 210 * cos(2 * x) + 126) / 512
Result: (cos(10 * x) + 10 * cos(8 * x) + 45 * cos(6 * x) + 120 * cos(4 * x) + 210 * cos(2 * x) + 126) / 512
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `cos(x)^10`
   - after: `(cos(10 · x) + 10 · cos(8 · x) + 45 · cos(6 · x) + 120 · cos(4 · x) + 210 · cos(2 · x) + 126)/512`
   - substeps:
     1. `Usar cos²(u) = (1 + cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_cosine_twelfth_power_reduction (trig_expand)

- Source: `cos(x)^12`
- Target: `(462+792*cos(2*x)+495*cos(4*x)+220*cos(6*x)+66*cos(8*x)+12*cos(10*x)+cos(12*x))/2048`
- Result: `(cos(12 * x) + 12 * cos(10 * x) + 66 * cos(8 * x) + 220 * cos(6 * x) + 495 * cos(4 * x) + 792 * cos(2 * x) + 462) / 2048`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x)^12
Target: (cos(12 * x) + 12 * cos(10 * x) + 66 * cos(8 * x) + 220 * cos(6 * x) + 495 * cos(4 * x) + 792 * cos(2 * x) + 462) / 2048
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce cos¹²(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: cos(x)^(12)
   Cambio local: cos(x)^(12) -> (cos(12 * x) + 12 * cos(10 * x) + 66 * cos(8 * x) + 220 * cos(6 * x) + 495 * cos(4 * x) + 792 * cos(2 * x) + 462) / 2048
   After: (cos(12 * x) + 12 * cos(10 * x) + 66 * cos(8 * x) + 220 * cos(6 * x) + 495 * cos(4 * x) + 792 * cos(2 * x) + 462) / 2048
Result: (cos(12 * x) + 12 * cos(10 * x) + 66 * cos(8 * x) + 220 * cos(6 * x) + 495 * cos(4 * x) + 792 * cos(2 * x) + 462) / 2048
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `cos(x)^12`
   - after: `(cos(12 · x) + 12 · cos(10 · x) + 66 · cos(8 · x) + 220 · cos(6 · x) + 495 · cos(4 · x) + 792 · cos(2 · x) + 462)/2048`
   - substeps:
     1. `Usar cos²(u) = (1 + cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_cosine_twentieth_power_reduction (trig_expand)

- Source: `cos(x)^20`
- Target: `(92378+167960*cos(2*x)+125970*cos(4*x)+77520*cos(6*x)+38760*cos(8*x)+15504*cos(10*x)+4845*cos(12*x)+1140*cos(14*x)+190*cos(16*x)+20*cos(18*x)+cos(20*x))/524288`
- Result: `(cos(20 * x) + 20 * cos(18 * x) + 190 * cos(16 * x) + 1140 * cos(14 * x) + 4845 * cos(12 * x) + 15504 * cos(10 * x) + 38760 * cos(8 * x) + 77520 * cos(6 * x) + 125970 * cos(4 * x) + 167960 * cos(2 * x) + 92378) / 524288`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x)^20
Target: (cos(20 * x) + 20 * cos(18 * x) + 190 * cos(16 * x) + 1140 * cos(14 * x) + 4845 * cos(12 * x) + 15504 * cos(10 * x) + 38760 * cos(8 * x) + 77520 * cos(6 * x) + 125970 * cos(4 * x) + 167960 * cos(2 * x) + 92378) / 524288
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce cos²⁰(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: cos(x)^(20)
   Cambio local: cos(x)^(20) -> (cos(20 * x) + 20 * cos(18 * x) + 190 * cos(16 * x) + 1140 * cos(14 * x) + 4845 * cos(12 * x) + 15504 * cos(10 * x) + 38760 * cos(8 * x) + 77520 * cos(6 * x) + 125970 * cos(4 * x) + 167960 * cos(2 * x) + 92378) / 524288
   After: (cos(20 * x) + 20 * cos(18 * x) + 190 * cos(16 * x) + 1140 * cos(14 * x) + 4845 * cos(12 * x) + 15504 * cos(10 * x) + 38760 * cos(8 * x) + 77520 * cos(6 * x) + 125970 * cos(4 * x) + 167960 * cos(2 * x) + 92378) / 524288
Result: (cos(20 * x) + 20 * cos(18 * x) + 190 * cos(16 * x) + 1140 * cos(14 * x) + 4845 * cos(12 * x) + 15504 * cos(10 * x) + 38760 * cos(8 * x) + 77520 * cos(6 * x) + 125970 * cos(4 * x) + 167960 * cos(2 * x) + 92378) / 524288
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `cos(x)^20`
   - after: `(cos(20 · x) + 20 · cos(18 · x) + 190 · cos(16 · x) + 1140 · cos(14 · x) + 4845 · cos(12 · x) + 15504 · cos(10 · x) + 38760 · cos(8 · x) + 77520 · cos(6 · x) + 125970 · cos(4 · x) + 167960 · cos(2 · x) + 92378)/524288`
   - substeps:
     1. `Usar cos²(u) = (1 + cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_cosine_twenty_fourth_power_reduction (trig_expand)

- Source: `cos(x)^24`
- Target: `(1352078+2496144*cos(2*x)+1961256*cos(4*x)+1307504*cos(6*x)+735471*cos(8*x)+346104*cos(10*x)+134596*cos(12*x)+42504*cos(14*x)+10626*cos(16*x)+2024*cos(18*x)+276*cos(20*x)+24*cos(22*x)+cos(24*x))/8388608`
- Result: `(cos(24 * x) + 24 * cos(22 * x) + 276 * cos(20 * x) + 2024 * cos(18 * x) + 10626 * cos(16 * x) + 42504 * cos(14 * x) + 134596 * cos(12 * x) + 346104 * cos(10 * x) + 735471 * cos(8 * x) + 1307504 * cos(6 * x) + 1961256 * cos(4 * x) + 2496144 * cos(2 * x) + 1352078) / 8388608`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x)^24
Target: (cos(24 * x) + 24 * cos(22 * x) + 276 * cos(20 * x) + 2024 * cos(18 * x) + 10626 * cos(16 * x) + 42504 * cos(14 * x) + 134596 * cos(12 * x) + 346104 * cos(10 * x) + 735471 * cos(8 * x) + 1307504 * cos(6 * x) + 1961256 * cos(4 * x) + 2496144 * cos(2 * x) + 1352078) / 8388608
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce higher even powers of cos(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: cos(x)^(24)
   Cambio local: cos(x)^(24) -> (cos(24 * x) + 24 * cos(22 * x) + 276 * cos(20 * x) + 2024 * cos(18 * x) + 10626 * cos(16 * x) + 42504 * cos(14 * x) + 134596 * cos(12 * x) + 346104 * cos(10 * x) + 735471 * cos(8 * x) + 1307504 * cos(6 * x) + 1961256 * cos(4 * x) + 2496144 * cos(2 * x) + 1352078) / 8388608
   After: (cos(24 * x) + 24 * cos(22 * x) + 276 * cos(20 * x) + 2024 * cos(18 * x) + 10626 * cos(16 * x) + 42504 * cos(14 * x) + 134596 * cos(12 * x) + 346104 * cos(10 * x) + 735471 * cos(8 * x) + 1307504 * cos(6 * x) + 1961256 * cos(4 * x) + 2496144 * cos(2 * x) + 1352078) / 8388608
Result: (cos(24 * x) + 24 * cos(22 * x) + 276 * cos(20 * x) + 2024 * cos(18 * x) + 10626 * cos(16 * x) + 42504 * cos(14 * x) + 134596 * cos(12 * x) + 346104 * cos(10 * x) + 735471 * cos(8 * x) + 1307504 * cos(6 * x) + 1961256 * cos(4 * x) + 2496144 * cos(2 * x) + 1352078) / 8388608
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `cos(x)^24`
   - after: `(cos(24 · x) + 24 · cos(22 · x) + 276 · cos(20 · x) + 2024 · cos(18 · x) + 10626 · cos(16 · x) + 42504 · cos(14 · x) + 134596 · cos(12 · x) + 346104 · cos(10 · x) + 735471 · cos(8 · x) + 1307504 · cos(6 · x) + 1961256 · cos(4 · x) + 2496144 · cos(2 · x) + 1352078)/8388608`
   - substeps:
     1. `Usar cos²(u) = (1 + cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_cosine_twenty_second_power_reduction (trig_expand)

- Source: `cos(x)^22`
- Target: `(352716+646646*cos(2*x)+497420*cos(4*x)+319770*cos(6*x)+170544*cos(8*x)+74613*cos(10*x)+26334*cos(12*x)+7315*cos(14*x)+1540*cos(16*x)+231*cos(18*x)+22*cos(20*x)+cos(22*x))/2097152`
- Result: `(cos(22 * x) + 22 * cos(20 * x) + 231 * cos(18 * x) + 1540 * cos(16 * x) + 7315 * cos(14 * x) + 26334 * cos(12 * x) + 74613 * cos(10 * x) + 170544 * cos(8 * x) + 319770 * cos(6 * x) + 497420 * cos(4 * x) + 646646 * cos(2 * x) + 352716) / 2097152`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x)^22
Target: (cos(22 * x) + 22 * cos(20 * x) + 231 * cos(18 * x) + 1540 * cos(16 * x) + 7315 * cos(14 * x) + 26334 * cos(12 * x) + 74613 * cos(10 * x) + 170544 * cos(8 * x) + 319770 * cos(6 * x) + 497420 * cos(4 * x) + 646646 * cos(2 * x) + 352716) / 2097152
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce cos²²(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: cos(x)^(22)
   Cambio local: cos(x)^(22) -> (cos(22 * x) + 22 * cos(20 * x) + 231 * cos(18 * x) + 1540 * cos(16 * x) + 7315 * cos(14 * x) + 26334 * cos(12 * x) + 74613 * cos(10 * x) + 170544 * cos(8 * x) + 319770 * cos(6 * x) + 497420 * cos(4 * x) + 646646 * cos(2 * x) + 352716) / 2097152
   After: (cos(22 * x) + 22 * cos(20 * x) + 231 * cos(18 * x) + 1540 * cos(16 * x) + 7315 * cos(14 * x) + 26334 * cos(12 * x) + 74613 * cos(10 * x) + 170544 * cos(8 * x) + 319770 * cos(6 * x) + 497420 * cos(4 * x) + 646646 * cos(2 * x) + 352716) / 2097152
Result: (cos(22 * x) + 22 * cos(20 * x) + 231 * cos(18 * x) + 1540 * cos(16 * x) + 7315 * cos(14 * x) + 26334 * cos(12 * x) + 74613 * cos(10 * x) + 170544 * cos(8 * x) + 319770 * cos(6 * x) + 497420 * cos(4 * x) + 646646 * cos(2 * x) + 352716) / 2097152
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `cos(x)^22`
   - after: `(cos(22 · x) + 22 · cos(20 · x) + 231 · cos(18 · x) + 1540 · cos(16 · x) + 7315 · cos(14 · x) + 26334 · cos(12 · x) + 74613 · cos(10 · x) + 170544 · cos(8 · x) + 319770 · cos(6 · x) + 497420 · cos(4 · x) + 646646 · cos(2 · x) + 352716)/2097152`
   - substeps:
     1. `Usar cos²(u) = (1 + cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_cot_quotient (trig_expand)

- Source: `cot(x)`
- Target: `cos(x)/sin(x)`
- Result: `cos(x) / sin(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cot(x)
Target: cos(x) / sin(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand cot(u) as cos(u) / sin(u)  [Aplicar identidad trigonométrica recíproca]
   Before: cot(x)
   Cambio local: cot(x) -> cos(x) / sin(x)
   After: cos(x) / sin(x)
Result: cos(x) / sin(x)
ℹ️ Requires:
  • sin(x) ≠ 0
```

### Web / JSON Steps

1. `Reescribir cotangente como coseno entre seno`
   - before: `cot(x)`
   - after: `cos(x)/sin(x)`
   - substeps: none

## expand_trig_csc_reciprocal (trig_expand)

- Source: `csc(x)`
- Target: `1/sin(x)`
- Result: `1 / sin(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: csc(x)
Target: 1 / sin(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand csc(u) as 1 / sin(u)  [Aplicar identidad trigonométrica recíproca]
   Before: csc(x)
   Cambio local: csc(x) -> 1 / sin(x)
   After: 1 / sin(x)
Result: 1 / sin(x)
ℹ️ Requires:
  • sin(x) ≠ 0
```

### Web / JSON Steps

1. `Reescribir cosecante como recíproco del seno`
   - before: `csc(x)`
   - after: `1/sin(x)`
   - substeps: none

## expand_trig_csc_squared (trig_expand)

- Source: `csc(x)^2`
- Target: `1 + cot(x)^2`
- Result: `cot(x)^2 + 1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: csc(x)^2
Target: cot(x)^2 + 1
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand csc²(u) as 1 + cot(u)^2  [Expandir cosecante cuadrada]
   Before: csc(x)^(2)
   Cambio local: csc(x)^(2) -> cot(x)^(2) + 1
   After: cot(x)^2 + 1
Result: cot(x)^(2) + 1
```

### Web / JSON Steps

1. `Expandir cosecante cuadrada`
   - before: `csc(x)^2`
   - after: `cot(x)^2 + 1`
   - substeps: none

## expand_trig_double_cos_as_one_minus_sin_sq (trig_expand)

- Source: `cos(2*x)`
- Target: `1 - 2*sin(x)^2`
- Result: `1 - 2 * sin(x)^2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cos(2 * x)
Target: 1 - 2 * sin(x)^2
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand cosine double-angle as 1 - 2·sin(u)^2  [Expandir ángulo doble]
   Before: cos(2 * x)
   Cambio local: cos(2 * x) -> 1 - 2 * sin(x)^(2)
   After: 1 - 2 * sin(x)^2
Result: 1 - 2 * sin(x)^(2)
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `cos(2 · x)`
   - after: `1 - 2 · sin(x)^2`
   - substeps: none

## expand_trig_double_cos_as_two_cos_sq_minus_one (trig_expand)

- Source: `cos(2*x)`
- Target: `2*cos(x)^2 - 1`
- Result: `2 * cos(x)^2 - 1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cos(2 * x)
Target: 2 * cos(x)^2 - 1
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand cosine double-angle as 2·cos(u)^2 - 1  [Expandir ángulo doble]
   Before: cos(2 * x)
   Cambio local: cos(2 * x) -> 2 * cos(x)^(2) - 1
   After: 2 * cos(x)^2 - 1
Result: 2 * cos(x)^(2) - 1
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `cos(2 · x)`
   - after: `2 · cos(x)^2 - 1`
   - substeps: none

## expand_trig_double_cos_inverse_arccos (trig_expand)

- Source: `cos(2*arccos(x))`
- Target: `2*x^2-1`
- Result: `2 * x^2 - 1`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: cos(2 * arccos(x))
Target: 2 * x^2 - 1
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand double-angle cosine  [Expandir ángulo doble]
   Before: cos(2 * arccos(x))
   Cambio local: cos(2 * arccos(x)) -> 2 * x^(2) - 1
   After: 2 * x^2 - 1
Result: 2 * x^(2) - 1
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `cos(2 · arccos(x))`
   - after: `2 · x^2 - 1`
   - substeps:
     1. `Expandir con la identidad de ángulo doble`
     2. `Sustituir las razones trigonométricas inversas`

## expand_trig_double_cos_inverse_arcsin (trig_expand)

- Source: `cos(2*arcsin(x))`
- Target: `1-2*x^2`
- Result: `1 - 2 * x^2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: cos(2 * arcsin(x))
Target: 1 - 2 * x^2
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand double-angle cosine  [Expandir ángulo doble]
   Before: cos(2 * arcsin(x))
   Cambio local: cos(2 * arcsin(x)) -> 1 - 2 * x^(2)
   After: 1 - 2 * x^2
Result: 1 - 2 * x^(2)
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `cos(2 · arcsin(x))`
   - after: `1 - 2 · x^2`
   - substeps:
     1. `Expandir con la identidad de ángulo doble`
     2. `Sustituir las razones trigonométricas inversas`

## expand_trig_double_sin (trig_expand)

- Source: `sin(2*x)`
- Target: `2*sin(x)*cos(x)`
- Result: `2 * sin(x) * cos(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(2 * x)
Target: 2 * sin(x) * cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand double-angle sine  [Expandir ángulo doble]
   Before: sin(2 * x)
   Cambio local: sin(2 * x) -> 2 * sin(x) * cos(x)
   After: 2 * sin(x) * cos(x)
Result: 2 * sin(x) * cos(x)
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `sin(2 · x)`
   - after: `2 · sin(x) · cos(x)`
   - substeps: none

## expand_trig_double_sin_arctan_projection (trig_expand)

- Source: `sin(2*arctan(x))`
- Target: `2*x/(1+x^2)`
- Result: `2 * x / (x^2 + 1)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sin(2 * arctan(x))
Target: 2 * x / (x^2 + 1)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand double-angle sine  [Expandir ángulo doble]
   Before: sin(2 * arctan(x))
   Cambio local: sin(2 * arctan(x)) -> 2 * x / (x^(2) + 1)
   After: 2 * x / (x^2 + 1)
Result: 2 * x / (x^(2) + 1)
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `sin(2 · arctan(x))`
   - after: `(2 · x)/(x^2 + 1)`
   - substeps:
     1. `Expandir con la identidad de ángulo doble`
     2. `Sustituir las razones trigonométricas inversas`

## expand_trig_double_sin_inverse_arccos (trig_expand)

- Source: `sin(2*arccos(x))`
- Target: `2*x*sqrt(1-x^2)`
- Result: `2 * x * sqrt(1 - x^2)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sin(2 * arccos(x))
Target: 2 * x * sqrt(1 - x^2)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand double-angle sine  [Expandir ángulo doble]
   Before: sin(2 * arccos(x))
   Cambio local: sin(2 * arccos(x)) -> 2 * x * sqrt(1 - x^(2))
   After: 2 * x * sqrt(1 - x^2)
Result: 2 * x * sqrt(1 - x^(2))
ℹ️ Requires:
  • -1 ≤ x ≤ 1
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `sin(2 · arccos(x))`
   - after: `2 · x · sqrt(1 - x^2)`
   - substeps:
     1. `Expandir con la identidad de ángulo doble`
     2. `Sustituir las razones trigonométricas inversas`

## expand_trig_double_sin_inverse_arcsin (trig_expand)

- Source: `sin(2*arcsin(x))`
- Target: `2*x*sqrt(1-x^2)`
- Result: `2 * x * sqrt(1 - x^2)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sin(2 * arcsin(x))
Target: 2 * x * sqrt(1 - x^2)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand double-angle sine  [Expandir ángulo doble]
   Before: sin(2 * arcsin(x))
   Cambio local: sin(2 * arcsin(x)) -> 2 * x * sqrt(1 - x^(2))
   After: 2 * x * sqrt(1 - x^2)
Result: 2 * x * sqrt(1 - x^(2))
ℹ️ Requires:
  • -1 ≤ x ≤ 1
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `sin(2 · arcsin(x))`
   - after: `2 · x · sqrt(1 - x^2)`
   - substeps:
     1. `Expandir con la identidad de ángulo doble`
     2. `Sustituir las razones trigonométricas inversas`

## expand_trig_double_tangent (trig_expand)

- Source: `tan(2*x)`
- Target: `2*tan(x)/(1-tan(x)^2)`
- Result: `2 * tan(x) / (1 - tan(x)^2)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: tan(2 * x)
Target: 2 * tan(x) / (1 - tan(x)^2)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand tangent double-angle form  [Aplicar identidad de tangente de ángulo doble]
   Before: tan(2 * x)
   Cambio local: tan(2 * x) -> 2 * tan(x) / (1 - tan(x)^(2))
   After: 2 * tan(x) / (1 - tan(x)^2)
Result: 2 * tan(x) / (1 - tan(x)^(2))
ℹ️ Requires:
  • 1 - tan(x) ≠ 0
  • tan(x) + 1 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de tangente de ángulo doble`
   - before: `tan(2 · x)`
   - after: `(2 · tan(x))/(1 - tan(x)^2)`
   - substeps: none

## expand_trig_half_angle_cos_squared (trig_expand)

- Source: `cos(x)^2`
- Target: `(1+cos(2*x))/2`
- Result: `(cos(2 * x) + 1) / 2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cos(x)^2
Target: (cos(2 * x) + 1) / 2
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand cos²(u) as (1 + cos(2u))/2  [Aplicar identidad de ángulo mitad]
   Before: cos(x)^(2)
   Cambio local: cos(x)^(2) -> (cos(2 * x) + 1) / 2
   After: (cos(2 * x) + 1) / 2
Result: (cos(2 * x) + 1) / 2
```

### Web / JSON Steps

1. `Aplicar identidad de ángulo mitad`
   - before: `cos(x)^2`
   - after: `(cos(2 · x) + 1)/2`
   - substeps: none

## expand_trig_half_angle_sin_squared (trig_expand)

- Source: `sin(x)^2`
- Target: `(1-cos(2*x))/2`
- Result: `(1 - cos(2 * x)) / 2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(x)^2
Target: (1 - cos(2 * x)) / 2
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand sin²(u) as (1 - cos(2u))/2  [Aplicar identidad de ángulo mitad]
   Before: sin(x)^(2)
   Cambio local: sin(x)^(2) -> (1 - cos(2 * x)) / 2
   After: (1 - cos(2 * x)) / 2
Result: (1 - cos(2 * x)) / 2
```

### Web / JSON Steps

1. `Aplicar identidad de ángulo mitad`
   - before: `sin(x)^2`
   - after: `(1 - cos(2 · x))/2`
   - substeps: none

## expand_trig_half_angle_tangent (trig_expand)

- Source: `tan(x)`
- Target: `(1-cos(2*x))/sin(2*x)`
- Result: `(1 - cos(2 * x)) / sin(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: tan(x)
Target: (1 - cos(2 * x)) / sin(2 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand tan(u) as (1 - cos(2u))/sin(2u)  [Aplicar identidad de tangente de ángulo mitad]
   Before: tan(x)
   Cambio local: tan(x) -> (1 - cos(2 * x)) / sin(2 * x)
   After: (1 - cos(2 * x)) / sin(2 * x)
Result: (1 - cos(2 * x)) / sin(2 * x)
ℹ️ Requires:
  • sin(2 * x) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de tangente de ángulo mitad`
   - before: `tan(x)`
   - after: `(1 - cos(2 · x))/sin(2 · x)`
   - substeps: none

## expand_trig_half_angle_tangent_alt (trig_expand)

- Source: `tan(x)`
- Target: `sin(2*x)/(1+cos(2*x))`
- Result: `sin(2 * x) / (cos(2 * x) + 1)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: tan(x)
Target: sin(2 * x) / (cos(2 * x) + 1)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand tan(u) as sin(2u)/(1 + cos(2u))  [Aplicar identidad de tangente de ángulo mitad]
   Before: tan(x)
   Cambio local: tan(x) -> sin(2 * x) / (cos(2 * x) + 1)
   After: sin(2 * x) / (cos(2 * x) + 1)
Result: sin(2 * x) / (cos(2 * x) + 1)
ℹ️ Requires:
  • cos(2 * x) + 1 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de tangente de ángulo mitad`
   - before: `tan(x)`
   - after: `sin(2 · x)/(cos(2 · x) + 1)`
   - substeps: none

## expand_trig_half_angle_tangent_one_minus_cos_over_sin (trig_expand)

- Source: `tan(x/2)`
- Target: `(1-cos(x))/sin(x)`
- Result: `(1 - cos(x)) / sin(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: tan(x / 2)
Target: (1 - cos(x)) / sin(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand tan(u) as (1 - cos(2u))/sin(2u)  [Aplicar identidad de tangente de ángulo mitad]
   Before: tan(x / 2)
   Cambio local: tan(x / 2) -> (1 - cos(x)) / sin(x)
   After: (1 - cos(x)) / sin(x)
Result: (1 - cos(x)) / sin(x)
ℹ️ Requires:
  • sin(x) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de tangente de ángulo mitad`
   - before: `tan(x/2)`
   - after: `(1 - cos(x))/sin(x)`
   - substeps: none

## expand_trig_half_angle_tangent_sin_over_one_plus_cos (trig_expand)

- Source: `tan(x/2)`
- Target: `sin(x)/(1+cos(x))`
- Result: `sin(x) / (cos(x) + 1)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: tan(x / 2)
Target: sin(x) / (cos(x) + 1)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand tan(u) as sin(2u)/(1 + cos(2u))  [Aplicar identidad de tangente de ángulo mitad]
   Before: tan(x / 2)
   Cambio local: tan(x / 2) -> sin(x) / (cos(x) + 1)
   After: sin(x) / (cos(x) + 1)
Result: sin(x) / (cos(x) + 1)
ℹ️ Requires:
  • cos(x) + 1 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de tangente de ángulo mitad`
   - before: `tan(x/2)`
   - after: `sin(x)/(cos(x) + 1)`
   - substeps: none

## expand_trig_negative_double_cos_as_square_diff (trig_expand)

- Source: `-cos(2*x)`
- Target: `sin(x)^2 - cos(x)^2`
- Result: `sin(x)^2 - cos(x)^2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: -cos(2 * x)
Target: sin(x)^2 - cos(x)^2
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand -cos(2u) using the double-angle identity  [Expandir ángulo doble]
   Before: -cos(2 * x)
   Cambio local: -cos(2 * x) -> sin(x)^(2) - cos(x)^(2)
   After: sin(x)^2 - cos(x)^2
Result: sin(x)^(2) - cos(x)^(2)
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `-cos(2 · x)`
   - after: `sin(x)^2 - cos(x)^2`
   - substeps: none

## expand_trig_negative_double_sin (trig_expand)

- Source: `-sin(2*x)`
- Target: `-2*sin(x)*cos(x)`
- Result: `-2 * sin(x) * cos(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: -sin(2 * x)
Target: -2 * sin(x) * cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand -sin(2u) using the double-angle identity  [Expandir ángulo doble]
   Before: -sin(2 * x)
   Cambio local: -sin(2 * x) -> -2 * sin(x) * cos(x)
   After: -2 * sin(x) * cos(x)
Result: -2 * sin(x) * cos(x)
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `-sin(2 · x)`
   - after: `-2 · sin(x) · cos(x)`
   - substeps: none

## expand_trig_negative_tangent_parity (trig_expand)

- Source: `tan(-x)`
- Target: `-tan(x)`
- Result: `-tan(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: tan(-x)
Target: -tan(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Apply a trigonometric odd/even parity identity  [Aplicar paridad trigonométrica]
   Before: tan(-x)
   Cambio local: tan(-x) -> -tan(x)
   After: -tan(x)
Result: -tan(x)
```

### Web / JSON Steps

1. `Aplicar paridad trigonométrica`
   - before: `tan(-x)`
   - after: `-tan(x)`
   - substeps:
     1. `Usar que una función impar cumple f(-u) = -f(u)`

## expand_trig_phase_shift_exact_sixth_shifted_sine_to_sum (trig_expand)

- Source: `2*sin(x+pi/6)`
- Target: `sqrt(3)*sin(x)+cos(x)`
- Result: `cos(x) + sin(x) * sqrt(3)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * sin(pi / 6 + x)
Target: cos(x) + sin(x) * sqrt(3)
Strategy: expand trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: 2 * sin(pi / 6 + x)
   Cambio local: 2 * sin(pi / 6 + x) -> cos(x) + sin(x) * sqrt(3)
   After: cos(x) + sin(x) * sqrt(3)
Result: cos(x) + sin(x) * sqrt(3)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `2 · sin(pi/6 + x)`
   - after: `cos(x) + sin(x) · sqrt(3)`
   - substeps:
     1. `Expandir R·sin(u + φ)`

## expand_trig_phase_shift_exact_third_scaled_shifted_sine_to_sum (trig_expand)

- Source: `4*sin(x+pi/3)`
- Target: `2*sin(x)+2*sqrt(3)*cos(x)`
- Result: `2 * sin(x) + 2 * cos(x) * sqrt(3)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 4 * sin(pi / 3 + x)
Target: 2 * sin(x) + 2 * cos(x) * sqrt(3)
Strategy: expand trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: 4 * sin(pi / 3 + x)
   Cambio local: 4 * sin(pi / 3 + x) -> 2 * sin(x) + 2 * cos(x) * sqrt(3)
   After: 2 * sin(x) + 2 * cos(x) * sqrt(3)
Result: 2 * sin(x) + 2 * cos(x) * sqrt(3)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `4 · sin(pi/3 + x)`
   - after: `2 · sin(x) + 2 · cos(x) · sqrt(3)`
   - substeps:
     1. `Expandir R·sin(u + φ)`

## expand_trig_phase_shift_general_shifted_sine_to_sum (trig_expand)

- Source: `5*sin(x+arctan(4/3))`
- Target: `3*sin(x)+4*cos(x)`
- Result: `3 * sin(x) + 4 * cos(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 5 * sin(arctan(4 / 3) + x)
Target: 3 * sin(x) + 4 * cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: 5 * sin(arctan(4 / 3) + x)
   Cambio local: 5 * sin(arctan(4 / 3) + x) -> 3 * sin(x) + 4 * cos(x)
   After: 3 * sin(x) + 4 * cos(x)
Result: 3 * sin(x) + 4 * cos(x)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `5 · sin(arctan(4/3) + x)`
   - after: `3 · sin(x) + 4 · cos(x)`
   - substeps:
     1. `Expandir R·sin(u + φ)`

## expand_trig_phase_shift_general_shifted_sine_to_sum_with_passthrough (trig_expand)

- Source: `5*sin(x+arctan(4/3))+a`
- Target: `3*sin(x)+4*cos(x)+a`
- Result: `3 * sin(x) + 4 * cos(x) + a`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 5 * sin(arctan(4 / 3) + x) + a
Target: 3 * sin(x) + 4 * cos(x) + a
Strategy: expand trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: 5 * sin(arctan(4 / 3) + x) + a
   Cambio local: 5 * sin(arctan(4 / 3) + x) + a -> 3 * sin(x) + 4 * cos(x) + a
   After: 3 * sin(x) + 4 * cos(x) + a
Result: 3 * sin(x) + 4 * cos(x) + a
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `5 · sin(arctan(4/3) + x) + a`
   - after: `3 · sin(x) + 4 · cos(x) + a`
   - substeps:
     1. `Aplicar la identidad de desfase al bloque que cambia`

## expand_trig_phase_shift_pair_sum_to_shifted_sine_pair (trig_expand)

- Source: `sin(x)+cos(x)+sin(y)+cos(y)`
- Target: `sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)`
- Result: `sin(pi / 4 + x) * sqrt(2) + sin(pi / 4 + y) * sqrt(2)`
- Web step count: `2`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x) + sin(y) + cos(x) + cos(y)
Target: sin(pi / 4 + x) * sqrt(2) + sin(pi / 4 + y) * sqrt(2)
Strategy: contract trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: sin(x) + sin(y) + cos(x) + cos(y)
   Cambio local: sin(x) + sin(y) + cos(x) + cos(y) -> sin(y) + cos(y) + sqrt(2) * sin(pi / 4 + x)
   After: sin(y) + cos(y) + sqrt(2) * sin(pi / 4 + x)
2. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: sin(y) + cos(y) + sqrt(2) * sin(pi / 4 + x)
   Cambio local: sin(y) + cos(y) + sqrt(2) * sin(pi / 4 + x) -> sin(pi / 4 + x) * sqrt(2) + sin(pi / 4 + y) * sqrt(2)
   After: sin(pi / 4 + x) * sqrt(2) + sin(pi / 4 + y) * sqrt(2)
Result: sin(pi / 4 + x) * sqrt(2) + sin(pi / 4 + y) * sqrt(2)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `sin(x) + sin(y) + cos(x) + cos(y)`
   - after: `sin(y) + cos(y) + sin(pi/4 + x) · sqrt(2)`
   - substeps:
     1. `Aplicar la identidad de desfase al bloque que cambia`
2. `Aplicar identidad de desfase`
   - before: `sin(y) + cos(y) + sin(pi/4 + x) · sqrt(2)`
   - after: `sin(pi/4 + x) · sqrt(2) + sin(pi/4 + y) · sqrt(2)`
   - substeps: none

## expand_trig_phase_shift_scaled_shifted_sine_to_sum (trig_expand)

- Source: `2*sqrt(2)*sin(x+pi/4)`
- Target: `2*sin(x)+2*cos(x)`
- Result: `2 * sin(x) + 2 * cos(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * sin(pi / 4 + x) * sqrt(2)
Target: 2 * sin(x) + 2 * cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: 2 * sin(pi / 4 + x) * sqrt(2)
   Cambio local: 2 * sin(pi / 4 + x) * sqrt(2) -> 2 * sin(x) + 2 * cos(x)
   After: 2 * sin(x) + 2 * cos(x)
Result: 2 * sin(x) + 2 * cos(x)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `2 · sin(pi/4 + x) · sqrt(2)`
   - after: `2 · sin(x) + 2 · cos(x)`
   - substeps:
     1. `Expandir R·sin(u + φ)`

## expand_trig_phase_shift_scaled_shifted_sine_to_sum_with_passthrough (trig_expand)

- Source: `2*sqrt(2)*sin(x+pi/4)+a`
- Target: `2*sin(x)+2*cos(x)+a`
- Result: `2 * sin(x) + 2 * cos(x) + a`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * sin(pi / 4 + x) * sqrt(2) + a
Target: 2 * sin(x) + 2 * cos(x) + a
Strategy: expand trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: 2 * sin(pi / 4 + x) * sqrt(2) + a
   Cambio local: 2 * sin(pi / 4 + x) * sqrt(2) + a -> 2 * sin(x) + 2 * cos(x) + a
   After: 2 * sin(x) + 2 * cos(x) + a
Result: 2 * sin(x) + 2 * cos(x) + a
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `2 · sin(pi/4 + x) · sqrt(2) + a`
   - after: `2 · sin(x) + 2 · cos(x) + a`
   - substeps:
     1. `Aplicar la identidad de desfase al bloque que cambia`

## expand_trig_phase_shift_shifted_cosine_to_sum (trig_expand)

- Source: `sqrt(2)*cos(x-pi/4)`
- Target: `sin(x)+cos(x)`
- Result: `sin(x) + cos(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x - pi / 4) * sqrt(2)
Target: sin(x) + cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: cos(x - pi / 4) * sqrt(2)
   Cambio local: cos(x - pi / 4) * sqrt(2) -> sin(x) + cos(x)
   After: sin(x) + cos(x)
Result: sin(x) + cos(x)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `cos(x - pi/4) · sqrt(2)`
   - after: `sin(x) + cos(x)`
   - substeps:
     1. `Expandir R·sin(u + φ)`

## expand_trig_phase_shift_shifted_sine_pair_to_sum_pair (trig_expand)

- Source: `sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)`
- Target: `sin(x)+cos(x)+sin(y)+cos(y)`
- Result: `sin(x) + sin(y) + cos(x) + cos(y)`
- Web step count: `2`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sin(pi / 4 + x) * sqrt(2) + sin(pi / 4 + y) * sqrt(2)
Target: sin(x) + sin(y) + cos(x) + cos(y)
Strategy: expand trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: sin(pi / 4 + x) * sqrt(2) + sin(pi / 4 + y) * sqrt(2)
   Cambio local: sin(pi / 4 + x) * sqrt(2) + sin(pi / 4 + y) * sqrt(2) -> sin(x) + cos(x) + sin(pi / 4 + y) * sqrt(2)
   After: sin(x) + cos(x) + sin(pi / 4 + y) * sqrt(2)
2. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: sin(x) + cos(x) + sin(pi / 4 + y) * sqrt(2)
   Cambio local: sin(x) + cos(x) + sin(pi / 4 + y) * sqrt(2) -> sin(x) + sin(y) + cos(x) + cos(y)
   After: sin(x) + sin(y) + cos(x) + cos(y)
Result: sin(x) + sin(y) + cos(x) + cos(y)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `sin(pi/4 + x) · sqrt(2) + sin(pi/4 + y) · sqrt(2)`
   - after: `sin(x) + cos(x) + sin(pi/4 + y) · sqrt(2)`
   - substeps:
     1. `Aplicar la identidad de desfase al bloque que cambia`
2. `Aplicar identidad de desfase`
   - before: `sin(x) + cos(x) + sin(pi/4 + y) · sqrt(2)`
   - after: `sin(x) + sin(y) + cos(x) + cos(y)`
   - substeps:
     1. `Aplicar la identidad de desfase al bloque que cambia`

## expand_trig_phase_shift_shifted_sine_to_sum (trig_expand)

- Source: `sqrt(2)*sin(x+pi/4)`
- Target: `sin(x)+cos(x)`
- Result: `sin(x) + cos(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(pi / 4 + x) * sqrt(2)
Target: sin(x) + cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Rewrite exact sine/cosine linear combinations using a phase shift  [Aplicar identidad de desfase]
   Before: sin(pi / 4 + x) * sqrt(2)
   Cambio local: sin(pi / 4 + x) * sqrt(2) -> sin(x) + cos(x)
   After: sin(x) + cos(x)
Result: sin(x) + cos(x)
```

### Web / JSON Steps

1. `Aplicar identidad de desfase`
   - before: `sin(pi/4 + x) · sqrt(2)`
   - after: `sin(x) + cos(x)`
   - substeps:
     1. `Expandir R·sin(u + φ)`

## expand_trig_product_to_sum_cos_cos (trig_expand)

- Source: `2*cos(x)*cos(y)`
- Target: `cos(x+y) + cos(x-y)`
- Result: `cos(x + y) + cos(x - y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * cos(x) * cos(y)
Target: cos(x + y) + cos(x - y)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand 2·cos(A)·cos(B) into cos(A+B) + cos(A-B)  [Aplicar producto a suma]
   Before: 2 * cos(x) * cos(y)
   Cambio local: 2 * cos(x) * cos(y) -> cos(x + y) + cos(x - y)
   After: cos(x + y) + cos(x - y)
Result: cos(x + y) + cos(x - y)
```

### Web / JSON Steps

1. `Aplicar producto a suma`
   - before: `2 · cos(x) · cos(y)`
   - after: `cos(x + y) + cos(x - y)`
   - substeps:
     1. `Usar 2·cos(A)·cos(B) = cos(A+B) + cos(A-B)`

## expand_trig_product_to_sum_cos_sin (trig_expand)

- Source: `2*cos(x)*sin(y)`
- Target: `sin(x+y) - sin(x-y)`
- Result: `sin(x + y) - sin(x - y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * sin(y) * cos(x)
Target: sin(x + y) - sin(x - y)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand 2·cos(A)·sin(B) into sin(A+B) - sin(A-B)  [Aplicar producto a suma]
   Before: 2 * sin(y) * cos(x)
   Cambio local: 2 * sin(y) * cos(x) -> sin(x + y) - sin(x - y)
   After: sin(x + y) - sin(x - y)
Result: sin(x + y) - sin(x - y)
```

### Web / JSON Steps

1. `Aplicar producto a suma`
   - before: `2 · sin(y) · cos(x)`
   - after: `sin(x + y) - sin(x - y)`
   - substeps:
     1. `Usar 2·cos(A)·sin(B) = sin(A+B) - sin(A-B)`

## expand_trig_product_to_sum_sin_cos (trig_expand)

- Source: `2*sin(x)*cos(y)`
- Target: `sin(x+y) + sin(x-y)`
- Result: `sin(x + y) + sin(x - y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * sin(x) * cos(y)
Target: sin(x + y) + sin(x - y)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand 2·sin(A)·cos(B) into sin(A+B) + sin(A-B)  [Aplicar producto a suma]
   Before: 2 * sin(x) * cos(y)
   Cambio local: 2 * sin(x) * cos(y) -> sin(x + y) + sin(x - y)
   After: sin(x + y) + sin(x - y)
Result: sin(x + y) + sin(x - y)
```

### Web / JSON Steps

1. `Aplicar producto a suma`
   - before: `2 · sin(x) · cos(y)`
   - after: `sin(x + y) + sin(x - y)`
   - substeps:
     1. `Usar 2·sin(A)·cos(B) = sin(A+B) + sin(A-B)`

## expand_trig_product_to_sum_sin_sin (trig_expand)

- Source: `2*sin(x)*sin(y)`
- Target: `cos(x-y) - cos(x+y)`
- Result: `cos(x - y) - cos(x + y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * sin(x) * sin(y)
Target: cos(x - y) - cos(x + y)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand 2·sin(A)·sin(B) into cos(A-B) - cos(A+B)  [Aplicar producto a suma]
   Before: 2 * sin(x) * sin(y)
   Cambio local: 2 * sin(x) * sin(y) -> cos(x - y) - cos(x + y)
   After: cos(x - y) - cos(x + y)
Result: cos(x - y) - cos(x + y)
```

### Web / JSON Steps

1. `Aplicar producto a suma`
   - before: `2 · sin(x) · sin(y)`
   - after: `cos(x - y) - cos(x + y)`
   - substeps:
     1. `Usar 2·sin(A)·sin(B) = cos(A-B) - cos(A+B)`

## expand_trig_product_to_sum_to_cosine_difference_polynomial (expand)

- Source: `2*sin(2*x)*sin(x)`
- Target: `4*cos(x)-4*cos(x)^3`
- Result: `4 * cos(x) - 4 * cos(x)^3`
- Web step count: `2`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 * sin(x) * sin(2 * x)
Target: 4 * cos(x) - 4 * cos(x)^3
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand 2·sin(A)·sin(B) into cos(A-B) - cos(A+B)  [Aplicar producto a suma]
   Before: 2 * sin(x) * sin(2 * x)
   Cambio local: 2 * sin(x) * sin(2 * x) -> cos(x) - cos(3 * x)
   After: cos(x) - cos(3 * x)
2. Expand or contract cosine triple-angle form  [Reescribir ángulo triple]
   Before: cos(x) - cos(3 * x)
   Cambio local: cos(x) - cos(3 * x) -> 4 * cos(x) - 4 * cos(x)^(3)
   After: 4 * cos(x) - 4 * cos(x)^3
Result: 4 * cos(x) - 4 * cos(x)^(3)
```

### Web / JSON Steps

1. `Aplicar producto a suma`
   - before: `2 · sin(x) · sin(2 · x)`
   - after: `cos(x) - cos(3 · x)`
   - substeps:
     1. `Usar 2·sin(A)·sin(B) = cos(A-B) - cos(A+B)`
2. `Reescribir ángulo triple`
   - before: `cos(x) - cos(3 · x)`
   - after: `4 · cos(x) - 4 · cos(x)^3`
   - substeps:
     1. `Usar cos(3u) = 4 · cos(u)^3 - 3 · cos(u), con u = x`

## expand_trig_product_to_sum_to_cosine_difference_polynomial_with_passthrough (expand)

- Source: `2*sin(2*x)*sin(x)+a`
- Target: `4*cos(x)-4*cos(x)^3+a`
- Result: `4 * cos(x) - 4 * cos(x)^3 + a`
- Web step count: `2`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 * sin(x) * sin(2 * x) + a
Target: 4 * cos(x) - 4 * cos(x)^3 + a
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand 2·sin(A)·sin(B) into cos(A-B) - cos(A+B)  [Aplicar producto a suma]
   Before: 2 * sin(x) * sin(2 * x) + a
   Cambio local: 2 * sin(x) * sin(2 * x) + a -> cos(x) - cos(3 * x) + a
   After: cos(x) - cos(3 * x) + a
2. Expand or contract cosine triple-angle form  [Reescribir ángulo triple]
   Before: cos(x) - cos(3 * x) + a
   Cambio local: cos(x) - cos(3 * x) + a -> 4 * cos(x) - 4 * cos(x)^(3) + a
   After: 4 * cos(x) - 4 * cos(x)^3 + a
Result: 4 * cos(x) - 4 * cos(x)^(3) + a
```

### Web / JSON Steps

1. `Aplicar producto a suma`
   - before: `2 · sin(x) · sin(2 · x) + a`
   - after: `cos(x) - cos(3 · x) + a`
   - substeps:
     1. `Usar 2·sin(A)·sin(B) = cos(A-B) - cos(A+B)`
2. `Reescribir ángulo triple`
   - before: `cos(x) - cos(3 · x) + a`
   - after: `4 · cos(x) - 4 · cos(x)^3 + a`
   - substeps:
     1. `Usar cos(3u) = 4 · cos(u)^3 - 3 · cos(u), con u = x`

## expand_trig_product_to_sum_to_cosine_sum_polynomial (expand)

- Source: `2*cos(2*x)*cos(x)`
- Target: `4*cos(x)^3-2*cos(x)`
- Result: `4 * cos(x)^3 - 2 * cos(x)`
- Web step count: `2`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 * cos(x) * cos(2 * x)
Target: 4 * cos(x)^3 - 2 * cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand 2·cos(A)·cos(B) into cos(A+B) + cos(A-B)  [Aplicar producto a suma]
   Before: 2 * cos(x) * cos(2 * x)
   Cambio local: 2 * cos(x) * cos(2 * x) -> cos(x) + cos(3 * x)
   After: cos(x) + cos(3 * x)
2. Expand or contract cosine triple-angle form  [Reescribir ángulo triple]
   Before: cos(x) + cos(3 * x)
   Cambio local: cos(x) + cos(3 * x) -> 4 * cos(x)^(3) - 2 * cos(x)
   After: 4 * cos(x)^3 - 2 * cos(x)
Result: 4 * cos(x)^(3) - 2 * cos(x)
```

### Web / JSON Steps

1. `Aplicar producto a suma`
   - before: `2 · cos(x) · cos(2 · x)`
   - after: `cos(x) + cos(3 · x)`
   - substeps:
     1. `Usar 2·cos(A)·cos(B) = cos(A+B) + cos(A-B)`
2. `Reescribir ángulo triple`
   - before: `cos(x) + cos(3 · x)`
   - after: `4 · cos(x)^3 - 2 · cos(x)`
   - substeps:
     1. `Usar cos(3u) = 4 · cos(u)^3 - 3 · cos(u), con u = x`

## expand_trig_product_to_sum_to_sine_difference_mixed_polynomial (expand)

- Source: `2*cos(2*x)*sin(x)`
- Target: `4*cos(x)^2*sin(x)-2*sin(x)`
- Result: `4 * sin(x) * cos(x)^2 - 2 * sin(x)`
- Web step count: `2`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 * sin(x) * cos(2 * x)
Target: 4 * sin(x) * cos(x)^2 - 2 * sin(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand 2·sin(A)·cos(B) into sin(A+B) + sin(A-B)  [Aplicar producto a suma]
   Before: 2 * sin(x) * cos(2 * x)
   Cambio local: 2 * sin(x) * cos(2 * x) -> sin(3 * x) - sin(x)
   After: sin(3 * x) - sin(x)
2. Expand or contract sine triple-angle form  [Reescribir ángulo triple]
   Before: sin(3 * x) - sin(x)
   Cambio local: sin(3 * x) - sin(x) -> 4 * sin(x) * cos(x)^(2) - 2 * sin(x)
   After: 4 * sin(x) * cos(x)^2 - 2 * sin(x)
Result: 4 * sin(x) * cos(x)^(2) - 2 * sin(x)
```

### Web / JSON Steps

1. `Aplicar producto a suma`
   - before: `2 · sin(x) · cos(2 · x)`
   - after: `sin(3 · x) - sin(x)`
   - substeps:
     1. `Usar 2·sin(A)·cos(B) = sin(A+B) + sin(A-B)`
2. `Reescribir ángulo triple`
   - before: `sin(3 · x) - sin(x)`
   - after: `4 · sin(x) · cos(x)^2 - 2 · sin(x)`
   - substeps:
     1. `Usar sin(3u) = 3 · sin(u) - 4 · sin(u)^3, con u = x`

## expand_trig_product_to_sum_to_sine_difference_mixed_polynomial_with_passthrough (expand)

- Source: `2*cos(2*x)*sin(x)+a`
- Target: `4*cos(x)^2*sin(x)-2*sin(x)+a`
- Result: `-2 * sin(x) + 4 * sin(x) * cos(x)^2 + a`
- Web step count: `2`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 * sin(x) * cos(2 * x) + a
Target: -2 * sin(x) + 4 * sin(x) * cos(x)^2 + a
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand 2·sin(A)·cos(B) into sin(A+B) + sin(A-B)  [Aplicar producto a suma]
   Before: 2 * sin(x) * cos(2 * x) + a
   Cambio local: 2 * sin(x) * cos(2 * x) + a -> sin(3 * x) + a - sin(x)
   After: sin(3 * x) + a - sin(x)
2. Expand or contract sine triple-angle form  [Reescribir ángulo triple]
   Before: sin(3 * x) + a - sin(x)
   Cambio local: sin(3 * x) + a - sin(x) -> -2 * sin(x) + 4 * sin(x) * cos(x)^(2) + a
   After: -2 * sin(x) + 4 * sin(x) * cos(x)^2 + a
Result: -2 * sin(x) + 4 * sin(x) * cos(x)^(2) + a
```

### Web / JSON Steps

1. `Aplicar producto a suma`
   - before: `2 · sin(x) · cos(2 · x) + a`
   - after: `sin(3 · x) + a - sin(x)`
   - substeps:
     1. `Usar 2·sin(A)·cos(B) = sin(A+B) + sin(A-B)`
2. `Reescribir ángulo triple`
   - before: `sin(3 · x) + a - sin(x)`
   - after: `4 · sin(x) · cos(x)^2 - 2 · sin(x) + a`
   - substeps:
     1. `Usar sin(3u) = 3 · sin(u) - 4 · sin(u)^3, con u = x`

## expand_trig_quadruple_angle_cosine (trig_expand)

- Source: `cos(4*x)`
- Target: `8*cos(x)^4-8*cos(x)^2+1`
- Result: `-8 * cos(x)^2 + 8 * cos(x)^4 + 1`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(4 * x)
Target: -8 * cos(x)^2 + 8 * cos(x)^4 + 1
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand or contract cosine quadruple-angle form  [Reescribir ángulo cuádruple]
   Before: cos(4 * x)
   Cambio local: cos(4 * x) -> -8 * cos(x)^(2) + 8 * cos(x)^(4) + 1
   After: -8 * cos(x)^2 + 8 * cos(x)^4 + 1
Result: -8 * cos(x)^(2) + 8 * cos(x)^(4) + 1
```

### Web / JSON Steps

1. `Reescribir ángulo cuádruple`
   - before: `cos(4 · x)`
   - after: `8 · cos(x)^4 - 8 · cos(x)^2 + 1`
   - substeps:
     1. `Usar cos(4u) = 8 · cos(u)^4 - 8 · cos(u)^2 + 1, con u = x`

## expand_trig_quadruple_angle_sine_expanded_product (trig_expand)

- Source: `sin(4*x)`
- Target: `4*sin(x)*cos(x)^3-4*sin(x)^3*cos(x)`
- Result: `4 * sin(x) * cos(x)^3 - 4 * cos(x) * sin(x)^3`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(4 * x)
Target: 4 * sin(x) * cos(x)^3 - 4 * cos(x) * sin(x)^3
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand or contract sine quadruple-angle form  [Reescribir ángulo cuádruple]
   Before: sin(4 * x)
   Cambio local: sin(4 * x) -> 4 * sin(x) * cos(x)^(3) - 4 * cos(x) * sin(x)^(3)
   After: 4 * sin(x) * cos(x)^3 - 4 * cos(x) * sin(x)^3
Result: 4 * sin(x) * cos(x)^(3) - 4 * cos(x) * sin(x)^(3)
```

### Web / JSON Steps

1. `Reescribir ángulo cuádruple`
   - before: `sin(4 · x)`
   - after: `4 · sin(x) · cos(x)^3 - 4 · cos(x) · sin(x)^3`
   - substeps:
     1. `Usar sin(4u) = 4 · sin(u) · cos(u)^3 - 4 · sin(u)^3 · cos(u), con u = x`

## expand_trig_quintuple_angle_cosine (trig_expand)

- Source: `cos(5*x)`
- Target: `16*cos(x)^5-20*cos(x)^3+5*cos(x)`
- Result: `5 * cos(x) + 16 * cos(x)^5 - 20 * cos(x)^3`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(5 * x)
Target: 5 * cos(x) + 16 * cos(x)^5 - 20 * cos(x)^3
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand or contract cosine quintuple-angle form  [Reescribir ángulo quíntuple]
   Before: cos(5 * x)
   Cambio local: cos(5 * x) -> 5 * cos(x) + 16 * cos(x)^(5) - 20 * cos(x)^(3)
   After: 5 * cos(x) + 16 * cos(x)^5 - 20 * cos(x)^3
Result: 5 * cos(x) + 16 * cos(x)^(5) - 20 * cos(x)^(3)
```

### Web / JSON Steps

1. `Reescribir ángulo quíntuple`
   - before: `cos(5 · x)`
   - after: `5 · cos(x) + 16 · cos(x)^5 - 20 · cos(x)^3`
   - substeps:
     1. `Usar cos(5u) = 16 · cos(u)^5 - 20 · cos(u)^3 + 5 · cos(u), con u = x`

## expand_trig_quintuple_angle_sine (trig_expand)

- Source: `sin(5*x)`
- Target: `5*sin(x)-20*sin(x)^3+16*sin(x)^5`
- Result: `5 * sin(x) + 16 * sin(x)^5 - 20 * sin(x)^3`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(5 * x)
Target: 5 * sin(x) + 16 * sin(x)^5 - 20 * sin(x)^3
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand or contract sine quintuple-angle form  [Reescribir ángulo quíntuple]
   Before: sin(5 * x)
   Cambio local: sin(5 * x) -> 5 * sin(x) + 16 * sin(x)^(5) - 20 * sin(x)^(3)
   After: 5 * sin(x) + 16 * sin(x)^5 - 20 * sin(x)^3
Result: 5 * sin(x) + 16 * sin(x)^(5) - 20 * sin(x)^(3)
```

### Web / JSON Steps

1. `Reescribir ángulo quíntuple`
   - before: `sin(5 · x)`
   - after: `16 · sin(x)^5 + 5 · sin(x) - 20 · sin(x)^3`
   - substeps:
     1. `Usar sin(5u) = 5 · sin(u) - 20 · sin(u)^3 + 16 · sin(u)^5, con u = x`

## expand_trig_recursive_six_cosine (trig_expand)

- Source: `cos(6*x)`
- Target: `cos(5*x)*cos(x)-sin(5*x)*sin(x)`
- Result: `cos(x) * cos(5 * x) - sin(x) * sin(5 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(6 * x)
Target: cos(x) * cos(5 * x) - sin(x) * sin(5 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand a trig multiple angle recursively via angle addition  [Aplicar suma/diferencia de ángulos]
   Before: cos(6 * x)
   Cambio local: cos(6 * x) -> cos(x) * cos(5 * x) - sin(x) * sin(5 * x)
   After: cos(x) * cos(5 * x) - sin(x) * sin(5 * x)
Result: cos(x) * cos(5 * x) - sin(x) * sin(5 * x)
```

### Web / JSON Steps

1. `Aplicar suma/diferencia de ángulos`
   - before: `cos(6 · x)`
   - after: `cos(x) · cos(5 · x) - sin(x) · sin(5 · x)`
   - substeps:
     1. `Usar cos(5u+u) = cos(5u) · cos(u) - sin(5u) · sin(u), con u = x`

## expand_trig_recursive_six_sine (trig_expand)

- Source: `sin(6*x)`
- Target: `sin(5*x)*cos(x)+cos(5*x)*sin(x)`
- Result: `sin(x) * cos(5 * x) + sin(5 * x) * cos(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(6 * x)
Target: sin(x) * cos(5 * x) + sin(5 * x) * cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand a trig multiple angle recursively via angle addition  [Aplicar suma/diferencia de ángulos]
   Before: sin(6 * x)
   Cambio local: sin(6 * x) -> sin(x) * cos(5 * x) + sin(5 * x) * cos(x)
   After: sin(x) * cos(5 * x) + sin(5 * x) * cos(x)
Result: sin(x) * cos(5 * x) + sin(5 * x) * cos(x)
```

### Web / JSON Steps

1. `Aplicar suma/diferencia de ángulos`
   - before: `sin(6 · x)`
   - after: `sin(x) · cos(5 · x) + sin(5 · x) · cos(x)`
   - substeps:
     1. `Usar sin(5u+u) = sin(5u) · cos(u) + cos(5u) · sin(u), con u = x`

## expand_trig_scaled_half_angle_sine_square_to_shifted_cosine (trig_expand)

- Source: `2*sin(x/2)^2`
- Target: `1-cos(x)`
- Result: `1 - cos(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * sin(x / 2)^2
Target: 1 - cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand 2·sin(u)^2 as 1 - cos(2u)  [Expandir ángulo doble]
   Before: 2 * sin(x / 2)^(2)
   Cambio local: 2 * sin(x / 2)^(2) -> 1 - cos(x)
   After: 1 - cos(x)
Result: 1 - cos(x)
```

### Web / JSON Steps

1. `Expandir ángulo doble`
   - before: `2 · (sin(x/2))^2`
   - after: `1 - cos(x)`
   - substeps: none

## expand_trig_sec_reciprocal (trig_expand)

- Source: `sec(x)`
- Target: `1/cos(x)`
- Result: `1 / cos(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sec(x)
Target: 1 / cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand sec(u) as 1 / cos(u)  [Aplicar identidad trigonométrica recíproca]
   Before: sec(x)
   Cambio local: sec(x) -> 1 / cos(x)
   After: 1 / cos(x)
Result: 1 / cos(x)
ℹ️ Requires:
  • cos(x) ≠ 0
```

### Web / JSON Steps

1. `Reescribir secante como recíproco del coseno`
   - before: `sec(x)`
   - after: `1/cos(x)`
   - substeps: none

## expand_trig_sec_squared (trig_expand)

- Source: `sec(x)^2`
- Target: `1 + tan(x)^2`
- Result: `tan(x)^2 + 1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sec(x)^2
Target: tan(x)^2 + 1
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand sec²(u) as 1 + tan(u)^2  [Expandir secante cuadrada]
   Before: sec(x)^(2)
   Cambio local: sec(x)^(2) -> tan(x)^(2) + 1
   After: tan(x)^2 + 1
Result: tan(x)^(2) + 1
```

### Web / JSON Steps

1. `Expandir secante cuadrada`
   - before: `sec(x)^2`
   - after: `tan(x)^2 + 1`
   - substeps: none

## expand_trig_sin_cos_square_diff (trig_expand)

- Source: `(sin(x)-cos(x))^2`
- Target: `1-sin(2*x)`
- Result: `1 - sin(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (sin(x) - cos(x))^2
Target: 1 - sin(2 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand (sin(u) - cos(u))² as 1 - sin(2u)  [Aplicar identidad del cuadrado trigonométrico]
   Before: (sin(x) - cos(x))^(2)
   Cambio local: (sin(x) - cos(x))^(2) -> 1 - sin(2 * x)
   After: 1 - sin(2 * x)
Result: 1 - sin(2 * x)
```

### Web / JSON Steps

1. `Aplicar identidad del cuadrado trigonométrico`
   - before: `(sin(x) - cos(x))^2`
   - after: `1 - sin(2 · x)`
   - substeps: none

## expand_trig_sin_cos_square_sum (trig_expand)

- Source: `(sin(x)+cos(x))^2`
- Target: `1+sin(2*x)`
- Result: `sin(2 * x) + 1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (sin(x) + cos(x))^2
Target: sin(2 * x) + 1
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand (sin(u) + cos(u))² as 1 + sin(2u)  [Aplicar identidad del cuadrado trigonométrico]
   Before: (sin(x) + cos(x))^(2)
   Cambio local: (sin(x) + cos(x))^(2) -> sin(2 * x) + 1
   After: sin(2 * x) + 1
Result: sin(2 * x) + 1
```

### Web / JSON Steps

1. `Aplicar identidad del cuadrado trigonométrico`
   - before: `(sin(x) + cos(x))^2`
   - after: `sin(2 · x) + 1`
   - substeps: none

## expand_trig_sine_cosine_square_product_reduction (simplify)

- Source: `sin(x)^2*cos(x)^2`
- Target: `(1-cos(4*x))/8`
- Result: `(1 - cos(4 * x)) / 8`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x)^2 * cos(x)^2
Target: (1 - cos(4 * x)) / 8
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce sin²(u)·cos²(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: sin(x)^(2) * cos(x)^(2)
   Cambio local: sin(x)^(2) * cos(x)^(2) -> (1 - cos(4 * x)) / 8
   After: (1 - cos(4 * x)) / 8
Result: (1 - cos(4 * x)) / 8
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `sin(x)^2 · cos(x)^2`
   - after: `(1 - cos(4 · x))/8`
   - substeps:
     1. `Usar sin²(u)·cos²(u) = (1 - cos(4u)) / 8, con u = x`

## expand_trig_sine_eighteenth_power_reduction (trig_expand)

- Source: `sin(x)^18`
- Target: `(24310-43758*cos(2*x)+31824*cos(4*x)-18564*cos(6*x)+8568*cos(8*x)-3060*cos(10*x)+816*cos(12*x)-153*cos(14*x)+18*cos(16*x)-cos(18*x))/131072`
- Result: `(18 * cos(16 * x) - 153 * cos(14 * x) + 816 * cos(12 * x) - 3060 * cos(10 * x) + 8568 * cos(8 * x) - 18564 * cos(6 * x) + 31824 * cos(4 * x) - 43758 * cos(2 * x) + 24310 - cos(18 * x)) / 131072`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x)^18
Target: (18 * cos(16 * x) - 153 * cos(14 * x) + 816 * cos(12 * x) - 3060 * cos(10 * x) + 8568 * cos(8 * x) - 18564 * cos(6 * x) + 31824 * cos(4 * x) - 43758 * cos(2 * x) + 24310 - cos(18 * x)) / 131072
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce sin¹⁸(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: sin(x)^(18)
   Cambio local: sin(x)^(18) -> (18 * cos(16 * x) - 153 * cos(14 * x) + 816 * cos(12 * x) - 3060 * cos(10 * x) + 8568 * cos(8 * x) - 18564 * cos(6 * x) + 31824 * cos(4 * x) - 43758 * cos(2 * x) + 24310 - cos(18 * x)) / 131072
   After: (18 * cos(16 * x) - 153 * cos(14 * x) + 816 * cos(12 * x) - 3060 * cos(10 * x) + 8568 * cos(8 * x) - 18564 * cos(6 * x) + 31824 * cos(4 * x) - 43758 * cos(2 * x) + 24310 - cos(18 * x)) / 131072
Result: (18 * cos(16 * x) - 153 * cos(14 * x) + 816 * cos(12 * x) - 3060 * cos(10 * x) + 8568 * cos(8 * x) - 18564 * cos(6 * x) + 31824 * cos(4 * x) - 43758 * cos(2 * x) + 24310 - cos(18 * x)) / 131072
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `sin(x)^18`
   - after: `(18 · cos(16 · x) + 816 · cos(12 · x) + 8568 · cos(8 · x) + 31824 · cos(4 · x) + 24310 - 43758 · cos(2 · x) - 18564 · cos(6 · x) - 3060 · cos(10 · x) - 153 · cos(14 · x) - cos(18 · x))/131072`
   - substeps:
     1. `Usar sin²(u) = (1 - cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_sine_eighth_power_reduction (trig_expand)

- Source: `sin(x)^8`
- Target: `(35-56*cos(2*x)+28*cos(4*x)-8*cos(6*x)+cos(8*x))/128`
- Result: `(cos(8 * x) - 8 * cos(6 * x) + 28 * cos(4 * x) - 56 * cos(2 * x) + 35) / 128`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x)^8
Target: (cos(8 * x) - 8 * cos(6 * x) + 28 * cos(4 * x) - 56 * cos(2 * x) + 35) / 128
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce sin⁸(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: sin(x)^(8)
   Cambio local: sin(x)^(8) -> (cos(8 * x) - 8 * cos(6 * x) + 28 * cos(4 * x) - 56 * cos(2 * x) + 35) / 128
   After: (cos(8 * x) - 8 * cos(6 * x) + 28 * cos(4 * x) - 56 * cos(2 * x) + 35) / 128
Result: (cos(8 * x) - 8 * cos(6 * x) + 28 * cos(4 * x) - 56 * cos(2 * x) + 35) / 128
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `sin(x)^8`
   - after: `(cos(8 · x) + 28 · cos(4 · x) + 35 - 56 · cos(2 · x) - 8 · cos(6 · x))/128`
   - substeps:
     1. `Usar sin²(u) = (1 - cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_sine_fourteenth_power_reduction (trig_expand)

- Source: `sin(x)^14`
- Target: `(1716-3003*cos(2*x)+2002*cos(4*x)-1001*cos(6*x)+364*cos(8*x)-91*cos(10*x)+14*cos(12*x)-cos(14*x))/8192`
- Result: `(14 * cos(12 * x) - 91 * cos(10 * x) + 364 * cos(8 * x) - 1001 * cos(6 * x) + 2002 * cos(4 * x) - 3003 * cos(2 * x) + 1716 - cos(14 * x)) / 8192`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x)^14
Target: (14 * cos(12 * x) - 91 * cos(10 * x) + 364 * cos(8 * x) - 1001 * cos(6 * x) + 2002 * cos(4 * x) - 3003 * cos(2 * x) + 1716 - cos(14 * x)) / 8192
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce sin¹⁴(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: sin(x)^(14)
   Cambio local: sin(x)^(14) -> (14 * cos(12 * x) - 91 * cos(10 * x) + 364 * cos(8 * x) - 1001 * cos(6 * x) + 2002 * cos(4 * x) - 3003 * cos(2 * x) + 1716 - cos(14 * x)) / 8192
   After: (14 * cos(12 * x) - 91 * cos(10 * x) + 364 * cos(8 * x) - 1001 * cos(6 * x) + 2002 * cos(4 * x) - 3003 * cos(2 * x) + 1716 - cos(14 * x)) / 8192
Result: (14 * cos(12 * x) - 91 * cos(10 * x) + 364 * cos(8 * x) - 1001 * cos(6 * x) + 2002 * cos(4 * x) - 3003 * cos(2 * x) + 1716 - cos(14 * x)) / 8192
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `sin(x)^14`
   - after: `(14 · cos(12 · x) + 364 · cos(8 · x) + 2002 · cos(4 · x) + 1716 - 3003 · cos(2 · x) - 1001 · cos(6 · x) - 91 · cos(10 · x) - cos(14 · x))/8192`
   - substeps:
     1. `Usar sin²(u) = (1 - cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_sine_fourth_power_reduction (simplify)

- Source: `sin(x)^4`
- Target: `(3-4*cos(2*x)+cos(4*x))/8`
- Result: `(cos(4 * x) - 4 * cos(2 * x) + 3) / 8`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x)^4
Target: (cos(4 * x) - 4 * cos(2 * x) + 3) / 8
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce sin⁴(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: sin(x)^(4)
   Cambio local: sin(x)^(4) -> (cos(4 * x) - 4 * cos(2 * x) + 3) / 8
   After: (cos(4 * x) - 4 * cos(2 * x) + 3) / 8
Result: (cos(4 * x) - 4 * cos(2 * x) + 3) / 8
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `sin(x)^4`
   - after: `(cos(4 · x) + 3 - 4 · cos(2 · x))/8`
   - substeps:
     1. `Usar sin²(u) = (1 - cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_sine_sixteenth_power_reduction (trig_expand)

- Source: `sin(x)^16`
- Target: `(6435-11440*cos(2*x)+8008*cos(4*x)-4368*cos(6*x)+1820*cos(8*x)-560*cos(10*x)+120*cos(12*x)-16*cos(14*x)+cos(16*x))/32768`
- Result: `(cos(16 * x) - 16 * cos(14 * x) + 120 * cos(12 * x) - 560 * cos(10 * x) + 1820 * cos(8 * x) - 4368 * cos(6 * x) + 8008 * cos(4 * x) - 11440 * cos(2 * x) + 6435) / 32768`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x)^16
Target: (cos(16 * x) - 16 * cos(14 * x) + 120 * cos(12 * x) - 560 * cos(10 * x) + 1820 * cos(8 * x) - 4368 * cos(6 * x) + 8008 * cos(4 * x) - 11440 * cos(2 * x) + 6435) / 32768
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce sin¹⁶(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: sin(x)^(16)
   Cambio local: sin(x)^(16) -> (cos(16 * x) - 16 * cos(14 * x) + 120 * cos(12 * x) - 560 * cos(10 * x) + 1820 * cos(8 * x) - 4368 * cos(6 * x) + 8008 * cos(4 * x) - 11440 * cos(2 * x) + 6435) / 32768
   After: (cos(16 * x) - 16 * cos(14 * x) + 120 * cos(12 * x) - 560 * cos(10 * x) + 1820 * cos(8 * x) - 4368 * cos(6 * x) + 8008 * cos(4 * x) - 11440 * cos(2 * x) + 6435) / 32768
Result: (cos(16 * x) - 16 * cos(14 * x) + 120 * cos(12 * x) - 560 * cos(10 * x) + 1820 * cos(8 * x) - 4368 * cos(6 * x) + 8008 * cos(4 * x) - 11440 * cos(2 * x) + 6435) / 32768
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `sin(x)^16`
   - after: `(cos(16 · x) + 120 · cos(12 · x) + 1820 · cos(8 · x) + 8008 · cos(4 · x) + 6435 - 11440 · cos(2 · x) - 4368 · cos(6 · x) - 560 · cos(10 · x) - 16 · cos(14 · x))/32768`
   - substeps:
     1. `Usar sin²(u) = (1 - cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_sine_sixth_power_reduction (trig_expand)

- Source: `sin(x)^6`
- Target: `(10-15*cos(2*x)+6*cos(4*x)-cos(6*x))/32`
- Result: `(6 * cos(4 * x) - 15 * cos(2 * x) + 10 - cos(6 * x)) / 32`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x)^6
Target: (6 * cos(4 * x) - 15 * cos(2 * x) + 10 - cos(6 * x)) / 32
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce sin⁶(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: sin(x)^(6)
   Cambio local: sin(x)^(6) -> (6 * cos(4 * x) - 15 * cos(2 * x) + 10 - cos(6 * x)) / 32
   After: (6 * cos(4 * x) - 15 * cos(2 * x) + 10 - cos(6 * x)) / 32
Result: (6 * cos(4 * x) - 15 * cos(2 * x) + 10 - cos(6 * x)) / 32
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `sin(x)^6`
   - after: `(6 · cos(4 · x) + 10 - 15 · cos(2 · x) - cos(6 · x))/32`
   - substeps:
     1. `Usar sin²(u) = (1 - cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_sine_tenth_power_reduction (trig_expand)

- Source: `sin(x)^10`
- Target: `(126-210*cos(2*x)+120*cos(4*x)-45*cos(6*x)+10*cos(8*x)-cos(10*x))/512`
- Result: `(10 * cos(8 * x) - 45 * cos(6 * x) + 120 * cos(4 * x) - 210 * cos(2 * x) + 126 - cos(10 * x)) / 512`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x)^10
Target: (10 * cos(8 * x) - 45 * cos(6 * x) + 120 * cos(4 * x) - 210 * cos(2 * x) + 126 - cos(10 * x)) / 512
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce sin¹⁰(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: sin(x)^(10)
   Cambio local: sin(x)^(10) -> (10 * cos(8 * x) - 45 * cos(6 * x) + 120 * cos(4 * x) - 210 * cos(2 * x) + 126 - cos(10 * x)) / 512
   After: (10 * cos(8 * x) - 45 * cos(6 * x) + 120 * cos(4 * x) - 210 * cos(2 * x) + 126 - cos(10 * x)) / 512
Result: (10 * cos(8 * x) - 45 * cos(6 * x) + 120 * cos(4 * x) - 210 * cos(2 * x) + 126 - cos(10 * x)) / 512
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `sin(x)^10`
   - after: `(10 · cos(8 · x) + 120 · cos(4 · x) + 126 - 210 · cos(2 · x) - 45 · cos(6 · x) - cos(10 · x))/512`
   - substeps:
     1. `Usar sin²(u) = (1 - cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_sine_twelfth_power_reduction (trig_expand)

- Source: `sin(x)^12`
- Target: `(462-792*cos(2*x)+495*cos(4*x)-220*cos(6*x)+66*cos(8*x)-12*cos(10*x)+cos(12*x))/2048`
- Result: `(cos(12 * x) - 12 * cos(10 * x) + 66 * cos(8 * x) - 220 * cos(6 * x) + 495 * cos(4 * x) - 792 * cos(2 * x) + 462) / 2048`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x)^12
Target: (cos(12 * x) - 12 * cos(10 * x) + 66 * cos(8 * x) - 220 * cos(6 * x) + 495 * cos(4 * x) - 792 * cos(2 * x) + 462) / 2048
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce sin¹²(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: sin(x)^(12)
   Cambio local: sin(x)^(12) -> (cos(12 * x) - 12 * cos(10 * x) + 66 * cos(8 * x) - 220 * cos(6 * x) + 495 * cos(4 * x) - 792 * cos(2 * x) + 462) / 2048
   After: (cos(12 * x) - 12 * cos(10 * x) + 66 * cos(8 * x) - 220 * cos(6 * x) + 495 * cos(4 * x) - 792 * cos(2 * x) + 462) / 2048
Result: (cos(12 * x) - 12 * cos(10 * x) + 66 * cos(8 * x) - 220 * cos(6 * x) + 495 * cos(4 * x) - 792 * cos(2 * x) + 462) / 2048
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `sin(x)^12`
   - after: `(cos(12 · x) + 66 · cos(8 · x) + 495 · cos(4 · x) + 462 - 792 · cos(2 · x) - 220 · cos(6 · x) - 12 · cos(10 · x))/2048`
   - substeps:
     1. `Usar sin²(u) = (1 - cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_sine_twentieth_power_reduction (trig_expand)

- Source: `sin(x)^20`
- Target: `(92378-167960*cos(2*x)+125970*cos(4*x)-77520*cos(6*x)+38760*cos(8*x)-15504*cos(10*x)+4845*cos(12*x)-1140*cos(14*x)+190*cos(16*x)-20*cos(18*x)+cos(20*x))/524288`
- Result: `(cos(20 * x) - 20 * cos(18 * x) + 190 * cos(16 * x) - 1140 * cos(14 * x) + 4845 * cos(12 * x) - 15504 * cos(10 * x) + 38760 * cos(8 * x) - 77520 * cos(6 * x) + 125970 * cos(4 * x) - 167960 * cos(2 * x) + 92378) / 524288`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x)^20
Target: (cos(20 * x) - 20 * cos(18 * x) + 190 * cos(16 * x) - 1140 * cos(14 * x) + 4845 * cos(12 * x) - 15504 * cos(10 * x) + 38760 * cos(8 * x) - 77520 * cos(6 * x) + 125970 * cos(4 * x) - 167960 * cos(2 * x) + 92378) / 524288
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce sin²⁰(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: sin(x)^(20)
   Cambio local: sin(x)^(20) -> (cos(20 * x) - 20 * cos(18 * x) + 190 * cos(16 * x) - 1140 * cos(14 * x) + 4845 * cos(12 * x) - 15504 * cos(10 * x) + 38760 * cos(8 * x) - 77520 * cos(6 * x) + 125970 * cos(4 * x) - 167960 * cos(2 * x) + 92378) / 524288
   After: (cos(20 * x) - 20 * cos(18 * x) + 190 * cos(16 * x) - 1140 * cos(14 * x) + 4845 * cos(12 * x) - 15504 * cos(10 * x) + 38760 * cos(8 * x) - 77520 * cos(6 * x) + 125970 * cos(4 * x) - 167960 * cos(2 * x) + 92378) / 524288
Result: (cos(20 * x) - 20 * cos(18 * x) + 190 * cos(16 * x) - 1140 * cos(14 * x) + 4845 * cos(12 * x) - 15504 * cos(10 * x) + 38760 * cos(8 * x) - 77520 * cos(6 * x) + 125970 * cos(4 * x) - 167960 * cos(2 * x) + 92378) / 524288
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `sin(x)^20`
   - after: `(cos(20 · x) + 190 · cos(16 · x) + 4845 · cos(12 · x) + 38760 · cos(8 · x) + 125970 · cos(4 · x) + 92378 - 167960 · cos(2 · x) - 77520 · cos(6 · x) - 15504 · cos(10 · x) - 1140 · cos(14 · x) - 20 · cos(18 · x))/524288`
   - substeps:
     1. `Usar sin²(u) = (1 - cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_sine_twenty_fourth_power_reduction (trig_expand)

- Source: `sin(x)^24`
- Target: `(1352078-2496144*cos(2*x)+1961256*cos(4*x)-1307504*cos(6*x)+735471*cos(8*x)-346104*cos(10*x)+134596*cos(12*x)-42504*cos(14*x)+10626*cos(16*x)-2024*cos(18*x)+276*cos(20*x)-24*cos(22*x)+cos(24*x))/8388608`
- Result: `(cos(24 * x) - 24 * cos(22 * x) + 276 * cos(20 * x) - 2024 * cos(18 * x) + 10626 * cos(16 * x) - 42504 * cos(14 * x) + 134596 * cos(12 * x) - 346104 * cos(10 * x) + 735471 * cos(8 * x) - 1307504 * cos(6 * x) + 1961256 * cos(4 * x) - 2496144 * cos(2 * x) + 1352078) / 8388608`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x)^24
Target: (cos(24 * x) - 24 * cos(22 * x) + 276 * cos(20 * x) - 2024 * cos(18 * x) + 10626 * cos(16 * x) - 42504 * cos(14 * x) + 134596 * cos(12 * x) - 346104 * cos(10 * x) + 735471 * cos(8 * x) - 1307504 * cos(6 * x) + 1961256 * cos(4 * x) - 2496144 * cos(2 * x) + 1352078) / 8388608
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce higher even powers of sin(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: sin(x)^(24)
   Cambio local: sin(x)^(24) -> (cos(24 * x) - 24 * cos(22 * x) + 276 * cos(20 * x) - 2024 * cos(18 * x) + 10626 * cos(16 * x) - 42504 * cos(14 * x) + 134596 * cos(12 * x) - 346104 * cos(10 * x) + 735471 * cos(8 * x) - 1307504 * cos(6 * x) + 1961256 * cos(4 * x) - 2496144 * cos(2 * x) + 1352078) / 8388608
   After: (cos(24 * x) - 24 * cos(22 * x) + 276 * cos(20 * x) - 2024 * cos(18 * x) + 10626 * cos(16 * x) - 42504 * cos(14 * x) + 134596 * cos(12 * x) - 346104 * cos(10 * x) + 735471 * cos(8 * x) - 1307504 * cos(6 * x) + 1961256 * cos(4 * x) - 2496144 * cos(2 * x) + 1352078) / 8388608
Result: (cos(24 * x) - 24 * cos(22 * x) + 276 * cos(20 * x) - 2024 * cos(18 * x) + 10626 * cos(16 * x) - 42504 * cos(14 * x) + 134596 * cos(12 * x) - 346104 * cos(10 * x) + 735471 * cos(8 * x) - 1307504 * cos(6 * x) + 1961256 * cos(4 * x) - 2496144 * cos(2 * x) + 1352078) / 8388608
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `sin(x)^24`
   - after: `(cos(24 · x) + 276 · cos(20 · x) + 10626 · cos(16 · x) + 134596 · cos(12 · x) + 735471 · cos(8 · x) + 1961256 · cos(4 · x) + 1352078 - 2496144 · cos(2 · x) - 1307504 · cos(6 · x) - 346104 · cos(10 · x) - 42504 · cos(14 · x) - 2024 · cos(18 · x) - 24 · cos(22 · x))/8388608`
   - substeps:
     1. `Usar sin²(u) = (1 - cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_sine_twenty_second_power_reduction (trig_expand)

- Source: `sin(x)^22`
- Target: `(352716-646646*cos(2*x)+497420*cos(4*x)-319770*cos(6*x)+170544*cos(8*x)-74613*cos(10*x)+26334*cos(12*x)-7315*cos(14*x)+1540*cos(16*x)-231*cos(18*x)+22*cos(20*x)-cos(22*x))/2097152`
- Result: `(22 * cos(20 * x) - 231 * cos(18 * x) + 1540 * cos(16 * x) - 7315 * cos(14 * x) + 26334 * cos(12 * x) - 74613 * cos(10 * x) + 170544 * cos(8 * x) - 319770 * cos(6 * x) + 497420 * cos(4 * x) - 646646 * cos(2 * x) + 352716 - cos(22 * x)) / 2097152`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x)^22
Target: (22 * cos(20 * x) - 231 * cos(18 * x) + 1540 * cos(16 * x) - 7315 * cos(14 * x) + 26334 * cos(12 * x) - 74613 * cos(10 * x) + 170544 * cos(8 * x) - 319770 * cos(6 * x) + 497420 * cos(4 * x) - 646646 * cos(2 * x) + 352716 - cos(22 * x)) / 2097152
Strategy: expand trig
Steps (Aggressive Mode):
1. Reduce sin²²(u) using power-reduction identities  [Aplicar reducción de potencias]
   Before: sin(x)^(22)
   Cambio local: sin(x)^(22) -> (22 * cos(20 * x) - 231 * cos(18 * x) + 1540 * cos(16 * x) - 7315 * cos(14 * x) + 26334 * cos(12 * x) - 74613 * cos(10 * x) + 170544 * cos(8 * x) - 319770 * cos(6 * x) + 497420 * cos(4 * x) - 646646 * cos(2 * x) + 352716 - cos(22 * x)) / 2097152
   After: (22 * cos(20 * x) - 231 * cos(18 * x) + 1540 * cos(16 * x) - 7315 * cos(14 * x) + 26334 * cos(12 * x) - 74613 * cos(10 * x) + 170544 * cos(8 * x) - 319770 * cos(6 * x) + 497420 * cos(4 * x) - 646646 * cos(2 * x) + 352716 - cos(22 * x)) / 2097152
Result: (22 * cos(20 * x) - 231 * cos(18 * x) + 1540 * cos(16 * x) - 7315 * cos(14 * x) + 26334 * cos(12 * x) - 74613 * cos(10 * x) + 170544 * cos(8 * x) - 319770 * cos(6 * x) + 497420 * cos(4 * x) - 646646 * cos(2 * x) + 352716 - cos(22 * x)) / 2097152
```

### Web / JSON Steps

1. `Aplicar reducción de potencias`
   - before: `sin(x)^22`
   - after: `(22 · cos(20 · x) + 1540 · cos(16 · x) + 26334 · cos(12 · x) + 170544 · cos(8 · x) + 497420 · cos(4 · x) + 352716 - 646646 · cos(2 · x) - 319770 · cos(6 · x) - 74613 · cos(10 · x) - 7315 · cos(14 · x) - 231 · cos(18 · x) - cos(22 · x))/2097152`
   - substeps:
     1. `Usar sin²(u) = (1 - cos(2u)) / 2 repetidamente, con u = x`

## expand_trig_sum_to_product_cos_diff_general (trig_expand)

- Source: `cos(5*x)-cos(x)`
- Target: `-2*sin(3*x)*sin(2*x)`
- Result: `-2 * sin(2 * x) * sin(3 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cos(5 * x) - cos(x)
Target: -2 * sin(2 * x) * sin(3 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand cosine difference to product  [Aplicar suma a producto]
   Before: cos(5 * x) - cos(x)
   Cambio local: cos(5 * x) - cos(x) -> -2 * sin(2 * x) * sin(3 * x)
   After: -2 * sin(2 * x) * sin(3 * x)
Result: -2 * sin(2 * x) * sin(3 * x)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `cos(5 · x) - cos(x)`
   - after: `-2 · sin(2 · x) · sin(3 · x)`
   - substeps: none

## expand_trig_sum_to_product_cos_diff_xy (trig_expand)

- Source: `cos(x)-cos(y)`
- Target: `-2*sin((x+y)/2)*sin((x-y)/2)`
- Result: `-2 * sin((x + y) / 2) * sin((x - y) / 2)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cos(x) - cos(y)
Target: -2 * sin((x + y) / 2) * sin((x - y) / 2)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand cosine difference to product  [Aplicar suma a producto]
   Before: cos(x) - cos(y)
   Cambio local: cos(x) - cos(y) -> -2 * sin((x + y) / 2) * sin((x - y) / 2)
   After: -2 * sin((x + y) / 2) * sin((x - y) / 2)
Result: -2 * sin((x + y) / 2) * sin((x - y) / 2)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `cos(x) - cos(y)`
   - after: `-2 · sin((x + y)/2) · sin((x - y)/2)`
   - substeps: none

## expand_trig_sum_to_product_cos_sum_general (trig_expand)

- Source: `cos(5*x)+cos(x)`
- Target: `2*cos(3*x)*cos(2*x)`
- Result: `2 * cos(2 * x) * cos(3 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cos(x) + cos(5 * x)
Target: 2 * cos(2 * x) * cos(3 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand cosine sum to product  [Aplicar suma a producto]
   Before: cos(x) + cos(5 * x)
   Cambio local: cos(x) + cos(5 * x) -> 2 * cos(2 * x) * cos(3 * x)
   After: 2 * cos(2 * x) * cos(3 * x)
Result: 2 * cos(2 * x) * cos(3 * x)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `cos(x) + cos(5 · x)`
   - after: `2 · cos(2 · x) · cos(3 · x)`
   - substeps: none

## expand_trig_sum_to_product_cos_sum_xy (trig_expand)

- Source: `cos(x)+cos(y)`
- Target: `2*cos((x+y)/2)*cos((x-y)/2)`
- Result: `2 * cos((x + y) / 2) * cos((x - y) / 2)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cos(x) + cos(y)
Target: 2 * cos((x + y) / 2) * cos((x - y) / 2)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand cosine sum to product  [Aplicar suma a producto]
   Before: cos(x) + cos(y)
   Cambio local: cos(x) + cos(y) -> 2 * cos((x + y) / 2) * cos((x - y) / 2)
   After: 2 * cos((x + y) / 2) * cos((x - y) / 2)
Result: 2 * cos((x + y) / 2) * cos((x - y) / 2)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `cos(x) + cos(y)`
   - after: `2 · cos((x + y)/2) · cos((x - y)/2)`
   - substeps: none

## expand_trig_sum_to_product_sin_diff_general (trig_expand)

- Source: `sin(5*x)-sin(x)`
- Target: `2*cos(3*x)*sin(2*x)`
- Result: `2 * sin(2 * x) * cos(3 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(5 * x) - sin(x)
Target: 2 * sin(2 * x) * cos(3 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand sine difference to product  [Aplicar suma a producto]
   Before: sin(5 * x) - sin(x)
   Cambio local: sin(5 * x) - sin(x) -> 2 * sin(2 * x) * cos(3 * x)
   After: 2 * sin(2 * x) * cos(3 * x)
Result: 2 * sin(2 * x) * cos(3 * x)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `sin(5 · x) - sin(x)`
   - after: `2 · sin(2 · x) · cos(3 · x)`
   - substeps: none

## expand_trig_sum_to_product_sin_sum_general (trig_expand)

- Source: `sin(5*x)+sin(x)`
- Target: `2*sin(3*x)*cos(2*x)`
- Result: `2 * sin(3 * x) * cos(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(x) + sin(5 * x)
Target: 2 * sin(3 * x) * cos(2 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand sine sum to product  [Aplicar suma a producto]
   Before: sin(x) + sin(5 * x)
   Cambio local: sin(x) + sin(5 * x) -> 2 * sin(3 * x) * cos(2 * x)
   After: 2 * sin(3 * x) * cos(2 * x)
Result: 2 * sin(3 * x) * cos(2 * x)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `sin(x) + sin(5 · x)`
   - after: `2 · sin(3 · x) · cos(2 · x)`
   - substeps: none

## expand_trig_sum_to_product_sin_sum_xy (trig_expand)

- Source: `sin(x)+sin(y)`
- Target: `2*sin((x+y)/2)*cos((x-y)/2)`
- Result: `2 * sin((x + y) / 2) * cos((x - y) / 2)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(x) + sin(y)
Target: 2 * sin((x + y) / 2) * cos((x - y) / 2)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand sine sum to product  [Aplicar suma a producto]
   Before: sin(x) + sin(y)
   Cambio local: sin(x) + sin(y) -> 2 * sin((x + y) / 2) * cos((x - y) / 2)
   After: 2 * sin((x + y) / 2) * cos((x - y) / 2)
Result: 2 * sin((x + y) / 2) * cos((x - y) / 2)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `sin(x) + sin(y)`
   - after: `2 · sin((x + y)/2) · cos((x - y)/2)`
   - substeps: none

## expand_trig_tan_to_sin_cos (trig_expand)

- Source: `tan(x)`
- Target: `sin(x)/cos(x)`
- Result: `sin(x) / cos(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: tan(x)
Target: sin(x) / cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand tangent to sine over cosine  [Expandir una identidad trigonométrica]
   Before: tan(x)
   Cambio local: tan(x) -> sin(x) / cos(x)
   After: sin(x) / cos(x)
Result: sin(x) / cos(x)
ℹ️ Requires:
  • cos(x) ≠ 0
```

### Web / JSON Steps

1. `Expandir tangente como seno entre coseno`
   - before: `tan(x)`
   - after: `sin(x)/cos(x)`
   - substeps: none

## expand_trig_tangent_angle_difference (trig_expand)

- Source: `tan(x-y)`
- Target: `(tan(x)-tan(y))/(1+tan(x)*tan(y))`
- Result: `(tan(x) - tan(y)) / (tan(x) * tan(y) + 1)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: tan(x - y)
Target: (tan(x) - tan(y)) / (tan(x) * tan(y) + 1)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand tangent angle sum/difference form  [Aplicar identidad de tangente de suma/diferencia de ángulos]
   Before: tan(x - y)
   Cambio local: tan(x - y) -> (tan(x) - tan(y)) / (tan(x) * tan(y) + 1)
   After: (tan(x) - tan(y)) / (tan(x) * tan(y) + 1)
Result: (tan(x) - tan(y)) / (tan(x) * tan(y) + 1)
ℹ️ Requires:
  • tan(x) * tan(y) + 1 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de tangente de suma/diferencia de ángulos`
   - before: `tan(x - y)`
   - after: `(tan(x) - tan(y))/(tan(x) · tan(y) + 1)`
   - substeps: none

## expand_trig_tangent_angle_sum (trig_expand)

- Source: `tan(x+y)`
- Target: `(tan(x)+tan(y))/(1-tan(x)*tan(y))`
- Result: `(tan(x) + tan(y)) / (1 - tan(x) * tan(y))`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: tan(x + y)
Target: (tan(x) + tan(y)) / (1 - tan(x) * tan(y))
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand tangent angle sum/difference form  [Aplicar identidad de tangente de suma/diferencia de ángulos]
   Before: tan(x + y)
   Cambio local: tan(x + y) -> (tan(x) + tan(y)) / (1 - tan(x) * tan(y))
   After: (tan(x) + tan(y)) / (1 - tan(x) * tan(y))
Result: (tan(x) + tan(y)) / (1 - tan(x) * tan(y))
ℹ️ Requires:
  • 1 - tan(x) * tan(y) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de tangente de suma/diferencia de ángulos`
   - before: `tan(x + y)`
   - after: `(tan(x) + tan(y))/(1 - tan(x) · tan(y))`
   - substeps: none

## expand_trig_tangent_half_angle_substitution_sine (trig_expand)

- Source: `sin(x)`
- Target: `2*tan(x/2)/(1+tan(x/2)^2)`
- Result: `2 * tan(x / 2) / (tan(x / 2)^2 + 1)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(x)
Target: 2 * tan(x / 2) / (tan(x / 2)^2 + 1)
Strategy: expand trig
Steps (Aggressive Mode):
1. Rewrite sin(u) using the tangent half-angle substitution  [Aplicar identidad de tangente de ángulo mitad]
   Before: sin(x)
   Cambio local: sin(x) -> 2 * tan(x / 2) / (tan(x / 2)^(2) + 1)
   After: 2 * tan(x / 2) / (tan(x / 2)^2 + 1)
Result: 2 * tan(x / 2) / (tan(x / 2)^(2) + 1)
```

### Web / JSON Steps

1. `Aplicar identidad de tangente de ángulo mitad`
   - before: `sin(x)`
   - after: `(2 · tan(x/2))/((tan(x/2))^2 + 1)`
   - substeps: none

## expand_trig_triple_angle_cosine (trig_expand)

- Source: `cos(3*x)`
- Target: `4*cos(x)^3-3*cos(x)`
- Result: `4 * cos(x)^3 - 3 * cos(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(3 * x)
Target: 4 * cos(x)^3 - 3 * cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand or contract cosine triple-angle form  [Reescribir ángulo triple]
   Before: cos(3 * x)
   Cambio local: cos(3 * x) -> 4 * cos(x)^(3) - 3 * cos(x)
   After: 4 * cos(x)^3 - 3 * cos(x)
Result: 4 * cos(x)^(3) - 3 * cos(x)
```

### Web / JSON Steps

1. `Reescribir ángulo triple`
   - before: `cos(3 · x)`
   - after: `4 · cos(x)^3 - 3 · cos(x)`
   - substeps:
     1. `Usar cos(3u) = 4 · cos(u)^3 - 3 · cos(u), con u = x`

## expand_trig_triple_angle_sine (trig_expand)

- Source: `sin(3*x)`
- Target: `3*sin(x)-4*sin(x)^3`
- Result: `3 * sin(x) - 4 * sin(x)^3`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(3 * x)
Target: 3 * sin(x) - 4 * sin(x)^3
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand or contract sine triple-angle form  [Reescribir ángulo triple]
   Before: sin(3 * x)
   Cambio local: sin(3 * x) -> 3 * sin(x) - 4 * sin(x)^(3)
   After: 3 * sin(x) - 4 * sin(x)^3
Result: 3 * sin(x) - 4 * sin(x)^(3)
```

### Web / JSON Steps

1. `Reescribir ángulo triple`
   - before: `sin(3 · x)`
   - after: `3 · sin(x) - 4 · sin(x)^3`
   - substeps:
     1. `Usar sin(3u) = 3 · sin(u) - 4 · sin(u)^3, con u = x`

## expand_trig_triple_angle_tangent (trig_expand)

- Source: `tan(3*x)`
- Target: `(3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)`
- Result: `(3 * tan(x) - tan(x)^3) / (1 - 3 * tan(x)^2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: tan(3 * x)
Target: (3 * tan(x) - tan(x)^3) / (1 - 3 * tan(x)^2)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand or contract tangent triple-angle form  [Reescribir ángulo triple]
   Before: tan(3 * x)
   Cambio local: tan(3 * x) -> (3 * tan(x) - tan(x)^(3)) / (1 - 3 * tan(x)^(2))
   After: (3 * tan(x) - tan(x)^3) / (1 - 3 * tan(x)^2)
Result: (3 * tan(x) - tan(x)^(3)) / (1 - 3 * tan(x)^(2))
ℹ️ Requires:
  • 1 - 3 * tan(x)^2 ≠ 0
```

### Web / JSON Steps

1. `Reescribir ángulo triple`
   - before: `tan(3 · x)`
   - after: `(3 · tan(x) - tan(x)^3)/(1 - 3 · tan(x)^2)`
   - substeps:
     1. `Usar tan(3u) = (3 · tan(u) - tan(u)^3) / (1 - 3 · tan(u)^2), con u = x`

## factor_alternating_cubic_vandermonde (factor)

- Source: `a^3*(b-c) + b^3*(c-a) + c^3*(a-b)`
- Target: `(a-b)*(a-c)*(b-c)*(a+b+c)`
- Result: `(a + b + c) * (a - b) * (a - c) * (b - c)`
- Web step count: `1`
- Web substep count: `4`
- Flags: none

### CLI

```text
Parsed: a^3 * (b - c) + b^3 * (c - a) + c^3 * (a - b)
Target: (a + b + c) * (a - b) * (a - c) * (b - c)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: a^(3) * (b - c) + b^(3) * (c - a) + c^(3) * (a - b)
   Cambio local: a^(3) * (b - c) + b^(3) * (c - a) + c^(3) * (a - b) -> (a + b + c) * (a - b) * (a - c) * (b - c)
   After: (a + b + c) * (a - b) * (a - c) * (b - c)
Result: (a + b + c) * (a - b) * (a - c) * (b - c)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a^3 · (b - c) + b^3 · (c - a) + c^3 · (a - b)`
   - after: `(a + b + c) · (a - b) · (a - c) · (b - c)`
   - substeps:
     1. `Si a = b, aparece el factor a - b`
     2. `Si a = c, aparece el factor a - c`
     3. `Si b = c, aparece el factor b - c`
     4. `El cociente restante es a + b + c`

## factor_common_factor_sum (factor)

- Source: `a*b + a*c`
- Target: `a*(b+c)`
- Result: `a * (b + c)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a * b + a * c
Target: a * (b + c)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: a * b + a * c
   Cambio local: a * b + a * c -> a * (b + c)
   After: a * (b + c)
Result: a * (b + c)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a · b + a · c`
   - after: `a · (b + c)`
   - substeps:
     1. `Aquí el factor común es a`

## factor_common_factor_sum_three_terms (factor)

- Source: `a*x + b*x + c*x`
- Target: `x*(a+b+c)`
- Result: `x * (a + b + c)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a * x + b * x + c * x
Target: x * (a + b + c)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: a * x + b * x + c * x
   Cambio local: a * x + b * x + c * x -> x * (a + b + c)
   After: x * (a + b + c)
Result: x * (a + b + c)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a · x + b · x + c · x`
   - after: `x · (a + b + c)`
   - substeps:
     1. `Aquí el factor común es x`

## factor_difference_cubes (factor)

- Source: `a^3-b^3`
- Target: `(a-b)*(a^2+a*b+b^2)`
- Result: `(a^2 + b^2 + a * b) * (a - b)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a^3 - b^3
Target: (a^2 + b^2 + a * b) * (a - b)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: a^(3) - b^(3)
   Cambio local: a^(3) - b^(3) -> (a^(2) + b^(2) + a * b) * (a - b)
   After: (a^2 + b^2 + a * b) * (a - b)
Result: (a^(2) + b^(2) + a * b) * (a - b)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a^3 - b^3`
   - after: `(a^2 + b^2 + a · b) · (a - b)`
   - substeps:
     1. `Reconocer la forma a^3 - b^3`

## factor_difference_squares (factor)

- Source: `a^2 - b^2`
- Target: `(a - b)*(a + b)`
- Result: `(a + b) * (a - b)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a^2 - b^2
Target: (a + b) * (a - b)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: a^(2) - b^(2)
   Cambio local: a^(2) - b^(2) -> (a + b) * (a - b)
   After: (a + b) * (a - b)
Result: (a + b) * (a - b)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a^2 - b^2`
   - after: `(a + b) · (a - b)`
   - substeps:
     1. `Aquí la diferencia de cuadrados usa bases a y b`

## factor_difference_squares_with_passthrough (factor)

- Source: `a+x^2-1`
- Target: `a+(x-1)*(x+1)`
- Result: `(x + 1) * (x - 1) + a`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: x^2 + a - 1
Target: (x + 1) * (x - 1) + a
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: x^(2) + a - 1
   Cambio local: x^(2) - 1 -> (x + 1) * (x - 1)
   After: (x + 1) * (x - 1) + a
Result: (x + 1) * (x - 1) + a
```

### Web / JSON Steps

1. `Factorizar`
   - before: `x^2 + a - 1`
   - after: `(x + 1) · (x - 1) + a`
   - substeps:
     1. `Aquí la diferencia de potencias usa base x y exponente 2`

## factor_full_cyclotomic_sixth_power_difference (factor)

- Source: `x^6 - 1`
- Target: `(x^2+x+1)*(x^2-x+1)*(x+1)*(x-1)`
- Result: `(x + 1) * (x^2 - x + 1) * (x^2 + x + 1) * (x - 1)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: x^6 - 1
Target: (x + 1) * (x^2 - x + 1) * (x^2 + x + 1) * (x - 1)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: x^(6) - 1
   Cambio local: x^(6) - 1 -> (x + 1) * (x^(2) - x + 1) * (x^(2) + x + 1) * (x - 1)
   After: (x + 1) * (x^2 - x + 1) * (x^2 + x + 1) * (x - 1)
Result: (x + 1) * (x^(2) - x + 1) * (x^(2) + x + 1) * (x - 1)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `x^6 - 1`
   - after: `(x + 1) · (x^2 - x + 1) · (x^2 + x + 1) · (x - 1)`
   - substeps:
     1. `Aquí la diferencia de sexto grado se factoriza completamente con base x`

## factor_geometric_difference_power_6 (factor)

- Source: `x^6 - 1`
- Target: `(x-1)*(x^5 + x^4 + x^3 + x^2 + x + 1)`
- Result: `(x^5 + x^4 + x^3 + x^2 + x + 1) * (x - 1)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: x^6 - 1
Target: (x^5 + x^4 + x^3 + x^2 + x + 1) * (x - 1)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: x^(6) - 1
   Cambio local: x^(6) - 1 -> (x^(5) + x^(4) + x^(3) + x^(2) + x + 1) * (x - 1)
   After: (x^5 + x^4 + x^3 + x^2 + x + 1) * (x - 1)
Result: (x^(5) + x^(4) + x^(3) + x^(2) + x + 1) * (x - 1)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `x^6 - 1`
   - after: `(x^5 + x^4 + x^3 + x^2 + x + 1) · (x - 1)`
   - substeps:
     1. `Aquí la diferencia de potencias usa base x y exponente 6`

## factor_out_cube_with_division_septic (conditional_factor)

- Source: `a*x^7 + b*x^5 + c*x^3 + d`
- Target: `x^3*(a*x^4 + b*x^2 + c + d/x^3)`
- Result: `x^3 * (d / x^3 + a * x^4 + b * x^2 + c)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * x^7 + b * x^5 + c * x^3 + d
Target: x^3 * (d / x^3 + a * x^4 + b * x^2 + c)
Strategy: factor out with division
Steps (Aggressive Mode):
1. Factor out x^3 from the whole expression  [Sacar factor usando división]
   Before: a * x^(7) + b * x^(5) + c * x^(3) + d
   Cambio local: a * x^(7) + b * x^(5) + c * x^(3) + d -> x^(3) * (d / x^(3) + a * x^(4) + b * x^(2) + c)
   After: x^3 * (d / x^3 + a * x^4 + b * x^2 + c)
Result: x^(3) * (d / x^(3) + a * x^(4) + b * x^(2) + c)
ℹ️ Requires:
  • x ≠ 0
```

### Web / JSON Steps

1. `Sacar factor usando división`
   - before: `a · x^7 + b · x^5 + c · x^3 + d`
   - after: `x^3 · (d/x^3 + a · x^4 + b · x^2 + c)`
   - substeps:
     1. `Reescribir cada término con el factor común x^3`
     2. `Sacar el factor común x^3`

## factor_out_square_with_division_quartic (conditional_factor)

- Source: `a*x^4 + b*x^3 + c*x^2 + d`
- Target: `x^2*(a*x^2 + b*x + c + d/x^2)`
- Result: `x^2 * (d / x^2 + a * x^2 + b * x + c)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * x^4 + b * x^3 + c * x^2 + d
Target: x^2 * (d / x^2 + a * x^2 + b * x + c)
Strategy: factor out with division
Steps (Aggressive Mode):
1. Factor out x^2 from the whole expression  [Sacar factor usando división]
   Before: a * x^(4) + b * x^(3) + c * x^(2) + d
   Cambio local: a * x^(4) + b * x^(3) + c * x^(2) + d -> x^(2) * (d / x^(2) + a * x^(2) + b * x + c)
   After: x^2 * (d / x^2 + a * x^2 + b * x + c)
Result: x^(2) * (d / x^(2) + a * x^(2) + b * x + c)
ℹ️ Requires:
  • x ≠ 0
```

### Web / JSON Steps

1. `Sacar factor usando división`
   - before: `a · x^4 + b · x^3 + c · x^2 + d`
   - after: `x^2 · (d/x^2 + a · x^2 + b · x + c)`
   - substeps:
     1. `Reescribir cada término con el factor común x^2`
     2. `Sacar el factor común x^2`

## factor_out_with_division (conditional_factor)

- Source: `a*x + b*x + c`
- Target: `x*(a + b + c/x)`
- Result: `x * (c / x + a + b)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * x + b * x + c
Target: x * (c / x + a + b)
Strategy: factor out with division
Steps (Aggressive Mode):
1. Factor out x from the whole expression  [Sacar factor usando división]
   Before: a * x + b * x + c
   Cambio local: a * x + b * x + c -> x * (c / x + a + b)
   After: x * (c / x + a + b)
Result: x * (c / x + a + b)
ℹ️ Requires:
  • x ≠ 0
```

### Web / JSON Steps

1. `Sacar factor usando división`
   - before: `a · x + b · x + c`
   - after: `x · (c/x + a + b)`
   - substeps:
     1. `Reescribir cada término con el factor común x`
     2. `Sacar el factor común x`

## factor_out_with_division_mixed_septic (conditional_factor)

- Source: `a*x^7 + b*x^5 + c*x^2 + d`
- Target: `x*(a*x^6 + b*x^4 + c*x + d/x)`
- Result: `x * (d / x + a * x^6 + b * x^4 + c * x)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * x^7 + b * x^5 + c * x^2 + d
Target: x * (d / x + a * x^6 + b * x^4 + c * x)
Strategy: factor out with division
Steps (Aggressive Mode):
1. Factor out x from the whole expression  [Sacar factor usando división]
   Before: a * x^(7) + b * x^(5) + c * x^(2) + d
   Cambio local: a * x^(7) + b * x^(5) + c * x^(2) + d -> x * (d / x + a * x^(6) + b * x^(4) + c * x)
   After: x * (d / x + a * x^6 + b * x^4 + c * x)
Result: x * (d / x + a * x^(6) + b * x^(4) + c * x)
ℹ️ Requires:
  • x ≠ 0
```

### Web / JSON Steps

1. `Sacar factor usando división`
   - before: `a · x^7 + b · x^5 + c · x^2 + d`
   - after: `x · (d/x + a · x^6 + b · x^4 + c · x)`
   - substeps:
     1. `Reescribir cada término con el factor común x`
     2. `Sacar el factor común x`

## factor_out_with_division_quadratic (conditional_factor)

- Source: `a*x^2 + b*x + c`
- Target: `x*(a*x + b + c/x)`
- Result: `x * (c / x + a * x + b)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * x^2 + b * x + c
Target: x * (c / x + a * x + b)
Strategy: factor out with division
Steps (Aggressive Mode):
1. Factor out x from the whole expression  [Sacar factor usando división]
   Before: a * x^(2) + b * x + c
   Cambio local: a * x^(2) + b * x + c -> x * (c / x + a * x + b)
   After: x * (c / x + a * x + b)
Result: x * (c / x + a * x + b)
ℹ️ Requires:
  • x ≠ 0
```

### Web / JSON Steps

1. `Sacar factor usando división`
   - before: `a · x^2 + b · x + c`
   - after: `x · (c/x + a · x + b)`
   - substeps:
     1. `Reescribir cada término con el factor común x`
     2. `Sacar el factor común x`

## factor_out_with_division_sparse_quintic (conditional_factor)

- Source: `a*x^5 + b*x^3 + c*x + d`
- Target: `x*(a*x^4 + b*x^2 + c + d/x)`
- Result: `x * (d / x + a * x^4 + b * x^2 + c)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * x^5 + b * x^3 + c * x + d
Target: x * (d / x + a * x^4 + b * x^2 + c)
Strategy: factor out with division
Steps (Aggressive Mode):
1. Factor out x from the whole expression  [Sacar factor usando división]
   Before: a * x^(5) + b * x^(3) + c * x + d
   Cambio local: a * x^(5) + b * x^(3) + c * x + d -> x * (d / x + a * x^(4) + b * x^(2) + c)
   After: x * (d / x + a * x^4 + b * x^2 + c)
Result: x * (d / x + a * x^(4) + b * x^(2) + c)
ℹ️ Requires:
  • x ≠ 0
```

### Web / JSON Steps

1. `Sacar factor usando división`
   - before: `a · x^5 + b · x^3 + c · x + d`
   - after: `x · (d/x + a · x^4 + b · x^2 + c)`
   - substeps:
     1. `Reescribir cada término con el factor común x`
     2. `Sacar el factor común x`

## factor_perfect_square_trinomial (factor)

- Source: `x^2 + 2*x + 1`
- Target: `(x + 1)^2`
- Result: `(x + 1)^2`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: x^2 + 2 * x + 1
Target: (x + 1)^2
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: x^(2) + 2 * x + 1
   Cambio local: x^(2) + 2 * x + 1 -> (x + 1)^(2)
   After: (x + 1)^2
Result: (x + 1)^(2)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `x^2 + 2 · x + 1`
   - after: `(x + 1)^2`
   - substeps:
     1. `Usar a^2 + 2ab + b^2 = (a + b)^2`

## factor_perfect_square_trinomial_minus (factor)

- Source: `a^2 - 2*a*b + b^2`
- Target: `(a - b)^2`
- Result: `(a - b)^2`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a^2 + b^2 - 2 * a * b
Target: (a - b)^2
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: a^(2) + b^(2) - 2 * a * b
   Cambio local: a^(2) + b^(2) - 2 * a * b -> (a - b)^(2)
   After: (a - b)^2
Result: (a - b)^(2)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a^2 - 2 · a · b + b^2`
   - after: `(a - b)^2`
   - substeps:
     1. `Usar a^2 - 2ab + b^2 = (a - b)^2`

## factor_perfect_square_trinomial_symbolic (factor)

- Source: `a^2 + 2*a*b + b^2`
- Target: `(a + b)^2`
- Result: `(a + b)^2`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a^2 + b^2 + 2 * a * b
Target: (a + b)^2
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: a^(2) + b^(2) + 2 * a * b
   Cambio local: a^(2) + b^(2) + 2 * a * b -> (a + b)^(2)
   After: (a + b)^2
Result: (a + b)^(2)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a^2 + b^2 + 2 · a · b`
   - after: `(a + b)^2`
   - substeps:
     1. `Usar a^2 + 2ab + b^2 = (a + b)^2`

## factor_sophie_germain (factor)

- Source: `x^4 + 4*y^4`
- Target: `(x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2)`
- Result: `(x^2 + 2 * y^2 - 2 * x * y) * (x^2 + 2 * y^2 + 2 * x * y)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: x^4 + 4 * y^4
Target: (x^2 + 2 * y^2 - 2 * x * y) * (x^2 + 2 * y^2 + 2 * x * y)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: x^(4) + 4 * y^(4)
   Cambio local: x^(4) + 4 * y^(4) -> (x^(2) + 2 * y^(2) - 2 * x * y) * (x^(2) + 2 * y^(2) + 2 * x * y)
   After: (x^2 + 2 * y^2 - 2 * x * y) * (x^2 + 2 * y^2 + 2 * x * y)
Result: (x^(2) + 2 * y^(2) - 2 * x * y) * (x^(2) + 2 * y^(2) + 2 * x * y)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `x^4 + 4 · y^4`
   - after: `(x^2 - 2 · x · y + 2 · y^2) · (x^2 + 2 · y^2 + 2 · x · y)`
   - substeps:
     1. `Convertir la suma en diferencia de cuadrados`
     2. `Factorizar la diferencia de cuadrados`

## factor_sum_cubes (factor)

- Source: `a^3+b^3`
- Target: `(a+b)*(a^2-a*b+b^2)`
- Result: `(a + b) * (a^2 + b^2 - a * b)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a^3 + b^3
Target: (a + b) * (a^2 + b^2 - a * b)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: a^(3) + b^(3)
   Cambio local: a^(3) + b^(3) -> (a + b) * (a^(2) + b^(2) - a * b)
   After: (a + b) * (a^2 + b^2 - a * b)
Result: (a + b) * (a^(2) + b^(2) - a * b)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a^3 + b^3`
   - after: `(a + b) · (a^2 - a · b + b^2)`
   - substeps:
     1. `Reconocer la forma a^3 + b^3`

## factor_symbolic_binomial_cube (factor)

- Source: `a^3 + 3*a^2*b + 3*a*b^2 + b^3`
- Target: `(a+b)^3`
- Result: `(a + b)^3`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a^3 + b^3 + 3 * a * b^2 + 3 * b * a^2
Target: (a + b)^3
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: a^(3) + b^(3) + 3 * a * b^(2) + 3 * b * a^(2)
   Cambio local: a^(3) + b^(3) + 3 * a * b^(2) + 3 * b * a^(2) -> (a + b)^(3)
   After: (a + b)^3
Result: (a + b)^(3)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a^3 + b^3 + 3 · a · b^2 + 3 · b · a^2`
   - after: `(a + b)^3`
   - substeps:
     1. `Usar a^3 + 3a^2b + 3ab^2 + b^3 = (a + b)^3`

## factor_symbolic_binomial_cube_minus (factor)

- Source: `a^3 - 3*a^2*b + 3*a*b^2 - b^3`
- Target: `(a-b)^3`
- Result: `(a - b)^3`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a^3 + 3 * a * b^2 - 3 * b * a^2 - b^3
Target: (a - b)^3
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: a^(3) + 3 * a * b^(2) - 3 * b * a^(2) - b^(3)
   Cambio local: a^(3) + 3 * a * b^(2) - 3 * b * a^(2) - b^(3) -> (a - b)^(3)
   After: (a - b)^3
Result: (a - b)^(3)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a^3 - 3 · b · a^2 + 3 · a · b^2 - b^3`
   - after: `(a - b)^3`
   - substeps:
     1. `Usar a^3 - 3a^2b + 3ab^2 - b^3 = (a - b)^3`

## factor_symbolic_sixth_power_difference (factor)

- Source: `x^6-a^6`
- Target: `(x^2-a^2)*(x^4+a^2*x^2+a^4)`
- Result: `(a^4 + x^4 + a^2 * x^2) * (x^2 - a^2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: x^6 - a^6
Target: (a^4 + x^4 + a^2 * x^2) * (x^2 - a^2)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: x^(6) - a^(6)
   Cambio local: x^(6) - a^(6) -> (a^(4) + x^(4) + a^(2) * x^(2)) * (x^(2) - a^(2))
   After: (a^4 + x^4 + a^2 * x^2) * (x^2 - a^2)
Result: (a^(4) + x^(4) + a^(2) * x^(2)) * (x^(2) - a^(2))
```

### Web / JSON Steps

1. `Factorizar`
   - before: `x^6 - a^6`
   - after: `(a^4 + x^4 + a^2 · x^2) · (x^2 - a^2)`
   - substeps:
     1. `Aplicar a^6 - b^6 = (a^2 - b^2)(a^4 + a^2b^2 + b^4)`

## factor_symbolic_sixth_power_sum (factor)

- Source: `x^6+a^6`
- Target: `(x^2+a^2)*(x^4-a^2*x^2+a^4)`
- Result: `(a^2 + x^2) * (a^4 + x^4 - a^2 * x^2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a^6 + x^6
Target: (a^2 + x^2) * (a^4 + x^4 - a^2 * x^2)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorizar]
   Before: a^(6) + x^(6)
   Cambio local: a^(6) + x^(6) -> (a^(2) + x^(2)) * (a^(4) + x^(4) - a^(2) * x^(2))
   After: (a^2 + x^2) * (a^4 + x^4 - a^2 * x^2)
Result: (a^(2) + x^(2)) * (a^(4) + x^(4) - a^(2) * x^(2))
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a^6 + x^6`
   - after: `(a^2 + x^2) · (x^4 - a^2 · x^2 + a^4)`
   - substeps:
     1. `Aplicar a^6 + b^6 = (a^2 + b^2)(a^4 - a^2b^2 + b^4)`

## finite_aggregate_product_constant_symbolic (finite_aggregate)

- Source: `product(c, k, 1, n)`
- Target: `c^n`
- Result: `c^n`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: product(c, k, 1, n)
Target: c^n
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Product of constant factor: Π(c, k) from 1 to n  [Aplicar producto de constante]
   Before: product(c, k, 1, n)
   Cambio local: product(c, k, 1, n) -> c^(n)
   After: c^n
Result: c^(n)
```

### Web / JSON Steps

1. `Aplicar producto de constante`
   - before: `prod_k=1^n c`
   - after: `c^n`
   - substeps:
     1. `Escribir el producto con sus extremos`
     2. `Contar factores iguales en el producto`

## finite_aggregate_product_constant_symbolic_lower_bound (finite_aggregate)

- Source: `product(c, k, m, n)`
- Target: `c^(n-m+1)`
- Result: `c^(-m + n + 1)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: product(c, k, m, n)
Target: c^(-m + n + 1)
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Product of constant factor: Π(c, k) from m to n  [Aplicar producto de constante]
   Before: product(c, k, m, n)
   Cambio local: product(c, k, m, n) -> c^(-m + n + 1)
   After: c^(-m + n + 1)
Result: c^(-m + n + 1)
```

### Web / JSON Steps

1. `Aplicar producto de constante`
   - before: `prod_k=m^n c`
   - after: `c^(n - m + 1)`
   - substeps:
     1. `Escribir el producto con sus extremos`
     2. `Contar factores iguales en el producto`

## finite_aggregate_product_first_integers_symbolic (finite_aggregate)

- Source: `product(k, k, 1, n)`
- Target: `n!`
- Result: `n!`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: product(k, k, 1, n)
Target: n!
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Product of first integers: Π(k, k) from 1 to n  [Aplicar producto factorial]
   Before: product(k, k, 1, n)
   Cambio local: product(k, k, 1, n) -> n!
   After: n!
Result: n!
```

### Web / JSON Steps

1. `Aplicar producto factorial`
   - before: `prod_k=1^n k`
   - after: `n!`
   - substeps:
     1. `Escribir el producto con sus extremos`
     2. `Usar factorial para el producto de enteros consecutivos`

## finite_aggregate_product_first_integers_symbolic_lower_bound (finite_aggregate)

- Source: `product(k, k, m, n)`
- Target: `n!/(m-1)!`
- Result: `n! / (m - 1)!`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: product(k, k, m, n)
Target: n! / (m - 1)!
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Product of first integers: Π(k, k) from m to n  [Aplicar producto factorial]
   Before: product(k, k, m, n)
   Cambio local: product(k, k, m, n) -> n! / (m - 1)!
   After: n! / (m - 1)!
Result: n! / (m - 1)!
ℹ️ Requires:
  • (m - 1)! ≠ 0
```

### Web / JSON Steps

1. `Aplicar producto factorial`
   - before: `prod_k=m^n k`
   - after: `n!/(m - 1)!`
   - substeps:
     1. `Escribir el producto con sus extremos`
     2. `Usar factorial para el producto de enteros consecutivos`

## finite_aggregate_product_of_cubes_symbolic_lower_bound (finite_aggregate)

- Source: `product(k^3, k, m, n)`
- Target: `(n!/(m-1)!)^3`
- Result: `(n! / (m - 1)!)^3`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: product(k^3, k, m, n)
Target: (n! / (m - 1)!)^3
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Product of powers: Π(k^3, k) from m to n  [Aplicar producto de potencias]
   Before: product(k^(3), k, m, n)
   Cambio local: product(k^(3), k, m, n) -> (n! / (m - 1)!)^(3)
   After: (n! / (m - 1)!)^3
Result: (n! / (m - 1)!)^(3)
ℹ️ Requires:
  • (m - 1)! ≠ 0
```

### Web / JSON Steps

1. `Aplicar producto de potencias`
   - before: `prod_k=m^n k^3`
   - after: `(n!/(m - 1)!)^3`
   - substeps:
     1. `Escribir el producto con sus extremos`
     2. `Convertir el producto de potencias en potencia de factoriales`

## finite_aggregate_product_of_squares_symbolic (finite_aggregate)

- Source: `product(k^2, k, 1, n)`
- Target: `(n!)^2`
- Result: `n!^2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: product(k^2, k, 1, n)
Target: n!^2
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Product of powers: Π(k^2, k) from 1 to n  [Aplicar producto de potencias]
   Before: product(k^(2), k, 1, n)
   Cambio local: product(k^(2), k, 1, n) -> n!^(2)
   After: n!^2
Result: n!^(2)
```

### Web / JSON Steps

1. `Aplicar producto de potencias`
   - before: `prod_k=1^n k^2`
   - after: `n!^2`
   - substeps:
     1. `Escribir el producto con sus extremos`
     2. `Convertir el producto de potencias en potencia de factoriales`

## finite_aggregate_product_of_squares_symbolic_lower_bound (finite_aggregate)

- Source: `product(k^2, k, m, n)`
- Target: `(n!/(m-1)!)^2`
- Result: `(n! / (m - 1)!)^2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: product(k^2, k, m, n)
Target: (n! / (m - 1)!)^2
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Product of powers: Π(k^2, k) from m to n  [Aplicar producto de potencias]
   Before: product(k^(2), k, m, n)
   Cambio local: product(k^(2), k, m, n) -> (n! / (m - 1)!)^(2)
   After: (n! / (m - 1)!)^2
Result: (n! / (m - 1)!)^(2)
ℹ️ Requires:
  • (m - 1)! ≠ 0
```

### Web / JSON Steps

1. `Aplicar producto de potencias`
   - before: `prod_k=m^n k^2`
   - after: `(n!/(m - 1)!)^2`
   - substeps:
     1. `Escribir el producto con sus extremos`
     2. `Convertir el producto de potencias en potencia de factoriales`

## finite_aggregate_sum_constant_symbolic (finite_aggregate)

- Source: `sum(c, k, 1, n)`
- Target: `c*n`
- Result: `c * n`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sum(c, k, 1, n)
Target: c * n
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Sum of constant term: Σ(c, k) from 1 to n  [Aplicar suma de constante]
   Before: sum(c, k, 1, n)
   Cambio local: sum(c, k, 1, n) -> c * n
   After: c * n
Result: c * n
```

### Web / JSON Steps

1. `Aplicar suma de constante`
   - before: `sum_k=1^n c`
   - after: `c · n`
   - substeps:
     1. `Escribir la suma con sus extremos`
     2. `Contar términos iguales en la suma`

## finite_aggregate_sum_constant_symbolic_lower_bound (finite_aggregate)

- Source: `sum(c, k, m, n)`
- Target: `c*(n-m+1)`
- Result: `c * (-m + n + 1)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sum(c, k, m, n)
Target: c * (-m + n + 1)
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Sum of constant term: Σ(c, k) from m to n  [Aplicar suma de constante]
   Before: sum(c, k, m, n)
   Cambio local: sum(c, k, m, n) -> c * (-m + n + 1)
   After: c * (-m + n + 1)
Result: c * (-m + n + 1)
```

### Web / JSON Steps

1. `Aplicar suma de constante`
   - before: `sum_k=m^n c`
   - after: `c · (n - m + 1)`
   - substeps:
     1. `Escribir la suma con sus extremos`
     2. `Contar términos iguales en la suma`

## finite_aggregate_sum_first_integers_symbolic (finite_aggregate)

- Source: `sum(k, k, 1, n)`
- Target: `n*(n+1)/2`
- Result: `n * (n + 1) / 2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sum(k, k, 1, n)
Target: n * (n + 1) / 2
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Sum of first integers: Σ(k, k) from 1 to n  [Aplicar fórmula de suma de enteros]
   Before: sum(k, k, 1, n)
   Cambio local: sum(k, k, 1, n) -> n * (n + 1) / 2
   After: n * (n + 1) / 2
Result: n * (n + 1) / 2
```

### Web / JSON Steps

1. `Aplicar fórmula de suma de enteros`
   - before: `sum_k=1^n k`
   - after: `(n · (n + 1))/2`
   - substeps:
     1. `Escribir la suma con sus extremos`
     2. `Usar la fórmula cerrada para la suma de enteros`

## finite_aggregate_sum_first_integers_symbolic_lower_bound (finite_aggregate)

- Source: `sum(k, k, m, n)`
- Target: `(n*(n+1)-m*(m-1))/2`
- Result: `(n * (n + 1) - m * (m - 1)) / 2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sum(k, k, m, n)
Target: (n * (n + 1) - m * (m - 1)) / 2
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Sum of first integers: Σ(k, k) from m to n  [Aplicar fórmula de suma de enteros]
   Before: sum(k, k, m, n)
   Cambio local: sum(k, k, m, n) -> (n * (n + 1) - m * (m - 1)) / 2
   After: (n * (n + 1) - m * (m - 1)) / 2
Result: (n * (n + 1) - m * (m - 1)) / 2
```

### Web / JSON Steps

1. `Aplicar fórmula de suma de enteros`
   - before: `sum_k=m^n k`
   - after: `(n · (n + 1) - m · (m - 1))/2`
   - substeps:
     1. `Escribir la suma con sus extremos`
     2. `Usar la fórmula cerrada para la suma de enteros`

## finite_aggregate_sum_geometric_power_base_two_symbolic (finite_aggregate)

- Source: `sum(2^k, k, 0, n)`
- Target: `2^(n+1)-1`
- Result: `2^(n + 1) - 1`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sum(2^k, k, 0, n)
Target: 2^(n + 1) - 1
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Geometric sum: Σ(2^k, k) from 0 to n  [Aplicar fórmula de suma geométrica]
   Before: sum(2^(k), k, 0, n)
   Cambio local: sum(2^(k), k, 0, n) -> 2^(n + 1) - 1
   After: 2^(n + 1) - 1
Result: 2^(n + 1) - 1
```

### Web / JSON Steps

1. `Aplicar fórmula de suma geométrica`
   - before: `sum_k=0^n 2^k`
   - after: `2^(n + 1) - 1`
   - substeps:
     1. `Escribir la suma con sus extremos`
     2. `Usar la fórmula cerrada para la suma geométrica`

## finite_aggregate_sum_geometric_power_base_two_symbolic_lower_bound (finite_aggregate)

- Source: `sum(2^k, k, m, n)`
- Target: `2^(n+1)-2^m`
- Result: `2^(n + 1) - 2^m`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sum(2^k, k, m, n)
Target: 2^(n + 1) - 2^m
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Geometric sum: Σ(2^k, k) from m to n  [Aplicar fórmula de suma geométrica]
   Before: sum(2^(k), k, m, n)
   Cambio local: sum(2^(k), k, m, n) -> 2^(n + 1) - 2^(m)
   After: 2^(n + 1) - 2^m
Result: 2^(n + 1) - 2^(m)
```

### Web / JSON Steps

1. `Aplicar fórmula de suma geométrica`
   - before: `sum_k=m^n 2^k`
   - after: `2^(n + 1) - 2^m`
   - substeps:
     1. `Escribir la suma con sus extremos`
     2. `Usar la fórmula cerrada para la suma geométrica`

## finite_aggregate_sum_of_cubes_symbolic (finite_aggregate)

- Source: `sum(k^3, k, 1, n)`
- Target: `(n*(n+1)/2)^2`
- Result: `(n * (n + 1) / 2)^2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sum(k^3, k, 1, n)
Target: (n * (n + 1) / 2)^2
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Sum of cubes: Σ(k^3, k) from 1 to n  [Aplicar fórmula de suma de cubos]
   Before: sum(k^(3), k, 1, n)
   Cambio local: sum(k^(3), k, 1, n) -> (n * (n + 1) / 2)^(2)
   After: (n * (n + 1) / 2)^2
Result: (n * (n + 1) / 2)^(2)
```

### Web / JSON Steps

1. `Aplicar fórmula de suma de cubos`
   - before: `sum_k=1^n k^3`
   - after: `((n · (n + 1))/2)^2`
   - substeps:
     1. `Escribir la suma con sus extremos`
     2. `Usar la fórmula cerrada para la suma de cubos`

## finite_aggregate_sum_of_cubes_symbolic_lower_bound (finite_aggregate)

- Source: `sum(k^3, k, m, n)`
- Target: `(n*(n+1)/2)^2-(m*(m-1)/2)^2`
- Result: `(n * (n + 1) / 2)^2 - (m * (m - 1) / 2)^2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sum(k^3, k, m, n)
Target: (n * (n + 1) / 2)^2 - (m * (m - 1) / 2)^2
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Sum of cubes: Σ(k^3, k) from m to n  [Aplicar fórmula de suma de cubos]
   Before: sum(k^(3), k, m, n)
   Cambio local: sum(k^(3), k, m, n) -> (n * (n + 1) / 2)^(2) - (m * (m - 1) / 2)^(2)
   After: (n * (n + 1) / 2)^2 - (m * (m - 1) / 2)^2
Result: (n * (n + 1) / 2)^(2) - (m * (m - 1) / 2)^(2)
```

### Web / JSON Steps

1. `Aplicar fórmula de suma de cubos`
   - before: `sum_k=m^n k^3`
   - after: `((n · (n + 1))/2)^2 - ((m · (m - 1))/2)^2`
   - substeps:
     1. `Escribir la suma con sus extremos`
     2. `Usar la fórmula cerrada para la suma de cubos`

## finite_aggregate_sum_of_squares_symbolic (finite_aggregate)

- Source: `sum(k^2, k, 1, n)`
- Target: `n*(n+1)*(2*n+1)/6`
- Result: `n * (n + 1) * (2 * n + 1) / 6`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sum(k^2, k, 1, n)
Target: n * (n + 1) * (2 * n + 1) / 6
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Sum of squares: Σ(k^2, k) from 1 to n  [Aplicar fórmula de suma de cuadrados]
   Before: sum(k^(2), k, 1, n)
   Cambio local: sum(k^(2), k, 1, n) -> n * (n + 1) * (2 * n + 1) / 6
   After: n * (n + 1) * (2 * n + 1) / 6
Result: n * (n + 1) * (2 * n + 1) / 6
```

### Web / JSON Steps

1. `Aplicar fórmula de suma de cuadrados`
   - before: `sum_k=1^n k^2`
   - after: `(n · (n + 1) · (2 · n + 1))/6`
   - substeps:
     1. `Escribir la suma con sus extremos`
     2. `Usar la fórmula cerrada para la suma de cuadrados`

## finite_aggregate_sum_of_squares_symbolic_lower_bound (finite_aggregate)

- Source: `sum(k^2, k, m, n)`
- Target: `(n*(n+1)*(2*n+1)-m*(m-1)*(2*m-1))/6`
- Result: `(n * (n + 1) * (2 * n + 1) - m * (m - 1) * (2 * m - 1)) / 6`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sum(k^2, k, m, n)
Target: (n * (n + 1) * (2 * n + 1) - m * (m - 1) * (2 * m - 1)) / 6
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Sum of squares: Σ(k^2, k) from m to n  [Aplicar fórmula de suma de cuadrados]
   Before: sum(k^(2), k, m, n)
   Cambio local: sum(k^(2), k, m, n) -> (n * (n + 1) * (2 * n + 1) - m * (m - 1) * (2 * m - 1)) / 6
   After: (n * (n + 1) * (2 * n + 1) - m * (m - 1) * (2 * m - 1)) / 6
Result: (n * (n + 1) * (2 * n + 1) - m * (m - 1) * (2 * m - 1)) / 6
```

### Web / JSON Steps

1. `Aplicar fórmula de suma de cuadrados`
   - before: `sum_k=m^n k^2`
   - after: `(n · (n + 1) · (2 · n + 1) - m · (m - 1) · (2 · m - 1))/6`
   - substeps:
     1. `Escribir la suma con sus extremos`
     2. `Usar la fórmula cerrada para la suma de cuadrados`

## finite_telescoping_product_affine_symbolic_shift_symbolic_lower (finite_telescoping)

- Source: `product((a*k+b+a)/(a*k+b), k, m, n)`
- Target: `(a*n+a+b)/(a*m+b)`
- Result: `(a * n + a + b) / (a * m + b)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: product((a * k + a + b) / (a * k + b), k, m, n)
Target: (a * n + a + b) / (a * m + b)
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Telescoping product: Π((a * k + a + b) / (a * k + b), k) from m to n  [Evaluar producto telescópico finito]
   Before: product((a * k + a + b) / (a * k + b), k, m, n)
   Cambio local: product((a * k + a + b) / (a * k + b), k, m, n) -> (a * n + a + b) / (a * m + b)
   After: (a * n + a + b) / (a * m + b)
Result: (a * n + a + b) / (a * m + b)
ℹ️ Requires:
  • a * k + b ≠ 0
  • a * m + b ≠ 0
```

### Web / JSON Steps

1. `Evaluar producto telescópico finito`
   - before: `prod_k=m^n (a · k + a + b)/(a · k + b)`
   - after: `(a · n + a + b)/(a · m + b)`
   - substeps:
     1. `Escribir los primeros y últimos factores del producto`
     2. `Los factores intermedios se cancelan por parejas`

## finite_telescoping_product_basic (finite_telescoping)

- Source: `product((k+1)/k, k, 1, n)`
- Target: `n+1`
- Result: `n + 1`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: product((k + 1) / k, k, 1, n)
Target: n + 1
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Telescoping product: Π((k + 1) / k, k) from 1 to n  [Evaluar producto telescópico finito]
   Before: product((k + 1) / k, k, 1, n)
   Cambio local: product((k + 1) / k, k, 1, n) -> n + 1
   After: n + 1
Result: n + 1
ℹ️ Requires:
  • k ≠ 0
```

### Web / JSON Steps

1. `Evaluar producto telescópico finito`
   - before: `prod_k=1^n (k + 1)/k`
   - after: `n + 1`
   - substeps:
     1. `Escribir los primeros y últimos factores del producto`
     2. `Los factores intermedios se cancelan por parejas`
     3. `Solo quedan el último numerador y el primer denominador`

## finite_telescoping_product_factorized_square_shifted_base_numeric_symbolic_lower (finite_telescoping)

- Source: `product(1 - 1/(k+2)^2, k, m, n)`
- Target: `((m+1)*(n+3))/((m+2)*(n+2))`
- Result: `(m + 1) * (n + 3) / ((m + 2) * (n + 2))`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: product(1 - 1 / (k + 2)^2, k, m, n)
Target: (m + 1) * (n + 3) / ((m + 2) * (n + 2))
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Factorized telescoping product: Π(1 - 1 / (k + 2)^2, k) from m to n  [Evaluar producto telescópico finito]
   Before: product(1 - 1 / (k + 2)^(2), k, m, n)
   Cambio local: product(1 - 1 / (k + 2)^(2), k, m, n) -> (m + 1) * (n + 3) / ((m + 2) * (n + 2))
   After: (m + 1) * (n + 3) / ((m + 2) * (n + 2))
Result: (m + 1) * (n + 3) / ((m + 2) * (n + 2))
ℹ️ Requires:
  • m * n + 2 * m + 2 * n + 4 ≠ 0
  • k ≠ -2
```

### Web / JSON Steps

1. `Evaluar producto telescópico finito`
   - before: `prod_k=m^n 1 - 1/(k + 2)^2`
   - after: `((m + 1) · (n + 3))/((m + 2) · (n + 2))`
   - substeps:
     1. `Usar (u^2 - 1) / u^2 = ((u - 1) · (u + 1)) / u^2`
     2. `Los factores (u + 1) y (u - 1) se cancelan telescópicamente`
     3. `Solo quedan el primer factor u - 1 y el último factor u + 1`

## finite_telescoping_product_factorized_square_shifted_base_symbolic_symbolic_lower (finite_telescoping)

- Source: `product(1 - 1/(k+a)^2, k, m, n)`
- Target: `((m+a-1)*(n+a+1))/((m+a)*(n+a))`
- Result: `(a + n + 1) * (a + m - 1) / ((a + m) * (a + n))`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: product(1 - 1 / (a + k)^2, k, m, n)
Target: (a + n + 1) * (a + m - 1) / ((a + m) * (a + n))
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Factorized telescoping product: Π(1 - 1 / (a + k)^2, k) from m to n  [Evaluar producto telescópico finito]
   Before: product(1 - 1 / (a + k)^(2), k, m, n)
   Cambio local: product(1 - 1 / (a + k)^(2), k, m, n) -> (a + n + 1) * (a + m - 1) / ((a + m) * (a + n))
   After: (a + n + 1) * (a + m - 1) / ((a + m) * (a + n))
Result: (a + n + 1) * (a + m - 1) / ((a + m) * (a + n))
ℹ️ Requires:
  • a^2 + a * m + a * n + m * n ≠ 0
  • a + k ≠ 0
```

### Web / JSON Steps

1. `Evaluar producto telescópico finito`
   - before: `prod_k=m^n 1 - 1/(a + k)^2`
   - after: `((a + n + 1) · (a + m - 1))/((a + m) · (a + n))`
   - substeps:
     1. `Usar (u^2 - 1) / u^2 = ((u - 1) · (u + 1)) / u^2`
     2. `Los factores (u + 1) y (u - 1) se cancelan telescópicamente`
     3. `Solo quedan el primer factor u - 1 y el último factor u + 1`

## finite_telescoping_product_symbolic_shift_symbolic_lower (finite_telescoping)

- Source: `product((k+a+1)/(k+a), k, m, n)`
- Target: `(n+a+1)/(m+a)`
- Result: `(a + n + 1) / (a + m)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: product((a + k + 1) / (a + k), k, m, n)
Target: (a + n + 1) / (a + m)
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Telescoping product: Π((a + k + 1) / (a + k), k) from m to n  [Evaluar producto telescópico finito]
   Before: product((a + k + 1) / (a + k), k, m, n)
   Cambio local: product((a + k + 1) / (a + k), k, m, n) -> (a + n + 1) / (a + m)
   After: (a + n + 1) / (a + m)
Result: (a + n + 1) / (a + m)
ℹ️ Requires:
  • a + k ≠ 0
  • a + m ≠ 0
```

### Web / JSON Steps

1. `Evaluar producto telescópico finito`
   - before: `prod_k=m^n (a + k + 1)/(a + k)`
   - after: `(a + n + 1)/(a + m)`
   - substeps:
     1. `Escribir los primeros y últimos factores del producto`
     2. `Los factores intermedios se cancelan por parejas`

## finite_telescoping_sum_affine_symbolic_arbitrary_shift_symbolic_lower (finite_telescoping)

- Source: `sum(1/((a*k+b+c)*(a*k+b+c+a)), k, m, n)`
- Target: `1/a*(1/(a*m+b+c) - 1/(a*n+a+b+c))`
- Result: `((1 / (a * m + b + c) - 1 / (a * n + a + b + c)) * 1)/a`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sum(1 / ((a * k + a + b + c) * (a * k + b + c)), k, m, n)
Target: ((1 / (a * m + b + c) - 1 / (a * n + a + b + c)))/a
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Telescoping sum: Σ(1 / ((a * k + a + b + c) * (a * k + b + c)), k) from m to n  [Evaluar suma telescópica finita]
   Before: sum(1 / ((a * k + a + b + c) * (a * k + b + c)), k, m, n)
   Cambio local: sum(1 / ((a * k + a + b + c) * (a * k + b + c)), k, m, n) -> 1 / a * (1 / (a * m + b + c) - 1 / (a * n + a + b + c))
   After: ((1 / (a * m + b + c) - 1 / (a * n + a + b + c)))/a
Result: 1 / a * (1 / (a * m + b + c) - 1 / (a * n + a + b + c))
ℹ️ Requires:
  • a^2 * k^2 + 2 * a * b * k + 2 * a * c * k + a^2 * k + b^2 + c^2 + 2 * b * c + a * b + a * c ≠ 0
  • a * m + b + c ≠ 0
  • a * n + a + b + c ≠ 0
  • a ≠ 0
```

### Web / JSON Steps

1. `Evaluar suma telescópica finita`
   - before: `sum_k=m^n 1/((a · k + a + b + c) · (a · k + b + c))`
   - after: `1/a · (1/(a · m + b + c) - 1/(a · n + a + b + c))`
   - substeps:
     1. `Usar 1 / (u · (u + g)) = 1 / g · (1 / u - 1 / (u + g))`
     2. `La suma telescópica cancela los términos intermedios`

## finite_telescoping_sum_affine_symbolic_shift_symbolic_lower (finite_telescoping)

- Source: `sum(1/((a*k+b)*(a*k+b+a)), k, m, n)`
- Target: `1/a*(1/(a*m+b) - 1/(a*n+a+b))`
- Result: `((1 / (a * m + b) - 1 / (a * n + a + b)) * 1)/a`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sum(1 / ((a * k + b) * (a * k + a + b)), k, m, n)
Target: ((1 / (a * m + b) - 1 / (a * n + a + b)))/a
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Telescoping sum: Σ(1 / ((a * k + b) * (a * k + a + b)), k) from m to n  [Evaluar suma telescópica finita]
   Before: sum(1 / ((a * k + b) * (a * k + a + b)), k, m, n)
   Cambio local: sum(1 / ((a * k + b) * (a * k + a + b)), k, m, n) -> 1 / a * (1 / (a * m + b) - 1 / (a * n + a + b))
   After: ((1 / (a * m + b) - 1 / (a * n + a + b)))/a
Result: 1 / a * (1 / (a * m + b) - 1 / (a * n + a + b))
ℹ️ Requires:
  • a^2 * k^2 + 2 * a * b * k + a^2 * k + b^2 + a * b ≠ 0
  • a * m + b ≠ 0
  • a * n + a + b ≠ 0
  • a ≠ 0
```

### Web / JSON Steps

1. `Evaluar suma telescópica finita`
   - before: `sum_k=m^n 1/((a · k + b) · (a · k + a + b))`
   - after: `1/a · (1/(a · m + b) - 1/(a · n + a + b))`
   - substeps:
     1. `Usar 1 / (u · (u + g)) = 1 / g · (1 / u - 1 / (u + g))`
     2. `La suma telescópica cancela los términos intermedios`

## finite_telescoping_sum_basic (finite_telescoping)

- Source: `sum(1/(k*(k+1)), k, 1, n)`
- Target: `1 - 1/(n+1)`
- Result: `1 - 1 / (n + 1)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sum(1 / (k * (k + 1)), k, 1, n)
Target: 1 - 1 / (n + 1)
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Telescoping sum: Σ(1 / (k * (k + 1)), k) from 1 to n  [Evaluar suma telescópica finita]
   Before: sum(1 / (k * (k + 1)), k, 1, n)
   Cambio local: sum(1 / (k * (k + 1)), k, 1, n) -> 1 - 1 / (n + 1)
   After: 1 - 1 / (n + 1)
Result: 1 - 1 / (n + 1)
ℹ️ Requires:
  • k ≠ 0
  • k ≠ -1
  • n ≠ -1
```

### Web / JSON Steps

1. `Evaluar suma telescópica finita`
   - before: `sum_k=1^n 1/(k · (k + 1))`
   - after: `1 - 1/(n + 1)`
   - substeps:
     1. `Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)`
     2. `La suma telescópica cancela los términos intermedios`

## finite_telescoping_sum_symbolic_shift_symbolic_lower (finite_telescoping)

- Source: `sum(1/((k+a)*(k+a+1)), k, m, n)`
- Target: `1/(m+a) - 1/(n+a+1)`
- Result: `1 / (a + m) - 1 / (a + n + 1)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sum(1 / ((a + k) * (a + k + 1)), k, m, n)
Target: 1 / (a + m) - 1 / (a + n + 1)
Strategy: finite sums/products
Steps (Aggressive Mode):
1. Telescoping sum: Σ(1 / ((a + k) * (a + k + 1)), k) from m to n  [Evaluar suma telescópica finita]
   Before: sum(1 / ((a + k) * (a + k + 1)), k, m, n)
   Cambio local: sum(1 / ((a + k) * (a + k + 1)), k, m, n) -> 1 / (a + m) - 1 / (a + n + 1)
   After: 1 / (a + m) - 1 / (a + n + 1)
Result: 1 / (a + m) - 1 / (a + n + 1)
ℹ️ Requires:
  • a^2 + k^2 + 2 * a * k + a + k ≠ 0
  • a + m ≠ 0
  • a + n + 1 ≠ 0
```

### Web / JSON Steps

1. `Evaluar suma telescópica finita`
   - before: `sum_k=m^n 1/((a + k) · (a + k + 1))`
   - after: `1/(a + m) - 1/(a + n + 1)`
   - substeps:
     1. `Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)`
     2. `La suma telescópica cancela los términos intermedios`

## hyperbolic_composition_sinh_asinh (simplify)

- Source: `sinh(asinh(x))`
- Target: `x`
- Result: `x`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sinh(asinh(x))
Target: x
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Cancel a hyperbolic function with its inverse  [Cancelar funciones hiperbólicas inversas]
   Before: sinh(asinh(x))
   Cambio local: sinh(asinh(x)) -> x
   After: x
Result: x
```

### Web / JSON Steps

1. `Cancelar funciones hiperbólicas inversas`
   - before: `sinh(asinh(x))`
   - after: `x`
   - substeps:
     1. `Usar que sinh y asinh son funciones inversas`
     2. `Aquí u = x`

## hyperbolic_contract_cosh_triple_angle (simplify)

- Source: `4*cosh(x)^3-3*cosh(x)`
- Target: `cosh(3*x)`
- Result: `cosh(3 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 4 * cosh(x)^3 - 3 * cosh(x)
Target: cosh(3 * x)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize 4·cosh(u)^3 - 3·cosh(u) as cosh(3u)  [Aplicar identidad hiperbólica de ángulo triple]
   Before: 4 * cosh(x)^(3) - 3 * cosh(x)
   Cambio local: 4 * cosh(x)^(3) - 3 * cosh(x) -> cosh(3 * x)
   After: cosh(3 * x)
Result: cosh(3 * x)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de ángulo triple`
   - before: `4 · cosh(x)^3 - 3 · cosh(x)`
   - after: `cosh(3 · x)`
   - substeps: none

## hyperbolic_contract_exp_decomposition (simplify)

- Source: `sinh(x)+cosh(x)`
- Target: `exp(x)`
- Result: `e^x`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sinh(x) + cosh(x)
Target: e^x
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize sinh(u) + cosh(u) as exp(u)  [Reconocer forma exponencial hiperbólica]
   Before: sinh(x) + cosh(x)
   Cambio local: sinh(x) + cosh(x) -> e^(x)
   After: e^x
Result: e^(x)
```

### Web / JSON Steps

1. `Aplicar identidad exponencial hiperbólica`
   - before: `sinh(x) + cosh(x)`
   - after: `e^x`
   - substeps: none

## hyperbolic_contract_negated_negative_exp_decomposition (simplify)

- Source: `sinh(x)-cosh(x)`
- Target: `-exp(-x)`
- Result: `-(e^(-x))`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sinh(x) - cosh(x)
Target: -(e^(-x))
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize sinh(u) - cosh(u) as -exp(-u)  [Reconocer forma exponencial hiperbólica]
   Before: sinh(x) - cosh(x)
   Cambio local: sinh(x) - cosh(x) -> -e^(-x)
   After: -(e^(-x))
Result: -e^(-x)
```

### Web / JSON Steps

1. `Aplicar identidad exponencial hiperbólica`
   - before: `sinh(x) - cosh(x)`
   - after: `-e^(-x)`
   - substeps: none

## hyperbolic_contract_negative_exp_decomposition (simplify)

- Source: `cosh(x)-sinh(x)`
- Target: `exp(-x)`
- Result: `e^(-x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cosh(x) - sinh(x)
Target: e^(-x)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize cosh(u) - sinh(u) as exp(-u)  [Reconocer forma exponencial hiperbólica]
   Before: cosh(x) - sinh(x)
   Cambio local: cosh(x) - sinh(x) -> e^(-x)
   After: e^(-x)
Result: e^(-x)
```

### Web / JSON Steps

1. `Aplicar identidad exponencial hiperbólica`
   - before: `cosh(x) - sinh(x)`
   - after: `e^(-x)`
   - substeps: none

## hyperbolic_contract_sinh_double_angle (simplify)

- Source: `2*sinh(x)*cosh(x)`
- Target: `sinh(2*x)`
- Result: `sinh(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * sinh(x) * cosh(x)
Target: sinh(2 * x)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize 2·sinh(u)·cosh(u) as sinh(2u)  [Aplicar identidad hiperbólica de ángulo doble]
   Before: 2 * sinh(x) * cosh(x)
   Cambio local: 2 * sinh(x) * cosh(x) -> sinh(2 * x)
   After: sinh(2 * x)
Result: sinh(2 * x)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de ángulo doble`
   - before: `2 · sinh(x) · cosh(x)`
   - after: `sinh(2 · x)`
   - substeps: none

## hyperbolic_contract_sinh_double_angle_with_passthrough (simplify)

- Source: `2*sinh(x)*cosh(x)+a`
- Target: `sinh(2*x)+a`
- Result: `sinh(2 * x) + a`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * sinh(x) * cosh(x) + a
Target: sinh(2 * x) + a
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize 2·sinh(u)·cosh(u) as sinh(2u)  [Aplicar identidad hiperbólica de ángulo doble]
   Before: 2 * sinh(x) * cosh(x) + a
   Cambio local: 2 * sinh(x) * cosh(x) + a -> sinh(2 * x) + a
   After: sinh(2 * x) + a
Result: sinh(2 * x) + a
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de ángulo doble`
   - before: `2 · sinh(x) · cosh(x) + a`
   - after: `sinh(2 · x) + a`
   - substeps: none

## hyperbolic_contract_sinh_triple_angle (simplify)

- Source: `3*sinh(x)+4*sinh(x)^3`
- Target: `sinh(3*x)`
- Result: `sinh(3 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 3 * sinh(x) + 4 * sinh(x)^3
Target: sinh(3 * x)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize 3·sinh(u) + 4·sinh(u)^3 as sinh(3u)  [Aplicar identidad hiperbólica de ángulo triple]
   Before: 3 * sinh(x) + 4 * sinh(x)^(3)
   Cambio local: 3 * sinh(x) + 4 * sinh(x)^(3) -> sinh(3 * x)
   After: sinh(3 * x)
Result: sinh(3 * x)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de ángulo triple`
   - before: `3 · sinh(x) + 4 · sinh(x)^3`
   - after: `sinh(3 · x)`
   - substeps: none

## hyperbolic_contract_tanh_double_angle (simplify)

- Source: `2*tanh(x)/(1+tanh(x)^2)`
- Target: `tanh(2*x)`
- Result: `tanh(2 * x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 2 * tanh(x) / (tanh(x)^2 + 1)
Target: tanh(2 * x)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize 2·tanh(u)/(1 + tanh(u)^2) as tanh(2u)  [Aplicar identidad hiperbólica de ángulo doble]
   Before: 2 * tanh(x) / (tanh(x)^(2) + 1)
   Cambio local: 2 * tanh(x) / (tanh(x)^(2) + 1) -> tanh(2 * x)
   After: tanh(2 * x)
Result: tanh(2 * x)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de ángulo doble`
   - before: `(2 · tanh(x))/(tanh(x)^2 + 1)`
   - after: `tanh(2 · x)`
   - substeps: none

## hyperbolic_contract_tanh_quotient (simplify)

- Source: `sinh(x)/cosh(x)`
- Target: `tanh(x)`
- Result: `tanh(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sinh(x) / cosh(x)
Target: tanh(x)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize sinh(u) / cosh(u) as tanh(u)  [Aplicar identidad hiperbólica de cociente]
   Before: sinh(x) / cosh(x)
   Cambio local: sinh(x) / cosh(x) -> tanh(x)
   After: tanh(x)
Result: tanh(x)
```

### Web / JSON Steps

1. `Reconocer tangente hiperbólica desde un cociente`
   - before: `sinh(x)/cosh(x)`
   - after: `tanh(x)`
   - substeps: none

## hyperbolic_expand_cosh_double_angle_cosh_sq (simplify)

- Source: `cosh(2*x)`
- Target: `2*cosh(x)^2 - 1`
- Result: `2 * cosh(x)^2 - 1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cosh(2 * x)
Target: 2 * cosh(x)^2 - 1
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Expand cosh(2u) as 2·cosh(u)^2 - 1  [Aplicar identidad hiperbólica de ángulo doble]
   Before: cosh(2 * x)
   Cambio local: cosh(2 * x) -> 2 * cosh(x)^(2) - 1
   After: 2 * cosh(x)^2 - 1
Result: 2 * cosh(x)^(2) - 1
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de ángulo doble`
   - before: `cosh(2 · x)`
   - after: `2 · cosh(x)^2 - 1`
   - substeps: none

## hyperbolic_expand_cosh_double_angle_sum (simplify)

- Source: `cosh(2*x)`
- Target: `cosh(x)^2 + sinh(x)^2`
- Result: `sinh(x)^2 + cosh(x)^2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cosh(2 * x)
Target: sinh(x)^2 + cosh(x)^2
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Expand cosh(2u) as cosh(u)^2 + sinh(u)^2  [Aplicar identidad hiperbólica de ángulo doble]
   Before: cosh(2 * x)
   Cambio local: cosh(2 * x) -> sinh(x)^(2) + cosh(x)^(2)
   After: sinh(x)^2 + cosh(x)^2
Result: sinh(x)^(2) + cosh(x)^(2)
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de ángulo doble`
   - before: `cosh(2 · x)`
   - after: `sinh(x)^2 + cosh(x)^2`
   - substeps: none

## hyperbolic_expand_cosh_to_exp_definition (simplify)

- Source: `cosh(x)`
- Target: `(e^x + e^(-x))/2`
- Result: `(e^x + e^(-x)) / 2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cosh(x)
Target: (e^x + e^(-x)) / 2
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Expand cosh(u) as (exp(u) + exp(-u)) / 2  [Expandir identidad exponencial hiperbólica]
   Before: cosh(x)
   Cambio local: cosh(x) -> (e^(x) + e^(-x)) / 2
   After: (e^x + e^(-x)) / 2
Result: (e^(x) + e^(-x)) / 2
```

### Web / JSON Steps

1. `Aplicar identidad exponencial hiperbólica`
   - before: `cosh(x)`
   - after: `(e^x + e^(-x))/2`
   - substeps: none

## hyperbolic_expand_exp_to_sum (simplify)

- Source: `exp(x)`
- Target: `sinh(x) + cosh(x)`
- Result: `sinh(x) + cosh(x)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: e^x
Target: sinh(x) + cosh(x)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Expand exp(u) as sinh(u) + cosh(u)  [Expandir identidad exponencial hiperbólica]
   Before: e^(x)
   Cambio local: e^(x) -> sinh(x) + cosh(x)
   After: sinh(x) + cosh(x)
Result: sinh(x) + cosh(x)
```

### Web / JSON Steps

1. `Aplicar identidad exponencial hiperbólica`
   - before: `e^x`
   - after: `sinh(x) + cosh(x)`
   - substeps: none

## hyperbolic_expand_sinh_to_exp_definition (simplify)

- Source: `sinh(x)`
- Target: `(e^x - e^(-x))/2`
- Result: `(e^x - e^(-x)) / 2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sinh(x)
Target: (e^x - e^(-x)) / 2
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Expand sinh(u) as (exp(u) - exp(-u)) / 2  [Expandir identidad exponencial hiperbólica]
   Before: sinh(x)
   Cambio local: sinh(x) -> (e^(x) - e^(-x)) / 2
   After: (e^x - e^(-x)) / 2
Result: (e^(x) - e^(-x)) / 2
```

### Web / JSON Steps

1. `Aplicar identidad exponencial hiperbólica`
   - before: `sinh(x)`
   - after: `(e^x - e^(-x))/2`
   - substeps: none

## hyperbolic_expand_tanh_to_exp_definition (simplify)

- Source: `tanh(x)`
- Target: `(e^x - e^(-x))/(e^x + e^(-x))`
- Result: `(e^x - e^(-x)) / (e^x + e^(-x))`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: tanh(x)
Target: (e^x - e^(-x)) / (e^x + e^(-x))
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Expand tanh(u) as (exp(u) - exp(-u)) / (exp(u) + exp(-u))  [Expandir identidad exponencial hiperbólica]
   Before: tanh(x)
   Cambio local: tanh(x) -> (e^(x) - e^(-x)) / (e^(x) + e^(-x))
   After: (e^x - e^(-x)) / (e^x + e^(-x))
Result: (e^(x) - e^(-x)) / (e^(x) + e^(-x))
```

### Web / JSON Steps

1. `Aplicar identidad exponencial hiperbólica`
   - before: `tanh(x)`
   - after: `(e^x - e^(-x))/(e^x + e^(-x))`
   - substeps: none

## hyperbolic_half_angle_cosh_forward (simplify)

- Source: `cosh(x/2)^2`
- Target: `(cosh(x)+1)/2`
- Result: `(cosh(x) + 1) / 2`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cosh(x / 2)^2
Target: (cosh(x) + 1) / 2
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Expand cosh(u/2)^2 as (cosh(u) + 1) / 2  [Aplicar identidad hiperbólica de ángulo mitad]
   Before: cosh(x / 2)^(2)
   Cambio local: cosh(x / 2)^(2) -> (cosh(x) + 1) / 2
   After: (cosh(x) + 1) / 2
Result: (cosh(x) + 1) / 2
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de ángulo mitad`
   - before: `(cosh(x/2))^2`
   - after: `(cosh(x) + 1)/2`
   - substeps:
     1. `Usar cosh²(u/2) = (cosh(u) + 1) / 2`

## hyperbolic_half_angle_sinh_forward (simplify)

- Source: `sinh(x/2)^2`
- Target: `(cosh(x)-1)/2`
- Result: `(cosh(x) - 1) / 2`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sinh(x / 2)^2
Target: (cosh(x) - 1) / 2
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Expand sinh(u/2)^2 as (cosh(u) - 1) / 2  [Aplicar identidad hiperbólica de ángulo mitad]
   Before: sinh(x / 2)^(2)
   Cambio local: sinh(x / 2)^(2) -> (cosh(x) - 1) / 2
   After: (cosh(x) - 1) / 2
Result: (cosh(x) - 1) / 2
```

### Web / JSON Steps

1. `Aplicar identidad hiperbólica de ángulo mitad`
   - before: `(sinh(x/2))^2`
   - after: `(cosh(x) - 1)/2`
   - substeps:
     1. `Usar sinh²(u/2) = (cosh(u) - 1) / 2`

## hyperbolic_negative_tanh_parity (simplify)

- Source: `tanh(-x)`
- Target: `-tanh(x)`
- Result: `-tanh(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: tanh(-x)
Target: -tanh(x)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Apply a hyperbolic odd/even parity identity  [Aplicar paridad hiperbólica]
   Before: tanh(-x)
   Cambio local: tanh(-x) -> -tanh(x)
   After: -tanh(x)
Result: -tanh(x)
```

### Web / JSON Steps

1. `Aplicar paridad hiperbólica`
   - before: `tanh(-x)`
   - after: `-tanh(x)`
   - substeps:
     1. `Usar que una función impar cumple f(-u) = -f(u)`

## hyperbolic_pythagorean_identity (simplify)

- Source: `cosh(x)^2 - sinh(x)^2`
- Target: `1`
- Result: `1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cosh(x)^2 - sinh(x)^2
Target: 1
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize cosh(u)^2 - sinh(u)^2 = 1  [Aplicar identidad pitagórica hiperbólica]
   Before: cosh(x)^(2) - sinh(x)^(2)
   Cambio local: cosh(x)^(2) - sinh(x)^(2) -> 1
   After: 1
Result: 1
```

### Web / JSON Steps

1. `Aplicar identidad pitagórica hiperbólica`
   - before: `cosh(x)^2 - sinh(x)^2`
   - after: `1`
   - substeps: none

## hyperbolic_pythagorean_identity_with_passthrough (simplify)

- Source: `cosh(x)^2-sinh(x)^2+a`
- Target: `1+a`
- Result: `a + 1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: -sinh(x)^2 + cosh(x)^2 + a
Target: a + 1
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize cosh(u)^2 - sinh(u)^2 = 1  [Aplicar identidad pitagórica hiperbólica]
   Before: -sinh(x)^(2) + cosh(x)^(2) + a
   Cambio local: -sinh(x)^(2) + cosh(x)^(2) + a -> a + 1
   After: a + 1
Result: a + 1
```

### Web / JSON Steps

1. `Aplicar identidad pitagórica hiperbólica`
   - before: `cosh(x)^2 - sinh(x)^2 + a`
   - after: `a + 1`
   - substeps: none

## hyperbolic_pythagorean_shifted_forward (simplify)

- Source: `cosh(x)^2 - 1`
- Target: `sinh(x)^2`
- Result: `sinh(x)^2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cosh(x)^2 - 1
Target: sinh(x)^2
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize cosh(u)^2 - 1 as sinh(u)^2  [Aplicar identidad pitagórica hiperbólica]
   Before: cosh(x)^(2) - 1
   Cambio local: cosh(x)^(2) - 1 -> sinh(x)^(2)
   After: sinh(x)^2
Result: sinh(x)^(2)
```

### Web / JSON Steps

1. `Aplicar identidad pitagórica hiperbólica`
   - before: `cosh(x)^2 - 1`
   - after: `sinh(x)^2`
   - substeps: none

## hyperbolic_special_value_sinh_zero (simplify)

- Source: `sinh(0)`
- Target: `0`
- Result: `0`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sinh(0)
Target: 0
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Evaluate a hyperbolic function at a special input  [Evaluar valor hiperbólico especial]
   Before: sinh(0)
   Cambio local: sinh(0) -> 0
   After: 0
Result: 0
```

### Web / JSON Steps

1. `Evaluar valor hiperbólico especial`
   - before: `sinh(0)`
   - after: `0`
   - substeps: none

## hyperbolic_tanh_pythagorean_forward (simplify)

- Source: `1 - tanh(x)^2`
- Target: `1/cosh(x)^2`
- Result: `1 / cosh(x)^2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 1 - tanh(x)^2
Target: 1 / cosh(x)^2
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize 1 - tanh(u)^2 as 1 / cosh(u)^2  [Aplicar identidad pitagórica hiperbólica]
   Before: 1 - tanh(x)^(2)
   Cambio local: 1 - tanh(x)^(2) -> 1 / cosh(x)^(2)
   After: 1 / cosh(x)^2
Result: 1 / cosh(x)^(2)
```

### Web / JSON Steps

1. `Aplicar identidad pitagórica hiperbólica`
   - before: `1 - tanh(x)^2`
   - after: `1/cosh(x)^2`
   - substeps: none

## hyperbolic_tanh_pythagorean_reverse (simplify)

- Source: `1/cosh(x)^2`
- Target: `1 - tanh(x)^2`
- Result: `1 - tanh(x)^2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 1 / cosh(x)^2
Target: 1 - tanh(x)^2
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Expand 1 / cosh(u)^2 as 1 - tanh(u)^2  [Aplicar identidad pitagórica hiperbólica]
   Before: 1 / cosh(x)^(2)
   Cambio local: 1 / cosh(x)^(2) -> 1 - tanh(x)^(2)
   After: 1 - tanh(x)^2
Result: 1 - tanh(x)^(2)
```

### Web / JSON Steps

1. `Aplicar identidad pitagórica hiperbólica`
   - before: `1/cosh(x)^2`
   - after: `1 - tanh(x)^2`
   - substeps: none

## integrate_prep_dirichlet_basic (integrate_prep)

- Source: `1 + 2*cos(x) + 2*cos(2*x)`
- Target: `sin(5*x/2)/sin(x/2)`
- Result: `sin(5 * x / 2) / sin(x / 2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * cos(x) + 2 * cos(2 * x) + 1
Target: sin(5 * x / 2) / sin(x / 2)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity  [Aplicar identidad del núcleo de Dirichlet]
   Before: 2 * cos(x) + 2 * cos(2 * x) + 1
   Cambio local: 2 * cos(x) + 2 * cos(2 * x) + 1 -> sin(5 * x / 2) / sin(x / 2)
   After: sin(5 * x / 2) / sin(x / 2)
Result: sin(5 * x / 2) / sin(x / 2)
ℹ️ Requires:
  • sin(x / 2) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `2 · cos(x) + 2 · cos(2 · x) + 1`
   - after: `sin((5 · x)/2)/sin(x/2)`
   - substeps:
     1. `Usar el núcleo de Dirichlet con n = 2 y u = x`

## integrate_prep_dirichlet_longer (integrate_prep)

- Source: `1 + 2*cos(x) + 2*cos(2*x) + 2*cos(3*x)`
- Target: `sin(7*x/2)/sin(x/2)`
- Result: `sin(7 * x / 2) / sin(x / 2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * cos(x) + 2 * cos(2 * x) + 2 * cos(3 * x) + 1
Target: sin(7 * x / 2) / sin(x / 2)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity  [Aplicar identidad del núcleo de Dirichlet]
   Before: 2 * cos(x) + 2 * cos(2 * x) + 2 * cos(3 * x) + 1
   Cambio local: 2 * cos(x) + 2 * cos(2 * x) + 2 * cos(3 * x) + 1 -> sin(7 * x / 2) / sin(x / 2)
   After: sin(7 * x / 2) / sin(x / 2)
Result: sin(7 * x / 2) / sin(x / 2)
ℹ️ Requires:
  • sin(x / 2) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `2 · cos(x) + 2 · cos(2 · x) + 2 · cos(3 · x) + 1`
   - after: `sin((7 · x)/2)/sin(x/2)`
   - substeps:
     1. `Usar el núcleo de Dirichlet con n = 3 y u = x`

## integrate_prep_dirichlet_reverse_basic (integrate_prep)

- Source: `sin(5*x/2)/sin(x/2)`
- Target: `1 + 2*cos(x) + 2*cos(2*x)`
- Result: `2 * cos(x) + 2 * cos(2 * x) + 1`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(5 * x / 2) / sin(x / 2)
Target: 2 * cos(x) + 2 * cos(2 * x) + 1
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity  [Aplicar identidad del núcleo de Dirichlet]
   Before: sin(5 * x / 2) / sin(x / 2)
   Cambio local: sin(5 * x / 2) / sin(x / 2) -> 2 * cos(x) + 2 * cos(2 * x) + 1
   After: 2 * cos(x) + 2 * cos(2 * x) + 1
Result: 2 * cos(x) + 2 * cos(2 * x) + 1
ℹ️ Requires:
  • sin(x / 2) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `sin((5 · x)/2)/sin(x/2)`
   - after: `2 · cos(x) + 2 · cos(2 · x) + 1`
   - substeps:
     1. `Expandir el núcleo de Dirichlet con n = 2 y u = x`

## integrate_prep_dirichlet_reverse_symbolic_scale_longer (integrate_prep)

- Source: `sin(7*a*x/2)/sin(a*x/2)`
- Target: `1 + 2*cos(a*x) + 2*cos(2*a*x) + 2*cos(3*a*x)`
- Result: `2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(7 * a * x / 2) / sin(a * x / 2)
Target: 2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity  [Aplicar identidad del núcleo de Dirichlet]
   Before: sin(7 * a * x / 2) / sin(a * x / 2)
   Cambio local: sin(7 * a * x / 2) / sin(a * x / 2) -> 2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1
   After: 2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1
Result: 2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1
ℹ️ Requires:
  • sin(a * x / 2) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `sin((7 · a · x)/2)/sin((a · x)/2)`
   - after: `2 · cos(a · x) + 2 · cos(2 · a · x) + 2 · cos(3 · a · x) + 1`
   - substeps:
     1. `Expandir el núcleo de Dirichlet con n = 3 y u = a · x`

## integrate_prep_dirichlet_symbolic_argument (integrate_prep)

- Source: `1 + 2*cos(u) + 2*cos(2*u) + 2*cos(3*u) + 2*cos(4*u)`
- Target: `sin(9*u/2)/sin(u/2)`
- Result: `sin(9 * u / 2) / sin(u / 2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * cos(u) + 2 * cos(2 * u) + 2 * cos(3 * u) + 2 * cos(4 * u) + 1
Target: sin(9 * u / 2) / sin(u / 2)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity  [Aplicar identidad del núcleo de Dirichlet]
   Before: 2 * cos(u) + 2 * cos(2 * u) + 2 * cos(3 * u) + 2 * cos(4 * u) + 1
   Cambio local: 2 * cos(u) + 2 * cos(2 * u) + 2 * cos(3 * u) + 2 * cos(4 * u) + 1 -> sin(9 * u / 2) / sin(u / 2)
   After: sin(9 * u / 2) / sin(u / 2)
Result: sin(9 * u / 2) / sin(u / 2)
ℹ️ Requires:
  • sin(u / 2) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `2 · cos(u) + 2 · cos(2 · u) + 2 · cos(3 · u) + 2 · cos(4 · u) + 1`
   - after: `sin((9 · u)/2)/sin(u/2)`
   - substeps:
     1. `Usar el núcleo de Dirichlet con n = 4`

## integrate_prep_dirichlet_symbolic_scale (integrate_prep)

- Source: `1 + 2*cos(a*x) + 2*cos(2*a*x)`
- Target: `sin(5*a*x/2)/sin(a*x/2)`
- Result: `sin(5 * a * x / 2) / sin(a * x / 2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * cos(a * x) + 2 * cos(2 * a * x) + 1
Target: sin(5 * a * x / 2) / sin(a * x / 2)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity  [Aplicar identidad del núcleo de Dirichlet]
   Before: 2 * cos(a * x) + 2 * cos(2 * a * x) + 1
   Cambio local: 2 * cos(a * x) + 2 * cos(2 * a * x) + 1 -> sin(5 * a * x / 2) / sin(a * x / 2)
   After: sin(5 * a * x / 2) / sin(a * x / 2)
Result: sin(5 * a * x / 2) / sin(a * x / 2)
ℹ️ Requires:
  • sin(a * x / 2) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `2 · cos(a · x) + 2 · cos(2 · a · x) + 1`
   - after: `sin((5 · a · x)/2)/sin((a · x)/2)`
   - substeps:
     1. `Usar el núcleo de Dirichlet con n = 2 y u = a · x`

## integrate_prep_dirichlet_symbolic_scale_longer (integrate_prep)

- Source: `1 + 2*cos(a*x) + 2*cos(2*a*x) + 2*cos(3*a*x)`
- Target: `sin(7*a*x/2)/sin(a*x/2)`
- Result: `sin(7 * a * x / 2) / sin(a * x / 2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1
Target: sin(7 * a * x / 2) / sin(a * x / 2)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity  [Aplicar identidad del núcleo de Dirichlet]
   Before: 2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1
   Cambio local: 2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1 -> sin(7 * a * x / 2) / sin(a * x / 2)
   After: sin(7 * a * x / 2) / sin(a * x / 2)
Result: sin(7 * a * x / 2) / sin(a * x / 2)
ℹ️ Requires:
  • sin(a * x / 2) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `2 · cos(a · x) + 2 · cos(2 · a · x) + 2 · cos(3 · a · x) + 1`
   - after: `sin((7 · a · x)/2)/sin((a · x)/2)`
   - substeps:
     1. `Usar el núcleo de Dirichlet con n = 3 y u = a · x`

## integrate_prep_morrie_basic (integrate_prep)

- Source: `cos(x)*cos(2*x)*cos(4*x)`
- Target: `sin(8*x)/(8*sin(x))`
- Result: `sin(8 * x) / (8 * sin(x))`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x) * cos(2 * x) * cos(4 * x)
Target: sin(8 * x) / (8 * sin(x))
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply Morrie's law  [Aplicar telescopado de cosenos]
   Before: cos(x) * cos(2 * x) * cos(4 * x)
   Cambio local: cos(x) * cos(2 * x) * cos(4 * x) -> sin(8 * x) / (8 * sin(x))
   After: sin(8 * x) / (8 * sin(x))
Result: sin(8 * x) / (8 * sin(x))
ℹ️ Requires:
  • sin(x) ≠ 0
```

### Web / JSON Steps

1. `Aplicar telescopado de cosenos`
   - before: `cos(x) · cos(2 · x) · cos(4 · x)`
   - after: `sin(8 · x)/(8 · sin(x))`
   - substeps:
     1. `Usar el telescopado de cosenos con u = x`

## integrate_prep_morrie_reverse_basic (integrate_prep)

- Source: `sin(8*x)/(8*sin(x))`
- Target: `cos(x)*cos(2*x)*cos(4*x)`
- Result: `cos(x) * cos(2 * x) * cos(4 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(8 * x) / (8 * sin(x))
Target: cos(x) * cos(2 * x) * cos(4 * x)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply Morrie's law  [Aplicar telescopado de cosenos]
   Before: sin(8 * x) / (8 * sin(x))
   Cambio local: sin(8 * x) / (8 * sin(x)) -> cos(x) * cos(2 * x) * cos(4 * x)
   After: cos(x) * cos(2 * x) * cos(4 * x)
Result: cos(x) * cos(2 * x) * cos(4 * x)
ℹ️ Requires:
  • sin(x) ≠ 0
```

### Web / JSON Steps

1. `Aplicar telescopado de cosenos`
   - before: `sin(8 · x)/(8 · sin(x))`
   - after: `cos(x) · cos(2 · x) · cos(4 · x)`
   - substeps:
     1. `Expandir la ley de Morrie con u = x`

## integrate_prep_morrie_reverse_symbolic_scale_longer (integrate_prep)

- Source: `sin(8*a*x)/(8*sin(a*x))`
- Target: `cos(a*x)*cos(2*a*x)*cos(4*a*x)`
- Result: `cos(a * x) * cos(2 * a * x) * cos(4 * a * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(8 * a * x) / (8 * sin(a * x))
Target: cos(a * x) * cos(2 * a * x) * cos(4 * a * x)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply Morrie's law  [Aplicar telescopado de cosenos]
   Before: sin(8 * a * x) / (8 * sin(a * x))
   Cambio local: sin(8 * a * x) / (8 * sin(a * x)) -> cos(a * x) * cos(2 * a * x) * cos(4 * a * x)
   After: cos(a * x) * cos(2 * a * x) * cos(4 * a * x)
Result: cos(a * x) * cos(2 * a * x) * cos(4 * a * x)
ℹ️ Requires:
  • sin(a * x) ≠ 0
```

### Web / JSON Steps

1. `Aplicar telescopado de cosenos`
   - before: `sin(8 · a · x)/(8 · sin(a · x))`
   - after: `cos(a · x) · cos(2 · a · x) · cos(4 · a · x)`
   - substeps:
     1. `Expandir la ley de Morrie con u = a · x`

## integrate_prep_morrie_symbolic_argument (integrate_prep)

- Source: `cos(u)*cos(2*u)*cos(4*u)*cos(8*u)`
- Target: `sin(16*u)/(16*sin(u))`
- Result: `sin(16 * u) / (16 * sin(u))`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(u) * cos(2 * u) * cos(4 * u) * cos(8 * u)
Target: sin(16 * u) / (16 * sin(u))
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply Morrie's law  [Aplicar telescopado de cosenos]
   Before: cos(u) * cos(2 * u) * cos(4 * u) * cos(8 * u)
   Cambio local: cos(u) * cos(2 * u) * cos(4 * u) * cos(8 * u) -> sin(16 * u) / (16 * sin(u))
   After: sin(16 * u) / (16 * sin(u))
Result: sin(16 * u) / (16 * sin(u))
ℹ️ Requires:
  • sin(u) ≠ 0
```

### Web / JSON Steps

1. `Aplicar telescopado de cosenos`
   - before: `cos(u) · cos(2 · u) · cos(4 · u) · cos(8 · u)`
   - after: `sin(16 · u)/(16 · sin(u))`
   - substeps:
     1. `Usar el telescopado de cosenos`

## integrate_prep_morrie_symbolic_scale (integrate_prep)

- Source: `cos(a*x)*cos(2*a*x)`
- Target: `sin(4*a*x)/(4*sin(a*x))`
- Result: `sin(4 * a * x) / (4 * sin(a * x))`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(a * x) * cos(2 * a * x)
Target: sin(4 * a * x) / (4 * sin(a * x))
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply Morrie's law  [Aplicar telescopado de cosenos]
   Before: cos(a * x) * cos(2 * a * x)
   Cambio local: cos(a * x) * cos(2 * a * x) -> sin(4 * a * x) / (4 * sin(a * x))
   After: sin(4 * a * x) / (4 * sin(a * x))
Result: sin(4 * a * x) / (4 * sin(a * x))
ℹ️ Requires:
  • sin(a * x) ≠ 0
```

### Web / JSON Steps

1. `Aplicar telescopado de cosenos`
   - before: `cos(a · x) · cos(2 · a · x)`
   - after: `sin(4 · a · x)/(4 · sin(a · x))`
   - substeps:
     1. `Usar el telescopado de cosenos con u = a · x`

## inverse_hyperbolic_atanh_square_ratio_log (simplify)

- Source: `atanh((x^2 - 1)/(x^2 + 1))`
- Target: `ln(x)`
- Result: `ln(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: atanh((x^2 - 1) / (x^2 + 1))
Target: ln(x)
Strategy: rewrite hyperbolics
Steps (Aggressive Mode):
1. Recognize atanh((u^2 - 1)/(u^2 + 1)) as ln(u)  [Convertir tangente hiperbólica inversa en logaritmo]
   Before: atanh((x^(2) - 1) / (x^(2) + 1))
   Cambio local: atanh((x^(2) - 1) / (x^(2) + 1)) -> ln(x)
   After: ln(x)
Result: ln(x)
ℹ️ Requires:
  • x > 0
```

### Web / JSON Steps

1. `Convertir tangente hiperbólica inversa en logaritmo`
   - before: `atanh((x^2 - 1)/(x^2 + 1))`
   - after: `ln(x)`
   - substeps:
     1. `Identificar el argumento como (u^2 - 1)/(u^2 + 1)`

## inverse_tan_identity (simplify)

- Source: `arctan(a)+arctan(1/a)`
- Target: `(pi/2)*sign(a)`
- Result: `(pi * sign(a))/2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: arctan(a) + arctan(1 / a)
Target: (pi * sign(a))/2
Strategy: rewrite inverse trigs
Steps (Aggressive Mode):
1. arctan(x) + arctan(1/x) = (π/2)·sign(x)  [Inverse Tan Relations]
   Before: arctan(a) + arctan(1 / a)
   Cambio local: arctan(a) + arctan(1 / a) -> sign(a) * pi / 2
   After: (pi * sign(a))/2
Result: sign(a) * pi / 2
ℹ️ Requires:
  • a ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de arctangentes`
   - before: `arctan(a) + arctan(1/a)`
   - after: `sign(a) · pi/2`
   - substeps: none

## inverse_trig_arcsin_arccos_complement_sum (simplify)

- Source: `arcsin(x)+arccos(x)`
- Target: `pi/2`
- Result: `pi / 2`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: arcsin(x) + arccos(x)
Target: pi / 2
Strategy: rewrite inverse trigs
Steps (Aggressive Mode):
1. arcsin(x) + arccos(x) = π/2  [Aplicar identidad complementaria arcsin/arccos]
   Before: arcsin(x) + arccos(x)
   Cambio local: arcsin(x) + arccos(x) -> pi / 2
   After: pi / 2
Result: pi / 2
```

### Web / JSON Steps

1. `Aplicar identidad complementaria arcsin/arccos`
   - before: `arcsin(x) + arccos(x)`
   - after: `pi/2`
   - substeps:
     1. `Aquí arcsin(x) y arccos(x) suman pi/2`

## inverse_trig_composition_sin_arcsin (simplify)

- Source: `sin(arcsin(x))`
- Target: `x`
- Result: `x`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sin(arcsin(x))
Target: x
Strategy: rewrite inverse trigs
Steps (Aggressive Mode):
1. sin(arcsin(x)) = x  [Aplicar composición trigonométrica inversa]
   Before: sin(arcsin(x))
   Cambio local: sin(arcsin(x)) -> x
   After: x
Result: x
```

### Web / JSON Steps

1. `Aplicar composición trigonométrica inversa`
   - before: `sin(arcsin(x))`
   - after: `x`
   - substeps:
     1. `Usar que sin y arcsin son funciones inversas`
     2. `Aquí u = x`

## inverse_trig_special_value_arctan_sqrt_three (simplify)

- Source: `arctan(sqrt(3))`
- Target: `pi/3`
- Result: `pi / 3`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: arctan(sqrt(3))
Target: pi / 3
Strategy: rewrite trigs
Steps (Aggressive Mode):
1. Evaluate a trigonometric function at a special input  [Evaluar valor trigonométrico especial]
   Before: arctan(sqrt(3))
   Cambio local: arctan(sqrt(3)) -> pi / 3
   After: pi / 3
Result: pi / 3
```

### Web / JSON Steps

1. `Evaluar valor trigonométrico especial`
   - before: `arctan(sqrt(3))`
   - after: `pi/3`
   - substeps: none

## log_exp_inverse_ln_exp (log_exp_inverse)

- Source: `ln(exp(x))`
- Target: `x`
- Result: `x`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(e^x)
Target: x
Strategy: rewrite exponentials
Steps (Aggressive Mode):
1. Cancel ln(exp(u)) to u  [Cancelar logaritmo natural y exponencial inversos]
   Before: ln(e^(x))
   Cambio local: ln(e^(x)) -> x
   After: x
Result: x
```

### Web / JSON Steps

1. `Cancelar logaritmo natural y exponencial inversos`
   - before: `ln(e^x)`
   - after: `x`
   - substeps: none

## log_exp_inverse_ln_exp_power (log_exp_inverse)

- Source: `ln(exp(x)^2)`
- Target: `2*x`
- Result: `2 * x`
- Web step count: `2`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: ln(e^x^2)
Target: 2 * x
Strategy: rewrite exponentials
Steps (Aggressive Mode):
1. Multiply exponents  [Multiplicar exponentes]
   Before: ln(e^(x)^(2))
   Cambio local: e^(x)^(2) -> exp(2 * x)
   After: ln(exp(2 * x))
2. Cancel ln(exp(u)) to u  [Cancelar logaritmo natural y exponencial inversos]
   Before: ln(exp(2 * x))
   Cambio local: ln(exp(2 * x)) -> 2 * x
   After: 2 * x
Result: 2 * x
```

### Web / JSON Steps

1. `Multiplicar exponentes`
   - before: `ln(e^x^2)`
   - after: `ln(e^(2 · x))`
   - substeps:
     1. `Usar (e^A)^n = e^(n·A)`
2. `Cancelar logaritmo natural y exponencial inversos`
   - before: `ln(e^(2 · x))`
   - after: `2 · x`
   - substeps: none

## log_exp_inverse_ln_exp_product (log_exp_inverse)

- Source: `ln(exp(x)*exp(y))`
- Target: `x+y`
- Result: `x + y`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: ln(e^x * e^y)
Target: x + y
Strategy: expand_log
Steps (Aggressive Mode):
1. Log expansion  [Expandir logaritmos]
   Before: ln(e^(x) * e^(y))
   Cambio local: ln(e^(x) * e^(y)) -> x + y
   After: x + y
   ℹ️ Requires: e^x > 0
   ℹ️ Requires: e^y > 0
Result: x + y
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln(e^x · e^y)`
   - after: `x + y`
   - substeps:
     1. `Cancelar cada logaritmo natural con su exponencial`

## log_exp_inverse_log10_power_alias (log_exp_inverse)

- Source: `10^(y*log10(x))`
- Target: `x^y`
- Result: `x^y`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 10^(y * log10(x))
Target: x^y
Strategy: rewrite exponentials
Steps (Aggressive Mode):
1. Recognize b^(k·log_b(u)) as u^k  [Cancelar exponencial con logaritmo y conservar exponente]
   Before: 10^(y * log10(x))
   Cambio local: 10^(y * log10(x)) -> x^(y)
   After: x^y
   ℹ️ Requires: x > 0
Result: x^(y)
ℹ️ Requires:
  • x > 0
```

### Web / JSON Steps

1. `Cancelar exponencial con logaritmo y conservar exponente`
   - before: `10^(y · log_10(x))`
   - after: `x^y`
   - substeps:
     1. `Usar que 10^(log10(u)) = u`
     2. `Aplicar el factor exterior como exponente`

## log_exp_inverse_natural_log_power_alias (log_exp_inverse)

- Source: `exp(y*log(x))`
- Target: `x^y`
- Result: `x^y`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: e^(y * ln(x))
Target: x^y
Strategy: rewrite exponentials
Steps (Aggressive Mode):
1. Recognize b^(k·log_b(u)) as u^k  [Cancelar exponencial con logaritmo y conservar exponente]
   Before: e^(y * ln(x))
   Cambio local: e^(y * ln(x)) -> x^(y)
   After: x^y
   ℹ️ Requires: x > 0
Result: x^(y)
ℹ️ Requires:
  • x > 0
```

### Web / JSON Steps

1. `Cancelar exponencial con logaritmo y conservar exponente`
   - before: `e^(y · ln(x))`
   - after: `x^y`
   - substeps:
     1. `Usar que e^(ln(u)) = u`
     2. `Aplicar el factor exterior como exponente`

## log_inverse_power_tower (log_inverse_power)

- Source: `x^(ln(y)/ln(x))`
- Target: `y`
- Result: `y`
- Web step count: `2`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: x^(ln(y) / ln(x))
Target: y
Strategy: log inverse power
Steps (Aggressive Mode):
1. x^(c/log(b, x)) = b^c  [Convertir potencia logarítmica inversa]
   Before: x^(ln(y) / ln(x))
   Cambio local: x^(ln(y) / ln(x)) -> e^(ln(y))
   After: e^(ln(y))
2. b^log(b, x) = x  [Cancelar exponencial y logaritmo inversos]
   Before: e^(ln(y))
   Cambio local: e^(ln(y)) -> y
   After: y
Result: y
ℹ️ Requires:
  • x ≠ 1
  • x > 0
  • y > 0
```

### Web / JSON Steps

1. `Convertir potencia logarítmica inversa`
   - before: `x^(ln(y)/ln(x))`
   - after: `e^ln(y)`
   - substeps:
     1. `Usar que e^(ln(u)) = u`
     2. `El exponente exterior cancela el ln del exponente interior`
2. `Cancelar exponencial y logaritmo inversos`
   - before: `e^ln(y)`
   - after: `y`
   - substeps: none

## log_inverse_power_unary_natural_alias (log_inverse_power)

- Source: `x^(log(log(x))/log(x))`
- Target: `log(x)`
- Result: `ln(x)`
- Web step count: `2`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: x^(ln(ln(x)) / ln(x))
Target: ln(x)
Strategy: log inverse power
Steps (Aggressive Mode):
1. x^(c/log(b, x)) = b^c  [Convertir potencia logarítmica inversa]
   Before: x^(ln(ln(x)) / ln(x))
   Cambio local: x^(ln(ln(x)) / ln(x)) -> e^(ln(ln(x)))
   After: e^(ln(ln(x)))
2. b^log(b, x) = x  [Cancelar exponencial y logaritmo inversos]
   Before: e^(ln(ln(x)))
   Cambio local: e^(ln(ln(x))) -> ln(x)
   After: ln(x)
Result: ln(x)
ℹ️ Requires:
  • ln(x) > 0
  • x > 0
```

### Web / JSON Steps

1. `Convertir potencia logarítmica inversa`
   - before: `x^(ln(ln(x))/ln(x))`
   - after: `e^ln(ln(x))`
   - substeps:
     1. `Usar que e^(ln(u)) = u`
     2. `El exponente exterior cancela el ln del exponente interior`
2. `Cancelar exponencial y logaritmo inversos`
   - before: `e^ln(ln(x))`
   - after: `ln(x)`
   - substeps: none

## log_sum_difference_cancels_to_zero (simplify)

- Source: `ln(x^3) + ln(y^2) - ln(x^3*y^2)`
- Target: `0`
- Result: `0`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(x^3) + ln(y^2) - ln(x^3 * y^2)
Target: 0
Strategy: expand_log
Steps (Aggressive Mode):
1. Log expansion  [Expandir logaritmos]
   Before: ln(x^(3)) + ln(y^(2)) - ln(x^(3) * y^(2))
   Cambio local: ln(x^(3)) + ln(y^(2)) - ln(x^(3) * y^(2)) -> 0
   After: 0
   ℹ️ Requires: x^3 > 0
   ℹ️ Requires: y^2 > 0
Result: 0
ℹ️ Requires:
  • x > 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln(x^3) + ln(y^2) - ln(x^3 · y^2)`
   - after: `0`
   - substeps: none

## merge_four_same_base_symbolic_powers (power_merge)

- Source: `x^a*x^b*x^c*x^d`
- Target: `x^(a+b+c+d)`
- Result: `x^(a + b + c + d)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: x^a * x^b * x^c * x^d
Target: x^(a + b + c + d)
Strategy: combine powers
Steps (Aggressive Mode):
1. Combine powers with same base (n-ary)  [Sumar exponentes de la misma base]
   Before: x^(a) * x^(b) * x^(c) * x^(d)
   Cambio local: x^(a) * x^(b) * x^(c) * x^(d) -> x^(a + b + c + d)
   After: x^(a + b + c + d)
Result: x^(a + b + c + d)
```

### Web / JSON Steps

1. `Sumar exponentes de la misma base`
   - before: `x^a · x^b · x^c · x^d`
   - after: `x^(a + b + c + d)`
   - substeps: none

## merge_mixed_root_and_fractional_power_five_sixths (power_merge)

- Source: `sqrt(x)*x^(1/3)`
- Target: `x^(5/6)`
- Result: `x^(5 / 6)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sqrt(x) * x^(1 / 3)
Target: x^(5 / 6)
Strategy: combine powers
Steps (Aggressive Mode):
1. Combine powers with same base (n-ary)  [Sumar exponentes de la misma base]
   Before: sqrt(x) * x^(1 / 3)
   Cambio local: sqrt(x) * x^(1 / 3) -> x^(5 / 6)
   After: x^(5 / 6)
Result: x^(5 / 6)
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Sumar exponentes de la misma base`
   - before: `sqrt(x) · sqrt[3]x`
   - after: `sqrt[6]x^5`
   - substeps: none

## merge_mixed_root_and_fractional_powers_to_integer_with_passthrough (power_merge)

- Source: `sqrt(x)*x^(3/2)+a`
- Target: `x^2+a`
- Result: `x^2 + a`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sqrt(x) * x^(3 / 2) + a
Target: x^2 + a
Strategy: combine powers
Steps (Aggressive Mode):
1. Combine powers with same base (n-ary)  [Sumar exponentes de la misma base]
   Before: sqrt(x) * sqrt(x^3) + a
   Cambio local: sqrt(x) * sqrt(x^3) + a -> x^(2) + a
   After: x^2 + a
Result: x^(2) + a
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Sumar exponentes de la misma base`
   - before: `sqrt(x) · sqrt(x^3) + a`
   - after: `x^2 + a`
   - substeps: none

## merge_mixed_root_and_symbolic_power (power_merge)

- Source: `sqrt(x)*x^a`
- Target: `x^(a+1/2)`
- Result: `x^(1 / 2 + a)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sqrt(x) * x^a
Target: x^(1 / 2 + a)
Strategy: combine powers
Steps (Aggressive Mode):
1. Combine powers with same base (n-ary)  [Sumar exponentes de la misma base]
   Before: sqrt(x) * x^(a)
   Cambio local: sqrt(x) * x^(a) -> x^(1 / 2 + a)
   After: x^(1 / 2 + a)
Result: x^(1 / 2 + a)
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Sumar exponentes de la misma base`
   - before: `sqrt(x) · x^a`
   - after: `x^(1/2 + a)`
   - substeps: none

## merge_same_base_fractional_powers (power_merge)

- Source: `x^(1/2)*x^(2/3)`
- Target: `x^(7/6)`
- Result: `x^(7 / 6)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: x^(1 / 2) * x^(2 / 3)
Target: x^(7 / 6)
Strategy: combine powers
Steps (Aggressive Mode):
1. Combine powers with same base (n-ary)  [Sumar exponentes de la misma base]
   Before: x^(1 / 2) * x^(2 / 3)
   Cambio local: x^(1 / 2) * x^(2 / 3) -> x^(7 / 6)
   After: x^(7 / 6)
Result: x^(7 / 6)
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Sumar exponentes de la misma base`
   - before: `sqrt(x) · sqrt[3]x^2`
   - after: `sqrt[6]x^7`
   - substeps: none

## merge_same_base_fractional_powers_to_integer (power_merge)

- Source: `x^(3/4)*x^(1/4)`
- Target: `x`
- Result: `x`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: x^(1 / 4) * x^(3 / 4)
Target: x
Strategy: combine powers
Steps (Aggressive Mode):
1. Combine powers with same base (n-ary)  [Sumar exponentes de la misma base]
   Before: x^(1 / 4) * x^(3 / 4)
   Cambio local: x^(1 / 4) * x^(3 / 4) -> x
   After: x
Result: x
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Sumar exponentes de la misma base`
   - before: `sqrt[4]x · sqrt[4]x^3`
   - after: `x`
   - substeps: none

## merge_same_base_integer_and_fractional_power (power_merge)

- Source: `x*x^(1/3)`
- Target: `x^(4/3)`
- Result: `x^(4 / 3)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: x * x^(1 / 3)
Target: x^(4 / 3)
Strategy: combine powers
Steps (Aggressive Mode):
1. Combine powers with same base (n-ary)  [Sumar exponentes de la misma base]
   Before: x * x^(1 / 3)
   Cambio local: x * x^(1 / 3) -> x^(4 / 3)
   After: x^(4 / 3)
Result: x^(4 / 3)
```

### Web / JSON Steps

1. `Sumar exponentes de la misma base`
   - before: `x · sqrt[3]x`
   - after: `sqrt[3]x^4`
   - substeps: none

## merge_same_base_integer_and_symbolic_power (power_merge)

- Source: `x*x^a`
- Target: `x^(a+1)`
- Result: `x^(a + 1)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: x * x^a
Target: x^(a + 1)
Strategy: combine powers
Steps (Aggressive Mode):
1. Combine powers with same base (n-ary)  [Sumar exponentes de la misma base]
   Before: x * x^(a)
   Cambio local: x * x^(a) -> x^(a + 1)
   After: x^(a + 1)
Result: x^(a + 1)
```

### Web / JSON Steps

1. `Sumar exponentes de la misma base`
   - before: `x · x^a`
   - after: `x^(a + 1)`
   - substeps: none

## merge_same_base_symbolic_powers (power_merge)

- Source: `x^a*x^b`
- Target: `x^(a+b)`
- Result: `x^(a + b)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: x^a * x^b
Target: x^(a + b)
Strategy: combine powers
Steps (Aggressive Mode):
1. Combine powers with same base (n-ary)  [Sumar exponentes de la misma base]
   Before: x^(a) * x^(b)
   Cambio local: x^(a) * x^(b) -> x^(a + b)
   After: x^(a + b)
Result: x^(a + b)
```

### Web / JSON Steps

1. `Sumar exponentes de la misma base`
   - before: `x^a · x^b`
   - after: `x^(a + b)`
   - substeps: none

## merge_same_base_symbolic_quotient_powers (power_merge)

- Source: `x^a/x^b`
- Target: `x^(a-b)`
- Result: `x^(a - b)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: x^a / x^b
Target: x^(a - b)
Strategy: combine powers
Steps (Aggressive Mode):
1. Combine powers with same base (n-ary)  [Sumar exponentes de la misma base]
   Before: x^(a) / x^(b)
   Cambio local: x^(a) / x^(b) -> x^(a - b)
   After: x^(a - b)
Result: x^(a - b)
ℹ️ Requires:
  • x^b ≠ 0
```

### Web / JSON Steps

1. `Sumar exponentes de la misma base`
   - before: `x^a/x^b`
   - after: `x^(a - b)`
   - substeps:
     1. `Reescribir la división como potencia negativa`
     2. `Sumar los exponentes de la misma base`

## merge_sqrt_product_requires_nonnegative (simplify)

- Source: `sqrt(x)*sqrt(y)`
- Target: `sqrt(x*y)`
- Result: `sqrt(x * y)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sqrt(x) * sqrt(y)
Target: sqrt(x * y)
Strategy: rewrite radicals
Steps (Aggressive Mode):
1. √a · √b = √(a·b)  [Combinar raíces en un producto]
   Before: sqrt(x) * sqrt(y)
   Cambio local: sqrt(x) * sqrt(y) -> sqrt(x * y)
   After: sqrt(x * y)
Result: sqrt(x * y)
ℹ️ Requires:
  • x ≥ 0
  • y ≥ 0
```

### Web / JSON Steps

1. `Combinar raíces en un producto`
   - before: `sqrt(x) · sqrt(y)`
   - after: `sqrt(x · y)`
   - substeps: none

## nested_fraction_fraction_over_sum_with_fraction_general (nested_fraction)

- Source: `a/(b + c/d)`
- Target: `a*d/(b*d+c)`
- Result: `a * d / (b * d + c)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a / (c / d + b)
Target: a * d / (b * d + c)
Strategy: nested fraction
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplificar fracción anidada]
   Before: a / (c / d + b)
   Cambio local: a / (c / d + b) -> a * d / (b * d + c)
   After: a * d / (b * d + c)
Result: a * d / (b * d + c)
ℹ️ Requires:
  • b * d + c ≠ 0
  • c / d + b ≠ 0
  • d ≠ 0
```

### Web / JSON Steps

1. `Cancelar factores en una fracción`
   - before: `a/(c/d + b)`
   - after: `(a · d)/(b · d + c)`
   - substeps:
     1. `Llevar a denominador común dentro del denominador`
     2. `Dividir entre una fracción es multiplicar por su inversa`

## nested_fraction_fraction_over_sum_with_fraction_general_reverse (nested_fraction)

- Source: `a*d/(b*d+c)`
- Target: `a/(b + c/d)`
- Result: `a / (c / d + b)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a * d / (b * d + c)
Target: a / (c / d + b)
Strategy: nested fraction
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplificar fracción anidada]
   Before: a * d / (b * d + c)
   Cambio local: a * d / (b * d + c) -> a / (c / d + b)
   After: a / (c / d + b)
Result: a / (c / d + b)
ℹ️ Requires:
  • b * d + c ≠ 0
  • c / d + b ≠ 0
  • d ≠ 0
```

### Web / JSON Steps

1. `Simplificar fracción anidada`
   - before: `(a · d)/(b · d + c)`
   - after: `a/(c/d + b)`
   - substeps:
     1. `Reescribir el denominador sacando factor común d`

## nested_fraction_one_over_sum (nested_fraction)

- Source: `1/(1/a + 1/b)`
- Target: `(a*b)/(a+b)`
- Result: `a * b / (a + b)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (1 / a + 1 / b)
Target: a * b / (a + b)
Strategy: nested fraction
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplificar fracción anidada]
   Before: 1 / (1 / a + 1 / b)
   Cambio local: 1 / (1 / a + 1 / b) -> a * b / (a + b)
   After: a * b / (a + b)
Result: a * b / (a + b)
ℹ️ Requires:
  • a + b ≠ 0
  • a ≠ 0
  • b ≠ 0
```

### Web / JSON Steps

1. `Cancelar factores en una fracción`
   - before: `1/(1/a + 1/b)`
   - after: `(a · b)/(a + b)`
   - substeps:
     1. `Llevar a denominador común dentro del denominador`
     2. `Invertir la fracción del denominador`

## nested_fraction_one_over_sum_with_fraction (nested_fraction)

- Source: `1/(x + y/z)`
- Target: `z/(x*z+y)`
- Result: `z / (x * z + y)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (y / z + x)
Target: z / (x * z + y)
Strategy: nested fraction
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplificar fracción anidada]
   Before: 1 / (y / z + x)
   Cambio local: 1 / (y / z + x) -> z / (x * z + y)
   After: z / (x * z + y)
Result: z / (x * z + y)
ℹ️ Requires:
  • x * z + y ≠ 0
  • y / z + x ≠ 0
  • z ≠ 0
```

### Web / JSON Steps

1. `Cancelar factores en una fracción`
   - before: `1/(y/z + x)`
   - after: `z/(x · z + y)`
   - substeps:
     1. `Llevar a denominador común dentro del denominador`
     2. `Invertir la fracción del denominador`

## nested_fraction_one_over_sum_with_fraction_reverse (nested_fraction)

- Source: `z/(x*z+y)`
- Target: `1/(x + y/z)`
- Result: `1 / (y / z + x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: z / (x * z + y)
Target: 1 / (y / z + x)
Strategy: nested fraction
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplificar fracción anidada]
   Before: z / (x * z + y)
   Cambio local: z / (x * z + y) -> 1 / (y / z + x)
   After: 1 / (y / z + x)
Result: 1 / (y / z + x)
ℹ️ Requires:
  • x * z + y ≠ 0
  • y / z + x ≠ 0
  • z ≠ 0
```

### Web / JSON Steps

1. `Simplificar fracción anidada`
   - before: `z/(x · z + y)`
   - after: `1/(y/z + x)`
   - substeps:
     1. `Reescribir el denominador sacando factor común z`

## nested_fraction_one_over_sum_with_passthrough (nested_fraction)

- Source: `a+1/(1/x + 1/y)`
- Target: `a+(x*y)/(x+y)`
- Result: `x * y / (x + y) + a`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (1 / x + 1 / y) + a
Target: x * y / (x + y) + a
Strategy: nested fraction
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplificar fracción anidada]
   Before: 1 / (1 / x + 1 / y) + a
   Cambio local: 1 / (1 / x + 1 / y) + a -> x * y / (x + y) + a
   After: x * y / (x + y) + a
Result: x * y / (x + y) + a
ℹ️ Requires:
  • x + y ≠ 0
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Cancelar factores en una fracción`
   - before: `1/(1/x + 1/y) + a`
   - after: `(x · y)/(x + y) + a`
   - substeps:
     1. `Llevar a denominador común dentro del denominador`
     2. `Invertir la fracción del denominador`

## nested_fraction_one_over_sum_with_subtractive_passthrough (nested_fraction)

- Source: `a-1/(1/x + 1/y)`
- Target: `a-(x*y)/(x+y)`
- Result: `a - x * y / (x + y)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a - 1 / (1 / x + 1 / y)
Target: a - x * y / (x + y)
Strategy: nested fraction
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplificar fracción anidada]
   Before: a - 1 / (1 / x + 1 / y)
   Cambio local: a - 1 / (1 / x + 1 / y) -> a - x * y / (x + y)
   After: a - x * y / (x + y)
Result: a - x * y / (x + y)
ℹ️ Requires:
  • x + y ≠ 0
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Cancelar factores en una fracción`
   - before: `a - 1/(1/x + 1/y)`
   - after: `a - x · y/(x + y)`
   - substeps:
     1. `Llevar a denominador común dentro del denominador`
     2. `Invertir la fracción del denominador`

## nested_fraction_one_over_three_reciprocals (nested_fraction)

- Source: `1/(1/a + 1/b + 1/c)`
- Target: `(a*b*c)/(a*b + a*c + b*c)`
- Result: `a * b * c / (a * b + a * c + b * c)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (1 / a + 1 / b + 1 / c)
Target: a * b * c / (a * b + a * c + b * c)
Strategy: nested fraction
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplificar fracción anidada]
   Before: 1 / (1 / a + 1 / b + 1 / c)
   Cambio local: 1 / (1 / a + 1 / b + 1 / c) -> a * b * c / (a * b + a * c + b * c)
   After: a * b * c / (a * b + a * c + b * c)
Result: a * b * c / (a * b + a * c + b * c)
ℹ️ Requires:
  • 1 / a + 1 / b + 1 / c ≠ 0
  • a * b + a * c + b * c ≠ 0
  • a ≠ 0
  • b ≠ 0
  • c ≠ 0
```

### Web / JSON Steps

1. `Cancelar factores en una fracción`
   - before: `1/(1/a + 1/b + 1/c)`
   - after: `(a · b · c)/(a · b + a · c + b · c)`
   - substeps:
     1. `Llevar a denominador común dentro del denominador`
     2. `Invertir la fracción del denominador`

## nested_fraction_reciprocal_inverse (nested_fraction)

- Source: `1/(1/x)`
- Target: `x`
- Result: `x`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (1 / x)
Target: x
Strategy: nested fraction
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplificar fracción anidada]
   Before: 1 / (1 / x)
   Cambio local: 1 / (1 / x) -> x
   After: x
Result: x
ℹ️ Requires:
  • x ≠ 0
```

### Web / JSON Steps

1. `Cancelar factores en una fracción`
   - before: `1/(1/x)`
   - after: `x`
   - substeps:
     1. `Invertir la fracción del denominador`
     2. `Simplificar el producto resultante`

## nested_fraction_sum_over_reciprocal (nested_fraction)

- Source: `(1/x + 1/y)/(1/z)`
- Target: `z*(x+y)/(x*y)`
- Result: `z * (x + y) / (x * y)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (1 / x + 1 / y) / (1 / z)
Target: z * (x + y) / (x * y)
Strategy: nested fraction
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplificar fracción anidada]
   Before: (1 / x + 1 / y) / (1 / z)
   Cambio local: (1 / x + 1 / y) / (1 / z) -> z * (x + y) / (x * y)
   After: z * (x + y) / (x * y)
Result: z * (x + y) / (x * y)
ℹ️ Requires:
  • z ≠ 0
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Cancelar factores en una fracción`
   - before: `(1/x + 1/y)/(1/z)`
   - after: `(z · (x + y))/(x · y)`
   - substeps:
     1. `Invertir la fracción del denominador`
     2. `Simplificar el producto resultante`

## nested_fraction_sum_with_fraction_over_scalar_general (nested_fraction)

- Source: `(a + b/c)/d`
- Target: `(a*c+b)/(c*d)`
- Result: `(a * c + b) / (c * d)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (b / c + a) / d
Target: (a * c + b) / (c * d)
Strategy: nested fraction
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplificar fracción anidada]
   Before: (b / c + a) / d
   Cambio local: (b / c + a) / d -> (a * c + b) / (c * d)
   After: (a * c + b) / (c * d)
Result: (a * c + b) / (c * d)
ℹ️ Requires:
  • c ≠ 0
  • d ≠ 0
```

### Web / JSON Steps

1. `Cancelar factores en una fracción`
   - before: `(b/c + a)/d`
   - after: `(a · c + b)/(c · d)`
   - substeps:
     1. `Llevar el numerador a denominador común`
     2. `Incorporar el denominador externo`

## nested_fraction_sum_with_fraction_over_scalar_general_reverse (nested_fraction)

- Source: `(a*c+b)/(c*d)`
- Target: `(a + b/c)/d`
- Result: `(b / c + a) / d`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a * c + b) / (c * d)
Target: (b / c + a) / d
Strategy: nested fraction
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplificar fracción anidada]
   Before: (a * c + b) / (c * d)
   Cambio local: (a * c + b) / (c * d) -> (b / c + a) / d
   After: (b / c + a) / d
Result: (b / c + a) / d
ℹ️ Requires:
  • c ≠ 0
  • d ≠ 0
```

### Web / JSON Steps

1. `Simplificar fracción anidada`
   - before: `(a · c + b)/(c · d)`
   - after: `(b/c + a)/d`
   - substeps:
     1. `Reescribir el numerador sacando factor común c`

## nested_radical_denesting (simplify)

- Source: `sqrt(6 + 2*sqrt(5))`
- Target: `sqrt(5)+1`
- Result: `sqrt(5) + 1`
- Web step count: `2`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sqrt(2 * sqrt(5) + 6)
Target: sqrt(5) + 1
Strategy: rewrite radicals
Steps (Aggressive Mode):
1. sqrt(A^2 ± 2AB + B^2) = |A ± B|  [Reconocer un cuadrado perfecto bajo la raíz]
   Before: sqrt(2 * sqrt(5) + 6)
   Cambio local: sqrt(2 * sqrt(5) + 6) -> |sqrt(5) + 1|
   After: |sqrt(5) + 1|
2. |x² + ...| = x² + ...  [Quitar valor absoluto de una expresión no negativa]
   Before: |sqrt(5) + 1|
   Cambio local: |sqrt(5) + 1| -> sqrt(5) + 1
   After: sqrt(5) + 1
Result: sqrt(5) + 1
```

### Web / JSON Steps

1. `Reconocer un cuadrado perfecto bajo la raíz`
   - before: `sqrt(2 · sqrt(5) + 6)`
   - after: `|sqrt(5) + 1|`
   - substeps:
     1. `Reescribir el radicando como un cuadrado perfecto`
     2. `La raíz de un cuadrado da un valor absoluto`
2. `Quitar valor absoluto de una expresión no negativa`
   - before: `|sqrt(5) + 1|`
   - after: `sqrt(5) + 1`
   - substeps: none

## perfect_square_root_direct_power_to_abs (simplify)

- Source: `sqrt(x^2)`
- Target: `abs(x)`
- Result: `|x|`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sqrt(x^2)
Target: |x|
Strategy: rewrite radicals
Steps (Aggressive Mode):
1. Take the square root of a perfect square  [Reconocer un cuadrado perfecto bajo la raíz]
   Before: sqrt(x^(2))
   Cambio local: sqrt(x^(2)) -> |x|
   After: |x|
Result: |x|
```

### Web / JSON Steps

1. `Reconocer un cuadrado perfecto bajo la raíz`
   - before: `sqrt(x^2)`
   - after: `|x|`
   - substeps:
     1. `Identificar la base del cuadrado`
     2. `La raíz de un cuadrado da un valor absoluto`

## perfect_square_root_to_abs (simplify)

- Source: `sqrt(a^2 + 2*a*b + b^2)`
- Target: `abs(a+b)`
- Result: `|a + b|`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sqrt(a^2 + b^2 + 2 * a * b)
Target: |a + b|
Strategy: rewrite radicals
Steps (Aggressive Mode):
1. Take the square root of a perfect square  [Reconocer un cuadrado perfecto bajo la raíz]
   Before: sqrt(a^(2) + b^(2) + 2 * a * b)
   Cambio local: sqrt(a^(2) + b^(2) + 2 * a * b) -> |a + b|
   After: |a + b|
Result: |a + b|
```

### Web / JSON Steps

1. `Reconocer un cuadrado perfecto bajo la raíz`
   - before: `sqrt(a^2 + b^2 + 2 · a · b)`
   - after: `|a + b|`
   - substeps:
     1. `Reescribir el radicando como un cuadrado perfecto`
     2. `La raíz de un cuadrado da un valor absoluto`

## perfect_square_root_to_abs_with_passthrough (simplify)

- Source: `sqrt(a^2 + 2*a*b + b^2)+c`
- Target: `abs(a+b)+c`
- Result: `|a + b| + c`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sqrt(a^2 + b^2 + 2 * a * b) + c
Target: |a + b| + c
Strategy: rewrite radicals
Steps (Aggressive Mode):
1. Take the square root of a perfect square  [Reconocer un cuadrado perfecto bajo la raíz]
   Before: sqrt(a^(2) + b^(2) + 2 * a * b) + c
   Cambio local: sqrt(a^(2) + b^(2) + 2 * a * b) + c -> |a + b| + c
   After: |a + b| + c
Result: |a + b| + c
```

### Web / JSON Steps

1. `Reconocer un cuadrado perfecto bajo la raíz`
   - before: `sqrt(a^2 + b^2 + 2 · a · b) + c`
   - after: `|a + b| + c`
   - substeps:
     1. `Reescribir el radicando como un cuadrado perfecto`
     2. `La raíz de un cuadrado da un valor absoluto`

## pythagorean_factor_form_from_sin_sq (simplify)

- Source: `sin(x)^2`
- Target: `1-cos(x)^2`
- Result: `1 - cos(x)^2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(x)^2
Target: 1 - cos(x)^2
Strategy: rewrite trigs
Steps (Aggressive Mode):
1. 1 - cos²(x) = sin²(x)  [Aplicar identidad pitagórica]
   Before: sin(x)^(2)
   Cambio local: sin(x)^(2) -> 1 - cos(x)^(2)
   After: 1 - cos(x)^2
Result: 1 - cos(x)^(2)
```

### Web / JSON Steps

1. `Aplicar identidad pitagórica`
   - before: `sin(x)^2`
   - after: `1 - cos(x)^2`
   - substeps: none

## pythagorean_factor_form_to_cos_sq (simplify)

- Source: `1 - sin(x)^2`
- Target: `cos(x)^2`
- Result: `cos(x)^2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 1 - sin(x)^2
Target: cos(x)^2
Strategy: rewrite trigs
Steps (Aggressive Mode):
1. 1 - sin²(x) = cos²(x)  [Aplicar identidad pitagórica]
   Before: 1 - sin(x)^(2)
   Cambio local: 1 - sin(x)^(2) -> cos(x)^(2)
   After: cos(x)^2
Result: cos(x)^(2)
```

### Web / JSON Steps

1. `Aplicar identidad pitagórica`
   - before: `1 - sin(x)^2`
   - after: `cos(x)^2`
   - substeps: none

## pythagorean_identity (simplify)

- Source: `sin(x)^2 + cos(x)^2`
- Target: `1`
- Result: `1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(x)^2 + cos(x)^2
Target: 1
Strategy: rewrite trigs
Steps (Aggressive Mode):
1. Recognize sin²(u) + cos²(u) = 1  [Aplicar la identidad pitagórica]
   Before: sin(x)^(2) + cos(x)^(2)
   Cambio local: sin(x)^(2) + cos(x)^(2) -> 1
   After: 1
Result: 1
```

### Web / JSON Steps

1. `Aplicar la identidad pitagórica`
   - before: `sin(x)^2 + cos(x)^2`
   - after: `1`
   - substeps: none

## radical_notable_quotient (rationalize)

- Source: `(x^(3/2)-1)/(sqrt(x)-1)`
- Target: `sqrt(x)+x+1`
- Result: `sqrt(x) + x + 1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (x^(3 / 2) - 1) / (sqrt(x) - 1)
Target: sqrt(x) + x + 1
Strategy: rationalize
Steps (Aggressive Mode):
1. Polynomial division with opaque substitution  [Reconocer un cociente notable]
   Before: (sqrt(x^3) - 1) / (sqrt(x) - 1)
   Cambio local: (sqrt(x^3) - 1) / (sqrt(x) - 1) -> sqrt(x) + sqrt(x)^(2) + 1
   After: sqrt(x) + x + 1
Result: sqrt(x) + x + 1
ℹ️ Requires:
  • sqrt(x) - 1 ≠ 0
  • x ≥ 0
```

### Web / JSON Steps

1. `Reconocer un cociente notable`
   - before: `(sqrt(x^3) - 1)/(sqrt(x) - 1)`
   - after: `sqrt(x) + x + 1`
   - substeps: none

## rationalize_cube_root_sum_denominator (rationalize)

- Source: `1/(1+x^(1/3))`
- Target: `(1-x^(1/3)+x^(2/3))/(1+x)`
- Result: `(-x^(1 / 3) + x^(2 / 3) + 1) / (x + 1)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (x^(1 / 3) + 1)
Target: (-x^(1 / 3) + x^(2 / 3) + 1) / (x + 1)
Strategy: rationalize
Steps (Aggressive Mode):
1. Rationalize: cube root denominator via sum of cubes  [Racionalizar el denominador]
   Before: 1 / (x^(1/3) + 1)
   Cambio local: 1 / (x^(1/3) + 1) -> (-x^(1 / 3) + x^(2 / 3) + 1) / (x + 1)
   After: (-x^(1 / 3) + x^(2 / 3) + 1) / (x + 1)
Result: (-x^(1 / 3) + x^(2 / 3) + 1) / (x + 1)
ℹ️ Requires:
  • x ≠ -1
  • x^(1 / 3) + 1 ≠ 0
```

### Web / JSON Steps

1. `Racionalizar el denominador`
   - before: `1/(sqrt[3]x + 1)`
   - after: `(sqrt[3]x^2 + 1 - sqrt[3]x)/(x + 1)`
   - substeps:
     1. `Multiplicar por el conjugado cúbico`
     2. `Aplicar suma de cubos en el denominador`

## rationalize_linear_root (rationalize)

- Source: `1/(sqrt(x)-1)`
- Target: `(sqrt(x)+1)/(x-1)`
- Result: `(sqrt(x) + 1) / (x - 1)`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: 1 / (sqrt(x) - 1)
Target: (sqrt(x) + 1) / (x - 1)
Strategy: rationalize
Steps (Aggressive Mode):
1. Rationalize: multiply by conjugate  [Racionalizar el denominador]
   Before: 1 / (sqrt(x) - 1)
   After: (sqrt(x) + 1) / (x - 1)
Result: (sqrt(x) + 1) / (x - 1)
ℹ️ Requires:
  • x ≠ 1
  • x ≥ 0
```

### Web / JSON Steps

1. `Racionalizar el denominador`
   - before: `1/(sqrt(x) - 1)`
   - after: `(sqrt(x) + 1)/(x - 1^2)`
   - substeps:
     1. `Cambiar el signo para formar el conjugado`
     2. `Multiplicar numerador y denominador por ese conjugado`
     3. `En el denominador aparece una diferencia de cuadrados`

## rationalize_linear_root_plus (rationalize)

- Source: `1/(sqrt(x)+1)`
- Target: `(sqrt(x)-1)/(x-1)`
- Result: `(sqrt(x) - 1) / (x - 1)`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: 1 / (sqrt(x) + 1)
Target: (sqrt(x) - 1) / (x - 1)
Strategy: rationalize
Steps (Aggressive Mode):
1. Rationalize: multiply by conjugate  [Racionalizar el denominador]
   Before: 1 / (sqrt(x) + 1)
   After: (sqrt(x) - 1) / (x - 1)
Result: (sqrt(x) - 1) / (x - 1)
ℹ️ Requires:
  • x ≠ 1
  • x ≥ 0
```

### Web / JSON Steps

1. `Racionalizar el denominador`
   - before: `1/(sqrt(x) + 1)`
   - after: `(sqrt(x) - 1)/(x - 1)`
   - substeps:
     1. `Cambiar el signo para formar el conjugado`
     2. `Multiplicar numerador y denominador por ese conjugado`
     3. `En el denominador aparece una diferencia de cuadrados`

## rationalize_shifted_linear_root (rationalize)

- Source: `1/(sqrt(x)-2)`
- Target: `(sqrt(x)+2)/(x-4)`
- Result: `(sqrt(x) + 2) / (x - 4)`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: 1 / (sqrt(x) - 2)
Target: (sqrt(x) + 2) / (x - 4)
Strategy: rationalize
Steps (Aggressive Mode):
1. Rationalize: multiply by conjugate  [Racionalizar el denominador]
   Before: 1 / (sqrt(x) - 2)
   After: (sqrt(x) + 2) / (x - 4)
Result: (sqrt(x) + 2) / (x - 4)
ℹ️ Requires:
  • x ≠ 4
  • x ≥ 0
```

### Web / JSON Steps

1. `Racionalizar el denominador`
   - before: `1/(sqrt(x) - 2)`
   - after: `(sqrt(x) + 2)/(x - 2^2)`
   - substeps:
     1. `Cambiar el signo para formar el conjugado`
     2. `Multiplicar numerador y denominador por ese conjugado`
     3. `En el denominador aparece una diferencia de cuadrados`

## rationalize_symbolic_linear_root (rationalize)

- Source: `1/(sqrt(x)-a)`
- Target: `(sqrt(x)+a)/(x-a^2)`
- Result: `(sqrt(x) + a) / (x - a^2)`
- Web step count: `2`
- Web substep count: `4`
- Flags: none

### CLI

```text
Parsed: 1 / (sqrt(x) - a)
Target: (sqrt(x) + a) / (x - a^2)
Strategy: rationalize
Steps (Aggressive Mode):
1. Rationalize denominator (diff squares)  [Racionalizar el denominador]
   Before: 1 / (sqrt(x) - a)
   Cambio local: 1 / (sqrt(x) - a) -> (sqrt(x) + a) / (sqrt(x)^(2) + a^(2))
   After: (sqrt(x) + a) / (sqrt(x)^(2) + a^(2))
2.   [Deshacer raíz y potencia]
   Before: (sqrt(x) + a) / (sqrt(x)^(2) + a^(2))
   Cambio local: sqrt(x)^(2) -> x
   After: (sqrt(x) + a) / (x - a^2)
   ℹ️ Requires: x > 0
Result: (sqrt(x) + a) / (x - a^(2))
ℹ️ Requires:
  • sqrt(x) - a ≠ 0
  • a^2 - x ≠ 0
  • x ≥ 0
```

### Web / JSON Steps

1. `Racionalizar el denominador`
   - before: `1/(sqrt(x) - a)`
   - after: `(sqrt(x) + a)/(sqrt(x)^2 + a^2)`
   - substeps:
     1. `Cambiar el signo para formar el conjugado`
     2. `Multiplicar numerador y denominador por ese conjugado`
     3. `En el denominador aparece una diferencia de cuadrados`
2. `Deshacer raíz y potencia`
   - before: `(sqrt(x) + a)/(sqrt(x)^2 + a^2)`
   - after: `(sqrt(x) + a)/(x - a^2)`
   - substeps:
     1. `Reemplazar ese bloque en la expresión`

## rationalize_symbolic_linear_root_alt_var (rationalize)

- Source: `1/(sqrt(y)-a)`
- Target: `(sqrt(y)+a)/(y-a^2)`
- Result: `(sqrt(y) + a) / (y - a^2)`
- Web step count: `2`
- Web substep count: `4`
- Flags: none

### CLI

```text
Parsed: 1 / (sqrt(y) - a)
Target: (sqrt(y) + a) / (y - a^2)
Strategy: rationalize
Steps (Aggressive Mode):
1. Rationalize denominator (diff squares)  [Racionalizar el denominador]
   Before: 1 / (sqrt(y) - a)
   Cambio local: 1 / (sqrt(y) - a) -> (sqrt(y) + a) / (sqrt(y)^(2) + a^(2))
   After: (sqrt(y) + a) / (sqrt(y)^(2) + a^(2))
2.   [Deshacer raíz y potencia]
   Before: (sqrt(y) + a) / (sqrt(y)^(2) + a^(2))
   Cambio local: sqrt(y)^(2) -> y
   After: (sqrt(y) + a) / (y - a^2)
   ℹ️ Requires: y > 0
Result: (sqrt(y) + a) / (y - a^(2))
ℹ️ Requires:
  • sqrt(y) - a ≠ 0
  • a^2 - y ≠ 0
  • y ≥ 0
```

### Web / JSON Steps

1. `Racionalizar el denominador`
   - before: `1/(sqrt(y) - a)`
   - after: `(sqrt(y) + a)/(sqrt(y)^2 + a^2)`
   - substeps:
     1. `Cambiar el signo para formar el conjugado`
     2. `Multiplicar numerador y denominador por ese conjugado`
     3. `En el denominador aparece una diferencia de cuadrados`
2. `Deshacer raíz y potencia`
   - before: `(sqrt(y) + a)/(sqrt(y)^2 + a^2)`
   - after: `(sqrt(y) + a)/(y - a^2)`
   - substeps:
     1. `Reemplazar ese bloque en la expresión`

## rationalize_symbolic_linear_root_plus (rationalize)

- Source: `1/(sqrt(x)+a)`
- Target: `(sqrt(x)-a)/(x-a^2)`
- Result: `(sqrt(x) - a) / (x - a^2)`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: 1 / (sqrt(x) + a)
Target: (sqrt(x) - a) / (x - a^2)
Strategy: rationalize
Steps (Aggressive Mode):
1. Rationalize: multiply by conjugate  [Racionalizar el denominador]
   Before: 1 / (sqrt(x) + a)
   After: (sqrt(x) - a) / (x - a^2)
Result: (sqrt(x) - a) / (x - a^(2))
ℹ️ Requires:
  • sqrt(x) + a ≠ 0
  • a^2 - x ≠ 0
  • x ≥ 0
```

### Web / JSON Steps

1. `Racionalizar el denominador`
   - before: `1/(sqrt(x) + a)`
   - after: `(sqrt(x) - a)/(x - a^2)`
   - substeps:
     1. `Cambiar el signo para formar el conjugado`
     2. `Multiplicar numerador y denominador por ese conjugado`
     3. `En el denominador aparece una diferencia de cuadrados`

## rationalize_then_cancel_to_zero (rationalize)

- Source: `1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)`
- Target: `0`
- Result: `0`
- Web step count: `2`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: 1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)
Target: 0
Strategy: rationalize
Steps (Aggressive Mode):
1. Rationalize linear sqrt denominator  [Racionalizar el denominador]
   Before: 1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)
   After: (sqrt(x) + 1) / (x - 1) - (sqrt(x) + 1) / (x - 1)
2. Subtract two identical expressions  [Restar dos expresiones iguales]
   Before: (sqrt(x) + 1) / (x - 1) - (sqrt(x) + 1) / (x - 1)
   Cambio local: (sqrt(x) + 1) / (x - 1) - (sqrt(x) + 1) / (x - 1) -> 0
   After: 0
Result: 0
ℹ️ Requires:
  • x ≠ 1
  • x ≥ 0
```

### Web / JSON Steps

1. `Racionalizar el denominador`
   - before: `1/(sqrt(x) - 1) - (sqrt(x) + 1)/(x - 1)`
   - after: `(sqrt(x) + 1)/(x - 1) - (sqrt(x) + 1)/(x - 1)`
   - substeps:
     1. `Cambiar el signo para formar el conjugado`
     2. `Multiplicar numerador y denominador por ese conjugado`
     3. `En el denominador aparece una diferencia de cuadrados`
2. `Restar dos expresiones iguales`
   - before: `(sqrt(x) + 1)/(x - 1) - (sqrt(x) + 1)/(x - 1)`
   - after: `0`
   - substeps: none

## reciprocal_trig_cos_sec_product_to_one (simplify)

- Source: `cos(x)*sec(x)`
- Target: `1`
- Result: `1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cos(x) * sec(x)
Target: 1
Strategy: rewrite trigs
Steps (Aggressive Mode):
1. Recognize cos(u) · sec(u) = 1  [Cancelar funciones trigonométricas recíprocas]
   Before: cos(x) * sec(x)
   Cambio local: cos(x) * sec(x) -> 1
   After: 1
Result: 1
```

### Web / JSON Steps

1. `Reconocer coseno por secante como 1`
   - before: `cos(x) · sec(x)`
   - after: `1`
   - substeps: none

## reciprocal_trig_product_to_one (simplify)

- Source: `tan(x)*cot(x)`
- Target: `1`
- Result: `1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: tan(x) * cot(x)
Target: 1
Strategy: rewrite trigs
Steps (Aggressive Mode):
1. Recognize tan(u) · cot(u) = 1  [Cancelar funciones trigonométricas recíprocas]
   Before: tan(x) * cot(x)
   Cambio local: tan(x) * cot(x) -> 1
   After: 1
Result: 1
```

### Web / JSON Steps

1. `Reconocer tangente por cotangente como 1`
   - before: `tan(x) · cot(x)`
   - after: `1`
   - substeps: none

## reciprocal_trig_product_to_one_with_passthrough (simplify)

- Source: `tan(x)*cot(x)+a`
- Target: `1+a`
- Result: `a + 1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: tan(x) * cot(x) + a
Target: a + 1
Strategy: rewrite trigs
Steps (Aggressive Mode):
1. Recognize tan(u) · cot(u) = 1  [Cancelar funciones trigonométricas recíprocas]
   Before: tan(x) * cot(x) + a
   Cambio local: tan(x) * cot(x) + a -> a + 1
   After: a + 1
Result: a + 1
```

### Web / JSON Steps

1. `Reconocer tangente por cotangente como 1`
   - before: `tan(x) · cot(x) + a`
   - after: `a + 1`
   - substeps: none

## reciprocal_trig_sin_csc_product_to_one (simplify)

- Source: `sin(x)*csc(x)`
- Target: `1`
- Result: `1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(x) * csc(x)
Target: 1
Strategy: rewrite trigs
Steps (Aggressive Mode):
1. Recognize sin(u) · csc(u) = 1  [Cancelar funciones trigonométricas recíprocas]
   Before: sin(x) * csc(x)
   Cambio local: sin(x) * csc(x) -> 1
   After: 1
Result: 1
```

### Web / JSON Steps

1. `Reconocer seno por cosecante como 1`
   - before: `sin(x) · csc(x)`
   - after: `1`
   - substeps: none

## reciprocal_trig_special_value_sec_pi_fourth (simplify)

- Source: `sec(pi/4)`
- Target: `sqrt(2)`
- Result: `sqrt(2)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sec(pi / 4)
Target: sqrt(2)
Strategy: rewrite trigs
Steps (Aggressive Mode):
1. Evaluate a trigonometric function at a special input  [Evaluar valor trigonométrico especial]
   Before: sec(pi / 4)
   Cambio local: sec(pi / 4) -> sqrt(2)
   After: sqrt(2)
Result: sqrt(2)
```

### Web / JSON Steps

1. `Evaluar valor trigonométrico especial`
   - before: `sec(pi/4)`
   - after: `sqrt(2)`
   - substeps: none

## sec_tan_pythagorean_to_one (simplify)

- Source: `sec(x)^2 - tan(x)^2`
- Target: `1`
- Result: `1`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sec(x)^2 - tan(x)^2
Target: 1
Strategy: rewrite trigs
Steps (Aggressive Mode):
1. Recognize sec²(u) - tan²(u) = 1  [Aplicar identidad pitagórica recíproca]
   Before: sec(x)^(2) - tan(x)^(2)
   Cambio local: sec(x)^(2) - tan(x)^(2) -> 1
   After: 1
Result: 1
```

### Web / JSON Steps

1. `Aplicar identidad pitagórica recíproca`
   - before: `sec(x)^2 - tan(x)^2`
   - after: `1`
   - substeps: none

## simplify_sqrt_arithmetic_difference (simplify)

- Source: `sqrt(18)-sqrt(2)`
- Target: `2*sqrt(2)`
- Result: `2 * sqrt(2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sqrt(18) - sqrt(2)
Target: 2 * sqrt(2)
Strategy: combine like terms
Steps (Aggressive Mode):
1. Combine like terms  [Agrupar términos semejantes]
   Before: 3 * sqrt(2) - sqrt(2)
   Cambio local: 3 * sqrt(2) - sqrt(2) -> 2 * sqrt(2)
   After: 2 * sqrt(2)
Result: 2 * sqrt(2)
```

### Web / JSON Steps

1. `Agrupar términos semejantes`
   - before: `3 · sqrt(2) - sqrt(2)`
   - after: `2 · sqrt(2)`
   - substeps:
     1. `Sumar los coeficientes que acompañan a sqrt(2)`

## simplify_sqrt_arithmetic_sum (simplify)

- Source: `sqrt(8)+sqrt(2)`
- Target: `3*sqrt(2)`
- Result: `3 * sqrt(2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sqrt(2) + sqrt(8)
Target: 3 * sqrt(2)
Strategy: combine like terms
Steps (Aggressive Mode):
1. Combine like terms  [Agrupar términos semejantes]
   Before: sqrt(2) + 2 * sqrt(2)
   Cambio local: sqrt(2) + 2 * sqrt(2) -> 3 * sqrt(2)
   After: 3 * sqrt(2)
Result: 3 * sqrt(2)
```

### Web / JSON Steps

1. `Agrupar términos semejantes`
   - before: `sqrt(2) + 2 · sqrt(2)`
   - after: `3 · sqrt(2)`
   - substeps:
     1. `Sumar los coeficientes que acompañan a sqrt(2)`

## sin_arccos_complement_projection (simplify)

- Source: `sin(arccos(x))`
- Target: `sqrt(1-x^2)`
- Result: `sqrt(1 - x^2)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sin(arccos(x))
Target: sqrt(1 - x^2)
Strategy: rewrite inverse trigs
Steps (Aggressive Mode):
1. sin(arccos(x)) = sqrt(1-x^2)  [Aplicar composición trigonométrica inversa]
   Before: sin(arccos(x))
   Cambio local: sin(arccos(x)) -> sqrt(1 - x^(2))
   After: sqrt(1 - x^2)
Result: sqrt(1 - x^(2))
ℹ️ Requires:
  • -1 ≤ x ≤ 1
```

### Web / JSON Steps

1. `Aplicar composición trigonométrica inversa`
   - before: `sin(arccos(x))`
   - after: `sqrt(1 - x^2)`
   - substeps:
     1. `Calcular el cateto restante del triángulo asociado a arccos(x)`
     2. `Leer el seno desde ese triángulo`

## sin_arctan_right_triangle_projection (simplify)

- Source: `sin(arctan(x))`
- Target: `x/sqrt(1+x^2)`
- Result: `x / sqrt(x^2 + 1)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sin(arctan(x))
Target: x / sqrt(x^2 + 1)
Strategy: rewrite inverse trigs
Steps (Aggressive Mode):
1. sin(arctan(x)) = x/sqrt(1+x^2)  [Aplicar composición trigonométrica inversa]
   Before: sin(arctan(x))
   Cambio local: sin(arctan(x)) -> x / sqrt(x^(2) + 1)
   After: x / sqrt(x^2 + 1)
Result: x / sqrt(x^(2) + 1)
```

### Web / JSON Steps

1. `Aplicar composición trigonométrica inversa`
   - before: `sin(arctan(x))`
   - after: `x/sqrt(x^2 + 1)`
   - substeps:
     1. `Calcular la hipotenusa del triángulo asociado a arctan(x)`
     2. `Leer el seno desde ese triángulo`

## solve_prep_complete_square_alt_variable_symbolic_leading_coeff (solve_prep)

- Source: `a*y^2 + b*y + c`
- Target: `a*(y + b/(2*a))^2 + c - b^2/(4*a)`
- Result: `a * (b / (2 * a) + y)^2 + c - b^2 / (4 * a)`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: a * y^2 + b * y + c
Target: a * (b / (2 * a) + y)^2 + c - b^2 / (4 * a)
Strategy: solve prep
Steps (Aggressive Mode):
1. Complete the square to rewrite the quadratic  [Completar el cuadrado]
   Before: a * y^(2) + b * y + c
   Cambio local: a * y^(2) + b * y + c -> a * (b / (2 * a) + y)^(2) + c - b^(2) / (4 * a)
   After: a * (b / (2 * a) + y)^2 + c - b^2 / (4 * a)
Result: a * (b / (2 * a) + y)^(2) + c - b^(2) / (4 * a)
ℹ️ Requires:
  • a ≠ 0
```

### Web / JSON Steps

1. `Completar el cuadrado`
   - before: `a · y^2 + b · y + c`
   - after: `a · (b/(2 · a) + y)^2 + c - b^2/(4 · a)`
   - substeps:
     1. `Extraer el coeficiente líder de los términos cuadráticos`
     2. `Añadir y restar el cuadrado del semicoeficiente dentro del paréntesis`
     3. `Agrupar el trinomio como cuadrado perfecto`

## solve_prep_complete_square_fractional_monic_numeric (solve_prep)

- Source: `x^2 + 3*x + 1`
- Target: `(x+3/2)^2 - 5/4`
- Result: `(3 / 2 + x)^2 - 5 / 4`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: x^2 + 3 * x + 1
Target: (3 / 2 + x)^2 - 5 / 4
Strategy: solve prep
Steps (Aggressive Mode):
1. Complete the square to rewrite the quadratic  [Completar el cuadrado]
   Before: x^(2) + 3 * x + 1
   Subpasos:
     1.1 Añadir y restar el cuadrado del semicoeficiente
         x^2 + 3 * x + 1 -> 3/2^2 + x^2 + 3 * x + 1 - 3/2^2
     1.2 Agrupar el trinomio como cuadrado perfecto
         3/2^2 + x^2 + 3 * x + 1 - 3/2^2 -> (x + 3/2)^2 - 5/4
   Cambio local: x^(2) + 3 * x + 1 -> (3 / 2 + x)^(2) - 5 / 4
   After: (3 / 2 + x)^2 - 5 / 4
Result: (3 / 2 + x)^(2) - 5 / 4
```

### Web / JSON Steps

1. `Completar el cuadrado`
   - before: `x^2 + 3 · x + 1`
   - after: `(3/2 + x)^2 - 5/4`
   - substeps:
     1. `Añadir y restar el cuadrado del semicoeficiente`
     2. `Agrupar el trinomio como cuadrado perfecto`

## solve_prep_complete_square_fractional_symbolic_leading_coeff (solve_prep)

- Source: `(a/2)*x^2 + b*x + c`
- Target: `(a/2)*(x + b/a)^2 + c - b^2/(2*a)`
- Result: `(a * (b / a + x)^2)/2 + c - b^2 / (2 * a)`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: (a * x^2)/2 + b * x + c
Target: (a * (b / a + x)^2)/2 + c - b^2 / (2 * a)
Strategy: solve prep
Steps (Aggressive Mode):
1. Complete the square to rewrite the quadratic  [Completar el cuadrado]
   Before: x^(2) * a / 2 + b * x + c
   Cambio local: x^(2) * a / 2 + b * x + c -> (b / a + x)^(2) * a / 2 + c - b^(2) / (2 * a)
   After: (a * (b / a + x)^2)/2 + c - b^2 / (2 * a)
Result: (b / a + x)^(2) * a / 2 + c - b^(2) / (2 * a)
ℹ️ Requires:
  • a ≠ 0
```

### Web / JSON Steps

1. `Completar el cuadrado`
   - before: `x^2 · a/2 + b · x + c`
   - after: `(b/a + x)^2 · a/2 + c - b^2/(2 · a)`
   - substeps:
     1. `Extraer el coeficiente líder de los términos cuadráticos`
     2. `Añadir y restar el cuadrado del semicoeficiente dentro del paréntesis`
     3. `Agrupar el trinomio como cuadrado perfecto`

## solve_prep_complete_square_monic_numeric (solve_prep)

- Source: `x^2 + 6*x + 5`
- Target: `(x+3)^2 - 4`
- Result: `(x + 3)^2 - 4`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: x^2 + 6 * x + 5
Target: (x + 3)^2 - 4
Strategy: solve prep
Steps (Aggressive Mode):
1. Complete the square to rewrite the quadratic  [Completar el cuadrado]
   Before: x^(2) + 6 * x + 5
   Subpasos:
     1.1 Añadir y restar el cuadrado del semicoeficiente
         x^2 + 6 * x + 5 -> 3^2 + x^2 + 6 * x + 5 - 3^2
     1.2 Agrupar el trinomio como cuadrado perfecto
         3^2 + x^2 + 6 * x + 5 - 3^2 -> (x + 3)^2 - 4
   Cambio local: x^(2) + 6 * x + 5 -> (x + 3)^(2) - 4
   After: (x + 3)^2 - 4
Result: (x + 3)^(2) - 4
```

### Web / JSON Steps

1. `Completar el cuadrado`
   - before: `x^2 + 6 · x + 5`
   - after: `(x + 3)^2 - 4`
   - substeps:
     1. `Añadir y restar el cuadrado del semicoeficiente`
     2. `Agrupar el trinomio como cuadrado perfecto`

## solve_prep_complete_square_negative_symbolic_leading_coeff (solve_prep)

- Source: `-a*x^2 + b*x + c`
- Target: `-a*(x - b/(2*a))^2 + c + b^2/(4*a)`
- Result: `b^2 / (4 * a) + c - a * (x - b / (2 * a))^2`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: b * x + c - a * x^2
Target: b^2 / (4 * a) + c - a * (x - b / (2 * a))^2
Strategy: solve prep
Steps (Aggressive Mode):
1. Complete the square to rewrite the quadratic  [Completar el cuadrado]
   Before: b * x + c - a * x^(2)
   Cambio local: b * x + c - a * x^(2) -> b^(2) / (4 * a) + c - a * (x - b / (2 * a))^(2)
   After: b^2 / (4 * a) + c - a * (x - b / (2 * a))^2
Result: b^(2) / (4 * a) + c - a * (x - b / (2 * a))^(2)
ℹ️ Requires:
  • a ≠ 0
```

### Web / JSON Steps

1. `Completar el cuadrado`
   - before: `b · x + c - a · x^2`
   - after: `b^2/(4 · a) + c - a · (x - b/(2 · a))^2`
   - substeps:
     1. `Extraer el coeficiente líder de los términos cuadráticos`
     2. `Añadir y restar el cuadrado del semicoeficiente dentro del paréntesis`
     3. `Agrupar el trinomio como cuadrado perfecto`

## solve_prep_complete_square_symbolic_leading_coeff (solve_prep)

- Source: `a*x^2 + b*x + c`
- Target: `a*(x + b/(2*a))^2 + c - b^2/(4*a)`
- Result: `a * (b / (2 * a) + x)^2 + c - b^2 / (4 * a)`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: a * x^2 + b * x + c
Target: a * (b / (2 * a) + x)^2 + c - b^2 / (4 * a)
Strategy: solve prep
Steps (Aggressive Mode):
1. Complete the square to rewrite the quadratic  [Completar el cuadrado]
   Before: a * x^(2) + b * x + c
   Cambio local: a * x^(2) + b * x + c -> a * (b / (2 * a) + x)^(2) + c - b^(2) / (4 * a)
   After: a * (b / (2 * a) + x)^2 + c - b^2 / (4 * a)
Result: a * (b / (2 * a) + x)^(2) + c - b^(2) / (4 * a)
ℹ️ Requires:
  • a ≠ 0
```

### Web / JSON Steps

1. `Completar el cuadrado`
   - before: `a · x^2 + b · x + c`
   - after: `a · (b/(2 · a) + x)^2 + c - b^2/(4 · a)`
   - substeps:
     1. `Extraer el coeficiente líder de los términos cuadráticos`
     2. `Añadir y restar el cuadrado del semicoeficiente dentro del paréntesis`
     3. `Agrupar el trinomio como cuadrado perfecto`

## solve_prep_complete_square_symbolic_monic_parametric (solve_prep)

- Source: `x^2 + 2*b*x + c`
- Target: `(x+b)^2 + c - b^2`
- Result: `(b + x)^2 + c - b^2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: x^2 + 2 * b * x + c
Target: (b + x)^2 + c - b^2
Strategy: solve prep
Steps (Aggressive Mode):
1. Complete the square to rewrite the quadratic  [Completar el cuadrado]
   Before: x^(2) + 2 * b * x + c
   Subpasos:
     1.1 Añadir y restar el cuadrado del semicoeficiente
         x^2 + 2 * b * x + c -> b^2 + x^2 + 2 * b * x + c - b^2
     1.2 Agrupar el trinomio como cuadrado perfecto
         b^2 + x^2 + 2 * b * x + c - b^2 -> (b + x)^2 + c - b^2
   Cambio local: x^(2) + 2 * b * x + c -> (b + x)^(2) + c - b^(2)
   After: (b + x)^2 + c - b^2
Result: (b + x)^(2) + c - b^(2)
```

### Web / JSON Steps

1. `Completar el cuadrado`
   - before: `x^2 + 2 · b · x + c`
   - after: `(b + x)^2 + c - b^2`
   - substeps:
     1. `Añadir y restar el cuadrado del semicoeficiente`
     2. `Agrupar el trinomio como cuadrado perfecto`

## solve_prep_complete_square_symbolic_negative_linear_coeff (solve_prep)

- Source: `a*x^2 - b*x + c`
- Target: `a*(x - b/(2*a))^2 + c - b^2/(4*a)`
- Result: `a * (x - b / (2 * a))^2 + c - b^2 / (4 * a)`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: a * x^2 - b * x + c
Target: a * (x - b / (2 * a))^2 + c - b^2 / (4 * a)
Strategy: solve prep
Steps (Aggressive Mode):
1. Complete the square to rewrite the quadratic  [Completar el cuadrado]
   Before: a * x^(2) - b * x + c
   Cambio local: a * x^(2) - b * x + c -> a * (x - b / (2 * a))^(2) + c - b^(2) / (4 * a)
   After: a * (x - b / (2 * a))^2 + c - b^2 / (4 * a)
Result: a * (x - b / (2 * a))^(2) + c - b^(2) / (4 * a)
ℹ️ Requires:
  • a ≠ 0
```

### Web / JSON Steps

1. `Completar el cuadrado`
   - before: `a · x^2 - b · x + c`
   - after: `a · (x - b/(2 · a))^2 + c - b^2/(4 · a)`
   - substeps:
     1. `Extraer el coeficiente líder de los términos cuadráticos`
     2. `Añadir y restar el cuadrado del semicoeficiente dentro del paréntesis`
     3. `Agrupar el trinomio como cuadrado perfecto`

## split_fraction_into_whole_plus_remainder (fraction_decompose)

- Source: `(x+1)/(x-1)`
- Target: `1 + 2/(x-1)`
- Result: `2 / (x - 1) + 1`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (x + 1) / (x - 1)
Target: 2 / (x - 1) + 1
Strategy: split fraction
Steps (Aggressive Mode):
1. Split a fraction into a whole part plus remainder  [Separar fracción en parte entera y resto]
   Before: (x + 1) / (x - 1)
   After: 2 / (x - 1) + 1
Result: 2 / (x - 1) + 1
ℹ️ Requires:
  • x ≠ 1
```

### Web / JSON Steps

1. `Separar parte entera y resto`
   - before: `(x + 1)/(x - 1)`
   - after: `2/(x - 1) + 1`
   - substeps:
     1. `Reescribir el numerador como parte entera por denominador más resto`
     2. `Separar la suma del numerador sobre el denominador`

## split_fraction_linear_over_scaled_linear (fraction_decompose)

- Source: `(4*x+7)/(2*x+1)`
- Target: `2 + 5/(2*x+1)`
- Result: `5 / (2 * x + 1) + 2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (4 * x + 7) / (2 * x + 1)
Target: 5 / (2 * x + 1) + 2
Strategy: split fraction
Steps (Aggressive Mode):
1. Split a fraction into a whole part plus remainder  [Separar fracción en parte entera y resto]
   Before: (4 * x + 7) / (2 * x + 1)
   After: 5 / (2 * x + 1) + 2
Result: 5 / (2 * x + 1) + 2
ℹ️ Requires:
  • x ≠ -1/2
```

### Web / JSON Steps

1. `Separar parte entera y resto`
   - before: `(4 · x + 7)/(2 · x + 1)`
   - after: `5/(2 · x + 1) + 2`
   - substeps:
     1. `Reescribir el numerador como parte entera por denominador más resto`
     2. `Separar la suma del numerador sobre el denominador`

## split_fraction_symbolic_over_general_shift (fraction_decompose)

- Source: `(a*x+b)/(x+c)`
- Target: `a + (b-a*c)/(x+c)`
- Result: `(b - a * c) / (c + x) + a`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a * x + b) / (c + x)
Target: (b - a * c) / (c + x) + a
Strategy: split fraction
Steps (Aggressive Mode):
1. Split a fraction into a whole part plus remainder  [Separar fracción en parte entera y resto]
   Before: (a * x + b) / (c + x)
   After: (b - a * c) / (c + x) + a
Result: (b - a * c) / (c + x) + a
ℹ️ Requires:
  • c + x ≠ 0
```

### Web / JSON Steps

1. `Separar parte entera y resto`
   - before: `(a · x + b)/(c + x)`
   - after: `(b - a · c)/(c + x) + a`
   - substeps:
     1. `Reescribir el numerador como parte entera por denominador más resto`
     2. `Separar la suma del numerador sobre el denominador`

## split_fraction_symbolic_over_negative_scaled_general_linear (fraction_decompose)

- Source: `(a*x+b)/(d-c*x)`
- Target: `-a/c + (b+a*d/c)/(d-c*x)`
- Result: `(a * d / c + b) / (d - c * x) - a / c`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a * x + b) / (d - c * x)
Target: (a * d / c + b) / (d - c * x) - a / c
Strategy: split fraction
Steps (Aggressive Mode):
1. Split a fraction into a whole part plus remainder  [Separar fracción en parte entera y resto]
   Before: (a * x + b) / (d - c * x)
   After: (a * d / c + b) / (d - c * x) - a / c
Result: (a * d / c + b) / (d - c * x) - a / c
ℹ️ Requires:
  • c ≠ 0
  • c * x - d ≠ 0
```

### Web / JSON Steps

1. `Separar parte entera y resto`
   - before: `(a · x + b)/(d - c · x)`
   - after: `((a · d)/c + b)/(d - c · x) - a/c`
   - substeps:
     1. `Reescribir el numerador como parte entera por denominador más resto`
     2. `Separar la suma del numerador sobre el denominador`

## split_fraction_symbolic_over_scaled_general_linear (fraction_decompose)

- Source: `(a*x+b)/(c*x+d)`
- Target: `a/c + (b-a*d/c)/(c*x+d)`
- Result: `a / c + (b - a * d / c) / (c * x + d)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a * x + b) / (c * x + d)
Target: a / c + (b - a * d / c) / (c * x + d)
Strategy: split fraction
Steps (Aggressive Mode):
1. Split a fraction into a whole part plus remainder  [Separar fracción en parte entera y resto]
   Before: (a * x + b) / (c * x + d)
   After: a / c + (b - a * d / c) / (c * x + d)
Result: a / c + (b - a * d / c) / (c * x + d)
ℹ️ Requires:
  • c * x + d ≠ 0
  • c ≠ 0
```

### Web / JSON Steps

1. `Separar parte entera y resto`
   - before: `(a · x + b)/(c · x + d)`
   - after: `a/c + (b - a · d/c)/(c · x + d)`
   - substeps:
     1. `Reescribir el numerador como parte entera por denominador más resto`
     2. `Separar la suma del numerador sobre el denominador`

## split_telescoping_fraction_affine_gap_two (telescoping_fraction)

- Source: `1/((2*n+1)*(2*n+3))`
- Target: `1/2*(1/(2*n+1) - 1/(2*n+3))`
- Result: `((1 / (2 * n + 1) - 1 / (2 * n + 3)) * 1)/2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / ((2 * n + 1) * (2 * n + 3))
Target: ((1 / (2 * n + 1) - 1 / (2 * n + 3)))/2
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Descomponer en fracciones parciales telescópicas]
   Before: 1 / ((2 * n + 1) * (2 * n + 3))
   Cambio local: 1 / ((2 * n + 1) * (2 * n + 3)) -> 1 / 2 * (1 / (2 * n + 1) - 1 / (2 * n + 3))
   After: ((1 / (2 * n + 1) - 1 / (2 * n + 3)))/2
Result: 1 / 2 * (1 / (2 * n + 1) - 1 / (2 * n + 3))
ℹ️ Requires:
  • n ≠ -1/2
  • n ≠ -3/2
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/((2 · n + 1) · (2 · n + 3))`
   - after: `1/2 · (1/(2 · n + 1) - 1/(2 · n + 3))`
   - substeps:
     1. `Introducir el numerador telescópico`
     2. `Separar sobre el denominador común`

## split_telescoping_fraction_affine_symbolic_shift_gap (telescoping_fraction)

- Source: `1/((a*n+b)*(a*n+c))`
- Target: `1/(c-b)*(1/(a*n+b) - 1/(a*n+c))`
- Result: `((1 / (a * n + b) - 1 / (a * n + c)) * 1)/(c - b)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / ((a * n + b) * (a * n + c))
Target: ((1 / (a * n + b) - 1 / (a * n + c)))/(c - b)
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Descomponer en fracciones parciales telescópicas]
   Before: 1 / ((a * n + b) * (a * n + c))
   Cambio local: 1 / ((a * n + b) * (a * n + c)) -> 1 / (c - b) * (1 / (a * n + b) - 1 / (a * n + c))
   After: ((1 / (a * n + b) - 1 / (a * n + c)))/(c - b)
Result: 1 / (c - b) * (1 / (a * n + b) - 1 / (a * n + c))
ℹ️ Requires:
  • a * n + b ≠ 0
  • a * n + c ≠ 0
  • b - c ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/((a · n + b) · (a · n + c))`
   - after: `1/(c - b) · (1/(a · n + b) - 1/(a · n + c))`
   - substeps:
     1. `Introducir el numerador telescópico`
     2. `Separar sobre el denominador común`

## split_telescoping_fraction_consecutive (telescoping_fraction)

- Source: `1/(n*(n+1))`
- Target: `1/n - 1/(n+1)`
- Result: `1 / n - 1 / (n + 1)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (n * (n + 1))
Target: 1 / n - 1 / (n + 1)
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Descomponer en fracciones parciales telescópicas]
   Before: 1 / (n * (n + 1))
   Subpasos:
     1.1 Introducir el numerador telescópico
         1 / (n * (n + 1)) -> (n + 1 - n) / (n * (n + 1))
     1.2 Separar sobre el denominador común
         (n + 1 - n) / (n * (n + 1)) -> 1 / n - 1 / (n + 1)
   Cambio local: 1 / (n * (n + 1)) -> 1 / n - 1 / (n + 1)
   After: 1 / n - 1 / (n + 1)
Result: 1 / n - 1 / (n + 1)
ℹ️ Requires:
  • n ≠ 0
  • n ≠ -1
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/(n · (n + 1))`
   - after: `1/n - 1/(n + 1)`
   - substeps:
     1. `Introducir el numerador telescópico`
     2. `Separar sobre el denominador común`

## split_telescoping_fraction_difference_squares_unfactored (telescoping_fraction)

- Source: `1/(x^2-1)`
- Target: `1/2*(1/(x-1) - 1/(x+1))`
- Result: `((1 / (x - 1) - 1 / (x + 1)) * 1)/2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (x^2 - 1)
Target: ((1 / (x - 1) - 1 / (x + 1)))/2
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Descomponer en fracciones parciales telescópicas]
   Before: 1 / (x^(2) - 1)
   Cambio local: 1 / (x^(2) - 1) -> 1 / 2 * (1 / (x - 1) - 1 / (x + 1))
   After: ((1 / (x - 1) - 1 / (x + 1)))/2
Result: 1 / 2 * (1 / (x - 1) - 1 / (x + 1))
ℹ️ Requires:
  • x ≠ -1
  • x ≠ 1
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/(x^2 - 1)`
   - after: `1/2 · (1/(x - 1) - 1/(x + 1))`
   - substeps:
     1. `Introducir el numerador telescópico`
     2. `Separar sobre el denominador común`

## split_telescoping_fraction_gap_two (telescoping_fraction)

- Source: `1/(n*(n+2))`
- Target: `1/2*(1/n - 1/(n+2))`
- Result: `((1 / n - 1 / (n + 2)) * 1)/2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (n * (n + 2))
Target: ((1 / n - 1 / (n + 2)))/2
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Descomponer en fracciones parciales telescópicas]
   Before: 1 / (n * (n + 2))
   Cambio local: 1 / (n * (n + 2)) -> 1 / 2 * (1 / n - 1 / (n + 2))
   After: ((1 / n - 1 / (n + 2)))/2
Result: 1 / 2 * (1 / n - 1 / (n + 2))
ℹ️ Requires:
  • n ≠ 0
  • n ≠ -2
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/(n · (n + 2))`
   - after: `1/2 · (1/n - 1/(n + 2))`
   - substeps:
     1. `Introducir el numerador telescópico`
     2. `Separar sobre el denominador común`

## split_telescoping_fraction_negative_gap_two (telescoping_fraction)

- Source: `1/(n*(n-2))`
- Target: `1/2*(1/(n-2) - 1/n)`
- Result: `((1 / (n - 2) - 1 / n) * 1)/2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (n * (n - 2))
Target: ((1 / (n - 2) - 1 / n))/2
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Descomponer en fracciones parciales telescópicas]
   Before: 1 / (n * (n - 2))
   Cambio local: 1 / (n * (n - 2)) -> 1 / 2 * (1 / (n - 2) - 1 / n)
   After: ((1 / (n - 2) - 1 / n))/2
Result: 1 / 2 * (1 / (n - 2) - 1 / n)
ℹ️ Requires:
  • n ≠ 0
  • n ≠ 2
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/(n · (n - 2))`
   - after: `1/2 · (1/(n - 2) - 1/n)`
   - substeps:
     1. `Introducir el numerador telescópico`
     2. `Separar sobre el denominador común`

## split_telescoping_fraction_symbolic_difference_squares_unfactored (telescoping_fraction)

- Source: `1/(x^2-a^2)`
- Target: `1/(2*a)*(1/(x-a) - 1/(x+a))`
- Result: `((1 / (x - a) - 1 / (a + x)) * 1)/(2 * a)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (x^2 - a^2)
Target: ((1 / (x - a) - 1 / (a + x)))/(2 * a)
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Descomponer en fracciones parciales telescópicas]
   Before: 1 / (x^(2) - a^(2))
   Cambio local: 1 / (x^(2) - a^(2)) -> 1 / (2 * a) * (1 / (x - a) - 1 / (a + x))
   After: ((1 / (x - a) - 1 / (a + x)))/(2 * a)
Result: 1 / (2 * a) * (1 / (x - a) - 1 / (a + x))
ℹ️ Requires:
  • a + x ≠ 0
  • a ≠ 0
  • a - x ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/(x^2 - a^2)`
   - after: `1/(2 · a) · (1/(x - a) - 1/(a + x))`
   - substeps:
     1. `Introducir el numerador telescópico`
     2. `Separar sobre el denominador común`

## square_of_square_root_requires_nonnegative (simplify)

- Source: `sqrt(x)^2`
- Target: `x`
- Result: `x`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sqrt(x)^2
Target: x
Strategy: rewrite radicals
Steps (Aggressive Mode):
1. Square a radical under its domain condition  [Deshacer raíz y potencia]
   Before: sqrt(x)^(2)
   Cambio local: sqrt(x)^(2) -> x
   After: x
Result: x
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Deshacer raíz y potencia`
   - before: `sqrt(x)^2`
   - after: `x`
   - substeps:
     1. `Identificar el radicando de la raíz principal`
     2. `El cuadrado deshace la raíz bajo la condición u ≥ 0`

## tan_arcsin_tangent_projection (simplify)

- Source: `tan(arcsin(x))`
- Target: `x/sqrt(1-x^2)`
- Result: `x / sqrt(1 - x^2)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: tan(arcsin(x))
Target: x / sqrt(1 - x^2)
Strategy: rewrite inverse trigs
Steps (Aggressive Mode):
1. tan(arcsin(x)) = x/sqrt(1-x^2)  [Aplicar composición trigonométrica inversa]
   Before: tan(arcsin(x))
   Cambio local: tan(arcsin(x)) -> x / sqrt(1 - x^(2))
   After: x / sqrt(1 - x^2)
Result: x / sqrt(1 - x^(2))
ℹ️ Requires:
  • -1 < x < 1
```

### Web / JSON Steps

1. `Aplicar composición trigonométrica inversa`
   - before: `tan(arcsin(x))`
   - after: `x/sqrt(1 - x^2)`
   - substeps:
     1. `Calcular el cateto restante del triángulo asociado a arcsin(x)`
     2. `Leer la tangente desde ese triángulo`

## trig_special_value_cos_two_pi_thirds_negative_half (simplify)

- Source: `cos(2*pi/3)`
- Target: `-1/2`
- Result: `-1 / 2`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: cos(2 * pi / 3)
Target: -1 / 2
Strategy: rewrite trigs
Steps (Aggressive Mode):
1. Evaluate a trigonometric function at a special input  [Evaluar valor trigonométrico especial]
   Before: cos(2 * pi / 3)
   Cambio local: cos(2 * pi / 3) -> -1 / 2
   After: -1 / 2
Result: -1 / 2
```

### Web / JSON Steps

1. `Evaluar valor trigonométrico especial`
   - before: `cos((2 · pi)/3)`
   - after: `-1/2`
   - substeps: none

## trig_special_value_sin_zero (simplify)

- Source: `sin(0)`
- Target: `0`
- Result: `0`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sin(0)
Target: 0
Strategy: rewrite trigs
Steps (Aggressive Mode):
1. Evaluate a trigonometric function at a special input  [Evaluar valor trigonométrico especial]
   Before: sin(0)
   Cambio local: sin(0) -> 0
   After: 0
Result: 0
```

### Web / JSON Steps

1. `Evaluar valor trigonométrico especial`
   - before: `sin(0)`
   - after: `0`
   - substeps: none
