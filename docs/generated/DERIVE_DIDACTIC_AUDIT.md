# Derive Didactic Audit

Generated from [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv).

Command: `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`

## Summary

- Derived cases audited: `226`
- Mean top-level step count: `1.07`
- Total web substeps: `304`

| id | family | web steps | web substeps | flags |
| --- | --- | ---: | ---: | --- |
| `combine_like_terms` | `simplify` | 1 | 1 | none |
| `collect_linear` | `collect` | 1 | 1 | none |
| `collect_common_symbolic_coefficients` | `collect` | 1 | 1 | none |
| `collect_linear_alt_variable` | `collect` | 1 | 1 | none |
| `collect_multiple_power_groups` | `collect` | 1 | 1 | none |
| `collect_composite_monomial_factor` | `collect` | 1 | 1 | none |
| `collect_two_composite_factor_groups` | `collect` | 1 | 1 | none |
| `factor_out_with_division_quadratic` | `conditional_factor` | 1 | 1 | none |
| `factor_out_with_division_sparse_quintic` | `conditional_factor` | 1 | 1 | none |
| `factor_out_with_division_mixed_septic` | `conditional_factor` | 1 | 1 | none |
| `factor_out_square_with_division_quartic` | `conditional_factor` | 1 | 1 | none |
| `factor_out_cube_with_division_septic` | `conditional_factor` | 1 | 1 | none |
| `factor_difference_squares` | `factor` | 1 | 2 | none |
| `factor_perfect_square_trinomial` | `factor` | 1 | 2 | none |
| `factor_perfect_square_trinomial_symbolic` | `factor` | 1 | 1 | none |
| `factor_perfect_square_trinomial_minus` | `factor` | 1 | 1 | none |
| `factor_sophie_germain` | `factor` | 1 | 2 | none |
| `factor_alternating_cubic_vandermonde` | `factor` | 1 | 2 | none |
| `factor_difference_cubes` | `factor` | 1 | 1 | none |
| `factor_sum_cubes` | `factor` | 1 | 1 | none |
| `factor_symbolic_sixth_power_difference` | `factor` | 1 | 1 | none |
| `factor_symbolic_sixth_power_sum` | `factor` | 1 | 1 | none |
| `pythagorean_identity` | `simplify` | 1 | 2 | none |
| `pythagorean_factor_form_to_cos_sq` | `simplify` | 1 | 0 | none |
| `pythagorean_factor_form_from_sin_sq` | `simplify` | 1 | 0 | none |
| `inverse_tan_identity` | `simplify` | 1 | 1 | none |
| `cancel_fraction_difference_squares` | `simplify` | 1 | 2 | none |
| `cancel_fraction_difference_cubes` | `simplify` | 1 | 2 | none |
| `cancel_fraction_sum_cubes` | `simplify` | 1 | 2 | none |
| `cancel_fraction_perfect_square_minus_symbolic` | `simplify` | 1 | 2 | none |
| `perfect_square_root_to_abs` | `simplify` | 1 | 2 | none |
| `solve_prep_complete_square_monic_numeric` | `solve_prep` | 1 | 2 | none |
| `solve_prep_complete_square_symbolic_leading_coeff` | `solve_prep` | 1 | 2 | none |
| `solve_prep_complete_square_symbolic_monic_parametric` | `solve_prep` | 1 | 2 | none |
| `solve_prep_complete_square_alt_variable_symbolic_leading_coeff` | `solve_prep` | 1 | 2 | none |
| `solve_prep_complete_square_fractional_monic_numeric` | `solve_prep` | 1 | 2 | none |
| `solve_prep_complete_square_symbolic_negative_linear_coeff` | `solve_prep` | 1 | 2 | none |
| `solve_prep_complete_square_negative_symbolic_leading_coeff` | `solve_prep` | 1 | 2 | none |
| `solve_prep_complete_square_fractional_symbolic_leading_coeff` | `solve_prep` | 1 | 2 | none |
| `combine_fraction_part_with_same_denominator` | `fraction_combine` | 1 | 0 | none |
| `combine_three_same_denominator_fractions` | `fraction_combine` | 1 | 0 | none |
| `combine_fraction_part_with_same_denominator_three_terms` | `fraction_combine` | 1 | 0 | none |
| `expand_fraction_part_with_same_denominator_three_terms` | `fraction_expand` | 1 | 1 | none |
| `factor_common_factor_sum` | `factor` | 1 | 2 | none |
| `expand_common_factor_sum` | `expand` | 1 | 2 | none |
| `expand_common_factor_difference` | `expand` | 1 | 2 | none |
| `factor_common_factor_sum_three_terms` | `factor` | 1 | 2 | none |
| `expand_common_factor_sum_three_terms` | `expand` | 1 | 2 | none |
| `expand_common_factor_difference_three_terms` | `expand` | 1 | 2 | none |
| `cancel_fraction_common_factor_numeric` | `simplify` | 1 | 2 | none |
| `cancel_fraction_monomial_common_factor` | `simplify` | 1 | 2 | none |
| `expand_binomial` | `expand` | 1 | 2 | none |
| `expand_symbolic_binomial` | `expand` | 1 | 1 | none |
| `expand_symbolic_binomial_minus` | `expand` | 1 | 1 | none |
| `expand_symbolic_binomial_cube` | `expand` | 1 | 1 | none |
| `expand_symbolic_binomial_cube_minus` | `expand` | 1 | 1 | none |
| `expand_symbolic_trinomial_square` | `expand` | 1 | 2 | none |
| `expand_symbolic_signed_trinomial_square` | `expand` | 1 | 2 | none |
| `expand_symbolic_trinomial_cube` | `expand` | 1 | 2 | none |
| `expand_sophie_germain` | `expand` | 1 | 2 | none |
| `expand_difference_cubes` | `expand` | 1 | 2 | none |
| `expand_sum_cubes` | `expand` | 1 | 2 | none |
| `expand_then_cancel_to_square` | `expand` | 1 | 1 | none |
| `expand_log_product` | `log_expand` | 1 | 0 | none |
| `expand_log_even_power_abs` | `log_expand` | 1 | 1 | none |
| `expand_log_quotient` | `log_expand` | 1 | 0 | none |
| `expand_log_product_over_quotient` | `log_expand` | 1 | 0 | none |
| `expand_log_powered_two_denominator_factors` | `log_expand` | 1 | 0 | none |
| `expand_log_general_base_product_over_quotient` | `log_expand` | 1 | 0 | none |
| `expand_log_general_base_powered_two_denominator_factors_with_powered_denominator` | `log_expand` | 1 | 0 | none |
| `expand_log_general_base_power` | `log_expand` | 1 | 1 | none |
| `contract_log_sum` | `log_contract` | 1 | 0 | none |
| `contract_log_difference` | `log_contract` | 1 | 0 | none |
| `contract_log_product_over_quotient` | `log_contract` | 1 | 0 | none |
| `contract_log_powered_two_denominator_factors` | `log_contract` | 1 | 0 | none |
| `contract_log_sum_with_scaled_powers` | `log_contract` | 1 | 1 | none |
| `contract_log_difference_with_scaled_powers` | `log_contract` | 1 | 1 | none |
| `contract_log_general_base_difference` | `log_contract` | 1 | 0 | none |
| `contract_log_general_base_difference_with_scaled_powers` | `log_contract` | 1 | 1 | none |
| `contract_log_general_base_powered_two_denominator_factors_with_powered_denominator` | `log_contract` | 1 | 0 | none |
| `contract_log_even_power_abs` | `log_contract` | 1 | 1 | none |
| `contract_log_general_base_power` | `log_contract` | 1 | 1 | none |
| `contract_log_change_of_base_chain` | `log_contract` | 1 | 0 | none |
| `expand_log_change_of_base_chain` | `log_expand` | 1 | 0 | none |
| `expand_trig_double_sin` | `trig_expand` | 1 | 0 | none |
| `expand_trig_product_to_sum_sin_cos` | `trig_expand` | 1 | 1 | none |
| `expand_trig_product_to_sum_cos_sin` | `trig_expand` | 1 | 1 | none |
| `expand_trig_product_to_sum_cos_cos` | `trig_expand` | 1 | 1 | none |
| `expand_trig_product_to_sum_sin_sin` | `trig_expand` | 1 | 1 | none |
| `contract_trig_double_sin` | `trig_contract` | 1 | 0 | none |
| `expand_trig_after_simplify` | `trig_expand` | 2 | 1 | none |
| `contract_trig_tan_quotient` | `trig_contract` | 1 | 0 | none |
| `contract_trig_sec_reciprocal` | `trig_contract` | 1 | 0 | none |
| `contract_trig_csc_reciprocal` | `trig_contract` | 1 | 0 | none |
| `contract_trig_cot_quotient` | `trig_contract` | 1 | 0 | none |
| `expand_trig_double_cos_as_one_minus_sin_sq` | `trig_expand` | 1 | 0 | none |
| `expand_trig_double_cos_as_two_cos_sq_minus_one` | `trig_expand` | 1 | 0 | none |
| `contract_trig_double_cos_from_one_minus_sin_sq` | `trig_contract` | 1 | 0 | none |
| `contract_trig_double_cos_from_two_cos_sq_minus_one` | `trig_contract` | 1 | 0 | none |
| `contract_trig_sec_squared` | `trig_contract` | 1 | 0 | none |
| `contract_trig_csc_squared` | `trig_contract` | 1 | 0 | none |
| `expand_trig_sec_squared` | `trig_expand` | 1 | 0 | none |
| `expand_trig_csc_squared` | `trig_expand` | 1 | 0 | none |
| `expand_trig_half_angle_sin_squared` | `trig_expand` | 1 | 0 | none |
| `expand_trig_half_angle_cos_squared` | `trig_expand` | 1 | 0 | none |
| `contract_trig_half_angle_sin_squared` | `trig_contract` | 1 | 0 | none |
| `contract_trig_half_angle_cos_squared` | `trig_contract` | 1 | 0 | none |
| `contract_trig_sin_diff_special` | `trig_contract` | 1 | 1 | none |
| `expand_trig_sum_to_product_sin_sum_general` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sum_to_product_sin_diff_general` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sum_to_product_cos_sum_general` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sum_to_product_cos_diff_general` | `trig_expand` | 1 | 1 | none |
| `contract_trig_half_angle_tangent` | `trig_contract` | 1 | 0 | none |
| `contract_trig_half_angle_tangent_alt` | `trig_contract` | 1 | 0 | none |
| `expand_trig_half_angle_tangent` | `trig_expand` | 1 | 0 | none |
| `expand_trig_half_angle_tangent_alt` | `trig_expand` | 1 | 0 | none |
| `rationalize_linear_root` | `rationalize` | 1 | 3 | none |
| `rationalize_linear_root_plus` | `rationalize` | 1 | 3 | none |
| `rationalize_then_cancel_to_zero` | `rationalize` | 2 | 3 | none |
| `rationalize_shifted_linear_root` | `rationalize` | 1 | 3 | none |
| `radical_notable_quotient` | `rationalize` | 2 | 5 | none |
| `expand_fraction_simple` | `fraction_expand` | 1 | 1 | none |
| `expand_fraction_with_term_cancellation` | `fraction_expand` | 1 | 2 | none |
| `expand_fraction_with_common_scalar_factor_in_denominator` | `fraction_expand` | 1 | 2 | none |
| `expand_fraction_exact_division_term_plus_remainder` | `fraction_expand` | 1 | 2 | none |
| `expand_fraction_mixed_variable_term_cancellation` | `fraction_expand` | 1 | 2 | none |
| `expand_fraction_three_factor_full_cancellation` | `fraction_expand` | 1 | 2 | none |
| `expand_fraction_two_cancellations_plus_remainder` | `fraction_expand` | 1 | 2 | none |
| `expand_fraction_three_factor_cross_cancellation_plus_remainder` | `fraction_expand` | 1 | 2 | none |
| `expand_fraction_three_factor_three_cancellations_to_constant` | `fraction_expand` | 1 | 2 | none |
| `nested_fraction_one_over_sum` | `nested_fraction` | 2 | 0 | none |
| `nested_fraction_one_over_three_reciprocals` | `nested_fraction` | 3 | 0 | none |
| `nested_fraction_sum_over_reciprocal` | `nested_fraction` | 2 | 0 | none |
| `nested_fraction_one_over_sum_with_fraction` | `nested_fraction` | 1 | 1 | none |
| `nested_fraction_fraction_over_sum_with_fraction_general` | `nested_fraction` | 1 | 1 | none |
| `nested_fraction_sum_with_fraction_over_scalar_general` | `nested_fraction` | 1 | 0 | none |
| `nested_fraction_one_over_sum_with_fraction_reverse` | `nested_fraction` | 1 | 1 | none |
| `nested_fraction_fraction_over_sum_with_fraction_general_reverse` | `nested_fraction` | 1 | 1 | none |
| `nested_fraction_sum_with_fraction_over_scalar_general_reverse` | `nested_fraction` | 1 | 1 | none |
| `combine_same_denominator_fraction_sum` | `fraction_combine` | 1 | 0 | none |
| `combine_general_fraction_sum` | `fraction_combine` | 1 | 0 | none |
| `combine_same_denominator_fraction_difference` | `fraction_combine` | 1 | 0 | none |
| `combine_general_fraction_difference` | `fraction_combine` | 1 | 0 | none |
| `combine_term_and_fraction_subtraction` | `fraction_combine` | 1 | 0 | none |
| `combine_symbolic_same_denominator_fraction_subset_with_passthrough` | `fraction_combine` | 1 | 0 | none |
| `split_fraction_into_whole_plus_remainder` | `fraction_decompose` | 1 | 2 | none |
| `split_fraction_linear_over_scaled_linear` | `fraction_decompose` | 1 | 2 | none |
| `split_fraction_symbolic_over_general_shift` | `fraction_decompose` | 1 | 2 | none |
| `split_fraction_symbolic_over_scaled_general_linear` | `fraction_decompose` | 1 | 2 | none |
| `split_fraction_symbolic_over_negative_scaled_general_linear` | `fraction_decompose` | 1 | 2 | none |
| `split_telescoping_fraction_consecutive` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_consecutive` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_negative_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_negative_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_affine_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_affine_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_affine_symbolic_shift_gap` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_affine_symbolic_shift_gap` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_difference_squares_unfactored` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_difference_squares_unfactored` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_shifted_quadratic_unfactored` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_symbolic_difference_squares_unfactored` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_symbolic_difference_squares_unfactored` | `telescoping_fraction` | 1 | 2 | none |
| `combine_whole_plus_remainder_into_fraction` | `fraction_combine` | 1 | 2 | none |
| `combine_symbolic_whole_plus_remainder_into_fraction` | `fraction_combine` | 1 | 2 | none |
| `combine_scaled_symbolic_whole_plus_remainder_into_fraction` | `fraction_combine` | 1 | 2 | none |
| `combine_negative_scaled_symbolic_whole_plus_remainder_into_fraction` | `fraction_combine` | 1 | 2 | none |
| `expand_cube_sum_product` | `polynomial_product` | 1 | 2 | none |
| `expand_cube_difference_product` | `polynomial_product` | 1 | 2 | none |
| `expand_difference_of_squares_quadratic_product` | `polynomial_product` | 1 | 3 | none |
| `expand_sixth_power_plus_product` | `polynomial_product` | 1 | 3 | none |
| `expand_sixth_power_minus_product` | `polynomial_product` | 1 | 3 | none |
| `expand_eighth_power_minus_multifactor_product` | `polynomial_product` | 1 | 1 | none |
| `expand_ninth_power_plus_product` | `polynomial_product` | 1 | 3 | none |
| `expand_symbolic_cube_sum_product` | `polynomial_product` | 1 | 2 | none |
| `expand_symbolic_cube_difference_product` | `polynomial_product` | 1 | 2 | none |
| `expand_symbolic_sixth_power_plus_product` | `polynomial_product` | 1 | 2 | none |
| `expand_symbolic_sixth_power_minus_product` | `polynomial_product` | 1 | 2 | none |
| `factor_geometric_difference_power_6` | `factor` | 1 | 2 | none |
| `merge_same_base_fractional_powers` | `power_merge` | 1 | 0 | none |
| `merge_same_base_fractional_powers_to_integer` | `power_merge` | 1 | 0 | none |
| `merge_mixed_root_and_fractional_power_five_sixths` | `power_merge` | 2 | 0 | none |
| `merge_same_base_integer_and_fractional_power` | `power_merge` | 1 | 0 | none |
| `merge_same_base_symbolic_powers` | `power_merge` | 1 | 0 | none |
| `merge_same_base_integer_and_symbolic_power` | `power_merge` | 1 | 0 | none |
| `merge_mixed_root_and_symbolic_power` | `power_merge` | 2 | 0 | none |
| `merge_four_same_base_symbolic_powers` | `power_merge` | 1 | 0 | none |
| `log_sum_difference_cancels_to_zero` | `simplify` | 1 | 0 | none |
| `expand_odd_half_power` | `radical_power` | 1 | 2 | none |
| `expand_odd_half_power_after_simplify` | `radical_power` | 1 | 2 | none |
| `expand_higher_odd_half_power` | `radical_power` | 1 | 2 | none |
| `expand_higher_odd_half_power_after_simplify` | `radical_power` | 1 | 2 | none |
| `expand_higher_odd_half_power_alt_var` | `radical_power` | 1 | 2 | none |
| `factor_out_with_division` | `conditional_factor` | 1 | 1 | none |
| `consecutive_factorial_ratio` | `simplify` | 1 | 2 | none |
| `contract_trig_cos_diff_sin_diff_quotient` | `trig_contract` | 3 | 4 | none |
| `reciprocal_trig_product_to_one` | `simplify` | 1 | 0 | none |
| `sec_tan_pythagorean_to_one` | `simplify` | 1 | 0 | none |
| `factor_symbolic_binomial_cube` | `factor` | 1 | 1 | none |
| `factor_symbolic_binomial_cube_minus` | `factor` | 1 | 1 | none |
| `integrate_prep_morrie_basic` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_morrie_symbolic_argument` | `integrate_prep` | 1 | 1 | none |
| `integrate_prep_morrie_symbolic_scale` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_morrie_reverse_basic` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_morrie_reverse_symbolic_scale_longer` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_dirichlet_basic` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_dirichlet_longer` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_dirichlet_symbolic_argument` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_dirichlet_symbolic_scale` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_dirichlet_symbolic_scale_longer` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_dirichlet_reverse_basic` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_dirichlet_reverse_symbolic_scale_longer` | `integrate_prep` | 1 | 2 | none |
| `finite_telescoping_product_basic` | `finite_telescoping` | 1 | 3 | none |
| `finite_telescoping_product_symbolic_shift_symbolic_lower` | `finite_telescoping` | 1 | 2 | none |
| `finite_telescoping_product_affine_symbolic_shift_symbolic_lower` | `finite_telescoping` | 1 | 2 | none |
| `finite_telescoping_product_factorized_square_shifted_base_numeric_symbolic_lower` | `finite_telescoping` | 2 | 4 | none |
| `finite_telescoping_product_factorized_square_shifted_base_symbolic_symbolic_lower` | `finite_telescoping` | 2 | 4 | none |
| `finite_telescoping_sum_basic` | `finite_telescoping` | 1 | 3 | none |
| `finite_telescoping_sum_symbolic_shift_symbolic_lower` | `finite_telescoping` | 1 | 3 | none |
| `finite_telescoping_sum_affine_symbolic_shift_symbolic_lower` | `finite_telescoping` | 1 | 3 | none |
| `finite_telescoping_sum_affine_symbolic_arbitrary_shift_symbolic_lower` | `finite_telescoping` | 1 | 3 | none |
| `rationalize_symbolic_linear_root` | `rationalize` | 2 | 5 | none |
| `rationalize_symbolic_linear_root_plus` | `rationalize` | 1 | 3 | none |
| `rationalize_symbolic_linear_root_alt_var` | `rationalize` | 2 | 5 | none |

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
Strategy: simplify
Steps (Aggressive Mode):
1. Combine like terms  [Combine Like Terms]
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

## collect_linear (collect)

- Source: `a*x + b*x + c`
- Target: `(a + b)*x + c`
- Result: `x * (a + b) + c`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a * x + b * x + c
Target: x * (a + b) + c
Strategy: collect
Steps (Aggressive Mode):
1. Collect terms by x  [Collect Terms]
   Before: a * x + b * x + c
   Cambio local: a * x + b * x + c -> x * (a + b) + c
   After: x * (a + b) + c
Result: x * (a + b) + c
```

### Web / JSON Steps

1. `Agrupar términos por variable`
   - before: `a · x + b · x + c`
   - after: `x · (a + b) + c`
   - substeps:
     1. `Agrupar los términos que llevan la misma potencia de x`

## collect_common_symbolic_coefficients (collect)

- Source: `x*y + x*z + w`
- Target: `x*(y + z) + w`
- Result: `x * (y + z) + w`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: x * y + x * z + w
Target: x * (y + z) + w
Strategy: collect
Steps (Aggressive Mode):
1. Collect terms by x  [Collect Terms]
   Before: x * y + x * z + w
   Cambio local: x * y + x * z + w -> x * (y + z) + w
   After: x * (y + z) + w
Result: x * (y + z) + w
```

### Web / JSON Steps

1. `Agrupar términos por variable`
   - before: `x · y + x · z + w`
   - after: `x · (y + z) + w`
   - substeps:
     1. `Agrupar los términos que llevan la misma potencia de x`

## collect_linear_alt_variable (collect)

- Source: `a*y + b*y + c`
- Target: `(a + b)*y + c`
- Result: `y * (a + b) + c`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a * y + b * y + c
Target: y * (a + b) + c
Strategy: collect
Steps (Aggressive Mode):
1. Collect terms by y  [Collect Terms]
   Before: a * y + b * y + c
   Cambio local: a * y + b * y + c -> y * (a + b) + c
   After: y * (a + b) + c
Result: y * (a + b) + c
```

### Web / JSON Steps

1. `Agrupar términos por variable`
   - before: `a · y + b · y + c`
   - after: `y · (a + b) + c`
   - substeps:
     1. `Agrupar los términos que llevan la misma potencia de y`

## collect_multiple_power_groups (collect)

- Source: `a*x^2 + b*x + c*x^2 + d*x + e*x^2 + f`
- Target: `(a + c + e)*x^2 + (b + d)*x + f`
- Result: `x * (b + d) + x^2 * (a + c + e) + f`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a * x^2 + c * x^2 + e * x^2 + b * x + d * x + f
Target: x * (b + d) + x^2 * (a + c + e) + f
Strategy: collect
Steps (Aggressive Mode):
1. Collect terms by x  [Collect Terms]
   Before: a * x^(2) + c * x^(2) + e * x^(2) + b * x + d * x + f
   Cambio local: a * x^(2) + c * x^(2) + e * x^(2) + b * x + d * x + f -> x * (b + d) + x^(2) * (a + c + e) + f
   After: x * (b + d) + x^2 * (a + c + e) + f
Result: x * (b + d) + x^(2) * (a + c + e) + f
```

### Web / JSON Steps

1. `Agrupar términos por variable`
   - before: `a · x^2 + c · x^2 + e · x^2 + b · x + d · x + f`
   - after: `x · (b + d) + x^2 · (a + c + e) + f`
   - substeps:
     1. `Agrupar los términos que llevan la misma potencia de x`

## collect_composite_monomial_factor (collect)

- Source: `a*x*y + b*x*y + c`
- Target: `(a + b)*x*y + c`
- Result: `x * y * (a + b) + c`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a * x * y + b * x * y + c
Target: x * y * (a + b) + c
Strategy: collect
Steps (Aggressive Mode):
1. Collect terms by x * y  [Collect Terms]
   Before: a * x * y + b * x * y + c
   Cambio local: a * x * y + b * x * y + c -> x * y * (a + b) + c
   After: x * y * (a + b) + c
Result: x * y * (a + b) + c
```

### Web / JSON Steps

1. `Agrupar términos por factor común`
   - before: `a · x · y + b · x · y + c`
   - after: `x · y · (a + b) + c`
   - substeps:
     1. `Agrupar los términos que llevan el mismo factor x·y`

## collect_two_composite_factor_groups (collect)

- Source: `a*x*y + b*x*y + c*x*z + d*x*z + e`
- Target: `(a + b)*x*y + (c + d)*x*z + e`
- Result: `x * y * (a + b) + x * z * (c + d) + e`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a * x * y + b * x * y + c * x * z + d * x * z + e
Target: x * y * (a + b) + x * z * (c + d) + e
Strategy: collect
Steps (Aggressive Mode):
1. Collect terms by x * y  [Collect Terms]
   Before: a * x * y + b * x * y + c * x * z + d * x * z + e
   Cambio local: a * x * y + b * x * y + c * x * z + d * x * z + e -> x * y * (a + b) + x * z * (c + d) + e
   After: x * y * (a + b) + x * z * (c + d) + e
Result: x * y * (a + b) + x * z * (c + d) + e
```

### Web / JSON Steps

1. `Agrupar términos por factor común`
   - before: `a · x · y + b · x · y + c · x · z + d · x · z + e`
   - after: `x · y · (a + b) + x · z · (c + d) + e`
   - substeps:
     1. `Agrupar los términos que llevan el mismo factor x·y`

## factor_out_with_division_quadratic (conditional_factor)

- Source: `a*x^2 + b*x + c`
- Target: `x*(a*x + b + c/x)`
- Result: `x * (c / x + a * x + b)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a * x^2 + b * x + c
Target: x * (c / x + a * x + b)
Strategy: factor out with division
Steps (Aggressive Mode):
1. Factor out x from the whole expression  [Factor Out With Division]
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
     1. `Si un término no lleva x, escribirlo como x · (t/x)`

## factor_out_with_division_sparse_quintic (conditional_factor)

- Source: `a*x^5 + b*x^3 + c*x + d`
- Target: `x*(a*x^4 + b*x^2 + c + d/x)`
- Result: `x * (d / x + a * x^4 + b * x^2 + c)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a * x^5 + b * x^3 + c * x + d
Target: x * (d / x + a * x^4 + b * x^2 + c)
Strategy: factor out with division
Steps (Aggressive Mode):
1. Factor out x from the whole expression  [Factor Out With Division]
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
     1. `Si un término no lleva x, escribirlo como x · (t/x)`

## factor_out_with_division_mixed_septic (conditional_factor)

- Source: `a*x^7 + b*x^5 + c*x^2 + d`
- Target: `x*(a*x^6 + b*x^4 + c*x + d/x)`
- Result: `x * (d / x + a * x^6 + b * x^4 + c * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a * x^7 + b * x^5 + c * x^2 + d
Target: x * (d / x + a * x^6 + b * x^4 + c * x)
Strategy: factor out with division
Steps (Aggressive Mode):
1. Factor out x from the whole expression  [Factor Out With Division]
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
     1. `Si un término no lleva x, escribirlo como x · (t/x)`

## factor_out_square_with_division_quartic (conditional_factor)

- Source: `a*x^4 + b*x^3 + c*x^2 + d`
- Target: `x^2*(a*x^2 + b*x + c + d/x^2)`
- Result: `x^2 * (d / x^2 + a * x^2 + b * x + c)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a * x^4 + b * x^3 + c * x^2 + d
Target: x^2 * (d / x^2 + a * x^2 + b * x + c)
Strategy: factor out with division
Steps (Aggressive Mode):
1. Factor out x^2 from the whole expression  [Factor Out With Division]
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
     1. `Si un término no lleva x^2, escribirlo como x^2 · (t/x^2)`

## factor_out_cube_with_division_septic (conditional_factor)

- Source: `a*x^7 + b*x^5 + c*x^3 + d`
- Target: `x^3*(a*x^4 + b*x^2 + c + d/x^3)`
- Result: `x^3 * (d / x^3 + a * x^4 + b * x^2 + c)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a * x^7 + b * x^5 + c * x^3 + d
Target: x^3 * (d / x^3 + a * x^4 + b * x^2 + c)
Strategy: factor out with division
Steps (Aggressive Mode):
1. Factor out x^3 from the whole expression  [Factor Out With Division]
   Before: a * x^(7) + b * x^(5) + c * x^(3) + d
   Cambio local: a * x^(7) + b * x^(5) + c * x^(3) + d -> x^(3) * (d / x^(3) + a * x^(4) + b * x^(2) + c)
   After: x^3 * (d / x^3 + a * x^4 + b * x^2 + c)
Result: x^(3) * (d / x^(3) + a * x^(4) + b * x^(2) + c)
ℹ️ Requires:
  • x^3 ≠ 0
  • x ≠ 0
```

### Web / JSON Steps

1. `Sacar factor usando división`
   - before: `a · x^7 + b · x^5 + c · x^3 + d`
   - after: `x^3 · (d/x^3 + a · x^4 + b · x^2 + c)`
   - substeps:
     1. `Si un término no lleva x^3, escribirlo como x^3 · (t/x^3)`

## factor_difference_squares (factor)

- Source: `x^2 - 1`
- Target: `(x - 1)*(x + 1)`
- Result: `(x + 1) * (x - 1)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: x^2 - 1
Target: (x + 1) * (x - 1)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorization]
   Before: x^(2) - 1
   Cambio local: x^(2) - 1 -> (x + 1) * (x - 1)
   After: (x + 1) * (x - 1)
Result: (x + 1) * (x - 1)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `x^2 - 1`
   - after: `(x + 1) · (x - 1)`
   - substeps:
     1. `Usar a^n - 1 = (a - 1) · (a^(n-1) + a^(n-2) + ... + a + 1)`
     2. `Aquí a = x y n = 2`

## factor_perfect_square_trinomial (factor)

- Source: `x^2 + 2*x + 1`
- Target: `(x + 1)^2`
- Result: `(x + 1)^2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: x^2 + 2 * x + 1
Target: (x + 1)^2
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorization]
   Before: x^(2) + 2 * x + 1
   Cambio local: x^(2) + 2 * x + 1 -> (x + 1)^(2)
   After: (x + 1)^2
Result: (x + 1)^(2)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `x^2 + 2 · x + 1`
   - after: `((x + 1))^2`
   - substeps:
     1. `Usar a^2 + 2ab + b^2 = (a + b)^2`
     2. `Aquí a = x y b = 1`

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
1. Factorization  [Factorization]
   Before: a^(2) + b^(2) + 2 * a * b
   Cambio local: a^(2) + b^(2) + 2 * a * b -> (a + b)^(2)
   After: (a + b)^2
Result: (a + b)^(2)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a^2 + b^2 + 2 · a · b`
   - after: `((a + b))^2`
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
1. Factorization  [Factorization]
   Before: a^(2) + b^(2) - 2 * a * b
   Cambio local: a^(2) + b^(2) - 2 * a * b -> (a - b)^(2)
   After: (a - b)^2
Result: (a - b)^(2)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a^2 - 2 · a · b + b^2`
   - after: `((a - b))^2`
   - substeps:
     1. `Usar a^2 - 2ab + b^2 = (a - b)^2`

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
1. Factorization  [Factorization]
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
     1. `Usar a^4 + 4b^4 = (a^2 - 2ab + 2b^2) · (a^2 + 2ab + 2b^2)`
     2. `Aquí a = x y b = y`

## factor_alternating_cubic_vandermonde (factor)

- Source: `a^3*(b-c) + b^3*(c-a) + c^3*(a-b)`
- Target: `(a-b)*(a-c)*(b-c)*(a+b+c)`
- Result: `(a + b + c) * (a - b) * (a - c) * (b - c)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a^3 * (b - c) + b^3 * (c - a) + c^3 * (a - b)
Target: (a + b + c) * (a - b) * (a - c) * (b - c)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorization]
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
     1. `Si dos variables coinciden, la expresión vale 0`
     2. `El factor restante es lineal y simétrico`

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
1. Factorization  [Factorization]
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
1. Factorization  [Factorization]
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
1. Factorization  [Factorization]
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
1. Factorization  [Factorization]
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

## pythagorean_identity (simplify)

- Source: `sin(x)^2 + cos(x)^2`
- Target: `1`
- Result: `1`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sin(x)^2 + cos(x)^2
Target: 1
Strategy: simplify
Steps (Aggressive Mode):
1. sin²(x) + cos²(x) = 1  [Pythagorean Chain Identity]
   Before: sin(x)^(2) + cos(x)^(2)
   Cambio local: sin(x)^(2) + cos(x)^(2) -> 1
   After: 1
Result: 1
```

### Web / JSON Steps

1. `Aplicar la identidad pitagórica`
   - before: `sin(x)^2 + cos(x)^2`
   - after: `1`
   - substeps:
     1. `Usar sin²(u) + cos²(u) = 1`
     2. `Aquí seno y coseno tienen el mismo ángulo`

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
Strategy: simplify
Steps (Aggressive Mode):
1. 1 - sin²(x) = cos²(x)  [Pythagorean Factor Form]
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
Strategy: simplify
Steps (Aggressive Mode):
1. 1 - cos²(x) = sin²(x)  [Pythagorean Factor Form]
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

## inverse_tan_identity (simplify)

- Source: `arctan(3)+arctan(1/3)`
- Target: `pi/2`
- Result: `pi / 2`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: arctan(3) + arctan(1 / 3)
Target: pi / 2
Strategy: simplify
Steps (Aggressive Mode):
1. arctan(x) + arctan(1/x) = π/2  [Inverse Tan Relations]
   Before: arctan(1/3) + arctan(3)
   Cambio local: arctan(1/3) + arctan(3) -> pi / 2
   After: pi / 2
Result: pi / 2
```

### Web / JSON Steps

1. `Aplicar identidad de arctangentes`
   - before: `arctan(1/3) + arctan(3)`
   - after: `pi/2`
   - substeps:
     1. `Usar arctan(u) + arctan(1/u) = pi/2`

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
Strategy: simplify
Steps (Aggressive Mode):
1. Cancel common factor  [Pre-order Difference of Squares Cancel]
   Before: (a + b) * (a - b) / (a - b)
   Cambio local: (a + b) * (a - b) / (a - b) -> a + b
   After: a + b
Result: a + b
```

### Web / JSON Steps

1. `Factorizar una diferencia de cuadrados y cancelar`
   - before: `((a + b) · (a - b))/(a - b)`
   - after: `a + b`
   - substeps:
     1. `Usar la diferencia de cuadrados: a^2 - b^2 = (a - b)(a + b)`
     2. `Ahora se cancela el factor a - b`

## cancel_fraction_difference_cubes (simplify)

- Source: `(a^3-b^3)/(a-b)`
- Target: `a^2+a*b+b^2`
- Result: `a^2 + b^2 + a * b`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a^3 - b^3) / (a - b)
Target: a^2 + b^2 + a * b
Strategy: simplify
Steps (Aggressive Mode):
1. Factor: a³ - b³ = (a-b)(a² + ab + b²)  [Cancel Sum/Difference of Cubes Fraction]
   Before: (a^(3) - b^(3)) / (a - b)
   Cambio local: (a^(3) - b^(3)) / (a - b) -> a^(2) + b^(2) + a * b
   After: a^2 + b^2 + a * b
Result: a^(2) + b^(2) + a * b
```

### Web / JSON Steps

1. `Factorizar cubos y cancelar`
   - before: `(a^3 - b^3)/(a - b)`
   - after: `a^2 + b^2 + a · b`
   - substeps:
     1. `Factorizar el numerador como suma o diferencia de cubos`
     2. `Ahora se cancela el factor (a - b)`

## cancel_fraction_sum_cubes (simplify)

- Source: `(a^3+b^3)/(a+b)`
- Target: `a^2-a*b+b^2`
- Result: `a^2 + b^2 - a * b`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a^3 + b^3) / (a + b)
Target: a^2 + b^2 - a * b
Strategy: simplify
Steps (Aggressive Mode):
1. Factor: a³ + b³ = (a+b)(a² - ab + b²)  [Cancel Sum/Difference of Cubes Fraction]
   Before: (a^(3) + b^(3)) / (a + b)
   Cambio local: (a^(3) + b^(3)) / (a + b) -> a^(2) + b^(2) - a * b
   After: a^2 + b^2 - a * b
Result: a^(2) + b^(2) - a * b
```

### Web / JSON Steps

1. `Factorizar cubos y cancelar`
   - before: `(a^3 + b^3)/(a + b)`
   - after: `a^2 - a · b + b^2`
   - substeps:
     1. `Factorizar el numerador como suma o diferencia de cubos`
     2. `Ahora se cancela el factor (a + b)`

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
Strategy: simplify
Steps (Aggressive Mode):
1. Cancel common factor  [Pre-order Perfect Square Minus Cancel]
   Before: (a^(2) + b^(2) - 2 * a * b) / (a - b)
   Cambio local: (a^(2) + b^(2) - 2 * a * b) / (a - b) -> a - b
   After: a - b
Result: a - b
```

### Web / JSON Steps

1. `Pre-order Perfect Square Minus Cancel`
   - before: `(a^2 - 2 · a · b + b^2)/(a - b)`
   - after: `a - b`
   - substeps:
     1. `Reconocer que el numerador es un cuadrado perfecto`
     2. `Si (a - b)^2 está dividido entre a - b, queda una sola copia`

## perfect_square_root_to_abs (simplify)

- Source: `sqrt(x^2 + 2*x + 1)`
- Target: `abs(x+1)`
- Result: `|x + 1|`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sqrt(x^2 + 2 * x + 1)
Target: |x + 1|
Strategy: simplify
Steps (Aggressive Mode):
1. sqrt(A^2 ± 2AB + B^2) = |A ± B|  [Sqrt Perfect Square]
   Before: sqrt(x^(2) + 2 * x + 1)
   Cambio local: sqrt(x^(2) + 2 * x + 1) -> |x + 1|
   After: |x + 1|
Result: |x + 1|
```

### Web / JSON Steps

1. `Reconocer un cuadrado perfecto bajo la raíz`
   - before: `sqrt(x^2 + 2 · x + 1)`
   - after: `|x + 1|`
   - substeps:
     1. `Reescribir el radicando como un cuadrado perfecto`
     2. `La raíz de un cuadrado da un valor absoluto`

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
1. Complete the square to rewrite the quadratic  [Complete the Square]
   Before: x^(2) + 6 * x + 5
   Cambio local: x^(2) + 6 * x + 5 -> (x + 3)^(2) - 4
   After: (x + 3)^2 - 4
Result: (x + 3)^(2) - 4
```

### Web / JSON Steps

1. `Completar el cuadrado`
   - before: `x^2 + 6 · x + 5`
   - after: `((x + 3))^2 - 4`
   - substeps:
     1. `Usar la fórmula de completar el cuadrado`
     2. `Aquí A = 1, B = 6 y C = 5`

## solve_prep_complete_square_symbolic_leading_coeff (solve_prep)

- Source: `a*x^2 + b*x + c`
- Target: `a*(x + b/(2*a))^2 + c - b^2/(4*a)`
- Result: `a * (b / (2 * a) + x)^2 + c - b^2 / (4 * a)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * x^2 + b * x + c
Target: a * (b / (2 * a) + x)^2 + c - b^2 / (4 * a)
Strategy: solve prep
Steps (Aggressive Mode):
1. Complete the square to rewrite the quadratic  [Complete the Square]
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
   - after: `a · ((b/(2 · a) + x))^2 + c - b^2/(4 · a)`
   - substeps:
     1. `Usar la fórmula de completar el cuadrado`
     2. `Aquí A = a, B = b y C = c`

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
1. Complete the square to rewrite the quadratic  [Complete the Square]
   Before: x^(2) + 2 * b * x + c
   Cambio local: x^(2) + 2 * b * x + c -> (b + x)^(2) + c - b^(2)
   After: (b + x)^2 + c - b^2
Result: (b + x)^(2) + c - b^(2)
```

### Web / JSON Steps

1. `Completar el cuadrado`
   - before: `x^2 + 2 · b · x + c`
   - after: `((b + x))^2 + c - b^2`
   - substeps:
     1. `Usar la fórmula de completar el cuadrado`
     2. `Aquí A = 1, B = 2 · b y C = c`

## solve_prep_complete_square_alt_variable_symbolic_leading_coeff (solve_prep)

- Source: `a*y^2 + b*y + c`
- Target: `a*(y + b/(2*a))^2 + c - b^2/(4*a)`
- Result: `a * (b / (2 * a) + y)^2 + c - b^2 / (4 * a)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * y^2 + b * y + c
Target: a * (b / (2 * a) + y)^2 + c - b^2 / (4 * a)
Strategy: solve prep
Steps (Aggressive Mode):
1. Complete the square to rewrite the quadratic  [Complete the Square]
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
   - after: `a · ((b/(2 · a) + y))^2 + c - b^2/(4 · a)`
   - substeps:
     1. `Usar la fórmula de completar el cuadrado`
     2. `Aquí A = a, B = b y C = c`

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
1. Complete the square to rewrite the quadratic  [Complete the Square]
   Before: x^(2) + 3 * x + 1
   Cambio local: x^(2) + 3 * x + 1 -> (3 / 2 + x)^(2) - 5 / 4
   After: (3 / 2 + x)^2 - 5 / 4
Result: (3 / 2 + x)^(2) - 5 / 4
ℹ️ Requires:
  • 4 ≠ 0
  • 2 ≠ 0
```

### Web / JSON Steps

1. `Completar el cuadrado`
   - before: `x^2 + 3 · x + 1`
   - after: `((3/2 + x))^2 - 5/4`
   - substeps:
     1. `Usar la fórmula de completar el cuadrado`
     2. `Aquí A = 1, B = 3 y C = 1`

## solve_prep_complete_square_symbolic_negative_linear_coeff (solve_prep)

- Source: `a*x^2 - b*x + c`
- Target: `a*(x - b/(2*a))^2 + c - b^2/(4*a)`
- Result: `a * (x - b / (2 * a))^2 + c - b^2 / (4 * a)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * x^2 - b * x + c
Target: a * (x - b / (2 * a))^2 + c - b^2 / (4 * a)
Strategy: solve prep
Steps (Aggressive Mode):
1. Complete the square to rewrite the quadratic  [Complete the Square]
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
   - after: `a · ((x - b/(2 · a)))^2 + c - b^2/(4 · a)`
   - substeps:
     1. `Usar la fórmula de completar el cuadrado`
     2. `Aquí A = a, B = -b y C = c`

## solve_prep_complete_square_negative_symbolic_leading_coeff (solve_prep)

- Source: `-a*x^2 + b*x + c`
- Target: `-a*(x - b/(2*a))^2 + c + b^2/(4*a)`
- Result: `b^2 / (4 * a) + c - a * (x - b / (2 * a))^2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: b * x + c - a * x^2
Target: b^2 / (4 * a) + c - a * (x - b / (2 * a))^2
Strategy: solve prep
Steps (Aggressive Mode):
1. Complete the square to rewrite the quadratic  [Complete the Square]
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
   - after: `b^2/(4 · a) + c - a · ((x - b/(2 · a)))^2`
   - substeps:
     1. `Usar la fórmula de completar el cuadrado`
     2. `Aquí A = -a, B = b y C = c`

## solve_prep_complete_square_fractional_symbolic_leading_coeff (solve_prep)

- Source: `(a/2)*x^2 + b*x + c`
- Target: `(a/2)*(x + b/a)^2 + c - b^2/(2*a)`
- Result: `(a * (b / a + x)^2)/2 + c - b^2 / (2 * a)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a * x^2)/2 + b * x + c
Target: (a * (b / a + x)^2)/2 + c - b^2 / (2 * a)
Strategy: solve prep
Steps (Aggressive Mode):
1. Complete the square to rewrite the quadratic  [Complete the Square]
   Before: x^(2) * a / 2 + b * x + c
   Cambio local: x^(2) * a / 2 + b * x + c -> (b / a + x)^(2) * a / 2 + c - b^(2) / (2 * a)
   After: (a * (b / a + x)^2)/2 + c - b^2 / (2 * a)
Result: (b / a + x)^(2) * a / 2 + c - b^(2) / (2 * a)
ℹ️ Requires:
  • a ≠ 0
  • 2 ≠ 0
```

### Web / JSON Steps

1. `Completar el cuadrado`
   - before: `x^2 · a/2 + b · x + c`
   - after: `((b/a + x))^2 · a/2 + c - b^2/(2 · a)`
   - substeps:
     1. `Usar la fórmula de completar el cuadrado`
     2. `Aquí A = a/2, B = b y C = c`

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
1. Combine fractions that already share the same denominator  [Combine Same Denominator Fractions]
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
1. Combine fractions that already share the same denominator  [Combine Same Denominator Fractions]
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
1. Combine fractions that already share the same denominator  [Combine Same Denominator Fractions]
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

## expand_fraction_part_with_same_denominator_three_terms (fraction_expand)

- Source: `1 + (a+b+c)/d`
- Target: `1 + a/d + b/d + c/d`
- Result: `a / d + b / d + c / d + 1`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a + b + c) / d + 1
Target: a / d + b / d + c / d + 1
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Distribute Division]
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
   - substeps:
     1. `Repartir el mismo denominador sobre cada término del numerador`

## factor_common_factor_sum (factor)

- Source: `a*b + a*c`
- Target: `a*(b+c)`
- Result: `a * (b + c)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * b + a * c
Target: a * (b + c)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorization]
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
     1. `Usar el factor común`
     2. `Aquí el factor común es a`

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
1. Expand the expression distributively  [Expand]
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
     1. `Usar la distributiva`
     2. `Aquí se distribuye a sobre cada término del paréntesis`

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
1. Expand the expression distributively  [Expand]
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
     1. `Usar la distributiva`
     2. `Aquí se distribuye a sobre cada término del paréntesis`

## factor_common_factor_sum_three_terms (factor)

- Source: `a*x + b*x + c*x`
- Target: `x*(a+b+c)`
- Result: `x * (a + b + c)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * x + b * x + c * x
Target: x * (a + b + c)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorization]
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
     1. `Usar el factor común`
     2. `Aquí el factor común es x`

## expand_common_factor_sum_three_terms (expand)

- Source: `x*(a+b+c)`
- Target: `a*x + b*x + c*x`
- Result: `a * x + b * x + c * x`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: x * (a + b + c)
Target: a * x + b * x + c * x
Strategy: expand
Steps (Aggressive Mode):
1. Expand the expression distributively  [Expand]
   Before: x * (a + b + c)
   Cambio local: x * (a + b + c) -> a * x + b * x + c * x
   After: a * x + b * x + c * x
Result: a * x + b * x + c * x
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `x · (a + b + c)`
   - after: `a · x + b · x + c · x`
   - substeps:
     1. `Usar la distributiva`
     2. `Aquí se distribuye x sobre cada término del paréntesis`

## expand_common_factor_difference_three_terms (expand)

- Source: `x*(a-b-c)`
- Target: `a*x - b*x - c*x`
- Result: `a * x - b * x - c * x`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: x * (a - b - c)
Target: a * x - b * x - c * x
Strategy: expand
Steps (Aggressive Mode):
1. Expand the expression distributively  [Expand]
   Before: x * (a - b - c)
   Cambio local: x * (a - b - c) -> a * x - b * x - c * x
   After: a * x - b * x - c * x
Result: a * x - b * x - c * x
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `x · (a - b - c)`
   - after: `a · x - b · x - c · x`
   - substeps:
     1. `Usar la distributiva`
     2. `Aquí se distribuye x sobre cada término del paréntesis`

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
Strategy: simplify
Steps (Aggressive Mode):
1. Cancel common factor  [Simplify Nested Fraction]
   Before: 2 * x / (2 * 2 * x)
   Cambio local: 2 * x / (2 * 2 * x) -> 1 / 2
   After: 1 / 2
Result: 1 / 2
```

### Web / JSON Steps

1. `Cancelar factores en una fracción`
   - before: `(2 · x)/(2 · 2 · x)`
   - after: `1/2`
   - substeps:
     1. `Cancelar el factor común 2`
     2. `Cancelar también el factor común x`

## cancel_fraction_monomial_common_factor (simplify)

- Source: `(6*x^2)/(3*x)`
- Target: `2*x`
- Result: `2 * x`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 6 * x^2 / (3 * x)
Target: 2 * x
Strategy: simplify
Steps (Aggressive Mode):
1. Cancel common factor  [Simplify Nested Fraction]
   Before: 2 * x * 3 * x / (3 * x)
   Cambio local: 2 * x * 3 * x / (3 * x) -> 2 * x
   After: 2 * x
Result: 2 * x
```

### Web / JSON Steps

1. `Cancelar factores en una fracción`
   - before: `(2 · x · 3 · x)/(3 · x)`
   - after: `2 · x`
   - substeps:
     1. `Cancelar el factor común x`
     2. `Cancelar también el factor común 3`

## expand_binomial (expand)

- Source: `(x + 1)^2`
- Target: `x^2 + 2*x + 1`
- Result: `x^2 + 2 * x + 1`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (x + 1)^2
Target: x^2 + 2 * x + 1
Strategy: expand
Steps (Aggressive Mode):
1. Expand the binomial power  [Binomial Expansion]
   Before: (x + 1)^(2)
   Cambio local: (x + 1)^(2) -> x^(2) + 2 * x + 1
   After: x^2 + 2 * x + 1
Result: x^(2) + 2 * x + 1
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `((x + 1))^2`
   - after: `x^2 + 2 · x + 1`
   - substeps:
     1. `Usar (a + b)^2 = a^2 + 2ab + b^2`
     2. `Aplicar la fórmula con a = 1, b = x`

## expand_symbolic_binomial (expand)

- Source: `(a + b)^2`
- Target: `a^2 + 2*a*b + b^2`
- Result: `a^2 + b^2 + 2 * a * b`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a + b)^2
Target: a^2 + b^2 + 2 * a * b
Strategy: expand
Steps (Aggressive Mode):
1. Expand the binomial power  [Binomial Expansion]
   Before: (a + b)^(2)
   Cambio local: (a + b)^(2) -> a^(2) + b^(2) + 2 * a * b
   After: a^2 + b^2 + 2 * a * b
Result: a^(2) + b^(2) + 2 * a * b
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `((a + b))^2`
   - after: `a^2 + b^2 + 2 · a · b`
   - substeps:
     1. `Usar (a + b)^2 = a^2 + 2ab + b^2`

## expand_symbolic_binomial_minus (expand)

- Source: `(a - b)^2`
- Target: `a^2 - 2*a*b + b^2`
- Result: `a^2 + b^2 - 2 * a * b`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a - b)^2
Target: a^2 + b^2 - 2 * a * b
Strategy: expand
Steps (Aggressive Mode):
1. Expand the binomial power  [Binomial Expansion]
   Before: (a - b)^(2)
   Cambio local: (a - b)^(2) -> a^(2) + b^(2) - 2 * a * b
   After: a^2 + b^2 - 2 * a * b
Result: a^(2) + b^(2) - 2 * a * b
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `((a - b))^2`
   - after: `a^2 - 2 · a · b + b^2`
   - substeps:
     1. `Usar (a - b)^2 = a^2 - 2ab + b^2`

## expand_symbolic_binomial_cube (expand)

- Source: `(a + b)^3`
- Target: `a^3 + 3*a^2*b + 3*a*b^2 + b^3`
- Result: `a^3 + b^3 + 3 * a * b^2 + 3 * b * a^2`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a + b)^3
Target: a^3 + b^3 + 3 * a * b^2 + 3 * b * a^2
Strategy: expand
Steps (Aggressive Mode):
1. Expand the binomial power  [Binomial Expansion]
   Before: (a + b)^(3)
   Cambio local: (a + b)^(3) -> a^(3) + b^(3) + 3 * a * b^(2) + 3 * b * a^(2)
   After: a^3 + b^3 + 3 * a * b^2 + 3 * b * a^2
Result: a^(3) + b^(3) + 3 * a * b^(2) + 3 * b * a^(2)
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `((a + b))^3`
   - after: `a^3 + b^3 + 3 · a · b^2 + 3 · b · a^2`
   - substeps:
     1. `Usar (a + b)^3 = a^3 + 3a^2b + 3ab^2 + b^3`

## expand_symbolic_binomial_cube_minus (expand)

- Source: `(a - b)^3`
- Target: `a^3 - 3*a^2*b + 3*a*b^2 - b^3`
- Result: `a^3 + 3 * a * b^2 - 3 * b * a^2 - b^3`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a - b)^3
Target: a^3 + 3 * a * b^2 - 3 * b * a^2 - b^3
Strategy: expand
Steps (Aggressive Mode):
1. Expand the binomial power  [Binomial Expansion]
   Before: (a - b)^(3)
   Cambio local: (a - b)^(3) -> a^(3) + 3 * a * b^(2) - 3 * b * a^(2) - b^(3)
   After: a^3 + 3 * a * b^2 - 3 * b * a^2 - b^3
Result: a^(3) + 3 * a * b^(2) - 3 * b * a^(2) - b^(3)
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `((a - b))^3`
   - after: `a^3 - 3 · b · a^2 + 3 · a · b^2 - b^3`
   - substeps:
     1. `Usar (a - b)^3 = a^3 - 3a^2b + 3ab^2 - b^3`

## expand_symbolic_trinomial_square (expand)

- Source: `(a + b + c)^2`
- Target: `a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c`
- Result: `a^2 + b^2 + c^2 + 2 * a * b + 2 * a * c + 2 * b * c`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a + b + c)^2
Target: a^2 + b^2 + c^2 + 2 * a * b + 2 * a * c + 2 * b * c
Strategy: expand
Steps (Aggressive Mode):
1. Expand the binomial power  [Binomial Expansion]
   Before: (a + b + c)^(2)
   Cambio local: (a + b + c)^(2) -> a^(2) + b^(2) + c^(2) + 2 * a * b + 2 * a * c + 2 * b * c
   After: a^2 + b^2 + c^2 + 2 * a * b + 2 * a * c + 2 * b * c
Result: a^(2) + b^(2) + c^(2) + 2 * a * b + 2 * a * c + 2 * b * c
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `((a + b + c))^2`
   - after: `a^2 + b^2 + c^2 + 2 · a · b + 2 · a · c + 2 · b · c`
   - substeps:
     1. `Usar (a + b)^2 = a^2 + 2ab + b^2`
     2. `Aplicar la fórmula con a = a + b, b = c`

## expand_symbolic_signed_trinomial_square (expand)

- Source: `(a - b + c)^2`
- Target: `a^2 + b^2 + c^2 - 2*a*b + 2*a*c - 2*b*c`
- Result: `a^2 + b^2 + c^2 - 2 * a * b + 2 * a * c - 2 * b * c`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a - b + c)^2
Target: a^2 + b^2 + c^2 - 2 * a * b + 2 * a * c - 2 * b * c
Strategy: expand
Steps (Aggressive Mode):
1. Expand the binomial power  [Binomial Expansion]
   Before: (a - b + c)^(2)
   Cambio local: (a - b + c)^(2) -> a^(2) + b^(2) + c^(2) - 2 * a * b + 2 * a * c - 2 * b * c
   After: a^2 + b^2 + c^2 - 2 * a * b + 2 * a * c - 2 * b * c
Result: a^(2) + b^(2) + c^(2) - 2 * a * b + 2 * a * c - 2 * b * c
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `((a - b + c))^2`
   - after: `a^2 + b^2 + c^2 - 2 · a · b + 2 · a · c - 2 · b · c`
   - substeps:
     1. `Usar (a + b)^2 = a^2 + 2ab + b^2`
     2. `Aplicar la fórmula con a = c, b = a - b`

## expand_symbolic_trinomial_cube (expand)

- Source: `(a + b + c)^3`
- Target: `a^3 + b^3 + c^3 + 3*a^2*b + 3*a^2*c + 3*a*b^2 + 6*a*b*c + 3*a*c^2 + 3*b^2*c + 3*b*c^2`
- Result: `a^3 + b^3 + c^3 + 3 * a * b^2 + 3 * a * c^2 + 3 * b * a^2 + 3 * b * c^2 + 3 * c * a^2 + 3 * c * b^2 + 6 * a * b * c`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a + b + c)^3
Target: a^3 + b^3 + c^3 + 3 * a * b^2 + 3 * a * c^2 + 3 * b * a^2 + 3 * b * c^2 + 3 * c * a^2 + 3 * c * b^2 + 6 * a * b * c
Strategy: expand
Steps (Aggressive Mode):
1. Expand the binomial power  [Binomial Expansion]
   Before: (a + b + c)^(3)
   Cambio local: (a + b + c)^(3) -> a^(3) + b^(3) + c^(3) + 3 * a * b^(2) + 3 * a * c^(2) + 3 * b * a^(2) + 3 * b * c^(2) + 3 * c * a^(2) + 3 * c * b^(2) + 6 * a * b * c
   After: a^3 + b^3 + c^3 + 3 * a * b^2 + 3 * a * c^2 + 3 * b * a^2 + 3 * b * c^2 + 3 * c * a^2 + 3 * c * b^2 + 6 * a * b * c
Result: a^(3) + b^(3) + c^(3) + 3 * a * b^(2) + 3 * a * c^(2) + 3 * b * a^(2) + 3 * b * c^(2) + 3 * c * a^(2) + 3 * c * b^(2) + 6 * a * b * c
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `((a + b + c))^3`
   - after: `a^3 + b^3 + c^3 + 3 · a · b^2 + 3 · a · c^2 + 3 · b · a^2 + 3 · b · c^2 + 3 · c · a^2 + 3 · c · b^2 + 6 · a · b · c`
   - substeps:
     1. `Usar (a + b)^3 = a^3 + 3a^2b + 3ab^2 + b^3`
     2. `Aplicar la fórmula con a = a + b, b = c`

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
1. Expand the expression distributively  [Expand]
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
     1. `Usar (a^2 - 2ab + 2b^2) · (a^2 + 2ab + 2b^2) = a^4 + 4b^4`
     2. `Aquí a = x y b = y`

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
1. Expand the expression distributively  [Expand]
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
1. Expand the expression distributively  [Expand]
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

## expand_then_cancel_to_square (expand)

- Source: `(a+b)^2 - a^2 - 2*a*b`
- Target: `b^2`
- Result: `b^2`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a + b)^2 - a^2 - 2 * a * b
Target: b^2
Strategy: simplify
Steps (Aggressive Mode):
1. Auto-expand (a+b)^2  [Auto Expand Power Sum]
   Before: (a + b)^(2) - a^(2) - 2 * a * b
   Cambio local: (a + b)^(2) -> b^(2) + 2 * a * b + a^(2)
   After: b^2
Result: b^(2)
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `((a + b))^2 - a^2 - 2 · a · b`
   - after: `b^2 + 2 · a · b + a^2 - a^2 - 2 · a · b`
   - substeps:
     1. `Usar (a + b)^2 = a^2 + 2ab + b^2`

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
1. Expand a logarithm into a sum of logarithms  [expand_log]
   Before: ln(x * y)
   Cambio local: ln(x * y) -> ln(x) + ln(y)
   After: ln(x) + ln(y)
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

## expand_log_even_power_abs (log_expand)

- Source: `ln(x^2)`
- Target: `2*ln(abs(x))`
- Result: `2 * ln(|x|)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: ln(x^2)
Target: 2 * ln(|x|)
Strategy: simplify
Steps (Aggressive Mode):
1. ln(x^(2k)) = 2·ln(|x^k|)  [Factor Perfect Square in Logarithm]
   Before: ln(x^(2))
   Cambio local: ln(x^(2)) -> 2 * ln(|x|)
   After: 2 * ln(|x|)
Result: 2 * ln(|x|)
ℹ️ Requires:
  • |x| > 0
```

### Web / JSON Steps

1. `Sacar un exponente fuera del logaritmo`
   - before: `ln(x^2)`
   - after: `2 · ln(|x|)`
   - substeps:
     1. `Sacar un exponente par fuera del logaritmo`

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
1. Expand a logarithm into a sum of logarithms  [expand_log]
   Before: ln(x / y)
   Cambio local: ln(x / y) -> ln(x) - ln(y)
   After: ln(x) - ln(y)
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
1. Expand a logarithm into a sum of logarithms  [expand_log]
   Before: ln(x * y / z)
   Cambio local: ln(x * y / z) -> ln(x) + ln(y) - ln(z)
   After: ln(x) + ln(y) - ln(z)
Result: ln(x) + ln(y) - ln(z)
ℹ️ Requires:
  • y > 0
  • x > 0
  • z > 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln((x · y)/z)`
   - after: `ln(x) + ln(y) - ln(z)`
   - substeps: none

## expand_log_powered_two_denominator_factors (log_expand)

- Source: `ln((x^2*y)/(z*t))`
- Target: `2*ln(abs(x)) + ln(y) - ln(z) - ln(t)`
- Result: `ln(y) + 2 * ln(|x|) - ln(z) - ln(t)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: ln(y * x^2 / (t * z))
Target: ln(y) + 2 * ln(|x|) - ln(z) - ln(t)
Strategy: expand_log
Steps (Aggressive Mode):
1. Expand a logarithm into a sum of logarithms  [expand_log]
   Before: ln(y * x^(2) / (t * z))
   Cambio local: ln(y * x^(2) / (t * z)) -> ln(y) + 2 * ln(|x|) - ln(z) - ln(t)
   After: ln(y) + 2 * ln(|x|) - ln(z) - ln(t)
Result: ln(y) + 2 * ln(|x|) - ln(z) - ln(t)
ℹ️ Requires:
  • z > 0
  • y > 0
  • |x| > 0
  • t > 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln((y · x^2)/(t · z))`
   - after: `ln(y) + 2 · ln(|x|) - ln(z) - ln(t)`
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
1. Expand a logarithm into a sum of logarithms  [expand_log]
   Before: log(b, x * y / z)
   Cambio local: log(b, x * y / z) -> log(b, x) + log(b, y) - log(b, z)
   After: log(b, x) + log(b, y) - log(b, z)
Result: log(b, x) + log(b, y) - log(b, z)
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `log_b((x · y)/z)`
   - after: `log_b(x) + log_b(y) - log_b(z)`
   - substeps: none

## expand_log_general_base_powered_two_denominator_factors_with_powered_denominator (log_expand)

- Source: `log(b, (x^2*y^3)/(z^2*t))`
- Target: `2*log(b, x) + 3*log(b, y) - 2*log(b, z) - log(b, t)`
- Result: `2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: log(b, x^2 * y^3 / (t * z^2))
Target: 2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t)
Strategy: expand_log
Steps (Aggressive Mode):
1. Expand a logarithm into a sum of logarithms  [expand_log]
   Before: log(b, x^(2) * y^(3) / (t * z^(2)))
   Cambio local: log(b, x^(2) * y^(3) / (t * z^(2))) -> 2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t)
   After: 2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t)
Result: 2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t)
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `log_b((x^2 · y^3)/(t · z^2))`
   - after: `2 · log_b(x) + 3 · log_b(y) - 2 · log_b(z) - log_b(t)`
   - substeps: none

## expand_log_general_base_power (log_expand)

- Source: `log(2, x^3)`
- Target: `3*log(2, x)`
- Result: `3 * log(2, x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: log(2, x^3)
Target: 3 * log(2, x)
Strategy: simplify
Steps (Aggressive Mode):
1. log(b, x^y) = y * log(b, x)  [Evaluate Logarithms]
   Before: log(2, x^(3))
   Cambio local: log(2, x^(3)) -> 3 * log(2, x)
   After: 3 * log(2, x)
   ⚠️ Assumes: x > 0
Result: 3 * log(2, x)
```

### Web / JSON Steps

1. `Sacar un exponente fuera del logaritmo`
   - before: `log_2(x^3)`
   - after: `3 · log_2(x)`
   - substeps:
     1. `Sacar el exponente fuera del logaritmo`

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
1. Combine logarithms into a single logarithm  [Log Contraction]
   Before: ln(x) + ln(y)
   Cambio local: ln(x) + ln(y) -> ln(x * y)
   After: ln(x * y)
Result: ln(x * y)
ℹ️ Requires:
  • x * y > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `ln(x) + ln(y)`
   - after: `ln(x · y)`
   - substeps: none

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
1. Combine logarithms into a single logarithm  [Log Contraction]
   Before: ln(x) - ln(y)
   Cambio local: ln(x) - ln(y) -> ln(x / y)
   After: ln(x / y)
Result: ln(x / y)
ℹ️ Requires:
  • x / y > 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `ln(x) - ln(y)`
   - after: `ln(x/y)`
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
1. Combine logarithms into a single logarithm  [Log Contraction]
   Before: ln(x) + ln(y) - ln(z)
   Cambio local: ln(x) + ln(y) - ln(z) -> ln(x * y / z)
   After: ln(x * y / z)
Result: ln(x * y / z)
ℹ️ Requires:
  • z ≠ 0
  • x * y / z > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `ln(x) + ln(y) - ln(z)`
   - after: `ln((x · y)/z)`
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
1. Combine logarithms into a single logarithm  [Log Contraction]
   Before: ln(y) + 2 * ln(|x|) - ln(z) - ln(t)
   Cambio local: ln(y) + 2 * ln(|x|) - ln(z) - ln(t) -> ln(y * x^(2) / (t * z))
   After: ln(y * x^2 / (t * z))
Result: ln(y * x^(2) / (t * z))
ℹ️ Requires:
  • t ≠ 0
  • z ≠ 0
  • y * x^2 / (t * z) > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `ln(y) + 2 · ln(|x|) - ln(z) - ln(t)`
   - after: `ln((y · x^2)/(t · z))`
   - substeps: none

## contract_log_sum_with_scaled_powers (log_contract)

- Source: `3*ln(x) + 2*ln(abs(y))`
- Target: `ln(x^3*y^2)`
- Result: `ln(x^3 * y^2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * ln(|y|) + 3 * ln(x)
Target: ln(x^3 * y^2)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Log Contraction]
   Before: 2 * ln(|y|) + 3 * ln(x)
   Cambio local: 2 * ln(|y|) + 3 * ln(x) -> ln(x^(3) * y^(2))
   After: ln(x^3 * y^2)
Result: ln(x^(3) * y^(2))
ℹ️ Requires:
  • x^3 * y^2 > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `2 · ln(|y|) + 3 · ln(x)`
   - after: `ln(x^3 · y^2)`
   - substeps:
     1. `Meter los coeficientes dentro de los logaritmos como exponentes`

## contract_log_difference_with_scaled_powers (log_contract)

- Source: `3*ln(x) - 2*ln(y)`
- Target: `ln(x^3/y^2)`
- Result: `ln(x^3 / y^2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 3 * ln(x) - 2 * ln(y)
Target: ln(x^3 / y^2)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Log Contraction]
   Before: 3 * ln(x) - 2 * ln(y)
   Cambio local: 3 * ln(x) - 2 * ln(y) -> ln(x^(3) / y^(2))
   After: ln(x^3 / y^2)
Result: ln(x^(3) / y^(2))
ℹ️ Requires:
  • y ≠ 0
  • x^3 / y^2 > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `3 · ln(x) - 2 · ln(y)`
   - after: `ln(x^3/y^2)`
   - substeps:
     1. `Meter los coeficientes dentro de los logaritmos y reunir la resta en un cociente`

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
1. Combine logarithms into a single logarithm  [Log Contraction]
   Before: log(2, x) - log(2, y)
   Cambio local: log(2, x) - log(2, y) -> log(2, x / y)
   After: log(2, x / y)
Result: log(2, x / y)
ℹ️ Requires:
  • y ≠ 0
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
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 3 * log(2, x) - 2 * log(2, y)
Target: log(2, x^3 / y^2)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Log Contraction]
   Before: 3 * log(2, x) - 2 * log(2, y)
   Cambio local: 3 * log(2, x) - 2 * log(2, y) -> log(2, x^(3) / y^(2))
   After: log(2, x^3 / y^2)
Result: log(2, x^(3) / y^(2))
ℹ️ Requires:
  • y ≠ 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `3 · log_2(x) - 2 · log_2(y)`
   - after: `log_2(x^3/y^2)`
   - substeps:
     1. `Meter los coeficientes dentro de los logaritmos y reunir la resta en un cociente`

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
1. Combine logarithms into a single logarithm  [Log Contraction]
   Before: 2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t)
   Cambio local: 2 * log(b, x) + 3 * log(b, y) - 2 * log(b, z) - log(b, t) -> log(b, x^(2) * y^(3) / (t * z^(2)))
   After: log(b, x^2 * y^3 / (t * z^2))
Result: log(b, x^(2) * y^(3) / (t * z^(2)))
ℹ️ Requires:
  • t ≠ 0
  • z ≠ 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `2 · log_b(x) + 3 · log_b(y) - 2 · log_b(z) - log_b(t)`
   - after: `log_b((x^2 · y^3)/(t · z^2))`
   - substeps: none

## contract_log_even_power_abs (log_contract)

- Source: `2*ln(abs(x))`
- Target: `ln(x^2)`
- Result: `ln(x^2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * ln(|x|)
Target: ln(x^2)
Strategy: simplify
Steps (Aggressive Mode):
1. Simplify the expression  [Simplify]
   Before: 2 * ln(|x|)
   Cambio local: 2 * ln(|x|) -> ln(x^(2))
   After: ln(x^2)
Result: ln(x^(2))
ℹ️ Requires:
  • x ≠ 0
```

### Web / JSON Steps

1. `Meter el coeficiente dentro del logaritmo`
   - before: `2 · ln(|x|)`
   - after: `ln(x^2)`
   - substeps:
     1. `Usar n · ln(|u|) = ln(u^n) cuando n es par`

## contract_log_general_base_power (log_contract)

- Source: `3*log(2, x)`
- Target: `log(2, x^3)`
- Result: `log(2, x^3)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 3 * log(2, x)
Target: log(2, x^3)
Strategy: simplify
Steps (Aggressive Mode):
1. Simplify the expression  [Simplify]
   Before: 3 * log(2, x)
   Cambio local: 3 * log(2, x) -> log(2, x^(3))
   After: log(2, x^3)
Result: log(2, x^(3))
```

### Web / JSON Steps

1. `Meter el coeficiente dentro del logaritmo`
   - before: `3 · log_2(x)`
   - after: `log_2(x^3)`
   - substeps:
     1. `Usar n · log_b(u) = log_b(u^n)`

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
1. Combine logarithms into a single logarithm  [Log Contraction]
   Before: log(a, c) * log(b, a)
   Cambio local: log(a, c) * log(b, a) -> log(b, c)
   After: log(b, c)
Result: log(b, c)
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `log_a(c) · log_b(a)`
   - after: `log_b(c)`
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
Strategy: simplify
Steps (Aggressive Mode):
1. Simplify the expression  [Simplify]
   Before: log(b, c)
   Cambio local: log(b, c) -> log(a, c) * log(b, a)
   After: log(a, c) * log(b, a)
Result: log(a, c) * log(b, a)
```

### Web / JSON Steps

1. `Expandir cambio de base`
   - before: `log_b(c)`
   - after: `log_a(c) · log_b(a)`
   - substeps: none

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
1. Expand double-angle sine  [Double Angle Expansion]
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
1. Expand 2·sin(A)·cos(B) into sin(A+B) + sin(A-B)  [Product-to-Sum Identity]
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
     1. `Usar 2 · sin(A) · cos(B) = sin(A+B) + sin(A-B)`

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
1. Expand 2·cos(A)·sin(B) into sin(A+B) - sin(A-B)  [Product-to-Sum Identity]
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
     1. `Usar 2 · cos(A) · sin(B) = sin(A+B) - sin(A-B)`

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
1. Expand 2·cos(A)·cos(B) into cos(A+B) + cos(A-B)  [Product-to-Sum Identity]
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
     1. `Usar 2 · cos(A) · cos(B) = cos(A+B) + cos(A-B)`

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
1. Expand 2·sin(A)·sin(B) into cos(A-B) - cos(A+B)  [Product-to-Sum Identity]
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
     1. `Usar 2 · sin(A) · sin(B) = cos(A-B) - cos(A+B)`

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
1. Expand double-angle sine  [Double Angle Expansion]
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

## expand_trig_after_simplify (trig_expand)

- Source: `sin(x + x)`
- Target: `2*sin(x)*cos(x)`
- Result: `2 * sin(x) * cos(x)`
- Web step count: `2`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x + x)
Target: 2 * sin(x) * cos(x)
Strategy: simplify -> expand trig
Steps (Aggressive Mode):
1. Combine like terms  [Combine Like Terms]
   Before: sin(x + x)
   Cambio local: x + x -> 2 * x
   After: sin(2 * x)
2. Expand double-angle sine  [Double Angle Expansion]
   Before: sin(2 * x)
   Cambio local: sin(2 * x) -> 2 * sin(x) * cos(x)
   After: 2 * sin(x) * cos(x)
Result: 2 * sin(x) * cos(x)
```

### Web / JSON Steps

1. `Agrupar términos semejantes`
   - before: `sin(x + x)`
   - after: `sin(2 · x)`
   - substeps:
     1. `Sumar los coeficientes que acompañan a x`
2. `Expandir ángulo doble`
   - before: `sin(2 · x)`
   - after: `2 · sin(x) · cos(x)`
   - substeps: none

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
1. Recognize sin(u) / cos(u) as tan(u)  [Trig Quotient]
   Before: sin(2 * x) / cos(2 * x)
   Cambio local: sin(2 * x) / cos(2 * x) -> tan(2 * x)
   After: tan(2 * x)
Result: tan(2 * x)
```

### Web / JSON Steps

1. `Convertir un cociente trigonométrico en tangente`
   - before: `sin(2 · x)/cos(2 · x)`
   - after: `tan(2 · x)`
   - substeps: none

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
1. Recognize 1 / cos(u) as sec(u)  [Reciprocal Trig Identity]
   Before: 1 / cos(x)
   Cambio local: 1 / cos(x) -> sec(x)
   After: sec(x)
Result: sec(x)
```

### Web / JSON Steps

1. `Aplicar identidad trigonométrica recíproca`
   - before: `1/cos(x)`
   - after: `sec(x)`
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
1. Recognize 1 / sin(u) as csc(u)  [Reciprocal Trig Identity]
   Before: 1 / sin(x)
   Cambio local: 1 / sin(x) -> csc(x)
   After: csc(x)
Result: csc(x)
```

### Web / JSON Steps

1. `Aplicar identidad trigonométrica recíproca`
   - before: `1/sin(x)`
   - after: `csc(x)`
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
1. Recognize cos(u) / sin(u) as cot(u)  [Reciprocal Trig Identity]
   Before: cos(x) / sin(x)
   Cambio local: cos(x) / sin(x) -> cot(x)
   After: cot(x)
Result: cot(x)
```

### Web / JSON Steps

1. `Aplicar identidad trigonométrica recíproca`
   - before: `cos(x)/sin(x)`
   - after: `cot(x)`
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
1. Expand cosine double-angle as 1 - 2·sin(u)^2  [Double Angle Expansion]
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
1. Expand cosine double-angle as 2·cos(u)^2 - 1  [Double Angle Expansion]
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
1. Expand cosine double-angle as 1 - 2·sin(u)^2  [Double Angle Expansion]
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
1. Expand cosine double-angle as 2·cos(u)^2 - 1  [Double Angle Expansion]
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
1. Recognize 1 + tan²(u) as sec²(u)  [Recognize Secant Squared]
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
1. Recognize 1 + cot²(u) as csc²(u)  [Recognize Cosecant Squared]
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
1. Expand sec²(u) as 1 + tan(u)^2  [Expand Secant Squared]
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
1. Expand csc²(u) as 1 + cot(u)^2  [Expand Cosecant Squared]
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
1. Expand sin²(u) as (1 - cos(2u))/2  [Half-Angle Square Identity]
   Before: sin(x)^(2)
   Cambio local: sin(x)^(2) -> (1 - cos(2 * x)) / 2
   After: (1 - cos(2 * x)) / 2
Result: (1 - cos(2 * x)) / 2
ℹ️ Requires:
  • 2 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de ángulo mitad`
   - before: `sin(x)^2`
   - after: `(1 - cos(2 · x))/2`
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
1. Expand cos²(u) as (1 + cos(2u))/2  [Half-Angle Square Identity]
   Before: cos(x)^(2)
   Cambio local: cos(x)^(2) -> (cos(2 * x) + 1) / 2
   After: (cos(2 * x) + 1) / 2
Result: (cos(2 * x) + 1) / 2
ℹ️ Requires:
  • 2 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad de ángulo mitad`
   - before: `cos(x)^2`
   - after: `(cos(2 · x) + 1)/2`
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
1. Recognize (1 - cos(2u))/2 as sin²(u)  [Half-Angle Square Identity]
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
1. Recognize (1 + cos(2u))/2 as cos²(u)  [Half-Angle Square Identity]
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

## contract_trig_sin_diff_special (trig_contract)

- Source: `sin(3*x)-sin(x)`
- Target: `2*cos(2*x)*sin(x)`
- Result: `2 * sin(x) * cos(2 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(3 * x) - sin(x)
Target: 2 * sin(x) * cos(2 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand sine difference to product  [Sum-to-Product Identity]
   Before: sin(3 * x) - sin(x)
   Cambio local: sin(3 * x) - sin(x) -> 2 * sin(x) * cos(2 * x)
   After: 2 * sin(x) * cos(2 * x)
Result: 2 * sin(x) * cos(2 * x)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `sin(3 · x) - sin(x)`
   - after: `2 · sin(x) · cos(2 · x)`
   - substeps:
     1. `Usar sin(A) - sin(B) = 2 · cos((A+B)/2) · sin((A-B)/2)`

## expand_trig_sum_to_product_sin_sum_general (trig_expand)

- Source: `sin(5*x)+sin(x)`
- Target: `2*sin(3*x)*cos(2*x)`
- Result: `2 * sin(3 * x) * cos(2 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x) + sin(5 * x)
Target: 2 * sin(3 * x) * cos(2 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand sine sum to product  [Sum-to-Product Identity]
   Before: sin(x) + sin(5 * x)
   Cambio local: sin(x) + sin(5 * x) -> 2 * sin(3 * x) * cos(2 * x)
   After: 2 * sin(3 * x) * cos(2 * x)
Result: 2 * sin(3 * x) * cos(2 * x)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `sin(x) + sin(5 · x)`
   - after: `2 · sin(3 · x) · cos(2 · x)`
   - substeps:
     1. `Usar sin(A) + sin(B) = 2 · sin((A+B)/2) · cos((A-B)/2)`

## expand_trig_sum_to_product_sin_diff_general (trig_expand)

- Source: `sin(5*x)-sin(x)`
- Target: `2*cos(3*x)*sin(2*x)`
- Result: `2 * sin(2 * x) * cos(3 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(5 * x) - sin(x)
Target: 2 * sin(2 * x) * cos(3 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand sine difference to product  [Sum-to-Product Identity]
   Before: sin(5 * x) - sin(x)
   Cambio local: sin(5 * x) - sin(x) -> 2 * sin(2 * x) * cos(3 * x)
   After: 2 * sin(2 * x) * cos(3 * x)
Result: 2 * sin(2 * x) * cos(3 * x)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `sin(5 · x) - sin(x)`
   - after: `2 · sin(2 · x) · cos(3 · x)`
   - substeps:
     1. `Usar sin(A) - sin(B) = 2 · cos((A+B)/2) · sin((A-B)/2)`

## expand_trig_sum_to_product_cos_sum_general (trig_expand)

- Source: `cos(5*x)+cos(x)`
- Target: `2*cos(3*x)*cos(2*x)`
- Result: `2 * cos(2 * x) * cos(3 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x) + cos(5 * x)
Target: 2 * cos(2 * x) * cos(3 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand cosine sum to product  [Sum-to-Product Identity]
   Before: cos(x) + cos(5 * x)
   Cambio local: cos(x) + cos(5 * x) -> 2 * cos(2 * x) * cos(3 * x)
   After: 2 * cos(2 * x) * cos(3 * x)
Result: 2 * cos(2 * x) * cos(3 * x)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `cos(x) + cos(5 · x)`
   - after: `2 · cos(2 · x) · cos(3 · x)`
   - substeps:
     1. `Usar cos(A) + cos(B) = 2 · cos((A+B)/2) · cos((A-B)/2)`

## expand_trig_sum_to_product_cos_diff_general (trig_expand)

- Source: `cos(5*x)-cos(x)`
- Target: `-2*sin(3*x)*sin(2*x)`
- Result: `-2 * sin(2 * x) * sin(3 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(5 * x) - cos(x)
Target: -2 * sin(2 * x) * sin(3 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand cosine difference to product  [Sum-to-Product Identity]
   Before: cos(5 * x) - cos(x)
   Cambio local: cos(5 * x) - cos(x) -> -2 * sin(2 * x) * sin(3 * x)
   After: -2 * sin(2 * x) * sin(3 * x)
Result: -2 * sin(2 * x) * sin(3 * x)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `cos(5 · x) - cos(x)`
   - after: `-2 · sin(2 · x) · sin(3 · x)`
   - substeps:
     1. `Usar cos(A) - cos(B) = -2 · sin((A+B)/2) · sin((A-B)/2)`

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
1. Contract half-angle tangent quotient  [Half-Angle Tangent Identity]
   Before: (1 - cos(2 * x)) / sin(2 * x)
   Cambio local: (1 - cos(2 * x)) / sin(2 * x) -> tan(x)
   After: tan(x)
Result: tan(x)
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
1. Contract half-angle tangent quotient  [Half-Angle Tangent Identity]
   Before: sin(2 * x) / (cos(2 * x) + 1)
   Cambio local: sin(2 * x) / (cos(2 * x) + 1) -> tan(x)
   After: tan(x)
Result: tan(x)
```

### Web / JSON Steps

1. `Aplicar identidad de tangente de ángulo mitad`
   - before: `sin(2 · x)/(cos(2 · x) + 1)`
   - after: `tan(x)`
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
1. Expand tan(u) as (1 - cos(2u))/sin(2u)  [Half-Angle Tangent Identity]
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
1. Expand tan(u) as sin(2u)/(1 + cos(2u))  [Half-Angle Tangent Identity]
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
1. Rationalize: multiply by conjugate  [Rationalize Linear Sqrt Denominator]
   Before: 1 / (sqrt(x) - 1)
   After: (sqrt(x) + 1) / (x - 1)
Result: (sqrt(x) + 1) / (x - 1)
ℹ️ Requires:
  • x - 1 ≠ 0
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
Strategy: simplify
Steps (Aggressive Mode):
1. Rationalize: multiply by conjugate  [Rationalize Linear Sqrt Denominator]
   Before: 1 / (sqrt(x) + 1)
   After: (sqrt(x) - 1) / (x - 1)
Result: (sqrt(x) - 1) / (x - 1)
ℹ️ Requires:
  • x ≥ 0
  • x - 1 ≠ 0
```

### Web / JSON Steps

1. `Racionalizar el denominador`
   - before: `1/(sqrt(x) + 1)`
   - after: `(sqrt(x) - 1)/(x - 1)`
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
Strategy: simplify
Steps (Aggressive Mode):
1. Rationalize: multiply by conjugate  [Rationalize Linear Sqrt Denominator]
   Before: 1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)
   After: (sqrt(x) + 1) / (x + 1^(2)) - (sqrt(x) + 1) / (x - 1)
2. a - a = 0  [Subtraction Self-Cancel]
   Before: (sqrt(x) + 1) / (x - 1) - (sqrt(x) + 1) / (x - 1)
   Cambio local: (sqrt(x) + 1) / (x - 1) - (sqrt(x) + 1) / (x - 1) -> 0
   After: 0
Result: 0
```

### Web / JSON Steps

1. `Racionalizar el denominador`
   - before: `1/(sqrt(x) - 1) - (sqrt(x) + 1)/(x - 1)`
   - after: `(sqrt(x) + 1)/(x - 1^2) - (sqrt(x) + 1)/(x - 1)`
   - substeps:
     1. `Cambiar el signo para formar el conjugado`
     2. `Multiplicar numerador y denominador por ese conjugado`
     3. `En el denominador aparece una diferencia de cuadrados`
2. `Restar dos expresiones iguales`
   - before: `(sqrt(x) + 1)/(x - 1) - (sqrt(x) + 1)/(x - 1)`
   - after: `0`
   - substeps: none

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
1. Rationalize: multiply by conjugate  [Rationalize Linear Sqrt Denominator]
   Before: 1 / (sqrt(x) - 2)
   After: (sqrt(x) + 2) / (x - 4)
Result: (sqrt(x) + 2) / (x - 4)
ℹ️ Requires:
  • x - 4 ≠ 0
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

## radical_notable_quotient (rationalize)

- Source: `(x^(3/2)-1)/(sqrt(x)-1)`
- Target: `sqrt(x)+x+1`
- Result: `sqrt(x) + x + 1`
- Web step count: `2`
- Web substep count: `5`
- Flags: none

### CLI

```text
Parsed: (x^(3 / 2) - 1) / (sqrt(x) - 1)
Target: sqrt(x) + x + 1
Strategy: rationalize
Steps (Aggressive Mode):
1. Polynomial division with opaque substitution  [Rationalize Linear Sqrt Denominator]
   Before: (sqrt(x^3) - 1) / (sqrt(x) - 1)
   After: sqrt(x) + sqrt(x)^(2) + 1
2. (u^y)^(1/y) = u  [Cancel Reciprocal Exponents]
   Before: sqrt(x) + sqrt(x)^(2) + 1
   Cambio local: sqrt(x)^(2) -> x
   After: sqrt(x) + x + 1
   ℹ️ Requires: x > 0
Result: sqrt(x) + x + 1
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Reconocer un cociente notable`
   - before: `(sqrt(x^3) - 1)/(sqrt(x) - 1)`
   - after: `sqrt(x) + sqrt(x)^2 + 1`
   - substeps:
     1. `Llamar t = sqrt(x) para reconocer la forma`
     2. `Ese cociente notable se convierte en t^2 + t + 1`
     3. `Volver a poner t = sqrt(x)`
2. `Deshacer raíz y potencia`
   - before: `sqrt(x) + sqrt(x)^2 + 1`
   - after: `sqrt(x) + x + 1`
   - substeps:
     1. `El cuadrado deshace la raíz`
     2. `Reemplazar ese bloque en la expresión`

## expand_fraction_simple (fraction_expand)

- Source: `(a+b)/x`
- Target: `a/x + b/x`
- Result: `a / x + b / x`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a + b) / x
Target: a / x + b / x
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Distribute Division]
   Before: (a + b) / x
   After: a / x + b / x
Result: a / x + b / x
ℹ️ Requires:
  • x ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(a + b)/x`
   - after: `a/x + b/x`
   - substeps:
     1. `Usar (a + b) / d = a/d + b/d`

## expand_fraction_with_term_cancellation (fraction_expand)

- Source: `(x+y)/(x*y)`
- Target: `1/x + 1/y`
- Result: `1 / x + 1 / y`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (x + y) / (x * y)
Target: 1 / x + 1 / y
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Distribute Division]
   Before: (x + y) / (x * y)
   After: 1 / x + 1 / y
Result: 1 / x + 1 / y
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(x + y)/(x · y)`
   - after: `1/x + 1/y`
   - substeps:
     1. `Usar (a + b) / d = a/d + b/d`
     2. `Cancelar los factores comunes en las fracciones resultantes`

## expand_fraction_with_common_scalar_factor_in_denominator (fraction_expand)

- Source: `(a*x+b)/(c*x)`
- Target: `a/c + b/(c*x)`
- Result: `a / c + b / (c * x)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a * x + b) / (c * x)
Target: a / c + b / (c * x)
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Distribute Division]
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
     1. `Usar (a + b) / d = a/d + b/d`
     2. `Cancelar los factores comunes en la fracción que queda`

## expand_fraction_exact_division_term_plus_remainder (fraction_expand)

- Source: `(a*d+b)/d`
- Target: `a + b/d`
- Result: `b / d + a`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a * d + b) / d
Target: b / d + a
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Distribute Division]
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
     1. `Usar (a + b) / d = a/d + b/d`
     2. `Cancelar los factores comunes en la fracción que queda`

## expand_fraction_mixed_variable_term_cancellation (fraction_expand)

- Source: `(a*x+b*y)/(x*y)`
- Target: `a/y + b/x`
- Result: `a / y + b / x`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a * x + b * y) / (x * y)
Target: a / y + b / x
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Distribute Division]
   Before: (a * x + b * y) / (x * y)
   After: a / y + b / x
Result: a / y + b / x
ℹ️ Requires:
  • y ≠ 0
  • x ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(a · x + b · y)/(x · y)`
   - after: `a/y + b/x`
   - substeps:
     1. `Usar (a + b) / d = a/d + b/d`
     2. `Cancelar los factores comunes en las fracciones resultantes`

## expand_fraction_three_factor_full_cancellation (fraction_expand)

- Source: `(a*x+b*y+c*z)/(x*y*z)`
- Target: `a/(y*z) + b/(x*z) + c/(x*y)`
- Result: `a / (y * z) + b / (x * z) + c / (x * y)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a * x + b * y + c * z) / (x * y * z)
Target: a / (y * z) + b / (x * z) + c / (x * y)
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Distribute Division]
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
     1. `Repartir el mismo denominador sobre cada término del numerador`
     2. `Cancelar los factores comunes en las fracciones resultantes`

## expand_fraction_two_cancellations_plus_remainder (fraction_expand)

- Source: `(a*x+b*y+c)/(x*y)`
- Target: `a/y + b/x + c/(x*y)`
- Result: `a / y + b / x + c / (x * y)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a * x + b * y + c) / (x * y)
Target: a / y + b / x + c / (x * y)
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Distribute Division]
   Before: (a * x + b * y + c) / (x * y)
   After: a / y + b / x + c / (x * y)
Result: a / y + b / x + c / (x * y)
ℹ️ Requires:
  • y ≠ 0
  • x ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(a · x + b · y + c)/(x · y)`
   - after: `a/y + b/x + c/(x · y)`
   - substeps:
     1. `Repartir el mismo denominador sobre cada término del numerador`
     2. `Cancelar los factores comunes en las fracciones resultantes`

## expand_fraction_three_factor_cross_cancellation_plus_remainder (fraction_expand)

- Source: `(a*x*y+b*y*z+c*x*z+d)/(x*y*z)`
- Target: `a/z + b/x + c/y + d/(x*y*z)`
- Result: `a / z + b / x + c / y + d / (x * y * z)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a * x * y + b * y * z + c * x * z + d) / (x * y * z)
Target: a / z + b / x + c / y + d / (x * y * z)
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Distribute Division]
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
     1. `Repartir el mismo denominador sobre cada término del numerador`
     2. `Cancelar los factores comunes en las fracciones resultantes`

## expand_fraction_three_factor_three_cancellations_to_constant (fraction_expand)

- Source: `(a*x*y+b*x*z+c*y*z+d*x*y*z)/(x*y*z)`
- Target: `a/z + b/y + c/x + d`
- Result: `a / z + b / y + c / x + d`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (d * x * y * z + a * x * y + b * x * z + c * y * z) / (x * y * z)
Target: a / z + b / y + c / x + d
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Distribute Division]
   Before: (d * x * y * z + a * x * y + b * x * z + c * y * z) / (x * y * z)
   After: a / z + b / y + c / x + d
Result: a / z + b / y + c / x + d
ℹ️ Requires:
  • y ≠ 0
  • z ≠ 0
  • x ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(d · x · y · z + a · x · y + b · x · z + c · y · z)/(x · y · z)`
   - after: `a/z + b/y + c/x + d`
   - substeps:
     1. `Repartir el mismo denominador sobre cada término del numerador`
     2. `Cancelar los factores comunes en las fracciones resultantes`

## nested_fraction_one_over_sum (nested_fraction)

- Source: `1/(1/x + 1/y)`
- Target: `(x*y)/(x+y)`
- Result: `x * y / (x + y)`
- Web step count: `2`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 1 / (1 / x + 1 / y)
Target: x * y / (x + y)
Strategy: simplify
Steps (Aggressive Mode):
1. Add fractions: a/b + c/d -> (ad+bc)/bd  [Add Fractions]
   Before: 1 / (1 / x + 1 / y)
   Cambio local: 1 / x + 1 / y -> (x + y) / (x * y)
   After: 1 / ((x + y) / (x * y))
2. Simplify nested fraction  [Simplify Complex Fraction]
   Before: 1 / ((x + y) / (x * y))
   Cambio local: 1 / ((x + y) / (x * y)) -> x * y / (x + y)
   After: x * y / (x + y)
Result: x * y / (x + y)
ℹ️ Requires:
  • x + y ≠ 0
```

### Web / JSON Steps

1. `Sumar fracciones`
   - before: `1/(1/x + 1/y)`
   - after: `1/((x + y)/(x · y))`
   - substeps: none
2. `Simplificar fracción anidada`
   - before: `1/((x + y)/(x · y))`
   - after: `(x · y)/(x + y)`
   - substeps: none

## nested_fraction_one_over_three_reciprocals (nested_fraction)

- Source: `1/(1/a + 1/b + 1/c)`
- Target: `(a*b*c)/(a*b + a*c + b*c)`
- Result: `a * b * c / (a * b + a * c + b * c)`
- Web step count: `3`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 1 / (1 / a + 1 / b + 1 / c)
Target: a * b * c / (a * b + a * c + b * c)
Strategy: simplify
Steps (Aggressive Mode):
1. Add fractions: a/b + c/d -> (ad+bc)/bd  [Add Fractions]
   Before: 1 / (1 / a + 1 / b + 1 / c)
   Cambio local: 1 / a + 1 / b -> (a + b) / (a * b)
   After: 1 / (1 / c + (a + b) / (a * b))
2. Add fractions: a/b + c/d -> (ad+bc)/bd  [Add Fractions]
   Before: 1 / (1 / c + (a + b) / (a * b))
   Cambio local: 1 / c + (a + b) / (a * b) -> ((a + b) * c + a * b) / (c * a * b)
   After: 1 / (((a + b) * c + a * b) / (c * a * b))
3. Simplify nested fraction  [Simplify Complex Fraction]
   Before: 1 / (((a + b) * c + a * b) / (c * a * b))
   Cambio local: 1 / ((a * b + a * c + b * c) / (a * b * c)) -> a * b * c / (a * b + a * c + b * c)
   After: a * b * c / (a * b + a * c + b * c)
Result: a * b * c / (a * b + a * c + b * c)
ℹ️ Requires:
  • a * b + a * c + b * c ≠ 0
```

### Web / JSON Steps

1. `Sumar fracciones`
   - before: `1/(1/a + 1/b + 1/c)`
   - after: `1/(1/c + (a + b)/(a · b))`
   - substeps: none
2. `Sumar fracciones`
   - before: `1/(1/c + (a + b)/(a · b))`
   - after: `1/(((a + b) · c + a · b)/(c · a · b))`
   - substeps: none
3. `Simplificar fracción anidada`
   - before: `1/(((a + b) · c + a · b)/(c · a · b))`
   - after: `(a · b · c)/(a · b + a · c + b · c)`
   - substeps: none

## nested_fraction_sum_over_reciprocal (nested_fraction)

- Source: `(1/x + 1/y)/(1/z)`
- Target: `z*(x+y)/(x*y)`
- Result: `z * (x + y) / (x * y)`
- Web step count: `2`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (1 / x + 1 / y) / (1 / z)
Target: z * (x + y) / (x * y)
Strategy: simplify
Steps (Aggressive Mode):
1. Add fractions: a/b + c/d -> (ad+bc)/bd  [Add Fractions]
   Before: (1 / x + 1 / y) / (1 / z)
   Cambio local: 1 / x + 1 / y -> (x + y) / (x * y)
   After: (x + y) / (x * y) / (1 / z)
2. Simplify nested fraction  [Simplify Complex Fraction]
   Before: (x + y) / (x * y) / (1 / z)
   Cambio local: (x + y) / (x * y) / (1 / z) -> (x + y) * z / (x * y)
   After: z * (x + y) / (x * y)
Result: z * (x + y) / (x * y)
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Sumar fracciones`
   - before: `(1/x + 1/y)/(1/z)`
   - after: `((x + y)/(x · y))/(1/z)`
   - substeps: none
2. `Simplificar fracción anidada`
   - before: `((x + y)/(x · y))/(1/z)`
   - after: `(z · (x + y))/(x · y)`
   - substeps: none

## nested_fraction_one_over_sum_with_fraction (nested_fraction)

- Source: `1/(x + y/z)`
- Target: `z/(x*z+y)`
- Result: `z / (x * z + y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 1 / (y / z + x)
Target: z / (x * z + y)
Strategy: simplify
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplify Complex Fraction]
   Before: 1 / (y / z + x)
   Cambio local: 1 / (y / z + x) -> z / (x * z + y)
   After: z / (x * z + y)
Result: z / (x * z + y)
ℹ️ Requires:
  • x * z + y ≠ 0
```

### Web / JSON Steps

1. `Simplificar fracción anidada`
   - before: `1/(y/z + x)`
   - after: `z/(x · z + y)`
   - substeps:
     1. `Primero simplificar la suma del denominador`

## nested_fraction_fraction_over_sum_with_fraction_general (nested_fraction)

- Source: `a/(b + c/d)`
- Target: `a*d/(b*d+c)`
- Result: `a * d / (b * d + c)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a / (c / d + b)
Target: a * d / (b * d + c)
Strategy: simplify
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplify Complex Fraction]
   Before: a / (c / d + b)
   Cambio local: a / (c / d + b) -> a * d / (b * d + c)
   After: a * d / (b * d + c)
Result: a * d / (b * d + c)
ℹ️ Requires:
  • b * d + c ≠ 0
```

### Web / JSON Steps

1. `Simplificar fracción anidada`
   - before: `a/(c/d + b)`
   - after: `(a · d)/(b · d + c)`
   - substeps:
     1. `Primero simplificar la suma del denominador`

## nested_fraction_sum_with_fraction_over_scalar_general (nested_fraction)

- Source: `(a + b/c)/d`
- Target: `(a*c+b)/(c*d)`
- Result: `(a * c + b) / (c * d)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: (b / c + a) / d
Target: (a * c + b) / (c * d)
Strategy: simplify
Steps (Aggressive Mode):
1. Simplify nested fraction  [Simplify Complex Fraction]
   Before: (b / c + a) / d
   Cambio local: (b / c + a) / d -> (a * c + b) / (d * c)
   After: (a * c + b) / (c * d)
Result: (a * c + b) / (c * d)
ℹ️ Requires:
  • c ≠ 0
  • d ≠ 0
```

### Web / JSON Steps

1. `Simplificar fracción anidada`
   - before: `(b/c + a)/d`
   - after: `(a · c + b)/(c · d)`
   - substeps: none

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
Strategy: simplify
Steps (Aggressive Mode):
1. Simplify the expression  [Simplify]
   Before: z / (x * z + y)
   Cambio local: z / (x * z + y) -> 1 / (y / z + x)
   After: 1 / (y / z + x)
Result: 1 / (y / z + x)
ℹ️ Requires:
  • z ≠ 0
  • y / z + x ≠ 0
```

### Web / JSON Steps

1. `Simplificar fracción anidada`
   - before: `z/(x · z + y)`
   - after: `1/(y/z + x)`
   - substeps:
     1. `Reescribir el denominador sacando factor común z`

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
Strategy: simplify
Steps (Aggressive Mode):
1. Simplify the expression  [Simplify]
   Before: a * d / (b * d + c)
   Cambio local: a * d / (b * d + c) -> a / (c / d + b)
   After: a / (c / d + b)
Result: a / (c / d + b)
ℹ️ Requires:
  • c / d + b ≠ 0
  • d ≠ 0
```

### Web / JSON Steps

1. `Simplificar fracción anidada`
   - before: `(a · d)/(b · d + c)`
   - after: `a/(c/d + b)`
   - substeps:
     1. `Reescribir el denominador sacando factor común d`

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
Strategy: simplify
Steps (Aggressive Mode):
1. Simplify the expression  [Simplify]
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

## combine_same_denominator_fraction_sum (fraction_combine)

- Source: `a/x + b/x`
- Target: `(a+b)/x`
- Result: `(a + b) / x`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a / x + b / x
Target: (a + b) / x
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine fractions that already share the same denominator  [Combine Same Denominator Fractions]
   Before: a / x + b / x
   Cambio local: a / x + b / x -> (a + b) / x
   After: (a + b) / x
Result: (a + b) / x
ℹ️ Requires:
  • x ≠ 0
```

### Web / JSON Steps

1. `Sumar fracciones con mismo denominador`
   - before: `a/x + b/x`
   - after: `(a + b)/x`
   - substeps: none

## combine_general_fraction_sum (fraction_combine)

- Source: `1/x + 1/y`
- Target: `(x+y)/(x*y)`
- Result: `(x + y) / (x * y)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 1 / x + 1 / y
Target: (x + y) / (x * y)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine two fractions into a single denominator  [Add Fractions]
   Before: 1 / x + 1 / y
   Cambio local: 1 / x + 1 / y -> (x + y) / (x * y)
   After: (x + y) / (x * y)
Result: (x + y) / (x * y)
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Sumar fracciones`
   - before: `1/x + 1/y`
   - after: `(x + y)/(x · y)`
   - substeps: none

## combine_same_denominator_fraction_difference (fraction_combine)

- Source: `a/x - b/x`
- Target: `(a-b)/x`
- Result: `(a - b) / x`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: a / x - b / x
Target: (a - b) / x
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine fractions with the same denominator into one subtraction  [Combine Same Denominator Sub]
   Before: a / x - b / x
   Cambio local: a / x - b / x -> (a - b) / x
   After: (a - b) / x
Result: (a - b) / x
ℹ️ Requires:
  • x ≠ 0
```

### Web / JSON Steps

1. `Restar fracciones con mismo denominador`
   - before: `a/x - b/x`
   - after: `(a - b)/x`
   - substeps: none

## combine_general_fraction_difference (fraction_combine)

- Source: `1/x - 1/y`
- Target: `(y-x)/(x*y)`
- Result: `(y - x) / (x * y)`
- Web step count: `1`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: 1 / x - 1 / y
Target: (y - x) / (x * y)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Subtract two fractions into a single denominator  [Subtract Fractions]
   Before: 1 / x - 1 / y
   Cambio local: 1 / x - 1 / y -> (y - x) / (x * y)
   After: (y - x) / (x * y)
Result: (y - x) / (x * y)
ℹ️ Requires:
  • x ≠ 0
  • y ≠ 0
```

### Web / JSON Steps

1. `Restar fracciones`
   - before: `1/x - 1/y`
   - after: `(y - x)/(x · y)`
   - substeps: none

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
1. Put the term and the fraction over the same denominator  [Combine Same Denominator Sub]
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
1. Combine fractions that already share the same denominator  [Combine Same Denominator Fractions]
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
1. Split a fraction into a whole part plus remainder  [Mixed Fraction Split]
   Before: (x + 1) / (x - 1)
   After: 2 / (x - 1) + 1
Result: 2 / (x - 1) + 1
ℹ️ Requires:
  • x - 1 ≠ 0
```

### Web / JSON Steps

1. `Separar parte entera y resto`
   - before: `(x + 1)/(x - 1)`
   - after: `2/(x - 1) + 1`
   - substeps:
     1. `Reescribir el numerador como denominador · parte entera + resto`
     2. `Separar la parte entera de la fracción restante`

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
1. Split a fraction into a whole part plus remainder  [Mixed Fraction Split]
   Before: (4 * x + 7) / (2 * x + 1)
   After: 5 / (2 * x + 1) + 2
Result: 5 / (2 * x + 1) + 2
ℹ️ Requires:
  • 2 * x + 1 ≠ 0
```

### Web / JSON Steps

1. `Separar parte entera y resto`
   - before: `(4 · x + 7)/(2 · x + 1)`
   - after: `5/(2 · x + 1) + 2`
   - substeps:
     1. `Reescribir el numerador como denominador · parte entera + resto`
     2. `Separar la parte entera de la fracción restante`

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
1. Split a fraction into a whole part plus remainder  [Mixed Fraction Split]
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
     1. `Reescribir el numerador como denominador · parte entera + resto`
     2. `Separar la parte entera de la fracción restante`

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
1. Split a fraction into a whole part plus remainder  [Mixed Fraction Split]
   Before: (a * x + b) / (c * x + d)
   After: a / c + (b - a * d / c) / (c * x + d)
Result: a / c + (b - a * d / c) / (c * x + d)
ℹ️ Requires:
  • c ≠ 0
  • c * x + d ≠ 0
```

### Web / JSON Steps

1. `Separar parte entera y resto`
   - before: `(a · x + b)/(c · x + d)`
   - after: `a/c + (b - a · d/c)/(c · x + d)`
   - substeps:
     1. `Reescribir el numerador como denominador · parte entera + resto`
     2. `Separar la parte entera de la fracción restante`

## split_fraction_symbolic_over_negative_scaled_general_linear (fraction_decompose)

- Source: `(a*x+b)/(d-c*x)`
- Target: `-a/c + (b+a*d/c)/(d-c*x)`
- Result: `-a / c + (a * d / c + b) / (d - c * x)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a * x + b) / (d - c * x)
Target: -a / c + (a * d / c + b) / (d - c * x)
Strategy: split fraction
Steps (Aggressive Mode):
1. Split a fraction into a whole part plus remainder  [Mixed Fraction Split]
   Before: (a * x + b) / (d - c * x)
   After: -a / c + (a * d / c + b) / (d - c * x)
Result: -a / c + (a * d / c + b) / (d - c * x)
ℹ️ Requires:
  • c * x - d ≠ 0
  • c ≠ 0
```

### Web / JSON Steps

1. `Separar parte entera y resto`
   - before: `(a · x + b)/(d - c · x)`
   - after: `-a/c + ((a · d)/c + b)/(d - c · x)`
   - substeps:
     1. `Reescribir el numerador como denominador · parte entera + resto`
     2. `Separar la parte entera de la fracción restante`

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
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / (n * (n + 1))
   Cambio local: 1 / (n * (n + 1)) -> 1 / n - 1 / (n + 1)
   After: 1 / n - 1 / (n + 1)
Result: 1 / n - 1 / (n + 1)
ℹ️ Requires:
  • n ≠ 0
  • n + 1 ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/(n · (n + 1))`
   - after: `1/n - 1/(n + 1)`
   - substeps:
     1. `Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)`
     2. `Aquí u = n`

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
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / (n * (n + 2))
   Cambio local: 1 / (n * (n + 2)) -> 1 / 2 * (1 / n - 1 / (n + 2))
   After: ((1 / n - 1 / (n + 2)))/2
Result: 1 / 2 * (1 / n - 1 / (n + 2))
ℹ️ Requires:
  • 2 ≠ 0
  • n ≠ 0
  • n + 2 ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/(n · (n + 2))`
   - after: `1/2 · (1/n - 1/(n + 2))`
   - substeps:
     1. `Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))`
     2. `Aquí u = n y k = 2`

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
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / n - 1 / (n + 1)
   Cambio local: 1 / n - 1 / (n + 1) -> 1 / (n * (n + 1))
   After: 1 / (n * (n + 1))
Result: 1 / (n * (n + 1))
ℹ️ Requires:
  • n ≠ 0
  • n + 1 ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/n - 1/(n + 1)`
   - after: `1/(n · (n + 1))`
   - substeps:
     1. `Usar 1 / u - 1 / (u + 1) = 1 / (u · (u + 1))`
     2. `Aquí u = n`

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
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / 2 * (1 / n - 1 / (n + 2))
   Cambio local: 1 / 2 * (1 / n - 1 / (n + 2)) -> 1 / (n * (n + 2))
   After: 1 / (n * (n + 2))
Result: 1 / (n * (n + 2))
ℹ️ Requires:
  • n ≠ 0
  • n + 2 ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/2 · (1/n - 1/(n + 2))`
   - after: `1/(n · (n + 2))`
   - substeps:
     1. `Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))`
     2. `Aquí u = n y k = 2`

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
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / (n * (n - 2))
   Cambio local: 1 / (n * (n - 2)) -> 1 / 2 * (1 / (n - 2) - 1 / n)
   After: ((1 / (n - 2) - 1 / n))/2
Result: 1 / 2 * (1 / (n - 2) - 1 / n)
ℹ️ Requires:
  • n - 2 ≠ 0
  • n ≠ 0
  • 2 ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/(n · (n - 2))`
   - after: `1/2 · (1/(n - 2) - 1/n)`
   - substeps:
     1. `Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))`
     2. `Aquí u = n - 2 y k = 2`

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
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / 2 * (1 / (n - 2) - 1 / n)
   Cambio local: 1 / 2 * (1 / (n - 2) - 1 / n) -> 1 / (n * (n - 2))
   After: 1 / (n * (n - 2))
Result: 1 / (n * (n - 2))
ℹ️ Requires:
  • n ≠ 0
  • n - 2 ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/2 · (1/(n - 2) - 1/n)`
   - after: `1/(n · (n - 2))`
   - substeps:
     1. `Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))`
     2. `Aquí u = n - 2 y k = 2`

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
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / ((2 * n + 1) * (2 * n + 3))
   Cambio local: 1 / ((2 * n + 1) * (2 * n + 3)) -> 1 / 2 * (1 / (2 * n + 1) - 1 / (2 * n + 3))
   After: ((1 / (2 * n + 1) - 1 / (2 * n + 3)))/2
Result: 1 / 2 * (1 / (2 * n + 1) - 1 / (2 * n + 3))
ℹ️ Requires:
  • 2 * n + 3 ≠ 0
  • 2 ≠ 0
  • 2 * n + 1 ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/((2 · n + 1) · (2 · n + 3))`
   - after: `1/2 · (1/(2 · n + 1) - 1/(2 · n + 3))`
   - substeps:
     1. `Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))`
     2. `Aquí u = 2 · n + 1 y k = 2`

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
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / 2 * (1 / (2 * n + 1) - 1 / (2 * n + 3))
   Cambio local: 1 / 2 * (1 / (2 * n + 1) - 1 / (2 * n + 3)) -> 1 / ((2 * n + 1) * (2 * n + 3))
   After: 1 / ((2 * n + 1) * (2 * n + 3))
Result: 1 / ((2 * n + 1) * (2 * n + 3))
ℹ️ Requires:
  • 2 * n + 1 ≠ 0
  • 2 * n + 3 ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/2 · (1/(2 · n + 1) - 1/(2 · n + 3))`
   - after: `1/((2 · n + 1) · (2 · n + 3))`
   - substeps:
     1. `Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))`
     2. `Aquí u = 2 · n + 1 y k = 2`

## split_telescoping_fraction_affine_symbolic_shift_gap (telescoping_fraction)

- Source: `1/((a*n+b)*(a*n+c))`
- Target: `1/(c-b)*(1/(a*n+b) - 1/(a*n+c))`
- Result: `((1 / (a * n + b) - 1 / (a * n + c)) * 1)/((c - b))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / ((a * n + b) * (a * n + c))
Target: ((1 / (a * n + b) - 1 / (a * n + c)))/((c - b))
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / ((a * n + b) * (a * n + c))
   Cambio local: 1 / ((a * n + b) * (a * n + c)) -> 1 / (c - b) * (1 / (a * n + b) - 1 / (a * n + c))
   After: ((1 / (a * n + b) - 1 / (a * n + c)))/((c - b))
Result: 1 / (c - b) * (1 / (a * n + b) - 1 / (a * n + c))
ℹ️ Requires:
  • a * n + c ≠ 0
  • a * n + b ≠ 0
  • b - c ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/((a · n + b) · (a · n + c))`
   - after: `1/(c - b) · (1/(a · n + b) - 1/(a · n + c))`
   - substeps:
     1. `Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))`
     2. `Aquí u = a · n + b y k = c - b`

## combine_telescoping_fraction_affine_symbolic_shift_gap (telescoping_fraction)

- Source: `1/(c-b)*(1/(a*n+b) - 1/(a*n+c))`
- Target: `1/((a*n+b)*(a*n+c))`
- Result: `1 / ((a * n + b) * (a * n + c))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ((1 / (a * n + b) - 1 / (a * n + c)) * 1)/((c - b))
Target: 1 / ((a * n + b) * (a * n + c))
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / (c - b) * (1 / (a * n + b) - 1 / (a * n + c))
   Cambio local: 1 / (c - b) * (1 / (a * n + b) - 1 / (a * n + c)) -> 1 / ((a * n + b) * (a * n + c))
   After: 1 / ((a * n + b) * (a * n + c))
Result: 1 / ((a * n + b) * (a * n + c))
ℹ️ Requires:
  • a^2 * n^2 + a * b * n + a * c * n + b * c ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/(c - b) · (1/(a · n + b) - 1/(a · n + c))`
   - after: `1/((a · n + b) · (a · n + c))`
   - substeps:
     1. `Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))`
     2. `Aquí u = a · n + b y k = c - b`

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
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / (x^(2) - 1)
   Cambio local: 1 / (x^(2) - 1) -> 1 / 2 * (1 / (x - 1) - 1 / (x + 1))
   After: ((1 / (x - 1) - 1 / (x + 1)))/2
Result: 1 / 2 * (1 / (x - 1) - 1 / (x + 1))
ℹ️ Requires:
  • x - 1 ≠ 0
  • x + 1 ≠ 0
  • 2 ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/(x^2 - 1)`
   - after: `1/2 · (1/(x - 1) - 1/(x + 1))`
   - substeps:
     1. `Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))`
     2. `Aquí u = x - 1 y k = 2`

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
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / 2 * (1 / (x - 1) - 1 / (x + 1))
   Cambio local: 1 / 2 * (1 / (x - 1) - 1 / (x + 1)) -> 1 / (x^(2) - 1)
   After: 1 / (x^2 - 1)
Result: 1 / (x^(2) - 1)
ℹ️ Requires:
  • x - 1 ≠ 0
  • x + 1 ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/2 · (1/(x - 1) - 1/(x + 1))`
   - after: `1/(x^2 - 1)`
   - substeps:
     1. `Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))`
     2. `Aquí u = x - 1 y k = 2`

## combine_telescoping_fraction_shifted_quadratic_unfactored (telescoping_fraction)

- Source: `1/(x+1) - 1/(x+2)`
- Target: `1/(x^2+3*x+2)`
- Result: `1 / (x^2 + 3 * x + 2)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (x + 1) - 1 / (x + 2)
Target: 1 / (x^2 + 3 * x + 2)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / (x + 1) - 1 / (x + 2)
   Cambio local: 1 / (x + 1) - 1 / (x + 2) -> 1 / (x^(2) + 3 * x + 2)
   After: 1 / (x^2 + 3 * x + 2)
Result: 1 / (x^(2) + 3 * x + 2)
ℹ️ Requires:
  • x + 1 ≠ 0
  • x + 2 ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/(x + 1) - 1/(x + 2)`
   - after: `1/(x^2 + 3 · x + 2)`
   - substeps:
     1. `Usar 1 / u - 1 / (u + 1) = 1 / (u · (u + 1))`
     2. `Aquí u = x + 1`

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
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / (x^(2) - a^(2))
   Cambio local: 1 / (x^(2) - a^(2)) -> 1 / (2 * a) * (1 / (x - a) - 1 / (a + x))
   After: ((1 / (x - a) - 1 / (a + x)))/(2 * a)
Result: 1 / (2 * a) * (1 / (x - a) - 1 / (a + x))
ℹ️ Requires:
  • a ≠ 0
  • a + x ≠ 0
  • a - x ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/(x^2 - a^2)`
   - after: `1/(2 · a) · (1/(x - a) - 1/(a + x))`
   - substeps:
     1. `Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))`
     2. `Aquí u = x - a y k = 2 · a`

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
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / (2 * a) * (1 / (x - a) - 1 / (a + x))
   Cambio local: 1 / (2 * a) * (1 / (x - a) - 1 / (a + x)) -> 1 / (x^(2) - a^(2))
   After: 1 / (x^2 - a^2)
Result: 1 / (x^(2) - a^(2))
ℹ️ Requires:
  • a - x ≠ 0
  • a + x ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/(2 · a) · (1/(x - a) - 1/(a + x))`
   - after: `1/(x^2 - a^2)`
   - substeps:
     1. `Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))`
     2. `Aquí u = x - a y k = 2 · a`

## combine_whole_plus_remainder_into_fraction (fraction_combine)

- Source: `1 + 2/(x-1)`
- Target: `(x+1)/(x-1)`
- Result: `(x + 1) / (x - 1)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 / (x - 1) + 1
Target: (x + 1) / (x - 1)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine the whole part with the remaining fraction  [Mixed Fraction Combine]
   Before: 2 / (x - 1) + 1
   After: (x + 1) / (x - 1)
Result: (x + 1) / (x - 1)
ℹ️ Requires:
  • x - 1 ≠ 0
```

### Web / JSON Steps

1. `Unir parte entera y fracción`
   - before: `2/(x - 1) + 1`
   - after: `(x + 1)/(x - 1)`
   - substeps:
     1. `Poner la parte entera sobre el mismo denominador`
     2. `Sumar los numeradores y conservar el denominador`

## combine_symbolic_whole_plus_remainder_into_fraction (fraction_combine)

- Source: `a + (b-a*c)/(x+c)`
- Target: `(a*x+b)/(x+c)`
- Result: `(a * x + b) / (c + x)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (b - a * c) / (c + x) + a
Target: (a * x + b) / (c + x)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine the whole part with the remaining fraction  [Mixed Fraction Combine]
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
   - substeps:
     1. `Poner la parte entera sobre el mismo denominador`
     2. `Sumar los numeradores y conservar el denominador`

## combine_scaled_symbolic_whole_plus_remainder_into_fraction (fraction_combine)

- Source: `a/c + (b-a*d/c)/(c*x+d)`
- Target: `(a*x+b)/(c*x+d)`
- Result: `(a * x + b) / (c * x + d)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a / c + (b - a * d / c) / (c * x + d)
Target: (a * x + b) / (c * x + d)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine the whole part with the remaining fraction  [Mixed Fraction Combine]
   Before: a / c + (b - a * d / c) / (c * x + d)
   After: (a * x + b) / (c * x + d)
Result: (a * x + b) / (c * x + d)
ℹ️ Requires:
  • c * x + d ≠ 0
```

### Web / JSON Steps

1. `Unir parte entera y fracción`
   - before: `a/c + (b - a · d/c)/(c · x + d)`
   - after: `(a · x + b)/(c · x + d)`
   - substeps:
     1. `Poner la parte entera sobre el mismo denominador`
     2. `Sumar los numeradores y conservar el denominador`

## combine_negative_scaled_symbolic_whole_plus_remainder_into_fraction (fraction_combine)

- Source: `-a/c + (b+a*d/c)/(d-c*x)`
- Target: `(a*x+b)/(d-c*x)`
- Result: `(a * x + b) / (d - c * x)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: -a / c + (a * d / c + b) / (d - c * x)
Target: (a * x + b) / (d - c * x)
Strategy: combine fraction
Steps (Aggressive Mode):
1. Combine the whole part with the remaining fraction  [Mixed Fraction Combine]
   Before: -a / c + (a * d / c + b) / (d - c * x)
   After: (a * x + b) / (d - c * x)
Result: (a * x + b) / (d - c * x)
ℹ️ Requires:
  • c * x - d ≠ 0
```

### Web / JSON Steps

1. `Unir parte entera y fracción`
   - before: `-a/c + ((a · d)/c + b)/(d - c · x)`
   - after: `(a · x + b)/(d - c · x)`
   - substeps:
     1. `Poner la parte entera sobre el mismo denominador`
     2. `Sumar los numeradores y conservar el denominador`

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
1. Expand the expression distributively  [Expand]
   Before: (x + 1) * (x^(2) - x + 1)
   Cambio local: (x + 1) * (x^(2) - x + 1) -> x^(3) + 1
   After: x^3 + 1
Result: x^(3) + 1
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(x + 1) · (x^2 - x + 1)`
   - after: `x^3 + 1`
   - substeps:
     1. `Reconocer el patrón (a + b)(a^2 - ab + b^2)`
     2. `Aplicar (a + b)(a^2 - ab + b^2) = a^3 + b^3`

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
1. Expand the expression distributively  [Expand]
   Before: (x^(2) + x + 1) * (x - 1)
   Cambio local: (x^(2) + x + 1) * (x - 1) -> x^(3) - 1
   After: x^3 - 1
Result: x^(3) - 1
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(x^2 + x + 1) · (x - 1)`
   - after: `x^3 - 1`
   - substeps:
     1. `Reconocer el patrón (a - b)(a^2 + ab + b^2)`
     2. `Aplicar (a - b)(a^2 + ab + b^2) = a^3 - b^3`

## expand_difference_of_squares_quadratic_product (polynomial_product)

- Source: `(x^2+1)*(x^2-1)`
- Target: `x^4-1`
- Result: `x^4 - 1`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: (x^2 + 1) * (x^2 - 1)
Target: x^4 - 1
Strategy: expand
Steps (Aggressive Mode):
1. Expand the expression distributively  [Expand]
   Before: (x^(2) + 1) * (x^(2) - 1)
   Cambio local: (x^(2) + 1) * (x^(2) - 1) -> x^(4) - 1
   After: x^4 - 1
Result: x^(4) - 1
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(x^2 + 1) · (x^2 - 1)`
   - after: `x^4 - 1`
   - substeps:
     1. `Distribuir cada término del producto`
     2. `Agrupar los términos del mismo grado`
     3. `Al combinar esos términos, se cancelan`

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
1. Expand the expression distributively  [Expand]
   Before: (x^(2) + 1) * (x^(4) - x^(2) + 1)
   Cambio local: (x^(2) + 1) * (x^(4) - x^(2) + 1) -> x^(6) + 1
   After: x^6 + 1
Result: x^(6) + 1
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(x^2 + 1) · (x^4 - x^2 + 1)`
   - after: `x^6 + 1`
   - substeps:
     1. `Distribuir cada término del producto`
     2. `Agrupar los términos del mismo grado`
     3. `Los términos intermedios se cancelan por parejas`

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
1. Expand the expression distributively  [Expand]
   Before: (x^(4) + x^(2) + 1) * (x^(2) - 1)
   Cambio local: (x^(4) + x^(2) + 1) * (x^(2) - 1) -> x^(6) - 1
   After: x^6 - 1
Result: x^(6) - 1
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(x^4 + x^2 + 1) · (x^2 - 1)`
   - after: `x^6 - 1`
   - substeps:
     1. `Distribuir cada término del producto`
     2. `Agrupar los términos del mismo grado`
     3. `Los términos intermedios se cancelan por parejas`

## expand_eighth_power_minus_multifactor_product (polynomial_product)

- Source: `(x-1)*(x+1)*(x^2+1)*(x^4+1)`
- Target: `x^8-1`
- Result: `x^8 - 1`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (x + 1) * (x^2 + 1) * (x^4 + 1) * (x - 1)
Target: x^8 - 1
Strategy: expand
Steps (Aggressive Mode):
1. Expand the expression distributively  [Expand]
   Before: (x + 1) * (x^(2) + 1) * (x^(4) + 1) * (x - 1)
   Cambio local: (x + 1) * (x^(2) + 1) * (x^(4) + 1) * (x - 1) -> x^(8) - 1
   After: x^8 - 1
Result: x^(8) - 1
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(x + 1) · (x^2 + 1) · (x^4 + 1) · (x - 1)`
   - after: `x^8 - 1`
   - substeps:
     1. `Multiplicar y reagrupar por grados para cancelar términos intermedios`

## expand_ninth_power_plus_product (polynomial_product)

- Source: `(x^3+1)*(x^6-x^3+1)`
- Target: `x^9+1`
- Result: `x^9 + 1`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: (x^3 + 1) * (x^6 - x^3 + 1)
Target: x^9 + 1
Strategy: expand
Steps (Aggressive Mode):
1. Expand the expression distributively  [Expand]
   Before: (x^(3) + 1) * (x^(6) - x^(3) + 1)
   Cambio local: (x^(3) + 1) * (x^(6) - x^(3) + 1) -> x^(9) + 1
   After: x^9 + 1
Result: x^(9) + 1
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(x^3 + 1) · (x^6 - x^3 + 1)`
   - after: `x^9 + 1`
   - substeps:
     1. `Distribuir cada término del producto`
     2. `Agrupar los términos del mismo grado`
     3. `Los términos intermedios se cancelan por parejas`

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
1. Expand the expression distributively  [Expand]
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
1. Expand the expression distributively  [Expand]
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
1. Expand the expression distributively  [Expand]
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
1. Expand the expression distributively  [Expand]
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

## factor_geometric_difference_power_6 (factor)

- Source: `x^6 - 1`
- Target: `(x-1)*(x^5 + x^4 + x^3 + x^2 + x + 1)`
- Result: `(x^5 + x^4 + x^3 + x^2 + x + 1) * (x - 1)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: x^6 - 1
Target: (x^5 + x^4 + x^3 + x^2 + x + 1) * (x - 1)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorization]
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
     1. `Usar a^n - 1 = (a - 1) · (a^(n-1) + a^(n-2) + ... + a + 1)`
     2. `Aquí a = x y n = 6`

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
1. Combine powers with same base (n-ary)  [Combine powers with same base (n-ary)]
   Before: x^(1 / 2) * x^(2 / 3)
   Cambio local: x^(1 / 2) * x^(2 / 3) -> x^(7 / 6)
   After: x^(7 / 6)
Result: x^(7 / 6)
ℹ️ Requires:
  • 6 ≠ 0
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
1. Combine powers with same base (n-ary)  [Combine powers with same base (n-ary)]
   Before: x^(1 / 4) * x^(3 / 4)
   Cambio local: x^(1 / 4) * x^(3 / 4) -> x
   After: x
Result: x
```

### Web / JSON Steps

1. `Sumar exponentes de la misma base`
   - before: `sqrt[4]x · sqrt[4]x^3`
   - after: `x`
   - substeps: none

## merge_mixed_root_and_fractional_power_five_sixths (power_merge)

- Source: `sqrt(x)*x^(1/3)`
- Target: `x^(5/6)`
- Result: `x^(5 / 6)`
- Web step count: `2`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sqrt(x) * x^(1 / 3)
Target: x^(5 / 6)
Strategy: combine powers
Steps (Aggressive Mode):
1. sqrt(x) = x^(1/2)  [Canonicalize Roots]
   Before: sqrt(x) * x^(1 / 3)
   Cambio local: sqrt(x) * x^(1 / 3) -> x^(1/2) * x^(1 / 3)
   After: x^(1/2) * x^(1 / 3)
2. Combine powers with same base (n-ary)  [Combine powers with same base (n-ary)]
   Before: x^(1/2) * x^(1 / 3)
   Cambio local: x^(1/2) * x^(1 / 3) -> x^(5 / 6)
   After: x^(5 / 6)
Result: x^(5 / 6)
ℹ️ Requires:
  • 6 ≠ 0
```

### Web / JSON Steps

1. `Reescribir la raíz como potencia fraccionaria`
   - before: `sqrt(x) · sqrt[3]x`
   - after: `sqrt(x) · sqrt[3]x`
   - substeps: none
2. `Sumar exponentes de la misma base`
   - before: `sqrt(x) · sqrt[3]x`
   - after: `sqrt[6]x^5`
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
1. Combine powers with same base (n-ary)  [Combine powers with same base (n-ary)]
   Before: x * x^(1 / 3)
   Cambio local: x * x^(1 / 3) -> x^(4 / 3)
   After: x^(4 / 3)
Result: x^(4 / 3)
ℹ️ Requires:
  • 3 ≠ 0
```

### Web / JSON Steps

1. `Sumar exponentes de la misma base`
   - before: `x · sqrt[3]x`
   - after: `sqrt[3]x^4`
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
1. Combine powers with same base (n-ary)  [Combine powers with same base (n-ary)]
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
1. Combine powers with same base (n-ary)  [Combine powers with same base (n-ary)]
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

## merge_mixed_root_and_symbolic_power (power_merge)

- Source: `sqrt(x)*x^a`
- Target: `x^(a+1/2)`
- Result: `x^(1 / 2 + a)`
- Web step count: `2`
- Web substep count: `0`
- Flags: none

### CLI

```text
Parsed: sqrt(x) * x^a
Target: x^(1 / 2 + a)
Strategy: combine powers
Steps (Aggressive Mode):
1. sqrt(x) = x^(1/2)  [Canonicalize Roots]
   Before: sqrt(x) * x^(a)
   After: sqrt(x) * x^(a)
2. Combine powers with same base (n-ary)  [Combine powers with same base (n-ary)]
   Before: sqrt(x) * x^(a)
   Cambio local: sqrt(x) * x^(a) -> x^(1 / 2 + a)
   After: x^(1 / 2 + a)
Result: x^(1 / 2 + a)
ℹ️ Requires:
  • 2 ≠ 0
```

### Web / JSON Steps

1. `Reescribir la raíz como potencia fraccionaria`
   - before: `sqrt(x) · x^a`
   - after: `sqrt(x) · x^a`
   - substeps: none
2. `Sumar exponentes de la misma base`
   - before: `sqrt(x) · x^a`
   - after: `x^(1/2 + a)`
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
1. Combine powers with same base (n-ary)  [Combine powers with same base (n-ary)]
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
1. Log expansion  [expand_log]
   Before: ln(x^(3)) + ln(y^(2)) - ln(x^(3) * y^(2))
   Cambio local: ln(x^(3)) + ln(y^(2)) - ln(x^(3) * y^(2)) -> ln(x^(3)) + ln(y^(2)) - (ln(x^(3)) + ln(y^(2)))
   After: 0
   ℹ️ Requires: x^3 > 0
   ℹ️ Requires: y^2 > 0
Result: 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln(x^3) + ln(y^2) - ln(x^3 · y^2)`
   - after: `ln(x^3) + ln(y^2) - (ln(x^3) + ln(y^2))`
   - substeps: none

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
1. Rewrite an odd half-integer power using a square root  [Expand Odd Half Power]
   Before: x^(3 / 2)
   After: sqrt(x) * |x|
Result: sqrt(x) * |x|
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Reescribir potencia semientera impar`
   - before: `sqrt(x^3)`
   - after: `sqrt(x) · |x|`
   - substeps:
     1. `Separar la mitad entera de la mitad radical`
     2. `Usar que queda una raíz cuadrada del mismo factor`

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
1. Rewrite an odd half-integer power using a square root  [Expand Odd Half Power]
   Before: sqrt(x^(3))
   After: sqrt(x) * |x|
Result: sqrt(x) * |x|
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Reescribir potencia semientera impar`
   - before: `sqrt(x^3)`
   - after: `sqrt(x) · |x|`
   - substeps:
     1. `Separar la mitad entera de la mitad radical`
     2. `Usar que queda una raíz cuadrada del mismo factor`

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
1. Rewrite an odd half-integer power using a square root  [Expand Odd Half Power]
   Before: x^(5 / 2)
   After: sqrt(x) * |x|^2
Result: sqrt(x) * |x|^(2)
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Reescribir potencia semientera impar`
   - before: `sqrt(x^5)`
   - after: `sqrt(x) · |x|^2`
   - substeps:
     1. `Separar la mitad entera de la mitad radical`
     2. `Usar que queda una raíz cuadrada del mismo factor`

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
1. Rewrite an odd half-integer power using a square root  [Expand Odd Half Power]
   Before: sqrt(x^(5))
   After: sqrt(x) * |x|^2
Result: sqrt(x) * |x|^(2)
ℹ️ Requires:
  • x ≥ 0
```

### Web / JSON Steps

1. `Reescribir potencia semientera impar`
   - before: `sqrt(x^5)`
   - after: `sqrt(x) · |x|^2`
   - substeps:
     1. `Separar la mitad entera de la mitad radical`
     2. `Usar que queda una raíz cuadrada del mismo factor`

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
1. Rewrite an odd half-integer power using a square root  [Expand Odd Half Power]
   Before: y^(7 / 2)
   After: sqrt(y) * |y|^3
Result: sqrt(y) * |y|^(3)
ℹ️ Requires:
  • y ≥ 0
```

### Web / JSON Steps

1. `Reescribir potencia semientera impar`
   - before: `sqrt(y^7)`
   - after: `sqrt(y) · |y|^3`
   - substeps:
     1. `Separar la mitad entera de la mitad radical`
     2. `Usar que queda una raíz cuadrada del mismo factor`

## factor_out_with_division (conditional_factor)

- Source: `a*x + b*x + c`
- Target: `x*(a + b + c/x)`
- Result: `x * (c / x + a + b)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: a * x + b * x + c
Target: x * (c / x + a + b)
Strategy: factor out with division
Steps (Aggressive Mode):
1. Factor out x from the whole expression  [Factor Out With Division]
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
     1. `Si un término no lleva x, escribirlo como x · (t/x)`

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
Strategy: simplify
Steps (Aggressive Mode):
1. Cancel consecutive factorials  [Consecutive Factorial Ratio]
   Before: (n + 1)! / n!
   Cambio local: (n + 1)! / n! -> n + 1
   After: n + 1
Result: n + 1
```

### Web / JSON Steps

1. `Cancelar factoriales consecutivos`
   - before: `(n + 1)!/n!`
   - after: `n + 1`
   - substeps:
     1. `Escribir el factorial superior como el siguiente número por el factorial anterior`
     2. `Cancelar el factorial común`

## contract_trig_cos_diff_sin_diff_quotient (trig_contract)

- Source: `(cos(x)-cos(3*x))/(sin(3*x)-sin(x))`
- Target: `tan(2*x)`
- Result: `tan(2 * x)`
- Web step count: `3`
- Web substep count: `4`
- Flags: none

### CLI

```text
Parsed: (cos(x) - cos(3 * x)) / (sin(3 * x) - sin(x))
Target: tan(2 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. cos(A)−cos(B) = 2·sin((A+B)/2)·sin((B−A)/2)  [Cos-Diff / Sin-Diff Quotient]
   Before: (cos(x) - cos(3 * x)) / (sin(3 * x) - sin(x))
   Cambio local: cos(x) - cos(3 * x) -> 2 * sin(2 * x) * sin(x)
   After: 2 * sin(2 * x) * sin(x) / (sin(3 * x) - sin(x))
2. sin(B)−sin(A) = 2·cos((A+B)/2)·sin((B−A)/2)  [Cos-Diff / Sin-Diff Quotient]
   Before: 2 * sin(2 * x) * sin(x) / (sin(3 * x) - sin(x))
   Cambio local: sin(3 * x) - sin(x) -> 2 * cos(2 * x) * sin(x)
   After: 2 * sin(2 * x) * sin(x) / (2 * cos(2 * x) * sin(x))
3. Cancel common factors 2 and sin(half_gap)  [Cos-Diff / Sin-Diff Quotient]
   Before: 2 * sin(2 * x) * sin(x) / (2 * cos(2 * x) * sin(x))
   Cambio local: 2 * sin(2 * x) * sin(x) / (2 * cos(2 * x) * sin(x)) -> tan(2 * x)
   After: tan(2 * x)
Result: tan(2 * x)
```

### Web / JSON Steps

1. `Convertir un cociente trigonométrico en tangente`
   - before: `(cos(x) - cos(3 · x))/(sin(3 · x) - sin(x))`
   - after: `(2 · sin(2 · x) · sin(x))/(sin(3 · x) - sin(x))`
   - substeps:
     1. `Usar cos(A) - cos(B) = 2 · sin((A+B)/2) · sin((B-A)/2)`
2. `Convertir un cociente trigonométrico en tangente`
   - before: `(2 · sin(2 · x) · sin(x))/(sin(3 · x) - sin(x))`
   - after: `(2 · sin(2 · x) · sin(x))/(2 · cos(2 · x) · sin(x))`
   - substeps:
     1. `Usar sin(B) - sin(A) = 2 · cos((A+B)/2) · sin((B-A)/2)`
3. `Convertir un cociente trigonométrico en tangente`
   - before: `(2 · sin(2 · x) · sin(x))/(2 · cos(2 · x) · sin(x))`
   - after: `tan(2 · x)`
   - substeps:
     1. `Cancelar el factor común del numerador y del denominador`
     2. `Reconocer el patrón sin(u) / cos(u) = tan(u)`

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
Strategy: simplify
Steps (Aggressive Mode):
1. Recognize tan(u) · cot(u) = 1  [Reciprocal Product Identity]
   Before: tan(x) * cot(x)
   Cambio local: tan(x) * cot(x) -> 1
   After: 1
Result: 1
```

### Web / JSON Steps

1. `Cancelar funciones trigonométricas recíprocas`
   - before: `tan(x) · cot(x)`
   - after: `1`
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
Strategy: simplify
Steps (Aggressive Mode):
1. Recognize sec²(u) - tan²(u) = 1  [Reciprocal Pythagorean Identity]
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
1. Factorization  [Factorization]
   Before: a^(3) + b^(3) + 3 * a * b^(2) + 3 * b * a^(2)
   Cambio local: a^(3) + b^(3) + 3 * a * b^(2) + 3 * b * a^(2) -> (a + b)^(3)
   After: (a + b)^3
Result: (a + b)^(3)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a^3 + b^3 + 3 · a · b^2 + 3 · b · a^2`
   - after: `((a + b))^3`
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
1. Factorization  [Factorization]
   Before: a^(3) + 3 * a * b^(2) - 3 * b * a^(2) - b^(3)
   Cambio local: a^(3) + 3 * a * b^(2) - 3 * b * a^(2) - b^(3) -> (a - b)^(3)
   After: (a - b)^3
Result: (a - b)^(3)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a^3 - 3 · b · a^2 + 3 · a · b^2 - b^3`
   - after: `((a - b))^3`
   - substeps:
     1. `Usar a^3 - 3a^2b + 3ab^2 - b^3 = (a - b)^3`

## integrate_prep_morrie_basic (integrate_prep)

- Source: `cos(x)*cos(2*x)*cos(4*x)`
- Target: `sin(8*x)/(8*sin(x))`
- Result: `sin(8 * x) / (8 * sin(x))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: cos(x) * cos(2 * x) * cos(4 * x)
Target: sin(8 * x) / (8 * sin(x))
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply Morrie's law  [Cos Product Telescoping]
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
     1. `Usar el telescopado de cosenos`
     2. `Aquí u = x`

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
1. Apply Morrie's law  [Cos Product Telescoping]
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
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: cos(a * x) * cos(2 * a * x)
Target: sin(4 * a * x) / (4 * sin(a * x))
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply Morrie's law  [Cos Product Telescoping]
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
     1. `Usar el telescopado de cosenos`
     2. `Aquí u = a · x`

## integrate_prep_morrie_reverse_basic (integrate_prep)

- Source: `sin(8*x)/(8*sin(x))`
- Target: `cos(x)*cos(2*x)*cos(4*x)`
- Result: `cos(x) * cos(2 * x) * cos(4 * x)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sin(8 * x) / (8 * sin(x))
Target: cos(x) * cos(2 * x) * cos(4 * x)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply Morrie's law  [Cos Product Telescoping]
   Before: sin(8 * x) / (8 * sin(x))
   Cambio local: sin(8 * x) / (8 * sin(x)) -> cos(x) * cos(2 * x) * cos(4 * x)
   After: cos(x) * cos(2 * x) * cos(4 * x)
Result: cos(x) * cos(2 * x) * cos(4 * x)
```

### Web / JSON Steps

1. `Aplicar telescopado de cosenos`
   - before: `sin(8 · x)/(8 · sin(x))`
   - after: `cos(x) · cos(2 · x) · cos(4 · x)`
   - substeps:
     1. `Expandir la ley de Morrie`
     2. `Aquí u = x`

## integrate_prep_morrie_reverse_symbolic_scale_longer (integrate_prep)

- Source: `sin(8*a*x)/(8*sin(a*x))`
- Target: `cos(a*x)*cos(2*a*x)*cos(4*a*x)`
- Result: `cos(a * x) * cos(2 * a * x) * cos(4 * a * x)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sin(8 * a * x) / (8 * sin(a * x))
Target: cos(a * x) * cos(2 * a * x) * cos(4 * a * x)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply Morrie's law  [Cos Product Telescoping]
   Before: sin(8 * a * x) / (8 * sin(a * x))
   Cambio local: sin(8 * a * x) / (8 * sin(a * x)) -> cos(a * x) * cos(2 * a * x) * cos(4 * a * x)
   After: cos(a * x) * cos(2 * a * x) * cos(4 * a * x)
Result: cos(a * x) * cos(2 * a * x) * cos(4 * a * x)
```

### Web / JSON Steps

1. `Aplicar telescopado de cosenos`
   - before: `sin(8 · a · x)/(8 · sin(a · x))`
   - after: `cos(a · x) · cos(2 · a · x) · cos(4 · a · x)`
   - substeps:
     1. `Expandir la ley de Morrie`
     2. `Aquí u = a · x`

## integrate_prep_dirichlet_basic (integrate_prep)

- Source: `1 + 2*cos(x) + 2*cos(2*x)`
- Target: `sin(5*x/2)/sin(x/2)`
- Result: `sin(5 * x / 2) / sin(x / 2)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 * cos(x) + 2 * cos(2 * x) + 1
Target: sin(5 * x / 2) / sin(x / 2)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity  [Dirichlet Kernel Identity]
   Before: 2 * cos(x) + 2 * cos(2 * x) + 1
   Cambio local: 2 * cos(x) + 2 * cos(2 * x) + 1 -> sin(5 * x / 2) / sin(x / 2)
   After: sin(5 * x / 2) / sin(x / 2)
Result: sin(5 * x / 2) / sin(x / 2)
ℹ️ Requires:
  • 2 ≠ 0
  • sin(x / 2) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `2 · cos(x) + 2 · cos(2 · x) + 1`
   - after: `sin((5 · x)/2)/sin(x/2)`
   - substeps:
     1. `Usar el núcleo de Dirichlet`
     2. `Aquí n = 2 y u = x`

## integrate_prep_dirichlet_longer (integrate_prep)

- Source: `1 + 2*cos(x) + 2*cos(2*x) + 2*cos(3*x)`
- Target: `sin(7*x/2)/sin(x/2)`
- Result: `sin(7 * x / 2) / sin(x / 2)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 * cos(x) + 2 * cos(2 * x) + 2 * cos(3 * x) + 1
Target: sin(7 * x / 2) / sin(x / 2)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity  [Dirichlet Kernel Identity]
   Before: 2 * cos(x) + 2 * cos(2 * x) + 2 * cos(3 * x) + 1
   Cambio local: 2 * cos(x) + 2 * cos(2 * x) + 2 * cos(3 * x) + 1 -> sin(7 * x / 2) / sin(x / 2)
   After: sin(7 * x / 2) / sin(x / 2)
Result: sin(7 * x / 2) / sin(x / 2)
ℹ️ Requires:
  • sin(x / 2) ≠ 0
  • 2 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `2 · cos(x) + 2 · cos(2 · x) + 2 · cos(3 · x) + 1`
   - after: `sin((7 · x)/2)/sin(x/2)`
   - substeps:
     1. `Usar el núcleo de Dirichlet`
     2. `Aquí n = 3 y u = x`

## integrate_prep_dirichlet_symbolic_argument (integrate_prep)

- Source: `1 + 2*cos(u) + 2*cos(2*u) + 2*cos(3*u) + 2*cos(4*u)`
- Target: `sin(9*u/2)/sin(u/2)`
- Result: `sin(9 * u / 2) / sin(u / 2)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 * cos(u) + 2 * cos(2 * u) + 2 * cos(3 * u) + 2 * cos(4 * u) + 1
Target: sin(9 * u / 2) / sin(u / 2)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity  [Dirichlet Kernel Identity]
   Before: 2 * cos(u) + 2 * cos(2 * u) + 2 * cos(3 * u) + 2 * cos(4 * u) + 1
   Cambio local: 2 * cos(u) + 2 * cos(2 * u) + 2 * cos(3 * u) + 2 * cos(4 * u) + 1 -> sin(9 * u / 2) / sin(u / 2)
   After: sin(9 * u / 2) / sin(u / 2)
Result: sin(9 * u / 2) / sin(u / 2)
ℹ️ Requires:
  • sin(u / 2) ≠ 0
  • 2 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `2 · cos(u) + 2 · cos(2 · u) + 2 · cos(3 · u) + 2 · cos(4 · u) + 1`
   - after: `sin((9 · u)/2)/sin(u/2)`
   - substeps:
     1. `Usar el núcleo de Dirichlet`
     2. `Aquí n = 4`

## integrate_prep_dirichlet_symbolic_scale (integrate_prep)

- Source: `1 + 2*cos(a*x) + 2*cos(2*a*x)`
- Target: `sin(5*a*x/2)/sin(a*x/2)`
- Result: `sin(5 * a * x / 2) / sin(a * x / 2)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 * cos(a * x) + 2 * cos(2 * a * x) + 1
Target: sin(5 * a * x / 2) / sin(a * x / 2)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity  [Dirichlet Kernel Identity]
   Before: 2 * cos(a * x) + 2 * cos(2 * a * x) + 1
   Cambio local: 2 * cos(a * x) + 2 * cos(2 * a * x) + 1 -> sin(5 * a * x / 2) / sin(a * x / 2)
   After: sin(5 * a * x / 2) / sin(a * x / 2)
Result: sin(5 * a * x / 2) / sin(a * x / 2)
ℹ️ Requires:
  • sin(a * x / 2) ≠ 0
  • 2 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `2 · cos(a · x) + 2 · cos(2 · a · x) + 1`
   - after: `sin((5 · a · x)/2)/sin((a · x)/2)`
   - substeps:
     1. `Usar el núcleo de Dirichlet`
     2. `Aquí n = 2 y u = a · x`

## integrate_prep_dirichlet_symbolic_scale_longer (integrate_prep)

- Source: `1 + 2*cos(a*x) + 2*cos(2*a*x) + 2*cos(3*a*x)`
- Target: `sin(7*a*x/2)/sin(a*x/2)`
- Result: `sin(7 * a * x / 2) / sin(a * x / 2)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1
Target: sin(7 * a * x / 2) / sin(a * x / 2)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity  [Dirichlet Kernel Identity]
   Before: 2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1
   Cambio local: 2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1 -> sin(7 * a * x / 2) / sin(a * x / 2)
   After: sin(7 * a * x / 2) / sin(a * x / 2)
Result: sin(7 * a * x / 2) / sin(a * x / 2)
ℹ️ Requires:
  • 2 ≠ 0
  • sin(a * x / 2) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `2 · cos(a · x) + 2 · cos(2 · a · x) + 2 · cos(3 · a · x) + 1`
   - after: `sin((7 · a · x)/2)/sin((a · x)/2)`
   - substeps:
     1. `Usar el núcleo de Dirichlet`
     2. `Aquí n = 3 y u = a · x`

## integrate_prep_dirichlet_reverse_basic (integrate_prep)

- Source: `sin(5*x/2)/sin(x/2)`
- Target: `1 + 2*cos(x) + 2*cos(2*x)`
- Result: `2 * cos(x) + 2 * cos(2 * x) + 1`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sin(5 * x / 2) / sin(x / 2)
Target: 2 * cos(x) + 2 * cos(2 * x) + 1
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity  [Dirichlet Kernel Identity]
   Before: sin(5 * x / 2) / sin(x / 2)
   Cambio local: sin(5 * x / 2) / sin(x / 2) -> 2 * cos(x) + 2 * cos(2 * x) + 1
   After: 2 * cos(x) + 2 * cos(2 * x) + 1
Result: 2 * cos(x) + 2 * cos(2 * x) + 1
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `sin((5 · x)/2)/sin(x/2)`
   - after: `2 · cos(x) + 2 · cos(2 · x) + 1`
   - substeps:
     1. `Expandir el núcleo de Dirichlet`
     2. `Aquí n = 2 y u = x`

## integrate_prep_dirichlet_reverse_symbolic_scale_longer (integrate_prep)

- Source: `sin(7*a*x/2)/sin(a*x/2)`
- Target: `1 + 2*cos(a*x) + 2*cos(2*a*x) + 2*cos(3*a*x)`
- Result: `2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sin(7 * a * x / 2) / sin(a * x / 2)
Target: 2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity  [Dirichlet Kernel Identity]
   Before: sin(7 * a * x / 2) / sin(a * x / 2)
   Cambio local: sin(7 * a * x / 2) / sin(a * x / 2) -> 2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1
   After: 2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1
Result: 2 * cos(a * x) + 2 * cos(2 * a * x) + 2 * cos(3 * a * x) + 1
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `sin((7 · a · x)/2)/sin((a · x)/2)`
   - after: `2 · cos(a · x) + 2 · cos(2 · a · x) + 2 · cos(3 · a · x) + 1`
   - substeps:
     1. `Expandir el núcleo de Dirichlet`
     2. `Aquí n = 3 y u = a · x`

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
Strategy: simplify
Steps (Aggressive Mode):
1. Telescoping product: Π((k + 1) / k, k) from 1 to n  [Finite Product]
   Before: product((k + 1) / k, k, 1, n)
   Cambio local: product((k + 1) / k, k, 1, n) -> n + 1
   After: n + 1
Result: n + 1
```

### Web / JSON Steps

1. `Evaluar producto telescópico finito`
   - before: `prod_k=1^n (k + 1)/k`
   - after: `n + 1`
   - substeps:
     1. `Escribir los primeros y últimos factores del producto`
     2. `Los factores intermedios se cancelan por parejas`
     3. `Solo quedan el último numerador y el primer denominador`

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
Strategy: simplify
Steps (Aggressive Mode):
1. Telescoping product: Π((a + k + 1) / (a + k), k) from m to n  [Finite Product]
   Before: product((a + k + 1) / (a + k), k, m, n)
   Cambio local: product((a + k + 1) / (a + k), k, m, n) -> (a + n + 1) / (a + m)
   After: (a + n + 1) / (a + m)
Result: (a + n + 1) / (a + m)
ℹ️ Requires:
  • a + m ≠ 0
```

### Web / JSON Steps

1. `Evaluar producto telescópico finito`
   - before: `prod_k=m^n (a + k + 1)/(a + k)`
   - after: `(a + n + 1)/(a + m)`
   - substeps:
     1. `Escribir los primeros y últimos factores del producto`
     2. `Los factores intermedios se cancelan por parejas`

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
Strategy: simplify
Steps (Aggressive Mode):
1. Telescoping product: Π((a * k + a + b) / (a * k + b), k) from m to n  [Finite Product]
   Before: product((a * k + a + b) / (a * k + b), k, m, n)
   Cambio local: product((a * k + a + b) / (a * k + b), k, m, n) -> (a * n + a + b) / (a * m + b)
   After: (a * n + a + b) / (a * m + b)
Result: (a * n + a + b) / (a * m + b)
ℹ️ Requires:
  • a * m + b ≠ 0
```

### Web / JSON Steps

1. `Evaluar producto telescópico finito`
   - before: `prod_k=m^n (a · k + a + b)/(a · k + b)`
   - after: `(a · n + a + b)/(a · m + b)`
   - substeps:
     1. `Escribir los primeros y últimos factores del producto`
     2. `Los factores intermedios se cancelan por parejas`

## finite_telescoping_product_factorized_square_shifted_base_numeric_symbolic_lower (finite_telescoping)

- Source: `product(1 - 1/(k+2)^2, k, m, n)`
- Target: `((m+1)*(n+3))/((m+2)*(n+2))`
- Result: `(m + 1) * (n + 3) / ((m + 2) * (n + 2))`
- Web step count: `2`
- Web substep count: `4`
- Flags: none

### CLI

```text
Parsed: product(1 - 1 / (k + 2)^2, k, m, n)
Target: (m + 1) * (n + 3) / ((m + 2) * (n + 2))
Strategy: simplify
Steps (Aggressive Mode):
1. Add fractions: a/b + c/d -> (ad+bc)/bd  [Add Fractions]
   Before: product(1 - 1 / (k + 2)^(2), k, m, n)
   Cambio local: 1 - 1 / (k + 2)^(2) -> ((k + 2)^(2) - 1) / ((k + 2)^(2))
   After: product(((k + 2)^(2) - 1) / ((k + 2)^(2)), k, m, n)
2. Factorized telescoping product: Π(((k + 2)^2 - 1) / (k + 2)^2, k) from m to n  [Finite Product]
   Before: product(((k + 2)^(2) - 1) / ((k + 2)^(2)), k, m, n)
   Cambio local: product(((k + 2)^(2) - 1) / (k + 2)^(2), k, m, n) -> (m + 1) * (n + 3) / ((m + 2) * (n + 2))
   After: (m + 1) * (n + 3) / ((m + 2) * (n + 2))
Result: (m + 1) * (n + 3) / ((m + 2) * (n + 2))
ℹ️ Requires:
  • m * n + 2 * m + 2 * n + 4 ≠ 0
```

### Web / JSON Steps

1. `Sumar fracciones`
   - before: `prod_k=m^n 1 - 1/((k + 2))^2`
   - after: `prod_k=m^n (((k + 2))^2 - 1)/((k + 2))^2`
   - substeps: none
2. `Evaluar producto telescópico finito`
   - before: `prod_k=m^n (((k + 2))^2 - 1)/((k + 2))^2`
   - after: `((m + 1) · (n + 3))/((m + 2) · (n + 2))`
   - substeps:
     1. `Usar (u^2 - 1) / u^2 = ((u - 1) · (u + 1)) / u^2`
     2. `Aquí u = k + 2`
     3. `Los factores (u + 1) y (u - 1) se cancelan telescópicamente`
     4. `Solo quedan el primer factor u - 1 y el último factor u + 1`

## finite_telescoping_product_factorized_square_shifted_base_symbolic_symbolic_lower (finite_telescoping)

- Source: `product(1 - 1/(k+a)^2, k, m, n)`
- Target: `((m+a-1)*(n+a+1))/((m+a)*(n+a))`
- Result: `(a + n + 1) * (a + m - 1) / ((a + m) * (a + n))`
- Web step count: `2`
- Web substep count: `4`
- Flags: none

### CLI

```text
Parsed: product(1 - 1 / (a + k)^2, k, m, n)
Target: (a + n + 1) * (a + m - 1) / ((a + m) * (a + n))
Strategy: simplify
Steps (Aggressive Mode):
1. Add fractions: a/b + c/d -> (ad+bc)/bd  [Add Fractions]
   Before: product(1 - 1 / (a + k)^(2), k, m, n)
   Cambio local: 1 - 1 / (a + k)^(2) -> ((a + k)^(2) - 1) / ((a + k)^(2))
   After: product(((a + k)^(2) - 1) / ((a + k)^(2)), k, m, n)
2. Factorized telescoping product: Π(((a + k)^2 - 1) / (a + k)^2, k) from m to n  [Finite Product]
   Before: product(((a + k)^(2) - 1) / ((a + k)^(2)), k, m, n)
   Cambio local: product(((a + k)^(2) - 1) / (a + k)^(2), k, m, n) -> (a + n + 1) * (a + m - 1) / ((a + m) * (a + n))
   After: (a + n + 1) * (a + m - 1) / ((a + m) * (a + n))
Result: (a + n + 1) * (a + m - 1) / ((a + m) * (a + n))
ℹ️ Requires:
  • a^2 + a * m + a * n + m * n ≠ 0
```

### Web / JSON Steps

1. `Sumar fracciones`
   - before: `prod_k=m^n 1 - 1/((a + k))^2`
   - after: `prod_k=m^n (((a + k))^2 - 1)/((a + k))^2`
   - substeps: none
2. `Evaluar producto telescópico finito`
   - before: `prod_k=m^n (((a + k))^2 - 1)/((a + k))^2`
   - after: `((a + n + 1) · (a + m - 1))/((a + m) · (a + n))`
   - substeps:
     1. `Usar (u^2 - 1) / u^2 = ((u - 1) · (u + 1)) / u^2`
     2. `Aquí u = a + k`
     3. `Los factores (u + 1) y (u - 1) se cancelan telescópicamente`
     4. `Solo quedan el primer factor u - 1 y el último factor u + 1`

## finite_telescoping_sum_basic (finite_telescoping)

- Source: `sum(1/(k*(k+1)), k, 1, n)`
- Target: `1 - 1/(n+1)`
- Result: `1 - 1 / (n + 1)`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: sum(1 / (k * (k + 1)), k, 1, n)
Target: 1 - 1 / (n + 1)
Strategy: simplify
Steps (Aggressive Mode):
1. Telescoping sum: Σ(1 / (k * (k + 1)), k) from 1 to n  [Finite Summation]
   Before: sum(1 / (k * (k + 1)), k, 1, n)
   Cambio local: sum(1 / (k * (k + 1)), k, 1, n) -> 1 - 1 / (n + 1)
   After: 1 - 1 / (n + 1)
Result: 1 - 1 / (n + 1)
ℹ️ Requires:
  • n + 1 ≠ 0
```

### Web / JSON Steps

1. `Evaluar suma telescópica finita`
   - before: `sum_k=1^n 1/(k · (k + 1))`
   - after: `1 - 1/(n + 1)`
   - substeps:
     1. `Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)`
     2. `Aquí u = k`
     3. `La suma telescópica cancela los términos intermedios`

## finite_telescoping_sum_symbolic_shift_symbolic_lower (finite_telescoping)

- Source: `sum(1/((k+a)*(k+a+1)), k, m, n)`
- Target: `1/(m+a) - 1/(n+a+1)`
- Result: `1 / (a + m) - 1 / (a + n + 1)`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: sum(1 / ((a + k) * (a + k + 1)), k, m, n)
Target: 1 / (a + m) - 1 / (a + n + 1)
Strategy: simplify
Steps (Aggressive Mode):
1. Telescoping sum: Σ(1 / ((a + k) * (a + k + 1)), k) from m to n  [Finite Summation]
   Before: sum(1 / ((a + k) * (a + k + 1)), k, m, n)
   Cambio local: sum(1 / ((a + k) * (a + k + 1)), k, m, n) -> 1 / (a + m) - 1 / (a + n + 1)
   After: 1 / (a + m) - 1 / (a + n + 1)
Result: 1 / (a + m) - 1 / (a + n + 1)
ℹ️ Requires:
  • a + n + 1 ≠ 0
  • a + m ≠ 0
```

### Web / JSON Steps

1. `Evaluar suma telescópica finita`
   - before: `sum_k=m^n 1/((a + k) · (a + k + 1))`
   - after: `1/(a + m) - 1/(a + n + 1)`
   - substeps:
     1. `Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)`
     2. `Aquí u = a + k`
     3. `La suma telescópica cancela los términos intermedios`

## finite_telescoping_sum_affine_symbolic_shift_symbolic_lower (finite_telescoping)

- Source: `sum(1/((a*k+b)*(a*k+b+a)), k, m, n)`
- Target: `1/a*(1/(a*m+b) - 1/(a*n+a+b))`
- Result: `((1 / (a * m + b) - 1 / (a * n + a + b)) * 1)/a`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: sum(1 / ((a * k + b) * (a * k + a + b)), k, m, n)
Target: ((1 / (a * m + b) - 1 / (a * n + a + b)))/a
Strategy: simplify
Steps (Aggressive Mode):
1. Telescoping sum: Σ(1 / ((a * k + b) * (a * k + a + b)), k) from m to n  [Finite Summation]
   Before: sum(1 / ((a * k + b) * (a * k + a + b)), k, m, n)
   Cambio local: sum(1 / ((a * k + b) * (a * k + a + b)), k, m, n) -> 1 / a * (1 / (a * m + b) - 1 / (a * n + a + b))
   After: ((1 / (a * m + b) - 1 / (a * n + a + b)))/a
Result: 1 / a * (1 / (a * m + b) - 1 / (a * n + a + b))
ℹ️ Requires:
  • a ≠ 0
  • a * n + a + b ≠ 0
  • a * m + b ≠ 0
```

### Web / JSON Steps

1. `Evaluar suma telescópica finita`
   - before: `sum_k=m^n 1/((a · k + b) · (a · k + a + b))`
   - after: `1/a · (1/(a · m + b) - 1/(a · n + a + b))`
   - substeps:
     1. `Usar 1 / (u · (u + g)) = 1 / g · (1 / u - 1 / (u + g))`
     2. `Aquí u = a * k + b y g = a`
     3. `La suma telescópica cancela los términos intermedios`

## finite_telescoping_sum_affine_symbolic_arbitrary_shift_symbolic_lower (finite_telescoping)

- Source: `sum(1/((a*k+b+c)*(a*k+b+c+a)), k, m, n)`
- Target: `1/a*(1/(a*m+b+c) - 1/(a*n+a+b+c))`
- Result: `((1 / (a * m + b + c) - 1 / (a * n + a + b + c)) * 1)/a`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: sum(1 / ((a * k + a + b + c) * (a * k + b + c)), k, m, n)
Target: ((1 / (a * m + b + c) - 1 / (a * n + a + b + c)))/a
Strategy: simplify
Steps (Aggressive Mode):
1. Telescoping sum: Σ(1 / ((a * k + a + b + c) * (a * k + b + c)), k) from m to n  [Finite Summation]
   Before: sum(1 / ((a * k + a + b + c) * (a * k + b + c)), k, m, n)
   Cambio local: sum(1 / ((a * k + a + b + c) * (a * k + b + c)), k, m, n) -> 1 / a * (1 / (a * m + b + c) - 1 / (a * n + a + b + c))
   After: ((1 / (a * m + b + c) - 1 / (a * n + a + b + c)))/a
Result: 1 / a * (1 / (a * m + b + c) - 1 / (a * n + a + b + c))
ℹ️ Requires:
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
     2. `Aquí u = a * k + b + c y g = a`
     3. `La suma telescópica cancela los términos intermedios`

## rationalize_symbolic_linear_root (rationalize)

- Source: `1/(sqrt(x)-a)`
- Target: `(sqrt(x)+a)/(x-a^2)`
- Result: `(sqrt(x) + a) / (x - a^2)`
- Web step count: `2`
- Web substep count: `5`
- Flags: none

### CLI

```text
Parsed: 1 / (sqrt(x) - a)
Target: (sqrt(x) + a) / (x - a^2)
Strategy: rationalize
Steps (Aggressive Mode):
1. Rationalize denominator (diff squares)  [Rationalize Denominator]
   Before: 1 / (sqrt(x) - a)
   Cambio local: 1 / (sqrt(x) - a) -> (sqrt(x) + a) / (sqrt(x)^(2) + a^(2))
   After: (sqrt(x) + a) / (sqrt(x)^(2) + a^(2))
2. (u^y)^(1/y) = u  [Cancel Reciprocal Exponents]
   Before: (sqrt(x) + a) / (sqrt(x)^(2) + a^(2))
   Cambio local: sqrt(x)^(2) -> x
   After: (sqrt(x) + a) / (x - a^2)
   ℹ️ Requires: x > 0
Result: (sqrt(x) + a) / (x - a^(2))
ℹ️ Requires:
  • x ≥ 0
  • a^2 - x ≠ 0
```

### Web / JSON Steps

1. `Racionalizar el denominador`
   - before: `1/(sqrt(x) - a)`
   - after: `(sqrt(x) + a)/(sqrt(x)^2 - ((-a))^2)`
   - substeps:
     1. `Cambiar el signo para formar el conjugado`
     2. `Multiplicar numerador y denominador por ese conjugado`
     3. `En el denominador aparece una diferencia de cuadrados`
2. `Deshacer raíz y potencia`
   - before: `(sqrt(x) + a)/(sqrt(x)^2 - ((-a))^2)`
   - after: `(sqrt(x) + a)/(x - a^2)`
   - substeps:
     1. `El cuadrado deshace la raíz`
     2. `Reemplazar ese bloque en la expresión`

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
Strategy: simplify
Steps (Aggressive Mode):
1. Rationalize: multiply by conjugate  [Rationalize Linear Sqrt Denominator]
   Before: 1 / (sqrt(x) + a)
   After: (sqrt(x) - a) / (x - a^2)
Result: (sqrt(x) - a) / (x - a^(2))
ℹ️ Requires:
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

## rationalize_symbolic_linear_root_alt_var (rationalize)

- Source: `1/(sqrt(y)-a)`
- Target: `(sqrt(y)+a)/(y-a^2)`
- Result: `(sqrt(y) + a) / (y - a^2)`
- Web step count: `2`
- Web substep count: `5`
- Flags: none

### CLI

```text
Parsed: 1 / (sqrt(y) - a)
Target: (sqrt(y) + a) / (y - a^2)
Strategy: rationalize
Steps (Aggressive Mode):
1. Rationalize denominator (diff squares)  [Rationalize Denominator]
   Before: 1 / (sqrt(y) - a)
   Cambio local: 1 / (sqrt(y) - a) -> (sqrt(y) + a) / (sqrt(y)^(2) + a^(2))
   After: (sqrt(y) + a) / (sqrt(y)^(2) + a^(2))
2. (u^y)^(1/y) = u  [Cancel Reciprocal Exponents]
   Before: (sqrt(y) + a) / (sqrt(y)^(2) + a^(2))
   Cambio local: sqrt(y)^(2) -> y
   After: (sqrt(y) + a) / (y - a^2)
   ℹ️ Requires: y > 0
Result: (sqrt(y) + a) / (y - a^(2))
ℹ️ Requires:
  • a^2 - y ≠ 0
  • y ≥ 0
```

### Web / JSON Steps

1. `Racionalizar el denominador`
   - before: `1/(sqrt(y) - a)`
   - after: `(sqrt(y) + a)/(sqrt(y)^2 - ((-a))^2)`
   - substeps:
     1. `Cambiar el signo para formar el conjugado`
     2. `Multiplicar numerador y denominador por ese conjugado`
     3. `En el denominador aparece una diferencia de cuadrados`
2. `Deshacer raíz y potencia`
   - before: `(sqrt(y) + a)/(sqrt(y)^2 - ((-a))^2)`
   - after: `(sqrt(y) + a)/(y - a^2)`
   - substeps:
     1. `El cuadrado deshace la raíz`
     2. `Reemplazar ese bloque en la expresión`
