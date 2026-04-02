# Derive Didactic Audit

Generated from [derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv).

Command: `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture`

## Summary

- Derived cases audited: `179`
- Mean top-level step count: `1.06`
- Total web substeps: `297`

| id | family | web steps | web substeps | flags |
| --- | --- | ---: | ---: | --- |
| `combine_like_terms` | `simplify` | 1 | 1 | none |
| `collect_linear` | `collect` | 1 | 1 | none |
| `factor_difference_squares` | `factor` | 1 | 2 | none |
| `factor_perfect_square_trinomial` | `factor` | 1 | 2 | none |
| `factor_perfect_square_trinomial_symbolic` | `factor` | 1 | 2 | none |
| `factor_perfect_square_trinomial_minus` | `factor` | 1 | 2 | none |
| `factor_sophie_germain` | `factor` | 1 | 2 | none |
| `factor_alternating_cubic_vandermonde` | `factor` | 1 | 2 | none |
| `factor_difference_cubes` | `factor` | 1 | 2 | none |
| `factor_sum_cubes` | `factor` | 1 | 2 | none |
| `pythagorean_identity` | `simplify` | 1 | 2 | none |
| `pythagorean_factor_form_to_cos_sq` | `simplify` | 1 | 1 | none |
| `pythagorean_factor_form_to_sin_sq` | `simplify` | 1 | 1 | none |
| `pythagorean_factor_form_from_sin_sq` | `simplify` | 1 | 1 | none |
| `pythagorean_factor_form_from_cos_sq` | `simplify` | 1 | 1 | none |
| `inverse_tan_identity` | `simplify` | 1 | 2 | none |
| `cancel_fraction_difference_squares` | `simplify` | 1 | 2 | none |
| `cancel_fraction_difference_squares_mirror` | `simplify` | 1 | 2 | none |
| `cancel_fraction_difference_cubes` | `simplify` | 1 | 2 | none |
| `cancel_fraction_sum_cubes` | `simplify` | 1 | 2 | none |
| `cancel_fraction_perfect_square_plus` | `simplify` | 1 | 1 | none |
| `cancel_fraction_perfect_square_symbolic` | `simplify` | 1 | 1 | none |
| `cancel_fraction_perfect_square_minus_numeric` | `simplify` | 1 | 2 | none |
| `cancel_fraction_perfect_square_minus_symbolic` | `simplify` | 1 | 2 | none |
| `perfect_square_root_to_abs` | `simplify` | 1 | 2 | none |
| `combine_fraction_part_with_same_denominator` | `fraction_combine` | 1 | 2 | none |
| `expand_fraction_part_with_same_denominator` | `fraction_expand` | 1 | 1 | none |
| `combine_three_same_denominator_fractions` | `fraction_combine` | 1 | 2 | none |
| `combine_fraction_part_with_same_denominator_three_terms` | `fraction_combine` | 1 | 2 | none |
| `expand_fraction_part_with_same_denominator_three_terms` | `fraction_expand` | 1 | 1 | none |
| `combine_like_terms_with_zero` | `simplify` | 1 | 1 | none |
| `factor_common_factor_sum` | `factor` | 1 | 2 | none |
| `factor_common_factor_difference` | `factor` | 1 | 2 | none |
| `expand_common_factor_sum` | `expand` | 1 | 2 | none |
| `expand_common_factor_difference` | `expand` | 1 | 2 | none |
| `factor_common_factor_sum_three_terms` | `factor` | 1 | 2 | none |
| `factor_common_factor_difference_three_terms` | `factor` | 1 | 2 | none |
| `expand_common_factor_sum_three_terms` | `expand` | 1 | 2 | none |
| `expand_common_factor_difference_three_terms` | `expand` | 1 | 2 | none |
| `cancel_fraction_common_factor_numeric` | `simplify` | 1 | 2 | none |
| `cancel_fraction_monomial_common_factor` | `simplify` | 1 | 2 | none |
| `expand_binomial` | `expand` | 1 | 2 | none |
| `expand_symbolic_binomial` | `expand` | 1 | 2 | none |
| `expand_symbolic_binomial_minus` | `expand` | 1 | 2 | none |
| `expand_symbolic_binomial_cube` | `expand` | 1 | 2 | none |
| `expand_symbolic_binomial_cube_minus` | `expand` | 1 | 2 | none |
| `expand_sophie_germain` | `expand` | 1 | 2 | none |
| `expand_difference_cubes` | `expand` | 1 | 2 | none |
| `expand_sum_cubes` | `expand` | 1 | 2 | none |
| `expand_then_cancel_to_square` | `expand` | 3 | 2 | none |
| `expand_log_product` | `log_expand` | 1 | 1 | none |
| `expand_log_product_and_power` | `log_expand` | 1 | 1 | none |
| `expand_log_even_power_abs` | `log_expand` | 1 | 1 | none |
| `expand_log_product_preserve_powers` | `log_expand` | 1 | 1 | none |
| `expand_log_quotient` | `log_expand` | 1 | 1 | none |
| `expand_log_general_base` | `log_expand` | 1 | 1 | none |
| `expand_log_general_base_quotient` | `log_expand` | 1 | 1 | none |
| `expand_log_general_base_power` | `log_expand` | 1 | 1 | none |
| `contract_log_sum` | `log_contract` | 1 | 1 | none |
| `contract_log_difference` | `log_contract` | 1 | 1 | none |
| `contract_log_sum_with_powers` | `log_contract` | 1 | 1 | none |
| `contract_log_sum_with_scaled_powers` | `log_contract` | 1 | 1 | none |
| `contract_log_difference_with_scaled_powers` | `log_contract` | 1 | 1 | none |
| `contract_log_general_base_difference` | `log_contract` | 1 | 1 | none |
| `contract_log_general_base_sum_with_scaled_powers` | `log_contract` | 1 | 1 | none |
| `contract_log_general_base_difference_with_scaled_powers` | `log_contract` | 1 | 1 | none |
| `contract_log_even_power_abs` | `log_contract` | 1 | 1 | none |
| `contract_log_general_base_power` | `log_contract` | 1 | 1 | none |
| `contract_log_change_of_base_chain` | `log_contract` | 1 | 1 | none |
| `expand_log_change_of_base_chain` | `log_expand` | 1 | 1 | none |
| `contract_log_change_of_base_chain_three` | `log_contract` | 1 | 1 | none |
| `expand_log_change_of_base_chain_three` | `log_expand` | 1 | 1 | none |
| `expand_trig_double_sin` | `trig_expand` | 1 | 1 | none |
| `expand_trig_product_to_sum_sin_cos` | `trig_expand` | 1 | 1 | none |
| `expand_trig_product_to_sum_cos_sin` | `trig_expand` | 1 | 1 | none |
| `expand_trig_product_to_sum_cos_cos` | `trig_expand` | 1 | 1 | none |
| `expand_trig_product_to_sum_sin_sin` | `trig_expand` | 1 | 1 | none |
| `contract_trig_double_sin` | `trig_contract` | 1 | 1 | none |
| `expand_trig_after_simplify` | `trig_expand` | 2 | 2 | none |
| `expand_trig_tan_quotient` | `trig_expand` | 1 | 1 | none |
| `contract_trig_tan_quotient` | `trig_contract` | 1 | 1 | none |
| `expand_trig_sec_reciprocal` | `trig_expand` | 1 | 1 | none |
| `contract_trig_sec_reciprocal` | `trig_contract` | 1 | 1 | none |
| `expand_trig_csc_reciprocal` | `trig_expand` | 1 | 1 | none |
| `contract_trig_csc_reciprocal` | `trig_contract` | 1 | 1 | none |
| `expand_trig_cot_quotient` | `trig_expand` | 1 | 1 | none |
| `contract_trig_cot_quotient` | `trig_contract` | 1 | 1 | none |
| `expand_trig_double_cos_as_one_minus_sin_sq` | `trig_expand` | 1 | 1 | none |
| `expand_trig_double_cos_as_two_cos_sq_minus_one` | `trig_expand` | 1 | 1 | none |
| `contract_trig_double_cos_from_one_minus_sin_sq` | `trig_contract` | 1 | 1 | none |
| `contract_trig_double_cos_from_two_cos_sq_minus_one` | `trig_contract` | 1 | 1 | none |
| `contract_trig_sec_squared` | `trig_contract` | 1 | 1 | none |
| `contract_trig_csc_squared` | `trig_contract` | 1 | 1 | none |
| `expand_trig_sec_squared` | `trig_expand` | 1 | 1 | none |
| `expand_trig_csc_squared` | `trig_expand` | 1 | 1 | none |
| `expand_trig_half_angle_sin_squared` | `trig_expand` | 1 | 1 | none |
| `expand_trig_half_angle_cos_squared` | `trig_expand` | 1 | 1 | none |
| `contract_trig_half_angle_sin_squared` | `trig_contract` | 1 | 1 | none |
| `contract_trig_half_angle_cos_squared` | `trig_contract` | 1 | 1 | none |
| `contract_trig_sin_sum_special` | `trig_expand` | 1 | 1 | none |
| `contract_trig_sin_diff_special` | `trig_contract` | 1 | 1 | none |
| `expand_trig_sum_to_product_sin_sum_general` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sum_to_product_sin_diff_general` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sum_to_product_cos_sum_general` | `trig_expand` | 1 | 1 | none |
| `expand_trig_sum_to_product_cos_diff_general` | `trig_expand` | 1 | 1 | none |
| `contract_trig_cos_sum_special` | `trig_expand` | 1 | 1 | none |
| `contract_trig_cos_diff_special` | `trig_expand` | 1 | 1 | none |
| `contract_trig_half_angle_tangent` | `trig_contract` | 1 | 1 | none |
| `contract_trig_half_angle_tangent_alt` | `trig_contract` | 1 | 1 | none |
| `expand_trig_half_angle_tangent` | `trig_expand` | 1 | 1 | none |
| `expand_trig_half_angle_tangent_alt` | `trig_expand` | 1 | 1 | none |
| `rationalize_linear_root` | `rationalize` | 1 | 3 | none |
| `rationalize_then_cancel_to_zero` | `rationalize` | 2 | 3 | none |
| `radical_notable_quotient` | `rationalize` | 2 | 5 | none |
| `radical_notable_quotient_sqrt_input` | `rationalize` | 2 | 5 | none |
| `expand_fraction_simple` | `fraction_expand` | 1 | 1 | none |
| `expand_fraction_same_denominator_three_terms` | `fraction_expand` | 1 | 1 | none |
| `expand_fraction_with_term_cancellation` | `fraction_expand` | 1 | 2 | none |
| `nested_fraction_one_over_sum` | `nested_fraction` | 2 | 3 | none |
| `nested_fraction_fraction_over_sum` | `nested_fraction` | 2 | 3 | none |
| `combine_same_denominator_fraction_sum` | `fraction_combine` | 1 | 2 | none |
| `combine_general_fraction_sum` | `fraction_combine` | 1 | 2 | none |
| `combine_same_denominator_fraction_difference` | `fraction_combine` | 1 | 2 | none |
| `combine_general_fraction_difference` | `fraction_combine` | 1 | 2 | none |
| `combine_term_and_fraction_subtraction` | `fraction_combine` | 1 | 2 | none |
| `split_fraction_into_whole_plus_remainder` | `fraction_decompose` | 1 | 2 | none |
| `split_telescoping_fraction_consecutive` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_gap_three` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_consecutive` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_gap_three` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_negative_consecutive` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_negative_consecutive` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_negative_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_negative_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_affine_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_affine_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_affine_shifted_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_affine_shifted_gap_two` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_affine_coeff_three_gap_three` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_affine_coeff_three_gap_three` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_affine_coeff_three_shifted_gap_three` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_affine_coeff_three_shifted_gap_three` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_affine_symbolic_coeff_gap_three` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_affine_symbolic_coeff_gap_three` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_affine_symbolic_coeff_shifted_gap_three` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_affine_symbolic_coeff_shifted_gap_three` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_symbolic_shift_gap` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_symbolic_shift_gap` | `telescoping_fraction` | 1 | 2 | none |
| `split_telescoping_fraction_affine_symbolic_shift_gap` | `telescoping_fraction` | 1 | 2 | none |
| `combine_telescoping_fraction_affine_symbolic_shift_gap` | `telescoping_fraction` | 1 | 2 | none |
| `combine_whole_plus_remainder_into_fraction` | `fraction_combine` | 1 | 2 | none |
| `small_polynomial_product` | `polynomial_product` | 1 | 3 | none |
| `factor_geometric_difference_power_6` | `factor` | 1 | 2 | none |
| `merge_same_base_fractional_powers` | `power_merge` | 1 | 1 | none |
| `merge_mixed_root_and_power` | `power_merge` | 2 | 2 | none |
| `log_sum_difference_cancels_to_zero` | `simplify` | 1 | 1 | none |
| `expand_odd_half_power` | `radical_power` | 1 | 2 | none |
| `expand_odd_half_power_after_simplify` | `radical_power` | 1 | 2 | none |
| `factor_out_with_division` | `conditional_factor` | 1 | 1 | none |
| `consecutive_factorial_ratio` | `simplify` | 1 | 2 | none |
| `inverse_tan_identity_cancels_to_zero` | `simplify` | 1 | 3 | none |
| `contract_trig_cos_diff_sin_diff_quotient` | `trig_contract` | 3 | 4 | none |
| `reciprocal_trig_product_to_one` | `simplify` | 1 | 1 | none |
| `sec_tan_pythagorean_to_one` | `simplify` | 1 | 1 | none |
| `csc_cot_pythagorean_to_one` | `simplify` | 1 | 1 | none |
| `factor_symbolic_binomial_cube` | `factor` | 1 | 2 | none |
| `factor_symbolic_binomial_cube_minus` | `factor` | 1 | 2 | none |
| `integrate_prep_morrie_basic` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_morrie_scaled` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_dirichlet_basic` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_dirichlet_longer` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_dirichlet_scaled` | `integrate_prep` | 1 | 2 | none |
| `integrate_prep_dirichlet_scaled_longer` | `integrate_prep` | 1 | 2 | none |
| `finite_telescoping_product_basic` | `finite_telescoping` | 1 | 3 | none |
| `finite_telescoping_product_shifted` | `finite_telescoping` | 1 | 3 | none |
| `finite_telescoping_sum_basic` | `finite_telescoping` | 1 | 3 | none |
| `finite_telescoping_sum_shifted` | `finite_telescoping` | 1 | 3 | none |

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
- Web substep count: `2`
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
     2. `Aquí a = a y b = b`

## factor_perfect_square_trinomial_minus (factor)

- Source: `a^2 - 2*a*b + b^2`
- Target: `(a - b)^2`
- Result: `(a - b)^2`
- Web step count: `1`
- Web substep count: `2`
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
     2. `Aquí a = a y b = b`

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
- Web substep count: `2`
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
     2. `Aplicar a^3 - b^3 = (a - b)(a^2 + ab + b^2)`

## factor_sum_cubes (factor)

- Source: `a^3+b^3`
- Target: `(a+b)*(a^2-a*b+b^2)`
- Result: `(a + b) * (a^2 + b^2 - a * b)`
- Web step count: `1`
- Web substep count: `2`
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
     2. `Aplicar a^3 + b^3 = (a + b)(a^2 - ab + b^2)`

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
- Web substep count: `1`
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
   - substeps:
     1. `Usar 1 - sin²(u) = cos²(u)`

## pythagorean_factor_form_to_sin_sq (simplify)

- Source: `1 - cos(x)^2`
- Target: `sin(x)^2`
- Result: `sin(x)^2`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 1 - cos(x)^2
Target: sin(x)^2
Strategy: simplify
Steps (Aggressive Mode):
1. 1 - cos²(x) = sin²(x)  [Pythagorean Factor Form]
   Before: 1 - cos(x)^(2)
   Cambio local: 1 - cos(x)^(2) -> sin(x)^(2)
   After: sin(x)^2
Result: sin(x)^(2)
```

### Web / JSON Steps

1. `Aplicar identidad pitagórica`
   - before: `1 - cos(x)^2`
   - after: `sin(x)^2`
   - substeps:
     1. `Usar 1 - cos²(u) = sin²(u)`

## pythagorean_factor_form_from_sin_sq (simplify)

- Source: `sin(x)^2`
- Target: `1-cos(x)^2`
- Result: `1 - cos(x)^2`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar 1 - cos²(u) = sin²(u)`

## pythagorean_factor_form_from_cos_sq (simplify)

- Source: `cos(x)^2`
- Target: `1-sin(x)^2`
- Result: `1 - sin(x)^2`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x)^2
Target: 1 - sin(x)^2
Strategy: simplify
Steps (Aggressive Mode):
1. 1 - sin²(x) = cos²(x)  [Pythagorean Factor Form]
   Before: cos(x)^(2)
   Cambio local: cos(x)^(2) -> 1 - sin(x)^(2)
   After: 1 - sin(x)^2
Result: 1 - sin(x)^(2)
```

### Web / JSON Steps

1. `Aplicar identidad pitagórica`
   - before: `cos(x)^2`
   - after: `1 - sin(x)^2`
   - substeps:
     1. `Usar 1 - sin²(u) = cos²(u)`

## inverse_tan_identity (simplify)

- Source: `arctan(3)+arctan(1/3)`
- Target: `pi/2`
- Result: `pi / 2`
- Web step count: `1`
- Web substep count: `2`
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
     2. `Esa pareja vale pi/2`

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
     1. `Reescribir el numerador como diferencia de cuadrados`
     2. `Ahora se cancela el factor a - b`

## cancel_fraction_difference_squares_mirror (simplify)

- Source: `(a^2-b^2)/(a+b)`
- Target: `a-b`
- Result: `a - b`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (a^2 - b^2) / (a + b)
Target: a - b
Strategy: simplify
Steps (Aggressive Mode):
1. Cancel common factor  [Pre-order Difference of Squares Cancel]
   Before: (a + b) * (a - b) / (a + b)
   Cambio local: (a + b) * (a - b) / (a + b) -> a - b
   After: a - b
Result: a - b
```

### Web / JSON Steps

1. `Factorizar una diferencia de cuadrados y cancelar`
   - before: `((a + b) · (a - b))/(a + b)`
   - after: `a - b`
   - substeps:
     1. `Reescribir el numerador como diferencia de cuadrados`
     2. `Ahora se cancela el factor a + b`

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

## cancel_fraction_perfect_square_plus (simplify)

- Source: `(x^2+2*x+1)/(x+1)`
- Target: `x+1`
- Result: `x + 1`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (x^2 + 2 * x + 1) / (x + 1)
Target: x + 1
Strategy: simplify
Steps (Aggressive Mode):
1. Cancel common factor  [Simplify Nested Fraction]
   Before: (x + 1) * (x + 1) / (x + 1)
   Cambio local: (x + 1) * (x + 1) / (x + 1) -> x + 1
   After: x + 1
Result: x + 1
```

### Web / JSON Steps

1. `Cancelar factores en una fracción`
   - before: `((x + 1) · (x + 1))/(x + 1)`
   - after: `x + 1`
   - substeps:
     1. `Si x + 1 aparece dos veces arriba y una abajo, queda una sola copia`

## cancel_fraction_perfect_square_symbolic (simplify)

- Source: `(a^2+2*a*b+b^2)/(a+b)`
- Target: `a+b`
- Result: `a + b`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a^2 + b^2 + 2 * a * b) / (a + b)
Target: a + b
Strategy: simplify
Steps (Aggressive Mode):
1. Cancel common factor  [Simplify Nested Fraction]
   Before: (a + b) * (a + b) / (a + b)
   Cambio local: (a + b) * (a + b) / (a + b) -> a + b
   After: a + b
Result: a + b
```

### Web / JSON Steps

1. `Cancelar factores en una fracción`
   - before: `((a + b) · (a + b))/(a + b)`
   - after: `a + b`
   - substeps:
     1. `Si a + b aparece dos veces arriba y una abajo, queda una sola copia`

## cancel_fraction_perfect_square_minus_numeric (simplify)

- Source: `(x^2-2*x+1)/(x-1)`
- Target: `x-1`
- Result: `x - 1`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: (x^2 - 2 * x + 1) / (x - 1)
Target: x - 1
Strategy: simplify
Steps (Aggressive Mode):
1. Cancel common factor  [Pre-order Perfect Square Minus Cancel]
   Before: (x^(2) - 2 * x + 1) / (x - 1)
   Cambio local: (x^(2) - 2 * x + 1) / (x - 1) -> x - 1
   After: x - 1
Result: x - 1
```

### Web / JSON Steps

1. `Pre-order Perfect Square Minus Cancel`
   - before: `(x^2 - 2 · x + 1)/(x - 1)`
   - after: `x - 1`
   - substeps:
     1. `Reconocer que el numerador es un cuadrado perfecto`
     2. `Si (x - 1)^2 está dividido entre x - 1, queda una sola copia`

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

## combine_fraction_part_with_same_denominator (fraction_combine)

- Source: `1 + a/d + b/d`
- Target: `1 + (a+b)/d`
- Result: `(a + b) / d + 1`
- Web step count: `1`
- Web substep count: `2`
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
   - substeps:
     1. `Como el denominador ya es el mismo, se mantiene igual`
     2. `Basta sumar los numeradores`

## expand_fraction_part_with_same_denominator (fraction_expand)

- Source: `1 + (a+b)/d`
- Target: `1 + a/d + b/d`
- Result: `a / d + b / d + 1`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a + b) / d + 1
Target: a / d + b / d + 1
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Distribute Division]
   Before: (a + b) / d + 1
   After: a / d + b / d + 1
Result: a / d + b / d + 1
ℹ️ Requires:
  • d ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(a + b)/d + 1`
   - after: `a/d + b/d + 1`
   - substeps:
     1. `Usar (a + b) / d = a/d + b/d`

## combine_three_same_denominator_fractions (fraction_combine)

- Source: `a/d + b/d + c/d`
- Target: `(a+b+c)/d`
- Result: `(a + b + c) / d`
- Web step count: `1`
- Web substep count: `2`
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
   - substeps:
     1. `Como el denominador ya es el mismo, se mantiene igual`
     2. `Basta sumar los numeradores`

## combine_fraction_part_with_same_denominator_three_terms (fraction_combine)

- Source: `1 + a/d + b/d + c/d`
- Target: `1 + (a+b+c)/d`
- Result: `(a + b + c) / d + 1`
- Web step count: `1`
- Web substep count: `2`
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
   - substeps:
     1. `Como el denominador ya es el mismo, se mantiene igual`
     2. `Basta sumar los numeradores`

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

## combine_like_terms_with_zero (simplify)

- Source: `2*x + 3*x + 0`
- Target: `5*x`
- Result: `5 * x`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * x + 3 * x + 0
Target: 5 * x
Strategy: simplify
Steps (Aggressive Mode):
1. Combine like terms  [Combine Like Terms]
   Before: 2 * x + 3 * x
   Cambio local: 2 * x + 3 * x -> 5 * x
   After: 5 * x
Result: 5 * x
```

### Web / JSON Steps

1. `Agrupar términos semejantes`
   - before: `2 · x + 3 · x`
   - after: `5 · x`
   - substeps:
     1. `Sumar los coeficientes que acompañan a x`

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

## factor_common_factor_difference (factor)

- Source: `a*b - a*c`
- Target: `a*(b-c)`
- Result: `a * (b - c)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * b - a * c
Target: a * (b - c)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorization]
   Before: a * b - a * c
   Cambio local: a * b - a * c -> a * (b - c)
   After: a * (b - c)
Result: a * (b - c)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a · b - a · c`
   - after: `a · (b - c)`
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

## factor_common_factor_difference_three_terms (factor)

- Source: `a*x - b*x - c*x`
- Target: `x*(a-b-c)`
- Result: `x * (a - b - c)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: a * x - b * x - c * x
Target: x * (a - b - c)
Strategy: factor
Steps (Aggressive Mode):
1. Factorization  [Factorization]
   Before: a * x - b * x - c * x
   Cambio local: a * x - b * x - c * x -> x * (a - b - c)
   After: x * (a - b - c)
Result: x * (a - b - c)
```

### Web / JSON Steps

1. `Factorizar`
   - before: `a · x - b · x - c · x`
   - after: `x · (a - b - c)`
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
     2. `Simplificar la fracción restante`

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
     2. `Simplificar la fracción restante`

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
     2. `Sustituir a = 1 y b = x`

## expand_symbolic_binomial (expand)

- Source: `(a + b)^2`
- Target: `a^2 + 2*a*b + b^2`
- Result: `a^2 + b^2 + 2 * a * b`
- Web step count: `1`
- Web substep count: `2`
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
     2. `Sustituir a = a y b = b`

## expand_symbolic_binomial_minus (expand)

- Source: `(a - b)^2`
- Target: `a^2 - 2*a*b + b^2`
- Result: `a^2 + b^2 - 2 * a * b`
- Web step count: `1`
- Web substep count: `2`
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
     2. `Sustituir a = a y b = b`

## expand_symbolic_binomial_cube (expand)

- Source: `(a + b)^3`
- Target: `a^3 + 3*a^2*b + 3*a*b^2 + b^3`
- Result: `a^3 + b^3 + 3 * a * b^2 + 3 * b * a^2`
- Web step count: `1`
- Web substep count: `2`
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
     2. `Sustituir a = a y b = b`

## expand_symbolic_binomial_cube_minus (expand)

- Source: `(a - b)^3`
- Target: `a^3 - 3*a^2*b + 3*a*b^2 - b^3`
- Result: `a^3 + 3 * a * b^2 - 3 * b * a^2 - b^3`
- Web step count: `1`
- Web substep count: `2`
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
     2. `Sustituir a = a y b = b`

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
     2. `Sustituir a = x y b = y`

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
- Web step count: `3`
- Web substep count: `2`
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
   After: b^(2) + 2 * a * b + a^(2) - a^(2) - 2 * a * b
2. Cancel opposite terms  [Combine Like Terms]
   Before: a^(2) + b^(2) + 2 * a * b - a^(2) - 2 * a * b
   Cambio local: a^(2) - a^(2) -> 0
   After: b^(2) + 2 * a * b - 2 * a * b
3. Cancel opposite terms  [Combine Like Terms]
   Before: b^(2) + 2 * a * b - 2 * a * b
   Cambio local: 2 * a * b - 2 * a * b -> 0
   After: b^2
Result: b^(2)
```

### Web / JSON Steps

1. `Expandir binomio`
   - before: `((a + b))^2 - a^2 - 2 · a · b`
   - after: `b^2 + 2 · a · b + a^2 - a^2 - 2 · a · b`
   - substeps:
     1. `Usar (a + b)^2 = a^2 + 2ab + b^2`
     2. `Sustituir a = a y b = b`
2. `Agrupar términos semejantes`
   - before: `a^2 + b^2 + 2 · a · b - a^2 - 2 · a · b`
   - after: `b^2 + 2 · a · b - 2 · a · b`
   - substeps: none
3. `Agrupar términos semejantes`
   - before: `b^2 + 2 · a · b - 2 · a · b`
   - after: `b^2`
   - substeps: none

## expand_log_product (log_expand)

- Source: `ln(x*y)`
- Target: `ln(x) + ln(y)`
- Result: `ln(x) + ln(y)`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar que el logaritmo de un producto se separa en una suma`

## expand_log_product_and_power (log_expand)

- Source: `ln(x^2*y)`
- Target: `ln(y) + 2*ln(abs(x))`
- Result: `ln(y) + 2 * ln(|x|)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: ln(y * x^2)
Target: ln(y) + 2 * ln(|x|)
Strategy: expand_log
Steps (Aggressive Mode):
1. Expand a logarithm into a sum of logarithms  [expand_log]
   Before: ln(y * x^(2))
   Cambio local: ln(y * x^(2)) -> ln(y) + 2 * ln(|x|)
   After: ln(y) + 2 * ln(|x|)
Result: ln(y) + 2 * ln(|x|)
ℹ️ Requires:
  • y > 0
  • |x| > 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln(y · x^2)`
   - after: `ln(y) + 2 · ln(|x|)`
   - substeps:
     1. `Usar que el logaritmo de un producto se separa en una suma`

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

## expand_log_product_preserve_powers (log_expand)

- Source: `ln(x^3*y^2)`
- Target: `ln(x^3) + ln(y^2)`
- Result: `ln(x^3) + ln(y^2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: ln(x^3 * y^2)
Target: ln(x^3) + ln(y^2)
Strategy: expand_log
Steps (Aggressive Mode):
1. Expand a logarithm into a sum of logarithms  [expand_log]
   Before: ln(x^(3) * y^(2))
   Cambio local: ln(x^(3) * y^(2)) -> ln(x^(3)) + ln(y^(2))
   After: ln(x^3) + ln(y^2)
Result: ln(x^(3)) + ln(y^(2))
ℹ️ Requires:
  • y ≠ 0
  • x^3 > 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln(x^3 · y^2)`
   - after: `ln(x^3) + ln(y^2)`
   - substeps:
     1. `Usar que el logaritmo de un producto se separa en una suma`

## expand_log_quotient (log_expand)

- Source: `ln(x/y)`
- Target: `ln(x) - ln(y)`
- Result: `ln(x) - ln(y)`
- Web step count: `1`
- Web substep count: `1`
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
  • y > 0
  • x > 0
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `ln(x/y)`
   - after: `ln(x) - ln(y)`
   - substeps:
     1. `Usar que el logaritmo de un cociente se separa en una resta`

## expand_log_general_base (log_expand)

- Source: `log(2, x*y)`
- Target: `log(2, x) + log(2, y)`
- Result: `log(2, x) + log(2, y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: log(2, x * y)
Target: log(2, x) + log(2, y)
Strategy: expand_log
Steps (Aggressive Mode):
1. Expand a logarithm into a sum of logarithms  [expand_log]
   Before: log(2, x * y)
   Cambio local: log(2, x * y) -> log(2, x) + log(2, y)
   After: log(2, x) + log(2, y)
Result: log(2, x) + log(2, y)
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `log_2(x · y)`
   - after: `log_2(x) + log_2(y)`
   - substeps:
     1. `Usar que el logaritmo de un producto se separa en una suma`

## expand_log_general_base_quotient (log_expand)

- Source: `log(2, x/y)`
- Target: `log(2, x) - log(2, y)`
- Result: `log(2, x) - log(2, y)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: log(2, x / y)
Target: log(2, x) - log(2, y)
Strategy: expand_log
Steps (Aggressive Mode):
1. Expand a logarithm into a sum of logarithms  [expand_log]
   Before: log(2, x / y)
   Cambio local: log(2, x / y) -> log(2, x) - log(2, y)
   After: log(2, x) - log(2, y)
Result: log(2, x) - log(2, y)
```

### Web / JSON Steps

1. `Expandir logaritmos`
   - before: `log_2(x/y)`
   - after: `log_2(x) - log_2(y)`
   - substeps:
     1. `Usar que el logaritmo de un cociente se separa en una resta`

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
- Web substep count: `1`
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
   - substeps:
     1. `Usar que una suma de logaritmos se puede reunir en un producto`

## contract_log_difference (log_contract)

- Source: `ln(x) - ln(y)`
- Target: `ln(x/y)`
- Result: `ln(x / y)`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar que una resta de logaritmos se puede reunir en un cociente`

## contract_log_sum_with_powers (log_contract)

- Source: `ln(x^3) + ln(y^2)`
- Target: `ln(x^3*y^2)`
- Result: `ln(x^3 * y^2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: ln(x^3) + ln(y^2)
Target: ln(x^3 * y^2)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Log Contraction]
   Before: ln(x^(3)) + ln(y^(2))
   Cambio local: ln(x^(3)) + ln(y^(2)) -> ln(x^(3) * y^(2))
   After: ln(x^3 * y^2)
Result: ln(x^(3) * y^(2))
ℹ️ Requires:
  • x^3 * y^2 > 0
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `ln(x^3) + ln(y^2)`
   - after: `ln(x^3 · y^2)`
   - substeps:
     1. `Usar que una suma de logaritmos se puede reunir en un producto`

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
  • x^3 / y^2 > 0
  • y ≠ 0
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
- Web substep count: `1`
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
   - substeps:
     1. `Usar que una resta de logaritmos se puede reunir en un cociente`

## contract_log_general_base_sum_with_scaled_powers (log_contract)

- Source: `3*log(2, x) + 2*log(2, y)`
- Target: `log(2, x^3*y^2)`
- Result: `log(2, x^3 * y^2)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: 2 * log(2, y) + 3 * log(2, x)
Target: log(2, x^3 * y^2)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Log Contraction]
   Before: 2 * log(2, y) + 3 * log(2, x)
   Cambio local: 2 * log(2, y) + 3 * log(2, x) -> log(2, x^(3) * y^(2))
   After: log(2, x^3 * y^2)
Result: log(2, x^(3) * y^(2))
```

### Web / JSON Steps

1. `Contraer logaritmos`
   - before: `2 · log_2(y) + 3 · log_2(x)`
   - after: `log_2(x^3 · y^2)`
   - substeps:
     1. `Meter los coeficientes dentro de los logaritmos como exponentes`

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
- Web substep count: `1`
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

1. `Contraer cadena de logaritmos`
   - before: `log_a(c) · log_b(a)`
   - after: `log_b(c)`
   - substeps:
     1. `Usar log_b(a) · log_a(c) = log_b(c)`

## expand_log_change_of_base_chain (log_expand)

- Source: `log(b,c)`
- Target: `log(b,a)*log(a,c)`
- Result: `log(a, c) * log(b, a)`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar log_b(c) = log_a(c) · log_b(a)`

## contract_log_change_of_base_chain_three (log_contract)

- Source: `log(a,b)*log(b,c)*log(c,d)`
- Target: `log(a,d)`
- Result: `log(a, d)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: log(a, b) * log(b, c) * log(c, d)
Target: log(a, d)
Strategy: contract logs
Steps (Aggressive Mode):
1. Combine logarithms into a single logarithm  [Log Contraction]
   Before: log(a, b) * log(b, c) * log(c, d)
   Cambio local: log(a, b) * log(b, c) * log(c, d) -> log(a, d)
   After: log(a, d)
Result: log(a, d)
```

### Web / JSON Steps

1. `Contraer cadena de logaritmos`
   - before: `log_a(b) · log_b(c) · log_c(d)`
   - after: `log_a(d)`
   - substeps:
     1. `Encadenar los cambios de base intermedios`

## expand_log_change_of_base_chain_three (log_expand)

- Source: `log(a,d)`
- Target: `log(a,b)*log(b,c)*log(c,d)`
- Result: `log(a, b) * log(b, c) * log(c, d)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: log(a, d)
Target: log(a, b) * log(b, c) * log(c, d)
Strategy: simplify
Steps (Aggressive Mode):
1. Simplify the expression  [Simplify]
   Before: log(a, d)
   Cambio local: log(a, d) -> log(a, b) * log(b, c) * log(c, d)
   After: log(a, b) * log(b, c) * log(c, d)
Result: log(a, b) * log(b, c) * log(c, d)
```

### Web / JSON Steps

1. `Expandir cambio de base`
   - before: `log_a(d)`
   - after: `log_a(b) · log_b(c) · log_c(d)`
   - substeps:
     1. `Desplegar un logaritmo en una cadena de cambios de base`

## expand_trig_double_sin (trig_expand)

- Source: `sin(2*x)`
- Target: `2*sin(x)*cos(x)`
- Result: `2 * sin(x) * cos(x)`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar la identidad de ángulo doble`

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
- Web substep count: `1`
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
   - substeps:
     1. `Usar la identidad de ángulo doble`

## expand_trig_after_simplify (trig_expand)

- Source: `sin(x + x)`
- Target: `2*sin(x)*cos(x)`
- Result: `2 * sin(x) * cos(x)`
- Web step count: `2`
- Web substep count: `2`
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
   - substeps:
     1. `Usar la identidad de ángulo doble`

## expand_trig_tan_quotient (trig_expand)

- Source: `tan(2*x)`
- Target: `(sin(2*x))/(cos(2*x))`
- Result: `sin(2 * x) / cos(2 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: tan(2 * x)
Target: sin(2 * x) / cos(2 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand tangent to sine over cosine  [Trig Expansion]
   Before: tan(2 * x)
   Cambio local: tan(2 * x) -> sin(2 * x) / cos(2 * x)
   After: sin(2 * x) / cos(2 * x)
Result: sin(2 * x) / cos(2 * x)
ℹ️ Requires:
  • cos(2 * x) ≠ 0
```

### Web / JSON Steps

1. `Expandir una identidad trigonométrica`
   - before: `tan(2 · x)`
   - after: `sin(2 · x)/cos(2 · x)`
   - substeps:
     1. `Usar tan(u) = sin(u) / cos(u)`

## contract_trig_tan_quotient (trig_contract)

- Source: `(sin(2*x))/(cos(2*x))`
- Target: `tan(2*x)`
- Result: `tan(2 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(2 * x) / cos(2 * x)
Target: tan(2 * x)
Strategy: contract trig
Steps (Aggressive Mode):
1. sin(x)/cos(x) → tan(x)  [Trig Quotient]
   Before: sin(2 * x) / cos(2 * x)
   Cambio local: sin(2 * x) / cos(2 * x) -> tan(2 * x)
   After: tan(2 * x)
Result: tan(2 * x)
```

### Web / JSON Steps

1. `Convertir un cociente trigonométrico en tangente`
   - before: `sin(2 · x)/cos(2 · x)`
   - after: `tan(2 · x)`
   - substeps:
     1. `Reconocer el patrón sin(u) / cos(u) = tan(u)`

## expand_trig_sec_reciprocal (trig_expand)

- Source: `sec(x)`
- Target: `1/cos(x)`
- Result: `1 / cos(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sec(x)
Target: 1 / cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand sec(u) as 1 / cos(u)  [Reciprocal Trig Identity]
   Before: sec(x)
   Cambio local: sec(x) -> 1 / cos(x)
   After: 1 / cos(x)
Result: 1 / cos(x)
ℹ️ Requires:
  • cos(x) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad trigonométrica recíproca`
   - before: `sec(x)`
   - after: `1/cos(x)`
   - substeps:
     1. `Usar sec(u) = 1 / cos(u)`

## contract_trig_sec_reciprocal (trig_contract)

- Source: `1/cos(x)`
- Target: `sec(x)`
- Result: `sec(x)`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar 1 / cos(u) = sec(u)`

## expand_trig_csc_reciprocal (trig_expand)

- Source: `csc(x)`
- Target: `1/sin(x)`
- Result: `1 / sin(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: csc(x)
Target: 1 / sin(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand csc(u) as 1 / sin(u)  [Reciprocal Trig Identity]
   Before: csc(x)
   Cambio local: csc(x) -> 1 / sin(x)
   After: 1 / sin(x)
Result: 1 / sin(x)
ℹ️ Requires:
  • sin(x) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad trigonométrica recíproca`
   - before: `csc(x)`
   - after: `1/sin(x)`
   - substeps:
     1. `Usar csc(u) = 1 / sin(u)`

## contract_trig_csc_reciprocal (trig_contract)

- Source: `1/sin(x)`
- Target: `csc(x)`
- Result: `csc(x)`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar 1 / sin(u) = csc(u)`

## expand_trig_cot_quotient (trig_expand)

- Source: `cot(x)`
- Target: `cos(x)/sin(x)`
- Result: `cos(x) / sin(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cot(x)
Target: cos(x) / sin(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand cot(u) as cos(u) / sin(u)  [Reciprocal Trig Identity]
   Before: cot(x)
   Cambio local: cot(x) -> cos(x) / sin(x)
   After: cos(x) / sin(x)
Result: cos(x) / sin(x)
ℹ️ Requires:
  • sin(x) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad trigonométrica recíproca`
   - before: `cot(x)`
   - after: `cos(x)/sin(x)`
   - substeps:
     1. `Usar cot(u) = cos(u) / sin(u)`

## contract_trig_cot_quotient (trig_contract)

- Source: `cos(x)/sin(x)`
- Target: `cot(x)`
- Result: `cot(x)`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar cos(u) / sin(u) = cot(u)`

## expand_trig_double_cos_as_one_minus_sin_sq (trig_expand)

- Source: `cos(2*x)`
- Target: `1 - 2*sin(x)^2`
- Result: `1 - 2 * sin(x)^2`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar la identidad de ángulo doble`

## expand_trig_double_cos_as_two_cos_sq_minus_one (trig_expand)

- Source: `cos(2*x)`
- Target: `2*cos(x)^2 - 1`
- Result: `2 * cos(x)^2 - 1`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar la identidad de ángulo doble`

## contract_trig_double_cos_from_one_minus_sin_sq (trig_contract)

- Source: `1 - 2*sin(x)^2`
- Target: `cos(2*x)`
- Result: `cos(2 * x)`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Reconocer el patrón 1 - 2 · sin(u)^2 = cos(2u)`

## contract_trig_double_cos_from_two_cos_sq_minus_one (trig_contract)

- Source: `2*cos(x)^2 - 1`
- Target: `cos(2*x)`
- Result: `cos(2 * x)`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar la identidad de ángulo doble`

## contract_trig_sec_squared (trig_contract)

- Source: `1 + tan(x)^2`
- Target: `sec(x)^2`
- Result: `sec(x)^2`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar 1 + tan²(u) = sec²(u)`

## contract_trig_csc_squared (trig_contract)

- Source: `1 + cot(x)^2`
- Target: `csc(x)^2`
- Result: `csc(x)^2`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar 1 + cot²(u) = csc²(u)`

## expand_trig_sec_squared (trig_expand)

- Source: `sec(x)^2`
- Target: `1 + tan(x)^2`
- Result: `tan(x)^2 + 1`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar sec²(u) = 1 + tan²(u)`

## expand_trig_csc_squared (trig_expand)

- Source: `csc(x)^2`
- Target: `1 + cot(x)^2`
- Result: `cot(x)^2 + 1`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar csc²(u) = 1 + cot²(u)`

## expand_trig_half_angle_sin_squared (trig_expand)

- Source: `sin(x)^2`
- Target: `(1-cos(2*x))/2`
- Result: `(1 - cos(2 * x)) / 2`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar sin²(u) = (1 - cos(2u)) / 2`

## expand_trig_half_angle_cos_squared (trig_expand)

- Source: `cos(x)^2`
- Target: `(1+cos(2*x))/2`
- Result: `(cos(2 * x) + 1) / 2`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar cos²(u) = (1 + cos(2u)) / 2`

## contract_trig_half_angle_sin_squared (trig_contract)

- Source: `(1-cos(2*x))/2`
- Target: `sin(x)^2`
- Result: `sin(x)^2`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar (1 - cos(2u)) / 2 = sin²(u)`

## contract_trig_half_angle_cos_squared (trig_contract)

- Source: `(1+cos(2*x))/2`
- Target: `cos(x)^2`
- Result: `cos(x)^2`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar (1 + cos(2u)) / 2 = cos²(u)`

## contract_trig_sin_sum_special (trig_expand)

- Source: `sin(3*x)+sin(x)`
- Target: `2*sin(2*x)*cos(x)`
- Result: `2 * sin(2 * x) * cos(x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: sin(x) + sin(3 * x)
Target: 2 * sin(2 * x) * cos(x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand sine sum to product  [Sum-to-Product Identity]
   Before: sin(x) + sin(3 * x)
   Cambio local: sin(x) + sin(3 * x) -> 2 * sin(2 * x) * cos(x)
   After: 2 * sin(2 * x) * cos(x)
Result: 2 * sin(2 * x) * cos(x)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `sin(x) + sin(3 · x)`
   - after: `2 · sin(2 · x) · cos(x)`
   - substeps:
     1. `Usar sin(A) + sin(B) = 2 · sin((A+B)/2) · cos((A-B)/2)`

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

## contract_trig_cos_sum_special (trig_expand)

- Source: `cos(3*x)+cos(x)`
- Target: `2*cos(2*x)*cos(x)`
- Result: `2 * cos(x) * cos(2 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x) + cos(3 * x)
Target: 2 * cos(x) * cos(2 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand cosine sum to product  [Sum-to-Product Identity]
   Before: cos(x) + cos(3 * x)
   Cambio local: cos(x) + cos(3 * x) -> 2 * cos(x) * cos(2 * x)
   After: 2 * cos(x) * cos(2 * x)
Result: 2 * cos(x) * cos(2 * x)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `cos(x) + cos(3 · x)`
   - after: `2 · cos(x) · cos(2 · x)`
   - substeps:
     1. `Usar cos(A) + cos(B) = 2 · cos((A+B)/2) · cos((A-B)/2)`

## contract_trig_cos_diff_special (trig_expand)

- Source: `cos(x)-cos(3*x)`
- Target: `2*sin(2*x)*sin(x)`
- Result: `2 * sin(x) * sin(2 * x)`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: cos(x) - cos(3 * x)
Target: 2 * sin(x) * sin(2 * x)
Strategy: expand trig
Steps (Aggressive Mode):
1. Expand cosine difference to product  [Sum-to-Product Identity]
   Before: cos(x) - cos(3 * x)
   Cambio local: cos(x) - cos(3 * x) -> 2 * sin(x) * sin(2 * x)
   After: 2 * sin(x) * sin(2 * x)
Result: 2 * sin(x) * sin(2 * x)
```

### Web / JSON Steps

1. `Aplicar suma a producto`
   - before: `cos(x) - cos(3 · x)`
   - after: `2 · sin(x) · sin(2 · x)`
   - substeps:
     1. `Usar cos(A) - cos(B) = -2 · sin((A+B)/2) · sin((A-B)/2)`

## contract_trig_half_angle_tangent (trig_contract)

- Source: `(1-cos(2*x))/sin(2*x)`
- Target: `tan(x)`
- Result: `tan(x)`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar (1 - cos(2u)) / sin(2u) = tan(u)`

## contract_trig_half_angle_tangent_alt (trig_contract)

- Source: `sin(2*x)/(1+cos(2*x))`
- Target: `tan(x)`
- Result: `tan(x)`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar sin(2u) / (1 + cos(2u)) = tan(u)`

## expand_trig_half_angle_tangent (trig_expand)

- Source: `tan(x)`
- Target: `(1-cos(2*x))/sin(2*x)`
- Result: `(1 - cos(2 * x)) / sin(2 * x)`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar (1 - cos(2u)) / sin(2u) = tan(u)`

## expand_trig_half_angle_tangent_alt (trig_expand)

- Source: `tan(x)`
- Target: `sin(2*x)/(1+cos(2*x))`
- Result: `sin(2 * x) / (cos(2 * x) + 1)`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar sin(2u) / (1 + cos(2u)) = tan(u)`

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
  • x ≥ 0
  • x - 1 ≠ 0
```

### Web / JSON Steps

1. `Racionalizar el denominador`
   - before: `1/(sqrt(x) - 1)`
   - after: `(sqrt(x) + 1)/(x - 1^2)`
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

## radical_notable_quotient_sqrt_input (rationalize)

- Source: `((sqrt(x))^3 - 1)/(sqrt(x) - 1)`
- Target: `sqrt(x)+x+1`
- Result: `sqrt(x) + x + 1`
- Web step count: `2`
- Web substep count: `5`
- Flags: none

### CLI

```text
Parsed: (sqrt(x)^3 - 1) / (sqrt(x) - 1)
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

## expand_fraction_same_denominator_three_terms (fraction_expand)

- Source: `(a+b+c)/d`
- Target: `a/d + b/d + c/d`
- Result: `a / d + b / d + c / d`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: (a + b + c) / d
Target: a / d + b / d + c / d
Strategy: expand fraction
Steps (Aggressive Mode):
1. Distribute a sum over the common denominator  [Distribute Division]
   Before: (a + b + c) / d
   After: a / d + b / d + c / d
Result: a / d + b / d + c / d
ℹ️ Requires:
  • d ≠ 0
```

### Web / JSON Steps

1. `Repartir el denominador común`
   - before: `(a + b + c)/d`
   - after: `a/d + b/d + c/d`
   - substeps:
     1. `Repartir el mismo denominador sobre cada término del numerador`

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
     2. `Simplificar cada fracción resultante por separado`

## nested_fraction_one_over_sum (nested_fraction)

- Source: `1/(1/x + 1/y)`
- Target: `(x*y)/(x+y)`
- Result: `x * y / (x + y)`
- Web step count: `2`
- Web substep count: `3`
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
   - substeps:
     1. `Llevar ambas fracciones al mismo denominador`
     2. `Juntar todo en una sola fracción`
2. `Simplificar fracción anidada`
   - before: `1/((x + y)/(x · y))`
   - after: `(x · y)/(x + y)`
   - substeps:
     1. `Dividir entre una fracción equivale a invertirla`

## nested_fraction_fraction_over_sum (nested_fraction)

- Source: `(1/x)/(1/y + 1/z)`
- Target: `(y*z)/(x*(y+z))`
- Result: `y * z / (x * (y + z))`
- Web step count: `2`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: 1 / x / (1 / y + 1 / z)
Target: y * z / (x * (y + z))
Strategy: simplify
Steps (Aggressive Mode):
1. Add fractions: a/b + c/d -> (ad+bc)/bd  [Add Fractions]
   Before: 1 / x / (1 / y + 1 / z)
   Cambio local: 1 / y + 1 / z -> (y + z) / (y * z)
   After: 1 / x / ((y + z) / (y * z))
2. Simplify nested fraction  [Simplify Complex Fraction]
   Before: 1 / x / ((y + z) / (y * z))
   Cambio local: 1 / x / ((y + z) / (y * z)) -> y * z / ((y + z) * x)
   After: y * z / (x * (y + z))
Result: y * z / (x * (y + z))
ℹ️ Requires:
  • x * y + x * z ≠ 0
```

### Web / JSON Steps

1. `Sumar fracciones`
   - before: `(1/x)/(1/y + 1/z)`
   - after: `(1/x)/((y + z)/(y · z))`
   - substeps:
     1. `Llevar ambas fracciones al mismo denominador`
     2. `Juntar todo en una sola fracción`
2. `Simplificar fracción anidada`
   - before: `(1/x)/((y + z)/(y · z))`
   - after: `(y · z)/(x · (y + z))`
   - substeps:
     1. `Dividir entre una fracción equivale a invertirla`

## combine_same_denominator_fraction_sum (fraction_combine)

- Source: `a/x + b/x`
- Target: `(a+b)/x`
- Result: `(a + b) / x`
- Web step count: `1`
- Web substep count: `2`
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
   - substeps:
     1. `Como el denominador ya es el mismo, se mantiene igual`
     2. `Basta sumar los numeradores`

## combine_general_fraction_sum (fraction_combine)

- Source: `1/x + 1/y`
- Target: `(x+y)/(x*y)`
- Result: `(x + y) / (x * y)`
- Web step count: `1`
- Web substep count: `2`
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
   - substeps:
     1. `Llevar ambas fracciones al mismo denominador`
     2. `Juntar todo en una sola fracción`

## combine_same_denominator_fraction_difference (fraction_combine)

- Source: `a/x - b/x`
- Target: `(a-b)/x`
- Result: `(a - b) / x`
- Web step count: `1`
- Web substep count: `2`
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
   - substeps:
     1. `Como el denominador ya es el mismo, se mantiene igual`
     2. `Basta restar los numeradores`

## combine_general_fraction_difference (fraction_combine)

- Source: `1/x - 1/y`
- Target: `(y-x)/(x*y)`
- Result: `(y - x) / (x * y)`
- Web step count: `1`
- Web substep count: `2`
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
   - substeps:
     1. `Llevar ambas fracciones al mismo denominador`
     2. `Restar los numeradores en una sola fracción`

## combine_term_and_fraction_subtraction (fraction_combine)

- Source: `a - b/a`
- Target: `(a^2-b)/a`
- Result: `(a^2 - b) / a`
- Web step count: `1`
- Web substep count: `2`
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
   - substeps:
     1. `Como el denominador ya es el mismo, se mantiene igual`
     2. `Basta restar los numeradores`

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
  • n + 2 ≠ 0
  • n ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/(n · (n + 2))`
   - after: `1/2 · (1/n - 1/(n + 2))`
   - substeps:
     1. `Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))`
     2. `Aquí u = n y k = 2`

## split_telescoping_fraction_gap_three (telescoping_fraction)

- Source: `1/(n*(n+3))`
- Target: `1/3*(1/n - 1/(n+3))`
- Result: `((1 / n - 1 / (n + 3)) * 1)/3`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (n * (n + 3))
Target: ((1 / n - 1 / (n + 3)))/3
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / (n * (n + 3))
   Cambio local: 1 / (n * (n + 3)) -> 1 / 3 * (1 / n - 1 / (n + 3))
   After: ((1 / n - 1 / (n + 3)))/3
Result: 1 / 3 * (1 / n - 1 / (n + 3))
ℹ️ Requires:
  • n ≠ 0
  • n + 3 ≠ 0
  • 3 ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/(n · (n + 3))`
   - after: `1/3 · (1/n - 1/(n + 3))`
   - substeps:
     1. `Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))`
     2. `Aquí u = n y k = 3`

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

## combine_telescoping_fraction_gap_three (telescoping_fraction)

- Source: `1/3*(1/n - 1/(n+3))`
- Target: `1/(n*(n+3))`
- Result: `1 / (n * (n + 3))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ((1 / n - 1 / (n + 3)) * 1)/3
Target: 1 / (n * (n + 3))
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / 3 * (1 / n - 1 / (n + 3))
   Cambio local: 1 / 3 * (1 / n - 1 / (n + 3)) -> 1 / (n * (n + 3))
   After: 1 / (n * (n + 3))
Result: 1 / (n * (n + 3))
ℹ️ Requires:
  • n ≠ 0
  • n + 3 ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/3 · (1/n - 1/(n + 3))`
   - after: `1/(n · (n + 3))`
   - substeps:
     1. `Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))`
     2. `Aquí u = n y k = 3`

## split_telescoping_fraction_negative_consecutive (telescoping_fraction)

- Source: `1/(n*(n-1))`
- Target: `1/(n-1) - 1/n`
- Result: `1 / (n - 1) - 1 / n`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (n * (n - 1))
Target: 1 / (n - 1) - 1 / n
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / (n * (n - 1))
   Cambio local: 1 / (n * (n - 1)) -> 1 / (n - 1) - 1 / n
   After: 1 / (n - 1) - 1 / n
Result: 1 / (n - 1) - 1 / n
ℹ️ Requires:
  • n ≠ 0
  • n - 1 ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/(n · (n - 1))`
   - after: `1/(n - 1) - 1/n`
   - substeps:
     1. `Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)`
     2. `Aquí u = n - 1`

## combine_telescoping_fraction_negative_consecutive (telescoping_fraction)

- Source: `1/(n-1) - 1/n`
- Target: `1/(n*(n-1))`
- Result: `1 / (n * (n - 1))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / (n - 1) - 1 / n
Target: 1 / (n * (n - 1))
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / (n - 1) - 1 / n
   Cambio local: 1 / (n - 1) - 1 / n -> 1 / (n * (n - 1))
   After: 1 / (n * (n - 1))
Result: 1 / (n * (n - 1))
ℹ️ Requires:
  • n ≠ 0
  • n - 1 ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/(n - 1) - 1/n`
   - after: `1/(n · (n - 1))`
   - substeps:
     1. `Usar 1 / u - 1 / (u + 1) = 1 / (u · (u + 1))`
     2. `Aquí u = n - 1`

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
  • 2 * n + 1 ≠ 0
  • 2 ≠ 0
  • 2 * n + 3 ≠ 0
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

## split_telescoping_fraction_affine_shifted_gap_two (telescoping_fraction)

- Source: `1/((2*n-1)*(2*n+1))`
- Target: `1/2*(1/(2*n-1) - 1/(2*n+1))`
- Result: `((1 / (2 * n - 1) - 1 / (2 * n + 1)) * 1)/2`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / ((2 * n + 1) * (2 * n - 1))
Target: ((1 / (2 * n - 1) - 1 / (2 * n + 1)))/2
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / ((2 * n + 1) * (2 * n - 1))
   Cambio local: 1 / ((2 * n + 1) * (2 * n - 1)) -> 1 / 2 * (1 / (2 * n - 1) - 1 / (2 * n + 1))
   After: ((1 / (2 * n - 1) - 1 / (2 * n + 1)))/2
Result: 1 / 2 * (1 / (2 * n - 1) - 1 / (2 * n + 1))
ℹ️ Requires:
  • 2 * n + 1 ≠ 0
  • 2 * n - 1 ≠ 0
  • 2 ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/((2 · n + 1) · (2 · n - 1))`
   - after: `1/2 · (1/(2 · n - 1) - 1/(2 · n + 1))`
   - substeps:
     1. `Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))`
     2. `Aquí u = 2 · n - 1 y k = 2`

## combine_telescoping_fraction_affine_shifted_gap_two (telescoping_fraction)

- Source: `1/2*(1/(2*n-1) - 1/(2*n+1))`
- Target: `1/((2*n-1)*(2*n+1))`
- Result: `1 / ((2 * n + 1) * (2 * n - 1))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ((1 / (2 * n - 1) - 1 / (2 * n + 1)) * 1)/2
Target: 1 / ((2 * n + 1) * (2 * n - 1))
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / 2 * (1 / (2 * n - 1) - 1 / (2 * n + 1))
   Cambio local: 1 / 2 * (1 / (2 * n - 1) - 1 / (2 * n + 1)) -> 1 / ((2 * n + 1) * (2 * n - 1))
   After: 1 / ((2 * n + 1) * (2 * n - 1))
Result: 1 / ((2 * n + 1) * (2 * n - 1))
ℹ️ Requires:
  • 2 * n - 1 ≠ 0
  • 2 * n + 1 ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/2 · (1/(2 · n - 1) - 1/(2 · n + 1))`
   - after: `1/((2 · n + 1) · (2 · n - 1))`
   - substeps:
     1. `Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))`
     2. `Aquí u = 2 · n - 1 y k = 2`

## split_telescoping_fraction_affine_coeff_three_gap_three (telescoping_fraction)

- Source: `1/((3*n+2)*(3*n+5))`
- Target: `1/3*(1/(3*n+2) - 1/(3*n+5))`
- Result: `((1 / (3 * n + 2) - 1 / (3 * n + 5)) * 1)/3`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / ((3 * n + 2) * (3 * n + 5))
Target: ((1 / (3 * n + 2) - 1 / (3 * n + 5)))/3
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / ((3 * n + 2) * (3 * n + 5))
   Cambio local: 1 / ((3 * n + 2) * (3 * n + 5)) -> 1 / 3 * (1 / (3 * n + 2) - 1 / (3 * n + 5))
   After: ((1 / (3 * n + 2) - 1 / (3 * n + 5)))/3
Result: 1 / 3 * (1 / (3 * n + 2) - 1 / (3 * n + 5))
ℹ️ Requires:
  • 3 ≠ 0
  • 3 * n + 5 ≠ 0
  • 3 * n + 2 ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/((3 · n + 2) · (3 · n + 5))`
   - after: `1/3 · (1/(3 · n + 2) - 1/(3 · n + 5))`
   - substeps:
     1. `Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))`
     2. `Aquí u = 3 · n + 2 y k = 3`

## combine_telescoping_fraction_affine_coeff_three_gap_three (telescoping_fraction)

- Source: `1/3*(1/(3*n+2) - 1/(3*n+5))`
- Target: `1/((3*n+2)*(3*n+5))`
- Result: `1 / ((3 * n + 2) * (3 * n + 5))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ((1 / (3 * n + 2) - 1 / (3 * n + 5)) * 1)/3
Target: 1 / ((3 * n + 2) * (3 * n + 5))
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / 3 * (1 / (3 * n + 2) - 1 / (3 * n + 5))
   Cambio local: 1 / 3 * (1 / (3 * n + 2) - 1 / (3 * n + 5)) -> 1 / ((3 * n + 2) * (3 * n + 5))
   After: 1 / ((3 * n + 2) * (3 * n + 5))
Result: 1 / ((3 * n + 2) * (3 * n + 5))
ℹ️ Requires:
  • 3 * n + 2 ≠ 0
  • 3 * n + 5 ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/3 · (1/(3 · n + 2) - 1/(3 · n + 5))`
   - after: `1/((3 · n + 2) · (3 · n + 5))`
   - substeps:
     1. `Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))`
     2. `Aquí u = 3 · n + 2 y k = 3`

## split_telescoping_fraction_affine_coeff_three_shifted_gap_three (telescoping_fraction)

- Source: `1/((3*n-1)*(3*n+2))`
- Target: `1/3*(1/(3*n-1) - 1/(3*n+2))`
- Result: `((1 / (3 * n - 1) - 1 / (3 * n + 2)) * 1)/3`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / ((3 * n + 2) * (3 * n - 1))
Target: ((1 / (3 * n - 1) - 1 / (3 * n + 2)))/3
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / ((3 * n + 2) * (3 * n - 1))
   Cambio local: 1 / ((3 * n + 2) * (3 * n - 1)) -> 1 / 3 * (1 / (3 * n - 1) - 1 / (3 * n + 2))
   After: ((1 / (3 * n - 1) - 1 / (3 * n + 2)))/3
Result: 1 / 3 * (1 / (3 * n - 1) - 1 / (3 * n + 2))
ℹ️ Requires:
  • 3 * n - 1 ≠ 0
  • 3 ≠ 0
  • 3 * n + 2 ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/((3 · n + 2) · (3 · n - 1))`
   - after: `1/3 · (1/(3 · n - 1) - 1/(3 · n + 2))`
   - substeps:
     1. `Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))`
     2. `Aquí u = 3 · n - 1 y k = 3`

## combine_telescoping_fraction_affine_coeff_three_shifted_gap_three (telescoping_fraction)

- Source: `1/3*(1/(3*n-1) - 1/(3*n+2))`
- Target: `1/((3*n-1)*(3*n+2))`
- Result: `1 / ((3 * n + 2) * (3 * n - 1))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ((1 / (3 * n - 1) - 1 / (3 * n + 2)) * 1)/3
Target: 1 / ((3 * n + 2) * (3 * n - 1))
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / 3 * (1 / (3 * n - 1) - 1 / (3 * n + 2))
   Cambio local: 1 / 3 * (1 / (3 * n - 1) - 1 / (3 * n + 2)) -> 1 / ((3 * n + 2) * (3 * n - 1))
   After: 1 / ((3 * n + 2) * (3 * n - 1))
Result: 1 / ((3 * n + 2) * (3 * n - 1))
ℹ️ Requires:
  • 3 * n - 1 ≠ 0
  • 3 * n + 2 ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/3 · (1/(3 · n - 1) - 1/(3 · n + 2))`
   - after: `1/((3 · n + 2) · (3 · n - 1))`
   - substeps:
     1. `Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))`
     2. `Aquí u = 3 · n - 1 y k = 3`

## split_telescoping_fraction_affine_symbolic_coeff_gap_three (telescoping_fraction)

- Source: `1/((a*n+2)*(a*n+5))`
- Target: `1/3*(1/(a*n+2) - 1/(a*n+5))`
- Result: `((1 / (a * n + 2) - 1 / (a * n + 5)) * 1)/3`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / ((a * n + 2) * (a * n + 5))
Target: ((1 / (a * n + 2) - 1 / (a * n + 5)))/3
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / ((a * n + 2) * (a * n + 5))
   Cambio local: 1 / ((a * n + 2) * (a * n + 5)) -> 1 / 3 * (1 / (a * n + 2) - 1 / (a * n + 5))
   After: ((1 / (a * n + 2) - 1 / (a * n + 5)))/3
Result: 1 / 3 * (1 / (a * n + 2) - 1 / (a * n + 5))
ℹ️ Requires:
  • a * n + 2 ≠ 0
  • 3 ≠ 0
  • a * n + 5 ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/((a · n + 2) · (a · n + 5))`
   - after: `1/3 · (1/(a · n + 2) - 1/(a · n + 5))`
   - substeps:
     1. `Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))`
     2. `Aquí u = a · n + 2 y k = 3`

## combine_telescoping_fraction_affine_symbolic_coeff_gap_three (telescoping_fraction)

- Source: `1/3*(1/(a*n+2) - 1/(a*n+5))`
- Target: `1/((a*n+2)*(a*n+5))`
- Result: `1 / ((a * n + 2) * (a * n + 5))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ((1 / (a * n + 2) - 1 / (a * n + 5)) * 1)/3
Target: 1 / ((a * n + 2) * (a * n + 5))
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / 3 * (1 / (a * n + 2) - 1 / (a * n + 5))
   Cambio local: 1 / 3 * (1 / (a * n + 2) - 1 / (a * n + 5)) -> 1 / ((a * n + 2) * (a * n + 5))
   After: 1 / ((a * n + 2) * (a * n + 5))
Result: 1 / ((a * n + 2) * (a * n + 5))
ℹ️ Requires:
  • a^2 * n^2 + 7 * a * n + 10 ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/3 · (1/(a · n + 2) - 1/(a · n + 5))`
   - after: `1/((a · n + 2) · (a · n + 5))`
   - substeps:
     1. `Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))`
     2. `Aquí u = a · n + 2 y k = 3`

## split_telescoping_fraction_affine_symbolic_coeff_shifted_gap_three (telescoping_fraction)

- Source: `1/((a*n-1)*(a*n+2))`
- Target: `1/3*(1/(a*n-1) - 1/(a*n+2))`
- Result: `((1 / (a * n - 1) - 1 / (a * n + 2)) * 1)/3`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / ((a * n + 2) * (a * n - 1))
Target: ((1 / (a * n - 1) - 1 / (a * n + 2)))/3
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / ((a * n + 2) * (a * n - 1))
   Cambio local: 1 / ((a * n + 2) * (a * n - 1)) -> 1 / 3 * (1 / (a * n - 1) - 1 / (a * n + 2))
   After: ((1 / (a * n - 1) - 1 / (a * n + 2)))/3
Result: 1 / 3 * (1 / (a * n - 1) - 1 / (a * n + 2))
ℹ️ Requires:
  • a * n + 2 ≠ 0
  • 3 ≠ 0
  • a * n - 1 ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/((a · n + 2) · (a · n - 1))`
   - after: `1/3 · (1/(a · n - 1) - 1/(a · n + 2))`
   - substeps:
     1. `Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))`
     2. `Aquí u = a · n - 1 y k = 3`

## combine_telescoping_fraction_affine_symbolic_coeff_shifted_gap_three (telescoping_fraction)

- Source: `1/3*(1/(a*n-1) - 1/(a*n+2))`
- Target: `1/((a*n-1)*(a*n+2))`
- Result: `1 / ((a * n + 2) * (a * n - 1))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ((1 / (a * n - 1) - 1 / (a * n + 2)) * 1)/3
Target: 1 / ((a * n + 2) * (a * n - 1))
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / 3 * (1 / (a * n - 1) - 1 / (a * n + 2))
   Cambio local: 1 / 3 * (1 / (a * n - 1) - 1 / (a * n + 2)) -> 1 / ((a * n + 2) * (a * n - 1))
   After: 1 / ((a * n + 2) * (a * n - 1))
Result: 1 / ((a * n + 2) * (a * n - 1))
ℹ️ Requires:
  • a^2 * n^2 + a * n - 2 ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/3 · (1/(a · n - 1) - 1/(a · n + 2))`
   - after: `1/((a · n + 2) · (a · n - 1))`
   - substeps:
     1. `Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))`
     2. `Aquí u = a · n - 1 y k = 3`

## split_telescoping_fraction_symbolic_shift_gap (telescoping_fraction)

- Source: `1/((n+a)*(n+b))`
- Target: `1/(b-a)*(1/(n+a) - 1/(n+b))`
- Result: `((1 / (a + n) - 1 / (b + n)) * 1)/((b - a))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 1 / ((a + n) * (b + n))
Target: ((1 / (a + n) - 1 / (b + n)))/((b - a))
Strategy: expand fraction
Steps (Aggressive Mode):
1. Split into telescoping partial fractions  [Telescoping Fraction Split]
   Before: 1 / ((a + n) * (b + n))
   Cambio local: 1 / ((a + n) * (b + n)) -> 1 / (b - a) * (1 / (a + n) - 1 / (b + n))
   After: ((1 / (a + n) - 1 / (b + n)))/((b - a))
Result: 1 / (b - a) * (1 / (a + n) - 1 / (b + n))
ℹ️ Requires:
  • b + n ≠ 0
  • a - b ≠ 0
  • a + n ≠ 0
```

### Web / JSON Steps

1. `Descomponer en fracciones telescópicas`
   - before: `1/((a + n) · (b + n))`
   - after: `1/(b - a) · (1/(a + n) - 1/(b + n))`
   - substeps:
     1. `Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))`
     2. `Aquí u = a + n y k = b - a`

## combine_telescoping_fraction_symbolic_shift_gap (telescoping_fraction)

- Source: `1/(b-a)*(1/(n+a) - 1/(n+b))`
- Target: `1/((n+a)*(n+b))`
- Result: `1 / ((a + n) * (b + n))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: ((1 / (a + n) - 1 / (b + n)) * 1)/((b - a))
Target: 1 / ((a + n) * (b + n))
Strategy: combine fraction
Steps (Aggressive Mode):
1. Recompose the telescoping partial fractions into a single fraction  [Telescoping Fraction Combine]
   Before: 1 / (b - a) * (1 / (a + n) - 1 / (b + n))
   Cambio local: 1 / (b - a) * (1 / (a + n) - 1 / (b + n)) -> 1 / ((a + n) * (b + n))
   After: 1 / ((a + n) * (b + n))
Result: 1 / ((a + n) * (b + n))
ℹ️ Requires:
  • n^2 + a * b + a * n + b * n ≠ 0
```

### Web / JSON Steps

1. `Recomponer fracción telescópica`
   - before: `1/(b - a) · (1/(a + n) - 1/(b + n))`
   - after: `1/((a + n) · (b + n))`
   - substeps:
     1. `Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))`
     2. `Aquí u = a + n y k = b - a`

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
  • a * n + b ≠ 0
  • a * n + c ≠ 0
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

## small_polynomial_product (polynomial_product)

- Source: `(x - 1)*(x^5 + x^4 + x^3 + x^2 + x + 1)`
- Target: `x^6 - 1`
- Result: `x^6 - 1`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: (x^5 + x^4 + x^3 + x^2 + x + 1) * (x - 1)
Target: x^6 - 1
Strategy: expand
Steps (Aggressive Mode):
1. Expand the expression distributively  [Expand]
   Before: (x^(5) + x^(4) + x^(3) + x^(2) + x + 1) * (x - 1)
   Cambio local: (x^(5) + x^(4) + x^(3) + x^(2) + x + 1) * (x - 1) -> x^(6) - 1
   After: x^6 - 1
Result: x^(6) - 1
```

### Web / JSON Steps

1. `Expandir la expresión`
   - before: `(x^5 + x^4 + x^3 + x^2 + x + 1) · (x - 1)`
   - after: `x^6 - 1`
   - substeps:
     1. `Distribuir cada término del producto`
     2. `Agrupar los términos del mismo grado`
     3. `Los términos intermedios se cancelan por parejas`

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
- Web substep count: `1`
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
   - substeps:
     1. `Usar x^a · x^b = x^(a+b)`

## merge_mixed_root_and_power (power_merge)

- Source: `sqrt(x)*x^(2/3)`
- Target: `x^(7/6)`
- Result: `x^(7 / 6)`
- Web step count: `2`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: sqrt(x) * x^(2 / 3)
Target: x^(7 / 6)
Strategy: combine powers
Steps (Aggressive Mode):
1. sqrt(x) = x^(1/2)  [Canonicalize Roots]
   Before: sqrt(x) * x^(2 / 3)
   Cambio local: sqrt(x) * x^(2 / 3) -> x^(1/2) * x^(2 / 3)
   After: x^(1/2) * x^(2 / 3)
2. Combine powers with same base (n-ary)  [Combine powers with same base (n-ary)]
   Before: x^(1/2) * x^(2 / 3)
   Cambio local: x^(1/2) * x^(2 / 3) -> x^(7 / 6)
   After: x^(7 / 6)
Result: x^(7 / 6)
ℹ️ Requires:
  • 6 ≠ 0
```

### Web / JSON Steps

1. `Reescribir la raíz como potencia fraccionaria`
   - before: `sqrt(x) · sqrt[3]x^2`
   - after: `sqrt(x) · sqrt[3]x^2`
   - substeps:
     1. `Usar sqrt(u) = u^(1/2)`
2. `Sumar exponentes de la misma base`
   - before: `sqrt(x) · sqrt[3]x^2`
   - after: `sqrt[6]x^7`
   - substeps:
     1. `Usar x^a · x^b = x^(a+b)`

## log_sum_difference_cancels_to_zero (simplify)

- Source: `ln(x^3) + ln(y^2) - ln(x^3*y^2)`
- Target: `0`
- Result: `0`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar que el logaritmo de un producto se separa en una suma`

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
Parsed: fact(n + 1) / fact(n)
Target: n + 1
Strategy: simplify
Steps (Aggressive Mode):
1. Cancel consecutive factorials  [Consecutive Factorial Ratio]
   Before: fact(n + 1) / fact(n)
   Cambio local: fact(n + 1) / fact(n) -> n + 1
   After: n + 1
Result: n + 1
```

### Web / JSON Steps

1. `Cancelar factoriales consecutivos`
   - before: `fact(n + 1)/fact(n)`
   - after: `n + 1`
   - substeps:
     1. `Escribir el factorial superior como el siguiente número por el factorial anterior`
     2. `Cancelar el factorial común`

## inverse_tan_identity_cancels_to_zero (simplify)

- Source: `atan(3) + (atan(1/3) - pi/2)`
- Target: `0`
- Result: `0`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: atan(3) + atan(1 / 3) - pi / 2
Target: 0
Strategy: simplify
Steps (Aggressive Mode):
1. arctan(x) + arctan(1/x) = π/2  [Inverse Tan Relations]
   Before: arctan(1/3) + arctan(3) - 1/2 * pi
   Cambio local: arctan(1/3) + arctan(3) -> pi / 2
   After: 0
Result: 0
```

### Web / JSON Steps

1. `Aplicar identidad de arctangentes`
   - before: `arctan(1/3) + arctan(3) - 1/2 · pi`
   - after: `pi/2 - 1/2 · pi`
   - substeps:
     1. `Usar arctan(u) + arctan(1/u) = pi/2`
     2. `Juntar la pareja que encaja con la identidad`
     3. `Esa pareja vale pi/2`

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
- Web substep count: `1`
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
   - substeps:
     1. `Usar tan(u) · cot(u) = 1`

## sec_tan_pythagorean_to_one (simplify)

- Source: `sec(x)^2 - tan(x)^2`
- Target: `1`
- Result: `1`
- Web step count: `1`
- Web substep count: `1`
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
   - substeps:
     1. `Usar sec²(u) - tan²(u) = 1`

## csc_cot_pythagorean_to_one (simplify)

- Source: `csc(x)^2 - cot(x)^2`
- Target: `1`
- Result: `1`
- Web step count: `1`
- Web substep count: `1`
- Flags: none

### CLI

```text
Parsed: csc(x)^2 - cot(x)^2
Target: 1
Strategy: simplify
Steps (Aggressive Mode):
1. Recognize csc²(u) - cot²(u) = 1  [Reciprocal Pythagorean Identity]
   Before: csc(x)^(2) - cot(x)^(2)
   Cambio local: csc(x)^(2) - cot(x)^(2) -> 1
   After: 1
Result: 1
```

### Web / JSON Steps

1. `Aplicar identidad pitagórica recíproca`
   - before: `csc(x)^2 - cot(x)^2`
   - after: `1`
   - substeps:
     1. `Usar csc²(u) - cot²(u) = 1`

## factor_symbolic_binomial_cube (factor)

- Source: `a^3 + 3*a^2*b + 3*a*b^2 + b^3`
- Target: `(a+b)^3`
- Result: `(a + b)^3`
- Web step count: `1`
- Web substep count: `2`
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
     2. `Aquí a = a y b = b`

## factor_symbolic_binomial_cube_minus (factor)

- Source: `a^3 - 3*a^2*b + 3*a*b^2 - b^3`
- Target: `(a-b)^3`
- Result: `(a - b)^3`
- Web step count: `1`
- Web substep count: `2`
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
     2. `Aquí a = a y b = b`

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
1. Apply Morrie's law to telescope the cosine product  [Cos Product Telescoping]
   Before: cos(x) * cos(2 * x) * cos(4 * x)
   Cambio local: cos(x) * cos(2 * x) * cos(4 * x) -> sin(8 * x) / (8 * sin(x))
   After: sin(8 * x) / (8 * sin(x))
Result: sin(8 * x) / (8 * sin(x))
ℹ️ Requires:
  • 8 * sin(x) ≠ 0
```

### Web / JSON Steps

1. `Aplicar telescopado de cosenos`
   - before: `cos(x) · cos(2 · x) · cos(4 · x)`
   - after: `sin(8 · x)/(8 · sin(x))`
   - substeps:
     1. `Usar el telescopado de cosenos`
     2. `Aquí u = x`

## integrate_prep_morrie_scaled (integrate_prep)

- Source: `cos(3*x)*cos(6*x)`
- Target: `sin(12*x)/(4*sin(3*x))`
- Result: `sin(12 * x) / (4 * sin(3 * x))`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: cos(3 * x) * cos(6 * x)
Target: sin(12 * x) / (4 * sin(3 * x))
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply Morrie's law to telescope the cosine product  [Cos Product Telescoping]
   Before: cos(3 * x) * cos(6 * x)
   Cambio local: cos(3 * x) * cos(6 * x) -> sin(12 * x) / (4 * sin(3 * x))
   After: sin(12 * x) / (4 * sin(3 * x))
Result: sin(12 * x) / (4 * sin(3 * x))
ℹ️ Requires:
  • 4 * sin(3 * x) ≠ 0
```

### Web / JSON Steps

1. `Aplicar telescopado de cosenos`
   - before: `cos(3 · x) · cos(6 · x)`
   - after: `sin(12 · x)/(4 · sin(3 · x))`
   - substeps:
     1. `Usar el telescopado de cosenos`
     2. `Aquí u = 3 · x`

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
1. Apply the Dirichlet kernel identity to rewrite the cosine sum  [Dirichlet Kernel Identity]
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
1. Apply the Dirichlet kernel identity to rewrite the cosine sum  [Dirichlet Kernel Identity]
   Before: 2 * cos(x) + 2 * cos(2 * x) + 2 * cos(3 * x) + 1
   Cambio local: 2 * cos(x) + 2 * cos(2 * x) + 2 * cos(3 * x) + 1 -> sin(7 * x / 2) / sin(x / 2)
   After: sin(7 * x / 2) / sin(x / 2)
Result: sin(7 * x / 2) / sin(x / 2)
ℹ️ Requires:
  • 2 ≠ 0
  • sin(x / 2) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `2 · cos(x) + 2 · cos(2 · x) + 2 · cos(3 · x) + 1`
   - after: `sin((7 · x)/2)/sin(x/2)`
   - substeps:
     1. `Usar el núcleo de Dirichlet`
     2. `Aquí n = 3 y u = x`

## integrate_prep_dirichlet_scaled (integrate_prep)

- Source: `1 + 2*cos(3*x) + 2*cos(6*x)`
- Target: `sin(15*x/2)/sin(3*x/2)`
- Result: `sin(15 * x / 2) / sin(3 * x / 2)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 * cos(3 * x) + 2 * cos(6 * x) + 1
Target: sin(15 * x / 2) / sin(3 * x / 2)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity to rewrite the cosine sum  [Dirichlet Kernel Identity]
   Before: 2 * cos(3 * x) + 2 * cos(6 * x) + 1
   Cambio local: 2 * cos(3 * x) + 2 * cos(6 * x) + 1 -> sin(15 * x / 2) / sin(3 * x / 2)
   After: sin(15 * x / 2) / sin(3 * x / 2)
Result: sin(15 * x / 2) / sin(3 * x / 2)
ℹ️ Requires:
  • sin(3 * x / 2) ≠ 0
  • 2 ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `2 · cos(3 · x) + 2 · cos(6 · x) + 1`
   - after: `sin((15 · x)/2)/sin((3 · x)/2)`
   - substeps:
     1. `Usar el núcleo de Dirichlet`
     2. `Aquí n = 2 y u = 3 · x`

## integrate_prep_dirichlet_scaled_longer (integrate_prep)

- Source: `1 + 2*cos(2*x) + 2*cos(4*x) + 2*cos(6*x)`
- Target: `sin(7*x)/sin(x)`
- Result: `sin(7 * x) / sin(x)`
- Web step count: `1`
- Web substep count: `2`
- Flags: none

### CLI

```text
Parsed: 2 * cos(2 * x) + 2 * cos(4 * x) + 2 * cos(6 * x) + 1
Target: sin(7 * x) / sin(x)
Strategy: integrate prep
Steps (Aggressive Mode):
1. Apply the Dirichlet kernel identity to rewrite the cosine sum  [Dirichlet Kernel Identity]
   Before: 2 * cos(2 * x) + 2 * cos(4 * x) + 2 * cos(6 * x) + 1
   Cambio local: 2 * cos(2 * x) + 2 * cos(4 * x) + 2 * cos(6 * x) + 1 -> sin(7 * x) / sin(x)
   After: sin(7 * x) / sin(x)
Result: sin(7 * x) / sin(x)
ℹ️ Requires:
  • sin(x) ≠ 0
```

### Web / JSON Steps

1. `Aplicar identidad del núcleo de Dirichlet`
   - before: `2 · cos(2 · x) + 2 · cos(4 · x) + 2 · cos(6 · x) + 1`
   - after: `sin(7 · x)/sin(x)`
   - substeps:
     1. `Usar el núcleo de Dirichlet`
     2. `Aquí n = 3 y u = 2 · x`

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

## finite_telescoping_product_shifted (finite_telescoping)

- Source: `product((k+2)/(k+1), k, 1, n)`
- Target: `(n+2)/2`
- Result: `(n + 2) / 2`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: product((k + 2) / (k + 1), k, 1, n)
Target: (n + 2) / 2
Strategy: simplify
Steps (Aggressive Mode):
1. Telescoping product: Π((k + 2) / (k + 1), k) from 1 to n  [Finite Product]
   Before: product((k + 2) / (k + 1), k, 1, n)
   Cambio local: product((k + 2) / (k + 1), k, 1, n) -> (n + 2) / 2
   After: (n + 2) / 2
Result: (n + 2) / 2
ℹ️ Requires:
  • 2 ≠ 0
```

### Web / JSON Steps

1. `Evaluar producto telescópico finito`
   - before: `prod_k=1^n (k + 2)/(k + 1)`
   - after: `(n + 2)/2`
   - substeps:
     1. `Escribir los primeros y últimos factores del producto`
     2. `Los factores intermedios se cancelan por parejas`
     3. `Solo quedan el último numerador y el primer denominador`

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

## finite_telescoping_sum_shifted (finite_telescoping)

- Source: `sum(1/((k+2)*(k+3)), k, 1, n)`
- Target: `1/3 - 1/(n+3)`
- Result: `1 / 3 - 1 / (n + 3)`
- Web step count: `1`
- Web substep count: `3`
- Flags: none

### CLI

```text
Parsed: sum(1 / ((k + 2) * (k + 3)), k, 1, n)
Target: 1 / 3 - 1 / (n + 3)
Strategy: simplify
Steps (Aggressive Mode):
1. Telescoping sum: Σ(1 / ((k + 2) * (k + 3)), k) from 1 to n  [Finite Summation]
   Before: sum(1 / ((k + 2) * (k + 3)), k, 1, n)
   Cambio local: sum(1 / ((k + 2) * (k + 3)), k, 1, n) -> 1 / 3 - 1 / (n + 3)
   After: 1 / 3 - 1 / (n + 3)
Result: 1 / 3 - 1 / (n + 3)
ℹ️ Requires:
  • n + 3 ≠ 0
  • 3 ≠ 0
```

### Web / JSON Steps

1. `Evaluar suma telescópica finita`
   - before: `sum_k=1^n 1/((k + 2) · (k + 3))`
   - after: `1/3 - 1/(n + 3)`
   - substeps:
     1. `Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)`
     2. `Aquí u = k + 2`
     3. `La suma telescópica cancela los términos intermedios`
