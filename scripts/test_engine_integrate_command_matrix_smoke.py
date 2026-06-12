import importlib.util
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "engine_integrate_command_matrix_smoke.py"
SPEC = importlib.util.spec_from_file_location(
    "engine_integrate_command_matrix_smoke",
    MODULE_PATH,
)
SMOKE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)


class IntegrateCommandMatrixSmokeTests(unittest.TestCase):
    def test_default_matrix_covers_integrate_policy_axes(self) -> None:
        cases = SMOKE.build_cases()

        self.assertEqual(len(cases), 227)
        names = {case.name for case in cases}
        self.assertIn(
            "algorithmic_backend_hermite_expanded_symbolic_affine_positive_radius_mixed_numerator",
            names,
        )
        self.assertIn("reciprocal_affine_log_abs_domain", names)
        self.assertIn("reciprocal_negative_affine_log_abs_domain", names)
        self.assertIn("reciprocal_negative_affine_derivative_log_abs_domain", names)
        self.assertIn(
            "algorithmic_backend_rational_affine_quotient_numeric_slope",
            names,
        )
        self.assertIn(
            "algorithmic_backend_rational_affine_quotient_symbolic_slope",
            names,
        )
        self.assertIn(
            "algorithmic_backend_rational_affine_quotient_external_scale_zero_intercept",
            names,
        )
        self.assertIn(
            "algorithmic_backend_hermite_symbolic_positive_radius_mixed_numerator",
            names,
        )
        self.assertIn(
            "algorithmic_backend_hermite_symbolic_affine_positive_radius_mixed_numerator",
            names,
        )
        self.assertIn(
            "algorithmic_backend_hermite_symbolic_indefinite_square_denominator",
            names,
        )
        self.assertIn(
            "algorithmic_backend_hermite_symbolic_affine_indefinite_square_mixed_numerator",
            names,
        )
        self.assertIn("log_derivative_positive_quadratic_substitution", names)
        self.assertIn(
            "log_derivative_positive_quadratic_negative_orientation_substitution",
            names,
        )
        self.assertIn("log_rational_positive_domain_residual", names)
        self.assertIn("non_elementary_tan_polynomial_residual_domain", names)
        self.assertIn("non_elementary_tan_presimplified_residual_domain", names)
        self.assertIn("non_elementary_csc_presimplified_residual_domain", names)
        self.assertIn("explicit_reciprocal_sine_presimplified_residual_domain", names)
        self.assertIn("explicit_reciprocal_sine_verified_log_domain", names)
        self.assertIn("explicit_reciprocal_cosine_verified_log_domain", names)
        self.assertIn(
            "explicit_reciprocal_cosine_symbolic_shift_verified_log_domain",
            names,
        )
        self.assertIn(
            "explicit_reciprocal_cosine_symbolic_external_scale_shift_verified_log_domain",
            names,
        )
        self.assertIn("explicit_reciprocal_tangent_presimplified_residual_domain", names)
        self.assertIn("explicit_tangent_log_residual_condition_alias_dedupe", names)
        self.assertIn(
            "explicit_tangent_log_numeric_shifted_residual_condition_alias_dedupe",
            names,
        )
        self.assertIn(
            "explicit_tangent_log_numeric_offset_residual_condition_alias_dedupe",
            names,
        )
        self.assertIn(
            "explicit_cotangent_log_numeric_offset_residual_condition_alias_dedupe",
            names,
        )
        self.assertIn(
            "explicit_cotangent_log_numeric_shifted_residual_condition_alias_dedupe",
            names,
        )
        self.assertIn(
            "inverse_trig_symbolic_affine_positive_rational_radius_positive_quadratic_table",
            names,
        )
        self.assertIn(
            "inverse_trig_named_positive_constant_radius_positive_quadratic_table",
            names,
        )
        self.assertIn(
            "inverse_trig_numeric_affine_named_positive_constant_radius_positive_quadratic_table",
            names,
        )
        self.assertIn(
            "rational_positive_quadratic_linear_numerator_expanded_named_positive_radius_decomposition",
            names,
        )
        self.assertIn(
            "inverse_trig_expanded_numeric_affine_named_positive_constant_radius_positive_quadratic_table",
            names,
        )
        self.assertIn(
            "inverse_trig_symbolic_square_radius_positive_quadratic_table",
            names,
        )
        self.assertIn(
            "inverse_trig_symbolic_shifted_square_radius_positive_quadratic_table",
            names,
        )
        self.assertIn("explicit_reciprocal_tangent_verified_log_domain", names)
        self.assertIn(
            "explicit_reciprocal_tangent_presimplified_verified_log_domain",
            names,
        )
        self.assertIn(
            "explicit_reciprocal_secant_presimplified_source_domain",
            names,
        )
        self.assertIn(
            "symbolic_external_scale_tangent_log_derivative_ratio_domain",
            names,
        )
        self.assertIn(
            "symbolic_external_scale_cotangent_log_derivative_ratio_domain",
            names,
        )
        self.assertIn(
            "nested_tangent_log_derivative_denominator_verified_domain",
            names,
        )
        self.assertIn(
            "explicit_reciprocal_hyperbolic_tangent_verified_log_domain",
            names,
        )
        self.assertIn(
            "explicit_reciprocal_hyperbolic_tangent_presimplified_condition_dedupe",
            names,
        )
        self.assertIn(
            "symbolic_external_scale_reciprocal_hyperbolic_tangent_log_domain",
            names,
        )
        self.assertIn(
            "negative_orientation_symbolic_external_scale_reciprocal_hyperbolic_tangent_log_domain",
            names,
        )
        self.assertIn(
            "symbolic_external_scale_hyperbolic_tanh_log_derivative_ratio_positive",
            names,
        )
        self.assertIn(
            "symbolic_external_scale_hyperbolic_tanh_direct_log_positive",
            names,
        )
        self.assertIn("additive_trig_pole_residual_domain", names)
        self.assertIn("constant_multiple_affine_trig_substitution", names)
        self.assertIn("affine_secant_table_log_domain", names)
        self.assertIn("affine_cosecant_table_log_domain", names)
        self.assertIn("external_constant_affine_secant_table_log_domain", names)
        self.assertIn(
            "rational_denominator_scaled_affine_secant_table_log_domain",
            names,
        )
        self.assertIn(
            "rational_denominator_scaled_affine_cosecant_table_log_domain",
            names,
        )
        self.assertIn("log_power_product_substitution", names)
        self.assertIn(
            "constant_base_log_power_product_positive_domain_substitution",
            names,
        )
        self.assertIn("inverse_trig_sqrt_reciprocal_bridge", names)
        self.assertIn("inverse_trig_scaled_sqrt_reciprocal_bridge", names)
        self.assertIn(
            "inverse_trig_symbolic_denominator_scale_sqrt_reciprocal_bridge",
            names,
        )
        self.assertIn(
            "inverse_trig_symbolic_denominator_numeric_square_scale_sqrt_reciprocal_bridge",
            names,
        )
        self.assertIn(
            "inverse_trig_symbolic_numerator_scale_sqrt_reciprocal_bridge",
            names,
        )
        self.assertIn("inverse_trig_shifted_scaled_sqrt_reciprocal_bridge", names)
        self.assertIn(
            "inverse_trig_affine_shifted_scaled_positive_quadratic_table",
            names,
        )
        self.assertIn(
            "inverse_trig_symbolic_affine_expanded_square_radius_positive_quadratic_table",
            names,
        )
        self.assertIn("rational_positive_quadratic_square_reduction", names)
        self.assertIn("affine_exp_substitution", names)
        self.assertIn("linear_exp_by_parts", names)
        self.assertIn("linear_exp_affine_slope_by_parts", names)
        self.assertIn("by_parts_affine_log_domain", names)
        self.assertIn("inverse_hyperbolic_rational_direct_atanh_domain", names)
        self.assertIn("rational_partial_fraction_two_real_linear_factors", names)
        self.assertIn("rational_partial_fraction_three_real_linear_factors", names)
        self.assertIn(
            "rational_partial_fraction_mixed_simple_repeated_linear_factors",
            names,
        )
        self.assertIn(
            "rational_partial_fraction_repeated_real_linear_factors",
            names,
        )
        self.assertIn(
            "rational_partial_fraction_mixed_linear_positive_quadratic",
            names,
        )
        self.assertIn(
            "rational_partial_fraction_repeated_linear_positive_quadratic",
            names,
        )
        self.assertIn(
            "rational_partial_fraction_repeated_origin_linear_positive_quadratic_no_log",
            names,
        )
        self.assertIn(
            "rational_partial_fraction_repeated_origin_scaled_positive_quadratic_no_log",
            names,
        )
        self.assertIn(
            "rational_improper_partial_fraction_polynomial_division",
            names,
        )
        self.assertIn(
            "rational_improper_positive_quadratic_polynomial_division",
            names,
        )
        self.assertIn(
            "rational_improper_positive_quadratic_negative_orientation",
            names,
        )
        self.assertIn("affine_inverse_hyperbolic_atanh_domain", names)
        self.assertIn("inverse_sqrt_direct_arcsin_domain", names)
        self.assertIn("additive_inverse_sqrt_interval_residual_domain", names)
        self.assertIn("inverse_sqrt_symbolic_radius_arcsin_domain", names)
        self.assertIn("inverse_sqrt_symbolic_radius_shifted_arcsin_domain", names)
        self.assertIn("inverse_sqrt_symbolic_slope_shifted_arcsin_domain", names)
        self.assertIn("inverse_sqrt_direct_asinh_unconditional", names)
        self.assertIn("inverse_hyperbolic_sqrt_symbolic_radius_table", names)
        self.assertIn(
            "inverse_hyperbolic_sqrt_symbolic_shifted_radius_table",
            names,
        )
        self.assertIn(
            "inverse_hyperbolic_sqrt_symbolic_slope_shifted_radius_table",
            names,
        )
        self.assertIn("polynomial_base_sqrt_substitution", names)
        self.assertIn("hyperbolic_sine_reciprocal_square_substitution", names)
        self.assertIn(
            "negative_affine_hyperbolic_sine_reciprocal_square_substitution",
            names,
        )
        self.assertIn(
            "symbolic_affine_exact_hyperbolic_cosh_reciprocal_square_substitution",
            names,
        )
        self.assertIn(
            "symbolic_affine_exact_hyperbolic_sinh_reciprocal_square_substitution",
            names,
        )
        self.assertIn(
            "symbolic_external_scale_shifted_hyperbolic_cosh_reciprocal_square_substitution",
            names,
        )
        self.assertIn(
            "symbolic_external_scale_shifted_hyperbolic_sinh_reciprocal_square_substitution",
            names,
        )
        self.assertIn("hyperbolic_cosh_reciprocal_fourth_substitution", names)
        self.assertIn(
            "symbolic_external_scale_shifted_hyperbolic_cosh_reciprocal_fourth_substitution",
            names,
        )
        self.assertIn("hyperbolic_sinh_reciprocal_fourth_substitution", names)
        self.assertIn(
            "symbolic_external_scale_shifted_hyperbolic_sinh_reciprocal_fourth_substitution",
            names,
        )
        self.assertIn(
            "symbolic_affine_exact_hyperbolic_sinh_over_cosh_square_substitution",
            names,
        )
        self.assertIn(
            "symbolic_affine_exact_hyperbolic_cosh_over_sinh_square_substitution",
            names,
        )
        self.assertIn(
            "symbolic_external_scale_hyperbolic_sinh_over_cosh_square_substitution",
            names,
        )
        self.assertIn(
            "symbolic_external_scale_hyperbolic_cosh_over_sinh_square_substitution",
            names,
        )
        self.assertIn(
            "polynomial_shifted_hyperbolic_sinh_over_cosh_square_substitution",
            names,
        )
        self.assertIn(
            "symbolic_external_scale_shifted_hyperbolic_sinh_over_cosh_square_substitution",
            names,
        )
        self.assertIn(
            "symbolic_external_scale_shifted_hyperbolic_cosh_over_sinh_square_substitution",
            names,
        )
        self.assertIn("affine_secant_tangent_derivative_product_domain", names)
        self.assertIn("affine_cosecant_cotangent_derivative_product_domain", names)
        self.assertIn(
            "negative_affine_secant_tangent_derivative_product_domain",
            names,
        )
        self.assertIn(
            "negative_affine_cosecant_cotangent_derivative_product_domain",
            names,
        )
        self.assertIn(
            "polynomial_shifted_secant_tangent_derivative_product_domain",
            names,
        )
        self.assertIn(
            "symbolic_affine_exact_secant_tangent_derivative_product_domain",
            names,
        )
        self.assertIn(
            "symbolic_affine_exact_cosecant_cotangent_derivative_product_domain",
            names,
        )
        self.assertIn(
            "symbolic_external_scale_secant_tangent_derivative_product_domain",
            names,
        )
        self.assertIn(
            "symbolic_external_scale_cosecant_cotangent_derivative_product_domain",
            names,
        )
        self.assertIn(
            "negative_symbolic_affine_exact_secant_tangent_derivative_product_domain",
            names,
        )
        self.assertIn(
            "negative_symbolic_affine_exact_cosecant_cotangent_derivative_product_domain",
            names,
        )
        self.assertIn(
            "negative_symbolic_external_scale_secant_tangent_derivative_product_domain",
            names,
        )
        self.assertIn(
            "negative_symbolic_external_scale_cosecant_cotangent_derivative_product_domain",
            names,
        )
        self.assertIn("sqrt_chain_secant_tangent_domain", names)
        self.assertIn(
            "external_symbolic_scale_sqrt_chain_secant_tangent_domain",
            names,
        )
        self.assertIn(
            "external_symbolic_scale_sqrt_chain_cosecant_cotangent_domain",
            names,
        )
        self.assertIn(
            "external_symbolic_scale_shifted_sqrt_chain_secant_tangent_domain",
            names,
        )
        self.assertIn(
            "external_symbolic_scale_shifted_sqrt_chain_cosecant_cotangent_domain",
            names,
        )
        self.assertIn(
            "external_symbolic_scale_sqrt_minus_symbol_chain_secant_tangent_domain",
            names,
        )
        self.assertIn(
            "external_symbolic_scale_sqrt_minus_symbol_chain_cosecant_cotangent_domain",
            names,
        )
        self.assertIn(
            "sqrt_chain_hyperbolic_tangent_presimplified_condition_dedupe",
            names,
        )
        self.assertIn("shifted_sqrt_chain_hyperbolic_tangent_log_domain", names)
        self.assertIn("affine_shifted_sqrt_chain_hyperbolic_tangent_log_domain", names)
        self.assertIn(
            "sqrt_chain_hyperbolic_cosh_over_sinh_square_domain",
            names,
        )
        self.assertIn(
            "negative_shifted_sqrt_chain_hyperbolic_cosh_over_sinh_square_symbolic_scale_domain",
            names,
        )
        self.assertIn("invalid_log_base_integrand_undefined", names)
        self.assertIn("nonfinite_integrand_undefined", names)
        self.assertIn("non_elementary_exp_quadratic_residual", names)
        self.assertEqual(
            SMOKE.count_by(cases, "outcome"),
            {"residual": 15, "supported": 207, "undefined": 5},
        )
        self.assertEqual(
            sum(
                1
                for case in cases
                if (
                    case.expected_derivative_result is not None
                    or case.expected_derivative_equivalent_to is not None
                )
            ),
            134,
        )
        self.assertEqual(
            SMOKE.count_verification_regimes(cases),
            {
                "residual_not_verified": 15,
                "undefined_not_verified": 5,
                "definite_ftc_from_verified_antiderivative": 34,
                "verification_gap": 7,
                "verified_by_diff": 31,
                "verified_by_diff_and_direct_diff_integrate": 103,
                "verified_by_direct_diff_integrate": 32,
            },
        )
        self.assertEqual(SMOKE.count_verified_supported_cases(cases), 166)
        self.assertEqual(
            SMOKE.count_residual_causes(cases),
            {
                "branch_sensitive_interval_residual": 1,
                "non_elementary_composition": 7,
                "special_function_method_required": 6,
                "definite_interval_not_certifiable": 1,
            },
        )
        self.assertEqual(
            SMOKE.count_residual_families(cases),
            {
                "definite_integral_ftc_residual": 1,
                "explicit_reciprocal_trig_residual_domain": 7,
                "inverse_sqrt_additive_residual_domain": 1,
                "log_rational_residual": 1,
                "non_elementary_exp_quadratic": 1,
                "trig_additive_residual_domain": 1,
                "trig_residual_domain": 3,
            },
        )
        self.assertEqual(
            SMOKE.count_residual_cause_families(cases),
            {
                "branch_sensitive_interval_residual/inverse_sqrt_additive_residual_domain": 1,
                "definite_interval_not_certifiable/definite_integral_ftc_residual": 1,
                "non_elementary_composition/explicit_reciprocal_trig_residual_domain": 2,
                "non_elementary_composition/non_elementary_exp_quadratic": 1,
                "non_elementary_composition/trig_additive_residual_domain": 1,
                "non_elementary_composition/trig_residual_domain": 3,
                "special_function_method_required/explicit_reciprocal_trig_residual_domain": 5,
                "special_function_method_required/log_rational_residual": 1,
            },
        )
        self.assertNotIn(
            "unsupported_reciprocal_trig_method",
            SMOKE.count_residual_causes(cases),
        )
        self.assertEqual(
            SMOKE.group_residual_cases_by_cause(cases)[
                "special_function_method_required"
            ][1:],
            [
                "explicit_tangent_log_residual_condition_alias_dedupe",
                "explicit_tangent_log_numeric_shifted_residual_condition_alias_dedupe",
                "explicit_tangent_log_numeric_offset_residual_condition_alias_dedupe",
                "explicit_cotangent_log_numeric_offset_residual_condition_alias_dedupe",
                "explicit_cotangent_log_numeric_shifted_residual_condition_alias_dedupe",
            ],
        )
        self.assertIn(
            "non_elementary_exp_quadratic_residual",
            SMOKE.group_residual_cases_by_cause(cases)[
                "non_elementary_composition"
            ],
        )
        self.assertNotIn(
            "symbolic_radius_verification_gap",
            SMOKE.group_residual_cases_by_cause(cases),
        )
        step_checked = {
            case.name
            for case in cases
            if case.expected_step_substrings
        }
        supported_step_unchecked = {
            case.name
            for case in cases
            if case.outcome == "supported"
            and not case.expected_step_substrings
            and not SMOKE.is_algorithmic_backend_boundary_case(case)
        }
        self.assertEqual(
            step_checked,
            {
                "algorithmic_backend_hermite_reciprocal_educational_substeps",
                "algorithmic_backend_hermite_symbolic_positive_radius_mixed_numerator",
                "algorithmic_backend_hermite_symbolic_affine_positive_radius_mixed_numerator",
                "algorithmic_backend_hermite_expanded_symbolic_affine_positive_radius_mixed_numerator",
                "algorithmic_backend_hermite_expanded_symbolic_affine_derivative_multiple_numerator",
                "algorithmic_backend_hermite_expanded_numeric_center_derivative_multiple_numerator",
                "algorithmic_backend_rational_multi_quadratic_mixed_numerator",
                "algorithmic_backend_rational_multi_quadratic_triple_product",
                "algorithmic_backend_rational_ostrogradsky_expanded_sextic",
                "algorithmic_backend_rational_general_pole_condition",
                "algorithmic_backend_rational_sophie_germain_quartic",
                "algorithmic_backend_rational_cyclotomic_style_quartic",
                "algorithmic_backend_rational_resolvent_cubic_quartic",
                "algorithmic_backend_rational_resolvent_cubic_quartic_mixed_factors",
                "definite_integral_polynomial_exact_value",
                "definite_integral_arctan_exact_pi",
                "definite_integral_log_certified_off_pole",
                "definite_integral_symbolic_bound_area_function",
                "definite_integral_improper_convergent_reciprocal_square",
                "definite_integral_improper_full_line_arctan",
                "definite_integral_improper_divergent_polynomial",
                "definite_integral_improper_exponential_decay",
                "reciprocal_exponential_normalized_negative_exponent",
                "by_parts_exponential_normalized_div_linear",
                "definite_integral_improper_gamma_two",
                "definite_integral_improper_log_divergent_positive",
                "definite_integral_improper_log_divergent_negative",
                "cyclic_by_parts_damped_sine_normalized_div",
                "definite_integral_improper_damped_sine",
                "definite_integral_cofactor_sqrt_chain",
                "definite_integral_positive_polynomial_certified",
                "definite_integral_trig_pole_free_certified",
                "definite_integral_trig_pole_inside_undefined",
                "definite_integral_pi_bound_sec_squared",
                "definite_integral_pi_bound_sec_squared_offset",
                "definite_integral_pi_bound_pole_inside_undefined",
                "definite_integral_boundary_convergent_log",
                "definite_integral_boundary_convergent_inverse_sqrt",
                "definite_integral_boundary_convergent_sqrt_domain",
                "definite_integral_boundary_convergent_sqrt_integrand",
                "definite_integral_boundary_convergent_cube_root",
                "definite_integral_boundary_convergent_power_radical_product",
                "definite_integral_boundary_divergent_endpoint_pole",
                "definite_integral_fourier_orthogonality_sines",
                "definite_integral_fourier_sine_cosine_value",
                "definite_integral_pole_inside_interval_undefined",
                "polynomial_power_direct",
                "polynomial_sum_linearity",
                "affine_trig_substitution",
                "constant_multiple_affine_trig_substitution",
                "affine_exp_substitution",
                "linear_exp_by_parts",
                "linear_exp_affine_slope_by_parts",
                "polynomial_exp_derivative_substitution",
                "log_power_product_substitution",
                "constant_base_log_power_product_positive_domain_substitution",
                "reciprocal_affine_log_abs_domain",
                "reciprocal_negative_affine_log_abs_domain",
                "reciprocal_negative_affine_derivative_log_abs_domain",
                "log_derivative_positive_quadratic_substitution",
                "log_derivative_positive_quadratic_negative_orientation_substitution",
                "log_rational_positive_domain_residual",
                "non_elementary_tan_polynomial_residual_domain",
                "non_elementary_tan_presimplified_residual_domain",
                "non_elementary_csc_presimplified_residual_domain",
                "explicit_reciprocal_sine_presimplified_residual_domain",
                "explicit_reciprocal_sine_verified_log_domain",
                "explicit_reciprocal_cosine_verified_log_domain",
                "explicit_reciprocal_cosine_symbolic_shift_verified_log_domain",
                "explicit_reciprocal_cosine_symbolic_external_scale_shift_verified_log_domain",
                "explicit_reciprocal_tangent_presimplified_residual_domain",
                "explicit_tangent_log_residual_condition_alias_dedupe",
                "explicit_tangent_log_numeric_shifted_residual_condition_alias_dedupe",
                "explicit_tangent_log_numeric_offset_residual_condition_alias_dedupe",
                "explicit_cotangent_log_numeric_offset_residual_condition_alias_dedupe",
                "explicit_cotangent_log_numeric_shifted_residual_condition_alias_dedupe",
                "explicit_reciprocal_tangent_verified_log_domain",
                "explicit_reciprocal_tangent_presimplified_verified_log_domain",
                "explicit_reciprocal_secant_presimplified_source_domain",
                "symbolic_external_scale_tangent_log_derivative_ratio_domain",
                "symbolic_external_scale_cotangent_log_derivative_ratio_domain",
                "nested_tangent_log_derivative_denominator_verified_domain",
                "explicit_reciprocal_hyperbolic_tangent_verified_log_domain",
                "explicit_reciprocal_hyperbolic_tangent_presimplified_condition_dedupe",
                "symbolic_external_scale_reciprocal_hyperbolic_tangent_log_domain",
                "negative_orientation_symbolic_external_scale_reciprocal_hyperbolic_tangent_log_domain",
                "symbolic_external_scale_hyperbolic_tanh_log_derivative_ratio_positive",
                "symbolic_external_scale_hyperbolic_tanh_direct_log_positive",
                "additive_trig_pole_residual_domain",
                "inverse_trig_table",
                "inverse_trig_affine_shifted_scaled_positive_quadratic_table",
                "inverse_trig_symbolic_affine_expanded_square_radius_positive_quadratic_table",
                "inverse_trig_symbolic_affine_positive_rational_radius_positive_quadratic_table",
                "inverse_trig_named_positive_constant_radius_positive_quadratic_table",
                "inverse_trig_numeric_affine_named_positive_constant_radius_positive_quadratic_table",
                "inverse_trig_expanded_numeric_affine_named_positive_constant_radius_positive_quadratic_table",
                "rational_positive_quadratic_linear_numerator_expanded_named_positive_radius_decomposition",
                "inverse_trig_symbolic_square_radius_positive_quadratic_table",
                "inverse_trig_symbolic_shifted_square_radius_positive_quadratic_table",
                "affine_secant_table_log_domain",
                "affine_cosecant_table_log_domain",
                "external_constant_affine_secant_table_log_domain",
                "rational_denominator_scaled_affine_secant_table_log_domain",
                "rational_denominator_scaled_affine_cosecant_table_log_domain",
                "rational_positive_quadratic_square_reduction",
                "inverse_trig_sqrt_reciprocal_bridge",
                "inverse_trig_scaled_sqrt_reciprocal_bridge",
                "inverse_trig_symbolic_denominator_scale_sqrt_reciprocal_bridge",
                "inverse_trig_symbolic_denominator_numeric_square_scale_sqrt_reciprocal_bridge",
                "inverse_trig_symbolic_numerator_scale_sqrt_reciprocal_bridge",
                "inverse_trig_shifted_scaled_sqrt_reciprocal_bridge",
                "inverse_hyperbolic_rational_direct_atanh_domain",
                "rational_partial_fraction_two_real_linear_factors",
                "rational_partial_fraction_three_real_linear_factors",
                "rational_partial_fraction_mixed_simple_repeated_linear_factors",
                "rational_partial_fraction_repeated_real_linear_factors",
                "rational_partial_fraction_mixed_linear_positive_quadratic",
                "rational_partial_fraction_repeated_linear_positive_quadratic",
                "rational_partial_fraction_repeated_origin_linear_positive_quadratic_no_log",
                "rational_partial_fraction_repeated_origin_scaled_positive_quadratic_no_log",
                "rational_improper_partial_fraction_polynomial_division",
                "rational_improper_positive_quadratic_polynomial_division",
                "rational_improper_positive_quadratic_negative_orientation",
                "affine_inverse_hyperbolic_atanh_domain",
                "inverse_sqrt_direct_arcsin_domain",
                "additive_inverse_sqrt_interval_residual_domain",
                "inverse_sqrt_symbolic_radius_arcsin_domain",
                "inverse_sqrt_symbolic_radius_shifted_arcsin_domain",
                "inverse_sqrt_symbolic_slope_shifted_arcsin_domain",
                "inverse_sqrt_direct_asinh_unconditional",
                "inverse_hyperbolic_sqrt_symbolic_radius_table",
                "inverse_hyperbolic_sqrt_symbolic_shifted_radius_table",
                "inverse_hyperbolic_sqrt_symbolic_slope_shifted_radius_table",
                "affine_inverse_sqrt_arcsin_domain",
                "polynomial_base_sqrt_substitution",
                "hyperbolic_reciprocal_square_substitution",
                "hyperbolic_sine_reciprocal_square_substitution",
                "negative_affine_hyperbolic_sine_reciprocal_square_substitution",
                "symbolic_affine_exact_hyperbolic_cosh_reciprocal_square_substitution",
                "symbolic_affine_exact_hyperbolic_sinh_reciprocal_square_substitution",
                "symbolic_external_scale_shifted_hyperbolic_cosh_reciprocal_square_substitution",
                "symbolic_external_scale_shifted_hyperbolic_sinh_reciprocal_square_substitution",
                "hyperbolic_cosh_reciprocal_fourth_substitution",
                "symbolic_external_scale_shifted_hyperbolic_cosh_reciprocal_fourth_substitution",
                "hyperbolic_sinh_reciprocal_fourth_substitution",
                "symbolic_external_scale_shifted_hyperbolic_sinh_reciprocal_fourth_substitution",
                "symbolic_affine_exact_hyperbolic_sinh_over_cosh_square_substitution",
                "symbolic_affine_exact_hyperbolic_cosh_over_sinh_square_substitution",
                "symbolic_external_scale_hyperbolic_sinh_over_cosh_square_substitution",
                "symbolic_external_scale_hyperbolic_cosh_over_sinh_square_substitution",
                "polynomial_shifted_hyperbolic_sinh_over_cosh_square_substitution",
                "symbolic_external_scale_shifted_hyperbolic_sinh_over_cosh_square_substitution",
                "symbolic_external_scale_shifted_hyperbolic_cosh_over_sinh_square_substitution",
                "by_parts_log_domain",
                "by_parts_monomial_arcsine_domain",
                "by_parts_monomial_arccosine_domain",
                "hermite_split_cross_terms_over_acosh_radical",
                "hermite_split_square_over_completed_square",
                "hermite_split_square_over_circle_shifted",
                "linear_over_sqrt_completed_square_asinh",
                "linear_over_sqrt_completed_square_arcsin",
                "monomial_over_sqrt_hyperbolic_asinh",
                "monomial_over_sqrt_hyperbolic_acosh",
                "by_parts_square_arcsine_radical_tail",
                "by_parts_monomial_scaled_arcsine_radical_tail",
                "monomial_over_sqrt_reduction_square",
                "monomial_over_sqrt_reduction_scaled_radicand",
                "definite_integral_monomial_over_sqrt_quarter_pi",
                "by_parts_log_negative_power_quotient",
                "by_parts_log_fractional_power_radical",
                "by_parts_log_square_negative_power_quotient",
                "exp_substitution_gaussian_damped_indefinite",
                "exp_substitution_cubic_damped_indefinite",
                "definite_integral_improper_gaussian_damped",
                "definite_integral_improper_scaled_arctan_pi",
                "multiple_angle_cosine_product_chebyshev",
                "sine_multiple_angle_ratio_quotient_surface",
                "definite_integral_fourier_orthogonality_cosines",
                "definite_integral_arccosine_unit_interval",
                "definite_integral_monomial_arcsine_unit_interval",
                "cotangent_odd_cube_reduction",
                "definite_integral_tan_cube_pi_bound",
                "tangent_odd_cube_reduction",
                "reciprocal_cosine_odd_cube_affine_reduction",
                "reciprocal_sine_odd_cube_reduction",
                "definite_integral_sec_cube_pi_bound",
                "reciprocal_cosine_odd_cube_reduction",
                "by_parts_affine_log_domain",
                "affine_secant_tangent_derivative_product_domain",
                "affine_cosecant_cotangent_derivative_product_domain",
                "negative_affine_secant_tangent_derivative_product_domain",
                "negative_affine_cosecant_cotangent_derivative_product_domain",
                "polynomial_shifted_secant_tangent_derivative_product_domain",
                "polynomial_shifted_cosecant_cotangent_derivative_product_domain",
                "symbolic_affine_exact_secant_tangent_derivative_product_domain",
                "symbolic_affine_exact_cosecant_cotangent_derivative_product_domain",
                "symbolic_external_scale_secant_tangent_derivative_product_domain",
                "symbolic_external_scale_cosecant_cotangent_derivative_product_domain",
                "negative_symbolic_affine_exact_secant_tangent_derivative_product_domain",
                "negative_symbolic_affine_exact_cosecant_cotangent_derivative_product_domain",
                "negative_symbolic_external_scale_secant_tangent_derivative_product_domain",
                "negative_symbolic_external_scale_cosecant_cotangent_derivative_product_domain",
                "sqrt_chain_secant_tangent_domain",
                "external_symbolic_scale_sqrt_chain_secant_tangent_domain",
                "external_symbolic_scale_sqrt_chain_cosecant_cotangent_domain",
                "external_symbolic_scale_shifted_sqrt_chain_secant_tangent_domain",
                "external_symbolic_scale_shifted_sqrt_chain_cosecant_cotangent_domain",
                "external_symbolic_scale_sqrt_minus_symbol_chain_secant_tangent_domain",
                "external_symbolic_scale_sqrt_minus_symbol_chain_cosecant_cotangent_domain",
                "sqrt_chain_tangent_log_domain",
                "shifted_sqrt_chain_tangent_log_domain",
                "sqrt_chain_hyperbolic_tangent_presimplified_condition_dedupe",
                "shifted_sqrt_chain_hyperbolic_tangent_log_domain",
                "affine_shifted_sqrt_chain_hyperbolic_tangent_log_domain",
                "sqrt_chain_hyperbolic_cosh_over_sinh_square_domain",
                "shifted_sqrt_chain_hyperbolic_sinh_over_cosh_square_symbolic_scale_domain",
                "negative_shifted_sqrt_chain_hyperbolic_cosh_over_sinh_square_symbolic_scale_domain",
                "invalid_log_base_integrand_undefined",
                "nonfinite_integrand_undefined",
                "non_elementary_exp_quadratic_residual",
            },
        )
        self.assertEqual(supported_step_unchecked, set())
        self.assertEqual(
            SMOKE.count_by(cases, "domain_regime"),
            {
                "affine_shifted_sqrt_chain_nonzero_positive": 1,
                "backend_verified_component_local_symbolic_poles_and_slope_required": 1,
                "backend_verified_denominator_and_slope_nonzero_required": 2,
                "backend_verified_denominator_nonzero_required": 1,
                "backend_verified_component_local_symbolic_poles_required": 1,
                "backend_verified_positive_radius_and_slope_nonzero_required": 3,
                "backend_verified_unconditional_real": 5,
                "backend_verified_conservative_denominator_nonzero": 2,
                "interval_certified_unconditional": 5,
                "interval_certified_trig_discharged": 5,
                "bounded_interval_required": 4,
                "boundary_touch_one_sided_limit": 7,
                "improper_interval_certified": 7,
                "improper_divergent_to_infinity": 4,
                "total_real_function": 9,
                "symbolic_bounds_condition_free": 4,
                "interval_certified_with_source_condition": 2,
                "interval_pole_divergent": 3,
                "interval_not_certifiable": 1,
                "backend_verified_pole_nonzero_required": 1,
                "backend_verified_positive_radius_required": 4,
                "empty_real_domain": 1,
                "explicit_denominator_source_condition": 1,
                "explicit_hyperbolic_tangent_presimplified_condition_dedupe": 1,
                "explicit_hyperbolic_tangent_denominator_verified_substitution": 1,
                "explicit_reciprocal_cosine_verified_substitution": 1,
                "explicit_reciprocal_cosine_symbolic_shift_verified_substitution": 1,
                "explicit_reciprocal_cosine_symbolic_external_scale_shift_verified_substitution": 1,
                "explicit_reciprocal_sine_verified_substitution": 1,
                "explicit_reciprocal_trig_source_defined_condition": 1,
                "explicit_tangent_denominator_source_condition": 1,
                "explicit_tangent_denominator_verified_substitution": 1,
                "explicit_tangent_log_shifted_condition_dedupe": 1,
                "explicit_tangent_log_numeric_shifted_condition_dedupe": 1,
                "explicit_tangent_log_numeric_offset_condition_dedupe": 1,
                "explicit_cotangent_log_numeric_offset_condition_dedupe": 1,
                "explicit_cotangent_log_numeric_shifted_condition_dedupe": 1,
                "explicit_tangent_presimplified_condition_dedupe": 1,
                "hyperbolic_sine_pole_required": 11,
                "linear_poles_required": 9,
                "open_interval_required": 6,
                "nonzero_required": 9,
                "nonfinite_undefined": 1,
                "positive_log_argument_required": 1,
                "positive_log_argument_and_trig_poles": 1,
                "positive_required": 8,
                "positive_required_residual": 1,
                "radical_interval_additive_residual": 1,
                "radical_interval": 2,
                "rational_interval": 2,
                "shifted_sqrt_chain_nonzero_positive": 5,
                "shifted_sqrt_chain_positive": 1,
                "symbolic_denominator_scale_positive_required": 2,
                "symbolic_radius_nonzero_required": 3,
                "symbolic_scale_nonzero_required": 2,
                "symbolic_positive_square_radius_verified": 1,
                "symbolic_slope_positive_square_radius_verified": 1,
                "symbolic_slope_shifted_radical_interval_verified": 1,
                "symbolic_radical_interval_verified": 1,
                "symbolic_shifted_radical_interval_verified": 1,
                "symbolic_numerator_scale_positive_required": 1,
                "sqrt_minus_symbol_chain_nonzero_positive": 2,
                "sqrt_chain_nonzero_positive": 4,
                "sqrt_chain_hyperbolic_presimplified_condition_dedupe": 1,
                "sqrt_chain_hyperbolic_sine_pole_required": 1,
                "structurally_nonzero_negative_quadratic_denominator": 1,
                "structurally_positive_denominator": 7,
                "structurally_positive_log_argument": 3,
                "trig_pole_additive_residual": 1,
                "trig_pole_presimplified_residual": 1,
                "trig_pole_residual": 1,
                "trig_log_derivative_pole_required": 2,
                "trig_reciprocal_table_pole_required": 5,
                "trig_reciprocal_product_pole_required": 6,
                "trig_reciprocal_product_exact_symbolic_derivative_pole_required": 8,
                "trig_sine_pole_presimplified_residual": 1,
                "unconditional": 21,
                "unconditional_cosh_positive": 2,
            },
        )
        self.assertEqual(
            SMOKE.count_trig_hyperbolic_policy_clusters(cases),
            {
                "block7_explicit_reciprocal_hyperbolic_tangent": 4,
                "block7_explicit_reciprocal_trig_log_substitution": 7,
                "block7_hyperbolic_reciprocal_derivative_product": 7,
                "block7_hyperbolic_log_derivative_ratio": 1,
                "block7_hyperbolic_tanh_log_derivative": 1,
                "block7_hyperbolic_reciprocal_fourth": 4,
                "block7_hyperbolic_reciprocal_square": 7,
                "block7_trig_log_derivative_ratio": 2,
                "block7_sqrt_chain_hyperbolic_reciprocal_derivative_product": 3,
                "block7_sqrt_chain_hyperbolic_log": 3,
                "block7_sqrt_chain_reciprocal_trig_product": 7,
                "block7_sqrt_chain_trig_log": 2,
                "block7_trig_reciprocal_derivative_product": 14,
                "block7_trig_reciprocal_log_table": 5,
                "block9_explicit_reciprocal_trig_residual": 7,
            },
        )
        self.assertEqual(
            SMOKE.count_radical_inverse_policy_clusters(cases),
            {
                "block8_bounded_inverse_trig_by_parts": 4,
                "block8_inverse_hyperbolic_rational_interval": 2,
                "block8_inverse_sqrt_tables": 9,
                "block8_inverse_trig_root_reciprocal": 6,
                "block8_linear_over_sqrt_shifted_quadratic": 2,
                "block8_polynomial_over_sqrt_hermite_split": 3,
                "block8_monomial_over_sqrt_reduction": 4,
            },
        )
        self.assertEqual(
            SMOKE.count_base_integration_policy_clusters(cases),
            {
                "block4_exponential_by_parts": 2,
                "block4_log_by_parts": 5,
                "block4_log_power_product_by_parts": 2,
            },
        )
        self.assertEqual(
            SMOKE.count_calculus_maturity_blocks(cases),
            {
                "block4_base_integration": 21,
                "block5_generalized_substitution": 13,
                "block6_rational_integration": 22,
                "block7_trig_hyperbolic_integration": 67,
                "block8_radical_inverse_families": 30,
                "block9_residuals_and_non_goals": 16,
                "block12_hybrid_algorithmic_backend": 20,
                "block13_definite_integrals": 38,
            },
        )
        self.assertEqual(
            SMOKE.count_calculus_block_gates(cases),
            {
                "algorithmic_backend_boundary_verified": 20,
                "didactic_trace_and_verified_antiderivative": 63,
                "domain_conditions_and_verified_antiderivative": 124,
                "explicit_undefined_domain_policy": 5,
                "safe_residual_policy": 15,
            },
        )
        self.assertGreaterEqual(len({case.family for case in cases}), 10)

    def test_direct_arctan_table_rows_count_as_rational_integration(self) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}

        for name in (
            "inverse_trig_table",
            "inverse_trig_affine_shifted_scaled_positive_quadratic_table",
            "inverse_trig_symbolic_affine_expanded_square_radius_positive_quadratic_table",
            "inverse_trig_symbolic_affine_positive_rational_radius_positive_quadratic_table",
            "inverse_trig_named_positive_constant_radius_positive_quadratic_table",
            "inverse_trig_numeric_affine_named_positive_constant_radius_positive_quadratic_table",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    SMOKE.calculus_maturity_block(cases[name]),
                    "block6_rational_integration",
                )

    def test_symbolic_positive_rational_radius_arctan_tracks_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases[
            "inverse_trig_symbolic_affine_positive_rational_radius_positive_quadratic_table"
        ]

        self.assertEqual(
            case.expected_direct_diff_integrate_result,
            "1 / ((a·x + b)^2 + 2)",
        )
        self.assertEqual(
            case.expected_direct_diff_integrate_required_display,
            ("a ≠ 0",),
        )

    def test_named_positive_constant_radius_arctan_tracks_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases[
            "inverse_trig_named_positive_constant_radius_positive_quadratic_table"
        ]

        self.assertEqual(case.expected_result, "arctan(x·phi^(-1/2)) / sqrt(phi)")
        self.assertEqual(
            case.expected_direct_diff_integrate_result,
            "1 / (x^2 + phi)",
        )
        self.assertEqual(case.expected_direct_diff_integrate_required_display, ())

    def test_positive_quadratic_arctan_table_tracks_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}

        expectations = {
            "inverse_trig_table": ("1 / (x^2 + 1)", ()),
            "inverse_trig_affine_shifted_scaled_positive_quadratic_table": (
                "1 / ((2·x + 3)^2 + 4)",
                (),
            ),
            "inverse_trig_symbolic_affine_expanded_square_radius_positive_quadratic_table": (
                "1 / ((a·x + b)^2 + 4)",
                ("a ≠ 0",),
            ),
            "inverse_trig_symbolic_square_radius_positive_quadratic_table": (
                "1 / (a^2 + x^2)",
                ("a ≠ 0",),
            ),
        }

        for name, (expected_result, expected_required_display) in expectations.items():
            with self.subTest(name=name):
                case = cases[name]
                self.assertEqual(
                    case.expected_direct_diff_integrate_result,
                    expected_result,
                )
                self.assertIsNone(case.expected_direct_diff_integrate_equivalent_to)
                self.assertEqual(
                    case.expected_direct_diff_integrate_required_display,
                    expected_required_display,
                )

    def test_numeric_affine_named_positive_constant_radius_arctan_is_verified(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases[
            "inverse_trig_numeric_affine_named_positive_constant_radius_positive_quadratic_table"
        ]

        self.assertEqual(
            case.expected_result,
            "arctan(phi^(-1/2)·(2·x + 3)) / (2·sqrt(phi))",
        )
        self.assertEqual(
            case.expected_derivative_equivalent_to,
            "1/((2*x+3)^2+phi)",
        )
        self.assertEqual(
            case.expected_direct_diff_integrate_result,
            "1 / ((2·x + 3)^2 + phi)",
        )
        self.assertEqual(case.expected_direct_diff_integrate_required_display, ())
        self.assertEqual(case.expected_required_display, ())

    def test_linear_numerator_named_positive_constant_radius_arctan_is_verified(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases[
            "rational_positive_quadratic_linear_numerator_expanded_named_positive_radius_decomposition"
        ]

        self.assertEqual(
            case.expected_result,
            "1/4·ln(4·x^2 + 12·x + 9 + phi) + (atan((2·x + 3) / sqrt(phi))·3)/(2·sqrt(phi))",
        )
        self.assertEqual(
            case.expected_derivative_equivalent_to,
            "(2*x+6)/(4*x^2+12*x+9+phi)",
        )
        self.assertEqual(
            case.expected_direct_diff_integrate_result,
            "(2·x + 6) / (4·x^2 + 12·x + 9 + phi)",
        )
        self.assertEqual(case.expected_direct_diff_integrate_required_display, ())
        self.assertEqual(case.expected_required_display, ())

    def test_repeated_linear_partial_fraction_tracks_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases["rational_partial_fraction_repeated_real_linear_factors"]

        self.assertEqual(
            case.expected_direct_diff_integrate_result,
            "1 / (x^2 - 1)^2",
        )
        self.assertEqual(
            case.expected_direct_diff_integrate_required_display,
            ("x ≠ -1", "x ≠ 1"),
        )

    def test_inverse_sqrt_arcsin_tracks_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases["inverse_sqrt_direct_arcsin_domain"]

        self.assertEqual(
            case.expected_direct_diff_integrate_result,
            "1 / sqrt(1 - x^2)",
        )
        self.assertEqual(
            case.expected_direct_diff_integrate_required_display,
            ("-1 < x < 1",),
        )

    def test_symbolic_shifted_inverse_sqrt_arcsin_tracks_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases["inverse_sqrt_symbolic_radius_shifted_arcsin_domain"]

        self.assertEqual(
            case.expected_result,
            "arcsin((b + x) / sqrt(a^2))",
        )
        self.assertEqual(
            case.expected_direct_diff_integrate_result,
            "1 / sqrt(a^2 - (b + x)^2)",
        )
        self.assertEqual(
            case.expected_direct_diff_integrate_required_display,
            ("a^2 - b^2 - x^2 - 2·b·x > 0",),
        )

    def test_symbolic_slope_shifted_inverse_sqrt_arcsin_tracks_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases["inverse_sqrt_symbolic_slope_shifted_arcsin_domain"]

        self.assertEqual(
            case.expected_result,
            "arcsin((m·x + b) / sqrt(a^2)) / m",
        )
        self.assertEqual(
            case.expected_direct_diff_integrate_result,
            "1 / sqrt(a^2 - (m·x + b)^2)",
        )
        self.assertEqual(
            case.expected_direct_diff_integrate_required_display,
            (
                "a^2 - m^2·x^2 - 2·b·m·x - b^2 > 0",
                "m ≠ 0",
            ),
        )

    def test_affine_inverse_sqrt_arcsin_tracks_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases["affine_inverse_sqrt_arcsin_domain"]

        self.assertEqual(
            case.expected_direct_diff_integrate_result,
            "1 / sqrt(4 - (x + 1)^2)",
        )
        self.assertEqual(
            case.expected_direct_diff_integrate_required_display,
            ("-3 < x < 1",),
        )

    def test_inverse_trig_root_reciprocal_cluster_tracks_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}

        expected = {
            "inverse_trig_sqrt_reciprocal_bridge": (
                "1 / (sqrt(x)·(x + 1))",
                ("x > 0",),
            ),
            "inverse_trig_scaled_sqrt_reciprocal_bridge": (
                "1 / (sqrt(x)·(4·x + 1))",
                ("x > 0",),
            ),
            "inverse_trig_symbolic_denominator_scale_sqrt_reciprocal_bridge": (
                "1 / (sqrt(x)·(a^2 + x))",
                ("a ≠ 0", "x > 0"),
            ),
            "inverse_trig_symbolic_denominator_numeric_square_scale_sqrt_reciprocal_bridge": (
                "1 / (sqrt(x)·(a^2 + 4·x))",
                ("a ≠ 0", "x > 0"),
            ),
            "inverse_trig_symbolic_numerator_scale_sqrt_reciprocal_bridge": (
                "1 / (sqrt(x)·(x·a^2 + 1))",
                ("a ≠ 0", "x > 0"),
            ),
            "inverse_trig_shifted_scaled_sqrt_reciprocal_bridge": (
                "1 / (sqrt(x + 1)·(4·x + 5))",
                ("x > -1",),
            ),
        }

        for name, (result, required_display) in expected.items():
            with self.subTest(name=name):
                case = cases[name]
                self.assertEqual(case.expected_direct_diff_integrate_result, result)
                self.assertEqual(
                    case.expected_direct_diff_integrate_required_display,
                    required_display,
                )

    def test_inverse_hyperbolic_rational_interval_tracks_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}

        expected = {
            "inverse_hyperbolic_rational_direct_atanh_domain": (
                "1 / (1 - x^2)",
                ("-1 < x < 1",),
            ),
            "affine_inverse_hyperbolic_atanh_domain": (
                "2 / (4 - (2·x + 1)^2)",
                ("-3/2 < x < 1/2",),
            ),
        }

        for name, (result, required_display) in expected.items():
            with self.subTest(name=name):
                case = cases[name]
                self.assertEqual(case.expected_direct_diff_integrate_result, result)
                self.assertEqual(
                    case.expected_direct_diff_integrate_required_display,
                    required_display,
                )

    def test_inverse_sqrt_asinh_tracks_direct_diff_integrate(self) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases["inverse_sqrt_direct_asinh_unconditional"]

        self.assertEqual(case.expected_result, "asinh(x)")
        self.assertEqual(
            case.expected_direct_diff_integrate_result,
            "1 / sqrt(x^2 + 1)",
        )
        self.assertEqual(case.expected_direct_diff_integrate_required_display, ())

    def test_hyperbolic_reciprocal_square_tracks_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases["hyperbolic_reciprocal_square_substitution"]

        self.assertEqual(
            case.expected_direct_diff_integrate_result,
            "1 / cosh(2·x + 1)^2",
        )
        self.assertEqual(case.expected_direct_diff_integrate_required_display, ())

    def test_hyperbolic_sine_reciprocal_square_tracks_exact_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases["hyperbolic_sine_reciprocal_square_substitution"]

        self.assertEqual(
            case.expected_direct_diff_integrate_result,
            "1 / sinh(2·x + 1)^2",
        )
        self.assertIsNone(case.expected_direct_diff_integrate_equivalent_to)
        self.assertEqual(
            case.expected_direct_diff_integrate_required_display,
            ("sinh(2·x + 1) ≠ 0",),
        )

    def test_negative_affine_hyperbolic_sine_reciprocal_square_tracks_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases["negative_affine_hyperbolic_sine_reciprocal_square_substitution"]

        self.assertEqual(
            case.expected_result,
            "1 / (2·tanh(1 - 2·x))",
        )
        self.assertEqual(
            case.expected_direct_diff_integrate_result,
            "1 / sinh(1 - 2·x)^2",
        )
        self.assertIsNone(case.expected_direct_diff_integrate_equivalent_to)
        self.assertEqual(
            case.expected_direct_diff_integrate_required_display,
            ("sinh(2·x - 1) ≠ 0",),
        )

    def test_hyperbolic_reciprocal_fourth_tracks_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}

        self.assertEqual(
            cases["hyperbolic_cosh_reciprocal_fourth_substitution"].expected_direct_diff_integrate_result,
            "1 / cosh(2·x + 1)^4",
        )
        self.assertEqual(
            cases["symbolic_external_scale_shifted_hyperbolic_cosh_reciprocal_fourth_substitution"].expected_direct_diff_integrate_result,
            "2·k·x / cosh(x^2 + b)^4",
        )

        sinh_case = cases["hyperbolic_sinh_reciprocal_fourth_substitution"]
        self.assertEqual(
            sinh_case.expected_direct_diff_integrate_result,
            "1 / sinh(2·x + 1)^4",
        )
        self.assertEqual(
            sinh_case.expected_direct_diff_integrate_required_display,
            ("sinh(2·x + 1) ≠ 0",),
        )

        shifted_sinh_case = cases[
            "symbolic_external_scale_shifted_hyperbolic_sinh_reciprocal_fourth_substitution"
        ]
        self.assertEqual(
            shifted_sinh_case.expected_direct_diff_integrate_result,
            "2·k·x / sinh(x^2 + b)^4",
        )
        self.assertEqual(
            shifted_sinh_case.expected_direct_diff_integrate_required_display,
            ("sinh(x^2 + b) ≠ 0",),
        )

    def test_trig_reciprocal_product_symbolic_and_polynomial_track_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}

        symbolic_sec = cases["symbolic_affine_exact_secant_tangent_derivative_product_domain"]
        self.assertEqual(
            symbolic_sec.expected_direct_diff_integrate_result,
            "a·tan(a·x + b)·sec(a·x + b)",
        )
        self.assertEqual(
            symbolic_sec.expected_direct_diff_integrate_required_display,
            ("cos(a·x + b) ≠ 0",),
        )

        symbolic_csc = cases[
            "symbolic_affine_exact_cosecant_cotangent_derivative_product_domain"
        ]
        self.assertEqual(
            symbolic_csc.expected_direct_diff_integrate_result,
            "a·csc(a·x + b)·cot(a·x + b)",
        )
        self.assertEqual(
            symbolic_csc.expected_direct_diff_integrate_required_display,
            ("sin(a·x + b) ≠ 0",),
        )

        symbolic_external_sec = cases[
            "symbolic_external_scale_secant_tangent_derivative_product_domain"
        ]
        self.assertEqual(
            symbolic_external_sec.expected_direct_diff_integrate_result,
            "a·k·tan(a·x + b)·sec(a·x + b)",
        )
        self.assertEqual(
            symbolic_external_sec.expected_direct_diff_integrate_required_display,
            ("cos(a·x + b) ≠ 0",),
        )

        negative_symbolic_sec = cases[
            "negative_symbolic_affine_exact_secant_tangent_derivative_product_domain"
        ]
        self.assertEqual(
            negative_symbolic_sec.expected_direct_diff_integrate_result,
            "-tan(b - a·x)·sec(b - a·x)·a",
        )
        self.assertEqual(
            negative_symbolic_sec.expected_direct_diff_integrate_required_display,
            ("cos(b - a·x) ≠ 0",),
        )

        negative_symbolic_csc = cases[
            "negative_symbolic_affine_exact_cosecant_cotangent_derivative_product_domain"
        ]
        self.assertEqual(
            negative_symbolic_csc.expected_direct_diff_integrate_result,
            "-csc(b - a·x)·cot(b - a·x)·a",
        )
        self.assertEqual(
            negative_symbolic_csc.expected_direct_diff_integrate_required_display,
            ("sin(b - a·x) ≠ 0",),
        )

        negative_symbolic_external_csc = cases[
            "negative_symbolic_external_scale_cosecant_cotangent_derivative_product_domain"
        ]
        self.assertEqual(
            negative_symbolic_external_csc.expected_direct_diff_integrate_result,
            "a·csc(b - a·x)·-cot(b - a·x)·k",
        )
        self.assertEqual(
            negative_symbolic_external_csc.expected_direct_diff_integrate_required_display,
            ("sin(b - a·x) ≠ 0",),
        )

        polynomial_sec = cases["polynomial_shifted_secant_tangent_derivative_product_domain"]
        self.assertEqual(
            polynomial_sec.expected_direct_diff_integrate_result,
            "2·x·tan(x^2 + b)·sec(x^2 + b)",
        )
        self.assertIsNone(polynomial_sec.expected_direct_diff_integrate_equivalent_to)
        self.assertEqual(
            polynomial_sec.expected_direct_diff_integrate_required_display,
            ("cos(x^2 + b) ≠ 0",),
        )

        polynomial_csc = cases[
            "polynomial_shifted_cosecant_cotangent_derivative_product_domain"
        ]
        self.assertEqual(
            polynomial_csc.expected_direct_diff_integrate_result,
            "2·x·csc(x^2 + b)·cot(x^2 + b)",
        )
        self.assertIsNone(polynomial_csc.expected_direct_diff_integrate_equivalent_to)
        self.assertEqual(
            polynomial_csc.expected_direct_diff_integrate_required_display,
            ("sin(x^2 + b) ≠ 0",),
        )

    def test_direct_diff_integrate_coverage_is_visible_by_block_and_cluster(
        self,
    ) -> None:
        cases = SMOKE.build_cases()
        cases_by_name = {case.name: case for case in cases}

        backend_case = cases_by_name[
            "algorithmic_backend_rational_affine_quotient_numeric_slope"
        ]
        self.assertEqual(
            backend_case.expected_direct_diff_integrate_result,
            "(c + 3·x) / (b + 2·x)",
        )
        self.assertEqual(
            backend_case.expected_direct_diff_integrate_required_display,
            ("b + 2·x ≠ 0",),
        )
        symbolic_backend_case = cases_by_name[
            "algorithmic_backend_rational_affine_quotient_symbolic_slope"
        ]
        self.assertEqual(
            symbolic_backend_case.expected_direct_diff_integrate_result,
            "(c + 3·x) / (2·a·x + b)",
        )
        self.assertEqual(
            symbolic_backend_case.expected_direct_diff_integrate_required_display,
            ("2·a·x + b ≠ 0", "a ≠ 0"),
        )
        external_scale_backend_case = cases_by_name[
            "algorithmic_backend_rational_affine_quotient_external_scale_zero_intercept"
        ]
        self.assertEqual(
            external_scale_backend_case.expected_direct_diff_integrate_result,
            "a·x / (c·x + d)",
        )
        self.assertEqual(
            external_scale_backend_case.expected_direct_diff_integrate_required_display,
            ("c ≠ 0", "c·x + d ≠ 0"),
        )
        hermite_backend_case = cases_by_name[
            "algorithmic_backend_hermite_symbolic_positive_radius_mixed_numerator"
        ]
        self.assertEqual(
            hermite_backend_case.expected_direct_diff_integrate_result,
            "(x + 1) / (x^2 + a)",
        )
        self.assertEqual(
            hermite_backend_case.expected_direct_diff_integrate_required_display,
            ("a > 0",),
        )
        affine_hermite_backend_case = cases_by_name[
            "algorithmic_backend_hermite_symbolic_affine_positive_radius_mixed_numerator"
        ]
        self.assertEqual(
            affine_hermite_backend_case.expected_direct_diff_integrate_result,
            "(m·(s·x + b) + c) / ((s·x + b)^2 + a)",
        )
        self.assertEqual(
            affine_hermite_backend_case.expected_direct_diff_integrate_required_display,
            ("a > 0", "s ≠ 0"),
        )

        self.assertEqual(
            SMOKE.count_direct_diff_integrate_calculus_maturity_blocks(cases),
            {
                "block12_hybrid_algorithmic_backend": 20,
                "block4_base_integration": 14,
                "block5_generalized_substitution": 13,
                "block6_rational_integration": 22,
                "block7_trig_hyperbolic_integration": 37,
                "block8_radical_inverse_families": 29,
            },
        )
        self.assertEqual(
            SMOKE.count_direct_diff_integrate_calculus_block_gates(cases),
            {
                "algorithmic_backend_boundary_verified": 20,
                "didactic_trace_and_verified_antiderivative": 38,
                "domain_conditions_and_verified_antiderivative": 77,
            },
        )
        self.assertEqual(
            SMOKE.count_direct_diff_integrate_base_integration_policy_clusters(cases),
            {
                "block4_exponential_by_parts": 2,
                "block4_log_by_parts": 5,
                "block4_log_power_product_by_parts": 2,
            },
        )
        self.assertEqual(
            SMOKE.count_direct_diff_integrate_radical_inverse_policy_clusters(cases),
            {
                "block8_bounded_inverse_trig_by_parts": 4,
                "block8_inverse_hyperbolic_rational_interval": 2,
                "block8_inverse_sqrt_tables": 9,
                "block8_inverse_trig_root_reciprocal": 6,
                "block8_linear_over_sqrt_shifted_quadratic": 2,
                "block8_polynomial_over_sqrt_hermite_split": 2,
                "block8_monomial_over_sqrt_reduction": 4,
            },
        )
        self.assertEqual(
            SMOKE.count_direct_diff_integrate_trig_hyperbolic_policy_clusters(cases),
            {
                "block7_hyperbolic_reciprocal_derivative_product": 7,
                "block7_hyperbolic_reciprocal_fourth": 4,
                "block7_hyperbolic_reciprocal_square": 7,
                "block7_trig_reciprocal_derivative_product": 14,
                "block7_trig_reciprocal_log_table": 5,
            },
        )
        self.assertEqual(len(SMOKE.direct_diff_integrate_exact_cases(cases)), 115)
        self.assertEqual(len(SMOKE.direct_diff_integrate_equivalence_cases(cases)), 20)
        self.assertEqual(
            len(SMOKE.derivative_verified_without_direct_diff_integrate_cases(cases)),
            31,
        )
        self.assertEqual(
            SMOKE.count_direct_diff_integrate_gap_calculus_maturity_blocks(cases),
            {
                "block7_trig_hyperbolic_integration": 30,
                "block8_radical_inverse_families": 1,
            },
        )
        self.assertEqual(
            SMOKE.count_direct_diff_integrate_gap_calculus_block_gates(cases),
            {
                "didactic_trace_and_verified_antiderivative": 3,
                "domain_conditions_and_verified_antiderivative": 28,
            },
        )
        self.assertEqual(
            SMOKE.count_direct_diff_integrate_gap_base_integration_policy_clusters(
                cases
            ),
            {},
        )
        self.assertEqual(
            SMOKE.count_direct_diff_integrate_gap_radical_inverse_policy_clusters(
                cases
            ),
            # The asinh Hermite row's composed direct-diff channel times
            # out (simplifier churn); derivative-equivalence plus numeric
            # unit round-trips pin it instead.
            {"block8_polynomial_over_sqrt_hermite_split": 1},
        )
        self.assertEqual(
            SMOKE.count_direct_diff_integrate_gap_trig_hyperbolic_policy_clusters(
                cases
            ),
            {
                "block7_explicit_reciprocal_hyperbolic_tangent": 4,
                "block7_explicit_reciprocal_trig_log_substitution": 7,
                "block7_hyperbolic_log_derivative_ratio": 1,
                "block7_hyperbolic_tanh_log_derivative": 1,
                "block7_sqrt_chain_hyperbolic_log": 3,
                "block7_sqrt_chain_hyperbolic_reciprocal_derivative_product": 3,
                "block7_sqrt_chain_reciprocal_trig_product": 7,
                "block7_sqrt_chain_trig_log": 2,
                "block7_trig_log_derivative_ratio": 2,
            },
        )
        self.assertEqual(
            SMOKE.direct_diff_integrate_gap_case_names_by_base_integration_policy_cluster(
                cases
            ),
            {},
        )
        gap_trig_examples = (
            SMOKE.direct_diff_integrate_gap_case_names_by_trig_hyperbolic_policy_cluster(
                cases
            )
        )
        self.assertEqual(
            gap_trig_examples["block7_hyperbolic_tanh_log_derivative"],
            ["symbolic_external_scale_hyperbolic_tanh_direct_log_positive"],
        )
        self.assertEqual(
            gap_trig_examples["block7_sqrt_chain_reciprocal_trig_product"][:3],
            [
                "sqrt_chain_secant_tangent_domain",
                "external_symbolic_scale_sqrt_chain_secant_tangent_domain",
                "external_symbolic_scale_sqrt_chain_cosecant_cotangent_domain",
            ],
        )
        self.assertEqual(
            SMOKE.direct_diff_integrate_gap_case_names_by_calculus_maturity_block(
                cases
            )["block7_trig_hyperbolic_integration"][:2],
            [
                "explicit_reciprocal_sine_verified_log_domain",
                "explicit_reciprocal_cosine_verified_log_domain",
            ],
        )
        self.assertEqual(
            SMOKE.count_direct_diff_integrate_equivalence_calculus_maturity_blocks(
                cases
            ),
            {
                "block4_base_integration": 1,
                "block5_generalized_substitution": 2,
                "block7_trig_hyperbolic_integration": 5,
                "block8_radical_inverse_families": 12,
            },
        )
        self.assertEqual(
            SMOKE.count_direct_diff_integrate_equivalence_trig_hyperbolic_policy_clusters(
                cases
            ),
            {
                "block7_hyperbolic_reciprocal_derivative_product": 3,
                "block7_hyperbolic_reciprocal_square": 2,
            },
        )
        self.assertEqual(
            SMOKE.count_direct_diff_integrate_equivalence_base_integration_policy_clusters(
                cases
            ),
            {},
        )

    def test_positive_quadratic_rational_rows_track_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}

        square_reduction = cases["rational_positive_quadratic_square_reduction"]
        self.assertTrue(square_reduction.direct_diff_integrate_from_derivative)
        self.assertEqual(
            SMOKE.direct_diff_integrate_expected_result(square_reduction),
            "1 / (x^2 + 1)^2",
        )

        improper = cases["rational_improper_positive_quadratic_polynomial_division"]
        self.assertTrue(improper.direct_diff_integrate_from_derivative)
        self.assertEqual(
            SMOKE.direct_diff_integrate_expected_result(improper),
            "(x^2 + 1) / (x^2 + 2·x + 2)",
        )

        negative_orientation = cases[
            "rational_improper_positive_quadratic_negative_orientation"
        ]
        self.assertEqual(
            SMOKE.direct_diff_integrate_expected_result(negative_orientation),
            "(-x^2 - 1) / (x^2 + 2·x + 2)",
        )

    def test_block6_linear_pole_rational_rows_track_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}

        expected = {
            "rational_partial_fraction_mixed_linear_positive_quadratic": (
                "1 / (x^4 - 1)",
                ("x ≠ -1", "x ≠ 1"),
            ),
            "rational_partial_fraction_repeated_linear_positive_quadratic": (
                "1 / ((x - 1)^2·(x^2 + 1))",
                ("x ≠ 1",),
            ),
            "rational_partial_fraction_repeated_origin_linear_positive_quadratic_no_log": (
                "1 / (x^2·(x^2 + 1))",
                ("x ≠ 0",),
            ),
            "rational_partial_fraction_repeated_origin_scaled_positive_quadratic_no_log": (
                "1 / (x^2·(x^2 + 4))",
                ("x ≠ 0",),
            ),
            "rational_improper_partial_fraction_polynomial_division": (
                "(x^2 + 1) / (x^2 - 1)",
                ("x ≠ -1", "x ≠ 1"),
            ),
        }

        for name, (result, required_display) in expected.items():
            with self.subTest(name=name):
                case = cases[name]
                self.assertEqual(SMOKE.direct_diff_integrate_expected_result(case), result)
                self.assertEqual(
                    SMOKE.direct_diff_integrate_expected_required_display(case),
                    required_display,
                )

    def test_block7_secant_cosecant_log_table_tracks_direct_diff_integrate(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}

        exact_expected = {
            "affine_secant_table_log_domain": (
                "sec(2·x + 1)",
                ("cos(2·x + 1) ≠ 0",),
            ),
            "affine_cosecant_table_log_domain": (
                "csc(2·x + 1)",
                ("sin(2·x + 1) ≠ 0",),
            ),
            "external_constant_affine_secant_table_log_domain": (
                "3·sec(2·x + 1)",
                ("cos(2·x + 1) ≠ 0",),
            ),
            "rational_denominator_scaled_affine_secant_table_log_domain": (
                "(2·sec(2·x + 1))/3",
                ("cos(2·x + 1) ≠ 0",),
            ),
            "rational_denominator_scaled_affine_cosecant_table_log_domain": (
                "(2·csc(2·x + 1))/3",
                ("sin(2·x + 1) ≠ 0",),
            ),
        }

        for name, (result, required_display) in exact_expected.items():
            with self.subTest(name=name):
                case = cases[name]
                self.assertEqual(SMOKE.direct_diff_integrate_expected_result(case), result)
                self.assertIsNone(
                    SMOKE.direct_diff_integrate_expected_equivalent_to(case)
                )
                self.assertEqual(
                    SMOKE.direct_diff_integrate_expected_required_display(case),
                    required_display,
                )

    def test_direct_diff_integrate_expectation_can_inherit_derivative_policy(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}

        polynomial = cases["polynomial_power_direct"]
        self.assertTrue(polynomial.direct_diff_integrate_from_derivative)
        self.assertIsNone(polynomial.expected_direct_diff_integrate_result)
        self.assertEqual(SMOKE.direct_diff_integrate_expected_result(polynomial), "x^2")
        self.assertEqual(
            SMOKE.direct_diff_integrate_expected_required_display(polynomial),
            (),
        )

        reciprocal = cases["reciprocal_affine_log_abs_domain"]
        self.assertTrue(reciprocal.direct_diff_integrate_from_derivative)
        self.assertIsNone(reciprocal.expected_direct_diff_integrate_result)
        self.assertEqual(
            SMOKE.direct_diff_integrate_expected_result(reciprocal),
            "1 / (2·x + 1)",
        )
        self.assertEqual(
            SMOKE.direct_diff_integrate_expected_required_display(reciprocal),
            ("x ≠ -1/2",),
        )

    def test_radical_inverse_policy_clusters_cover_supported_block8_cases(self) -> None:
        cases = SMOKE.build_cases()

        supported_block8_cases = [
            case
            for case in cases
            if case.outcome == "supported"
            and SMOKE.calculus_maturity_block(case)
            == "block8_radical_inverse_families"
        ]
        clustered_count = sum(SMOKE.count_radical_inverse_policy_clusters(cases).values())

        self.assertEqual(clustered_count, len(supported_block8_cases))

    def test_shifted_sqrt_tangent_uses_bounded_derivative_verification(self) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases["shifted_sqrt_chain_tangent_log_domain"]

        self.assertIsNotNone(case.expected_derivative_result)
        self.assertIsNone(case.expected_derivative_equivalent_to)
        self.assertIn("tan(b - sqrt(x))", case.expected_derivative_result)

    def test_affine_shifted_sqrt_hyperbolic_uses_bounded_derivative_verification(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        shifted_case = cases["shifted_sqrt_chain_hyperbolic_tangent_log_domain"]
        case = cases["affine_shifted_sqrt_chain_hyperbolic_tangent_log_domain"]

        self.assertIsNotNone(shifted_case.expected_derivative_result)
        self.assertIsNone(shifted_case.expected_derivative_equivalent_to)
        self.assertIn("tanh(x^(1/2) - b)", shifted_case.expected_derivative_result)
        self.assertIsNotNone(case.expected_derivative_result)
        self.assertIsNone(case.expected_derivative_equivalent_to)
        self.assertIn("tanh((3·x + 1)^(1/2) - b)", case.expected_derivative_result)

    def test_inverse_trig_root_scale_siblings_use_direct_derivative_verification(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        for name in (
            "inverse_trig_symbolic_denominator_numeric_square_scale_sqrt_reciprocal_bridge",
            "inverse_trig_symbolic_numerator_scale_sqrt_reciprocal_bridge",
        ):
            case = cases[name]
            self.assertIsNotNone(case.expected_derivative_result)
            self.assertIsNone(case.expected_derivative_equivalent_to)

    def test_hyperbolic_reciprocal_fourth_uses_direct_derivative_verification(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        for name in (
            "hyperbolic_cosh_reciprocal_fourth_substitution",
            "symbolic_external_scale_shifted_hyperbolic_cosh_reciprocal_fourth_substitution",
            "hyperbolic_sinh_reciprocal_fourth_substitution",
            "symbolic_external_scale_shifted_hyperbolic_sinh_reciprocal_fourth_substitution",
        ):
            case = cases[name]
            self.assertIsNotNone(case.expected_derivative_result)
            self.assertIsNone(case.expected_derivative_equivalent_to)

    def test_hyperbolic_log_cosh_rows_use_direct_derivative_verification(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        for name in (
            "symbolic_external_scale_hyperbolic_tanh_log_derivative_ratio_positive",
            "symbolic_external_scale_hyperbolic_tanh_direct_log_positive",
        ):
            case = cases[name]
            self.assertIsNotNone(case.expected_derivative_result)
            self.assertIsNone(case.expected_derivative_equivalent_to)

    def test_symbolic_shifted_hyperbolic_sinh_square_uses_direct_derivative_verification(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases[
            "symbolic_external_scale_shifted_hyperbolic_sinh_reciprocal_square_substitution"
        ]

        self.assertIsNotNone(case.expected_derivative_result)
        self.assertIsNone(case.expected_derivative_equivalent_to)
        self.assertIn("sinh(x^2 + b)^2", case.expected_derivative_result)

    def test_shifted_sqrt_hyperbolic_reciprocal_product_rows_use_direct_derivative_verification(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        for name in (
            "shifted_sqrt_chain_hyperbolic_sinh_over_cosh_square_symbolic_scale_domain",
            "negative_shifted_sqrt_chain_hyperbolic_cosh_over_sinh_square_symbolic_scale_domain",
        ):
            case = cases[name]
            self.assertIsNotNone(case.expected_derivative_result)
            self.assertIsNone(case.expected_derivative_equivalent_to)

    def test_external_symbolic_scale_sqrt_reciprocal_trig_rows_use_direct_derivative_verification(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        for name in (
            "external_symbolic_scale_sqrt_chain_secant_tangent_domain",
            "external_symbolic_scale_sqrt_chain_cosecant_cotangent_domain",
            "external_symbolic_scale_shifted_sqrt_chain_secant_tangent_domain",
            "external_symbolic_scale_shifted_sqrt_chain_cosecant_cotangent_domain",
            "external_symbolic_scale_sqrt_minus_symbol_chain_secant_tangent_domain",
            "external_symbolic_scale_sqrt_minus_symbol_chain_cosecant_cotangent_domain",
        ):
            case = cases[name]
            self.assertIsNotNone(case.expected_derivative_result)
            self.assertIsNone(case.expected_derivative_equivalent_to)

    def test_repeated_linear_positive_quadratic_partial_fraction_uses_direct_derivative_verification(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases["rational_partial_fraction_repeated_linear_positive_quadratic"]

        self.assertIsNotNone(case.expected_derivative_result)
        self.assertIsNone(case.expected_derivative_equivalent_to)
        self.assertIn("x^4 + 2·x^2 + 1", case.expected_derivative_result)

    def test_mixed_linear_positive_quadratic_partial_fraction_uses_direct_derivative_verification(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases["rational_partial_fraction_mixed_linear_positive_quadratic"]

        self.assertIsNotNone(case.expected_derivative_result)
        self.assertIsNone(case.expected_derivative_equivalent_to)
        self.assertIn("x^2 - 1", case.expected_derivative_result)

    def test_mixed_simple_repeated_linear_partial_fraction_uses_direct_derivative_verification(
        self,
    ) -> None:
        cases = {case.name: case for case in SMOKE.build_cases()}
        case = cases["rational_partial_fraction_mixed_simple_repeated_linear_factors"]

        self.assertIsNotNone(case.expected_derivative_result)
        self.assertIsNone(case.expected_derivative_equivalent_to)
        self.assertIn("x^3 + 2·x^2 + x", case.expected_derivative_result)

    def test_run_matrix_accepts_expected_required_display_and_residual(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    case "$*" in
                    *"diff(integrate(x^2, x), x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"x^2","warnings":[],"required_display":[]}
                    OUT
                    ;;
                    *"diff(integrate(2*x + 3, x), x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"2·x + 3","warnings":[],"required_display":[]}
                    OUT
                    ;;
                    *"diff(integrate(1/(2*x + 1), x), x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"1 / (2·x + 1)","warnings":[],"required_display":["x ≠ -1/2"]}
                    OUT
                    ;;
                    *"integrate(x^2, x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"1/3·x^3","warnings":[],"required_display":[],"steps":[{"substeps":[{"title":"Usar regla de potencia para integrales"}]}]}
                    OUT
                    ;;
                    *"diff(1/3·x^3, x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"x^2","warnings":[],"required_display":[]}
                    OUT
                    ;;
                    *"integrate(2*x + 3, x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"x^2 + 3·x","warnings":[],"required_display":[],"steps":[{"substeps":[{"title":"Usar linealidad de la integral"},{"title":"Integrar cada término"}]}]}
                    OUT
                    ;;
                    *"diff(x^2 + 3·x, x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"2·x + 3","warnings":[],"required_display":[]}
                    OUT
                    ;;
                    *"integrate(1/(2*x + 1), x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"1/2·ln(|2·x + 1|)","warnings":[],"required_display":["x ≠ -1/2"],"steps":[{"substeps":[{"title":"Usar la regla de ln|u| con derivada interna"},{"title":"Identificar el denominador afín"},{"title":"Ajustar el factor constante"}]}]}
                    OUT
                    ;;
                    *"diff(1/2·ln("*)
                    cat <<'OUT'
                    {"ok":true,"result":"1 / (2·x + 1)","warnings":[],"required_display":["x ≠ -1/2"]}
                    OUT
                    ;;
                    *"integrate(exp(x^2), x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"integrate(e^(x^2), x)","warnings":[],"required_display":[],"timings_us":{"parse_us":100,"simplify_us":3000,"total_us":3500},"steps":[{"rule":"Conservar integral residual","before":"e^(x^2)","after":"integrate(e^(x^2), x)"}]}
                    OUT
                    ;;
                    *)
                    echo '{"ok":false,"result":"unexpected","warnings":[],"required_display":[]}'
                    ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(
                (
                    "polynomial_power_direct",
                    "polynomial_sum_linearity",
                    "reciprocal_affine_log_abs_domain",
                    "non_elementary_exp_quadratic_residual",
                )
            )
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "pass", matrix)
        self.assertEqual(matrix["total"], 4)
        self.assertEqual(matrix["status_counts"]["pass"], 4)
        self.assertEqual(matrix["supported_case_count"], 3)
        self.assertEqual(matrix["residual_case_count"], 1)
        self.assertEqual(matrix["warning_expected_case_count"], 0)
        self.assertEqual(matrix["required_display_case_count"], 1)
        self.assertEqual(matrix["step_checked_case_count"], 4)
        self.assertEqual(matrix["supported_step_unchecked_case_count"], 0)
        self.assertEqual(matrix["supported_step_unchecked_cases"], [])
        self.assertEqual(matrix["antiderivative_verification_case_count"], 3)
        self.assertEqual(matrix["verified_supported_case_count"], 3)
        self.assertEqual(matrix["direct_diff_integrate_case_count"], 3)
        self.assertEqual(matrix["trig_hyperbolic_policy_cluster_counts"], {})
        self.assertEqual(matrix["base_integration_policy_cluster_counts"], {})
        self.assertEqual(matrix["radical_inverse_policy_cluster_counts"], {})
        self.assertEqual(
            matrix["direct_diff_integrate_base_integration_policy_cluster_counts"],
            {},
        )
        self.assertEqual(
            matrix["verification_regime_counts"],
            {
                "residual_not_verified": 1,
                "verified_by_diff_and_direct_diff_integrate": 3,
            },
        )
        self.assertEqual(
            matrix["residual_cause_counts"],
            {"non_elementary_composition": 1},
        )
        self.assertEqual(
            matrix["residual_family_counts"], {"non_elementary_exp_quadratic": 1}
        )
        self.assertEqual(
            matrix["residual_cause_family_counts"],
            {"non_elementary_composition/non_elementary_exp_quadratic": 1},
        )
        self.assertEqual(matrix["expected_step_substring_count"], 7)
        self.assertEqual(matrix["required_display_counts"], {"x ≠ -1/2": 1})
        self.assertIn("slowest_integrate_evaluations", matrix)
        self.assertIn("slowest_antiderivative_verifications", matrix)
        self.assertIn("runtime_by_antiderivative_verification_mode", matrix)
        self.assertEqual(
            matrix["runtime_by_residual_cause"][0]["cause"],
            "non_elementary_composition",
        )
        self.assertEqual(matrix["runtime_by_residual_cause"][0]["case_count"], 1)
        self.assertEqual(
            matrix["runtime_by_residual_cause_family"][0]["cause_family"],
            "non_elementary_composition/non_elementary_exp_quadratic",
        )
        self.assertEqual(
            matrix["runtime_by_residual_cause_family"][0]["case_count"], 1
        )
        self.assertIn("residual_public_phase_slowest_cases", matrix)
        self.assertIn("residual_public_phase_by_cause", matrix)
        self.assertEqual(
            matrix["residual_public_phase_by_cause"][0]["cause"],
            "non_elementary_composition",
        )
        self.assertIn("residual_public_phase_by_cause_family", matrix)
        self.assertEqual(
            matrix["residual_public_phase_by_cause_family"][0]["cause_family"],
            "non_elementary_composition/non_elementary_exp_quadratic",
        )
        self.assertEqual(
            matrix["residual_public_phase_slowest_cases"][0]["cli_simplify_ms"],
            3.0,
        )
        self.assertIn("largest_stdout_payload_cases", matrix)
        self.assertIn("largest_step_trace_cases", matrix)
        self.assertIn("integrate_elapsed_seconds", matrix["cases"][0])
        self.assertIn("cli_total_us", matrix["cases"][0])
        self.assertIn("stdout_bytes", matrix["cases"][0])
        self.assertIn("step_text_char_count", matrix["cases"][0])
        self.assertIn(
            "antiderivative_verification_elapsed_seconds",
            matrix["cases"][0],
        )
        self.assertIn("direct_diff_integrate_elapsed_seconds", matrix["cases"][0])
        self.assertEqual(
            matrix["cases"][0]["antiderivative_verification_mode"],
            "direct_derivative",
        )

    def test_residual_shape_orientation_probes_are_best_effort_timed_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cas_cli = Path(tmpdir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    cat <<'OUT'
                    {"ok":true,"result":"integrate(1 / ((tan(x) - 2)·ln(tan(x))), x)","warnings":[],"required_display":["cos(x) ≠ 0","tan(x) - 1 ≠ 0","tan(x) - 2 ≠ 0","tan(x) > 0"],"timings_us":{"parse_us":100,"simplify_us":2500,"total_us":3000}}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            rows = SMOKE.residual_shape_orientation_probe_rows(
                cas_cli=cas_cli,
                timeout_seconds=2.0,
            )

        self.assertEqual(len(rows), 24)
        self.assertEqual({row["status"] for row in rows}, {"pass"})
        self.assertEqual(
            {row["expression_shape"] for row in rows},
            {"source", "factored_residual"},
        )
        self.assertEqual(
            {row["orientation"] for row in rows},
            {
                "tan_minus_offset",
                "cot_minus_offset",
                "offset_minus_tan",
                "offset_minus_cot",
                "tan_minus_symbolic_offset",
                "symbolic_offset_minus_tan",
                "cot_minus_symbolic_offset",
                "symbolic_offset_minus_cot",
            },
        )
        self.assertEqual({row["steps_mode"] for row in rows}, {"off", "on"})
        self.assertEqual(
            [(row["name"], row["steps_mode"]) for row in rows[:2]],
            [
                ("shifted_tangent_log_source", "off"),
                ("shifted_tangent_log_source", "on"),
            ],
        )
        self.assertEqual(rows[0]["cli_simplify_us"], 2500)
        self.assertEqual(rows[0]["required_display_count"], 4)
        self.assertIn("tan(x) - 2 ≠ 0", rows[0]["required_display"])
        self.assertIn("wall_elapsed_seconds", rows[0])

    def test_run_matrix_reports_required_display_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    cat <<'OUT'
                    {"ok":true,"result":"1/2·ln(|2·x + 1|)","warnings":[],"required_display":[]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(("reciprocal_affine_log_abs_domain",))
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "fail")
        self.assertEqual(matrix["status_counts"]["fail"], 1)
        self.assertEqual(matrix["issue_kind_counts"], {"required_display_mismatch": 1})
        self.assertEqual(
            matrix["problem_cases"][0]["name"],
            "reciprocal_affine_log_abs_domain",
        )

    def test_run_matrix_reports_antiderivative_verification_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    case "$*" in
                    *"integrate(x^2, x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"1/3·x^3","warnings":[],"required_display":[],"steps":[{"substeps":[{"title":"Usar regla de potencia para integrales"}]}]}
                    OUT
                    ;;
                    *"diff(1/3·x^3, x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"0","warnings":[],"required_display":[]}
                    OUT
                    ;;
                    *)
                    echo '{"ok":false,"result":"unexpected","warnings":[],"required_display":[]}'
                    ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(("polynomial_power_direct",))
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "fail")
        self.assertEqual(matrix["status_counts"]["fail"], 1)
        self.assertEqual(
            matrix["issue_kind_counts"],
            {"antiderivative_verification_mismatch": 1},
        )
        self.assertEqual(matrix["problem_cases"][0]["derivative_result"], "0")

    def test_run_matrix_accepts_antiderivative_equivalence_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    case "$*" in
                    *"integrate(f, x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"F","warnings":[],"required_display":[],"steps":[{"substeps":[{"title":"Explicar integral"}]}]}
                    OUT
                    ;;
                    *"diff(F, x) - (f)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"0","warnings":[],"required_display":[]}
                    OUT
                    ;;
                    *)
                    echo '{"ok":false,"result":"unexpected","warnings":[],"required_display":[]}'
                    ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            case = SMOKE.IntegrateCommandMatrixCase(
                name="equivalence_verified",
                expr="integrate(f, x)",
                expected_result="F",
                expected_derivative_equivalent_to="f",
                expected_step_substrings=("Explicar integral",),
            )
            matrix = SMOKE.run_matrix((case,), cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "pass")
        self.assertEqual(matrix["antiderivative_verification_case_count"], 1)
        self.assertEqual(matrix["verified_supported_case_count"], 1)
        self.assertEqual(
            matrix["verification_regime_counts"],
            {"verified_by_diff": 1},
        )
        self.assertEqual(matrix["cases"][0]["derivative_equivalence_result"], "0")
        self.assertEqual(
            matrix["cases"][0]["antiderivative_verification_mode"],
            "residual_equivalence",
        )
        self.assertEqual(
            matrix["runtime_by_antiderivative_verification_mode"][0]["mode"],
            "residual_equivalence",
        )

    def test_run_matrix_accepts_direct_diff_integrate_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    case "$*" in
                    *"diff(integrate(g, x), x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"g","warnings":[],"required_display":["a ≠ 0"]}
                    OUT
                    ;;
                    *"integrate(g, x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"G","warnings":[],"required_display":["a ≠ 0"],"steps":[{"substeps":[{"title":"Explicar integral"}]}]}
                    OUT
                    ;;
                    *"diff(G, x) - (g)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"0","warnings":[],"required_display":["a ≠ 0"]}
                    OUT
                    ;;
                    *)
                    echo '{"ok":false,"result":"unexpected","warnings":[],"required_display":[]}'
                    ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            case = SMOKE.IntegrateCommandMatrixCase(
                name="direct_nested_verified",
                expr="integrate(g, x)",
                expected_result="G",
                expected_required_display=("a ≠ 0",),
                expected_derivative_equivalent_to="g",
                expected_derivative_required_display=("a ≠ 0",),
                expected_direct_diff_integrate_result="g",
                expected_direct_diff_integrate_required_display=("a ≠ 0",),
                expected_step_substrings=("Explicar integral",),
            )
            matrix = SMOKE.run_matrix((case,), cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "pass", matrix)
        self.assertEqual(matrix["direct_diff_integrate_case_count"], 1)
        self.assertEqual(matrix["direct_diff_integrate_gap_case_count"], 0)
        self.assertEqual(matrix["verified_supported_case_count"], 1)
        self.assertEqual(
            matrix["verification_regime_counts"],
            {"verified_by_diff_and_direct_diff_integrate": 1},
        )
        self.assertIn("slowest_direct_diff_integrate_checks", matrix)
        self.assertEqual(matrix["cases"][0]["direct_diff_integrate_result"], "g")
        self.assertEqual(
            matrix["cases"][0]["direct_diff_integrate_required_display"],
            ["a ≠ 0"],
        )

    def test_run_matrix_reports_missing_expected_step_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    cat <<'OUT'
                    {"ok":true,"result":"1/2·ln(|2·x + 1|)","warnings":[],"required_display":["x ≠ -1/2"],"steps":[{"rule":"Calcular la integral"}]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(("reciprocal_affine_log_abs_domain",))
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "fail")
        self.assertEqual(matrix["status_counts"]["fail"], 1)
        self.assertEqual(matrix["issue_kind_counts"], {"step_trace_mismatch": 1})
        self.assertEqual(
            matrix["problem_cases"][0]["name"],
            "reciprocal_affine_log_abs_domain",
        )

    def test_run_matrix_reports_fragile_derivative_stderr_even_when_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    case "$*" in
                    *"integrate(x^2, x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"1/3·x^3","warnings":[],"required_display":[],"steps":[{"substeps":[{"title":"Usar regla de potencia para integrales"}]}]}
                    OUT
                    ;;
                    *"diff(1/3·x^3, x)"*)
                    echo 'WARN simplify: depth_overflow' >&2
                    cat <<'OUT'
                    {"ok":true,"result":"x^2","warnings":[],"required_display":[]}
                    OUT
                    ;;
                    *)
                    echo '{"ok":false,"result":"unexpected","warnings":[],"required_display":[]}'
                    ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(("polynomial_power_direct",))
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "fail")
        self.assertEqual(matrix["status_counts"]["fail"], 1)
        self.assertEqual(matrix["issue_kind_counts"], {"stderr_fragility": 1})
        self.assertEqual(matrix["problem_cases"][0]["name"], "polynomial_power_direct")
        self.assertIn("fragile substring", matrix["problem_cases"][0]["error"])


if __name__ == "__main__":
    unittest.main()
