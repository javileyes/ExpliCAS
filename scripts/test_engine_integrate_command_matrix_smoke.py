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

        self.assertEqual(len(cases), 115)
        names = {case.name for case in cases}
        self.assertIn("reciprocal_affine_log_abs_domain", names)
        self.assertIn("reciprocal_negative_affine_log_abs_domain", names)
        self.assertIn("reciprocal_negative_affine_derivative_log_abs_domain", names)
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
        self.assertIn("rational_positive_quadratic_square_reduction", names)
        self.assertIn("affine_exp_substitution", names)
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
        self.assertIn("inverse_sqrt_direct_asinh_unconditional", names)
        self.assertIn("polynomial_base_sqrt_substitution", names)
        self.assertIn("hyperbolic_sine_reciprocal_square_substitution", names)
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
            {"residual": 13, "supported": 100, "undefined": 2},
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
            100,
        )
        self.assertEqual(
            SMOKE.count_verification_regimes(cases),
            {
                "residual_not_verified": 13,
                "undefined_not_verified": 2,
                "verified_by_diff": 100,
            },
        )
        self.assertEqual(
            SMOKE.count_residual_causes(cases),
            {
                "branch_sensitive_interval_residual": 1,
                "non_elementary_composition": 7,
                "special_function_method_required": 1,
                "unsupported_reciprocal_trig_method": 4,
            },
        )
        step_checked = {
            case.name
            for case in cases
            if case.expected_step_substrings
        }
        supported_step_unchecked = {
            case.name
            for case in cases
            if case.outcome == "supported" and not case.expected_step_substrings
        }
        self.assertEqual(
            step_checked,
            {
                "polynomial_power_direct",
                "polynomial_sum_linearity",
                "affine_trig_substitution",
                "constant_multiple_affine_trig_substitution",
                "affine_exp_substitution",
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
                "explicit_reciprocal_tangent_presimplified_residual_domain",
                "explicit_tangent_log_residual_condition_alias_dedupe",
                "explicit_tangent_log_numeric_shifted_residual_condition_alias_dedupe",
                "explicit_tangent_log_numeric_offset_residual_condition_alias_dedupe",
                "explicit_cotangent_log_numeric_offset_residual_condition_alias_dedupe",
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
                "affine_secant_table_log_domain",
                "affine_cosecant_table_log_domain",
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
                "inverse_sqrt_direct_asinh_unconditional",
                "affine_inverse_sqrt_arcsin_domain",
                "polynomial_base_sqrt_substitution",
                "hyperbolic_reciprocal_square_substitution",
                "hyperbolic_sine_reciprocal_square_substitution",
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
                "by_parts_affine_log_domain",
                "affine_secant_tangent_derivative_product_domain",
                "affine_cosecant_cotangent_derivative_product_domain",
                "negative_affine_secant_tangent_derivative_product_domain",
                "negative_affine_cosecant_cotangent_derivative_product_domain",
                "polynomial_shifted_secant_tangent_derivative_product_domain",
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
                "empty_real_domain": 1,
                "explicit_denominator_source_condition": 1,
                "explicit_hyperbolic_tangent_presimplified_condition_dedupe": 1,
                "explicit_hyperbolic_tangent_denominator_verified_substitution": 1,
                "explicit_reciprocal_sine_verified_substitution": 1,
                "explicit_reciprocal_trig_source_defined_condition": 1,
                "explicit_tangent_denominator_source_condition": 1,
                "explicit_tangent_denominator_verified_substitution": 1,
                "explicit_tangent_log_shifted_condition_dedupe": 1,
                "explicit_tangent_log_numeric_shifted_condition_dedupe": 1,
                "explicit_tangent_log_numeric_offset_condition_dedupe": 1,
                "explicit_cotangent_log_numeric_offset_condition_dedupe": 1,
                "explicit_tangent_presimplified_condition_dedupe": 1,
                "hyperbolic_sine_pole_required": 10,
                "linear_poles_required": 9,
                "nonzero_required": 3,
                "nonfinite_undefined": 1,
                "positive_log_argument_required": 1,
                "positive_log_argument_and_trig_poles": 1,
                "positive_required": 5,
                "positive_required_residual": 1,
                "radical_interval_additive_residual": 1,
                "radical_interval": 2,
                "rational_interval": 2,
                "shifted_sqrt_chain_nonzero_positive": 5,
                "shifted_sqrt_chain_positive": 1,
                "symbolic_denominator_scale_positive_required": 2,
                "symbolic_numerator_scale_positive_required": 1,
                "sqrt_minus_symbol_chain_nonzero_positive": 2,
                "sqrt_chain_nonzero_positive": 4,
                "sqrt_chain_hyperbolic_presimplified_condition_dedupe": 1,
                "sqrt_chain_hyperbolic_sine_pole_required": 1,
                "structurally_nonzero_negative_quadratic_denominator": 1,
                "structurally_positive_denominator": 2,
                "structurally_positive_log_argument": 3,
                "trig_pole_additive_residual": 1,
                "trig_pole_presimplified_residual": 1,
                "trig_pole_residual": 1,
                "trig_log_derivative_pole_required": 2,
                "trig_reciprocal_table_pole_required": 2,
                "trig_reciprocal_product_pole_required": 5,
                "trig_reciprocal_product_exact_symbolic_derivative_pole_required": 8,
                "trig_sine_pole_presimplified_residual": 1,
                "unconditional": 19,
                "unconditional_cosh_positive": 2,
            },
        )
        self.assertEqual(
            SMOKE.count_trig_hyperbolic_policy_clusters(cases),
            {
                "block7_explicit_reciprocal_hyperbolic_tangent": 4,
                "block7_explicit_reciprocal_trig_log_substitution": 4,
                "block7_hyperbolic_reciprocal_derivative_product": 7,
                "block7_hyperbolic_log_derivative_ratio": 1,
                "block7_hyperbolic_tanh_log_derivative": 1,
                "block7_hyperbolic_reciprocal_fourth": 4,
                "block7_hyperbolic_reciprocal_square": 6,
                "block7_trig_log_derivative_ratio": 2,
                "block7_sqrt_chain_hyperbolic_reciprocal_derivative_product": 3,
                "block7_sqrt_chain_hyperbolic_log": 3,
                "block7_sqrt_chain_reciprocal_trig_product": 7,
                "block7_sqrt_chain_trig_log": 2,
                "block7_trig_reciprocal_derivative_product": 13,
                "block7_trig_reciprocal_log_table": 2,
                "block9_explicit_reciprocal_trig_residual": 6,
            },
        )
        self.assertEqual(
            SMOKE.count_radical_inverse_policy_clusters(cases),
            {"block8_inverse_trig_root_reciprocal": 6},
        )
        self.assertEqual(
            SMOKE.count_calculus_maturity_blocks(cases),
            {
                "block4_base_integration": 6,
                "block5_generalized_substitution": 11,
                "block6_rational_integration": 12,
                "block7_trig_hyperbolic_integration": 59,
                "block8_radical_inverse_families": 12,
                "block9_residuals_and_non_goals": 15,
            },
        )
        self.assertEqual(
            SMOKE.count_calculus_block_gates(cases),
            {
                "didactic_trace_and_verified_antiderivative": 26,
                "domain_conditions_and_verified_antiderivative": 74,
                "explicit_undefined_domain_policy": 2,
                "safe_residual_policy": 13,
            },
        )
        self.assertGreaterEqual(len({case.family for case in cases}), 10)

    def test_run_matrix_accepts_expected_required_display_and_residual(self) -> None:
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
                    {"ok":true,"result":"integrate(e^(x^2), x)","warnings":[],"required_display":[],"steps":[{"rule":"Conservar integral residual","before":"e^(x^2)","after":"integrate(e^(x^2), x)"}]}
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
        self.assertEqual(matrix["trig_hyperbolic_policy_cluster_counts"], {})
        self.assertEqual(matrix["radical_inverse_policy_cluster_counts"], {})
        self.assertEqual(
            matrix["verification_regime_counts"],
            {"residual_not_verified": 1, "verified_by_diff": 3},
        )
        self.assertEqual(
            matrix["residual_cause_counts"],
            {"non_elementary_composition": 1},
        )
        self.assertEqual(matrix["expected_step_substring_count"], 7)
        self.assertEqual(matrix["required_display_counts"], {"x ≠ -1/2": 1})

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
        self.assertEqual(
            matrix["verification_regime_counts"],
            {"verified_by_diff": 1},
        )
        self.assertEqual(matrix["cases"][0]["derivative_equivalence_result"], "0")

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
