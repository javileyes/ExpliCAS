import importlib.util
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "engine_limit_command_matrix_smoke.py"
SPEC = importlib.util.spec_from_file_location("engine_limit_command_matrix_smoke", MODULE_PATH)
SMOKE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)


class LimitCommandMatrixSmokeTests(unittest.TestCase):
    def test_default_matrix_covers_limit_policy_axes(self) -> None:
        cases = SMOKE.build_cases()

        self.assertEqual(len(cases), 161)
        names = {case.name for case in cases}
        self.assertIn("finite_removable_rational_cancellation", names)
        self.assertIn("finite_rational_simple_pole_residual", names)
        self.assertIn("finite_even_order_rational_pole_bilateral_supported", names)
        self.assertIn("finite_even_cosh_reciprocal_pole_bilateral_supported", names)
        self.assertIn("finite_oscillating_outer_even_pole_residual", names)
        self.assertIn("infinity_polylog_fractional_power_dominance", names)
        self.assertIn("infinity_polylog_higher_power_dominance", names)
        self.assertIn("infinity_power_over_polylog_diverges", names)
        self.assertIn("infinity_bounded_noise_equal_degree_ratio", names)
        self.assertIn("infinity_bounded_noise_higher_degree_diverges", names)
        self.assertIn("infinity_log_difference_finite_ratio", names)
        self.assertIn("infinity_log_difference_diverges", names)
        self.assertIn("finite_one_sided_removable_rational_cancellation", names)
        self.assertIn(
            "finite_one_sided_removable_rational_cancellation_with_nonlocal_pole",
            names,
        )
        self.assertIn("finite_one_sided_rational_simple_pole_supported", names)
        self.assertIn(
            "finite_one_sided_shifted_scaled_odd_order_rational_pole_supported",
            names,
        )
        self.assertIn("finite_one_sided_abs_orientation_quotient_supported", names)
        self.assertIn(
            "finite_one_sided_abs_quadratic_orientation_quotient_supported", names
        )
        self.assertIn(
            "finite_one_sided_sign_quadratic_orientation_supported", names
        )
        self.assertIn(
            "finite_bilateral_sign_even_order_orientation_supported", names
        )
        self.assertIn("finite_abs_even_order_pole_bilateral_supported", names)
        self.assertIn("finite_one_sided_log_zero_endpoint_supported", names)
        self.assertIn(
            "finite_bilateral_log_even_order_endpoint_supported", names
        )
        self.assertIn(
            "finite_bilateral_reciprocal_base_log_even_order_endpoint_supported",
            names,
        )
        self.assertIn("finite_one_sided_fixed_base_log_zero_endpoint_supported", names)
        self.assertIn(
            "finite_one_sided_binary_log_constant_base_less_than_one_endpoint_supported",
            names,
        )
        self.assertIn(
            "finite_one_sided_log_rational_zero_tail_endpoint_supported", names
        )
        self.assertIn(
            "finite_one_sided_variable_base_log_rational_zero_tail_endpoint_supported",
            names,
        )
        self.assertIn(
            "finite_one_sided_unit_base_boundary_log_rational_zero_tail_endpoint_supported",
            names,
        )
        self.assertIn(
            "finite_one_sided_rational_unit_base_boundary_log_rational_zero_tail_endpoint_supported",
            names,
        )
        self.assertIn("finite_one_sided_sqrt_zero_endpoint_supported", names)
        self.assertIn("finite_one_sided_sqrt_shifted_zero_endpoint_supported", names)
        self.assertIn("finite_one_sided_half_power_zero_endpoint_supported", names)
        self.assertIn(
            "finite_one_sided_sqrt_rational_zero_tail_endpoint_supported", names
        )
        self.assertIn(
            "finite_one_sided_sqrt_rational_reverse_zero_tail_endpoint_supported",
            names,
        )
        self.assertIn(
            "finite_one_sided_sqrt_rational_domain_path_conflict_residual", names
        )
        self.assertIn("finite_one_sided_sqrt_domain_path_conflict_residual", names)
        self.assertIn(
            "finite_one_sided_sqrt_shifted_domain_path_conflict_residual", names
        )
        self.assertIn("finite_one_sided_acosh_lower_bound_endpoint_supported", names)
        self.assertIn(
            "finite_one_sided_acosh_lower_bound_domain_path_conflict_residual",
            names,
        )
        self.assertIn("finite_one_sided_inverse_trig_upper_endpoint_supported", names)
        self.assertIn(
            "finite_one_sided_inverse_trig_upper_domain_path_conflict_residual",
            names,
        )
        self.assertIn("finite_one_sided_atanh_upper_endpoint_supported", names)
        self.assertIn("finite_one_sided_atanh_lower_endpoint_supported", names)
        self.assertIn(
            "finite_one_sided_atanh_upper_domain_path_conflict_residual",
            names,
        )
        self.assertIn("finite_log_argument_zero_endpoint_residual", names)
        self.assertIn("finite_log_positive_scaled_abs_quotient_domain", names)
        self.assertIn("finite_log_exact_e_point_required_condition", names)
        self.assertIn("finite_fixed_base_log_argument_zero_endpoint_residual", names)
        self.assertIn(
            "finite_binary_log_constant_base_argument_zero_endpoint_residual",
            names,
        )
        self.assertIn("finite_acosh_bilateral_lower_bound_endpoint_supported", names)
        self.assertIn("finite_inverse_trig_bilateral_upper_endpoint_supported", names)
        self.assertIn(
            "finite_inverse_trig_empty_punctured_upper_endpoint_residual", names
        )
        self.assertIn(
            "finite_inverse_trig_empty_punctured_lower_endpoint_residual", names
        )
        self.assertIn("finite_inverse_trig_bilateral_lower_endpoint_supported", names)
        self.assertIn("finite_log_root_structurally_positive_composition", names)
        self.assertIn("finite_total_real_over_positive_sublimit_composition", names)
        self.assertIn("finite_log_rational_positive_sublimit_domain", names)
        self.assertIn("finite_sqrt_structurally_positive_radical_presentation", names)
        self.assertIn("finite_inverse_trig_root_interior_special_angle", names)
        self.assertIn("finite_binary_log_variable_base_domain", names)
        self.assertIn("finite_binary_log_unit_base_boundary_residual", names)
        self.assertIn("finite_binary_log_argument_zero_boundary_residual", names)
        self.assertIn("finite_static_invalid_log_undefined", names)
        self.assertIn(
            "finite_invalid_binary_log_base_dependent_argument_undefined",
            names,
        )
        self.assertIn(
            "finite_inverse_hyperbolic_empty_open_interval_undefined",
            names,
        )
        self.assertIn("finite_sqrt_bilateral_even_gap_endpoint_supported", names)
        self.assertIn("finite_sqrt_empty_punctured_endpoint_residual", names)
        self.assertIn(
            "finite_acosh_empty_punctured_lower_bound_endpoint_residual", names
        )
        self.assertIn("finite_sqrt_endpoint_residual_presentation_cleanup", names)
        self.assertIn("finite_discontinuous_sign_residual_presentation_cleanup", names)
        self.assertIn("finite_abs_orientation_quotient_residual_boundary", names)
        self.assertIn("finite_negative_integer_power_nonzero_root_base", names)
        self.assertIn("finite_trig_sine_even_power_pole_bilateral_supported", names)
        self.assertIn(
            "finite_trig_sine_special_point_even_power_pole_bilateral_supported",
            names,
        )
        self.assertIn(
            "finite_trig_sine_rational_pi_multiple_even_power_pole_bilateral_supported",
            names,
        )
        self.assertIn(
            "finite_trig_sine_higher_even_power_pole_bilateral_supported",
            names,
        )
        self.assertIn("finite_trig_cosine_even_power_pole_bilateral_supported", names)
        self.assertIn(
            "finite_trig_cosine_special_point_even_power_pole_bilateral_supported",
            names,
        )
        self.assertIn("finite_trig_special_angle_structural_domain", names)
        self.assertIn("finite_trig_table_undefined_pole_residual", names)
        self.assertIn(
            "finite_reciprocal_trig_sine_even_power_pole_bilateral_supported",
            names,
        )
        self.assertIn("finite_one_sided_tangent_pole_supported", names)
        self.assertIn("finite_reciprocal_trig_sine_pole_residual", names)
        self.assertIn("finite_one_sided_reciprocal_trig_sine_pole_supported", names)
        self.assertIn(
            "finite_one_sided_scaled_reciprocal_trig_sine_pole_supported", names
        )
        self.assertIn(
            "finite_one_sided_explicit_tangent_ratio_pole_supported", names
        )
        self.assertIn(
            "finite_one_sided_cross_argument_explicit_trig_ratio_pole_supported",
            names,
        )
        self.assertIn(
            "finite_one_sided_scaled_cross_argument_trig_ratio_orientation_supported",
            names,
        )
        self.assertIn(
            "finite_one_sided_symbolic_orientation_trig_ratio_pole_residual",
            names,
        )
        self.assertIn("finite_trig_small_angle_scaled_quotient", names)
        self.assertIn("finite_exp_zero_scaled_quotient", names)
        self.assertIn("finite_log_unit_scaled_quotient", names)
        self.assertIn("finite_fixed_base_log_unit_scaled_quotient", names)
        self.assertIn("finite_binary_log_constant_base_unit_scaled_quotient", names)
        self.assertIn("finite_binary_log_variable_base_unit_scaled_quotient", names)
        self.assertIn("finite_binary_log_resolved_base_unit_scaled_quotient", names)
        self.assertIn(
            "finite_binary_log_resolved_radical_base_unit_scaled_quotient", names
        )
        self.assertIn("negative_infinity_log_domain_conflict_residual", names)
        self.assertIn(
            "negative_infinity_log_domain_compatible_growth_supported",
            names,
        )
        self.assertIn("infinity_root_polynomial_tail_domain_suppression", names)
        self.assertIn("infinity_cbrt_polynomial_signed_tail_standalone", names)
        self.assertIn("infinity_asinh_polynomial_signed_tail_standalone", names)
        self.assertIn("infinity_atan_polynomial_bounded_tail_standalone", names)
        self.assertIn("infinity_sinh_polynomial_signed_tail_standalone", names)
        self.assertIn("infinity_exp_polynomial_signed_tail_standalone", names)
        self.assertIn("infinity_exp_polynomial_dominance_decay", names)
        self.assertIn(
            "infinity_exp_polynomial_subpolynomial_dominance_decay",
            names,
        )
        self.assertIn("infinity_root_rational_finite_tail_standalone", names)
        self.assertIn("infinity_root_rational_zero_tail_standalone", names)
        self.assertIn("infinity_cbrt_rational_signed_finite_tail_standalone", names)
        self.assertIn("infinity_log_polynomial_argument_standalone", names)
        self.assertIn("infinity_log_polynomial_argument_dominance", names)
        self.assertIn("infinity_log_rational_argument_dominance", names)
        self.assertIn("infinity_log_rational_zero_tail_dominance", names)
        self.assertIn("infinity_log_rational_finite_tail_standalone", names)
        self.assertIn("infinity_acosh_rational_finite_tail_standalone", names)
        self.assertIn("infinity_acosh_polynomial_argument_dominance", names)
        self.assertIn("infinity_bounded_polynomial_exp_decay", names)
        self.assertIn("negative_infinity_bounded_over_divergent_orientation_supported", names)
        self.assertIn("infinity_bounded_over_divergent_domain_conflict_residual", names)
        self.assertIn("finite_squeeze_bounded_oscillator_zero", names)
        self.assertIn("finite_squeeze_even_power_cosine_zero", names)
        self.assertIn("finite_squeeze_scaled_oscillator_residual", names)
        self.assertIn("finite_first_order_equiv_inversion_quotient", names)
        self.assertIn("finite_first_order_equiv_sin_sin_composition", names)
        self.assertIn("finite_first_order_equiv_sin_sin_half_composition", names)
        self.assertIn("finite_first_order_equiv_tan_quotient", names)
        self.assertIn("finite_first_order_equiv_asin_quotient", names)
        self.assertIn("finite_first_order_equiv_arctan_quotient", names)
        self.assertIn("finite_first_order_equiv_sinh_quotient", names)
        self.assertIn("finite_first_order_equiv_tanh_quotient", names)
        self.assertEqual(
            SMOKE.count_by(cases, "point_regime"),
            {"finite": 78, "finite_one_sided": 39, "infinity": 44},
        )
        self.assertEqual(
            SMOKE.count_by(cases, "outcome"),
            {"residual": 28, "supported": 130, "undefined": 3},
        )
        self.assertEqual(
            SMOKE.count_residual_causes(cases),
            {
                "finite_discontinuity_or_orientation": 3,
                "finite_endpoint_empty_punctured_domain_policy": 4,
                "finite_endpoint_or_boundary_policy": 7,
                "finite_oscillatory_no_infinitesimal": 2,
                "finite_rational_pole_policy": 1,
                "finite_trig_symbolic_orientation_policy": 1,
                "finite_trig_pole_policy": 2,
                "infinity_domain_path_conflict": 2,
                "one_sided_domain_path_conflict": 6,
            },
        )
        self.assertEqual(
            SMOKE.count_residual_families(cases),
            {
                "abs_orientation": 1,
                "binary_log_domain_policy": 3,
                "bounded_domain_policy": 1,
                "discontinuous": 2,
                "even_saturating_composition": 1,
                "finite_squeeze_bounded_product": 1,
                "fixed_base_log_domain_policy": 1,
                "inverse_hyperbolic": 2,
                "inverse_hyperbolic_domain_policy": 1,
                "inverse_trig": 3,
                "log": 1,
                "log_domain_policy": 1,
                "rational": 1,
                "reciprocal_trig": 1,
                "root": 3,
                "root_endpoint": 2,
                "root_rational_endpoint": 1,
                "trig_ratio": 1,
                "trig_special_angle": 1,
            },
        )
        self.assertEqual(
            SMOKE.count_residual_cause_families(cases),
            {
                "finite_discontinuity_or_orientation/abs_orientation": 1,
                "finite_discontinuity_or_orientation/discontinuous": 2,
                (
                    "finite_endpoint_empty_punctured_domain_policy/"
                    "inverse_hyperbolic_domain_policy"
                ): 1,
                "finite_endpoint_empty_punctured_domain_policy/inverse_trig": 2,
                "finite_endpoint_empty_punctured_domain_policy/root": 1,
                "finite_endpoint_or_boundary_policy/binary_log_domain_policy": 3,
                "finite_endpoint_or_boundary_policy/fixed_base_log_domain_policy": 1,
                "finite_endpoint_or_boundary_policy/log_domain_policy": 1,
                "finite_endpoint_or_boundary_policy/root": 2,
                "finite_oscillatory_no_infinitesimal/even_saturating_composition": 1,
                "finite_oscillatory_no_infinitesimal/finite_squeeze_bounded_product": 1,
                "finite_rational_pole_policy/rational": 1,
                "finite_trig_pole_policy/reciprocal_trig": 1,
                "finite_trig_pole_policy/trig_special_angle": 1,
                "finite_trig_symbolic_orientation_policy/trig_ratio": 1,
                "infinity_domain_path_conflict/bounded_domain_policy": 1,
                "infinity_domain_path_conflict/log": 1,
                "one_sided_domain_path_conflict/inverse_hyperbolic": 2,
                "one_sided_domain_path_conflict/inverse_trig": 1,
                "one_sided_domain_path_conflict/root_endpoint": 2,
                "one_sided_domain_path_conflict/root_rational_endpoint": 1,
            },
        )
        self.assertEqual(
            SMOKE.group_residual_cases_by_cause(cases)[
                "finite_endpoint_or_boundary_policy"
            ][:3],
            [
                "finite_log_argument_zero_endpoint_residual",
                "finite_fixed_base_log_argument_zero_endpoint_residual",
                "finite_binary_log_constant_base_argument_zero_endpoint_residual",
            ],
        )
        self.assertIn(
            "finite_one_sided_sqrt_rational_domain_path_conflict_residual",
            SMOKE.group_residual_cases_by_cause(cases)[
                "one_sided_domain_path_conflict"
            ],
        )
        self.assertEqual(
            SMOKE.count_calculus_maturity_blocks(cases),
            {
                "block3_real_domain_limits": 130,
                "block9_residuals_and_non_goals": 31,
            },
        )
        self.assertEqual(
            SMOKE.count_calculus_block_gates(cases),
            {
                "didactic_trace_and_limit_policy": 50,
                "domain_conditions_and_limit_policy": 80,
                "explicit_undefined_domain_policy": 3,
                "safe_residual_policy": 28,
            },
        )
        self.assertEqual(
            SMOKE.count_by(cases, "required_condition_regime"),
            {
                "finite_acosh_bilateral_lower_bound_endpoint_domain": 1,
                "finite_abs_orientation_denominator_domain": 1,
                "finite_binary_log_constant_base_endpoint_residual_domain": 1,
                "finite_binary_log_constant_base_unit_denominator_domain": 1,
                "finite_binary_log_resolved_base_unit_denominator_and_base_domain": 1,
                "finite_binary_log_resolved_radical_base_unit_denominator_base_and_radical_domain": 1,
                "finite_binary_log_variable_base_unit_denominator_and_base_domain": 1,
                "finite_bilateral_reciprocal_trig_sine_even_power_pole_domain": 1,
                "finite_bilateral_abs_even_order_pole_domain": 1,
                "finite_bilateral_even_order_rational_pole_domain": 1,
                "finite_bilateral_log_even_order_endpoint_domain": 1,
                "finite_bilateral_reciprocal_base_log_even_order_endpoint_domain": 1,
                "finite_bilateral_trig_cosine_power_pole_domain": 1,
                "finite_bilateral_trig_cosine_special_point_power_pole_domain": 1,
                "finite_bilateral_trig_sine_higher_even_power_pole_domain": 1,
                "finite_bilateral_trig_sine_power_pole_domain": 1,
                "finite_bilateral_trig_sine_rational_pi_multiple_power_pole_domain": 1,
                "finite_bilateral_trig_sine_special_point_power_pole_domain": 1,
                "finite_boundary_residual_domain": 2,
                "finite_empty_punctured_endpoint_residual_domain": 1,
                "finite_empty_punctured_inverse_trig_endpoint_residual_domain": 2,
                "finite_empty_punctured_lower_bound_endpoint_residual_domain": 1,
                "finite_endpoint_residual_domain": 2,
                "finite_first_order_equiv_denominator_domain": 8,
                "finite_fixed_base_log_endpoint_residual_domain": 1,
                "finite_fixed_base_log_unit_denominator_domain": 1,
                "finite_interval_domain": 1,
                "finite_inverse_trig_bilateral_lower_endpoint_domain": 1,
                "finite_inverse_trig_bilateral_upper_endpoint_domain": 1,
                "finite_log_base_argument_domain": 1,
                "finite_log_exact_e_point_positive_domain": 1,
                "finite_log_endpoint_residual_domain": 1,
                "finite_log_rational_positive_sublimit_domain": 1,
                "finite_log_unit_denominator_domain": 1,
                "finite_positive_scaled_abs_quotient_domain": 1,
                "finite_one_sided_acosh_lower_bound_endpoint_domain": 1,
                "finite_one_sided_atanh_endpoint_domain": 1,
                "finite_one_sided_atanh_lower_endpoint_domain": 1,
                "finite_one_sided_atanh_path_conflict": 1,
                "finite_one_sided_inverse_trig_endpoint_domain": 1,
                "finite_one_sided_inverse_trig_path_conflict": 1,
                "finite_one_sided_lower_bound_path_conflict": 1,
                "finite_one_sided_constant_base_less_than_one_log_endpoint_domain": 1,
                "finite_one_sided_fixed_base_log_endpoint_domain": 1,
                "finite_positive_domain": 1,
                "finite_positive_nonzero_power_domain": 1,
                "finite_exp_zero_denominator_domain": 1,
                "finite_one_sided_log_endpoint_domain": 1,
                "finite_one_sided_log_rational_zero_tail_endpoint_domain": 1,
                "finite_one_sided_rational_unit_base_boundary_log_zero_tail_endpoint_domain": 1,
                "finite_one_sided_unit_base_boundary_log_rational_zero_tail_endpoint_domain": 1,
                "finite_one_sided_variable_base_log_rational_zero_tail_endpoint_domain": 1,
                "finite_one_sided_factored_abs_orientation_domain": 1,
                "finite_one_sided_orientation_domain": 1,
                "finite_one_sided_path_conflict": 2,
                "finite_one_sided_rational_pole_domain": 1,
                "finite_one_sided_reciprocal_trig_sine_pole_domain": 1,
                "finite_one_sided_scaled_reciprocal_trig_sine_pole_domain": 1,
                "finite_one_sided_cross_argument_explicit_trig_ratio_pole_domain": 1,
                "finite_one_sided_scaled_cross_argument_trig_ratio_orientation_domain": 1,
                "finite_one_sided_explicit_tangent_ratio_pole_domain": 1,
                "finite_one_sided_symbolic_orientation_trig_ratio_domain": 1,
                "finite_one_sided_tangent_pole_domain": 1,
                "finite_one_sided_rational_path_conflict": 1,
                "finite_one_sided_removable_hole": 1,
                "finite_one_sided_removable_hole_with_nonlocal_pole_domain": 1,
                "finite_one_sided_root_endpoint_domain": 3,
                "finite_one_sided_root_rational_zero_tail_endpoint_domain": 2,
                "finite_one_sided_shifted_scaled_odd_order_pole_domain": 1,
                "finite_rational_pole_residual_domain": 1,
                "finite_removable_hole": 1,
                "finite_removable_radical_hole": 3,
                "finite_small_angle_denominator_domain": 1,
                "finite_source_definedness": 13,
                "finite_trig_sine_pole_residual_domain": 1,
                "finite_trig_pole_residual_domain": 1,
                "infinity_path_compatible_domain": 3,
                "infinity_path_compatible_polynomial_domain": 1,
                "infinity_path_compatible_polynomial_positive_domain": 6,
                "infinity_path_compatible_rational_finite_tail_domain": 2,
                "infinity_path_compatible_rational_lower_bound_finite_tail_domain": 1,
                "infinity_path_compatible_rational_positive_domain": 1,
                "infinity_path_compatible_rational_zero_tail_domain": 2,
                "infinity_path_conflict": 2,
                "infinity_path_total_real_polynomial_tail_domain": 5,
                "infinity_path_radical_unbounded_tail_domain": 2,
                "infinity_path_total_real_function": 1,
                "infinity_path_total_real_rational_finite_tail_domain": 1,
                "infinity_source_definedness": 1,
                "none": 29,
            },
        )
        self.assertEqual(
            sum(1 for case in cases if case.expected_step_substrings),
            161,
        )
        self.assertEqual(
            [
                case.name
                for case in cases
                if case.outcome == "supported" and not case.expected_step_substrings
            ],
            [],
        )
        self.assertGreaterEqual(len({case.family for case in cases}), 7)

    def test_run_matrix_accepts_expected_warning_and_required_display(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    case "$*" in
                    *"limit(x^2 + x + 1, x, -2)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"3","warnings":[],"required_display":[],"steps":[{"rule":"Evaluar límite finito","before":"x^2 + x + 1","after":"3"}]}
                    OUT
                    ;;
                    *"limit((x^2-1)/(x-1), x, 1)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"2","warnings":[],"required_display":["x ≠ 1"],"steps":[{"rule":"Evaluar límite finito","before":"(x^2 - 1)/(x - 1)","after":"2"}]}
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
                    "finite_polynomial_supported",
                    "finite_removable_rational_cancellation",
                )
            )
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "pass", matrix)
        self.assertEqual(matrix["total"], 2)
        self.assertEqual(matrix["status_counts"]["pass"], 2)
        self.assertEqual(matrix["supported_case_count"], 2)
        self.assertEqual(matrix["residual_case_count"], 0)
        self.assertEqual(matrix["warning_expected_case_count"], 0)
        self.assertEqual(matrix["required_display_case_count"], 1)
        self.assertEqual(matrix["step_checked_case_count"], 2)
        self.assertEqual(matrix["supported_step_unchecked_case_count"], 0)
        self.assertEqual(matrix["supported_step_unchecked_cases"], [])
        self.assertEqual(matrix["expected_step_substring_count"], 2)
        self.assertEqual(matrix["required_display_counts"], {"x ≠ 1": 1})
        self.assertEqual(
            matrix["calculus_maturity_block_counts"],
            {"block3_real_domain_limits": 2},
        )
        self.assertEqual(
            matrix["calculus_block_gate_counts"],
            {
                "didactic_trace_and_limit_policy": 1,
                "domain_conditions_and_limit_policy": 1,
            },
        )

    def test_run_matrix_reports_required_display_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    cat <<'OUT'
                    {"ok":true,"result":"1","warnings":[],"required_display":[]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(("finite_rational_required_condition",))
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "fail")
        self.assertEqual(matrix["status_counts"]["fail"], 1)
        self.assertEqual(matrix["issue_kind_counts"], {"required_display_mismatch": 1})
        self.assertEqual(matrix["problem_cases"][0]["name"], "finite_rational_required_condition")

    def test_run_matrix_reports_missing_expected_step_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    cat <<'OUT'
                    {"ok":true,"result":"3","warnings":[],"required_display":[],"steps":[]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(("finite_polynomial_supported",))
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "fail")
        self.assertEqual(matrix["status_counts"]["fail"], 1)
        self.assertEqual(matrix["issue_kind_counts"], {"step_trace_mismatch": 1})
        self.assertEqual(matrix["problem_cases"][0]["name"], "finite_polynomial_supported")

    def test_run_matrix_reports_fragile_stderr_even_when_result_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    echo 'WARN simplify: depth_overflow' >&2
                    cat <<'OUT'
                    {"ok":true,"result":"3","warnings":[],"required_display":[],"steps":[{"rule":"Evaluar límite finito","before":"x^2 + x + 1","after":"3"}]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(("finite_polynomial_supported",))
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "fail")
        self.assertEqual(matrix["status_counts"]["fail"], 1)
        self.assertEqual(matrix["issue_kind_counts"], {"stderr_fragility": 1})
        self.assertEqual(matrix["problem_cases"][0]["name"], "finite_polynomial_supported")
        self.assertIn("fragile substring", matrix["problem_cases"][0]["error"])


if __name__ == "__main__":
    unittest.main()
