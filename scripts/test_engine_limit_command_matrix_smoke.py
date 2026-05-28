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

        self.assertEqual(len(cases), 84)
        names = {case.name for case in cases}
        self.assertIn("finite_removable_rational_cancellation", names)
        self.assertIn("finite_rational_simple_pole_residual", names)
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
        self.assertIn("finite_one_sided_log_zero_endpoint_supported", names)
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
        self.assertIn(
            "finite_one_sided_sqrt_rational_zero_tail_endpoint_supported", names
        )
        self.assertIn(
            "finite_one_sided_sqrt_rational_domain_path_conflict_residual", names
        )
        self.assertIn("finite_one_sided_sqrt_domain_path_conflict_residual", names)
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
        self.assertIn("finite_fixed_base_log_argument_zero_endpoint_residual", names)
        self.assertIn(
            "finite_binary_log_constant_base_argument_zero_endpoint_residual",
            names,
        )
        self.assertIn("finite_acosh_bilateral_lower_bound_endpoint_supported", names)
        self.assertIn("finite_inverse_trig_bilateral_upper_endpoint_supported", names)
        self.assertIn("finite_inverse_trig_bilateral_lower_endpoint_supported", names)
        self.assertIn("finite_log_root_structurally_positive_composition", names)
        self.assertIn("finite_log_rational_positive_sublimit_domain", names)
        self.assertIn("finite_sqrt_structurally_positive_radical_presentation", names)
        self.assertIn("finite_inverse_trig_root_interior_special_angle", names)
        self.assertIn("finite_binary_log_variable_base_domain", names)
        self.assertIn("finite_binary_log_unit_base_boundary_residual", names)
        self.assertIn("finite_binary_log_argument_zero_boundary_residual", names)
        self.assertIn("finite_static_invalid_log_undefined", names)
        self.assertIn("finite_sqrt_endpoint_residual_presentation_cleanup", names)
        self.assertIn("finite_discontinuous_sign_residual_presentation_cleanup", names)
        self.assertIn("finite_abs_orientation_quotient_residual_boundary", names)
        self.assertIn("finite_negative_integer_power_nonzero_root_base", names)
        self.assertIn("finite_trig_special_angle_structural_domain", names)
        self.assertIn("finite_trig_table_undefined_pole_residual", names)
        self.assertIn("finite_reciprocal_trig_sine_pole_residual", names)
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
        self.assertEqual(
            SMOKE.count_by(cases, "point_regime"),
            {"finite": 36, "finite_one_sided": 21, "infinity": 27},
        )
        self.assertEqual(
            SMOKE.count_by(cases, "outcome"),
            {"residual": 20, "supported": 63, "undefined": 1},
        )
        self.assertEqual(
            SMOKE.count_calculus_maturity_blocks(cases),
            {
                "block3_real_domain_limits": 63,
                "block9_residuals_and_non_goals": 21,
            },
        )
        self.assertEqual(
            SMOKE.count_calculus_block_gates(cases),
            {
                "didactic_trace_and_limit_policy": 29,
                "domain_conditions_and_limit_policy": 34,
                "explicit_undefined_domain_policy": 1,
                "safe_residual_policy": 20,
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
                "finite_boundary_residual_domain": 2,
                "finite_endpoint_residual_domain": 2,
                "finite_fixed_base_log_endpoint_residual_domain": 1,
                "finite_fixed_base_log_unit_denominator_domain": 1,
                "finite_interval_domain": 1,
                "finite_inverse_trig_bilateral_lower_endpoint_domain": 1,
                "finite_inverse_trig_bilateral_upper_endpoint_domain": 1,
                "finite_log_base_argument_domain": 1,
                "finite_log_endpoint_residual_domain": 1,
                "finite_log_rational_positive_sublimit_domain": 1,
                "finite_log_unit_denominator_domain": 1,
                "finite_one_sided_acosh_lower_bound_endpoint_domain": 1,
                "finite_one_sided_atanh_endpoint_domain": 1,
                "finite_one_sided_atanh_lower_endpoint_domain": 1,
                "finite_one_sided_atanh_path_conflict": 1,
                "finite_one_sided_inverse_trig_endpoint_domain": 1,
                "finite_one_sided_inverse_trig_path_conflict": 1,
                "finite_one_sided_lower_bound_path_conflict": 1,
                "finite_positive_domain": 1,
                "finite_positive_nonzero_power_domain": 1,
                "finite_exp_zero_denominator_domain": 1,
                "finite_one_sided_log_endpoint_domain": 1,
                "finite_one_sided_log_rational_zero_tail_endpoint_domain": 1,
                "finite_one_sided_rational_unit_base_boundary_log_zero_tail_endpoint_domain": 1,
                "finite_one_sided_unit_base_boundary_log_rational_zero_tail_endpoint_domain": 1,
                "finite_one_sided_variable_base_log_rational_zero_tail_endpoint_domain": 1,
                "finite_one_sided_orientation_domain": 1,
                "finite_one_sided_path_conflict": 1,
                "finite_one_sided_rational_pole_domain": 1,
                "finite_one_sided_rational_path_conflict": 1,
                "finite_one_sided_removable_hole": 1,
                "finite_one_sided_removable_hole_with_nonlocal_pole_domain": 1,
                "finite_one_sided_root_endpoint_domain": 1,
                "finite_one_sided_root_rational_zero_tail_endpoint_domain": 1,
                "finite_one_sided_shifted_scaled_odd_order_pole_domain": 1,
                "finite_rational_pole_residual_domain": 1,
                "finite_removable_hole": 1,
                "finite_small_angle_denominator_domain": 1,
                "finite_source_definedness": 1,
                "finite_trig_sine_pole_residual_domain": 1,
                "finite_trig_pole_residual_domain": 1,
                "infinity_path_compatible_domain": 3,
                "infinity_path_compatible_polynomial_domain": 1,
                "infinity_path_compatible_polynomial_positive_domain": 3,
                "infinity_path_compatible_rational_finite_tail_domain": 2,
                "infinity_path_compatible_rational_lower_bound_finite_tail_domain": 1,
                "infinity_path_compatible_rational_positive_domain": 1,
                "infinity_path_compatible_rational_zero_tail_domain": 2,
                "infinity_path_conflict": 2,
                "infinity_path_total_real_polynomial_tail_domain": 5,
                "infinity_path_total_real_rational_finite_tail_domain": 1,
                "infinity_source_definedness": 1,
                "none": 12,
            },
        )
        self.assertEqual(
            sum(1 for case in cases if case.expected_step_substrings),
            84,
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
