import importlib.util
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "engine_diff_command_matrix_smoke.py"
SPEC = importlib.util.spec_from_file_location("engine_diff_command_matrix_smoke", MODULE_PATH)
SMOKE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)


class DiffCommandMatrixSmokeTests(unittest.TestCase):
    def test_default_matrix_covers_diff_policy_axes(self) -> None:
        cases = SMOKE.build_cases()

        self.assertEqual(len(cases), 40)
        names = {case.name for case in cases}
        self.assertIn(
            "log_quadratic_empty_positive_argument_domain_undefined",
            names,
        )
        self.assertIn("general_base_log_unit_base_domain_undefined", names)
        self.assertIn("polynomial_inner_chain_power", names)
        self.assertIn("elementary_exp_affine_chain_trace", names)
        self.assertIn("elementary_trig_affine_chain_trace", names)
        self.assertIn("elementary_trig_tan_affine_chain_required_condition", names)
        self.assertIn("log_affine_chain_required_domain", names)
        self.assertIn(
            "log_affine_chain_negative_orientation_required_domain",
            names,
        )
        self.assertIn("constant_base_exp_affine_chain_positive_base", names)
        self.assertIn("constant_base_exp_affine_chain_negative_base_undefined", names)
        self.assertIn(
            "constant_base_exp_affine_chain_zero_base_positive_exponent_domain",
            names,
        )
        self.assertIn(
            "constant_base_exp_quadratic_zero_base_empty_domain_undefined",
            names,
        )
        self.assertIn("nonfinite_constant_derivative_undefined", names)
        self.assertIn(
            "sqrt_quadratic_empty_positive_argument_domain_undefined",
            names,
        )
        self.assertIn("inverse_trig_root_compact_presentation", names)
        self.assertIn("inverse_trig_root_constant_multiple_trace", names)
        self.assertIn("inverse_trig_root_symbolic_denominator_scale", names)
        self.assertIn("inverse_trig_root_symbolic_rational_denominator_scale", names)
        self.assertIn(
            "inverse_trig_root_symbolic_rational_denominator_affine_radicand_shortcut",
            names,
        )
        self.assertIn(
            "inverse_trig_root_symbolic_rational_denominator_affine_radicand_external_scale_shortcut",
            names,
        )
        self.assertIn(
            "inverse_trig_root_symbolic_numerator_scale_positive_gap",
            names,
        )
        self.assertIn("inverse_trig_root_symbolic_denominator_internal_scale", names)
        self.assertIn(
            "inverse_trig_root_symbolic_denominator_scale_dual_orientation",
            names,
        )
        self.assertIn("inverse_trig_root_interval_orientation", names)
        self.assertIn("inverse_trig_root_empty_open_interval_residual", names)
        self.assertIn(
            "inverse_trig_shifted_quadratic_empty_open_interval_residual",
            names,
        )
        self.assertIn(
            "inverse_trig_symbolic_constant_empty_open_interval_residual",
            names,
        )
        self.assertIn("inverse_hyperbolic_atanh_empty_open_interval_residual", names)
        self.assertIn(
            "inverse_hyperbolic_root_atanh_symbolic_numerator_scale_open_interval",
            names,
        )
        self.assertIn(
            "inverse_hyperbolic_root_symbolic_numerator_scale_positive_gap",
            names,
        )
        self.assertIn("inverse_trig_root_negative_argument", names)
        self.assertIn(
            "sqrt_chain_trig_log_presimplified_condition_dedupe",
            names,
        )
        self.assertIn("discontinuous_sign_residual_boundary", names)
        self.assertEqual(
            SMOKE.count_by(cases, "outcome"),
            {"residual": 5, "supported": 29, "undefined": 6},
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
                "product_log_required_condition",
                "log_quadratic_empty_positive_argument_domain_undefined",
                "general_base_log_unit_base_domain_undefined",
                "polynomial_power_direct",
                "polynomial_inner_chain_power",
                "elementary_exp_affine_chain_trace",
                "elementary_trig_affine_chain_trace",
                "elementary_trig_tan_affine_chain_required_condition",
                "log_affine_chain_required_domain",
                "log_affine_chain_negative_orientation_required_domain",
                "constant_base_exp_affine_chain_positive_base",
                "constant_base_exp_affine_chain_negative_base_undefined",
                "constant_base_exp_affine_chain_zero_base_positive_exponent_domain",
                "constant_base_exp_quadratic_zero_base_empty_domain_undefined",
                "nonfinite_constant_derivative_undefined",
                "rational_quotient_required_condition",
                "sqrt_variable_open_domain",
                "sqrt_quadratic_empty_positive_argument_domain_undefined",
                "inverse_trig_root_compact_presentation",
                "inverse_trig_root_constant_multiple_trace",
                "inverse_trig_root_symbolic_denominator_scale",
                "inverse_trig_root_symbolic_rational_denominator_scale",
                "inverse_trig_root_symbolic_rational_denominator_affine_radicand_shortcut",
                "inverse_trig_root_symbolic_rational_denominator_affine_radicand_external_scale_shortcut",
                "inverse_trig_root_symbolic_numerator_scale_positive_gap",
                "inverse_trig_root_symbolic_denominator_internal_scale",
                "inverse_trig_root_symbolic_denominator_scale_dual_orientation",
                "inverse_trig_root_interval_orientation",
                "inverse_trig_shifted_quadratic_empty_open_interval_residual",
                "inverse_hyperbolic_root_atanh_symbolic_numerator_scale_open_interval",
                "inverse_hyperbolic_root_symbolic_numerator_scale_positive_gap",
                "inverse_trig_root_negative_argument",
                "trig_product_rule",
                "sqrt_chain_trig_log_presimplified_condition_dedupe",
                "variable_power_log_domain",
                "abs_piecewise_required_condition",
                "discontinuous_sign_residual_boundary",
            },
        )
        self.assertEqual(supported_step_unchecked, set())
        self.assertEqual(
            SMOKE.count_by(cases, "domain_regime"),
            {
                "discontinuous_residual": 1,
                "empty_open_interval_domain": 4,
                "empty_positive_argument_domain": 2,
                "empty_positive_exponent_domain": 1,
                "invalid_log_base_domain": 1,
                "interval_required": 1,
                "negative_base_undefined": 1,
                "nonfinite_undefined": 1,
                "open_interval_required": 1,
                "positive_exponent_required": 1,
                "required_condition": 21,
                "unconditional": 5,
            },
        )
        self.assertGreaterEqual(len({case.family for case in cases}), 11)

    def test_run_matrix_accepts_expected_required_display_and_residual(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    case "$*" in
                    *"diff(x^3, x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"3·x^2","warnings":[],"required_display":[],"steps":[{"substeps":[{"title":"Usar regla de la potencia"}]}]}
                    OUT
                    ;;
                    *"diff((x^2+1)^3, x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"6·x·(x^2 + 1)^2","warnings":[],"required_display":[],"steps":[{"substeps":[{"title":"Usar regla de la potencia con cadena"},{"title":"Identificar u y du"}]}]}
                    OUT
                    ;;
                    *"diff(sqrt(x), x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"1 / (2·sqrt(x))","warnings":[],"required_display":["x > 0"],"steps":[{"substeps":[{"title":"Usar regla de la potencia"}]}]}
                    OUT
                    ;;
                    *"diff(sign(x), x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"diff(sign(x), x)","warnings":[],"required_display":[],"steps":[{"rule":"Conservar derivada residual","before":"sign(x)","after":"diff(sign(x), x)"}]}
                    OUT
                    ;;
                    *)
                    echo '{"ok":false,"result":"unexpected","warnings":[],"required_display":[],"steps":[]}'
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
                    "polynomial_inner_chain_power",
                    "sqrt_variable_open_domain",
                    "discontinuous_sign_residual_boundary",
                )
            )
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "pass", matrix)
        self.assertEqual(matrix["total"], 4)
        self.assertEqual(matrix["status_counts"]["pass"], 4)
        self.assertEqual(matrix["supported_case_count"], 3)
        self.assertEqual(matrix["residual_case_count"], 1)
        self.assertEqual(matrix["warning_expected_case_count"], 0)
        self.assertEqual(matrix["blocked_hint_expected_case_count"], 0)
        self.assertEqual(matrix["required_display_case_count"], 1)
        self.assertEqual(matrix["required_display_counts"], {"x > 0": 1})
        self.assertEqual(matrix["step_checked_case_count"], 4)
        self.assertEqual(matrix["supported_step_unchecked_case_count"], 0)
        self.assertEqual(matrix["supported_step_unchecked_cases"], [])
        self.assertEqual(matrix["expected_step_substring_count"], 5)

    def test_run_matrix_reports_result_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    cat <<'OUT'
                    {"ok":true,"result":"0","warnings":[],"required_display":[],"steps":[]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(("polynomial_power_direct",))
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "fail")
        self.assertEqual(matrix["status_counts"]["fail"], 1)
        self.assertEqual(matrix["issue_kind_counts"], {"result_mismatch": 1})
        self.assertEqual(matrix["problem_cases"][0]["name"], "polynomial_power_direct")

    def test_run_matrix_reports_missing_expected_step_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    cat <<'OUT'
                    {"ok":true,"result":"6·x·(x^2 + 1)^2","warnings":[],"required_display":[],"steps":[{"rule":"Calcular la derivada"}]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(("polynomial_inner_chain_power",))
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "fail")
        self.assertEqual(matrix["status_counts"]["fail"], 1)
        self.assertEqual(matrix["issue_kind_counts"], {"step_trace_mismatch": 1})
        self.assertEqual(matrix["problem_cases"][0]["name"], "polynomial_inner_chain_power")


if __name__ == "__main__":
    unittest.main()
