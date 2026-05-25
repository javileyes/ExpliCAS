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

        self.assertEqual(len(cases), 39)
        names = {case.name for case in cases}
        self.assertIn("reciprocal_affine_log_abs_domain", names)
        self.assertIn("reciprocal_negative_affine_log_abs_domain", names)
        self.assertIn("reciprocal_negative_affine_derivative_log_abs_domain", names)
        self.assertIn("log_derivative_positive_quadratic_substitution", names)
        self.assertIn("log_rational_positive_domain_residual", names)
        self.assertIn("non_elementary_tan_polynomial_residual_domain", names)
        self.assertIn("non_elementary_tan_presimplified_residual_domain", names)
        self.assertIn("non_elementary_csc_presimplified_residual_domain", names)
        self.assertIn("explicit_reciprocal_sine_presimplified_residual_domain", names)
        self.assertIn("explicit_reciprocal_tangent_presimplified_residual_domain", names)
        self.assertIn("explicit_reciprocal_tangent_verified_log_domain", names)
        self.assertIn(
            "explicit_reciprocal_hyperbolic_tangent_verified_log_domain",
            names,
        )
        self.assertIn(
            "explicit_reciprocal_hyperbolic_tangent_presimplified_condition_dedupe",
            names,
        )
        self.assertIn("additive_trig_pole_residual_domain", names)
        self.assertIn("constant_multiple_affine_trig_substitution", names)
        self.assertIn("inverse_trig_sqrt_reciprocal_bridge", names)
        self.assertIn("inverse_trig_scaled_sqrt_reciprocal_bridge", names)
        self.assertIn("affine_exp_substitution", names)
        self.assertIn("by_parts_affine_log_domain", names)
        self.assertIn("inverse_hyperbolic_rational_direct_atanh_domain", names)
        self.assertIn("affine_inverse_hyperbolic_atanh_domain", names)
        self.assertIn("inverse_sqrt_direct_arcsin_domain", names)
        self.assertIn("additive_inverse_sqrt_interval_residual_domain", names)
        self.assertIn("inverse_sqrt_direct_asinh_unconditional", names)
        self.assertIn("polynomial_base_sqrt_substitution", names)
        self.assertIn("sqrt_chain_secant_tangent_domain", names)
        self.assertIn(
            "sqrt_chain_hyperbolic_tangent_presimplified_condition_dedupe",
            names,
        )
        self.assertIn("invalid_log_base_integrand_undefined", names)
        self.assertIn("nonfinite_integrand_undefined", names)
        self.assertIn("non_elementary_exp_quadratic_residual", names)
        self.assertEqual(
            SMOKE.count_by(cases, "outcome"),
            {"residual": 9, "supported": 28, "undefined": 2},
        )
        self.assertEqual(
            sum(1 for case in cases if case.expected_derivative_result is not None),
            28,
        )
        self.assertEqual(
            SMOKE.count_verification_regimes(cases),
            {
                "residual_not_verified": 9,
                "undefined_not_verified": 2,
                "verified_by_diff": 28,
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
                "reciprocal_affine_log_abs_domain",
                "reciprocal_negative_affine_log_abs_domain",
                "reciprocal_negative_affine_derivative_log_abs_domain",
                "log_derivative_positive_quadratic_substitution",
                "log_rational_positive_domain_residual",
                "non_elementary_tan_polynomial_residual_domain",
                "non_elementary_tan_presimplified_residual_domain",
                "non_elementary_csc_presimplified_residual_domain",
                "explicit_reciprocal_sine_presimplified_residual_domain",
                "explicit_reciprocal_tangent_presimplified_residual_domain",
                "explicit_reciprocal_tangent_verified_log_domain",
                "explicit_reciprocal_hyperbolic_tangent_verified_log_domain",
                "explicit_reciprocal_hyperbolic_tangent_presimplified_condition_dedupe",
                "additive_trig_pole_residual_domain",
                "inverse_trig_table",
                "inverse_trig_sqrt_reciprocal_bridge",
                "inverse_trig_scaled_sqrt_reciprocal_bridge",
                "inverse_hyperbolic_rational_direct_atanh_domain",
                "affine_inverse_hyperbolic_atanh_domain",
                "inverse_sqrt_direct_arcsin_domain",
                "additive_inverse_sqrt_interval_residual_domain",
                "inverse_sqrt_direct_asinh_unconditional",
                "affine_inverse_sqrt_arcsin_domain",
                "polynomial_base_sqrt_substitution",
                "hyperbolic_reciprocal_square_substitution",
                "by_parts_log_domain",
                "by_parts_affine_log_domain",
                "sqrt_chain_secant_tangent_domain",
                "sqrt_chain_tangent_log_domain",
                "sqrt_chain_hyperbolic_tangent_presimplified_condition_dedupe",
                "invalid_log_base_integrand_undefined",
                "nonfinite_integrand_undefined",
                "non_elementary_exp_quadratic_residual",
            },
        )
        self.assertEqual(supported_step_unchecked, set())
        self.assertEqual(
            SMOKE.count_by(cases, "domain_regime"),
            {
                "empty_real_domain": 1,
                "explicit_denominator_source_condition": 1,
                "explicit_hyperbolic_tangent_presimplified_condition_dedupe": 1,
                "explicit_hyperbolic_tangent_denominator_verified_substitution": 1,
                "explicit_tangent_denominator_source_condition": 1,
                "explicit_tangent_denominator_verified_substitution": 1,
                "nonzero_required": 3,
                "nonfinite_undefined": 1,
                "positive_required": 4,
                "positive_required_residual": 1,
                "radical_interval_additive_residual": 1,
                "radical_interval": 2,
                "rational_interval": 2,
                "sqrt_chain_nonzero_positive": 2,
                "sqrt_chain_hyperbolic_presimplified_condition_dedupe": 1,
                "structurally_positive_log_argument": 1,
                "trig_pole_additive_residual": 1,
                "trig_pole_presimplified_residual": 1,
                "trig_pole_residual": 1,
                "trig_sine_pole_presimplified_residual": 1,
                "unconditional": 11,
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
        self.assertEqual(
            matrix["verification_regime_counts"],
            {"residual_not_verified": 1, "verified_by_diff": 3},
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


if __name__ == "__main__":
    unittest.main()
