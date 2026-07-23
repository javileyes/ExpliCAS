import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "engine_dsolve_command_matrix_smoke.py"
SPEC = importlib.util.spec_from_file_location("engine_dsolve_command_matrix_smoke", MODULE_PATH)
SMOKE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)


class DsolveCommandMatrixSmokeTests(unittest.TestCase):
    def test_default_matrix_covers_o0_axes(self) -> None:
        cases = SMOKE.build_cases()

        # O0-O9+μ registry: 7 separable + 4 linear + 3 exact + 3
        # first-order-IVP + 5 second-order + 5 UC + 3 Bernoulli + 2
        # homogeneous + 3 system + 3 Cauchy-Euler + 2 integrating-factor
        # supported rows + 11 honest residual rows.
        self.assertEqual(len(cases), 51)
        names = {case.name for case in cases}
        self.assertIn("separable_growth_textbook", names)
        self.assertIn("separable_implicit_circle", names)
        self.assertIn("separable_parametric_rate", names)
        self.assertIn("separable_direct_integration", names)
        self.assertIn("separable_arity2_sugar", names)
        self.assertIn("linear_basic_integrating_factor", names)
        self.assertIn("linear_mu_display_strip_abs", names)
        self.assertIn("linear_first_order_resonance", names)
        self.assertIn("linear_trig_rhs", names)
        self.assertIn("exact_polynomial_potential", names)
        self.assertIn("exact_transcendental_full_eval_level2", names)
        self.assertIn("bernoulli_basic_n2", names)
        self.assertIn("homogeneous_explicit_h18", names)
        self.assertIn("homogeneous_implicit_h19", names)
        self.assertIn("residual_bernoulli_n3_branch_verification", names)
        self.assertIn("system_2x2_real_eigenvalues", names)
        self.assertIn("system_2x2_complex_real_solutions", names)
        self.assertIn("system_2x2_defective_generalized_vector", names)
        self.assertIn("residual_system_nonlinear", names)
        self.assertIn("residual_system_ivp_future_cycle", names)
        self.assertIn("cauchy_euler_distinct_roots", names)
        self.assertIn("integrating_factor_mu_x_boyce", names)
        self.assertIn("residual_mu_rational_potential_ceiling", names)
        self.assertIn("cauchy_euler_complex_trig_ln", names)
        self.assertIn("residual_riccati_never_fabricate", names)
        self.assertIn("residual_airy_variable_coefficients", names)
        self.assertIn("second_order_complex_envelope_o23", names)
        self.assertIn("second_order_ivp_v31", names)
        self.assertIn("residual_third_order", names)
        self.assertIn("uc_trig_resonance_shift", names)
        self.assertIn("uc_ivp_nonhomogeneous", names)
        self.assertIn("residual_uc_out_of_table", names)
        self.assertIn("ivp_separable_pinned_constant", names)
        self.assertIn("ivp_implicit_circle", names)
        self.assertIn("residual_ivp_derivative_condition_order_mismatch", names)
        self.assertIn("residual_ivp_inconsistent_condition", names)
        self.assertIn("residual_pendulum_never_fabricate", names)

        supported = [case for case in cases if case.outcome == "supported"]
        residual = [case for case in cases if case.outcome == "residual"]
        self.assertEqual(len(supported), 40)
        self.assertEqual(len(residual), 11)

        # Verification-gated emission: every supported row is verified; every
        # residual row is declined (never fabricated).
        for case in supported:
            self.assertIn(
                case.verification_regime,
                (
                    "verified_by_substitution",
                    "verified_by_implicit_differentiation",
                    "verified_per_component",
                    "verified_per_basis",
                    "verified_affine",
                ),
                case.name,
            )
        for case in residual:
            self.assertEqual(case.verification_regime, "declined", case.name)
            self.assertEqual(case.presentation_regime, "echo", case.name)
            self.assertTrue(case.expected_result.startswith("dsolve("), case.name)
            self.assertNotEqual(case.residual_cause, "not_applicable", case.name)

        # Axis coverage minimums for O0+O1+O2.
        families = {case.family for case in cases}
        self.assertEqual(
            families,
            {
                "separable",
                "lineal_1o",
                "exacta",
                "coef_const_2o",
                "UC",
                "bernoulli",
                "homogenea",
                "sistema",
                "cauchy_euler",
                "factor_integrante",
            },
        )
        self.assertEqual(
            {case.order_regime for case in cases},
            {"first", "second", "third_plus"},
        )
        self.assertIn("implicit", {case.presentation_regime for case in cases})
        self.assertIn("sugar_arity2", {case.presentation_regime for case in cases})
        self.assertIn(
            "condition_order_mismatch",
            {case.residual_cause for case in cases},
        )
        self.assertIn("ivp_resolved", {case.constant_regime for case in cases})

    def test_case_filter_selects_named_cases(self) -> None:
        cases = SMOKE.build_cases(("separable_growth_textbook",))
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].name, "separable_growth_textbook")
        with self.assertRaises(SystemExit):
            SMOKE.build_cases(("nonexistent_case",))

    def test_summarize_matrix_drops_case_rows_and_keeps_axis_counts(self) -> None:
        matrix = {
            "total": 2,
            "status": "pass",
            "status_counts": {"pass": 2},
            "problem_cases": [],
            "family_counts": {"separable": 2},
            "outcome_counts": {"supported": 1, "residual": 1},
            "cases": [{"name": "a"}, {"name": "b"}],
        }
        summary = SMOKE.summarize_matrix(matrix)
        self.assertNotIn("cases", summary)
        self.assertEqual(summary["family_counts"], {"separable": 2})
        self.assertEqual(summary["outcome_counts"], {"supported": 1, "residual": 1})


if __name__ == "__main__":
    unittest.main()
