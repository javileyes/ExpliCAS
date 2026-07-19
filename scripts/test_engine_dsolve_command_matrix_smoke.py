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

        # O0+O1 registry: 7 separable + 4 linear supported rows + 4 honest
        # residual rows.
        self.assertEqual(len(cases), 15)
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
        self.assertIn("residual_riccati_never_fabricate", names)
        self.assertIn("residual_airy_higher_order", names)
        self.assertIn("residual_ivp_conditions_future_cycle", names)
        self.assertIn("residual_pendulum_never_fabricate", names)

        supported = [case for case in cases if case.outcome == "supported"]
        residual = [case for case in cases if case.outcome == "residual"]
        self.assertEqual(len(supported), 11)
        self.assertEqual(len(residual), 4)

        # Verification-gated emission: every supported row is verified; every
        # residual row is declined (never fabricated).
        for case in supported:
            self.assertIn(
                case.verification_regime,
                ("verified_by_substitution", "verified_by_implicit_differentiation"),
                case.name,
            )
        for case in residual:
            self.assertEqual(case.verification_regime, "declined", case.name)
            self.assertEqual(case.presentation_regime, "echo", case.name)
            self.assertTrue(case.expected_result.startswith("dsolve("), case.name)
            self.assertNotEqual(case.residual_cause, "not_applicable", case.name)

        # Axis coverage minimums for O0+O1.
        families = {case.family for case in cases}
        self.assertEqual(families, {"separable", "lineal_1o"})
        self.assertEqual(
            {case.order_regime for case in cases}, {"first", "second"}
        )
        self.assertIn("implicit", {case.presentation_regime for case in cases})
        self.assertIn("sugar_arity2", {case.presentation_regime for case in cases})
        self.assertIn(
            "conditions_future_cycle", {case.residual_cause for case in cases}
        )

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
