#!/usr/bin/env python3
"""Registry counters for the solve-system command matrix lane (frente S)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "engine_solve_system_command_matrix_smoke.py"
SPEC = importlib.util.spec_from_file_location(
    "engine_solve_system_command_matrix_smoke", MODULE_PATH
)
SMOKE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)

AXES = SMOKE.AXES
DEFAULT_SOLVE_SYSTEM_COMMAND_MATRIX_CASES = SMOKE.DEFAULT_SOLVE_SYSTEM_COMMAND_MATRIX_CASES


class SolveSystemCommandMatrixRegistryTest(unittest.TestCase):
    def test_registry_counters(self) -> None:
        cases = DEFAULT_SOLVE_SYSTEM_COMMAND_MATRIX_CASES
        # S1+S2 registry: 7 linear (2×2/3×3/4×4 + edges + surd-coef vía
        # composición S2) + 3 degenerate + 4 parametric supported + 5
        # nonlinear supported + 1 nonlinear proven-empty + 2 honest residual
        # rows. Las DOS filas residuales de S1 (no-lineal, coef surd)
        # GRADUARON en S2 — el contrato migró de decline a resultado.
        self.assertEqual(len(cases), 27)

        names = {case.name for case in cases}
        self.assertIn("parametric_2x2_symbolic_det_condition", names)
        self.assertIn("residual_parametric_degenerate_det_zero", names)
        self.assertIn("surd_coefficient_linear_via_substitution_composition", names)
        self.assertIn("degenerate_all_zero_rows_edge", names)
        self.assertIn("nonlinear_circle_line_textbook", names)
        self.assertIn("nonlinear_parabola_line_surd_pairs_verified", names)
        self.assertIn("nonlinear_two_conics_resultant_sylvester", names)
        self.assertIn("nonlinear_concentric_circles_proven_empty", names)

        supported = [c for c in cases if c.outcome == "supported"]
        residual = [c for c in cases if c.outcome == "honest_residual"]
        degenerate = [c for c in cases if c.outcome in {"dependent", "inconsistent"}]
        self.assertEqual(len(supported), 21)
        self.assertEqual(len(residual), 1)
        self.assertEqual(len(degenerate), 5)

        families = {c.family for c in cases}
        self.assertEqual(
            families, {"lineal", "degenerado", "parametrico", "no_lineal"}
        )

        conditioned = [c for c in cases if c.condition_regime == "det_nonzero"]
        self.assertEqual(len(conditioned), 2)
        for case in conditioned:
            self.assertIn("requires:", case.expected_result)

        parity = [c for c in cases if c.presentation_regime == "list_syntax_parity"]
        self.assertEqual(len(parity), 2)

        narrated = [c for c in cases if c.expected_solve_step_substrings]
        self.assertEqual(len(narrated), 5)

        verified_pairs = [c for c in cases if c.condition_regime == "verified_pairs"]
        self.assertEqual(len(verified_pairs), 10)
        for case in verified_pairs:
            self.assertIn("{", case.expected_result)

        for case in cases:
            for axis in AXES:
                self.assertTrue(getattr(case, axis))


if __name__ == "__main__":
    unittest.main()
