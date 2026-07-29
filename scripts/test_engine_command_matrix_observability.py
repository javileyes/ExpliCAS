"""Tests del predicado de fragilidad compartido por las lanes de command-matrix."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


# Mismo molde que las demás pruebas de `scripts/`: no hay paquete, así que el
# módulo se carga por ruta y el test corre igual desde la raíz del repo.
ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "engine_command_matrix_observability",
    ROOT / "scripts" / "engine_command_matrix_observability.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
stderr_fragility_error = MODULE.stderr_fragility_error


PHASE_TIMEOUT = (
    "2026-07-29T02:42:11Z  WARN simplify: phase_timeout_after_pass "
    "phase=Core iters=1 rewrites=26"
)
PHASE_TIMEOUT_BEFORE = (
    "2026-07-29T02:42:11Z  WARN simplify: phase_timeout_before_iteration phase=Core iters=3"
)
CYCLE_DETECTED = "2026-07-29T02:42:11Z  WARN simplify: cycle_detected phase=Core iters=7"
DEPTH_OVERFLOW = (
    "2026-07-29T02:42:11Z  WARN simplify: depth_overflow - returning expression unsimplified"
)


class StderrFragilityTests(unittest.TestCase):
    def test_phase_budget_warnings_are_not_fragility(self) -> None:
        # Dependen del RELOJ: un caso pegado al presupuesto los emite o no según
        # la carga. Medido en un mismo HEAD, un caso de integrate fallaba 3 de 8
        # corridas sin que nada cambiara en el motor.
        self.assertIsNone(stderr_fragility_error(PHASE_TIMEOUT))
        self.assertIsNone(stderr_fragility_error(PHASE_TIMEOUT_BEFORE))
        self.assertIsNone(
            stderr_fragility_error(f"{PHASE_TIMEOUT}\n{PHASE_TIMEOUT_BEFORE}")
        )

    def test_deterministic_warnings_stay_fragility(self) -> None:
        # Dependen de la EXPRESIÓN, no del reloj: siguen tumbando la corrida.
        for stderr in (CYCLE_DETECTED, DEPTH_OVERFLOW):
            self.assertIsNotNone(stderr_fragility_error(stderr), stderr)

    def test_a_budget_warning_does_not_mask_a_real_one(self) -> None:
        self.assertIsNotNone(
            stderr_fragility_error(f"{PHASE_TIMEOUT}\n{CYCLE_DETECTED}")
        )

    def test_hard_markers_and_empty_stderr(self) -> None:
        self.assertIsNone(stderr_fragility_error(""))
        for stderr in (
            "thread 'main' panicked at foo",
            "fatal runtime error: x",
            "SIGSEGV",
            "stack overflow",
        ):
            self.assertIsNotNone(stderr_fragility_error(stderr), stderr)

    def test_extra_forbidden_substrings_still_apply(self) -> None:
        self.assertIsNotNone(
            stderr_fragility_error("algo raro", forbidden_substrings=("raro",))
        )


if __name__ == "__main__":
    unittest.main()
