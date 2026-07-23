#!/usr/bin/env python3
"""Command-level solve-system policy matrix smoke (frente S · S1).

`solve([eq1, eq2, ...], [v1, v2, ...])` and `solve_system(...)` share one
pipeline. Its contract after S1: the unknowns list drives linearity (every
other variable is a PARAMETER), symbolic 2×2 systems solve by exact Cramer
carrying `det ≠ 0` on the canonical required-conditions channel, and every
well-formed system the solver cannot handle declines as an HONEST ok-result
(never an internal error on the wire). No other scorecard lane exercises this
pipeline — each cycle that widens it MUST add its rows here in the same
commit.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Literal


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cas_cli_release import ensure_release_cas_cli
from engine_command_matrix_observability import stderr_fragility_error
from engine_smoke_common import parse_json, terminate_process_group


ROOT = SCRIPT_DIR.parent
DEFAULT_CAS_CLI = ROOT / "target" / "release" / "cas_cli"
Status = Literal["pass", "slow", "fail", "timeout"]

AXES = (
    "family",
    "size_regime",
    "coefficient_regime",
    "outcome",
    "condition_regime",
    "presentation_regime",
)


@dataclass(frozen=True)
class SolveSystemCommandMatrixCase:
    name: str
    expr: str
    expected_result: str
    expected_required: tuple[str, ...] = ()
    expected_solve_step_substrings: tuple[str, ...] = ()
    family: str = "lineal"
    size_regime: str = "two_by_two"
    coefficient_regime: str = "rational"
    outcome: str = "supported"
    condition_regime: str = "none"
    presentation_regime: str = "solution_set"


DEFAULT_SOLVE_SYSTEM_COMMAND_MATRIX_CASES = (
    SolveSystemCommandMatrixCase(
        name="linear_2x2_integer",
        expr="solve([x+y=3, x-y=1], [x, y])",
        expected_result="{ x = 2, y = 1 }",
        expected_solve_step_substrings=(
            "Identificar sistema de 2 ecuaciones",
            "Cramer/Gauss",
        ),
    ),
    SolveSystemCommandMatrixCase(
        name="linear_2x2_fraction_solution",
        expr="solve([2*x+y=1, x+3*y=2], [x, y])",
        expected_result="{ x = 1/5, y = 3/5 }",
    ),
    SolveSystemCommandMatrixCase(
        name="linear_2x2_div_coefficients",
        expr="solve([x/2+y=1, x-y/3=2], [x, y])",
        expected_result="{ x = 2, y = 0 }",
    ),
    SolveSystemCommandMatrixCase(
        name="linear_2x2_swapped_var_order",
        expr="solve([x+y=3, x-y=1], [y, x])",
        expected_result="{ y = 1, x = 2 }",
        presentation_regime="var_order_follows_request",
    ),
    SolveSystemCommandMatrixCase(
        name="linear_3x3_unique",
        expr="solve([x+y+z=6, x-y=0, z=3], [x, y, z])",
        expected_result="{ x = 3/2, y = 3/2, z = 3 }",
        size_regime="three_by_three",
    ),
    SolveSystemCommandMatrixCase(
        name="linear_4x4_triangular_gauss",
        expr="solve([x+y+z+w=10, y+z+w=9, z+w=7, w=4], [x, y, z, w])",
        expected_result="{ x = 1, y = 2, z = 3, w = 4 }",
        size_regime="n_by_n",
    ),
    SolveSystemCommandMatrixCase(
        name="degenerate_dependent",
        expr="solve([x+y=1, 2*x+2*y=2], [x, y])",
        expected_result=(
            "System has infinitely many solutions.\nThe equations are dependent."
        ),
        family="degenerado",
        outcome="dependent",
        presentation_regime="prose_outcome",
    ),
    SolveSystemCommandMatrixCase(
        name="degenerate_inconsistent",
        expr="solve([x+y=1, x+y=2], [x, y])",
        expected_result="System has no solution.\nThe equations are inconsistent.",
        expected_solve_step_substrings=("inconsistentes",),
        family="degenerado",
        outcome="inconsistent",
        presentation_regime="prose_outcome",
    ),
    SolveSystemCommandMatrixCase(
        name="degenerate_all_zero_rows_edge",
        expr="solve([0*x+0*y=0, 0*x+0*y=0], [x, y])",
        expected_result=(
            "System has infinitely many solutions.\nThe equations are dependent."
        ),
        family="degenerado",
        coefficient_regime="zero_edge",
        outcome="dependent",
        presentation_regime="prose_outcome",
    ),
    SolveSystemCommandMatrixCase(
        name="parametric_2x2_symbolic_det_condition",
        expr="solve([a*x+y=1, x-y=0], [x, y])",
        expected_result=(
            "{ x = 1 / (a + 1), y = 1 / (a + 1) }\n  requires: a + 1 != 0"
        ),
        expected_required=("a ≠ -1",),
        expected_solve_step_substrings=(
            "Coeficientes simbólicos",
            "determinante debe ser distinto de cero",
        ),
        family="parametrico",
        coefficient_regime="symbolic_parameter",
        condition_regime="det_nonzero",
    ),
    SolveSystemCommandMatrixCase(
        name="parametric_2x2_two_parameters_compound",
        expr="solve([a*x+b*y=1, x-y=0], [x, y])",
        expected_result=(
            "{ x = 1 / (a + b), y = 1 / (a + b) }\n  requires: a + b != 0"
        ),
        family="parametrico",
        coefficient_regime="symbolic_parameter",
        condition_regime="det_nonzero",
    ),
    SolveSystemCommandMatrixCase(
        name="parametric_2x2_symbolic_rhs_exact",
        expr="solve([x+y=u+v, x-y=u-v], [x, y])",
        expected_result="{ x = u, y = v }",
        family="parametrico",
        coefficient_regime="symbolic_parameter",
        presentation_regime="polynomial_quotient_folds",
    ),
    SolveSystemCommandMatrixCase(
        name="parametric_2x2_constant_det_symbolic_numerators",
        expr="solve([2*x+y=a, x-y=b], [x, y])",
        expected_result="{ x = 1/3·a + 1/3·b, y = 1/3·a - 2/3·b }",
        family="parametrico",
        coefficient_regime="symbolic_parameter",
    ),
    SolveSystemCommandMatrixCase(
        name="nonlinear_parabola_line_rational_pairs",
        expr="solve([y=x^2, y=x+2], [x, y])",
        expected_result="{ x = -1, y = 1 } or { x = 2, y = 4 }",
        family="no_lineal",
        coefficient_regime="rational",
        condition_regime="verified_pairs",
        presentation_regime="solution_pairs",
    ),
    SolveSystemCommandMatrixCase(
        name="nonlinear_hyperbola_line_product",
        expr="solve([x*y=6, x+y=5], [x, y])",
        expected_result="{ x = 2, y = 3 } or { x = 3, y = 2 }",
        expected_solve_step_substrings=(
            "Aislar y de la ecuación 2",
            "resolver la univariable en x",
            "2 pares verificados",
        ),
        family="no_lineal",
        coefficient_regime="rational",
        condition_regime="verified_pairs",
        presentation_regime="solution_pairs",
    ),
    SolveSystemCommandMatrixCase(
        name="nonlinear_circle_line_textbook",
        expr="solve([x^2+y^2=25, x+y=7], [x, y])",
        expected_result="{ x = 3, y = 4 } or { x = 4, y = 3 }",
        family="no_lineal",
        coefficient_regime="rational",
        condition_regime="verified_pairs",
        presentation_regime="solution_pairs",
    ),
    SolveSystemCommandMatrixCase(
        name="nonlinear_parabola_line_surd_pairs_verified",
        expr="solve([x^2+y=3, x-y=1], [x, y])",
        expected_result=(
            "{ x = 1/2·(-sqrt(17) - 1), y = 1/2·(-sqrt(17) - 3) } or "
            "{ x = 1/2·(sqrt(17) - 1), y = 1/2·(sqrt(17) - 3) }"
        ),
        family="no_lineal",
        coefficient_regime="surd_solutions",
        condition_regime="verified_pairs",
        presentation_regime="solution_pairs",
    ),
    SolveSystemCommandMatrixCase(
        name="nonlinear_circle_line_no_real_intersection",
        expr="solve([x^2+y^2=1, x+y=5], [x, y])",
        expected_result="System has no solution.\nThe equations are inconsistent.",
        family="no_lineal",
        coefficient_regime="rational",
        outcome="inconsistent",
        presentation_regime="prose_outcome",
    ),
    SolveSystemCommandMatrixCase(
        name="nonlinear_two_conics_resultant_sylvester",
        expr="solve([x*y=6, x^2+y^2=13], [x, y])",
        expected_result=(
            "{ x = -3, y = -2 } or { x = -2, y = -3 } or "
            "{ x = 2, y = 3 } or { x = 3, y = 2 }"
        ),
        expected_solve_step_substrings=(
            "resultante de Sylvester",
            "Resolver la resultante univariable en x",
            "4 pares verificados",
        ),
        family="no_lineal",
        coefficient_regime="rational",
        condition_regime="verified_pairs",
        presentation_regime="solution_pairs",
    ),
    SolveSystemCommandMatrixCase(
        name="nonlinear_ellipse_hyperbola_no_linear_unknown",
        expr="solve([x^2+4*y^2=25, x^2-y^2=5], [x, y])",
        expected_result=(
            "{ x = -3, y = -2 } or { x = -3, y = 2 } or "
            "{ x = 3, y = -2 } or { x = 3, y = 2 }"
        ),
        family="no_lineal",
        coefficient_regime="rational",
        condition_regime="verified_pairs",
        presentation_regime="solution_pairs",
    ),
    SolveSystemCommandMatrixCase(
        name="nonlinear_circle_circle_intersection",
        expr="solve([x^2+y^2=25, (x-1)^2+y^2=18], [x, y])",
        expected_result="{ x = 4, y = -3 } or { x = 4, y = 3 }",
        family="no_lineal",
        coefficient_regime="rational",
        condition_regime="verified_pairs",
        presentation_regime="solution_pairs",
    ),
    SolveSystemCommandMatrixCase(
        name="nonlinear_concentric_circles_proven_empty",
        expr="solve([x^2+y^2=1, x^2+y^2=4], [x, y])",
        expected_result="System has no solution.\nThe equations are inconsistent.",
        family="no_lineal",
        coefficient_regime="rational",
        outcome="inconsistent",
        presentation_regime="prose_outcome",
    ),
    SolveSystemCommandMatrixCase(
        name="solve_system_list_form_parity",
        expr="solve_system([x+y=3, x-y=1], [x, y])",
        expected_result="{ x = 2, y = 1 }",
        presentation_regime="list_syntax_parity",
    ),
    SolveSystemCommandMatrixCase(
        name="solve_system_list_form_nonlinear_parity",
        expr="solve_system([x*y=6, x+y=5], [x, y])",
        expected_result="{ x = 2, y = 3 } or { x = 3, y = 2 }",
        family="no_lineal",
        condition_regime="verified_pairs",
        presentation_regime="list_syntax_parity",
    ),
    SolveSystemCommandMatrixCase(
        name="declared_constant_name_e_is_unknown",
        expr="solve([a+b+c+d+e=15, b+c+d+e=14, c+d+e=12, d+e=9, e=5], [a, b, c, d, e])",
        expected_result="{ a = 1, b = 2, c = 3, d = 4, e = 5 }",
        size_regime="n_by_n",
        presentation_regime="constant_name_shadowed",
    ),
    SolveSystemCommandMatrixCase(
        name="undeclared_e_stays_euler_coefficient",
        expr="solve([x+e*y=1, x-y=0], [x, y])",
        expected_result="{ x = 1 / (1 + e), y = 1 / (1 + e) }",
        family="no_lineal",
        coefficient_regime="transcendental_constant",
        condition_regime="verified_pairs",
        presentation_regime="solution_pairs",
    ),
    SolveSystemCommandMatrixCase(
        name="parametric_3x3_symbolic_det_condition",
        expr="solve([a*x+y+z=1, x-y=0, y-z=0], [x, y, z])",
        expected_result=(
            "{ x = 1 / (a + 2), y = 1 / (a + 2), z = 1 / (a + 2) }\n"
            "  requires: a + 2 != 0"
        ),
        expected_required=("a ≠ -2",),
        family="parametrico",
        size_regime="three_by_three",
        coefficient_regime="symbolic_parameter",
        condition_regime="det_nonzero",
    ),
    SolveSystemCommandMatrixCase(
        name="parametric_3x3_two_parameters_dense",
        expr="solve([a*x+b*y+z=1, x+y+z=0, x-y=2], [x, y, z])",
        expected_result=(
            "{ x = (2·b - 1) / (a + b - 2), y = (3 - 2·a) / (a + b - 2), "
            "z = (2·a - 2·b - 2) / (a + b - 2) }\n  requires: a + b - 2 != 0"
        ),
        family="parametrico",
        size_regime="three_by_three",
        coefficient_regime="symbolic_parameter",
        condition_regime="det_nonzero",
    ),
    SolveSystemCommandMatrixCase(
        name="parametric_3x3_symbolic_rhs_constant_det",
        expr="solve([x+y+z=p, x-y=0, y-z=0], [x, y, z])",
        expected_result="{ x = 1/3·p, y = 1/3·p, z = 1/3·p }",
        family="parametrico",
        size_regime="three_by_three",
        coefficient_regime="symbolic_parameter",
    ),
    SolveSystemCommandMatrixCase(
        name="residual_parametric_3x3_degenerate_det_zero",
        expr="solve([a*x+y+z=1, a*x+y+z=2, x-y=0], [x, y, z])",
        expected_result=(
            "Error: non-linear term: symbolic coefficients with det = 0: "
            "rank classification is a future rung\n"
            "solve_system() only handles linear equations."
        ),
        family="parametrico",
        size_regime="three_by_three",
        coefficient_regime="symbolic_parameter",
        outcome="honest_residual",
        presentation_regime="prose_outcome",
    ),
    SolveSystemCommandMatrixCase(
        name="parametric_4x4_symbolic_det_condition",
        expr="solve([a*x+y+z+w=1, x-y=0, y-z=0, z-w=0], [x, y, z, w])",
        expected_result=(
            "{ x = 1 / (a + 3), y = 1 / (a + 3), z = 1 / (a + 3), "
            "w = 1 / (a + 3) }\n  requires: a + 3 != 0"
        ),
        family="parametrico",
        size_regime="n_by_n",
        coefficient_regime="symbolic_parameter",
        condition_regime="det_nonzero",
    ),
    SolveSystemCommandMatrixCase(
        name="parametric_4x4_symbolic_rhs_constant_det",
        expr="solve([x+y+z+w=p, x-y=0, y-z=0, z-w=0], [x, y, z, w])",
        expected_result="{ x = 1/4·p, y = 1/4·p, z = 1/4·p, w = 1/4·p }",
        family="parametrico",
        size_regime="n_by_n",
        coefficient_regime="symbolic_parameter",
    ),
    SolveSystemCommandMatrixCase(
        name="residual_parametric_degenerate_det_zero",
        expr="solve([a*x+y=1, a*x+y=2], [x, y])",
        expected_result=(
            "Error: non-linear term: symbolic coefficients with det = 0: "
            "rank classification is a future rung\n"
            "solve_system() only handles linear equations."
        ),
        family="parametrico",
        coefficient_regime="symbolic_parameter",
        outcome="honest_residual",
        presentation_regime="prose_outcome",
    ),
    SolveSystemCommandMatrixCase(
        name="nonlinear_parabola_identity_line_surd_pairs",
        expr="solve([x^2+y=1, x-y=0], [x, y])",
        expected_result=(
            "{ x = 1/2·(-sqrt(5) - 1), y = 1/2·(-sqrt(5) - 1) } or "
            "{ x = 1/2·(sqrt(5) - 1), y = 1/2·(sqrt(5) - 1) }"
        ),
        family="no_lineal",
        coefficient_regime="surd_solutions",
        condition_regime="verified_pairs",
        presentation_regime="solution_pairs",
    ),
    SolveSystemCommandMatrixCase(
        name="surd_coefficient_linear_via_substitution_composition",
        expr="solve([sqrt(2)*x+y=1, x-y=0], [x, y])",
        expected_result="{ x = sqrt(2) - 1, y = sqrt(2) - 1 }",
        family="lineal",
        coefficient_regime="surd_constant",
        condition_regime="verified_pairs",
        presentation_regime="solution_pairs",
    ),
)


def extract_solve_step_text(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        return ""
    chunks = []
    for step in parsed.get("solve_steps") or []:
        if isinstance(step, dict):
            chunks.append(str(step.get("description", "")))
    return "\n".join(chunks)


def extract_required_display(parsed: Any) -> tuple[str, ...]:
    if not isinstance(parsed, dict):
        return ()
    required = parsed.get("required_display")
    if not isinstance(required, list):
        return ()
    return tuple(str(item) for item in required)


def run_case(
    case: SolveSystemCommandMatrixCase,
    *,
    cas_cli: str | pathlib.Path,
    timeout_seconds: float,
    slow_wall_seconds: float | None = None,
) -> dict[str, Any]:
    command = [str(cas_cli), "eval", case.expr, "--format", "json", "--steps", "on"]
    start = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        terminate_process_group(process)
        stdout, stderr = process.communicate()
        return {
            "name": case.name,
            "status": "timeout",
            "error": "timeout",
            "returncode": None,
            "wall_elapsed_seconds": round(time.monotonic() - start, 3),
            "stdout": stdout,
            "stderr": stderr,
        }

    wall_elapsed = time.monotonic() - start
    parsed, parse_error = parse_json(stdout)
    result = parsed.get("result") if isinstance(parsed, dict) else None
    required = extract_required_display(parsed)
    ok = parsed.get("ok") if isinstance(parsed, dict) else None

    error: str | None = None
    if process.returncode != 0:
        error = f"returncode={process.returncode}"
    elif parse_error:
        error = parse_error
    elif ok is not True:
        # The S1 wire-honesty contract: well-formed systems NEVER surface as
        # internal errors — declines are ok-results with the reason as text.
        error = "ok was not true (well-formed system must never be a wire error)"
    elif result != case.expected_result:
        error = f"expected result {case.expected_result!r}, got {result!r}"
    elif (stderr_error := stderr_fragility_error(stderr)) is not None:
        error = stderr_error
    else:
        for expected in case.expected_required:
            if expected not in required:
                error = f"missing expected required condition {expected!r}"
                break
        if error is None:
            step_text = extract_solve_step_text(parsed)
            for expected in case.expected_solve_step_substrings:
                if expected not in step_text:
                    error = f"missing expected solve step containing {expected!r}"
                    break

    status: Status = "pass" if error is None else "fail"
    if status == "pass" and slow_wall_seconds is not None and wall_elapsed > slow_wall_seconds:
        status = "slow"
        error = f"slow: {wall_elapsed:.3f}s > {slow_wall_seconds:.3f}s"

    row: dict[str, Any] = {
        "name": case.name,
        "status": status,
        "error": error,
        "returncode": process.returncode,
        "wall_elapsed_seconds": round(wall_elapsed, 3),
        "result": result if isinstance(result, str) else None,
        "required_display": list(required),
        "expected_result": case.expected_result,
    }
    for axis in AXES:
        row[axis] = getattr(case, axis)
    return row


def axis_counts(rows: list[dict[str, Any]], axis: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(axis))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cas-cli", default=str(DEFAULT_CAS_CLI))
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--slow-wall-seconds", type=float, default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument(
        "--ensure-release-cas-cli", action="store_true", default=True
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cas_cli = pathlib.Path(args.cas_cli)
    if args.ensure_release_cas_cli:
        ensure_release_cas_cli(cas_cli)

    rows = [
        run_case(
            case,
            cas_cli=cas_cli,
            timeout_seconds=args.timeout_seconds,
            slow_wall_seconds=args.slow_wall_seconds,
        )
        for case in DEFAULT_SOLVE_SYSTEM_COMMAND_MATRIX_CASES
    ]

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    status = "pass" if set(counts) <= {"pass"} else "fail"

    payload: dict[str, Any] = {
        "status": status,
        "total_cases": len(rows),
        "counts": counts,
        "cases": rows,
    }
    for axis in AXES:
        payload[f"solve_system_{axis}_counts"] = axis_counts(rows, axis)

    if args.json_output:
        pathlib.Path(args.json_output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for row in rows:
            marker = "PASS" if row["status"] == "pass" else row["status"].upper()
            print(f"[{marker}] {row['name']}: {row.get('error') or 'ok'}")
        print(f"total={len(rows)} status={status} counts={counts}")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
