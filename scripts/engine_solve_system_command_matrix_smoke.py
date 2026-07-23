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
        name="residual_nonlinear_square_future_cycle",
        expr="solve([x^2+y=1, x-y=0], [x, y])",
        expected_result=(
            "Error in equation 1: non-linear term: degree > 1 in the system\n"
            "solve_system() only handles linear equations."
        ),
        family="no_lineal",
        outcome="honest_residual",
        presentation_regime="prose_outcome",
    ),
    SolveSystemCommandMatrixCase(
        name="residual_surd_coefficient_rational_ceiling",
        expr="solve([sqrt(2)*x+y=1, x-y=0], [x, y])",
        expected_result=(
            "Error solving system: polynomial conversion: "
            "expression is not a polynomial over Q"
        ),
        family="parametrico",
        coefficient_regime="surd_constant",
        outcome="honest_residual",
        presentation_regime="prose_outcome",
    ),
)


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
    command = [str(cas_cli), "eval", case.expr, "--format", "json"]
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
