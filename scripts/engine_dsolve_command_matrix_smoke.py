#!/usr/bin/env python3
"""Command-level dsolve policy matrix smoke (Fase 4 · D15).

`dsolve(...)` is a special-command surface (wire + REPL parity) with
verification-gated emission: every supported family must emit ONLY after the
candidate reduced the ODE residue to an exact symbolic 0, and every
out-of-scope form must decline to an honest `dsolve(...)` residual echo.
No existing scorecard lane exercises the solve/special-command pipeline, so
this lane is dsolve's only safety net besides the CLI contract tests — each
cycle that graduates a family MUST add its rows here in the same commit.
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
from engine_command_matrix_observability import (
    extract_warning_messages,
    stderr_fragility_error,
)
from engine_smoke_common import parse_json, terminate_process_group


ROOT = SCRIPT_DIR.parent
DEFAULT_CAS_CLI = ROOT / "target" / "release" / "cas_cli"
Status = Literal["pass", "slow", "fail", "timeout"]

AXES = (
    "family",
    "order_regime",
    "verification_regime",
    "constant_regime",
    "outcome",
    "residual_cause",
    "trace_regime",
    "presentation_regime",
)


@dataclass(frozen=True)
class DsolveCommandMatrixCase:
    name: str
    expr: str
    expected_result: str
    expected_warning_substrings: tuple[str, ...] = ()
    expected_solve_step_substrings: tuple[str, ...] = ()
    family: str = "separable"
    order_regime: str = "first"
    verification_regime: str = "verified_by_substitution"
    constant_regime: str = "general"
    outcome: str = "supported"
    residual_cause: str = "not_applicable"
    trace_regime: str = "separable_narrated"
    presentation_regime: str = "explicit"


DEFAULT_DSOLVE_COMMAND_MATRIX_CASES = (
    DsolveCommandMatrixCase(
        name="separable_growth_textbook",
        expr="dsolve(diff(y,x)=x*y, y, x)",
        expected_result="y = C·e^(x^2 / 2)",
        expected_warning_substrings=("constante arbitraria", "solución singular"),
        expected_solve_step_substrings=(
            "Identificar EDO separable",
            "Integrar ambos lados",
            "Verificar por sustitución",
        ),
    ),
    DsolveCommandMatrixCase(
        name="separable_reciprocal_quadratic",
        expr="dsolve(diff(y,x)=y^2, y, x)",
        expected_result="y = -1 / (C + x)",
        expected_warning_substrings=("constante arbitraria",),
        expected_solve_step_substrings=("Identificar EDO separable",),
    ),
    DsolveCommandMatrixCase(
        name="separable_implicit_circle",
        expr="dsolve(diff(y,x)=-x/y, y, x)",
        expected_result="x^2 + y^2 = C",
        expected_warning_substrings=("Solución implícita",),
        expected_solve_step_substrings=("solución implícita",),
        verification_regime="verified_by_implicit_differentiation",
        presentation_regime="implicit",
    ),
    DsolveCommandMatrixCase(
        name="separable_proportional",
        expr="dsolve(diff(y,x)=y/x, y, x)",
        expected_result="y = C·x",
        expected_warning_substrings=("se absorbe en C",),
    ),
    DsolveCommandMatrixCase(
        name="separable_parametric_rate",
        expr="dsolve(diff(y,x)=k*y, y, x)",
        expected_result="y = C·e^(k·x)",
        expected_warning_substrings=("constante arbitraria",),
        presentation_regime="parametric_rhs",
    ),
    DsolveCommandMatrixCase(
        name="separable_direct_integration",
        expr="dsolve(diff(y,x)=cos(x), y, x)",
        expected_result="y = sin(x) + C",
        expected_warning_substrings=("constante arbitraria",),
        trace_regime="direct_integration",
    ),
    DsolveCommandMatrixCase(
        name="separable_arity2_sugar",
        expr="dsolve(diff(y,x)=x*y, y)",
        expected_result="y = C·e^(x^2 / 2)",
        expected_warning_substrings=("constante arbitraria",),
        presentation_regime="sugar_arity2",
    ),
    DsolveCommandMatrixCase(
        name="residual_riccati_never_fabricate",
        expr="dsolve(diff(y,x)=x^2+y^2, y, x)",
        expected_result="dsolve(diff(y, x) = x^2 + y^2, y, x)",
        expected_warning_substrings=("no es separable",),
        verification_regime="declined",
        outcome="residual",
        residual_cause="not_separable",
        trace_regime="none",
        presentation_regime="echo",
    ),
    DsolveCommandMatrixCase(
        name="residual_airy_higher_order",
        expr="dsolve(diff(y,x,2)+x*y=0, y, x)",
        expected_result="dsolve(diff(y, x, 2) + x·y = 0, y, x)",
        expected_warning_substrings=("orden superior",),
        order_regime="second",
        verification_regime="declined",
        outcome="residual",
        residual_cause="higher_order_future_cycle",
        trace_regime="none",
        presentation_regime="echo",
    ),
    DsolveCommandMatrixCase(
        name="residual_ivp_conditions_future_cycle",
        expr="dsolve(diff(y,x)=-y, y, x, y(0)=3)",
        expected_result="dsolve(diff(y, x) = -y, y, x)",
        expected_warning_substrings=("Condiciones iniciales",),
        verification_regime="declined",
        constant_regime="ivp_declined",
        outcome="residual",
        residual_cause="conditions_future_cycle",
        trace_regime="none",
        presentation_regime="echo",
    ),
    DsolveCommandMatrixCase(
        name="residual_pendulum_never_fabricate",
        expr="dsolve(diff(y,x,2)+sin(y)=0, y, x)",
        expected_result="dsolve(sin(y) + diff(y, x, 2) = 0, y, x)",
        expected_warning_substrings=("orden superior",),
        order_regime="second",
        verification_regime="declined",
        outcome="residual",
        residual_cause="higher_order_future_cycle",
        trace_regime="none",
        presentation_regime="echo",
    ),
)


def extract_solve_step_text(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        return ""
    chunks: list[str] = []
    for step in parsed.get("solve_steps") or []:
        if isinstance(step, dict):
            desc = step.get("description")
            if isinstance(desc, str):
                chunks.append(desc)
            for sub in step.get("substeps") or []:
                if isinstance(sub, dict):
                    sub_desc = sub.get("description")
                    if isinstance(sub_desc, str):
                        chunks.append(sub_desc)
    return "\n".join(chunks)


def run_case(
    case: DsolveCommandMatrixCase,
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
    warnings = extract_warning_messages(parsed)
    solve_step_text = extract_solve_step_text(parsed)
    ok = parsed.get("ok") if isinstance(parsed, dict) else None

    error: str | None = None
    if process.returncode != 0:
        error = f"returncode={process.returncode}"
    elif parse_error:
        error = parse_error
    elif ok is not True:
        error = "ok was not true"
    elif result != case.expected_result:
        error = f"expected result {case.expected_result!r}, got {result!r}"
    elif (stderr_error := stderr_fragility_error(stderr)) is not None:
        error = stderr_error
    else:
        warning_text = "\n".join(warnings)
        for expected in case.expected_warning_substrings:
            if expected not in warning_text:
                error = f"missing expected warning containing {expected!r}"
                break
        if error is None:
            for expected in case.expected_solve_step_substrings:
                if expected not in solve_step_text:
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
        "warnings": list(warnings),
        "expected_result": case.expected_result,
        "expected_warning_substrings": list(case.expected_warning_substrings),
        "expected_solve_step_substrings": list(case.expected_solve_step_substrings),
    }
    for axis in AXES:
        row[axis] = getattr(case, axis)
    return row


def build_cases(case_filters: tuple[str, ...] = ()) -> tuple[DsolveCommandMatrixCase, ...]:
    if not case_filters:
        return DEFAULT_DSOLVE_COMMAND_MATRIX_CASES
    requested = set(case_filters)
    selected = tuple(
        case for case in DEFAULT_DSOLVE_COMMAND_MATRIX_CASES if case.name in requested
    )
    missing = requested - {case.name for case in selected}
    if missing:
        raise SystemExit(f"unknown case name(s): {', '.join(sorted(missing))}")
    return selected


def run_matrix(
    cases: tuple[DsolveCommandMatrixCase, ...],
    *,
    cas_cli: str | pathlib.Path,
    timeout_seconds: float,
    slow_wall_seconds: float | None = None,
) -> dict[str, Any]:
    rows = [
        run_case(
            case,
            cas_cli=cas_cli,
            timeout_seconds=timeout_seconds,
            slow_wall_seconds=slow_wall_seconds,
        )
        for case in cases
    ]
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    axis_counts: dict[str, dict[str, int]] = {}
    for axis in AXES:
        counts: dict[str, int] = {}
        for case in cases:
            value = getattr(case, axis)
            counts[value] = counts.get(value, 0) + 1
        axis_counts[f"{axis}_counts"] = counts
    problem_cases = [
        {"name": row["name"], "status": row["status"], "error": row["error"]}
        for row in rows
        if row["status"] not in ("pass", "slow")
    ]
    matrix: dict[str, Any] = {
        "total": len(rows),
        "status": "pass" if not problem_cases else "fail",
        "status_counts": status_counts,
        "problem_cases": problem_cases,
        "cases": rows,
    }
    matrix.update(axis_counts)
    return matrix


def summarize_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    summary = {key: value for key, value in matrix.items() if key != "cases"}
    return summary


def print_human(matrix: dict[str, Any]) -> None:
    for row in matrix["cases"]:
        marker = "✓" if row["status"] == "pass" else "✗"
        print(f"{marker} {row['name']} [{row['status']}] {row.get('error') or ''}")
    print(
        f"total={matrix['total']} status={matrix['status']} "
        f"counts={matrix['status_counts']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cas-cli", default=str(DEFAULT_CAS_CLI))
    parser.add_argument("--ensure-release-cas-cli", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--slow-wall-seconds", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--summary-json", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only the named case(s); repeatable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cas_cli = pathlib.Path(args.cas_cli)
    if args.ensure_release_cas_cli:
        ensure_release_cas_cli(cas_cli)
    cases = build_cases(tuple(args.case))
    matrix = run_matrix(
        cases,
        cas_cli=cas_cli,
        timeout_seconds=args.timeout_seconds,
        slow_wall_seconds=args.slow_wall_seconds,
    )
    payload = summarize_matrix(matrix) if args.summary_json else matrix
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print_human(matrix)
    return 0 if matrix["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
