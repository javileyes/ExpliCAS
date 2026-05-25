#!/usr/bin/env python3
"""Command-level differentiation policy matrix smoke.

This lane complements the broad Rust diff contract. It keeps a small public
`diff(...)` support matrix visible at scorecard level so future calculus work is
selected by family, domain, trace, and presentation regime instead of by raw
test counts alone.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Literal


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cas_cli_release import ensure_release_cas_cli


ROOT = SCRIPT_DIR.parent
DEFAULT_CAS_CLI = ROOT / "target" / "release" / "cas_cli"
Status = Literal["pass", "slow", "fail", "timeout"]


@dataclass(frozen=True)
class DiffCommandMatrixCase:
    name: str
    expr: str
    expected_result: str
    expected_required_display: tuple[str, ...] = ()
    expected_warning_substrings: tuple[str, ...] = ()
    expected_blocked_hint_substrings: tuple[str, ...] = ()
    expected_step_substrings: tuple[str, ...] = ()
    family: str = "unknown"
    argument_regime: str = "variable"
    domain_regime: str = "unconditional"
    outcome: str = "supported"
    trace_regime: str = "direct"
    presentation_regime: str = "canonical"


DEFAULT_DIFF_COMMAND_MATRIX_CASES = (
    DiffCommandMatrixCase(
        name="polynomial_power_direct",
        expr="diff(x^3, x)",
        expected_result="3·x^2",
        expected_step_substrings=("Usar regla de la potencia",),
        family="polynomial",
        argument_regime="variable",
        trace_regime="power_rule",
    ),
    DiffCommandMatrixCase(
        name="product_log_required_condition",
        expr="diff(x*ln(x), x)",
        expected_result="ln(x) + 1",
        expected_required_display=("x > 0",),
        expected_step_substrings=(
            "Usar regla del producto",
            "Derivar el primer factor",
            "Derivar el segundo factor",
        ),
        family="product_log",
        argument_regime="product",
        domain_regime="required_condition",
        trace_regime="product_rule_log",
    ),
    DiffCommandMatrixCase(
        name="log_quadratic_empty_positive_argument_domain_undefined",
        expr="diff(ln(-x^2-1), x)",
        expected_result="undefined",
        expected_step_substrings=(
            "Detectar dominio real vacío del logaritmo",
            "undefined",
        ),
        family="logarithmic",
        argument_regime="quadratic_argument",
        domain_regime="empty_positive_argument_domain",
        outcome="undefined",
        trace_regime="log_empty_domain_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="general_base_log_unit_base_domain_undefined",
        expr="diff(log(1,x), x)",
        expected_result="undefined",
        expected_step_substrings=(
            "Detectar base inválida del logaritmo",
            "undefined",
        ),
        family="logarithmic",
        argument_regime="variable_argument",
        domain_regime="invalid_log_base_domain",
        outcome="undefined",
        trace_regime="log_invalid_base_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="polynomial_inner_chain_power",
        expr="diff((x^2+1)^3, x)",
        expected_result="6·x·(x^2 + 1)^2",
        expected_step_substrings=(
            "Usar regla de la potencia con cadena",
            "Identificar u y du",
        ),
        family="polynomial_chain",
        argument_regime="polynomial_inner",
        trace_regime="chain_rule",
        presentation_regime="factored",
    ),
    DiffCommandMatrixCase(
        name="elementary_exp_affine_chain_trace",
        expr="diff(exp(2*x+1), x)",
        expected_result="2·e^(2·x + 1)",
        expected_step_substrings=(
            "Usar regla exponencial",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="elementary_exp",
        argument_regime="affine_argument",
        trace_regime="exponential_chain_rule",
    ),
    DiffCommandMatrixCase(
        name="elementary_trig_affine_chain_trace",
        expr="diff(sin(2*x+1), x)",
        expected_result="2·cos(2·x + 1)",
        expected_step_substrings=(
            "Usar regla de sin(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="elementary_trig_chain",
        argument_regime="affine_argument",
        trace_regime="trig_chain_rule",
    ),
    DiffCommandMatrixCase(
        name="elementary_trig_tan_affine_chain_required_condition",
        expr="diff(tan(2*x+1), x)",
        expected_result="2 / cos(2·x + 1)^2",
        expected_required_display=("cos(2·x + 1) ≠ 0",),
        expected_step_substrings=(
            "Usar regla de tan(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="elementary_trig_chain",
        argument_regime="affine_argument_with_trig_pole",
        domain_regime="required_condition",
        trace_regime="trig_chain_rule_with_pole",
        presentation_regime="reciprocal_trig_power",
    ),
    DiffCommandMatrixCase(
        name="log_affine_chain_required_domain",
        expr="diff(ln(2*x+1), x)",
        expected_result="2 / (2·x + 1)",
        expected_required_display=("x > -1/2",),
        expected_step_substrings=(
            "Usar regla de ln(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="log_affine_chain",
        argument_regime="affine_argument",
        domain_regime="required_condition",
        trace_regime="log_chain_rule",
        presentation_regime="compact_quotient",
    ),
    DiffCommandMatrixCase(
        name="log_affine_chain_negative_orientation_required_domain",
        expr="diff(ln(1-2*x), x)",
        expected_result="-2 / (1 - 2·x)",
        expected_required_display=("x < 1/2",),
        expected_step_substrings=(
            "Usar regla de ln(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="log_affine_chain",
        argument_regime="negative_affine_argument",
        domain_regime="required_condition",
        trace_regime="log_chain_rule",
        presentation_regime="signed_compact_quotient",
    ),
    DiffCommandMatrixCase(
        name="constant_base_exp_affine_chain_positive_base",
        expr="diff(a^(2*x+1), x)",
        expected_result="2·ln(a)·a^(2·x + 1)",
        expected_required_display=("a > 0",),
        expected_step_substrings=(
            "Usar regla exponencial",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="constant_base_exponential",
        argument_regime="affine_exponent",
        domain_regime="required_condition",
        trace_regime="constant_base_exponential_chain_rule",
    ),
    DiffCommandMatrixCase(
        name="constant_base_exp_affine_chain_negative_base_undefined",
        expr="diff((-2)^(2*x+1), x)",
        expected_result="undefined",
        expected_step_substrings=(
            "Detectar base negativa con exponente variable",
            "undefined",
        ),
        family="constant_base_exponential",
        argument_regime="affine_exponent",
        domain_regime="negative_base_undefined",
        outcome="undefined",
        trace_regime="undefined_domain_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="constant_base_exp_affine_chain_zero_base_positive_exponent_domain",
        expr="diff(0^(2*x+1), x)",
        expected_result="0",
        expected_required_display=("x > -1/2",),
        expected_step_substrings=(
            "Detectar base cero con exponente variable",
            "0",
        ),
        family="constant_base_exponential",
        argument_regime="affine_exponent",
        domain_regime="positive_exponent_required",
        trace_regime="zero_base_domain_policy",
        presentation_regime="constant_zero",
    ),
    DiffCommandMatrixCase(
        name="constant_base_exp_quadratic_zero_base_empty_domain_undefined",
        expr="diff(0^(-x^2-1), x)",
        expected_result="undefined",
        expected_step_substrings=(
            "Detectar dominio real vacío de base cero",
            "undefined",
        ),
        family="constant_base_exponential",
        argument_regime="quadratic_exponent",
        domain_regime="empty_positive_exponent_domain",
        outcome="undefined",
        trace_regime="zero_base_empty_domain_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="nonfinite_constant_derivative_undefined",
        expr="diff(infinity, x)",
        expected_result="undefined",
        expected_step_substrings=(
            "Detectar constante no finita en la derivada",
            "undefined",
        ),
        family="nonfinite",
        argument_regime="constant",
        domain_regime="nonfinite_undefined",
        outcome="undefined",
        trace_regime="nonfinite_domain_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="rational_quotient_required_condition",
        expr="diff(x/(x+1), x)",
        expected_result="1 / (x + 1)^2",
        expected_required_display=("x ≠ -1",),
        expected_step_substrings=(
            "Usar regla del cociente",
            "Derivar el numerador",
            "Derivar el denominador",
        ),
        family="rational",
        argument_regime="rational_expression",
        domain_regime="required_condition",
        trace_regime="quotient_rule",
        presentation_regime="compact_quotient",
    ),
    DiffCommandMatrixCase(
        name="sqrt_variable_open_domain",
        expr="diff(sqrt(x), x)",
        expected_result="1 / (2·sqrt(x))",
        expected_required_display=("x > 0",),
        expected_step_substrings=("Usar regla de la potencia",),
        family="root",
        argument_regime="variable",
        domain_regime="required_condition",
        trace_regime="chain_rule",
        presentation_regime="reciprocal_root",
    ),
    DiffCommandMatrixCase(
        name="sqrt_quadratic_empty_positive_argument_domain_undefined",
        expr="diff(sqrt(-x^2-1), x)",
        expected_result="undefined",
        expected_step_substrings=(
            "Detectar dominio real vacío de la raíz",
            "undefined",
        ),
        family="root",
        argument_regime="quadratic_argument",
        domain_regime="empty_positive_argument_domain",
        outcome="undefined",
        trace_regime="sqrt_empty_domain_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_root_compact_presentation",
        expr="diff(arctan(sqrt(x)), x)",
        expected_result="1 / (2·sqrt(x)·(x + 1))",
        expected_required_display=("x > 0",),
        expected_step_substrings=(
            "Usar regla de arctan(u)",
            "Identificar u y du",
        ),
        family="inverse_trig_root",
        argument_regime="nested_root",
        domain_regime="required_condition",
        trace_regime="chain_rule",
        presentation_regime="post_calculus_compact",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_root_constant_multiple_trace",
        expr="diff(2*arctan(sqrt(x)), x)",
        expected_result="1 / ((x + 1)·sqrt(x))",
        expected_required_display=("x > 0",),
        expected_step_substrings=(
            "Usar factor constante de la derivada",
            "Usar regla de arctan(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="inverse_trig_root",
        argument_regime="scaled_nested_root",
        domain_regime="required_condition",
        trace_regime="constant_multiple_chain_rule",
        presentation_regime="scaled_post_calculus_compact",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_root_interval_orientation",
        expr="diff(arccos(sqrt(x)), x)",
        expected_result="-1 / (2·sqrt(x)·sqrt(1 - x))",
        expected_required_display=("x > 0", "x < 1"),
        expected_step_substrings=(
            "Usar regla de arccos(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="inverse_trig_root",
        argument_regime="nested_bounded_root",
        domain_regime="interval_required",
        trace_regime="chain_rule",
        presentation_regime="signed_reciprocal_root_interval",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_root_empty_open_interval_residual",
        expr="diff(arcsin(sqrt(x^2+1)), x)",
        expected_result="diff(arcsin(sqrt(x^2 + 1)), x)",
        expected_blocked_hint_substrings=(
            "real domain is empty",
            "> 0",
        ),
        family="inverse_trig_root",
        argument_regime="quadratic_root_argument",
        domain_regime="empty_open_interval_domain",
        outcome="residual",
        trace_regime="empty_open_interval_policy",
        presentation_regime="residual",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_shifted_quadratic_empty_open_interval_residual",
        expr="diff(arcsin((x+1)^2+1), x)",
        expected_result="diff(arcsin(x^2 + 2·x + 2), x)",
        expected_blocked_hint_substrings=(
            "real domain is empty",
            "> 0",
        ),
        family="inverse_trig_root",
        argument_regime="shifted_quadratic_argument",
        domain_regime="empty_open_interval_domain",
        outcome="residual",
        trace_regime="empty_open_interval_policy",
        presentation_regime="residual",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_symbolic_constant_empty_open_interval_residual",
        expr="diff(arcsin(pi), x)",
        expected_result="diff(arcsin(pi), x)",
        expected_blocked_hint_substrings=(
            "real domain is empty",
            "pi",
            "> 0",
        ),
        family="inverse_trig",
        argument_regime="symbolic_constant",
        domain_regime="empty_open_interval_domain",
        outcome="residual",
        trace_regime="empty_open_interval_policy",
        presentation_regime="residual",
    ),
    DiffCommandMatrixCase(
        name="inverse_hyperbolic_atanh_empty_open_interval_residual",
        expr="diff(atanh(x^2+1), x)",
        expected_result="diff(atanh(x^2 + 1), x)",
        expected_blocked_hint_substrings=(
            "real domain is empty",
            "> 0",
        ),
        family="inverse_hyperbolic",
        argument_regime="quadratic_argument",
        domain_regime="empty_open_interval_domain",
        outcome="residual",
        trace_regime="empty_open_interval_policy",
        presentation_regime="residual",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_root_negative_argument",
        expr="diff(arctan(-sqrt(x)), x)",
        expected_result="-1 / (2·sqrt(x)·(x + 1))",
        expected_required_display=("x > 0",),
        expected_step_substrings=(
            "Inverse Trig Negative Argument",
            "Usar linealidad de la derivada",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="inverse_trig_root",
        argument_regime="negated_nested_root",
        domain_regime="required_condition",
        trace_regime="negative_argument_chain_rule",
        presentation_regime="signed_post_calculus_compact",
    ),
    DiffCommandMatrixCase(
        name="trig_product_rule",
        expr="diff(sin(x)*cos(x), x)",
        expected_result="cos(x)^2 - sin(x)^2",
        expected_step_substrings=(
            "Usar regla del producto",
            "Derivar el primer factor",
            "Derivar el segundo factor",
        ),
        family="trig_product",
        argument_regime="product",
        trace_regime="product_rule",
        presentation_regime="trig_power_difference",
    ),
    DiffCommandMatrixCase(
        name="variable_power_log_domain",
        expr="diff(x^x, x)",
        expected_result="x^x·(ln(x) + 1)",
        expected_required_display=("x > 0",),
        expected_step_substrings=("Usar derivación logarítmica",),
        family="variable_power",
        argument_regime="variable_power",
        domain_regime="required_condition",
        trace_regime="logarithmic_derivative",
        presentation_regime="factored",
    ),
    DiffCommandMatrixCase(
        name="abs_piecewise_required_condition",
        expr="diff(abs(x), x)",
        expected_result="x / |x|",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Usar regla de la cadena",),
        family="abs",
        argument_regime="variable",
        domain_regime="required_condition",
        trace_regime="piecewise_abs",
        presentation_regime="quotient_abs",
    ),
    DiffCommandMatrixCase(
        name="discontinuous_sign_residual_boundary",
        expr="diff(sign(x), x)",
        expected_result="diff(sign(x), x)",
        expected_step_substrings=("Conservar derivada residual",),
        family="discontinuous",
        argument_regime="variable",
        domain_regime="discontinuous_residual",
        outcome="residual",
        trace_regime="residual_policy",
        presentation_regime="residual",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only the named matrix case. Repeatable.",
    )
    parser.add_argument(
        "--cas-cli",
        default=str(DEFAULT_CAS_CLI),
        help="Path to cas_cli.",
    )
    parser.add_argument(
        "--ensure-release-cas-cli",
        action="store_true",
        help="Build target/release/cas_cli if it is missing or stale.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=4.0,
        help="Per-case timeout.",
    )
    parser.add_argument(
        "--slow-wall-seconds",
        type=float,
        default=None,
        help="Mark an otherwise passing case as slow beyond this wall time.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help="When emitting JSON, omit passing case payloads.",
    )
    return parser.parse_args()


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=1.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def parse_json(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(value, dict):
        return None, "json output is not an object"
    return value, None


def extract_required_display(payload: dict[str, Any] | None) -> tuple[str, ...]:
    if not payload:
        return ()
    raw = payload.get("required_display") or []
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, str))


def extract_warning_messages(payload: dict[str, Any] | None) -> tuple[str, ...]:
    if not payload:
        return ()
    raw = payload.get("warnings") or []
    if not isinstance(raw, list):
        return ()
    messages: list[str] = []
    for item in raw:
        if isinstance(item, str):
            messages.append(item)
        elif isinstance(item, dict):
            rule = item.get("rule")
            assumption = item.get("assumption") or item.get("message") or item.get("text")
            parts = [part for part in (rule, assumption) if isinstance(part, str)]
            if parts:
                messages.append(": ".join(parts))
    return tuple(messages)


def extract_blocked_hint_messages(payload: dict[str, Any] | None) -> tuple[str, ...]:
    if not payload:
        return ()
    raw = payload.get("blocked_hints") or []
    if not isinstance(raw, list):
        return ()
    messages: list[str] = []
    for item in raw:
        if isinstance(item, str):
            messages.append(item)
        elif isinstance(item, dict):
            parts: list[str] = []
            for key in ("rule", "tip"):
                value = item.get(key)
                if isinstance(value, str):
                    parts.append(value)
            requires = item.get("requires")
            if isinstance(requires, list):
                parts.extend(value for value in requires if isinstance(value, str))
            if parts:
                messages.append(": ".join(parts))
    return tuple(messages)


def extract_step_text(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    steps = payload.get("steps") or []
    if not isinstance(steps, list):
        return ""

    strings: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(steps)
    return "\n".join(strings)


def warning_expectations_met(
    expected_substrings: tuple[str, ...],
    actual_warnings: tuple[str, ...],
) -> tuple[bool, str | None]:
    if not expected_substrings and actual_warnings:
        return False, f"unexpected warnings {actual_warnings!r}"
    for expected in expected_substrings:
        if not any(expected in warning for warning in actual_warnings):
            return False, f"missing expected warning containing {expected!r}"
    return True, None


def blocked_hint_expectations_met(
    expected_substrings: tuple[str, ...],
    actual_hints: tuple[str, ...],
) -> tuple[bool, str | None]:
    for expected in expected_substrings:
        if not any(expected in hint for hint in actual_hints):
            return False, f"missing expected blocked hint containing {expected!r}"
    return True, None


def classify_error_kind(error: str | None) -> str | None:
    if error is None:
        return None
    if error == "timeout":
        return "timeout"
    if "warning" in error:
        return "warning_mismatch"
    if "blocked hint" in error:
        return "blocked_hint_mismatch"
    if "step trace" in error:
        return "step_trace_mismatch"
    if "required_display" in error:
        return "required_display_mismatch"
    if "result" in error:
        return "result_mismatch"
    if "returncode" in error:
        return "process_error"
    if "json" in error:
        return "parse_error"
    return "unknown"


def run_case(
    case: DiffCommandMatrixCase,
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
            "error_kind": "timeout",
            "returncode": None,
            "wall_elapsed_seconds": round(time.monotonic() - start, 3),
            "stdout": stdout,
            "stderr": stderr,
        }

    wall_elapsed = time.monotonic() - start
    parsed, parse_error = parse_json(stdout)
    result = parsed.get("result") if isinstance(parsed, dict) else None
    required_display = extract_required_display(parsed)
    warnings = extract_warning_messages(parsed)
    blocked_hints = extract_blocked_hint_messages(parsed)
    step_text = extract_step_text(parsed)
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
    elif required_display != case.expected_required_display:
        error = (
            "expected required_display "
            f"{case.expected_required_display!r}, got {required_display!r}"
        )
    else:
        warnings_ok, warning_error = warning_expectations_met(
            case.expected_warning_substrings,
            warnings,
        )
        if not warnings_ok:
            error = warning_error
        else:
            blocked_hints_ok, blocked_hint_error = blocked_hint_expectations_met(
                case.expected_blocked_hint_substrings,
                blocked_hints,
            )
            if not blocked_hints_ok:
                error = blocked_hint_error
            else:
                for expected in case.expected_step_substrings:
                    if expected not in step_text:
                        error = f"missing expected step trace containing {expected!r}"
                        break

    status: Status = "pass" if error is None else "fail"
    error_kind = classify_error_kind(error)
    if status == "pass" and slow_wall_seconds is not None and wall_elapsed > slow_wall_seconds:
        status = "slow"
        error = f"slow: wall_elapsed_seconds={wall_elapsed:.3f}"
        error_kind = "slow"

    return {
        "name": case.name,
        "status": status,
        "error": error,
        "error_kind": error_kind,
        "returncode": process.returncode,
        "wall_elapsed_seconds": round(wall_elapsed, 3),
        "result": result,
        "expected_result": case.expected_result,
        "required_display": list(required_display),
        "expected_required_display": list(case.expected_required_display),
        "warnings": list(warnings),
        "expected_warning_substrings": list(case.expected_warning_substrings),
        "blocked_hints": list(blocked_hints),
        "expected_blocked_hint_substrings": list(case.expected_blocked_hint_substrings),
        "expected_step_substrings": list(case.expected_step_substrings),
        "family": case.family,
        "argument_regime": case.argument_regime,
        "domain_regime": case.domain_regime,
        "outcome": case.outcome,
        "trace_regime": case.trace_regime,
        "presentation_regime": case.presentation_regime,
        "stderr": stderr,
    }


def build_cases(filters: tuple[str, ...] = ()) -> tuple[DiffCommandMatrixCase, ...]:
    if not filters:
        return DEFAULT_DIFF_COMMAND_MATRIX_CASES
    selected = {case.name: case for case in DEFAULT_DIFF_COMMAND_MATRIX_CASES}
    missing = [case for case in filters if case not in selected]
    if missing:
        raise SystemExit(f"unknown diff command matrix case(s): {', '.join(missing)}")
    return tuple(selected[name] for name in filters)


def count_by(cases: tuple[DiffCommandMatrixCase, ...], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        key = getattr(case, attr)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def count_required_display_items(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for item in result.get("required_display", []):
            if not isinstance(item, str):
                continue
            counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items()))


def run_matrix(
    cases: tuple[DiffCommandMatrixCase, ...],
    *,
    cas_cli: str | pathlib.Path,
    timeout_seconds: float,
    slow_wall_seconds: float | None = None,
) -> dict[str, Any]:
    results = [
        run_case(
            case,
            cas_cli=cas_cli,
            timeout_seconds=timeout_seconds,
            slow_wall_seconds=slow_wall_seconds,
        )
        for case in cases
    ]
    status_counts = {"pass": 0, "slow": 0, "fail": 0, "timeout": 0}
    issue_kind_counts: dict[str, int] = {}
    for result in results:
        status_counts[result["status"]] += 1
        error_kind = result.get("error_kind")
        if error_kind:
            issue_kind_counts[error_kind] = issue_kind_counts.get(error_kind, 0) + 1

    problem_cases = [case for case in results if case["status"] != "pass"]
    status = "pass" if not problem_cases else "fail"
    required_displays = {
        item
        for result in results
        for item in result.get("required_display", [])
        if isinstance(item, str)
    }
    supported_step_unchecked_cases = [
        case.name
        for case in cases
        if case.outcome == "supported" and not case.expected_step_substrings
    ]
    return {
        "status": status,
        "total": len(results),
        "status_counts": status_counts,
        "issue_kind_counts": dict(sorted(issue_kind_counts.items())),
        "problem_case_count": len(problem_cases),
        "problem_cases": problem_cases,
        "supported_case_count": sum(1 for case in cases if case.outcome == "supported"),
        "residual_case_count": sum(1 for case in cases if case.outcome == "residual"),
        "warning_expected_case_count": sum(
            1 for case in cases if case.expected_warning_substrings
        ),
        "blocked_hint_expected_case_count": sum(
            1 for case in cases if case.expected_blocked_hint_substrings
        ),
        "required_display_case_count": sum(
            1 for result in results if result.get("required_display")
        ),
        "step_checked_case_count": sum(
            1 for case in cases if case.expected_step_substrings
        ),
        "supported_step_unchecked_case_count": len(supported_step_unchecked_cases),
        "supported_step_unchecked_cases": supported_step_unchecked_cases,
        "expected_step_substring_count": sum(
            len(case.expected_step_substrings) for case in cases
        ),
        "distinct_required_display_count": len(required_displays),
        "required_display_counts": count_required_display_items(results),
        "family_count": len({case.family for case in cases}),
        "argument_regime_counts": count_by(cases, "argument_regime"),
        "domain_regime_counts": count_by(cases, "domain_regime"),
        "outcome_counts": count_by(cases, "outcome"),
        "trace_regime_counts": count_by(cases, "trace_regime"),
        "presentation_regime_counts": count_by(cases, "presentation_regime"),
        "case_filters": [case.name for case in cases],
        "cases": results,
    }


def print_text_summary(matrix: dict[str, Any]) -> None:
    counts = matrix["status_counts"]
    print(
        "diff_command_matrix "
        f"status={matrix['status']} total={matrix['total']} "
        f"pass={counts['pass']} slow={counts['slow']} "
        f"fail={counts['fail']} timeout={counts['timeout']} "
        f"supported={matrix['supported_case_count']} "
        f"residual={matrix['residual_case_count']} "
        f"required_display={matrix['required_display_case_count']} "
        f"step_checked={matrix['step_checked_case_count']}"
    )
    for case in matrix["problem_cases"]:
        print(
            f"- {case['name']}: {case['status']} {case.get('error') or ''}".rstrip()
        )


def main() -> int:
    args = parse_args()
    if args.ensure_release_cas_cli:
        ensure_release_cas_cli(args.cas_cli)
    cases = build_cases(tuple(args.case))
    matrix = run_matrix(
        cases,
        cas_cli=args.cas_cli,
        timeout_seconds=args.timeout_seconds,
        slow_wall_seconds=args.slow_wall_seconds,
    )

    if args.json:
        payload = dict(matrix)
        if args.summary_json:
            payload = {
                key: value
                for key, value in payload.items()
                if key not in {"cases"}
            }
            payload["problem_cases"] = matrix["problem_cases"]
        print(json.dumps(payload, sort_keys=True))
    else:
        print_text_summary(matrix)
    return 0 if matrix["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
