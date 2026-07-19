#!/usr/bin/env python3
"""Command-level limit policy matrix smoke.

This lane complements the algebraic residual matrix. `limit(...)` is a command
surface with finite/infinity policy, so it is checked at top level instead of
inside algebraic wrappers.
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
from engine_command_matrix_observability import (
    extract_warning_messages,
    runtime_observability_summary,
    stderr_fragility_error,
)
from engine_smoke_common import extract_cli_timings_us, parse_json, terminate_process_group


ROOT = SCRIPT_DIR.parent
DEFAULT_CAS_CLI = ROOT / "target" / "release" / "cas_cli"
Status = Literal["pass", "slow", "fail", "timeout"]


@dataclass(frozen=True)
class LimitCommandMatrixCase:
    name: str
    expr: str
    expected_result: str
    expected_required_display: tuple[str, ...] = ()
    expected_warning_substrings: tuple[str, ...] = ()
    expected_step_substrings: tuple[str, ...] = ()
    family: str = "unknown"
    point_regime: str = "finite"
    domain_regime: str = "unconditional"
    required_condition_regime: str = "none"
    outcome: str = "supported"
    residual_cause: str = "not_applicable"
    trace_regime: str = "direct"
    presentation_regime: str = "canonical"


DEFAULT_LIMIT_COMMAND_MATRIX_CASES = (
    LimitCommandMatrixCase(
        name="finite_polynomial_supported",
        expr="limit(x^2 + x + 1, x, -2)",
        expected_result="3",
        expected_step_substrings=("Evaluar límite finito",),
        family="polynomial",
        point_regime="finite",
        trace_regime="substitution",
    ),
    LimitCommandMatrixCase(
        name="finite_rational_required_condition",
        expr="limit((x^2+3*x+2)/(x+2), x, 0)",
        expected_result="1",
        expected_required_display=("x ≠ -2",),
        expected_step_substrings=("Evaluar límite finito",),
        family="rational",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_source_definedness",
        trace_regime="safe_rational_substitution",
    ),
    LimitCommandMatrixCase(
        name="finite_removable_rational_cancellation",
        expr="limit((x^2-1)/(x-1), x, 1)",
        expected_result="2",
        expected_required_display=("x ≠ 1",),
        expected_step_substrings=("Evaluar límite finito",),
        family="rational",
        point_regime="finite",
        domain_regime="removable_hole",
        required_condition_regime="finite_removable_hole",
        trace_regime="rational_multiplicity_policy",
    ),
    LimitCommandMatrixCase(
        name="finite_rational_simple_pole_residual",
        expr="limit(1/x, x, 0)",
        expected_result="undefined",
        expected_required_display=("x ≠ 0",),
        expected_warning_substrings=("the bilateral limit does not exist",),
        expected_step_substrings=("undefined",),
        family="rational",
        point_regime="finite",
        domain_regime="simple_pole_dne",
        required_condition_regime="finite_rational_pole_residual_domain",
        outcome="undefined",
        trace_regime="finite_bilateral_dne_policy",
        presentation_regime="undefined",
    ),
    LimitCommandMatrixCase(
        name="finite_even_order_rational_pole_bilateral_supported",
        expr="limit(2/(x-1)^2, x, 1)",
        expected_result="infinity",
        expected_required_display=("x ≠ 1",),
        expected_step_substrings=("Evaluar límite finito",),
        family="rational",
        point_regime="finite",
        domain_regime="bilateral_even_order_rational_pole",
        required_condition_regime="finite_bilateral_even_order_rational_pole_domain",
        trace_regime="finite_bilateral_rational_pole_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_removable_rational_cancellation",
        expr="limit((x^2-1)/(x-1), x, 1+)",
        expected_result="2",
        expected_required_display=("x ≠ 1",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="rational",
        point_regime="finite_one_sided",
        domain_regime="one_sided_removable_hole",
        required_condition_regime="finite_one_sided_removable_hole",
        trace_regime="one_sided_rational_multiplicity_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_removable_rational_cancellation_with_nonlocal_pole",
        expr="limit((x^2-1)/((x-1)*(x+3)), x, 1+)",
        expected_result="1/2",
        expected_required_display=("x ≠ 1", "x ≠ -3"),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="rational",
        point_regime="finite_one_sided",
        domain_regime="one_sided_removable_hole_with_nonlocal_pole",
        required_condition_regime=(
            "finite_one_sided_removable_hole_with_nonlocal_pole_domain"
        ),
        trace_regime="one_sided_rational_multiplicity_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_rational_simple_pole_supported",
        expr="limit(1/x, x, 0+)",
        expected_result="infinity",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="rational",
        point_regime="finite_one_sided",
        domain_regime="one_sided_simple_pole",
        required_condition_regime="finite_one_sided_rational_pole_domain",
        trace_regime="one_sided_rational_pole_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_shifted_scaled_odd_order_rational_pole_supported",
        expr="limit(2/(x-1)^3, x, 1-)",
        expected_result="-infinity",
        expected_required_display=("x ≠ 1",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="rational",
        point_regime="finite_one_sided",
        domain_regime="one_sided_shifted_scaled_odd_order_pole",
        required_condition_regime=(
            "finite_one_sided_shifted_scaled_odd_order_pole_domain"
        ),
        trace_regime="one_sided_rational_pole_policy",
        presentation_regime="negative_infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_abs_orientation_quotient_supported",
        expr="limit(abs(x)/x, x, 0+)",
        expected_result="1",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="abs_orientation",
        point_regime="finite_one_sided",
        domain_regime="one_sided_orientation",
        required_condition_regime="finite_one_sided_orientation_domain",
        trace_regime="one_sided_orientation_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_abs_quadratic_orientation_quotient_supported",
        expr="limit(abs(x^2-1)/(x^2-1), x, 1-)",
        expected_result="-1",
        expected_required_display=("x ≠ 1", "x ≠ -1"),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="abs_orientation",
        point_regime="finite_one_sided",
        domain_regime="one_sided_factored_orientation",
        required_condition_regime="finite_one_sided_factored_abs_orientation_domain",
        trace_regime="one_sided_orientation_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_sign_quadratic_orientation_supported",
        expr="limit(sign(x^2-1), x, 1-)",
        expected_result="-1",
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="discontinuous",
        point_regime="finite_one_sided",
        domain_regime="one_sided_sign_orientation",
        trace_regime="one_sided_sign_orientation_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_abs_even_order_pole_bilateral_supported",
        expr="limit(abs(x)/(x^2), x, 0)",
        expected_result="infinity",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="abs_orientation",
        point_regime="finite",
        domain_regime="bilateral_abs_even_order_pole",
        required_condition_regime="finite_bilateral_abs_even_order_pole_domain",
        trace_regime="finite_bilateral_abs_orientation_pole_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_log_zero_endpoint_supported",
        expr="limit(ln(x), x, 0+)",
        expected_result="-infinity",
        expected_required_display=("x > 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="log_endpoint",
        point_regime="finite_one_sided",
        domain_regime="one_sided_log_endpoint",
        required_condition_regime="finite_one_sided_log_endpoint_domain",
        trace_regime="one_sided_log_endpoint_policy",
        presentation_regime="negative_infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_bilateral_log_even_order_endpoint_supported",
        expr="limit(ln(x^2), x, 0)",
        expected_result="-infinity",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="log_endpoint",
        point_regime="finite",
        domain_regime="bilateral_log_even_order_endpoint",
        required_condition_regime="finite_bilateral_log_even_order_endpoint_domain",
        trace_regime="finite_bilateral_log_endpoint_policy",
        presentation_regime="negative_infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_bilateral_reciprocal_base_log_even_order_endpoint_supported",
        expr="limit(log(1/2,x^2), x, 0)",
        expected_result="infinity",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="binary_log_endpoint",
        point_regime="finite",
        domain_regime="bilateral_reciprocal_base_log_even_order_endpoint",
        required_condition_regime=(
            "finite_bilateral_reciprocal_base_log_even_order_endpoint_domain"
        ),
        trace_regime="finite_bilateral_log_endpoint_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_fixed_base_log_zero_endpoint_supported",
        expr="limit(log2(x), x, 0+)",
        expected_result="-infinity",
        expected_required_display=("x > 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="fixed_base_log_endpoint",
        point_regime="finite_one_sided",
        domain_regime="one_sided_fixed_base_log_endpoint",
        required_condition_regime="finite_one_sided_fixed_base_log_endpoint_domain",
        trace_regime="one_sided_log_endpoint_policy",
        presentation_regime="negative_infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_binary_log_constant_base_less_than_one_endpoint_supported",
        expr="limit(log(1/2,x), x, 0+)",
        expected_result="infinity",
        expected_required_display=("x > 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="binary_log_endpoint",
        point_regime="finite_one_sided",
        domain_regime="one_sided_constant_base_less_than_one_log_endpoint",
        required_condition_regime=(
            "finite_one_sided_constant_base_less_than_one_log_endpoint_domain"
        ),
        trace_regime="one_sided_log_endpoint_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_log_rational_zero_tail_endpoint_supported",
        expr="limit(ln((x-1)/(x+3)), x, 1+)",
        expected_result="-infinity",
        expected_required_display=("x < -3 or x > 1",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="log_rational_endpoint",
        point_regime="finite_one_sided",
        domain_regime="one_sided_rational_log_zero_tail_endpoint",
        required_condition_regime=(
            "finite_one_sided_log_rational_zero_tail_endpoint_domain"
        ),
        trace_regime="one_sided_rational_log_endpoint_policy",
        presentation_regime="negative_infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_variable_base_log_rational_zero_tail_endpoint_supported",
        expr="limit(log(x-1/2, (x-1)/(x+3)), x, 1+)",
        expected_result="infinity",
        expected_required_display=("x < -3 or x > 1", "x > 1/2", "x ≠ 3/2"),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="log_rational_endpoint",
        point_regime="finite_one_sided",
        domain_regime="one_sided_variable_base_rational_log_zero_tail_endpoint",
        required_condition_regime=(
            "finite_one_sided_variable_base_log_rational_zero_tail_endpoint_domain"
        ),
        trace_regime="one_sided_rational_log_endpoint_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_unit_base_boundary_log_rational_zero_tail_endpoint_supported",
        expr="limit(log(x, (x-1)/(x+3)), x, 1+)",
        expected_result="-infinity",
        expected_required_display=("x < -3 or x > 1", "x > 0"),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="log_rational_endpoint",
        point_regime="finite_one_sided",
        domain_regime="one_sided_unit_base_boundary_rational_log_zero_tail_endpoint",
        required_condition_regime=(
            "finite_one_sided_unit_base_boundary_log_rational_zero_tail_endpoint_domain"
        ),
        trace_regime="one_sided_unit_base_boundary_rational_log_endpoint_policy",
        presentation_regime="negative_infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_rational_unit_base_boundary_log_rational_zero_tail_endpoint_supported",
        expr="limit(log((x+2)/(2*x+1), (x-1)/(x+3)), x, 1+)",
        expected_result="infinity",
        expected_required_display=("x < -2 or x > -1/2", "x < -3 or x > 1"),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="log_rational_endpoint",
        point_regime="finite_one_sided",
        domain_regime="one_sided_rational_unit_base_boundary_log_zero_tail_endpoint",
        required_condition_regime=(
            "finite_one_sided_rational_unit_base_boundary_log_zero_tail_endpoint_domain"
        ),
        trace_regime="one_sided_rational_unit_base_boundary_log_endpoint_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_sqrt_zero_endpoint_supported",
        expr="limit(sqrt(x), x, 0+)",
        expected_result="0",
        expected_required_display=("x ≥ 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="root_endpoint",
        point_regime="finite_one_sided",
        domain_regime="one_sided_root_endpoint",
        required_condition_regime="finite_one_sided_root_endpoint_domain",
        trace_regime="one_sided_root_endpoint_policy",
        presentation_regime="exact_zero",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_sqrt_shifted_zero_endpoint_supported",
        expr="limit(sqrt(x+1), x, -1+)",
        expected_result="0",
        expected_required_display=("x ≥ -1",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="root_endpoint",
        point_regime="finite_one_sided",
        domain_regime="one_sided_shifted_root_endpoint",
        required_condition_regime="finite_one_sided_root_endpoint_domain",
        trace_regime="one_sided_root_endpoint_policy",
        presentation_regime="exact_zero",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_half_power_zero_endpoint_supported",
        expr="limit(x^(1/2), x, 0+)",
        expected_result="0",
        expected_required_display=("x ≥ 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="root_endpoint",
        point_regime="finite_one_sided",
        domain_regime="one_sided_half_power_endpoint",
        required_condition_regime="finite_one_sided_root_endpoint_domain",
        trace_regime="one_sided_root_endpoint_policy",
        presentation_regime="exact_zero",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_sqrt_rational_zero_tail_endpoint_supported",
        expr="limit(sqrt((x-1)/(x+3)), x, 1+)",
        expected_result="0",
        expected_required_display=("(x - 1) / (x + 3) ≥ 0", "x ≠ -3"),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="root_rational_endpoint",
        point_regime="finite_one_sided",
        domain_regime="one_sided_rational_root_zero_tail_endpoint",
        required_condition_regime=(
            "finite_one_sided_root_rational_zero_tail_endpoint_domain"
        ),
        trace_regime="one_sided_rational_root_endpoint_policy",
        presentation_regime="exact_zero",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_sqrt_rational_reverse_zero_tail_endpoint_supported",
        expr="limit(sqrt((1-x)/(x+3)), x, 1-)",
        expected_result="0",
        expected_required_display=("(1 - x) / (x + 3) ≥ 0", "x ≠ -3"),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="root_rational_endpoint",
        point_regime="finite_one_sided",
        domain_regime="one_sided_rational_root_reverse_zero_tail_endpoint",
        required_condition_regime=(
            "finite_one_sided_root_rational_zero_tail_endpoint_domain"
        ),
        trace_regime="one_sided_rational_root_endpoint_policy",
        presentation_regime="exact_zero",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_sqrt_rational_domain_path_conflict_residual",
        expr="limit(sqrt((x-1)/(x+3)), x, 1-)",
        expected_result="limit(sqrt((x - 1) / (x + 3)), x, 1, left)",
        expected_required_display=("(x - 1) / (x + 3) ≥ 0", "x ≠ -3"),
        expected_warning_substrings=(
            "One-sided finite point limits are not supported safely for this expression yet",
            "Limit path conflicts with the input domain",
        ),
        expected_step_substrings=("Conservar límite residual",),
        family="root_rational_endpoint",
        point_regime="finite_one_sided",
        domain_regime="rational_domain_path_conflict",
        required_condition_regime="finite_one_sided_rational_path_conflict",
        outcome="residual",
        residual_cause="one_sided_domain_path_conflict",
        trace_regime="one_sided_rational_domain_path_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_sqrt_domain_path_conflict_residual",
        expr="limit(sqrt(x), x, 0-)",
        expected_result="limit(sqrt(x), x, 0, left)",
        expected_required_display=("x ≥ 0",),
        expected_warning_substrings=(
            "One-sided finite point limits are not supported safely for this expression yet",
            "Limit path conflicts with the input domain",
        ),
        expected_step_substrings=("Conservar límite residual",),
        family="root_endpoint",
        point_regime="finite_one_sided",
        domain_regime="domain_path_conflict",
        required_condition_regime="finite_one_sided_path_conflict",
        outcome="residual",
        residual_cause="one_sided_domain_path_conflict",
        trace_regime="one_sided_domain_path_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_sqrt_shifted_domain_path_conflict_residual",
        expr="limit(sqrt(x+1), x, -1-)",
        expected_result="limit(sqrt(x + 1), x, -1, left)",
        expected_required_display=("x ≥ -1",),
        expected_warning_substrings=(
            "One-sided finite point limits are not supported safely for this expression yet",
            "Limit path conflicts with the input domain",
        ),
        expected_step_substrings=("Conservar límite residual",),
        family="root_endpoint",
        point_regime="finite_one_sided",
        domain_regime="shifted_root_domain_path_conflict",
        required_condition_regime="finite_one_sided_path_conflict",
        outcome="residual",
        residual_cause="one_sided_domain_path_conflict",
        trace_regime="one_sided_domain_path_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_acosh_lower_bound_endpoint_supported",
        expr="limit(acosh(x), x, 1+)",
        expected_result="0",
        expected_required_display=("x ≥ 1",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="inverse_hyperbolic",
        point_regime="finite_one_sided",
        domain_regime="one_sided_acosh_lower_bound_endpoint",
        required_condition_regime="finite_one_sided_acosh_lower_bound_endpoint_domain",
        trace_regime="one_sided_acosh_lower_bound_endpoint_policy",
        presentation_regime="exact_zero",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_acosh_lower_bound_domain_path_conflict_residual",
        expr="limit(acosh(x), x, 1-)",
        expected_result="limit(acosh(x), x, 1, left)",
        expected_required_display=("x ≥ 1",),
        expected_warning_substrings=(
            "One-sided finite point limits are not supported safely for this expression yet",
            "Limit path conflicts with the input domain",
        ),
        expected_step_substrings=("Conservar límite residual",),
        family="inverse_hyperbolic",
        point_regime="finite_one_sided",
        domain_regime="lower_bound_domain_path_conflict",
        required_condition_regime="finite_one_sided_lower_bound_path_conflict",
        outcome="residual",
        residual_cause="one_sided_domain_path_conflict",
        trace_regime="one_sided_domain_path_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_inverse_trig_upper_endpoint_supported",
        expr="limit(acos(x), x, 1-)",
        expected_result="0",
        expected_required_display=("-1 ≤ x ≤ 1",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="inverse_trig",
        point_regime="finite_one_sided",
        domain_regime="one_sided_inverse_trig_upper_endpoint",
        required_condition_regime="finite_one_sided_inverse_trig_endpoint_domain",
        trace_regime="one_sided_inverse_trig_endpoint_policy",
        presentation_regime="exact_zero",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_inverse_trig_upper_domain_path_conflict_residual",
        expr="limit(acos(x), x, 1+)",
        expected_result="limit(acos(x), x, 1, right)",
        expected_required_display=("-1 ≤ x ≤ 1",),
        expected_warning_substrings=(
            "One-sided finite point limits are not supported safely for this expression yet",
            "Limit path conflicts with the input domain",
        ),
        expected_step_substrings=("Conservar límite residual",),
        family="inverse_trig",
        point_regime="finite_one_sided",
        domain_regime="inverse_trig_domain_path_conflict",
        required_condition_regime="finite_one_sided_inverse_trig_path_conflict",
        outcome="residual",
        residual_cause="one_sided_domain_path_conflict",
        trace_regime="one_sided_domain_path_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_atanh_upper_endpoint_supported",
        expr="limit(atanh(x), x, 1-)",
        expected_result="infinity",
        expected_required_display=("-1 < x < 1",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="inverse_hyperbolic",
        point_regime="finite_one_sided",
        domain_regime="one_sided_atanh_open_interval_endpoint",
        required_condition_regime="finite_one_sided_atanh_endpoint_domain",
        trace_regime="one_sided_atanh_endpoint_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_atanh_lower_endpoint_supported",
        expr="limit(atanh(x), x, -1+)",
        expected_result="-infinity",
        expected_required_display=("-1 < x < 1",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="inverse_hyperbolic",
        point_regime="finite_one_sided",
        domain_regime="one_sided_atanh_lower_open_interval_endpoint",
        required_condition_regime="finite_one_sided_atanh_lower_endpoint_domain",
        trace_regime="one_sided_atanh_endpoint_policy",
        presentation_regime="negative_infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_atanh_upper_domain_path_conflict_residual",
        expr="limit(atanh(x), x, 1+)",
        expected_result="limit(atanh(x), x, 1, right)",
        expected_required_display=("-1 < x < 1",),
        expected_warning_substrings=(
            "One-sided finite point limits are not supported safely for this expression yet",
            "Limit path conflicts with the input domain",
        ),
        expected_step_substrings=("Conservar límite residual",),
        family="inverse_hyperbolic",
        point_regime="finite_one_sided",
        domain_regime="atanh_open_interval_domain_path_conflict",
        required_condition_regime="finite_one_sided_atanh_path_conflict",
        outcome="residual",
        residual_cause="one_sided_domain_path_conflict",
        trace_regime="one_sided_domain_path_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_log_required_condition",
        expr="limit(ln(x+3), x, 0)",
        expected_result="ln(3)",
        expected_required_display=("x > -3",),
        expected_step_substrings=("Evaluar límite finito",),
        family="log",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_positive_domain",
        trace_regime="positive_domain_substitution",
    ),
    LimitCommandMatrixCase(
        name="finite_log_positive_scaled_abs_quotient_domain",
        expr="limit(ln(3*abs((x-1)/(x+1))), x, 0)",
        expected_result="ln(3)",
        expected_required_display=("x ≠ 1", "x ≠ -1"),
        expected_step_substrings=("Evaluar límite finito",),
        family="log_abs_quotient",
        point_regime="finite",
        domain_regime="positive_scaled_abs_quotient_domain",
        required_condition_regime="finite_positive_scaled_abs_quotient_domain",
        trace_regime="positive_scaled_abs_quotient_substitution",
        presentation_regime="canonical_log_with_atomic_domain",
    ),
    LimitCommandMatrixCase(
        name="finite_log_exact_e_point_required_condition",
        expr="limit(ln(x), x, e)",
        expected_result="1",
        expected_required_display=("x > 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="log",
        point_regime="finite",
        domain_regime="exact_constant_required_condition",
        required_condition_regime="finite_log_exact_e_point_positive_domain",
        trace_regime="positive_domain_substitution",
        presentation_regime="exact_one",
    ),
    LimitCommandMatrixCase(
        name="finite_log_argument_zero_endpoint_residual",
        expr="limit(ln(x), x, 0)",
        expected_result="limit(ln(x), x, 0)",
        expected_required_display=("x > 0",),
        expected_warning_substrings=("Finite point limits are not supported safely yet",),
        expected_step_substrings=("Conservar límite residual",),
        family="log_domain_policy",
        point_regime="finite",
        domain_regime="argument_zero_endpoint_residual",
        required_condition_regime="finite_log_endpoint_residual_domain",
        outcome="residual",
        residual_cause="finite_endpoint_or_boundary_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_fixed_base_log_argument_zero_endpoint_residual",
        expr="limit(log2(x), x, 0)",
        expected_result="limit(log2(x), x, 0)",
        expected_required_display=("x > 0",),
        expected_warning_substrings=("Finite point limits are not supported safely yet",),
        expected_step_substrings=("Conservar límite residual",),
        family="fixed_base_log_domain_policy",
        point_regime="finite",
        domain_regime="fixed_base_argument_zero_endpoint_residual",
        required_condition_regime="finite_fixed_base_log_endpoint_residual_domain",
        outcome="residual",
        residual_cause="finite_endpoint_or_boundary_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_binary_log_constant_base_argument_zero_endpoint_residual",
        expr="limit(log(2,x), x, 0)",
        expected_result="limit(log(2, x), x, 0)",
        expected_required_display=("x > 0",),
        expected_warning_substrings=("Finite point limits are not supported safely yet",),
        expected_step_substrings=("Conservar límite residual",),
        family="binary_log_domain_policy",
        point_regime="finite",
        domain_regime="constant_base_argument_zero_endpoint_residual",
        required_condition_regime="finite_binary_log_constant_base_endpoint_residual_domain",
        outcome="residual",
        residual_cause="finite_endpoint_or_boundary_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="binary_log_residual",
    ),
    LimitCommandMatrixCase(
        name="finite_acosh_bilateral_lower_bound_endpoint_supported",
        expr="limit(acosh(1+x^2), x, 0)",
        expected_result="0",
        expected_required_display=("x^2 + 1 ≥ 1",),
        expected_step_substrings=("Evaluar límite finito",),
        family="inverse_hyperbolic",
        point_regime="finite",
        domain_regime="bilateral_lower_bound_endpoint",
        required_condition_regime="finite_acosh_bilateral_lower_bound_endpoint_domain",
        trace_regime="finite_acosh_bilateral_lower_bound_endpoint_policy",
        presentation_regime="exact_zero",
    ),
    LimitCommandMatrixCase(
        name="finite_inverse_trig_bilateral_upper_endpoint_supported",
        expr="limit(acos(1-x^2), x, 0)",
        expected_result="0",
        expected_required_display=("-1 ≤ x^2 - 1 ≤ 1",),
        expected_step_substrings=("Evaluar límite finito",),
        family="inverse_trig",
        point_regime="finite",
        domain_regime="bilateral_upper_bound_endpoint",
        required_condition_regime="finite_inverse_trig_bilateral_upper_endpoint_domain",
        trace_regime="finite_inverse_trig_bilateral_upper_endpoint_policy",
        presentation_regime="exact_zero",
    ),
    LimitCommandMatrixCase(
        name="finite_inverse_trig_empty_punctured_upper_endpoint_residual",
        expr="limit(acos(1+x^2), x, 0)",
        expected_result="limit(acos(x^2 + 1), x, 0)",
        expected_required_display=("-1 ≤ x^2 + 1 ≤ 1",),
        expected_warning_substrings=(
            "Finite point limits are not supported safely yet",
            "no punctured real neighbourhood",
        ),
        expected_step_substrings=("Conservar límite residual",),
        family="inverse_trig",
        point_regime="finite",
        domain_regime="empty_punctured_interval_endpoint_residual",
        required_condition_regime="finite_empty_punctured_inverse_trig_endpoint_residual_domain",
        outcome="residual",
        residual_cause="finite_endpoint_empty_punctured_domain_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_inverse_trig_empty_punctured_lower_endpoint_residual",
        expr="limit(acos(-1-x^2), x, 0)",
        expected_result="limit(acos(-1 - x^2), x, 0)",
        expected_required_display=("-1 ≤ x^2 + 1 ≤ 1",),
        expected_warning_substrings=(
            "Finite point limits are not supported safely yet",
            "no punctured real neighbourhood",
        ),
        expected_step_substrings=("Conservar límite residual",),
        family="inverse_trig",
        point_regime="finite",
        domain_regime="empty_punctured_interval_lower_endpoint_residual",
        required_condition_regime="finite_empty_punctured_inverse_trig_endpoint_residual_domain",
        outcome="residual",
        residual_cause="finite_endpoint_empty_punctured_domain_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_inverse_trig_bilateral_lower_endpoint_supported",
        expr="limit(acos(-1+x^2), x, 0)",
        expected_result="pi",
        expected_required_display=("-1 ≤ x^2 - 1 ≤ 1",),
        expected_step_substrings=("Evaluar límite finito",),
        family="inverse_trig",
        point_regime="finite",
        domain_regime="bilateral_lower_bound_endpoint",
        required_condition_regime="finite_inverse_trig_bilateral_lower_endpoint_domain",
        trace_regime="finite_inverse_trig_bilateral_lower_endpoint_policy",
        presentation_regime="exact_pi",
    ),
    LimitCommandMatrixCase(
        name="finite_log_root_structurally_positive_composition",
        expr="limit(ln(sqrt(x^2+1)), x, 0)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite finito",),
        family="log_root_composition",
        point_regime="finite",
        domain_regime="structurally_positive_composition",
        trace_regime="nested_positive_domain_substitution",
        presentation_regime="canonical_no_required_conditions",
    ),
    LimitCommandMatrixCase(
        name="finite_total_real_over_positive_sublimit_composition",
        expr="limit(sin(sqrt(x^2+1)), x, -2)",
        expected_result="sin(sqrt(5))",
        expected_step_substrings=("Evaluar límite finito",),
        family="total_real_root_composition",
        point_regime="finite",
        domain_regime="structurally_positive_nested_composition",
        trace_regime="total_real_over_positive_sublimit_substitution",
        presentation_regime="symbolic_trig_over_radical",
    ),
    LimitCommandMatrixCase(
        name="finite_log_rational_positive_sublimit_domain",
        expr="limit(ln((x+2)/(x+3)), x, 0)",
        expected_result="ln(2/3)",
        expected_required_display=("x < -3 or x > -2",),
        expected_step_substrings=("Evaluar límite finito",),
        family="log_rational_composition",
        point_regime="finite",
        domain_regime="rational_positive_sublimit_domain",
        required_condition_regime="finite_log_rational_positive_sublimit_domain",
        trace_regime="finite_positive_domain_substitution",
        presentation_regime="exact_log_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_sqrt_structurally_positive_radical_presentation",
        expr="limit(sqrt((x+1)^2+1), x, 0)",
        expected_result="sqrt(2)",
        expected_step_substrings=("Evaluar límite finito",),
        family="root",
        point_regime="finite",
        domain_regime="structurally_positive_argument",
        trace_regime="positive_radical_substitution",
        presentation_regime="exact_radical",
    ),
    LimitCommandMatrixCase(
        name="finite_negative_integer_power_nonzero_root_base",
        expr="limit((sqrt(x+4))^(-2), x, 0)",
        expected_result="1/4",
        expected_required_display=("x > -4",),
        expected_step_substrings=("Evaluar límite finito",),
        family="finite_power",
        point_regime="finite",
        domain_regime="positive_root_nonzero_power_required",
        required_condition_regime="finite_positive_nonzero_power_domain",
        trace_regime="finite_negative_integer_power_policy",
        presentation_regime="reciprocal_power",
    ),
    LimitCommandMatrixCase(
        name="finite_inverse_trig_root_interior_special_angle",
        expr="limit(arcsin(sqrt(x)), x, 1/4)",
        expected_result="1/6·pi",
        expected_required_display=("x ≤ 1", "x ≥ 0"),
        expected_step_substrings=("Evaluar límite finito",),
        family="inverse_trig_root",
        point_regime="finite",
        domain_regime="root_interior_interval_required",
        required_condition_regime="finite_interval_domain",
        trace_regime="inverse_trig_root_finite_substitution",
        presentation_regime="special_angle",
    ),
    LimitCommandMatrixCase(
        name="finite_binary_log_variable_base_domain",
        expr="limit(log(x+2, x+3), x, 0)",
        expected_result="log(2, 3)",
        expected_required_display=("x > -2", "x ≠ -1"),
        expected_step_substrings=("Evaluar límite finito",),
        family="binary_log",
        point_regime="finite",
        domain_regime="valid_base_and_positive_argument",
        required_condition_regime="finite_log_base_argument_domain",
        trace_regime="binary_log_finite_substitution",
        presentation_regime="binary_log",
    ),
    LimitCommandMatrixCase(
        name="finite_binary_log_unit_base_boundary_residual",
        expr="limit(log(x+1, x+3), x, 0)",
        expected_result="limit(log(x + 1, x + 3), x, 0)",
        expected_required_display=("x > -1", "x ≠ 0"),
        expected_warning_substrings=("Finite point limits are not supported safely yet",),
        expected_step_substrings=("Conservar límite residual",),
        family="binary_log_domain_policy",
        point_regime="finite",
        domain_regime="unit_base_boundary_residual",
        required_condition_regime="finite_boundary_residual_domain",
        outcome="residual",
        residual_cause="finite_endpoint_or_boundary_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_binary_log_argument_zero_boundary_residual",
        expr="limit(log(x+2, x), x, 0)",
        expected_result="limit(log(x + 2, x), x, 0)",
        expected_required_display=("x > 0",),
        expected_warning_substrings=("Finite point limits are not supported safely yet",),
        expected_step_substrings=("Conservar límite residual",),
        family="binary_log_domain_policy",
        point_regime="finite",
        domain_regime="argument_zero_boundary_residual",
        required_condition_regime="finite_boundary_residual_domain",
        outcome="residual",
        residual_cause="finite_endpoint_or_boundary_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_static_invalid_log_undefined",
        expr="limit(log(1,2), x, 0)",
        expected_result="undefined",
        expected_step_substrings=("undefined",),
        family="log_domain_policy",
        point_regime="finite",
        domain_regime="empty_real_domain",
        outcome="undefined",
        trace_regime="static_empty_domain_policy",
        presentation_regime="undefined",
    ),
    LimitCommandMatrixCase(
        name="finite_invalid_binary_log_base_dependent_argument_undefined",
        expr="limit(log(1,x^2), x, 0)",
        expected_result="undefined",
        expected_step_substrings=("undefined",),
        family="binary_log_domain_policy",
        point_regime="finite",
        domain_regime="invalid_constant_base_dependent_argument_empty_real_domain",
        outcome="undefined",
        trace_regime="static_empty_domain_policy",
        presentation_regime="undefined",
    ),
    LimitCommandMatrixCase(
        name="finite_inverse_hyperbolic_empty_open_interval_undefined",
        expr="limit(atanh(x^2+2), x, 0)",
        expected_result="undefined",
        expected_step_substrings=("undefined",),
        family="inverse_hyperbolic_domain_policy",
        point_regime="finite",
        domain_regime="empty_open_interval_domain",
        outcome="undefined",
        trace_regime="static_empty_domain_policy",
        presentation_regime="undefined",
    ),
    LimitCommandMatrixCase(
        name="finite_sqrt_bilateral_even_gap_endpoint_supported",
        expr="limit(sqrt(x^2), x, 0)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite finito",),
        family="root",
        point_regime="finite",
        domain_regime="bilateral_even_gap_endpoint",
        trace_regime="finite_sqrt_bilateral_endpoint_policy",
        presentation_regime="exact_zero",
    ),
    LimitCommandMatrixCase(
        name="finite_sqrt_empty_punctured_endpoint_residual",
        expr="limit(sqrt(-x^2), x, 0)",
        expected_result="limit(sqrt(-(x^2)), x, 0)",
        expected_required_display=("0 ≤ x ≤ 0",),
        expected_warning_substrings=(
            "Finite point limits are not supported safely yet",
            "no punctured real neighbourhood",
        ),
        expected_step_substrings=("Conservar límite residual",),
        family="root",
        point_regime="finite",
        domain_regime="empty_punctured_endpoint_residual",
        required_condition_regime="finite_empty_punctured_endpoint_residual_domain",
        outcome="residual",
        residual_cause="finite_endpoint_empty_punctured_domain_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_acosh_empty_punctured_lower_bound_endpoint_residual",
        expr="limit(acosh(1-x^2), x, 0)",
        expected_result="limit(acosh(1 - x^2), x, 0)",
        expected_required_display=("1 - x^2 ≥ 1",),
        expected_warning_substrings=(
            "Finite point limits are not supported safely yet",
            "no punctured real neighbourhood",
        ),
        expected_step_substrings=("Conservar límite residual",),
        family="inverse_hyperbolic_domain_policy",
        point_regime="finite",
        domain_regime="empty_punctured_lower_bound_endpoint_residual",
        required_condition_regime="finite_empty_punctured_lower_bound_endpoint_residual_domain",
        outcome="residual",
        residual_cause="finite_endpoint_empty_punctured_domain_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_sqrt_endpoint_residual_boundary",
        expr="limit(x^(1/2), x, 0)",
        expected_result="limit(x^(1 / 2), x, 0)",
        expected_required_display=("x ≥ 0",),
        expected_warning_substrings=("Finite point limits are not supported safely yet",),
        expected_step_substrings=("Conservar límite residual",),
        family="root",
        point_regime="finite",
        domain_regime="endpoint_residual",
        required_condition_regime="finite_endpoint_residual_domain",
        outcome="residual",
        residual_cause="finite_endpoint_or_boundary_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_sqrt_endpoint_residual_presentation_cleanup",
        expr="limit(sqrt(x + 0), x, 0)",
        expected_result="limit(sqrt(x), x, 0)",
        expected_required_display=("x ≥ 0",),
        expected_warning_substrings=("Finite point limits are not supported safely yet",),
        expected_step_substrings=("Conservar límite residual", "limit(sqrt(x), x, 0)"),
        family="root",
        point_regime="finite",
        domain_regime="endpoint_residual",
        required_condition_regime="finite_endpoint_residual_domain",
        outcome="residual",
        residual_cause="finite_endpoint_or_boundary_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="residual_presentation_cleanup",
    ),
    LimitCommandMatrixCase(
        name="finite_discontinuous_sign_residual_boundary",
        expr="limit(sign(x), x, 0)",
        expected_result="undefined",
        expected_warning_substrings=("the bilateral limit does not exist",),
        expected_step_substrings=("undefined",),
        family="discontinuous",
        point_regime="finite",
        domain_regime="discontinuous_residual",
        outcome="undefined",
        trace_regime="finite_bilateral_dne_policy",
        presentation_regime="undefined",
    ),
    LimitCommandMatrixCase(
        name="finite_discontinuous_sign_residual_presentation_cleanup",
        expr="limit(sign(x + 0), x, 0)",
        expected_result="undefined",
        expected_warning_substrings=("the bilateral limit does not exist",),
        expected_step_substrings=("undefined",),
        family="discontinuous",
        point_regime="finite",
        domain_regime="discontinuous_residual",
        outcome="undefined",
        trace_regime="finite_bilateral_dne_policy",
        presentation_regime="undefined",
    ),
    LimitCommandMatrixCase(
        name="finite_bilateral_sign_even_order_orientation_supported",
        expr="limit(sign(x^2), x, 0)",
        expected_result="1",
        expected_step_substrings=("Evaluar límite finito",),
        family="discontinuous",
        point_regime="finite",
        domain_regime="bilateral_sign_even_order_orientation",
        trace_regime="finite_bilateral_sign_orientation_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_abs_orientation_quotient_residual_boundary",
        expr="limit(abs(x)/x, x, 0)",
        expected_result="undefined",
        expected_required_display=("x ≠ 0",),
        expected_warning_substrings=("the bilateral limit does not exist",),
        expected_step_substrings=("undefined",),
        family="abs_orientation",
        point_regime="finite",
        domain_regime="orientation_discontinuity_residual",
        required_condition_regime="finite_abs_orientation_denominator_domain",
        outcome="undefined",
        trace_regime="finite_bilateral_dne_policy",
        presentation_regime="undefined",
    ),
    LimitCommandMatrixCase(
        name="finite_trig_sine_even_power_pole_bilateral_supported",
        expr="limit(1/(sin(x)^2), x, 0)",
        expected_result="infinity",
        expected_required_display=("sin(x) ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="trig_pole",
        point_regime="finite",
        domain_regime="bilateral_even_order_trig_sine_pole",
        required_condition_regime="finite_bilateral_trig_sine_power_pole_domain",
        trace_regime="finite_bilateral_trig_pole_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_trig_sine_special_point_even_power_pole_bilateral_supported",
        expr="limit(1/(sin(x)^2), x, pi)",
        expected_result="infinity",
        expected_required_display=("sin(x) ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="trig_pole",
        point_regime="finite",
        domain_regime="bilateral_even_order_trig_sine_special_point_pole",
        required_condition_regime="finite_bilateral_trig_sine_special_point_power_pole_domain",
        trace_regime="finite_bilateral_trig_pole_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_trig_sine_rational_pi_multiple_even_power_pole_bilateral_supported",
        expr="limit(1/(sin(x)^2), x, 2*pi)",
        expected_result="infinity",
        expected_required_display=("sin(x) ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="trig_pole",
        point_regime="finite",
        domain_regime="bilateral_even_order_trig_sine_rational_pi_multiple_pole",
        required_condition_regime="finite_bilateral_trig_sine_rational_pi_multiple_power_pole_domain",
        trace_regime="finite_bilateral_trig_pole_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_trig_sine_higher_even_power_pole_bilateral_supported",
        expr="limit(1/(sin(x)^4), x, 0)",
        expected_result="infinity",
        expected_required_display=("sin(x) ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="trig_pole",
        point_regime="finite",
        domain_regime="bilateral_higher_even_order_trig_sine_pole",
        required_condition_regime="finite_bilateral_trig_sine_higher_even_power_pole_domain",
        trace_regime="finite_bilateral_trig_pole_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_trig_cosine_even_power_pole_bilateral_supported",
        expr="limit(1/(cos(pi/2 + x)^2), x, 0)",
        expected_result="infinity",
        expected_required_display=("cos(pi / 2 + x) ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="trig_pole",
        point_regime="finite",
        domain_regime="bilateral_even_order_trig_cosine_pole",
        required_condition_regime="finite_bilateral_trig_cosine_power_pole_domain",
        trace_regime="finite_bilateral_trig_pole_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_trig_cosine_special_point_even_power_pole_bilateral_supported",
        expr="limit(1/(cos(x)^2), x, pi/2)",
        expected_result="infinity",
        expected_required_display=("cos(x) ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="trig_pole",
        point_regime="finite",
        domain_regime="bilateral_even_order_trig_cosine_special_point_pole",
        required_condition_regime="finite_bilateral_trig_cosine_special_point_power_pole_domain",
        trace_regime="finite_bilateral_trig_pole_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_trig_special_angle_structural_domain",
        expr="limit(tan(pi/4 + x - x), x, 0)",
        expected_result="1",
        expected_step_substrings=("Evaluar límite finito",),
        family="trig_special_angle",
        point_regime="finite",
        domain_regime="structurally_satisfied_domain",
        trace_regime="finite_special_angle_table",
        presentation_regime="canonical_no_required_conditions",
    ),
    LimitCommandMatrixCase(
        name="finite_trig_table_undefined_pole_residual",
        expr="limit(tan(x + pi/2), x, 0)",
        expected_result="undefined",
        expected_required_display=("cos(pi / 2 + x) ≠ 0",),
        expected_warning_substrings=("the bilateral limit does not exist",),
        expected_step_substrings=("undefined",),
        family="trig_special_angle",
        point_regime="finite",
        domain_regime="table_undefined_trig_pole_residual",
        required_condition_regime="finite_trig_pole_residual_domain",
        outcome="undefined",
        trace_regime="finite_bilateral_dne_policy",
        presentation_regime="undefined",
    ),
    LimitCommandMatrixCase(
        name="finite_reciprocal_trig_sine_even_power_pole_bilateral_supported",
        expr="limit(csc(x)^2, x, 0)",
        expected_result="infinity",
        expected_required_display=("sin(x) ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="reciprocal_trig",
        point_regime="finite",
        domain_regime="bilateral_even_order_reciprocal_trig_sine_pole",
        required_condition_regime=(
            "finite_bilateral_reciprocal_trig_sine_even_power_pole_domain"
        ),
        trace_regime="finite_bilateral_trig_pole_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_tangent_pole_supported",
        expr="limit(tan(x + pi/2), x, 0+)",
        expected_result="-infinity",
        expected_required_display=("cos(pi / 2 + x) ≠ 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="trig_special_angle",
        point_regime="finite_one_sided",
        domain_regime="one_sided_tangent_pole",
        required_condition_regime="finite_one_sided_tangent_pole_domain",
        trace_regime="finite_one_sided_trig_function_pole_policy",
        presentation_regime="negative_infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_explicit_tangent_ratio_pole_supported",
        expr="limit(sin(x + pi/2)/cos(x + pi/2), x, 0+)",
        expected_result="-infinity",
        expected_required_display=("cos(pi / 2 + x) ≠ 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="trig_ratio",
        point_regime="finite_one_sided",
        domain_regime="one_sided_explicit_tangent_ratio_pole",
        required_condition_regime="finite_one_sided_explicit_tangent_ratio_pole_domain",
        trace_regime="finite_one_sided_trig_ratio_pole_policy",
        presentation_regime="negative_infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_cross_argument_explicit_trig_ratio_pole_supported",
        expr="limit(sin(x + pi/2)/cos(pi/2 - x), x, 0+)",
        expected_result="infinity",
        expected_required_display=("cos(pi / 2 - x) ≠ 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="trig_ratio",
        point_regime="finite_one_sided",
        domain_regime="one_sided_cross_argument_explicit_trig_ratio_pole",
        required_condition_regime="finite_one_sided_cross_argument_explicit_trig_ratio_pole_domain",
        trace_regime="finite_one_sided_trig_ratio_pole_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_scaled_cross_argument_trig_ratio_orientation_supported",
        expr="limit(sin(pi/2 - 2*x)/cos(pi/2 + 2*x), x, 0+)",
        expected_result="-infinity",
        expected_required_display=("cos(pi / 2 + 2·x) ≠ 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="trig_ratio",
        point_regime="finite_one_sided",
        domain_regime="one_sided_scaled_cross_argument_trig_ratio_orientation",
        required_condition_regime=(
            "finite_one_sided_scaled_cross_argument_trig_ratio_orientation_domain"
        ),
        trace_regime="finite_one_sided_trig_ratio_pole_policy",
        presentation_regime="negative_infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_symbolic_orientation_trig_ratio_pole_residual",
        expr="limit(sin(pi/2 + a*x)/cos(pi/2 - a*x), x, 0+)",
        expected_result="limit(sin(pi / 2 + a·x) / cos(pi / 2 - a·x), x, 0, right)",
        expected_required_display=("cos(pi / 2 - a·x) ≠ 0",),
        expected_warning_substrings=(
            "One-sided finite point limits are not supported safely",
        ),
        expected_step_substrings=("Conservar límite residual",),
        family="trig_ratio",
        point_regime="finite_one_sided",
        domain_regime="symbolic_orientation_trig_ratio_residual",
        required_condition_regime="finite_one_sided_symbolic_orientation_trig_ratio_domain",
        outcome="residual",
        residual_cause="finite_trig_symbolic_orientation_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_reciprocal_trig_sine_pole_residual",
        expr="limit(csc(x + pi), x, 0)",
        expected_result="undefined",
        expected_required_display=("sin(x + pi) ≠ 0",),
        expected_warning_substrings=("the bilateral limit does not exist",),
        expected_step_substrings=("undefined",),
        family="reciprocal_trig",
        point_regime="finite",
        domain_regime="table_undefined_trig_sine_pole_residual",
        required_condition_regime="finite_trig_sine_pole_residual_domain",
        outcome="undefined",
        trace_regime="finite_bilateral_dne_policy",
        presentation_regime="undefined",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_reciprocal_trig_sine_pole_supported",
        expr="limit(csc(x + pi), x, 0+)",
        expected_result="-infinity",
        expected_required_display=("sin(x + pi) ≠ 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="reciprocal_trig",
        point_regime="finite_one_sided",
        domain_regime="one_sided_reciprocal_trig_sine_pole",
        required_condition_regime="finite_one_sided_reciprocal_trig_sine_pole_domain",
        trace_regime="finite_one_sided_trig_pole_policy",
        presentation_regime="negative_infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_scaled_reciprocal_trig_sine_pole_supported",
        expr="limit(-2*csc(x + pi), x, 0+)",
        expected_result="infinity",
        expected_required_display=("sin(x + pi) ≠ 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="reciprocal_trig",
        point_regime="finite_one_sided",
        domain_regime="one_sided_scaled_reciprocal_trig_sine_pole",
        required_condition_regime="finite_one_sided_scaled_reciprocal_trig_sine_pole_domain",
        trace_regime="finite_one_sided_trig_pole_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_trig_small_angle_scaled_quotient",
        expr="limit(sin(2*x)/x, x, 0)",
        expected_result="2",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="trig_small_angle",
        point_regime="finite",
        domain_regime="small_angle_removable_quotient",
        required_condition_regime="finite_small_angle_denominator_domain",
        trace_regime="finite_small_angle_quotient_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_exp_zero_scaled_quotient",
        expr="limit((exp(2*x)-1)/x, x, 0)",
        expected_result="2",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="exp_zero_quotient",
        point_regime="finite",
        domain_regime="exp_zero_removable_quotient",
        required_condition_regime="finite_exp_zero_denominator_domain",
        trace_regime="finite_exp_zero_quotient_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_general_base_exp_zero_quotient",
        expr="limit((2^x-1)/x, x, 0)",
        expected_result="ln(2)",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="exp_zero_quotient",
        point_regime="finite",
        domain_regime="general_base_exp_zero_removable_quotient",
        required_condition_regime="finite_exp_zero_denominator_domain",
        trace_regime="finite_general_exp_zero_quotient_policy",
        presentation_regime="log_of_base",
    ),
    LimitCommandMatrixCase(
        name="finite_general_base_exp_zero_scaled_quotient",
        expr="limit((2^(3*x)-1)/x, x, 0)",
        expected_result="3·ln(2)",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="exp_zero_quotient",
        point_regime="finite",
        domain_regime="general_base_exp_zero_removable_quotient",
        required_condition_regime="finite_exp_zero_denominator_domain",
        trace_regime="finite_general_exp_zero_quotient_policy",
        presentation_regime="scaled_log_of_base",
    ),
    LimitCommandMatrixCase(
        name="finite_general_exp_ratio_log_quotient",
        expr="limit((3^x-1)/(2^x-1), x, 0)",
        expected_result="ln(3) / ln(2)",
        expected_required_display=("2^x - 1 ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="exp_zero_quotient",
        point_regime="finite",
        domain_regime="general_base_exp_ratio_removable_quotient",
        required_condition_regime="finite_exp_zero_denominator_domain",
        trace_regime="finite_general_exp_ratio_policy",
        presentation_regime="ratio_of_logs",
    ),
    LimitCommandMatrixCase(
        name="finite_general_exp_ratio_same_base_rational",
        expr="limit((2^(2*x)-1)/(2^x-1), x, 0)",
        expected_result="2",
        expected_required_display=("2^x - 1 ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="exp_zero_quotient",
        point_regime="finite",
        domain_regime="general_base_exp_ratio_removable_quotient",
        required_condition_regime="finite_exp_zero_denominator_domain",
        trace_regime="finite_general_exp_ratio_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_exp_combination_ratio_of_derivatives",
        expr="limit((2^x-3^x)/(5^x-7^x), x, 0)",
        expected_result="ln(2/3) / ln(5/7)",
        expected_required_display=("5^x - 7^x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="exp_zero_quotient",
        point_regime="finite",
        domain_regime="general_base_exp_ratio_removable_quotient",
        required_condition_regime="finite_exp_zero_denominator_domain",
        trace_regime="finite_exp_combination_ratio_policy",
        presentation_regime="ratio_of_log_combinations",
    ),
    LimitCommandMatrixCase(
        name="finite_general_exp_difference_quotient",
        expr="limit((2^x-3^x)/x, x, 0)",
        expected_result="ln(2/3)",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="general_exp_difference_quotient",
        point_regime="finite",
        domain_regime="general_base_exp_zero_removable_quotient",
        required_condition_regime="finite_exp_zero_denominator_domain",
        trace_regime="finite_exp_linear_combination_policy",
        presentation_regime="difference_of_logs",
    ),
    LimitCommandMatrixCase(
        name="finite_general_exp_scaled_difference_quotient",
        expr="limit((2^(3*x)-3^x)/x, x, 0)",
        expected_result="3·ln(2) - ln(3)",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="general_exp_difference_quotient",
        point_regime="finite",
        domain_regime="general_base_exp_zero_removable_quotient",
        required_condition_regime="finite_exp_zero_denominator_domain",
        trace_regime="finite_exp_linear_combination_policy",
        presentation_regime="scaled_difference_of_logs",
    ),
    LimitCommandMatrixCase(
        name="finite_higher_order_taylor_cosine_quotient",
        expr="limit((1-cos(x))/x^2, x, 0)",
        expected_result="1/2",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="higher_order_taylor_quotient",
        point_regime="finite",
        domain_regime="higher_order_taylor_removable_quotient",
        required_condition_regime="finite_taylor_quotient_denominator_domain",
        trace_regime="finite_taylor_quotient_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_higher_order_taylor_sine_cubic_quotient",
        expr="limit((sin(x)-x)/x^3, x, 0)",
        expected_result="-1/6",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="higher_order_taylor_quotient",
        point_regime="finite",
        domain_regime="higher_order_taylor_removable_quotient",
        required_condition_regime="finite_taylor_quotient_denominator_domain",
        trace_regime="finite_taylor_quotient_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_higher_order_taylor_arctan_cubic_quotient",
        expr="limit((arctan(x)-x)/x^3, x, 0)",
        expected_result="-1/3",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="higher_order_taylor_quotient",
        point_regime="finite",
        domain_regime="higher_order_taylor_removable_quotient",
        required_condition_regime="finite_taylor_quotient_denominator_domain",
        trace_regime="finite_taylor_quotient_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_log_unit_scaled_quotient",
        expr="limit(ln(1+2*x)/x, x, 0)",
        expected_result="2",
        expected_required_display=("x > -1/2", "x ≠ 0"),
        expected_step_substrings=("Evaluar límite finito",),
        family="log_unit_quotient",
        point_regime="finite",
        domain_regime="log_unit_removable_quotient",
        required_condition_regime="finite_log_unit_denominator_domain",
        trace_regime="finite_log_unit_quotient_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_fixed_base_log_unit_scaled_quotient",
        expr="limit(log2(1+2*x)/x, x, 0)",
        expected_result="2 / ln(2)",
        expected_required_display=("x > -1/2", "x ≠ 0"),
        expected_step_substrings=("Evaluar límite finito",),
        family="fixed_base_log_unit_quotient",
        point_regime="finite",
        domain_regime="fixed_base_log_unit_removable_quotient",
        required_condition_regime="finite_fixed_base_log_unit_denominator_domain",
        trace_regime="finite_fixed_base_log_unit_quotient_policy",
        presentation_regime="exact_log_base_factor",
    ),
    LimitCommandMatrixCase(
        name="finite_binary_log_constant_base_unit_scaled_quotient",
        expr="limit(log(3,1+2*x)/x, x, 0)",
        expected_result="2 / ln(3)",
        expected_required_display=("x > -1/2", "x ≠ 0"),
        expected_step_substrings=("Evaluar límite finito",),
        family="binary_log_constant_base_unit_quotient",
        point_regime="finite",
        domain_regime="binary_log_constant_base_unit_removable_quotient",
        required_condition_regime="finite_binary_log_constant_base_unit_denominator_domain",
        trace_regime="finite_binary_log_constant_base_unit_quotient_policy",
        presentation_regime="exact_log_base_factor",
    ),
    LimitCommandMatrixCase(
        name="finite_binary_log_variable_base_unit_scaled_quotient",
        expr="limit(log(x+1/4,1+2*x)/x, x, 0)",
        expected_result="2 / ln(1/4)",
        expected_required_display=("x > -1/4", "x ≠ 0", "x ≠ 3/4"),
        expected_step_substrings=("Evaluar límite finito",),
        family="binary_log_variable_base_unit_quotient",
        point_regime="finite",
        domain_regime="binary_log_variable_base_unit_removable_quotient",
        required_condition_regime="finite_binary_log_variable_base_unit_denominator_and_base_domain",
        trace_regime="finite_binary_log_variable_base_unit_quotient_policy",
        presentation_regime="exact_log_base_factor",
    ),
    LimitCommandMatrixCase(
        name="finite_binary_log_resolved_base_unit_scaled_quotient",
        expr="limit(log(exp(x)+2,1+2*x)/x, x, 0)",
        expected_result="2 / ln(3)",
        expected_required_display=("x > -1/2", "x ≠ 0"),
        expected_step_substrings=("Evaluar límite finito",),
        family="binary_log_variable_base_unit_quotient",
        point_regime="finite",
        domain_regime="binary_log_resolved_base_unit_removable_quotient",
        required_condition_regime="finite_binary_log_resolved_base_unit_denominator_and_base_domain",
        trace_regime="finite_binary_log_resolved_base_unit_quotient_policy",
        presentation_regime="exact_log_base_factor",
    ),
    LimitCommandMatrixCase(
        name="finite_binary_log_resolved_radical_base_unit_scaled_quotient",
        expr="limit(log(sqrt(x+4)+1,1+2*x)/x, x, 0)",
        expected_result="2 / ln(3)",
        expected_required_display=("x > -1/2", "x ≠ 0"),
        expected_step_substrings=("Evaluar límite finito",),
        family="binary_log_variable_base_unit_quotient",
        point_regime="finite",
        domain_regime="binary_log_resolved_radical_base_unit_removable_quotient",
        required_condition_regime="finite_binary_log_resolved_radical_base_unit_denominator_base_and_radical_domain",
        trace_regime="finite_binary_log_resolved_radical_base_unit_quotient_policy",
        presentation_regime="exact_log_base_factor",
    ),
    LimitCommandMatrixCase(
        name="finite_first_order_equiv_inversion_quotient",
        expr="limit(x/sin(x), x, 0)",
        expected_result="1",
        expected_required_display=("sin(x) ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="first_order_equivalent_infinitesimal",
        point_regime="finite",
        domain_regime="first_order_equivalent_quotient",
        required_condition_regime="finite_first_order_equiv_denominator_domain",
        trace_regime="finite_first_order_equiv_quotient_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_first_order_equiv_sin_sin_composition",
        expr="limit(sin(3*x)/sin(5*x), x, 0)",
        expected_result="3/5",
        expected_required_display=("sin(5·x) ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="first_order_equivalent_infinitesimal",
        point_regime="finite",
        domain_regime="first_order_equivalent_quotient",
        required_condition_regime="finite_first_order_equiv_denominator_domain",
        trace_regime="finite_first_order_equiv_quotient_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_first_order_equiv_sin_sin_half_composition",
        expr="limit(sin(x)/sin(2*x), x, 0)",
        expected_result="1/2",
        expected_required_display=("sin(2·x) ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="first_order_equivalent_infinitesimal",
        point_regime="finite",
        domain_regime="first_order_equivalent_quotient",
        required_condition_regime="finite_first_order_equiv_denominator_domain",
        trace_regime="finite_first_order_equiv_quotient_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_first_order_equiv_tan_quotient",
        expr="limit(tan(x)/x, x, 0)",
        expected_result="1",
        expected_required_display=("cos(x) ≠ 0", "x ≠ 0"),
        expected_step_substrings=("Evaluar límite finito",),
        family="first_order_equivalent_infinitesimal",
        point_regime="finite",
        domain_regime="first_order_equivalent_quotient",
        required_condition_regime="finite_first_order_equiv_denominator_domain",
        trace_regime="finite_first_order_equiv_quotient_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_first_order_equiv_asin_quotient",
        expr="limit(asin(x)/x, x, 0)",
        expected_result="1",
        expected_required_display=("-1 ≤ x ≤ 1", "x ≠ 0"),
        expected_step_substrings=("Evaluar límite finito",),
        family="first_order_equivalent_infinitesimal",
        point_regime="finite",
        domain_regime="first_order_equivalent_quotient",
        required_condition_regime="finite_first_order_equiv_denominator_domain",
        trace_regime="finite_first_order_equiv_quotient_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_first_order_equiv_arctan_quotient",
        expr="limit(arctan(x)/x, x, 0)",
        expected_result="1",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="first_order_equivalent_infinitesimal",
        point_regime="finite",
        domain_regime="first_order_equivalent_quotient",
        required_condition_regime="finite_first_order_equiv_denominator_domain",
        trace_regime="finite_first_order_equiv_quotient_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_first_order_equiv_sinh_quotient",
        expr="limit(sinh(x)/x, x, 0)",
        expected_result="1",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="first_order_equivalent_infinitesimal",
        point_regime="finite",
        domain_regime="first_order_equivalent_quotient",
        required_condition_regime="finite_first_order_equiv_denominator_domain",
        trace_regime="finite_first_order_equiv_quotient_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_first_order_equiv_tanh_quotient",
        expr="limit(tanh(x)/x, x, 0)",
        expected_result="1",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="first_order_equivalent_infinitesimal",
        point_regime="finite",
        domain_regime="first_order_equivalent_quotient",
        required_condition_regime="finite_first_order_equiv_denominator_domain",
        trace_regime="finite_first_order_equiv_quotient_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_rational_equal_degree",
        expr="limit((x^2+1)/(2*x^2-3), x, infinity)",
        expected_result="1/2",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="rational",
        point_regime="infinity",
        domain_regime="required_condition",
        required_condition_regime="infinity_source_definedness",
        trace_regime="rational_degree_policy",
        presentation_regime="infinity_tail_polynomial_condition_suppression",
    ),
    LimitCommandMatrixCase(
        name="infinity_bounded_noise_equal_degree_ratio",
        expr="limit((x+sin(x))/x, x, infinity)",
        expected_result="1",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="rational",
        point_regime="infinity",
        domain_regime="bounded_additive_noise",
        required_condition_regime="none",
        trace_regime="bounded_noise_rational_degree_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_bounded_noise_higher_degree_diverges",
        expr="limit((x^2+sin(x))/x, x, infinity)",
        expected_result="infinity",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="rational",
        point_regime="infinity",
        domain_regime="bounded_additive_noise",
        required_condition_regime="none",
        trace_regime="bounded_noise_rational_degree_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="infinity_log_difference_finite_ratio",
        expr="limit(ln(2*x)-ln(x), x, infinity)",
        expected_result="ln(2)",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="logarithmic_growth",
        point_regime="infinity",
        domain_regime="log_difference_ratio",
        required_condition_regime="none",
        trace_regime="log_difference_ratio_policy",
        presentation_regime="log_of_leading_ratio",
    ),
    LimitCommandMatrixCase(
        name="infinity_log_difference_diverges",
        expr="limit(ln(x^2)-ln(x), x, infinity)",
        expected_result="infinity",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="logarithmic_growth",
        point_regime="infinity",
        domain_regime="log_difference_ratio",
        required_condition_regime="none",
        trace_regime="log_difference_ratio_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="infinity_log_exp_sum_dominant_base",
        expr="limit(ln(2^x+3^x)/x, x, infinity)",
        expected_result="ln(3)",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="logarithmic_growth",
        point_regime="infinity",
        domain_regime="log_exp_sum_dominance",
        required_condition_regime="none",
        trace_regime="log_exp_sum_dominance_policy",
        presentation_regime="log_of_dominant_base",
    ),
    LimitCommandMatrixCase(
        name="infinity_log_exp_sum_three_terms",
        expr="limit(ln(2^x+3^x+5^x)/x, x, infinity)",
        expected_result="ln(5)",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="logarithmic_growth",
        point_regime="infinity",
        domain_regime="log_exp_sum_dominance",
        required_condition_regime="none",
        trace_regime="log_exp_sum_dominance_policy",
        presentation_regime="log_of_dominant_base",
    ),
    LimitCommandMatrixCase(
        name="infinity_rational_difference_finite_value",
        expr="limit((x^2+1)/(x+1) - x, x, infinity)",
        expected_result="-1",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="rational",
        point_regime="infinity",
        domain_regime="rational_common_denominator",
        required_condition_regime="none",
        trace_regime="rational_difference_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_rational_difference_of_quotients",
        expr="limit(x^2/(x+1) - x^2/(x+2), x, infinity)",
        expected_result="1",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="rational",
        point_regime="infinity",
        domain_regime="rational_common_denominator",
        required_condition_regime="none",
        trace_regime="rational_difference_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_one_to_infinity_power_e",
        expr="limit((1+1/x)^x, x, infinity)",
        expected_result="e",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="exponential_indeterminate",
        point_regime="infinity",
        domain_regime="one_to_infinity_unit_base",
        required_condition_regime="none",
        trace_regime="one_to_infinity_power_policy",
        presentation_regime="euler_constant",
    ),
    LimitCommandMatrixCase(
        name="infinity_one_to_infinity_power_scaled",
        expr="limit((1+2/x)^x, x, infinity)",
        expected_result="e^2",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="exponential_indeterminate",
        point_regime="infinity",
        domain_regime="one_to_infinity_unit_base",
        required_condition_regime="none",
        trace_regime="one_to_infinity_power_policy",
        presentation_regime="euler_power",
    ),
    LimitCommandMatrixCase(
        name="infinity_general_base_exponential_growth",
        expr="limit(2^x, x, infinity)",
        expected_result="infinity",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="exponential_growth",
        point_regime="infinity",
        domain_regime="total_real_function",
        required_condition_regime="none",
        trace_regime="general_base_exponential_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="infinity_inf_to_zero_power_dominant_base",
        expr="limit((2^x+3^x)^(1/x), x, infinity)",
        expected_result="3",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="exponential_indeterminate",
        point_regime="infinity",
        domain_regime="inf_to_zero_divergent_base",
        required_condition_regime="none",
        trace_regime="inf_to_zero_power_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_exp_quotient_dominance_diverges",
        expr="limit(3^x/2^x, x, infinity)",
        expected_result="infinity",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="exponential_growth",
        point_regime="infinity",
        domain_regime="exponential_dominance",
        required_condition_regime="none",
        trace_regime="exp_sum_quotient_dominance_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="infinity_exp_quotient_dominance_finite_ratio",
        expr="limit((2^x+3^x)/3^x, x, infinity)",
        expected_result="1",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="exponential_growth",
        point_regime="infinity",
        domain_regime="exponential_dominance",
        required_condition_regime="none",
        trace_regime="exp_sum_quotient_dominance_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_exponential_beats_polynomial",
        expr="limit(2^x/x^2, x, infinity)",
        expected_result="infinity",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="exponential_growth",
        point_regime="infinity",
        domain_regime="exponential_dominance",
        required_condition_regime="none",
        trace_regime="general_exp_polynomial_dominance_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="infinity_polynomial_over_exponential_decays",
        expr="limit(x^10/2^x, x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="exponential_growth",
        point_regime="infinity",
        domain_regime="exponential_dominance",
        required_condition_regime="none",
        trace_regime="general_exp_polynomial_dominance_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_polynomial_times_decaying_exponential",
        expr="limit(x*2^(-x), x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="exponential_growth",
        point_regime="infinity",
        domain_regime="exponential_dominance",
        required_condition_regime="none",
        trace_regime="polynomial_times_decaying_exponential_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_one_to_infinity_power_e_definition",
        expr="limit((1+x)^(1/x), x, 0)",
        expected_result="e",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="exponential_indeterminate",
        point_regime="finite",
        domain_regime="one_to_infinity_unit_base",
        required_condition_regime="finite_source_definedness",
        trace_regime="one_to_infinity_power_policy",
        presentation_regime="euler_constant",
    ),
    LimitCommandMatrixCase(
        name="finite_one_to_infinity_power_second_order",
        expr="limit(cos(x)^(1/x^2), x, 0)",
        expected_result="1 / sqrt(e)",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="exponential_indeterminate",
        point_regime="finite",
        domain_regime="one_to_infinity_unit_base",
        required_condition_regime="finite_source_definedness",
        trace_regime="one_to_infinity_power_policy",
        presentation_regime="euler_power",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_zero_base_power_x_to_x",
        expr="limit(x^x, x, 0+)",
        expected_result="1",
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="exponential_indeterminate",
        point_regime="finite_one_sided",
        domain_regime="zero_to_zero_positive_base",
        required_condition_regime="none",
        trace_regime="zero_base_power_policy",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_root_polynomial_tail_domain_suppression",
        expr="limit(sqrt(x^2-3)/x, x, infinity)",
        expected_result="1",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="eventual_polynomial_domain",
        required_condition_regime="infinity_path_compatible_polynomial_domain",
        trace_regime="root_polynomial_tail_policy",
        presentation_regime="infinity_tail_polynomial_sign_condition_suppression",
    ),
    LimitCommandMatrixCase(
        name="infinity_radical_conjugate_product_quadratic_decay",
        expr="limit(x*(sqrt(x^2+1)-x), x, infinity)",
        expected_result="1/2",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="unconditional",
        required_condition_regime="none",
        trace_regime="radical_conjugate_product_policy",
        presentation_regime="infinity_radical_conjugate_product_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_radical_conjugate_product_shifted_linear_tail",
        expr="limit(x*(sqrt(x^2+2*x)-x-1), x, infinity)",
        expected_result="-1/2",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="unconditional",
        required_condition_regime="none",
        trace_regime="radical_conjugate_product_policy",
        presentation_regime="infinity_radical_conjugate_product_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_radical_conjugate_product_sqrt_factor_root_difference",
        expr="limit(sqrt(x)*(sqrt(x+1)-sqrt(x)), x, infinity)",
        expected_result="1/2",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="unconditional",
        required_condition_regime="none",
        trace_regime="radical_conjugate_product_policy",
        presentation_regime="infinity_radical_conjugate_product_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_radical_conjugate_product_quadratic_root_difference",
        expr="limit(x*(sqrt(x^2+x+1)-sqrt(x^2+x)), x, infinity)",
        expected_result="1/2",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="unconditional",
        required_condition_regime="none",
        trace_regime="radical_conjugate_product_policy",
        presentation_regime="infinity_radical_conjugate_product_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_cbrt_conjugate_bare_difference_cubic_tail",
        expr="limit(cbrt(x^3+x^2)-x, x, infinity)",
        expected_result="1/3",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="unconditional",
        required_condition_regime="none",
        trace_regime="cbrt_conjugate_policy",
        presentation_regime="infinity_cbrt_conjugate_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_cbrt_conjugate_bare_difference_scaled_tail",
        expr="limit(cbrt(x^3+2*x^2)-x, x, infinity)",
        expected_result="2/3",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="unconditional",
        required_condition_regime="none",
        trace_regime="cbrt_conjugate_policy",
        presentation_regime="infinity_cbrt_conjugate_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_cbrt_conjugate_zero_times_infinity_product",
        expr="limit(x^2*(cbrt(x^3+1)-x), x, infinity)",
        expected_result="1/3",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="unconditional",
        required_condition_regime="none",
        trace_regime="cbrt_conjugate_policy",
        presentation_regime="infinity_cbrt_conjugate_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_nth_root_conjugate_fourth_root_tail",
        expr="limit((x^4+x^3)^(1/4)-x, x, infinity)",
        expected_result="1/4",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="unconditional",
        required_condition_regime="none",
        trace_regime="nth_root_conjugate_policy",
        presentation_regime="infinity_nth_root_conjugate_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_nth_root_conjugate_fifth_root_tail",
        expr="limit((x^5+x^4)^(1/5)-x, x, infinity)",
        expected_result="1/5",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="unconditional",
        required_condition_regime="none",
        trace_regime="nth_root_conjugate_policy",
        presentation_regime="infinity_nth_root_conjugate_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_nth_root_conjugate_zero_times_infinity_product",
        expr="limit(x^3*((x^4+1)^(1/4)-x), x, infinity)",
        expected_result="1/4",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="unconditional",
        required_condition_regime="none",
        trace_regime="nth_root_conjugate_policy",
        presentation_regime="infinity_nth_root_conjugate_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_cbrt_polynomial_signed_tail_standalone",
        expr="limit(cbrt(2 - x^4), x, infinity)",
        expected_result="-infinity",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="total_real_polynomial_tail_domain",
        required_condition_regime="infinity_path_total_real_polynomial_tail_domain",
        trace_regime="cbrt_polynomial_signed_tail_policy",
        presentation_regime="infinity_tail_polynomial_signed_cube_root",
    ),
    LimitCommandMatrixCase(
        name="infinity_asinh_polynomial_signed_tail_standalone",
        expr="limit(asinh(2 - x^4), x, infinity)",
        expected_result="-infinity",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="inverse_hyperbolic",
        point_regime="infinity",
        domain_regime="total_real_polynomial_tail_domain",
        required_condition_regime="infinity_path_total_real_polynomial_tail_domain",
        trace_regime="asinh_polynomial_signed_tail_policy",
        presentation_regime="infinity_tail_polynomial_signed_inverse_hyperbolic",
    ),
    LimitCommandMatrixCase(
        name="infinity_atan_polynomial_bounded_tail_standalone",
        expr="limit(atan(2 - x^4), x, infinity)",
        expected_result="-1/2·pi",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="inverse_trig",
        point_regime="infinity",
        domain_regime="total_real_polynomial_tail_domain",
        required_condition_regime="infinity_path_total_real_polynomial_tail_domain",
        trace_regime="bounded_total_real_polynomial_tail_policy",
        presentation_regime="infinity_tail_polynomial_bounded_inverse_trig",
    ),
    LimitCommandMatrixCase(
        name="infinity_atan_sqrt_unbounded_tail_standalone",
        expr="limit(arctan(sqrt(x)), x, infinity)",
        expected_result="1/2·pi",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="inverse_trig",
        point_regime="infinity",
        domain_regime="radical_unbounded_tail_domain",
        required_condition_regime="infinity_path_radical_unbounded_tail_domain",
        trace_regime="bounded_radical_unbounded_tail_policy",
        presentation_regime="infinity_tail_radical_bounded_inverse_trig",
    ),
    LimitCommandMatrixCase(
        name="infinity_exp_negated_sqrt_decay_standalone",
        expr="limit(e^(-sqrt(x)), x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="exponential",
        point_regime="infinity",
        domain_regime="radical_unbounded_tail_domain",
        required_condition_regime="infinity_path_radical_unbounded_tail_domain",
        trace_regime="exponential_radical_negative_tail_policy",
        presentation_regime="infinity_tail_radical_exponential_decay",
    ),
    LimitCommandMatrixCase(
        name="infinity_scaled_arctan_symbolic_product_standalone",
        expr="limit(2*arctan(x), x, infinity)",
        expected_result="pi",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="inverse_trig",
        point_regime="infinity",
        domain_regime="total_real_function",
        required_condition_regime="infinity_path_total_real_function",
        trace_regime="multiplicative_symbolic_finite_composition_policy",
        presentation_regime="symbolic_finite_product_unfolded",
    ),
    LimitCommandMatrixCase(
        name="infinity_sinh_polynomial_signed_tail_standalone",
        expr="limit(sinh(2 - x^4), x, infinity)",
        expected_result="-infinity",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="hyperbolic",
        point_regime="infinity",
        domain_regime="total_real_polynomial_tail_domain",
        required_condition_regime="infinity_path_total_real_polynomial_tail_domain",
        trace_regime="hyperbolic_polynomial_signed_tail_policy",
        presentation_regime="infinity_tail_polynomial_signed_hyperbolic_sine",
    ),
    LimitCommandMatrixCase(
        name="infinity_exp_polynomial_signed_tail_standalone",
        expr="limit(exp(2 - x^4), x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="exponential",
        point_regime="infinity",
        domain_regime="total_real_polynomial_tail_domain",
        required_condition_regime="infinity_path_total_real_polynomial_tail_domain",
        trace_regime="exponential_polynomial_signed_tail_policy",
        presentation_regime="infinity_tail_polynomial_exponential_decay",
    ),
    LimitCommandMatrixCase(
        name="infinity_exp_polynomial_dominance_decay",
        expr="limit(x^2*exp(2 - x^4), x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="exponential_growth",
        point_regime="infinity",
        domain_regime="total_real_polynomial_exponential_dominance",
        trace_regime="exponential_polynomial_dominance_policy",
        presentation_regime="infinity_tail_polynomial_exponential_dominance_decay",
    ),
    LimitCommandMatrixCase(
        name="infinity_exp_polynomial_subpolynomial_dominance_decay",
        expr="limit(ln(x)/exp(x^2), x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="exponential_growth",
        point_regime="infinity",
        domain_regime="eventual_positive_log_exponential_dominance",
        required_condition_regime="infinity_path_compatible_domain",
        trace_regime="exponential_polynomial_subpolynomial_dominance_policy",
        presentation_regime="infinity_tail_polynomial_exponential_subpolynomial_decay",
    ),
    LimitCommandMatrixCase(
        name="infinity_root_rational_finite_tail_standalone",
        expr="limit(sqrt((2*x^2 + 1)/(x^2 + 1)), x, infinity)",
        expected_result="sqrt(2)",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="eventual_positive_rational_finite_tail_domain",
        required_condition_regime="infinity_path_compatible_rational_finite_tail_domain",
        trace_regime="root_rational_finite_tail_policy",
        presentation_regime="infinity_tail_rational_finite_positive_radical",
    ),
    LimitCommandMatrixCase(
        name="infinity_root_rational_zero_tail_standalone",
        expr="limit(sqrt((x + 1)/(x^2 + 1)), x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="eventual_positive_rational_zero_tail_domain",
        required_condition_regime="infinity_path_compatible_rational_zero_tail_domain",
        trace_regime="root_rational_zero_tail_policy",
        presentation_regime="infinity_tail_rational_zero_positive_radical",
    ),
    LimitCommandMatrixCase(
        name="infinity_cbrt_rational_signed_finite_tail_standalone",
        expr="limit(cbrt((1 - 8*x^2)/(x^2 + 1)), x, infinity)",
        expected_result="-2",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="root",
        point_regime="infinity",
        domain_regime="total_real_rational_finite_tail_domain",
        required_condition_regime="infinity_path_total_real_rational_finite_tail_domain",
        trace_regime="cbrt_rational_signed_tail_policy",
        presentation_regime="infinity_tail_rational_signed_cube_root",
    ),
    LimitCommandMatrixCase(
        name="negative_infinity_polynomial_parity",
        expr="limit(x - 2*x^3, x, -infinity)",
        expected_result="infinity",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="polynomial",
        point_regime="infinity",
        trace_regime="polynomial_leading_term_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="infinity_exponential_decay",
        expr="limit(exp(-x), x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="exponential",
        point_regime="infinity",
        trace_regime="exponential_tail_policy",
    ),
    LimitCommandMatrixCase(
        name="infinity_bounded_polynomial_exp_decay",
        expr="limit(sin(x)*exp(2 - x^4), x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="bounded_exponential_decay",
        point_regime="infinity",
        domain_regime="total_real_bounded_polynomial_exponential_decay",
        trace_regime="bounded_polynomial_exponential_decay_policy",
        presentation_regime="infinity_tail_bounded_polynomial_exponential_decay",
    ),
    LimitCommandMatrixCase(
        name="negative_infinity_log_domain_conflict_residual",
        expr="limit(ln(x), x, -infinity)",
        expected_result="limit(ln(x), x, -infinity)",
        expected_required_display=("x > 0",),
        expected_warning_substrings=(
            "Could not determine limit safely",
            "Limit path conflicts with the input domain",
        ),
        expected_step_substrings=("Conservar límite residual",),
        family="log",
        point_regime="infinity",
        domain_regime="domain_path_conflict",
        required_condition_regime="infinity_path_conflict",
        outcome="residual",
        residual_cause="infinity_domain_path_conflict",
        trace_regime="infinity_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="negative_infinity_log_domain_compatible_growth_supported",
        expr="limit(x/ln(1-x), x, -infinity)",
        expected_result="-infinity",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="logarithmic_growth",
        point_regime="infinity",
        domain_regime="negative_infinity_log_domain_required",
        required_condition_regime="infinity_path_compatible_domain",
        trace_regime="logarithmic_growth_negative_infinity_policy",
        presentation_regime="negative_infinity_tail_condition_suppression",
    ),
    LimitCommandMatrixCase(
        name="infinity_log_polynomial_argument_standalone",
        expr="limit(ln(x^2 - 3), x, infinity)",
        expected_result="infinity",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="log",
        point_regime="infinity",
        domain_regime="eventual_positive_polynomial_domain",
        required_condition_regime="infinity_path_compatible_polynomial_positive_domain",
        trace_regime="log_polynomial_argument_tail_policy",
        presentation_regime="infinity_tail_polynomial_positive_condition_suppression",
    ),
    LimitCommandMatrixCase(
        name="infinity_log_polynomial_argument_dominance",
        expr="limit(ln(x^2 - 3)/x, x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="logarithmic_growth",
        point_regime="infinity",
        domain_regime="eventual_positive_polynomial_domain",
        required_condition_regime="infinity_path_compatible_polynomial_positive_domain",
        trace_regime="logarithmic_growth_polynomial_argument_policy",
        presentation_regime="infinity_tail_polynomial_positive_condition_suppression",
    ),
    LimitCommandMatrixCase(
        name="infinity_log_rational_argument_dominance",
        expr="limit(ln((x^2 + 1)/(x + 1))/x, x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="logarithmic_growth",
        point_regime="infinity",
        domain_regime="eventual_positive_rational_domain",
        required_condition_regime="infinity_path_compatible_rational_positive_domain",
        trace_regime="logarithmic_growth_rational_argument_policy",
        presentation_regime="infinity_tail_rational_positive_condition_suppression",
    ),
    LimitCommandMatrixCase(
        name="infinity_log_rational_zero_tail_dominance",
        expr="limit(ln((x + 1)/(x^2 + 1))/x, x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="logarithmic_growth",
        point_regime="infinity",
        domain_regime="eventual_positive_rational_zero_tail_domain",
        required_condition_regime="infinity_path_compatible_rational_zero_tail_domain",
        trace_regime="logarithmic_growth_rational_zero_tail_policy",
        presentation_regime="infinity_tail_rational_zero_positive_condition_suppression",
    ),
    LimitCommandMatrixCase(
        name="infinity_polylog_fractional_power_dominance",
        expr="limit(ln(x)/sqrt(x), x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="logarithmic_growth",
        point_regime="infinity",
        domain_regime="polylog_over_positive_power",
        required_condition_regime="infinity_path_compatible_polynomial_positive_domain",
        trace_regime="polylog_power_dominance_policy",
        presentation_regime="zero",
    ),
    LimitCommandMatrixCase(
        name="infinity_polylog_higher_power_dominance",
        expr="limit(ln(x)^2/x, x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="logarithmic_growth",
        point_regime="infinity",
        domain_regime="polylog_over_positive_power",
        required_condition_regime="infinity_path_compatible_polynomial_positive_domain",
        trace_regime="polylog_power_dominance_policy",
        presentation_regime="zero",
    ),
    LimitCommandMatrixCase(
        name="infinity_power_over_polylog_diverges",
        expr="limit(sqrt(x)/ln(x), x, infinity)",
        expected_result="infinity",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="logarithmic_growth",
        point_regime="infinity",
        domain_regime="positive_power_over_polylog",
        required_condition_regime="infinity_path_compatible_polynomial_positive_domain",
        trace_regime="polylog_power_dominance_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="infinity_fractional_power_negative_exponent_decays",
        expr="limit(x^(-1/2), x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="fractional_power_growth",
        point_regime="infinity",
        trace_regime="fractional_power_infinity_tail_policy",
        presentation_regime="zero",
    ),
    LimitCommandMatrixCase(
        name="infinity_fractional_power_positive_exponent_diverges",
        expr="limit(x^(3/2), x, infinity)",
        expected_result="infinity",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="fractional_power_growth",
        point_regime="infinity",
        trace_regime="fractional_power_infinity_tail_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="infinity_fractional_power_odd_denominator_diverges",
        expr="limit(x^(2/3), x, infinity)",
        expected_result="infinity",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="fractional_power_growth",
        point_regime="infinity",
        trace_regime="fractional_power_infinity_tail_policy",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="infinity_log_rational_finite_tail_standalone",
        expr="limit(log(2, (2*x^2 + 1)/(x^2 + 1)), x, infinity)",
        expected_result="1",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="log",
        point_regime="infinity",
        domain_regime="eventual_positive_rational_finite_tail_domain",
        required_condition_regime="infinity_path_compatible_rational_finite_tail_domain",
        trace_regime="log_rational_finite_tail_policy",
        presentation_regime="infinity_tail_rational_finite_positive_condition_suppression",
    ),
    LimitCommandMatrixCase(
        name="infinity_acosh_rational_finite_tail_standalone",
        expr="limit(acosh((2*x^2 + 1)/(x^2 + 1)), x, infinity)",
        expected_result="acosh(2)",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="inverse_hyperbolic",
        point_regime="infinity",
        domain_regime="eventual_lower_bound_rational_finite_tail_domain",
        required_condition_regime="infinity_path_compatible_rational_lower_bound_finite_tail_domain",
        trace_regime="inverse_hyperbolic_rational_finite_tail_policy",
        presentation_regime="infinity_tail_rational_lower_bound_condition_suppression",
    ),
    LimitCommandMatrixCase(
        name="infinity_acosh_polynomial_argument_dominance",
        expr="limit(acosh(x^2 - 3)/x, x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="inverse_hyperbolic_growth",
        point_regime="infinity",
        domain_regime="eventual_positive_polynomial_domain",
        required_condition_regime="infinity_path_compatible_polynomial_positive_domain",
        trace_regime="inverse_hyperbolic_growth_polynomial_argument_policy",
        presentation_regime="infinity_tail_polynomial_positive_condition_suppression",
    ),
    LimitCommandMatrixCase(
        name="infinity_bounded_cross_family_over_divergent",
        expr="limit((sin(x)+cos(x)*arctan(x))/(x^2+1), x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="cross_family_bounded",
        point_regime="infinity",
        trace_regime="bounded_over_divergent_policy",
    ),
    LimitCommandMatrixCase(
        name="negative_infinity_bounded_over_divergent_orientation_supported",
        expr="limit(sin(sqrt(-x))/x, x, -infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="bounded_orientation_policy",
        point_regime="infinity",
        domain_regime="negative_infinity_bounded_composed_domain_required",
        required_condition_regime="infinity_path_compatible_domain",
        trace_regime="bounded_over_divergent_policy",
        presentation_regime="negative_infinity_tail_bounded_condition_suppression",
    ),
    LimitCommandMatrixCase(
        name="finite_saturating_exp_decay_to_zero",
        expr="limit(e^(-1/x^2), x, 0)",
        expected_result="0",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="saturating_composition",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_source_definedness",
        trace_regime="saturating_function_at_infinity",
        presentation_regime="zero",
    ),
    LimitCommandMatrixCase(
        name="finite_saturating_exp_growth_to_infinity",
        expr="limit(e^(1/x^2), x, 0)",
        expected_result="infinity",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="saturating_composition",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_source_definedness",
        trace_regime="saturating_function_at_infinity",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_saturating_arctan_to_half_pi",
        expr="limit(atan(1/x^2), x, 0)",
        expected_result="1/2·pi",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="saturating_composition",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_source_definedness",
        trace_regime="saturating_function_at_infinity",
        presentation_regime="half_pi",
    ),
    LimitCommandMatrixCase(
        name="finite_saturating_tanh_to_one",
        expr="limit(tanh(1/x^2), x, 0)",
        expected_result="1",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="saturating_composition",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_source_definedness",
        trace_regime="saturating_function_at_infinity",
        presentation_regime="one",
    ),
    LimitCommandMatrixCase(
        name="infinity_sqrt_quadratic_minus_linear_half",
        expr="limit(sqrt(x^2+x)-x, x, infinity)",
        expected_result="1/2",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="radical_difference_conjugate",
        point_regime="infinity",
        domain_regime="total_real_function",
        trace_regime="sqrt_quadratic_leading_cancellation",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_sqrt_quadratic_minus_linear_zero",
        expr="limit(sqrt(x^2+1)-x, x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="radical_difference_conjugate",
        point_regime="infinity",
        domain_regime="total_real_function",
        trace_regime="sqrt_quadratic_leading_cancellation",
        presentation_regime="zero",
    ),
    LimitCommandMatrixCase(
        name="infinity_sqrt_scaled_quadratic_minus_linear",
        expr="limit(sqrt(4*x^2+x)-2*x, x, infinity)",
        expected_result="1/4",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="radical_difference_conjugate",
        point_regime="infinity",
        domain_regime="total_real_function",
        trace_regime="sqrt_quadratic_leading_cancellation",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_linear_minus_sqrt_quadratic",
        expr="limit(x-sqrt(x^2-x), x, infinity)",
        expected_result="1/2",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="radical_difference_conjugate",
        point_regime="infinity",
        domain_regime="total_real_function",
        trace_regime="sqrt_quadratic_leading_cancellation",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_sqrt_linear_difference_zero",
        expr="limit(sqrt(x+1)-sqrt(x), x, infinity)",
        expected_result="0",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="radical_difference_conjugate",
        point_regime="infinity",
        domain_regime="total_real_function",
        trace_regime="sqrt_minus_sqrt_leading_cancellation",
        presentation_regime="zero",
    ),
    LimitCommandMatrixCase(
        name="infinity_sqrt_quadratic_difference_unit",
        expr="limit(sqrt(x^2+x)-sqrt(x^2-x), x, infinity)",
        expected_result="1",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="radical_difference_conjugate",
        point_regime="infinity",
        domain_regime="total_real_function",
        trace_regime="sqrt_minus_sqrt_leading_cancellation",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="infinity_sqrt_scaled_quadratic_difference_half",
        expr="limit(sqrt(4*x^2+x)-sqrt(4*x^2-x), x, infinity)",
        expected_result="1/2",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="radical_difference_conjugate",
        point_regime="infinity",
        domain_regime="total_real_function",
        trace_regime="sqrt_minus_sqrt_leading_cancellation",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_radical_conjugate_removable_quarter",
        expr="limit((sqrt(x)-2)/(x-4), x, 4)",
        expected_result="1/4",
        expected_required_display=("x ≠ 4", "x ≥ 0"),
        expected_step_substrings=("Evaluar límite finito",),
        family="radical_conjugate_removable",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_removable_radical_hole",
        trace_regime="radical_conjugate_cancellation",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_radical_conjugate_removable_sixth",
        expr="limit((sqrt(x)-3)/(x-9), x, 9)",
        expected_result="1/6",
        expected_required_display=("x ≠ 9", "x ≥ 0"),
        expected_step_substrings=("Evaluar límite finito",),
        family="radical_conjugate_removable",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_removable_radical_hole",
        trace_regime="radical_conjugate_cancellation",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_radical_conjugate_affine_radicand",
        expr="limit((sqrt(2*x+1)-3)/(x-4), x, 4)",
        expected_result="1/3",
        expected_required_display=("x ≠ 4", "x ≥ -1/2"),
        expected_step_substrings=("Evaluar límite finito",),
        family="radical_conjugate_removable",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_removable_radical_hole",
        trace_regime="radical_conjugate_cancellation",
        presentation_regime="exact_rational",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_exp_reciprocal_diverges",
        expr="limit(e^(1/x), x, 0, +)",
        expected_result="infinity",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="one_sided_saturating_composition",
        point_regime="finite_one_sided",
        domain_regime="required_condition",
        required_condition_regime="finite_source_definedness",
        trace_regime="one_sided_saturation_at_infinity",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_exp_reciprocal_decays",
        expr="limit(e^(1/x), x, 0, -)",
        expected_result="0",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="one_sided_saturating_composition",
        point_regime="finite_one_sided",
        domain_regime="required_condition",
        required_condition_regime="finite_source_definedness",
        trace_regime="one_sided_saturation_at_infinity",
        presentation_regime="zero",
    ),
    LimitCommandMatrixCase(
        name="finite_one_sided_arctan_reciprocal_half_pi",
        expr="limit(atan(1/x), x, 0, +)",
        expected_result="1/2·pi",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite unilateral finito",),
        family="one_sided_saturating_composition",
        point_regime="finite_one_sided",
        domain_regime="required_condition",
        required_condition_regime="finite_source_definedness",
        trace_regime="one_sided_saturation_at_infinity",
        presentation_regime="half_pi",
    ),
    LimitCommandMatrixCase(
        name="finite_squeeze_bounded_oscillator_zero",
        expr="limit(x*sin(1/x), x, 0)",
        expected_result="0",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="finite_squeeze_bounded_product",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_source_definedness",
        trace_regime="squeeze_bounded_product",
        presentation_regime="zero",
    ),
    LimitCommandMatrixCase(
        name="finite_squeeze_even_power_cosine_zero",
        expr="limit(x^2*cos(1/x), x, 0)",
        expected_result="0",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="finite_squeeze_bounded_product",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_source_definedness",
        trace_regime="squeeze_bounded_product",
        presentation_regime="zero",
    ),
    LimitCommandMatrixCase(
        name="finite_squeeze_scaled_oscillator_residual",
        expr="limit(2*sin(1/x), x, 0)",
        expected_result="limit(2·sin(1 / x), x, 0)",
        expected_required_display=("x ≠ 0",),
        expected_warning_substrings=("Finite point limits are not supported safely yet",),
        expected_step_substrings=("Conservar límite residual",),
        family="finite_squeeze_bounded_product",
        point_regime="finite",
        domain_regime="oscillatory_no_limit_residual",
        required_condition_regime="finite_source_definedness",
        outcome="residual",
        residual_cause="finite_oscillatory_no_infinitesimal",
        trace_regime="finite_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_even_cosh_reciprocal_pole_bilateral_supported",
        expr="limit(cosh(1/x), x, 0)",
        expected_result="infinity",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="even_saturating_composition",
        point_regime="finite",
        domain_regime="bilateral_even_cosh_reciprocal_pole",
        required_condition_regime="finite_source_definedness",
        trace_regime="bilateral_even_cosh_saturation",
        presentation_regime="infinity",
    ),
    LimitCommandMatrixCase(
        # GRADUADO 2026-07-18 (tanda-3, oscilación): la no-existencia ahora se
        # PRUEBA desde la divergencia del argumento interior — undefined con
        # motivo de oscilación, en vez del residual genérico que este caso
        # pineaba como mejor-answer-de-entonces.
        name="finite_oscillating_outer_even_pole_residual",
        expr="limit(cos(1/x^2), x, 0)",
        expected_result="undefined",
        expected_required_display=("x ≠ 0",),
        expected_warning_substrings=("OSCILLATES",),
        expected_step_substrings=("Conservar límite residual",),
        family="even_saturating_composition",
        point_regime="finite",
        domain_regime="oscillatory_no_limit_dne",
        required_condition_regime="finite_source_definedness",
        outcome="undefined",
        trace_regime="finite_residual_policy",
        presentation_regime="undefined",
    ),
    LimitCommandMatrixCase(
        name="infinity_bounded_over_divergent_domain_conflict_residual",
        expr="limit(sin(sqrt(1 - x))/x, x, infinity)",
        expected_result="limit(sin(sqrt(1 - x)) / x, x, infinity)",
        expected_required_display=("x ≠ 0", "x ≤ 1"),
        expected_warning_substrings=(
            "Could not determine limit safely",
            "Limit path conflicts with the input domain",
        ),
        expected_step_substrings=("Conservar límite residual",),
        family="bounded_domain_policy",
        point_regime="infinity",
        domain_regime="domain_path_conflict",
        required_condition_regime="infinity_path_conflict",
        outcome="residual",
        residual_cause="infinity_domain_path_conflict",
        trace_regime="infinity_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_lhopital_sine_shifted_zero_supported",
        expr="limit(sin(x)/(x-pi), x, pi)",
        expected_result="-1",
        expected_required_display=("x - pi ≠ 0",),
        expected_step_substrings=("Evaluar límite finito",),
        family="trig_lhopital",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_lhopital_nonzero_point_denominator_domain",
        trace_regime="lhopital_nonzero_point",
    ),
    LimitCommandMatrixCase(
        name="finite_lhopital_tangent_sine_ratio_supported",
        expr="limit(tan(x)/sin(x), x, pi)",
        expected_result="-1",
        expected_required_display=("cos(x) ≠ 0", "sin(x) ≠ 0"),
        expected_step_substrings=("Evaluar límite finito",),
        family="trig_lhopital",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_lhopital_nonzero_point_denominator_domain",
        trace_regime="lhopital_nonzero_point",
    ),
    LimitCommandMatrixCase(
        name="finite_lhopital_second_order_shifted_supported",
        expr="limit((1-cos(x-1))/(x-1)^2, x, 1)",
        expected_result="1/2",
        expected_required_display=("x ≠ 1",),
        expected_step_substrings=("Evaluar límite finito",),
        family="trig_lhopital",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_lhopital_nonzero_point_denominator_domain",
        trace_regime="lhopital_nonzero_point",
    ),
    LimitCommandMatrixCase(
        name="finite_lhopital_third_order_shifted_supported",
        expr="limit((sin(x-1)-(x-1))/(x-1)^3, x, 1)",
        expected_result="-1/6",
        expected_required_display=("x ≠ 1",),
        expected_step_substrings=("Evaluar límite finito",),
        family="trig_lhopital",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_lhopital_nonzero_point_denominator_domain",
        trace_regime="lhopital_nonzero_point",
    ),
    LimitCommandMatrixCase(
        name="infinity_product_log_unit_argument_unit_slope_supported",
        expr="limit(x*ln(1+1/x), x, infinity)",
        expected_result="1",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="product_log_unit_argument",
        point_regime="infinity",
        trace_regime="inf_times_zero_log_reduction",
    ),
    LimitCommandMatrixCase(
        name="infinity_product_log_unit_argument_scaled_slope_supported",
        expr="limit(x*ln(1+2/x), x, infinity)",
        expected_result="2",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="product_log_unit_argument",
        point_regime="infinity",
        trace_regime="inf_times_zero_log_reduction",
    ),
    LimitCommandMatrixCase(
        name="infinity_product_log_unit_argument_quadratic_cofactor_supported",
        expr="limit(x^2*ln(1+1/x^2), x, infinity)",
        expected_result="1",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="product_log_unit_argument",
        point_regime="infinity",
        trace_regime="inf_times_zero_log_reduction",
    ),
    LimitCommandMatrixCase(
        name="finite_radical_difference_conjugate_unit_radius_supported",
        expr="limit((sqrt(1+x)-sqrt(1-x))/x, x, 0)",
        expected_result="1",
        expected_required_display=("x ≠ 0", "x ≤ 1", "x ≥ -1"),
        expected_step_substrings=("Evaluar límite finito",),
        family="radical_difference_conjugate",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_radical_difference_denominator_and_radicand_domain",
        trace_regime="radical_difference_conjugate",
    ),
    LimitCommandMatrixCase(
        name="finite_radical_difference_conjugate_scaled_radius_supported",
        expr="limit((sqrt(4+x)-sqrt(4-x))/x, x, 0)",
        expected_result="1/2",
        expected_required_display=("x ≠ 0", "x ≤ 4", "x ≥ -4"),
        expected_step_substrings=("Evaluar límite finito",),
        family="radical_difference_conjugate",
        point_regime="finite",
        domain_regime="required_condition",
        required_condition_regime="finite_radical_difference_denominator_and_radicand_domain",
        trace_regime="radical_difference_conjugate",
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


def extract_required_display(payload: dict[str, Any] | None) -> tuple[str, ...]:
    if not payload:
        return ()
    raw = payload.get("required_display") or []
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, str))


def timing_seconds(timings_us: dict[str, int], key: str) -> float | None:
    value = timings_us.get(key)
    if not isinstance(value, int):
        return None
    return value / 1_000_000.0


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


def classify_error_kind(error: str | None) -> str | None:
    if error is None:
        return None
    if error == "timeout":
        return "timeout"
    if "fragile substring" in error:
        return "stderr_fragility"
    if "warning" in error:
        return "warning_mismatch"
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
    case: LimitCommandMatrixCase,
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
    step_text = extract_step_text(parsed)
    cli_timings_us = extract_cli_timings_us(parsed)
    cli_parse_seconds = timing_seconds(cli_timings_us, "parse_us")
    cli_simplify_seconds = timing_seconds(cli_timings_us, "simplify_us")
    cli_total_seconds = timing_seconds(cli_timings_us, "total_us")
    cli_public_overhead_seconds = (
        max(0.0, wall_elapsed - cli_total_seconds)
        if cli_total_seconds is not None
        else None
    )
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
    elif (stderr_error := stderr_fragility_error(stderr)) is not None:
        error = stderr_error
    else:
        warnings_ok, warning_error = warning_expectations_met(
            case.expected_warning_substrings,
            warnings,
        )
        if not warnings_ok:
            error = warning_error
        else:
            for expected in case.expected_step_substrings:
                if expected not in step_text:
                    error = f"missing expected step trace containing {expected!r}"
                    break

    status: Status = "pass" if error is None else "fail"
    error_kind = classify_error_kind(error)
    if status == "pass" and slow_wall_seconds is not None and wall_elapsed > slow_wall_seconds:
        status = "slow"
        error = f"slow: {wall_elapsed:.3f}s > {slow_wall_seconds:.3f}s"
        error_kind = "slow"

    return {
        "name": case.name,
        "status": status,
        "error": error,
        "error_kind": error_kind,
        "returncode": process.returncode,
        "wall_elapsed_seconds": round(wall_elapsed, 3),
        "cli_parse_us": cli_timings_us.get("parse_us"),
        "cli_simplify_us": cli_timings_us.get("simplify_us"),
        "cli_total_us": cli_timings_us.get("total_us"),
        "cli_parse_elapsed_seconds": (
            round(cli_parse_seconds, 6) if cli_parse_seconds is not None else None
        ),
        "cli_simplify_elapsed_seconds": (
            round(cli_simplify_seconds, 6)
            if cli_simplify_seconds is not None
            else None
        ),
        "cli_total_elapsed_seconds": (
            round(cli_total_seconds, 6) if cli_total_seconds is not None else None
        ),
        "cli_public_overhead_seconds": (
            round(cli_public_overhead_seconds, 6)
            if cli_public_overhead_seconds is not None
            else None
        ),
        "result": result if isinstance(result, str) else None,
        "required_display": list(required_display),
        "warnings": list(warnings),
        "expected_result": case.expected_result,
        "expected_required_display": list(case.expected_required_display),
        "expected_warning_substrings": list(case.expected_warning_substrings),
        "expected_step_substrings": list(case.expected_step_substrings),
        "family": case.family,
        "point_regime": case.point_regime,
        "domain_regime": case.domain_regime,
        "required_condition_regime": case.required_condition_regime,
        "outcome": case.outcome,
        "residual_cause": case.residual_cause,
        "trace_regime": case.trace_regime,
        "presentation_regime": case.presentation_regime,
    }


def build_cases(case_filters: tuple[str, ...] = ()) -> tuple[LimitCommandMatrixCase, ...]:
    if not case_filters:
        return DEFAULT_LIMIT_COMMAND_MATRIX_CASES
    requested = set(case_filters)
    return tuple(case for case in DEFAULT_LIMIT_COMMAND_MATRIX_CASES if case.name in requested)


def count_by(cases: tuple[LimitCommandMatrixCase, ...], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        value = getattr(case, attr)
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def calculus_maturity_block(case: LimitCommandMatrixCase) -> str:
    if case.outcome in {"residual", "undefined"}:
        return "block9_residuals_and_non_goals"
    return "block3_real_domain_limits"


def calculus_block_gate(case: LimitCommandMatrixCase) -> str:
    if case.outcome == "residual":
        return "safe_residual_policy"
    if case.outcome == "undefined":
        return "explicit_undefined_domain_policy"
    if case.expected_required_display:
        return "domain_conditions_and_limit_policy"
    if case.expected_step_substrings:
        return "didactic_trace_and_limit_policy"
    return "supported_limit_policy"


def count_calculus_maturity_blocks(
    cases: tuple[LimitCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        block = calculus_maturity_block(case)
        counts[block] = counts.get(block, 0) + 1
    return dict(sorted(counts.items()))


def count_calculus_block_gates(
    cases: tuple[LimitCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        gate = calculus_block_gate(case)
        counts[gate] = counts.get(gate, 0) + 1
    return dict(sorted(counts.items()))


def count_required_display_items(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for item in result.get("required_display", []):
            if not isinstance(item, str):
                continue
            counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items()))


def count_residual_causes(
    cases: tuple[LimitCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        if case.outcome != "residual":
            continue
        cause = case.residual_cause
        if cause == "not_applicable":
            cause = "unclassified_residual"
        counts[cause] = counts.get(cause, 0) + 1
    return dict(sorted(counts.items()))


def count_residual_families(
    cases: tuple[LimitCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        if case.outcome != "residual":
            continue
        counts[case.family] = counts.get(case.family, 0) + 1
    return dict(sorted(counts.items()))


def count_residual_cause_families(
    cases: tuple[LimitCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        if case.outcome != "residual":
            continue
        cause = case.residual_cause
        if cause == "not_applicable":
            cause = "unclassified_residual"
        key = f"{cause}/{case.family}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def group_residual_cases_by_cause(
    cases: tuple[LimitCommandMatrixCase, ...],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for case in cases:
        if case.outcome != "residual":
            continue
        cause = case.residual_cause
        if cause == "not_applicable":
            cause = "unclassified_residual"
        grouped.setdefault(cause, []).append(case.name)
    return {cause: grouped[cause] for cause in sorted(grouped)}


def phase_runtime_distribution(
    results: list[dict[str, Any]],
    *,
    phase_key: str,
) -> dict[str, Any]:
    values = [
        float(result[phase_key])
        for result in results
        if isinstance(result.get(phase_key), (int, float))
    ]
    if not values:
        return {}
    values.sort()
    p95_index = min(len(values) - 1, int(len(values) * 0.95))
    total_elapsed = sum(values)
    return {
        "timed_case_count": len(values),
        "total_elapsed_seconds": round(total_elapsed, 3),
        "avg_case_ms": round(total_elapsed * 1000.0 / len(values), 3),
        "p95_case_ms": round(values[p95_index] * 1000.0, 3),
        "max_case_ms": round(max(values) * 1000.0, 3),
    }


def phase_runtime_case_rows(
    results: list[dict[str, Any]],
    *,
    phase_key: str,
    output_key: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    timed_results = [
        result
        for result in results
        if isinstance(result.get(phase_key), (int, float))
    ]
    timed_results.sort(
        key=lambda result: (
            -float(result.get(phase_key, 0.0)),
            str(result.get("name", "")),
        )
    )

    rows: list[dict[str, Any]] = []
    for result in timed_results[:limit]:
        row: dict[str, Any] = {
            "name": result.get("name"),
            output_key: round(float(result[phase_key]), 3),
        }
        for key in (
            "family",
            "point_regime",
            "domain_regime",
            "required_condition_regime",
            "trace_regime",
            "presentation_regime",
            "calculus_maturity_block",
            "calculus_block_gate",
        ):
            value = result.get(key)
            if isinstance(value, str):
                row[key] = value
        rows.append(row)
    return rows


def phase_runtime_observability_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for phase_name, phase_key, output_key in (
        ("cli_parse", "cli_parse_elapsed_seconds", "cli_parse_elapsed_seconds"),
        (
            "cli_simplify",
            "cli_simplify_elapsed_seconds",
            "cli_simplify_elapsed_seconds",
        ),
        ("cli_total", "cli_total_elapsed_seconds", "cli_total_elapsed_seconds"),
        (
            "cli_public_overhead",
            "cli_public_overhead_seconds",
            "cli_public_overhead_seconds",
        ),
    ):
        distribution = phase_runtime_distribution(results, phase_key=phase_key)
        if distribution:
            summary[f"{phase_name}_runtime_distribution"] = distribution
        rows = phase_runtime_case_rows(
            results,
            phase_key=phase_key,
            output_key=output_key,
        )
        if rows:
            summary[f"slowest_{phase_name}_evaluations"] = rows
    return summary


def increment_issue_kind(issue_kind_counts: dict[str, int], error_kind: str | None) -> None:
    if error_kind is None:
        return
    issue_kind_counts[error_kind] = issue_kind_counts.get(error_kind, 0) + 1


def run_matrix(
    cases: tuple[LimitCommandMatrixCase, ...],
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
        increment_issue_kind(issue_kind_counts, result.get("error_kind"))

    if not results:
        overall_status: Status = "fail"
        increment_issue_kind(issue_kind_counts, "no_matching_cases")
    elif status_counts["timeout"]:
        overall_status = "timeout"
    elif status_counts["fail"]:
        overall_status = "fail"
    elif status_counts["slow"]:
        overall_status = "slow"
    else:
        overall_status = "pass"

    problem_cases = [
        result
        for result in results
        if result["status"] != "pass"
    ]
    warning_expected_cases = sum(1 for case in cases if case.expected_warning_substrings)
    required_display_cases = sum(1 for case in cases if case.expected_required_display)
    step_checked_cases = sum(1 for case in cases if case.expected_step_substrings)
    expected_step_substrings = sum(
        len(case.expected_step_substrings) for case in cases
    )
    supported_step_unchecked_cases = [
        case.name
        for case in cases
        if case.outcome == "supported" and not case.expected_step_substrings
    ]
    residual_case_names = [case.name for case in cases if case.outcome == "residual"]

    return {
        "status": overall_status,
        "total": len(results),
        "status_counts": status_counts,
        "issue_kind_counts": issue_kind_counts,
        "problem_case_count": len(problem_cases),
        "problem_cases": problem_cases,
        "cases": results,
        "supported_case_count": sum(1 for case in cases if case.outcome == "supported"),
        "residual_case_count": sum(1 for case in cases if case.outcome == "residual"),
        "residual_case_names": residual_case_names,
        "warning_expected_case_count": warning_expected_cases,
        "required_display_case_count": required_display_cases,
        "step_checked_case_count": step_checked_cases,
        "supported_step_unchecked_case_count": len(supported_step_unchecked_cases),
        "supported_step_unchecked_cases": supported_step_unchecked_cases,
        "expected_step_substring_count": expected_step_substrings,
        "distinct_required_display_count": len(
            {
                required
                for case in cases
                for required in case.expected_required_display
            }
        ),
        "required_display_counts": count_required_display_items(results),
        "family_count": len({case.family for case in cases}),
        "point_regime_counts": count_by(cases, "point_regime"),
        "domain_regime_counts": count_by(cases, "domain_regime"),
        "required_condition_regime_counts": count_by(cases, "required_condition_regime"),
        "outcome_counts": count_by(cases, "outcome"),
        "residual_cause_counts": count_residual_causes(cases),
        "residual_family_counts": count_residual_families(cases),
        "residual_cause_family_counts": count_residual_cause_families(cases),
        "residual_cases_by_cause": group_residual_cases_by_cause(cases),
        "calculus_maturity_block_counts": count_calculus_maturity_blocks(cases),
        "calculus_block_gate_counts": count_calculus_block_gates(cases),
        "trace_regime_counts": count_by(cases, "trace_regime"),
        "presentation_regime_counts": count_by(cases, "presentation_regime"),
        "case_filters": [case.name for case in cases],
        **runtime_observability_summary(
            results,
            group_keys=(
                "family",
                "point_regime",
                "domain_regime",
                "trace_regime",
            ),
        ),
        **phase_runtime_observability_summary(results),
    }


def summarize_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    summary = {key: value for key, value in matrix.items() if key != "cases"}
    problem_cases = matrix.get("problem_cases")
    if isinstance(problem_cases, list):
        summary["problem_cases"] = problem_cases
    return summary


def print_human(matrix: dict[str, Any]) -> None:
    print(
        f"status={matrix['status']} total={matrix['total']} "
        f"counts={matrix['status_counts']} issue_kinds={matrix['issue_kind_counts']}"
    )
    for case in matrix.get("cases", []):
        print(
            f"- {case['name']}: {case['status']} result={case.get('result')} "
            f"required={case.get('required_display')} warnings={len(case.get('warnings', []))}"
        )
        if case.get("error"):
            print(f"  error={case['error']}")


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
