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
from engine_command_matrix_observability import stderr_fragility_error


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
        expected_result="limit(1 / x, x, 0)",
        expected_required_display=("x ≠ 0",),
        expected_warning_substrings=("Finite point limits are not supported safely yet",),
        expected_step_substrings=("Conservar límite residual",),
        family="rational",
        point_regime="finite",
        domain_regime="simple_pole_residual",
        required_condition_regime="finite_rational_pole_residual_domain",
        outcome="residual",
        residual_cause="finite_rational_pole_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="rational_pole_residual",
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
        expected_result="pi / 6",
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
        expected_result="limit(sign(x), x, 0)",
        expected_warning_substrings=("Finite point limits are not supported safely yet",),
        expected_step_substrings=("Conservar límite residual",),
        family="discontinuous",
        point_regime="finite",
        domain_regime="discontinuous_residual",
        outcome="residual",
        residual_cause="finite_discontinuity_or_orientation",
        trace_regime="finite_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_discontinuous_sign_residual_presentation_cleanup",
        expr="limit(sign(x + 0), x, 0)",
        expected_result="limit(sign(x), x, 0)",
        expected_warning_substrings=("Finite point limits are not supported safely yet",),
        expected_step_substrings=("Conservar límite residual", "limit(sign(x), x, 0)"),
        family="discontinuous",
        point_regime="finite",
        domain_regime="discontinuous_residual",
        outcome="residual",
        residual_cause="finite_discontinuity_or_orientation",
        trace_regime="finite_residual_policy",
        presentation_regime="residual_presentation_cleanup",
    ),
    LimitCommandMatrixCase(
        name="finite_abs_orientation_quotient_residual_boundary",
        expr="limit(abs(x)/x, x, 0)",
        expected_result="limit(|x| / x, x, 0)",
        expected_required_display=("x ≠ 0",),
        expected_warning_substrings=("Finite point limits are not supported safely yet",),
        expected_step_substrings=("Conservar límite residual",),
        family="abs_orientation",
        point_regime="finite",
        domain_regime="orientation_discontinuity_residual",
        required_condition_regime="finite_abs_orientation_denominator_domain",
        outcome="residual",
        residual_cause="finite_discontinuity_or_orientation",
        trace_regime="finite_residual_policy",
        presentation_regime="abs_orientation_residual",
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
        expected_result="limit(tan(pi / 2 + x), x, 0)",
        expected_required_display=("cos(pi / 2 + x) ≠ 0",),
        expected_warning_substrings=("Finite point limits are not supported safely yet",),
        expected_step_substrings=("Conservar límite residual",),
        family="trig_special_angle",
        point_regime="finite",
        domain_regime="table_undefined_trig_pole_residual",
        required_condition_regime="finite_trig_pole_residual_domain",
        outcome="residual",
        residual_cause="finite_trig_pole_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="residual",
    ),
    LimitCommandMatrixCase(
        name="finite_reciprocal_trig_sine_pole_residual",
        expr="limit(csc(x + pi), x, 0)",
        expected_result="limit(csc(x + pi), x, 0)",
        expected_required_display=("sin(x + pi) ≠ 0",),
        expected_warning_substrings=("Finite point limits are not supported safely yet",),
        expected_step_substrings=("Conservar límite residual",),
        family="reciprocal_trig",
        point_regime="finite",
        domain_regime="table_undefined_trig_sine_pole_residual",
        required_condition_regime="finite_trig_sine_pole_residual_domain",
        outcome="residual",
        residual_cause="finite_trig_pole_policy",
        trace_regime="finite_residual_policy",
        presentation_regime="residual",
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
        expected_result="-pi / 2",
        expected_step_substrings=("Evaluar límite en infinito",),
        family="inverse_trig",
        point_regime="infinity",
        domain_regime="total_real_polynomial_tail_domain",
        required_condition_regime="infinity_path_total_real_polynomial_tail_domain",
        trace_regime="bounded_total_real_polynomial_tail_policy",
        presentation_regime="infinity_tail_polynomial_bounded_inverse_trig",
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
        "calculus_maturity_block_counts": count_calculus_maturity_blocks(cases),
        "calculus_block_gate_counts": count_calculus_block_gates(cases),
        "trace_regime_counts": count_by(cases, "trace_regime"),
        "presentation_regime_counts": count_by(cases, "presentation_regime"),
        "case_filters": [case.name for case in cases],
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
