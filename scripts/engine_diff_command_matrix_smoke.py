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
from engine_command_matrix_observability import (
    payload_observability_summary,
    runtime_observability_summary,
    stderr_fragility_error,
)


ROOT = SCRIPT_DIR.parent
DEFAULT_CAS_CLI = ROOT / "target" / "release" / "cas_cli"
Status = Literal["pass", "slow", "fail", "timeout"]


EXACT_SQUARE_RUNTIME_PAIR_CASES = (
    {
        "name": "inverse_hyperbolic_root_atanh_denominator_scale_exact_square",
        "baseline_case": "inverse_hyperbolic_root_atanh_symbolic_denominator_scale_open_interval",
        "exact_square_case": "inverse_hyperbolic_root_atanh_exact_square_denominator_scale_open_interval",
    },
)


@dataclass(frozen=True)
class DiffCommandMatrixCase:
    name: str
    expr: str
    expected_result: str
    expected_required_display: tuple[str, ...] = ()
    expected_warning_substrings: tuple[str, ...] = ()
    expected_blocked_hint_substrings: tuple[str, ...] = ()
    expected_step_substrings: tuple[str, ...] = ()
    forbidden_stderr_substrings: tuple[str, ...] = ()
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
        name="product_log_root_domain_dedupe_compact",
        expr="diff(ln(x)*sqrt(x), x)",
        expected_result="(ln(x) + 2) / (2·sqrt(x))",
        expected_required_display=("x > 0",),
        expected_step_substrings=(
            "Usar regla del producto",
            "Derivar el primer factor",
            "Derivar el segundo factor",
            "Extract Common Multiplicative Factor",
            "Sumar fracciones",
        ),
        family="product_log_root",
        argument_regime="product_domain_bearing_factors",
        domain_regime="required_condition",
        trace_regime="product_rule_log_root",
        presentation_regime="compact_log_root_quotient",
    ),
    DiffCommandMatrixCase(
        name="log_over_sqrt_root_denominator_compact",
        expr="diff(ln(x)/sqrt(x), x)",
        expected_result="(2 - ln(x)) / (2·x·sqrt(x))",
        expected_required_display=("x > 0",),
        expected_step_substrings=(
            "Calcular la derivada",
            "Presentar resultado de cálculo en forma compacta",
        ),
        family="log_over_sqrt",
        argument_regime="reciprocal_root_same_argument",
        domain_regime="required_condition",
        trace_regime="product_rule_log_root_reciprocal",
        presentation_regime="compact_log_over_root_denominator",
    ),
    DiffCommandMatrixCase(
        name="log_over_sqrt_denominator_scale_compact",
        expr="diff(ln(x)/(2*sqrt(x)), x)",
        expected_result="(2 - ln(x)) / (4·x·sqrt(x))",
        expected_required_display=("x > 0",),
        expected_step_substrings=(
            "Calcular la derivada",
            "Presentar resultado de cálculo en forma compacta",
        ),
        family="log_over_sqrt",
        argument_regime="denominator_scaled_reciprocal_root_same_argument",
        domain_regime="required_condition",
        trace_regime="quotient_rule_log_root_reciprocal_denominator_scale",
        presentation_regime="denominator_scaled_compact_log_over_root_denominator",
    ),
    DiffCommandMatrixCase(
        name="log_over_sqrt_symbolic_denominator_scale_compact",
        expr="diff(ln(x)/(a*sqrt(x)), x)",
        expected_result="(2 - ln(x)) / (2·a·x·sqrt(x))",
        expected_required_display=("a ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Calcular la derivada",
            "Presentar resultado de cálculo en forma compacta",
        ),
        family="log_over_sqrt",
        argument_regime="symbolic_denominator_scaled_reciprocal_root_same_argument",
        domain_regime="required_condition",
        trace_regime="quotient_rule_log_root_reciprocal_symbolic_denominator_scale",
        presentation_regime="symbolic_denominator_scaled_compact_log_over_root_denominator",
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
        name="log_abs_negative_scale_empty_positive_argument_domain_undefined",
        expr="diff(ln(-3*abs(x-1)), x)",
        expected_result="undefined",
        expected_step_substrings=(
            "Detectar dominio real vacío del logaritmo",
            "undefined",
        ),
        family="log_abs_empty_domain",
        argument_regime="negative_scaled_abs_argument",
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
        name="elementary_reciprocal_trig_csc_affine_chain_sine_pole",
        expr="diff(csc(2*x+1), x)",
        expected_result="-2·csc(2·x + 1)·cot(2·x + 1)",
        expected_required_display=("sin(2·x + 1) ≠ 0",),
        expected_step_substrings=(
            "Expandir cosecante como recíproco de seno",
            "Calcular la derivada",
            "Usar regla del cociente",
            "Derivar el denominador",
            "Presentar resultado de cálculo en forma compacta",
        ),
        family="elementary_reciprocal_trig_chain",
        argument_regime="affine_argument_with_sine_pole",
        domain_regime="trig_sine_pole_required",
        trace_regime="reciprocal_trig_quotient_chain_rule",
        presentation_regime="reciprocal_trig_compact_product",
    ),
    DiffCommandMatrixCase(
        name="elementary_hyperbolic_sinh_affine_chain_trace",
        expr="diff(sinh(2*x+1), x)",
        expected_result="2·cosh(2·x + 1)",
        expected_step_substrings=(
            "Usar regla de sinh(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="elementary_hyperbolic_chain",
        argument_regime="affine_argument",
        trace_regime="hyperbolic_chain_rule",
        presentation_regime="hyperbolic_chain",
    ),
    DiffCommandMatrixCase(
        name="elementary_hyperbolic_tanh_affine_chain_reciprocal_square",
        expr="diff(tanh(2*x+1), x)",
        expected_result="2 / cosh(2·x + 1)^2",
        expected_step_substrings=(
            "Usar regla de tanh(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="elementary_hyperbolic_chain",
        argument_regime="affine_argument",
        domain_regime="unconditional_cosh_positive",
        trace_regime="hyperbolic_tanh_chain_rule",
        presentation_regime="hyperbolic_reciprocal_square",
    ),
    DiffCommandMatrixCase(
        name="log_tangent_positive_source_domain",
        expr="diff(ln(tan(x)), x)",
        expected_result="1 / (sin(x)·cos(x))",
        expected_required_display=("tan(x) > 0",),
        expected_step_substrings=(
            "Expandir tangente como seno entre coseno",
            "Usar regla de ln(u)",
            "Aplicar la identidad pitagórica",
        ),
        family="log_trig_chain",
        argument_regime="tangent_argument",
        domain_regime="positive_source_trig_domain",
        trace_regime="log_chain_rule_after_trig_expansion",
        presentation_regime="positive_source_trig_condition",
    ),
    DiffCommandMatrixCase(
        name="log_cotangent_positive_source_domain",
        expr="diff(ln(cot(x)), x)",
        expected_result="-1 / (sin(x)·cos(x))",
        expected_required_display=("cot(x) > 0",),
        expected_step_substrings=(
            "Expandir cotangente como coseno entre seno",
            "Usar regla de ln(u)",
            "Simplificar fracción anidada",
        ),
        family="log_trig_chain",
        argument_regime="cotangent_argument",
        domain_regime="positive_source_trig_domain",
        trace_regime="log_chain_rule_after_trig_expansion",
        presentation_regime="positive_source_trig_condition",
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
        name="quotient_root_over_log_domain_pole_compact",
        expr="diff(sqrt(x)/ln(x), x)",
        expected_result="(ln(x) - 2) / (2·ln(x)^2·sqrt(x))",
        expected_required_display=("x ≠ 1", "x > 0"),
        expected_step_substrings=(
            "Calcular la derivada",
            "Usar regla del cociente",
            "Derivar el numerador",
            "Derivar el denominador",
        ),
        family="quotient_log_root",
        argument_regime="quotient_domain_bearing_denominator",
        domain_regime="required_condition",
        trace_regime="quotient_rule_log_root",
        presentation_regime="compact_log_denominator_root_quotient",
    ),
    DiffCommandMatrixCase(
        name="quotient_root_over_log_symbolic_denominator_scale_compact",
        expr="diff(sqrt(x)/(a*ln(x)), x)",
        expected_result="(ln(x) - 2) / (2·a·ln(x)^2·sqrt(x))",
        expected_required_display=("a ≠ 0", "x ≠ 1", "x > 0"),
        expected_step_substrings=(
            "Calcular la derivada",
            "Usar regla del cociente",
            "Derivar el denominador",
        ),
        family="quotient_log_root",
        argument_regime="symbolic_denominator_scaled_quotient_domain_bearing_denominator",
        domain_regime="required_condition",
        trace_regime="quotient_rule_log_root_symbolic_denominator_scale",
        presentation_regime="symbolic_denominator_scaled_compact_log_denominator_root_quotient",
    ),
    DiffCommandMatrixCase(
        name="positive_quadratic_log_arctan_polynomial_primitive_compact",
        expr="diff(ln(x^2+2*x+2)-arctan(x+1)-x, x)",
        expected_result="(-x^2 - 1) / (x^2 + 2·x + 2)",
        expected_step_substrings=("Usar linealidad de la derivada",),
        family="positive_quadratic_log_arctan_primitive",
        argument_regime="log_arctan_polynomial_primitive",
        domain_regime="unconditional_positive_quadratic",
        trace_regime="positive_quadratic_log_arctan_linearity",
        presentation_regime="compact_positive_quadratic_quotient",
    ),
    DiffCommandMatrixCase(
        name="positive_quadratic_log_arctan_surd_primitive_compact",
        expr="diff(1/2*ln(x^2+x+1)-arctan((2*x+1)/sqrt(3))/sqrt(3), x)",
        expected_result="x / (x^2 + x + 1)",
        expected_step_substrings=("Usar linealidad de la derivada",),
        family="positive_quadratic_log_arctan_primitive",
        argument_regime="surd_discriminant_log_arctan_primitive",
        domain_regime="unconditional_positive_quadratic",
        trace_regime="positive_quadratic_log_arctan_surd_linearity",
        presentation_regime="compact_positive_quadratic_surd_quotient",
    ),
    DiffCommandMatrixCase(
        name="positive_quadratic_log_arctan_surd_negative_orientation_compact",
        expr="diff(1/2*ln(x^2-x+1)-arctan((2*x-1)/sqrt(3))/sqrt(3), x)",
        expected_result="(x - 1) / (x^2 + 1 - x)",
        expected_step_substrings=("Usar linealidad de la derivada",),
        family="positive_quadratic_log_arctan_primitive",
        argument_regime="negative_orientation_surd_discriminant_log_arctan_primitive",
        domain_regime="unconditional_positive_quadratic",
        trace_regime="positive_quadratic_log_arctan_surd_negative_orientation_linearity",
        presentation_regime="compact_positive_quadratic_surd_negative_orientation_quotient",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_symbolic_radius_shifted_arctan_primitive_no_depth_overflow",
        expr="diff(arctan((b+x)/a)/a, x)",
        expected_result="1 / ((b + x)^2 + a^2)",
        expected_required_display=("a ≠ 0",),
        expected_step_substrings=("Calcular la derivada",),
        forbidden_stderr_substrings=("depth_overflow",),
        family="positive_quadratic_arctan_primitive",
        argument_regime="shifted_symbolic_radius_affine_argument",
        domain_regime="symbolic_radius_minimal_nonzero_parameter",
        trace_regime="direct_inverse_tangent_chain_rule",
        presentation_regime="compact_shifted_symbolic_radius_positive_quadratic",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_symbolic_radius_non_unit_slope_arctan_primitive_compact",
        expr="diff(arctan((c*x+b)/a)/a, x)",
        expected_result="c / ((c·x + b)^2 + a^2)",
        expected_required_display=("a ≠ 0",),
        expected_step_substrings=("Calcular la derivada",),
        forbidden_stderr_substrings=("depth_overflow",),
        family="positive_quadratic_arctan_primitive",
        argument_regime="non_unit_slope_symbolic_radius_affine_argument",
        domain_regime="symbolic_radius_minimal_nonzero_parameter",
        trace_regime="direct_inverse_tangent_chain_rule",
        presentation_regime="compact_shifted_symbolic_radius_positive_quadratic",
    ),
    DiffCommandMatrixCase(
        name="log_ratio_single_pole_scaled_shifted_linear_compact",
        expr="diff(ln(abs((2*x+3)/(x-5)))+1/(x-5), x)",
        expected_result="(62 - 15·x) / (2·x^3 + 20·x + 75 - 17·x^2)",
        expected_required_display=("x ≠ -3/2", "x ≠ 5"),
        expected_step_substrings=("Calcular la derivada",),
        family="log_ratio_single_pole_primitive",
        argument_regime="scaled_shifted_linear_factors",
        domain_regime="linear_poles_required",
        trace_regime="log_ratio_single_pole_linearity",
        presentation_regime="compact_integer_rational_quotient",
    ),
    DiffCommandMatrixCase(
        name="log_ratio_single_pole_positive_scaled_abs_argument_compact",
        expr="diff(ln(3*abs((2*x+3)/(x-5)))+1/(x-5), x)",
        expected_result="(62 - 15·x) / (2·x^3 + 20·x + 75 - 17·x^2)",
        expected_required_display=("x ≠ -3/2", "x ≠ 5"),
        expected_step_substrings=("Calcular la derivada",),
        forbidden_stderr_substrings=("depth_overflow",),
        family="log_ratio_single_pole_primitive",
        argument_regime="positive_scaled_abs_scaled_shifted_linear_factors",
        domain_regime="linear_poles_required",
        trace_regime="log_ratio_single_pole_linearity",
        presentation_regime="compact_integer_rational_quotient",
    ),
    DiffCommandMatrixCase(
        name="separated_log_abs_linear_pole_raw_preserved",
        expr="diff((-1/2)*ln(abs(x-1)) - 4/(x-1) + (1/2)*ln(abs(x+1)), x)",
        expected_result="(3·x + 5) / (x^3 + 1 - x^2 - x)",
        expected_required_display=("x ≠ -1", "x ≠ 1"),
        expected_step_substrings=("Usar linealidad de la derivada",),
        forbidden_stderr_substrings=("depth_overflow",),
        family="separated_log_abs_linear_pole_primitive",
        argument_regime="separated_linear_logs_plus_linear_pole",
        domain_regime="linear_poles_required",
        trace_regime="separated_log_abs_linear_pole_linearity",
        presentation_regime="compact_integer_rational_quotient",
    ),
    DiffCommandMatrixCase(
        name="positive_quadratic_log_abs_pole_scaled_quadratic_compact",
        expr="diff(1/4*ln((2*x)^2+1)-1/2*ln(abs(x-1))-1/(2*(x-1)), x)",
        expected_result="(3·x + 2) / (8·x^4 + 10·x^2 + 2 - 16·x^3 - 4·x)",
        expected_required_display=("x ≠ 1",),
        expected_step_substrings=("Usar linealidad de la derivada",),
        forbidden_stderr_substrings=("depth_overflow",),
        family="positive_quadratic_log_abs_pole_primitive",
        argument_regime="scaled_positive_quadratic_plus_linear_pole",
        domain_regime="linear_poles_required",
        trace_regime="positive_quadratic_log_abs_pole_linearity",
        presentation_regime="compact_integer_rational_quotient",
    ),
    DiffCommandMatrixCase(
        name="positive_quadratic_log_abs_pole_scaled_linear_pole_compact",
        expr="diff(1/6*ln(2*x^2+2)-1/2*ln(abs(2*x-2))-1/(2*(2*x-2)), x)",
        expected_result="(x^2 + 9 - 2·x^3 - 2·x) / (12·x^4 + 24·x^2 + 12 - 24·x^3 - 24·x)",
        expected_required_display=("x ≠ 1",),
        expected_step_substrings=("Usar linealidad de la derivada",),
        forbidden_stderr_substrings=("depth_overflow",),
        family="positive_quadratic_log_abs_pole_primitive",
        argument_regime="scaled_positive_quadratic_plus_scaled_linear_pole",
        domain_regime="linear_poles_required",
        trace_regime="positive_quadratic_log_abs_pole_linearity",
        presentation_regime="compact_integer_rational_quotient",
    ),
    DiffCommandMatrixCase(
        name="positive_quadratic_log_abs_pole_positive_orientation_combined_source_compact",
        expr="diff(1/4*ln(x^2+1)-1/2*ln(abs(x-1))+1/(2*x-2), x)",
        expected_result="-x^2 / (x^4 + 2·x^2 + 1 - 2·x^3 - 2·x)",
        expected_required_display=("x ≠ 1",),
        expected_step_substrings=("Usar linealidad de la derivada",),
        forbidden_stderr_substrings=("depth_overflow",),
        family="positive_quadratic_log_abs_pole_primitive",
        argument_regime="positive_orientation_combined_linear_denominator_source",
        domain_regime="linear_poles_required",
        trace_regime="positive_quadratic_log_abs_pole_raw_linearity",
        presentation_regime="compact_integer_rational_quotient",
    ),
    DiffCommandMatrixCase(
        name="positive_quadratic_log_abs_pole_linear_wrapper_compact",
        expr="diff((ln(abs(x^2+x+1))+1/(x+2))/(x+2), x)",
        expected_result="(2·x + 1) / ((x^2 + x + 1)·(x + 2)) - 2 / (x + 2)^3 - ln(x^2 + x + 1) / (x + 2)^2",
        expected_required_display=("x ≠ -2",),
        expected_step_substrings=("Calcular la derivada",),
        forbidden_stderr_substrings=("depth_overflow",),
        family="positive_quadratic_log_abs_pole_primitive",
        argument_regime="linear_denominator_wrapper",
        domain_regime="linear_wrapper_pole_required",
        trace_regime="positive_quadratic_log_abs_pole_wrapper_linearity",
        presentation_regime="compact_log_wrapper",
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
        name="sqrt_tangent_positive_dominates_nonnegative_domain",
        expr="diff(sqrt(tan(x)), x)",
        expected_result="1 / (2·cos(x)^2·sqrt(tan(x)))",
        expected_required_display=("tan(x) > 0",),
        expected_step_substrings=(
            "Calcular la derivada",
            "Convertir un cociente trigonométrico en tangente",
        ),
        family="root_trig_chain",
        argument_regime="source_trig_argument",
        domain_regime="positive_dominates_nonnegative_source_trig_domain",
        trace_regime="root_trig_chain_rule",
        presentation_regime="positive_dominates_nonnegative_source_condition",
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
        name="inverse_trig_root_symbolic_denominator_scale",
        expr="diff(arctan(sqrt(x)/a), x)",
        expected_result="a / (2·sqrt(x)·(a^2 + x))",
        expected_required_display=("x > 0", "a ≠ 0"),
        expected_step_substrings=(
            "Usar regla de arctan(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="inverse_trig_root",
        argument_regime="symbolic_denominator_scaled_nested_root",
        domain_regime="required_condition",
        trace_regime="parameter_scaled_chain_rule",
        presentation_regime="symbolic_denominator_post_calculus_compact",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_root_external_symbolic_denominator_scale",
        expr="diff((2*arctan(sqrt(x)/a))/a, x)",
        expected_result="(x^(1/2)·2)/(2·(x·a^2 + x^2))",
        expected_required_display=("a ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Sacar constante de una fracción",
            "Calcular la derivada",
            "Usar factor constante de la derivada",
        ),
        family="inverse_trig_root",
        argument_regime="external_symbolic_denominator_scaled_nested_root",
        domain_regime="required_condition",
        trace_regime="constant_multiple_parameter_scaled_chain_rule",
        presentation_regime="external_symbolic_denominator_post_calculus_compact",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_root_symbolic_rational_denominator_scale",
        expr="diff(arctan(sqrt(x)/(2*a)), x)",
        expected_result="a / (sqrt(x)·(4·a^2 + x))",
        expected_required_display=("a ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar regla de arctan(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="inverse_trig_root",
        argument_regime="symbolic_rational_denominator_scaled_nested_root",
        domain_regime="required_condition",
        trace_regime="parameter_denominator_scaled_chain_rule",
        presentation_regime="symbolic_rational_denominator_post_calculus_compact",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_root_symbolic_rational_denominator_affine_radicand_shortcut",
        expr="diff(arctan(sqrt(2*x+2)/(2*a)), x)",
        expected_result="a / (sqrt(2·x + 2)·(2·a^2 + x + 1))",
        expected_required_display=("a ≠ 0", "x > -1"),
        expected_step_substrings=(
            "Usar regla de arctan(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        forbidden_stderr_substrings=("depth_overflow",),
        family="inverse_trig_root",
        argument_regime="symbolic_rational_denominator_affine_radicand",
        domain_regime="required_condition",
        trace_regime="parameter_denominator_scaled_affine_radicand_chain_rule",
        presentation_regime="symbolic_rational_denominator_affine_radicand_compact",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_root_symbolic_rational_denominator_affine_radicand_external_scale_shortcut",
        expr="diff(3*arctan(sqrt(2*x+2)/(2*a)), x)",
        expected_result="3·a / (sqrt(2·x + 2)·(2·a^2 + x + 1))",
        expected_required_display=("a ≠ 0", "x > -1"),
        expected_step_substrings=(
            "Usar factor constante de la derivada",
            "Usar regla de arctan(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        forbidden_stderr_substrings=("depth_overflow",),
        family="inverse_trig_root",
        argument_regime="external_scale_symbolic_rational_denominator_affine_radicand",
        domain_regime="required_condition",
        trace_regime="constant_multiple_parameter_denominator_scaled_affine_radicand_chain_rule",
        presentation_regime="external_scale_symbolic_rational_denominator_affine_radicand_compact",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_root_exact_square_symbolic_denominator_scale_condition_dedupe",
        expr="diff(arctan(sqrt(4*x+4)/a), x)",
        expected_result="a / (sqrt(x + 1)·(4·(x + 1) + a^2))",
        expected_required_display=("a ≠ 0", "x > -1"),
        expected_step_substrings=(
            "Reconocer un cuadrado perfecto bajo la raíz",
            "Sacar constante de una fracción",
            "Usar regla de arctan(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        forbidden_stderr_substrings=("depth_overflow",),
        family="inverse_trig_root",
        argument_regime="exact_square_symbolic_denominator_scaled_shifted_root",
        domain_regime="required_condition",
        trace_regime="parameter_denominator_exact_square_chain_rule",
        presentation_regime="exact_square_symbolic_denominator_condition_deduped_compact",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_root_symbolic_numerator_scale_positive_gap",
        expr="diff(arctan(a*sqrt(x+1)), x)",
        expected_result="a / ((2·x·a^2 + 2·a^2 + 2)·sqrt(x + 1))",
        expected_required_display=("x > -1",),
        expected_step_substrings=(
            "Usar regla de arctan(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="inverse_trig_root",
        argument_regime="symbolic_numerator_scaled_shifted_root",
        domain_regime="required_condition",
        trace_regime="parameter_numerator_scaled_chain_rule",
        presentation_regime="symbolic_numerator_positive_gap_post_calculus_compact",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_root_symbolic_denominator_internal_scale",
        expr="diff(arctan(2*sqrt(x)/a), x)",
        expected_result="a / (sqrt(x)·(a^2 + 4·x))",
        expected_required_display=("a ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar regla de arctan(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="inverse_trig_root",
        argument_regime="symbolic_denominator_internal_scaled_nested_root",
        domain_regime="required_condition",
        trace_regime="parameter_internal_scaled_chain_rule",
        presentation_regime="symbolic_denominator_internal_scale_post_calculus_compact",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_root_symbolic_denominator_scale_dual_orientation",
        expr="diff(arccot(sqrt(x)/a), x)",
        expected_result="-a / (2·sqrt(x)·(a^2 + x))",
        expected_required_display=("x > 0", "a ≠ 0"),
        expected_step_substrings=(
            "arccot(x) → arctan(1/x)",
            "Usar regla de arctan(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="inverse_trig_root",
        argument_regime="dual_symbolic_denominator_scaled_nested_root",
        domain_regime="required_condition",
        trace_regime="dual_parameter_scaled_chain_rule",
        presentation_regime="symbolic_denominator_dual_post_calculus_compact",
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
        name="inverse_trig_root_empty_open_interval_undefined",
        expr="diff(arcsin(sqrt(x^2+1)), x)",
        expected_result="undefined",
        expected_step_substrings=(
            "Detectar dominio real vacío de la derivada de la función inversa",
        ),
        family="inverse_trig_root",
        argument_regime="quadratic_root_argument",
        domain_regime="empty_derivative_domain",
        outcome="undefined",
        trace_regime="empty_derivative_domain_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_shifted_quadratic_empty_open_interval_undefined",
        expr="diff(arcsin((x+1)^2+1), x)",
        expected_result="undefined",
        expected_step_substrings=(
            "Detectar dominio real vacío de la derivada de la función inversa",
        ),
        family="inverse_trig_root",
        argument_regime="shifted_quadratic_argument",
        domain_regime="empty_derivative_domain",
        outcome="undefined",
        trace_regime="empty_derivative_domain_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_arcsin_point_domain_empty_derivative_undefined",
        expr="diff(asin(1+x^2), x)",
        expected_result="undefined",
        expected_step_substrings=(
            "Detectar dominio real vacío de la derivada de la función inversa",
        ),
        family="inverse_trig",
        argument_regime="quadratic_boundary_argument",
        domain_regime="empty_derivative_domain",
        outcome="undefined",
        trace_regime="empty_derivative_domain_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_symbolic_constant_empty_open_interval_undefined",
        expr="diff(arcsin(pi), x)",
        expected_result="undefined",
        expected_step_substrings=("Detectar dominio real vacío de la función inversa",),
        family="inverse_trig",
        argument_regime="symbolic_constant",
        domain_regime="empty_open_interval_domain",
        outcome="undefined",
        trace_regime="empty_open_interval_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="inverse_trig_named_constant_positive_offset_empty_open_interval_undefined",
        expr="diff(arcsin(phi+x^2), x)",
        expected_result="undefined",
        expected_step_substrings=("Detectar dominio real vacío de la función inversa",),
        family="inverse_trig",
        argument_regime="named_positive_offset_quadratic_argument",
        domain_regime="empty_open_interval_domain",
        outcome="undefined",
        trace_regime="empty_open_interval_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="inverse_hyperbolic_atanh_empty_open_interval_undefined",
        expr="diff(atanh(x^2+1), x)",
        expected_result="undefined",
        expected_step_substrings=("Detectar dominio real vacío de la función inversa",),
        family="inverse_hyperbolic",
        argument_regime="quadratic_argument",
        domain_regime="empty_open_interval_domain",
        outcome="undefined",
        trace_regime="empty_open_interval_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="inverse_hyperbolic_atanh_named_constant_positive_offset_empty_open_interval_undefined",
        expr="diff(atanh(phi+x^2), x)",
        expected_result="undefined",
        expected_step_substrings=("Detectar dominio real vacío de la función inversa",),
        family="inverse_hyperbolic",
        argument_regime="named_positive_offset_quadratic_argument",
        domain_regime="empty_open_interval_domain",
        outcome="undefined",
        trace_regime="empty_open_interval_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="inverse_hyperbolic_acosh_empty_lower_bound_undefined",
        expr="diff(acosh(-x^2), x)",
        expected_result="undefined",
        expected_step_substrings=("Detectar dominio real vacío de la función inversa",),
        family="inverse_hyperbolic",
        argument_regime="quadratic_argument",
        domain_regime="empty_lower_bound_domain",
        outcome="undefined",
        trace_regime="empty_lower_bound_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="inverse_hyperbolic_acosh_empty_derivative_domain_undefined",
        expr="diff(acosh(1-x^2), x)",
        expected_result="undefined",
        expected_step_substrings=(
            "Detectar dominio real vacío de la derivada de la función inversa",
        ),
        family="inverse_hyperbolic",
        argument_regime="quadratic_boundary_argument",
        domain_regime="empty_derivative_domain",
        outcome="undefined",
        trace_regime="empty_derivative_domain_policy",
        presentation_regime="undefined",
    ),
    DiffCommandMatrixCase(
        name="inverse_hyperbolic_acosh_named_positive_offset_satisfied_lower_bound",
        expr="diff(acosh(phi+x^2), x)",
        expected_result="2·x·(x^2 + phi - 1)^(-1/2) / sqrt(x^2 + 1 + phi)",
        expected_step_substrings=(
            "Usar regla de la cadena",
            "Identificar u y du",
            "Presentar resultado de cálculo en forma compacta",
        ),
        family="inverse_hyperbolic",
        argument_regime="named_positive_offset_quadratic_argument",
        domain_regime="structurally_satisfied_lower_bound_domain",
        trace_regime="chain_rule",
        presentation_regime="compact_domain_suppressed_radical",
    ),
    DiffCommandMatrixCase(
        name="inverse_hyperbolic_root_atanh_symbolic_numerator_scale_open_interval",
        expr="diff(atanh(a*sqrt(x+1)), x)",
        expected_result="a / ((2 - 2·x·a^2 - 2·a^2)·sqrt(x + 1))",
        expected_required_display=(
            "x > -1",
            "1 - a^2·x - a^2 > 0",
        ),
        expected_step_substrings=(
            "Usar regla de la cadena",
            "Identificar u y du",
            "Presentar resultado de cálculo en forma compacta",
        ),
        family="inverse_hyperbolic_root",
        argument_regime="symbolic_numerator_scaled_shifted_root",
        domain_regime="open_interval_required",
        trace_regime="chain_rule",
        presentation_regime="open_interval_boundary_deduped_compact",
    ),
    DiffCommandMatrixCase(
        name="inverse_hyperbolic_root_atanh_symbolic_denominator_scale_open_interval",
        expr="diff(atanh(sqrt(x+1)/a), x)",
        expected_result="a / ((2·a^2 - 2·x - 2)·sqrt(x + 1))",
        expected_required_display=(
            "x > -1",
            "a ≠ 0",
            "a^2 - x - 1 > 0",
        ),
        expected_step_substrings=(
            "Usar regla de la cadena",
            "Identificar u y du",
            "Presentar resultado de cálculo en forma compacta",
        ),
        forbidden_stderr_substrings=("depth_overflow",),
        family="inverse_hyperbolic_root",
        argument_regime="symbolic_denominator_scaled_shifted_root",
        domain_regime="open_interval_required",
        trace_regime="chain_rule",
        presentation_regime="symbolic_denominator_open_interval_compact",
    ),
    DiffCommandMatrixCase(
        name="inverse_hyperbolic_root_atanh_exact_square_denominator_scale_open_interval",
        expr="diff(atanh(sqrt(4*x+4)/a), x)",
        expected_result="a / (sqrt(x + 1)·(a^2 - 4·x - 4))",
        expected_required_display=(
            "a ≠ 0",
            "a^2 - 4·x - 4 > 0",
            "x > -1",
        ),
        expected_step_substrings=(
            "Reconocer un cuadrado perfecto bajo la raíz",
            "Sacar constante de una fracción",
            "Usar regla de la cadena",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        forbidden_stderr_substrings=("depth_overflow",),
        family="inverse_hyperbolic_root",
        argument_regime="exact_square_symbolic_denominator_scaled_shifted_root",
        domain_regime="open_interval_required",
        trace_regime="chain_rule",
        presentation_regime="exact_square_symbolic_denominator_open_interval_compact",
    ),
    DiffCommandMatrixCase(
        name="inverse_hyperbolic_root_symbolic_numerator_scale_positive_gap",
        expr="diff(asinh(a*sqrt(x+1)), x)",
        expected_result="a·(x·a^2 + a^2 + 1)^(-1/2) / (2·sqrt(x + 1))",
        expected_required_display=("x > -1",),
        expected_step_substrings=(
            "Usar regla de la cadena",
            "Identificar u y du",
            "Presentar resultado de cálculo en forma compacta",
        ),
        family="inverse_hyperbolic_root",
        argument_regime="symbolic_numerator_scaled_shifted_root",
        domain_regime="required_condition",
        trace_regime="chain_rule",
        presentation_regime="positive_gap_domain_condition_compact",
    ),
    DiffCommandMatrixCase(
        name="inverse_hyperbolic_root_asinh_symbolic_denominator_scale_positive_gap",
        expr="diff(asinh(sqrt(x+1)/a), x)",
        expected_result="1 / (2·a·sqrt((a^2 + x + 1) / a^2)·sqrt(x + 1))",
        expected_required_display=(
            "a ≠ 0",
            "x > -1",
        ),
        expected_step_substrings=(
            "Usar regla de la cadena",
            "Identificar u y du",
        ),
        family="inverse_hyperbolic_root",
        argument_regime="symbolic_denominator_scaled_shifted_root",
        domain_regime="positive_gap_with_symbolic_denominator_scale_deduped",
        trace_regime="chain_rule",
        presentation_regime="symbolic_denominator_positive_gap_compact",
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
        name="sqrt_chain_trig_log_presimplified_condition_dedupe",
        expr="diff(-ln(|cos(sqrt(x+0))|), x)",
        expected_result="tan(sqrt(x + 0)) / (2·sqrt(x + 0))",
        expected_required_display=("cos(sqrt(x)) ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Calcular la derivada",
            "Presentar resultado de cálculo en forma compacta",
        ),
        family="trig_log_sqrt_chain",
        argument_regime="presimplified_nested_root",
        domain_regime="required_condition",
        trace_regime="log_abs_trig_chain_rule",
        presentation_regime="same_function_zero_set_condition_dedupe",
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
        name="shifted_linear_base_variable_power_log_domain",
        expr="diff((x+1)^x, x)",
        expected_result="(x·(x + 1)^x + ln(x + 1)·(x + 1)^(x + 1)) / (x + 1)",
        expected_required_display=("x > -1",),
        expected_step_substrings=("Usar derivación logarítmica",),
        family="variable_power",
        argument_regime="shifted_linear_base_variable_power",
        domain_regime="shifted_positive_base_required",
        trace_regime="logarithmic_derivative",
        presentation_regime="shifted_linear_base_fraction",
    ),
    DiffCommandMatrixCase(
        name="quadratic_base_variable_power_disconnected_domain",
        expr="diff((x^2-1)^x, x)",
        expected_result=(
            "(x^2 - 1)^x·(ln(x^2 - 1)·(x^2 - 1) + 2·x^2) "
            "/ (x^2 - 1)"
        ),
        expected_required_display=("x < -1 or x > 1",),
        expected_step_substrings=("Usar derivación logarítmica",),
        family="variable_power",
        argument_regime="quadratic_base_variable_power",
        domain_regime="disconnected_positive_base_required",
        trace_regime="logarithmic_derivative",
        presentation_regime="quadratic_base_fraction",
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
        name="abs_quadratic_factored_nondifferentiable_domain",
        expr="diff(abs(x^2-1), x)",
        expected_result="((x^2 - 1)·x·2)/|x^2 - 1|",
        expected_required_display=("x ≠ 1", "x ≠ -1"),
        expected_step_substrings=(
            "Usar regla de la cadena",
            "Identificar u y du",
        ),
        family="abs",
        argument_regime="quadratic_argument",
        domain_regime="factored_nondifferentiable_points_required",
        trace_regime="piecewise_abs_chain_rule",
        presentation_regime="quotient_abs_factored_domain",
    ),
    DiffCommandMatrixCase(
        name="discontinuous_sign_polynomial_nonzero_domain",
        expr="diff(sign(x), x)",
        expected_result="0",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Usar derivada de sign(u) fuera de u = 0",),
        family="discontinuous",
        argument_regime="variable",
        domain_regime="required_condition",
        trace_regime="sign_derivative",
        presentation_regime="constant_zero",
    ),
    DiffCommandMatrixCase(
        name="discontinuous_sign_quadratic_factored_domain",
        expr="diff(sign(x^2-1), x)",
        expected_result="0",
        expected_required_display=("x ≠ 1", "x ≠ -1"),
        expected_step_substrings=(
            "Usar derivada de sign(u) fuera de u = 0",
            "Identificar u y du",
        ),
        family="discontinuous",
        argument_regime="quadratic_argument",
        domain_regime="factored_discontinuity_points_required",
        trace_regime="sign_derivative_chain_rule",
        presentation_regime="constant_zero_factored_domain",
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


def extract_step_rule_names(payload: dict[str, Any] | None) -> tuple[str, ...]:
    if not payload:
        return ()
    steps = payload.get("steps") or []
    if not isinstance(steps, list):
        return ()

    names: list[str] = []
    seen: set[str] = set()

    def add_name(value: Any) -> None:
        if not isinstance(value, str):
            return
        stripped = value.strip()
        if not stripped or stripped in seen:
            return
        seen.add(stripped)
        names.append(stripped)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            add_name(value.get("rule"))
            add_name(value.get("title"))
            for nested in value.get("substeps") or []:
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(steps)
    return tuple(names)


def extract_timing_us(payload: dict[str, Any] | None, key: str) -> int | None:
    if not payload:
        return None
    timings = payload.get("timings_us")
    if not isinstance(timings, dict):
        return None
    value = timings.get(key)
    if not isinstance(value, int):
        return None
    return value


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
    if "fragile substring" in error:
        return "stderr_fragility"
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
    harness_check_start = time.monotonic()
    parsed, parse_error = parse_json(stdout)
    result = parsed.get("result") if isinstance(parsed, dict) else None
    required_display = extract_required_display(parsed)
    warnings = extract_warning_messages(parsed)
    blocked_hints = extract_blocked_hint_messages(parsed)
    step_text = extract_step_text(parsed)
    step_rule_names = extract_step_rule_names(parsed)
    ok = parsed.get("ok") if isinstance(parsed, dict) else None
    cli_parse_us = extract_timing_us(parsed, "parse_us")
    cli_simplify_us = extract_timing_us(parsed, "simplify_us")
    cli_total_us = extract_timing_us(parsed, "total_us")
    cli_parse_elapsed_seconds = (
        round(cli_parse_us / 1_000_000.0, 6) if cli_parse_us is not None else None
    )
    cli_simplify_elapsed_seconds = (
        round(cli_simplify_us / 1_000_000.0, 6)
        if cli_simplify_us is not None
        else None
    )
    cli_total_elapsed_seconds = (
        round(cli_total_us / 1_000_000.0, 6) if cli_total_us is not None else None
    )
    cli_public_overhead_seconds = (
        round(max(0.0, wall_elapsed - cli_total_us / 1_000_000.0), 6)
        if cli_total_us is not None
        else None
    )

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
    elif (
        stderr_error := stderr_fragility_error(
            stderr,
            forbidden_substrings=case.forbidden_stderr_substrings,
        )
    ) is not None:
        error = stderr_error
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

    harness_check_elapsed = time.monotonic() - harness_check_start
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
        "process_elapsed_seconds": round(wall_elapsed, 3),
        "harness_check_elapsed_seconds": round(harness_check_elapsed, 3),
        "cli_parse_us": cli_parse_us,
        "cli_simplify_us": cli_simplify_us,
        "cli_total_us": cli_total_us,
        "cli_parse_elapsed_seconds": cli_parse_elapsed_seconds,
        "cli_simplify_elapsed_seconds": cli_simplify_elapsed_seconds,
        "cli_total_elapsed_seconds": cli_total_elapsed_seconds,
        "cli_public_overhead_seconds": cli_public_overhead_seconds,
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "step_text_char_count": len(step_text),
        "step_rule_names": list(step_rule_names),
        "steps_count": len(parsed.get("steps") or []) if isinstance(parsed, dict) else 0,
        "result": result,
        "expected_result": case.expected_result,
        "required_display": list(required_display),
        "expected_required_display": list(case.expected_required_display),
        "warnings": list(warnings),
        "expected_warning_substrings": list(case.expected_warning_substrings),
        "blocked_hints": list(blocked_hints),
        "expected_blocked_hint_substrings": list(case.expected_blocked_hint_substrings),
        "expected_step_substrings": list(case.expected_step_substrings),
        "forbidden_stderr_substrings": list(case.forbidden_stderr_substrings),
        "family": case.family,
        "argument_regime": case.argument_regime,
        "domain_regime": case.domain_regime,
        "outcome": case.outcome,
        "calculus_maturity_block": calculus_maturity_block(case),
        "calculus_block_gate": calculus_block_gate(case),
        "positive_quadratic_policy_cluster": positive_quadratic_policy_cluster(case),
        "variable_power_policy_cluster": variable_power_policy_cluster(case),
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


def calculus_maturity_block(case: DiffCommandMatrixCase) -> str:
    if case.outcome in {"residual", "undefined"}:
        return "block9_residuals_and_non_goals"
    return "block2_real_domain_differentiation"


def calculus_block_gate(case: DiffCommandMatrixCase) -> str:
    if case.outcome == "residual":
        return "safe_residual_policy"
    if case.outcome == "undefined":
        return "explicit_undefined_domain_policy"
    if case.expected_required_display:
        return "domain_conditions_and_diff_policy"
    if case.expected_step_substrings:
        return "didactic_trace_and_diff_policy"
    return "supported_diff_policy"


def count_calculus_maturity_blocks(
    cases: tuple[DiffCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        block = calculus_maturity_block(case)
        counts[block] = counts.get(block, 0) + 1
    return dict(sorted(counts.items()))


def count_calculus_block_gates(
    cases: tuple[DiffCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        gate = calculus_block_gate(case)
        counts[gate] = counts.get(gate, 0) + 1
    return dict(sorted(counts.items()))


def symbolic_radius_policy_cluster(case: DiffCommandMatrixCase) -> str | None:
    if (
        case.family == "positive_quadratic_arctan_primitive"
        and case.domain_regime == "symbolic_radius_minimal_nonzero_parameter"
        and case.presentation_regime
        == "compact_shifted_symbolic_radius_positive_quadratic"
    ):
        return "block2_symbolic_radius_arctan_positive_quadratic"
    return None


def positive_quadratic_policy_cluster(case: DiffCommandMatrixCase) -> str | None:
    if case.family == "positive_quadratic_log_arctan_primitive":
        return "block2_positive_quadratic_log_arctan_primitive"
    if case.family == "positive_quadratic_log_abs_pole_primitive":
        return "block2_positive_quadratic_log_abs_pole_primitive"
    return symbolic_radius_policy_cluster(case)


def variable_power_policy_cluster(case: DiffCommandMatrixCase) -> str | None:
    if case.family == "variable_power":
        return "block2_variable_power_logarithmic_derivative_domain"
    return None


def count_symbolic_radius_policy_clusters(
    cases: tuple[DiffCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        cluster = symbolic_radius_policy_cluster(case)
        if cluster is None:
            continue
        counts[cluster] = counts.get(cluster, 0) + 1
    return dict(sorted(counts.items()))


def count_positive_quadratic_policy_clusters(
    cases: tuple[DiffCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        cluster = positive_quadratic_policy_cluster(case)
        if cluster is None:
            continue
        counts[cluster] = counts.get(cluster, 0) + 1
    return dict(sorted(counts.items()))


def count_variable_power_policy_clusters(
    cases: tuple[DiffCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        cluster = variable_power_policy_cluster(case)
        if cluster is None:
            continue
        counts[cluster] = counts.get(cluster, 0) + 1
    return dict(sorted(counts.items()))


def count_required_display_items(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for item in result.get("required_display", []):
            if not isinstance(item, str):
                continue
            counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items()))


def phase_runtime_distribution(
    results: list[dict[str, Any]],
    *,
    phase_key: str,
) -> dict[str, Any]:
    elapsed_values = sorted(
        float(result[phase_key])
        for result in results
        if isinstance(result.get(phase_key), (int, float))
    )
    if not elapsed_values:
        return {}

    total_elapsed = sum(elapsed_values)
    p95_index = max(0, (95 * len(elapsed_values) + 99) // 100 - 1)
    return {
        "timed_case_count": len(elapsed_values),
        "total_elapsed_seconds": round(total_elapsed, 3),
        "avg_case_ms": round(total_elapsed * 1000.0 / len(elapsed_values), 3),
        "p95_case_ms": round(elapsed_values[p95_index] * 1000.0, 3),
        "max_case_ms": round(elapsed_values[-1] * 1000.0, 3),
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
            "argument_regime",
            "domain_regime",
            "trace_regime",
            "presentation_regime",
            "calculus_maturity_block",
            "calculus_block_gate",
            "positive_quadratic_policy_cluster",
            "variable_power_policy_cluster",
        ):
            value = result.get(key)
            if isinstance(value, str):
                row[key] = value
        steps_count = result.get("steps_count")
        if isinstance(steps_count, int):
            row["steps_count"] = steps_count
        step_rule_names = result.get("step_rule_names")
        if isinstance(step_rule_names, list):
            names = [value for value in step_rule_names if isinstance(value, str)]
            if names:
                row["step_rule_names"] = names[:8]
        rows.append(row)
    return rows


def phase_runtime_observability_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    process_rows = phase_runtime_case_rows(
        results,
        phase_key="process_elapsed_seconds",
        output_key="process_elapsed_seconds",
    )
    if process_rows:
        summary["slowest_process_evaluations"] = process_rows

    harness_rows = phase_runtime_case_rows(
        results,
        phase_key="harness_check_elapsed_seconds",
        output_key="harness_check_elapsed_seconds",
    )
    if harness_rows:
        summary["harness_check_runtime_distribution"] = phase_runtime_distribution(
            results,
            phase_key="harness_check_elapsed_seconds",
        )
        summary["slowest_harness_checks"] = harness_rows
    for phase_name, phase_key, output_key in (
        (
            "cli_parse",
            "cli_parse_elapsed_seconds",
            "cli_parse_elapsed_seconds",
        ),
        (
            "cli_simplify",
            "cli_simplify_elapsed_seconds",
            "cli_simplify_elapsed_seconds",
        ),
        (
            "cli_total",
            "cli_total_elapsed_seconds",
            "cli_total_elapsed_seconds",
        ),
        (
            "cli_public_overhead",
            "cli_public_overhead_seconds",
            "cli_public_overhead_seconds",
        ),
    ):
        rows = phase_runtime_case_rows(
            results,
            phase_key=phase_key,
            output_key=output_key,
        )
        if not rows:
            continue
        summary[f"{phase_name}_runtime_distribution"] = phase_runtime_distribution(
            results,
            phase_key=phase_key,
        )
        summary[f"slowest_{phase_name}_evaluations"] = rows
    return summary


def exact_square_runtime_pair_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {
        result.get("name"): result
        for result in results
        if isinstance(result.get("name"), str)
    }
    rows: list[dict[str, Any]] = []
    for pair in EXACT_SQUARE_RUNTIME_PAIR_CASES:
        baseline = by_name.get(pair["baseline_case"])
        exact_square = by_name.get(pair["exact_square_case"])
        if baseline is None or exact_square is None:
            continue
        baseline_elapsed = baseline.get("wall_elapsed_seconds")
        exact_square_elapsed = exact_square.get("wall_elapsed_seconds")
        if not isinstance(baseline_elapsed, (int, float)) or not isinstance(
            exact_square_elapsed,
            (int, float),
        ):
            continue
        baseline_ms = float(baseline_elapsed) * 1000.0
        exact_square_ms = float(exact_square_elapsed) * 1000.0
        row: dict[str, Any] = {
            "name": pair["name"],
            "baseline_case": pair["baseline_case"],
            "exact_square_case": pair["exact_square_case"],
            "baseline_case_ms": round(baseline_ms, 3),
            "exact_square_case_ms": round(exact_square_ms, 3),
            "delta_ms": round(exact_square_ms - baseline_ms, 3),
        }
        if baseline_ms > 0.0:
            row["ratio"] = round(exact_square_ms / baseline_ms, 3)
        for source, prefix in (
            (baseline, "baseline"),
            (exact_square, "exact_square"),
        ):
            for key in ("argument_regime", "presentation_regime"):
                value = source.get(key)
                if isinstance(value, str):
                    row[f"{prefix}_{key}"] = value
        family = exact_square.get("family")
        if isinstance(family, str):
            row["family"] = family
        rows.append(row)
    return rows


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
    residual_case_names = [case.name for case in cases if case.outcome == "residual"]
    return {
        "status": status,
        "total": len(results),
        "status_counts": status_counts,
        "issue_kind_counts": dict(sorted(issue_kind_counts.items())),
        "problem_case_count": len(problem_cases),
        "problem_cases": problem_cases,
        "supported_case_count": sum(1 for case in cases if case.outcome == "supported"),
        "residual_case_count": sum(1 for case in cases if case.outcome == "residual"),
        "residual_case_names": residual_case_names,
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
        "calculus_maturity_block_counts": count_calculus_maturity_blocks(cases),
        "calculus_block_gate_counts": count_calculus_block_gates(cases),
        "symbolic_radius_policy_cluster_counts": (
            count_symbolic_radius_policy_clusters(cases)
        ),
        "positive_quadratic_policy_cluster_counts": (
            count_positive_quadratic_policy_clusters(cases)
        ),
        "variable_power_policy_cluster_counts": (
            count_variable_power_policy_clusters(cases)
        ),
        "trace_regime_counts": count_by(cases, "trace_regime"),
        "presentation_regime_counts": count_by(cases, "presentation_regime"),
        "case_filters": [case.name for case in cases],
        **runtime_observability_summary(
            results,
            group_keys=(
                "family",
                "calculus_maturity_block",
                "calculus_block_gate",
                "domain_regime",
                "trace_regime",
                "presentation_regime",
                "positive_quadratic_policy_cluster",
                "variable_power_policy_cluster",
            ),
        ),
        **phase_runtime_observability_summary(results),
        **payload_observability_summary(results),
        "exact_square_runtime_pairs": exact_square_runtime_pair_rows(results),
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
