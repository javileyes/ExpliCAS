#!/usr/bin/env python3
"""Command-level integration policy matrix smoke.

This lane complements the broad Rust integration contract. It keeps a small
public `integrate(...)` support matrix visible at scorecard level so future
calculus work can be selected by family, argument, domain, trace, and
presentation regime instead of by raw test counts alone.
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
from typing import Any, Callable, Literal


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


@dataclass(frozen=True)
class IntegrateCommandMatrixCase:
    name: str
    expr: str
    expected_result: str
    expected_derivative_result: str | None = None
    expected_derivative_equivalent_to: str | None = None
    expected_derivative_required_display: tuple[str, ...] = ()
    expected_direct_diff_integrate_result: str | None = None
    expected_direct_diff_integrate_equivalent_to: str | None = None
    expected_direct_diff_integrate_required_display: tuple[str, ...] = ()
    direct_diff_integrate_from_derivative: bool = False
    expected_required_display: tuple[str, ...] = ()
    expected_warning_substrings: tuple[str, ...] = ()
    expected_step_substrings: tuple[str, ...] = ()
    family: str = "unknown"
    argument_regime: str = "variable"
    domain_regime: str = "unconditional"
    outcome: str = "supported"
    residual_cause: str = "not_applicable"
    trace_regime: str = "direct"
    presentation_regime: str = "canonical"


def direct_diff_integrate_expected_result(
    case: IntegrateCommandMatrixCase,
) -> str | None:
    if case.direct_diff_integrate_from_derivative:
        return case.expected_derivative_result
    return case.expected_direct_diff_integrate_result


def direct_diff_integrate_expected_equivalent_to(
    case: IntegrateCommandMatrixCase,
) -> str | None:
    if case.direct_diff_integrate_from_derivative:
        return case.expected_derivative_equivalent_to
    return case.expected_direct_diff_integrate_equivalent_to


def direct_diff_integrate_expected_required_display(
    case: IntegrateCommandMatrixCase,
) -> tuple[str, ...]:
    if case.direct_diff_integrate_from_derivative:
        return case.expected_derivative_required_display
    return case.expected_direct_diff_integrate_required_display


def has_direct_diff_integrate_probe(case: IntegrateCommandMatrixCase) -> bool:
    return (
        direct_diff_integrate_expected_result(case) is not None
        or direct_diff_integrate_expected_equivalent_to(case) is not None
    )


def validate_direct_diff_integrate_expectations(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> None:
    for case in cases:
        if not case.direct_diff_integrate_from_derivative:
            continue
        if (
            case.expected_direct_diff_integrate_result is not None
            or case.expected_direct_diff_integrate_equivalent_to is not None
            or case.expected_direct_diff_integrate_required_display
        ):
            raise SystemExit(
                f"{case.name} mixes inherited and explicit direct diff(integrate) expectations"
            )
        if (
            case.expected_derivative_result is None
            and case.expected_derivative_equivalent_to is None
        ):
            raise SystemExit(
                f"{case.name} inherits direct diff(integrate) expectations without derivative expectations"
            )


@dataclass(frozen=True)
class ResidualShapeOrientationProbe:
    name: str
    expr: str
    expression_shape: str
    orientation: str


RESIDUAL_SHAPE_ORIENTATION_PROBES = (
    ResidualShapeOrientationProbe(
        name="shifted_tangent_log_source",
        expr="integrate(1/((tan(x)-2)*ln(tan(x))), x)",
        expression_shape="source",
        orientation="tan_minus_offset",
    ),
    ResidualShapeOrientationProbe(
        name="shifted_tangent_log_factored_residual",
        expr=(
            "integrate(cos(x)/(ln(sin(x)/cos(x))*(sin(x)-2*cos(x))), x)"
        ),
        expression_shape="factored_residual",
        orientation="tan_minus_offset",
    ),
    ResidualShapeOrientationProbe(
        name="shifted_cotangent_log_source",
        expr="integrate(1/((cot(x)-2)*ln(cot(x))), x)",
        expression_shape="source",
        orientation="cot_minus_offset",
    ),
    ResidualShapeOrientationProbe(
        name="shifted_cotangent_log_factored_residual",
        expr=(
            "integrate(sin(x)/(ln(cos(x)/sin(x))*(cos(x)-2*sin(x))), x)"
        ),
        expression_shape="factored_residual",
        orientation="cot_minus_offset",
    ),
    ResidualShapeOrientationProbe(
        name="offset_tangent_log_source",
        expr="integrate(1/((2-tan(x))*ln(tan(x))), x)",
        expression_shape="source",
        orientation="offset_minus_tan",
    ),
    ResidualShapeOrientationProbe(
        name="offset_tangent_log_factored_residual",
        expr=(
            "integrate(cos(x)/(ln(sin(x)/cos(x))*(2*cos(x)-sin(x))), x)"
        ),
        expression_shape="factored_residual",
        orientation="offset_minus_tan",
    ),
    ResidualShapeOrientationProbe(
        name="offset_cotangent_log_source",
        expr="integrate(1/((2-cot(x))*ln(cot(x))), x)",
        expression_shape="source",
        orientation="offset_minus_cot",
    ),
    ResidualShapeOrientationProbe(
        name="offset_cotangent_log_factored_residual",
        expr=(
            "integrate(sin(x)/(ln(cos(x)/sin(x))*(2*sin(x)-cos(x))), x)"
        ),
        expression_shape="factored_residual",
        orientation="offset_minus_cot",
    ),
    ResidualShapeOrientationProbe(
        name="symbolic_shifted_tangent_log_source",
        expr="integrate(1/((tan(x)-a)*ln(tan(x))), x)",
        expression_shape="source",
        orientation="tan_minus_symbolic_offset",
    ),
    ResidualShapeOrientationProbe(
        name="symbolic_offset_tangent_log_source",
        expr="integrate(1/((a-tan(x))*ln(tan(x))), x)",
        expression_shape="source",
        orientation="symbolic_offset_minus_tan",
    ),
    ResidualShapeOrientationProbe(
        name="symbolic_shifted_cotangent_log_source",
        expr="integrate(1/((cot(x)-a)*ln(cot(x))), x)",
        expression_shape="source",
        orientation="cot_minus_symbolic_offset",
    ),
    ResidualShapeOrientationProbe(
        name="symbolic_offset_cotangent_log_source",
        expr="integrate(1/((a-cot(x))*ln(cot(x))), x)",
        expression_shape="source",
        orientation="symbolic_offset_minus_cot",
    ),
)


DEFAULT_INTEGRATE_COMMAND_MATRIX_CASES = (
    IntegrateCommandMatrixCase(
        name="polynomial_power_direct",
        expr="integrate(x^2, x)",
        expected_result="1/3·x^3",
        expected_derivative_result="x^2",
        direct_diff_integrate_from_derivative=True,
        expected_step_substrings=("Usar regla de potencia para integrales",),
        family="polynomial",
        argument_regime="variable_power",
        trace_regime="power_rule",
        presentation_regime="compact_power",
    ),
    IntegrateCommandMatrixCase(
        name="polynomial_sum_linearity",
        expr="integrate(2*x + 3, x)",
        expected_result="x^2 + 3·x",
        expected_derivative_result="2·x + 3",
        direct_diff_integrate_from_derivative=True,
        expected_step_substrings=(
            "Usar linealidad de la integral",
            "Integrar cada término",
        ),
        family="polynomial_sum",
        argument_regime="sum",
        trace_regime="linearity",
        presentation_regime="polynomial_sum",
    ),
    IntegrateCommandMatrixCase(
        name="affine_trig_substitution",
        expr="integrate(sin(2*x), x)",
        expected_result="-1/2·cos(2·x)",
        expected_derivative_result="sin(2·x)",
        direct_diff_integrate_from_derivative=True,
        expected_step_substrings=(
            "Usar la regla de sin con derivada interna",
            "Identificar el argumento afín",
            "Ajustar el factor constante",
        ),
        family="trig_affine",
        argument_regime="affine_argument",
        trace_regime="linear_substitution",
        presentation_regime="scaled_trig",
    ),
    IntegrateCommandMatrixCase(
        name="constant_multiple_affine_trig_substitution",
        expr="integrate(3*cos(2*x+1), x)",
        expected_result="3/2·sin(2·x + 1)",
        expected_derivative_result="3·cos(2·x + 1)",
        direct_diff_integrate_from_derivative=True,
        expected_step_substrings=(
            "Usar la regla de cos(u) -> sin(u)",
            "Identificar u y du",
            "Ajustar el factor constante",
        ),
        family="trig_affine",
        argument_regime="external_constant_affine_argument",
        trace_regime="constant_multiple_linear_substitution",
        presentation_regime="external_constant_scaled_trig",
    ),
    IntegrateCommandMatrixCase(
        name="affine_secant_table_log_domain",
        expr="integrate(sec(2*x+1), x)",
        expected_result="1/2·ln(|tan(2·x + 1) + sec(2·x + 1)|)",
        expected_derivative_equivalent_to="sec(2*x+1)",
        expected_derivative_required_display=("cos(2·x + 1) ≠ 0",),
        expected_direct_diff_integrate_result="sec(2·x + 1)",
        expected_direct_diff_integrate_required_display=("cos(2·x + 1) ≠ 0",),
        expected_required_display=("cos(2·x + 1) ≠ 0",),
        expected_step_substrings=(
            "Expandir secante como recíproco de coseno",
            "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
            "Identificar el argumento afín",
        ),
        family="trig_reciprocal_log_table",
        argument_regime="affine_reciprocal_trig_table",
        domain_regime="trig_reciprocal_table_pole_required",
        trace_regime="affine_secant_log_table",
        presentation_regime="abs_log_reciprocal_trig_affine",
    ),
    IntegrateCommandMatrixCase(
        name="affine_cosecant_table_log_domain",
        expr="integrate(csc(2*x+1), x)",
        expected_result="1/2·ln(|csc(2·x + 1) - cot(2·x + 1)|)",
        expected_derivative_equivalent_to="csc(2*x+1)",
        expected_derivative_required_display=("sin(2·x + 1) ≠ 0",),
        expected_direct_diff_integrate_result="csc(2·x + 1)",
        expected_direct_diff_integrate_required_display=("sin(2·x + 1) ≠ 0",),
        expected_required_display=("sin(2·x + 1) ≠ 0",),
        expected_step_substrings=(
            "Expandir cosecante como recíproco de seno",
            "Usar la regla de csc(u) -> ln|csc(u)-cot(u)|",
            "Identificar el argumento afín",
        ),
        family="trig_reciprocal_log_table",
        argument_regime="affine_reciprocal_trig_table",
        domain_regime="trig_reciprocal_table_pole_required",
        trace_regime="affine_cosecant_log_table",
        presentation_regime="abs_log_reciprocal_trig_affine",
    ),
    IntegrateCommandMatrixCase(
        name="external_constant_affine_secant_table_log_domain",
        expr="integrate(3*sec(2*x+1), x)",
        expected_result="3/2·ln(|tan(2·x + 1) + sec(2·x + 1)|)",
        expected_derivative_equivalent_to="3*sec(2*x+1)",
        expected_derivative_required_display=("cos(2·x + 1) ≠ 0",),
        expected_direct_diff_integrate_result="3·sec(2·x + 1)",
        expected_direct_diff_integrate_required_display=("cos(2·x + 1) ≠ 0",),
        expected_required_display=("cos(2·x + 1) ≠ 0",),
        expected_step_substrings=(
            "Expandir secante como recíproco de coseno",
            "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
            "Identificar el argumento afín",
            "Ajustar el factor constante",
        ),
        family="trig_reciprocal_log_table",
        argument_regime="external_constant_affine_reciprocal_trig_table",
        domain_regime="trig_reciprocal_table_pole_required",
        trace_regime="external_constant_affine_secant_log_table",
        presentation_regime="external_constant_abs_log_reciprocal_trig_affine",
    ),
    IntegrateCommandMatrixCase(
        name="rational_denominator_scaled_affine_secant_table_log_domain",
        expr="integrate(2/3*sec(2*x+1), x)",
        expected_result="ln(|tan(2·x + 1) + sec(2·x + 1)|) / 3",
        expected_derivative_equivalent_to="2/3*sec(2*x+1)",
        expected_derivative_required_display=("cos(2·x + 1) ≠ 0",),
        expected_direct_diff_integrate_result="(2·sec(2·x + 1))/3",
        expected_direct_diff_integrate_required_display=("cos(2·x + 1) ≠ 0",),
        expected_required_display=("cos(2·x + 1) ≠ 0",),
        expected_step_substrings=(
            "Expandir secante como recíproco de coseno",
            "Combinar fracciones en una multiplicación",
            "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
            "Identificar el argumento afín",
            "Ajustar el factor constante",
        ),
        family="trig_reciprocal_log_table",
        argument_regime="rational_denominator_scaled_affine_reciprocal_trig_table",
        domain_regime="trig_reciprocal_table_pole_required",
        trace_regime="rational_denominator_scaled_affine_secant_log_table",
        presentation_regime="rational_denominator_scaled_abs_log_reciprocal_trig_affine",
    ),
    IntegrateCommandMatrixCase(
        name="rational_denominator_scaled_affine_cosecant_table_log_domain",
        expr="integrate(2/3*csc(2*x+1), x)",
        expected_result="ln(|csc(2·x + 1) - cot(2·x + 1)|) / 3",
        expected_derivative_equivalent_to="2/3*csc(2*x+1)",
        expected_derivative_required_display=("sin(2·x + 1) ≠ 0",),
        expected_direct_diff_integrate_result="(2·csc(2·x + 1))/3",
        expected_direct_diff_integrate_required_display=("sin(2·x + 1) ≠ 0",),
        expected_required_display=("sin(2·x + 1) ≠ 0",),
        expected_step_substrings=(
            "Expandir cosecante como recíproco de seno",
            "Combinar fracciones en una multiplicación",
            "Usar la regla de csc(u) -> ln|csc(u)-cot(u)|",
            "Identificar el argumento afín",
            "Ajustar el factor constante",
        ),
        family="trig_reciprocal_log_table",
        argument_regime="rational_denominator_scaled_affine_reciprocal_trig_table",
        domain_regime="trig_reciprocal_table_pole_required",
        trace_regime="rational_denominator_scaled_affine_cosecant_log_table",
        presentation_regime="rational_denominator_scaled_abs_log_reciprocal_trig_affine",
    ),
    IntegrateCommandMatrixCase(
        name="affine_exp_substitution",
        expr="integrate(exp(2*x+1), x)",
        expected_result="1/2·e^(2·x + 1)",
        expected_derivative_result="e^(2·x + 1)",
        direct_diff_integrate_from_derivative=True,
        expected_step_substrings=(
            "Usar la regla de exp con derivada interna",
            "Identificar el argumento afín",
            "Ajustar el factor constante",
        ),
        family="exp_affine",
        argument_regime="affine_argument",
        trace_regime="exp_affine_substitution",
        presentation_regime="exponential",
    ),
    IntegrateCommandMatrixCase(
        name="linear_exp_by_parts",
        expr="integrate(x*exp(x), x)",
        expected_result="(x - 1)·e^x",
        expected_derivative_result="x·e^x",
        direct_diff_integrate_from_derivative=True,
        expected_step_substrings=("Usar integración por partes",),
        family="by_parts_exp",
        argument_regime="linear_exp_product",
        trace_regime="by_parts_exp",
        presentation_regime="compact_exp_product",
    ),
    IntegrateCommandMatrixCase(
        name="linear_exp_affine_slope_by_parts",
        expr="integrate(x*exp(2*x), x)",
        expected_result="(1/2·x - 1/4)·e^(2·x)",
        expected_derivative_result="x·e^(2·x)",
        direct_diff_integrate_from_derivative=True,
        expected_step_substrings=("Usar integración por partes",),
        family="by_parts_exp",
        argument_regime="linear_product_affine_exp_argument",
        trace_regime="by_parts_exp_affine_slope",
        presentation_regime="compact_affine_exp_product",
    ),
    IntegrateCommandMatrixCase(
        name="polynomial_exp_derivative_substitution",
        expr="integrate(2*x*exp(x^2), x)",
        expected_result="e^(x^2)",
        expected_derivative_result="2·x·e^(x^2)",
        direct_diff_integrate_from_derivative=True,
        expected_step_substrings=(
            "Usar la regla de exp(u) -> exp(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="exp_polynomial_derivative",
        argument_regime="nonlinear_polynomial_derivative",
        trace_regime="polynomial_derivative_substitution",
        presentation_regime="exponential",
    ),
    IntegrateCommandMatrixCase(
        name="log_power_product_substitution",
        expr="integrate(2*x*ln(x^2+1)^2, x)",
        expected_result="(x^2 + 1)·(ln(x^2 + 1)^2 - 2·ln(x^2 + 1) + 2)",
        expected_derivative_result="2·x·ln(x^2 + 1)^2",
        direct_diff_integrate_from_derivative=True,
        expected_step_substrings=(
            "Usar la regla de u'·ln(u)^n por partes",
            "Identificar u y du",
        ),
        family="log_power_product_substitution",
        argument_regime="polynomial_log_power_product_derivative",
        domain_regime="structurally_positive_log_argument",
        trace_regime="log_power_product_substitution",
        presentation_regime="log_power_by_parts_product",
    ),
    IntegrateCommandMatrixCase(
        name="constant_base_log_power_product_positive_domain_substitution",
        expr="integrate(2*x*log(2,x^2-1)^2, x)",
        expected_result="(x^2 - 1)·(log(2, x^2 - 1)^2 + 2 / ln(2)^2 - 2·log(2, x^2 - 1) / ln(2))",
        expected_derivative_equivalent_to="2*x*log(2,x^2-1)^2",
        expected_direct_diff_integrate_result="2·x·log(2, x^2 - 1)^2",
        expected_direct_diff_integrate_required_display=("x < -1 or x > 1",),
        expected_derivative_required_display=("x < -1 or x > 1",),
        expected_required_display=("x < -1 or x > 1",),
        expected_step_substrings=(
            "Usar la regla de u'·log_b(u)^n por partes",
            "Identificar u y du",
        ),
        family="log_power_product_substitution",
        argument_regime="constant_base_polynomial_log_power_product_derivative",
        domain_regime="positive_log_argument_required",
        trace_regime="constant_base_log_power_product_substitution",
        presentation_regime="constant_base_log_power_by_parts_product",
    ),
    IntegrateCommandMatrixCase(
        name="reciprocal_affine_log_abs_domain",
        expr="integrate(1/(2*x + 1), x)",
        expected_result="1/2·ln(|2·x + 1|)",
        expected_derivative_result="1 / (2·x + 1)",
        expected_derivative_required_display=("x ≠ -1/2",),
        direct_diff_integrate_from_derivative=True,
        expected_required_display=("x ≠ -1/2",),
        expected_step_substrings=(
            "Usar la regla de ln|u| con derivada interna",
            "Identificar el denominador afín",
            "Ajustar el factor constante",
        ),
        family="reciprocal_log",
        argument_regime="rational_expression",
        domain_regime="nonzero_required",
        trace_regime="log_reciprocal_derivative",
        presentation_regime="abs_log",
    ),
    IntegrateCommandMatrixCase(
        name="reciprocal_negative_affine_log_abs_domain",
        expr="integrate(1/(1-2*x), x)",
        expected_result="-1/2·ln(|1 - 2·x|)",
        expected_derivative_result="1 / (1 - 2·x)",
        expected_derivative_required_display=("x ≠ 1/2",),
        direct_diff_integrate_from_derivative=True,
        expected_required_display=("x ≠ 1/2",),
        expected_step_substrings=(
            "Usar la regla de ln|u| con derivada interna",
            "Identificar el denominador afín",
            "Ajustar el factor constante",
        ),
        family="reciprocal_log",
        argument_regime="negative_affine_rational_expression",
        domain_regime="nonzero_required",
        trace_regime="log_reciprocal_derivative",
        presentation_regime="signed_abs_log",
    ),
    IntegrateCommandMatrixCase(
        name="reciprocal_negative_affine_derivative_log_abs_domain",
        expr="integrate(-2/(1-2*x), x)",
        expected_result="ln(|1 - 2·x|)",
        expected_derivative_result="-2 / (1 - 2·x)",
        expected_derivative_required_display=("x ≠ 1/2",),
        direct_diff_integrate_from_derivative=True,
        expected_required_display=("x ≠ 1/2",),
        expected_step_substrings=(
            "Usar la regla de ln|u| con derivada interna",
            "Identificar el denominador afín",
            "Ajustar el factor constante",
        ),
        family="reciprocal_log",
        argument_regime="negative_affine_derivative_rational_expression",
        domain_regime="nonzero_required",
        trace_regime="log_reciprocal_exact_derivative",
        presentation_regime="abs_log_exact_derivative",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_rational_affine_quotient_numeric_slope",
        expr="integrate((3*x+c)/(2*x+b), x)",
        expected_result="1/2·ln(|b + 2·x|)·(c - 3/2·b) + 3/2·x",
        expected_direct_diff_integrate_result="(c + 3·x) / (b + 2·x)",
        expected_direct_diff_integrate_required_display=("b + 2·x ≠ 0",),
        expected_required_display=("b + 2·x ≠ 0",),
        family="algorithmic_backend_rational_affine_quotient",
        argument_regime="linear_numerator_affine_denominator_numeric_slope",
        domain_regime="backend_verified_denominator_nonzero_required",
        trace_regime="algorithmic_backend_rational_summary",
        presentation_regime="backend_summary_log_affine_quotient",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_rational_affine_quotient_symbolic_slope",
        expr="integrate((3*x+c)/(2*a*x+b), x)",
        expected_result="(3·x)/(2·a) + ((c - (3·b)/(2·a))·ln(|2·a·x + b|))/(2·a)",
        expected_direct_diff_integrate_result="(c + 3·x) / (2·a·x + b)",
        expected_direct_diff_integrate_required_display=("2·a·x + b ≠ 0", "a ≠ 0"),
        expected_required_display=("2·a·x + b ≠ 0", "a ≠ 0"),
        family="algorithmic_backend_rational_affine_quotient",
        argument_regime="linear_numerator_affine_denominator_symbolic_slope",
        domain_regime="backend_verified_denominator_and_slope_nonzero_required",
        trace_regime="algorithmic_backend_rational_summary",
        presentation_regime="backend_summary_log_affine_quotient_symbolic_slope",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_rational_affine_quotient_external_scale_zero_intercept",
        expr="integrate(a*x/(c*x+d), x)",
        expected_result="(a·x)/c - a·d·ln(|c·x + d|)/(c·c)",
        expected_direct_diff_integrate_result="a·x / (c·x + d)",
        expected_direct_diff_integrate_required_display=("c ≠ 0", "c·x + d ≠ 0"),
        expected_required_display=("c ≠ 0", "c·x + d ≠ 0"),
        family="algorithmic_backend_rational_affine_quotient",
        argument_regime="external_scaled_zero_intercept_linear_numerator_affine_denominator_symbolic_slope",
        domain_regime="backend_verified_denominator_and_slope_nonzero_required",
        trace_regime="algorithmic_backend_rational_summary",
        presentation_regime="backend_summary_log_affine_quotient_external_scale_zero_intercept",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_hermite_reciprocal_educational_substeps",
        expr="integrate(c/((x+b)^2+a), x)",
        expected_result="(arctan((b + x) / sqrt(a))·c)/sqrt(a)",
        expected_direct_diff_integrate_result="c / ((b + x)^2 + a)",
        expected_direct_diff_integrate_required_display=("a > 0",),
        expected_required_display=("a > 0",),
        expected_step_substrings=(
            "Usar la regla de arctan con derivada interna",
            "Identificar el argumento afín",
            "Ajustar el factor constante",
        ),
        family="algorithmic_backend_hermite_positive_quadratic",
        argument_regime="scaled_reciprocal_positive_quadratic_symbolic_radius",
        domain_regime="backend_verified_positive_radius_required",
        trace_regime="algorithmic_backend_hermite_educational_substeps",
        presentation_regime="backend_arctan_symbolic_radius_with_substeps",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_hermite_symbolic_positive_radius_mixed_numerator",
        expr="integrate((x+1)/(x^2+a), x)",
        expected_result="1/2·ln(x^2 + a) + (1·arctan(x / sqrt(a)))/sqrt(a)",
        expected_direct_diff_integrate_result="(x + 1) / (x^2 + a)",
        expected_direct_diff_integrate_required_display=("a > 0",),
        expected_required_display=("a > 0",),
        expected_step_substrings=(
            "Separar la parte logarítmica del numerador",
            "Integrar la derivada del denominador como logaritmo",
            "Usar la regla de arctan con derivada interna",
        ),
        family="algorithmic_backend_hermite_positive_quadratic",
        argument_regime="linear_numerator_unit_positive_quadratic_symbolic_radius",
        domain_regime="backend_verified_positive_radius_required",
        trace_regime="algorithmic_backend_hermite_summary",
        presentation_regime="backend_summary_log_arctan_symbolic_positive_radius",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_hermite_symbolic_affine_positive_radius_mixed_numerator",
        expr="integrate((m*(s*x+b)+c)/((s*x+b)^2+a), x)",
        expected_result=(
            "(c·arctan((s·x + b) / sqrt(a)))/(sqrt(a)·s) + "
            "(m·ln((s·x + b)^2 + a))/(s·2)"
        ),
        expected_direct_diff_integrate_result=(
            "(m·(s·x + b) + c) / ((s·x + b)^2 + a)"
        ),
        expected_direct_diff_integrate_required_display=("a > 0", "s ≠ 0"),
        expected_required_display=("a > 0", "s ≠ 0"),
        expected_step_substrings=(
            "Separar la parte logarítmica del numerador",
            "Integrar la derivada del denominador como logaritmo",
            "Usar la regla de arctan con derivada interna",
        ),
        family="algorithmic_backend_hermite_positive_quadratic",
        argument_regime="symbolic_affine_positive_quadratic_mixed_numerator",
        domain_regime="backend_verified_positive_radius_and_slope_nonzero_required",
        trace_regime="algorithmic_backend_hermite_summary",
        presentation_regime="backend_summary_log_arctan_symbolic_affine_positive_radius",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_hermite_expanded_symbolic_affine_positive_radius_mixed_numerator",
        expr="integrate((m*s*x+b*m+c)/(s^2*x^2+2*b*s*x+b^2+a), x)",
        expected_result=(
            "(c·arctan((s·x + b) / sqrt(a)))/(sqrt(a)·s) + "
            "(m·ln((s·x + b)^2 + a))/(s·2)"
        ),
        expected_direct_diff_integrate_result=(
            "(m·s·x + b·m + c) / (s^2·x^2 + 2·b·s·x + b^2 + a)"
        ),
        expected_direct_diff_integrate_required_display=("a > 0", "s ≠ 0"),
        expected_required_display=("a > 0", "s ≠ 0"),
        expected_step_substrings=(
            "Completar el cuadrado en el denominador",
            "Separar la parte logarítmica del numerador",
            "Integrar la derivada del denominador como logaritmo",
            "Usar la regla de arctan con derivada interna",
        ),
        family="algorithmic_backend_hermite_positive_quadratic",
        argument_regime="expanded_symbolic_affine_positive_quadratic_mixed_numerator",
        domain_regime="backend_verified_positive_radius_and_slope_nonzero_required",
        trace_regime="algorithmic_backend_hermite_summary",
        presentation_regime="backend_summary_log_arctan_symbolic_affine_positive_radius",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_hermite_expanded_symbolic_affine_derivative_multiple_numerator",
        expr="integrate((m*s*x+b*m)/(s^2*x^2+2*b*s*x+b^2+a), x)",
        expected_result="(m·ln((s·x + b)^2 + a))/(s·2)",
        expected_direct_diff_integrate_result=(
            "(m·s·x + b·m) / (s^2·x^2 + 2·b·s·x + b^2 + a)"
        ),
        expected_direct_diff_integrate_required_display=("a > 0", "s ≠ 0"),
        expected_required_display=("a > 0", "s ≠ 0"),
        expected_step_substrings=(
            "Completar el cuadrado en el denominador",
            "Integrar la derivada del denominador como logaritmo",
        ),
        family="algorithmic_backend_hermite_positive_quadratic",
        argument_regime="expanded_symbolic_affine_positive_quadratic_derivative_multiple_numerator",
        domain_regime="backend_verified_positive_radius_and_slope_nonzero_required",
        trace_regime="algorithmic_backend_hermite_summary",
        presentation_regime="backend_summary_log_symbolic_affine_positive_radius",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_hermite_expanded_numeric_center_derivative_multiple_numerator",
        expr="integrate((x+2)/(x^2+4*x+4+a), x)",
        expected_result="1/2·ln((x + 2)^2 + a)",
        expected_direct_diff_integrate_result="(x + 2) / (x^2 + a + 4·x + 4)",
        expected_direct_diff_integrate_required_display=("a > 0",),
        expected_required_display=("a > 0",),
        expected_step_substrings=(
            "Completar el cuadrado en el denominador",
            "Integrar la derivada del denominador como logaritmo",
        ),
        family="algorithmic_backend_hermite_positive_quadratic",
        argument_regime="expanded_numeric_center_positive_quadratic_derivative_multiple_numerator",
        domain_regime="backend_verified_positive_radius_required",
        trace_regime="algorithmic_backend_hermite_summary",
        presentation_regime="backend_summary_log_numeric_center_positive_radius",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_hermite_expanded_numeric_center_mixed_numerator",
        expr="integrate((x+2+c)/(x^2+4*x+4+a), x)",
        expected_result=(
            "1/2·ln((x + 2)^2 + a) + (c·arctan((x + 2) / sqrt(a)))/sqrt(a)"
        ),
        expected_direct_diff_integrate_result="(c + x + 2) / (x^2 + a + 4·x + 4)",
        expected_direct_diff_integrate_required_display=("a > 0",),
        expected_required_display=("a > 0",),
        family="algorithmic_backend_hermite_positive_quadratic",
        argument_regime="expanded_numeric_center_positive_quadratic_mixed_numerator",
        domain_regime="backend_verified_positive_radius_required",
        trace_regime="algorithmic_backend_hermite_summary",
        presentation_regime="backend_summary_log_arctan_numeric_center_positive_radius",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_rational_multi_quadratic_mixed_numerator",
        expr="integrate((x^3+x+1)/((x^2+1)*(x^2+4)), x)",
        expected_result="1/3·arctan(x) + 1/2·ln(x^2 + 4) - 1/3·arctan(x / 2)/2",
        expected_direct_diff_integrate_result=(
            "(x^3 + x + 1) / ((x^2 + 1)·(x^2 + 4))"
        ),
        expected_direct_diff_integrate_required_display=(),
        expected_required_display=(),
        expected_step_substrings=(
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="algorithmic_backend_rational_multi_quadratic",
        argument_regime="proper_numerator_two_distinct_irreducible_quadratics",
        domain_regime="backend_verified_unconditional_real",
        trace_regime="algorithmic_backend_rational_summary",
        presentation_regime="backend_summary_arctan_log_multi_quadratic",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_rational_multi_quadratic_triple_product",
        expr="integrate(1/((x^2+1)*(x^2+4)*(x^2+9)), x)",
        expected_result=(
            "1/24·arctan(x) - 1/15·arctan(x / 2)/2 + (1/40·arctan(x / 3))/3"
        ),
        expected_direct_diff_integrate_result=(
            "1 / ((x^2 + 1)·(x^2 + 4)·(x^2 + 9))"
        ),
        expected_direct_diff_integrate_required_display=(),
        expected_required_display=(),
        expected_step_substrings=(
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="algorithmic_backend_rational_multi_quadratic",
        argument_regime="reciprocal_three_distinct_irreducible_quadratics",
        domain_regime="backend_verified_unconditional_real",
        trace_regime="algorithmic_backend_rational_summary",
        presentation_regime="backend_summary_arctan_multi_quadratic",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_rational_ostrogradsky_expanded_sextic",
        expr="integrate(1/(x^6+9*x^4+24*x^2+16), x)",
        expected_result=(
            "1/9·arctan(x) - 11/72·arctan(x / 2)/2 - 1/24·x / (x^2 + 4)"
        ),
        expected_direct_diff_integrate_result=(
            "1 / (x^6 + 9·x^4 + 24·x^2 + 16)"
        ),
        expected_direct_diff_integrate_required_display=(),
        expected_required_display=(),
        expected_step_substrings=(
            "Separar la parte racional (reducción de Ostrogradsky)",
            "Factorizar el denominador",
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="algorithmic_backend_rational_general",
        argument_regime="expanded_sextic_repeated_quadratic_ostrogradsky",
        domain_regime="backend_verified_unconditional_real",
        trace_regime="algorithmic_backend_rational_summary",
        presentation_regime="backend_summary_arctan_rational_part",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_rational_general_pole_condition",
        expr="integrate(1/(x^5+2*x^3+x), x)",
        expected_result="ln(|x|) + 1/2 / (x^2 + 1) - 1/2·ln(x^2 + 1)",
        expected_direct_diff_integrate_result=(
            "1 / (x^5 + 2·x^3 + x)"
        ),
        expected_direct_diff_integrate_required_display=("x ≠ 0",),
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=(
            "Separar la parte racional (reducción de Ostrogradsky)",
            "Factorizar el denominador",
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="algorithmic_backend_rational_general",
        argument_regime="expanded_quintic_rational_pole_repeated_quadratic",
        domain_regime="backend_verified_pole_nonzero_required",
        trace_regime="algorithmic_backend_rational_summary",
        presentation_regime="backend_summary_log_abs_rational_part",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_rational_sophie_germain_quartic",
        expr="integrate(1/(x^4+4), x)",
        expected_result=(
            "1/16·ln(x^2 + 2·x + 2) + 1/8·arctan(x + 1) + 1/8·arctan(x - 1)"
            " - 1/16·ln(x^2 + 2 - 2·x)"
        ),
        expected_direct_diff_integrate_result="1 / (x^4 + 4)",
        expected_direct_diff_integrate_required_display=(),
        expected_required_display=(),
        expected_step_substrings=(
            "Factorizar el denominador",
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="algorithmic_backend_rational_general",
        argument_regime="even_quartic_symmetric_descent_factorization",
        domain_regime="backend_verified_unconditional_real",
        trace_regime="algorithmic_backend_rational_summary",
        presentation_regime="backend_summary_arctan_log_shifted_pair",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_rational_cyclotomic_style_quartic",
        expr="integrate((x^2+1)/(x^4+x^2+1), x)",
        expected_result=(
            "(1·arctan((2·x + 1) / sqrt(3)))/sqrt(3)"
            " + (1·arctan((2·x - 1) / sqrt(3)))/sqrt(3)"
        ),
        expected_direct_diff_integrate_result=(
            "(x^2 + 1) / (x^4 + x^2 + 1)"
        ),
        expected_direct_diff_integrate_required_display=(),
        expected_required_display=(),
        expected_step_substrings=(
            "Factorizar el denominador",
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="algorithmic_backend_rational_general",
        argument_regime="even_quartic_descent_irrational_radius_pair",
        domain_regime="backend_verified_unconditional_real",
        trace_regime="algorithmic_backend_rational_summary",
        presentation_regime="backend_summary_arctan_doubled_center",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_rational_resolvent_cubic_quartic",
        expr="integrate(1/(x^4+2*x^3+4*x^2+2*x+3), x)",
        expected_result=(
            "1/8·ln(x^2 + 2·x + 3) + 1/4·arctan(x) - 1/8·ln(x^2 + 1)"
        ),
        expected_direct_diff_integrate_result=(
            "1 / (x^4 + 2·x^3 + 4·x^2 + 2·x + 3)"
        ),
        expected_direct_diff_integrate_required_display=(
            "x^4 + 2·x^3 + 4·x^2 + 2·x + 3 ≠ 0",
        ),
        expected_required_display=("x^4 + 2·x^3 + 4·x^2 + 2·x + 3 ≠ 0",),
        expected_step_substrings=(
            "Factorizar el denominador",
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="algorithmic_backend_rational_general",
        argument_regime="non_even_quartic_resolvent_cubic_descent",
        domain_regime="backend_verified_conservative_denominator_nonzero",
        trace_regime="algorithmic_backend_rational_summary",
        presentation_regime="backend_summary_arctan_log_shifted_pair",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_rational_resolvent_cubic_quartic_mixed_factors",
        expr="integrate(1/(x^4+x^3+3*x^2+2*x+2), x)",
        expected_result=(
            "1/6·ln(x^2 + x + 1) - 1/3·arctan(x / sqrt(2))/sqrt(2)"
            " + (1·arctan((2·x + 1) / sqrt(3)))/sqrt(3) - 1/6·ln(x^2 + 2)"
        ),
        expected_direct_diff_integrate_result=(
            "1 / (x^4 + x^3 + 3·x^2 + 2·x + 2)"
        ),
        expected_direct_diff_integrate_required_display=(
            "x^4 + x^3 + 3·x^2 + 2·x + 2 ≠ 0",
        ),
        expected_required_display=("x^4 + x^3 + 3·x^2 + 2·x + 2 ≠ 0",),
        expected_step_substrings=(
            "Factorizar el denominador",
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="algorithmic_backend_rational_general",
        argument_regime="non_even_quartic_resolvent_cubic_mixed_radii",
        domain_regime="backend_verified_conservative_denominator_nonzero",
        trace_regime="algorithmic_backend_rational_summary",
        presentation_regime="backend_summary_arctan_log_mixed_radii",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_polynomial_exact_value",
        expr="integrate(x^2, x, 0, 1)",
        expected_result="1/3",
        expected_required_display=(),
        expected_step_substrings=(
            "int_0^1",
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="polynomial_numeric_bounds",
        domain_regime="interval_certified_unconditional",
        trace_regime="definite_ftc_evaluation",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_arctan_exact_pi",
        expr="integrate(1/(x^2+1), x, 0, 1)",
        expected_result="1/4·pi",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="positive_quadratic_numeric_bounds",
        domain_regime="interval_certified_unconditional",
        trace_regime="definite_ftc_evaluation",
        presentation_regime="exact_pi_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_log_certified_off_pole",
        expr="integrate(1/x, x, 1, 2)",
        expected_result="ln(2)",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="reciprocal_pole_outside_interval",
        domain_regime="interval_certified_with_source_condition",
        trace_regime="definite_ftc_evaluation",
        presentation_regime="exact_log_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_pole_inside_interval_undefined",
        expr="integrate(1/x, x, -1, 1)",
        expected_result="undefined",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=(
            "Detectar un polo dentro del intervalo de integración",
        ),
        family="definite_integral_ftc_undefined",
        argument_regime="reciprocal_pole_inside_interval",
        domain_regime="interval_pole_divergent",
        trace_regime="definite_ftc_interval_check",
        presentation_regime="undefined",
        outcome="undefined",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_symbolic_bound_area_function",
        expr="integrate(x^2, x, 0, t)",
        expected_result="1/3·t^3",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="symbolic_upper_bound_unconditional",
        domain_regime="symbolic_bounds_condition_free",
        trace_regime="definite_ftc_evaluation",
        presentation_regime="area_function_polynomial",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_symbolic_bound_conditional_residual",
        expr="integrate(1/x, x, 1, t)",
        expected_result="integrate(1 / x, x, 1, t)",
        expected_required_display=("x ≠ 0",),
        family="definite_integral_ftc_residual",
        argument_regime="symbolic_bound_over_conditional_antiderivative",
        domain_regime="interval_not_certifiable",
        trace_regime="definite_ftc_residual",
        presentation_regime="residual_echo",
        outcome="residual",
        residual_cause="definite_interval_not_certifiable",
    ),
    IntegrateCommandMatrixCase(
        name="reciprocal_exponential_normalized_negative_exponent",
        expr="integrate(exp(-x), x)",
        expected_result="-1 / e^x",
        expected_direct_diff_integrate_result="1 / e^x",
        expected_direct_diff_integrate_required_display=(),
        expected_required_display=(),
        expected_step_substrings=(
            "Reescribir el cociente como producto exponencial",
            "Usar la regla de la exponencial",
        ),
        family="elementary_exponential_table",
        argument_regime="reciprocal_exp_unit_negative_slope",
        domain_regime="total_real_function",
        trace_regime="exponential_table_normalized_reciprocal",
        presentation_regime="reciprocal_exponential_display",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_improper_exponential_decay",
        expr="integrate(exp(-x), x, 0, infinity)",
        expected_result="1",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="improper_exponential_decay",
        domain_regime="improper_interval_certified",
        trace_regime="definite_ftc_improper_limit",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="by_parts_exponential_normalized_div_linear",
        expr="integrate(x/e^x, x)",
        expected_result="(-x - 1) / e^x",
        expected_direct_diff_integrate_equivalent_to="x / e^x",
        expected_direct_diff_integrate_required_display=(),
        expected_required_display=(),
        expected_step_substrings=(
            "Reescribir el cociente como producto exponencial",
            "Usar integración por partes",
        ),
        family="by_parts_exponential_table",
        argument_regime="normalized_div_linear_times_exp",
        domain_regime="total_real_function",
        trace_regime="by_parts_exponential_normalized_div",
        presentation_regime="reciprocal_exponential_display",
    ),
    IntegrateCommandMatrixCase(
        name="exp_substitution_gaussian_damped_indefinite",
        expr="integrate(x*e^(-x^2), x)",
        expected_result="-1 / (2·e^(x^2))",
        expected_direct_diff_integrate_equivalent_to="x / e^(x^2)",
        expected_direct_diff_integrate_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Usar sustitución",),
        family="exp_polynomial_substitution",
        argument_regime="normalized_div_derivative_times_exp_quadratic",
        domain_regime="total_real_function",
        trace_regime="exp_polynomial_derivative_substitution",
        presentation_regime="reciprocal_exponential_display",
    ),
    IntegrateCommandMatrixCase(
        name="exp_substitution_cubic_damped_indefinite",
        expr="integrate(x^2*e^(-x^3), x)",
        expected_result="-1 / (3·e^(x^3))",
        expected_direct_diff_integrate_equivalent_to="x^2 / e^(x^3)",
        expected_direct_diff_integrate_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Usar sustitución",),
        family="exp_polynomial_substitution",
        argument_regime="normalized_div_derivative_times_exp_cubic",
        domain_regime="total_real_function",
        trace_regime="exp_polynomial_derivative_substitution",
        presentation_regime="reciprocal_exponential_display",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_improper_gaussian_damped",
        expr="integrate(x*e^(-x^2), x, 0, infinity)",
        expected_result="1/2",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="improper_gaussian_damped_integral",
        domain_regime="improper_interval_certified",
        trace_regime="definite_ftc_improper_limit",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_improper_scaled_arctan_pi",
        expr="integrate(2/(1+x^2), x, 0, infinity)",
        expected_result="pi",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="improper_scaled_arctan_integral",
        domain_regime="improper_interval_certified",
        trace_regime="definite_ftc_improper_limit",
        presentation_regime="exact_pi_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_improper_inf_alias_spelling",
        expr="integrate(e^(-x), x, 0, inf)",
        expected_result="1",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="improper_exponential_decay_inf_alias",
        domain_regime="improper_interval_certified",
        trace_regime="definite_ftc_improper_limit",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_improper_gamma_two",
        expr="integrate(x*exp(-x), x, 0, infinity)",
        expected_result="1",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="improper_gamma_integral",
        domain_regime="improper_interval_certified",
        trace_regime="definite_ftc_improper_limit",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_improper_log_divergent_positive",
        expr="integrate(1/x, x, 1, infinity)",
        expected_result="infinity",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="improper_log_divergence_positive_tail",
        domain_regime="improper_divergent_to_infinity",
        trace_regime="definite_ftc_improper_limit",
        presentation_regime="honest_infinite_divergence",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_improper_log_divergent_negative",
        expr="integrate(1/x, x, -infinity, -1)",
        expected_result="-infinity",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="improper_log_divergence_negative_tail",
        domain_regime="improper_divergent_to_infinity",
        trace_regime="definite_ftc_improper_limit",
        presentation_regime="honest_infinite_divergence",
    ),
    IntegrateCommandMatrixCase(
        name="cyclic_by_parts_damped_sine_normalized_div",
        expr="integrate(sin(x)/e^x, x)",
        expected_result="(-sin(x) - cos(x)) / (2·e^x)",
        expected_direct_diff_integrate_result="sin(x) / e^x",
        expected_direct_diff_integrate_required_display=(),
        expected_required_display=(),
        expected_step_substrings=(
            "Reescribir el cociente como producto exponencial",
            "Usar integración por partes",
        ),
        family="by_parts_exponential_table",
        argument_regime="normalized_div_damped_sine",
        domain_regime="total_real_function",
        trace_regime="by_parts_exponential_normalized_div",
        presentation_regime="reciprocal_exponential_display",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_improper_damped_sine",
        expr="integrate(sin(x)/e^x, x, 0, infinity)",
        expected_result="1/2",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="improper_damped_oscillation",
        domain_regime="improper_interval_certified",
        trace_regime="definite_ftc_improper_limit",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_cofactor_sqrt_chain",
        expr="integrate(x/sqrt(x^2+1), x, 0, 1)",
        expected_result="sqrt(2) - 1",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="derivative_cofactor_sqrt_chain",
        domain_regime="interval_certified_unconditional",
        trace_regime="definite_ftc_evaluation",
        presentation_regime="exact_radical_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_positive_polynomial_certified",
        expr="integrate(x/sqrt(1-x^2), x, 0, 1/2)",
        expected_result="1/2·(2 - sqrt(3))",
        expected_required_display=("-1 < x < 1",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="positive_polynomial_roots_outside_interval",
        domain_regime="interval_certified_with_source_condition",
        trace_regime="definite_ftc_evaluation",
        presentation_regime="exact_radical_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_trig_pole_free_certified",
        expr="integrate(sec(x)^2, x, 0, 1)",
        expected_result="tan(1)",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="trig_nonzero_certified_pi_enclosure",
        domain_regime="interval_certified_trig_discharged",
        trace_regime="definite_ftc_evaluation",
        presentation_regime="exact_trig_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_trig_pole_inside_undefined",
        expr="integrate(tan(x), x, 0, 2)",
        expected_result="undefined",
        expected_required_display=(),
        expected_step_substrings=(
            "Detectar un polo dentro del intervalo de integración",
        ),
        family="definite_integral_ftc_undefined",
        argument_regime="trig_pole_inside_interval",
        domain_regime="interval_pole_divergent",
        trace_regime="definite_ftc_interval_check",
        presentation_regime="undefined",
        outcome="undefined",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_pi_bound_sec_squared",
        expr="integrate(sec(x)^2, x, 0, pi/4)",
        expected_result="1",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="pi_multiple_bound_trig_certified",
        domain_regime="interval_certified_trig_discharged",
        trace_regime="definite_ftc_evaluation",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_pi_bound_sec_squared_offset",
        expr="integrate(sec(x)^2, x, pi/4, pi/3)",
        expected_result="sqrt(3) - 1",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="pi_multiple_bounds_both_sides",
        domain_regime="interval_certified_trig_discharged",
        trace_regime="definite_ftc_evaluation",
        presentation_regime="exact_radical_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_pi_bound_pole_inside_undefined",
        expr="integrate(tan(x), x, 0, 3*pi/4)",
        expected_result="undefined",
        expected_required_display=(),
        expected_step_substrings=(
            "Detectar un polo dentro del intervalo de integración",
        ),
        family="definite_integral_ftc_undefined",
        argument_regime="pi_multiple_bound_pole_inside",
        domain_regime="interval_pole_divergent",
        trace_regime="definite_ftc_interval_check",
        presentation_regime="undefined",
        outcome="undefined",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_boundary_convergent_log",
        expr="integrate(ln(x), x, 0, 1)",
        expected_result="-1",
        expected_required_display=("x > 0",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="boundary_touch_log_endpoint",
        domain_regime="boundary_touch_one_sided_limit",
        trace_regime="definite_ftc_boundary_limit",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_boundary_convergent_inverse_sqrt",
        expr="integrate(1/sqrt(x), x, 0, 1)",
        expected_result="2",
        expected_required_display=("x > 0",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="boundary_touch_inverse_sqrt",
        domain_regime="boundary_touch_one_sided_limit",
        trace_regime="definite_ftc_boundary_limit",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_boundary_convergent_sqrt_domain",
        expr="integrate(x/sqrt(1-x^2), x, 0, 1)",
        expected_result="1",
        expected_required_display=("-1 < x < 1",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="boundary_touch_sqrt_domain_root",
        domain_regime="boundary_touch_one_sided_limit",
        trace_regime="definite_ftc_boundary_limit",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_boundary_divergent_endpoint_pole",
        expr="integrate(1/x, x, 0, 1)",
        expected_result="infinity",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="boundary_touch_divergent_pole",
        domain_regime="improper_divergent_to_infinity",
        trace_regime="definite_ftc_boundary_limit",
        presentation_regime="honest_infinite_divergence",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_boundary_convergent_sqrt_integrand",
        expr="integrate(sqrt(x), x, 0, 4)",
        expected_result="16/3",
        expected_required_display=("x ≥ 0",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="boundary_touch_sqrt_integrand",
        domain_regime="boundary_touch_one_sided_limit",
        trace_regime="definite_ftc_boundary_limit",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_boundary_convergent_cube_root",
        expr="integrate(x^(1/3), x, 0, 8)",
        expected_result="12",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="boundary_touch_cube_root_integrand",
        domain_regime="boundary_touch_one_sided_limit",
        trace_regime="definite_ftc_boundary_limit",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_boundary_convergent_power_radical_product",
        expr="integrate(x*sqrt(x), x, 0, 1)",
        expected_result="2/5",
        expected_required_display=("x ≥ 0",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="boundary_touch_power_radical_product",
        domain_regime="boundary_touch_one_sided_limit",
        trace_regime="definite_ftc_boundary_limit",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="multiple_angle_cosine_product_chebyshev",
        expr="integrate(cos(x)*cos(2*x), x)",
        expected_result="1/3·(3·sin(x) - 2·sin(x)^3)",
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="sine_multiple_angle_ratio",
        argument_regime="multiple_angle_cosine_product",
        domain_regime="total_real_function",
        trace_regime="chebyshev_multiple_angle_rewrite",
        presentation_regime="sine_polynomial_compact",
    ),
    IntegrateCommandMatrixCase(
        name="sine_multiple_angle_ratio_quotient_surface",
        expr="integrate(sin(4*x)/(4*sin(x)), x)",
        expected_result="1/3·(3·sin(x) - 2·sin(x)^3)",
        expected_required_display=("sin(x) ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="sine_multiple_angle_ratio",
        argument_regime="dirichlet_style_sine_ratio",
        domain_regime="nonzero_required",
        trace_regime="chebyshev_multiple_angle_rewrite",
        presentation_regime="sine_polynomial_compact",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_fourier_orthogonality_cosines",
        expr="integrate(cos(x)*cos(2*x), x, 0, pi)",
        expected_result="0",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="fourier_orthogonality_distinct_cosines",
        domain_regime="symbolic_bounds_condition_free",
        trace_regime="definite_ftc_pi_bounds",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_fourier_orthogonality_sines",
        expr="integrate(sin(2*x)*sin(3*x), x, 0, 2*pi)",
        expected_result="0",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="fourier_orthogonality_distinct_sines",
        domain_regime="symbolic_bounds_condition_free",
        trace_regime="definite_ftc_evaluation",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_fourier_sine_cosine_value",
        expr="integrate(sin(3*x)*cos(2*x), x, 0, pi)",
        expected_result="6/5",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="fourier_sine_cosine_half_period",
        domain_regime="symbolic_bounds_condition_free",
        trace_regime="definite_ftc_evaluation",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_improper_convergent_reciprocal_square",
        expr="integrate(1/x^2, x, 1, infinity)",
        expected_result="1",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="improper_upper_infinity_pole_outside",
        domain_regime="improper_interval_certified",
        trace_regime="definite_ftc_improper_limit",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_improper_full_line_arctan",
        expr="integrate(1/(x^2+1), x, -infinity, infinity)",
        expected_result="pi",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="improper_full_line_unconditional",
        domain_regime="improper_interval_certified",
        trace_regime="definite_ftc_improper_limit",
        presentation_regime="exact_pi_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_improper_divergent_polynomial",
        expr="integrate(x^2, x, 0, infinity)",
        expected_result="infinity",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="improper_divergent_polynomial",
        domain_regime="improper_divergent_to_infinity",
        trace_regime="definite_ftc_improper_limit",
        presentation_regime="honest_infinite_divergence",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_hermite_symbolic_indefinite_square_denominator",
        expr="integrate(1/(x^2-a^2), x)",
        expected_result="((ln(|x - a|) - ln(|a + x|))·1·1/2)/a",
        expected_direct_diff_integrate_result="1 / (x^2 - a^2)",
        expected_direct_diff_integrate_required_display=(
            "a + x ≠ 0",
            "a ≠ 0",
            "a - x ≠ 0",
        ),
        expected_required_display=("a + x ≠ 0", "a ≠ 0", "a - x ≠ 0"),
        family="algorithmic_backend_hermite_indefinite_square",
        argument_regime="reciprocal_indefinite_square_symbolic_radius",
        domain_regime="backend_verified_component_local_symbolic_poles_required",
        trace_regime="algorithmic_backend_hermite_summary",
        presentation_regime="backend_summary_log_abs_component_local_indefinite_square",
    ),
    IntegrateCommandMatrixCase(
        name="algorithmic_backend_hermite_symbolic_affine_indefinite_square_mixed_numerator",
        expr="integrate((m*(s*x+b)+c)/((s*x+b)^2-a^2), x)",
        expected_result=(
            "(m·ln(|(s·x + b)^2 - a^2|))/(s·2) + "
            "((ln(|s·x + b - a|) - ln(|s·x + a + b|))·c·1/2)/(a·s)"
        ),
        expected_direct_diff_integrate_result=(
            "(m·(s·x + b) + c) / ((s·x + b)^2 - a^2)"
        ),
        expected_direct_diff_integrate_required_display=(
            "a ≠ 0",
            "s ≠ 0",
            "s·x + a + b ≠ 0",
            "a - s·x - b ≠ 0",
        ),
        expected_required_display=(
            "a ≠ 0",
            "s ≠ 0",
            "s·x + a + b ≠ 0",
            "a - s·x - b ≠ 0",
        ),
        family="algorithmic_backend_hermite_indefinite_square",
        argument_regime="symbolic_affine_indefinite_square_mixed_numerator",
        domain_regime="backend_verified_component_local_symbolic_poles_and_slope_required",
        trace_regime="algorithmic_backend_hermite_summary",
        presentation_regime="backend_summary_log_abs_component_local_affine_indefinite_square_mixed_numerator",
    ),
    IntegrateCommandMatrixCase(
        name="log_derivative_positive_quadratic_substitution",
        expr="integrate((2*x+2)/(x^2+2*x+2), x)",
        expected_result="ln(x^2 + 2·x + 2)",
        expected_derivative_result="(2·x + 2) / (x^2 + 2·x + 2)",
        expected_direct_diff_integrate_result="(2·x + 2) / (x^2 + 2·x + 2)",
        expected_step_substrings=(
            "Usar la regla de u'/u -> ln|u|",
            "Identificar u y du",
            "Abs Under Positivity",
        ),
        family="log_derivative_polynomial",
        argument_regime="positive_quadratic_derivative_ratio",
        domain_regime="structurally_positive_log_argument",
        trace_regime="log_derivative_positive_quadratic_substitution",
        presentation_regime="positive_log_no_abs",
    ),
    IntegrateCommandMatrixCase(
        name="log_derivative_positive_quadratic_negative_orientation_substitution",
        expr="integrate((2*x-1)/(x^2-x+1), x)",
        expected_result="ln(x^2 + 1 - x)",
        expected_derivative_result="(2·x - 1) / (x^2 + 1 - x)",
        expected_direct_diff_integrate_result="(2·x - 1) / (x^2 + 1 - x)",
        expected_step_substrings=(
            "Usar la regla de u'/u -> ln|u|",
            "Identificar u y du",
            "Abs Under Positivity",
        ),
        family="log_derivative_polynomial",
        argument_regime="negative_orientation_positive_quadratic_derivative_ratio",
        domain_regime="structurally_positive_log_argument",
        trace_regime="log_derivative_positive_quadratic_negative_orientation_substitution",
        presentation_regime="positive_log_no_abs_negative_orientation",
    ),
    IntegrateCommandMatrixCase(
        name="log_rational_positive_domain_residual",
        expr="integrate(ln(x)/(x+1), x)",
        expected_result="integrate(ln(x) / (x + 1), x)",
        expected_required_display=("x > 0",),
        expected_step_substrings=("Conservar integral residual",),
        family="log_rational_residual",
        argument_regime="log_over_affine_rational_expression",
        domain_regime="positive_required_residual",
        outcome="residual",
        residual_cause="special_function_method_required",
        trace_regime="residual_policy_with_domain",
        presentation_regime="residual",
    ),
    IntegrateCommandMatrixCase(
        name="non_elementary_tan_polynomial_residual_domain",
        expr="integrate(tan(x^2), x)",
        expected_result="integrate(tan(x^2), x)",
        expected_required_display=("cos(x^2) ≠ 0",),
        expected_step_substrings=("Conservar integral residual",),
        family="trig_residual_domain",
        argument_regime="nonlinear_polynomial_argument",
        domain_regime="trig_pole_residual",
        outcome="residual",
        residual_cause="non_elementary_composition",
        trace_regime="residual_policy_with_domain",
        presentation_regime="residual",
    ),
    IntegrateCommandMatrixCase(
        name="non_elementary_tan_presimplified_residual_domain",
        expr="integrate(tan(x^2+0), x)",
        expected_result="integrate(tan(x^2), x)",
        expected_required_display=("cos(x^2) ≠ 0",),
        expected_step_substrings=("Conservar integral residual",),
        family="trig_residual_domain",
        argument_regime="nonlinear_polynomial_argument_presimplified",
        domain_regime="trig_pole_presimplified_residual",
        outcome="residual",
        residual_cause="non_elementary_composition",
        trace_regime="residual_presimplification_with_domain",
        presentation_regime="residual_presentation_cleanup",
    ),
    IntegrateCommandMatrixCase(
        name="non_elementary_csc_presimplified_residual_domain",
        expr="integrate(csc(x^2+0), x)",
        expected_result="integrate(csc(x^2), x)",
        expected_required_display=("sin(x^2) ≠ 0",),
        expected_step_substrings=("Conservar integral residual",),
        family="trig_residual_domain",
        argument_regime="nonlinear_polynomial_argument_presimplified",
        domain_regime="trig_sine_pole_presimplified_residual",
        outcome="residual",
        residual_cause="non_elementary_composition",
        trace_regime="residual_presimplification_with_domain",
        presentation_regime="residual_presentation_cleanup",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_reciprocal_sine_presimplified_residual_domain",
        expr="integrate(1/sin(x^2+0), x)",
        expected_result="integrate(csc(x^2), x)",
        expected_required_display=("sin(x^2) ≠ 0",),
        expected_step_substrings=(
            "Reconocer cosecante desde un recíproco",
            "Conservar integral residual",
        ),
        family="explicit_reciprocal_trig_residual_domain",
        argument_regime="explicit_reciprocal_presimplified_argument",
        domain_regime="explicit_denominator_source_condition",
        outcome="residual",
        residual_cause="non_elementary_composition",
        trace_regime="residual_presimplification_with_domain",
        presentation_regime="result_cleanup_source_condition",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_reciprocal_sine_verified_log_domain",
        expr="integrate(2*x/sin(x^2), x)",
        expected_result="ln(|csc(x^2) - cot(x^2)|)",
        expected_derivative_equivalent_to="2*x/sin(x^2)",
        expected_derivative_required_display=("sin(x^2) ≠ 0",),
        expected_required_display=("sin(x^2) ≠ 0",),
        expected_step_substrings=(
            "Sacar constante de una fracción",
            "Usar la regla de csc(u) -> ln|csc(u)-cot(u)|",
            "Identificar u y du",
        ),
        family="explicit_reciprocal_trig_substitution",
        argument_regime="explicit_reciprocal_sine_polynomial_derivative",
        domain_regime="explicit_reciprocal_sine_verified_substitution",
        trace_regime="reciprocal_sine_verified_substitution",
        presentation_regime="abs_log_reciprocal_trig_source_condition",
    ),
    IntegrateCommandMatrixCase(
        name="reciprocal_cosine_odd_cube_reduction",
        expr="integrate(1/cos(x)^3, x)",
        expected_result="(ln(|tan(x) + sec(x)|) + tan(x)·sec(x)) / 2",
        expected_required_display=("cos(x) ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="reciprocal_trig_odd_power_reduction",
        argument_regime="reciprocal_cosine_cube",
        domain_regime="nonzero_required",
        trace_regime="sec_odd_power_reduction",
        presentation_regime="sec_tan_log_reduction_closed_form",
    ),
    IntegrateCommandMatrixCase(
        name="reciprocal_sine_odd_cube_reduction",
        expr="integrate(1/sin(x)^3, x)",
        expected_result="(ln(|csc(x) - cot(x)|) - csc(x)·cot(x)) / 2",
        expected_required_display=("sin(x) ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="reciprocal_trig_odd_power_reduction",
        argument_regime="reciprocal_sine_cube",
        domain_regime="nonzero_required",
        trace_regime="csc_odd_power_reduction",
        presentation_regime="csc_cot_log_reduction_closed_form",
    ),
    IntegrateCommandMatrixCase(
        name="reciprocal_cosine_odd_cube_affine_reduction",
        expr="integrate(1/cos(2*x+1)^3, x)",
        expected_result="(ln(|tan(2·x + 1) + sec(2·x + 1)|) + tan(2·x + 1)·sec(2·x + 1)) / 4",
        expected_required_display=("cos(2·x + 1) ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="reciprocal_trig_odd_power_reduction",
        argument_regime="reciprocal_cosine_cube_affine",
        domain_regime="nonzero_required",
        trace_regime="sec_odd_power_reduction",
        presentation_regime="sec_tan_log_reduction_closed_form",
    ),
    IntegrateCommandMatrixCase(
        name="tangent_odd_cube_reduction",
        expr="integrate(tan(x)^3, x)",
        expected_result="1/2·(2·ln(|cos(x)|) + sin(x)^2 / cos(x)^2)",
        expected_required_display=("cos(x) ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="tangent_odd_power_reduction",
        argument_regime="tangent_cube",
        domain_regime="nonzero_required",
        trace_regime="tan_odd_power_reduction",
        presentation_regime="tan_square_log_reduction_closed_form",
    ),
    IntegrateCommandMatrixCase(
        name="cotangent_odd_cube_reduction",
        expr="integrate(cot(x)^3, x)",
        expected_result="1/2·(-cos(x)^2 / sin(x)^2 - 2·ln(|sin(x)|))",
        expected_required_display=("sin(x) ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="tangent_odd_power_reduction",
        argument_regime="cotangent_cube",
        domain_regime="nonzero_required",
        trace_regime="cot_odd_power_reduction",
        presentation_regime="cot_square_log_reduction_closed_form",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_tan_cube_pi_bound",
        expr="integrate(tan(x)^3, x, 0, pi/4)",
        expected_result="1/2·(-ln(2) + 1)",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="tan_cube_pi_bound_integral",
        domain_regime="interval_certified_trig_discharged",
        trace_regime="definite_ftc_pi_bounds",
        presentation_regime="exact_log_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_monomial_arcsine_unit_interval",
        expr="integrate(x*arcsin(x), x, 0, 1)",
        expected_result="1/8·pi",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="bounded_inverse_trig_unit_interval",
        domain_regime="interval_certified_unconditional",
        trace_regime="definite_ftc_from_verified_antiderivative",
        presentation_regime="exact_pi_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_arccosine_unit_interval",
        expr="integrate(arccos(x), x, 0, 1)",
        expected_result="1",
        expected_required_display=(),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="bounded_inverse_trig_unit_interval",
        domain_regime="interval_certified_unconditional",
        trace_regime="definite_ftc_from_verified_antiderivative",
        presentation_regime="exact_rational_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_sec_cube_pi_bound",
        expr="integrate(1/cos(x)^3, x, 0, pi/4)",
        expected_result="1/2·(ln(sqrt(2) + 1) + sqrt(2))",
        expected_required_display=("cos(x) ≠ 0",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="sec_cube_pi_bound_integral",
        domain_regime="interval_certified_trig_discharged",
        trace_regime="definite_ftc_pi_bounds",
        presentation_regime="exact_radical_log_value",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_reciprocal_cosine_verified_log_domain",
        expr="integrate(2*x/cos(x^2), x)",
        expected_result="ln(|tan(x^2) + sec(x^2)|)",
        expected_derivative_equivalent_to="2*x/cos(x^2)",
        expected_derivative_required_display=("cos(x^2) ≠ 0",),
        expected_required_display=("cos(x^2) ≠ 0",),
        expected_step_substrings=(
            "Sacar constante de una fracción",
            "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
            "Identificar u y du",
        ),
        family="explicit_reciprocal_trig_substitution",
        argument_regime="explicit_reciprocal_cosine_polynomial_derivative",
        domain_regime="explicit_reciprocal_cosine_verified_substitution",
        trace_regime="reciprocal_cosine_verified_substitution",
        presentation_regime="abs_log_reciprocal_trig_source_condition",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_reciprocal_cosine_symbolic_shift_verified_log_domain",
        expr="integrate(2*x/cos(x^2+b), x)",
        expected_result="ln(|tan(x^2 + b) + sec(x^2 + b)|)",
        expected_derivative_equivalent_to="2*x/cos(x^2+b)",
        expected_derivative_required_display=("cos(x^2 + b) ≠ 0",),
        expected_required_display=("cos(x^2 + b) ≠ 0",),
        expected_step_substrings=(
            "Sacar constante de una fracción",
            "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
            "Identificar u y du",
        ),
        family="explicit_reciprocal_trig_substitution",
        argument_regime="explicit_reciprocal_cosine_symbolic_shift_polynomial_derivative",
        domain_regime="explicit_reciprocal_cosine_symbolic_shift_verified_substitution",
        trace_regime="reciprocal_cosine_symbolic_shift_verified_substitution",
        presentation_regime="abs_log_reciprocal_trig_symbolic_shift_source_condition",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_reciprocal_cosine_symbolic_external_scale_shift_verified_log_domain",
        expr="integrate(2*k*x/cos(x^2+b), x)",
        expected_result="k·ln(|tan(x^2 + b) + sec(x^2 + b)|)",
        expected_derivative_equivalent_to="2*k*x/cos(x^2+b)",
        expected_derivative_required_display=("cos(x^2 + b) ≠ 0",),
        expected_required_display=("cos(x^2 + b) ≠ 0",),
        expected_step_substrings=(
            "Sacar constante de una fracción",
            "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
            "Identificar u y du",
            "Ajustar el factor constante",
        ),
        family="explicit_reciprocal_trig_substitution",
        argument_regime="explicit_reciprocal_cosine_symbolic_external_scale_shift_polynomial_derivative",
        domain_regime="explicit_reciprocal_cosine_symbolic_external_scale_shift_verified_substitution",
        trace_regime="reciprocal_cosine_symbolic_external_scale_shift_verified_substitution",
        presentation_regime="abs_log_reciprocal_trig_symbolic_external_scale_shift_source_condition",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_reciprocal_tangent_presimplified_residual_domain",
        expr="integrate(1/tan(x^2+0), x)",
        expected_result="integrate(cot(x^2), x)",
        expected_required_display=("sin(x^2) ≠ 0", "cos(x^2) ≠ 0"),
        expected_step_substrings=(
            "Expandir tangente como seno entre coseno",
            "Simplificar fracción anidada",
            "Reconocer cotangente desde un cociente",
            "Conservar integral residual",
        ),
        family="explicit_reciprocal_trig_residual_domain",
        argument_regime="explicit_tangent_reciprocal_presimplified_argument",
        domain_regime="explicit_tangent_denominator_source_condition",
        outcome="residual",
        residual_cause="non_elementary_composition",
        trace_regime="reciprocal_tangent_residual_prep",
        presentation_regime="compound_source_condition_residual",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_tangent_log_residual_condition_alias_dedupe",
        expr="integrate(1/(tan(x)*ln(tan(x))), x)",
        expected_result="integrate(cos(x) / (sin(x)·ln(sin(x) / cos(x))), x)",
        expected_required_display=(
            "tan(x) > 0",
            "cos(x) ≠ 0",
            "tan(x) - 1 ≠ 0",
            "sin(x) ≠ 0",
        ),
        expected_step_substrings=(
            "Expandir tangente como seno entre coseno",
            "Conservar integral residual",
        ),
        family="explicit_reciprocal_trig_residual_domain",
        argument_regime="explicit_tangent_log_residual_argument",
        domain_regime="explicit_tangent_log_shifted_condition_dedupe",
        outcome="residual",
        residual_cause="special_function_method_required",
        trace_regime="residual_presimplification_with_domain",
        presentation_regime="shifted_tangent_condition_alias_dedupe",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_tangent_log_numeric_shifted_residual_condition_alias_dedupe",
        expr="integrate(1/((tan(x)-2)*ln(tan(x))), x)",
        expected_result="integrate(cos(x) / (ln(sin(x) / cos(x))·(sin(x) - 2·cos(x))), x)",
        expected_required_display=(
            "cos(x) ≠ 0",
            "tan(x) - 1 ≠ 0",
            "tan(x) - 2 ≠ 0",
            "tan(x) > 0",
        ),
        expected_step_substrings=(
            "Expandir tangente como seno entre coseno",
            "Conservar integral residual",
        ),
        family="explicit_reciprocal_trig_residual_domain",
        argument_regime="explicit_tangent_log_numeric_shifted_residual_argument",
        domain_regime="explicit_tangent_log_numeric_shifted_condition_dedupe",
        outcome="residual",
        residual_cause="special_function_method_required",
        trace_regime="residual_presimplification_with_domain",
        presentation_regime="shifted_tangent_numeric_condition_alias_dedupe",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_tangent_log_numeric_offset_residual_condition_alias_dedupe",
        expr="integrate(1/((2-tan(x))*ln(tan(x))), x)",
        expected_result="integrate(cos(x) / (ln(sin(x) / cos(x))·(2·cos(x) - sin(x))), x)",
        expected_required_display=(
            "2 - tan(x) ≠ 0",
            "cos(x) ≠ 0",
            "tan(x) - 1 ≠ 0",
            "tan(x) > 0",
        ),
        expected_step_substrings=(
            "Expandir tangente como seno entre coseno",
            "Conservar integral residual",
        ),
        family="explicit_reciprocal_trig_residual_domain",
        argument_regime="explicit_tangent_log_numeric_offset_residual_argument",
        domain_regime="explicit_tangent_log_numeric_offset_condition_dedupe",
        outcome="residual",
        residual_cause="special_function_method_required",
        trace_regime="residual_presimplification_with_domain",
        presentation_regime="offset_tangent_numeric_condition_alias_dedupe",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_cotangent_log_numeric_offset_residual_condition_alias_dedupe",
        expr="integrate(1/((2-cot(x))*ln(cot(x))), x)",
        expected_result="integrate(sin(x) / (ln(cos(x) / sin(x))·(2·sin(x) - cos(x))), x)",
        expected_required_display=(
            "2 - cot(x) ≠ 0",
            "cot(x) > 0",
            "cot(x) - 1 ≠ 0",
            "sin(x) ≠ 0",
        ),
        expected_step_substrings=(
            "Expandir cotangente como coseno entre seno",
            "Conservar integral residual",
        ),
        family="explicit_reciprocal_trig_residual_domain",
        argument_regime="explicit_cotangent_log_numeric_offset_residual_argument",
        domain_regime="explicit_cotangent_log_numeric_offset_condition_dedupe",
        outcome="residual",
        residual_cause="special_function_method_required",
        trace_regime="residual_presimplification_with_domain",
        presentation_regime="offset_cotangent_numeric_condition_alias_dedupe",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_cotangent_log_numeric_shifted_residual_condition_alias_dedupe",
        expr="integrate(1/((cot(x)-2)*ln(cot(x))), x)",
        expected_result="integrate(sin(x) / (ln(cos(x) / sin(x))·(cos(x) - 2·sin(x))), x)",
        expected_required_display=(
            "cot(x) > 0",
            "cot(x) - 1 ≠ 0",
            "cot(x) - 2 ≠ 0",
            "sin(x) ≠ 0",
        ),
        expected_step_substrings=(
            "Expandir cotangente como coseno entre seno",
            "Conservar integral residual",
        ),
        family="explicit_reciprocal_trig_residual_domain",
        argument_regime="explicit_cotangent_log_numeric_shifted_residual_argument",
        domain_regime="explicit_cotangent_log_numeric_shifted_condition_dedupe",
        outcome="residual",
        residual_cause="special_function_method_required",
        trace_regime="residual_presimplification_with_domain",
        presentation_regime="shifted_cotangent_numeric_condition_alias_dedupe",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_reciprocal_tangent_verified_log_domain",
        expr="integrate(2*x/tan(x^2), x)",
        expected_result="ln(|sin(x^2)|)",
        expected_derivative_result="(cos(x^2)·x·2)/sin(x^2)",
        expected_derivative_required_display=("sin(x^2) ≠ 0",),
        expected_required_display=("sin(x^2) ≠ 0", "cos(x^2) ≠ 0"),
        expected_step_substrings=(
            "Expandir tangente como seno entre coseno",
            "Simplificar fracción anidada",
            "Sacar constante de una fracción",
            "Usar la regla de cot(u) -> ln|sin(u)|",
            "Identificar u y du",
        ),
        family="explicit_reciprocal_trig_substitution",
        argument_regime="explicit_tangent_reciprocal_polynomial_derivative",
        domain_regime="explicit_tangent_denominator_verified_substitution",
        trace_regime="reciprocal_tangent_verified_substitution",
        presentation_regime="abs_log_source_condition",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_reciprocal_tangent_presimplified_verified_log_domain",
        expr="integrate(2*x/tan(x^2+0), x)",
        expected_result="ln(|sin(x^2)|)",
        expected_derivative_result="(cos(x^2)·x·2)/sin(x^2)",
        expected_derivative_required_display=("sin(x^2) ≠ 0",),
        expected_required_display=("sin(x^2) ≠ 0", "cos(x^2) ≠ 0"),
        expected_step_substrings=(
            "Agrupar términos semejantes",
            "Expandir tangente como seno entre coseno",
            "Simplificar fracción anidada",
            "Sacar constante de una fracción",
            "Usar la regla de cot(u) -> ln|sin(u)|",
            "Identificar u y du",
        ),
        family="explicit_reciprocal_trig_substitution",
        argument_regime="explicit_tangent_reciprocal_presimplified_polynomial_derivative",
        domain_regime="explicit_tangent_presimplified_condition_dedupe",
        trace_regime="presimplified_reciprocal_tangent_verified_substitution",
        presentation_regime="abs_log_condition_dedupe",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_reciprocal_secant_presimplified_source_domain",
        expr="integrate(2*x/sec(x^2+0), x)",
        expected_result="sin(x^2)",
        expected_derivative_equivalent_to="2*x/sec(x^2)",
        expected_derivative_required_display=("cos(x^2) ≠ 0",),
        expected_required_display=("cos(x^2) ≠ 0",),
        expected_step_substrings=(
            "Agrupar términos semejantes",
            "Expandir secante como recíproco de coseno",
            "Simplificar fracción anidada",
            "Usar la regla de cos(u) -> sin(u)",
            "Identificar u y du",
        ),
        family="explicit_reciprocal_trig_substitution",
        argument_regime="explicit_secant_reciprocal_presimplified_polynomial_derivative",
        domain_regime="explicit_reciprocal_trig_source_defined_condition",
        trace_regime="presimplified_reciprocal_secant_verified_substitution",
        presentation_regime="source_defined_condition_cleanup",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_tangent_log_derivative_ratio_domain",
        expr="integrate(2*k*x*tan(x^2+b), x)",
        expected_result="-(k·ln(|cos(x^2 + b)|))",
        expected_derivative_equivalent_to="2*k*x*tan(x^2+b)",
        expected_derivative_required_display=("cos(x^2 + b) ≠ 0",),
        expected_required_display=("cos(x^2 + b) ≠ 0",),
        expected_step_substrings=(
            "Expandir tangente como seno entre coseno",
            "Sacar constante de una fracción",
            "Usar la regla de tan(u) -> -ln|cos(u)|",
            "Identificar u y du",
            "Ajustar el factor constante",
        ),
        family="trig_log_derivative_ratio",
        argument_regime="symbolic_external_scale_polynomial_trig_log_derivative",
        domain_regime="trig_log_derivative_pole_required",
        trace_regime="symbolic_external_scale_trig_log_derivative_substitution",
        presentation_regime="symbolic_external_scale_abs_log_trig_denominator",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_cotangent_log_derivative_ratio_domain",
        expr="integrate(2*k*x*cot(x^2+b), x)",
        expected_result="k·ln(|sin(x^2 + b)|)",
        expected_derivative_equivalent_to="2*k*x*cot(x^2+b)",
        expected_derivative_required_display=("sin(x^2 + b) ≠ 0",),
        expected_required_display=("sin(x^2 + b) ≠ 0",),
        expected_step_substrings=(
            "Expandir cotangente como coseno entre seno",
            "Sacar constante de una fracción",
            "Usar la regla de cot(u) -> ln|sin(u)|",
            "Identificar u y du",
            "Ajustar el factor constante",
        ),
        family="trig_log_derivative_ratio",
        argument_regime="symbolic_external_scale_polynomial_trig_log_derivative",
        domain_regime="trig_log_derivative_pole_required",
        trace_regime="symbolic_external_scale_trig_log_derivative_substitution",
        presentation_regime="symbolic_external_scale_abs_log_trig_denominator",
    ),
    IntegrateCommandMatrixCase(
        name="nested_tangent_log_derivative_denominator_verified_domain",
        expr="integrate(1/(sin(x)*cos(x)*ln(tan(x))), x)",
        expected_result="ln(|ln(tan(x))|)",
        expected_derivative_equivalent_to="1/(sin(x)*cos(x)*ln(tan(x)))",
        expected_direct_diff_integrate_result="1 / (sin(x)·cos(x)·ln(tan(x)))",
        expected_direct_diff_integrate_required_display=(
            "cos(x) ≠ 0",
            "tan(x) - 1 ≠ 0",
            "tan(x) > 0",
        ),
        expected_derivative_required_display=(
            "cos(x) ≠ 0",
            "tan(x) - 1 ≠ 0",
            "tan(x) > 0",
        ),
        expected_required_display=(
            "tan(x) - 1 ≠ 0",
            "tan(x) > 0",
        ),
        expected_step_substrings=(
            "Expandir tangente como seno entre coseno",
            "Usar la regla de u'/u -> ln|u|",
            "Identificar u y du",
            "Convertir un cociente trigonométrico en tangente",
        ),
        family="nested_trig_log_derivative_substitution",
        argument_regime="affine_trig_log_derivative_denominator",
        domain_regime="positive_log_argument_and_trig_poles",
        trace_regime="nested_log_derivative_substitution",
        presentation_regime="nested_abs_log",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_reciprocal_hyperbolic_tangent_verified_log_domain",
        expr="integrate(2*x/tanh(x^2), x)",
        expected_result="ln(|sinh(x^2)|)",
        expected_derivative_result="(x·2)/tanh(x^2)",
        expected_derivative_required_display=("sinh(x^2) ≠ 0",),
        expected_required_display=("sinh(x^2) ≠ 0",),
        expected_step_substrings=(
            "Sacar constante de una fracción",
            "Usar la regla de 1/tanh(u) -> ln|sinh(u)|",
            "Identificar u y du",
        ),
        family="explicit_reciprocal_hyperbolic_substitution",
        argument_regime="explicit_hyperbolic_tangent_reciprocal_polynomial_derivative",
        domain_regime="explicit_hyperbolic_tangent_denominator_verified_substitution",
        trace_regime="reciprocal_hyperbolic_tangent_verified_substitution",
        presentation_regime="abs_log_hyperbolic_source_condition",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_reciprocal_hyperbolic_tangent_presimplified_condition_dedupe",
        expr="integrate(2*x/tanh(x^2+0), x)",
        expected_result="ln(|sinh(x^2)|)",
        expected_derivative_result="(x·2)/tanh(x^2)",
        expected_derivative_required_display=("sinh(x^2) ≠ 0",),
        expected_required_display=("sinh(x^2) ≠ 0",),
        expected_step_substrings=(
            "Agrupar términos semejantes",
            "Sacar constante de una fracción",
            "Usar la regla de 1/tanh(u) -> ln|sinh(u)|",
            "Identificar u y du",
        ),
        family="explicit_reciprocal_hyperbolic_substitution",
        argument_regime="explicit_hyperbolic_tangent_reciprocal_presimplified_polynomial_derivative",
        domain_regime="explicit_hyperbolic_tangent_presimplified_condition_dedupe",
        trace_regime="presimplified_reciprocal_hyperbolic_tangent_verified_substitution",
        presentation_regime="abs_log_hyperbolic_condition_dedupe",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_reciprocal_hyperbolic_tangent_log_domain",
        expr="integrate(2*k*x/tanh(x^2+b), x)",
        expected_result="k·ln(|sinh(x^2 + b)|)",
        expected_derivative_equivalent_to="2*k*x/tanh(x^2+b)",
        expected_derivative_required_display=("sinh(x^2 + b) ≠ 0",),
        expected_required_display=("sinh(x^2 + b) ≠ 0",),
        expected_step_substrings=(
            "Sacar constante de una fracción",
            "Usar la regla de 1/tanh(u) -> ln|sinh(u)|",
            "Identificar u y du",
            "Ajustar el factor constante",
        ),
        family="explicit_reciprocal_hyperbolic_substitution",
        argument_regime="symbolic_external_scale_hyperbolic_tangent_reciprocal_polynomial_derivative",
        domain_regime="hyperbolic_sine_pole_required",
        trace_regime="symbolic_external_scale_reciprocal_hyperbolic_tangent",
        presentation_regime="reciprocal_hyperbolic_tangent",
    ),
    IntegrateCommandMatrixCase(
        name="negative_orientation_symbolic_external_scale_reciprocal_hyperbolic_tangent_log_domain",
        expr="integrate(2*k*x/tanh(b-x^2), x)",
        expected_result="-(k·ln(|sinh(x^2 - b)|))",
        expected_derivative_equivalent_to="2*k*x/tanh(b-x^2)",
        expected_derivative_required_display=("sinh(b - x^2) ≠ 0",),
        expected_required_display=("sinh(b - x^2) ≠ 0",),
        expected_step_substrings=(
            "Hyperbolic Negative Argument",
            "Sacar constante de una fracción",
            "Usar la regla de 1/tanh(u) -> ln|sinh(u)|",
            "Identificar u y du",
            "Ajustar el factor constante",
        ),
        family="explicit_reciprocal_hyperbolic_substitution",
        argument_regime="negative_orientation_symbolic_external_scale_hyperbolic_tangent_reciprocal_polynomial_derivative",
        domain_regime="hyperbolic_sine_pole_required",
        trace_regime="negative_orientation_symbolic_external_scale_reciprocal_hyperbolic_tangent",
        presentation_regime="negative_orientation_reciprocal_hyperbolic_tangent",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_hyperbolic_tanh_log_derivative_ratio_positive",
        expr="integrate(2*k*x*sinh(x^2+b)/cosh(x^2+b), x)",
        expected_result="k·ln(cosh(x^2 + b))",
        expected_derivative_result="(sinh(x^2 + b)·x·k·2)/cosh(x^2 + b)",
        expected_step_substrings=(
            "Sacar constante de una fracción",
            "Usar la regla de tanh(u) -> ln(cosh(u))",
            "Identificar u y du",
            "Ajustar el factor constante",
            "Abs Under Positivity",
        ),
        family="hyperbolic_log_derivative_ratio",
        argument_regime="symbolic_external_scale_polynomial_hyperbolic_log_derivative",
        domain_regime="unconditional_cosh_positive",
        trace_regime="symbolic_external_scale_hyperbolic_log_derivative_substitution",
        presentation_regime="symbolic_external_scale_positive_cosh_log",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_hyperbolic_tanh_direct_log_positive",
        expr="integrate(2*k*x*tanh(x^2+b), x)",
        expected_result="k·ln(cosh(x^2 + b))",
        expected_derivative_result="(sinh(x^2 + b)·x·k·2)/cosh(x^2 + b)",
        expected_step_substrings=(
            "Usar la regla de tanh(u) -> ln(cosh(u))",
            "Identificar u y du",
            "Ajustar el factor constante",
            "Abs Under Positivity",
        ),
        family="hyperbolic_tanh_log_derivative",
        argument_regime="symbolic_external_scale_polynomial_hyperbolic_tanh",
        domain_regime="unconditional_cosh_positive",
        trace_regime="symbolic_external_scale_hyperbolic_tanh_substitution",
        presentation_regime="symbolic_external_scale_positive_cosh_log",
    ),
    IntegrateCommandMatrixCase(
        name="additive_trig_pole_residual_domain",
        expr="integrate(tan(x^2)+sin(x^2), x)",
        expected_result="integrate(sin(x^2) + tan(x^2), x)",
        expected_required_display=("cos(x^2) ≠ 0",),
        expected_step_substrings=("Conservar integral residual",),
        family="trig_additive_residual_domain",
        argument_regime="additive_nonlinear_trig_residual",
        domain_regime="trig_pole_additive_residual",
        outcome="residual",
        residual_cause="non_elementary_composition",
        trace_regime="residual_policy_with_domain",
        presentation_regime="residual",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_table",
        expr="integrate(1/(x^2+1), x)",
        expected_result="arctan(x)",
        expected_derivative_result="1 / (x^2 + 1)",
        expected_direct_diff_integrate_result="1 / (x^2 + 1)",
        expected_step_substrings=("Usar la regla de arctan con derivada interna",),
        family="inverse_trig_table",
        argument_regime="rational_expression",
        trace_regime="inverse_trig_table",
        presentation_regime="inverse_trig",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_affine_shifted_scaled_positive_quadratic_table",
        expr="integrate(1/((2*x+3)^2+4), x)",
        expected_result="1/4·arctan((2·x + 3) / 2)",
        expected_derivative_result="1 / (4·x^2 + 12·x + 13)",
        expected_direct_diff_integrate_result="1 / ((2·x + 3)^2 + 4)",
        expected_step_substrings=(
            "Usar la regla de arctan con derivada interna",
            "Identificar el argumento afín",
            "Ajustar el factor constante",
        ),
        family="inverse_trig_table",
        argument_regime="affine_shifted_scaled_positive_quadratic",
        domain_regime="structurally_positive_denominator",
        trace_regime="arctan_affine_positive_quadratic_table",
        presentation_regime="scaled_arctan_affine",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_symbolic_affine_expanded_square_radius_positive_quadratic_table",
        expr="integrate(1/((a*x+b)^2+4), x)",
        expected_result="arctan((a·x + b) / 2) / (2·a)",
        expected_derivative_equivalent_to="1/((a*x+b)^2+4)",
        expected_required_display=("a ≠ 0",),
        expected_derivative_required_display=("a ≠ 0",),
        expected_direct_diff_integrate_result="1 / ((a·x + b)^2 + 4)",
        expected_direct_diff_integrate_required_display=("a ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="inverse_trig_table",
        argument_regime="symbolic_affine_expanded_square_radius_positive_quadratic",
        domain_regime="symbolic_scale_nonzero_required",
        trace_regime="arctan_symbolic_affine_expanded_square_radius_positive_quadratic_table",
        presentation_regime="symbolic_affine_expanded_square_radius_arctan",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_symbolic_affine_positive_rational_radius_positive_quadratic_table",
        expr="integrate(1/((a*x+b)^2+2), x)",
        expected_result="arctan(sqrt(2)·(a·x + b) / 2) / (sqrt(2)·a)",
        expected_derivative_equivalent_to="1/((a*x+b)^2+2)",
        expected_required_display=("a ≠ 0",),
        expected_derivative_required_display=("a ≠ 0",),
        expected_direct_diff_integrate_result="1 / ((a·x + b)^2 + 2)",
        expected_direct_diff_integrate_required_display=("a ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="inverse_trig_table",
        argument_regime="symbolic_affine_positive_rational_radius_positive_quadratic",
        domain_regime="symbolic_scale_nonzero_required",
        trace_regime="arctan_symbolic_affine_positive_rational_radius_positive_quadratic_table",
        presentation_regime="symbolic_affine_positive_rational_radius_arctan",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_named_positive_constant_radius_positive_quadratic_table",
        expr="integrate(1/(x^2+phi), x)",
        expected_result="arctan(x·phi^(-1/2)) / sqrt(phi)",
        expected_derivative_equivalent_to="1/(x^2+phi)",
        expected_direct_diff_integrate_result="1 / (x^2 + phi)",
        expected_step_substrings=(
            "Usar la regla de arctan con derivada interna",
            "Identificar el argumento afín",
            "Ajustar el factor constante",
        ),
        family="inverse_trig_table",
        argument_regime="named_positive_constant_radius_positive_quadratic",
        domain_regime="structurally_positive_denominator",
        trace_regime="arctan_named_positive_constant_radius_positive_quadratic_table",
        presentation_regime="named_positive_constant_radius_arctan",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_numeric_affine_named_positive_constant_radius_positive_quadratic_table",
        expr="integrate(1/((2*x+3)^2+phi), x)",
        expected_result="arctan(phi^(-1/2)·(2·x + 3)) / (2·sqrt(phi))",
        expected_derivative_equivalent_to="1/((2*x+3)^2+phi)",
        expected_direct_diff_integrate_result="1 / ((2·x + 3)^2 + phi)",
        expected_step_substrings=(
            "Usar la regla de arctan con derivada interna",
            "Identificar el argumento afín",
            "Ajustar el factor constante",
        ),
        family="inverse_trig_table",
        argument_regime="numeric_affine_named_positive_constant_radius_positive_quadratic",
        domain_regime="structurally_positive_denominator",
        trace_regime="arctan_numeric_affine_named_positive_constant_radius_positive_quadratic_table",
        presentation_regime="numeric_affine_named_positive_constant_radius_arctan",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_expanded_numeric_affine_named_positive_constant_radius_positive_quadratic_table",
        expr="integrate(1/(4*x^2+12*x+9+phi), x)",
        expected_result="arctan(phi^(-1/2)·(2·x + 3)) / (2·sqrt(phi))",
        expected_derivative_equivalent_to="1/(4*x^2+12*x+9+phi)",
        expected_direct_diff_integrate_result="1 / (4·x^2 + 12·x + 9 + phi)",
        expected_step_substrings=(
            "Usar la regla de arctan con derivada interna",
            "Identificar el argumento afín",
            "Ajustar el factor constante",
        ),
        family="inverse_trig_table",
        argument_regime="expanded_numeric_affine_named_positive_constant_radius_positive_quadratic",
        domain_regime="structurally_positive_denominator",
        trace_regime="arctan_expanded_numeric_affine_named_positive_constant_radius_positive_quadratic_table",
        presentation_regime="expanded_numeric_affine_named_positive_constant_radius_arctan",
    ),
    IntegrateCommandMatrixCase(
        name="rational_positive_quadratic_linear_numerator_expanded_named_positive_radius_decomposition",
        expr="integrate((2*x+6)/(4*x^2+12*x+9+phi), x)",
        expected_result="1/4·ln(4·x^2 + 12·x + 9 + phi) + (atan((2·x + 3) / sqrt(phi))·3)/(2·sqrt(phi))",
        expected_derivative_equivalent_to="(2*x+6)/(4*x^2+12*x+9+phi)",
        expected_direct_diff_integrate_result="(2·x + 6) / (4·x^2 + 12·x + 9 + phi)",
        expected_step_substrings=(
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="rational_positive_quadratic_linear_numerator",
        argument_regime="expanded_numeric_affine_named_positive_constant_radius_linear_numerator",
        domain_regime="structurally_positive_denominator",
        trace_regime="positive_quadratic_linear_numerator_named_positive_radius_decomposition",
        presentation_regime="positive_log_plus_named_positive_radius_arctan",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_symbolic_square_radius_positive_quadratic_table",
        expr="integrate(1/(x^2+a^2), x)",
        expected_result="arctan(x / a) / a",
        expected_derivative_equivalent_to="1/(x^2+a^2)",
        expected_required_display=("a ≠ 0",),
        expected_derivative_required_display=("a ≠ 0",),
        expected_direct_diff_integrate_result="1 / (a^2 + x^2)",
        expected_direct_diff_integrate_required_display=("a ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="inverse_trig_table",
        argument_regime="symbolic_square_radius_positive_quadratic",
        domain_regime="symbolic_radius_nonzero_required",
        trace_regime="arctan_symbolic_square_radius_positive_quadratic_table",
        presentation_regime="symbolic_square_radius_arctan",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_symbolic_shifted_square_radius_positive_quadratic_table",
        expr="integrate(1/((x+b)^2+a^2), x)",
        expected_result="arctan((b + x) / a) / a",
        expected_required_display=("a ≠ 0",),
        expected_direct_diff_integrate_result="1 / ((b + x)^2 + a^2)",
        expected_direct_diff_integrate_required_display=("a ≠ 0",),
        expected_step_substrings=(
            "Usar la regla de arctan con derivada interna",
            "Identificar el argumento afín",
            "Ajustar el factor constante",
        ),
        family="inverse_trig_table",
        argument_regime="symbolic_shifted_square_radius_positive_quadratic",
        domain_regime="symbolic_radius_nonzero_required",
        trace_regime="arctan_symbolic_shifted_square_radius_positive_quadratic_table",
        presentation_regime="symbolic_shifted_square_radius_arctan",
    ),
    IntegrateCommandMatrixCase(
        name="rational_positive_quadratic_square_reduction",
        expr="integrate(1/(x^2+1)^2, x)",
        expected_result="1/2·arctan(x) + x / (2·(x^2 + 1))",
        expected_derivative_result="1 / (x^2 + 1)^2",
        direct_diff_integrate_from_derivative=True,
        expected_step_substrings=(
            "Reducir el cuadrático positivo al cuadrado",
            "Integrar la parte arctan y la parte racional",
        ),
        family="rational_positive_quadratic_power",
        argument_regime="repeated_irreducible_positive_quadratic_denominator",
        domain_regime="structurally_positive_denominator",
        trace_regime="positive_quadratic_square_reduction",
        presentation_regime="arctan_plus_rational_term",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_sqrt_reciprocal_bridge",
        expr="integrate(1/(sqrt(x)*(x+1)), x)",
        expected_result="2·arctan(sqrt(x))",
        expected_derivative_result="1 / ((x + 1)·sqrt(x))",
        expected_derivative_required_display=("x > 0",),
        expected_required_display=("x > 0",),
        expected_direct_diff_integrate_result="1 / (sqrt(x)·(x + 1))",
        expected_direct_diff_integrate_required_display=("x > 0",),
        expected_step_substrings=(
            "Usar la regla de u'/(1+u^2) -> arctan(u)",
            "Identificar u y du",
            "u = \\sqrt{x}",
            "du = \\frac{1}{2\\sqrt{x}}\\,dx",
        ),
        family="inverse_trig_root_reciprocal",
        argument_regime="sqrt_reciprocal_linear",
        domain_regime="positive_required",
        trace_regime="arctan_sqrt_reciprocal_substitution",
        presentation_regime="scaled_inverse_trig_root",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_scaled_sqrt_reciprocal_bridge",
        expr="integrate(1/(sqrt(x)*(4*x+1)), x)",
        expected_result="arctan(2·sqrt(x))",
        expected_derivative_result="1 / (sqrt(x)·(4·x + 1))",
        expected_derivative_required_display=("x > 0",),
        expected_required_display=("x > 0",),
        expected_direct_diff_integrate_result="1 / (sqrt(x)·(4·x + 1))",
        expected_direct_diff_integrate_required_display=("x > 0",),
        expected_step_substrings=(
            "Usar la regla de u'/(1+u^2) -> arctan(u)",
            "Identificar u y du",
            "u = 2\\cdot \\sqrt{x}",
            "du = \\frac{1}{\\sqrt{x}}\\,dx",
        ),
        family="inverse_trig_root_reciprocal",
        argument_regime="scaled_sqrt_reciprocal_linear",
        domain_regime="positive_required",
        trace_regime="arctan_scaled_sqrt_reciprocal_substitution",
        presentation_regime="scaled_root_argument_inverse_trig",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_symbolic_denominator_scale_sqrt_reciprocal_bridge",
        expr="integrate(1/(sqrt(x)*(x+a^2)), x)",
        expected_result="(2·arctan(sqrt(x) / a))/a",
        expected_derivative_result="(x^(1/2)·2)/(2·(x·a^2 + x^2))",
        expected_derivative_required_display=("a ≠ 0", "x > 0"),
        expected_required_display=("a ≠ 0", "x > 0"),
        expected_direct_diff_integrate_result="1 / (sqrt(x)·(a^2 + x))",
        expected_direct_diff_integrate_required_display=("a ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar la regla de u'/(1+u^2) -> arctan(u)",
            "Identificar u y du",
        ),
        family="inverse_trig_root_reciprocal",
        argument_regime="symbolic_denominator_scaled_sqrt_reciprocal_linear",
        domain_regime="symbolic_denominator_scale_positive_required",
        trace_regime="arctan_symbolic_denominator_scaled_sqrt_reciprocal_substitution",
        presentation_regime="symbolic_denominator_scaled_root_argument_inverse_trig",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_symbolic_denominator_numeric_square_scale_sqrt_reciprocal_bridge",
        expr="integrate(1/(sqrt(x)*(4*x+a^2)), x)",
        expected_result="arctan(2·sqrt(x) / a) / a",
        expected_derivative_result="2·a^2 / ((2·a^4 + 8·x·a^2)·sqrt(x))",
        expected_derivative_required_display=("a ≠ 0", "x > 0"),
        expected_required_display=("a ≠ 0", "x > 0"),
        expected_direct_diff_integrate_result="1 / (sqrt(x)·(a^2 + 4·x))",
        expected_direct_diff_integrate_required_display=("a ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar la regla de u'/(1+u^2) -> arctan(u)",
            "Identificar u y du",
            "u = \\frac{2\\cdot \\sqrt{x}}{a}",
            "du = \\frac{1}{a\\cdot \\sqrt{x}}\\,dx",
        ),
        family="inverse_trig_root_reciprocal",
        argument_regime="symbolic_denominator_numeric_square_scaled_sqrt_reciprocal_linear",
        domain_regime="symbolic_denominator_scale_positive_required",
        trace_regime="arctan_symbolic_denominator_numeric_square_scaled_sqrt_reciprocal_substitution",
        presentation_regime="symbolic_denominator_numeric_square_scaled_root_argument_inverse_trig",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_symbolic_numerator_scale_sqrt_reciprocal_bridge",
        expr="integrate(1/(sqrt(x)*(a^2*x+1)), x)",
        expected_result="(2·arctan(a·sqrt(x)))/a",
        expected_derivative_result="2 / ((2·x·a^2 + 2)·sqrt(x))",
        expected_derivative_required_display=("a ≠ 0", "x > 0"),
        expected_required_display=("a ≠ 0", "x > 0"),
        expected_direct_diff_integrate_result="1 / (sqrt(x)·(x·a^2 + 1))",
        expected_direct_diff_integrate_required_display=("a ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar la regla de u'/(1+u^2) -> arctan(u)",
            "Identificar u y du",
            "u = a\\cdot \\sqrt{x}",
            "du = \\frac{a}{2\\cdot \\sqrt{x}}\\,dx",
        ),
        family="inverse_trig_root_reciprocal",
        argument_regime="symbolic_numerator_scaled_sqrt_reciprocal_linear",
        domain_regime="symbolic_numerator_scale_positive_required",
        trace_regime="arctan_symbolic_numerator_scaled_sqrt_reciprocal_substitution",
        presentation_regime="symbolic_numerator_scaled_root_argument_inverse_trig",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_shifted_scaled_sqrt_reciprocal_bridge",
        expr="integrate(1/(sqrt(x+1)*(4*x+5)), x)",
        expected_result="arctan(2·sqrt(x + 1))",
        expected_derivative_result="1 / (sqrt(x + 1)·(4·x + 5))",
        expected_derivative_required_display=("x > -1",),
        expected_required_display=("x > -1",),
        expected_direct_diff_integrate_result="1 / (sqrt(x + 1)·(4·x + 5))",
        expected_direct_diff_integrate_required_display=("x > -1",),
        expected_step_substrings=(
            "Usar la regla de u'/(1+u^2) -> arctan(u)",
            "Identificar u y du",
            "u = 2\\cdot \\sqrt{x + 1}",
            "du = \\frac{1}{\\sqrt{x + 1}}\\,dx",
        ),
        family="inverse_trig_root_reciprocal",
        argument_regime="shifted_scaled_sqrt_reciprocal_linear",
        domain_regime="positive_required",
        trace_regime="arctan_shifted_scaled_sqrt_reciprocal_substitution",
        presentation_regime="scaled_shifted_root_argument_inverse_trig",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_hyperbolic_rational_direct_atanh_domain",
        expr="integrate(1/(1-x^2), x)",
        expected_result="atanh(x)",
        expected_derivative_result="1 / (1 - x^2)",
        expected_derivative_required_display=("-1 < x < 1",),
        expected_required_display=("-1 < x < 1",),
        expected_direct_diff_integrate_result="1 / (1 - x^2)",
        expected_direct_diff_integrate_required_display=("-1 < x < 1",),
        expected_step_substrings=("Usar la regla de atanh con derivada interna",),
        family="inverse_hyperbolic_rational_table",
        argument_regime="bounded_rational_expression",
        domain_regime="rational_interval",
        trace_regime="inverse_hyperbolic_rational_direct_table",
        presentation_regime="inverse_hyperbolic",
    ),
    IntegrateCommandMatrixCase(
        name="rational_partial_fraction_two_real_linear_factors",
        expr="integrate(1/(x^2-1), x)",
        expected_result="1/2·ln(|(x - 1) / (x + 1)|)",
        expected_derivative_result="1 / (x^2 - 1)",
        expected_derivative_required_display=("x ≠ 1", "x ≠ -1"),
        expected_direct_diff_integrate_result="1 / (x^2 - 1)",
        expected_direct_diff_integrate_required_display=("x ≠ -1", "x ≠ 1"),
        expected_required_display=("x ≠ 1", "x ≠ -1"),
        expected_step_substrings=(
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="rational_partial_fraction",
        argument_regime="two_real_linear_factors",
        domain_regime="linear_poles_required",
        trace_regime="partial_fraction_linear_factors",
        presentation_regime="log_ratio_abs",
    ),
    IntegrateCommandMatrixCase(
        name="rational_partial_fraction_three_real_linear_factors",
        expr="integrate(1/(x*(x-1)*(x+1)), x)",
        expected_result="1/2·ln(|x - 1|) + 1/2·ln(|x + 1|) - ln(|x|)",
        expected_derivative_result="1 / (x·(x^2 - 1))",
        expected_derivative_required_display=("x ≠ 1", "x ≠ -1", "x ≠ 0"),
        expected_direct_diff_integrate_result="1 / (x·(x + 1)·(x - 1))",
        expected_direct_diff_integrate_required_display=(
            "x ≠ -1",
            "x ≠ 0",
            "x ≠ 1",
        ),
        expected_required_display=("x ≠ -1", "x ≠ 1", "x ≠ 0"),
        expected_step_substrings=(
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="rational_three_linear_partial_fraction",
        argument_regime="three_real_linear_factors",
        domain_regime="linear_poles_required",
        trace_regime="partial_fraction_three_linear_factors",
        presentation_regime="three_linear_log_terms",
    ),
    IntegrateCommandMatrixCase(
        name="rational_partial_fraction_mixed_simple_repeated_linear_factors",
        expr="integrate(1/(x*(x+1)^2), x)",
        expected_result="ln(|x / (x + 1)|) + 1 / (x + 1)",
        expected_derivative_result="1 / (x^3 + 2·x^2 + x)",
        expected_derivative_required_display=("x ≠ 0", "x ≠ -1"),
        expected_direct_diff_integrate_result="1 / (x·(x + 1)^2)",
        expected_direct_diff_integrate_required_display=("x ≠ -1", "x ≠ 0"),
        expected_required_display=("x ≠ 0", "x ≠ -1"),
        expected_step_substrings=(
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="rational_mixed_simple_repeated_linear_partial_fraction",
        argument_regime="mixed_simple_repeated_real_linear_factors",
        domain_regime="linear_poles_required",
        trace_regime="partial_fraction_mixed_simple_repeated_linear_factors",
        presentation_regime="log_ratio_plus_single_rational_pole",
    ),
    IntegrateCommandMatrixCase(
        name="rational_partial_fraction_repeated_real_linear_factors",
        expr="integrate(1/(x^2-1)^2, x)",
        expected_result="1/4·ln(|(x + 1) / (x - 1)|) - 1 / (4·(x - 1)) - 1 / (4·(x + 1))",
        expected_derivative_equivalent_to="1/(x^2-1)^2",
        expected_derivative_required_display=("x ≠ -1", "x ≠ 1"),
        expected_required_display=("x ≠ -1", "x ≠ 1"),
        expected_direct_diff_integrate_result="1 / (x^2 - 1)^2",
        expected_direct_diff_integrate_required_display=("x ≠ -1", "x ≠ 1"),
        expected_step_substrings=(
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="rational_repeated_linear_partial_fraction",
        argument_regime="repeated_two_real_linear_factors",
        domain_regime="linear_poles_required",
        trace_regime="partial_fraction_repeated_linear_factors",
        presentation_regime="log_ratio_plus_rational_poles",
    ),
    IntegrateCommandMatrixCase(
        name="rational_partial_fraction_mixed_linear_positive_quadratic",
        expr="integrate(1/(x^4-1), x)",
        expected_result="1/4·ln(|x - 1|) - 1/2·arctan(x) - 1/4·ln(|x + 1|)",
        expected_derivative_result="1 / (2·(x^2 - 1)) - 1 / (2·(x^2 + 1))",
        expected_derivative_required_display=("x ≠ -1", "x ≠ 1"),
        expected_direct_diff_integrate_result="1 / (x^4 - 1)",
        expected_direct_diff_integrate_required_display=("x ≠ -1", "x ≠ 1"),
        expected_required_display=("x ≠ -1", "x ≠ 1"),
        expected_step_substrings=(
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="rational_mixed_linear_positive_quadratic_partial_fraction",
        argument_regime="multi_linear_positive_quadratic_factors",
        domain_regime="linear_poles_required",
        trace_regime="partial_fraction_mixed_linear_positive_quadratic",
        presentation_regime="linear_logs_plus_arctan",
    ),
    IntegrateCommandMatrixCase(
        name="rational_partial_fraction_repeated_linear_positive_quadratic",
        expr="integrate(1/((x-1)^2*(x^2+1)), x)",
        expected_result="1/4·ln(x^2 + 1) - 1/2·ln(|x - 1|) - 1 / (2·(x - 1))",
        expected_derivative_result="1 / (x^4 + 2·x^2 + 1 - 2·x^3 - 2·x)",
        expected_derivative_required_display=("x ≠ 1",),
        expected_direct_diff_integrate_result="1 / ((x - 1)^2·(x^2 + 1))",
        expected_direct_diff_integrate_required_display=("x ≠ 1",),
        expected_required_display=("x ≠ 1",),
        expected_step_substrings=(
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="rational_repeated_linear_positive_quadratic_partial_fraction",
        argument_regime="repeated_linear_positive_quadratic_factors",
        domain_regime="linear_poles_required",
        trace_regime="partial_fraction_repeated_linear_positive_quadratic",
        presentation_regime="positive_log_abs_log_plus_rational_pole",
    ),
    IntegrateCommandMatrixCase(
        name="rational_partial_fraction_repeated_origin_linear_positive_quadratic_no_log",
        expr="integrate(1/(x^2*(x^2+1)), x)",
        expected_result="-1 / x - arctan(x)",
        expected_derivative_equivalent_to="1/(x^2*(x^2+1))",
        expected_derivative_required_display=("x ≠ 0",),
        expected_direct_diff_integrate_result="1 / (x^2·(x^2 + 1))",
        expected_direct_diff_integrate_required_display=("x ≠ 0",),
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=(
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="rational_repeated_origin_linear_positive_quadratic_partial_fraction",
        argument_regime="repeated_origin_linear_positive_quadratic_factors",
        domain_regime="linear_poles_required",
        trace_regime="partial_fraction_repeated_origin_linear_positive_quadratic",
        presentation_regime="arctan_plus_rational_pole_no_log",
    ),
    IntegrateCommandMatrixCase(
        name="rational_partial_fraction_repeated_origin_scaled_positive_quadratic_no_log",
        expr="integrate(1/(x^2*(x^2+4)), x)",
        expected_result="-1/8·arctan(x / 2) - 1 / (4·x)",
        expected_derivative_equivalent_to="1/(x^2*(x^2+4))",
        expected_derivative_required_display=("x ≠ 0",),
        expected_direct_diff_integrate_result="1 / (x^2·(x^2 + 4))",
        expected_direct_diff_integrate_required_display=("x ≠ 0",),
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=(
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="rational_repeated_origin_linear_positive_quadratic_partial_fraction",
        argument_regime="repeated_origin_linear_scaled_positive_quadratic_factors",
        domain_regime="linear_poles_required",
        trace_regime="partial_fraction_repeated_origin_scaled_positive_quadratic",
        presentation_regime="scaled_arctan_plus_rational_pole_no_log",
    ),
    IntegrateCommandMatrixCase(
        name="rational_improper_partial_fraction_polynomial_division",
        expr="integrate((x^2+1)/(x^2-1), x)",
        expected_result="ln(|x - 1|) + x - ln(|x + 1|)",
        expected_derivative_result="2 / (x^2 - 1) + 1",
        expected_derivative_required_display=("x ≠ -1", "x ≠ 1"),
        expected_direct_diff_integrate_result="(x^2 + 1) / (x^2 - 1)",
        expected_direct_diff_integrate_required_display=("x ≠ -1", "x ≠ 1"),
        expected_required_display=("x ≠ -1", "x ≠ 1"),
        expected_step_substrings=(
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="rational_improper_partial_fraction",
        argument_regime="polynomial_division_two_real_linear_factors",
        domain_regime="linear_poles_required",
        trace_regime="polynomial_division_partial_fraction",
        presentation_regime="polynomial_plus_log_terms",
    ),
    IntegrateCommandMatrixCase(
        name="rational_improper_positive_quadratic_polynomial_division",
        expr="integrate((x^2+1)/(x^2+2*x+2), x)",
        expected_result="arctan(x + 1) + x - ln(x^2 + 2·x + 2)",
        expected_derivative_result="(x^2 + 1) / (x^2 + 2·x + 2)",
        direct_diff_integrate_from_derivative=True,
        expected_step_substrings=(
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="rational_improper_positive_quadratic",
        argument_regime="polynomial_division_irreducible_positive_quadratic",
        domain_regime="structurally_positive_denominator",
        trace_regime="polynomial_division_positive_quadratic_decomposition",
        presentation_regime="polynomial_plus_positive_log_and_arctan",
    ),
    IntegrateCommandMatrixCase(
        name="rational_improper_positive_quadratic_negative_orientation",
        expr="integrate((x^2+1)/(-x^2-2*x-2), x)",
        expected_result="ln(x^2 + 2·x + 2) - arctan(x + 1) - x",
        expected_derivative_equivalent_to="(x^2+1)/(-x^2-2*x-2)",
        expected_direct_diff_integrate_result="(-x^2 - 1) / (x^2 + 2·x + 2)",
        expected_step_substrings=(
            "Descomponer en fracciones parciales",
            "Integrar los términos simples",
        ),
        family="rational_improper_positive_quadratic",
        argument_regime="negative_orientation_polynomial_division_irreducible_positive_quadratic",
        domain_regime="structurally_nonzero_negative_quadratic_denominator",
        trace_regime="negative_orientation_polynomial_division_positive_quadratic_decomposition",
        presentation_regime="negative_orientation_polynomial_plus_positive_log_and_arctan",
    ),
    IntegrateCommandMatrixCase(
        name="affine_inverse_hyperbolic_atanh_domain",
        expr="integrate(2/(4-(2*x+1)^2), x)",
        expected_result="1/2·atanh((2·x + 1) / 2)",
        expected_derivative_result="2 / (3 - 4·x^2 - 4·x)",
        expected_derivative_required_display=("-3/2 < x < 1/2",),
        expected_required_display=("-3/2 < x < 1/2",),
        expected_direct_diff_integrate_result="2 / (4 - (2·x + 1)^2)",
        expected_direct_diff_integrate_required_display=("-3/2 < x < 1/2",),
        expected_step_substrings=(
            "Usar la regla de atanh con derivada interna",
            "Identificar el argumento afín",
            "Ajustar el factor constante",
        ),
        family="inverse_hyperbolic_rational_affine",
        argument_regime="affine_bounded_rational",
        domain_regime="rational_interval",
        trace_regime="inverse_hyperbolic_rational_affine_table",
        presentation_regime="affine_atanh",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_sqrt_direct_arcsin_domain",
        expr="integrate(1/sqrt(1-x^2), x)",
        expected_result="arcsin(x)",
        expected_derivative_result="1 / sqrt(1 - x^2)",
        expected_derivative_required_display=("-1 < x < 1",),
        expected_required_display=("-1 < x < 1",),
        expected_direct_diff_integrate_result="1 / sqrt(1 - x^2)",
        expected_direct_diff_integrate_required_display=("-1 < x < 1",),
        expected_step_substrings=("Usar la regla de arcsin con derivada interna",),
        family="inverse_sqrt_table",
        argument_regime="bounded_variable_radical",
        domain_regime="radical_interval",
        trace_regime="inverse_sqrt_direct_table",
        presentation_regime="inverse_trig",
    ),
    IntegrateCommandMatrixCase(
        name="additive_inverse_sqrt_interval_residual_domain",
        expr="integrate(1/sqrt(1-x^2)+sin(x^2), x)",
        expected_result="integrate(sin(x^2) + (1 - x^2)^(-1/2), x)",
        expected_required_display=("-1 < x < 1",),
        expected_step_substrings=(
            "Reescribir la raíz como potencia fraccionaria",
            "Canonicalize Reciprocal Sqrt",
            "Conservar integral residual",
        ),
        family="inverse_sqrt_additive_residual_domain",
        argument_regime="additive_interval_radical_residual",
        domain_regime="radical_interval_additive_residual",
        outcome="residual",
        residual_cause="branch_sensitive_interval_residual",
        trace_regime="residual_presimplification_with_domain",
        presentation_regime="residual_reciprocal_sqrt_power",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_sqrt_symbolic_radius_arcsin_domain",
        expr="integrate(1/sqrt(a^2-x^2), x)",
        expected_result="arcsin(x / sqrt(a^2))",
        expected_required_display=("a^2 - x^2 > 0", "a ≠ 0"),
        expected_direct_diff_integrate_result="1 / sqrt(a^2 - x^2)",
        expected_direct_diff_integrate_required_display=("a^2 - x^2 > 0",),
        expected_step_substrings=(
            "Usar la regla de arcsin con derivada interna",
        ),
        family="inverse_sqrt_symbolic_radius",
        argument_regime="symbolic_interval_radical",
        domain_regime="symbolic_radical_interval_verified",
        trace_regime="inverse_sqrt_symbolic_radius_table",
        presentation_regime="symbolic_radius_arcsin",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_sqrt_symbolic_radius_shifted_arcsin_domain",
        expr="integrate(1/sqrt(a^2-(x+b)^2), x)",
        expected_result="arcsin((b + x) / sqrt(a^2))",
        expected_required_display=(
            "a^2 - b^2 - x^2 - 2·b·x > 0",
            "a ≠ 0",
        ),
        expected_direct_diff_integrate_result="1 / sqrt(a^2 - (b + x)^2)",
        expected_direct_diff_integrate_required_display=(
            "a^2 - b^2 - x^2 - 2·b·x > 0",
        ),
        expected_step_substrings=(
            "Usar la regla de arcsin con derivada interna",
            "Identificar el argumento afín",
        ),
        family="inverse_sqrt_symbolic_radius",
        argument_regime="symbolic_shifted_interval_radical",
        domain_regime="symbolic_shifted_radical_interval_verified",
        trace_regime="inverse_sqrt_symbolic_radius_shifted_table",
        presentation_regime="symbolic_shifted_radius_arcsin",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_sqrt_symbolic_slope_shifted_arcsin_domain",
        expr="integrate(1/sqrt(a^2-(m*x+b)^2), x)",
        expected_result="arcsin((m·x + b) / sqrt(a^2)) / m",
        expected_required_display=(
            "a^2 - m^2·x^2 - 2·b·m·x - b^2 > 0",
            "m ≠ 0",
            "a ≠ 0",
        ),
        expected_direct_diff_integrate_result="1 / sqrt(a^2 - (m·x + b)^2)",
        expected_direct_diff_integrate_required_display=(
            "a^2 - m^2·x^2 - 2·b·m·x - b^2 > 0",
            "m ≠ 0",
        ),
        expected_step_substrings=(
            "Usar la regla de arcsin con derivada interna",
            "Identificar el argumento afín",
            "Ajustar el factor constante",
        ),
        family="inverse_sqrt_symbolic_radius",
        argument_regime="symbolic_slope_shifted_interval_radical",
        domain_regime="symbolic_slope_shifted_radical_interval_verified",
        trace_regime="inverse_sqrt_symbolic_slope_shifted_table",
        presentation_regime="symbolic_slope_shifted_radius_arcsin",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_sqrt_direct_asinh_unconditional",
        expr="integrate(1/sqrt(x^2+1), x)",
        expected_result="asinh(x)",
        expected_derivative_result="1 / sqrt(x^2 + 1)",
        expected_direct_diff_integrate_result="1 / sqrt(x^2 + 1)",
        expected_step_substrings=("Usar la regla de asinh con derivada interna",),
        family="inverse_hyperbolic_sqrt_table",
        argument_regime="unbounded_variable_radical",
        trace_regime="inverse_sqrt_hyperbolic_direct_table",
        presentation_regime="inverse_hyperbolic",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_hyperbolic_sqrt_symbolic_radius_table",
        expr="integrate(1/sqrt(x^2+a^2), x)",
        expected_result="asinh(x / sqrt(a^2))",
        expected_required_display=("a ≠ 0",),
        expected_direct_diff_integrate_result="1 / sqrt(a^2 + x^2)",
        expected_direct_diff_integrate_required_display=("a^2 + x^2 > 0",),
        expected_step_substrings=(
            "Usar la regla de asinh con derivada interna",
            "Identificar el argumento afín",
        ),
        family="inverse_hyperbolic_sqrt_symbolic_radius",
        argument_regime="symbolic_positive_square_radius",
        domain_regime="symbolic_radius_nonzero_required",
        trace_regime="asinh_symbolic_radius_table",
        presentation_regime="symbolic_radius_asinh",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_hyperbolic_sqrt_symbolic_shifted_radius_table",
        expr="integrate(1/sqrt((x+b)^2+a^2), x)",
        expected_result="asinh((b + x) / sqrt(a^2))",
        expected_required_display=("a^2 + b^2 + x^2 + 2·b·x > 0", "a ≠ 0"),
        expected_direct_diff_integrate_result="1 / sqrt((b + x)^2 + a^2)",
        expected_direct_diff_integrate_required_display=(
            "a^2 + b^2 + x^2 + 2·b·x > 0",
        ),
        expected_step_substrings=(
            "Usar la regla de asinh con derivada interna",
            "Identificar el argumento afín",
        ),
        family="inverse_hyperbolic_sqrt_symbolic_radius",
        argument_regime="symbolic_shifted_positive_square_radius",
        domain_regime="symbolic_positive_square_radius_verified",
        trace_regime="asinh_symbolic_shifted_radius_table",
        presentation_regime="symbolic_shifted_radius_asinh",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_hyperbolic_sqrt_symbolic_slope_shifted_radius_table",
        expr="integrate(1/sqrt((m*x+b)^2+a^2), x)",
        expected_result="asinh((m·x + b) / sqrt(a^2)) / m",
        expected_required_display=(
            "m ≠ 0",
            "m^2·x^2 + 2·b·m·x + a^2 + b^2 > 0",
            "a ≠ 0",
        ),
        expected_direct_diff_integrate_result="1 / sqrt((m·x + b)^2 + a^2)",
        expected_direct_diff_integrate_required_display=(
            "m ≠ 0",
            "m^2·x^2 + 2·b·m·x + a^2 + b^2 > 0",
        ),
        expected_step_substrings=(
            "Usar la regla de asinh con derivada interna",
            "Identificar el argumento afín",
            "Ajustar el factor constante",
        ),
        family="inverse_hyperbolic_sqrt_symbolic_radius",
        argument_regime="symbolic_slope_shifted_positive_square_radius",
        domain_regime="symbolic_slope_positive_square_radius_verified",
        trace_regime="asinh_symbolic_slope_shifted_radius_table",
        presentation_regime="symbolic_slope_shifted_radius_asinh",
    ),
    IntegrateCommandMatrixCase(
        name="affine_inverse_sqrt_arcsin_domain",
        expr="integrate(1/sqrt(4-(x+1)^2), x)",
        expected_result="arcsin((x + 1) / 2)",
        expected_derivative_result="1 / sqrt(3 - x^2 - 2·x)",
        expected_derivative_required_display=("-3 < x < 1",),
        expected_required_display=("-3 < x < 1",),
        expected_direct_diff_integrate_result="1 / sqrt(4 - (x + 1)^2)",
        expected_direct_diff_integrate_required_display=("-3 < x < 1",),
        expected_step_substrings=(
            "Usar la regla de arcsin con derivada interna",
            "Identificar el argumento afín",
        ),
        family="inverse_sqrt_affine",
        argument_regime="affine_radical",
        domain_regime="radical_interval",
        trace_regime="inverse_sqrt_affine_table",
        presentation_regime="affine_arcsin",
    ),
    IntegrateCommandMatrixCase(
        name="polynomial_base_sqrt_substitution",
        expr="integrate(2*x/sqrt(x^2+1), x)",
        expected_result="2·sqrt(x^2 + 1)",
        expected_derivative_result="2·x / sqrt(x^2 + 1)",
        expected_direct_diff_integrate_result="2·x / sqrt(x^2 + 1)",
        expected_step_substrings=(
            "Usar la regla de u'/sqrt(u) -> 2*sqrt(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="polynomial_base",
        argument_regime="nonlinear_polynomial_base",
        trace_regime="polynomial_base_substitution",
        presentation_regime="radical_power",
    ),
    IntegrateCommandMatrixCase(
        name="hyperbolic_reciprocal_square_substitution",
        expr="integrate(1/cosh(2*x + 1)^2, x)",
        expected_result="1/2·tanh(2·x + 1)",
        expected_derivative_result="1 / cosh(2·x + 1)^2",
        expected_direct_diff_integrate_result="1 / cosh(2·x + 1)^2",
        expected_step_substrings=(
            "Usar la regla de 1/cosh(u)^2 -> tanh(u)",
            "Identificar u y du",
            "Ajustar el factor constante",
        ),
        family="hyperbolic_reciprocal_square",
        argument_regime="affine_hyperbolic",
        trace_regime="hyperbolic_substitution",
        presentation_regime="scaled_hyperbolic",
    ),
    IntegrateCommandMatrixCase(
        name="hyperbolic_sine_reciprocal_square_substitution",
        expr="integrate(1/sinh(2*x + 1)^2, x)",
        expected_result="-1 / (2·tanh(2·x + 1))",
        expected_derivative_result="1 / sinh(2·x + 1)^2",
        expected_derivative_required_display=("sinh(2·x + 1) ≠ 0",),
        expected_required_display=("sinh(2·x + 1) ≠ 0",),
        expected_direct_diff_integrate_result="1 / sinh(2·x + 1)^2",
        expected_direct_diff_integrate_required_display=("sinh(2·x + 1) ≠ 0",),
        expected_step_substrings=(
            "Usar la regla de 1/sinh(u)^2 -> -1/tanh(u)",
            "Identificar u y du",
            "Ajustar el factor constante",
        ),
        family="hyperbolic_reciprocal_square",
        argument_regime="affine_hyperbolic",
        domain_regime="hyperbolic_sine_pole_required",
        trace_regime="hyperbolic_sine_reciprocal_square_substitution",
        presentation_regime="reciprocal_hyperbolic_tangent",
    ),
    IntegrateCommandMatrixCase(
        name="negative_affine_hyperbolic_sine_reciprocal_square_substitution",
        expr="integrate(1/sinh(-2*x + 1)^2, x)",
        expected_result="1 / (2·tanh(1 - 2·x))",
        expected_derivative_result="1 / sinh(1 - 2·x)^2",
        expected_derivative_required_display=("sinh(2·x - 1) ≠ 0",),
        expected_required_display=("sinh(2·x - 1) ≠ 0",),
        expected_direct_diff_integrate_result="1 / sinh(1 - 2·x)^2",
        expected_direct_diff_integrate_required_display=("sinh(2·x - 1) ≠ 0",),
        expected_step_substrings=(
            "Usar la regla de 1/sinh(u)^2 -> -1/tanh(u)",
            "Identificar u y du",
            "Ajustar el factor constante",
        ),
        family="hyperbolic_reciprocal_square",
        argument_regime="negative_affine_hyperbolic",
        domain_regime="hyperbolic_sine_pole_required",
        trace_regime="negative_affine_hyperbolic_sine_reciprocal_square_substitution",
        presentation_regime="negative_affine_reciprocal_hyperbolic_tangent",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_affine_exact_hyperbolic_cosh_reciprocal_square_substitution",
        expr="integrate(a/cosh(a*x+b)^2, x)",
        expected_result="tanh(a·x + b)",
        expected_derivative_result="a / cosh(a·x + b)^2",
        expected_direct_diff_integrate_result="a / cosh(a·x + b)^2",
        expected_step_substrings=(
            "Usar la regla de 1/cosh(u)^2 -> tanh(u)",
            "Identificar u y du",
            "du = a",
        ),
        family="hyperbolic_reciprocal_square",
        argument_regime="symbolic_affine_exact_cofactor_hyperbolic",
        trace_regime="symbolic_exact_hyperbolic_substitution",
        presentation_regime="unscaled_hyperbolic_tangent",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_affine_exact_hyperbolic_sinh_reciprocal_square_substitution",
        expr="integrate(a/sinh(a*x+b)^2, x)",
        expected_result="-1 / tanh(a·x + b)",
        expected_derivative_result="a / sinh(a·x + b)^2",
        expected_derivative_required_display=("sinh(a·x + b) ≠ 0",),
        expected_required_display=("sinh(a·x + b) ≠ 0",),
        expected_direct_diff_integrate_result="a / sinh(a·x + b)^2",
        expected_direct_diff_integrate_required_display=("sinh(a·x + b) ≠ 0",),
        expected_step_substrings=(
            "Usar la regla de 1/sinh(u)^2 -> -1/tanh(u)",
            "Identificar u y du",
            "du = a",
        ),
        family="hyperbolic_reciprocal_square",
        argument_regime="symbolic_affine_exact_cofactor_hyperbolic",
        domain_regime="hyperbolic_sine_pole_required",
        trace_regime="symbolic_exact_hyperbolic_sine_reciprocal_square_substitution",
        presentation_regime="reciprocal_hyperbolic_tangent",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_shifted_hyperbolic_cosh_reciprocal_square_substitution",
        expr="integrate(2*k*x/cosh(x^2+b)^2, x)",
        expected_result="k·tanh(x^2 + b)",
        expected_derivative_equivalent_to="2*k*x/cosh(x^2+b)^2",
        expected_direct_diff_integrate_equivalent_to="2*k*x/cosh(x^2+b)^2",
        expected_step_substrings=(
            "Usar la regla de 1/cosh(u)^2 -> tanh(u)",
            "Identificar u y du",
            "du = 2\\cdot x",
            "Ajustar el factor constante",
        ),
        family="hyperbolic_reciprocal_square",
        argument_regime="symbolic_external_scale_shifted_polynomial_hyperbolic_square",
        trace_regime="symbolic_external_scale_polynomial_hyperbolic_reciprocal_square",
        presentation_regime="external_scale_hyperbolic_tangent",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_shifted_hyperbolic_sinh_reciprocal_square_substitution",
        expr="integrate(2*k*x/sinh(x^2+b)^2, x)",
        expected_result="-k·cosh(x^2 + b) / sinh(x^2 + b)",
        expected_derivative_result="(x·k·2)/sinh(x^2 + b)^2",
        expected_derivative_required_display=("sinh(x^2 + b) ≠ 0",),
        expected_required_display=("sinh(x^2 + b) ≠ 0",),
        expected_direct_diff_integrate_equivalent_to="2*k*x/sinh(x^2+b)^2",
        expected_direct_diff_integrate_required_display=("sinh(x^2 + b) ≠ 0",),
        expected_step_substrings=(
            "Usar la regla de 1/sinh(u)^2 -> -1/tanh(u)",
            "Identificar u y du",
            "du = 2\\cdot x",
            "Ajustar el factor constante",
        ),
        family="hyperbolic_reciprocal_square",
        argument_regime="symbolic_external_scale_shifted_polynomial_hyperbolic_square",
        domain_regime="hyperbolic_sine_pole_required",
        trace_regime="symbolic_external_scale_polynomial_hyperbolic_sine_reciprocal_square",
        presentation_regime="external_scale_reciprocal_hyperbolic_tangent",
    ),
    IntegrateCommandMatrixCase(
        name="hyperbolic_cosh_reciprocal_fourth_substitution",
        expr="integrate(1/cosh(2*x+1)^4, x)",
        expected_result="1/6·(3·tanh(2·x + 1) - tanh(2·x + 1)^3)",
        expected_derivative_result="1 / cosh(2·x + 1)^4",
        expected_direct_diff_integrate_result="1 / cosh(2·x + 1)^4",
        expected_step_substrings=(
            "Usar la regla de 1/cosh(u)^4 -> tanh(u) - tanh(u)^3/3",
            "Identificar u y du",
            "du = 2",
            "Ajustar el factor constante",
        ),
        family="hyperbolic_reciprocal_fourth",
        argument_regime="affine_hyperbolic_fourth",
        trace_regime="hyperbolic_reciprocal_fourth_substitution",
        presentation_regime="scaled_hyperbolic_tangent_minus_cubic",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_shifted_hyperbolic_cosh_reciprocal_fourth_substitution",
        expr="integrate(2*k*x/cosh(x^2+b)^4, x)",
        expected_result="1/3·(3·k·tanh(x^2 + b) - k·tanh(x^2 + b)^3)",
        expected_derivative_result="(x·k·6)/(3·cosh(x^2 + b)^4)",
        expected_direct_diff_integrate_result="2·k·x / cosh(x^2 + b)^4",
        expected_step_substrings=(
            "Usar la regla de 1/cosh(u)^4 -> tanh(u) - tanh(u)^3/3",
            "Identificar u y du",
            "du = 2\\cdot x",
            "Ajustar el factor constante",
            "k\\cdot 2\\cdot x",
        ),
        family="hyperbolic_reciprocal_fourth",
        argument_regime="symbolic_external_scale_shifted_polynomial_hyperbolic_fourth",
        trace_regime="symbolic_external_scale_polynomial_hyperbolic_reciprocal_fourth",
        presentation_regime="external_scale_hyperbolic_tangent_minus_cubic",
    ),
    IntegrateCommandMatrixCase(
        name="hyperbolic_sinh_reciprocal_fourth_substitution",
        expr="integrate(1/sinh(2*x+1)^4, x)",
        expected_result="1/2 / tanh(2·x + 1) - 1/6 / tanh(2·x + 1)^3",
        expected_derivative_result="1 / (cosh(2·x + 1)^2·tanh(2·x + 1)^4) - 1 / sinh(2·x + 1)^2",
        expected_derivative_required_display=("sinh(2·x + 1) ≠ 0",),
        expected_direct_diff_integrate_result="1 / sinh(2·x + 1)^4",
        expected_direct_diff_integrate_required_display=("sinh(2·x + 1) ≠ 0",),
        expected_required_display=("sinh(2·x + 1) ≠ 0",),
        expected_step_substrings=(
            "Usar la regla de 1/sinh(u)^4 -> 1/tanh(u) - 1/(3*tanh(u)^3)",
            "Identificar u y du",
            "du = 2",
            "Ajustar el factor constante",
        ),
        family="hyperbolic_reciprocal_fourth",
        argument_regime="affine_hyperbolic_fourth",
        domain_regime="hyperbolic_sine_pole_required",
        trace_regime="hyperbolic_sine_reciprocal_fourth_substitution",
        presentation_regime="scaled_reciprocal_hyperbolic_tangent_minus_cubic",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_shifted_hyperbolic_sinh_reciprocal_fourth_substitution",
        expr="integrate(2*k*x/sinh(x^2+b)^4, x)",
        expected_result="k / tanh(x^2 + b) - k / (3·tanh(x^2 + b)^3)",
        expected_derivative_result=(
            "k·2·x / (cosh(x^2 + b)^2·tanh(x^2 + b)^4) "
            "- k·2·x / (cosh(x^2 + b)^2·tanh(x^2 + b)^2)"
        ),
        expected_derivative_required_display=("sinh(x^2 + b) ≠ 0",),
        expected_direct_diff_integrate_result="2·k·x / sinh(x^2 + b)^4",
        expected_direct_diff_integrate_required_display=("sinh(x^2 + b) ≠ 0",),
        expected_required_display=("sinh(x^2 + b) ≠ 0",),
        expected_step_substrings=(
            "Usar la regla de 1/sinh(u)^4 -> 1/tanh(u) - 1/(3*tanh(u)^3)",
            "Identificar u y du",
            "du = 2\\cdot x",
            "Ajustar el factor constante",
            "k\\cdot 2\\cdot x",
        ),
        family="hyperbolic_reciprocal_fourth",
        argument_regime="symbolic_external_scale_shifted_polynomial_hyperbolic_fourth",
        domain_regime="hyperbolic_sine_pole_required",
        trace_regime="symbolic_external_scale_polynomial_hyperbolic_sine_reciprocal_fourth",
        presentation_regime="external_scale_reciprocal_hyperbolic_tangent_minus_cubic",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_affine_exact_hyperbolic_sinh_over_cosh_square_substitution",
        expr="integrate(a*sinh(a*x+b)/cosh(a*x+b)^2, x)",
        expected_result="-1 / cosh(a·x + b)",
        expected_derivative_result="a·sinh(a·x + b) / cosh(a·x + b)^2",
        expected_direct_diff_integrate_result="a·sinh(a·x + b) / cosh(a·x + b)^2",
        expected_step_substrings=(
            "Usar la regla de sinh(u)/cosh(u)^2 -> -1/cosh(u)",
            "Identificar u y du",
            "du = a",
        ),
        family="hyperbolic_reciprocal_derivative_product",
        argument_regime="symbolic_affine_exact_cofactor_hyperbolic_product",
        trace_regime="symbolic_exact_hyperbolic_reciprocal_derivative_product",
        presentation_regime="compact_hyperbolic_reciprocal",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_affine_exact_hyperbolic_cosh_over_sinh_square_substitution",
        expr="integrate(a*cosh(a*x+b)/sinh(a*x+b)^2, x)",
        expected_result="-1 / sinh(a·x + b)",
        expected_derivative_result="a·cosh(a·x + b) / sinh(a·x + b)^2",
        expected_derivative_required_display=("sinh(a·x + b) ≠ 0",),
        expected_required_display=("sinh(a·x + b) ≠ 0",),
        expected_direct_diff_integrate_result="a·cosh(a·x + b) / sinh(a·x + b)^2",
        expected_direct_diff_integrate_required_display=("sinh(a·x + b) ≠ 0",),
        expected_step_substrings=(
            "Usar la regla de cosh(u)/sinh(u)^2 -> -1/sinh(u)",
            "Identificar u y du",
            "du = a",
        ),
        family="hyperbolic_reciprocal_derivative_product",
        argument_regime="symbolic_affine_exact_cofactor_hyperbolic_product",
        domain_regime="hyperbolic_sine_pole_required",
        trace_regime="symbolic_exact_hyperbolic_reciprocal_derivative_product",
        presentation_regime="compact_hyperbolic_reciprocal",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_hyperbolic_sinh_over_cosh_square_substitution",
        expr="integrate(k*a*sinh(a*x+b)/cosh(a*x+b)^2, x)",
        expected_result="-k / cosh(a·x + b)",
        expected_derivative_result="a·k·sinh(a·x + b) / cosh(a·x + b)^2",
        expected_direct_diff_integrate_result="a·k·sinh(a·x + b) / cosh(a·x + b)^2",
        expected_step_substrings=(
            "Usar la regla de sinh(u)/cosh(u)^2 -> -1/cosh(u)",
            "Identificar u y du",
            "du = a",
        ),
        family="hyperbolic_reciprocal_derivative_product",
        argument_regime="symbolic_external_scale_hyperbolic_product",
        trace_regime="symbolic_external_scale_hyperbolic_reciprocal_derivative_product",
        presentation_regime="external_scale_compact_hyperbolic_reciprocal",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_hyperbolic_cosh_over_sinh_square_substitution",
        expr="integrate(k*a*cosh(a*x+b)/sinh(a*x+b)^2, x)",
        expected_result="-k / sinh(a·x + b)",
        expected_derivative_result="a·k·cosh(a·x + b) / sinh(a·x + b)^2",
        expected_derivative_required_display=("sinh(a·x + b) ≠ 0",),
        expected_required_display=("sinh(a·x + b) ≠ 0",),
        expected_direct_diff_integrate_result="a·k·cosh(a·x + b) / sinh(a·x + b)^2",
        expected_direct_diff_integrate_required_display=("sinh(a·x + b) ≠ 0",),
        expected_step_substrings=(
            "Usar la regla de cosh(u)/sinh(u)^2 -> -1/sinh(u)",
            "Identificar u y du",
            "du = a",
        ),
        family="hyperbolic_reciprocal_derivative_product",
        argument_regime="symbolic_external_scale_hyperbolic_product",
        domain_regime="hyperbolic_sine_pole_required",
        trace_regime="symbolic_external_scale_hyperbolic_reciprocal_derivative_product",
        presentation_regime="external_scale_compact_hyperbolic_reciprocal",
    ),
    IntegrateCommandMatrixCase(
        name="polynomial_shifted_hyperbolic_sinh_over_cosh_square_substitution",
        expr="integrate(2*x*sinh(x^2+b)/cosh(x^2+b)^2, x)",
        expected_result="-1 / cosh(x^2 + b)",
        expected_derivative_equivalent_to="2*x*sinh(x^2+b)/cosh(x^2+b)^2",
        expected_direct_diff_integrate_equivalent_to="2*x*sinh(x^2+b)/cosh(x^2+b)^2",
        expected_step_substrings=(
            "Usar la regla de sinh(u)/cosh(u)^2 -> -1/cosh(u)",
            "Identificar u y du",
            "du = 2\\cdot x",
        ),
        family="hyperbolic_reciprocal_derivative_product",
        argument_regime="shifted_polynomial_hyperbolic_product",
        trace_regime="polynomial_hyperbolic_reciprocal_derivative_product",
        presentation_regime="compact_hyperbolic_reciprocal",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_shifted_hyperbolic_sinh_over_cosh_square_substitution",
        expr="integrate(2*k*x*sinh(x^2+b)/cosh(x^2+b)^2, x)",
        expected_result="-k / cosh(x^2 + b)",
        expected_derivative_equivalent_to="2*k*x*sinh(x^2+b)/cosh(x^2+b)^2",
        expected_direct_diff_integrate_equivalent_to="2*k*x*sinh(x^2+b)/cosh(x^2+b)^2",
        expected_step_substrings=(
            "Usar la regla de sinh(u)/cosh(u)^2 -> -1/cosh(u)",
            "Identificar u y du",
            "du = 2\\cdot x",
            "Ajustar el factor constante",
        ),
        family="hyperbolic_reciprocal_derivative_product",
        argument_regime="symbolic_external_scale_shifted_polynomial_hyperbolic_product",
        trace_regime="symbolic_external_scale_polynomial_hyperbolic_reciprocal_derivative_product",
        presentation_regime="external_scale_compact_hyperbolic_reciprocal",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_shifted_hyperbolic_cosh_over_sinh_square_substitution",
        expr="integrate(2*k*x*cosh(x^2+b)/sinh(x^2+b)^2, x)",
        expected_result="-k / sinh(x^2 + b)",
        expected_derivative_equivalent_to="2*k*x*cosh(x^2+b)/sinh(x^2+b)^2",
        expected_derivative_required_display=("sinh(x^2 + b) ≠ 0",),
        expected_required_display=("sinh(x^2 + b) ≠ 0",),
        expected_direct_diff_integrate_equivalent_to="2*k*x*cosh(x^2+b)/sinh(x^2+b)^2",
        expected_direct_diff_integrate_required_display=("sinh(x^2 + b) ≠ 0",),
        expected_step_substrings=(
            "Usar la regla de cosh(u)/sinh(u)^2 -> -1/sinh(u)",
            "Identificar u y du",
            "du = 2\\cdot x",
            "Ajustar el factor constante",
        ),
        family="hyperbolic_reciprocal_derivative_product",
        argument_regime="symbolic_external_scale_shifted_polynomial_hyperbolic_product",
        domain_regime="hyperbolic_sine_pole_required",
        trace_regime="symbolic_external_scale_polynomial_hyperbolic_reciprocal_derivative_product",
        presentation_regime="external_scale_compact_hyperbolic_reciprocal",
    ),
    IntegrateCommandMatrixCase(
        name="by_parts_log_domain",
        expr="integrate(x*ln(x), x)",
        expected_result="1/4·x^2·(2·ln(x) - 1)",
        expected_derivative_result="x·ln(x)",
        expected_direct_diff_integrate_result="x·ln(x)",
        expected_direct_diff_integrate_required_display=("x > 0",),
        expected_derivative_required_display=("x > 0",),
        expected_required_display=("x > 0",),
        expected_step_substrings=(
            "Usar integración por partes",
            "Elegir u y dv",
            "Calcular du y v",
            "Aplicar la fórmula de integración por partes",
        ),
        family="by_parts_log",
        argument_regime="product",
        domain_regime="positive_required",
        trace_regime="by_parts_log",
        presentation_regime="factored_by_parts",
    ),
    IntegrateCommandMatrixCase(
        name="radical_numerator_unit_circle",
        expr="integrate(sqrt(1-x^2), x)",
        expected_result="1/2·(arcsin(x) + x·(1 - x^2)^(1/2))",
        expected_derivative_equivalent_to="sqrt(1-x^2)",
        expected_direct_diff_integrate_equivalent_to="sqrt(1-x^2)",
        expected_direct_diff_integrate_required_display=("-1 < x < 1",),
        expected_derivative_required_display=("-1 < x < 1",),
        expected_required_display=("-1 ≤ x ≤ 1",),
        expected_step_substrings=("Calcular la integral",),
        family="radical_numerator_polynomial",
        argument_regime="unit_circle_radical_numerator",
        domain_regime="bounded_interval_required",
        trace_regime="radical_numerator_quotient_delegation",
        presentation_regime="arcsin_plus_radical_half",
    ),
    IntegrateCommandMatrixCase(
        name="radical_numerator_square_cofactor",
        expr="integrate(x^2*sqrt(1-x^2), x)",
        expected_result="1/8·(arcsin(x) + (1 - x^2)^(1/2)·(2·x^3 - x))",
        expected_derivative_equivalent_to="x^2*sqrt(1-x^2)",
        expected_derivative_required_display=("-1 < x < 1",),
        expected_required_display=("-1 ≤ x ≤ 1",),
        expected_step_substrings=("Calcular la integral",),
        family="radical_numerator_polynomial",
        argument_regime="square_times_unit_circle_radical",
        domain_regime="bounded_interval_required",
        trace_regime="radical_numerator_quotient_delegation",
        presentation_regime="arcsin_plus_radical_eighth",
    ),
    IntegrateCommandMatrixCase(
        name="radical_numerator_hyperbolic_asinh",
        expr="integrate(sqrt(x^2+1), x)",
        expected_result="1/2·(asinh(x) + x·(x^2 + 1)^(1/2))",
        expected_derivative_equivalent_to="sqrt(x^2+1)",
        expected_direct_diff_integrate_equivalent_to="sqrt(x^2+1)",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="radical_numerator_polynomial",
        argument_regime="hyperbolic_radical_numerator",
        domain_regime="total_real_function",
        trace_regime="radical_numerator_quotient_delegation",
        presentation_regime="asinh_plus_radical_half",
    ),
    IntegrateCommandMatrixCase(
        name="exponential_rational_logistic_reciprocal",
        expr="integrate(1/(1+e^x), x)",
        expected_result="ln(e^x / (e^x + 1))",
        expected_derivative_equivalent_to="1/(1+e^x)",
        expected_direct_diff_integrate_equivalent_to="1/(1+e^x)",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="exponential_rational_substitution",
        argument_regime="logistic_reciprocal",
        domain_regime="total_real_function",
        trace_regime="exponential_rational_substitution_delegation",
        presentation_regime="log_quotient_exponential",
    ),
    IntegrateCommandMatrixCase(
        name="exponential_rational_arctan_gudermannian",
        expr="integrate(e^x/(1+e^(2*x)), x)",
        expected_result="arctan(e^x)",
        expected_derivative_equivalent_to="e^x/(1+e^(2*x))",
        expected_direct_diff_integrate_equivalent_to="e^x/(1+e^(2*x))",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="exponential_rational_substitution",
        argument_regime="arctan_exponential_square",
        domain_regime="total_real_function",
        trace_regime="exponential_rational_substitution_delegation",
        presentation_regime="compact_arctan_exponential",
    ),
    IntegrateCommandMatrixCase(
        name="exponential_rational_improper_quotient",
        expr="integrate(e^(2*x)/(1+e^x), x)",
        expected_result="e^x - ln(e^x + 1)",
        expected_derivative_equivalent_to="e^(2*x)/(1+e^x)",
        expected_direct_diff_integrate_equivalent_to="e^(2*x)/(1+e^x)",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="exponential_rational_substitution",
        argument_regime="improper_exponential_quotient",
        domain_regime="total_real_function",
        trace_regime="exponential_rational_substitution_delegation",
        presentation_regime="exponential_minus_log",
    ),
    IntegrateCommandMatrixCase(
        name="exponential_rational_tanh_half_surface",
        expr="integrate((e^x-1)/(e^x+1), x)",
        expected_result="2·ln(e^x + 1) - x",
        expected_derivative_equivalent_to="(e^x-1)/(e^x+1)",
        expected_direct_diff_integrate_equivalent_to="(e^x-1)/(e^x+1)",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="exponential_rational_substitution",
        argument_regime="balanced_exponential_quotient",
        domain_regime="total_real_function",
        trace_regime="exponential_rational_substitution_delegation",
        presentation_regime="log_minus_linear",
    ),
    IntegrateCommandMatrixCase(
        name="exponential_rational_double_slope_pole",
        expr="integrate(1/(e^(2*x)-1), x)",
        expected_result="1/2·ln(|(e^(2·x) - 1) / e^(2·x)|)",
        expected_derivative_equivalent_to="1/(e^(2*x)-1)",
        expected_direct_diff_integrate_equivalent_to="1/(e^(2*x)-1)",
        expected_direct_diff_integrate_required_display=("e^(2·x) - 1 ≠ 0",),
        expected_derivative_required_display=("e^(2·x) - 1 ≠ 0",),
        expected_required_display=("e^(2·x) - 1 ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="exponential_rational_substitution",
        argument_regime="double_slope_pole_quotient",
        domain_regime="nonzero_denominator_required",
        trace_regime="exponential_rational_substitution_delegation",
        presentation_regime="half_log_abs_quotient",
    ),
    IntegrateCommandMatrixCase(
        name="exponential_rational_half_slope_arctan",
        expr="integrate(e^(x/2)/(1+e^x), x)",
        expected_result="2·arctan(e^(x / 2))",
        expected_derivative_equivalent_to="e^(x/2)/(1+e^x)",
        expected_direct_diff_integrate_equivalent_to="e^(x/2)/(1+e^x)",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="exponential_rational_substitution",
        argument_regime="fractional_slope_gcd",
        domain_regime="total_real_function",
        trace_regime="exponential_rational_substitution_delegation",
        presentation_regime="scaled_arctan_half_exponential",
    ),
    IntegrateCommandMatrixCase(
        name="linear_radical_monomial_cofactor",
        expr="integrate(x*sqrt(x+1), x)",
        expected_result="sqrt(x + 1)·(2/5·x^2 + 2/15·x - 4/15)",
        expected_derivative_equivalent_to="x*sqrt(x+1)",
        expected_direct_diff_integrate_equivalent_to="x*sqrt(x+1)",
        expected_direct_diff_integrate_required_display=("x ≥ -1",),
        expected_derivative_required_display=("x > -1",),
        expected_required_display=("x ≥ -1",),
        expected_step_substrings=("Calcular la integral",),
        family="linear_radical_substitution",
        argument_regime="monomial_radical_cofactor",
        domain_regime="radicand_halfline_required",
        trace_regime="linear_radical_substitution_delegation",
        presentation_regime="radical_times_polynomial",
    ),
    IntegrateCommandMatrixCase(
        name="linear_radical_square_cofactor",
        expr="integrate(x^2*sqrt(x+1), x)",
        expected_result="sqrt(x + 1)·(2/7·x^3 + 2/35·x^2 + 16/105 - 8/105·x)",
        expected_derivative_equivalent_to="x^2*sqrt(x+1)",
        expected_direct_diff_integrate_equivalent_to="x^2*sqrt(x+1)",
        expected_direct_diff_integrate_required_display=("x ≥ -1",),
        expected_derivative_required_display=("x > -1",),
        expected_required_display=("x ≥ -1",),
        expected_step_substrings=("Calcular la integral",),
        family="linear_radical_substitution",
        argument_regime="square_radical_cofactor",
        domain_regime="radicand_halfline_required",
        trace_regime="linear_radical_substitution_delegation",
        presentation_regime="radical_times_polynomial",
    ),
    IntegrateCommandMatrixCase(
        name="linear_radical_scaled_radicand",
        expr="integrate(x*sqrt(2*x-1), x)",
        expected_result="1/30·(3·(2·x - 1)^(5/2) + 5·(2·x - 1)^(3/2))",
        expected_derivative_equivalent_to="x*sqrt(2*x-1)",
        expected_direct_diff_integrate_equivalent_to="x*sqrt(2*x-1)",
        expected_direct_diff_integrate_required_display=("x ≥ 1/2",),
        expected_derivative_required_display=("x ≥ 1/2",),
        expected_required_display=("x ≥ 1/2",),
        expected_step_substrings=("Calcular la integral",),
        family="linear_radical_substitution",
        argument_regime="scaled_shifted_radicand",
        domain_regime="radicand_halfline_required",
        trace_regime="linear_radical_substitution_delegation",
        presentation_regime="half_power_sum",
    ),
    IntegrateCommandMatrixCase(
        name="linear_radical_rationalized_reciprocal",
        expr="integrate(1/(sqrt(x)+1), x)",
        expected_result="2·x^(1/2) - 2·ln(x^(1/2) + 1)",
        expected_required_display=("x ≥ 0",),
        expected_step_substrings=(
            "Racionalizar el denominador",
            "Calcular la integral",
        ),
        family="linear_radical_substitution",
        argument_regime="rationalized_reciprocal_surface",
        domain_regime="radicand_halfline_required",
        trace_regime="linear_radical_substitution_delegation",
        presentation_regime="radical_minus_log",
    ),
    IntegrateCommandMatrixCase(
        name="linear_radical_arctan_quotient",
        expr="integrate(sqrt(x)/(1+x), x)",
        expected_result="2·x^(1/2) - 2·arctan(sqrt(x))",
        expected_required_display=("x ≥ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="linear_radical_substitution",
        argument_regime="radical_over_shifted_square",
        domain_regime="radicand_halfline_required",
        trace_regime="linear_radical_substitution_delegation",
        presentation_regime="radical_minus_arctan",
    ),
    IntegrateCommandMatrixCase(
        name="linear_radical_three_half_power",
        expr="integrate(x*(x+1)^(3/2), x)",
        expected_result="sqrt(x + 1)·(2/7·x^3 + 16/35·x^2 + 2/35·x - 4/35)",
        expected_derivative_equivalent_to="x*(x+1)^(3/2)",
        expected_direct_diff_integrate_equivalent_to="x*(x+1)^(3/2)",
        expected_direct_diff_integrate_required_display=("x > -1",),
        expected_derivative_required_display=("x > -1",),
        expected_required_display=("x ≥ -1",),
        expected_step_substrings=("Calcular la integral",),
        family="linear_radical_substitution",
        argument_regime="three_half_power_cofactor",
        domain_regime="radicand_halfline_required",
        trace_regime="linear_radical_substitution_delegation",
        presentation_regime="radical_times_polynomial",
    ),
    IntegrateCommandMatrixCase(
        name="weierstrass_shifted_cosine_arctan",
        expr="integrate(1/(2+cos(x)), x)",
        expected_result="2·arctan(tan(x / 2) / sqrt(3)) / sqrt(3)",
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="weierstrass_rational_substitution",
        argument_regime="shifted_cosine_reciprocal",
        domain_regime="total_real_function",
        trace_regime="weierstrass_substitution_delegation",
        presentation_regime="scaled_arctan_half_tangent",
    ),
    IntegrateCommandMatrixCase(
        name="weierstrass_shifted_sine_reciprocal",
        expr="integrate(1/(1+sin(x)), x)",
        expected_result="-2·cos(x / 2)/(sin(x / 2) + cos(x / 2))",
        expected_required_display=(
            "sin(x / 2) + cos(x / 2) ≠ 0",
            "sin(x) + 1 ≠ 0",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="weierstrass_rational_substitution",
        argument_regime="shifted_sine_reciprocal",
        domain_regime="nonzero_denominator_required",
        trace_regime="weierstrass_substitution_delegation",
        presentation_regime="compact_half_angle_quotient",
    ),
    IntegrateCommandMatrixCase(
        name="weierstrass_half_angle_tangent",
        expr="integrate(1/(1+cos(x)), x)",
        expected_result="tan(x / 2)",
        expected_derivative_equivalent_to="1/(1+cos(x))",
        expected_direct_diff_integrate_equivalent_to="1/(1+cos(x))",
        expected_direct_diff_integrate_required_display=("cos(x) + 1 ≠ 0",),
        expected_derivative_required_display=("cos(x) + 1 ≠ 0",),
        expected_required_display=("cos(x) + 1 ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="weierstrass_rational_substitution",
        argument_regime="shifted_cosine_pole_touch",
        domain_regime="nonzero_denominator_required",
        trace_regime="weierstrass_substitution_delegation",
        presentation_regime="bare_half_tangent",
    ),
    IntegrateCommandMatrixCase(
        name="weierstrass_textbook_sine_arctan",
        expr="integrate(1/(5+4*sin(x)), x)",
        expected_result="2/3·arctan((sin(x / 2)·5)/(3·cos(x / 2)) + 4/3)",
        expected_required_display=("cos(x / 2) ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="weierstrass_rational_substitution",
        argument_regime="textbook_sine_coefficients",
        domain_regime="nonzero_denominator_required",
        trace_regime="weierstrass_substitution_delegation",
        presentation_regime="arctan_offset_half_tangent",
    ),
    IntegrateCommandMatrixCase(
        name="weierstrass_sine_over_shifted_sine",
        expr="integrate(sin(x)/(1+sin(x)), x)",
        expected_result="2·arctan(tan(x / 2)) + 2 / (tan(x / 2) + 1)",
        expected_required_display=(
            "sin(x) + 1 ≠ 0",
            "tan(x / 2) + 1 ≠ 0",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="weierstrass_rational_substitution",
        argument_regime="proper_quotient_split",
        domain_regime="nonzero_denominator_required",
        trace_regime="weierstrass_substitution_delegation",
        presentation_regime="arctan_plus_reciprocal",
    ),
    IntegrateCommandMatrixCase(
        name="weierstrass_double_angle_slope",
        expr="integrate(1/(2+cos(2*x)), x)",
        expected_result="arctan(tan(x) / sqrt(3)) / sqrt(3)",
        expected_derivative_equivalent_to="1/(2+cos(2*x))",
        expected_direct_diff_integrate_equivalent_to="1/(2+cos(2*x))",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="weierstrass_rational_substitution",
        argument_regime="double_angle_slope",
        domain_regime="total_real_function",
        trace_regime="weierstrass_substitution_delegation",
        presentation_regime="scaled_arctan_full_tangent",
    ),
    IntegrateCommandMatrixCase(
        name="arcsec_reciprocal_hyperbola_radical",
        expr="integrate(1/(x*sqrt(x^2-1)), x)",
        expected_result="arctan(sqrt(x^2 - 1))",
        expected_derivative_equivalent_to="1/(x*sqrt(x^2-1))",
        expected_direct_diff_integrate_equivalent_to="1/(x*sqrt(x^2-1))",
        expected_direct_diff_integrate_required_display=("x < -1 or x > 1",),
        expected_derivative_required_display=("x < -1 or x > 1",),
        expected_required_display=("x < -1 or x > 1",),
        expected_step_substrings=("Calcular la integral",),
        family="quadratic_radical_over_monomial",
        argument_regime="arcsec_kernel",
        domain_regime="exterior_interval_required",
        trace_regime="quadratic_radical_monomial_delegation",
        presentation_regime="arctan_of_radical",
    ),
    IntegrateCommandMatrixCase(
        name="arcsec_inverse_square_cofactor",
        expr="integrate(1/(x^2*sqrt(x^2+4)), x)",
        expected_result="-(x^2 + 4)^(1/2) / (4·x)",
        expected_derivative_equivalent_to="1/(x^2*sqrt(x^2+4))",
        expected_direct_diff_integrate_equivalent_to="1/(x^2*sqrt(x^2+4))",
        expected_direct_diff_integrate_required_display=("x ≠ 0",),
        expected_derivative_required_display=("x ≠ 0",),
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="quadratic_radical_over_monomial",
        argument_regime="inverse_square_radical_cofactor",
        domain_regime="punctured_real_line",
        trace_regime="quadratic_radical_monomial_delegation",
        presentation_regime="closed_form_radical_quotient",
    ),
    IntegrateCommandMatrixCase(
        name="arcsec_circle_radical_over_x",
        expr="integrate(sqrt(4-x^2)/x, x)",
        expected_result="ln(-|(4 - x^2)^(1/2) - 2|·((4 - x^2)^(1/2) - 2) / x^2) + (4 - x^2)^(1/2)",
        expected_required_display=(
            "-2 ≤ x ≤ 2",
            "-|(4 - x^2)^(1/2) - 2|·((4 - x^2)^(1/2) - 2) / x^2 > 0",
            "x ≠ 0",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="quadratic_radical_over_monomial",
        argument_regime="circle_radical_over_monomial",
        domain_regime="punctured_bounded_interval",
        trace_regime="quadratic_radical_monomial_delegation",
        presentation_regime="log_radical_shifted_quotient",
    ),
    IntegrateCommandMatrixCase(
        name="arcsec_circle_radical_over_square",
        expr="integrate(sqrt(1-x^2)/x^2, x)",
        expected_result="(-(1 - x^2)^(1/2) - x·arcsin(x)) / x",
        expected_derivative_equivalent_to="sqrt(1-x^2)/x^2",
        expected_direct_diff_integrate_equivalent_to="sqrt(1-x^2)/x^2",
        expected_direct_diff_integrate_required_display=(
            "-1 ≤ x ≤ 1",
            "x ≠ 0",
        ),
        expected_derivative_required_display=(
            "-1 ≤ x ≤ 1",
            "x ≠ 0",
        ),
        expected_required_display=(
            "-1 ≤ x ≤ 1",
            "x ≠ 0",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="quadratic_radical_over_monomial",
        argument_regime="circle_radical_over_square",
        domain_regime="punctured_bounded_interval",
        trace_regime="quadratic_radical_monomial_delegation",
        presentation_regime="radical_plus_arcsin_quotient",
    ),
    IntegrateCommandMatrixCase(
        name="arcsec_log_kernel_positive_shift",
        expr="integrate(1/(x*sqrt(x^2+4)), x)",
        expected_result="1/4·ln(|((x^2 + 4)^(1/2) - 2)^2 / x^2|)",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="quadratic_radical_over_monomial",
        argument_regime="log_kernel_positive_shift",
        domain_regime="punctured_real_line",
        trace_regime="quadratic_radical_monomial_delegation",
        presentation_regime="quarter_log_radical_square",
    ),
    IntegrateCommandMatrixCase(
        name="arcsec_hyperbola_radical_over_x",
        expr="integrate(sqrt(x^2-1)/x, x)",
        expected_result="(x^2 - 1)^(1/2) - arctan(sqrt(x^2 - 1))",
        expected_required_display=(
            "x ≠ 0",
            "x ≤ -1 or x ≥ 1",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="quadratic_radical_over_monomial",
        argument_regime="hyperbola_radical_over_monomial",
        domain_regime="punctured_exterior_interval",
        trace_regime="quadratic_radical_monomial_delegation",
        presentation_regime="radical_minus_arctan_radical",
    ),
    IntegrateCommandMatrixCase(
        name="mixed_trig_odd_cosine_cube",
        expr="integrate(sin(x)^2*cos(x)^3, x)",
        expected_result="1/15·(5·sin(x)^3 - 3·sin(x)^5)",
        expected_derivative_equivalent_to="sin(x)^2*cos(x)^3",
        expected_direct_diff_integrate_equivalent_to="sin(x)^2*cos(x)^3",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="mixed_trig_power_substitution",
        argument_regime="even_sin_odd_cos_cube",
        domain_regime="total_real_function",
        trace_regime="mixed_trig_power_substitution_delegation",
        presentation_regime="polynomial_in_trig",
    ),
    IntegrateCommandMatrixCase(
        name="mixed_trig_odd_sine_cube",
        expr="integrate(sin(x)^3*cos(x)^2, x)",
        expected_result="1/15·(3·cos(x)^5 - 5·cos(x)^3)",
        expected_derivative_equivalent_to="sin(x)^3*cos(x)^2",
        expected_direct_diff_integrate_equivalent_to="sin(x)^3*cos(x)^2",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="mixed_trig_power_substitution",
        argument_regime="odd_sin_cube_even_cos",
        domain_regime="total_real_function",
        trace_regime="mixed_trig_power_substitution_delegation",
        presentation_regime="polynomial_in_trig",
    ),
    IntegrateCommandMatrixCase(
        name="mixed_trig_quartic_sine_odd_cosine",
        expr="integrate(sin(x)^4*cos(x)^3, x)",
        expected_result="1/35·(7·sin(x)^5 - 5·sin(x)^7)",
        expected_derivative_equivalent_to="sin(x)^4*cos(x)^3",
        expected_direct_diff_integrate_equivalent_to="sin(x)^4*cos(x)^3",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="mixed_trig_power_substitution",
        argument_regime="quartic_sin_odd_cos",
        domain_regime="total_real_function",
        trace_regime="mixed_trig_power_substitution_delegation",
        presentation_regime="polynomial_in_trig",
    ),
    IntegrateCommandMatrixCase(
        name="mixed_trig_quintic_sine_even_cosine",
        expr="integrate(sin(x)^5*cos(x)^2, x)",
        expected_result="1/21·(42/5·cos(x)^5 - 7·cos(x)^3 - 3·cos(x)^7)",
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="mixed_trig_power_substitution",
        argument_regime="quintic_sin_even_cos",
        domain_regime="total_real_function",
        trace_regime="mixed_trig_power_substitution_delegation",
        presentation_regime="polynomial_in_trig",
    ),
    IntegrateCommandMatrixCase(
        name="mixed_trig_odd_sine_quartic_cosine",
        expr="integrate(sin(x)^3*cos(x)^4, x)",
        expected_result="1/35·(5·cos(x)^7 - 7·cos(x)^5)",
        expected_derivative_equivalent_to="sin(x)^3*cos(x)^4",
        expected_direct_diff_integrate_equivalent_to="sin(x)^3*cos(x)^4",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="mixed_trig_power_substitution",
        argument_regime="odd_sin_quartic_cos",
        domain_regime="total_real_function",
        trace_regime="mixed_trig_power_substitution_delegation",
        presentation_regime="polynomial_in_trig",
    ),
    IntegrateCommandMatrixCase(
        name="mixed_trig_shared_double_angle",
        expr="integrate(sin(2*x)^3*cos(2*x)^2, x)",
        expected_result="1/30·(3·cos(2·x)^5 - 5·cos(2·x)^3)",
        expected_derivative_equivalent_to="sin(2*x)^3*cos(2*x)^2",
        expected_direct_diff_integrate_equivalent_to="sin(2*x)^3*cos(2*x)^2",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="mixed_trig_power_substitution",
        argument_regime="shared_double_angle_odd",
        domain_regime="total_real_function",
        trace_regime="mixed_trig_power_substitution_delegation",
        presentation_regime="polynomial_in_trig",
    ),
    IntegrateCommandMatrixCase(
        name="reciprocal_quadratic_irrational_monic",
        expr="integrate(1/(x^2-2), x)",
        expected_result="(1/2·ln(|(x - sqrt(2)) / (sqrt(2) + x)|))/sqrt(2)",
        expected_required_display=(
            "x - sqrt(2) ≠ 0",
            "sqrt(2) + x ≠ 0",
            "x^2 - 2 ≠ 0",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="reciprocal_quadratic_irrational_log",
        argument_regime="monic_irrational_difference",
        domain_regime="irrational_pole_pair_required",
        trace_regime="reciprocal_quadratic_irrational_log_form",
        presentation_regime="log_ratio_over_surd",
    ),
    IntegrateCommandMatrixCase(
        name="reciprocal_quadratic_irrational_scaled",
        expr="integrate(2/(x^2-3), x)",
        expected_result="ln(|(x - sqrt(3)) / (sqrt(3) + x)|) / sqrt(3)",
        expected_derivative_equivalent_to="2/(x^2-3)",
        expected_derivative_required_display=(
            "x - sqrt(3) ≠ 0",
            "sqrt(3) + x ≠ 0",
            "x^2 - 3 ≠ 0",
        ),
        expected_required_display=(
            "x - sqrt(3) ≠ 0",
            "sqrt(3) + x ≠ 0",
            "x^2 - 3 ≠ 0",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="reciprocal_quadratic_irrational_log",
        argument_regime="scaled_irrational_difference",
        domain_regime="irrational_pole_pair_required",
        trace_regime="reciprocal_quadratic_irrational_log_form",
        presentation_regime="log_ratio_over_surd",
    ),
    IntegrateCommandMatrixCase(
        name="reciprocal_quadratic_irrational_completed_square",
        expr="integrate(1/(x^2+2*x-1), x)",
        expected_result="(1/2·ln(|(x + 1 - sqrt(2)) / (sqrt(2) + x + 1)|))/sqrt(2)",
        expected_required_display=(
            "x + 1 - sqrt(2) ≠ 0",
            "sqrt(2) + x + 1 ≠ 0",
            "x^2 + 2·x - 1 ≠ 0",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="reciprocal_quadratic_irrational_log",
        argument_regime="completed_square_irrational",
        domain_regime="irrational_pole_pair_required",
        trace_regime="reciprocal_quadratic_irrational_log_form",
        presentation_regime="log_ratio_over_surd",
    ),
    IntegrateCommandMatrixCase(
        name="reciprocal_quadratic_irrational_shifted",
        expr="integrate(1/(x^2-2*x-2), x)",
        expected_result="(1/2·ln(|(x - 1 - sqrt(3)) / (sqrt(3) + x - 1)|))/sqrt(3)",
        expected_required_display=(
            "x - 1 - sqrt(3) ≠ 0",
            "sqrt(3) + x - 1 ≠ 0",
            "x^2 - 2·x - 2 ≠ 0",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="reciprocal_quadratic_irrational_log",
        argument_regime="shifted_completed_square_irrational",
        domain_regime="irrational_pole_pair_required",
        trace_regime="reciprocal_quadratic_irrational_log_form",
        presentation_regime="log_ratio_over_surd",
    ),
    IntegrateCommandMatrixCase(
        name="quartic_symmetric_reciprocal",
        expr="integrate(1/(x^4+1), x)",
        expected_result="(arctan((x^2 - 1) / x / sqrt(2))·1/2)/sqrt(2) - 1/2·(1/2·ln(|((x^2 + 1) / x - sqrt(2)) / (sqrt(2) + (x^2 + 1) / x)|))/sqrt(2)",
        expected_required_display=(
            "(x^2 + 1) / x - sqrt(2) ≠ 0",
            "sqrt(2) + (x^2 + 1) / x ≠ 0",
            "x ≠ 0",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="quartic_symmetric_substitution",
        argument_regime="famous_reciprocal_quartic",
        domain_regime="symmetric_substitution_pole_required",
        trace_regime="quartic_symmetric_substitution_delegation",
        presentation_regime="arctan_log_symmetric_combination",
    ),
    IntegrateCommandMatrixCase(
        name="quartic_symmetric_square_numerator",
        expr="integrate(x^2/(x^4+1), x)",
        expected_result="(1/2·ln(|((x^2 + 1) / x - sqrt(2)) / (sqrt(2) + (x^2 + 1) / x)|)·1/2)/sqrt(2) + (arctan((x^2 - 1) / x / sqrt(2))·1/2)/sqrt(2)",
        expected_required_display=(
            "(x^2 + 1) / x - sqrt(2) ≠ 0",
            "sqrt(2) + (x^2 + 1) / x ≠ 0",
            "x ≠ 0",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="quartic_symmetric_substitution",
        argument_regime="square_over_quartic",
        domain_regime="symmetric_substitution_pole_required",
        trace_regime="quartic_symmetric_substitution_delegation",
        presentation_regime="arctan_log_symmetric_combination",
    ),
    IntegrateCommandMatrixCase(
        name="quartic_symmetric_even_arctan_piece",
        expr="integrate((x^2+1)/(x^4+1), x)",
        expected_result="arctan((x^2 - 1) / x / sqrt(2)) / sqrt(2)",
        expected_required_display=(
            "x ≠ 0",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="quartic_symmetric_substitution",
        argument_regime="symmetric_arctan_piece",
        domain_regime="symmetric_substitution_pole_required",
        trace_regime="quartic_symmetric_substitution_delegation",
        presentation_regime="arctan_log_symmetric_combination",
    ),
    IntegrateCommandMatrixCase(
        name="quartic_symmetric_odd_log_piece",
        expr="integrate((x^2-1)/(x^4+1), x)",
        expected_result="(ln(|((x^2 + 1) / x - sqrt(2)) / (sqrt(2) + (x^2 + 1) / x)|)·1/2)/sqrt(2)",
        expected_required_display=(
            "(x^2 + 1) / x - sqrt(2) ≠ 0",
            "sqrt(2) + (x^2 + 1) / x ≠ 0",
            "x ≠ 0",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="quartic_symmetric_substitution",
        argument_regime="symmetric_log_piece",
        domain_regime="symmetric_substitution_pole_required",
        trace_regime="quartic_symmetric_substitution_delegation",
        presentation_regime="arctan_log_symmetric_combination",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_reciprocal_e_bound_unit_log",
        expr="integrate(1/x, x, 1, e)",
        expected_result="1",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="reciprocal_pole_outside_e_interval",
        domain_regime="interval_certified_with_source_condition",
        trace_regime="definite_ftc_evaluation",
        presentation_regime="exact_unit_value",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_reciprocal_square_e_bound",
        expr="integrate(1/x^2, x, 1, e)",
        expected_result="(e - 1) / e",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="reciprocal_square_pole_outside_e_interval",
        domain_regime="interval_certified_with_source_condition",
        trace_regime="definite_ftc_evaluation",
        presentation_regime="exact_e_quotient",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_scaled_reciprocal_e_bound",
        expr="integrate(2/x, x, 1, e)",
        expected_result="2",
        expected_required_display=("x ≠ 0",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="scaled_reciprocal_pole_outside_e_interval",
        domain_regime="interval_certified_with_source_condition",
        trace_regime="definite_ftc_evaluation",
        presentation_regime="exact_integer_value",
    ),
    IntegrateCommandMatrixCase(
        name="reciprocal_hyperbolic_sech_squared_tanh",
        expr="integrate(sech(x)^2, x)",
        expected_result="tanh(x)",
        expected_derivative_equivalent_to="sech(x)^2",
        expected_direct_diff_integrate_equivalent_to="sech(x)^2",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="reciprocal_hyperbolic_desugar",
        argument_regime="sech_squared_parse_desugar",
        domain_regime="total_real_function",
        trace_regime="cosh_reciprocal_square_delegation",
        presentation_regime="hyperbolic_tangent",
    ),
    IntegrateCommandMatrixCase(
        name="reciprocal_hyperbolic_csch_squared_coth",
        expr="integrate(csch(x)^2, x)",
        expected_result="-1 / tanh(x)",
        expected_derivative_equivalent_to="csch(x)^2",
        expected_direct_diff_integrate_equivalent_to="csch(x)^2",
        expected_direct_diff_integrate_required_display=("sinh(x) ≠ 0",),
        expected_derivative_required_display=("sinh(x) ≠ 0",),
        expected_required_display=("sinh(x) ≠ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="reciprocal_hyperbolic_desugar",
        argument_regime="csch_squared_parse_desugar",
        domain_regime="nonzero_denominator_required",
        trace_regime="sinh_reciprocal_square_delegation",
        presentation_regime="negated_hyperbolic_cotangent",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_semicircle_area",
        expr="integrate(sqrt(1-x^2), x, -1, 1)",
        expected_result="1/2·pi",
        expected_required_display=("-1 ≤ x ≤ 1",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="semicircle_area_double_touch",
        domain_regime="boundary_touch_one_sided_limit",
        trace_regime="definite_ftc_boundary_limit",
        presentation_regime="exact_pi_value",
    ),
    IntegrateCommandMatrixCase(
        name="hermite_split_square_over_completed_square",
        expr="integrate(x^2/sqrt(x^2+x+1), x)",
        expected_result="(1/2·x - 3/4)·sqrt(x^2 + x + 1) - 1/8·asinh((2·x + 1) / sqrt(3))",
        expected_derivative_equivalent_to="x^2/sqrt(x^2+x+1)",
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="polynomial_over_sqrt_hermite_split",
        argument_regime="square_over_completed_square_positive",
        domain_regime="total_real_function",
        trace_regime="hermite_radical_split",
        presentation_regime="polynomial_sqrt_plus_asinh_split",
    ),
    IntegrateCommandMatrixCase(
        name="hermite_split_square_over_circle_shifted",
        expr="integrate(x^2/sqrt(2*x-x^2), x)",
        expected_result="1/2·(3·arcsin(x - 1) + (2·x - x^2)^(1/2)·(-x - 3))",
        expected_derivative_equivalent_to="x^2/sqrt(2*x-x^2)",
        expected_direct_diff_integrate_equivalent_to="x^2/sqrt(2*x-x^2)",
        expected_direct_diff_integrate_required_display=("0 < x < 2",),
        expected_derivative_required_display=("0 < x < 2",),
        expected_required_display=("0 < x < 2",),
        expected_step_substrings=("Calcular la integral",),
        family="polynomial_over_sqrt_hermite_split",
        argument_regime="square_over_completed_square_circle",
        domain_regime="open_interval_required",
        trace_regime="hermite_radical_split",
        presentation_regime="polynomial_sqrt_plus_arcsin_split",
    ),
    IntegrateCommandMatrixCase(
        name="hermite_split_cross_terms_over_acosh_radical",
        expr="integrate((x^2+1)/sqrt(x^2+2*x), x)",
        expected_result="1/2·(5·acosh(x + 1) + (x^2 + 2·x)^(1/2)·(x - 3))",
        expected_derivative_equivalent_to="(x^2+1)/sqrt(x^2+2*x)",
        expected_direct_diff_integrate_equivalent_to="(x^2+1)/sqrt(x^2+2*x)",
        expected_direct_diff_integrate_required_display=("x > 0",),
        expected_derivative_required_display=("x > 0",),
        expected_required_display=(
            "x < -2 or x > 0",
            "x ≥ 0",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="polynomial_over_sqrt_hermite_split",
        argument_regime="cross_terms_over_acosh_radical",
        domain_regime="open_interval_required",
        trace_regime="hermite_radical_split",
        presentation_regime="polynomial_sqrt_plus_acosh_split",
    ),
    IntegrateCommandMatrixCase(
        name="linear_over_sqrt_completed_square_asinh",
        expr="integrate(x/sqrt(x^2+x+1), x)",
        expected_result="sqrt(x^2 + x + 1) - 1/2·asinh((2·x + 1) / sqrt(3))",
        expected_derivative_equivalent_to="x/sqrt(x^2+x+1)",
        expected_direct_diff_integrate_equivalent_to="x/sqrt(x^2+x+1)",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="linear_over_sqrt_shifted_quadratic",
        argument_regime="linear_over_completed_square_positive",
        domain_regime="total_real_function",
        trace_regime="linear_radical_split",
        presentation_regime="sqrt_plus_asinh_split",
    ),
    IntegrateCommandMatrixCase(
        name="linear_over_sqrt_completed_square_arcsin",
        expr="integrate(x/sqrt(2*x-x^2), x)",
        expected_result="arcsin(x - 1) - (2·x - x^2)^(1/2)",
        expected_derivative_equivalent_to="x/sqrt(2*x-x^2)",
        expected_direct_diff_integrate_equivalent_to="x/sqrt(2*x-x^2)",
        expected_direct_diff_integrate_required_display=("0 < x < 2",),
        expected_derivative_required_display=("0 < x < 2",),
        expected_required_display=("0 < x < 2",),
        expected_step_substrings=("Calcular la integral",),
        family="linear_over_sqrt_shifted_quadratic",
        argument_regime="linear_over_completed_square_circle",
        domain_regime="open_interval_required",
        trace_regime="linear_radical_split",
        presentation_regime="sqrt_plus_arcsin_split",
    ),
    IntegrateCommandMatrixCase(
        name="monomial_over_sqrt_hyperbolic_asinh",
        expr="integrate(x^2/sqrt(1+x^2), x)",
        expected_result="1/2·(x·(x^2 + 1)^(1/2) - asinh(x))",
        expected_derivative_equivalent_to="x^2/sqrt(1+x^2)",
        expected_direct_diff_integrate_equivalent_to="x^2/sqrt(1+x^2)",
        expected_direct_diff_integrate_required_display=(),
        expected_derivative_required_display=(),
        expected_required_display=(),
        expected_step_substrings=("Calcular la integral",),
        family="monomial_over_sqrt_reduction",
        argument_regime="square_over_hyperbolic_radical",
        domain_regime="total_real_function",
        trace_regime="monomial_radical_reduction",
        presentation_regime="asinh_radical_reduction_closed_form",
    ),
    IntegrateCommandMatrixCase(
        name="monomial_over_sqrt_hyperbolic_acosh",
        expr="integrate(x^2/sqrt(x^2-1), x)",
        expected_result="1/2·(acosh(x) + x·(x^2 - 1)^(1/2))",
        expected_derivative_equivalent_to="x^2/sqrt(x^2-1)",
        expected_direct_diff_integrate_equivalent_to="x^2/sqrt(x^2-1)",
        expected_direct_diff_integrate_required_display=("x > 1",),
        expected_derivative_required_display=("x > 1",),
        expected_required_display=(
            "x < -1 or x > 1",
            "x ≥ 1",
        ),
        expected_step_substrings=("Calcular la integral",),
        family="monomial_over_sqrt_reduction",
        argument_regime="square_over_acosh_radical",
        domain_regime="open_interval_required",
        trace_regime="monomial_radical_reduction",
        presentation_regime="acosh_radical_reduction_closed_form",
    ),
    IntegrateCommandMatrixCase(
        name="monomial_over_sqrt_reduction_square",
        expr="integrate(x^2/sqrt(1-x^2), x)",
        expected_result="1/2·(arcsin(x) - x·(1 - x^2)^(1/2))",
        expected_derivative_equivalent_to="x^2/sqrt(1-x^2)",
        expected_direct_diff_integrate_equivalent_to="x^2/sqrt(1-x^2)",
        expected_direct_diff_integrate_required_display=("-1 < x < 1",),
        expected_derivative_required_display=("-1 < x < 1",),
        expected_required_display=("-1 < x < 1",),
        expected_step_substrings=("Calcular la integral",),
        family="monomial_over_sqrt_reduction",
        argument_regime="square_over_unit_circle_radical",
        domain_regime="open_interval_required",
        trace_regime="monomial_radical_reduction",
        presentation_regime="arcsin_radical_reduction_closed_form",
    ),
    IntegrateCommandMatrixCase(
        name="monomial_over_sqrt_reduction_scaled_radicand",
        expr="integrate(x^2/sqrt(4-x^2), x)",
        expected_result="1/2·(4·arcsin(x / 2) - x·(4 - x^2)^(1/2))",
        expected_derivative_equivalent_to="x^2/sqrt(4-x^2)",
        expected_direct_diff_integrate_equivalent_to="x^2/sqrt(4-x^2)",
        expected_direct_diff_integrate_required_display=("-2 < x < 2",),
        expected_derivative_required_display=("-2 < x < 2",),
        expected_required_display=("-2 < x < 2",),
        expected_step_substrings=("Calcular la integral",),
        family="monomial_over_sqrt_reduction",
        argument_regime="square_over_scaled_circle_radical",
        domain_regime="open_interval_required",
        trace_regime="monomial_radical_reduction",
        presentation_regime="arcsin_radical_reduction_closed_form",
    ),
    IntegrateCommandMatrixCase(
        name="definite_integral_monomial_over_sqrt_quarter_pi",
        expr="integrate(x^2/sqrt(1-x^2), x, 0, 1)",
        expected_result="1/4·pi",
        expected_required_display=("-1 < x < 1",),
        expected_step_substrings=(
            "Hallar la antiderivada",
            "Evaluar la antiderivada en los límites",
        ),
        family="definite_integral_ftc",
        argument_regime="monomial_radical_boundary_touch",
        domain_regime="boundary_touch_one_sided_limit",
        trace_regime="definite_ftc_boundary_limit",
        presentation_regime="exact_pi_value",
    ),
    IntegrateCommandMatrixCase(
        name="by_parts_square_arcsine_radical_tail",
        expr="integrate(x^2*arcsin(x), x)",
        expected_result="1/9·(3·arcsin(x)·x^3 + (1 - x^2)^(1/2)·(x^2 + 2))",
        expected_derivative_equivalent_to="x^2*arcsin(x)",
        expected_direct_diff_integrate_equivalent_to="x^2*arcsin(x)",
        expected_direct_diff_integrate_required_display=("-1 < x < 1",),
        expected_derivative_required_display=("-1 < x < 1",),
        expected_required_display=("-1 ≤ x ≤ 1",),
        expected_step_substrings=("Calcular la integral",),
        family="by_parts_bounded_inverse_trig",
        argument_regime="square_times_arcsine_radical_tail",
        domain_regime="bounded_interval_required",
        trace_regime="by_parts_bounded_inverse_trig",
        presentation_regime="ninth_factored_inverse_radical",
    ),
    IntegrateCommandMatrixCase(
        name="by_parts_monomial_scaled_arcsine_radical_tail",
        expr="integrate(x*arcsin(2*x), x)",
        expected_result="1/16·(2·x·(1 - 4·x^2)^(1/2) + 8·arcsin(2·x)·x^2 - arcsin(2·x))",
        expected_derivative_equivalent_to="x*arcsin(2*x)",
        expected_direct_diff_integrate_equivalent_to="x*arcsin(2*x)",
        expected_direct_diff_integrate_required_display=("-1/2 < x < 1/2",),
        expected_derivative_required_display=("-1/2 < x < 1/2",),
        expected_required_display=("-1/2 ≤ x ≤ 1/2",),
        expected_step_substrings=("Calcular la integral",),
        family="by_parts_bounded_inverse_trig",
        argument_regime="monomial_times_scaled_arcsine",
        domain_regime="bounded_interval_required",
        trace_regime="by_parts_bounded_inverse_trig",
        presentation_regime="sixteenth_factored_inverse_radical",
    ),
    IntegrateCommandMatrixCase(
        name="by_parts_monomial_shifted_arcsine_domain",
        expr="integrate(x*arcsin(x+1), x)",
        expected_result="1/4·(2·arcsin(x + 1)·x^2 + (-x^2 - 2·x)^(1/2)·(x - 3) - 3·arcsin(x + 1))",
        expected_derivative_equivalent_to="x*arcsin(x+1)",
        expected_derivative_required_display=("-2 < x < 0",),
        expected_required_display=("-2 ≤ x ≤ 0",),
        expected_step_substrings=("Calcular la integral",),
        family="by_parts_bounded_inverse_trig",
        argument_regime="monomial_times_shifted_arcsine",
        domain_regime="bounded_interval_required",
        trace_regime="by_parts_bounded_inverse_trig",
        presentation_regime="quarter_factored_shifted_inverse_radical",
    ),
    IntegrateCommandMatrixCase(
        name="by_parts_monomial_arcsine_domain",
        expr="integrate(x*arcsin(x), x)",
        expected_result="1/4·(2·arcsin(x)·x^2 + x·(1 - x^2)^(1/2) - arcsin(x))",
        expected_derivative_equivalent_to="x*arcsin(x)",
        expected_direct_diff_integrate_equivalent_to="x*arcsin(x)",
        expected_direct_diff_integrate_required_display=("-1 < x < 1",),
        expected_derivative_required_display=("-1 < x < 1",),
        expected_required_display=("-1 ≤ x ≤ 1",),
        expected_step_substrings=("Calcular la integral",),
        family="by_parts_bounded_inverse_trig",
        argument_regime="monomial_times_arcsine",
        domain_regime="bounded_interval_required",
        trace_regime="by_parts_bounded_inverse_trig",
        presentation_regime="quarter_factored_inverse_radical",
    ),
    IntegrateCommandMatrixCase(
        name="by_parts_monomial_arccosine_domain",
        expr="integrate(x*arccos(x), x)",
        expected_result="1/4·(arccos(x)·(2·x^2 - 1) - x·(1 - x^2)^(1/2))",
        expected_derivative_equivalent_to="x*arccos(x)",
        expected_direct_diff_integrate_equivalent_to="x*arccos(x)",
        expected_direct_diff_integrate_required_display=("-1 < x < 1",),
        expected_derivative_required_display=("-1 < x < 1",),
        expected_required_display=("-1 ≤ x ≤ 1",),
        expected_step_substrings=("Calcular la integral",),
        family="by_parts_bounded_inverse_trig",
        argument_regime="monomial_times_arccosine",
        domain_regime="bounded_interval_required",
        trace_regime="by_parts_bounded_inverse_trig",
        presentation_regime="quarter_factored_inverse_radical",
    ),
    IntegrateCommandMatrixCase(
        name="by_parts_log_negative_power_quotient",
        expr="integrate(ln(x)/x^2, x)",
        expected_result="(-ln(x) - 1)/x",
        expected_derivative_result="ln(x) / x^2",
        expected_direct_diff_integrate_result="ln(x) / x^2",
        expected_direct_diff_integrate_required_display=("x > 0",),
        expected_derivative_required_display=("x > 0",),
        expected_required_display=("x > 0",),
        expected_step_substrings=("Usar integración por partes",),
        family="by_parts_log",
        argument_regime="negative_power_log_quotient",
        domain_regime="positive_required",
        trace_regime="by_parts_log",
        presentation_regime="negative_power_by_parts_quotient",
    ),
    IntegrateCommandMatrixCase(
        name="by_parts_log_fractional_power_radical",
        expr="integrate(ln(x)/sqrt(x), x)",
        expected_result="4·x^(1/2)·(1/2·ln(x) - 1)",
        expected_derivative_result="ln(x) / sqrt(x)",
        expected_direct_diff_integrate_result="ln(x) / sqrt(x)",
        expected_direct_diff_integrate_required_display=("x > 0",),
        expected_derivative_required_display=("x > 0",),
        expected_required_display=("x > 0",),
        expected_step_substrings=("Usar integración por partes",),
        family="by_parts_log",
        argument_regime="fractional_power_log_quotient",
        domain_regime="positive_required",
        trace_regime="by_parts_log",
        presentation_regime="factored_by_parts",
    ),
    IntegrateCommandMatrixCase(
        name="by_parts_log_square_negative_power_quotient",
        expr="integrate(ln(x)^2/x^2, x)",
        expected_result="-((ln(x)^2 + 2·ln(x) + 2)/x)",
        expected_derivative_result="ln(x)^2 / x^2",
        expected_direct_diff_integrate_result="ln(x)^2 / x^2",
        expected_direct_diff_integrate_required_display=("x > 0",),
        expected_derivative_required_display=("x > 0",),
        expected_required_display=("x > 0",),
        expected_step_substrings=("Usar integración por partes",),
        family="by_parts_log",
        argument_regime="log_square_negative_power_quotient",
        domain_regime="positive_required",
        trace_regime="by_parts_log",
        presentation_regime="negative_power_by_parts_quotient",
    ),
    IntegrateCommandMatrixCase(
        name="by_parts_affine_log_domain",
        expr="integrate((2*x+1)*ln(2*x+1), x)",
        expected_result="1/4·((2·x + 1)^2·ln(2·x + 1) - 2·x^2 - 2·x)",
        expected_derivative_result="ln(2·x + 1)·(2·x + 1)",
        expected_direct_diff_integrate_result="ln(2·x + 1)·(2·x + 1)",
        expected_direct_diff_integrate_required_display=("x > -1/2",),
        expected_derivative_required_display=("x > -1/2",),
        expected_required_display=("x > -1/2",),
        expected_step_substrings=(
            "Usar integración por partes",
            "Elegir u y dv",
            "Calcular du y v",
            "Aplicar la fórmula de integración por partes",
        ),
        family="by_parts_affine_log",
        argument_regime="affine_product",
        domain_regime="positive_required",
        trace_regime="by_parts_affine_log",
        presentation_regime="factored_by_parts",
    ),
    IntegrateCommandMatrixCase(
        name="affine_secant_tangent_derivative_product_domain",
        expr="integrate(sec(2*x+1)*tan(2*x+1), x)",
        expected_result="sec(2·x + 1) / 2",
        expected_derivative_equivalent_to="sec(2*x+1)*tan(2*x+1)",
        expected_direct_diff_integrate_result="tan(2·x + 1)·sec(2·x + 1)",
        expected_direct_diff_integrate_required_display=("cos(2·x + 1) ≠ 0",),
        expected_derivative_required_display=("cos(2·x + 1) ≠ 0",),
        expected_required_display=("cos(2·x + 1) ≠ 0",),
        expected_step_substrings=(
            "Expandir secante como recíproco de coseno",
            "Combinar fracciones en una multiplicación",
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            "Identificar u y du",
            "Ajustar el factor constante",
        ),
        family="reciprocal_trig_derivative_product",
        argument_regime="affine_reciprocal_trig_product",
        domain_regime="trig_reciprocal_product_pole_required",
        trace_regime="reciprocal_trig_derivative_product",
        presentation_regime="scaled_reciprocal_trig_product",
    ),
    IntegrateCommandMatrixCase(
        name="affine_cosecant_cotangent_derivative_product_domain",
        expr="integrate(csc(2*x+1)*cot(2*x+1), x)",
        expected_result="-csc(2·x + 1) / 2",
        expected_derivative_equivalent_to="csc(2*x+1)*cot(2*x+1)",
        expected_direct_diff_integrate_result="csc(2·x + 1)·cot(2·x + 1)",
        expected_direct_diff_integrate_required_display=("sin(2·x + 1) ≠ 0",),
        expected_derivative_required_display=("sin(2·x + 1) ≠ 0",),
        expected_required_display=("sin(2·x + 1) ≠ 0",),
        expected_step_substrings=(
            "Expandir cosecante como recíproco de seno",
            "Expandir cotangente como coseno entre seno",
            "Combinar fracciones en una multiplicación",
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            "Identificar u y du",
            "Ajustar el factor constante",
        ),
        family="reciprocal_trig_derivative_product",
        argument_regime="affine_reciprocal_trig_product",
        domain_regime="trig_reciprocal_product_pole_required",
        trace_regime="reciprocal_trig_derivative_product",
        presentation_regime="scaled_reciprocal_trig_product",
    ),
    IntegrateCommandMatrixCase(
        name="negative_affine_secant_tangent_derivative_product_domain",
        expr="integrate(sec(1-2*x)*tan(1-2*x), x)",
        expected_result="-sec(1 - 2·x) / 2",
        expected_derivative_equivalent_to="sec(1-2*x)*tan(1-2*x)",
        expected_direct_diff_integrate_result="tan(1 - 2·x)·sec(1 - 2·x)",
        expected_direct_diff_integrate_required_display=("cos(1 - 2·x) ≠ 0",),
        expected_derivative_required_display=("cos(1 - 2·x) ≠ 0",),
        expected_required_display=("cos(1 - 2·x) ≠ 0",),
        expected_step_substrings=(
            "Expandir secante como recíproco de coseno",
            "Combinar fracciones en una multiplicación",
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            "Identificar u y du",
            "du = -2",
            "Ajustar el factor constante",
        ),
        family="reciprocal_trig_derivative_product",
        argument_regime="negative_affine_reciprocal_trig_product",
        domain_regime="trig_reciprocal_product_pole_required",
        trace_regime="negative_orientation_reciprocal_trig_derivative_product",
        presentation_regime="negative_orientation_scaled_reciprocal_trig_product",
    ),
    IntegrateCommandMatrixCase(
        name="negative_affine_cosecant_cotangent_derivative_product_domain",
        expr="integrate(csc(1-2*x)*cot(1-2*x), x)",
        expected_result="csc(1 - 2·x) / 2",
        expected_derivative_equivalent_to="csc(1-2*x)*cot(1-2*x)",
        expected_direct_diff_integrate_result="csc(1 - 2·x)·cot(1 - 2·x)",
        expected_direct_diff_integrate_required_display=("sin(1 - 2·x) ≠ 0",),
        expected_derivative_required_display=("sin(1 - 2·x) ≠ 0",),
        expected_required_display=("sin(1 - 2·x) ≠ 0",),
        expected_step_substrings=(
            "Expandir cosecante como recíproco de seno",
            "Expandir cotangente como coseno entre seno",
            "Combinar fracciones en una multiplicación",
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            "Identificar u y du",
            "du = -2",
            "Ajustar el factor constante",
        ),
        family="reciprocal_trig_derivative_product",
        argument_regime="negative_affine_reciprocal_trig_product",
        domain_regime="trig_reciprocal_product_pole_required",
        trace_regime="negative_orientation_reciprocal_trig_derivative_product",
        presentation_regime="negative_orientation_scaled_reciprocal_trig_product",
    ),
    IntegrateCommandMatrixCase(
        name="polynomial_shifted_secant_tangent_derivative_product_domain",
        expr="integrate(2*x*sec(x^2+b)*tan(x^2+b), x)",
        expected_result="sec(x^2 + b)",
        expected_derivative_equivalent_to="2*x*sec(x^2+b)*tan(x^2+b)",
        expected_derivative_required_display=("cos(x^2 + b) ≠ 0",),
        expected_direct_diff_integrate_result="2·x·tan(x^2 + b)·sec(x^2 + b)",
        expected_direct_diff_integrate_required_display=("cos(x^2 + b) ≠ 0",),
        expected_required_display=("cos(x^2 + b) ≠ 0",),
        expected_step_substrings=(
            "Expandir tangente como seno entre coseno",
            "Expandir secante como recíproco de coseno",
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            "Identificar u y du",
            "du = 2\\cdot x",
        ),
        family="reciprocal_trig_derivative_product",
        argument_regime="shifted_polynomial_reciprocal_trig_product",
        domain_regime="trig_reciprocal_product_pole_required",
        trace_regime="polynomial_reciprocal_trig_derivative_product",
        presentation_regime="compact_reciprocal_trig_product",
    ),
    IntegrateCommandMatrixCase(
        name="polynomial_shifted_cosecant_cotangent_derivative_product_domain",
        expr="integrate(2*x*csc(x^2+b)*cot(x^2+b), x)",
        expected_result="-csc(x^2 + b)",
        expected_derivative_equivalent_to="2*x*csc(x^2+b)*cot(x^2+b)",
        expected_derivative_required_display=("sin(x^2 + b) ≠ 0",),
        expected_direct_diff_integrate_result="2·x·csc(x^2 + b)·cot(x^2 + b)",
        expected_direct_diff_integrate_required_display=("sin(x^2 + b) ≠ 0",),
        expected_required_display=("sin(x^2 + b) ≠ 0",),
        expected_step_substrings=(
            "Expandir cosecante como recíproco de seno",
            "Expandir cotangente como coseno entre seno",
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            "Identificar u y du",
            "du = 2\\cdot x",
        ),
        family="reciprocal_trig_derivative_product",
        argument_regime="shifted_polynomial_reciprocal_trig_product",
        domain_regime="trig_reciprocal_product_pole_required",
        trace_regime="polynomial_reciprocal_trig_derivative_product",
        presentation_regime="compact_reciprocal_trig_product",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_affine_exact_secant_tangent_derivative_product_domain",
        expr="integrate(a*sec(a*x+b)*tan(a*x+b), x)",
        expected_result="sec(a·x + b)",
        expected_derivative_equivalent_to="a*sec(a*x+b)*tan(a*x+b)",
        expected_derivative_required_display=("cos(a·x + b) ≠ 0",),
        expected_direct_diff_integrate_result="a·tan(a·x + b)·sec(a·x + b)",
        expected_direct_diff_integrate_required_display=("cos(a·x + b) ≠ 0",),
        expected_required_display=("cos(a·x + b) ≠ 0",),
        expected_step_substrings=(
            "Expandir secante como recíproco de coseno",
            "Combinar fracciones en una multiplicación",
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            "Identificar u y du",
            "du = a",
        ),
        family="reciprocal_trig_derivative_product",
        argument_regime="symbolic_affine_exact_reciprocal_trig_product",
        domain_regime="trig_reciprocal_product_exact_symbolic_derivative_pole_required",
        trace_regime="symbolic_exact_derivative_reciprocal_trig_product",
        presentation_regime="compact_symbolic_exact_reciprocal_trig_product",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_affine_exact_cosecant_cotangent_derivative_product_domain",
        expr="integrate(a*csc(a*x+b)*cot(a*x+b), x)",
        expected_result="-csc(a·x + b)",
        expected_derivative_equivalent_to="a*csc(a*x+b)*cot(a*x+b)",
        expected_derivative_required_display=("sin(a·x + b) ≠ 0",),
        expected_direct_diff_integrate_result="a·csc(a·x + b)·cot(a·x + b)",
        expected_direct_diff_integrate_required_display=("sin(a·x + b) ≠ 0",),
        expected_required_display=("sin(a·x + b) ≠ 0",),
        expected_step_substrings=(
            "Expandir cosecante como recíproco de seno",
            "Expandir cotangente como coseno entre seno",
            "Combinar fracciones en una multiplicación",
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            "Identificar u y du",
            "du = a",
        ),
        family="reciprocal_trig_derivative_product",
        argument_regime="symbolic_affine_exact_reciprocal_trig_product",
        domain_regime="trig_reciprocal_product_exact_symbolic_derivative_pole_required",
        trace_regime="symbolic_exact_derivative_reciprocal_trig_product",
        presentation_regime="compact_symbolic_exact_reciprocal_trig_product",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_secant_tangent_derivative_product_domain",
        expr="integrate(k*a*sec(a*x+b)*tan(a*x+b), x)",
        expected_result="k·sec(a·x + b)",
        expected_derivative_equivalent_to="k*a*sec(a*x+b)*tan(a*x+b)",
        expected_derivative_required_display=("cos(a·x + b) ≠ 0",),
        expected_direct_diff_integrate_result="a·k·tan(a·x + b)·sec(a·x + b)",
        expected_direct_diff_integrate_required_display=("cos(a·x + b) ≠ 0",),
        expected_required_display=("cos(a·x + b) ≠ 0",),
        expected_step_substrings=(
            "Expandir secante como recíproco de coseno",
            "Combinar fracciones en una multiplicación",
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            "Identificar u y du",
            "du = a",
        ),
        family="reciprocal_trig_derivative_product",
        argument_regime="symbolic_external_scale_reciprocal_trig_product",
        domain_regime="trig_reciprocal_product_exact_symbolic_derivative_pole_required",
        trace_regime="symbolic_external_scale_derivative_reciprocal_trig_product",
        presentation_regime="external_scale_compact_reciprocal_trig_product",
    ),
    IntegrateCommandMatrixCase(
        name="symbolic_external_scale_cosecant_cotangent_derivative_product_domain",
        expr="integrate(k*a*csc(a*x+b)*cot(a*x+b), x)",
        expected_result="-k·csc(a·x + b)",
        expected_derivative_equivalent_to="k*a*csc(a*x+b)*cot(a*x+b)",
        expected_derivative_required_display=("sin(a·x + b) ≠ 0",),
        expected_direct_diff_integrate_result="a·k·csc(a·x + b)·cot(a·x + b)",
        expected_direct_diff_integrate_required_display=("sin(a·x + b) ≠ 0",),
        expected_required_display=("sin(a·x + b) ≠ 0",),
        expected_step_substrings=(
            "Expandir cosecante como recíproco de seno",
            "Expandir cotangente como coseno entre seno",
            "Combinar fracciones en una multiplicación",
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            "Identificar u y du",
            "du = a",
        ),
        family="reciprocal_trig_derivative_product",
        argument_regime="symbolic_external_scale_reciprocal_trig_product",
        domain_regime="trig_reciprocal_product_exact_symbolic_derivative_pole_required",
        trace_regime="symbolic_external_scale_derivative_reciprocal_trig_product",
        presentation_regime="external_scale_compact_reciprocal_trig_product",
    ),
    IntegrateCommandMatrixCase(
        name="negative_symbolic_affine_exact_secant_tangent_derivative_product_domain",
        expr="integrate(-a*sec(b-a*x)*tan(b-a*x), x)",
        expected_result="sec(b - a·x)",
        expected_derivative_equivalent_to="-a*sec(b-a*x)*tan(b-a*x)",
        expected_derivative_required_display=("cos(b - a·x) ≠ 0",),
        expected_direct_diff_integrate_result="-tan(b - a·x)·sec(b - a·x)·a",
        expected_direct_diff_integrate_required_display=("cos(b - a·x) ≠ 0",),
        expected_required_display=("cos(b - a·x) ≠ 0",),
        expected_step_substrings=(
            "Expandir secante como recíproco de coseno",
            "Combinar fracciones en una multiplicación",
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            "Identificar u y du",
            "du = -a",
        ),
        family="reciprocal_trig_derivative_product",
        argument_regime="negative_symbolic_affine_exact_reciprocal_trig_product",
        domain_regime="trig_reciprocal_product_exact_symbolic_derivative_pole_required",
        trace_regime="negative_symbolic_exact_derivative_reciprocal_trig_product",
        presentation_regime="compact_negative_symbolic_exact_reciprocal_trig_product",
    ),
    IntegrateCommandMatrixCase(
        name="negative_symbolic_affine_exact_cosecant_cotangent_derivative_product_domain",
        expr="integrate(-a*csc(b-a*x)*cot(b-a*x), x)",
        expected_result="-csc(b - a·x)",
        expected_derivative_equivalent_to="-a*csc(b-a*x)*cot(b-a*x)",
        expected_derivative_required_display=("sin(b - a·x) ≠ 0",),
        expected_direct_diff_integrate_result="-csc(b - a·x)·cot(b - a·x)·a",
        expected_direct_diff_integrate_required_display=("sin(b - a·x) ≠ 0",),
        expected_required_display=("sin(b - a·x) ≠ 0",),
        expected_step_substrings=(
            "Expandir cosecante como recíproco de seno",
            "Expandir cotangente como coseno entre seno",
            "Combinar fracciones en una multiplicación",
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            "Identificar u y du",
            "du = -a",
        ),
        family="reciprocal_trig_derivative_product",
        argument_regime="negative_symbolic_affine_exact_reciprocal_trig_product",
        domain_regime="trig_reciprocal_product_exact_symbolic_derivative_pole_required",
        trace_regime="negative_symbolic_exact_derivative_reciprocal_trig_product",
        presentation_regime="compact_negative_symbolic_exact_reciprocal_trig_product",
    ),
    IntegrateCommandMatrixCase(
        name="negative_symbolic_external_scale_secant_tangent_derivative_product_domain",
        expr="integrate(-k*a*sec(b-a*x)*tan(b-a*x), x)",
        expected_result="k·sec(b - a·x)",
        expected_derivative_equivalent_to="-k*a*sec(b-a*x)*tan(b-a*x)",
        expected_derivative_required_display=("cos(b - a·x) ≠ 0",),
        expected_direct_diff_integrate_result="a·tan(b - a·x)·-sec(b - a·x)·k",
        expected_direct_diff_integrate_required_display=("cos(b - a·x) ≠ 0",),
        expected_required_display=("cos(b - a·x) ≠ 0",),
        expected_step_substrings=(
            "Expandir secante como recíproco de coseno",
            "Combinar fracciones en una multiplicación",
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            "Identificar u y du",
            "du = -a",
            "Ajustar el factor constante",
        ),
        family="reciprocal_trig_derivative_product",
        argument_regime="negative_symbolic_external_scale_reciprocal_trig_product",
        domain_regime="trig_reciprocal_product_exact_symbolic_derivative_pole_required",
        trace_regime="negative_symbolic_external_scale_derivative_reciprocal_trig_product",
        presentation_regime="compact_negative_symbolic_external_scale_reciprocal_trig_product",
    ),
    IntegrateCommandMatrixCase(
        name="negative_symbolic_external_scale_cosecant_cotangent_derivative_product_domain",
        expr="integrate(-k*a*csc(b-a*x)*cot(b-a*x), x)",
        expected_result="-k·csc(b - a·x)",
        expected_derivative_equivalent_to="-k*a*csc(b-a*x)*cot(b-a*x)",
        expected_derivative_required_display=("sin(b - a·x) ≠ 0",),
        expected_direct_diff_integrate_result="a·csc(b - a·x)·-cot(b - a·x)·k",
        expected_direct_diff_integrate_required_display=("sin(b - a·x) ≠ 0",),
        expected_required_display=("sin(b - a·x) ≠ 0",),
        expected_step_substrings=(
            "Expandir cosecante como recíproco de seno",
            "Expandir cotangente como coseno entre seno",
            "Combinar fracciones en una multiplicación",
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            "Identificar u y du",
            "du = -a",
            "Ajustar el factor constante",
        ),
        family="reciprocal_trig_derivative_product",
        argument_regime="negative_symbolic_external_scale_reciprocal_trig_product",
        domain_regime="trig_reciprocal_product_exact_symbolic_derivative_pole_required",
        trace_regime="negative_symbolic_external_scale_derivative_reciprocal_trig_product",
        presentation_regime="compact_negative_symbolic_external_scale_reciprocal_trig_product",
    ),
    IntegrateCommandMatrixCase(
        name="sqrt_chain_secant_tangent_domain",
        expr="integrate(sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)",
        expected_result="sec(sqrt(x))",
        expected_derivative_result="sec(sqrt(x))·tan(sqrt(x)) / (2·sqrt(x))",
        expected_derivative_required_display=("cos(sqrt(x)) ≠ 0", "x > 0"),
        expected_required_display=("cos(sqrt(x)) ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="sqrt_chain_reciprocal_trig",
        argument_regime="sqrt_chain",
        domain_regime="sqrt_chain_nonzero_positive",
        trace_regime="sqrt_chain_reciprocal_trig",
        presentation_regime="reciprocal_trig",
    ),
    IntegrateCommandMatrixCase(
        name="external_symbolic_scale_sqrt_chain_secant_tangent_domain",
        expr="integrate(k*sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)",
        expected_result="sec(sqrt(x))·k",
        expected_derivative_result="k·sin(sqrt(x)) / (2·cos(sqrt(x))^2·sqrt(x))",
        expected_derivative_required_display=("cos(sqrt(x)) ≠ 0", "x > 0"),
        expected_required_display=("cos(sqrt(x)) ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            "Identificar u y du",
            "u =",
            "du =",
            "Ajustar el factor constante",
        ),
        family="sqrt_chain_reciprocal_trig",
        argument_regime="external_symbolic_scale_sqrt_chain",
        domain_regime="sqrt_chain_nonzero_positive",
        trace_regime="external_symbolic_scale_sqrt_chain_reciprocal_trig",
        presentation_regime="external_scale_reciprocal_trig",
    ),
    IntegrateCommandMatrixCase(
        name="external_symbolic_scale_sqrt_chain_cosecant_cotangent_domain",
        expr="integrate(k*csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x)",
        expected_result="-csc(sqrt(x))·k",
        expected_derivative_result="k·cos(sqrt(x)) / (2·sin(sqrt(x))^2·sqrt(x))",
        expected_derivative_required_display=("sin(sqrt(x)) ≠ 0", "x > 0"),
        expected_required_display=("sin(sqrt(x)) ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            "Identificar u y du",
            "u =",
            "du =",
            "Ajustar el factor constante",
        ),
        family="sqrt_chain_reciprocal_trig",
        argument_regime="external_symbolic_scale_sqrt_chain",
        domain_regime="sqrt_chain_nonzero_positive",
        trace_regime="external_symbolic_scale_sqrt_chain_reciprocal_trig",
        presentation_regime="external_scale_reciprocal_trig",
    ),
    IntegrateCommandMatrixCase(
        name="external_symbolic_scale_shifted_sqrt_chain_secant_tangent_domain",
        expr="integrate(-k*sec(b-sqrt(x))*tan(b-sqrt(x))/(2*sqrt(x)), x)",
        expected_result="sec(b - sqrt(x))·k",
        expected_derivative_result="-k·sec(b - sqrt(x))·tan(b - sqrt(x)) / (2·sqrt(x))",
        expected_derivative_required_display=(
            "cos(b - sqrt(x)) ≠ 0",
            "x > 0",
        ),
        expected_required_display=("cos(b - sqrt(x)) ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            "Identificar u y du",
            "u =",
            "du =",
            "Ajustar el factor constante",
        ),
        family="sqrt_chain_reciprocal_trig",
        argument_regime="external_symbolic_scale_shifted_sqrt_chain",
        domain_regime="shifted_sqrt_chain_nonzero_positive",
        trace_regime="external_symbolic_scale_shifted_sqrt_chain_reciprocal_trig",
        presentation_regime="external_scale_shifted_reciprocal_trig",
    ),
    IntegrateCommandMatrixCase(
        name="external_symbolic_scale_shifted_sqrt_chain_cosecant_cotangent_domain",
        expr="integrate(-k*csc(b-sqrt(x))*cot(b-sqrt(x))/(2*sqrt(x)), x)",
        expected_result="-csc(b - sqrt(x))·k",
        expected_derivative_result="-k·csc(b - sqrt(x))·cot(b - sqrt(x)) / (2·sqrt(x))",
        expected_derivative_required_display=(
            "sin(b - sqrt(x)) ≠ 0",
            "x > 0",
        ),
        expected_required_display=("sin(b - sqrt(x)) ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            "Identificar u y du",
            "u =",
            "du =",
            "Ajustar el factor constante",
        ),
        family="sqrt_chain_reciprocal_trig",
        argument_regime="external_symbolic_scale_shifted_sqrt_chain",
        domain_regime="shifted_sqrt_chain_nonzero_positive",
        trace_regime="external_symbolic_scale_shifted_sqrt_chain_reciprocal_trig",
        presentation_regime="external_scale_shifted_reciprocal_trig",
    ),
    IntegrateCommandMatrixCase(
        name="external_symbolic_scale_sqrt_minus_symbol_chain_secant_tangent_domain",
        expr="integrate(k*sec(sqrt(x)-b)*tan(sqrt(x)-b)/(2*sqrt(x)), x)",
        expected_result="sec(sqrt(x) - b)·k",
        expected_derivative_result="k·sec(sqrt(x) - b)·tan(sqrt(x) - b) / (2·sqrt(x))",
        expected_derivative_required_display=(
            "cos(sqrt(x) - b) ≠ 0",
            "x > 0",
        ),
        expected_required_display=("cos(sqrt(x) - b) ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            "Identificar u y du",
            "u =",
            "du =",
            "Ajustar el factor constante",
        ),
        family="sqrt_chain_reciprocal_trig",
        argument_regime="external_symbolic_scale_sqrt_minus_symbol_chain",
        domain_regime="sqrt_minus_symbol_chain_nonzero_positive",
        trace_regime="external_symbolic_scale_sqrt_minus_symbol_chain_reciprocal_trig",
        presentation_regime="external_scale_sqrt_minus_symbol_reciprocal_trig",
    ),
    IntegrateCommandMatrixCase(
        name="external_symbolic_scale_sqrt_minus_symbol_chain_cosecant_cotangent_domain",
        expr="integrate(k*csc(sqrt(x)-b)*cot(sqrt(x)-b)/(2*sqrt(x)), x)",
        expected_result="-csc(sqrt(x) - b)·k",
        expected_derivative_result="k·csc(sqrt(x) - b)·cot(sqrt(x) - b) / (2·sqrt(x))",
        expected_derivative_required_display=(
            "sin(sqrt(x) - b) ≠ 0",
            "x > 0",
        ),
        expected_required_display=("sin(sqrt(x) - b) ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            "Identificar u y du",
            "u =",
            "du =",
            "Ajustar el factor constante",
        ),
        family="sqrt_chain_reciprocal_trig",
        argument_regime="external_symbolic_scale_sqrt_minus_symbol_chain",
        domain_regime="sqrt_minus_symbol_chain_nonzero_positive",
        trace_regime="external_symbolic_scale_sqrt_minus_symbol_chain_reciprocal_trig",
        presentation_regime="external_scale_sqrt_minus_symbol_reciprocal_trig",
    ),
    IntegrateCommandMatrixCase(
        name="sqrt_chain_tangent_log_domain",
        expr="integrate(tan(sqrt(x))/(2*sqrt(x)), x)",
        expected_result="-ln(|cos(sqrt(x))|)",
        expected_derivative_result="tan(sqrt(x)) / (2·sqrt(x))",
        expected_derivative_required_display=("cos(sqrt(x)) ≠ 0", "x > 0"),
        expected_required_display=("cos(sqrt(x)) ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar la regla de tan(u) -> -ln|cos(u)|",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="sqrt_chain_trig_log",
        argument_regime="sqrt_chain",
        domain_regime="sqrt_chain_nonzero_positive",
        trace_regime="sqrt_chain_trig_log",
        presentation_regime="abs_log_sqrt_chain",
    ),
    IntegrateCommandMatrixCase(
        name="shifted_sqrt_chain_tangent_log_domain",
        expr="integrate(tan(b-sqrt(x))/(2*sqrt(x)), x)",
        expected_result="ln(|cos(b - sqrt(x))|)",
        expected_derivative_result="tan(b - sqrt(x)) / (2·sqrt(x))",
        expected_derivative_required_display=("cos(b - sqrt(x)) ≠ 0", "x > 0"),
        expected_required_display=("cos(b - sqrt(x)) ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar la regla de tan(u) -> -ln|cos(u)|",
            "Identificar u y du",
            "u =",
            "du =",
            "Ajustar el factor constante",
        ),
        family="sqrt_chain_trig_log",
        argument_regime="shifted_sqrt_chain",
        domain_regime="shifted_sqrt_chain_nonzero_positive",
        trace_regime="shifted_sqrt_chain_trig_log",
        presentation_regime="abs_log_shifted_sqrt_chain",
    ),
    IntegrateCommandMatrixCase(
        name="sqrt_chain_hyperbolic_tangent_presimplified_condition_dedupe",
        expr="integrate(1/(2*sqrt(x+0)*tanh(sqrt(x+0))), x)",
        expected_result="ln(|sinh(sqrt(x))|)",
        expected_derivative_result="1 / (2·tanh(sqrt(x))·sqrt(x))",
        expected_derivative_required_display=("sinh(sqrt(x)) ≠ 0", "x > 0"),
        expected_required_display=("sinh(sqrt(x)) ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Agrupar términos semejantes",
            "Usar la regla de 1/tanh(u) -> ln|sinh(u)|",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="sqrt_chain_hyperbolic_log",
        argument_regime="sqrt_chain_presimplified",
        domain_regime="sqrt_chain_hyperbolic_presimplified_condition_dedupe",
        trace_regime="sqrt_chain_hyperbolic_log",
        presentation_regime="abs_log_sqrt_chain_hyperbolic_condition_dedupe",
    ),
    IntegrateCommandMatrixCase(
        name="shifted_sqrt_chain_hyperbolic_tangent_log_domain",
        expr="integrate(1/(2*sqrt(x)*tanh(b-sqrt(x))), x)",
        expected_result="-ln(|sinh(sqrt(x) - b)|)",
        expected_derivative_result="-1 / (2·tanh(x^(1/2) - b)·sqrt(x))",
        expected_derivative_required_display=("sinh(sqrt(x) - b) ≠ 0", "x > 0"),
        expected_required_display=("sinh(b - sqrt(x)) ≠ 0", "x > 0"),
        expected_step_substrings=(
            "Usar la regla de 1/tanh(u) -> ln|sinh(u)|",
            "Identificar u y du",
            "u =",
            "du =",
            "Ajustar el factor constante",
        ),
        family="sqrt_chain_hyperbolic_log",
        argument_regime="shifted_sqrt_chain",
        domain_regime="shifted_sqrt_chain_nonzero_positive",
        trace_regime="shifted_sqrt_chain_hyperbolic_log",
        presentation_regime="abs_log_shifted_sqrt_chain_hyperbolic",
    ),
    IntegrateCommandMatrixCase(
        name="affine_shifted_sqrt_chain_hyperbolic_tangent_log_domain",
        expr="integrate(3/(2*sqrt(3*x+1)*tanh(b-sqrt(3*x+1))), x)",
        expected_result="-ln(|sinh(sqrt(3·x + 1) - b)|)",
        expected_derivative_result="-3 / (2·tanh((3·x + 1)^(1/2) - b)·sqrt(3·x + 1))",
        expected_derivative_required_display=(
            "sinh(sqrt(3·x + 1) - b) ≠ 0",
            "x > -1/3",
        ),
        expected_required_display=("sinh(b - sqrt(3·x + 1)) ≠ 0", "x > -1/3"),
        expected_step_substrings=(
            "Usar la regla de 1/tanh(u) -> ln|sinh(u)|",
            "Identificar u y du",
            "u =",
            "du =",
            "Ajustar el factor constante",
        ),
        family="sqrt_chain_hyperbolic_log",
        argument_regime="affine_shifted_sqrt_chain",
        domain_regime="affine_shifted_sqrt_chain_nonzero_positive",
        trace_regime="affine_shifted_sqrt_chain_hyperbolic_log",
        presentation_regime="abs_log_affine_shifted_sqrt_chain_hyperbolic",
    ),
    IntegrateCommandMatrixCase(
        name="sqrt_chain_hyperbolic_cosh_over_sinh_square_domain",
        expr="integrate(cosh(sqrt(x))/(2*sqrt(x)*sinh(sqrt(x))^2), x)",
        expected_result="-1 / sinh(sqrt(x))",
        expected_derivative_result="cosh(sqrt(x)) / (2·sqrt(x)·sinh(sqrt(x))^2)",
        expected_derivative_required_display=("x > 0", "sinh(sqrt(x)) ≠ 0"),
        expected_required_display=("x > 0", "sinh(sqrt(x)) ≠ 0"),
        expected_step_substrings=(
            "Usar la regla de cosh(u)/sinh(u)^2 -> -1/sinh(u)",
            "Identificar u y du",
            "u =",
            "du =",
        ),
        family="hyperbolic_reciprocal_derivative_product",
        argument_regime="sqrt_chain_hyperbolic_product",
        domain_regime="sqrt_chain_hyperbolic_sine_pole_required",
        trace_regime="sqrt_chain_hyperbolic_reciprocal_derivative_product",
        presentation_regime="compact_hyperbolic_reciprocal",
    ),
    IntegrateCommandMatrixCase(
        name="shifted_sqrt_chain_hyperbolic_sinh_over_cosh_square_symbolic_scale_domain",
        expr="integrate(k*sinh(sqrt(x)-b)/(2*sqrt(x)*cosh(sqrt(x)-b)^2), x)",
        expected_result="-k / cosh(sqrt(x) - b)",
        expected_derivative_result=(
            "k·sinh(x^(1/2) - b) / (2·cosh(x^(1/2) - b)^2·sqrt(x))"
        ),
        expected_derivative_required_display=("x > 0",),
        expected_required_display=("x > 0",),
        expected_step_substrings=(
            "Usar la regla de sinh(u)/cosh(u)^2 -> -1/cosh(u)",
            "Identificar u y du",
            "u =",
            "du =",
            "Ajustar el factor constante",
        ),
        family="hyperbolic_reciprocal_derivative_product",
        argument_regime="shifted_sqrt_chain_hyperbolic_product",
        domain_regime="shifted_sqrt_chain_positive",
        trace_regime="symbolic_external_scale_shifted_sqrt_chain_hyperbolic_reciprocal_derivative_product",
        presentation_regime="external_scale_shifted_hyperbolic_reciprocal",
    ),
    IntegrateCommandMatrixCase(
        name="negative_shifted_sqrt_chain_hyperbolic_cosh_over_sinh_square_symbolic_scale_domain",
        expr="integrate(k*cosh(b-sqrt(x))/(2*sqrt(x)*sinh(b-sqrt(x))^2), x)",
        expected_result="-k / sinh(sqrt(x) - b)",
        expected_derivative_result=(
            "k·cosh(x^(1/2) - b) / (2·sinh(x^(1/2) - b)^2·sqrt(x))"
        ),
        expected_derivative_required_display=(
            "x > 0",
            "sinh(sqrt(x) - b) ≠ 0",
        ),
        expected_required_display=("x > 0", "sinh(b - sqrt(x)) ≠ 0"),
        expected_step_substrings=(
            "Hyperbolic Negative Argument",
            "Usar la regla de cosh(u)/sinh(u)^2 -> -1/sinh(u)",
            "Identificar u y du",
            "u =",
            "du =",
            "Ajustar el factor constante",
        ),
        family="hyperbolic_reciprocal_derivative_product",
        argument_regime="negative_shifted_sqrt_chain_hyperbolic_product",
        domain_regime="shifted_sqrt_chain_nonzero_positive",
        trace_regime="negative_symbolic_external_scale_shifted_sqrt_chain_hyperbolic_reciprocal_derivative_product",
        presentation_regime="negative_external_scale_shifted_hyperbolic_reciprocal",
    ),
    IntegrateCommandMatrixCase(
        name="invalid_log_base_integrand_undefined",
        expr="integrate(log(1,x), x)",
        expected_result="undefined",
        expected_step_substrings=("undefined",),
        family="log_domain_policy",
        argument_regime="invalid_base_variable_argument",
        domain_regime="empty_real_domain",
        outcome="undefined",
        trace_regime="invalid_integrand_domain_policy",
        presentation_regime="undefined",
    ),
    IntegrateCommandMatrixCase(
        name="nonfinite_integrand_undefined",
        expr="integrate(infinity, x)",
        expected_result="undefined",
        expected_step_substrings=(
            "Detectar integrando no finito",
            "undefined",
        ),
        family="nonfinite",
        argument_regime="constant",
        domain_regime="nonfinite_undefined",
        outcome="undefined",
        trace_regime="nonfinite_domain_policy",
        presentation_regime="undefined",
    ),
    IntegrateCommandMatrixCase(
        name="non_elementary_exp_quadratic_residual",
        expr="integrate(exp(x^2), x)",
        expected_result="integrate(e^(x^2), x)",
        expected_step_substrings=("Conservar integral residual",),
        family="non_elementary_exp_quadratic",
        argument_regime="unsupported_core",
        outcome="residual",
        residual_cause="non_elementary_composition",
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


def extract_cli_timings_us(payload: dict[str, Any] | None) -> dict[str, int]:
    if not payload:
        return {}
    raw = payload.get("timings_us")
    if not isinstance(raw, dict):
        return {}

    timings: dict[str, int] = {}
    for key in ("parse_us", "simplify_us", "total_us"):
        value = raw.get(key)
        if isinstance(value, int):
            timings[key] = value
    return timings


def timing_seconds(timings_us: dict[str, int], key: str) -> float | None:
    value = timings_us.get(key)
    if not isinstance(value, int):
        return None
    return value / 1_000_000.0


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
    raw_steps = payload.get("steps") or []
    if not isinstance(raw_steps, list):
        return ""

    fragments: list[str] = []
    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        for key in ("rule", "title", "before", "after", "before_latex", "after_latex"):
            value = step.get(key)
            if isinstance(value, str):
                fragments.append(value)
        raw_substeps = step.get("substeps") or []
        if not isinstance(raw_substeps, list):
            continue
        for substep in raw_substeps:
            if not isinstance(substep, dict):
                continue
            for key in ("title", "before", "after", "before_latex", "after_latex"):
                value = substep.get(key)
                if isinstance(value, str):
                    fragments.append(value)
    return "\n".join(fragments)


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
    if "fragile substring" in error:
        return "stderr_fragility"
    if "antiderivative verification timeout" in error:
        return "antiderivative_verification_timeout"
    if "antiderivative verification" in error:
        return "antiderivative_verification_mismatch"
    if "direct diff(integrate) timeout" in error:
        return "direct_diff_integrate_timeout"
    if "direct diff(integrate)" in error:
        return "direct_diff_integrate_mismatch"
    if error == "timeout":
        return "timeout"
    if "warning" in error:
        return "warning_mismatch"
    if "required_display" in error:
        return "required_display_mismatch"
    if "step trace" in error:
        return "step_trace_mismatch"
    if "result" in error:
        return "result_mismatch"
    if "returncode" in error:
        return "process_error"
    if "json" in error:
        return "parse_error"
    return "unknown"


def run_case(
    case: IntegrateCommandMatrixCase,
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
        integrate_elapsed = time.monotonic() - start
        return {
            "name": case.name,
            "status": "timeout",
            "error": "timeout",
            "error_kind": "timeout",
            "returncode": None,
            "wall_elapsed_seconds": round(integrate_elapsed, 3),
            "integrate_elapsed_seconds": round(integrate_elapsed, 3),
            "antiderivative_verification_mode": verification_mode(case),
            "stdout": stdout,
            "stderr": stderr,
        }

    wall_elapsed = time.monotonic() - start
    integrate_elapsed = wall_elapsed
    parsed, parse_error = parse_json(stdout)
    result = parsed.get("result") if isinstance(parsed, dict) else None
    required_display = extract_required_display(parsed)
    warnings = extract_warning_messages(parsed)
    step_text = extract_step_text(parsed)
    cli_timings_us = extract_cli_timings_us(parsed)
    cli_parse_seconds = timing_seconds(cli_timings_us, "parse_us")
    cli_simplify_seconds = timing_seconds(cli_timings_us, "simplify_us")
    cli_total_seconds = timing_seconds(cli_timings_us, "total_us")
    public_overhead_seconds = (
        max(0.0, integrate_elapsed - cli_total_seconds)
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

    derivative_result: str | None = None
    derivative_required_display: tuple[str, ...] = ()
    derivative_stderr = ""
    derivative_equivalence_result: str | None = None
    derivative_elapsed: float | None = None
    derivative_residual_simplify_elapsed: float | None = None
    if error is None and (
        case.expected_derivative_result is not None
        or case.expected_derivative_equivalent_to is not None
    ):
        if case.expected_derivative_equivalent_to is None:
            derivative_expr = f"diff({case.expected_result}, x)"
        else:
            derivative_expr = (
                f"diff({case.expected_result}, x) - "
                f"({case.expected_derivative_equivalent_to})"
            )
        derivative_command = [
            str(cas_cli),
            "eval",
            derivative_expr,
            "--format",
            "json",
        ]
        derivative_start = time.monotonic()
        derivative_process = subprocess.Popen(
            derivative_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            derivative_stdout, derivative_stderr = derivative_process.communicate(
                timeout=timeout_seconds
            )
        except subprocess.TimeoutExpired:
            terminate_process_group(derivative_process)
            derivative_stdout, derivative_stderr = derivative_process.communicate()
            derivative_elapsed = time.monotonic() - derivative_start
            error = "antiderivative verification timeout"
        else:
            derivative_elapsed = time.monotonic() - derivative_start
            derivative_parsed, derivative_parse_error = parse_json(derivative_stdout)
            derivative_result = (
                derivative_parsed.get("result")
                if isinstance(derivative_parsed, dict)
                else None
            )
            if case.expected_derivative_equivalent_to is not None:
                derivative_equivalence_result = derivative_result
            derivative_required_display = extract_required_display(derivative_parsed)
            derivative_ok = (
                derivative_parsed.get("ok")
                if isinstance(derivative_parsed, dict)
                else None
            )
            if derivative_process.returncode != 0:
                error = (
                    "antiderivative verification "
                    f"returncode={derivative_process.returncode}"
                )
            elif derivative_parse_error:
                error = f"antiderivative verification {derivative_parse_error}"
            elif derivative_ok is not True:
                error = "antiderivative verification ok was not true"
            elif (
                case.expected_derivative_equivalent_to is not None
                and derivative_equivalence_result not in {None, "0"}
            ):
                simplified_command = [
                    str(cas_cli),
                    "eval",
                    f"simplify({derivative_equivalence_result})",
                    "--format",
                    "json",
                ]
                simplified_start = time.monotonic()
                simplified_process = subprocess.Popen(
                    simplified_command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                try:
                    simplified_stdout, simplified_stderr = simplified_process.communicate(
                        timeout=timeout_seconds
                    )
                except subprocess.TimeoutExpired:
                    terminate_process_group(simplified_process)
                    simplified_stdout, simplified_stderr = simplified_process.communicate()
                    derivative_residual_simplify_elapsed = (
                        time.monotonic() - simplified_start
                    )
                    error = "antiderivative verification timeout"
                else:
                    derivative_residual_simplify_elapsed = (
                        time.monotonic() - simplified_start
                    )
                    derivative_stderr += simplified_stderr
                    simplified_parsed, simplified_parse_error = parse_json(simplified_stdout)
                    simplified_result = (
                        simplified_parsed.get("result")
                        if isinstance(simplified_parsed, dict)
                        else None
                    )
                    simplified_ok = (
                        simplified_parsed.get("ok")
                        if isinstance(simplified_parsed, dict)
                        else None
                    )
                    if simplified_process.returncode != 0:
                        error = (
                            "antiderivative verification "
                            f"returncode={simplified_process.returncode}"
                        )
                    elif simplified_parse_error:
                        error = f"antiderivative verification {simplified_parse_error}"
                    elif simplified_ok is not True:
                        error = "antiderivative verification ok was not true"
                    else:
                        derivative_equivalence_result = simplified_result

            if error is not None:
                pass
            elif (
                case.expected_derivative_equivalent_to is None
                and derivative_result != case.expected_derivative_result
            ):
                error = (
                    "antiderivative verification expected derivative result "
                    f"{case.expected_derivative_result!r}, got {derivative_result!r}"
                )
            elif (
                case.expected_derivative_equivalent_to is not None
                and derivative_equivalence_result != "0"
            ):
                error = (
                    "antiderivative verification expected residual result "
                    f"'0', got {derivative_equivalence_result!r}"
                )
            elif set(derivative_required_display) != set(
                case.expected_derivative_required_display
            ):
                error = (
                    "antiderivative verification expected required_display "
                    f"{case.expected_derivative_required_display!r}, "
                    f"got {derivative_required_display!r}"
                )
            elif (
                derivative_stderr_error := stderr_fragility_error(
                    derivative_stderr,
                    label="antiderivative verification stderr",
                )
            ) is not None:
                error = derivative_stderr_error

    direct_diff_integrate_result: str | None = None
    direct_diff_integrate_required_display: tuple[str, ...] = ()
    direct_diff_integrate_stderr = ""
    direct_diff_integrate_elapsed: float | None = None
    expected_direct_diff_integrate_result = direct_diff_integrate_expected_result(case)
    expected_direct_diff_integrate_equivalent_to = (
        direct_diff_integrate_expected_equivalent_to(case)
    )
    expected_direct_diff_integrate_required_display = (
        direct_diff_integrate_expected_required_display(case)
    )
    direct_diff_integrate_exact = expected_direct_diff_integrate_result is not None
    direct_diff_integrate_equivalence = (
        expected_direct_diff_integrate_equivalent_to is not None
    )
    if error is None and (direct_diff_integrate_exact or direct_diff_integrate_equivalence):
        direct_diff_integrate_expr = f"diff({case.expr}, x)"
        if direct_diff_integrate_equivalence:
            direct_diff_integrate_expr = (
                f"{direct_diff_integrate_expr} - "
                f"({expected_direct_diff_integrate_equivalent_to})"
            )
        direct_diff_integrate_command = [
            str(cas_cli),
            "eval",
            direct_diff_integrate_expr,
            "--format",
            "json",
        ]
        direct_diff_integrate_start = time.monotonic()
        direct_diff_integrate_process = subprocess.Popen(
            direct_diff_integrate_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            direct_diff_integrate_stdout, direct_diff_integrate_stderr = (
                direct_diff_integrate_process.communicate(timeout=timeout_seconds)
            )
        except subprocess.TimeoutExpired:
            terminate_process_group(direct_diff_integrate_process)
            direct_diff_integrate_stdout, direct_diff_integrate_stderr = (
                direct_diff_integrate_process.communicate()
            )
            direct_diff_integrate_elapsed = (
                time.monotonic() - direct_diff_integrate_start
            )
            error = "direct diff(integrate) timeout"
        else:
            direct_diff_integrate_elapsed = (
                time.monotonic() - direct_diff_integrate_start
            )
            direct_diff_integrate_parsed, direct_diff_integrate_parse_error = parse_json(
                direct_diff_integrate_stdout
            )
            direct_diff_integrate_result = (
                direct_diff_integrate_parsed.get("result")
                if isinstance(direct_diff_integrate_parsed, dict)
                else None
            )
            direct_diff_integrate_required_display = extract_required_display(
                direct_diff_integrate_parsed
            )
            direct_diff_integrate_warnings = extract_warning_messages(
                direct_diff_integrate_parsed
            )
            direct_diff_integrate_ok = (
                direct_diff_integrate_parsed.get("ok")
                if isinstance(direct_diff_integrate_parsed, dict)
                else None
            )
            if direct_diff_integrate_process.returncode != 0:
                error = (
                    "direct diff(integrate) "
                    f"returncode={direct_diff_integrate_process.returncode}"
                )
            elif direct_diff_integrate_parse_error:
                error = f"direct diff(integrate) {direct_diff_integrate_parse_error}"
            elif direct_diff_integrate_ok is not True:
                error = "direct diff(integrate) ok was not true"
            elif direct_diff_integrate_equivalence and direct_diff_integrate_result != "0":
                error = (
                    "direct diff(integrate) expected residual result "
                    f"'0', got {direct_diff_integrate_result!r}"
                )
            elif direct_diff_integrate_exact and (
                direct_diff_integrate_result
                != expected_direct_diff_integrate_result
            ):
                error = (
                    "direct diff(integrate) expected result "
                    f"{expected_direct_diff_integrate_result!r}, "
                    f"got {direct_diff_integrate_result!r}"
                )
            elif set(direct_diff_integrate_required_display) != set(
                expected_direct_diff_integrate_required_display
            ):
                error = (
                    "direct diff(integrate) expected required_display "
                    f"{expected_direct_diff_integrate_required_display!r}, "
                    f"got {direct_diff_integrate_required_display!r}"
                )
            else:
                direct_warnings_ok, direct_warning_error = warning_expectations_met(
                    (),
                    direct_diff_integrate_warnings,
                )
                if not direct_warnings_ok:
                    error = f"direct diff(integrate) {direct_warning_error}"
                elif (
                    direct_stderr_error := stderr_fragility_error(
                        direct_diff_integrate_stderr,
                        label="direct diff(integrate) stderr",
                    )
                ) is not None:
                    error = direct_stderr_error

    wall_elapsed = time.monotonic() - start
    status: Status = "pass" if error is None else "fail"
    error_kind = classify_error_kind(error)
    if error_kind in {
        "antiderivative_verification_timeout",
        "direct_diff_integrate_timeout",
    }:
        status = "timeout"
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
        "integrate_elapsed_seconds": round(integrate_elapsed, 3),
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
        "cli_total_seconds": (
            round(cli_total_seconds, 6) if cli_total_seconds is not None else None
        ),
        "cli_public_overhead_seconds": (
            round(public_overhead_seconds, 6)
            if public_overhead_seconds is not None
            else None
        ),
        "public_overhead_seconds": (
            round(public_overhead_seconds, 6)
            if public_overhead_seconds is not None
            else None
        ),
        "stdout_bytes": len(stdout.encode("utf-8")),
        "step_text_char_count": len(step_text),
        "antiderivative_verification_elapsed_seconds": (
            round(derivative_elapsed, 3) if derivative_elapsed is not None else None
        ),
        "antiderivative_residual_simplify_elapsed_seconds": (
            round(derivative_residual_simplify_elapsed, 3)
            if derivative_residual_simplify_elapsed is not None
            else None
        ),
        "direct_diff_integrate_elapsed_seconds": (
            round(direct_diff_integrate_elapsed, 3)
            if direct_diff_integrate_elapsed is not None
            else None
        ),
        "antiderivative_verification_mode": verification_mode(case),
        "result": result if isinstance(result, str) else None,
        "required_display": list(required_display),
        "warnings": list(warnings),
        "expected_result": case.expected_result,
        "expected_derivative_result": case.expected_derivative_result,
        "expected_derivative_equivalent_to": case.expected_derivative_equivalent_to,
        "derivative_result": derivative_result,
        "derivative_equivalence_result": derivative_equivalence_result,
        "expected_direct_diff_integrate_result": (
            expected_direct_diff_integrate_result
        ),
        "expected_direct_diff_integrate_equivalent_to": (
            expected_direct_diff_integrate_equivalent_to
        ),
        "direct_diff_integrate_result": direct_diff_integrate_result,
        "expected_direct_diff_integrate_required_display": list(
            expected_direct_diff_integrate_required_display
        ),
        "direct_diff_integrate_required_display": list(
            direct_diff_integrate_required_display
        ),
        "expected_derivative_required_display": list(
            case.expected_derivative_required_display
        ),
        "derivative_required_display": list(derivative_required_display),
        "expected_required_display": list(case.expected_required_display),
        "expected_warning_substrings": list(case.expected_warning_substrings),
        "expected_step_substrings": list(case.expected_step_substrings),
        "family": case.family,
        "argument_regime": case.argument_regime,
        "domain_regime": case.domain_regime,
        "outcome": case.outcome,
        "residual_cause": case.residual_cause,
        "trace_regime": case.trace_regime,
        "presentation_regime": case.presentation_regime,
        "calculus_maturity_block": calculus_maturity_block(case),
        "calculus_block_gate": calculus_block_gate(case),
        "stderr": stderr,
        "derivative_stderr": derivative_stderr,
        "direct_diff_integrate_stderr": direct_diff_integrate_stderr,
    }


def build_cases(
    case_filters: tuple[str, ...] = (),
) -> tuple[IntegrateCommandMatrixCase, ...]:
    validate_direct_diff_integrate_expectations(DEFAULT_INTEGRATE_COMMAND_MATRIX_CASES)
    if not case_filters:
        return DEFAULT_INTEGRATE_COMMAND_MATRIX_CASES
    selected = {case.name: case for case in DEFAULT_INTEGRATE_COMMAND_MATRIX_CASES}
    missing = [name for name in case_filters if name not in selected]
    if missing:
        raise SystemExit(
            f"unknown integrate command matrix case(s): {', '.join(missing)}"
        )
    return tuple(selected[name] for name in case_filters)


def count_by(
    cases: tuple[IntegrateCommandMatrixCase, ...],
    attr: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        value = getattr(case, attr)
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def trig_hyperbolic_policy_cluster(
    case: IntegrateCommandMatrixCase,
) -> str | None:
    if case.family == "reciprocal_trig_derivative_product":
        return "block7_trig_reciprocal_derivative_product"
    if case.family == "sqrt_chain_reciprocal_trig":
        return "block7_sqrt_chain_reciprocal_trig_product"
    if case.family == "hyperbolic_reciprocal_derivative_product":
        if "sqrt_chain" in case.argument_regime:
            return "block7_sqrt_chain_hyperbolic_reciprocal_derivative_product"
        return "block7_hyperbolic_reciprocal_derivative_product"
    if case.family == "hyperbolic_reciprocal_square":
        return "block7_hyperbolic_reciprocal_square"
    if case.family == "hyperbolic_reciprocal_fourth":
        return "block7_hyperbolic_reciprocal_fourth"
    if case.family == "explicit_reciprocal_hyperbolic_substitution":
        return "block7_explicit_reciprocal_hyperbolic_tangent"
    if case.family == "explicit_reciprocal_trig_substitution":
        return "block7_explicit_reciprocal_trig_log_substitution"
    if case.family == "trig_log_derivative_ratio":
        return "block7_trig_log_derivative_ratio"
    if case.family == "trig_reciprocal_log_table":
        return "block7_trig_reciprocal_log_table"
    if case.family == "hyperbolic_log_derivative_ratio":
        return "block7_hyperbolic_log_derivative_ratio"
    if case.family == "hyperbolic_tanh_log_derivative":
        return "block7_hyperbolic_tanh_log_derivative"
    if case.family == "sqrt_chain_trig_log":
        return "block7_sqrt_chain_trig_log"
    if case.family == "sqrt_chain_hyperbolic_log":
        return "block7_sqrt_chain_hyperbolic_log"
    if case.family == "explicit_reciprocal_trig_residual_domain":
        return "block9_explicit_reciprocal_trig_residual"
    return None


def count_trig_hyperbolic_policy_clusters(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        cluster = trig_hyperbolic_policy_cluster(case)
        if cluster is None:
            continue
        counts[cluster] = counts.get(cluster, 0) + 1
    return dict(sorted(counts.items()))


def base_integration_policy_cluster(
    case: IntegrateCommandMatrixCase,
) -> str | None:
    if case.family == "by_parts_exp":
        return "block4_exponential_by_parts"
    if case.family in {"by_parts_log", "by_parts_affine_log"}:
        return "block4_log_by_parts"
    if case.family == "log_power_product_substitution":
        return "block4_log_power_product_by_parts"
    return None


def count_base_integration_policy_clusters(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        cluster = base_integration_policy_cluster(case)
        if cluster is None:
            continue
        counts[cluster] = counts.get(cluster, 0) + 1
    return dict(sorted(counts.items()))


def radical_inverse_policy_cluster(
    case: IntegrateCommandMatrixCase,
) -> str | None:
    if case.family == "inverse_trig_root_reciprocal":
        return "block8_inverse_trig_root_reciprocal"
    if case.family == "by_parts_bounded_inverse_trig":
        return "block8_bounded_inverse_trig_by_parts"
    if case.family == "monomial_over_sqrt_reduction":
        return "block8_monomial_over_sqrt_reduction"
    if case.family == "linear_over_sqrt_shifted_quadratic":
        return "block8_linear_over_sqrt_shifted_quadratic"
    if case.family == "polynomial_over_sqrt_hermite_split":
        return "block8_polynomial_over_sqrt_hermite_split"
    if case.family == "radical_numerator_polynomial":
        return "block8_radical_numerator_polynomial"
    if case.family == "linear_radical_substitution":
        return "block8_linear_radical_substitution"
    if case.family == "quadratic_radical_over_monomial":
        return "block8_quadratic_radical_over_monomial"
    if case.family in {
        "inverse_hyperbolic_rational_affine",
        "inverse_hyperbolic_rational_table",
    }:
        return "block8_inverse_hyperbolic_rational_interval"
    if case.family in {
        "inverse_hyperbolic_sqrt_table",
        "inverse_hyperbolic_sqrt_symbolic_radius",
        "inverse_sqrt_affine",
        "inverse_sqrt_symbolic_radius",
        "inverse_sqrt_table",
    }:
        return "block8_inverse_sqrt_tables"
    return None


def count_radical_inverse_policy_clusters(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        cluster = radical_inverse_policy_cluster(case)
        if cluster is None:
            continue
        counts[cluster] = counts.get(cluster, 0) + 1
    return dict(sorted(counts.items()))


def direct_diff_integrate_cases(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> tuple[IntegrateCommandMatrixCase, ...]:
    return tuple(case for case in cases if has_direct_diff_integrate_probe(case))


def direct_diff_integrate_exact_cases(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> tuple[IntegrateCommandMatrixCase, ...]:
    return tuple(
        case
        for case in cases
        if direct_diff_integrate_expected_result(case) is not None
    )


def direct_diff_integrate_equivalence_cases(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> tuple[IntegrateCommandMatrixCase, ...]:
    return tuple(
        case
        for case in cases
        if direct_diff_integrate_expected_equivalent_to(case) is not None
    )


def derivative_verified_without_direct_diff_integrate_cases(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> tuple[IntegrateCommandMatrixCase, ...]:
    return tuple(
        case
        for case in cases
        if (
            case.expected_derivative_result is not None
            or case.expected_derivative_equivalent_to is not None
        )
        and not has_direct_diff_integrate_probe(case)
    )


def count_direct_diff_integrate_calculus_maturity_blocks(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in direct_diff_integrate_cases(cases):
        block = calculus_maturity_block(case)
        counts[block] = counts.get(block, 0) + 1
    return dict(sorted(counts.items()))


def count_direct_diff_integrate_calculus_block_gates(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in direct_diff_integrate_cases(cases):
        gate = calculus_block_gate(case)
        counts[gate] = counts.get(gate, 0) + 1
    return dict(sorted(counts.items()))


def count_direct_diff_integrate_trig_hyperbolic_policy_clusters(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    return count_trig_hyperbolic_policy_clusters(direct_diff_integrate_cases(cases))


def count_direct_diff_integrate_base_integration_policy_clusters(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    return count_base_integration_policy_clusters(direct_diff_integrate_cases(cases))


def count_direct_diff_integrate_radical_inverse_policy_clusters(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    return count_radical_inverse_policy_clusters(direct_diff_integrate_cases(cases))


def count_direct_diff_integrate_gap_calculus_maturity_blocks(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in derivative_verified_without_direct_diff_integrate_cases(cases):
        block = calculus_maturity_block(case)
        counts[block] = counts.get(block, 0) + 1
    return dict(sorted(counts.items()))


def count_direct_diff_integrate_gap_calculus_block_gates(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in derivative_verified_without_direct_diff_integrate_cases(cases):
        gate = calculus_block_gate(case)
        counts[gate] = counts.get(gate, 0) + 1
    return dict(sorted(counts.items()))


def count_direct_diff_integrate_gap_trig_hyperbolic_policy_clusters(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    return count_trig_hyperbolic_policy_clusters(
        derivative_verified_without_direct_diff_integrate_cases(cases)
    )


def count_direct_diff_integrate_gap_base_integration_policy_clusters(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    return count_base_integration_policy_clusters(
        derivative_verified_without_direct_diff_integrate_cases(cases)
    )


def count_direct_diff_integrate_gap_radical_inverse_policy_clusters(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    return count_radical_inverse_policy_clusters(
        derivative_verified_without_direct_diff_integrate_cases(cases)
    )


def group_case_names_by_key(
    cases: tuple[IntegrateCommandMatrixCase, ...],
    key_fn: Callable[[IntegrateCommandMatrixCase], str | None],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for case in cases:
        key = key_fn(case)
        if key is None:
            continue
        grouped.setdefault(key, []).append(case.name)
    return dict(sorted(grouped.items()))


def direct_diff_integrate_gap_case_names_by_calculus_maturity_block(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, list[str]]:
    return group_case_names_by_key(
        derivative_verified_without_direct_diff_integrate_cases(cases),
        calculus_maturity_block,
    )


def direct_diff_integrate_gap_case_names_by_calculus_block_gate(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, list[str]]:
    return group_case_names_by_key(
        derivative_verified_without_direct_diff_integrate_cases(cases),
        calculus_block_gate,
    )


def direct_diff_integrate_gap_case_names_by_trig_hyperbolic_policy_cluster(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, list[str]]:
    return group_case_names_by_key(
        derivative_verified_without_direct_diff_integrate_cases(cases),
        trig_hyperbolic_policy_cluster,
    )


def direct_diff_integrate_gap_case_names_by_base_integration_policy_cluster(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, list[str]]:
    return group_case_names_by_key(
        derivative_verified_without_direct_diff_integrate_cases(cases),
        base_integration_policy_cluster,
    )


def direct_diff_integrate_gap_case_names_by_radical_inverse_policy_cluster(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, list[str]]:
    return group_case_names_by_key(
        derivative_verified_without_direct_diff_integrate_cases(cases),
        radical_inverse_policy_cluster,
    )


def count_direct_diff_integrate_equivalence_calculus_maturity_blocks(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in direct_diff_integrate_equivalence_cases(cases):
        block = calculus_maturity_block(case)
        counts[block] = counts.get(block, 0) + 1
    return dict(sorted(counts.items()))


def count_direct_diff_integrate_equivalence_trig_hyperbolic_policy_clusters(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    return count_trig_hyperbolic_policy_clusters(
        direct_diff_integrate_equivalence_cases(cases)
    )


def count_direct_diff_integrate_equivalence_base_integration_policy_clusters(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    return count_base_integration_policy_clusters(
        direct_diff_integrate_equivalence_cases(cases)
    )


def calculus_maturity_block(case: IntegrateCommandMatrixCase) -> str:
    if case.family.startswith("definite_integral"):
        return "block13_definite_integrals"

    if case.outcome in {"residual", "undefined"}:
        return "block9_residuals_and_non_goals"

    if "algorithmic_backend" in case.family or "algorithmic_backend" in case.trace_regime:
        return "block12_hybrid_algorithmic_backend"

    cluster = trig_hyperbolic_policy_cluster(case)
    if cluster is not None and cluster.startswith("block7_"):
        return "block7_trig_hyperbolic_integration"

    # Direct arctan table rows are rational integration cells; block 8 is for
    # radical/inverse-family regimes where domain and orientation policy matter.
    if case.family == "inverse_trig_table":
        return "block6_rational_integration"

    if (
        "inverse" in case.family
        or "sqrt" in case.family
        or "radical" in case.family
        or "inverse_" in case.trace_regime
        or "arctan" in case.trace_regime
        or "asinh" in case.trace_regime
        or "atanh" in case.trace_regime
        or "radical" in case.argument_regime
    ):
        return "block8_radical_inverse_families"

    if (
        "partial_fraction" in case.family
        or "positive_quadratic" in case.family
        or "partial_fraction" in case.trace_regime
        or "polynomial_division" in case.trace_regime
        or "positive_quadratic_decomposition" in case.trace_regime
    ):
        return "block6_rational_integration"

    if "substitution" in case.trace_regime or "derivative" in case.argument_regime:
        return "block5_generalized_substitution"

    return "block4_base_integration"


def is_algorithmic_backend_boundary_case(case: IntegrateCommandMatrixCase) -> bool:
    return case.outcome == "supported" and (
        "algorithmic_backend" in case.family
        or "algorithmic_backend" in case.trace_regime
    )


def calculus_block_gate(case: IntegrateCommandMatrixCase) -> str:
    if is_algorithmic_backend_boundary_case(case):
        return "algorithmic_backend_boundary_verified"
    if case.outcome == "residual":
        return "safe_residual_policy"
    if case.outcome == "undefined":
        return "explicit_undefined_domain_policy"
    if case.expected_required_display:
        return "domain_conditions_and_verified_antiderivative"
    if case.expected_step_substrings:
        return "didactic_trace_and_verified_antiderivative"
    return "verified_antiderivative"


def count_calculus_maturity_blocks(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        block = calculus_maturity_block(case)
        counts[block] = counts.get(block, 0) + 1
    return dict(sorted(counts.items()))


def count_calculus_block_gates(
    cases: tuple[IntegrateCommandMatrixCase, ...],
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


def verification_regime(case: IntegrateCommandMatrixCase) -> str:
    has_diff_verification = (
        case.expected_derivative_result is not None
        or case.expected_derivative_equivalent_to is not None
    )
    has_direct_diff_integrate_probe_value = has_direct_diff_integrate_probe(case)
    if has_diff_verification and has_direct_diff_integrate_probe_value:
        return "verified_by_diff_and_direct_diff_integrate"
    if has_diff_verification:
        return "verified_by_diff"
    if has_direct_diff_integrate_probe_value:
        return "verified_by_direct_diff_integrate"
    if is_algorithmic_backend_boundary_case(case):
        return "verified_by_algorithmic_backend_boundary"
    if case.outcome == "residual":
        return "residual_not_verified"
    if case.outcome == "undefined":
        return "undefined_not_verified"
    if case.family.startswith("definite_integral"):
        # FTC values are constants: verification lives upstream in the
        # diff-verified antiderivative plus exact bound arithmetic.
        return "definite_ftc_from_verified_antiderivative"
    return "verification_gap"


def verification_mode(case: IntegrateCommandMatrixCase) -> str:
    if case.expected_derivative_equivalent_to is not None:
        return "residual_equivalence"
    if case.expected_derivative_result is not None:
        return "direct_derivative"
    if direct_diff_integrate_expected_equivalent_to(case) is not None:
        return "direct_diff_integrate_equivalence"
    if direct_diff_integrate_expected_result(case) is not None:
        return "direct_diff_integrate_exact"
    if is_algorithmic_backend_boundary_case(case):
        return "algorithmic_backend_boundary"
    if case.outcome == "residual":
        return "residual_not_verified"
    if case.outcome == "undefined":
        return "undefined_not_verified"
    if case.family.startswith("definite_integral"):
        return "definite_ftc_from_verified_antiderivative"
    return "verification_gap"


def count_verification_regimes(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        regime = verification_regime(case)
        counts[regime] = counts.get(regime, 0) + 1
    return dict(sorted(counts.items()))


def count_verified_supported_cases(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> int:
    verified_regimes = {
        "verified_by_algorithmic_backend_boundary",
        "verified_by_diff",
        "verified_by_diff_and_direct_diff_integrate",
        "verified_by_direct_diff_integrate",
    }
    return sum(
        1
        for case in cases
        if case.outcome == "supported"
        and verification_regime(case) in verified_regimes
    )


def count_residual_causes(
    cases: tuple[IntegrateCommandMatrixCase, ...],
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
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        if case.outcome != "residual":
            continue
        counts[case.family] = counts.get(case.family, 0) + 1
    return dict(sorted(counts.items()))


def count_residual_cause_families(
    cases: tuple[IntegrateCommandMatrixCase, ...],
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
    cases: tuple[IntegrateCommandMatrixCase, ...],
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
            "trace_regime",
            "calculus_maturity_block",
            "calculus_block_gate",
            "antiderivative_verification_mode",
        ):
            value = result.get(key)
            if isinstance(value, str):
                row[key] = value
        rows.append(row)
    return rows


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


def verification_mode_runtime_rows(
    results: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        mode = result.get("antiderivative_verification_mode")
        elapsed = result.get("antiderivative_verification_elapsed_seconds")
        if not isinstance(mode, str) or not isinstance(elapsed, (int, float)):
            continue
        groups.setdefault(mode, []).append(result)

    rows: list[dict[str, Any]] = []
    for mode, mode_results in groups.items():
        elapsed_values = [
            float(result["antiderivative_verification_elapsed_seconds"])
            for result in mode_results
        ]
        total_elapsed = sum(elapsed_values)
        slowest = max(
            mode_results,
            key=lambda result: float(
                result.get("antiderivative_verification_elapsed_seconds", 0.0)
            ),
        )
        rows.append(
            {
                "mode": mode,
                "case_count": len(mode_results),
                "total_elapsed_seconds": round(total_elapsed, 3),
                "avg_case_ms": round(
                    total_elapsed * 1000.0 / len(mode_results),
                    3,
                ),
                "max_elapsed_seconds": round(max(elapsed_values), 3),
                "slowest_case": slowest.get("name"),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["total_elapsed_seconds"]),
            -int(row["case_count"]),
            str(row["mode"]),
        )
    )
    return rows[:limit]


def residual_cause_runtime_rows(
    results: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("outcome") != "residual":
            continue
        cause = result.get("residual_cause")
        elapsed = result.get("integrate_elapsed_seconds")
        if not isinstance(cause, str) or not isinstance(elapsed, (int, float)):
            continue
        if cause == "not_applicable":
            cause = "unclassified_residual"
        groups.setdefault(cause, []).append(result)

    rows: list[dict[str, Any]] = []
    for cause, cause_results in groups.items():
        elapsed_values = [
            float(result["integrate_elapsed_seconds"]) for result in cause_results
        ]
        total_elapsed = sum(elapsed_values)
        slowest = max(
            cause_results,
            key=lambda result: float(result.get("integrate_elapsed_seconds", 0.0)),
        )
        rows.append(
            {
                "cause": cause,
                "case_count": len(cause_results),
                "total_elapsed_seconds": round(total_elapsed, 3),
                "avg_case_ms": round(total_elapsed * 1000.0 / len(cause_results), 3),
                "max_elapsed_seconds": round(max(elapsed_values), 3),
                "slowest_case": slowest.get("name"),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["total_elapsed_seconds"]),
            -int(row["case_count"]),
            str(row["cause"]),
        )
    )
    return rows[:limit]


def residual_cause_family_runtime_rows(
    results: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("outcome") != "residual":
            continue
        cause = result.get("residual_cause")
        family = result.get("family")
        elapsed = result.get("integrate_elapsed_seconds")
        if (
            not isinstance(cause, str)
            or not isinstance(family, str)
            or not isinstance(elapsed, (int, float))
        ):
            continue
        if cause == "not_applicable":
            cause = "unclassified_residual"
        groups.setdefault(f"{cause}/{family}", []).append(result)

    rows: list[dict[str, Any]] = []
    for cause_family, cause_family_results in groups.items():
        elapsed_values = [
            float(result["integrate_elapsed_seconds"])
            for result in cause_family_results
        ]
        total_elapsed = sum(elapsed_values)
        slowest = max(
            cause_family_results,
            key=lambda result: float(result.get("integrate_elapsed_seconds", 0.0)),
        )
        rows.append(
            {
                "cause_family": cause_family,
                "case_count": len(cause_family_results),
                "total_elapsed_seconds": round(total_elapsed, 3),
                "avg_case_ms": round(
                    total_elapsed * 1000.0 / len(cause_family_results), 3
                ),
                "max_elapsed_seconds": round(max(elapsed_values), 3),
                "slowest_case": slowest.get("name"),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["total_elapsed_seconds"]),
            -int(row["case_count"]),
            str(row["cause_family"]),
        )
    )
    return rows[:limit]


def residual_public_phase_case_rows(
    results: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    residual_results = [
        result
        for result in results
        if result.get("outcome") == "residual"
        and isinstance(result.get("integrate_elapsed_seconds"), (int, float))
        and isinstance(result.get("cli_total_seconds"), (int, float))
    ]
    residual_results.sort(
        key=lambda result: (
            -float(result.get("integrate_elapsed_seconds", 0.0)),
            str(result.get("name", "")),
        )
    )

    rows: list[dict[str, Any]] = []
    for result in residual_results[:limit]:
        integrate_elapsed = float(result.get("integrate_elapsed_seconds", 0.0))
        cli_total = float(result.get("cli_total_seconds", 0.0))
        public_overhead = float(result.get("public_overhead_seconds") or 0.0)
        row: dict[str, Any] = {
            "name": result.get("name"),
            "integrate_elapsed_seconds": round(integrate_elapsed, 3),
            "cli_total_seconds": round(cli_total, 6),
            "cli_simplify_ms": round(
                float(result.get("cli_simplify_us") or 0) / 1000.0,
                3,
            ),
            "public_overhead_seconds": round(public_overhead, 6),
            "public_overhead_share_percent": round(
                public_overhead * 100.0 / integrate_elapsed,
                1,
            )
            if integrate_elapsed > 0.0
            else 0.0,
            "required_display_count": len(result.get("required_display", [])),
            "step_text_char_count": int(result.get("step_text_char_count") or 0),
            "stdout_bytes": int(result.get("stdout_bytes") or 0),
        }
        for key in (
            "residual_cause",
            "family",
            "trace_regime",
            "domain_regime",
            "presentation_regime",
            "calculus_maturity_block",
            "calculus_block_gate",
        ):
            value = result.get(key)
            if isinstance(value, str):
                row[key] = value
        rows.append(row)
    return rows


def residual_public_phase_group_rows(
    results: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("outcome") != "residual":
            continue
        cause = result.get("residual_cause")
        if not isinstance(cause, str):
            continue
        if cause == "not_applicable":
            cause = "unclassified_residual"
        if not isinstance(result.get("cli_total_seconds"), (int, float)):
            continue
        groups.setdefault(cause, []).append(result)

    rows: list[dict[str, Any]] = []
    for cause, cause_results in groups.items():
        integrate_total = sum(
            float(result.get("integrate_elapsed_seconds") or 0.0)
            for result in cause_results
        )
        cli_total = sum(
            float(result.get("cli_total_seconds") or 0.0)
            for result in cause_results
        )
        public_overhead_total = sum(
            float(result.get("public_overhead_seconds") or 0.0)
            for result in cause_results
        )
        slowest = max(
            cause_results,
            key=lambda result: float(result.get("integrate_elapsed_seconds") or 0.0),
        )
        rows.append(
            {
                "cause": cause,
                "case_count": len(cause_results),
                "integrate_total_seconds": round(integrate_total, 3),
                "cli_total_seconds": round(cli_total, 6),
                "public_overhead_total_seconds": round(public_overhead_total, 6),
                "public_overhead_share_percent": round(
                    public_overhead_total * 100.0 / integrate_total,
                    1,
                )
                if integrate_total > 0.0
                else 0.0,
                "avg_required_display_count": round(
                    sum(len(result.get("required_display", [])) for result in cause_results)
                    / len(cause_results),
                    3,
                ),
                "avg_step_text_char_count": round(
                    sum(int(result.get("step_text_char_count") or 0) for result in cause_results)
                    / len(cause_results),
                    3,
                ),
                "slowest_case": slowest.get("name"),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["integrate_total_seconds"]),
            -int(row["case_count"]),
            str(row["cause"]),
        )
    )
    return rows[:limit]


def residual_public_phase_cause_family_group_rows(
    results: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("outcome") != "residual":
            continue
        cause = result.get("residual_cause")
        family = result.get("family")
        if not isinstance(cause, str) or not isinstance(family, str):
            continue
        if cause == "not_applicable":
            cause = "unclassified_residual"
        if not isinstance(result.get("cli_total_seconds"), (int, float)):
            continue
        groups.setdefault(f"{cause}/{family}", []).append(result)

    rows: list[dict[str, Any]] = []
    for cause_family, cause_family_results in groups.items():
        integrate_total = sum(
            float(result.get("integrate_elapsed_seconds") or 0.0)
            for result in cause_family_results
        )
        cli_total = sum(
            float(result.get("cli_total_seconds") or 0.0)
            for result in cause_family_results
        )
        public_overhead_total = sum(
            float(result.get("public_overhead_seconds") or 0.0)
            for result in cause_family_results
        )
        slowest = max(
            cause_family_results,
            key=lambda result: float(result.get("integrate_elapsed_seconds") or 0.0),
        )
        rows.append(
            {
                "cause_family": cause_family,
                "case_count": len(cause_family_results),
                "integrate_total_seconds": round(integrate_total, 3),
                "cli_total_seconds": round(cli_total, 6),
                "public_overhead_total_seconds": round(public_overhead_total, 6),
                "public_overhead_share_percent": round(
                    public_overhead_total * 100.0 / integrate_total,
                    1,
                )
                if integrate_total > 0.0
                else 0.0,
                "avg_required_display_count": round(
                    sum(
                        len(result.get("required_display", []))
                        for result in cause_family_results
                    )
                    / len(cause_family_results),
                    3,
                ),
                "avg_step_text_char_count": round(
                    sum(
                        int(result.get("step_text_char_count") or 0)
                        for result in cause_family_results
                    )
                    / len(cause_family_results),
                    3,
                ),
                "slowest_case": slowest.get("name"),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["integrate_total_seconds"]),
            -int(row["case_count"]),
            str(row["cause_family"]),
        )
    )
    return rows[:limit]


def run_residual_shape_orientation_probe(
    probe: ResidualShapeOrientationProbe,
    *,
    cas_cli: str | pathlib.Path,
    timeout_seconds: float,
    steps_mode: str,
) -> dict[str, Any]:
    command = [str(cas_cli), "eval", probe.expr, "--format", "json", "--steps", steps_mode]
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
        elapsed = time.monotonic() - start
        return {
            "name": probe.name,
            "status": "timeout",
            "error": "timeout",
            "expression_shape": probe.expression_shape,
            "orientation": probe.orientation,
            "steps_mode": steps_mode,
            "wall_elapsed_seconds": round(elapsed, 3),
            "stdout_bytes": len(stdout.encode("utf-8")),
            "stderr_bytes": len(stderr.encode("utf-8")),
        }

    elapsed = time.monotonic() - start
    parsed, parse_error = parse_json(stdout)
    ok = parsed.get("ok") if isinstance(parsed, dict) else None
    cli_timings_us = extract_cli_timings_us(parsed)
    error: str | None = None
    if process.returncode != 0:
        error = f"returncode={process.returncode}"
    elif parse_error:
        error = parse_error
    elif ok is not True:
        error = "ok was not true"

    row: dict[str, Any] = {
        "name": probe.name,
        "status": "pass" if error is None else "fail",
        "error": error,
        "expression_shape": probe.expression_shape,
        "orientation": probe.orientation,
        "steps_mode": steps_mode,
        "wall_elapsed_seconds": round(elapsed, 3),
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
    }
    result = parsed.get("result") if isinstance(parsed, dict) else None
    if isinstance(result, str):
        row["result"] = result
    required_display = extract_required_display(parsed)
    row["required_display"] = list(required_display)
    row["required_display_count"] = len(required_display)
    for source_key, output_key in (
        ("parse_us", "cli_parse_us"),
        ("simplify_us", "cli_simplify_us"),
        ("total_us", "cli_total_us"),
    ):
        value = cli_timings_us.get(source_key)
        if isinstance(value, int):
            row[output_key] = value
    return row


def residual_shape_orientation_probe_rows(
    *,
    cas_cli: str | pathlib.Path,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    return [
        run_residual_shape_orientation_probe(
            probe,
            cas_cli=cas_cli,
            timeout_seconds=timeout_seconds,
            steps_mode=steps_mode,
        )
        for probe in RESIDUAL_SHAPE_ORIENTATION_PROBES
        for steps_mode in ("off", "on")
    ]


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

    integrate_rows = phase_runtime_case_rows(
        results,
        phase_key="integrate_elapsed_seconds",
        output_key="integrate_elapsed_seconds",
    )
    if integrate_rows:
        summary["slowest_integrate_evaluations"] = integrate_rows

    verification_rows = phase_runtime_case_rows(
        results,
        phase_key="antiderivative_verification_elapsed_seconds",
        output_key="antiderivative_verification_elapsed_seconds",
    )
    if verification_rows:
        summary["slowest_antiderivative_verifications"] = verification_rows

    residual_simplify_rows = phase_runtime_case_rows(
        results,
        phase_key="antiderivative_residual_simplify_elapsed_seconds",
        output_key="antiderivative_residual_simplify_elapsed_seconds",
    )
    if residual_simplify_rows:
        summary["slowest_antiderivative_residual_simplifications"] = (
            residual_simplify_rows
        )

    direct_diff_integrate_rows = phase_runtime_case_rows(
        results,
        phase_key="direct_diff_integrate_elapsed_seconds",
        output_key="direct_diff_integrate_elapsed_seconds",
    )
    if direct_diff_integrate_rows:
        summary["slowest_direct_diff_integrate_checks"] = direct_diff_integrate_rows

    verification_mode_rows = verification_mode_runtime_rows(results)
    if verification_mode_rows:
        summary["runtime_by_antiderivative_verification_mode"] = (
            verification_mode_rows
        )
    residual_cause_rows = residual_cause_runtime_rows(results)
    if residual_cause_rows:
        summary["runtime_by_residual_cause"] = residual_cause_rows
    residual_cause_family_rows = residual_cause_family_runtime_rows(results)
    if residual_cause_family_rows:
        summary["runtime_by_residual_cause_family"] = residual_cause_family_rows
    residual_phase_rows = residual_public_phase_case_rows(results)
    if residual_phase_rows:
        summary["residual_public_phase_slowest_cases"] = residual_phase_rows
    residual_phase_group_rows = residual_public_phase_group_rows(results)
    if residual_phase_group_rows:
        summary["residual_public_phase_by_cause"] = residual_phase_group_rows
    residual_phase_cause_family_rows = residual_public_phase_cause_family_group_rows(
        results
    )
    if residual_phase_cause_family_rows:
        summary["residual_public_phase_by_cause_family"] = (
            residual_phase_cause_family_rows
        )
    return summary


def increment_issue_kind(issue_kind_counts: dict[str, int], error_kind: str | None) -> None:
    if error_kind is None:
        return
    issue_kind_counts[error_kind] = issue_kind_counts.get(error_kind, 0) + 1


def run_matrix(
    cases: tuple[IntegrateCommandMatrixCase, ...],
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
    antiderivative_verification_cases = sum(
        1
        for case in cases
        if (
            case.expected_derivative_result is not None
            or case.expected_derivative_equivalent_to is not None
        )
    )
    direct_diff_integrate_cases = sum(
        1 for case in cases if has_direct_diff_integrate_probe(case)
    )
    direct_diff_integrate_exact_cases = sum(
        1
        for case in cases
        if direct_diff_integrate_expected_result(case) is not None
    )
    direct_diff_integrate_equivalence_cases = sum(
        1
        for case in cases
        if direct_diff_integrate_expected_equivalent_to(case) is not None
    )
    direct_diff_integrate_gap_cases = len(
        derivative_verified_without_direct_diff_integrate_cases(cases)
    )
    expected_step_substrings = sum(
        len(case.expected_step_substrings) for case in cases
    )
    supported_step_unchecked_cases = [
        case.name
        for case in cases
        if case.outcome == "supported"
        and not case.expected_step_substrings
        and not is_algorithmic_backend_boundary_case(case)
    ]
    residual_case_names = [case.name for case in cases if case.outcome == "residual"]

    return {
        "status": overall_status,
        "total": len(results),
        "status_counts": status_counts,
        "issue_kind_counts": dict(sorted(issue_kind_counts.items())),
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
        "antiderivative_verification_case_count": antiderivative_verification_cases,
        "verified_supported_case_count": count_verified_supported_cases(cases),
        "direct_diff_integrate_case_count": direct_diff_integrate_cases,
        "direct_diff_integrate_exact_case_count": direct_diff_integrate_exact_cases,
        "direct_diff_integrate_equivalence_case_count": (
            direct_diff_integrate_equivalence_cases
        ),
        "direct_diff_integrate_gap_case_count": direct_diff_integrate_gap_cases,
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
        "argument_regime_counts": count_by(cases, "argument_regime"),
        "domain_regime_counts": count_by(cases, "domain_regime"),
        "outcome_counts": count_by(cases, "outcome"),
        "residual_cause_counts": count_residual_causes(cases),
        "residual_family_counts": count_residual_families(cases),
        "residual_cause_family_counts": count_residual_cause_families(cases),
        "residual_cases_by_cause": group_residual_cases_by_cause(cases),
        "verification_regime_counts": count_verification_regimes(cases),
        "calculus_maturity_block_counts": count_calculus_maturity_blocks(cases),
        "calculus_block_gate_counts": count_calculus_block_gates(cases),
        "trace_regime_counts": count_by(cases, "trace_regime"),
        "presentation_regime_counts": count_by(cases, "presentation_regime"),
        "trig_hyperbolic_policy_cluster_counts": (
            count_trig_hyperbolic_policy_clusters(cases)
        ),
        "base_integration_policy_cluster_counts": (
            count_base_integration_policy_clusters(cases)
        ),
        "radical_inverse_policy_cluster_counts": (
            count_radical_inverse_policy_clusters(cases)
        ),
        "direct_diff_integrate_calculus_maturity_block_counts": (
            count_direct_diff_integrate_calculus_maturity_blocks(cases)
        ),
        "direct_diff_integrate_calculus_block_gate_counts": (
            count_direct_diff_integrate_calculus_block_gates(cases)
        ),
        "direct_diff_integrate_trig_hyperbolic_policy_cluster_counts": (
            count_direct_diff_integrate_trig_hyperbolic_policy_clusters(cases)
        ),
        "direct_diff_integrate_base_integration_policy_cluster_counts": (
            count_direct_diff_integrate_base_integration_policy_clusters(cases)
        ),
        "direct_diff_integrate_radical_inverse_policy_cluster_counts": (
            count_direct_diff_integrate_radical_inverse_policy_clusters(cases)
        ),
        "direct_diff_integrate_gap_calculus_maturity_block_counts": (
            count_direct_diff_integrate_gap_calculus_maturity_blocks(cases)
        ),
        "direct_diff_integrate_gap_calculus_block_gate_counts": (
            count_direct_diff_integrate_gap_calculus_block_gates(cases)
        ),
        "direct_diff_integrate_gap_trig_hyperbolic_policy_cluster_counts": (
            count_direct_diff_integrate_gap_trig_hyperbolic_policy_clusters(cases)
        ),
        "direct_diff_integrate_gap_base_integration_policy_cluster_counts": (
            count_direct_diff_integrate_gap_base_integration_policy_clusters(cases)
        ),
        "direct_diff_integrate_gap_radical_inverse_policy_cluster_counts": (
            count_direct_diff_integrate_gap_radical_inverse_policy_clusters(cases)
        ),
        "direct_diff_integrate_gap_cases_by_calculus_maturity_block": (
            direct_diff_integrate_gap_case_names_by_calculus_maturity_block(cases)
        ),
        "direct_diff_integrate_gap_cases_by_calculus_block_gate": (
            direct_diff_integrate_gap_case_names_by_calculus_block_gate(cases)
        ),
        "direct_diff_integrate_gap_cases_by_trig_hyperbolic_policy_cluster": (
            direct_diff_integrate_gap_case_names_by_trig_hyperbolic_policy_cluster(
                cases
            )
        ),
        "direct_diff_integrate_gap_cases_by_base_integration_policy_cluster": (
            direct_diff_integrate_gap_case_names_by_base_integration_policy_cluster(
                cases
            )
        ),
        "direct_diff_integrate_gap_cases_by_radical_inverse_policy_cluster": (
            direct_diff_integrate_gap_case_names_by_radical_inverse_policy_cluster(
                cases
            )
        ),
        "direct_diff_integrate_equivalence_calculus_maturity_block_counts": (
            count_direct_diff_integrate_equivalence_calculus_maturity_blocks(cases)
        ),
        "direct_diff_integrate_equivalence_trig_hyperbolic_policy_cluster_counts": (
            count_direct_diff_integrate_equivalence_trig_hyperbolic_policy_clusters(
                cases
            )
        ),
        "direct_diff_integrate_equivalence_base_integration_policy_cluster_counts": (
            count_direct_diff_integrate_equivalence_base_integration_policy_clusters(
                cases
            )
        ),
        "case_filters": [case.name for case in cases],
        **runtime_observability_summary(
            results,
            group_keys=(
                "family",
                "calculus_maturity_block",
                "calculus_block_gate",
                "trace_regime",
            ),
        ),
        **phase_runtime_observability_summary(results),
        **payload_observability_summary(results),
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
    if args.summary_json:
        matrix["residual_shape_orientation_probes"] = (
            residual_shape_orientation_probe_rows(
                cas_cli=cas_cli,
                timeout_seconds=args.timeout_seconds,
            )
        )
    payload = summarize_matrix(matrix) if args.summary_json else matrix
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print_human(matrix)
    return 0 if matrix["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
