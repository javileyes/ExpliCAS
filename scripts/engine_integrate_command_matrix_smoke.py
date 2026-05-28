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
class IntegrateCommandMatrixCase:
    name: str
    expr: str
    expected_result: str
    expected_derivative_result: str | None = None
    expected_derivative_equivalent_to: str | None = None
    expected_derivative_required_display: tuple[str, ...] = ()
    expected_required_display: tuple[str, ...] = ()
    expected_warning_substrings: tuple[str, ...] = ()
    expected_step_substrings: tuple[str, ...] = ()
    family: str = "unknown"
    argument_regime: str = "variable"
    domain_regime: str = "unconditional"
    outcome: str = "supported"
    trace_regime: str = "direct"
    presentation_regime: str = "canonical"


DEFAULT_INTEGRATE_COMMAND_MATRIX_CASES = (
    IntegrateCommandMatrixCase(
        name="polynomial_power_direct",
        expr="integrate(x^2, x)",
        expected_result="1/3·x^3",
        expected_derivative_result="x^2",
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
        name="affine_exp_substitution",
        expr="integrate(exp(2*x+1), x)",
        expected_result="1/2·e^(2·x + 1)",
        expected_derivative_result="e^(2·x + 1)",
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
        name="polynomial_exp_derivative_substitution",
        expr="integrate(2*x*exp(x^2), x)",
        expected_result="e^(x^2)",
        expected_derivative_result="2·x·e^(x^2)",
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
        name="log_derivative_positive_quadratic_substitution",
        expr="integrate((2*x+2)/(x^2+2*x+2), x)",
        expected_result="ln(x^2 + 2·x + 2)",
        expected_derivative_result="(2·x + 2) / (x^2 + 2·x + 2)",
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
        trace_regime="residual_presimplification_with_domain",
        presentation_regime="residual_presentation_cleanup",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_reciprocal_sine_presimplified_residual_domain",
        expr="integrate(1/sin(x^2+0), x)",
        expected_result="integrate(csc(x^2), x)",
        expected_required_display=("sin(x^2 + 0) ≠ 0",),
        expected_step_substrings=(
            "Reconocer cosecante desde un recíproco",
            "Conservar integral residual",
        ),
        family="explicit_reciprocal_trig_residual_domain",
        argument_regime="explicit_reciprocal_presimplified_argument",
        domain_regime="explicit_denominator_source_condition",
        outcome="residual",
        trace_regime="residual_presimplification_with_domain",
        presentation_regime="result_cleanup_source_condition",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_reciprocal_tangent_presimplified_residual_domain",
        expr="integrate(1/tan(x^2+0), x)",
        expected_result="integrate(cot(x^2), x)",
        expected_required_display=("sin(x^2) ≠ 0", "tan(x^2 + 0) ≠ 0"),
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
        trace_regime="reciprocal_tangent_residual_prep",
        presentation_regime="compound_source_condition_residual",
    ),
    IntegrateCommandMatrixCase(
        name="explicit_reciprocal_tangent_verified_log_domain",
        expr="integrate(2*x/tan(x^2), x)",
        expected_result="ln(|sin(x^2)|)",
        expected_derivative_result="(cos(x^2)·x·2)/sin(x^2)",
        expected_derivative_required_display=("sin(x^2) ≠ 0",),
        expected_required_display=("sin(x^2) ≠ 0", "tan(x^2) ≠ 0"),
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
        name="additive_trig_pole_residual_domain",
        expr="integrate(tan(x^2)+sin(x^2), x)",
        expected_result="integrate(sin(x^2) + tan(x^2), x)",
        expected_required_display=("cos(x^2) ≠ 0",),
        expected_step_substrings=("Conservar integral residual",),
        family="trig_additive_residual_domain",
        argument_regime="additive_nonlinear_trig_residual",
        domain_regime="trig_pole_additive_residual",
        outcome="residual",
        trace_regime="residual_policy_with_domain",
        presentation_regime="residual",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_trig_table",
        expr="integrate(1/(x^2+1), x)",
        expected_result="arctan(x)",
        expected_derivative_result="1 / (x^2 + 1)",
        expected_step_substrings=("Usar la regla de arctan con derivada interna",),
        family="inverse_trig_table",
        argument_regime="rational_expression",
        trace_regime="inverse_trig_table",
        presentation_regime="inverse_trig",
    ),
    IntegrateCommandMatrixCase(
        name="rational_positive_quadratic_square_reduction",
        expr="integrate(1/(x^2+1)^2, x)",
        expected_result="1/2·arctan(x) + x / (2·(x^2 + 1))",
        expected_derivative_result="1 / (x^2 + 1)^2",
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
        name="inverse_hyperbolic_rational_direct_atanh_domain",
        expr="integrate(1/(1-x^2), x)",
        expected_result="atanh(x)",
        expected_derivative_result="1 / (1 - x^2)",
        expected_derivative_required_display=("-1 < x < 1",),
        expected_required_display=("-1 < x < 1",),
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
        expected_derivative_equivalent_to="1/(x*(x+1)^2)",
        expected_derivative_required_display=("x ≠ 0", "x ≠ -1"),
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
        expected_derivative_equivalent_to="1/(x^4-1)",
        expected_derivative_required_display=("x ≠ -1", "x ≠ 1"),
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
        expected_derivative_equivalent_to="1/((x-1)^2*(x^2+1))",
        expected_derivative_required_display=("x ≠ 1",),
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
        trace_regime="residual_presimplification_with_domain",
        presentation_regime="residual_reciprocal_sqrt_power",
    ),
    IntegrateCommandMatrixCase(
        name="inverse_sqrt_direct_asinh_unconditional",
        expr="integrate(1/sqrt(x^2+1), x)",
        expected_result="asinh(x)",
        expected_derivative_result="1 / sqrt(x^2 + 1)",
        expected_step_substrings=("Usar la regla de asinh con derivada interna",),
        family="inverse_hyperbolic_sqrt_table",
        argument_regime="unbounded_variable_radical",
        trace_regime="inverse_sqrt_hyperbolic_direct_table",
        presentation_regime="inverse_hyperbolic",
    ),
    IntegrateCommandMatrixCase(
        name="affine_inverse_sqrt_arcsin_domain",
        expr="integrate(1/sqrt(4-(x+1)^2), x)",
        expected_result="arcsin((x + 1) / 2)",
        expected_derivative_result="1 / sqrt(3 - x^2 - 2·x)",
        expected_derivative_required_display=("-3 < x < 1",),
        expected_required_display=("-3 < x < 1",),
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
        name="symbolic_affine_exact_hyperbolic_cosh_reciprocal_square_substitution",
        expr="integrate(a/cosh(a*x+b)^2, x)",
        expected_result="tanh(a·x + b)",
        expected_derivative_result="a / cosh(a·x + b)^2",
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
        expected_derivative_equivalent_to="2*k*x/sinh(x^2+b)^2",
        expected_derivative_required_display=("sinh(x^2 + b) ≠ 0",),
        expected_required_display=("sinh(x^2 + b) ≠ 0",),
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
        expected_derivative_equivalent_to="1/cosh(2*x+1)^4",
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
        expected_derivative_equivalent_to="2*k*x/cosh(x^2+b)^4",
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
        expected_derivative_equivalent_to="1/sinh(2*x+1)^4",
        expected_derivative_required_display=("sinh(2·x + 1) ≠ 0",),
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
        expected_derivative_equivalent_to="2*k*x/sinh(x^2+b)^4",
        expected_derivative_required_display=("sinh(x^2 + b) ≠ 0",),
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
        name="by_parts_affine_log_domain",
        expr="integrate((2*x+1)*ln(2*x+1), x)",
        expected_result="1/4·((2·x + 1)^2·ln(2·x + 1) - 2·x^2 - 2·x)",
        expected_derivative_result="ln(2·x + 1)·(2·x + 1)",
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
        name="symbolic_affine_exact_secant_tangent_derivative_product_domain",
        expr="integrate(a*sec(a*x+b)*tan(a*x+b), x)",
        expected_result="sec(a·x + b)",
        expected_derivative_equivalent_to="a*sec(a*x+b)*tan(a*x+b)",
        expected_derivative_required_display=("cos(a·x + b) ≠ 0",),
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
        expected_derivative_equivalent_to="k*sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x))",
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
        expected_derivative_equivalent_to="k*csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x))",
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
        expected_derivative_equivalent_to="-k*sec(b-sqrt(x))*tan(b-sqrt(x))/(2*sqrt(x))",
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
        expected_derivative_equivalent_to="-k*csc(b-sqrt(x))*cot(b-sqrt(x))/(2*sqrt(x))",
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
        expected_derivative_equivalent_to="k*sec(sqrt(x)-b)*tan(sqrt(x)-b)/(2*sqrt(x))",
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
        expected_derivative_equivalent_to="k*csc(sqrt(x)-b)*cot(sqrt(x)-b)/(2*sqrt(x))",
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
        expected_derivative_equivalent_to="tan(b-sqrt(x))/(2*sqrt(x))",
        expected_derivative_required_display=("x > 0", "cos(b - sqrt(x)) ≠ 0"),
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
        expected_result="-ln(|sinh(x^(1/2) - b)|)",
        expected_derivative_equivalent_to="1/(2*sqrt(x)*tanh(b-sqrt(x)))",
        expected_derivative_required_display=("sinh(b - sqrt(x)) ≠ 0", "x > 0"),
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
        expected_result="-k / cosh(x^(1/2) - b)",
        expected_derivative_equivalent_to=(
            "k*sinh(sqrt(x)-b)/(2*sqrt(x)*cosh(sqrt(x)-b)^2)"
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
        expected_result="-k / sinh(x^(1/2) - b)",
        expected_derivative_equivalent_to=(
            "k*cosh(b-sqrt(x))/(2*sqrt(x)*sinh(b-sqrt(x))^2)"
        ),
        expected_derivative_required_display=("x > 0", "sinh(b - sqrt(x)) ≠ 0"),
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

    derivative_result: str | None = None
    derivative_required_display: tuple[str, ...] = ()
    derivative_stderr = ""
    derivative_equivalence_result: str | None = None
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
            error = "antiderivative verification timeout"
        else:
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

    wall_elapsed = time.monotonic() - start
    status: Status = "pass" if error is None else "fail"
    error_kind = classify_error_kind(error)
    if error_kind == "antiderivative_verification_timeout":
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
        "result": result if isinstance(result, str) else None,
        "required_display": list(required_display),
        "warnings": list(warnings),
        "expected_result": case.expected_result,
        "expected_derivative_result": case.expected_derivative_result,
        "expected_derivative_equivalent_to": case.expected_derivative_equivalent_to,
        "derivative_result": derivative_result,
        "derivative_equivalence_result": derivative_equivalence_result,
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
        "trace_regime": case.trace_regime,
        "presentation_regime": case.presentation_regime,
        "calculus_maturity_block": calculus_maturity_block(case),
        "calculus_block_gate": calculus_block_gate(case),
        "stderr": stderr,
        "derivative_stderr": derivative_stderr,
    }


def build_cases(
    case_filters: tuple[str, ...] = (),
) -> tuple[IntegrateCommandMatrixCase, ...]:
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
        return "block7_explicit_reciprocal_trig_tangent"
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


def calculus_maturity_block(case: IntegrateCommandMatrixCase) -> str:
    if case.outcome in {"residual", "undefined"}:
        return "block9_residuals_and_non_goals"

    cluster = trig_hyperbolic_policy_cluster(case)
    if cluster is not None and cluster.startswith("block7_"):
        return "block7_trig_hyperbolic_integration"

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


def calculus_block_gate(case: IntegrateCommandMatrixCase) -> str:
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
    if (
        case.expected_derivative_result is not None
        or case.expected_derivative_equivalent_to is not None
    ):
        return "verified_by_diff"
    if case.outcome == "residual":
        return "residual_not_verified"
    if case.outcome == "undefined":
        return "undefined_not_verified"
    return "verification_gap"


def count_verification_regimes(
    cases: tuple[IntegrateCommandMatrixCase, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        regime = verification_regime(case)
        counts[regime] = counts.get(regime, 0) + 1
    return dict(sorted(counts.items()))


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
    expected_step_substrings = sum(
        len(case.expected_step_substrings) for case in cases
    )
    supported_step_unchecked_cases = [
        case.name
        for case in cases
        if case.outcome == "supported" and not case.expected_step_substrings
    ]

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
        "warning_expected_case_count": warning_expected_cases,
        "required_display_case_count": required_display_cases,
        "step_checked_case_count": step_checked_cases,
        "supported_step_unchecked_case_count": len(supported_step_unchecked_cases),
        "supported_step_unchecked_cases": supported_step_unchecked_cases,
        "antiderivative_verification_case_count": antiderivative_verification_cases,
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
        "verification_regime_counts": count_verification_regimes(cases),
        "calculus_maturity_block_counts": count_calculus_maturity_blocks(cases),
        "calculus_block_gate_counts": count_calculus_block_gates(cases),
        "trace_regime_counts": count_by(cases, "trace_regime"),
        "presentation_regime_counts": count_by(cases, "presentation_regime"),
        "trig_hyperbolic_policy_cluster_counts": (
            count_trig_hyperbolic_policy_clusters(cases)
        ),
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
