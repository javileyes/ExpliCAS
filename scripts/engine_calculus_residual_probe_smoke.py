#!/usr/bin/env python3
"""Smoke-test one calculus residual expression through the public CLI.

This is a discovery harness, not a promotion lane. It lets an improvement
iteration classify a calculus-shaped residual wrapper as pass/fail/timeout/slow
before deciding whether the engine needs a retained rule or only a corpus case.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CAS_CLI = ROOT / "target" / "release" / "cas_cli"

Status = Literal["pass", "fail", "timeout", "slow"]
IssueKind = Literal[
    "nonzero_exit",
    "invalid_json",
    "non_object_json",
    "result_mismatch",
    "missing_required_conditions",
    "impossible_required_conditions",
    "unexpected_warnings",
    "no_matching_cases",
    "timeout",
    "slow",
]


@dataclass(frozen=True)
class ProbeResult:
    status: Status
    returncode: int | None
    wall_elapsed_seconds: float
    result: str | None
    required_conditions: tuple[str, ...]
    impossible_required_conditions: tuple[str, ...]
    warnings: tuple[str, ...]
    stdout: str
    stderr: str
    slow_wall_seconds: float | None = None
    error: str | None = None
    error_kind: IssueKind | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "returncode": self.returncode,
            "wall_elapsed_seconds": round(self.wall_elapsed_seconds, 3),
            "slow_wall_seconds": self.slow_wall_seconds,
            "result": self.result,
            "required_conditions": list(self.required_conditions),
            "impossible_required_conditions": list(self.impossible_required_conditions),
            "warnings": list(self.warnings),
            "error": self.error,
            "error_kind": self.error_kind,
        }


@dataclass(frozen=True)
class MatrixProbeCase:
    name: str
    expr: str
    expected_result: str
    required_conditions: tuple[str, ...]


DEFAULT_MATRIX_WRAPPERS = (
    ("plus", "(({residual})+1)/(x+2)", "1 / (x + 2)", ("x + 2",)),
    ("one_minus", "(1-({residual}))/(x+2)", "1 / (x + 2)", ("x + 2",)),
    (
        "scaled_negative",
        "(-1)*((({residual})+1)/(x+2))",
        "-1 / (x + 2)",
        ("x + 2",),
    ),
    ("minus_const", "(({residual})-1)/(x+2)", "-1 / (x + 2)", ("x + 2",)),
    (
        "plus_noise",
        "((({residual})+1)/(x+2))+(x-x)",
        "1 / (x + 2)",
        ("x + 2",),
    ),
    (
        "plus_double_noise",
        "((((({residual})+1)/(x+2))+(x-x))+(y-y))",
        "1 / (x + 2)",
        ("x + 2",),
    ),
    (
        "plus_triple_noise",
        "(((((({residual})+1)/(x+2))+(x-x))+(y-y))+(z-z))",
        "1 / (x + 2)",
        ("x + 2",),
    ),
    (
        "plus_triple_noise_times_one",
        "(((((({residual})+1)/(x+2))+(x-x))+(y-y))+(z-z))*1",
        "1 / (x + 2)",
        ("x + 2",),
    ),
    (
        "product_den",
        "(({residual})+1)/((x+2)*(x+3))",
        "1 / ((x + 2)·(x + 3))",
        ("x + 2", "x + 3"),
    ),
    (
        "product_den_triple_noise_times_one",
        "(((((({residual})+1)/((x+2)*(x+3)))+(x-x))+(y-y))+(z-z))*1",
        "1 / ((x + 2)·(x + 3))",
        ("x + 2", "x + 3"),
    ),
    (
        "nested_den",
        "((({residual})+1)/(x+2))/(x+3)",
        None,
        ("x + 2", "x + 3"),
    ),
)

NESTED_DEN_EXPECTED_RESULTS = {
    "exp_poly": "1 / ((x + 2)·(x + 3))",
    "exp_affine": "1 / ((x + 2)·(x + 3))",
    "arctan_sqrt_additive_trig": "1 / ((x + 2)·(x + 3))",
    "integrate_exp_trig_sin": "1 / ((x + 2)·(x + 3))",
    "integrate_exp_trig_cos": "1 / ((x + 2)·(x + 3))",
    "integrate_exp_trig_neg_sin": "1 / ((x + 2)·(x + 3))",
    "integrate_exp_trig_neg_cos": "1 / ((x + 2)·(x + 3))",
    "plain_trig_sin": "1 / ((x + 2)·(x + 3))",
    "plain_trig_cos": "1 / ((x + 2)·(x + 3))",
    "plain_trig_cot_fourth": "1 / ((x + 2)·(x + 3))",
    "plain_trig_sec_fourth": "1 / ((x + 2)·(x + 3))",
    "plain_trig_neg_sin": "1 / ((x + 2)·(x + 3))",
    "plain_trig_neg_cos": "1 / ((x + 2)·(x + 3))",
    "plain_trig_sparse_neg_sin": "1 / ((x + 2)·(x + 3))",
    "hyperbolic_sinh": "1 / ((x + 2)·(x + 3))",
    "hyperbolic_cosh": "1 / ((x + 2)·(x + 3))",
    "inverse_trig_arctan": "1 / ((x + 2)·(x + 3))",
    "rational_quad": "1 / ((x + 2)·(x + 3))",
    "recip_trig": "1 / ((x + 2)·(x + 3))",
    "fractional_den_power": "1 / ((x + 2)·(x + 3))",
    "quartic_arcsin_kernel": "1 / ((x + 2)·(x + 3))",
    "constant_base_log_power": "1 / ((x + 2)·(x + 3))",
    "log10_power_alias": "1 / ((x + 2)·(x + 3))",
    "reciprocal_trig_csc": "1 / ((x + 2)·(x + 3))",
    "reciprocal_trig_sec": "1 / ((x + 2)·(x + 3))",
    "sqrt_chain_csc_log": "1 / ((x + 2)·(x + 3))",
    "sqrt_chain_cosh_recip_square": "1 / ((x + 2)·(x + 3))",
    "sqrt_chain_sec_log": "1 / ((x + 2)·(x + 3))",
    "sqrt_chain_sinh_recip_square": "1 / ((x + 2)·(x + 3))",
}

CUSTOM_RESIDUAL_EXPECTED_RESULTS_BY_WRAPPER = {
    "nested_den": "1 / ((x + 2)·(x + 3))",
}

CUSTOM_RESIDUAL_MATRIX_WRAPPERS = (
    *DEFAULT_MATRIX_WRAPPERS,
    (
        "double_nested_den",
        "((((({residual})+1)/(x+2))/(x+3))/(x+4))",
        "1 / ((x + 2)·(x + 3)·(x + 4))",
        ("x + 2", "x + 3", "x + 4"),
    ),
)

DEFAULT_MATRIX_BASES = (
    ("exp_poly", "diff(integrate(x^5*exp(x),x),x)-x^5*exp(x)", ()),
    ("exp_affine", "diff(integrate(x^4*exp(2*x+1),x),x)-x^4*exp(2*x+1)", ()),
    (
        "arctan_sqrt_additive_trig",
        (
            "diff(arctan(sqrt(sin(2*x)+cos(x)+4)),x)"
            "-(cos(2*x)-sin(x)/2)/(sqrt(sin(2*x)+cos(x)+4)*(sin(2*x)+cos(x)+5))"
        ),
        (),
    ),
    (
        "integrate_exp_trig_sin",
        "diff(integrate(exp(2*x+1)*sin(2*x+1),x),x)-exp(2*x+1)*sin(2*x+1)",
        (),
    ),
    (
        "integrate_exp_trig_cos",
        "diff(integrate(exp(2*x+1)*cos(2*x+1),x),x)-exp(2*x+1)*cos(2*x+1)",
        (),
    ),
    (
        "integrate_exp_trig_neg_sin",
        "diff(integrate(exp(1-2*x)*sin(1-2*x),x),x)-exp(1-2*x)*sin(1-2*x)",
        (),
    ),
    (
        "integrate_exp_trig_neg_cos",
        "diff(integrate(exp(1-2*x)*cos(1-2*x),x),x)-exp(1-2*x)*cos(1-2*x)",
        (),
    ),
    (
        "plain_trig_sin",
        "diff(integrate(x^4*sin(2*x+1),x),x)-x^4*sin(2*x+1)",
        (),
    ),
    ("plain_trig_cos", "diff(integrate(x^3*cos(x),x),x)-x^3*cos(x)", ()),
    (
        "plain_trig_cot_fourth",
        "diff(integrate(cot(2*x+1)^4,x),x)-cot(2*x+1)^4",
        ("sin(2·x + 1)",),
    ),
    (
        "plain_trig_sec_fourth",
        "diff(integrate(sec(2*x+1)^4,x),x)-sec(2*x+1)^4",
        ("cos(2·x + 1)",),
    ),
    (
        "plain_trig_neg_sin",
        "diff(integrate(x^4*sin(1-2*x),x),x)-x^4*sin(1-2*x)",
        (),
    ),
    (
        "plain_trig_neg_cos",
        "diff(integrate(x^3*cos(1-2*x),x),x)-x^3*cos(1-2*x)",
        (),
    ),
    (
        "plain_trig_sparse_neg_sin",
        "diff(integrate((x^6+1)*sin(1-2*x),x),x)-(x^6+1)*sin(1-2*x)",
        (),
    ),
    (
        "hyperbolic_sinh",
        "diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1)",
        (),
    ),
    (
        "hyperbolic_cosh",
        "diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1)",
        (),
    ),
    (
        "inverse_trig_arctan",
        "diff(integrate(x^6*arctan(x),x),x)-x^6*arctan(x)",
        (),
    ),
    (
        "rational_quad",
        (
            "diff(integrate(1/((x+1)^3*(x^2+2*x+2)),x),x)"
            "-1/((x+1)^3*(x^2+2*x+2))"
        ),
        ("x + 1",),
    ),
    ("recip_trig", "diff(integrate(1/(x^2+1),x),x)-1/(x^2+1)", ()),
    (
        "fractional_den_power",
        (
            "diff(integrate((2*x+1)/(x^2+x+1)^(3/2),x),x)"
            "-(2*x+1)/(x^2+x+1)^(3/2)"
        ),
        (),
    ),
    (
        "quartic_arcsin_kernel",
        "diff(integrate(2*x/sqrt(4-x^4),x),x)-2*x/sqrt(4-x^4)",
        ("4 - x^4",),
    ),
    (
        "constant_base_log_power",
        "diff(integrate(2*x*log(2,x^2+1)^2,x),x)-2*x*log(2,x^2+1)^2",
        (),
    ),
    (
        "log10_power_alias",
        "diff(integrate(2*x*log10(x^2+1)^2,x),x)-2*x*log10(x^2+1)^2",
        (),
    ),
    (
        "reciprocal_trig_csc",
        "diff(integrate(csc(2*x+1),x),x)-csc(2*x+1)",
        ("sin(2·x + 1)",),
    ),
    (
        "reciprocal_trig_sec",
        "diff(integrate(sec(2*x+1),x),x)-sec(2*x+1)",
        ("cos(2·x + 1)",),
    ),
    (
        "sqrt_chain_sec_log",
        (
            "diff(integrate(3/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))),x),x)"
            "-3/(2*sqrt(3*x+1)*cos(sqrt(3*x+1)))"
        ),
        ("cos(sqrt(3·x + 1))", "x > -1/3"),
    ),
    (
        "sqrt_chain_csc_log",
        (
            "diff(integrate(3/(2*sqrt(3*x+1)*sin(sqrt(3*x+1))),x),x)"
            "-3/(2*sqrt(3*x+1)*sin(sqrt(3*x+1)))"
        ),
        ("sin(sqrt(3·x + 1))", "x > -1/3"),
    ),
    (
        "sqrt_chain_cosh_recip_square",
        (
            "diff(integrate(3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2),x),x)"
            "-3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2)"
        ),
        ("x > -1/3",),
    ),
    (
        "sqrt_chain_sinh_recip_square",
        (
            "diff(integrate(3/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2),x),x)"
            "-3/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2)"
        ),
        ("sinh(sqrt(3·x + 1))", "x > -1/3"),
    ),
)

DEFAULT_DOUBLE_NESTED_DEN_BASES = (
    "exp_poly",
    "arctan_sqrt_additive_trig",
    "inverse_trig_arctan",
    "fractional_den_power",
    "hyperbolic_sinh",
    "hyperbolic_cosh",
    "quartic_arcsin_kernel",
    "sqrt_chain_sec_log",
    "sqrt_chain_csc_log",
    "sqrt_chain_sinh_recip_square",
    "integrate_exp_trig_sin",
    "integrate_exp_trig_neg_sin",
    "rational_quad",
    "plain_trig_sin",
    "plain_trig_cot_fourth",
    "plain_trig_sec_fourth",
)

DEFAULT_SHIFTED_QUOTIENT_RESIDUAL_DEN_CASES = (
    MatrixProbeCase(
        name="hyperbolic_sinh_over_cosh_shifted_quotient",
        expr=(
            "((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)"
            "/((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))+1)"
        ),
        expected_result="1",
        required_conditions=(),
    ),
    MatrixProbeCase(
        name="hyperbolic_sinh_over_cosh_negative_shifted_quotient",
        expr=(
            "((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)"
            "/((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))-1)"
        ),
        expected_result="-1",
        required_conditions=(),
    ),
    MatrixProbeCase(
        name="hyperbolic_sinh_over_scaled_cosh_shifted_quotient",
        expr=(
            "((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)"
            "/(2*((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))+1))"
        ),
        expected_result="1/2",
        required_conditions=(),
    ),
    MatrixProbeCase(
        name="scaled_hyperbolic_sinh_over_scaled_cosh_shifted_quotient",
        expr=(
            "3*((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)"
            "/(2*((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))+1))"
        ),
        expected_result="3/2",
        required_conditions=(),
    ),
    MatrixProbeCase(
        name="scaled_hyperbolic_sinh_over_cosh_product_denominator_shifted_quotient",
        expr=(
            "3*((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)"
            "/(((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))+1)*(x+2))"
        ),
        expected_result="3 / (x + 2)",
        required_conditions=("x + 2",),
    ),
    MatrixProbeCase(
        name="scaled_hyperbolic_sinh_over_negative_cosh_product_denominator_shifted_quotient",
        expr=(
            "3*((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)"
            "/((x+2)*((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))-1)*(x+3))"
        ),
        expected_result="-3 / ((x + 2)·(x + 3))",
        required_conditions=("x + 2", "x + 3"),
    ),
    MatrixProbeCase(
        name="scaled_hyperbolic_sinh_over_negative_cosh_fraction_denominator_shifted_quotient",
        expr=(
            "3*((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)"
            "/(((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))-1)/(x+2))"
        ),
        expected_result="-3·(x + 2)",
        required_conditions=("x + 2",),
    ),
    MatrixProbeCase(
        name="scaled_hyperbolic_sinh_over_scaled_negative_cosh_fraction_denominator_shifted_quotient",
        expr=(
            "3*((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)"
            "/(2*((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1)-1)/(x+2)))"
        ),
        expected_result="-3/2·(x + 2)",
        required_conditions=("x + 2",),
    ),
    MatrixProbeCase(
        name="scaled_hyperbolic_sinh_over_negative_cosh_fraction_denominator_positive_quadratic_shifted_quotient",
        expr=(
            "3*((diff(integrate(x^5*sinh(2*x+1),x),x)-x^5*sinh(2*x+1))+1)"
            "/(((diff(integrate(x^4*cosh(2*x+1),x),x)-x^4*cosh(2*x+1))-1)/(x^2+1))"
        ),
        expected_result="-3·(x^2 + 1)",
        required_conditions=(),
    ),
    MatrixProbeCase(
        name="rational_quad_over_recip_trig_shifted_quotient",
        expr=(
            "((diff(integrate(1/((x+1)^3*(x^2+2*x+2)),x),x)"
            "-1/((x+1)^3*(x^2+2*x+2)))+1)"
            "/((diff(integrate(1/(x^2+1),x),x)-1/(x^2+1))+1)"
        ),
        expected_result="1",
        required_conditions=("x + 1",),
    ),
    MatrixProbeCase(
        name="rational_quad_over_negative_recip_trig_shifted_quotient",
        expr=(
            "((diff(integrate(1/((x+1)^3*(x^2+2*x+2)),x),x)"
            "-1/((x+1)^3*(x^2+2*x+2)))+1)"
            "/((diff(integrate(1/(x^2+1),x),x)-1/(x^2+1))-1)"
        ),
        expected_result="-1",
        required_conditions=("x + 1",),
    ),
    MatrixProbeCase(
        name="quartic_arcsin_over_reciprocal_trig_csc_shifted_quotient",
        expr=(
            "((diff(integrate(2*x/sqrt(4-x^4),x),x)-2*x/sqrt(4-x^4))+1)"
            "/((diff(integrate(csc(2*x+1),x),x)-csc(2*x+1))+1)"
        ),
        expected_result="1",
        required_conditions=(
            "4 - x^4",
            "sin(2·x + 1)",
        ),
    ),
    MatrixProbeCase(
        name="quartic_arcsin_over_negative_reciprocal_trig_csc_shifted_quotient",
        expr=(
            "((diff(integrate(2*x/sqrt(4-x^4),x),x)-2*x/sqrt(4-x^4))+1)"
            "/((diff(integrate(csc(2*x+1),x),x)-csc(2*x+1))-1)"
        ),
        expected_result="-1",
        required_conditions=(
            "4 - x^4",
            "sin(2·x + 1)",
        ),
    ),
    MatrixProbeCase(
        name="sqrt_chain_sec_over_csc_shifted_quotient",
        expr=(
            "((diff(integrate(3/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))),x),x)"
            "-3/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))))+1)"
            "/((diff(integrate(3/(2*sqrt(3*x+1)*sin(sqrt(3*x+1))),x),x)"
            "-3/(2*sqrt(3*x+1)*sin(sqrt(3*x+1))))+1)"
        ),
        expected_result="1",
        required_conditions=(
            "cos(sqrt(3·x + 1))",
            "sin(sqrt(3·x + 1))",
            "x > -1/3",
        ),
    ),
    MatrixProbeCase(
        name="sqrt_chain_sec_over_negative_csc_shifted_quotient",
        expr=(
            "((diff(integrate(3/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))),x),x)"
            "-3/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))))+1)"
            "/((diff(integrate(3/(2*sqrt(3*x+1)*sin(sqrt(3*x+1))),x),x)"
            "-3/(2*sqrt(3*x+1)*sin(sqrt(3*x+1))))-1)"
        ),
        expected_result="-1",
        required_conditions=(
            "cos(sqrt(3·x + 1))",
            "sin(sqrt(3·x + 1))",
            "x > -1/3",
        ),
    ),
    MatrixProbeCase(
        name="sqrt_chain_cosh_over_sinh_shifted_quotient",
        expr=(
            "((diff(integrate(3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2),x),x)"
            "-3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2))+1)"
            "/((diff(integrate(3/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2),x),x)"
            "-3/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2))+1)"
        ),
        expected_result="1",
        required_conditions=(
            "sinh(sqrt(3·x + 1))",
            "x > -1/3",
        ),
    ),
    MatrixProbeCase(
        name="sqrt_chain_cosh_over_negative_sinh_shifted_quotient",
        expr=(
            "((diff(integrate(3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2),x),x)"
            "-3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2))+1)"
            "/((diff(integrate(3/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2),x),x)"
            "-3/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2))-1)"
        ),
        expected_result="-1",
        required_conditions=(
            "sinh(sqrt(3·x + 1))",
            "x > -1/3",
        ),
    ),
)

DEFAULT_RATIONAL_QUADRATIC_NONCANONICAL_CASES = (
    MatrixProbeCase(
        name="rational_quad_positive_quadratic:double_nested_den",
        expr=(
            "(((((diff(integrate((3*x + 5)/(x^2+x+1),x),x)"
            "-((3*x + 5)/(x^2+x+1)))+1)/(x+2))/(x+3))/(x+4))"
        ),
        expected_result="1 / ((x + 2)·(x + 3)·(x + 4))",
        required_conditions=("x + 2", "x + 3", "x + 4"),
    ),
)

DEFAULT_PRODUCT_ZERO_FACTOR_CASES = (
    MatrixProbeCase(
        name="recip_trig_csc_product_zero_factor",
        expr=(
            "((diff(integrate(csc(2*x+1),x),x)-csc(2*x+1))+x+2)*(y-y)"
        ),
        expected_result="0",
        required_conditions=("sin(2·x + 1)",),
    ),
    MatrixProbeCase(
        name="recip_trig_csc_product_zero_factor_reversed",
        expr=(
            "(y-y)*((diff(integrate(csc(2*x+1),x),x)-csc(2*x+1))+x+2)"
        ),
        expected_result="0",
        required_conditions=("sin(2·x + 1)",),
    ),
    MatrixProbeCase(
        name="plain_trig_by_parts_product_zero_factor",
        expr=(
            "((diff(integrate(x^6*sin(x),x),x)-x^6*sin(x))+x+2)*(y-y)"
        ),
        expected_result="0",
        required_conditions=(),
    ),
    MatrixProbeCase(
        name="hyperbolic_by_parts_product_zero_factor",
        expr=(
            "((diff(integrate((x^3+x)*cosh(2*x+1),x),x)"
            "-((x^3+x)*cosh(2*x+1)))+x+2)*(y-y)"
        ),
        expected_result="0",
        required_conditions=(),
    ),
)


def matrix_case_matches_filters(
    case: MatrixProbeCase,
    base_filters: tuple[str, ...] = (),
    wrapper_filters: tuple[str, ...] = (),
) -> bool:
    base_name, _, wrapper_name = case.name.partition(":")
    if base_filters and case.name not in base_filters and base_name not in base_filters:
        return False
    if wrapper_filters and wrapper_name not in wrapper_filters and case.name not in wrapper_filters:
        return False
    return True


def build_default_matrix_cases(
    base_filters: tuple[str, ...] = (),
    wrapper_filters: tuple[str, ...] = (),
) -> tuple[MatrixProbeCase, ...]:
    cases: list[MatrixProbeCase] = []
    base_by_family = {
        family: (residual, extra_required)
        for family, residual, extra_required in DEFAULT_MATRIX_BASES
    }
    for family, residual, extra_required in DEFAULT_MATRIX_BASES:
        for wrapper_name, template, expected_result, wrapper_required in DEFAULT_MATRIX_WRAPPERS:
            if expected_result is None:
                expected_result = NESTED_DEN_EXPECTED_RESULTS[family]
            cases.append(
                MatrixProbeCase(
                    name=f"{family}:{wrapper_name}",
                    expr=template.format(residual=residual),
                    expected_result=expected_result,
                    required_conditions=(*extra_required, *wrapper_required),
                )
            )
    for family in DEFAULT_DOUBLE_NESTED_DEN_BASES:
        residual, extra_required = base_by_family[family]
        cases.append(
            MatrixProbeCase(
                name=f"{family}:double_nested_den",
                expr=f"((((({residual})+1)/(x+2))/(x+3))/(x+4))",
                expected_result="1 / ((x + 2)·(x + 3)·(x + 4))",
                required_conditions=(*extra_required, "x + 2", "x + 3", "x + 4"),
            )
        )
    cases.extend(DEFAULT_SHIFTED_QUOTIENT_RESIDUAL_DEN_CASES)
    cases.extend(DEFAULT_RATIONAL_QUADRATIC_NONCANONICAL_CASES)
    cases.extend(DEFAULT_PRODUCT_ZERO_FACTOR_CASES)
    if base_filters or wrapper_filters:
        cases = [
            case
            for case in cases
            if matrix_case_matches_filters(case, base_filters, wrapper_filters)
        ]
    return tuple(cases)


def build_custom_residual_matrix_cases(
    residual_name: str,
    residual: str,
    required_conditions: tuple[str, ...] = (),
    wrapper_filters: tuple[str, ...] = (),
    include_wrapper_required: bool = True,
) -> tuple[MatrixProbeCase, ...]:
    cases: list[MatrixProbeCase] = []
    for wrapper_name, template, expected_result, wrapper_required in CUSTOM_RESIDUAL_MATRIX_WRAPPERS:
        case_required = required_conditions
        if include_wrapper_required:
            case_required = (*required_conditions, *wrapper_required)
        case_expected_result = expected_result
        if case_expected_result is None:
            case_expected_result = CUSTOM_RESIDUAL_EXPECTED_RESULTS_BY_WRAPPER.get(
                wrapper_name
            )
        case = MatrixProbeCase(
            name=f"{residual_name}:{wrapper_name}",
            expr=template.format(residual=residual),
            expected_result=case_expected_result,
            required_conditions=case_required,
        )
        if matrix_case_matches_filters(case, wrapper_filters=wrapper_filters):
            cases.append(case)
    return tuple(cases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one calculus residual expression with eval --format json, "
            "a process-group timeout, and optional result/condition checks."
        )
    )
    parser.add_argument("--expr", default=None, help="Expression passed to cas_cli eval.")
    parser.add_argument(
        "--default-matrix",
        action="store_true",
        help="Run the built-in representative calculus residual wrapper matrix.",
    )
    parser.add_argument(
        "--matrix-residual",
        default=None,
        help=(
            "Discovery helper: wrap this residual expression with the standard "
            "zero-residual matrix wrappers. Uses --require as extra required "
            "conditions and --matrix-wrapper for filtering."
        ),
    )
    parser.add_argument(
        "--matrix-residual-name",
        default="custom_residual",
        help="Base name used for cases generated by --matrix-residual.",
    )
    parser.add_argument(
        "--matrix-suppress-wrapper-requires",
        action="store_true",
        help=(
            "When used with --matrix-residual, do not require the standard "
            "wrapper denominator conditions. Use only when stronger base "
            "conditions are expected to imply those denominators."
        ),
    )
    parser.add_argument(
        "--matrix-base",
        action="append",
        default=[],
        help=(
            "When used with --default-matrix, run only cases with this base "
            "name before ':' or this exact standalone case name. Can be repeated."
        ),
    )
    parser.add_argument(
        "--matrix-wrapper",
        action="append",
        default=[],
        help=(
            "When used with --default-matrix, run only cases with this wrapper "
            "name after ':' or this exact standalone case name. Can be repeated."
        ),
    )
    parser.add_argument(
        "--expect-result",
        default=None,
        help="Exact expected public result string. Omit for discovery-only status.",
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="Required condition display that must be present. Can be repeated.",
    )
    parser.add_argument(
        "--forbid-warnings",
        action="store_true",
        help="Fail if the JSON response or stderr contains warnings.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=8.0,
        help="Hard wall timeout for the CLI subprocess.",
    )
    parser.add_argument(
        "--slow-wall-seconds",
        type=float,
        default=None,
        help="Classify an otherwise passing probe as slow above this wall budget.",
    )
    parser.add_argument(
        "--expect",
        choices=("pass", "fail", "timeout", "slow", "any"),
        default="pass",
        help="Expected smoke status. Non-matching status exits nonzero.",
    )
    parser.add_argument(
        "--cas-cli",
        default=str(DEFAULT_CAS_CLI),
        help="Path to cas_cli. Defaults to target/release/cas_cli.",
    )
    parser.add_argument(
        "--ensure-release-cas-cli",
        action="store_true",
        help=(
            "Build target/release/cas_cli if the default release binary is "
            "missing. Ignored when --cas-cli points at a custom path."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable summary JSON.",
    )
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help=(
            "When emitting a matrix with --json, omit passing case payloads and "
            "retain only aggregate counts plus non-pass problem cases."
        ),
    )
    return parser.parse_args()


def run_probe(
    expr: str,
    timeout_seconds: float,
    cas_cli: str | pathlib.Path = DEFAULT_CAS_CLI,
    expected_result: str | None = None,
    required_conditions: tuple[str, ...] = (),
    forbid_warnings: bool = False,
    slow_wall_seconds: float | None = None,
) -> ProbeResult:
    command = [str(cas_cli), "eval", expr, "--format", "json"]
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
        return ProbeResult(
            status="timeout",
            returncode=None,
            wall_elapsed_seconds=time.monotonic() - start,
            result=None,
            required_conditions=(),
            impossible_required_conditions=(),
            warnings=(),
            stdout=stdout,
            stderr=stderr,
            slow_wall_seconds=slow_wall_seconds,
            error="timeout",
            error_kind="timeout",
        )

    wall_elapsed = time.monotonic() - start
    parsed, parse_error = parse_json(stdout)
    result = extract_result(parsed)
    actual_required = extract_required_conditions(parsed)
    impossible_required = extract_impossible_required_conditions(parsed)
    warnings = (*extract_warnings(parsed), *extract_stderr_warnings(stderr))

    error = classify_error(
        returncode=process.returncode,
        parse_error=parse_error,
        result=result,
        expected_result=expected_result,
        actual_required=actual_required,
        expected_required=required_conditions,
        impossible_required=impossible_required,
        warnings=warnings,
        forbid_warnings=forbid_warnings,
    )
    status: Status = "pass" if error is None else "fail"
    error_kind = classify_error_kind(error)
    if status == "pass" and slow_wall_seconds is not None and wall_elapsed > slow_wall_seconds:
        status = "slow"
        error_kind = "slow"

    return ProbeResult(
        status=status,
        returncode=process.returncode,
        wall_elapsed_seconds=wall_elapsed,
        result=result,
        required_conditions=actual_required,
        impossible_required_conditions=impossible_required,
        warnings=warnings,
        stdout=stdout,
        stderr=stderr,
        slow_wall_seconds=slow_wall_seconds,
        error=error,
        error_kind=error_kind,
    )


def ensure_release_cas_cli(cas_cli: str | pathlib.Path) -> None:
    path = pathlib.Path(cas_cli)
    if path != DEFAULT_CAS_CLI or path.exists():
        return
    try:
        subprocess.run(
            ["cargo", "build", "--release", "-q", "-p", "cas_cli"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stdout or "")
        sys.stderr.write(exc.stderr or "")
        raise


def run_default_matrix(
    timeout_seconds: float,
    cas_cli: str | pathlib.Path = DEFAULT_CAS_CLI,
    forbid_warnings: bool = True,
    slow_wall_seconds: float | None = None,
    base_filters: tuple[str, ...] = (),
    wrapper_filters: tuple[str, ...] = (),
) -> dict[str, object]:
    results = []
    status_counts = {"pass": 0, "slow": 0, "fail": 0, "timeout": 0}
    issue_kind_counts: dict[str, int] = {}
    for case in build_default_matrix_cases(base_filters, wrapper_filters):
        result = run_probe(
            case.expr,
            timeout_seconds,
            cas_cli=cas_cli,
            expected_result=case.expected_result,
            required_conditions=case.required_conditions,
            forbid_warnings=forbid_warnings,
            slow_wall_seconds=slow_wall_seconds,
        )
        status_counts[result.status] += 1
        result_dict = {
            "name": case.name,
            "expected_required_conditions": list(case.required_conditions),
            **result.as_dict(),
        }
        increment_issue_kind(issue_kind_counts, result.error_kind)
        results.append(result_dict)

    overall_status: Status = "pass"
    if not results:
        overall_status = "fail"
        increment_issue_kind(issue_kind_counts, "no_matching_cases")
    elif status_counts["timeout"]:
        overall_status = "timeout"
    elif status_counts["fail"]:
        overall_status = "fail"
    elif status_counts["slow"]:
        overall_status = "slow"

    return {
        "status": overall_status,
        "total": len(results),
        "status_counts": status_counts,
        "issue_kind_counts": issue_kind_counts,
        "base_filters": list(base_filters),
        "wrapper_filters": list(wrapper_filters),
        "cases": results,
    }


def run_matrix_cases(
    cases: tuple[MatrixProbeCase, ...],
    timeout_seconds: float,
    cas_cli: str | pathlib.Path = DEFAULT_CAS_CLI,
    forbid_warnings: bool = True,
    slow_wall_seconds: float | None = None,
) -> dict[str, object]:
    results = []
    status_counts = {"pass": 0, "slow": 0, "fail": 0, "timeout": 0}
    issue_kind_counts: dict[str, int] = {}
    for case in cases:
        result = run_probe(
            case.expr,
            timeout_seconds,
            cas_cli=cas_cli,
            expected_result=case.expected_result,
            required_conditions=case.required_conditions,
            forbid_warnings=forbid_warnings,
            slow_wall_seconds=slow_wall_seconds,
        )
        status_counts[result.status] += 1
        result_dict = {
            "name": case.name,
            "expected_required_conditions": list(case.required_conditions),
            **result.as_dict(),
        }
        increment_issue_kind(issue_kind_counts, result.error_kind)
        results.append(result_dict)

    overall_status: Status = "pass"
    if not results:
        overall_status = "fail"
        increment_issue_kind(issue_kind_counts, "no_matching_cases")
    elif status_counts["timeout"]:
        overall_status = "timeout"
    elif status_counts["fail"]:
        overall_status = "fail"
    elif status_counts["slow"]:
        overall_status = "slow"

    return {
        "status": overall_status,
        "total": len(results),
        "status_counts": status_counts,
        "issue_kind_counts": issue_kind_counts,
        "cases": results,
    }


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
    except json.JSONDecodeError as err:
        return None, f"invalid json: {err}"
    if not isinstance(value, dict):
        return None, "json output is not an object"
    return value, None


def extract_result(parsed: dict[str, Any] | None) -> str | None:
    if not parsed:
        return None
    value = parsed.get("result")
    return value if isinstance(value, str) else None


def extract_required_conditions(parsed: dict[str, Any] | None) -> tuple[str, ...]:
    if not parsed:
        return ()
    raw = parsed.get("required_conditions") or parsed.get("requires") or []
    if not isinstance(raw, list):
        return ()
    displays: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            display = item.get("expr_display")
            if isinstance(display, str):
                displays.append(display)
        elif isinstance(item, str):
            displays.append(item)
    public_display = parsed.get("required_display") or []
    if isinstance(public_display, list):
        displays.extend(item for item in public_display if isinstance(item, str))
    return tuple(displays)


def extract_impossible_required_conditions(parsed: dict[str, Any] | None) -> tuple[str, ...]:
    if not parsed:
        return ()
    raw = parsed.get("required_conditions") or []
    if not isinstance(raw, list):
        return ()
    impossible: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "NonZero":
            continue
        display = item.get("expr_display")
        canonical = item.get("expr_canonical")
        witness = canonical if isinstance(canonical, str) else display
        if witness == "0":
            impossible.append("NonZero(0)")
    return tuple(impossible)


def extract_warnings(parsed: dict[str, Any] | None) -> tuple[str, ...]:
    if not parsed:
        return ()
    raw = parsed.get("warnings") or []
    if not isinstance(raw, list):
        return ()
    warnings: list[str] = []
    for item in raw:
        if isinstance(item, str):
            warnings.append(item)
        elif isinstance(item, dict):
            message = item.get("message") or item.get("text")
            if isinstance(message, str):
                warnings.append(message)
    return tuple(warnings)


def extract_stderr_warnings(stderr: str) -> tuple[str, ...]:
    warnings: list[str] = []
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("WARN ") or " WARN " in stripped:
            warnings.append(stripped)
    return tuple(warnings)


def classify_error_kind(error: str | None) -> IssueKind | None:
    if error is None:
        return None
    if error == "timeout":
        return "timeout"
    if error.startswith("nonzero exit:"):
        return "nonzero_exit"
    if error.startswith("invalid json:"):
        return "invalid_json"
    if error == "json output is not an object":
        return "non_object_json"
    if error.startswith("expected result "):
        return "result_mismatch"
    if error.startswith("missing required conditions:"):
        return "missing_required_conditions"
    if error.startswith("impossible required conditions:"):
        return "impossible_required_conditions"
    if error.startswith("unexpected warnings:"):
        return "unexpected_warnings"
    return None


def increment_issue_kind(issue_kind_counts: dict[str, int], issue_kind: IssueKind | None) -> None:
    if issue_kind is None:
        return
    issue_kind_counts[issue_kind] = issue_kind_counts.get(issue_kind, 0) + 1


def summarize_matrix(matrix: dict[str, object]) -> dict[str, object]:
    summary = {key: value for key, value in matrix.items() if key != "cases"}
    cases = matrix.get("cases")
    problem_cases: list[dict[str, object]] = []
    expected_required_condition_counts: dict[str, int] = {}
    expected_required_condition_case_count = 0
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, dict):
                continue
            expected_required = case.get("expected_required_conditions")
            if isinstance(expected_required, list):
                expected_conditions = [
                    condition for condition in expected_required if isinstance(condition, str)
                ]
                if expected_conditions:
                    expected_required_condition_case_count += 1
                for condition in expected_conditions:
                    expected_required_condition_counts[condition] = (
                        expected_required_condition_counts.get(condition, 0) + 1
                    )
            if case.get("status") == "pass" and case.get("error_kind") is None:
                continue
            problem_cases.append(
                {
                    "name": case.get("name"),
                    "status": case.get("status"),
                    "error_kind": case.get("error_kind"),
                    "error": case.get("error"),
                    "wall_elapsed_seconds": case.get("wall_elapsed_seconds"),
                    "result": case.get("result"),
                    "required_conditions": case.get("required_conditions"),
                }
            )
    summary["expected_required_condition_case_count"] = (
        expected_required_condition_case_count
    )
    summary["distinct_expected_required_conditions"] = len(
        expected_required_condition_counts
    )
    summary["expected_required_condition_counts"] = dict(
        sorted(expected_required_condition_counts.items())
    )
    summary["problem_case_count"] = len(problem_cases)
    summary["problem_cases"] = problem_cases
    return summary


def classify_error(
    returncode: int | None,
    parse_error: str | None,
    result: str | None,
    expected_result: str | None,
    actual_required: tuple[str, ...],
    expected_required: tuple[str, ...],
    impossible_required: tuple[str, ...],
    warnings: tuple[str, ...],
    forbid_warnings: bool,
) -> str | None:
    if returncode != 0:
        return f"nonzero exit: {returncode}"
    if parse_error is not None:
        return parse_error
    if expected_result is not None and result != expected_result:
        return f"expected result {expected_result!r}, got {result!r}"
    missing = [
        condition
        for condition in expected_required
        if not required_condition_satisfied(condition, actual_required)
    ]
    if missing:
        return f"missing required conditions: {', '.join(missing)}"
    if impossible_required:
        return f"impossible required conditions: {', '.join(impossible_required)}"
    if forbid_warnings and warnings:
        return f"unexpected warnings: {', '.join(warnings)}"
    return None


def required_condition_satisfied(condition: str, actual_required: tuple[str, ...]) -> bool:
    if condition in actual_required:
        return True
    root = affine_x_nonzero_root(condition)
    if root is None:
        return False
    return any(bound_excludes_root(actual, root) for actual in actual_required)


def affine_x_nonzero_root(condition: str) -> Fraction | None:
    compact = condition.replace(" ", "")
    if compact == "x":
        return Fraction(0)
    match = re.fullmatch(r"x([+-])(\d+)(?:/(\d+))?", compact)
    if not match:
        return None
    sign, numerator, denominator = match.groups()
    offset = Fraction(int(numerator), int(denominator) if denominator else 1)
    if sign == "-":
        offset = -offset
    return -offset


def bound_excludes_root(display: str, root: Fraction) -> bool:
    compact = display.replace(" ", "")
    lower = re.fullmatch(r"x[>≥](-?\d+)(?:/(\d+))?", compact)
    if lower:
        numerator, denominator = lower.groups()
        bound = Fraction(int(numerator), int(denominator) if denominator else 1)
        return root <= bound
    upper = re.fullmatch(r"x[<≤](-?\d+)(?:/(\d+))?", compact)
    if upper:
        numerator, denominator = upper.groups()
        bound = Fraction(int(numerator), int(denominator) if denominator else 1)
        return root >= bound
    return False


def expected_status_matches(status: Status, expect: str) -> bool:
    return expect == "any" or status == expect


def print_human(result: ProbeResult, expect: str) -> None:
    fields = [
        f"status={result.status}",
        f"expect={expect}",
        f"wall={result.wall_elapsed_seconds:.3f}s",
    ]
    if result.error_kind is not None:
        fields.append(f"kind={result.error_kind}")
    if result.result is not None:
        fields.append(f"result={result.result!r}")
    if result.required_conditions:
        fields.append(f"required={list(result.required_conditions)!r}")
    if result.impossible_required_conditions:
        fields.append(f"impossible_required={list(result.impossible_required_conditions)!r}")
    if result.warnings:
        fields.append(f"warnings={list(result.warnings)!r}")
    if result.slow_wall_seconds is not None:
        fields.append(f"slow_wall={result.slow_wall_seconds:.3f}s")
    if result.error is not None:
        fields.append(f"error={result.error!r}")
    print(" ".join(fields))
    if result.status != "pass" and result.stderr.strip():
        print(result.stderr.strip())


def print_matrix_human(matrix: dict[str, object], expect: str) -> None:
    print(
        f"status={matrix['status']} expect={expect} total={matrix['total']} "
        f"counts={matrix['status_counts']} issue_kinds={matrix.get('issue_kind_counts', {})}"
    )
    cases = matrix["cases"]
    if not isinstance(cases, list):
        return
    for case in cases:
        if not isinstance(case, dict):
            continue
        print(
            f"{case.get('name')} status={case.get('status')} "
            f"kind={case.get('error_kind')} "
            f"wall={case.get('wall_elapsed_seconds')}s "
            f"result={case.get('result')!r} required={case.get('required_conditions')}"
        )


def main() -> int:
    args = parse_args()
    if args.ensure_release_cas_cli:
        ensure_release_cas_cli(args.cas_cli)
    if args.matrix_residual is not None:
        cases = build_custom_residual_matrix_cases(
            args.matrix_residual_name,
            args.matrix_residual,
            required_conditions=tuple(args.require),
            wrapper_filters=tuple(args.matrix_wrapper),
            include_wrapper_required=not args.matrix_suppress_wrapper_requires,
        )
        matrix = run_matrix_cases(
            cases,
            args.timeout_seconds,
            cas_cli=args.cas_cli,
            forbid_warnings=args.forbid_warnings,
            slow_wall_seconds=args.slow_wall_seconds,
        )
        matrix["custom_residual_name"] = args.matrix_residual_name
        matrix["wrapper_filters"] = list(args.matrix_wrapper)
        if args.json:
            payload = summarize_matrix(matrix) if args.summary_json else matrix
            print(json.dumps(payload, sort_keys=True))
        else:
            print_matrix_human(matrix, args.expect)
        status = matrix["status"]
        return 0 if isinstance(status, str) and expected_status_matches(status, args.expect) else 1

    if args.default_matrix:
        matrix = run_default_matrix(
            args.timeout_seconds,
            cas_cli=args.cas_cli,
            forbid_warnings=args.forbid_warnings,
            slow_wall_seconds=args.slow_wall_seconds,
            base_filters=tuple(args.matrix_base),
            wrapper_filters=tuple(args.matrix_wrapper),
        )
        if args.json:
            payload = summarize_matrix(matrix) if args.summary_json else matrix
            print(json.dumps(payload, sort_keys=True))
        else:
            print_matrix_human(matrix, args.expect)
        status = matrix["status"]
        return 0 if isinstance(status, str) and expected_status_matches(status, args.expect) else 1

    if args.expr is None:
        raise SystemExit("--expr is required unless --default-matrix is used")

    result = run_probe(
        args.expr,
        args.timeout_seconds,
        cas_cli=args.cas_cli,
        expected_result=args.expect_result,
        required_conditions=tuple(args.require),
        forbid_warnings=args.forbid_warnings,
        slow_wall_seconds=args.slow_wall_seconds,
    )
    if args.json:
        print(json.dumps(result.as_dict(), sort_keys=True))
    else:
        print_human(result, args.expect)
    return 0 if expected_status_matches(result.status, args.expect) else 1


if __name__ == "__main__":
    sys.exit(main())
