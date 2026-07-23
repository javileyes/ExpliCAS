#!/usr/bin/env python3
"""Smoke checks for public web examples with objective expected results."""

from __future__ import annotations

import csv
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_CSV = ROOT / "web" / "examples.csv"
CAS_CLI_BIN = ROOT / "target" / "release" / "cas_cli"

EXPECTED_SOLVE_SYSTEM_RESULTS = {
    "solve([x+y=3, x-y=1], [x, y])": "{ x = 2, y = 1 }",
    "solve([a*x+y=1, x-y=0], [x, y])": (
        "{ x = 1 / (a + 1), y = 1 / (a + 1) }\n  requires: a + 1 != 0"
    ),
    "solve([x^2+y^2=25, x+y=7], [x, y])": "{ x = 3, y = 4 } or { x = 4, y = 3 }",
    "solve([x*y=6, x+y=5], [x, y])": "{ x = 2, y = 3 } or { x = 3, y = 2 }",
}

EXPECTED_DSOLVE_RESULTS = {
    "dsolve(diff(y,x) = x*y, y, x)": "y = C\u00b7e^(x^2 / 2)",
    "dsolve(diff(y,x) + y = x, y, x)": "y = C / e^x + x - 1",
    "dsolve((2*x*y+1) + (x^2+2*y)*diff(y,x) = 0, y, x)": "y\u00b7x^2 + y^2 + x = C",
    "dsolve(diff(y,x) = -y, y, x, y(0) = 3)": "y = 3 / e^x",
    "dsolve(diff(y,x,2) + 4*y = 0, y, x)": "y = C1\u00b7sin(2\u00b7x) + C2\u00b7cos(2\u00b7x)",
    "dsolve(diff(y,x,2) + y = cos(x), y, x)": "y = 1/2\u00b7x\u00b7sin(x) + C1\u00b7sin(x) + C2\u00b7cos(x)",
    "dsolve(diff(y,x) + y = y^2, y, x)": "y = 1 / (C\u00b7e^x + 1)",
    "dsolve([diff(x,t) = -y, diff(y,t) = x], [x,y], t)": "{ x = -C1\u00b7cos(t) - C2\u00b7sin(t), y = C2\u00b7cos(t) - C1\u00b7sin(t) }",
    "dsolve(diff(y,x) = x^2 + y^2, y, x)": "dsolve(diff(y, x) = x^2 + y^2, y, x)",
}

EXPECTED_LIMIT_RESULTS = {
    "limit((x^2+1)/(2*x^2-3), x, infinity)": "1/2",
    "limit(x^2/exp(2*x), x, infinity)": "0",
    "limit(ln(x)/exp(x), x, infinity)": "0",
    "limit(sqrt(x)*exp(-x), x, infinity)": "0",
    "limit(x/ln(1-x), x, -infinity)": "-infinity",
    "limit((x^2 + 3*x)/(2*x^2 - x), x, inf)": "1/2",
}


def load_example_expressions() -> list[str]:
    with EXAMPLES_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row["expression"] for row in reader]


def calculus_residual_examples(expressions: list[str]) -> list[str]:
    return [
        expression
        for expression in expressions
        if expression.startswith("diff(integrate(") and " - " in expression
    ]


def calculus_equivalence_examples(expressions: list[str]) -> list[str]:
    return [
        expression
        for expression in expressions
        if expression.startswith("equiv(") and "diff(integrate(" in expression
    ]


def eval_result(expression: str, *, timeout_seconds: float = 8.0) -> str:
    completed = subprocess.run(
        [
            str(CAS_CLI_BIN),
            "eval",
            expression,
            "--format",
            "json",
            "--steps",
            "off",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
    )
    payload = json.loads(completed.stdout)
    return str(payload.get("result", ""))


class WebExamplesSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            ["cargo", "build", "--release", "-q", "-p", "cas_cli"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180.0,
        )

    def test_public_calculus_verification_examples_stay_true(self) -> None:
        expressions = load_example_expressions()
        residual_examples = calculus_residual_examples(expressions)
        equivalence_examples = calculus_equivalence_examples(expressions)

        self.assertGreaterEqual(
            len(residual_examples),
            1,
            "expected at least one web example that verifies a derivative residual",
        )
        self.assertGreaterEqual(
            len(equivalence_examples),
            1,
            "expected at least one web example that verifies calculus equivalence",
        )

        for expression in residual_examples:
            with self.subTest(expression=expression):
                self.assertEqual(eval_result(expression), "0")

        for expression in equivalence_examples:
            with self.subTest(expression=expression):
                self.assertEqual(eval_result(expression), "true")

    def test_public_limit_examples_keep_expected_results(self) -> None:
        expressions = set(load_example_expressions())

        for expression, expected_result in EXPECTED_LIMIT_RESULTS.items():
            with self.subTest(expression=expression):
                self.assertIn(expression, expressions)
                self.assertEqual(eval_result(expression), expected_result)

    def test_public_dsolve_examples_keep_expected_results(self) -> None:
        # Fase 4 · O7: every published ODE example auto-verifies its exact
        # emitted result (including the honest Riccati residual echo).
        expressions = set(load_example_expressions())

        for expression, expected_result in EXPECTED_DSOLVE_RESULTS.items():
            with self.subTest(expression=expression):
                self.assertIn(expression, expressions)
                self.assertEqual(eval_result(expression), expected_result)


    def test_public_solve_system_examples_keep_expected_results(self) -> None:
        # Frente S · S4: every published system example auto-verifies its
        # exact emitted result (list form, parametric condition, verified
        # nonlinear pairs).
        expressions = set(load_example_expressions())

        for expression, expected_result in EXPECTED_SOLVE_SYSTEM_RESULTS.items():
            with self.subTest(expression=expression):
                self.assertIn(expression, expressions)
                self.assertEqual(eval_result(expression), expected_result)


if __name__ == "__main__":
    unittest.main()
