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


if __name__ == "__main__":
    unittest.main()
