#!/usr/bin/env python3
"""Focused tests for the calculus residual probe smoke harness."""

from __future__ import annotations

import importlib.util
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent / "engine_calculus_residual_probe_smoke.py"
SPEC = importlib.util.spec_from_file_location("engine_calculus_residual_probe_smoke", SCRIPT_PATH)
assert SPEC is not None
SMOKE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)


class CalculusResidualProbeSmokeTests(unittest.TestCase):
    def test_extracts_public_result_required_conditions_and_warnings(self) -> None:
        parsed, error = SMOKE.parse_json(
            textwrap.dedent(
                """\
                {
                  "result": "1 / (x + 2)",
                  "required_conditions": [
                    {"kind": "NonZero", "expr_display": "x + 1"},
                    {"kind": "NonZero", "expr_display": "x + 2"}
                  ],
                  "warnings": [{"message": "diagnostic"}]
                }
                """
            )
        )

        self.assertIsNone(error)
        self.assertEqual(SMOKE.extract_result(parsed), "1 / (x + 2)")
        self.assertEqual(
            SMOKE.extract_required_conditions(parsed),
            ("x + 1", "x + 2"),
        )
        self.assertEqual(SMOKE.extract_impossible_required_conditions(parsed), ())
        self.assertEqual(SMOKE.extract_warnings(parsed), ("diagnostic",))
        self.assertEqual(
            SMOKE.extract_stderr_warnings(
                "2026-05-15T05:36:56Z  WARN simplify: depth_overflow\n"
                "ordinary stderr line\n"
            ),
            ("2026-05-15T05:36:56Z  WARN simplify: depth_overflow",),
        )

    def test_classify_error_checks_result_required_conditions_and_warnings(self) -> None:
        self.assertIsNone(
            SMOKE.classify_error(
                returncode=0,
                parse_error=None,
                result="1 / (x + 2)",
                expected_result="1 / (x + 2)",
                actual_required=("x + 1", "x + 2"),
                expected_required=("x + 2",),
                impossible_required=(),
                warnings=(),
                forbid_warnings=True,
            )
        )

        self.assertIn(
            "missing required conditions",
            SMOKE.classify_error(
                returncode=0,
                parse_error=None,
                result="1 / (x + 2)",
                expected_result="1 / (x + 2)",
                actual_required=("x + 1",),
                expected_required=("x + 2",),
                impossible_required=(),
                warnings=(),
                forbid_warnings=False,
            ),
        )
        self.assertIn(
            "impossible required conditions",
            SMOKE.classify_error(
                returncode=0,
                parse_error=None,
                result="undefined",
                expected_result="undefined",
                actual_required=("0",),
                expected_required=(),
                impossible_required=("NonZero(0)",),
                warnings=(),
                forbid_warnings=False,
            ),
        )
        self.assertIn(
            "unexpected warnings",
            SMOKE.classify_error(
                returncode=0,
                parse_error=None,
                result="1 / (x + 2)",
                expected_result="1 / (x + 2)",
                actual_required=("x + 2",),
                expected_required=("x + 2",),
                impossible_required=(),
                warnings=("diagnostic",),
                forbid_warnings=True,
            ),
        )

    def test_extracts_literal_impossible_nonzero_required_conditions(self) -> None:
        parsed, error = SMOKE.parse_json(
            textwrap.dedent(
                """\
                {
                  "result": "undefined",
                  "required_conditions": [
                    {"kind": "NonZero", "expr_display": "0", "expr_canonical": "0"},
                    {"kind": "Positive", "expr_display": "0", "expr_canonical": "0"},
                    {"kind": "NonZero", "expr_display": "x", "expr_canonical": "x"}
                  ],
                  "warnings": []
                }
                """
            )
        )

        self.assertIsNone(error)
        self.assertEqual(
            SMOKE.extract_impossible_required_conditions(parsed),
            ("NonZero(0)",),
        )

    def test_run_probe_classifies_pass_and_slow_with_cli_stub(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    sleep 0.1
                    cat <<'OUT'
                    {"result":"1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            result = SMOKE.run_probe(
                "ignored",
                timeout_seconds=2.0,
                cas_cli=cas_cli,
                expected_result="1 / (x + 2)",
                required_conditions=("x + 2",),
                forbid_warnings=True,
                slow_wall_seconds=0.01,
            )

        self.assertEqual(result.status, "slow")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.result, "1 / (x + 2)")
        self.assertEqual(result.required_conditions, ("x + 2",))
        self.assertEqual(result.impossible_required_conditions, ())

    def test_run_probe_fails_on_literal_impossible_nonzero_required_condition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    cat <<'OUT'
                    {"result":"undefined","required_conditions":[{"kind":"NonZero","expr_display":"0","expr_canonical":"0"}],"warnings":[]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            result = SMOKE.run_probe(
                "ignored",
                timeout_seconds=2.0,
                cas_cli=cas_cli,
                expected_result="undefined",
            )

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.impossible_required_conditions, ("NonZero(0)",))
        self.assertIn("impossible required conditions", result.error or "")

    def test_run_probe_forbid_warnings_includes_stderr_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    echo '2026-05-15T05:36:56Z  WARN simplify: depth_overflow' >&2
                    cat <<'OUT'
                    {"result":"1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            result = SMOKE.run_probe(
                "ignored",
                timeout_seconds=2.0,
                cas_cli=cas_cli,
                expected_result="1 / (x + 2)",
                required_conditions=("x + 2",),
                forbid_warnings=True,
            )

        self.assertEqual(result.status, "fail")
        self.assertIn("depth_overflow", result.error or "")
        self.assertEqual(
            result.warnings,
            ("2026-05-15T05:36:56Z  WARN simplify: depth_overflow",),
        )

    def test_run_probe_timeout_terminates_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text("#!/bin/sh\nsleep 10\n", encoding="utf-8")
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            started = time.monotonic()
            result = SMOKE.run_probe("ignored", timeout_seconds=0.2, cas_cli=cas_cli)
            elapsed = time.monotonic() - started

        self.assertEqual(result.status, "timeout")
        self.assertIsNone(result.returncode)
        self.assertLess(elapsed, 3.0)

    def test_expected_status_matches_any(self) -> None:
        self.assertTrue(SMOKE.expected_status_matches("timeout", "any"))
        self.assertTrue(SMOKE.expected_status_matches("slow", "slow"))
        self.assertFalse(SMOKE.expected_status_matches("pass", "timeout"))

    def test_default_matrix_contains_representative_wrapped_residuals(self) -> None:
        cases = SMOKE.build_default_matrix_cases()

        self.assertEqual(len(cases), 212)
        self.assertIn("arctan_sqrt_additive_trig:plus", {case.name for case in cases})
        self.assertIn("arctan_sqrt_additive_trig:nested_den", {case.name for case in cases})
        self.assertIn("integrate_exp_trig_sin:plus", {case.name for case in cases})
        self.assertIn("integrate_exp_trig_sin:nested_den", {case.name for case in cases})
        self.assertIn("integrate_exp_trig_cos:plus", {case.name for case in cases})
        self.assertIn("integrate_exp_trig_cos:nested_den", {case.name for case in cases})
        self.assertIn("integrate_exp_trig_neg_sin:plus", {case.name for case in cases})
        self.assertIn("integrate_exp_trig_neg_sin:nested_den", {case.name for case in cases})
        self.assertIn("integrate_exp_trig_neg_cos:plus", {case.name for case in cases})
        self.assertIn("integrate_exp_trig_neg_cos:nested_den", {case.name for case in cases})
        self.assertIn("plain_trig_neg_sin:plus", {case.name for case in cases})
        self.assertIn("plain_trig_neg_sin:nested_den", {case.name for case in cases})
        self.assertIn("plain_trig_neg_cos:plus", {case.name for case in cases})
        self.assertIn("plain_trig_neg_cos:nested_den", {case.name for case in cases})
        self.assertIn("plain_trig_sparse_neg_sin:plus", {case.name for case in cases})
        self.assertIn("plain_trig_sparse_neg_sin:nested_den", {case.name for case in cases})
        self.assertIn("rational_quad:plus", {case.name for case in cases})
        self.assertIn("hyperbolic_sinh:scaled_negative", {case.name for case in cases})
        self.assertIn("hyperbolic_sinh:minus_const", {case.name for case in cases})
        self.assertIn("hyperbolic_sinh:plus_noise", {case.name for case in cases})
        self.assertIn("hyperbolic_sinh:plus_double_noise", {case.name for case in cases})
        self.assertIn("hyperbolic_sinh:plus_triple_noise", {case.name for case in cases})
        self.assertIn(
            "hyperbolic_sinh:plus_triple_noise_times_one", {case.name for case in cases}
        )
        self.assertIn("rational_quad:product_den", {case.name for case in cases})
        self.assertIn(
            "hyperbolic_sinh:product_den_triple_noise_times_one",
            {case.name for case in cases},
        )
        self.assertIn("hyperbolic_sinh:nested_den", {case.name for case in cases})
        self.assertIn("hyperbolic_sinh_over_cosh_shifted_quotient", {case.name for case in cases})
        self.assertIn(
            "hyperbolic_sinh_over_cosh_negative_shifted_quotient",
            {case.name for case in cases},
        )
        self.assertIn(
            "hyperbolic_sinh_over_scaled_cosh_shifted_quotient",
            {case.name for case in cases},
        )
        self.assertIn(
            "scaled_hyperbolic_sinh_over_scaled_cosh_shifted_quotient",
            {case.name for case in cases},
        )
        self.assertIn(
            "scaled_hyperbolic_sinh_over_cosh_product_denominator_shifted_quotient",
            {case.name for case in cases},
        )
        self.assertIn(
            "scaled_hyperbolic_sinh_over_negative_cosh_product_denominator_shifted_quotient",
            {case.name for case in cases},
        )
        self.assertIn(
            "scaled_hyperbolic_sinh_over_negative_cosh_fraction_denominator_shifted_quotient",
            {case.name for case in cases},
        )
        self.assertIn(
            "scaled_hyperbolic_sinh_over_scaled_negative_cosh_fraction_denominator_shifted_quotient",
            {case.name for case in cases},
        )
        self.assertIn(
            "scaled_hyperbolic_sinh_over_negative_cosh_fraction_denominator_positive_quadratic_shifted_quotient",
            {case.name for case in cases},
        )
        self.assertIn(
            "rational_quad_over_recip_trig_shifted_quotient", {case.name for case in cases}
        )
        self.assertIn("fractional_den_power:plus", {case.name for case in cases})
        self.assertIn("fractional_den_power:nested_den", {case.name for case in cases})
        rational_case = next(case for case in cases if case.name == "rational_quad:plus")
        self.assertEqual(rational_case.expected_result, "1 / (x + 2)")
        self.assertEqual(rational_case.required_conditions, ("x + 1", "x + 2"))
        scaled_negative_case = next(
            case for case in cases if case.name == "rational_quad:scaled_negative"
        )
        self.assertEqual(scaled_negative_case.expected_result, "-1 / (x + 2)")
        self.assertEqual(scaled_negative_case.required_conditions, ("x + 1", "x + 2"))
        minus_const_case = next(case for case in cases if case.name == "rational_quad:minus_const")
        self.assertEqual(minus_const_case.expected_result, "-1 / (x + 2)")
        self.assertEqual(minus_const_case.required_conditions, ("x + 1", "x + 2"))
        plus_noise_case = next(case for case in cases if case.name == "rational_quad:plus_noise")
        self.assertEqual(plus_noise_case.expected_result, "1 / (x + 2)")
        self.assertEqual(plus_noise_case.required_conditions, ("x + 1", "x + 2"))
        plus_double_noise_case = next(
            case for case in cases if case.name == "rational_quad:plus_double_noise"
        )
        self.assertEqual(plus_double_noise_case.expected_result, "1 / (x + 2)")
        self.assertEqual(plus_double_noise_case.required_conditions, ("x + 1", "x + 2"))
        plus_triple_noise_case = next(
            case for case in cases if case.name == "rational_quad:plus_triple_noise"
        )
        self.assertEqual(plus_triple_noise_case.expected_result, "1 / (x + 2)")
        self.assertEqual(plus_triple_noise_case.required_conditions, ("x + 1", "x + 2"))
        plus_triple_noise_times_one_case = next(
            case for case in cases if case.name == "rational_quad:plus_triple_noise_times_one"
        )
        self.assertEqual(plus_triple_noise_times_one_case.expected_result, "1 / (x + 2)")
        self.assertEqual(
            plus_triple_noise_times_one_case.required_conditions, ("x + 1", "x + 2")
        )
        product_case = next(case for case in cases if case.name == "rational_quad:product_den")
        self.assertEqual(product_case.expected_result, "1 / ((x + 2)·(x + 3))")
        self.assertEqual(product_case.required_conditions, ("x + 1", "x + 2", "x + 3"))
        product_noise_case = next(
            case
            for case in cases
            if case.name == "rational_quad:product_den_triple_noise_times_one"
        )
        self.assertEqual(product_noise_case.expected_result, "1 / ((x + 2)·(x + 3))")
        self.assertEqual(product_noise_case.required_conditions, ("x + 1", "x + 2", "x + 3"))
        exp_nested_case = next(case for case in cases if case.name == "exp_poly:nested_den")
        self.assertEqual(exp_nested_case.expected_result, "1 / (x·(x + 2) + 3·x + 6)")
        self.assertEqual(exp_nested_case.required_conditions, ("x + 2", "x + 3"))
        arctan_sqrt_nested_case = next(
            case for case in cases if case.name == "arctan_sqrt_additive_trig:nested_den"
        )
        self.assertEqual(
            arctan_sqrt_nested_case.expected_result,
            "1 / (x·(x + 2) + 3·x + 6)",
        )
        self.assertEqual(arctan_sqrt_nested_case.required_conditions, ("x + 2", "x + 3"))
        exp_trig_nested_case = next(
            case for case in cases if case.name == "integrate_exp_trig_sin:nested_den"
        )
        self.assertEqual(
            exp_trig_nested_case.expected_result,
            "1 / ((x + 2)·(x + 3))",
        )
        self.assertEqual(exp_trig_nested_case.required_conditions, ("x + 2", "x + 3"))
        exp_trig_neg_nested_case = next(
            case for case in cases if case.name == "integrate_exp_trig_neg_sin:nested_den"
        )
        self.assertEqual(
            exp_trig_neg_nested_case.expected_result,
            "1 / ((x + 2)·(x + 3))",
        )
        self.assertEqual(exp_trig_neg_nested_case.required_conditions, ("x + 2", "x + 3"))
        plain_trig_neg_nested_case = next(
            case for case in cases if case.name == "plain_trig_neg_sin:nested_den"
        )
        self.assertEqual(
            plain_trig_neg_nested_case.expected_result,
            "1 / ((x + 2)·(x + 3))",
        )
        self.assertEqual(plain_trig_neg_nested_case.required_conditions, ("x + 2", "x + 3"))
        sparse_plain_trig_neg_nested_case = next(
            case for case in cases if case.name == "plain_trig_sparse_neg_sin:nested_den"
        )
        self.assertEqual(
            sparse_plain_trig_neg_nested_case.expected_result,
            "1 / ((x + 2)·(x + 3))",
        )
        self.assertEqual(
            sparse_plain_trig_neg_nested_case.required_conditions, ("x + 2", "x + 3")
        )

    def test_default_matrix_filters_by_base_and_wrapper(self) -> None:
        base_cases = SMOKE.build_default_matrix_cases(base_filters=("arctan_sqrt_additive_trig",))
        self.assertEqual(len(base_cases), 11)
        self.assertTrue(
            all(case.name.startswith("arctan_sqrt_additive_trig:") for case in base_cases)
        )

        wrapper_cases = SMOKE.build_default_matrix_cases(wrapper_filters=("nested_den",))
        self.assertTrue(wrapper_cases)
        self.assertTrue(all(case.name.endswith(":nested_den") for case in wrapper_cases))

        exact_case = SMOKE.build_default_matrix_cases(
            base_filters=("hyperbolic_sinh_over_cosh_shifted_quotient",)
        )
        self.assertEqual(len(exact_case), 1)
        self.assertEqual(exact_case[0].name, "hyperbolic_sinh_over_cosh_shifted_quotient")

        cases = SMOKE.build_default_matrix_cases()
        fractional_den_power_case = next(
            case for case in cases if case.name == "fractional_den_power:plus"
        )
        self.assertEqual(fractional_den_power_case.expected_result, "1 / (x + 2)")
        self.assertEqual(fractional_den_power_case.required_conditions, ("x + 2",))
        fractional_den_power_nested_case = next(
            case for case in cases if case.name == "fractional_den_power:nested_den"
        )
        self.assertEqual(
            fractional_den_power_nested_case.expected_result,
            "1 / (x·(x + 2) + 3·x + 6)",
        )
        self.assertEqual(fractional_den_power_nested_case.required_conditions, ("x + 2", "x + 3"))
        shifted_quotient_case = next(
            case for case in cases if case.name == "rational_quad_over_recip_trig_shifted_quotient"
        )
        self.assertEqual(shifted_quotient_case.expected_result, "1")
        self.assertEqual(shifted_quotient_case.required_conditions, ("x + 1",))
        negative_shifted_quotient_case = next(
            case
            for case in cases
            if case.name == "hyperbolic_sinh_over_cosh_negative_shifted_quotient"
        )
        self.assertEqual(negative_shifted_quotient_case.expected_result, "-1")
        self.assertEqual(negative_shifted_quotient_case.required_conditions, ())
        scaled_shifted_quotient_case = next(
            case
            for case in cases
            if case.name == "hyperbolic_sinh_over_scaled_cosh_shifted_quotient"
        )
        self.assertEqual(scaled_shifted_quotient_case.expected_result, "1/2")
        self.assertEqual(scaled_shifted_quotient_case.required_conditions, ())
        scaled_numerator_and_denominator_case = next(
            case
            for case in cases
            if case.name == "scaled_hyperbolic_sinh_over_scaled_cosh_shifted_quotient"
        )
        self.assertEqual(scaled_numerator_and_denominator_case.expected_result, "3/2")
        self.assertEqual(scaled_numerator_and_denominator_case.required_conditions, ())
        product_denominator_residual_factor_case = next(
            case
            for case in cases
            if case.name
            == "scaled_hyperbolic_sinh_over_cosh_product_denominator_shifted_quotient"
        )
        self.assertEqual(
            product_denominator_residual_factor_case.expected_result,
            "3 / (x + 2)",
        )
        self.assertEqual(
            product_denominator_residual_factor_case.required_conditions,
            ("x + 2",),
        )
        negative_product_denominator_residual_factor_case = next(
            case
            for case in cases
            if case.name
            == "scaled_hyperbolic_sinh_over_negative_cosh_product_denominator_shifted_quotient"
        )
        self.assertEqual(
            negative_product_denominator_residual_factor_case.expected_result,
            "-3 / ((x + 2)·(x + 3))",
        )
        self.assertEqual(
            negative_product_denominator_residual_factor_case.required_conditions,
            ("x + 2", "x + 3"),
        )
        negative_fraction_denominator_residual_factor_case = next(
            case
            for case in cases
            if case.name
            == "scaled_hyperbolic_sinh_over_negative_cosh_fraction_denominator_shifted_quotient"
        )
        self.assertEqual(
            negative_fraction_denominator_residual_factor_case.expected_result,
            "-3·(x + 2)",
        )
        self.assertEqual(
            negative_fraction_denominator_residual_factor_case.required_conditions,
            ("x + 2",),
        )
        scaled_negative_fraction_denominator_residual_factor_case = next(
            case
            for case in cases
            if case.name
            == "scaled_hyperbolic_sinh_over_scaled_negative_cosh_fraction_denominator_shifted_quotient"
        )
        self.assertEqual(
            scaled_negative_fraction_denominator_residual_factor_case.expected_result,
            "-3/2·(x + 2)",
        )
        self.assertEqual(
            scaled_negative_fraction_denominator_residual_factor_case.required_conditions,
            ("x + 2",),
        )
        positive_quadratic_fraction_denominator_case = next(
            case
            for case in cases
            if case.name
            == "scaled_hyperbolic_sinh_over_negative_cosh_fraction_denominator_positive_quadratic_shifted_quotient"
        )
        self.assertEqual(
            positive_quadratic_fraction_denominator_case.expected_result,
            "-3·(x^2 + 1)",
        )
        self.assertEqual(
            positive_quadratic_fraction_denominator_case.required_conditions,
            (),
        )
        product_zero_csc_case = next(
            case for case in cases if case.name == "recip_trig_csc_product_zero_factor"
        )
        self.assertEqual(product_zero_csc_case.expected_result, "0")
        self.assertEqual(product_zero_csc_case.required_conditions, ("sin(2·x + 1)",))
        product_zero_csc_reversed_case = next(
            case
            for case in cases
            if case.name == "recip_trig_csc_product_zero_factor_reversed"
        )
        self.assertEqual(product_zero_csc_reversed_case.expected_result, "0")
        self.assertEqual(
            product_zero_csc_reversed_case.required_conditions,
            ("sin(2·x + 1)",),
        )
        product_zero_trig_case = next(
            case for case in cases if case.name == "plain_trig_by_parts_product_zero_factor"
        )
        self.assertEqual(product_zero_trig_case.expected_result, "0")
        self.assertEqual(product_zero_trig_case.required_conditions, ())
        product_zero_hyperbolic_case = next(
            case for case in cases if case.name == "hyperbolic_by_parts_product_zero_factor"
        )
        self.assertEqual(product_zero_hyperbolic_case.expected_result, "0")
        self.assertEqual(product_zero_hyperbolic_case.required_conditions, ())

    def test_custom_residual_matrix_builds_standard_wrappers(self) -> None:
        cases = SMOKE.build_custom_residual_matrix_cases(
            "candidate",
            "diff(f(x),x)-g(x)",
            required_conditions=("x > 0",),
            wrapper_filters=("plus",),
        )

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].name, "candidate:plus")
        self.assertEqual(cases[0].expr, "((diff(f(x),x)-g(x))+1)/(x+2)")
        self.assertEqual(cases[0].expected_result, "1 / (x + 2)")
        self.assertEqual(cases[0].required_conditions, ("x > 0", "x + 2"))

    def test_cli_accepts_custom_residual_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    cat <<'OUT'
                    {"result":"1 / (x + 2)","required_conditions":[{"expr_display":"x > 0"},{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--matrix-residual",
                    "diff(f(x),x)-g(x)",
                    "--matrix-residual-name",
                    "candidate",
                    "--matrix-wrapper",
                    "plus",
                    "--require",
                    "x > 0",
                    "--cas-cli",
                    str(cas_cli),
                    "--timeout-seconds",
                    "2",
                    "--json",
                ],
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn('"custom_residual_name": "candidate"', completed.stdout)
        self.assertIn('"total": 1', completed.stdout)
        self.assertIn('"status": "pass"', completed.stdout)

    def test_run_default_matrix_summarizes_all_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    case "$*" in
                    *"csc(2*x+1)"*"(y-y)"*)
                    cat <<'OUT'
                    {"result":"0","required_conditions":[{"expr_display":"sin(2·x + 1)"}],"warnings":[]}
                    OUT
                    ;;
                    *"(y-y)"*"csc(2*x+1)"*)
                    cat <<'OUT'
                    {"result":"0","required_conditions":[{"expr_display":"sin(2·x + 1)"}],"warnings":[]}
                    OUT
                    ;;
                    *"*(y-y)"*)
                    cat <<'OUT'
                    {"result":"0","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(((diff("*")-1)/(x^2+1)"*)
                    cat <<'OUT'
                    {"result":"-3·(x^2 + 1)","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(2*((diff("*")-1)/(x+2)"*)
                    cat <<'OUT'
                    {"result":"-3/2·(x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(((diff("*")-1)/(x+2)"*)
                    cat <<'OUT'
                    {"result":"-3·(x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"/(x+2))/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / (x·(x + 2) + 3·x + 6)","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"x+3"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*")-1)/(x+2)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"(-1)"*"^(3/2)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"(-1)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*)
                    cat <<'OUT'
                    {"result":"1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *")-1)/(x+2)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"(-1)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(2*((diff("*)
                    cat <<'OUT'
                    {"result":"3/2","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/((x+2)*((diff("*")-1)*(x+3)"*)
                    cat <<'OUT'
                    {"result":"-3 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(((diff("*")*(x+2)"*)
                    cat <<'OUT'
                    {"result":"3 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *")/(2*((diff("*)
                    cat <<'OUT'
                    {"result":"1/2","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *")+1)/((diff("*")-1)"*)
                    cat <<'OUT'
                    {"result":"-1","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *")+1)/((diff("*)
                    cat <<'OUT'
                    {"result":"1","required_conditions":[{"expr_display":"x + 1"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"/(x+2))/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / (x·(x + 2) + 3·x + 6)","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"x+3"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*")-1)/(x+2)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"(-1)"*"^(3/2)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"(-1)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*)
                    cat <<'OUT'
                    {"result":"1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *arctan*sqrt*"/(x+2))/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / (x·(x + 2) + 3·x + 6)","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *arctan*sqrt*"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *sin*"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *cos*"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"exp("*"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / (x·(x + 2) + 3·x + 6)","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"x+3"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *)
                    cat <<'OUT'
                    {"result":"1 / (x + 2)","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            matrix = SMOKE.run_default_matrix(
                timeout_seconds=2.0,
                cas_cli=cas_cli,
                forbid_warnings=True,
                slow_wall_seconds=1.0,
            )

        self.assertEqual(matrix["status"], "pass", matrix)
        self.assertEqual(matrix["total"], 212)
        self.assertEqual(matrix["status_counts"]["pass"], 212)

    def test_cli_accepts_repeated_require_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    case "$*" in
                    *"csc(2*x+1)"*"(y-y)"*)
                    cat <<'OUT'
                    {"result":"0","required_conditions":[{"expr_display":"sin(2·x + 1)"}],"warnings":[]}
                    OUT
                    ;;
                    *"(y-y)"*"csc(2*x+1)"*)
                    cat <<'OUT'
                    {"result":"0","required_conditions":[{"expr_display":"sin(2·x + 1)"}],"warnings":[]}
                    OUT
                    ;;
                    *"*(y-y)"*)
                    cat <<'OUT'
                    {"result":"0","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(((diff("*")-1)/(x^2+1)"*)
                    cat <<'OUT'
                    {"result":"-3·(x^2 + 1)","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(2*((diff("*")-1)/(x+2)"*)
                    cat <<'OUT'
                    {"result":"-3/2·(x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(((diff("*")-1)/(x+2)"*)
                    cat <<'OUT'
                    {"result":"-3·(x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"/(x+2))/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / (x·(x + 2) + 3·x + 6)","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"x+3"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*")-1)/(x+2)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"(-1)"*"^(3/2)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"(-1)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*)
                    cat <<'OUT'
                    {"result":"1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *")-1)/(x+2)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"(-1)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(2*((diff("*)
                    cat <<'OUT'
                    {"result":"3/2","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/((x+2)*((diff("*")-1)*(x+3)"*)
                    cat <<'OUT'
                    {"result":"-3 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(((diff("*")*(x+2)"*)
                    cat <<'OUT'
                    {"result":"3 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *")/(2*((diff("*)
                    cat <<'OUT'
                    {"result":"1/2","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *")+1)/((diff("*")-1)"*)
                    cat <<'OUT'
                    {"result":"-1","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *")+1)/((diff("*)
                    cat <<'OUT'
                    {"result":"1","required_conditions":[{"expr_display":"x + 1"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"/(x+2))/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / (x·(x + 2) + 3·x + 6)","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"x+3"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*")-1)/(x+2)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"(-1)"*"^(3/2)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"(-1)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*)
                    cat <<'OUT'
                    {"result":"1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *arctan*sqrt*"/(x+2))/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / (x·(x + 2) + 3·x + 6)","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *arctan*sqrt*"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *sin*"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *cos*"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"exp("*"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / (x·(x + 2) + 3·x + 6)","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"x+3"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *)
                    cat <<'OUT'
                    {"result":"1 / (x + 2)","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--expr",
                    "ignored",
                    "--cas-cli",
                    str(cas_cli),
                    "--expect-result",
                    "1 / (x + 2)",
                    "--require",
                    "x + 1",
                    "--require",
                    "x + 2",
                    "--forbid-warnings",
                    "--json",
                ],
                cwd=SCRIPT_PATH.parent.parent,
                text=True,
                capture_output=True,
                timeout=3,
            )

        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn('"status": "pass"', completed.stdout)

    def test_cli_runs_default_matrix_without_expr(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    case "$*" in
                    *"csc(2*x+1)"*"(y-y)"*)
                    cat <<'OUT'
                    {"result":"0","required_conditions":[{"expr_display":"sin(2·x + 1)"}],"warnings":[]}
                    OUT
                    ;;
                    *"(y-y)"*"csc(2*x+1)"*)
                    cat <<'OUT'
                    {"result":"0","required_conditions":[{"expr_display":"sin(2·x + 1)"}],"warnings":[]}
                    OUT
                    ;;
                    *"*(y-y)"*)
                    cat <<'OUT'
                    {"result":"0","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(((diff("*")-1)/(x^2+1)"*)
                    cat <<'OUT'
                    {"result":"-3·(x^2 + 1)","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(2*((diff("*")-1)/(x+2)"*)
                    cat <<'OUT'
                    {"result":"-3/2·(x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(((diff("*")-1)/(x+2)"*)
                    cat <<'OUT'
                    {"result":"-3·(x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *")-1)/(x+2)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"(-1)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(2*((diff("*)
                    cat <<'OUT'
                    {"result":"3/2","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/((x+2)*((diff("*")-1)*(x+3)"*)
                    cat <<'OUT'
                    {"result":"-3 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"3*(("*")/(((diff("*")*(x+2)"*)
                    cat <<'OUT'
                    {"result":"3 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *")/(2*((diff("*)
                    cat <<'OUT'
                    {"result":"1/2","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *")+1)/((diff("*")-1)"*)
                    cat <<'OUT'
                    {"result":"-1","required_conditions":[],"warnings":[]}
                    OUT
                    ;;
                    *")+1)/((diff("*)
                    cat <<'OUT'
                    {"result":"1","required_conditions":[{"expr_display":"x + 1"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"/(x+2))/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / (x·(x + 2) + 3·x + 6)","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"x+3"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*")-1)/(x+2)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"(-1)"*"^(3/2)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*"(-1)"*)
                    cat <<'OUT'
                    {"result":"-1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *"^(3/2)"*)
                    cat <<'OUT'
                    {"result":"1 / (x + 2)","required_conditions":[{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    *arctan*sqrt*"/(x+2))/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / (x·(x + 2) + 3·x + 6)","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *arctan*sqrt*"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *sin*"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *cos*"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"exp("*"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / (x·(x + 2) + 3·x + 6)","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"/(x+3)"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *"x+3"*)
                    cat <<'OUT'
                    {"result":"1 / ((x + 2)·(x + 3))","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"},{"expr_display":"x + 3"}],"warnings":[]}
                    OUT
                    ;;
                    *)
                    cat <<'OUT'
                    {"result":"1 / (x + 2)","required_conditions":[{"expr_display":"x + 1"},{"expr_display":"x + 2"}],"warnings":[]}
                    OUT
                    ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--default-matrix",
                    "--matrix-base",
                    "arctan_sqrt_additive_trig",
                    "--cas-cli",
                    str(cas_cli),
                    "--forbid-warnings",
                    "--json",
                ],
                cwd=SCRIPT_PATH.parent.parent,
                text=True,
                capture_output=True,
                timeout=3,
            )

        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn('"base_filters": ["arctan_sqrt_additive_trig"]', completed.stdout)
        self.assertIn('"total": 11', completed.stdout)


if __name__ == "__main__":
    unittest.main()
