import importlib.util
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "engine_limit_command_matrix_smoke.py"
SPEC = importlib.util.spec_from_file_location("engine_limit_command_matrix_smoke", MODULE_PATH)
SMOKE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)


class LimitCommandMatrixSmokeTests(unittest.TestCase):
    def test_default_matrix_covers_limit_policy_axes(self) -> None:
        cases = SMOKE.build_cases()

        self.assertEqual(len(cases), 12)
        names = {case.name for case in cases}
        self.assertIn("finite_removable_rational_cancellation", names)
        self.assertIn("finite_static_invalid_log_undefined", names)
        self.assertIn("negative_infinity_log_domain_conflict_residual", names)
        self.assertEqual(SMOKE.count_by(cases, "point_regime"), {"finite": 7, "infinity": 5})
        self.assertEqual(
            SMOKE.count_by(cases, "outcome"),
            {"residual": 3, "supported": 8, "undefined": 1},
        )
        self.assertEqual(
            sum(1 for case in cases if case.expected_step_substrings),
            12,
        )
        self.assertEqual(
            [
                case.name
                for case in cases
                if case.outcome == "supported" and not case.expected_step_substrings
            ],
            [],
        )
        self.assertGreaterEqual(len({case.family for case in cases}), 7)

    def test_run_matrix_accepts_expected_warning_and_required_display(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    case "$*" in
                    *"limit(x^2 + x + 1, x, -2)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"3","warnings":[],"required_display":[],"steps":[{"rule":"Evaluar límite finito","before":"x^2 + x + 1","after":"3"}]}
                    OUT
                    ;;
                    *"limit((x^2-1)/(x-1), x, 1)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"2","warnings":[],"required_display":["x ≠ 1"],"steps":[{"rule":"Evaluar límite finito","before":"(x^2 - 1)/(x - 1)","after":"2"}]}
                    OUT
                    ;;
                    *)
                    echo '{"ok":false,"result":"unexpected","warnings":[],"required_display":[]}'
                    ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(
                (
                    "finite_polynomial_supported",
                    "finite_removable_rational_cancellation",
                )
            )
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "pass", matrix)
        self.assertEqual(matrix["total"], 2)
        self.assertEqual(matrix["status_counts"]["pass"], 2)
        self.assertEqual(matrix["supported_case_count"], 2)
        self.assertEqual(matrix["residual_case_count"], 0)
        self.assertEqual(matrix["warning_expected_case_count"], 0)
        self.assertEqual(matrix["required_display_case_count"], 1)
        self.assertEqual(matrix["step_checked_case_count"], 2)
        self.assertEqual(matrix["supported_step_unchecked_case_count"], 0)
        self.assertEqual(matrix["supported_step_unchecked_cases"], [])
        self.assertEqual(matrix["expected_step_substring_count"], 2)
        self.assertEqual(matrix["required_display_counts"], {"x ≠ 1": 1})

    def test_run_matrix_reports_required_display_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    cat <<'OUT'
                    {"ok":true,"result":"1","warnings":[],"required_display":[]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(("finite_rational_required_condition",))
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "fail")
        self.assertEqual(matrix["status_counts"]["fail"], 1)
        self.assertEqual(matrix["issue_kind_counts"], {"required_display_mismatch": 1})
        self.assertEqual(matrix["problem_cases"][0]["name"], "finite_rational_required_condition")

    def test_run_matrix_reports_missing_expected_step_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    cat <<'OUT'
                    {"ok":true,"result":"3","warnings":[],"required_display":[],"steps":[]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(("finite_polynomial_supported",))
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "fail")
        self.assertEqual(matrix["status_counts"]["fail"], 1)
        self.assertEqual(matrix["issue_kind_counts"], {"step_trace_mismatch": 1})
        self.assertEqual(matrix["problem_cases"][0]["name"], "finite_polynomial_supported")


if __name__ == "__main__":
    unittest.main()
