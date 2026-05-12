#!/usr/bin/env python3
"""Focused tests for the embedded candidate smoke harness."""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parent / "engine_embedded_candidate_smoke.py"
SPEC = importlib.util.spec_from_file_location("engine_embedded_candidate_smoke", SCRIPT_PATH)
assert SPEC is not None
SMOKE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)


ROW = (
    "(sec(x) - 1/cos(x)),0,combined_additive_zero,test_pair,"
    "trig_contract,sec(x),1/cos(x),collapse exact zero additive subexpressions"
)


class EmbeddedCandidateSmokeTests(unittest.TestCase):
    def test_parse_candidate_row_accepts_exact_corpus_shape(self) -> None:
        row = SMOKE.parse_candidate_row(ROW)

        self.assertEqual(row[2], "combined_additive_zero")
        self.assertEqual(row[4], "trig_contract")
        self.assertEqual(row[5], "sec(x)")
        self.assertEqual(row[6], "1/cos(x)")
        self.assertEqual(len(row), 8)

    def test_parse_candidate_row_rejects_wrong_column_count(self) -> None:
        with self.assertRaisesRegex(SystemExit, "expected 8 CSV columns"):
            SMOKE.parse_candidate_row("expr,0,too_few")

    def test_expected_status_matches_timeout_and_any(self) -> None:
        self.assertTrue(SMOKE.expected_status_matches("timeout", "timeout"))
        self.assertTrue(SMOKE.expected_status_matches("slow", "slow"))
        self.assertTrue(SMOKE.expected_status_matches("fail", "any"))
        self.assertFalse(SMOKE.expected_status_matches("pass", "timeout"))

    def test_build_result_extracts_runner_metrics(self) -> None:
        stdout = textwrap.dedent(
            """
            Corpus file: /tmp/candidate.csv
            Failures file: /tmp/failures.csv
            Total cases: 1
            Passed: 1
            Failed: 0
            Elapsed: 12.34ms
            By complexity level:
              l3_nested_or_composed: total=1 passed=1 failed=0 avg_wrapper_overhead_nodes=10.00 avg_shell_depth=3.00 max_shell_depth=3
            """
        ).strip()

        result = SMOKE.build_result("pass", 0, 0.25, stdout, "")

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.total_cases, 1)
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.runner_elapsed, "12.34ms")
        self.assertEqual(result.runner_elapsed_seconds, 0.01234)
        self.assertEqual(
            result.complexity_rows,
            (
                "l3_nested_or_composed: total=1 passed=1 failed=0 "
                "avg_wrapper_overhead_nodes=10.00 avg_shell_depth=3.00 "
                "max_shell_depth=3",
            ),
        )

    def test_run_candidate_timeout_uses_path_cargo_stub(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cargo_stub = temp_path / "cargo"
            cargo_stub.write_text("#!/bin/sh\nsleep 10\n", encoding="utf-8")
            cargo_stub.chmod(cargo_stub.stat().st_mode | stat.S_IXUSR)

            row = SMOKE.parse_candidate_row(ROW)
            corpus_path = SMOKE.write_temp_corpus(row, keep_temp=False)
            old_path = os.environ.get("PATH", "")
            try:
                with mock.patch.dict(os.environ, {"PATH": f"{temp_path}{os.pathsep}{old_path}"}):
                    started = time.monotonic()
                    result = SMOKE.run_candidate(row, corpus_path, timeout_seconds=0.2)
                    elapsed = time.monotonic() - started
            finally:
                corpus_path.unlink(missing_ok=True)

        self.assertEqual(result.status, "timeout")
        self.assertIsNone(result.returncode)
        self.assertIsNone(result.total_cases)
        self.assertLess(elapsed, 3.0)

    def test_run_candidate_can_classify_pass_as_slow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cargo_stub = temp_path / "cargo"
            cargo_stub.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    cat <<'OUT'
                    Total cases: 1
                    Passed: 1
                    Failed: 0
                    Elapsed: 15.00ms
                      l3_nested_or_composed: total=1 passed=1 failed=0 avg_wrapper_overhead_nodes=10.00 avg_shell_depth=3.00 max_shell_depth=3
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cargo_stub.chmod(cargo_stub.stat().st_mode | stat.S_IXUSR)

            row = SMOKE.parse_candidate_row(ROW)
            corpus_path = SMOKE.write_temp_corpus(row, keep_temp=False)
            old_path = os.environ.get("PATH", "")
            try:
                with mock.patch.dict(
                    os.environ, {"PATH": f"{temp_path}{os.pathsep}{old_path}"}
                ):
                    result = SMOKE.run_candidate(
                        row,
                        corpus_path,
                        timeout_seconds=3.0,
                        slow_wall_seconds=0.0,
                    )
            finally:
                corpus_path.unlink(missing_ok=True)

        self.assertEqual(result.status, "slow")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.total_cases, 1)
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.slow_wall_seconds, 0.0)

    def test_parse_duration_seconds_accepts_ms_and_seconds(self) -> None:
        self.assertEqual(SMOKE.parse_duration_seconds("15.00ms"), 0.015)
        self.assertEqual(SMOKE.parse_duration_seconds("1.31s"), 1.31)
        self.assertIsNone(SMOKE.parse_duration_seconds("not a duration"))
        self.assertIsNone(SMOKE.parse_duration_seconds(None))

    def test_run_candidate_can_classify_pass_as_slow_by_runner_elapsed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cargo_stub = temp_path / "cargo"
            cargo_stub.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    cat <<'OUT'
                    Total cases: 1
                    Passed: 1
                    Failed: 0
                    Elapsed: 1.25s
                      l3_nested_or_composed: total=1 passed=1 failed=0 avg_wrapper_overhead_nodes=10.00 avg_shell_depth=3.00 max_shell_depth=3
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cargo_stub.chmod(cargo_stub.stat().st_mode | stat.S_IXUSR)

            row = SMOKE.parse_candidate_row(ROW)
            corpus_path = SMOKE.write_temp_corpus(row, keep_temp=False)
            old_path = os.environ.get("PATH", "")
            try:
                with mock.patch.dict(os.environ, {"PATH": f"{temp_path}{os.pathsep}{old_path}"}):
                    result = SMOKE.run_candidate(
                        row,
                        corpus_path,
                        timeout_seconds=3.0,
                        slow_runner_seconds=1.0,
                    )
            finally:
                corpus_path.unlink(missing_ok=True)

        self.assertEqual(result.status, "slow")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.runner_elapsed, "1.25s")
        self.assertEqual(result.runner_elapsed_seconds, 1.25)
        self.assertEqual(result.slow_runner_seconds, 1.0)


if __name__ == "__main__":
    unittest.main()
