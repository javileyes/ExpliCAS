#!/usr/bin/env python3
"""Smoke-test one embedded-equivalence candidate row with a hard timeout.

This is a discovery harness, not a promotion lane. It lets an improvement
iteration classify a generated candidate as pass/fail/timeout before deciding
whether a minimal representative belongs in the live corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Literal


ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADER = [
    "expression",
    "expected_result",
    "wrapper",
    "pair_id",
    "family",
    "source",
    "target",
    "expected_strategy",
]

Status = Literal["pass", "fail", "timeout", "slow"]


@dataclass(frozen=True)
class SmokeResult:
    status: Status
    returncode: int | None
    wall_elapsed_seconds: float
    runner_elapsed: str | None
    runner_elapsed_seconds: float | None
    total_cases: int | None
    passed: int | None
    failed: int | None
    complexity_rows: tuple[str, ...]
    stdout: str
    stderr: str
    slow_wall_seconds: float | None = None
    slow_runner_seconds: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "returncode": self.returncode,
            "wall_elapsed_seconds": round(self.wall_elapsed_seconds, 3),
            "slow_wall_seconds": self.slow_wall_seconds,
            "runner_elapsed": self.runner_elapsed,
            "runner_elapsed_seconds": self.runner_elapsed_seconds,
            "slow_runner_seconds": self.slow_runner_seconds,
            "total_cases": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "complexity_rows": list(self.complexity_rows),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one embedded_equivalence_context CSV row in an isolated temp "
            "corpus with a process-group timeout."
        )
    )
    parser.add_argument(
        "--row",
        required=True,
        help="One CSV data row with the 8 embedded corpus columns, without header.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=6.0,
        help="Hard wall timeout for cargo + runner subprocess.",
    )
    parser.add_argument(
        "--expect",
        choices=("pass", "fail", "timeout", "slow", "any"),
        default="pass",
        help="Expected smoke status. Non-matching status exits nonzero.",
    )
    parser.add_argument(
        "--slow-wall-seconds",
        type=float,
        default=None,
        help=(
            "Classify an otherwise passing candidate as slow when its wall "
            "time exceeds this explicit promotion budget."
        ),
    )
    parser.add_argument(
        "--slow-runner-seconds",
        type=float,
        default=None,
        help=(
            "Classify an otherwise passing candidate as slow when the corpus "
            "runner's reported Elapsed time exceeds this explicit budget."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable summary JSON.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary one-row CSV and print its path.",
    )
    return parser.parse_args()


def parse_candidate_row(row_text: str) -> list[str]:
    try:
        rows = list(csv.reader([row_text]))
    except csv.Error as err:
        raise SystemExit(f"invalid CSV row: {err}") from err
    if len(rows) != 1 or len(rows[0]) != len(HEADER):
        got = 0 if not rows else len(rows[0])
        raise SystemExit(
            f"expected {len(HEADER)} CSV columns ({', '.join(HEADER)}), got {got}"
        )
    return rows[0]


def write_temp_corpus(row: list[str], keep_temp: bool) -> pathlib.Path:
    temp = tempfile.NamedTemporaryFile(
        "w", newline="", delete=False, suffix=".csv", prefix="embedded_candidate_"
    )
    with temp:
        writer = csv.writer(temp)
        writer.writerow(HEADER)
        writer.writerow(row)
    return pathlib.Path(temp.name)


def run_candidate(
    row: list[str],
    csv_path: pathlib.Path,
    timeout_seconds: float,
    slow_wall_seconds: float | None = None,
    slow_runner_seconds: float | None = None,
) -> SmokeResult:
    command = [
        "cargo",
        "run",
        "--release",
        "-q",
        "-p",
        "cas_solver",
        "--example",
        "run_embedded_equivalence_context_corpus",
        "--",
        "--csv",
        str(csv_path),
        "--wrapper",
        row[2],
        "--family",
        row[4],
    ]
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
        wall_elapsed = time.monotonic() - start
        status: Status = "pass" if process.returncode == 0 else "fail"
        runner_elapsed_seconds = find_runner_elapsed_seconds(stdout + "\n" + stderr)
        if (
            status == "pass"
            and slow_wall_seconds is not None
            and wall_elapsed > slow_wall_seconds
        ):
            status = "slow"
        if (
            status == "pass"
            and slow_runner_seconds is not None
            and runner_elapsed_seconds is not None
            and runner_elapsed_seconds > slow_runner_seconds
        ):
            status = "slow"
        return build_result(
            status,
            process.returncode,
            wall_elapsed,
            stdout,
            stderr,
            slow_wall_seconds=slow_wall_seconds,
            slow_runner_seconds=slow_runner_seconds,
        )
    except subprocess.TimeoutExpired:
        terminate_process_group(process)
        stdout, stderr = process.communicate()
        wall_elapsed = time.monotonic() - start
        return build_result(
            "timeout",
            None,
            wall_elapsed,
            stdout,
            stderr,
            slow_wall_seconds=slow_wall_seconds,
            slow_runner_seconds=slow_runner_seconds,
        )


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=1.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def build_result(
    status: Status,
    returncode: int | None,
    wall_elapsed_seconds: float,
    stdout: str,
    stderr: str,
    slow_wall_seconds: float | None = None,
    slow_runner_seconds: float | None = None,
) -> SmokeResult:
    output = stdout + "\n" + stderr
    runner_elapsed = find_text(r"^Elapsed:\s*(.+)$", output)
    return SmokeResult(
        status=status,
        returncode=returncode,
        wall_elapsed_seconds=wall_elapsed_seconds,
        runner_elapsed=runner_elapsed,
        runner_elapsed_seconds=parse_duration_seconds(runner_elapsed),
        total_cases=find_int(r"^Total cases:\s*(\d+)$", output),
        passed=find_int(r"^Passed:\s*(\d+)$", output),
        failed=find_int(r"^Failed:\s*(\d+)$", output),
        complexity_rows=tuple(
            line.strip()
            for line in output.splitlines()
            if line.startswith("  l") and "passed=" in line and "failed=" in line
        ),
        stdout=stdout,
        stderr=stderr,
        slow_wall_seconds=slow_wall_seconds,
        slow_runner_seconds=slow_runner_seconds,
    )


def find_text(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def find_int(pattern: str, text: str) -> int | None:
    value = find_text(pattern, text)
    return int(value) if value is not None else None


def find_runner_elapsed_seconds(output: str) -> float | None:
    return parse_duration_seconds(find_text(r"^Elapsed:\s*(.+)$", output))


def parse_duration_seconds(duration: str | None) -> float | None:
    if duration is None:
        return None
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s)\s*", duration)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "ms":
        return value / 1000.0
    return value


def expected_status_matches(status: Status, expect: str) -> bool:
    return expect == "any" or status == expect


def print_human(result: SmokeResult, expect: str, csv_path: pathlib.Path | None) -> None:
    fields = [
        f"status={result.status}",
        f"expect={expect}",
        f"wall={result.wall_elapsed_seconds:.3f}s",
    ]
    if result.runner_elapsed:
        fields.append(f"runner_elapsed={result.runner_elapsed}")
    if result.slow_wall_seconds is not None:
        fields.append(f"slow_wall={result.slow_wall_seconds:.3f}s")
    if result.slow_runner_seconds is not None:
        fields.append(f"slow_runner={result.slow_runner_seconds:.3f}s")
    if result.total_cases is not None:
        fields.append(f"cases={result.total_cases}")
    if result.passed is not None and result.failed is not None:
        fields.append(f"passed={result.passed}")
        fields.append(f"failed={result.failed}")
    if csv_path is not None:
        fields.append(f"csv={csv_path}")
    print(" ".join(fields))
    for row in result.complexity_rows:
        print(row)
    if result.status != "pass" and result.stderr.strip():
        print(result.stderr.strip())


def main() -> int:
    args = parse_args()
    row = parse_candidate_row(args.row)
    csv_path = write_temp_corpus(row, args.keep_temp)
    try:
        result = run_candidate(
            row,
            csv_path,
            args.timeout_seconds,
            slow_wall_seconds=args.slow_wall_seconds,
            slow_runner_seconds=args.slow_runner_seconds,
        )
    finally:
        if not args.keep_temp:
            try:
                csv_path.unlink()
            except FileNotFoundError:
                pass

    if args.json:
        print(json.dumps(result.as_dict(), sort_keys=True))
    else:
        print_human(result, args.expect, csv_path if args.keep_temp else None)

    return 0 if expected_status_matches(result.status, args.expect) else 1


if __name__ == "__main__":
    sys.exit(main())
