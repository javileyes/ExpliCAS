#!/usr/bin/env python3
"""Unified scorecard runner for metamorphic engine improvement work.

This script centralizes the core regression and pressure suites we use to grow:
- simplification
- equivalence
- derive reachability

It is intentionally profile-based:
- fast: iteration loop, cheap enough to run often
- guardrail: stable, frequent, merge-friendly
- pressure: slower, deeper, used for improvement campaigns
- full: both
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "docs" / "generated" / "engine_improvement_scorecard.json"
EMBEDDED_RUNTIME_DELTA_RATIO_THRESHOLD = 0.10
EMBEDDED_RUNTIME_DELTA_SECONDS_THRESHOLD = 5.0
EMBEDDED_ORCHESTRATOR_PROFILE_LIMIT = 480
NF_FIRST_FULL_TIMEOUT_SECONDS = 15 * 60
SIMPLIFY_ZERO_MIXED_PRESSURE_WINDOWS = (
    ("sum", 0, 100),
    ("sum", 700, 100),
    ("difference", 0, 50),
    ("product", 0, 100),
    ("shifted_quotient", 0, 100),
)


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    category: str
    profile_tags: tuple[str, ...]
    command: list[str]
    env: dict[str, str]
    parser: str
    description: str
    timeout_seconds: float | None = None


def build_simplify_zero_mixed_pressure_command() -> list[str]:
    command = [
        "cargo",
        "run",
        "--release",
        "-q",
        "-p",
        "cas_solver",
        "--example",
        "run_simplify_zero_mixed_corpus",
        "--",
    ]
    for composition, offset, limit in SIMPLIFY_ZERO_MIXED_PRESSURE_WINDOWS:
        command.extend(["--window", f"{composition}:{offset}:{limit}"])
    return command


SUITES: dict[str, SuiteSpec] = {
    "embedded_equivalence_context": SuiteSpec(
        name="embedded_equivalence_context",
        category="equivalence",
        profile_tags=("guardrail", "fast_embedded", "full"),
        command=[
            "cargo",
            "run",
            "--release",
            "-q",
            "-p",
            "cas_solver",
            "--example",
            "run_embedded_equivalence_context_corpus",
            "--",
        ],
        env={},
        parser="corpus",
        description="Embedded equivalence context corpus over wire eval.",
    ),
    "simplify_zero_mixed": SuiteSpec(
        name="simplify_zero_mixed",
        category="simplify",
        profile_tags=("pressure", "full"),
        command=build_simplify_zero_mixed_pressure_command(),
        env={},
        parser="corpus",
        description="Mixed zero pressure windows over fixed composed simplification/equivalence slices.",
    ),
    "derive_contract": SuiteSpec(
        name="derive_contract",
        category="derive",
        profile_tags=("guardrail", "full"),
        command=[
            "cargo",
            "test",
            "--release",
            "-q",
            "-p",
            "cas_solver",
            "--test",
            "derive_contract_tests",
            "derive_pairs_follow_expected_outcomes",
            "--",
            "--exact",
            "--nocapture",
        ],
        env={},
        parser="derive",
        description="Derive reachability/equivalence corpus and step quality stats.",
    ),
    "simplify_strict": SuiteSpec(
        name="simplify_strict",
        category="simplify",
        profile_tags=("guardrail", "full"),
        command=[
            "cargo",
            "test",
            "--release",
            "-p",
            "cas_solver",
            "--test",
            "metamorphic_simplification_tests",
            "metatest_unified_benchmark",
            "--",
            "--ignored",
            "--exact",
            "--nocapture",
        ],
        env={
            "METATEST_VERBOSE": "1",
            "METATEST_PROGRESS_EVERY": "1000",
            "METATEST_MAX_EXAMPLES": "20",
        },
        parser="unified_benchmark",
        description="Unified metamorphic regression benchmark in strict mode.",
    ),
    "simplify_nf_first": SuiteSpec(
        name="simplify_nf_first",
        category="simplify",
        profile_tags=("full",),
        command=[
            "cargo",
            "test",
            "--release",
            "-p",
            "cas_solver",
            "--test",
            "metamorphic_simplification_tests",
            "metatest_unified_benchmark_nf_first",
            "--",
            "--ignored",
            "--exact",
            "--nocapture",
        ],
        env={
            "METATEST_VERBOSE": "1",
            "METATEST_PROGRESS_EVERY": "1000",
            "METATEST_MAX_EXAMPLES": "20",
        },
        parser="unified_benchmark",
        description="Unified metamorphic regression benchmark in NF-first mode.",
        timeout_seconds=NF_FIRST_FULL_TIMEOUT_SECONDS,
    ),
    "simplify_add_small": SuiteSpec(
        name="simplify_add_small",
        category="simplify",
        profile_tags=("fast", "fast_embedded"),
        command=[
            "cargo",
            "test",
            "-q",
            "-p",
            "cas_solver",
            "--test",
            "metamorphic_simplification_tests",
            "metatest_csv_combinations_small",
            "--",
            "--exact",
            "--nocapture",
        ],
        env={},
        parser="cargo_test_basic",
        description="Small additive metamorphic combination smoke for rapid iteration.",
    ),
    "contextual_strict_fast": SuiteSpec(
        name="contextual_strict_fast",
        category="equivalence",
        profile_tags=("fast", "fast_embedded"),
        command=[
            "cargo",
            "test",
            "--release",
            "-q",
            "-p",
            "cas_solver",
            "--test",
            "metamorphic_simplification_tests",
            "metatest_csv_contextual_pairs_strict",
            "--",
            "--ignored",
            "--exact",
            "--nocapture",
        ],
        env={},
        parser="cargo_test_basic",
        description="Strict contextual metamorphic lane for quick wrapper-level sanity.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a unified engine-improvement scorecard."
    )
    parser.add_argument(
        "--profile",
        choices=("fast", "fast_embedded", "guardrail", "pressure", "full"),
        default="guardrail",
        help="Predefined suite set to run.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        choices=sorted(SUITES),
        help="Specific suite to run. Repeatable. Overrides --profile selection.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON output path.",
    )
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Optional Markdown summary output path. Defaults to JSON sibling with .md extension.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional previous JSON scorecard to diff against.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and exit without executing them.",
    )
    parser.add_argument(
        "--orchestrator-profile",
        action="store_true",
        help=(
            "Enable orchestrator shortcut profiling for the embedded equivalence "
            "corpus suite."
        ),
    )
    return parser.parse_args()


def selected_suites(args: argparse.Namespace) -> list[SuiteSpec]:
    if args.suite:
        return [SUITES[name] for name in args.suite]
    return [
        spec
        for spec in SUITES.values()
        if args.profile in spec.profile_tags
    ]


def git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def effective_suite_env(spec: SuiteSpec, args: argparse.Namespace) -> dict[str, str]:
    del args
    return dict(spec.env)


def orchestrator_profile_env(
    spec: SuiteSpec, args: argparse.Namespace
) -> dict[str, str] | None:
    if not args.orchestrator_profile or spec.name != "embedded_equivalence_context":
        return None
    return {
        "CAS_PROFILE_ORCHESTRATOR_SHORTCUTS": "1",
        "CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER": "pipeline.,root.",
    }


def orchestrator_profile_command(spec: SuiteSpec) -> list[str] | None:
    if spec.name != "embedded_equivalence_context":
        return None
    return [*spec.command, "--limit", str(EMBEDDED_ORCHESTRATOR_PROFILE_LIMIT)]


def run_command(
    spec: SuiteSpec,
    suite_env: dict[str, str] | None = None,
    command: list[str] | None = None,
) -> tuple[int, str, float, bool]:
    env = os.environ.copy()
    env.update(spec.env)
    if suite_env:
        env.update(suite_env)
    start = time.time()
    effective_command = command or spec.command
    process = subprocess.Popen(
        effective_command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    assert process.stdout is not None
    timeout_seconds = spec.timeout_seconds
    if timeout_seconds is not None:
        chunks: list[str] = []
        reader = threading.Thread(
            target=stream_process_stdout,
            args=(process.stdout, chunks),
            daemon=True,
        )
        reader.start()
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                terminate_process_group(process, signal.SIGKILL)
                returncode = process.wait(timeout=5)
        reader.join(timeout=5)
        process.stdout.close()
        elapsed = time.time() - start
        output = "".join(chunks)
        if timed_out:
            timeout_message = (
                f"\n[scorecard] suite timeout after {timeout_seconds:.1f}s; "
                "process group terminated.\n"
            )
            output += timeout_message
            sys.stdout.write(timeout_message)
            return returncode or -int(signal.SIGTERM), output, elapsed, True
        return returncode, output, elapsed, False

    chunks: list[str] = []
    for line in process.stdout:
        sys.stdout.write(line)
        chunks.append(line)

    returncode = process.wait()
    process.stdout.close()
    elapsed = time.time() - start
    return returncode, "".join(chunks), elapsed, False


def stream_process_stdout(stdout: Any, chunks: list[str]) -> None:
    for line in stdout:
        sys.stdout.write(line)
        chunks.append(line)


def terminate_process_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


def leading_int(text: str) -> int:
    match = re.search(r"-?\d+", text)
    if not match:
        raise ValueError(f"expected integer in {text!r}")
    return int(match.group(0))


def parse_corpus(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, pattern in (
        ("total_cases", r"Total cases:\s+(\d+)"),
        ("passed", r"Passed:\s+(\d+)"),
        ("failed", r"Failed:\s+(\d+)"),
        ("wrapper_count", r"Distinct wrappers:\s+(\d+)"),
        ("family_count", r"Distinct families:\s+(\d+)"),
        ("complexity_level_count", r"Distinct complexity levels:\s+(\d+)"),
        ("shell_depth_count", r"Distinct shell depths:\s+(\d+)"),
        ("max_shell_depth", r"Max shell depth:\s+(\d+)"),
        ("max_expression_depth", r"Max expression depth:\s+(\d+)"),
    ):
        match = re.search(pattern, output)
        if not match and key in {
            "wrapper_count",
            "family_count",
            "complexity_level_count",
            "shell_depth_count",
            "max_shell_depth",
            "max_expression_depth",
        }:
            continue
        if not match:
            raise ValueError(f"missing {key} in corpus output")
        metrics[key] = int(match.group(1))

    match = re.search(r"Largest wrapper share:\s+([0-9.]+)%", output)
    if match:
        metrics["largest_wrapper_share_percent"] = float(match.group(1))

    match = re.search(r"Largest wrapper x complexity share:\s+([0-9.]+)%", output)
    if match:
        metrics["largest_wrapper_complexity_share_percent"] = float(match.group(1))

    match = re.search(r"Average wrapper overhead nodes:\s+([0-9.]+)", output)
    if match:
        metrics["average_wrapper_overhead_nodes"] = float(match.group(1))

    match = re.search(r"Wrappers:\s+(.+)", output)
    if match:
        metrics["wrapper_names"] = [
            part.strip() for part in match.group(1).split(",") if part.strip()
        ]

    wrapper_rows = parse_named_summary_rows(output, "By wrapper:")
    if wrapper_rows:
        metrics["wrapper_rows"] = wrapper_rows

    match = re.search(r"Elapsed:\s+([0-9.]+)([a-z]+)", output)
    if match:
        metrics["reported_elapsed"] = f"{match.group(1)}{match.group(2)}"
        metrics["reported_elapsed_seconds"] = duration_to_seconds(
            float(match.group(1)), match.group(2)
        )

    complexity_rows = parse_complexity_rows(output)
    if complexity_rows:
        metrics["complexity_rows"] = complexity_rows

    composition_rows = parse_composition_rows(output)
    if composition_rows:
        metrics["composition_rows"] = composition_rows

    window_rows = parse_window_rows(output)
    if window_rows:
        metrics["window_rows"] = window_rows

    steady_engine_heavy_rows = parse_steady_engine_heavy_rows(output)
    if steady_engine_heavy_rows:
        metrics["steady_engine_heavy_rows"] = steady_engine_heavy_rows

    shell_depth_rows = parse_shell_depth_rows(output)
    if shell_depth_rows:
        metrics["shell_depth_rows"] = shell_depth_rows

    wrapper_shell_depth_rows = parse_wrapper_shell_depth_rows(output)
    if wrapper_shell_depth_rows:
        metrics["wrapper_shell_depth_rows"] = wrapper_shell_depth_rows

    sparse_wrapper_noise_budget_rows = parse_sparse_wrapper_noise_budget_rows(output)
    if sparse_wrapper_noise_budget_rows:
        metrics["sparse_wrapper_noise_budget_rows"] = sparse_wrapper_noise_budget_rows

    wrapper_complexity_rows = parse_wrapper_complexity_rows(output)
    if wrapper_complexity_rows:
        metrics["wrapper_complexity_rows"] = wrapper_complexity_rows

    wrapper_complexity_family_rows = parse_wrapper_complexity_family_rows(output)
    if wrapper_complexity_family_rows:
        metrics["wrapper_complexity_family_rows"] = wrapper_complexity_family_rows

    wrapper_family_rows = parse_wrapper_family_rows(output)
    if wrapper_family_rows:
        metrics["wrapper_family_rows"] = wrapper_family_rows

    orchestrator_profile = parse_orchestrator_profile(output)
    if orchestrator_profile:
        metrics["orchestrator_profile"] = orchestrator_profile
    return metrics


def duration_to_seconds(value: float, unit: str) -> float:
    scale = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "µs": 1e-6,
        "ns": 1e-9,
    }.get(unit)
    if scale is None:
        raise ValueError(f"unsupported duration unit: {unit}")
    return value * scale


def parse_complexity_rows(output: str) -> dict[str, dict[str, Any]]:
    complexity_rows: dict[str, dict[str, Any]] = {}
    in_block = False
    row_pattern = re.compile(
        r"^\s+(?P<level>\S+): total=(?P<total>\d+) passed=(?P<passed>\d+) failed=(?P<failed>\d+) "
        r"avg_wrapper_overhead_nodes=(?P<avg_wrapper_overhead_nodes>[0-9.]+) "
        r"avg_shell_depth=(?P<avg_shell_depth>[0-9.]+) "
        r"max_shell_depth=(?P<max_shell_depth>\d+)$"
    )

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "By complexity level:":
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped:
            break
        match = row_pattern.match(line)
        if not match:
            continue
        complexity_rows[match.group("level")] = {
            "total": int(match.group("total")),
            "passed": int(match.group("passed")),
            "failed": int(match.group("failed")),
            "avg_wrapper_overhead_nodes": float(
                match.group("avg_wrapper_overhead_nodes")
            ),
            "avg_shell_depth": float(match.group("avg_shell_depth")),
            "max_shell_depth": int(match.group("max_shell_depth")),
        }
    return complexity_rows


def parse_composition_rows(output: str) -> dict[str, dict[str, Any]]:
    composition_rows: dict[str, dict[str, Any]] = {}
    in_block = False

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "By composition:":
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped:
            break
        if ":" not in stripped:
            continue
        label, payload = stripped.split(":", 1)
        row = parse_summary_kv_payload(payload.strip())
        if row:
            composition_rows[label] = row
    return composition_rows


def parse_window_rows(output: str) -> dict[str, dict[str, Any]]:
    window_rows: dict[str, dict[str, Any]] = {}
    in_block = False

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "By window:":
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped:
            break
        if ":" not in stripped:
            continue
        label, payload = stripped.split(":", 1)
        row = parse_summary_kv_payload(payload.strip())
        if row:
            window_rows[label] = row
    return window_rows


def parse_named_summary_rows(output: str, heading: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    in_block = False

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == heading:
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped:
            break
        if ":" not in stripped:
            continue
        label, payload = stripped.split(":", 1)
        row = parse_summary_kv_payload(payload.strip())
        if row:
            rows[label] = row
    return rows


def parse_summary_kv_payload(payload: str) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for token in payload.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in {"total", "passed", "failed"}:
            row[key] = int(value)
        elif key in {
            "elapsed",
            "wire_elapsed",
            "parse_elapsed",
            "simplify_elapsed",
        }:
            match = re.fullmatch(r"([0-9.]+)([a-zµ]+)", value)
            if not match:
                continue
            seconds = duration_to_seconds(float(match.group(1)), match.group(2))
            normalized_key = {
                "elapsed": "elapsed_seconds",
                "wire_elapsed": "wire_elapsed_seconds",
                "parse_elapsed": "parse_elapsed_seconds",
                "simplify_elapsed": "simplify_elapsed_seconds",
            }[key]
            row[normalized_key] = seconds
        elif key in {
            "avg_case_ms",
            "avg_wire_ms",
            "avg_parse_ms",
            "avg_simplify_ms",
        }:
            row[key] = float(value)
    return row


def parse_steady_engine_heavy_rows(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    in_block = False
    row_pattern = re.compile(
        r"^\s+\[(?:(?P<window_label>.+?) )?#(?P<case_number>\d+) (?P<composition>\S+)\] "
        r"runs=(?P<runs>\d+) "
        r"median_simplify=(?P<median_simplify>[0-9.]+[a-zµ]+) "
        r"median_wire=(?P<median_wire>[0-9.]+[a-zµ]+) "
        r"median_parse=(?P<median_parse>[0-9.]+[a-zµ]+) "
        r"median_elapsed=(?P<median_elapsed>[0-9.]+[a-zµ]+) "
        r"expr=(?P<expr>.+)$"
    )

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "Steady-state engine-heavy reruns:":
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped:
            break
        match = row_pattern.match(line)
        if not match:
            continue
        rows.append(
            {
                "window_label": match.group("window_label") or "",
                "case_number": int(match.group("case_number")),
                "composition": match.group("composition"),
                "runs": int(match.group("runs")),
                "median_simplify_seconds": parse_duration_token(
                    match.group("median_simplify")
                ),
                "median_wire_seconds": parse_duration_token(match.group("median_wire")),
                "median_parse_seconds": parse_duration_token(match.group("median_parse")),
                "median_elapsed_seconds": parse_duration_token(
                    match.group("median_elapsed")
                ),
                "expression": match.group("expr"),
            }
        )
    return rows


def parse_duration_token(token: str) -> float:
    match = re.fullmatch(r"([0-9.]+)([a-zµ]+)", token)
    if not match:
        raise ValueError(f"invalid duration token: {token}")
    return duration_to_seconds(float(match.group(1)), match.group(2))


def parse_shell_depth_rows(output: str) -> dict[int, dict[str, int]]:
    shell_depth_rows: dict[int, dict[str, int]] = {}
    in_block = False
    row_pattern = re.compile(
        r"^\s+depth (?P<depth>\d+): total=(?P<total>\d+) passed=(?P<passed>\d+) failed=(?P<failed>\d+)$"
    )

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "By shell depth:":
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped:
            break
        match = row_pattern.match(line)
        if not match:
            continue
        depth = int(match.group("depth"))
        shell_depth_rows[depth] = {
            "total": int(match.group("total")),
            "passed": int(match.group("passed")),
            "failed": int(match.group("failed")),
        }
    return shell_depth_rows


def parse_wrapper_complexity_rows(output: str) -> list[dict[str, Any]]:
    wrapper_complexity_rows: list[dict[str, Any]] = []
    in_block = False
    row_pattern = re.compile(
        r"^\s+(?P<wrapper>.+?) x (?P<level>\S+): total=(?P<total>\d+) passed=(?P<passed>\d+) failed=(?P<failed>\d+) "
        r"avg_wrapper_overhead_nodes=(?P<avg_wrapper_overhead_nodes>[0-9.]+) "
        r"avg_shell_depth=(?P<avg_shell_depth>[0-9.]+) "
        r"max_shell_depth=(?P<max_shell_depth>\d+)$"
    )

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "Top wrapper x complexity buckets:":
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped:
            break
        match = row_pattern.match(line)
        if not match:
            continue
        wrapper_complexity_rows.append(
            {
                "wrapper": match.group("wrapper"),
                "level": match.group("level"),
                "total": int(match.group("total")),
                "passed": int(match.group("passed")),
                "failed": int(match.group("failed")),
                "avg_wrapper_overhead_nodes": float(
                    match.group("avg_wrapper_overhead_nodes")
                ),
                "avg_shell_depth": float(match.group("avg_shell_depth")),
                "max_shell_depth": int(match.group("max_shell_depth")),
            }
        )
    return wrapper_complexity_rows


def parse_wrapper_shell_depth_rows(output: str) -> list[dict[str, Any]]:
    wrapper_shell_depth_rows: list[dict[str, Any]] = []
    in_block = False
    row_pattern = re.compile(
        r"^\s+(?P<wrapper>.+?) x depth (?P<shell_depth>\d+): total=(?P<total>\d+) passed=(?P<passed>\d+) failed=(?P<failed>\d+)$"
    )

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "Sparse wrapper x shell-depth buckets:":
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped:
            break
        match = row_pattern.match(line)
        if not match:
            continue
        wrapper_shell_depth_rows.append(
            {
                "wrapper": match.group("wrapper"),
                "shell_depth": int(match.group("shell_depth")),
                "total": int(match.group("total")),
                "passed": int(match.group("passed")),
                "failed": int(match.group("failed")),
            }
        )
    return wrapper_shell_depth_rows


def parse_sparse_wrapper_noise_budget_rows(output: str) -> list[dict[str, Any]]:
    noise_budget_rows: list[dict[str, Any]] = []
    in_block = False
    row_pattern = re.compile(
        r"^\s+(?P<wrapper>.+?): total=(?P<total>\d+) passed=(?P<passed>\d+) failed=(?P<failed>\d+) "
        r"avg_wrapper_overhead_nodes=(?P<avg_wrapper_overhead_nodes>[0-9.]+) "
        r"max_wrapper_overhead_nodes=(?P<max_wrapper_overhead_nodes>\d+) "
        r"avg_shell_depth=(?P<avg_shell_depth>[0-9.]+) "
        r"max_shell_depth=(?P<max_shell_depth>\d+)$"
    )

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "Sparse wrapper noise-budget rows:":
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped:
            break
        match = row_pattern.match(line)
        if not match:
            continue
        noise_budget_rows.append(
            {
                "wrapper": match.group("wrapper"),
                "total": int(match.group("total")),
                "passed": int(match.group("passed")),
                "failed": int(match.group("failed")),
                "avg_wrapper_overhead_nodes": float(
                    match.group("avg_wrapper_overhead_nodes")
                ),
                "max_wrapper_overhead_nodes": int(
                    match.group("max_wrapper_overhead_nodes")
                ),
                "avg_shell_depth": float(match.group("avg_shell_depth")),
                "max_shell_depth": int(match.group("max_shell_depth")),
            }
        )
    return noise_budget_rows


def parse_wrapper_family_rows(output: str) -> list[dict[str, Any]]:
    wrapper_family_rows: list[dict[str, Any]] = []
    in_block = False
    row_pattern = re.compile(
        r"^\s+(?P<wrapper>.+?) x (?P<family>.+?): total=(?P<total>\d+) passed=(?P<passed>\d+) failed=(?P<failed>\d+)$"
    )

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "Sparse wrapper x family buckets:":
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped:
            break
        match = row_pattern.match(line)
        if not match:
            continue
        wrapper_family_rows.append(
            {
                "wrapper": match.group("wrapper"),
                "family": match.group("family"),
                "total": int(match.group("total")),
                "passed": int(match.group("passed")),
                "failed": int(match.group("failed")),
            }
        )
    return wrapper_family_rows


def parse_wrapper_complexity_family_rows(output: str) -> list[dict[str, Any]]:
    wrapper_complexity_family_rows: list[dict[str, Any]] = []
    in_block = False
    row_pattern = re.compile(
        r"^\s+(?P<wrapper>.+?) x (?P<level>\S+) x (?P<family>.+?): "
        r"total=(?P<total>\d+) passed=(?P<passed>\d+) failed=(?P<failed>\d+)$"
    )

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "Sparse wrapper x complexity family buckets:":
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped:
            break
        match = row_pattern.match(line)
        if not match:
            continue
        wrapper_complexity_family_rows.append(
            {
                "wrapper": match.group("wrapper"),
                "level": match.group("level"),
                "family": match.group("family"),
                "total": int(match.group("total")),
                "passed": int(match.group("passed")),
                "failed": int(match.group("failed")),
            }
        )
    return wrapper_complexity_family_rows


def parse_orchestrator_profile(output: str) -> dict[str, Any] | None:
    lines = output.splitlines()
    header_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if line.strip() == "Orchestrator Profiling Report"
        ),
        None,
    )
    if header_idx is None:
        return None

    row_pattern = re.compile(
        r"^(?P<section>.+?)\s+(?P<attempts>\d+)\s+(?P<hits>\d+)\s+(?P<misses>\d+)\s+"
        r"(?P<total_ms>[0-9.]+)\s+(?P<avg_us>[0-9.]+)$"
    )
    sections: list[dict[str, Any]] = []
    total_row: dict[str, Any] | None = None
    samples_by_section: dict[str, list[str]] = {}
    in_samples = False
    current_sample_section: str | None = None

    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "Sample expressions":
            in_samples = True
            current_sample_section = None
            continue
        if set(stripped) == {"─"}:
            continue

        if in_samples:
            if line.startswith("  - "):
                if current_sample_section is not None:
                    samples_by_section.setdefault(current_sample_section, []).append(
                        line[4:].strip()
                    )
                continue
            if not line.startswith(" "):
                current_sample_section = stripped
                samples_by_section.setdefault(current_sample_section, [])
            continue

        match = row_pattern.match(line.rstrip())
        if not match:
            continue

        attempts = int(match.group("attempts"))
        hits = int(match.group("hits"))
        misses = int(match.group("misses"))
        row = {
            "section": match.group("section").strip(),
            "attempts": attempts,
            "hits": hits,
            "misses": misses,
            "hit_rate_percent": round((hits * 100.0 / attempts), 2) if attempts else 0.0,
            "miss_rate_percent": round((misses * 100.0 / attempts), 2)
            if attempts
            else 0.0,
            "total_ms": float(match.group("total_ms")),
            "avg_us": float(match.group("avg_us")),
        }
        if row["section"] == "TOTAL":
            total_row = row
        else:
            sections.append(row)

    for row in sections:
        row["samples"] = samples_by_section.get(row["section"], [])

    if total_row is None:
        total_attempts = sum(row["attempts"] for row in sections)
        total_hits = sum(row["hits"] for row in sections)
        total_misses = sum(row["misses"] for row in sections)
        total_ms = round(sum(row["total_ms"] for row in sections), 3)
        total_avg_us = round(
            (total_ms * 1000.0 / total_attempts) if total_attempts else 0.0,
            2,
        )
        total_row = {
            "section": "TOTAL",
            "attempts": total_attempts,
            "hits": total_hits,
            "misses": total_misses,
            "hit_rate_percent": round(
                (total_hits * 100.0 / total_attempts) if total_attempts else 0.0,
                2,
            ),
            "miss_rate_percent": round(
                (total_misses * 100.0 / total_attempts) if total_attempts else 0.0,
                2,
            ),
            "total_ms": total_ms,
            "avg_us": total_avg_us,
        }
    else:
        total_row["samples"] = []

    hot_sections = sorted(
        sections,
        key=lambda row: (-row["total_ms"], -row["attempts"], row["section"]),
    )
    no_match_cost_sections = sorted(
        (row for row in sections if row["misses"] > 0),
        key=lambda row: (-row["total_ms"], -row["misses"], row["section"]),
    )

    return {
        "section_count": len(sections),
        "sections": sections,
        "totals": total_row,
        "top_hot_sections": hot_sections[:5],
        "top_no_match_cost_sections": no_match_cost_sections[:5],
        "sample_section_count": sum(1 for row in sections if row["samples"]),
    }


def parse_derive(output: str) -> dict[str, Any]:
    summary = re.search(
        r"derive corpus summary: derived=(\d+) unsupported=(\d+) not_equivalent=(\d+)",
        output,
    )
    stats = re.search(
        r"derive stats: reachability_rate=([0-9.]+) supported_equiv_rate=([0-9.]+) mean_step_count=([0-9.]+) long_path_rate=([0-9.]+)",
        output,
    )
    if not summary or not stats:
        raise ValueError("missing derive summary block")

    return {
        "derived": int(summary.group(1)),
        "unsupported": int(summary.group(2)),
        "not_equivalent": int(summary.group(3)),
        "reachability_rate": float(stats.group(1)),
        "supported_equiv_rate": float(stats.group(2)),
        "mean_step_count": float(stats.group(3)),
        "long_path_rate": float(stats.group(4)),
    }


def parse_unified_benchmark(output: str) -> dict[str, Any]:
    lines = output.splitlines()
    total_line = next((line for line in lines if "║ TOTAL │" in line), None)
    if total_line is None:
        raise ValueError("missing unified TOTAL row")

    parts = [part.strip(" ║") for part in total_line.split("│")]
    if len(parts) < 10:
        raise ValueError(f"unexpected TOTAL row shape: {total_line}")

    metrics = {
        "total_combos": leading_int(parts[1]),
        "nf_convergent": leading_int(parts[2]),
        "proved_symbolic": leading_int(parts[3]),
        "numeric_only": leading_int(parts[4]),
        "inconclusive": leading_int(parts[5]),
        "failed": leading_int(parts[6]),
        "timeouts": leading_int(parts[7]),
        "cycles": leading_int(parts[8]),
        "skipped": leading_int(parts[9]),
    }

    suites: dict[str, dict[str, int]] = {}
    suite_name_counts: dict[str, int] = {}
    table_rows = [
        line for line in lines if line.startswith("║ ") and " TOTAL " not in line
    ]
    for line in table_rows:
        if "Suite │" in line or "══" in line:
            continue
        cols = [part.strip(" ║") for part in line.split("│")]
        if len(cols) < 10:
            continue
        suite_name = cols[0].strip()
        if not suite_name or suite_name in {"Suite"}:
            continue
        suite_name_counts[suite_name] = suite_name_counts.get(suite_name, 0) + 1
        suite_key = suite_name
        if suite_name_counts[suite_name] > 1:
            suite_key = f"{suite_name}#{suite_name_counts[suite_name]}"
        suites[suite_key] = {
            "combos": leading_int(cols[1]),
            "nf_convergent": leading_int(cols[2]),
            "proved_symbolic": leading_int(cols[3]),
            "numeric_only": leading_int(cols[4]),
            "inconclusive": leading_int(cols[5]),
            "failed": leading_int(cols[6]),
            "timeouts": leading_int(cols[7]),
            "cycles": leading_int(cols[8]),
            "skipped": leading_int(cols[9]),
        }

    proved_breakdown = re.search(
        r"🔢 Proved-symbolic breakdown: quotient (\d+) \| diff (\d+) \| composed (\d+)",
        output,
    )
    if proved_breakdown:
        quotient = int(proved_breakdown.group(1))
        difference = int(proved_breakdown.group(2))
        composed = int(proved_breakdown.group(3))
        metrics["proved_breakdown"] = {
            "quotient": quotient,
            "difference": difference,
            "composed": composed,
        }
        total_symbolic = quotient + difference + composed
        if total_symbolic > 0:
            metrics["proof_shape_mix"] = {
                "non_composed_symbolic": quotient + difference,
                "non_composed_share_percent": round(
                    (quotient + difference) * 100.0 / total_symbolic, 1
                ),
                "composed_share_percent": round(composed * 100.0 / total_symbolic, 1),
            }

    metrics["suite_rows"] = suites
    return metrics


def parse_cargo_test_basic(output: str) -> dict[str, Any]:
    result = re.search(
        r"test result:\s+(ok|FAILED)\.\s+(\d+) passed;\s+(\d+) failed;\s+(\d+) ignored;\s+(\d+) measured;\s+(\d+) filtered out",
        output,
    )
    if not result:
        raise ValueError("missing cargo test result summary")

    metrics = {
        "cargo_status": result.group(1),
        "passed": int(result.group(2)),
        "failed": int(result.group(3)),
        "ignored": int(result.group(4)),
        "measured": int(result.group(5)),
        "filtered_out": int(result.group(6)),
    }

    nf_line = re.search(
        r"📐 NF-convergent:\s+(\d+)\s+\|\s+🔢 Proved-symbolic:\s+(\d+).*?\|\s+🌡️ Numeric-only:\s+(\d+)\s+\|\s+◐ Inconclusive:\s+(\d+)",
        output,
    )
    if nf_line:
        metrics.update(
            {
                "nf_convergent": int(nf_line.group(1)),
                "proved_symbolic": int(nf_line.group(2)),
                "numeric_only": int(nf_line.group(3)),
                "inconclusive": int(nf_line.group(4)),
            }
        )

    timeout_line = re.search(
        r"✅ .*?:\s+\d+\s+passed,\s+\d+\s+failed,\s+(\d+)\s+(?:skipped \(timeout\)|timed out)",
        output,
    )
    if timeout_line:
        metrics["timeouts"] = int(timeout_line.group(1))
    else:
        metrics["timeouts"] = 0

    return metrics


PARSERS = {
    "corpus": parse_corpus,
    "derive": parse_derive,
    "unified_benchmark": parse_unified_benchmark,
    "cargo_test_basic": parse_cargo_test_basic,
}


def suite_status(name: str, metrics: dict[str, Any], returncode: int) -> str:
    if returncode != 0:
        return "fail"
    if name in {"embedded_equivalence_context", "simplify_zero_mixed"}:
        return "pass" if metrics["failed"] == 0 else "fail"
    if name == "derive_contract":
        return "warn" if metrics["unsupported"] > 0 else "pass"
    if name in {"simplify_strict", "simplify_nf_first"}:
        if metrics["failed"] > 0:
            return "fail"
        if metrics["timeouts"] > 0 or metrics["numeric_only"] > 0:
            return "warn"
        return "pass"
    if name in {"simplify_add_small", "contextual_strict_fast"}:
        if metrics["failed"] > 0:
            return "fail"
        if metrics.get("timeouts", 0) > 0 or metrics.get("numeric_only", 0) > 0:
            return "warn"
        return "pass"
    return "pass"


def numeric_delta(value: Any, baseline: Any) -> Any:
    if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return value - baseline


def compute_deltas(
    baseline_data: dict[str, Any] | None,
    suite_name: str,
    metrics: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    if not baseline_data:
        return {}
    suites = baseline_data.get("suites", {})
    previous_suite = suites.get(suite_name, {})
    previous = previous_suite.get("metrics", {})
    deltas: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "suite_rows" or key == "proved_breakdown":
            continue
        delta = numeric_delta(value, previous.get(key))
        if delta is not None and delta != 0:
            deltas[key] = delta
    elapsed_delta = numeric_delta(elapsed_seconds, previous_suite.get("elapsed_seconds"))
    if elapsed_delta is not None and elapsed_delta != 0:
        deltas["elapsed_seconds"] = round(elapsed_delta, 3)
    return deltas


def compute_embedded_runtime_guardrail(
    baseline_data: dict[str, Any] | None,
    suite_name: str,
    elapsed_seconds: float,
) -> dict[str, Any] | None:
    if suite_name != "embedded_equivalence_context" or not baseline_data:
        return None

    baseline_suite = baseline_data.get("suites", {}).get(suite_name, {})
    baseline_elapsed = baseline_suite.get("elapsed_seconds")
    if not isinstance(baseline_elapsed, (int, float)) or baseline_elapsed <= 0:
        return None

    delta_seconds = elapsed_seconds - baseline_elapsed
    delta_ratio = delta_seconds / baseline_elapsed
    threshold_seconds = max(
        EMBEDDED_RUNTIME_DELTA_SECONDS_THRESHOLD,
        baseline_elapsed * EMBEDDED_RUNTIME_DELTA_RATIO_THRESHOLD,
    )

    assessment = "stable"
    if delta_seconds >= threshold_seconds:
        assessment = "regression"
    elif delta_seconds <= -threshold_seconds:
        assessment = "improvement"

    return {
        "assessment": assessment,
        "baseline_elapsed_seconds": round(baseline_elapsed, 3),
        "delta_seconds": round(delta_seconds, 3),
        "delta_ratio": round(delta_ratio, 4),
        "threshold_seconds": round(threshold_seconds, 3),
    }


def format_elapsed_with_delta(
    elapsed_seconds: float, suite_delta: dict[str, Any]
) -> str:
    summary = f"{elapsed_seconds:.2f}s"
    delta = suite_delta.get("elapsed_seconds")
    if isinstance(delta, (int, float)):
        summary += f" (Δ {delta:+.2f}s)"
    return summary


def effective_elapsed_seconds(metrics: dict[str, Any], process_elapsed_seconds: float) -> float:
    reported = metrics.get("reported_elapsed_seconds")
    if isinstance(reported, (int, float)):
        return float(reported)
    return process_elapsed_seconds


def sparse_wrapper_names(wrapper_rows: dict[str, dict[str, Any]]) -> set[str]:
    wrapper_totals = [
        total
        for row in wrapper_rows.values()
        if isinstance(total := row.get("total"), int)
    ]
    largest_wrapper_total = max(wrapper_totals) if wrapper_totals else 0
    sparse_cutoff = largest_wrapper_total * 0.25
    return {
        wrapper
        for wrapper, row in wrapper_rows.items()
        if row.get("total", 0) <= sparse_cutoff
    }


def sparse_wrapper_family_breadth_summary(
    wrapper_family_rows: list[dict[str, Any]],
    sparse_wrappers: set[str],
    family_count: int | None,
) -> str:
    family_names_by_wrapper: dict[str, set[str]] = {}
    case_totals_by_wrapper: dict[str, int] = {}
    for row in wrapper_family_rows:
        wrapper = row.get("wrapper")
        family = row.get("family")
        total = row.get("total")
        if not isinstance(wrapper, str) or wrapper not in sparse_wrappers:
            continue
        if not isinstance(family, str):
            continue
        family_names_by_wrapper.setdefault(wrapper, set()).add(family)
        if isinstance(total, int):
            case_totals_by_wrapper[wrapper] = (
                case_totals_by_wrapper.get(wrapper, 0) + total
            )

    fragments = []
    for wrapper in sorted(family_names_by_wrapper):
        covered_families = len(family_names_by_wrapper[wrapper])
        total_cases = case_totals_by_wrapper.get(wrapper, 0)
        if isinstance(family_count, int) and family_count > 0:
            fragments.append(
                f"{wrapper} families={covered_families}/{family_count} "
                f"cases={total_cases}"
            )
        else:
            fragments.append(f"{wrapper} families={covered_families} cases={total_cases}")
    return ", ".join(fragments)


def sparse_wrapper_family_gap_summary(
    wrapper_family_rows: list[dict[str, Any]],
    sparse_wrappers: set[str],
    family_count: int | None,
) -> str:
    if not isinstance(family_count, int) or family_count <= 0:
        return ""

    family_names_by_wrapper: dict[str, set[str]] = {}
    case_totals_by_wrapper: dict[str, int] = {}
    for row in wrapper_family_rows:
        wrapper = row.get("wrapper")
        family = row.get("family")
        total = row.get("total")
        if not isinstance(wrapper, str) or wrapper not in sparse_wrappers:
            continue
        if not isinstance(family, str):
            continue
        family_names_by_wrapper.setdefault(wrapper, set()).add(family)
        if isinstance(total, int):
            case_totals_by_wrapper[wrapper] = (
                case_totals_by_wrapper.get(wrapper, 0) + total
            )

    gap_rows = []
    for wrapper, family_names in family_names_by_wrapper.items():
        covered_families = len(family_names)
        missing_families = max(family_count - covered_families, 0)
        gap_rows.append(
            (
                missing_families,
                covered_families,
                wrapper,
                case_totals_by_wrapper.get(wrapper, 0),
            )
        )

    fragments = []
    for missing_families, covered_families, wrapper, total_cases in sorted(
        gap_rows,
        key=lambda row: (-row[0], row[1], row[2]),
    ):
        fragments.append(
            f"{wrapper} missing_families={missing_families}/{family_count} "
            f"covered={covered_families} cases={total_cases}"
        )
    return ", ".join(fragments)


def sparse_wrapper_complexity_family_breadth_summary(
    wrapper_complexity_family_rows: list[dict[str, Any]],
    sparse_wrappers: set[str],
    family_count: int | None,
    level: str,
) -> str:
    family_names_by_wrapper: dict[str, set[str]] = {}
    case_totals_by_wrapper: dict[str, int] = {}
    for row in wrapper_complexity_family_rows:
        wrapper = row.get("wrapper")
        row_level = row.get("level")
        family = row.get("family")
        total = row.get("total")
        if not isinstance(wrapper, str) or wrapper not in sparse_wrappers:
            continue
        if row_level != level or not isinstance(family, str):
            continue
        family_names_by_wrapper.setdefault(wrapper, set()).add(family)
        if isinstance(total, int):
            case_totals_by_wrapper[wrapper] = (
                case_totals_by_wrapper.get(wrapper, 0) + total
            )

    fragments = []
    for wrapper in sorted(family_names_by_wrapper):
        covered_families = len(family_names_by_wrapper[wrapper])
        total_cases = case_totals_by_wrapper.get(wrapper, 0)
        if isinstance(family_count, int) and family_count > 0:
            missing_families = max(family_count - covered_families, 0)
            fragments.append(
                f"{wrapper} l3_families={covered_families}/{family_count} "
                f"missing={missing_families} cases={total_cases}"
            )
        else:
            fragments.append(
                f"{wrapper} l3_families={covered_families} cases={total_cases}"
            )
    return ", ".join(fragments)


def shell_depth_summary_rows(
    shell_depth_rows: dict[int, dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    ordered_rows = sorted(shell_depth_rows.items())
    summary_rows = ordered_rows[:4]
    if ordered_rows and ordered_rows[-1][0] not in {depth for depth, _ in summary_rows}:
        summary_rows.append(ordered_rows[-1])
    return summary_rows


def render_markdown(scorecard: dict[str, Any]) -> str:
    lines = [
        "# Engine Improvement Scorecard",
        "",
        f"- Generated: {scorecard['generated_at']}",
        f"- Git branch: {scorecard['git']['branch']}",
        f"- Git commit: `{scorecard['git']['commit']}`",
        f"- Profile: `{scorecard['profile']}`",
        "",
    ]

    embedded_suite = scorecard["suites"].get("embedded_equivalence_context")
    if embedded_suite:
        embedded_metrics = embedded_suite["metrics"]
        lines.extend(
            [
                "## Embedded Runtime Guardrail",
                "",
                "- Dimension: contextual simplify/equivalence under real wrappers.",
                "- Interpretation: strong for simplify/orchestration quality; not a derive-path metric.",
                f"- Elapsed: {embedded_suite['elapsed_seconds']:.2f}s",
            ]
        )
        wrapper_count = embedded_metrics.get("wrapper_count")
        family_count = embedded_metrics.get("family_count")
        largest_wrapper_share = embedded_metrics.get("largest_wrapper_share_percent")
        wrapper_names = embedded_metrics.get("wrapper_names")
        if isinstance(wrapper_count, int) and isinstance(family_count, int):
            lines.append(
                f"- Coverage axes: {wrapper_count} wrappers across {family_count} families"
            )
        complexity_level_count = embedded_metrics.get("complexity_level_count")
        shell_depth_count = embedded_metrics.get("shell_depth_count")
        if isinstance(complexity_level_count, int) and isinstance(shell_depth_count, int):
            lines.append(
                f"- Context axes: {complexity_level_count} complexity levels across {shell_depth_count} shell depths"
            )
        if isinstance(largest_wrapper_share, (int, float)):
            lines.append(f"- Largest wrapper share: {largest_wrapper_share:.1f}%")
        largest_wrapper_complexity_share = embedded_metrics.get(
            "largest_wrapper_complexity_share_percent"
        )
        if isinstance(largest_wrapper_complexity_share, (int, float)):
            lines.append(
                f"- Largest wrapper x complexity share: {largest_wrapper_complexity_share:.1f}%"
            )
        if isinstance(wrapper_names, list) and wrapper_names:
            lines.append(f"- Wrappers: {', '.join(wrapper_names)}")
        wrapper_rows = embedded_metrics.get("wrapper_rows")
        sparse_wrappers: set[str] = set()
        if isinstance(wrapper_rows, dict) and wrapper_rows:
            sparse_wrappers = sparse_wrapper_names(wrapper_rows)
            sparse_wrapper_rows = [
                (wrapper, row)
                for wrapper, row in sorted(
                    wrapper_rows.items(),
                    key=lambda item: (item[1].get("total", 0), item[0]),
                )
                if wrapper in sparse_wrappers
            ][:3]
            sparse_summary = ", ".join(
                f"{wrapper} total={row['total']} failed={row['failed']}"
                for wrapper, row in sparse_wrapper_rows
                if "total" in row and "failed" in row
            )
            if sparse_summary:
                lines.append(f"- Sparse wrappers: {sparse_summary}")
        max_shell_depth = embedded_metrics.get("max_shell_depth")
        max_expression_depth = embedded_metrics.get("max_expression_depth")
        avg_wrapper_overhead_nodes = embedded_metrics.get("average_wrapper_overhead_nodes")
        shell_depth_fragments = []
        if isinstance(max_shell_depth, int):
            shell_depth_fragments.append(f"max_shell_depth={max_shell_depth}")
        if isinstance(max_expression_depth, int):
            shell_depth_fragments.append(f"max_expression_depth={max_expression_depth}")
        if isinstance(avg_wrapper_overhead_nodes, (int, float)):
            shell_depth_fragments.append(
                f"avg_wrapper_overhead_nodes={avg_wrapper_overhead_nodes:.2f}"
            )
        if shell_depth_fragments:
            lines.append(f"- Structural depth: {' '.join(shell_depth_fragments)}")
        complexity_rows = embedded_metrics.get("complexity_rows")
        if isinstance(complexity_rows, dict) and complexity_rows:
            ordered_rows = sorted(
                complexity_rows.items(),
                key=lambda item: (-item[1]["total"], item[0]),
            )
            complexity_summary = ", ".join(
                (
                    f"{level} total={row['total']} failed={row['failed']} "
                    f"avg_shell_depth={row['avg_shell_depth']:.2f}"
                )
                for level, row in ordered_rows[:3]
            )
            lines.append(f"- Complexity mix: {complexity_summary}")
        shell_depth_rows = embedded_metrics.get("shell_depth_rows")
        if isinstance(shell_depth_rows, dict) and shell_depth_rows:
            shell_depth_summary = ", ".join(
                f"depth {depth} total={row['total']} failed={row['failed']}"
                for depth, row in shell_depth_summary_rows(shell_depth_rows)
            )
            lines.append(f"- Shell-depth mix: {shell_depth_summary}")
        wrapper_shell_depth_rows = embedded_metrics.get("wrapper_shell_depth_rows")
        if isinstance(wrapper_shell_depth_rows, list) and wrapper_shell_depth_rows:
            sparse_shell_depth_summary = ", ".join(
                (
                    f"{row['wrapper']} x depth {row['shell_depth']} "
                    f"total={row['total']} failed={row['failed']}"
                )
                for row in wrapper_shell_depth_rows
            )
            lines.append(
                "- Sparse wrapper x shell-depth buckets: "
                f"{sparse_shell_depth_summary}"
            )
        sparse_wrapper_noise_budget_rows = embedded_metrics.get(
            "sparse_wrapper_noise_budget_rows"
        )
        if (
            isinstance(sparse_wrapper_noise_budget_rows, list)
            and sparse_wrapper_noise_budget_rows
        ):
            sparse_noise_summary = ", ".join(
                (
                    f"{row['wrapper']} total={row['total']} failed={row['failed']} "
                    f"avg_overhead={row['avg_wrapper_overhead_nodes']:.2f} "
                    f"max_overhead={row['max_wrapper_overhead_nodes']}"
                )
                for row in sparse_wrapper_noise_budget_rows
            )
            lines.append(
                "- Sparse wrapper noise budgets: "
                f"{sparse_noise_summary}"
            )
        wrapper_complexity_rows = embedded_metrics.get("wrapper_complexity_rows")
        if isinstance(wrapper_complexity_rows, list) and wrapper_complexity_rows:
            sparse_wrapper_complexity_rows = [
                row
                for row in wrapper_complexity_rows
                if row.get("wrapper") in sparse_wrappers
            ]
            if sparse_wrapper_complexity_rows:
                sparse_bucket_summary = ", ".join(
                    (
                        f"{row['wrapper']} x {row['level']} total={row['total']} "
                        f"failed={row['failed']} avg_shell_depth={row['avg_shell_depth']:.2f}"
                    )
                    for row in sorted(
                        sparse_wrapper_complexity_rows,
                        key=lambda row: (row["wrapper"], row["level"]),
                    )[:4]
                )
                lines.append(
                    "- Sparse wrapper x complexity buckets: "
                    f"{sparse_bucket_summary}"
                )
            dominant_bucket_summary = ", ".join(
                (
                    f"{row['wrapper']} x {row['level']} total={row['total']} "
                    f"failed={row['failed']} avg_shell_depth={row['avg_shell_depth']:.2f}"
                )
                for row in wrapper_complexity_rows[:3]
            )
            lines.append(
                "- Dominant wrapper x complexity buckets: "
                f"{dominant_bucket_summary}"
            )
        wrapper_complexity_family_rows = embedded_metrics.get(
            "wrapper_complexity_family_rows"
        )
        if (
            isinstance(wrapper_complexity_family_rows, list)
            and wrapper_complexity_family_rows
        ):
            family_count_for_complexity = (
                family_count if isinstance(family_count, int) else None
            )
            l3_family_breadth_summary = (
                sparse_wrapper_complexity_family_breadth_summary(
                    wrapper_complexity_family_rows,
                    sparse_wrappers,
                    family_count_for_complexity,
                    "l3_nested_or_composed",
                )
            )
            if l3_family_breadth_summary:
                lines.append(
                    "- Sparse wrapper l3 family breadth: "
                    f"{l3_family_breadth_summary}"
                )
        wrapper_family_rows = embedded_metrics.get("wrapper_family_rows")
        if isinstance(wrapper_family_rows, list) and wrapper_family_rows:
            family_count_for_breadth = (
                family_count if isinstance(family_count, int) else None
            )
            family_breadth_summary = sparse_wrapper_family_breadth_summary(
                wrapper_family_rows,
                sparse_wrappers,
                family_count_for_breadth,
            )
            if family_breadth_summary:
                lines.append(
                    "- Sparse wrapper family breadth: "
                    f"{family_breadth_summary}"
                )
            family_gap_summary = sparse_wrapper_family_gap_summary(
                wrapper_family_rows,
                sparse_wrappers,
                family_count_for_breadth,
            )
            if family_gap_summary:
                lines.append(
                    "- Sparse wrapper family gaps: "
                    f"{family_gap_summary}"
                )
            sparse_family_summary = ", ".join(
                (
                    f"{row['wrapper']} x {row['family']} total={row['total']} "
                    f"failed={row['failed']}"
                )
                for row in wrapper_family_rows
            )
            lines.append(
                "- Sparse wrapper x family buckets: "
                f"{sparse_family_summary}"
            )
        embedded_guardrail = embedded_suite.get("guardrail")
        if embedded_guardrail:
            lines.extend(
                [
                    f"- Baseline: {embedded_guardrail['baseline_elapsed_seconds']:.2f}s",
                    (
                        "- Assessment: "
                        f"`{embedded_guardrail['assessment']}` "
                        f"(Δ {embedded_guardrail['delta_seconds']:+.2f}s, "
                        f"{embedded_guardrail['delta_ratio']:+.1%}, "
                        f"threshold {embedded_guardrail['threshold_seconds']:.2f}s)"
                    ),
                ]
            )
        lines.append("")

        orchestrator_profile = embedded_metrics.get("orchestrator_profile")
        if isinstance(orchestrator_profile, dict):
            profile_slice = embedded_metrics.get("orchestrator_profile_slice")
            totals = orchestrator_profile["totals"]
            lines.extend(
                [
                    "## Embedded Orchestrator Profile",
                    "",
                    "- Purpose: identify hot shortcut groups and expensive no-match traffic under the embedded live guardrail.",
                ]
            )
            if isinstance(profile_slice, dict):
                lines.append(
                    (
                        "- Profiled slice: "
                        f"{profile_slice['total_cases']} cases "
                        f"(limit {profile_slice['limit']}), "
                        f"{profile_slice['wrapper_count']} wrappers, "
                        f"{profile_slice['family_count']} families, "
                        f"{profile_slice['elapsed_seconds']:.2f}s elapsed."
                    )
                )
            lines.append(
                (
                    "- Coverage: "
                    f"{orchestrator_profile['section_count']} sections, "
                    f"{totals['attempts']} attempts, "
                    f"{totals['hits']} hits, "
                    f"{totals['misses']} misses, "
                    f"{totals['total_ms']:.3f}ms total profiled time."
                )
            )
            for idx, row in enumerate(orchestrator_profile["top_hot_sections"][:3], start=1):
                sample_suffix = ""
                if row["samples"]:
                    sample_suffix = f" sample=`{row['samples'][0]}`"
                lines.append(
                    (
                        f"- Hot {idx}: `{row['section']}` {row['total_ms']:.3f}ms over "
                        f"{row['attempts']} attempts "
                        f"(hits {row['hits']}, misses {row['misses']}){sample_suffix}"
                    )
                )
            no_match_sections = orchestrator_profile["top_no_match_cost_sections"][:3]
            if no_match_sections:
                for idx, row in enumerate(no_match_sections, start=1):
                    sample_suffix = ""
                    if row["samples"]:
                        sample_suffix = f" sample=`{row['samples'][0]}`"
                    lines.append(
                        (
                            f"- No-match hotspot {idx}: `{row['section']}` "
                            f"{row['total_ms']:.3f}ms "
                            f"(misses {row['misses']} of {row['attempts']})"
                            f"{sample_suffix}"
                        )
                    )
            lines.append("")

    derive_suite = scorecard["suites"].get("derive_contract")
    if derive_suite:
        derive_metrics = derive_suite["metrics"]
        lines.extend(
            [
                "## Derive Reachability Guardrail",
                "",
                "- Dimension: source-to-target bridgeability and path quality.",
                "- Interpretation: measures planner/strategy reachability; not contextual wrapper strength.",
            ]
        )
        if "parse_error" in derive_metrics:
            lines.extend(
                [
                    f"- Parser status: `parse_error={derive_metrics['parse_error']}`",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    (
                        f"- Outcomes: derived={derive_metrics['derived']} "
                        f"unsupported={derive_metrics['unsupported']} "
                        f"not_equivalent={derive_metrics['not_equivalent']}"
                    ),
                    (
                        f"- Path quality: mean_step_count={derive_metrics['mean_step_count']:.2f} "
                        f"long_path_rate={derive_metrics['long_path_rate']:.2f}"
                    ),
                    "",
                ]
            )

    strict_suite = scorecard["suites"].get("simplify_strict")
    if strict_suite:
        strict_metrics = strict_suite["metrics"]
        proof_shape_mix = strict_metrics.get("proof_shape_mix")
        proved_breakdown = strict_metrics.get("proved_breakdown")
        if isinstance(proof_shape_mix, dict) and isinstance(proved_breakdown, dict):
            lines.extend(
                [
                    "## Simplify Benchmark Interpretation",
                    "",
                    "- Dimension: broad semantic closure under metamorphic composition.",
                    (
                        "- Proof-shape mix: "
                        f"quotient={proved_breakdown['quotient']} "
                        f"diff={proved_breakdown['difference']} "
                        f"composed={proved_breakdown['composed']} "
                        f"(non-composed {proof_shape_mix['non_composed_share_percent']:.1f}%, "
                        f"composed {proof_shape_mix['composed_share_percent']:.1f}%)."
                    ),
                    (
                        "- Caveat: high `proved-composed` is a strong semantic/robustness signal, "
                        "but a weaker direct runtime proxy because part of the closure is shaped by "
                        "the benchmark harness rather than by one raw engine path."
                    ),
                    (
                        "- Runtime interpretation: use `embedded_equivalence_context`, "
                        "`simplify_zero_mixed`, and orchestrator profiling to localize real engine cost."
                    ),
                    "",
                ]
            )

    mixed_suite = scorecard["suites"].get("simplify_zero_mixed")
    if mixed_suite:
        mixed_metrics = mixed_suite["metrics"]
        lines.extend(
            [
                "## Mixed Zero Pressure",
                "",
                "- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.",
                "- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.",
                "- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.",
            ]
        )
        composition_rows = mixed_metrics.get("composition_rows")
        if isinstance(composition_rows, dict) and composition_rows:
            ordered_rows = sorted(
                composition_rows.items(),
                key=lambda item: (
                    -item[1].get("elapsed_seconds", 0.0),
                    -item[1].get("failed", 0),
                    item[0],
                ),
            )
            top_rows = []
            for composition, row in ordered_rows[:4]:
                fragment = (
                    f"{composition} total={row['total']} failed={row['failed']}"
                )
                if "elapsed_seconds" in row:
                    fragment += f" elapsed={row['elapsed_seconds']:.2f}s"
                if "avg_case_ms" in row:
                    fragment += f" avg_case_ms={row['avg_case_ms']:.2f}"
                if "simplify_elapsed_seconds" in row:
                    fragment += f" simplify={row['simplify_elapsed_seconds']:.2f}s"
                if "avg_simplify_ms" in row:
                    fragment += f" avg_simplify_ms={row['avg_simplify_ms']:.2f}"
                top_rows.append(fragment)
            lines.append(f"- Composition hotspots: {', '.join(top_rows)}")
            engine_rows = sorted(
                composition_rows.items(),
                key=lambda item: (
                    -item[1].get("simplify_elapsed_seconds", 0.0),
                    -item[1].get("elapsed_seconds", 0.0),
                    item[0],
                ),
            )
            top_engine_rows = []
            for composition, row in engine_rows[:4]:
                if "simplify_elapsed_seconds" not in row:
                    break
                fragment = (
                    f"{composition} simplify={row['simplify_elapsed_seconds']:.2f}s"
                )
                if "avg_simplify_ms" in row:
                    fragment += f" avg_simplify_ms={row['avg_simplify_ms']:.2f}"
                if "elapsed_seconds" in row:
                    fragment += f" wall={row['elapsed_seconds']:.2f}s"
                top_engine_rows.append(fragment)
            if top_engine_rows:
                lines.append(f"- Engine hotspots: {', '.join(top_engine_rows)}")
        window_rows = mixed_metrics.get("window_rows")
        if isinstance(window_rows, dict) and window_rows:
            ordered_windows = sorted(
                window_rows.items(),
                key=lambda item: (
                    -item[1].get("failed", 0),
                    -item[1].get("elapsed_seconds", 0.0),
                    item[0],
                ),
            )
            window_fragments = []
            for window, row in ordered_windows[:5]:
                fragment = f"{window} failed={row['failed']}"
                if "elapsed_seconds" in row:
                    fragment += f" elapsed={row['elapsed_seconds']:.2f}s"
                if "avg_case_ms" in row:
                    fragment += f" avg_case_ms={row['avg_case_ms']:.2f}"
                if "avg_simplify_ms" in row:
                    fragment += f" avg_simplify_ms={row['avg_simplify_ms']:.2f}"
                window_fragments.append(fragment)
            lines.append(f"- Window slices: {', '.join(window_fragments)}")
        steady_engine_heavy_rows = mixed_metrics.get("steady_engine_heavy_rows")
        if isinstance(steady_engine_heavy_rows, list) and steady_engine_heavy_rows:
            steady_fragments = []
            for row in steady_engine_heavy_rows[:5]:
                label = f"#{row['case_number']} {row['composition']}"
                if row.get("window_label"):
                    label = f"{row['window_label']} {label}"
                fragment = (
                    f"{label} runs={row['runs']} "
                    f"median_simplify={row['median_simplify_seconds']:.2f}s "
                    f"median_wire={row['median_wire_seconds']:.2f}s "
                    f"median_wall={row['median_elapsed_seconds']:.2f}s"
                )
                steady_fragments.append(fragment)
            lines.append(
                "- Steady-state engine reruns: " + ", ".join(steady_fragments)
            )
        lines.append("")

    lines.extend(
        [
            "| Suite | Status | Elapsed | Key metrics |",
            "| --- | --- | --- | --- |",
        ]
    )

    for name, suite in scorecard["suites"].items():
        metrics = suite["metrics"]
        status = suite["status"]
        delta = suite.get("delta", {})
        if "parse_error" in metrics:
            summary = f"parse_error={metrics['parse_error']}"
        elif "total_cases" in metrics:
            pieces = [
                f"passed={metrics['passed']}",
                f"failed={metrics['failed']}",
                f"total={metrics['total_cases']}",
            ]
            if "wrapper_count" in metrics:
                pieces.append(f"wrappers={metrics['wrapper_count']}")
            if "family_count" in metrics:
                pieces.append(f"families={metrics['family_count']}")
            summary = " ".join(pieces)
        elif "derived" in metrics:
            summary = (
                f"derived={metrics['derived']} unsupported={metrics['unsupported']} "
                f"not_equivalent={metrics['not_equivalent']} mean_step_count={metrics['mean_step_count']:.2f}"
            )
        elif "cargo_status" in metrics:
            pieces = [
                f"passed={metrics['passed']}",
                f"failed={metrics['failed']}",
            ]
            if "nf_convergent" in metrics:
                pieces.extend(
                    [
                        f"nf={metrics['nf_convergent']}",
                        f"proved={metrics['proved_symbolic']}",
                        f"timeouts={metrics.get('timeouts', 0)}",
                    ]
                )
            summary = " ".join(pieces)
        else:
            summary = (
                f"nf={metrics['nf_convergent']} proved={metrics['proved_symbolic']} "
                f"numeric={metrics['numeric_only']} inconclusive={metrics['inconclusive']} "
                f"timeouts={metrics['timeouts']}"
            )
        elapsed_summary = format_elapsed_with_delta(suite["elapsed_seconds"], delta)
        lines.append(
            f"| `{name}` | `{status}` | {elapsed_summary} | {summary} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    specs = selected_suites(args)
    output_path = pathlib.Path(args.output)
    markdown_path = (
        pathlib.Path(args.markdown_output)
        if args.markdown_output
        else output_path.with_suffix(".md")
    )
    baseline_data = None
    if args.baseline:
        baseline_path = pathlib.Path(args.baseline)
        if baseline_path.exists():
            baseline_data = json.loads(baseline_path.read_text())

    scorecard: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile": args.profile,
        "orchestrator_profile_requested": args.orchestrator_profile,
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": git_value(["rev-parse", "HEAD"]),
        },
        "suites": {},
    }

    if args.dry_run:
        for spec in specs:
            suite_env = effective_suite_env(spec, args)
            env_prefix = " ".join(f"{k}={v}" for k, v in sorted(suite_env.items()))
            command = " ".join(spec.command)
            timeout_suffix = (
                f" [timeout={spec.timeout_seconds:.1f}s]"
                if spec.timeout_seconds is not None
                else ""
            )
            print(
                f"[dry-run] {spec.name}: {env_prefix} {command}{timeout_suffix}".strip()
            )
            profile_env = orchestrator_profile_env(spec, args)
            profile_command = orchestrator_profile_command(spec)
            if profile_env and profile_command:
                profile_prefix = " ".join(
                    f"{k}={v}" for k, v in sorted(profile_env.items())
                )
                print(
                    f"[dry-run] {spec.name}[orchestrator-profile]: "
                    f"{profile_prefix} {' '.join(profile_command)}"
                )
        return 0

    overall_exit = 0
    for spec in specs:
        print()
        print(f"=== {spec.name} ===")
        print(spec.description)
        suite_env = effective_suite_env(spec, args)
        returncode, output, elapsed, timed_out = run_command(spec, suite_env)
        if timed_out:
            metrics = {
                "parse_error": "timeout",
                "timeout_seconds": spec.timeout_seconds,
            }
            status = "fail"
            overall_exit = 1
        else:
            try:
                metrics = PARSERS[spec.parser](output)
            except ValueError as exc:
                parse_error = str(exc)
                if "stack overflow" in output:
                    parse_error = "stack overflow"
                elif "fatal runtime error" in output:
                    parse_error = "fatal runtime error"
                metrics = {"parse_error": parse_error}
                status = "fail"
                overall_exit = 1
            else:
                status = suite_status(spec.name, metrics, returncode)
        measured_elapsed = effective_elapsed_seconds(metrics, elapsed)
        guardrail = compute_embedded_runtime_guardrail(
            baseline_data, spec.name, measured_elapsed
        )
        if guardrail and guardrail["assessment"] == "regression" and status == "pass":
            status = "warn"
        if status == "fail":
            overall_exit = 1

        scorecard["suites"][spec.name] = {
            "category": spec.category,
            "status": status,
            "returncode": returncode,
            "elapsed_seconds": round(measured_elapsed, 3),
            "process_elapsed_seconds": round(elapsed, 3),
            "command": spec.command,
            "env": suite_env,
            "timed_out": timed_out,
            "timeout_seconds": spec.timeout_seconds,
            "metrics": metrics,
            "delta": compute_deltas(
                baseline_data, spec.name, metrics, measured_elapsed
            ),
            "guardrail": guardrail,
        }

        profile_env = orchestrator_profile_env(spec, args)
        profile_command = orchestrator_profile_command(spec)
        if profile_env and profile_command and "parse_error" not in metrics:
            print()
            print(
                f"=== {spec.name}.orchestrator_profile ===\n"
                f"Profiled observability slice (limit {EMBEDDED_ORCHESTRATOR_PROFILE_LIMIT})"
            )
            (
                profile_returncode,
                profile_output,
                profile_elapsed,
                profile_timed_out,
            ) = run_command(
                spec,
                profile_env,
                profile_command,
            )
            if profile_timed_out:
                metrics["orchestrator_profile_error"] = "timeout"
            else:
                try:
                    profile_metrics = PARSERS[spec.parser](profile_output)
                except ValueError as exc:
                    metrics["orchestrator_profile_error"] = str(exc)
                else:
                    if profile_returncode != 0:
                        metrics["orchestrator_profile_error"] = (
                            f"returncode={profile_returncode}"
                        )
                    elif "orchestrator_profile" in profile_metrics:
                        metrics["orchestrator_profile"] = profile_metrics["orchestrator_profile"]
                        metrics["orchestrator_profile_slice"] = {
                            "limit": EMBEDDED_ORCHESTRATOR_PROFILE_LIMIT,
                            "total_cases": profile_metrics["total_cases"],
                            "wrapper_count": profile_metrics.get("wrapper_count", 0),
                            "family_count": profile_metrics.get("family_count", 0),
                            "elapsed_seconds": round(
                                effective_elapsed_seconds(profile_metrics, profile_elapsed),
                                3,
                            ),
                        }
                    else:
                        metrics["orchestrator_profile_error"] = (
                            "missing orchestrator_profile in profiled slice output"
                        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(scorecard))

    print()
    print(f"JSON scorecard: {output_path}")
    print(f"Markdown scorecard: {markdown_path}")
    print()
    for name, suite in scorecard["suites"].items():
        print(f"{name}: {suite['status']}")

    return overall_exit


if __name__ == "__main__":
    raise SystemExit(main())
