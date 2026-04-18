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
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "docs" / "generated" / "engine_improvement_scorecard.json"
EMBEDDED_RUNTIME_DELTA_RATIO_THRESHOLD = 0.10
EMBEDDED_RUNTIME_DELTA_SECONDS_THRESHOLD = 5.0


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    category: str
    profile_tags: tuple[str, ...]
    command: list[str]
    env: dict[str, str]
    parser: str
    description: str


SUITES: dict[str, SuiteSpec] = {
    "embedded_equivalence_context": SuiteSpec(
        name="embedded_equivalence_context",
        category="equivalence",
        profile_tags=("guardrail", "full"),
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
        command=[
            "cargo",
            "run",
            "--release",
            "-q",
            "-p",
            "cas_solver",
            "--example",
            "run_simplify_zero_mixed_corpus",
            "--",
        ],
        env={},
        parser="corpus",
        description="Mixed zero corpus exercising composed simplification/equivalence.",
    ),
    "derive_contract": SuiteSpec(
        name="derive_contract",
        category="derive",
        profile_tags=("guardrail", "full"),
        command=[
            "cargo",
            "test",
            "-q",
            "-p",
            "cas_solver",
            "--test",
            "derive_contract_tests",
            "--",
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
        profile_tags=("pressure", "full"),
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
    ),
    "simplify_add_small": SuiteSpec(
        name="simplify_add_small",
        category="simplify",
        profile_tags=("fast",),
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
        profile_tags=("fast",),
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
        choices=("fast", "guardrail", "pressure", "full"),
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


def run_command(spec: SuiteSpec) -> tuple[int, str, float]:
    env = os.environ.copy()
    env.update(spec.env)
    start = time.time()
    process = subprocess.Popen(
        spec.command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    chunks: list[str] = []
    for line in process.stdout:
        sys.stdout.write(line)
        chunks.append(line)

    returncode = process.wait()
    elapsed = time.time() - start
    return returncode, "".join(chunks), elapsed


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
    ):
        match = re.search(pattern, output)
        if not match and key in {"wrapper_count", "family_count"}:
            continue
        if not match:
            raise ValueError(f"missing {key} in corpus output")
        metrics[key] = int(match.group(1))

    match = re.search(r"Largest wrapper share:\s+([0-9.]+)%", output)
    if match:
        metrics["largest_wrapper_share_percent"] = float(match.group(1))

    match = re.search(r"Wrappers:\s+(.+)", output)
    if match:
        metrics["wrapper_names"] = [
            part.strip() for part in match.group(1).split(",") if part.strip()
        ]

    match = re.search(r"Elapsed:\s+([0-9.]+)([a-z]+)", output)
    if match:
        metrics["reported_elapsed"] = f"{match.group(1)}{match.group(2)}"
        metrics["reported_elapsed_seconds"] = duration_to_seconds(
            float(match.group(1)), match.group(2)
        )
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
        metrics["proved_breakdown"] = {
            "quotient": int(proved_breakdown.group(1)),
            "difference": int(proved_breakdown.group(2)),
            "composed": int(proved_breakdown.group(3)),
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
        if isinstance(largest_wrapper_share, (int, float)):
            lines.append(f"- Largest wrapper share: {largest_wrapper_share:.1f}%")
        if isinstance(wrapper_names, list) and wrapper_names:
            lines.append(f"- Wrappers: {', '.join(wrapper_names)}")
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

    derive_suite = scorecard["suites"].get("derive_contract")
    if derive_suite:
        derive_metrics = derive_suite["metrics"]
        lines.extend(
            [
                "## Derive Reachability Guardrail",
                "",
                "- Dimension: source-to-target bridgeability and path quality.",
                "- Interpretation: measures planner/strategy reachability; not contextual wrapper strength.",
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
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": git_value(["rev-parse", "HEAD"]),
        },
        "suites": {},
    }

    if args.dry_run:
        for spec in specs:
            env_prefix = " ".join(f"{k}={v}" for k, v in sorted(spec.env.items()))
            command = " ".join(spec.command)
            print(f"[dry-run] {spec.name}: {env_prefix} {command}".strip())
        return 0

    overall_exit = 0
    for spec in specs:
        print()
        print(f"=== {spec.name} ===")
        print(spec.description)
        returncode, output, elapsed = run_command(spec)
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
            "env": spec.env,
            "metrics": metrics,
            "delta": compute_deltas(
                baseline_data, spec.name, metrics, measured_elapsed
            ),
            "guardrail": guardrail,
        }

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
