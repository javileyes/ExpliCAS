#!/usr/bin/env python3
"""Unified scorecard runner for metamorphic engine improvement work.

This script centralizes the core regression and pressure suites we use to grow:
- simplification
- equivalence
- derive reachability
- calculus support-matrix visibility

It is intentionally profile-based:
- fast: iteration loop, cheap enough to run often
- guardrail: stable, frequent, merge-friendly
- pressure: slower, deeper, used for improvement campaigns
- full: both
"""

from __future__ import annotations

import argparse
import csv
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
EMBEDDED_EQUIVALENCE_CONTEXT_CORPUS = (
    ROOT / "docs" / "embedded_equivalence_context_corpus.csv"
)
ENGINE_COMBINATION_LEDGER = ROOT / "docs" / "ENGINE_COMBINATION_LEDGER.md"
EMBEDDED_RUNTIME_DELTA_RATIO_THRESHOLD = 0.10
EMBEDDED_RUNTIME_DELTA_SECONDS_THRESHOLD = 5.0
EMBEDDED_ORCHESTRATOR_PROFILE_LIMIT = 480
NF_FIRST_FULL_TIMEOUT_SECONDS = 15 * 60
COMBINED_ADDITIVE_FAMILY_TARGET_CASE_COUNT = 6
CALCULUS_POLICY_CLUSTER_CONSOLIDATION_THRESHOLD = 6
CALCULUS_RUNTIME_PRESSURE_WATCH_MAX_MS = 150.0
CALCULUS_RUNTIME_PRESSURE_RISK_MAX_MS = 500.0
CALCULUS_RUNTIME_PRESSURE_WATCH_P95_MS = 75.0
CALCULUS_RUNTIME_PRESSURE_RISK_P95_MS = 250.0
CALCULUS_RUNTIME_PRESSURE_WATCH_TOP3_SHARE_PERCENT = 50.0
CALCULUS_RUNTIME_PRESSURE_RISK_TOP3_SHARE_PERCENT = 80.0
CALCULUS_RUNTIME_PRESSURE_CONCENTRATION_MIN_CASES = 10
BACKEND_VERIFICATION_PRESSURE_WATCH_CHECKS_PER_ATTEMPT = 1.0
BACKEND_VERIFICATION_PRESSURE_RISK_CHECKS_PER_ATTEMPT = 2.0
BACKEND_VERIFICATION_PRESSURE_WATCH_MAX_NORMALIZATION_PASSES = 2
BACKEND_VERIFICATION_PRESSURE_RISK_MAX_NORMALIZATION_PASSES = 3
BACKEND_VERIFICATION_PRESSURE_WATCH_ELAPSED_MS = 10.0
BACKEND_VERIFICATION_PRESSURE_RISK_ELAPSED_MS = 50.0
CALCULUS_CLUSTER_RUNTIME_CANDIDATE_MIN_CASES = 2
CALCULUS_CLUSTER_RUNTIME_CANDIDATE_MIN_AVG_MS = 75.0
CALCULUS_CLUSTER_RUNTIME_CANDIDATE_RATIO = 2.0
CALCULUS_COMMAND_MATRIX_PROCESS_OVERHEAD_FLOOR_MAX_MS = 75.0
# Keep these clusters visible, but do not keep re-selecting them as raw
# consolidation candidates once a shared engine policy owns the family.
CALCULUS_POLICY_CLUSTERS_WITH_SHARED_POLICY = frozenset(
    {
        "block7_explicit_reciprocal_trig_log_substitution",
        "block7_hyperbolic_reciprocal_derivative_product",
        "block7_hyperbolic_reciprocal_fourth",
        "block7_hyperbolic_reciprocal_square",
        "block7_sqrt_chain_hyperbolic_reciprocal_derivative_product",
        "block7_sqrt_chain_reciprocal_trig_product",
        "block7_trig_reciprocal_derivative_product",
        "block9_explicit_reciprocal_trig_residual",
    }
)
CALCULUS_RADICAL_INVERSE_POLICY_CLUSTERS_WITH_SHARED_POLICY = frozenset(
    {
        "block8_inverse_hyperbolic_rational_interval",
        "block8_inverse_sqrt_tables",
        "block8_inverse_trig_root_reciprocal",
    }
)
CALCULUS_BASE_INTEGRATION_POLICY_CLUSTERS_WITH_SHARED_POLICY = frozenset(
    {
        "block4_exponential_by_parts",
        "block4_log_by_parts",
        "block4_log_power_product_by_parts",
    }
)
CALCULUS_DIFF_SYMBOLIC_RADIUS_POLICY_CLUSTERS_WITH_SHARED_POLICY = frozenset(
    {
        "block2_symbolic_radius_arctan_positive_quadratic",
    }
)
CALCULUS_DIFF_POSITIVE_QUADRATIC_POLICY_CLUSTERS_WITH_SHARED_POLICY = frozenset(
    {
        "block2_positive_quadratic_log_abs_pole_primitive",
        "block2_positive_quadratic_log_arctan_primitive",
        "block2_symbolic_radius_arctan_positive_quadratic",
    }
)
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
    "derive_shadow_pressure": SuiteSpec(
        name="derive_shadow_pressure",
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
            "derive_engine_identity_shadow_pressure_reports_reachability",
            "--",
            "--exact",
            "--nocapture",
        ],
        env={},
        parser="derive_shadow",
        description=(
            "Diagnostic engine-to-derive shadow pressure over representative "
            "engine equivalence rows."
        ),
    ),
    "derive_didactic_audit": SuiteSpec(
        name="derive_didactic_audit",
        category="didactic",
        profile_tags=("guardrail", "full"),
        command=[
            "cargo",
            "test",
            "--release",
            "-q",
            "-p",
            "cas_didactic",
            "--test",
            "derive_didactic_audit",
            "derive_didactic_audit_generates_markdown_report",
            "--",
            "--ignored",
            "--exact",
            "--nocapture",
        ],
        env={},
        parser="derive_didactic",
        description="Diagnostic derive didactic trace audit over curated derive pairs.",
    ),
    "simplify_didactic_audit": SuiteSpec(
        name="simplify_didactic_audit",
        category="didactic",
        profile_tags=("guardrail", "full"),
        command=[
            "cargo",
            "test",
            "--release",
            "-q",
            "-p",
            "cas_didactic",
            "--test",
            "didactic_step_quality_audit",
            "didactic_step_quality_audit_generates_markdown_report",
            "--",
            "--ignored",
            "--exact",
            "--nocapture",
        ],
        env={},
        parser="simplify_didactic",
        description="Compact simplify didactic trace audit over representative web-step cases.",
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
    "contextual_radical_fast": SuiteSpec(
        name="contextual_radical_fast",
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
            "metatest_csv_contextual_radical_pairs",
            "--",
            "--ignored",
            "--exact",
            "--nocapture",
        ],
        env={},
        parser="cargo_test_basic",
        description="Specialized contextual radical lane for cheap wrapper-level radical coverage.",
    ),
    "calculus_diff_contract": SuiteSpec(
        name="calculus_diff_contract",
        category="calculus",
        profile_tags=("fast", "fast_embedded", "guardrail", "full"),
        command=[
            "cargo",
            "test",
            "--release",
            "-q",
            "-p",
            "cas_solver",
            "--test",
            "diff_step_contract_tests",
            "--",
            "--nocapture",
        ],
        env={},
        parser="cargo_test_basic",
        description=(
            "Public differentiation contract lane for calculus support-matrix "
            "visibility."
        ),
    ),
    "calculus_diff_command_matrix_smoke": SuiteSpec(
        name="calculus_diff_command_matrix_smoke",
        category="calculus",
        profile_tags=("fast", "fast_embedded", "guardrail", "full"),
        command=[
            sys.executable,
            str(ROOT / "scripts" / "engine_diff_command_matrix_smoke.py"),
            "--ensure-release-cas-cli",
            "--timeout-seconds",
            "4",
            "--json",
            "--summary-json",
        ],
        env={},
        parser="calculus_diff_command_matrix",
        description=(
            "Command-level differentiation matrix over family, argument, "
            "domain, residual, trace, and presentation regimes."
        ),
    ),
    "calculus_diff_exhaustive_contract": SuiteSpec(
        name="calculus_diff_exhaustive_contract",
        category="calculus",
        profile_tags=("pressure", "full"),
        command=[
            "cargo",
            "test",
            "--release",
            "-q",
            "-p",
            "cas_solver",
            "--test",
            "diff_step_contract_tests",
            "inverse_reciprocal_trig_diff_evaluates_with_explicit_domain_conditions_exhaustive",
            "--",
            "--ignored",
            "--exact",
            "--nocapture",
        ],
        env={},
        parser="cargo_test_basic",
        description=(
            "Pressure-only exhaustive differentiation domain-condition "
            "contract over inverse reciprocal trig families."
        ),
    ),
    "calculus_limit_contract": SuiteSpec(
        name="calculus_limit_contract",
        category="calculus",
        profile_tags=("guardrail", "full"),
        command=[
            "cargo",
            "test",
            "--release",
            "-q",
            "-p",
            "cas_cli",
            "--test",
            "limit_contract_tests",
            "--",
            "--nocapture",
        ],
        env={},
        parser="cargo_test_basic",
        description=(
            "Public limit contract lane for calculus support-matrix visibility."
        ),
    ),
    "calculus_limit_compact_contract": SuiteSpec(
        name="calculus_limit_compact_contract",
        category="calculus",
        profile_tags=("fast", "fast_embedded"),
        command=[
            "cargo",
            "test",
            "--release",
            "-q",
            "-p",
            "cas_cli",
            "--test",
            "limit_contract_tests",
            "test_limit_sqrt_quadratic_over_noisy_scaled_linear_denominator",
            "--",
            "--exact",
            "--nocapture",
        ],
        env={},
        parser="cargo_test_basic",
        description=(
            "Cheap public limit compactness contract for fast calculus "
            "support-matrix visibility."
        ),
    ),
    "calculus_limit_presimplify_contract": SuiteSpec(
        name="calculus_limit_presimplify_contract",
        category="calculus",
        profile_tags=("fast", "fast_embedded", "guardrail", "full"),
        command=[
            "cargo",
            "test",
            "--release",
            "-q",
            "-p",
            "cas_cli",
            "--test",
            "presimplify_contract_tests",
            "--",
            "--nocapture",
        ],
        env={},
        parser="cargo_test_basic",
        description=(
            "Public limit safe pre-simplification contract lane for "
            "calculus/pre-calculus support-matrix visibility."
        ),
    ),
    "calculus_limit_command_matrix_smoke": SuiteSpec(
        name="calculus_limit_command_matrix_smoke",
        category="calculus",
        profile_tags=("fast", "fast_embedded", "guardrail", "full"),
        command=[
            sys.executable,
            str(ROOT / "scripts" / "engine_limit_command_matrix_smoke.py"),
            "--ensure-release-cas-cli",
            "--timeout-seconds",
            "4",
            "--json",
            "--summary-json",
        ],
        env={},
        parser="calculus_limit_command_matrix",
        description=(
            "Command-level limit policy matrix over finite, infinity, domain, "
            "and residual regimes."
        ),
    ),
    "calculus_integrate_compact_contract": SuiteSpec(
        name="calculus_integrate_compact_contract",
        category="calculus",
        profile_tags=("fast", "fast_embedded"),
        command=[
            "cargo",
            "test",
            "--release",
            "-q",
            "-p",
            "cas_cli",
            "--test",
            "integrate_contract_tests",
            "integrate_contract_polynomial_derivative_over_fractional_denominator_power_substitution",
            "--",
            "--exact",
            "--nocapture",
        ],
        env={},
        parser="cargo_test_basic",
        description=(
            "Cheap public integration compactness contract for fast calculus "
            "support-matrix visibility."
        ),
    ),
    "calculus_integrate_backend_observability": SuiteSpec(
        name="calculus_integrate_backend_observability",
        category="observability",
        profile_tags=("fast", "fast_embedded", "guardrail", "full"),
        command=[
            "cargo",
            "test",
            "-q",
            "-p",
            "cas_math",
            "--lib",
            "backend_observability_reports_boundary_metrics",
            "--",
            "--nocapture",
        ],
        env={},
        parser="algorithmic_backend_observability",
        description=(
            "Algorithmic integration backend boundary metrics over attempts, "
            "verification outcomes, residual reasons, and verifier runtime."
        ),
    ),
    "calculus_integrate_command_matrix_smoke": SuiteSpec(
        name="calculus_integrate_command_matrix_smoke",
        category="calculus",
        profile_tags=("fast", "fast_embedded", "guardrail", "full"),
        command=[
            sys.executable,
            str(ROOT / "scripts" / "engine_integrate_command_matrix_smoke.py"),
            "--ensure-release-cas-cli",
            "--timeout-seconds",
            "4",
            "--json",
            "--summary-json",
        ],
        env={},
        parser="calculus_integrate_command_matrix",
        description=(
            "Command-level integration matrix over family, argument, domain, "
            "residual, trace, and presentation regimes."
        ),
    ),
    "calculus_residual_matrix_smoke": SuiteSpec(
        name="calculus_residual_matrix_smoke",
        category="calculus",
        profile_tags=("fast", "fast_embedded", "guardrail", "full"),
        command=[
            sys.executable,
            str(ROOT / "scripts" / "engine_calculus_residual_probe_smoke.py"),
            "--default-matrix",
            "--ensure-release-cas-cli",
            "--timeout-seconds",
            "8",
            "--json",
            "--summary-json",
        ],
        env={},
        parser="calculus_residual_matrix",
        description=(
            "Public calculus residual matrix smoke over promoted residual "
            "wrappers and families."
        ),
    ),
    "calculus_integrate_contract": SuiteSpec(
        name="calculus_integrate_contract",
        category="calculus",
        profile_tags=("guardrail", "full"),
        command=[
            "cargo",
            "test",
            "--release",
            "-q",
            "-p",
            "cas_cli",
            "--test",
            "integrate_contract_tests",
            "--",
            "--nocapture",
        ],
        env={},
        parser="cargo_test_basic",
        description=(
            "Public integration contract lane for calculus support-matrix "
            "visibility."
        ),
    ),
    "calculus_integrate_exhaustive_contract": SuiteSpec(
        name="calculus_integrate_exhaustive_contract",
        category="calculus",
        profile_tags=("pressure", "full"),
        command=[
            "cargo",
            "test",
            "--release",
            "-q",
            "-p",
            "cas_cli",
            "--test",
            "integrate_contract_tests",
            "integrate_contract_supported_antiderivatives_verify_by_differentiation_exhaustive",
            "--",
            "--ignored",
            "--exact",
            "--nocapture",
        ],
        env={},
        parser="cargo_test_basic",
        description=(
            "Pressure-only exhaustive integration antiderivative verification "
            "over supported public families."
        ),
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
            "Enable orchestrator shortcut profiling for supported observability "
            "slices."
        ),
    )
    parser.add_argument(
        "--orchestrator-profile-filter",
        default="pipeline.,root.",
        help=(
            "Comma-separated orchestrator shortcut profile filter used with "
            "--orchestrator-profile."
        ),
    )
    parser.add_argument(
        "--orchestrator-profile-limit",
        type=int,
        default=EMBEDDED_ORCHESTRATOR_PROFILE_LIMIT,
        help=(
            "Embedded corpus case limit for the orchestrator profiling slice. "
            "Pressure-window suites keep their configured windows."
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


ORCHESTRATOR_PROFILE_SUITES = {
    "embedded_equivalence_context",
    "simplify_zero_mixed",
}


def orchestrator_profile_env(
    spec: SuiteSpec, args: argparse.Namespace
) -> dict[str, str] | None:
    if not args.orchestrator_profile or spec.name not in ORCHESTRATOR_PROFILE_SUITES:
        return None
    return {
        "CAS_PROFILE_ORCHESTRATOR_SHORTCUTS": "1",
        "CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER": args.orchestrator_profile_filter,
    }


def orchestrator_profile_command(
    spec: SuiteSpec, args: argparse.Namespace
) -> list[str] | None:
    if spec.name == "embedded_equivalence_context":
        return [*spec.command, "--limit", str(args.orchestrator_profile_limit)]
    if spec.name == "simplify_zero_mixed":
        return list(spec.command)
    return None


def orchestrator_profile_case_limit(
    spec: SuiteSpec, args: argparse.Namespace
) -> int | None:
    if spec.name == "embedded_equivalence_context":
        return args.orchestrator_profile_limit
    return None


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
        total_cases = metrics.get("total_cases")
        if isinstance(total_cases, int) and total_cases > 0:
            metrics["reported_elapsed_per_case_ms"] = round(
                metrics["reported_elapsed_seconds"] * 1000.0 / total_cases,
                4,
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

    wrapper_shell_depth_family_rows = parse_wrapper_shell_depth_family_rows(output)
    if wrapper_shell_depth_family_rows:
        metrics["wrapper_shell_depth_family_rows"] = wrapper_shell_depth_family_rows

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


def embedded_corpus_structure_metrics(
    csv_path: pathlib.Path = EMBEDDED_EQUIVALENCE_CONTEXT_CORPUS,
) -> dict[str, Any]:
    """Summarize corpus-only axes the runner output does not expose."""
    if not csv_path.exists():
        return {"parse_error": f"missing corpus csv: {csv_path}"}

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    families = sorted({row["family"] for row in rows if row.get("family")})
    combined_rows = [
        row
        for row in rows
        if row.get("wrapper") == "combined_additive_zero"
    ]
    combined_family_counts = count_by_field(combined_rows, "family")
    collapse_rows = [
        row
        for row in combined_rows
        if row.get("expected_strategy") == "collapse exact zero additive subexpressions"
    ]
    depth4_rows = [
        row
        for row in combined_rows
        if "depth4" in row.get("pair_id", "")
    ]
    orientation_rows = [
        row
        for row in combined_rows
        if is_orientation_variation_pair_id(row.get("pair_id", ""))
    ]
    multi_core_rows = [
        row
        for row in combined_rows
        if is_multi_core_pair_id(row.get("pair_id", ""))
    ]
    depth4_families = unique_field_values(depth4_rows, "family")
    orientation_families = unique_field_values(orientation_rows, "family")
    multi_core_families = unique_field_values(multi_core_rows, "family")
    family_set = set(families)

    min_family_case_count = (
        min(combined_family_counts.values()) if combined_family_counts else 0
    )
    low_family_counts = {
        family: count
        for family, count in sorted(combined_family_counts.items())
        if count == min_family_case_count
    }
    above_min_family_counts = {
        family: count
        for family, count in sorted(combined_family_counts.items())
        if count > min_family_case_count
    }
    under_target_family_counts = {
        family: count
        for family, count in sorted(combined_family_counts.items())
        if count < COMBINED_ADDITIVE_FAMILY_TARGET_CASE_COUNT
    }

    return {
        "row_count": len(rows),
        "family_count": len(families),
        "combined_additive_zero": {
            "total": len(combined_rows),
            "family_count": len(combined_family_counts),
            "family_counts": combined_family_counts,
            "collapse_rows": len(collapse_rows),
            "collapse_family_count": unique_field_count(collapse_rows, "family"),
            "depth4_rows": len(depth4_rows),
            "depth4_family_count": len(depth4_families),
            "depth4_missing_families": sorted(family_set - depth4_families),
            "orientation_rows": len(orientation_rows),
            "orientation_family_count": len(orientation_families),
            "orientation_missing_families": sorted(
                family_set - orientation_families
            ),
            "multi_core_rows": len(multi_core_rows),
            "multi_core_family_count": len(multi_core_families),
            "multi_core_missing_families": sorted(
                family_set - multi_core_families
            ),
            "min_family_case_count": min_family_case_count,
            "target_family_case_count": COMBINED_ADDITIVE_FAMILY_TARGET_CASE_COUNT,
            "low_family_count": len(low_family_counts),
            "low_family_counts": low_family_counts,
            "under_target_family_count": len(under_target_family_counts),
            "under_target_family_counts": under_target_family_counts,
            "above_min_family_counts": above_min_family_counts,
        },
    }


def generated_discovery_ledger_metrics(
    ledger_path: pathlib.Path = ENGINE_COMBINATION_LEDGER,
) -> dict[str, Any]:
    """Summarize observe-only generated discoveries retained outside live corpus."""
    if not ledger_path.exists():
        return {"parse_error": f"missing combination ledger: {ledger_path}"}
    return parse_generated_discovery_ledger(ledger_path.read_text())


def parse_generated_discovery_ledger(text: str) -> dict[str, Any]:
    discoveries: list[dict[str, str]] = []
    section_pattern = re.compile(
        r"^#{2,3}\s+(?P<title>.+?)\n(?P<body>.*?)(?=^#{2,3}\s+|\Z)",
        re.M | re.S,
    )
    axis_pattern = re.compile(r"`(?P<wrapper>[^`]+)`\s+x\s+`(?P<family>[^`]+)`")

    for match in section_pattern.finditer(text):
        title = match.group("title").strip()
        body = match.group("body")
        if not is_observe_only_discovery_section(title, body):
            continue
        if is_closed_observe_only_discovery_section(body):
            continue
        axis_match = axis_pattern.search(body)
        discoveries.append(
            {
                "title": title,
                "status": "observe-only discovery",
                "area": extract_discovery_area(body),
                "wrapper": axis_match.group("wrapper") if axis_match else "unknown",
                "family": axis_match.group("family") if axis_match else "unknown",
            }
        )

    by_family: dict[str, int] = {}
    by_wrapper: dict[str, int] = {}
    by_area: dict[str, int] = {}
    for discovery in discoveries:
        area = discovery["area"]
        family = discovery["family"]
        wrapper = discovery["wrapper"]
        if area != "unknown":
            by_area[area] = by_area.get(area, 0) + 1
        if family != "unknown":
            by_family[family] = by_family.get(family, 0) + 1
        if wrapper != "unknown":
            by_wrapper[wrapper] = by_wrapper.get(wrapper, 0) + 1

    return {
        "observe_only_discoveries": len(discoveries),
        "areas": by_area,
        "families": by_family,
        "wrappers": by_wrapper,
        "recent": discoveries[:5],
    }


def is_observe_only_discovery_section(title: str, body: str) -> bool:
    title_status = normalized_discovery_status(title)
    title_marks_discovery = (
        "discovery" in title_status and "observe-only" in title_status
    )
    if title_marks_discovery:
        return True

    for status in re.findall(r"^\s*-\s+`([^`]+)`", body, re.M):
        normalized = normalized_discovery_status(status)
        if normalized in {
            "observe-only-discovery",
            "discovery-observe-only",
        }:
            return True
        if normalized == "observe-only" and "discovery" in body.lower():
            return True
    return False


def is_closed_observe_only_discovery_section(body: str) -> bool:
    lowered_body = body.lower()
    closed_follow_up_phrases = (
        "should no longer be treated as an open observe-only candidate",
        "should no longer count as an open observe-only candidate",
        "no longer an open observe-only discovery",
    )
    if any(phrase in lowered_body for phrase in closed_follow_up_phrases):
        return True
    if re.search(
        r"^\s*-\s+(resolved by|superseded by|follow-up resolution):",
        body,
        re.M | re.I,
    ):
        return True
    for status in re.findall(r"^\s*-\s+`([^`]+)`", body, re.M):
        normalized = normalized_discovery_status(status)
        if normalized in {"closed", "rejected", "resolved", "superseded"}:
            return True
        if normalized.startswith(("closed-", "resolved-", "superseded-")):
            return True
    return False


def extract_discovery_area(body: str) -> str:
    lines = body.splitlines()
    for idx, line in enumerate(lines):
        if not re.match(r"^-\s+area:\s*$", line):
            continue

        items: list[str] = []
        current: str | None = None
        for following in lines[idx + 1 :]:
            if following.startswith("- "):
                break
            bullet_match = re.match(r"^\s{2,}-\s+(?P<item>.+)$", following)
            if bullet_match:
                if current:
                    items.append(current)
                current = bullet_match.group("item").strip()
                continue
            continuation_match = re.match(r"^\s{4,}(?P<text>\S.*)$", following)
            if continuation_match and current:
                current = f"{current} {continuation_match.group('text').strip()}"
        if current:
            items.append(current)

        for item in items:
            area = discovery_area_bucket(item)
            if area != "unknown":
                return area
        break
    return "unknown"


def discovery_area_bucket(raw_area: str) -> str:
    cleaned = re.sub(r"`([^`]+)`", r"\1", raw_area)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "unknown"

    parts = [
        canonical_discovery_area_part(part.strip())
        for part in cleaned.split("/")
        if part.strip()
    ]
    if not parts:
        return "unknown"
    if len(parts) == 1:
        return parts[0]
    return " / ".join(parts[:2])


def canonical_discovery_area_part(part: str) -> str:
    return {
        "diff": "differentiation",
        "integrate": "integration",
    }.get(part.lower(), part)


def normalized_discovery_status(text: str) -> str:
    return re.sub(r"[\s/_-]+", "-", text.strip().lower())


def embedded_coverage_saturation_metrics(
    embedded_metrics: dict[str, Any],
) -> dict[str, Any]:
    corpus_structure = embedded_metrics.get("corpus_structure")
    if not isinstance(corpus_structure, dict):
        return {}

    combined = corpus_structure.get("combined_additive_zero")
    if not isinstance(combined, dict):
        return {}

    family_counts = combined.get("family_counts")
    if not isinstance(family_counts, dict) or not family_counts:
        return {}

    families = sorted(family for family in family_counts if isinstance(family, str))
    if not families:
        return {}

    family_set = set(families)
    checks: list[dict[str, Any]] = []

    def append_missing_family_check(name: str, missing: Any) -> None:
        missing_families = (
            sorted(family for family in missing if isinstance(family, str))
            if isinstance(missing, list)
            else []
        )
        checks.append(
            {
                "name": name,
                "status": "balanced" if not missing_families else "needs_attention",
                "missing_family_count": len(missing_families),
                "missing_families": missing_families,
            }
        )

    under_target_counts = combined.get("under_target_family_counts")
    if isinstance(under_target_counts, dict):
        under_target_families = sorted(
            family for family in under_target_counts if isinstance(family, str)
        )
        checks.append(
            {
                "name": "combined_additive_family_target",
                "status": "balanced" if not under_target_families else "needs_attention",
                "under_target_family_count": len(under_target_families),
                "under_target_families": under_target_families,
            }
        )

    append_missing_family_check(
        "combined_additive_depth4",
        combined.get("depth4_missing_families"),
    )
    append_missing_family_check(
        "combined_additive_orientation",
        combined.get("orientation_missing_families"),
    )
    append_missing_family_check(
        "combined_additive_multi_core",
        combined.get("multi_core_missing_families"),
    )

    wrapper_shell_depth_family_rows = embedded_metrics.get(
        "wrapper_shell_depth_family_rows"
    )
    if isinstance(wrapper_shell_depth_family_rows, list):
        reciprocal_depth4_families = {
            row.get("family")
            for row in wrapper_shell_depth_family_rows
            if isinstance(row, dict)
            and row.get("wrapper") == "reciprocal_shifted_difference_zero"
            and row.get("shell_depth") == 4
            and isinstance(row.get("family"), str)
        }
        reciprocal_missing = sorted(family_set - reciprocal_depth4_families)
        checks.append(
            {
                "name": "reciprocal_shifted_difference_depth4",
                "status": "balanced" if not reciprocal_missing else "needs_attention",
                "missing_family_count": len(reciprocal_missing),
                "missing_families": reciprocal_missing,
            }
        )

    if not checks:
        return {}

    open_checks = [check for check in checks if check["status"] != "balanced"]
    status = "balanced" if not open_checks else "needs_attention"
    recommendation = (
        "defer embedded corpus padding unless a new reproducible failure appears; "
        "prefer calculus, robustness, or presentation candidates"
        if status == "balanced"
        else "prefer the listed embedded corpus gaps before promoting broader variants"
    )

    return {
        "status": status,
        "checked_family_count": len(families),
        "balanced_check_count": len(checks) - len(open_checks),
        "open_check_count": len(open_checks),
        "checks": checks,
        "recommendation": recommendation,
    }


def generated_discovery_pressure_metrics(scorecard: dict[str, Any]) -> dict[str, Any]:
    discovery_metrics = scorecard.get("generated_discovery")
    if not isinstance(discovery_metrics, dict):
        return {}
    discovery_families = discovery_metrics.get("families")
    if not isinstance(discovery_families, dict):
        return {}

    embedded_suite = scorecard.get("suites", {}).get("embedded_equivalence_context")
    if not isinstance(embedded_suite, dict):
        return {}
    metrics = embedded_suite.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    corpus_structure = metrics.get("corpus_structure")
    if not isinstance(corpus_structure, dict):
        return {}
    combined = corpus_structure.get("combined_additive_zero")
    if not isinstance(combined, dict):
        return {}
    target_family_counts = combined.get("under_target_family_counts")
    if not isinstance(target_family_counts, dict):
        target_family_counts = combined.get("low_family_counts")
    if not isinstance(target_family_counts, dict):
        return {}

    blocked: dict[str, dict[str, int]] = {}
    unblocked: dict[str, int] = {}
    for family, live_count in sorted(target_family_counts.items()):
        if not isinstance(family, str) or not isinstance(live_count, int):
            continue
        discovery_count = discovery_families.get(family, 0)
        if isinstance(discovery_count, int) and discovery_count > 0:
            blocked[family] = {
                "live_count": live_count,
                "observe_only_discoveries": discovery_count,
            }
        else:
            unblocked[family] = live_count

    return {
        "low_family_count": len(blocked) + len(unblocked),
        "blocked_low_family_count": len(blocked),
        "blocked_low_families": blocked,
        "unblocked_low_families": unblocked,
    }


def count_by_field(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(field, "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def unique_field_count(rows: list[dict[str, str]], field: str) -> int:
    return len({row[field] for row in rows if row.get(field)})


def unique_field_values(rows: list[dict[str, str]], field: str) -> set[str]:
    return {row[field] for row in rows if row.get(field)}


def is_orientation_variation_pair_id(pair_id: str) -> bool:
    tokens = pair_id.split("_")
    return (
        "reversed" in tokens
        or "reverse" in tokens
        or "negative" in tokens
    )


def is_multi_core_pair_id(pair_id: str) -> bool:
    return any(
        marker in pair_id
        for marker in (
            "three_core",
            "four_",
            "five_",
            "_mix",
            "mixed_core",
        )
    )


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


def parse_wrapper_shell_depth_family_rows(output: str) -> list[dict[str, Any]]:
    wrapper_shell_depth_family_rows: list[dict[str, Any]] = []
    in_block = False
    row_pattern = re.compile(
        r"^\s+(?P<wrapper>.+?) x depth (?P<shell_depth>\d+) x (?P<family>.+?): "
        r"total=(?P<total>\d+) passed=(?P<passed>\d+) failed=(?P<failed>\d+)$"
    )

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "Sparse wrapper x shell-depth family buckets:":
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped:
            break
        match = row_pattern.match(line)
        if not match:
            continue
        wrapper_shell_depth_family_rows.append(
            {
                "wrapper": match.group("wrapper"),
                "shell_depth": int(match.group("shell_depth")),
                "family": match.group("family"),
                "total": int(match.group("total")),
                "passed": int(match.group("passed")),
                "failed": int(match.group("failed")),
            }
        )
    return wrapper_shell_depth_family_rows


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
        section = row["section"]
        samples = samples_by_section.get(section)
        if samples is None and section.endswith("…"):
            prefix = section[:-1]
            sample_section_matches = [
                sample_section
                for sample_section in samples_by_section
                if sample_section.startswith(prefix)
            ]
            if len(sample_section_matches) == 1:
                section = sample_section_matches[0]
                row["section"] = section
                samples = samples_by_section[section]
        row["samples"] = samples or []
        row["hit_samples"] = [
            sample.removeprefix("hit: ")
            for sample in row["samples"]
            if sample.startswith("hit: ")
        ]
        row["miss_samples"] = [
            sample.removeprefix("miss: ")
            for sample in row["samples"]
            if sample.startswith("miss: ")
        ]

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


def orchestrator_profile_sample_suffix(
    row: dict[str, Any], *, prefer_miss: bool = False
) -> str:
    if prefer_miss and row.get("miss_samples"):
        return f" miss_sample=`{row['miss_samples'][0]}`"
    if row.get("hit_samples"):
        return f" hit_sample=`{row['hit_samples'][0]}`"
    if row.get("miss_samples"):
        return f" miss_sample=`{row['miss_samples'][0]}`"
    if row.get("samples"):
        return f" sample=`{row['samples'][0]}`"
    return ""


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

    path_signal = re.search(
        r"derive path signal: single_step_successes=(\d+) multi_step_successes=(\d+) max_step_count=(\d+)",
        output,
    )
    multi_step_ids = re.search(r"derive multi-step-ids: ([^\n]*)", output)
    specificity = re.search(
        r"derive strategy specificity: generic_simplify_expected=(\d+) distinct_expected_strategies=(\d+)",
        output,
    )
    derived_by_family = parse_debug_count_map(output, "derive derived-by-family")
    unsupported_by_family = parse_debug_count_map(
        output, "derive unsupported-equivalent-by-family"
    )
    not_equivalent_by_family = parse_debug_count_map(
        output, "derive not-equivalent-by-family"
    )
    expected_strategy_counts = parse_debug_count_map(
        output, "derive expected-strategy-counts"
    )
    single_step_successes = int(path_signal.group(1)) if path_signal else None
    multi_step_successes = int(path_signal.group(2)) if path_signal else None
    max_step_count = int(path_signal.group(3)) if path_signal else None
    multi_step_success_ids = []
    if multi_step_ids:
        raw_ids = multi_step_ids.group(1).strip()
        if raw_ids and raw_ids != "none":
            multi_step_success_ids = [
                item.strip() for item in raw_ids.split(",") if item.strip()
            ]

    return {
        "derived": int(summary.group(1)),
        "unsupported": int(summary.group(2)),
        "not_equivalent": int(summary.group(3)),
        "reachability_rate": float(stats.group(1)),
        "supported_equiv_rate": float(stats.group(2)),
        "mean_step_count": float(stats.group(3)),
        "long_path_rate": float(stats.group(4)),
        "single_step_successes": single_step_successes,
        "multi_step_successes": multi_step_successes,
        "multi_step_success_ids": multi_step_success_ids,
        "max_step_count": max_step_count,
        "generic_simplify_expected": int(specificity.group(1))
        if specificity
        else 0,
        "distinct_expected_strategies": int(specificity.group(2))
        if specificity
        else None,
        "expected_strategy_counts": expected_strategy_counts,
        "derived_by_family": derived_by_family,
        "unsupported_by_family": unsupported_by_family,
        "not_equivalent_by_family": not_equivalent_by_family,
    }


def parse_debug_count_map(output: str, label: str) -> dict[str, int]:
    match = re.search(rf"{re.escape(label)}: ({{[^\n]*}})", output)
    if not match:
        return {}
    try:
        raw = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        key: value
        for key, value in raw.items()
        if isinstance(key, str) and isinstance(value, int)
    }


def parse_derive_shadow(output: str) -> dict[str, Any]:
    summary = re.search(
        r"derive shadow pressure summary: sampled=(\d+) derived=(\d+) unsupported=(\d+) not_equivalent=(\d+)",
        output,
    )
    stats = re.search(
        r"derive shadow pressure stats: reachability_rate=([0-9.]+) mean_step_count=([0-9.]+) single_step_successes=(\d+) multi_step_successes=(\d+)",
        output,
    )
    if not summary or not stats:
        raise ValueError("missing derive shadow pressure summary block")

    specificity = re.search(
        r"derive shadow pressure strategy specificity: generic_simplify_strategy_successes=(\d+) distinct_actual_strategies=(\d+)",
        output,
    )
    embedded_family_coverage = re.search(
        r"derive shadow pressure embedded-family coverage: sampled_families=(\d+) total_families=(\d+) missing=([^\n]*)",
        output,
    )
    generic_ids = re.search(
        r"derive shadow pressure generic-simplify-ids: ([^\n]*)",
        output,
    )
    multi_step_ids = re.search(
        r"derive shadow pressure multi-step-ids: ([^\n]*)",
        output,
    )
    actual_strategy_counts = parse_debug_count_map(
        output, "derive shadow pressure actual-strategy-counts"
    )
    derived_by_family = parse_debug_count_map(
        output, "derive shadow pressure derived-by-family"
    )
    unsupported_by_family = parse_debug_count_map(
        output, "derive shadow pressure unsupported-equivalent-by-family"
    )
    not_equivalent_by_family = parse_debug_count_map(
        output, "derive shadow pressure not-equivalent-by-family"
    )

    sampled = int(summary.group(1))
    derived = int(summary.group(2))
    unsupported = int(summary.group(3))
    not_equivalent = int(summary.group(4))
    classified = derived + unsupported + not_equivalent
    generic_simplify_ids = []
    if generic_ids:
        raw_ids = generic_ids.group(1).strip()
        if raw_ids and raw_ids != "none":
            generic_simplify_ids = [
                item.strip() for item in raw_ids.split(",") if item.strip()
            ]
    single_step_successes = int(stats.group(3))
    multi_step_successes = int(stats.group(4))
    multi_step_success_ids = []
    multi_step_success_step_counts = {}
    if multi_step_ids:
        raw_ids = multi_step_ids.group(1).strip()
        if raw_ids and raw_ids != "none":
            multi_step_success_ids = [
                item.strip() for item in raw_ids.split(",") if item.strip()
            ]
            for item in multi_step_success_ids:
                step_count = re.match(r"(.+):(\d+)$", item)
                if step_count:
                    multi_step_success_step_counts[step_count.group(1)] = int(
                        step_count.group(2)
                    )
    parsed_step_counts = list(multi_step_success_step_counts.values())
    if parsed_step_counts:
        max_step_count = max(
            ([1] if single_step_successes else []) + parsed_step_counts
        )
    elif single_step_successes and not multi_step_successes:
        max_step_count = 1
    elif derived == 0:
        max_step_count = 0
    else:
        max_step_count = None
    embedded_missing_families = []
    if embedded_family_coverage:
        raw_missing = embedded_family_coverage.group(3).strip()
        if raw_missing and raw_missing != "none":
            embedded_missing_families = [
                item.strip() for item in raw_missing.split(",") if item.strip()
            ]

    return {
        "sampled": sampled,
        "derived": derived,
        "unsupported": unsupported,
        "not_equivalent": not_equivalent,
        "classified": classified,
        "classification_rate": classified / sampled if sampled else 0.0,
        "reachability_rate": float(stats.group(1)),
        "mean_step_count": float(stats.group(2)),
        "single_step_successes": single_step_successes,
        "multi_step_successes": multi_step_successes,
        "multi_step_success_ids": multi_step_success_ids,
        "multi_step_success_step_counts": multi_step_success_step_counts,
        "max_step_count": max_step_count,
        "generic_simplify_strategy_successes": int(specificity.group(1))
        if specificity
        else 0,
        "generic_simplify_strategy_ids": generic_simplify_ids,
        "distinct_actual_strategies": int(specificity.group(2))
        if specificity
        else None,
        "embedded_family_sampled_count": int(embedded_family_coverage.group(1))
        if embedded_family_coverage
        else None,
        "embedded_family_total_count": int(embedded_family_coverage.group(2))
        if embedded_family_coverage
        else None,
        "embedded_family_missing": embedded_missing_families,
        "actual_strategy_counts": actual_strategy_counts,
        "derived_by_family": derived_by_family,
        "unsupported_by_family": unsupported_by_family,
        "not_equivalent_by_family": not_equivalent_by_family,
    }


def parse_simplify_didactic(output: str) -> dict[str, Any]:
    summary = re.search(
        r"simplify didactic audit summary: cases=(\d+) flagged=(\d+) "
        r"no_wire_substeps=(\d+) single_step_no_substeps=(\d+) "
        r"missing_math_sides=(\d+) total_wire_substeps=(\d+) "
        r"mean_step_count=([0-9.]+)",
        output,
    )
    if not summary:
        raise ValueError("missing simplify didactic audit summary block")

    cases = int(summary.group(1))
    flagged_cases = int(summary.group(2))
    no_wire_substeps = int(summary.group(3))
    single_step_no_substeps = int(summary.group(4))
    missing_math_sides = int(summary.group(5))
    total_wire_substeps = int(summary.group(6))
    mean_step_count = float(summary.group(7))

    return {
        "cases": cases,
        "flagged_cases": flagged_cases,
        "flagged_rate": flagged_cases / cases if cases else 0.0,
        "no_wire_substeps": no_wire_substeps,
        "single_step_no_substeps": single_step_no_substeps,
        "missing_math_sides": missing_math_sides,
        "total_wire_substeps": total_wire_substeps,
        "mean_step_count": mean_step_count,
    }


def parse_derive_didactic(output: str) -> dict[str, Any]:
    summary = re.search(
        r"derive didactic audit summary: cases=(\d+) flagged=(\d+) "
        r"no_web_substeps=(\d+) no_web_steps=(\d+) "
        r"total_web_substeps=(\d+) mean_step_count=([0-9.]+)",
        output,
    )
    if not summary:
        raise ValueError("missing derive didactic audit summary block")

    cases = int(summary.group(1))
    flagged_cases = int(summary.group(2))
    no_web_substeps = int(summary.group(3))
    no_web_steps = int(summary.group(4))
    total_web_substeps = int(summary.group(5))
    mean_step_count = float(summary.group(6))

    metrics = {
        "cases": cases,
        "flagged_cases": flagged_cases,
        "flagged_rate": flagged_cases / cases if cases else 0.0,
        "no_web_substeps": no_web_substeps,
        "no_web_steps": no_web_steps,
        "total_web_substeps": total_web_substeps,
        "mean_step_count": mean_step_count,
    }
    timings = re.search(
        r"derive didactic audit timings: "
        r"artifacts_seconds=([0-9.]+) "
        r"cli_seconds=([0-9.]+) "
        r"report_seconds=([0-9.]+) "
        r"total_seconds=([0-9.]+) "
        r"worker_count=(\d+)",
        output,
    )
    if timings:
        metrics.update(
            {
                "artifact_seconds": float(timings.group(1)),
                "cli_seconds": float(timings.group(2)),
                "report_seconds": float(timings.group(3)),
                "reported_total_seconds": float(timings.group(4)),
                "worker_count": int(timings.group(5)),
            }
        )
    family_hotspots = re.search(
        r"derive didactic audit family hotspots: "
        r"artifacts=([^\n]*) cli=([^\n]*)",
        output,
    )
    if family_hotspots:
        metrics["artifact_family_hotspots"] = family_hotspots.group(1).strip()
        metrics["cli_family_hotspots"] = family_hotspots.group(2).strip()
    case_hotspots = re.search(
        r"derive didactic audit case hotspots: "
        r"artifacts=([^\n]*) cli=([^\n]*)",
        output,
    )
    if case_hotspots:
        metrics["artifact_case_hotspots"] = case_hotspots.group(1).strip()
        metrics["cli_case_hotspots"] = case_hotspots.group(2).strip()

    return metrics


def add_unified_benchmark_rate_metrics(metrics: dict[str, Any]) -> None:
    total = metrics.get("total_combos", metrics.get("combos", 0))
    skipped = metrics.get("skipped", 0)
    timeouts = metrics.get("timeouts", 0)
    effective = max(0, total - skipped - timeouts)
    nf_convergent = metrics.get("nf_convergent", 0)
    proved_symbolic = metrics.get("proved_symbolic", 0)
    numeric_only = metrics.get("numeric_only", 0)
    inconclusive = metrics.get("inconclusive", 0)
    symbolic_closure = nf_convergent + proved_symbolic

    def rate(count: int) -> float:
        return round(count * 100.0 / effective, 1) if effective else 0.0

    metrics["effective_combos"] = effective
    metrics["symbolic_closure"] = symbolic_closure
    metrics["symbolic_closure_rate_percent"] = rate(symbolic_closure)
    metrics["nf_rate_percent"] = rate(nf_convergent)
    metrics["proved_symbolic_rate_percent"] = rate(proved_symbolic)
    metrics["numeric_only_rate_percent"] = rate(numeric_only)
    metrics["inconclusive_rate_percent"] = rate(inconclusive)
    metrics["normalization_gap"] = proved_symbolic
    metrics["normalization_gap_rate_percent"] = rate(proved_symbolic)


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
        add_unified_benchmark_rate_metrics(suites[suite_key])

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
    add_unified_benchmark_rate_metrics(metrics)
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

    if "nf_convergent" in metrics:
        metrics["total_combos"] = (
            metrics["nf_convergent"]
            + metrics["proved_symbolic"]
            + metrics["numeric_only"]
            + metrics["inconclusive"]
            + metrics["timeouts"]
        )
        add_unified_benchmark_rate_metrics(metrics)

    return metrics


def cargo_test_source_path(command: list[str]) -> pathlib.Path | None:
    package = command_arg_value(command, "-p")
    test_name = command_arg_value(command, "--test")
    if not package or not test_name:
        return None
    path = ROOT / "crates" / package / "tests" / f"{test_name}.rs"
    return path if path.exists() else None


def command_arg_value(command: list[str], flag: str) -> str | None:
    for idx, token in enumerate(command):
        if token == flag and idx + 1 < len(command):
            return command[idx + 1]
    return None


def parse_ignored_rust_tests(source: str) -> list[dict[str, str]]:
    ignored: list[dict[str, str]] = []
    pattern = re.compile(
        r"(?ms)^\s*#\[ignore(?:\s*=\s*\"(?P<reason>[^\"]*)\")?\]\s*"
        r"(?:#\[[^\n]*\]\s*)*fn\s+(?P<name>[A-Za-z0-9_]+)\s*\("
    )
    for match in pattern.finditer(source):
        reason = match.group("reason") or ""
        ignored.append({"name": match.group("name"), "reason": reason})
    return ignored


def ignored_cargo_tests_for_suite(spec: SuiteSpec) -> list[dict[str, str]]:
    source_path = cargo_test_source_path(spec.command)
    if source_path is None:
        return []
    try:
        source = source_path.read_text()
    except OSError:
        return []
    return parse_ignored_rust_tests(source)


def sanitize_residual_problem_cases(raw_cases: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_cases, list):
        return []

    sanitized: list[dict[str, Any]] = []
    string_fields = {"name", "status", "error_kind", "error", "result"}
    numeric_fields = {"wall_elapsed_seconds"}
    for case in raw_cases:
        if not isinstance(case, dict):
            continue
        row: dict[str, Any] = {}
        for key in string_fields:
            value = case.get(key)
            if isinstance(value, str):
                row[key] = value
        for key in numeric_fields:
            value = case.get(key)
            if isinstance(value, (int, float)):
                row[key] = value
        required_conditions = case.get("required_conditions")
        if isinstance(required_conditions, list):
            conditions = [
                condition
                for condition in required_conditions
                if isinstance(condition, str)
            ]
            if conditions:
                row["required_conditions"] = conditions
        if row:
            sanitized.append(row)
    return sanitized


def parse_calculus_residual_matrix(output: str) -> dict[str, Any]:
    try:
        raw = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid residual matrix json: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("residual matrix json output is not an object")

    total = raw.get("total")
    status = raw.get("status")
    status_counts = raw.get("status_counts")
    issue_kind_counts = raw.get("issue_kind_counts", {})
    if not isinstance(total, int):
        raise ValueError("missing residual matrix total")
    if not isinstance(status, str):
        raise ValueError("missing residual matrix status")
    if not isinstance(status_counts, dict):
        raise ValueError("missing residual matrix status_counts")
    if not isinstance(issue_kind_counts, dict):
        issue_kind_counts = {}

    passed = status_counts.get("pass", 0)
    raw_failed = status_counts.get("fail", 0)
    slow = status_counts.get("slow", 0)
    timeouts = status_counts.get("timeout", 0)
    if not all(
        isinstance(value, int)
        for value in (passed, raw_failed, slow, timeouts)
    ):
        raise ValueError("invalid residual matrix status_counts")

    problem_cases = sanitize_residual_problem_cases(raw.get("problem_cases"))
    problem_case_count = raw.get("problem_case_count")
    if not isinstance(problem_case_count, int):
        problem_case_count = len(problem_cases)
    expected_required_condition_case_count = raw.get(
        "expected_required_condition_case_count"
    )
    distinct_expected_required_conditions = raw.get(
        "distinct_expected_required_conditions"
    )
    expected_required_condition_counts = raw.get("expected_required_condition_counts")
    if isinstance(expected_required_condition_counts, dict):
        sanitized_condition_counts = {
            key: value
            for key, value in expected_required_condition_counts.items()
            if isinstance(key, str) and isinstance(value, int)
        }
    else:
        sanitized_condition_counts = {}
    if not isinstance(distinct_expected_required_conditions, int):
        distinct_expected_required_conditions = len(sanitized_condition_counts)
    matrix_base_count = raw.get("matrix_base_count")
    matrix_wrapped_base_count = raw.get("matrix_wrapped_base_count")
    matrix_standalone_base_count = raw.get("matrix_standalone_base_count")
    matrix_wrapper_count = raw.get("matrix_wrapper_count")
    matrix_wrapped_case_count = raw.get("matrix_wrapped_case_count")
    matrix_standalone_case_count = raw.get("matrix_standalone_case_count")
    matrix_expected_wrapped_case_count = raw.get("matrix_expected_wrapped_case_count")
    matrix_missing_wrapped_pair_count = raw.get("matrix_missing_wrapped_pair_count")
    matrix_full_wrapper_base_count = raw.get("matrix_full_wrapper_base_count")
    matrix_partial_wrapper_base_count = raw.get("matrix_partial_wrapper_base_count")
    matrix_largest_wrapper_gap_count = raw.get("matrix_largest_wrapper_gap_count")
    matrix_wrapper_gap_examples = raw.get("matrix_wrapper_gap_examples")

    metrics = {
        "matrix_status": status,
        "total_cases": total,
        "passed": passed,
        "failed": raw_failed + slow + timeouts,
        "raw_failed": raw_failed,
        "slow": slow,
        "timeouts": timeouts,
        "problem_case_count": problem_case_count,
        "problem_cases": problem_cases,
        "issue_kind_counts": {
            key: value
            for key, value in issue_kind_counts.items()
            if isinstance(key, str) and isinstance(value, int)
        },
    }
    if isinstance(expected_required_condition_case_count, int):
        metrics["expected_required_condition_case_count"] = (
            expected_required_condition_case_count
        )
    if sanitized_condition_counts or isinstance(
        raw.get("distinct_expected_required_conditions"), int
    ):
        metrics["distinct_expected_required_conditions"] = (
            distinct_expected_required_conditions
        )
        metrics["expected_required_condition_counts"] = sanitized_condition_counts
    if isinstance(matrix_base_count, int):
        metrics["matrix_base_count"] = matrix_base_count
    if isinstance(matrix_wrapped_base_count, int):
        metrics["matrix_wrapped_base_count"] = matrix_wrapped_base_count
    if isinstance(matrix_standalone_base_count, int):
        metrics["matrix_standalone_base_count"] = matrix_standalone_base_count
    if isinstance(matrix_wrapper_count, int):
        metrics["matrix_wrapper_count"] = matrix_wrapper_count
    if isinstance(matrix_wrapped_case_count, int):
        metrics["matrix_wrapped_case_count"] = matrix_wrapped_case_count
    if isinstance(matrix_standalone_case_count, int):
        metrics["matrix_standalone_case_count"] = matrix_standalone_case_count
    if isinstance(matrix_expected_wrapped_case_count, int):
        metrics["matrix_expected_wrapped_case_count"] = (
            matrix_expected_wrapped_case_count
        )
    if isinstance(matrix_missing_wrapped_pair_count, int):
        metrics["matrix_missing_wrapped_pair_count"] = (
            matrix_missing_wrapped_pair_count
        )
    if isinstance(matrix_full_wrapper_base_count, int):
        metrics["matrix_full_wrapper_base_count"] = matrix_full_wrapper_base_count
    if isinstance(matrix_partial_wrapper_base_count, int):
        metrics["matrix_partial_wrapper_base_count"] = matrix_partial_wrapper_base_count
    if isinstance(matrix_largest_wrapper_gap_count, int):
        metrics["matrix_largest_wrapper_gap_count"] = matrix_largest_wrapper_gap_count
    if isinstance(matrix_wrapper_gap_examples, list):
        sanitized_gap_examples = []
        for example in matrix_wrapper_gap_examples:
            if not isinstance(example, dict):
                continue
            base = example.get("base")
            missing_count = example.get("missing_count")
            missing_wrappers = example.get("missing_wrappers")
            if not isinstance(base, str) or not isinstance(missing_count, int):
                continue
            sanitized_missing_wrappers = (
                [
                    wrapper
                    for wrapper in missing_wrappers
                    if isinstance(wrapper, str)
                ]
                if isinstance(missing_wrappers, list)
                else []
            )
            sanitized_gap_examples.append(
                {
                    "base": base,
                    "missing_count": missing_count,
                    "missing_wrappers": sanitized_missing_wrappers,
                }
            )
        if sanitized_gap_examples:
            metrics["matrix_wrapper_gap_examples"] = sanitized_gap_examples
    return metrics


def sanitize_limit_command_problem_cases(raw_cases: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_cases, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for case in raw_cases:
        if not isinstance(case, dict):
            continue
        row: dict[str, Any] = {}
        for key in (
            "name",
            "status",
            "error_kind",
            "error",
            "result",
            "expected_result",
            "family",
            "argument_regime",
            "point_regime",
            "domain_regime",
            "outcome",
        ):
            value = case.get(key)
            if isinstance(value, str):
                row[key] = value
        for key in ("required_display", "expected_required_display", "warnings"):
            value = case.get(key)
            if isinstance(value, list):
                row[key] = [item for item in value if isinstance(item, str)]
        wall_elapsed = case.get("wall_elapsed_seconds")
        if isinstance(wall_elapsed, (int, float)):
            row["wall_elapsed_seconds"] = wall_elapsed
        if row:
            sanitized.append(row)
    return sanitized


def sanitize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def sanitize_string_list_map(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, list[str]] = {}
    for key, raw_items in value.items():
        if not isinstance(key, str):
            continue
        items = sanitize_string_list(raw_items)
        if items:
            sanitized[key] = items
    return dict(sorted(sanitized.items()))


def sanitize_int_count_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, int] = {}
    for key, count in value.items():
        if isinstance(key, str) and isinstance(count, int):
            sanitized[key] = count
    return dict(sorted(sanitized.items()))


def split_compound_count_map(
    counts: dict[str, int], *, label: str
) -> tuple[dict[str, int], dict[str, int]]:
    first_parts: dict[str, int] = {}
    second_parts: dict[str, int] = {}
    for key, count in counts.items():
        first, separator, second = key.partition("/")
        if separator != "/" or not first or not second:
            raise ValueError(f"{label} keys must have method/status shape")
        first_parts[first] = first_parts.get(first, 0) + count
        second_parts[second] = second_parts.get(second, 0) + count
    return dict(sorted(first_parts.items())), dict(sorted(second_parts.items()))


def compound_count_map_prefix_for_suffix(
    counts: dict[str, int], *, suffix: str, label: str
) -> dict[str, int]:
    prefix_counts: dict[str, int] = {}
    for key, count in counts.items():
        first, separator, second = key.partition("/")
        if separator != "/" or not first or not second:
            raise ValueError(f"{label} keys must have method/status shape")
        if second == suffix:
            prefix_counts[first] = prefix_counts.get(first, 0) + count
    return dict(sorted(prefix_counts.items()))


def validate_nonnegative_integer_key_count_map(
    counts: dict[str, int], *, label: str
) -> None:
    for key, count in counts.items():
        if count < 0:
            raise ValueError(f"{label} counts must be non-negative")
        try:
            parsed_key = int(key)
        except ValueError as exc:
            raise ValueError(f"{label} keys must be non-negative integers") from exc
        if parsed_key < 0 or str(parsed_key) != key:
            raise ValueError(f"{label} keys must be non-negative integers")


def classify_backend_verification_pressure(
    *,
    attempts: int,
    verification_checks_used_total: int,
    verification_elapsed_ms: float,
    max_verification_normalization_passes: int,
) -> dict[str, Any]:
    checks_per_attempt = (
        verification_checks_used_total / attempts if attempts else 0.0
    )
    signals: list[tuple[str, str, str]] = []
    if (
        checks_per_attempt
        >= BACKEND_VERIFICATION_PRESSURE_RISK_CHECKS_PER_ATTEMPT
    ):
        signals.append(
            (
                "risk",
                "checks_per_attempt",
                f"checks_per_attempt={checks_per_attempt:.3f}",
            )
        )
    elif (
        checks_per_attempt
        >= BACKEND_VERIFICATION_PRESSURE_WATCH_CHECKS_PER_ATTEMPT
    ):
        signals.append(
            (
                "watch",
                "checks_per_attempt",
                f"checks_per_attempt={checks_per_attempt:.3f}",
            )
        )

    if (
        max_verification_normalization_passes
        >= BACKEND_VERIFICATION_PRESSURE_RISK_MAX_NORMALIZATION_PASSES
    ):
        signals.append(
            (
                "risk",
                "normalization_passes",
                f"max_passes={max_verification_normalization_passes}",
            )
        )
    elif (
        max_verification_normalization_passes
        >= BACKEND_VERIFICATION_PRESSURE_WATCH_MAX_NORMALIZATION_PASSES
    ):
        signals.append(
            (
                "watch",
                "normalization_passes",
                f"max_passes={max_verification_normalization_passes}",
            )
        )

    if verification_elapsed_ms >= BACKEND_VERIFICATION_PRESSURE_RISK_ELAPSED_MS:
        signals.append(
            ("risk", "elapsed_ms", f"elapsed_ms={verification_elapsed_ms:.3f}")
        )
    elif verification_elapsed_ms >= BACKEND_VERIFICATION_PRESSURE_WATCH_ELAPSED_MS:
        signals.append(
            ("watch", "elapsed_ms", f"elapsed_ms={verification_elapsed_ms:.3f}")
        )

    severity_rank = {"ok": 0, "watch": 1, "risk": 2}
    status = "ok"
    primary_signal = "none"
    reason = "within backend verification pressure thresholds"
    if signals:
        status, primary_signal, reason = max(
            signals,
            key=lambda signal: severity_rank[signal[0]],
        )

    return {
        "status": status,
        "primary_signal": primary_signal,
        "reason": reason,
        "attempts": attempts,
        "verification_checks_used_total": verification_checks_used_total,
        "checks_per_attempt": round(checks_per_attempt, 3),
        "max_verification_normalization_passes": (
            max_verification_normalization_passes
        ),
        "verification_elapsed_ms": round(float(verification_elapsed_ms), 3),
    }


def parse_algorithmic_backend_observability(output: str) -> dict[str, Any]:
    match = re.search(r"algorithmic_backend_observability:\s*(\{[^\n]+\})", output)
    if not match:
        raise ValueError("missing algorithmic_backend_observability output")

    raw = json.loads(match.group(1))
    if not isinstance(raw, dict):
        raise ValueError("algorithmic_backend_observability must be a JSON object")

    attempts = raw.get("attempts")
    public_accepted = raw.get("public_accepted")
    unverified_public_acceptances = raw.get("unverified_public_acceptances")
    fallback_eligible = raw.get("fallback_eligible", 0)
    unverified_fallback_acceptances = raw.get("unverified_fallback_acceptances", 0)
    method_probe_budget_exhausted = raw.get("method_probe_budget_exhausted", 0)
    verification_budget_exceeded = raw.get("verification_budget_exceeded", 0)
    method_probes_used_total = raw.get("method_probes_used_total", 0)
    verification_checks_used_total = raw.get("verification_checks_used_total", 0)
    verification_elapsed_ms = raw.get("verification_elapsed_ms")
    max_verification_normalization_passes = raw.get(
        "max_verification_normalization_passes", 0
    )
    public_max_verification_normalization_passes = raw.get(
        "public_max_verification_normalization_passes", 0
    )
    fallback_max_verification_normalization_passes = raw.get(
        "fallback_max_verification_normalization_passes", 0
    )
    assumption_exprs = raw.get("assumption_exprs", 0)
    public_assumption_exprs = raw.get("public_assumption_exprs", 0)
    fallback_assumption_exprs = raw.get("fallback_assumption_exprs", 0)
    if not isinstance(attempts, int) or attempts < 0:
        raise ValueError("algorithmic backend attempts must be a non-negative integer")
    if not isinstance(public_accepted, int) or public_accepted < 0:
        raise ValueError(
            "algorithmic backend public_accepted must be a non-negative integer"
        )
    if (
        not isinstance(unverified_public_acceptances, int)
        or unverified_public_acceptances < 0
    ):
        raise ValueError(
            "algorithmic backend unverified_public_acceptances must be a "
            "non-negative integer"
        )
    if not isinstance(fallback_eligible, int) or fallback_eligible < 0:
        raise ValueError(
            "algorithmic backend fallback_eligible must be a non-negative integer"
        )
    if (
        not isinstance(unverified_fallback_acceptances, int)
        or unverified_fallback_acceptances < 0
    ):
        raise ValueError(
            "algorithmic backend unverified_fallback_acceptances must be a "
            "non-negative integer"
        )
    if (
        not isinstance(method_probe_budget_exhausted, int)
        or method_probe_budget_exhausted < 0
    ):
        raise ValueError(
            "algorithmic backend method_probe_budget_exhausted must be a "
            "non-negative integer"
        )
    if (
        not isinstance(verification_budget_exceeded, int)
        or verification_budget_exceeded < 0
    ):
        raise ValueError(
            "algorithmic backend verification_budget_exceeded must be a "
            "non-negative integer"
        )
    if (
        not isinstance(method_probes_used_total, int)
        or method_probes_used_total < 0
    ):
        raise ValueError(
            "algorithmic backend method_probes_used_total must be a "
            "non-negative integer"
        )
    if (
        not isinstance(verification_checks_used_total, int)
        or verification_checks_used_total < 0
    ):
        raise ValueError(
            "algorithmic backend verification_checks_used_total must be a "
            "non-negative integer"
        )
    if not isinstance(verification_elapsed_ms, (int, float)):
        raise ValueError("algorithmic backend verification_elapsed_ms must be numeric")
    if (
        not isinstance(max_verification_normalization_passes, int)
        or max_verification_normalization_passes < 0
    ):
        raise ValueError(
            "algorithmic backend max_verification_normalization_passes must be "
            "a non-negative integer"
        )
    if (
        not isinstance(public_max_verification_normalization_passes, int)
        or public_max_verification_normalization_passes < 0
    ):
        raise ValueError(
            "algorithmic backend public_max_verification_normalization_passes "
            "must be a non-negative integer"
        )
    if (
        not isinstance(fallback_max_verification_normalization_passes, int)
        or fallback_max_verification_normalization_passes < 0
    ):
        raise ValueError(
            "algorithmic backend fallback_max_verification_normalization_passes "
            "must be a non-negative integer"
        )
    if not isinstance(assumption_exprs, int) or assumption_exprs < 0:
        raise ValueError(
            "algorithmic backend assumption_exprs must be a non-negative integer"
        )
    if (
        not isinstance(public_assumption_exprs, int)
        or public_assumption_exprs < 0
    ):
        raise ValueError(
            "algorithmic backend public_assumption_exprs must be a "
            "non-negative integer"
        )
    if (
        not isinstance(fallback_assumption_exprs, int)
        or fallback_assumption_exprs < 0
    ):
        raise ValueError(
            "algorithmic backend fallback_assumption_exprs must be a "
            "non-negative integer"
        )
    if (
        public_accepted > attempts
        or unverified_public_acceptances > attempts
        or fallback_eligible > attempts
        or unverified_fallback_acceptances > attempts
        or method_probe_budget_exhausted > attempts
        or verification_budget_exceeded > attempts
    ):
        raise ValueError("algorithmic backend counts exceed attempts")
    if (
        public_assumption_exprs > assumption_exprs
        or fallback_assumption_exprs > assumption_exprs
    ):
        raise ValueError("algorithmic backend assumption counts exceed total")

    mode_counts = sanitize_int_count_map(raw.get("mode_counts"))
    method_counts = sanitize_int_count_map(raw.get("method_counts"))
    method_probe_usage_by_method = sanitize_int_count_map(
        raw.get("method_probe_usage_by_method")
    )
    method_probe_attempt_counts = sanitize_int_count_map(
        raw.get("method_probe_attempt_counts")
    )
    method_probe_candidate_counts = sanitize_int_count_map(
        raw.get("method_probe_candidate_counts")
    )
    method_probe_no_match_counts = sanitize_int_count_map(
        raw.get("method_probe_no_match_counts")
    )
    method_probe_no_match_reason_counts = sanitize_int_count_map(
        raw.get("method_probe_no_match_reason_counts")
    )
    verification_check_usage_by_method = sanitize_int_count_map(
        raw.get("verification_check_usage_by_method")
    )
    verification_status_by_method = sanitize_int_count_map(
        raw.get("verification_status_by_method")
    )
    residual_reason_by_method = sanitize_int_count_map(
        raw.get("residual_reason_by_method")
    )
    verification_blocker_counts = sanitize_int_count_map(
        raw.get("verification_blocker_counts")
    )
    verification_blocker_by_method = sanitize_int_count_map(
        raw.get("verification_blocker_by_method")
    )
    failure_class_counts = sanitize_int_count_map(raw.get("failure_class_counts"))
    failure_class_by_method = sanitize_int_count_map(raw.get("failure_class_by_method"))
    verification_residual_counts = sanitize_int_count_map(
        raw.get("verification_residual_counts")
    )
    verification_residual_by_method = sanitize_int_count_map(
        raw.get("verification_residual_by_method")
    )
    verification_residual_kind_counts = sanitize_int_count_map(
        raw.get("verification_residual_kind_counts")
    )
    verification_residual_kind_by_method = sanitize_int_count_map(
        raw.get("verification_residual_kind_by_method")
    )
    verification_residual_signature_counts = sanitize_int_count_map(
        raw.get("verification_residual_signature_counts")
    )
    verification_residual_signature_by_method = sanitize_int_count_map(
        raw.get("verification_residual_signature_by_method")
    )
    publication_status_counts = sanitize_int_count_map(
        raw.get("publication_status_counts")
    )
    publication_status_by_method = sanitize_int_count_map(
        raw.get("publication_status_by_method")
    )
    fallback_status_counts = sanitize_int_count_map(raw.get("fallback_status_counts"))
    fallback_status_by_method = sanitize_int_count_map(
        raw.get("fallback_status_by_method")
    )
    trace_level_counts = sanitize_int_count_map(raw.get("trace_level_counts"))
    constant_policy_counts = sanitize_int_count_map(raw.get("constant_policy_counts"))
    public_trace_level_counts = sanitize_int_count_map(
        raw.get("public_trace_level_counts")
    )
    public_constant_policy_counts = sanitize_int_count_map(
        raw.get("public_constant_policy_counts")
    )
    fallback_trace_level_counts = sanitize_int_count_map(
        raw.get("fallback_trace_level_counts")
    )
    fallback_constant_policy_counts = sanitize_int_count_map(
        raw.get("fallback_constant_policy_counts")
    )
    verification_evidence_counts = sanitize_int_count_map(
        raw.get("verification_evidence_counts")
    )
    public_verification_evidence_counts = sanitize_int_count_map(
        raw.get("public_verification_evidence_counts")
    )
    fallback_verification_evidence_counts = sanitize_int_count_map(
        raw.get("fallback_verification_evidence_counts")
    )
    verification_evidence_by_method = sanitize_int_count_map(
        raw.get("verification_evidence_by_method")
    )
    public_verification_evidence_by_method = sanitize_int_count_map(
        raw.get("public_verification_evidence_by_method")
    )
    fallback_verification_evidence_by_method = sanitize_int_count_map(
        raw.get("fallback_verification_evidence_by_method")
    )
    verification_normalization_reason_counts = sanitize_int_count_map(
        raw.get("verification_normalization_reason_counts")
    )
    public_verification_normalization_reason_counts = sanitize_int_count_map(
        raw.get("public_verification_normalization_reason_counts")
    )
    fallback_verification_normalization_reason_counts = sanitize_int_count_map(
        raw.get("fallback_verification_normalization_reason_counts")
    )
    verification_normalization_reason_by_method = sanitize_int_count_map(
        raw.get("verification_normalization_reason_by_method")
    )
    public_verification_normalization_reason_by_method = sanitize_int_count_map(
        raw.get("public_verification_normalization_reason_by_method")
    )
    fallback_verification_normalization_reason_by_method = sanitize_int_count_map(
        raw.get("fallback_verification_normalization_reason_by_method")
    )
    verification_normalization_pass_count_counts = sanitize_int_count_map(
        raw.get("verification_normalization_pass_count_counts")
    )
    public_verification_normalization_pass_count_counts = sanitize_int_count_map(
        raw.get("public_verification_normalization_pass_count_counts")
    )
    fallback_verification_normalization_pass_count_counts = sanitize_int_count_map(
        raw.get("fallback_verification_normalization_pass_count_counts")
    )
    verification_normalization_pass_count_by_method = sanitize_int_count_map(
        raw.get("verification_normalization_pass_count_by_method")
    )
    public_verification_normalization_pass_count_by_method = sanitize_int_count_map(
        raw.get("public_verification_normalization_pass_count_by_method")
    )
    fallback_verification_normalization_pass_count_by_method = sanitize_int_count_map(
        raw.get("fallback_verification_normalization_pass_count_by_method")
    )
    verification_status_counts = sanitize_int_count_map(
        raw.get("verification_status_counts")
    )
    residual_reason_counts = sanitize_int_count_map(raw.get("residual_reason_counts"))
    required_condition_counts = sanitize_int_count_map(
        raw.get("required_condition_counts")
    )
    if mode_counts and sum(mode_counts.values()) != attempts:
        raise ValueError("algorithmic backend mode_counts do not match attempts")
    if sum(method_counts.values()) != attempts:
        raise ValueError("algorithmic backend method_counts do not match attempts")
    if sum(verification_status_counts.values()) != attempts:
        raise ValueError(
            "algorithmic backend verification_status_counts do not match attempts"
        )
    if sum(method_probe_usage_by_method.values()) != method_probes_used_total:
        raise ValueError(
            "algorithmic backend method_probe_usage_by_method does not match total"
        )
    if method_probes_used_total and not method_probe_attempt_counts:
        raise ValueError(
            "algorithmic backend method_probe_attempt_counts missing method probes"
        )
    if method_probes_used_total and "method_probe_candidate_counts" not in raw:
        raise ValueError(
            "algorithmic backend method_probe_candidate_counts missing method probes"
        )
    if method_probes_used_total and "method_probe_no_match_counts" not in raw:
        raise ValueError(
            "algorithmic backend method_probe_no_match_counts missing method probes"
        )
    if sum(method_probe_attempt_counts.values()) != method_probes_used_total:
        raise ValueError(
            "algorithmic backend method_probe_attempt_counts does not match total"
        )
    if sum(method_probe_candidate_counts.values()) > method_probes_used_total:
        raise ValueError(
            "algorithmic backend method_probe_candidate_counts exceed method probes"
        )
    method_probe_methods = (
        set(method_probe_attempt_counts)
        | set(method_probe_candidate_counts)
        | set(method_probe_no_match_counts)
    )
    for method in method_probe_methods:
        attempt_count = method_probe_attempt_counts.get(method, 0)
        candidate_count = method_probe_candidate_counts.get(method, 0)
        no_match_count = method_probe_no_match_counts.get(method, 0)
        if candidate_count + no_match_count != attempt_count:
            raise ValueError(
                "algorithmic backend method_probe candidate/no-match split "
                "does not match attempts"
            )
    method_probe_no_match_total = sum(method_probe_no_match_counts.values())
    if (
        method_probe_no_match_total
        and "method_probe_no_match_reason_counts" not in raw
    ):
        raise ValueError(
            "algorithmic backend method_probe_no_match_reason_counts missing "
            "method probe no-matches"
        )
    if (
        sum(method_probe_no_match_reason_counts.values())
        != method_probe_no_match_total
    ):
        raise ValueError(
            "algorithmic backend method_probe_no_match_reason_counts do not "
            "match method_probe_no_match_counts"
        )
    if method_probe_no_match_reason_counts:
        reason_by_method_prefix, _ = split_compound_count_map(
            method_probe_no_match_reason_counts,
            label="algorithmic backend method_probe_no_match_reason_counts",
        )
        if reason_by_method_prefix != method_probe_no_match_counts:
            raise ValueError(
                "algorithmic backend method_probe_no_match_reason_counts do not "
                "match method_probe_no_match_counts by method"
            )
    if (
        sum(verification_check_usage_by_method.values())
        != verification_checks_used_total
    ):
        raise ValueError(
            "algorithmic backend verification_check_usage_by_method does not "
            "match total"
        )
    if verification_status_by_method:
        if sum(verification_status_by_method.values()) != attempts:
            raise ValueError(
                "algorithmic backend verification_status_by_method does not "
                "match attempts"
            )
        status_by_method_prefix, status_by_method_suffix = split_compound_count_map(
            verification_status_by_method,
            label="algorithmic backend verification_status_by_method",
        )
        if status_by_method_prefix != method_counts:
            raise ValueError(
                "algorithmic backend verification_status_by_method does not "
                "match method_counts"
            )
        if status_by_method_suffix != verification_status_counts:
            raise ValueError(
                "algorithmic backend verification_status_by_method does not "
                "match verification_status_counts"
            )
    if residual_reason_by_method:
        if sum(residual_reason_by_method.values()) != sum(
            residual_reason_counts.values()
        ):
            raise ValueError(
                "algorithmic backend residual_reason_by_method does not match "
                "residual_reason_counts"
            )
        residual_by_method_prefix, residual_by_method_suffix = split_compound_count_map(
            residual_reason_by_method,
            label="algorithmic backend residual_reason_by_method",
        )
        unknown_residual_methods = set(residual_by_method_prefix) - set(method_counts)
        if unknown_residual_methods:
            raise ValueError(
                "algorithmic backend residual_reason_by_method has unknown methods"
            )
        if residual_by_method_suffix != residual_reason_counts:
            raise ValueError(
                "algorithmic backend residual_reason_by_method does not match "
                "residual_reason_counts"
            )
    if verification_blocker_counts:
        if sum(verification_blocker_counts.values()) > attempts:
            raise ValueError(
                "algorithmic backend verification_blocker_counts exceed attempts"
            )
    if verification_blocker_by_method:
        if sum(verification_blocker_by_method.values()) != sum(
            verification_blocker_counts.values()
        ):
            raise ValueError(
                "algorithmic backend verification_blocker_by_method does not "
                "match verification_blocker_counts"
            )
        blocker_by_method_prefix, blocker_by_method_suffix = split_compound_count_map(
            verification_blocker_by_method,
            label="algorithmic backend verification_blocker_by_method",
        )
        unknown_blocker_methods = set(blocker_by_method_prefix) - set(method_counts)
        if unknown_blocker_methods:
            raise ValueError(
                "algorithmic backend verification_blocker_by_method has unknown "
                "methods"
            )
        if blocker_by_method_suffix != verification_blocker_counts:
            raise ValueError(
                "algorithmic backend verification_blocker_by_method does not "
                "match verification_blocker_counts"
            )
    rejected_count = attempts - public_accepted
    if rejected_count and "failure_class_counts" not in raw:
        raise ValueError(
            "algorithmic backend failure_class_counts missing rejected candidates"
        )
    if failure_class_counts:
        if sum(failure_class_counts.values()) != rejected_count:
            raise ValueError(
                "algorithmic backend failure_class_counts do not match rejected "
                "candidates"
            )
    if failure_class_counts and "failure_class_by_method" not in raw:
        raise ValueError(
            "algorithmic backend failure_class_by_method missing rejected candidates"
        )
    if failure_class_by_method:
        if sum(failure_class_by_method.values()) != sum(failure_class_counts.values()):
            raise ValueError(
                "algorithmic backend failure_class_by_method does not match "
                "failure_class_counts"
            )
        failure_by_method_prefix, failure_by_method_suffix = split_compound_count_map(
            failure_class_by_method,
            label="algorithmic backend failure_class_by_method",
        )
        unknown_failure_methods = set(failure_by_method_prefix) - set(method_counts)
        if unknown_failure_methods:
            raise ValueError(
                "algorithmic backend failure_class_by_method has unknown methods"
            )
        if failure_by_method_suffix != failure_class_counts:
            raise ValueError(
                "algorithmic backend failure_class_by_method does not match "
                "failure_class_counts"
            )
    if verification_residual_counts:
        if (
            sum(verification_residual_counts.values())
            > verification_evidence_counts.get("failed_differentiation", 0)
        ):
            raise ValueError(
                "algorithmic backend verification_residual_counts exceed "
                "failed_differentiation evidence"
            )
    if verification_residual_by_method:
        if sum(verification_residual_by_method.values()) != sum(
            verification_residual_counts.values()
        ):
            raise ValueError(
                "algorithmic backend verification_residual_by_method does not "
                "match verification_residual_counts"
            )
        residual_by_method_prefix, residual_by_method_suffix = (
            split_compound_count_map(
                verification_residual_by_method,
                label="algorithmic backend verification_residual_by_method",
            )
        )
        unknown_residual_methods = set(residual_by_method_prefix) - set(
            method_counts
        )
        if unknown_residual_methods:
            raise ValueError(
                "algorithmic backend verification_residual_by_method has "
                "unknown methods"
            )
        if residual_by_method_suffix != verification_residual_counts:
            raise ValueError(
                "algorithmic backend verification_residual_by_method does not "
                "match verification_residual_counts"
            )
        if verification_evidence_by_method:
            failed_method_counts = compound_count_map_prefix_for_suffix(
                verification_evidence_by_method,
                suffix="failed_differentiation",
                label="algorithmic backend verification_evidence_by_method",
            )
            if any(
                residual_by_method_prefix.get(method, 0)
                > failed_method_counts.get(method, 0)
                for method in residual_by_method_prefix
            ):
                raise ValueError(
                    "algorithmic backend verification_residual_by_method "
                    "exceeds failed_differentiation evidence by method"
                )
    verification_residual_count_total = sum(verification_residual_counts.values())
    if verification_residual_count_total and not verification_residual_kind_counts:
        raise ValueError(
            "algorithmic backend verification_residual_kind_counts missing "
            "verification residuals"
        )
    if verification_residual_kind_counts:
        if sum(verification_residual_kind_counts.values()) != sum(
            verification_residual_counts.values()
        ):
            raise ValueError(
                "algorithmic backend verification_residual_kind_counts does not "
                "match verification_residual_counts"
            )
    if verification_residual_by_method and not verification_residual_kind_by_method:
        raise ValueError(
            "algorithmic backend verification_residual_kind_by_method missing "
            "verification residual methods"
        )
    if verification_residual_kind_by_method:
        if sum(verification_residual_kind_by_method.values()) != sum(
            verification_residual_kind_counts.values()
        ):
            raise ValueError(
                "algorithmic backend verification_residual_kind_by_method does "
                "not match verification_residual_kind_counts"
            )
        kind_by_method_prefix, kind_by_method_suffix = split_compound_count_map(
            verification_residual_kind_by_method,
            label="algorithmic backend verification_residual_kind_by_method",
        )
        unknown_kind_methods = set(kind_by_method_prefix) - set(method_counts)
        if unknown_kind_methods:
            raise ValueError(
                "algorithmic backend verification_residual_kind_by_method has "
                "unknown methods"
            )
        if kind_by_method_suffix != verification_residual_kind_counts:
            raise ValueError(
                "algorithmic backend verification_residual_kind_by_method does "
                "not match verification_residual_kind_counts"
            )
        if verification_residual_by_method:
            residual_by_method_prefix, _ = split_compound_count_map(
                verification_residual_by_method,
                label="algorithmic backend verification_residual_by_method",
            )
            if kind_by_method_prefix != residual_by_method_prefix:
                raise ValueError(
                    "algorithmic backend verification_residual_kind_by_method "
                    "does not match verification_residual_by_method methods"
                )
    if verification_residual_count_total and not verification_residual_signature_counts:
        raise ValueError(
            "algorithmic backend verification_residual_signature_counts missing "
            "verification residuals"
        )
    if verification_residual_signature_counts:
        if sum(verification_residual_signature_counts.values()) != sum(
            verification_residual_counts.values()
        ):
            raise ValueError(
                "algorithmic backend verification_residual_signature_counts does "
                "not match verification_residual_counts"
            )
    if verification_residual_by_method and not verification_residual_signature_by_method:
        raise ValueError(
            "algorithmic backend verification_residual_signature_by_method missing "
            "verification residual methods"
        )
    if verification_residual_signature_by_method:
        if sum(verification_residual_signature_by_method.values()) != sum(
            verification_residual_signature_counts.values()
        ):
            raise ValueError(
                "algorithmic backend verification_residual_signature_by_method "
                "does not match verification_residual_signature_counts"
            )
        signature_by_method_prefix, signature_by_method_suffix = split_compound_count_map(
            verification_residual_signature_by_method,
            label="algorithmic backend verification_residual_signature_by_method",
        )
        unknown_signature_methods = set(signature_by_method_prefix) - set(method_counts)
        if unknown_signature_methods:
            raise ValueError(
                "algorithmic backend verification_residual_signature_by_method has "
                "unknown methods"
            )
        if signature_by_method_suffix != verification_residual_signature_counts:
            raise ValueError(
                "algorithmic backend verification_residual_signature_by_method "
                "does not match verification_residual_signature_counts"
            )
        if verification_residual_by_method:
            residual_by_method_prefix, _ = split_compound_count_map(
                verification_residual_by_method,
                label="algorithmic backend verification_residual_by_method",
            )
            if signature_by_method_prefix != residual_by_method_prefix:
                raise ValueError(
                    "algorithmic backend verification_residual_signature_by_method "
                    "does not match verification_residual_by_method methods"
                )
    if publication_status_counts:
        if sum(publication_status_counts.values()) != attempts:
            raise ValueError(
                "algorithmic backend publication_status_counts do not match attempts"
            )
        if publication_status_counts.get("accepted", 0) != public_accepted:
            raise ValueError(
                "algorithmic backend publication_status_counts accepted does not "
                "match public_accepted"
            )
    if publication_status_by_method:
        if sum(publication_status_by_method.values()) != attempts:
            raise ValueError(
                "algorithmic backend publication_status_by_method does not "
                "match attempts"
            )
        publication_by_method_prefix, publication_by_method_suffix = (
            split_compound_count_map(
                publication_status_by_method,
                label="algorithmic backend publication_status_by_method",
            )
        )
        if publication_by_method_prefix != method_counts:
            raise ValueError(
                "algorithmic backend publication_status_by_method does not "
                "match method_counts"
            )
        if (
            publication_status_counts
            and publication_by_method_suffix != publication_status_counts
        ):
            raise ValueError(
                "algorithmic backend publication_status_by_method does not "
                "match publication_status_counts"
            )
    if fallback_status_counts:
        if sum(fallback_status_counts.values()) != attempts:
            raise ValueError(
                "algorithmic backend fallback_status_counts do not match attempts"
            )
        if fallback_status_counts.get("eligible", 0) != fallback_eligible:
            raise ValueError(
                "algorithmic backend fallback_status_counts eligible does not "
                "match fallback_eligible"
            )
    if fallback_status_by_method:
        if sum(fallback_status_by_method.values()) != attempts:
            raise ValueError(
                "algorithmic backend fallback_status_by_method does not "
                "match attempts"
            )
        fallback_by_method_prefix, fallback_by_method_suffix = (
            split_compound_count_map(
                fallback_status_by_method,
                label="algorithmic backend fallback_status_by_method",
            )
        )
        if fallback_by_method_prefix != method_counts:
            raise ValueError(
                "algorithmic backend fallback_status_by_method does not "
                "match method_counts"
            )
        if fallback_status_counts and fallback_by_method_suffix != fallback_status_counts:
            raise ValueError(
                "algorithmic backend fallback_status_by_method does not "
                "match fallback_status_counts"
            )
    if trace_level_counts and sum(trace_level_counts.values()) != attempts:
        raise ValueError("algorithmic backend trace_level_counts do not match attempts")
    if constant_policy_counts and sum(constant_policy_counts.values()) != attempts:
        raise ValueError(
            "algorithmic backend constant_policy_counts do not match attempts"
        )
    if (
        public_trace_level_counts
        and sum(public_trace_level_counts.values()) != public_accepted
    ):
        raise ValueError(
            "algorithmic backend public_trace_level_counts do not match public_accepted"
        )
    if (
        public_constant_policy_counts
        and sum(public_constant_policy_counts.values()) != public_accepted
    ):
        raise ValueError(
            "algorithmic backend public_constant_policy_counts do not match "
            "public_accepted"
        )
    if (
        fallback_trace_level_counts
        and sum(fallback_trace_level_counts.values()) != fallback_eligible
    ):
        raise ValueError(
            "algorithmic backend fallback_trace_level_counts do not match "
            "fallback_eligible"
        )
    if (
        fallback_constant_policy_counts
        and sum(fallback_constant_policy_counts.values()) != fallback_eligible
    ):
        raise ValueError(
            "algorithmic backend fallback_constant_policy_counts do not match "
            "fallback_eligible"
        )
    if (
        verification_evidence_counts
        and sum(verification_evidence_counts.values()) != attempts
    ):
        raise ValueError(
            "algorithmic backend verification_evidence_counts do not match attempts"
        )
    if (
        public_verification_evidence_counts
        and sum(public_verification_evidence_counts.values()) != public_accepted
    ):
        raise ValueError(
            "algorithmic backend public_verification_evidence_counts do not match "
            "public_accepted"
        )
    if (
        fallback_verification_evidence_counts
        and sum(fallback_verification_evidence_counts.values()) != fallback_eligible
    ):
        raise ValueError(
            "algorithmic backend fallback_verification_evidence_counts do not match "
            "fallback_eligible"
        )
    if verification_evidence_by_method:
        if sum(verification_evidence_by_method.values()) != attempts:
            raise ValueError(
                "algorithmic backend verification_evidence_by_method does not "
                "match attempts"
            )
        evidence_by_method_prefix, evidence_by_method_suffix = (
            split_compound_count_map(
                verification_evidence_by_method,
                label="algorithmic backend verification_evidence_by_method",
            )
        )
        if evidence_by_method_prefix != method_counts:
            raise ValueError(
                "algorithmic backend verification_evidence_by_method does not "
                "match method_counts"
            )
        if (
            verification_evidence_counts
            and evidence_by_method_suffix != verification_evidence_counts
        ):
            raise ValueError(
                "algorithmic backend verification_evidence_by_method does not "
                "match verification_evidence_counts"
            )
    if public_verification_evidence_by_method:
        if sum(public_verification_evidence_by_method.values()) != public_accepted:
            raise ValueError(
                "algorithmic backend public_verification_evidence_by_method does "
                "not match public_accepted"
            )
        public_evidence_by_method_prefix, public_evidence_by_method_suffix = (
            split_compound_count_map(
                public_verification_evidence_by_method,
                label=(
                    "algorithmic backend "
                    "public_verification_evidence_by_method"
                ),
            )
        )
        unknown_public_evidence_methods = set(public_evidence_by_method_prefix) - set(
            method_counts
        )
        if unknown_public_evidence_methods:
            raise ValueError(
                "algorithmic backend public_verification_evidence_by_method has "
                "unknown methods"
            )
        if (
            public_verification_evidence_counts
            and public_evidence_by_method_suffix != public_verification_evidence_counts
        ):
            raise ValueError(
                "algorithmic backend public_verification_evidence_by_method does "
                "not match public_verification_evidence_counts"
            )
    if fallback_verification_evidence_by_method:
        if sum(fallback_verification_evidence_by_method.values()) != fallback_eligible:
            raise ValueError(
                "algorithmic backend fallback_verification_evidence_by_method does "
                "not match fallback_eligible"
            )
        fallback_evidence_by_method_prefix, fallback_evidence_by_method_suffix = (
            split_compound_count_map(
                fallback_verification_evidence_by_method,
                label=(
                    "algorithmic backend "
                    "fallback_verification_evidence_by_method"
                ),
            )
        )
        unknown_fallback_evidence_methods = set(
            fallback_evidence_by_method_prefix
        ) - set(method_counts)
        if unknown_fallback_evidence_methods:
            raise ValueError(
                "algorithmic backend fallback_verification_evidence_by_method has "
                "unknown methods"
            )
        if (
            fallback_verification_evidence_counts
            and fallback_evidence_by_method_suffix
            != fallback_verification_evidence_counts
        ):
            raise ValueError(
                "algorithmic backend fallback_verification_evidence_by_method does "
                "not match fallback_verification_evidence_counts"
            )
    normalized_evidence_count = verification_evidence_counts.get(
        "normalized_differentiation", 0
    )
    public_normalized_evidence_count = public_verification_evidence_counts.get(
        "normalized_differentiation", 0
    )
    fallback_normalized_evidence_count = fallback_verification_evidence_counts.get(
        "normalized_differentiation", 0
    )
    if normalized_evidence_count and not verification_normalization_reason_counts:
        raise ValueError(
            "algorithmic backend verification_normalization_reason_counts missing "
            "normalized evidence"
        )
    if (
        sum(verification_normalization_reason_counts.values())
        != normalized_evidence_count
    ):
        raise ValueError(
            "algorithmic backend verification_normalization_reason_counts do not "
            "match normalized_differentiation evidence"
        )
    if (
        public_normalized_evidence_count
        and not public_verification_normalization_reason_counts
    ):
        raise ValueError(
            "algorithmic backend public_verification_normalization_reason_counts "
            "missing normalized evidence"
        )
    if (
        sum(public_verification_normalization_reason_counts.values())
        != public_normalized_evidence_count
    ):
        raise ValueError(
            "algorithmic backend public_verification_normalization_reason_counts "
            "do not match public normalized_differentiation evidence"
        )
    if (
        fallback_normalized_evidence_count
        and not fallback_verification_normalization_reason_counts
    ):
        raise ValueError(
            "algorithmic backend fallback_verification_normalization_reason_counts "
            "missing normalized evidence"
        )
    if (
        sum(fallback_verification_normalization_reason_counts.values())
        != fallback_normalized_evidence_count
    ):
        raise ValueError(
            "algorithmic backend fallback_verification_normalization_reason_counts "
            "do not match fallback normalized_differentiation evidence"
        )
    if verification_normalization_reason_by_method:
        if sum(verification_normalization_reason_by_method.values()) != sum(
            verification_normalization_reason_counts.values()
        ):
            raise ValueError(
                "algorithmic backend verification_normalization_reason_by_method "
                "does not match verification_normalization_reason_counts"
            )
        reason_by_method_prefix, reason_by_method_suffix = split_compound_count_map(
            verification_normalization_reason_by_method,
            label="algorithmic backend verification_normalization_reason_by_method",
        )
        unknown_reason_methods = set(reason_by_method_prefix) - set(method_counts)
        if unknown_reason_methods:
            raise ValueError(
                "algorithmic backend verification_normalization_reason_by_method "
                "has unknown methods"
            )
        if reason_by_method_suffix != verification_normalization_reason_counts:
            raise ValueError(
                "algorithmic backend verification_normalization_reason_by_method "
                "does not match verification_normalization_reason_counts"
            )
        if verification_evidence_by_method:
            normalized_method_counts = compound_count_map_prefix_for_suffix(
                verification_evidence_by_method,
                suffix="normalized_differentiation",
                label="algorithmic backend verification_evidence_by_method",
            )
            if reason_by_method_prefix != normalized_method_counts:
                raise ValueError(
                    "algorithmic backend verification_normalization_reason_by_method "
                    "does not match normalized evidence by method"
                )
    if public_verification_normalization_reason_by_method:
        if sum(public_verification_normalization_reason_by_method.values()) != sum(
            public_verification_normalization_reason_counts.values()
        ):
            raise ValueError(
                "algorithmic backend "
                "public_verification_normalization_reason_by_method does not match "
                "public_verification_normalization_reason_counts"
            )
        public_reason_by_method_prefix, public_reason_by_method_suffix = (
            split_compound_count_map(
                public_verification_normalization_reason_by_method,
                label=(
                    "algorithmic backend "
                    "public_verification_normalization_reason_by_method"
                ),
            )
        )
        unknown_public_reason_methods = set(public_reason_by_method_prefix) - set(
            method_counts
        )
        if unknown_public_reason_methods:
            raise ValueError(
                "algorithmic backend "
                "public_verification_normalization_reason_by_method has unknown "
                "methods"
            )
        if (
            public_reason_by_method_suffix
            != public_verification_normalization_reason_counts
        ):
            raise ValueError(
                "algorithmic backend "
                "public_verification_normalization_reason_by_method does not match "
                "public_verification_normalization_reason_counts"
            )
        if public_verification_evidence_by_method:
            public_normalized_method_counts = compound_count_map_prefix_for_suffix(
                public_verification_evidence_by_method,
                suffix="normalized_differentiation",
                label=(
                    "algorithmic backend "
                    "public_verification_evidence_by_method"
                ),
            )
            if public_reason_by_method_prefix != public_normalized_method_counts:
                raise ValueError(
                    "algorithmic backend "
                    "public_verification_normalization_reason_by_method does not "
                    "match public normalized evidence by method"
                )
    if fallback_verification_normalization_reason_by_method:
        if sum(fallback_verification_normalization_reason_by_method.values()) != sum(
            fallback_verification_normalization_reason_counts.values()
        ):
            raise ValueError(
                "algorithmic backend "
                "fallback_verification_normalization_reason_by_method does not "
                "match fallback_verification_normalization_reason_counts"
            )
        fallback_reason_by_method_prefix, fallback_reason_by_method_suffix = (
            split_compound_count_map(
                fallback_verification_normalization_reason_by_method,
                label=(
                    "algorithmic backend "
                    "fallback_verification_normalization_reason_by_method"
                ),
            )
        )
        unknown_fallback_reason_methods = set(
            fallback_reason_by_method_prefix
        ) - set(method_counts)
        if unknown_fallback_reason_methods:
            raise ValueError(
                "algorithmic backend "
                "fallback_verification_normalization_reason_by_method has unknown "
                "methods"
            )
        if (
            fallback_reason_by_method_suffix
            != fallback_verification_normalization_reason_counts
        ):
            raise ValueError(
                "algorithmic backend "
                "fallback_verification_normalization_reason_by_method does not "
                "match fallback_verification_normalization_reason_counts"
            )
        if fallback_verification_evidence_by_method:
            fallback_normalized_method_counts = compound_count_map_prefix_for_suffix(
                fallback_verification_evidence_by_method,
                suffix="normalized_differentiation",
                label=(
                    "algorithmic backend "
                    "fallback_verification_evidence_by_method"
                ),
            )
            if fallback_reason_by_method_prefix != fallback_normalized_method_counts:
                raise ValueError(
                    "algorithmic backend "
                    "fallback_verification_normalization_reason_by_method does not "
                    "match fallback normalized evidence by method"
                )
    if attempts and "verification_normalization_pass_count_counts" not in raw:
        raise ValueError(
            "algorithmic backend verification_normalization_pass_count_counts "
            "missing attempts"
        )
    if verification_normalization_pass_count_counts:
        validate_nonnegative_integer_key_count_map(
            verification_normalization_pass_count_counts,
            label="algorithmic backend verification_normalization_pass_count_counts",
        )
        if sum(verification_normalization_pass_count_counts.values()) != attempts:
            raise ValueError(
                "algorithmic backend verification_normalization_pass_count_counts "
                "do not match attempts"
            )
        if max(
            int(key) for key in verification_normalization_pass_count_counts
        ) != max_verification_normalization_passes:
            raise ValueError(
                "algorithmic backend max_verification_normalization_passes does "
                "not match pass counts"
            )
    elif attempts:
        raise ValueError(
            "algorithmic backend verification_normalization_pass_count_counts "
            "missing attempts"
        )
    if public_accepted and not public_verification_normalization_pass_count_counts:
        raise ValueError(
            "algorithmic backend public_verification_normalization_pass_count_counts "
            "missing public accepted candidates"
        )
    if public_verification_normalization_pass_count_counts:
        validate_nonnegative_integer_key_count_map(
            public_verification_normalization_pass_count_counts,
            label=(
                "algorithmic backend "
                "public_verification_normalization_pass_count_counts"
            ),
        )
        if (
            sum(public_verification_normalization_pass_count_counts.values())
            != public_accepted
        ):
            raise ValueError(
                "algorithmic backend "
                "public_verification_normalization_pass_count_counts do not "
                "match public_accepted"
            )
        if (
            max(int(key) for key in public_verification_normalization_pass_count_counts)
            != public_max_verification_normalization_passes
        ):
            raise ValueError(
                "algorithmic backend public_max_verification_normalization_passes "
                "does not match public pass counts"
            )
    if fallback_eligible and not fallback_verification_normalization_pass_count_counts:
        raise ValueError(
            "algorithmic backend fallback_verification_normalization_pass_count_counts "
            "missing fallback accepted candidates"
        )
    if fallback_verification_normalization_pass_count_counts:
        validate_nonnegative_integer_key_count_map(
            fallback_verification_normalization_pass_count_counts,
            label=(
                "algorithmic backend "
                "fallback_verification_normalization_pass_count_counts"
            ),
        )
        if (
            sum(fallback_verification_normalization_pass_count_counts.values())
            != fallback_eligible
        ):
            raise ValueError(
                "algorithmic backend "
                "fallback_verification_normalization_pass_count_counts do not "
                "match fallback_eligible"
            )
        if (
            max(
                int(key)
                for key in fallback_verification_normalization_pass_count_counts
            )
            != fallback_max_verification_normalization_passes
        ):
            raise ValueError(
                "algorithmic backend fallback_max_verification_normalization_passes "
                "does not match fallback pass counts"
            )
    if attempts and "verification_normalization_pass_count_by_method" not in raw:
        raise ValueError(
            "algorithmic backend verification_normalization_pass_count_by_method "
            "missing attempts"
        )
    if verification_normalization_pass_count_by_method:
        if sum(verification_normalization_pass_count_by_method.values()) != attempts:
            raise ValueError(
                "algorithmic backend verification_normalization_pass_count_by_method "
                "does not match attempts"
            )
        pass_by_method_prefix, pass_by_method_suffix = split_compound_count_map(
            verification_normalization_pass_count_by_method,
            label=(
                "algorithmic backend "
                "verification_normalization_pass_count_by_method"
            ),
        )
        validate_nonnegative_integer_key_count_map(
            pass_by_method_suffix,
            label=(
                "algorithmic backend "
                "verification_normalization_pass_count_by_method suffixes"
            ),
        )
        if pass_by_method_prefix != method_counts:
            raise ValueError(
                "algorithmic backend verification_normalization_pass_count_by_method "
                "does not match method_counts"
            )
        if pass_by_method_suffix != verification_normalization_pass_count_counts:
            raise ValueError(
                "algorithmic backend verification_normalization_pass_count_by_method "
                "does not match pass counts"
            )
    elif attempts:
        raise ValueError(
            "algorithmic backend verification_normalization_pass_count_by_method "
            "missing attempts"
        )
    if public_verification_normalization_pass_count_by_method:
        if (
            sum(public_verification_normalization_pass_count_by_method.values())
            != public_accepted
        ):
            raise ValueError(
                "algorithmic backend "
                "public_verification_normalization_pass_count_by_method does "
                "not match public_accepted"
            )
        public_pass_by_method_prefix, public_pass_by_method_suffix = (
            split_compound_count_map(
                public_verification_normalization_pass_count_by_method,
                label=(
                    "algorithmic backend "
                    "public_verification_normalization_pass_count_by_method"
                ),
            )
        )
        validate_nonnegative_integer_key_count_map(
            public_pass_by_method_suffix,
            label=(
                "algorithmic backend "
                "public_verification_normalization_pass_count_by_method suffixes"
            ),
        )
        unknown_public_pass_methods = set(public_pass_by_method_prefix) - set(
            method_counts
        )
        if unknown_public_pass_methods:
            raise ValueError(
                "algorithmic backend "
                "public_verification_normalization_pass_count_by_method has "
                "unknown methods"
            )
        if (
            public_pass_by_method_suffix
            != public_verification_normalization_pass_count_counts
        ):
            raise ValueError(
                "algorithmic backend "
                "public_verification_normalization_pass_count_by_method does "
                "not match public pass counts"
            )
    if fallback_verification_normalization_pass_count_by_method:
        if (
            sum(fallback_verification_normalization_pass_count_by_method.values())
            != fallback_eligible
        ):
            raise ValueError(
                "algorithmic backend "
                "fallback_verification_normalization_pass_count_by_method does "
                "not match fallback_eligible"
            )
        fallback_pass_by_method_prefix, fallback_pass_by_method_suffix = (
            split_compound_count_map(
                fallback_verification_normalization_pass_count_by_method,
                label=(
                    "algorithmic backend "
                    "fallback_verification_normalization_pass_count_by_method"
                ),
            )
        )
        validate_nonnegative_integer_key_count_map(
            fallback_pass_by_method_suffix,
            label=(
                "algorithmic backend "
                "fallback_verification_normalization_pass_count_by_method suffixes"
            ),
        )
        unknown_fallback_pass_methods = set(fallback_pass_by_method_prefix) - set(
            method_counts
        )
        if unknown_fallback_pass_methods:
            raise ValueError(
                "algorithmic backend "
                "fallback_verification_normalization_pass_count_by_method has "
                "unknown methods"
            )
        if (
            fallback_pass_by_method_suffix
            != fallback_verification_normalization_pass_count_counts
        ):
            raise ValueError(
                "algorithmic backend "
                "fallback_verification_normalization_pass_count_by_method does "
                "not match fallback pass counts"
            )

    verified_count = verification_status_counts.get("verified", 0) + (
        verification_status_counts.get("verified_under_conditions", 0)
    )
    failed_or_blocked_count = rejected_count
    required_condition_count = sum(required_condition_counts.values())
    budget_exceeded_count = residual_reason_counts.get("budget_exceeded", 0)
    if (
        method_probe_budget_exhausted + verification_budget_exceeded
        > budget_exceeded_count
    ):
        raise ValueError("algorithmic backend budget split exceeds budget count")
    backend_verification_pressure = classify_backend_verification_pressure(
        attempts=attempts,
        verification_checks_used_total=verification_checks_used_total,
        verification_elapsed_ms=float(verification_elapsed_ms),
        max_verification_normalization_passes=(
            max_verification_normalization_passes
        ),
    )
    failed = unverified_public_acceptances + unverified_fallback_acceptances
    return {
        "total_cases": attempts,
        "passed": attempts - failed,
        "failed": failed,
        "backend_attempts": attempts,
        "backend_public_accepted": public_accepted,
        "backend_unverified_public_acceptances": unverified_public_acceptances,
        "backend_fallback_eligible": fallback_eligible,
        "backend_unverified_fallback_acceptances": unverified_fallback_acceptances,
        "backend_verified_count": verified_count,
        "backend_failed_or_blocked_count": failed_or_blocked_count,
        "backend_required_condition_count": required_condition_count,
        "backend_budget_exceeded_count": budget_exceeded_count,
        "backend_method_probe_budget_exhausted_count": method_probe_budget_exhausted,
        "backend_verification_budget_exceeded_count": verification_budget_exceeded,
        "backend_method_probes_used_total": method_probes_used_total,
        "backend_verification_checks_used_total": verification_checks_used_total,
        "backend_verification_elapsed_ms": round(float(verification_elapsed_ms), 3),
        "backend_verification_pressure": backend_verification_pressure,
        "backend_verification_pressure_status": (
            backend_verification_pressure["status"]
        ),
        "backend_max_verification_normalization_passes": (
            max_verification_normalization_passes
        ),
        "backend_public_max_verification_normalization_passes": (
            public_max_verification_normalization_passes
        ),
        "backend_fallback_max_verification_normalization_passes": (
            fallback_max_verification_normalization_passes
        ),
        "backend_assumption_exprs": assumption_exprs,
        "backend_public_assumption_exprs": public_assumption_exprs,
        "backend_fallback_assumption_exprs": fallback_assumption_exprs,
        "backend_mode_counts": mode_counts,
        "backend_method_counts": method_counts,
        "backend_method_probe_usage_by_method": method_probe_usage_by_method,
        "backend_method_probe_attempt_counts": method_probe_attempt_counts,
        "backend_method_probe_candidate_counts": method_probe_candidate_counts,
        "backend_method_probe_no_match_counts": method_probe_no_match_counts,
        "backend_method_probe_no_match_reason_counts": (
            method_probe_no_match_reason_counts
        ),
        "backend_verification_check_usage_by_method": verification_check_usage_by_method,
        "backend_verification_status_by_method": verification_status_by_method,
        "backend_residual_reason_by_method": residual_reason_by_method,
        "backend_verification_blocker_counts": verification_blocker_counts,
        "backend_verification_blocker_by_method": verification_blocker_by_method,
        "backend_failure_class_counts": failure_class_counts,
        "backend_failure_class_by_method": failure_class_by_method,
        "backend_verification_residual_counts": verification_residual_counts,
        "backend_verification_residual_by_method": verification_residual_by_method,
        "backend_verification_residual_kind_counts": verification_residual_kind_counts,
        "backend_verification_residual_kind_by_method": (
            verification_residual_kind_by_method
        ),
        "backend_verification_residual_signature_counts": (
            verification_residual_signature_counts
        ),
        "backend_verification_residual_signature_by_method": (
            verification_residual_signature_by_method
        ),
        "backend_publication_status_counts": publication_status_counts,
        "backend_publication_status_by_method": publication_status_by_method,
        "backend_fallback_status_counts": fallback_status_counts,
        "backend_fallback_status_by_method": fallback_status_by_method,
        "backend_trace_level_counts": trace_level_counts,
        "backend_constant_policy_counts": constant_policy_counts,
        "backend_public_trace_level_counts": public_trace_level_counts,
        "backend_public_constant_policy_counts": public_constant_policy_counts,
        "backend_fallback_trace_level_counts": fallback_trace_level_counts,
        "backend_fallback_constant_policy_counts": fallback_constant_policy_counts,
        "backend_verification_evidence_counts": verification_evidence_counts,
        "backend_public_verification_evidence_counts": public_verification_evidence_counts,
        "backend_fallback_verification_evidence_counts": fallback_verification_evidence_counts,
        "backend_verification_evidence_by_method": verification_evidence_by_method,
        "backend_public_verification_evidence_by_method": (
            public_verification_evidence_by_method
        ),
        "backend_fallback_verification_evidence_by_method": (
            fallback_verification_evidence_by_method
        ),
        "backend_verification_normalization_reason_counts": (
            verification_normalization_reason_counts
        ),
        "backend_public_verification_normalization_reason_counts": (
            public_verification_normalization_reason_counts
        ),
        "backend_fallback_verification_normalization_reason_counts": (
            fallback_verification_normalization_reason_counts
        ),
        "backend_verification_normalization_reason_by_method": (
            verification_normalization_reason_by_method
        ),
        "backend_public_verification_normalization_reason_by_method": (
            public_verification_normalization_reason_by_method
        ),
        "backend_fallback_verification_normalization_reason_by_method": (
            fallback_verification_normalization_reason_by_method
        ),
        "backend_verification_normalization_pass_count_counts": (
            verification_normalization_pass_count_counts
        ),
        "backend_public_verification_normalization_pass_count_counts": (
            public_verification_normalization_pass_count_counts
        ),
        "backend_fallback_verification_normalization_pass_count_counts": (
            fallback_verification_normalization_pass_count_counts
        ),
        "backend_verification_normalization_pass_count_by_method": (
            verification_normalization_pass_count_by_method
        ),
        "backend_public_verification_normalization_pass_count_by_method": (
            public_verification_normalization_pass_count_by_method
        ),
        "backend_fallback_verification_normalization_pass_count_by_method": (
            fallback_verification_normalization_pass_count_by_method
        ),
        "backend_verification_status_counts": verification_status_counts,
        "backend_residual_reason_counts": residual_reason_counts,
        "backend_required_condition_counts": required_condition_counts,
    }


def sanitize_runtime_case_rows(raw_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        row: dict[str, Any] = {}
        for key in (
            "name",
            "family",
            "argument_regime",
            "point_regime",
            "domain_regime",
            "trace_regime",
            "presentation_regime",
            "outcome",
            "status",
        ):
            value = raw_row.get(key)
            if isinstance(value, str):
                row[key] = value
        elapsed = raw_row.get("wall_elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            row["wall_elapsed_seconds"] = round(float(elapsed), 3)
        if row:
            rows.append(row)
    return rows


def sanitize_runtime_group_rows(raw_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        row: dict[str, Any] = {}
        for key in ("axis", "group", "slowest_case"):
            value = raw_row.get(key)
            if isinstance(value, str):
                row[key] = value
        case_count = raw_row.get("case_count")
        if isinstance(case_count, int):
            row["case_count"] = case_count
        for key in (
            "total_elapsed_seconds",
            "avg_case_ms",
            "max_elapsed_seconds",
        ):
            value = raw_row.get(key)
            if isinstance(value, (int, float)):
                row[key] = round(float(value), 3)
        if row:
            rows.append(row)
    return rows


def sanitize_runtime_distribution(raw_row: Any) -> dict[str, Any]:
    if not isinstance(raw_row, dict):
        return {}
    row: dict[str, Any] = {}
    timed_case_count = raw_row.get("timed_case_count")
    if isinstance(timed_case_count, int):
        row["timed_case_count"] = timed_case_count
    for key in (
        "total_elapsed_seconds",
        "avg_case_ms",
        "p95_case_ms",
        "max_case_ms",
    ):
        value = raw_row.get(key)
        if isinstance(value, (int, float)):
            row[key] = round(float(value), 3)
    return row


def sanitize_runtime_concentration(raw_row: Any) -> dict[str, Any]:
    if not isinstance(raw_row, dict):
        return {}
    row: dict[str, Any] = {}
    timed_case_count = raw_row.get("timed_case_count")
    if isinstance(timed_case_count, int):
        row["timed_case_count"] = timed_case_count
    for key in (
        "slowest_case",
        "slowest_family",
        "slowest_argument_regime",
        "slowest_point_regime",
        "slowest_domain_regime",
        "slowest_trace_regime",
        "slowest_presentation_regime",
    ):
        value = raw_row.get(key)
        if isinstance(value, str):
            row[key] = value
    for key in (
        "total_elapsed_seconds",
        "slowest_case_ms",
        "slowest_case_share_percent",
        "top_3_share_percent",
    ):
        value = raw_row.get(key)
        if isinstance(value, (int, float)):
            row[key] = round(float(value), 3)
    return row


def classify_calculus_runtime_pressure(
    distribution: dict[str, Any],
    concentration: dict[str, Any],
    *,
    sample_basis: str = "cold_inclusive",
) -> dict[str, Any]:
    signals: list[tuple[str, str, str]] = []
    max_case_ms = distribution.get("max_case_ms")
    if isinstance(max_case_ms, (int, float)):
        if max_case_ms >= CALCULUS_RUNTIME_PRESSURE_RISK_MAX_MS:
            signals.append(("risk", "max", f"max_case_ms={max_case_ms:.3f}"))
        elif max_case_ms >= CALCULUS_RUNTIME_PRESSURE_WATCH_MAX_MS:
            signals.append(("watch", "max", f"max_case_ms={max_case_ms:.3f}"))
    p95_case_ms = distribution.get("p95_case_ms")
    if isinstance(p95_case_ms, (int, float)):
        if p95_case_ms >= CALCULUS_RUNTIME_PRESSURE_RISK_P95_MS:
            signals.append(("risk", "p95", f"p95_case_ms={p95_case_ms:.3f}"))
        elif p95_case_ms >= CALCULUS_RUNTIME_PRESSURE_WATCH_P95_MS:
            signals.append(("watch", "p95", f"p95_case_ms={p95_case_ms:.3f}"))

    timed_case_count = distribution.get("timed_case_count")
    if not isinstance(timed_case_count, int):
        timed_case_count = concentration.get("timed_case_count")
    top_3_share = concentration.get("top_3_share_percent")
    if (
        isinstance(timed_case_count, int)
        and timed_case_count >= CALCULUS_RUNTIME_PRESSURE_CONCENTRATION_MIN_CASES
        and isinstance(top_3_share, (int, float))
    ):
        if top_3_share >= CALCULUS_RUNTIME_PRESSURE_RISK_TOP3_SHARE_PERCENT:
            signals.append(
                ("risk", "top3", f"top3_share_percent={top_3_share:.1f}")
            )
        elif top_3_share >= CALCULUS_RUNTIME_PRESSURE_WATCH_TOP3_SHARE_PERCENT:
            signals.append(
                ("watch", "top3", f"top3_share_percent={top_3_share:.1f}")
            )

    severity_rank = {"ok": 0, "watch": 1, "risk": 2}
    status = "ok"
    primary_signal = "none"
    reason = "within runtime pressure thresholds"
    if signals:
        status, primary_signal, reason = max(
            signals,
            key=lambda signal: severity_rank[signal[0]],
        )

    pressure: dict[str, Any] = {
        "status": status,
        "primary_signal": primary_signal,
        "reason": reason,
        "sample_basis": sample_basis,
    }
    for source, key in (
        (distribution, "timed_case_count"),
        (distribution, "avg_case_ms"),
        (distribution, "p95_case_ms"),
        (distribution, "max_case_ms"),
        (concentration, "top_3_share_percent"),
        (concentration, "slowest_case"),
        (concentration, "slowest_family"),
    ):
        value = source.get(key)
        if isinstance(value, (int, float, str)):
            pressure[key] = value
    return pressure


def classify_calculus_runtime_measurement(
    distribution: dict[str, Any],
    pressure: dict[str, Any],
) -> dict[str, Any]:
    max_case_ms = distribution.get("max_case_ms")
    p95_case_ms = distribution.get("p95_case_ms")
    pressure_status = pressure.get("status")
    measurement: dict[str, Any] = {
        "mode": "process_per_case",
        "status": "unclassified",
        "guidance": "inspect runtime distribution before selecting a runtime candidate",
    }
    if isinstance(max_case_ms, (int, float)):
        measurement["max_case_ms"] = round(float(max_case_ms), 3)
    if isinstance(p95_case_ms, (int, float)):
        measurement["p95_case_ms"] = round(float(p95_case_ms), 3)
    if isinstance(pressure_status, str):
        measurement["pressure_status"] = pressure_status

    if pressure_status in {"watch", "risk"}:
        measurement["status"] = "actionable_pressure"
        measurement["guidance"] = "triage runtime pressure with focused probes"
        return measurement

    if (
        pressure_status == "ok"
        and isinstance(max_case_ms, (int, float))
        and max_case_ms < CALCULUS_COMMAND_MATRIX_PROCESS_OVERHEAD_FLOOR_MAX_MS
    ):
        measurement["status"] = "process_overhead_floor"
        measurement["guidance"] = (
            "require embedded or profiler evidence before treating this as an engine hotspot"
        )
        return measurement

    if pressure_status == "ok":
        measurement["status"] = "routine_process_timing"
        measurement["guidance"] = "prefer embedded or profiler confirmation before optimizing"
    return measurement


def add_runtime_observability_metrics(
    metrics: dict[str, Any],
    raw: dict[str, Any],
    *,
    prefix: str,
    group_keys: tuple[str, ...],
) -> None:
    distribution = sanitize_runtime_distribution(raw.get("runtime_distribution"))
    if distribution:
        metrics[f"{prefix}_runtime_distribution"] = distribution
    concentration = sanitize_runtime_concentration(raw.get("runtime_concentration"))
    if concentration:
        metrics[f"{prefix}_runtime_concentration"] = concentration
    warm_distribution = sanitize_runtime_distribution(
        raw.get("warm_runtime_distribution")
    )
    if warm_distribution:
        metrics[f"{prefix}_warm_runtime_distribution"] = warm_distribution
    warm_concentration = sanitize_runtime_concentration(
        raw.get("warm_runtime_concentration")
    )
    if warm_concentration:
        metrics[f"{prefix}_warm_runtime_concentration"] = warm_concentration
    pressure_distribution = warm_distribution or distribution
    pressure_concentration = warm_concentration or concentration
    if pressure_distribution:
        runtime_pressure = classify_calculus_runtime_pressure(
            pressure_distribution,
            pressure_concentration,
            sample_basis="warm" if warm_distribution else "cold_inclusive",
        )
        metrics[f"{prefix}_runtime_pressure"] = runtime_pressure
        metrics[f"{prefix}_runtime_measurement"] = (
            classify_calculus_runtime_measurement(
                pressure_distribution,
                runtime_pressure,
            )
        )
    slowest_cases = sanitize_runtime_case_rows(raw.get("slowest_cases"))
    if slowest_cases:
        metrics[f"{prefix}_slowest_cases"] = slowest_cases
    cold_start_case = sanitize_runtime_case_rows([raw.get("cold_start_case")])
    if cold_start_case:
        metrics[f"{prefix}_cold_start_case"] = cold_start_case[0]
    warm_slowest_cases = sanitize_runtime_case_rows(raw.get("warm_slowest_cases"))
    if warm_slowest_cases:
        metrics[f"{prefix}_warm_slowest_cases"] = warm_slowest_cases
    for group_key in group_keys:
        rows = sanitize_runtime_group_rows(raw.get(f"runtime_by_{group_key}"))
        if rows:
            metrics[f"{prefix}_runtime_by_{group_key}"] = rows
        warm_rows = sanitize_runtime_group_rows(
            raw.get(f"warm_runtime_by_{group_key}")
        )
        if warm_rows:
            metrics[f"{prefix}_warm_runtime_by_{group_key}"] = warm_rows


def sanitize_phase_runtime_case_rows(
    raw_rows: Any,
    *,
    elapsed_key: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        name = raw_row.get("name")
        elapsed = raw_row.get(elapsed_key)
        if not isinstance(name, str) or not isinstance(elapsed, (int, float)):
            continue
        row: dict[str, Any] = {
            "name": name,
            elapsed_key: round(float(elapsed), 3),
        }
        for key in (
            "family",
            "argument_regime",
            "domain_regime",
            "trace_regime",
            "presentation_regime",
            "calculus_maturity_block",
            "calculus_block_gate",
            "positive_quadratic_policy_cluster",
            "variable_power_policy_cluster",
            "antiderivative_verification_mode",
        ):
            value = raw_row.get(key)
            if isinstance(value, str):
                row[key] = value
        steps_count = raw_row.get("steps_count")
        if isinstance(steps_count, int):
            row["steps_count"] = steps_count
        step_rule_names = raw_row.get("step_rule_names")
        if isinstance(step_rule_names, list):
            names = [
                value
                for value in step_rule_names
                if isinstance(value, str) and value.strip()
            ]
            if names:
                row["step_rule_names"] = names[:8]
        rows.append(row)
    return rows


def sanitize_diff_exact_square_runtime_pair_rows(
    raw_rows: Any,
) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        name = raw_row.get("name")
        baseline_case = raw_row.get("baseline_case")
        exact_square_case = raw_row.get("exact_square_case")
        if (
            not isinstance(name, str)
            or not isinstance(baseline_case, str)
            or not isinstance(exact_square_case, str)
        ):
            continue
        row: dict[str, Any] = {
            "name": name,
            "baseline_case": baseline_case,
            "exact_square_case": exact_square_case,
        }
        for key in (
            "family",
            "baseline_argument_regime",
            "exact_square_argument_regime",
            "baseline_presentation_regime",
            "exact_square_presentation_regime",
        ):
            value = raw_row.get(key)
            if isinstance(value, str):
                row[key] = value
        for key in (
            "baseline_case_ms",
            "exact_square_case_ms",
            "delta_ms",
            "ratio",
        ):
            value = raw_row.get(key)
            if isinstance(value, (int, float)):
                row[key] = round(float(value), 3)
        rows.append(row)
    return rows


def sanitize_payload_case_rows(
    raw_rows: Any,
    *,
    metric_key: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        name = raw_row.get("name")
        metric = raw_row.get(metric_key)
        if not isinstance(name, str) or not isinstance(metric, int):
            continue
        row: dict[str, Any] = {
            "name": name,
            metric_key: metric,
        }
        for key in (
            "required_display_count",
            "expected_step_substring_count",
        ):
            value = raw_row.get(key)
            if isinstance(value, int):
                row[key] = value
        for key in (
            "family",
            "argument_regime",
            "domain_regime",
            "trace_regime",
            "presentation_regime",
            "calculus_maturity_block",
            "calculus_block_gate",
        ):
            value = raw_row.get(key)
            if isinstance(value, str):
                row[key] = value
        rows.append(row)
    return rows


def sanitize_verification_mode_runtime_rows(raw_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        mode = raw_row.get("mode")
        case_count = raw_row.get("case_count")
        if not isinstance(mode, str) or not isinstance(case_count, int):
            continue
        row: dict[str, Any] = {
            "mode": mode,
            "case_count": case_count,
        }
        for key in ("total_elapsed_seconds", "avg_case_ms", "max_elapsed_seconds"):
            value = raw_row.get(key)
            if isinstance(value, (int, float)):
                row[key] = round(float(value), 3)
        slowest_case = raw_row.get("slowest_case")
        if isinstance(slowest_case, str):
            row["slowest_case"] = slowest_case
        rows.append(row)
    return rows


def sanitize_residual_cause_runtime_rows(raw_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        cause = raw_row.get("cause")
        case_count = raw_row.get("case_count")
        if not isinstance(cause, str) or not isinstance(case_count, int):
            continue
        row: dict[str, Any] = {
            "cause": cause,
            "case_count": case_count,
        }
        for key in ("total_elapsed_seconds", "avg_case_ms", "max_elapsed_seconds"):
            value = raw_row.get(key)
            if isinstance(value, (int, float)):
                row[key] = round(float(value), 3)
        slowest_case = raw_row.get("slowest_case")
        if isinstance(slowest_case, str):
            row["slowest_case"] = slowest_case
        rows.append(row)
    return rows


def sanitize_residual_cause_family_runtime_rows(raw_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        cause_family = raw_row.get("cause_family")
        case_count = raw_row.get("case_count")
        if not isinstance(cause_family, str) or not isinstance(case_count, int):
            continue
        row: dict[str, Any] = {
            "cause_family": cause_family,
            "case_count": case_count,
        }
        for key in ("total_elapsed_seconds", "avg_case_ms", "max_elapsed_seconds"):
            value = raw_row.get(key)
            if isinstance(value, (int, float)):
                row[key] = round(float(value), 3)
        slowest_case = raw_row.get("slowest_case")
        if isinstance(slowest_case, str):
            row["slowest_case"] = slowest_case
        rows.append(row)
    return rows


def sanitize_residual_public_phase_case_rows(raw_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        name = raw_row.get("name")
        if not isinstance(name, str):
            continue
        row: dict[str, Any] = {"name": name}
        for key in (
            "integrate_elapsed_seconds",
            "cli_total_seconds",
            "cli_simplify_ms",
            "public_overhead_seconds",
            "public_overhead_share_percent",
        ):
            value = raw_row.get(key)
            if isinstance(value, (int, float)):
                row[key] = round(float(value), 6)
        for key in (
            "required_display_count",
            "step_text_char_count",
            "stdout_bytes",
        ):
            value = raw_row.get(key)
            if isinstance(value, int):
                row[key] = value
        for key in (
            "residual_cause",
            "family",
            "domain_regime",
            "trace_regime",
            "presentation_regime",
            "calculus_maturity_block",
            "calculus_block_gate",
        ):
            value = raw_row.get(key)
            if isinstance(value, str):
                row[key] = value
        rows.append(row)
    return rows


def sanitize_residual_public_phase_group_rows(raw_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        cause = raw_row.get("cause")
        case_count = raw_row.get("case_count")
        if not isinstance(cause, str) or not isinstance(case_count, int):
            continue
        row: dict[str, Any] = {
            "cause": cause,
            "case_count": case_count,
        }
        for key in (
            "integrate_total_seconds",
            "cli_total_seconds",
            "public_overhead_total_seconds",
            "public_overhead_share_percent",
            "avg_required_display_count",
            "avg_step_text_char_count",
        ):
            value = raw_row.get(key)
            if isinstance(value, (int, float)):
                row[key] = round(float(value), 6)
        slowest_case = raw_row.get("slowest_case")
        if isinstance(slowest_case, str):
            row["slowest_case"] = slowest_case
        rows.append(row)
    return rows


def sanitize_residual_public_phase_cause_family_group_rows(
    raw_rows: Any,
) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        cause_family = raw_row.get("cause_family")
        case_count = raw_row.get("case_count")
        if not isinstance(cause_family, str) or not isinstance(case_count, int):
            continue
        row: dict[str, Any] = {
            "cause_family": cause_family,
            "case_count": case_count,
        }
        for key in (
            "integrate_total_seconds",
            "cli_total_seconds",
            "public_overhead_total_seconds",
            "public_overhead_share_percent",
            "avg_required_display_count",
            "avg_step_text_char_count",
        ):
            value = raw_row.get(key)
            if isinstance(value, (int, float)):
                row[key] = round(float(value), 6)
        slowest_case = raw_row.get("slowest_case")
        if isinstance(slowest_case, str):
            row["slowest_case"] = slowest_case
        rows.append(row)
    return rows


def sanitize_residual_shape_orientation_probe_rows(raw_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        name = raw_row.get("name")
        expression_shape = raw_row.get("expression_shape")
        orientation = raw_row.get("orientation")
        status = raw_row.get("status")
        if not all(
            isinstance(value, str)
            for value in (name, expression_shape, orientation, status)
        ):
            continue
        row: dict[str, Any] = {
            "name": name,
            "expression_shape": expression_shape,
            "orientation": orientation,
            "status": status,
        }
        steps_mode = raw_row.get("steps_mode")
        if isinstance(steps_mode, str):
            row["steps_mode"] = steps_mode
        for key in (
            "wall_elapsed_seconds",
            "cli_parse_us",
            "cli_simplify_us",
            "cli_total_us",
        ):
            value = raw_row.get(key)
            if isinstance(value, (int, float)):
                row[key] = round(float(value), 6)
        for key in ("stdout_bytes", "stderr_bytes"):
            value = raw_row.get(key)
            if isinstance(value, int):
                row[key] = value
        required_display_count = raw_row.get("required_display_count")
        if isinstance(required_display_count, int):
            row["required_display_count"] = required_display_count
        for key in ("error", "result"):
            value = raw_row.get(key)
            if isinstance(value, str):
                row[key] = value
        rows.append(row)
    return rows


def residual_shape_orientation_probe_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    first_problem_row: dict[str, Any] | None = None
    for row in rows:
        status = row.get("status")
        if not isinstance(status, str):
            continue
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "pass" and first_problem_row is None:
            first_problem_row = row

    counted_rows = [
        row for row in rows if isinstance(row.get("required_display_count"), int)
    ]
    if not counted_rows:
        summary: dict[str, Any] = {"probe_count": len(rows)}
        if status_counts:
            summary["status_counts"] = status_counts
        if first_problem_row is not None:
            add_residual_shape_orientation_problem_summary(summary, first_problem_row)
        return summary
    max_row = max(
        counted_rows,
        key=lambda row: (
            int(row["required_display_count"]),
            str(row.get("name", "")),
            str(row.get("steps_mode", "")),
        ),
    )
    total_required_display_count = sum(
        int(row["required_display_count"]) for row in counted_rows
    )
    summary: dict[str, Any] = {
        "probe_count": len(rows),
        "counted_probe_count": len(counted_rows),
        "max_required_display_count": int(max_row["required_display_count"]),
        "avg_required_display_count": round(
            total_required_display_count / len(counted_rows), 3
        ),
    }
    if status_counts:
        summary["status_counts"] = status_counts
    if first_problem_row is not None:
        add_residual_shape_orientation_problem_summary(summary, first_problem_row)
    for key in ("name", "expression_shape", "orientation", "steps_mode", "status"):
        value = max_row.get(key)
        if isinstance(value, str):
            summary[f"max_{key}"] = value
    return summary


def add_residual_shape_orientation_problem_summary(
    summary: dict[str, Any],
    row: dict[str, Any],
) -> None:
    for source, target in (
        ("name", "first_problem_name"),
        ("status", "first_problem_status"),
        ("expression_shape", "first_problem_expression_shape"),
        ("orientation", "first_problem_orientation"),
        ("steps_mode", "first_problem_steps_mode"),
        ("error", "first_problem_error"),
    ):
        value = row.get(source)
        if isinstance(value, str):
            summary[target] = value


def parse_calculus_limit_command_matrix(output: str) -> dict[str, Any]:
    try:
        raw = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid limit command matrix json: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("limit command matrix json output is not an object")

    total = raw.get("total")
    status = raw.get("status")
    status_counts = raw.get("status_counts")
    issue_kind_counts = raw.get("issue_kind_counts", {})
    if not isinstance(total, int):
        raise ValueError("missing limit command matrix total")
    if not isinstance(status, str):
        raise ValueError("missing limit command matrix status")
    if not isinstance(status_counts, dict):
        raise ValueError("missing limit command matrix status_counts")
    if not isinstance(issue_kind_counts, dict):
        issue_kind_counts = {}

    passed = status_counts.get("pass", 0)
    raw_failed = status_counts.get("fail", 0)
    slow = status_counts.get("slow", 0)
    timeouts = status_counts.get("timeout", 0)
    if not all(
        isinstance(value, int)
        for value in (passed, raw_failed, slow, timeouts)
    ):
        raise ValueError("invalid limit command matrix status_counts")

    def int_metric(key: str) -> int | None:
        value = raw.get(key)
        return value if isinstance(value, int) else None

    def count_map(key: str) -> dict[str, int]:
        value = raw.get(key)
        if not isinstance(value, dict):
            return {}
        return {
            name: count
            for name, count in value.items()
            if isinstance(name, str) and isinstance(count, int)
        }

    problem_cases = sanitize_limit_command_problem_cases(raw.get("problem_cases"))
    residual_case_names = sanitize_string_list(raw.get("residual_case_names"))
    residual_cases_by_cause = sanitize_string_list_map(
        raw.get("residual_cases_by_cause")
    )
    problem_case_count = raw.get("problem_case_count")
    if not isinstance(problem_case_count, int):
        problem_case_count = len(problem_cases)

    metrics = {
        "matrix_status": status,
        "total_cases": total,
        "passed": passed,
        "failed": raw_failed + slow + timeouts,
        "raw_failed": raw_failed,
        "slow": slow,
        "timeouts": timeouts,
        "problem_case_count": problem_case_count,
        "problem_cases": problem_cases,
        "limit_residual_case_names": residual_case_names,
        "limit_residual_cases_by_cause": residual_cases_by_cause,
        "issue_kind_counts": {
            key: value
            for key, value in issue_kind_counts.items()
            if isinstance(key, str) and isinstance(value, int)
        },
    }
    for raw_key, metric_key in (
        ("supported_case_count", "limit_supported_case_count"),
        ("residual_case_count", "limit_residual_case_count"),
        ("warning_expected_case_count", "limit_warning_expected_case_count"),
        ("required_display_case_count", "limit_required_display_case_count"),
        ("step_checked_case_count", "limit_step_checked_case_count"),
        (
            "supported_step_unchecked_case_count",
            "limit_supported_step_unchecked_case_count",
        ),
        ("expected_step_substring_count", "limit_expected_step_substring_count"),
        ("distinct_required_display_count", "limit_distinct_required_display_count"),
        ("family_count", "limit_family_count"),
    ):
        value = int_metric(raw_key)
        if value is not None:
            metrics[metric_key] = value
    for raw_key, metric_key in (
        ("point_regime_counts", "limit_point_regime_counts"),
        ("domain_regime_counts", "limit_domain_regime_counts"),
        ("required_condition_regime_counts", "limit_required_condition_regime_counts"),
        ("outcome_counts", "limit_outcome_counts"),
        ("residual_cause_counts", "limit_residual_cause_counts"),
        ("residual_family_counts", "limit_residual_family_counts"),
        (
            "residual_cause_family_counts",
            "limit_residual_cause_family_counts",
        ),
        ("calculus_maturity_block_counts", "limit_calculus_maturity_block_counts"),
        ("calculus_block_gate_counts", "limit_calculus_block_gate_counts"),
        ("required_display_counts", "limit_required_display_counts"),
        ("trace_regime_counts", "limit_trace_regime_counts"),
        ("presentation_regime_counts", "limit_presentation_regime_counts"),
    ):
        counts = count_map(raw_key)
        if counts:
            metrics[metric_key] = counts
    add_runtime_observability_metrics(
        metrics,
        raw,
        prefix="limit",
        group_keys=("family", "point_regime", "domain_regime", "trace_regime"),
    )
    for raw_key, metric_key in (
        (
            "cli_parse_runtime_distribution",
            "limit_cli_parse_runtime_distribution",
        ),
        (
            "cli_simplify_runtime_distribution",
            "limit_cli_simplify_runtime_distribution",
        ),
        (
            "cli_total_runtime_distribution",
            "limit_cli_total_runtime_distribution",
        ),
        (
            "cli_public_overhead_runtime_distribution",
            "limit_cli_public_overhead_runtime_distribution",
        ),
    ):
        distribution = sanitize_runtime_distribution(raw.get(raw_key))
        if distribution:
            metrics[metric_key] = distribution
    for raw_key, metric_key, elapsed_key in (
        (
            "slowest_cli_parse_evaluations",
            "limit_slowest_cli_parse_evaluations",
            "cli_parse_elapsed_seconds",
        ),
        (
            "slowest_cli_simplify_evaluations",
            "limit_slowest_cli_simplify_evaluations",
            "cli_simplify_elapsed_seconds",
        ),
        (
            "slowest_cli_total_evaluations",
            "limit_slowest_cli_total_evaluations",
            "cli_total_elapsed_seconds",
        ),
        (
            "slowest_cli_public_overhead_evaluations",
            "limit_slowest_cli_public_overhead_evaluations",
            "cli_public_overhead_seconds",
        ),
    ):
        rows = sanitize_phase_runtime_case_rows(
            raw.get(raw_key),
            elapsed_key=elapsed_key,
        )
        if rows:
            metrics[metric_key] = rows
    return metrics


def parse_calculus_diff_command_matrix(output: str) -> dict[str, Any]:
    try:
        raw = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid diff command matrix json: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("diff command matrix json output is not an object")

    total = raw.get("total")
    status = raw.get("status")
    status_counts = raw.get("status_counts")
    issue_kind_counts = raw.get("issue_kind_counts", {})
    if not isinstance(total, int):
        raise ValueError("missing diff command matrix total")
    if not isinstance(status, str):
        raise ValueError("missing diff command matrix status")
    if not isinstance(status_counts, dict):
        raise ValueError("missing diff command matrix status_counts")
    if not isinstance(issue_kind_counts, dict):
        issue_kind_counts = {}

    passed = status_counts.get("pass", 0)
    raw_failed = status_counts.get("fail", 0)
    slow = status_counts.get("slow", 0)
    timeouts = status_counts.get("timeout", 0)
    if not all(
        isinstance(value, int)
        for value in (passed, raw_failed, slow, timeouts)
    ):
        raise ValueError("invalid diff command matrix status_counts")

    def int_metric(key: str) -> int | None:
        value = raw.get(key)
        return value if isinstance(value, int) else None

    def count_map(key: str) -> dict[str, int]:
        value = raw.get(key)
        if not isinstance(value, dict):
            return {}
        return {
            name: count
            for name, count in value.items()
            if isinstance(name, str) and isinstance(count, int)
        }

    problem_cases = sanitize_limit_command_problem_cases(raw.get("problem_cases"))
    residual_case_names = sanitize_string_list(raw.get("residual_case_names"))
    problem_case_count = raw.get("problem_case_count")
    if not isinstance(problem_case_count, int):
        problem_case_count = len(problem_cases)

    metrics = {
        "matrix_status": status,
        "total_cases": total,
        "passed": passed,
        "failed": raw_failed + slow + timeouts,
        "raw_failed": raw_failed,
        "slow": slow,
        "timeouts": timeouts,
        "problem_case_count": problem_case_count,
        "problem_cases": problem_cases,
        "diff_residual_case_names": residual_case_names,
        "issue_kind_counts": {
            key: value
            for key, value in issue_kind_counts.items()
            if isinstance(key, str) and isinstance(value, int)
        },
    }
    for raw_key, metric_key in (
        ("supported_case_count", "diff_supported_case_count"),
        ("residual_case_count", "diff_residual_case_count"),
        ("warning_expected_case_count", "diff_warning_expected_case_count"),
        ("required_display_case_count", "diff_required_display_case_count"),
        ("step_checked_case_count", "diff_step_checked_case_count"),
        (
            "supported_step_unchecked_case_count",
            "diff_supported_step_unchecked_case_count",
        ),
        ("expected_step_substring_count", "diff_expected_step_substring_count"),
        ("distinct_required_display_count", "diff_distinct_required_display_count"),
        ("family_count", "diff_family_count"),
    ):
        value = int_metric(raw_key)
        if value is not None:
            metrics[metric_key] = value
    for raw_key, metric_key in (
        ("argument_regime_counts", "diff_argument_regime_counts"),
        ("domain_regime_counts", "diff_domain_regime_counts"),
        ("outcome_counts", "diff_outcome_counts"),
        ("calculus_maturity_block_counts", "diff_calculus_maturity_block_counts"),
        ("calculus_block_gate_counts", "diff_calculus_block_gate_counts"),
        (
            "symbolic_radius_policy_cluster_counts",
            "diff_symbolic_radius_policy_cluster_counts",
        ),
        (
            "positive_quadratic_policy_cluster_counts",
            "diff_positive_quadratic_policy_cluster_counts",
        ),
        (
            "variable_power_policy_cluster_counts",
            "diff_variable_power_policy_cluster_counts",
        ),
        ("required_display_counts", "diff_required_display_counts"),
        ("trace_regime_counts", "diff_trace_regime_counts"),
        ("presentation_regime_counts", "diff_presentation_regime_counts"),
    ):
        counts = count_map(raw_key)
        if counts:
            metrics[metric_key] = counts
    add_runtime_observability_metrics(
        metrics,
        raw,
        prefix="diff",
        group_keys=(
            "family",
            "calculus_maturity_block",
            "calculus_block_gate",
            "domain_regime",
            "trace_regime",
            "presentation_regime",
            "positive_quadratic_policy_cluster",
            "variable_power_policy_cluster",
        ),
    )
    harness_check_distribution = sanitize_runtime_distribution(
        raw.get("harness_check_runtime_distribution")
    )
    if harness_check_distribution:
        metrics["diff_harness_check_runtime_distribution"] = (
            harness_check_distribution
        )
    for raw_key, metric_key in (
        (
            "cli_parse_runtime_distribution",
            "diff_cli_parse_runtime_distribution",
        ),
        (
            "cli_simplify_runtime_distribution",
            "diff_cli_simplify_runtime_distribution",
        ),
        (
            "cli_total_runtime_distribution",
            "diff_cli_total_runtime_distribution",
        ),
        (
            "cli_public_overhead_runtime_distribution",
            "diff_cli_public_overhead_runtime_distribution",
        ),
    ):
        distribution = sanitize_runtime_distribution(raw.get(raw_key))
        if distribution:
            metrics[metric_key] = distribution
    for raw_key, metric_key, elapsed_key in (
        (
            "slowest_process_evaluations",
            "diff_slowest_process_evaluations",
            "process_elapsed_seconds",
        ),
        (
            "slowest_cli_parse_evaluations",
            "diff_slowest_cli_parse_evaluations",
            "cli_parse_elapsed_seconds",
        ),
        (
            "slowest_cli_simplify_evaluations",
            "diff_slowest_cli_simplify_evaluations",
            "cli_simplify_elapsed_seconds",
        ),
        (
            "slowest_cli_total_evaluations",
            "diff_slowest_cli_total_evaluations",
            "cli_total_elapsed_seconds",
        ),
        (
            "slowest_cli_public_overhead_evaluations",
            "diff_slowest_cli_public_overhead_evaluations",
            "cli_public_overhead_seconds",
        ),
        (
            "slowest_harness_checks",
            "diff_slowest_harness_checks",
            "harness_check_elapsed_seconds",
        ),
    ):
        rows = sanitize_phase_runtime_case_rows(
            raw.get(raw_key),
            elapsed_key=elapsed_key,
        )
        if rows:
            metrics[metric_key] = rows
    exact_square_pairs = sanitize_diff_exact_square_runtime_pair_rows(
        raw.get("exact_square_runtime_pairs")
    )
    if exact_square_pairs:
        metrics["diff_exact_square_runtime_pairs"] = exact_square_pairs
    for raw_key, metric_key, payload_key in (
        (
            "largest_stdout_payload_cases",
            "diff_largest_stdout_payload_cases",
            "stdout_bytes",
        ),
        (
            "largest_step_trace_cases",
            "diff_largest_step_trace_cases",
            "step_text_char_count",
        ),
    ):
        rows = sanitize_payload_case_rows(
            raw.get(raw_key),
            metric_key=payload_key,
        )
        if rows:
            metrics[metric_key] = rows
    add_policy_cluster_consolidation_metrics(
        metrics,
        source_key="diff_symbolic_radius_policy_cluster_counts",
        consolidated_key="diff_symbolic_radius_consolidated_policy_cluster_counts",
        candidate_key="diff_symbolic_radius_consolidation_candidate_counts",
        shared_policy_clusters=(
            CALCULUS_DIFF_SYMBOLIC_RADIUS_POLICY_CLUSTERS_WITH_SHARED_POLICY
        ),
    )
    add_policy_cluster_consolidation_metrics(
        metrics,
        source_key="diff_positive_quadratic_policy_cluster_counts",
        consolidated_key="diff_positive_quadratic_consolidated_policy_cluster_counts",
        candidate_key="diff_positive_quadratic_consolidation_candidate_counts",
        shared_policy_clusters=(
            CALCULUS_DIFF_POSITIVE_QUADRATIC_POLICY_CLUSTERS_WITH_SHARED_POLICY
        ),
    )
    add_runtime_cluster_candidate_metrics(
        metrics,
        source_key="diff_warm_runtime_by_positive_quadratic_policy_cluster",
        candidate_key="diff_positive_quadratic_runtime_candidate_clusters",
    )
    add_runtime_cluster_candidate_metrics(
        metrics,
        source_key="diff_warm_runtime_by_variable_power_policy_cluster",
        candidate_key="diff_variable_power_runtime_candidate_clusters",
    )
    return metrics


def add_policy_cluster_consolidation_metrics(
    metrics: dict[str, Any],
    *,
    source_key: str,
    consolidated_key: str,
    candidate_key: str,
    shared_policy_clusters: frozenset[str],
) -> None:
    policy_cluster_counts = metrics.get(source_key)
    if not isinstance(policy_cluster_counts, dict):
        return

    consolidated_policy_clusters = {
        cluster: count
        for cluster, count in sorted(policy_cluster_counts.items())
        if isinstance(cluster, str)
        and isinstance(count, int)
        and cluster in shared_policy_clusters
    }
    consolidation_candidates = {
        cluster: count
        for cluster, count in sorted(policy_cluster_counts.items())
        if isinstance(cluster, str)
        and isinstance(count, int)
        and count >= CALCULUS_POLICY_CLUSTER_CONSOLIDATION_THRESHOLD
        and cluster not in shared_policy_clusters
    }
    if consolidated_policy_clusters:
        metrics[consolidated_key] = consolidated_policy_clusters
    if consolidation_candidates:
        metrics[candidate_key] = consolidation_candidates


def add_shared_policy_cluster_metrics(
    metrics: dict[str, Any],
    *,
    source_key: str,
    shared_key: str,
    shared_policy_clusters: frozenset[str],
) -> None:
    policy_cluster_counts = metrics.get(source_key)
    if not isinstance(policy_cluster_counts, dict):
        return

    shared_policy_cluster_counts = {
        cluster: count
        for cluster, count in sorted(policy_cluster_counts.items())
        if isinstance(cluster, str)
        and isinstance(count, int)
        and cluster in shared_policy_clusters
    }
    if shared_policy_cluster_counts:
        metrics[shared_key] = shared_policy_cluster_counts


def runtime_cluster_candidate_rows(
    raw_rows: Any,
    *,
    min_case_count: int = CALCULUS_CLUSTER_RUNTIME_CANDIDATE_MIN_CASES,
    min_avg_ms: float = CALCULUS_CLUSTER_RUNTIME_CANDIDATE_MIN_AVG_MS,
    min_ratio: float = CALCULUS_CLUSTER_RUNTIME_CANDIDATE_RATIO,
) -> list[dict[str, Any]]:
    rows = sanitize_runtime_group_rows(raw_rows)
    eligible = [
        row
        for row in rows
        if isinstance(row.get("group"), str)
        and isinstance(row.get("case_count"), int)
        and row["case_count"] >= min_case_count
        and isinstance(row.get("avg_case_ms"), (int, float))
    ]
    baseline_avg_ms = (
        min(float(row["avg_case_ms"]) for row in eligible)
        if len(eligible) >= 2
        else None
    )
    if baseline_avg_ms is not None and baseline_avg_ms <= 0:
        return []
    candidates: list[dict[str, Any]] = []
    for row in eligible:
        avg_case_ms = float(row["avg_case_ms"])
        max_elapsed_seconds = row.get("max_elapsed_seconds")
        max_case_ms = (
            float(max_elapsed_seconds) * 1000.0
            if isinstance(max_elapsed_seconds, (int, float))
            else 0.0
        )
        avg_ratio = (
            avg_case_ms / baseline_avg_ms
            if baseline_avg_ms is not None
            else 1.0
        )
        avg_is_candidate = (
            baseline_avg_ms is not None
            and avg_case_ms >= min_avg_ms
            and avg_ratio >= min_ratio
        )
        max_is_candidate = max_case_ms >= min_avg_ms
        if not avg_is_candidate and not max_is_candidate:
            continue
        candidate: dict[str, Any] = {
            "group": row["group"],
            "case_count": row["case_count"],
            "avg_case_ms": round(avg_case_ms, 3),
            "avg_ratio": round(avg_ratio, 3),
        }
        if baseline_avg_ms is not None:
            candidate["baseline_avg_case_ms"] = round(baseline_avg_ms, 3)
        for key in ("total_elapsed_seconds", "max_elapsed_seconds"):
            value = row.get(key)
            if isinstance(value, (int, float)):
                candidate[key] = round(float(value), 3)
        slowest_case = row.get("slowest_case")
        if isinstance(slowest_case, str):
            candidate["slowest_case"] = slowest_case
        candidates.append(candidate)
    candidates.sort(
        key=lambda row: (
            -float(row.get("avg_case_ms", 0.0)),
            str(row.get("group", "")),
        )
    )
    return candidates


def add_runtime_cluster_candidate_metrics(
    metrics: dict[str, Any],
    *,
    source_key: str,
    candidate_key: str,
) -> None:
    candidates = runtime_cluster_candidate_rows(metrics.get(source_key))
    if candidates:
        metrics[candidate_key] = candidates


def parse_calculus_integrate_command_matrix(output: str) -> dict[str, Any]:
    try:
        raw = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid integrate command matrix json: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("integrate command matrix json output is not an object")

    total = raw.get("total")
    status = raw.get("status")
    status_counts = raw.get("status_counts")
    issue_kind_counts = raw.get("issue_kind_counts", {})
    if not isinstance(total, int):
        raise ValueError("missing integrate command matrix total")
    if not isinstance(status, str):
        raise ValueError("missing integrate command matrix status")
    if not isinstance(status_counts, dict):
        raise ValueError("missing integrate command matrix status_counts")
    if not isinstance(issue_kind_counts, dict):
        issue_kind_counts = {}

    passed = status_counts.get("pass", 0)
    raw_failed = status_counts.get("fail", 0)
    slow = status_counts.get("slow", 0)
    timeouts = status_counts.get("timeout", 0)
    if not all(
        isinstance(value, int)
        for value in (passed, raw_failed, slow, timeouts)
    ):
        raise ValueError("invalid integrate command matrix status_counts")

    def int_metric(key: str) -> int | None:
        value = raw.get(key)
        return value if isinstance(value, int) else None

    def count_map(key: str) -> dict[str, int]:
        value = raw.get(key)
        if not isinstance(value, dict):
            return {}
        return {
            name: count
            for name, count in value.items()
            if isinstance(name, str) and isinstance(count, int)
        }

    problem_cases = sanitize_limit_command_problem_cases(raw.get("problem_cases"))
    residual_case_names = sanitize_string_list(raw.get("residual_case_names"))
    residual_cases_by_cause = sanitize_string_list_map(
        raw.get("residual_cases_by_cause")
    )
    direct_gap_cases_by_maturity_block = sanitize_string_list_map(
        raw.get("direct_diff_integrate_gap_cases_by_calculus_maturity_block")
    )
    direct_gap_cases_by_block_gate = sanitize_string_list_map(
        raw.get("direct_diff_integrate_gap_cases_by_calculus_block_gate")
    )
    direct_gap_cases_by_trig_hyperbolic_cluster = sanitize_string_list_map(
        raw.get("direct_diff_integrate_gap_cases_by_trig_hyperbolic_policy_cluster")
    )
    direct_gap_cases_by_base_integration_cluster = sanitize_string_list_map(
        raw.get("direct_diff_integrate_gap_cases_by_base_integration_policy_cluster")
    )
    direct_gap_cases_by_radical_inverse_cluster = sanitize_string_list_map(
        raw.get("direct_diff_integrate_gap_cases_by_radical_inverse_policy_cluster")
    )
    problem_case_count = raw.get("problem_case_count")
    if not isinstance(problem_case_count, int):
        problem_case_count = len(problem_cases)

    metrics = {
        "matrix_status": status,
        "total_cases": total,
        "passed": passed,
        "failed": raw_failed + slow + timeouts,
        "raw_failed": raw_failed,
        "slow": slow,
        "timeouts": timeouts,
        "problem_case_count": problem_case_count,
        "problem_cases": problem_cases,
        "integrate_residual_case_names": residual_case_names,
        "integrate_residual_cases_by_cause": residual_cases_by_cause,
        "integrate_direct_diff_integrate_gap_cases_by_calculus_maturity_block": (
            direct_gap_cases_by_maturity_block
        ),
        "integrate_direct_diff_integrate_gap_cases_by_calculus_block_gate": (
            direct_gap_cases_by_block_gate
        ),
        "integrate_direct_diff_integrate_gap_cases_by_trig_hyperbolic_policy_cluster": (
            direct_gap_cases_by_trig_hyperbolic_cluster
        ),
        "integrate_direct_diff_integrate_gap_cases_by_base_integration_policy_cluster": (
            direct_gap_cases_by_base_integration_cluster
        ),
        "integrate_direct_diff_integrate_gap_cases_by_radical_inverse_policy_cluster": (
            direct_gap_cases_by_radical_inverse_cluster
        ),
        "issue_kind_counts": {
            key: value
            for key, value in issue_kind_counts.items()
            if isinstance(key, str) and isinstance(value, int)
        },
    }
    for raw_key, metric_key in (
        ("supported_case_count", "integrate_supported_case_count"),
        ("residual_case_count", "integrate_residual_case_count"),
        ("warning_expected_case_count", "integrate_warning_expected_case_count"),
        ("required_display_case_count", "integrate_required_display_case_count"),
        ("step_checked_case_count", "integrate_step_checked_case_count"),
        (
            "supported_step_unchecked_case_count",
            "integrate_supported_step_unchecked_case_count",
        ),
        (
            "antiderivative_verification_case_count",
            "integrate_antiderivative_verification_case_count",
        ),
        ("verified_supported_case_count", "integrate_verified_supported_case_count"),
        (
            "direct_diff_integrate_case_count",
            "integrate_direct_diff_integrate_case_count",
        ),
        (
            "direct_diff_integrate_exact_case_count",
            "integrate_direct_diff_integrate_exact_case_count",
        ),
        (
            "direct_diff_integrate_equivalence_case_count",
            "integrate_direct_diff_integrate_equivalence_case_count",
        ),
        (
            "direct_diff_integrate_gap_case_count",
            "integrate_direct_diff_integrate_gap_case_count",
        ),
        ("expected_step_substring_count", "integrate_expected_step_substring_count"),
        (
            "distinct_required_display_count",
            "integrate_distinct_required_display_count",
        ),
        ("family_count", "integrate_family_count"),
    ):
        value = int_metric(raw_key)
        if value is not None:
            metrics[metric_key] = value
    for raw_key, metric_key in (
        ("argument_regime_counts", "integrate_argument_regime_counts"),
        ("domain_regime_counts", "integrate_domain_regime_counts"),
        ("outcome_counts", "integrate_outcome_counts"),
        ("residual_cause_counts", "integrate_residual_cause_counts"),
        ("residual_family_counts", "integrate_residual_family_counts"),
        (
            "residual_cause_family_counts",
            "integrate_residual_cause_family_counts",
        ),
        ("required_display_counts", "integrate_required_display_counts"),
        ("verification_regime_counts", "integrate_verification_regime_counts"),
        (
            "calculus_maturity_block_counts",
            "integrate_calculus_maturity_block_counts",
        ),
        ("calculus_block_gate_counts", "integrate_calculus_block_gate_counts"),
        ("trace_regime_counts", "integrate_trace_regime_counts"),
        ("presentation_regime_counts", "integrate_presentation_regime_counts"),
        (
            "trig_hyperbolic_policy_cluster_counts",
            "integrate_trig_hyperbolic_policy_cluster_counts",
        ),
        (
            "base_integration_policy_cluster_counts",
            "integrate_base_integration_policy_cluster_counts",
        ),
        (
            "radical_inverse_policy_cluster_counts",
            "integrate_radical_inverse_policy_cluster_counts",
        ),
        (
            "direct_diff_integrate_calculus_maturity_block_counts",
            "integrate_direct_diff_integrate_calculus_maturity_block_counts",
        ),
        (
            "direct_diff_integrate_calculus_block_gate_counts",
            "integrate_direct_diff_integrate_calculus_block_gate_counts",
        ),
        (
            "direct_diff_integrate_trig_hyperbolic_policy_cluster_counts",
            "integrate_direct_diff_integrate_trig_hyperbolic_policy_cluster_counts",
        ),
        (
            "direct_diff_integrate_base_integration_policy_cluster_counts",
            "integrate_direct_diff_integrate_base_integration_policy_cluster_counts",
        ),
        (
            "direct_diff_integrate_radical_inverse_policy_cluster_counts",
            "integrate_direct_diff_integrate_radical_inverse_policy_cluster_counts",
        ),
        (
            "direct_diff_integrate_gap_calculus_maturity_block_counts",
            "integrate_direct_diff_integrate_gap_calculus_maturity_block_counts",
        ),
        (
            "direct_diff_integrate_gap_calculus_block_gate_counts",
            "integrate_direct_diff_integrate_gap_calculus_block_gate_counts",
        ),
        (
            "direct_diff_integrate_gap_trig_hyperbolic_policy_cluster_counts",
            "integrate_direct_diff_integrate_gap_trig_hyperbolic_policy_cluster_counts",
        ),
        (
            "direct_diff_integrate_gap_base_integration_policy_cluster_counts",
            "integrate_direct_diff_integrate_gap_base_integration_policy_cluster_counts",
        ),
        (
            "direct_diff_integrate_gap_radical_inverse_policy_cluster_counts",
            "integrate_direct_diff_integrate_gap_radical_inverse_policy_cluster_counts",
        ),
        (
            "direct_diff_integrate_equivalence_calculus_maturity_block_counts",
            "integrate_direct_diff_integrate_equivalence_calculus_maturity_block_counts",
        ),
        (
            "direct_diff_integrate_equivalence_trig_hyperbolic_policy_cluster_counts",
            "integrate_direct_diff_integrate_equivalence_trig_hyperbolic_policy_cluster_counts",
        ),
        (
            "direct_diff_integrate_equivalence_base_integration_policy_cluster_counts",
            "integrate_direct_diff_integrate_equivalence_base_integration_policy_cluster_counts",
        ),
    ):
        counts = count_map(raw_key)
        if counts:
            metrics[metric_key] = counts

    add_runtime_observability_metrics(
        metrics,
        raw,
        prefix="integrate",
        group_keys=(
            "family",
            "calculus_maturity_block",
            "calculus_block_gate",
            "trace_regime",
        ),
    )
    for raw_key, metric_key in (
        (
            "cli_parse_runtime_distribution",
            "integrate_cli_parse_runtime_distribution",
        ),
        (
            "cli_simplify_runtime_distribution",
            "integrate_cli_simplify_runtime_distribution",
        ),
        (
            "cli_total_runtime_distribution",
            "integrate_cli_total_runtime_distribution",
        ),
        (
            "cli_public_overhead_runtime_distribution",
            "integrate_cli_public_overhead_runtime_distribution",
        ),
    ):
        distribution = sanitize_runtime_distribution(raw.get(raw_key))
        if distribution:
            metrics[metric_key] = distribution
    for raw_key, metric_key, elapsed_key in (
        (
            "slowest_cli_parse_evaluations",
            "integrate_slowest_cli_parse_evaluations",
            "cli_parse_elapsed_seconds",
        ),
        (
            "slowest_cli_simplify_evaluations",
            "integrate_slowest_cli_simplify_evaluations",
            "cli_simplify_elapsed_seconds",
        ),
        (
            "slowest_cli_total_evaluations",
            "integrate_slowest_cli_total_evaluations",
            "cli_total_elapsed_seconds",
        ),
        (
            "slowest_cli_public_overhead_evaluations",
            "integrate_slowest_cli_public_overhead_evaluations",
            "cli_public_overhead_seconds",
        ),
        (
            "slowest_integrate_evaluations",
            "integrate_slowest_integrate_evaluations",
            "integrate_elapsed_seconds",
        ),
        (
            "slowest_antiderivative_verifications",
            "integrate_slowest_antiderivative_verifications",
            "antiderivative_verification_elapsed_seconds",
        ),
        (
            "slowest_antiderivative_residual_simplifications",
            "integrate_slowest_antiderivative_residual_simplifications",
            "antiderivative_residual_simplify_elapsed_seconds",
        ),
        (
            "slowest_direct_diff_integrate_checks",
            "integrate_slowest_direct_diff_integrate_checks",
            "direct_diff_integrate_elapsed_seconds",
        ),
    ):
        rows = sanitize_phase_runtime_case_rows(
            raw.get(raw_key),
            elapsed_key=elapsed_key,
        )
        if rows:
            metrics[metric_key] = rows
    for raw_key, metric_key, payload_key in (
        (
            "largest_stdout_payload_cases",
            "integrate_largest_stdout_payload_cases",
            "stdout_bytes",
        ),
        (
            "largest_step_trace_cases",
            "integrate_largest_step_trace_cases",
            "step_text_char_count",
        ),
    ):
        rows = sanitize_payload_case_rows(
            raw.get(raw_key),
            metric_key=payload_key,
        )
        if rows:
            metrics[metric_key] = rows
    verification_mode_rows = sanitize_verification_mode_runtime_rows(
        raw.get("runtime_by_antiderivative_verification_mode")
    )
    if verification_mode_rows:
        metrics["integrate_runtime_by_antiderivative_verification_mode"] = (
            verification_mode_rows
        )
    residual_cause_runtime_rows = sanitize_residual_cause_runtime_rows(
        raw.get("runtime_by_residual_cause")
    )
    if residual_cause_runtime_rows:
        metrics["integrate_runtime_by_residual_cause"] = (
            residual_cause_runtime_rows
        )
    residual_cause_family_runtime_rows = sanitize_residual_cause_family_runtime_rows(
        raw.get("runtime_by_residual_cause_family")
    )
    if residual_cause_family_runtime_rows:
        metrics["integrate_runtime_by_residual_cause_family"] = (
            residual_cause_family_runtime_rows
        )
    residual_public_phase_rows = sanitize_residual_public_phase_case_rows(
        raw.get("residual_public_phase_slowest_cases")
    )
    if residual_public_phase_rows:
        metrics["integrate_residual_public_phase_slowest_cases"] = (
            residual_public_phase_rows
        )
    residual_public_phase_group_rows = sanitize_residual_public_phase_group_rows(
        raw.get("residual_public_phase_by_cause")
    )
    if residual_public_phase_group_rows:
        metrics["integrate_residual_public_phase_by_cause"] = (
            residual_public_phase_group_rows
        )
    residual_public_phase_cause_family_rows = (
        sanitize_residual_public_phase_cause_family_group_rows(
            raw.get("residual_public_phase_by_cause_family")
        )
    )
    if residual_public_phase_cause_family_rows:
        metrics["integrate_residual_public_phase_by_cause_family"] = (
            residual_public_phase_cause_family_rows
        )
    residual_shape_orientation_rows = sanitize_residual_shape_orientation_probe_rows(
        raw.get("residual_shape_orientation_probes")
    )
    if residual_shape_orientation_rows:
        metrics["integrate_residual_shape_orientation_probes"] = (
            residual_shape_orientation_rows
        )
        residual_shape_orientation_summary = (
            residual_shape_orientation_probe_summary(residual_shape_orientation_rows)
        )
        if residual_shape_orientation_summary:
            metrics["integrate_residual_shape_orientation_summary"] = (
                residual_shape_orientation_summary
            )

    add_policy_cluster_consolidation_metrics(
        metrics,
        source_key="integrate_trig_hyperbolic_policy_cluster_counts",
        consolidated_key=(
            "integrate_trig_hyperbolic_consolidated_policy_cluster_counts"
        ),
        candidate_key="integrate_trig_hyperbolic_consolidation_candidate_counts",
        shared_policy_clusters=CALCULUS_POLICY_CLUSTERS_WITH_SHARED_POLICY,
    )
    add_policy_cluster_consolidation_metrics(
        metrics,
        source_key="integrate_radical_inverse_policy_cluster_counts",
        consolidated_key="integrate_radical_inverse_consolidated_policy_cluster_counts",
        candidate_key="integrate_radical_inverse_consolidation_candidate_counts",
        shared_policy_clusters=(
            CALCULUS_RADICAL_INVERSE_POLICY_CLUSTERS_WITH_SHARED_POLICY
        ),
    )
    add_policy_cluster_consolidation_metrics(
        metrics,
        source_key="integrate_base_integration_policy_cluster_counts",
        consolidated_key="integrate_base_integration_consolidated_policy_cluster_counts",
        candidate_key="integrate_base_integration_consolidation_candidate_counts",
        shared_policy_clusters=(
            CALCULUS_BASE_INTEGRATION_POLICY_CLUSTERS_WITH_SHARED_POLICY
        ),
    )
    add_shared_policy_cluster_metrics(
        metrics,
        source_key=(
            "integrate_direct_diff_integrate_trig_hyperbolic_policy_cluster_counts"
        ),
        shared_key=(
            "integrate_direct_diff_integrate_trig_hyperbolic_shared_policy_cluster_counts"
        ),
        shared_policy_clusters=CALCULUS_POLICY_CLUSTERS_WITH_SHARED_POLICY,
    )
    add_shared_policy_cluster_metrics(
        metrics,
        source_key=(
            "integrate_direct_diff_integrate_radical_inverse_policy_cluster_counts"
        ),
        shared_key=(
            "integrate_direct_diff_integrate_radical_inverse_shared_policy_cluster_counts"
        ),
        shared_policy_clusters=(
            CALCULUS_RADICAL_INVERSE_POLICY_CLUSTERS_WITH_SHARED_POLICY
        ),
    )
    add_shared_policy_cluster_metrics(
        metrics,
        source_key=(
            "integrate_direct_diff_integrate_base_integration_policy_cluster_counts"
        ),
        shared_key=(
            "integrate_direct_diff_integrate_base_integration_shared_policy_cluster_counts"
        ),
        shared_policy_clusters=(
            CALCULUS_BASE_INTEGRATION_POLICY_CLUSTERS_WITH_SHARED_POLICY
        ),
    )
    return metrics


def matrix_wrapper_gap_fragments(
    metrics: dict[str, Any], limit: int = 5
) -> list[str]:
    examples = metrics.get("matrix_wrapper_gap_examples")
    if not isinstance(examples, list):
        return []
    fragments = []
    for example in examples[:limit]:
        if not isinstance(example, dict):
            continue
        base = example.get("base")
        missing_count = example.get("missing_count")
        if not isinstance(base, str) or not isinstance(missing_count, int):
            continue
        fragments.append(f"{base} missing={missing_count}")
    return fragments


def sparse_expected_condition_fragments(
    metrics: dict[str, Any], limit: int = 5
) -> list[str]:
    counts = metrics.get("expected_required_condition_counts")
    if not isinstance(counts, dict):
        return []
    rows = [
        (condition, count)
        for condition, count in counts.items()
        if isinstance(condition, str) and isinstance(count, int)
    ]
    rows.sort(key=lambda row: (row[1], row[0]))
    return [f"{condition}={count}" for condition, count in rows[:limit]]


def domain_expected_condition_fragments(
    metrics: dict[str, Any], limit: int = 5
) -> list[str]:
    counts = metrics.get("expected_required_condition_counts")
    if not isinstance(counts, dict):
        return []
    inequality_symbols = ("<", ">", "\u2264", "\u2265")
    rows = [
        (condition, count)
        for condition, count in counts.items()
        if isinstance(condition, str)
        and isinstance(count, int)
        and any(symbol in condition for symbol in inequality_symbols)
    ]
    rows.sort(key=lambda row: (row[1], row[0]))
    return [f"{condition}={count}" for condition, count in rows[:limit]]


def calculus_runtime_case_fragments(raw_rows: Any, limit: int = 5) -> list[str]:
    if not isinstance(raw_rows, list):
        return []
    fragments: list[str] = []
    for row in raw_rows[:limit]:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        elapsed = row.get("wall_elapsed_seconds")
        if not isinstance(name, str) or not isinstance(elapsed, (int, float)):
            continue
        fragment = f"{name}={elapsed:.3f}s"
        family = row.get("family")
        if isinstance(family, str):
            fragment += f" family={family}"
        fragments.append(fragment)
    return fragments


def calculus_runtime_group_fragments(raw_rows: Any, limit: int = 5) -> list[str]:
    if not isinstance(raw_rows, list):
        return []
    fragments: list[str] = []
    for row in raw_rows[:limit]:
        if not isinstance(row, dict):
            continue
        group = row.get("group")
        total_elapsed = row.get("total_elapsed_seconds")
        avg_case_ms = row.get("avg_case_ms")
        case_count = row.get("case_count")
        if (
            not isinstance(group, str)
            or not isinstance(total_elapsed, (int, float))
            or not isinstance(avg_case_ms, (int, float))
            or not isinstance(case_count, int)
        ):
            continue
        fragment = (
            f"{group} total={total_elapsed:.3f}s "
            f"avg={avg_case_ms:.3f}ms cases={case_count}"
        )
        slowest_case = row.get("slowest_case")
        if isinstance(slowest_case, str):
            fragment += f" slowest={slowest_case}"
        fragments.append(fragment)
    return fragments


def runtime_cluster_candidate_fragments(raw_rows: Any, limit: int = 5) -> list[str]:
    if not isinstance(raw_rows, list):
        return []
    fragments: list[str] = []
    for row in raw_rows[:limit]:
        if not isinstance(row, dict):
            continue
        group = row.get("group")
        avg_case_ms = row.get("avg_case_ms")
        case_count = row.get("case_count")
        avg_ratio = row.get("avg_ratio")
        if (
            not isinstance(group, str)
            or not isinstance(avg_case_ms, (int, float))
            or not isinstance(case_count, int)
            or not isinstance(avg_ratio, (int, float))
        ):
            continue
        fragment = (
            f"{group} avg={avg_case_ms:.3f}ms "
            f"ratio={avg_ratio:.3f}x cases={case_count}"
        )
        slowest_case = row.get("slowest_case")
        if isinstance(slowest_case, str):
            fragment += f" slowest={slowest_case}"
        fragments.append(fragment)
    return fragments


def calculus_runtime_distribution_fragment(raw_row: Any) -> str | None:
    if not isinstance(raw_row, dict):
        return None
    timed_case_count = raw_row.get("timed_case_count")
    avg_case_ms = raw_row.get("avg_case_ms")
    p95_case_ms = raw_row.get("p95_case_ms")
    max_case_ms = raw_row.get("max_case_ms")
    total_elapsed = raw_row.get("total_elapsed_seconds")
    if (
        not isinstance(timed_case_count, int)
        or not isinstance(avg_case_ms, (int, float))
        or not isinstance(p95_case_ms, (int, float))
        or not isinstance(max_case_ms, (int, float))
    ):
        return None
    fragment = (
        f"timed_cases={timed_case_count} "
        f"avg={avg_case_ms:.3f}ms "
        f"p95={p95_case_ms:.3f}ms "
        f"max={max_case_ms:.3f}ms"
    )
    if isinstance(total_elapsed, (int, float)):
        fragment += f" total={total_elapsed:.3f}s"
    return fragment


def calculus_runtime_concentration_fragment(raw_row: Any) -> str | None:
    if not isinstance(raw_row, dict):
        return None
    slowest_case = raw_row.get("slowest_case")
    slowest_case_ms = raw_row.get("slowest_case_ms")
    slowest_share = raw_row.get("slowest_case_share_percent")
    top_3_share = raw_row.get("top_3_share_percent")
    total_elapsed = raw_row.get("total_elapsed_seconds")
    if (
        not isinstance(slowest_case, str)
        or not isinstance(slowest_case_ms, (int, float))
        or not isinstance(slowest_share, (int, float))
        or not isinstance(top_3_share, (int, float))
    ):
        return None
    fragment = (
        f"slowest={slowest_case} "
        f"slowest_ms={slowest_case_ms:.3f} "
        f"slowest_share={slowest_share:.1f}% "
        f"top3_share={top_3_share:.1f}%"
    )
    if isinstance(total_elapsed, (int, float)):
        fragment += f" total={total_elapsed:.3f}s"
    family = raw_row.get("slowest_family")
    if isinstance(family, str):
        fragment += f" family={family}"
    return fragment


def calculus_runtime_pressure_fragment(raw_row: Any) -> str | None:
    if not isinstance(raw_row, dict):
        return None
    status = raw_row.get("status")
    primary_signal = raw_row.get("primary_signal")
    reason = raw_row.get("reason")
    if (
        not isinstance(status, str)
        or not isinstance(primary_signal, str)
        or not isinstance(reason, str)
    ):
        return None
    fragment = f"status={status} primary={primary_signal} reason={reason}"
    sample_basis = raw_row.get("sample_basis")
    if isinstance(sample_basis, str):
        fragment += f" basis={sample_basis}"
    for key, label, precision in (
        ("timed_case_count", "timed_cases", None),
        ("p95_case_ms", "p95", 3),
        ("max_case_ms", "max", 3),
        ("top_3_share_percent", "top3_share", 1),
    ):
        value = raw_row.get(key)
        if not isinstance(value, (int, float)):
            continue
        if precision is None:
            fragment += f" {label}={int(value)}"
        elif key == "top_3_share_percent":
            fragment += f" {label}={value:.{precision}f}%"
        else:
            fragment += f" {label}={value:.{precision}f}ms"
    slowest_case = raw_row.get("slowest_case")
    if isinstance(slowest_case, str):
        fragment += f" slowest={slowest_case}"
    family = raw_row.get("slowest_family")
    if isinstance(family, str):
        fragment += f" family={family}"
    return fragment


def backend_verification_pressure_fragment(raw_row: Any) -> str | None:
    if not isinstance(raw_row, dict):
        return None
    status = raw_row.get("status")
    primary_signal = raw_row.get("primary_signal")
    reason = raw_row.get("reason")
    if (
        not isinstance(status, str)
        or not isinstance(primary_signal, str)
        or not isinstance(reason, str)
    ):
        return None
    fragment = f"status={status} primary={primary_signal} reason={reason}"
    for key, label, precision in (
        ("attempts", "attempts", None),
        ("verification_checks_used_total", "checks", None),
        ("checks_per_attempt", "checks_per_attempt", 3),
        ("max_verification_normalization_passes", "max_passes", None),
        ("verification_elapsed_ms", "elapsed_ms", 3),
    ):
        value = raw_row.get(key)
        if not isinstance(value, (int, float)):
            continue
        if precision is None:
            fragment += f" {label}={int(value)}"
        else:
            fragment += f" {label}={value:.{precision}f}"
    return fragment


def calculus_runtime_measurement_fragment(raw_row: Any) -> str | None:
    if not isinstance(raw_row, dict):
        return None
    mode = raw_row.get("mode")
    status = raw_row.get("status")
    guidance = raw_row.get("guidance")
    if not isinstance(mode, str) or not isinstance(status, str):
        return None
    fragment = f"mode={mode} status={status}"
    pressure_status = raw_row.get("pressure_status")
    if isinstance(pressure_status, str):
        fragment += f" pressure={pressure_status}"
    for key, label in (
        ("p95_case_ms", "p95"),
        ("max_case_ms", "max"),
    ):
        value = raw_row.get(key)
        if isinstance(value, (int, float)):
            fragment += f" {label}={value:.3f}ms"
    if isinstance(guidance, str):
        fragment += f" guidance={guidance}"
    return fragment


def calculus_runtime_guardrail_rows(scorecard: dict[str, Any]) -> list[dict[str, Any]]:
    suites = scorecard.get("suites")
    if not isinstance(suites, dict):
        return []

    rows: list[dict[str, Any]] = []
    command_matrix_suites = (
        ("diff_command_matrix", "calculus_diff_command_matrix_smoke", "diff"),
        ("limit_command_matrix", "calculus_limit_command_matrix_smoke", "limit"),
        (
            "integrate_command_matrix",
            "calculus_integrate_command_matrix_smoke",
            "integrate",
        ),
    )
    for label, suite_name, prefix in command_matrix_suites:
        suite = suites.get(suite_name)
        if not isinstance(suite, dict):
            continue
        metrics = suite.get("metrics")
        if not isinstance(metrics, dict):
            continue
        distribution = metrics.get(f"{prefix}_warm_runtime_distribution")
        sample_basis = "warm"
        if not isinstance(distribution, dict):
            distribution = metrics.get(f"{prefix}_runtime_distribution")
            sample_basis = "cold_inclusive"
        if not isinstance(distribution, dict):
            continue

        row: dict[str, Any] = {
            "label": label,
            "status": suite.get("status"),
            "sample_basis": sample_basis,
        }
        total_cases = metrics.get("total_cases")
        if isinstance(total_cases, int):
            row["total_cases"] = total_cases
        for key in (
            "avg_case_ms",
            "p95_case_ms",
            "max_case_ms",
            "total_elapsed_seconds",
        ):
            value = distribution.get(key)
            if isinstance(value, (int, float)):
                row[key] = round(float(value), 3)
        pressure = metrics.get(f"{prefix}_runtime_pressure")
        if isinstance(pressure, dict):
            pressure_status = pressure.get("status")
            primary_signal = pressure.get("primary_signal")
            if isinstance(pressure_status, str):
                row["pressure_status"] = pressure_status
            if isinstance(primary_signal, str):
                row["pressure_signal"] = primary_signal
        measurement = metrics.get(f"{prefix}_runtime_measurement")
        if isinstance(measurement, dict):
            measurement_status = measurement.get("status")
            if isinstance(measurement_status, str):
                row["measurement_status"] = measurement_status
        rows.append(row)

    residual_suite = suites.get("calculus_residual_matrix_smoke")
    if isinstance(residual_suite, dict):
        metrics = residual_suite.get("metrics")
        elapsed = residual_suite.get("elapsed_seconds")
        if isinstance(metrics, dict) and isinstance(elapsed, (int, float)):
            total_cases = metrics.get("total_cases")
            if isinstance(total_cases, int) and total_cases > 0:
                row: dict[str, Any] = {
                    "label": "residual_matrix",
                    "status": residual_suite.get("status"),
                    "sample_basis": "suite_elapsed",
                    "total_cases": total_cases,
                    "avg_case_ms": round(float(elapsed) * 1000.0 / total_cases, 3),
                    "total_elapsed_seconds": round(float(elapsed), 3),
                    "pressure_status": "suite_elapsed_only",
                    "measurement_status": "suite_elapsed_per_case",
                }
                for key in ("failed", "timeouts"):
                    value = metrics.get(key)
                    if isinstance(value, int):
                        row[key] = value
                rows.append(row)
    return rows


def calculus_runtime_guardrail_fragments(
    scorecard: dict[str, Any],
    *,
    limit: int = 6,
) -> list[str]:
    fragments: list[str] = []
    rows = calculus_runtime_guardrail_rows(scorecard)
    for row in rows[:limit]:
        label = row.get("label")
        avg_case_ms = row.get("avg_case_ms")
        total_cases = row.get("total_cases")
        if (
            not isinstance(label, str)
            or not isinstance(avg_case_ms, (int, float))
            or not isinstance(total_cases, int)
        ):
            continue
        fragment = f"{label} cases={total_cases} avg={avg_case_ms:.3f}ms"
        p95_case_ms = row.get("p95_case_ms")
        if isinstance(p95_case_ms, (int, float)):
            fragment += f" p95={p95_case_ms:.3f}ms"
        max_case_ms = row.get("max_case_ms")
        if isinstance(max_case_ms, (int, float)):
            fragment += f" max={max_case_ms:.3f}ms"
        total_elapsed = row.get("total_elapsed_seconds")
        if isinstance(total_elapsed, (int, float)):
            fragment += f" total={total_elapsed:.3f}s"
        status = row.get("status")
        if isinstance(status, str):
            fragment += f" status={status}"
        failed = row.get("failed")
        if isinstance(failed, int):
            fragment += f" failed={failed}"
        timeouts = row.get("timeouts")
        if isinstance(timeouts, int):
            fragment += f" timeouts={timeouts}"
        pressure_status = row.get("pressure_status")
        if isinstance(pressure_status, str):
            fragment += f" pressure={pressure_status}"
        pressure_signal = row.get("pressure_signal")
        if isinstance(pressure_signal, str):
            fragment += f" signal={pressure_signal}"
        measurement_status = row.get("measurement_status")
        if isinstance(measurement_status, str):
            fragment += f" measurement={measurement_status}"
        sample_basis = row.get("sample_basis")
        if isinstance(sample_basis, str):
            fragment += f" basis={sample_basis}"
        fragments.append(fragment)
    remaining = len(rows) - limit
    if remaining > 0:
        fragments.append(f"+{remaining} more")
    return fragments


def phase_runtime_case_fragments(
    raw_rows: Any,
    *,
    elapsed_key: str,
    limit: int = 5,
) -> list[str]:
    if not isinstance(raw_rows, list):
        return []
    fragments: list[str] = []
    for row in raw_rows[:limit]:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        elapsed = row.get(elapsed_key)
        if not isinstance(name, str) or not isinstance(elapsed, (int, float)):
            continue
        fragment = f"{name}={elapsed:.3f}s"
        family = row.get("family")
        if isinstance(family, str):
            fragment += f" family={family}"
        mode = row.get("antiderivative_verification_mode")
        if isinstance(mode, str):
            fragment += f" mode={mode}"
        cluster = row.get("positive_quadratic_policy_cluster")
        if isinstance(cluster, str):
            fragment += f" cluster={cluster}"
        variable_power_cluster = row.get("variable_power_policy_cluster")
        if isinstance(variable_power_cluster, str):
            fragment += f" variable_power_cluster={variable_power_cluster}"
        steps_count = row.get("steps_count")
        if isinstance(steps_count, int):
            fragment += f" steps={steps_count}"
        step_rule_names = row.get("step_rule_names")
        if isinstance(step_rule_names, list):
            names = [
                value
                for value in step_rule_names
                if isinstance(value, str) and value.strip()
            ]
            if names:
                fragment += " rules=" + "|".join(names[:4])
        fragments.append(fragment)
    return fragments


def diff_exact_square_runtime_pair_fragments(
    raw_rows: Any,
    *,
    limit: int = 5,
) -> list[str]:
    rows = sanitize_diff_exact_square_runtime_pair_rows(raw_rows)
    fragments: list[str] = []
    for row in rows[:limit]:
        baseline_ms = row.get("baseline_case_ms")
        exact_square_ms = row.get("exact_square_case_ms")
        if not isinstance(baseline_ms, (int, float)) or not isinstance(
            exact_square_ms,
            (int, float),
        ):
            continue
        fragment = (
            f"{row['name']} exact_square={exact_square_ms:.3f}ms "
            f"baseline={baseline_ms:.3f}ms"
        )
        delta_ms = row.get("delta_ms")
        if isinstance(delta_ms, (int, float)):
            fragment += f" delta={delta_ms:.3f}ms"
        ratio = row.get("ratio")
        if isinstance(ratio, (int, float)):
            fragment += f" ratio={ratio:.3f}x"
        exact_square_case = row.get("exact_square_case")
        baseline_case = row.get("baseline_case")
        if isinstance(exact_square_case, str) and isinstance(baseline_case, str):
            fragment += f" pair={exact_square_case}/{baseline_case}"
        family = row.get("family")
        if isinstance(family, str):
            fragment += f" family={family}"
        fragments.append(fragment)
    remaining = len(rows) - limit
    if remaining > 0:
        fragments.append(f"+{remaining} more")
    return fragments


def payload_case_fragments(
    raw_rows: Any,
    *,
    metric_key: str,
    unit: str,
    limit: int = 5,
) -> list[str]:
    if not isinstance(raw_rows, list):
        return []
    fragments: list[str] = []
    for row in raw_rows[:limit]:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        metric = row.get(metric_key)
        if not isinstance(name, str) or not isinstance(metric, int):
            continue
        fragment = f"{name}={metric}{unit}"
        family = row.get("family")
        if isinstance(family, str):
            fragment += f" family={family}"
        required_count = row.get("required_display_count")
        if isinstance(required_count, int):
            fragment += f" required={required_count}"
        expected_step_count = row.get("expected_step_substring_count")
        if isinstance(expected_step_count, int):
            fragment += f" expected_steps={expected_step_count}"
        fragments.append(fragment)
    return fragments


def verification_mode_runtime_fragments(raw_rows: Any, limit: int = 5) -> list[str]:
    if not isinstance(raw_rows, list):
        return []
    fragments: list[str] = []
    for row in raw_rows[:limit]:
        if not isinstance(row, dict):
            continue
        mode = row.get("mode")
        total_elapsed = row.get("total_elapsed_seconds")
        avg_case_ms = row.get("avg_case_ms")
        case_count = row.get("case_count")
        if (
            not isinstance(mode, str)
            or not isinstance(total_elapsed, (int, float))
            or not isinstance(avg_case_ms, (int, float))
            or not isinstance(case_count, int)
        ):
            continue
        fragment = (
            f"{mode} total={total_elapsed:.3f}s "
            f"avg={avg_case_ms:.3f}ms cases={case_count}"
        )
        slowest_case = row.get("slowest_case")
        if isinstance(slowest_case, str):
            fragment += f" slowest={slowest_case}"
        fragments.append(fragment)
    return fragments


def residual_cause_runtime_fragments(
    raw_rows: Any,
    limit: int = 5,
    *,
    group_key: str = "cause",
) -> list[str]:
    if not isinstance(raw_rows, list):
        return []
    fragments: list[str] = []
    for row in raw_rows[:limit]:
        if not isinstance(row, dict):
            continue
        cause = row.get(group_key)
        total_elapsed = row.get("total_elapsed_seconds")
        avg_case_ms = row.get("avg_case_ms")
        case_count = row.get("case_count")
        if (
            not isinstance(cause, str)
            or not isinstance(total_elapsed, (int, float))
            or not isinstance(avg_case_ms, (int, float))
            or not isinstance(case_count, int)
        ):
            continue
        fragment = (
            f"{cause} total={total_elapsed:.3f}s "
            f"avg={avg_case_ms:.3f}ms cases={case_count}"
        )
        slowest_case = row.get("slowest_case")
        if isinstance(slowest_case, str):
            fragment += f" slowest={slowest_case}"
        fragments.append(fragment)
    return fragments


def residual_public_phase_group_fragments(
    raw_rows: Any,
    limit: int = 5,
    *,
    group_key: str = "cause",
) -> list[str]:
    if not isinstance(raw_rows, list):
        return []
    fragments: list[str] = []
    for row in raw_rows[:limit]:
        if not isinstance(row, dict):
            continue
        cause = row.get(group_key)
        integrate_total = row.get("integrate_total_seconds")
        cli_total = row.get("cli_total_seconds")
        overhead_share = row.get("public_overhead_share_percent")
        case_count = row.get("case_count")
        if (
            not isinstance(cause, str)
            or not isinstance(integrate_total, (int, float))
            or not isinstance(cli_total, (int, float))
            or not isinstance(overhead_share, (int, float))
            or not isinstance(case_count, int)
        ):
            continue
        fragment = (
            f"{cause} total={integrate_total:.3f}s "
            f"cli={cli_total:.3f}s overhead_share={overhead_share:.1f}% "
            f"cases={case_count}"
        )
        slowest_case = row.get("slowest_case")
        if isinstance(slowest_case, str):
            fragment += f" slowest={slowest_case}"
        fragments.append(fragment)
    return fragments


def residual_public_phase_case_fragments(raw_rows: Any, limit: int = 5) -> list[str]:
    if not isinstance(raw_rows, list):
        return []
    fragments: list[str] = []
    for row in raw_rows[:limit]:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        integrate_elapsed = row.get("integrate_elapsed_seconds")
        cli_simplify_ms = row.get("cli_simplify_ms")
        overhead_share = row.get("public_overhead_share_percent")
        if (
            not isinstance(name, str)
            or not isinstance(integrate_elapsed, (int, float))
            or not isinstance(cli_simplify_ms, (int, float))
            or not isinstance(overhead_share, (int, float))
        ):
            continue
        fragment = (
            f"{name} total={integrate_elapsed:.3f}s "
            f"cli_simplify={cli_simplify_ms:.3f}ms "
            f"overhead_share={overhead_share:.1f}%"
        )
        cause = row.get("residual_cause")
        if isinstance(cause, str):
            fragment += f" cause={cause}"
        fragments.append(fragment)
    return fragments


def residual_shape_orientation_probe_fragments(
    raw_rows: Any,
    limit: int = 5,
) -> list[str]:
    rows = sanitize_residual_shape_orientation_probe_rows(raw_rows)
    fragments: list[str] = []
    for row in rows[:limit]:
        name = row.get("name")
        expression_shape = row.get("expression_shape")
        orientation = row.get("orientation")
        status = row.get("status")
        if not all(
            isinstance(value, str)
            for value in (name, expression_shape, orientation, status)
        ):
            continue
        fragment = (
            f"{name} status={status} shape={expression_shape} "
            f"orientation={orientation}"
        )
        steps_mode = row.get("steps_mode")
        if isinstance(steps_mode, str):
            fragment += f" steps={steps_mode}"
        required_display_count = row.get("required_display_count")
        if isinstance(required_display_count, int):
            fragment += f" required_display={required_display_count}"
        simplify_us = row.get("cli_simplify_us")
        if isinstance(simplify_us, (int, float)):
            fragment += f" cli_simplify={simplify_us / 1000.0:.3f}ms"
        total_us = row.get("cli_total_us")
        if isinstance(total_us, (int, float)):
            fragment += f" cli_total={total_us / 1000.0:.3f}ms"
        fragments.append(fragment)
    remaining = len(rows) - limit
    if remaining > 0:
        fragments.append(f"+{remaining} more")
    return fragments


def residual_shape_orientation_summary_fragment(raw_summary: Any) -> str | None:
    if not isinstance(raw_summary, dict):
        return None
    probe_count = raw_summary.get("probe_count")
    counted_probe_count = raw_summary.get("counted_probe_count")
    max_required_display_count = raw_summary.get("max_required_display_count")
    avg_required_display_count = raw_summary.get("avg_required_display_count")
    if (
        not isinstance(probe_count, int)
        or not isinstance(counted_probe_count, int)
        or not isinstance(max_required_display_count, int)
        or not isinstance(avg_required_display_count, (int, float))
    ):
        return None
    fragment = (
        f"probes={probe_count} counted={counted_probe_count} "
        f"max_required_display={max_required_display_count} "
        f"avg_required_display={float(avg_required_display_count):.3f}"
    )
    status_counts = raw_summary.get("status_counts")
    if isinstance(status_counts, dict):
        status_fragments = sorted(
            f"{status}={count}"
            for status, count in status_counts.items()
            if isinstance(status, str) and isinstance(count, int)
        )
        if status_fragments:
            fragment += " status_counts=" + ",".join(status_fragments)
    max_name = raw_summary.get("max_name")
    if isinstance(max_name, str):
        fragment += f" max_case={max_name}"
    max_expression_shape = raw_summary.get("max_expression_shape")
    if isinstance(max_expression_shape, str):
        fragment += f" shape={max_expression_shape}"
    max_orientation = raw_summary.get("max_orientation")
    if isinstance(max_orientation, str):
        fragment += f" orientation={max_orientation}"
    max_steps_mode = raw_summary.get("max_steps_mode")
    if isinstance(max_steps_mode, str):
        fragment += f" steps={max_steps_mode}"
    first_problem_name = raw_summary.get("first_problem_name")
    first_problem_status = raw_summary.get("first_problem_status")
    if isinstance(first_problem_name, str) and isinstance(first_problem_status, str):
        fragment += f" first_problem={first_problem_name}:{first_problem_status}"
        first_problem_orientation = raw_summary.get("first_problem_orientation")
        if isinstance(first_problem_orientation, str):
            fragment += f" orientation={first_problem_orientation}"
        first_problem_steps_mode = raw_summary.get("first_problem_steps_mode")
        if isinstance(first_problem_steps_mode, str):
            fragment += f" steps={first_problem_steps_mode}"
    return fragment


def calculus_runtime_lines(
    label: str,
    metrics: dict[str, Any],
    *,
    prefix: str,
    group_keys: tuple[str, ...],
) -> list[str]:
    lines: list[str] = []
    distribution = calculus_runtime_distribution_fragment(
        metrics.get(f"{prefix}_runtime_distribution")
    )
    if distribution:
        lines.append(f"- `{label}` runtime distribution: {distribution}")
    concentration = calculus_runtime_concentration_fragment(
        metrics.get(f"{prefix}_runtime_concentration")
    )
    if concentration:
        lines.append(f"- `{label}` runtime concentration: {concentration}")
    pressure = calculus_runtime_pressure_fragment(
        metrics.get(f"{prefix}_runtime_pressure")
    )
    if pressure:
        lines.append(f"- `{label}` runtime pressure: {pressure}")
    measurement = calculus_runtime_measurement_fragment(
        metrics.get(f"{prefix}_runtime_measurement")
    )
    if measurement:
        lines.append(f"- `{label}` runtime measurement: {measurement}")
    warm_distribution = calculus_runtime_distribution_fragment(
        metrics.get(f"{prefix}_warm_runtime_distribution")
    )
    if warm_distribution:
        lines.append(f"- `{label}` warm runtime distribution: {warm_distribution}")
    warm_concentration = calculus_runtime_concentration_fragment(
        metrics.get(f"{prefix}_warm_runtime_concentration")
    )
    if warm_concentration:
        lines.append(f"- `{label}` warm runtime concentration: {warm_concentration}")
    cold_start = calculus_runtime_case_fragments(
        [metrics.get(f"{prefix}_cold_start_case")]
    )
    if cold_start:
        lines.append(f"- `{label}` cold-start case: " + ", ".join(cold_start))
    slowest_cases = calculus_runtime_case_fragments(
        metrics.get(f"{prefix}_slowest_cases")
    )
    if slowest_cases:
        lines.append(f"- `{label}` slowest cases: " + ", ".join(slowest_cases))
    warm_slowest_cases = calculus_runtime_case_fragments(
        metrics.get(f"{prefix}_warm_slowest_cases")
    )
    if warm_slowest_cases:
        lines.append(
            f"- `{label}` warm slowest cases: " + ", ".join(warm_slowest_cases)
        )
    for group_key in group_keys:
        fragments = calculus_runtime_group_fragments(
            metrics.get(f"{prefix}_runtime_by_{group_key}")
        )
        if fragments:
            axis_label = group_key.replace("_", " ")
            lines.append(
                f"- `{label}` runtime by {axis_label}: "
                + ", ".join(fragments)
            )
        warm_fragments = calculus_runtime_group_fragments(
            metrics.get(f"{prefix}_warm_runtime_by_{group_key}")
        )
        if warm_fragments:
            axis_label = group_key.replace("_", " ")
            lines.append(
                f"- `{label}` warm runtime by {axis_label}: "
                + ", ".join(warm_fragments)
            )
    return lines


def count_map_fragments(value: Any, limit: int | None = 8) -> list[str]:
    if not isinstance(value, dict):
        return []
    rows = [
        (name, count)
        for name, count in value.items()
        if isinstance(name, str) and isinstance(count, int)
    ]
    rows.sort(key=lambda row: (row[0], row[1]))
    selected_rows = rows if limit is None else rows[:limit]
    return [f"{name}={count}" for name, count in selected_rows]


def calculus_matrix_count_map_fragments(value: Any) -> list[str]:
    return count_map_fragments(value, limit=None)


def calculus_residual_case_fragments(value: Any, limit: int = 12) -> list[str]:
    names = sanitize_string_list(value)
    if not names:
        return []
    selected = names[:limit]
    fragments = list(selected)
    remaining = len(names) - len(selected)
    if remaining > 0:
        fragments.append(f"+{remaining} more")
    return fragments


def calculus_residual_cases_by_cause_fragments(
    value: Any,
    *,
    cause_limit: int = 8,
    case_limit: int = 3,
) -> list[str]:
    grouped = sanitize_string_list_map(value)
    fragments: list[str] = []
    for cause, case_names in list(grouped.items())[:cause_limit]:
        selected = case_names[:case_limit]
        cases_fragment = ", ".join(selected)
        remaining = len(case_names) - len(selected)
        if remaining > 0:
            cases_fragment = f"{cases_fragment}, +{remaining} more"
        fragments.append(f"{cause}: {cases_fragment}")
    remaining_causes = len(grouped) - cause_limit
    if remaining_causes > 0:
        fragments.append(f"+{remaining_causes} more causes")
    return fragments


PARSERS = {
    "corpus": parse_corpus,
    "derive": parse_derive,
    "derive_shadow": parse_derive_shadow,
    "derive_didactic": parse_derive_didactic,
    "simplify_didactic": parse_simplify_didactic,
    "unified_benchmark": parse_unified_benchmark,
    "cargo_test_basic": parse_cargo_test_basic,
    "calculus_residual_matrix": parse_calculus_residual_matrix,
    "calculus_diff_command_matrix": parse_calculus_diff_command_matrix,
    "calculus_limit_command_matrix": parse_calculus_limit_command_matrix,
    "calculus_integrate_command_matrix": parse_calculus_integrate_command_matrix,
    "algorithmic_backend_observability": parse_algorithmic_backend_observability,
}


def suite_status(name: str, metrics: dict[str, Any], returncode: int) -> str:
    if returncode != 0:
        return "fail"
    if name in {"embedded_equivalence_context", "simplify_zero_mixed"}:
        return "pass" if metrics["failed"] == 0 else "fail"
    if name == "derive_contract":
        return "warn" if metrics["unsupported"] > 0 else "pass"
    if name == "derive_shadow_pressure":
        return "pass"
    if name == "derive_didactic_audit":
        return "warn" if metrics["flagged_cases"] > 0 else "pass"
    if name == "simplify_didactic_audit":
        return "warn" if metrics["flagged_cases"] > 0 else "pass"
    if name in {"simplify_strict", "simplify_nf_first"}:
        if metrics["failed"] > 0:
            return "fail"
        if metrics["timeouts"] > 0 or metrics["numeric_only"] > 0:
            return "warn"
        return "pass"
    if name in {
        "simplify_add_small",
        "contextual_strict_fast",
        "contextual_radical_fast",
    }:
        if metrics["failed"] > 0:
            return "fail"
        if metrics.get("timeouts", 0) > 0 or metrics.get("numeric_only", 0) > 0:
            return "warn"
        return "pass"
    if name in {
        "calculus_residual_matrix_smoke",
        "calculus_diff_command_matrix_smoke",
        "calculus_limit_command_matrix_smoke",
        "calculus_integrate_command_matrix_smoke",
    }:
        if metrics.get("matrix_status") != "pass":
            return "fail"
        unchecked_supported_step_keys = (
            "diff_supported_step_unchecked_case_count",
            "limit_supported_step_unchecked_case_count",
            "integrate_supported_step_unchecked_case_count",
        )
        if any(
            metrics.get(key, 0) > 0
            for key in unchecked_supported_step_keys
            if isinstance(metrics.get(key, 0), int)
        ):
            return "fail"
        return "pass"
    if name == "calculus_integrate_backend_observability":
        if metrics.get("backend_unverified_public_acceptances", 0) > 0:
            return "fail"
        if metrics.get("backend_unverified_fallback_acceptances", 0) > 0:
            return "fail"
        public_trace_levels = metrics.get("backend_public_trace_level_counts", {})
        if (
            isinstance(public_trace_levels, dict)
            and public_trace_levels.get("diagnostic_only", 0) > 0
        ):
            return "fail"
        fallback_trace_levels = metrics.get("backend_fallback_trace_level_counts", {})
        if (
            isinstance(fallback_trace_levels, dict)
            and fallback_trace_levels.get("diagnostic_only", 0) > 0
        ):
            return "fail"
        public_constant_policies = metrics.get(
            "backend_public_constant_policy_counts", {}
        )
        if (
            isinstance(public_constant_policies, dict)
            and public_constant_policies.get("unspecified", 0) > 0
        ):
            return "fail"
        fallback_constant_policies = metrics.get(
            "backend_fallback_constant_policy_counts", {}
        )
        if (
            isinstance(fallback_constant_policies, dict)
            and fallback_constant_policies.get("unspecified", 0) > 0
        ):
            return "fail"
        if metrics.get("backend_public_assumption_exprs", 0) > 0:
            return "fail"
        if metrics.get("backend_fallback_assumption_exprs", 0) > 0:
            return "fail"
        public_verification_evidence = metrics.get(
            "backend_public_verification_evidence_counts", {}
        )
        if isinstance(public_verification_evidence, dict) and (
            public_verification_evidence.get("none", 0) > 0
            or public_verification_evidence.get("failed_differentiation", 0) > 0
        ):
            return "fail"
        fallback_verification_evidence = metrics.get(
            "backend_fallback_verification_evidence_counts", {}
        )
        if isinstance(fallback_verification_evidence, dict) and (
            fallback_verification_evidence.get("none", 0) > 0
            or fallback_verification_evidence.get("failed_differentiation", 0) > 0
        ):
            return "fail"
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


def format_runtime_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000.0:.2f}ms"
    return f"{seconds:.2f}s"


def compact_pressure_expression(expr: Any, max_chars: int = 140) -> str:
    text = " ".join(str(expr).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def effective_elapsed_seconds(metrics: dict[str, Any], process_elapsed_seconds: float) -> float:
    reported = metrics.get("reported_elapsed_seconds")
    if isinstance(reported, (int, float)):
        return float(reported)
    return process_elapsed_seconds


def add_elapsed_per_case_metric(
    metrics: dict[str, Any], elapsed_seconds: float
) -> None:
    if "reported_elapsed_per_case_ms" in metrics:
        return
    total_cases = metrics.get("total_cases")
    if not isinstance(total_cases, int) or total_cases <= 0:
        return
    metrics["reported_elapsed_per_case_ms"] = round(
        elapsed_seconds * 1000.0 / total_cases,
        4,
    )


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


def combined_additive_structure_summary(corpus_structure: dict[str, Any]) -> str:
    combined = corpus_structure.get("combined_additive_zero")
    if not isinstance(combined, dict):
        return ""

    family_count = corpus_structure.get("family_count")
    total_families = family_count if isinstance(family_count, int) else None
    fragments = [
        f"total={combined.get('total', 0)}",
        f"families={combined.get('family_count', 0)}",
        f"collapse_rows={combined.get('collapse_rows', 0)}",
        family_coverage_fragment(
            "collapse_families",
            combined.get("collapse_family_count"),
            total_families,
        ),
        f"depth4_rows={combined.get('depth4_rows', 0)}",
        family_coverage_fragment(
            "depth4_families",
            combined.get("depth4_family_count"),
            total_families,
        ),
        f"orientation_rows={combined.get('orientation_rows', 0)}",
        family_coverage_fragment(
            "orientation_families",
            combined.get("orientation_family_count"),
            total_families,
        ),
        f"multi_core_rows={combined.get('multi_core_rows', 0)}",
        family_coverage_fragment(
            "multi_core_families",
            combined.get("multi_core_family_count"),
            total_families,
        ),
    ]

    for label, field in (
        ("depth4_missing", "depth4_missing_families"),
        ("orientation_missing", "orientation_missing_families"),
    ):
        missing = combined.get(field)
        if isinstance(missing, list) and missing:
            fragments.append(f"{label}={format_family_list(missing)}")

    min_case_count = combined.get("min_family_case_count")
    target_case_count = combined.get("target_family_case_count")
    low_family_count = combined.get("low_family_count")
    if isinstance(min_case_count, int):
        fragments.append(f"min_family_case_count={min_case_count}")
    if isinstance(target_case_count, int):
        fragments.append(f"target_family_case_count={target_case_count}")
    if isinstance(low_family_count, int):
        if isinstance(total_families, int) and total_families > 0:
            fragments.append(f"families_at_min={low_family_count}/{total_families}")
        else:
            fragments.append(f"families_at_min={low_family_count}")

    under_target_family_count = combined.get("under_target_family_count")
    if isinstance(under_target_family_count, int):
        if isinstance(total_families, int) and total_families > 0:
            fragments.append(
                f"families_under_target={under_target_family_count}/{total_families}"
            )
        else:
            fragments.append(f"families_under_target={under_target_family_count}")

    low_family_counts = combined.get("low_family_counts")
    if isinstance(low_family_counts, dict) and low_family_counts:
        fragments.append(f"low_family_counts={format_family_counts(low_family_counts)}")

    under_target_counts = combined.get("under_target_family_counts")
    if isinstance(under_target_counts, dict) and under_target_counts:
        fragments.append(
            f"under_target_family_counts={format_family_counts(under_target_counts)}"
        )

    above_min_counts = combined.get("above_min_family_counts")
    if isinstance(above_min_counts, dict) and above_min_counts:
        fragments.append(
            f"above_min_family_counts={format_family_counts(above_min_counts)}"
        )

    return " ".join(fragment for fragment in fragments if fragment)


def format_family_counts(counts: dict[Any, Any]) -> str:
    items = [
        (family, count)
        for family, count in counts.items()
        if isinstance(family, str) and isinstance(count, int)
    ]
    return ", ".join(
        f"{family}:{count}"
        for family, count in sorted(items)
    )


def format_top_counts(counts: dict[Any, Any], max_items: int = 12) -> str:
    items = [
        (family, count)
        for family, count in counts.items()
        if isinstance(family, str) and isinstance(count, int)
    ]
    if not items:
        return "none"
    ordered = sorted(items, key=lambda item: (-item[1], item[0]))
    visible = ordered[:max_items]
    summary = ", ".join(f"{family}:{count}" for family, count in visible)
    if len(ordered) > max_items:
        summary += f", +{len(ordered) - max_items} more"
    return summary


def format_blocked_low_family_discoveries(counts: dict[Any, Any]) -> str:
    fragments = []
    for family, row in sorted(counts.items()):
        if not isinstance(family, str) or not isinstance(row, dict):
            continue
        live_count = row.get("live_count")
        discovery_count = row.get("observe_only_discoveries")
        if not isinstance(live_count, int) or not isinstance(discovery_count, int):
            continue
        fragments.append(
            f"{family}:live={live_count},observe_only={discovery_count}"
        )
    return ", ".join(fragments)


def format_ignored_tests(tests: Any, max_items: int = 3) -> str:
    if not isinstance(tests, list) or not tests:
        return "none"
    fragments = []
    for row in tests[:max_items]:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        reason = row.get("reason")
        if not isinstance(name, str):
            continue
        fragment = f"`{name}`"
        if isinstance(reason, str) and reason:
            fragment += f" ({reason})"
        fragments.append(fragment)
    if not fragments:
        return "none"
    if len(tests) > max_items:
        fragments.append(f"+{len(tests) - max_items} more")
    return ", ".join(fragments)


def format_family_list(families: list[Any], max_items: int = 8) -> str:
    names = sorted(family for family in families if isinstance(family, str))
    if not names:
        return ""
    if len(names) <= max_items:
        return ",".join(names)
    visible = ",".join(names[:max_items])
    return f"{visible},+{len(names) - max_items}"


def family_coverage_fragment(
    label: str,
    covered: Any,
    total_families: int | None,
) -> str:
    if not isinstance(covered, int):
        return ""
    if isinstance(total_families, int) and total_families > 0:
        return f"{label}={covered}/{total_families}"
    return f"{label}={covered}"


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


def sparse_wrapper_shell_depth_family_breadth_summary(
    wrapper_shell_depth_family_rows: list[dict[str, Any]],
    sparse_wrappers: set[str],
    family_count: int | None,
    shell_depth: int,
) -> str:
    family_names_by_wrapper: dict[str, set[str]] = {}
    case_totals_by_wrapper: dict[str, int] = {}
    for row in wrapper_shell_depth_family_rows:
        wrapper = row.get("wrapper")
        row_shell_depth = row.get("shell_depth")
        family = row.get("family")
        total = row.get("total")
        if not isinstance(wrapper, str) or wrapper not in sparse_wrappers:
            continue
        if row_shell_depth != shell_depth or not isinstance(family, str):
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
                f"{wrapper} depth{shell_depth}_families="
                f"{covered_families}/{family_count} missing={missing_families} "
                f"cases={total_cases}"
            )
        else:
            fragments.append(
                f"{wrapper} depth{shell_depth}_families={covered_families} "
                f"cases={total_cases}"
            )
    return ", ".join(fragments)


def shell_depth_summary_rows(
    shell_depth_rows: dict[int, dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    ordered_rows = sorted(shell_depth_rows.items())
    if len(ordered_rows) <= 8:
        return ordered_rows
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
        elapsed_per_case_ms = embedded_metrics.get("reported_elapsed_per_case_ms")
        if isinstance(elapsed_per_case_ms, (int, float)):
            lines.append(f"- Per-case runtime: {elapsed_per_case_ms:.3f}ms/case")
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
        corpus_structure = embedded_metrics.get("corpus_structure")
        if isinstance(corpus_structure, dict):
            combined_summary = combined_additive_structure_summary(corpus_structure)
            if combined_summary:
                lines.append(
                    "- Combined additive composition: "
                    f"{combined_summary}"
                )
        coverage_saturation = embedded_metrics.get("coverage_saturation")
        if not isinstance(coverage_saturation, dict):
            coverage_saturation = embedded_coverage_saturation_metrics(embedded_metrics)
        if isinstance(coverage_saturation, dict) and coverage_saturation:
            status = coverage_saturation.get("status")
            balanced_count = coverage_saturation.get("balanced_check_count")
            open_count = coverage_saturation.get("open_check_count")
            recommendation = coverage_saturation.get("recommendation")
            if (
                isinstance(status, str)
                and isinstance(balanced_count, int)
                and isinstance(open_count, int)
                and isinstance(recommendation, str)
            ):
                total_checks = balanced_count + open_count
                lines.append(
                    "- Embedded coverage saturation: "
                    f"{status} ({balanced_count}/{total_checks} checks); "
                    f"{recommendation}."
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
        wrapper_shell_depth_family_rows = embedded_metrics.get(
            "wrapper_shell_depth_family_rows"
        )
        if (
            isinstance(wrapper_shell_depth_family_rows, list)
            and wrapper_shell_depth_family_rows
        ):
            sparse_shell_depth_family_summary = ", ".join(
                (
                    f"{row['wrapper']} x depth {row['shell_depth']} x {row['family']} "
                    f"total={row['total']} failed={row['failed']}"
                )
                for row in wrapper_shell_depth_family_rows
            )
            lines.append(
                "- Sparse wrapper x shell-depth family buckets: "
                f"{sparse_shell_depth_family_summary}"
            )
            family_count_for_shell_depth = (
                family_count if isinstance(family_count, int) else None
            )
            depth4_family_breadth_summary = (
                sparse_wrapper_shell_depth_family_breadth_summary(
                    wrapper_shell_depth_family_rows,
                    sparse_wrappers,
                    family_count_for_shell_depth,
                    4,
                )
            )
            if depth4_family_breadth_summary:
                lines.append(
                    "- Sparse wrapper depth 4 family breadth: "
                    f"{depth4_family_breadth_summary}"
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
                profile_filter = profile_slice.get("filter", "unknown")
                lines.append(
                    (
                        "- Profiled slice: "
                        f"{profile_slice['total_cases']} cases "
                        f"(limit {profile_slice['limit']}), "
                        f"{profile_slice['wrapper_count']} wrappers, "
                        f"{profile_slice['family_count']} families, "
                        f"{profile_slice['elapsed_seconds']:.2f}s elapsed, "
                        f"filter `{profile_filter}`."
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
                sample_suffix = orchestrator_profile_sample_suffix(row)
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
                    sample_suffix = orchestrator_profile_sample_suffix(row, prefer_miss=True)
                    lines.append(
                        (
                            f"- No-match hotspot {idx}: `{row['section']}` "
                            f"{row['total_ms']:.3f}ms "
                            f"(misses {row['misses']} of {row['attempts']})"
                            f"{sample_suffix}"
                        )
                    )
            lines.append("")

    discovery_metrics = scorecard.get("generated_discovery")
    if isinstance(discovery_metrics, dict):
        observe_only_total = discovery_metrics.get("observe_only_discoveries")
        if isinstance(observe_only_total, int):
            families = discovery_metrics.get("families")
            wrappers = discovery_metrics.get("wrappers")
            recent = discovery_metrics.get("recent")
            discovery_pressure = scorecard.get("generated_discovery_pressure")
            if not isinstance(discovery_pressure, dict):
                discovery_pressure = generated_discovery_pressure_metrics(scorecard)
            lines.extend(
                [
                    "## Generated Discovery Ledger",
                    "",
                    "- Purpose: keep failed generated candidates visible without promoting them to live corpus.",
                    f"- Observe-only discoveries: total={observe_only_total}",
                ]
            )
            if observe_only_total == 0:
                lines.append("- Status: no open observe-only generated discoveries.")
            areas = discovery_metrics.get("areas")
            if isinstance(areas, dict) and areas:
                lines.append(f"- By area: {format_top_counts(areas, max_items=8)}")
            if isinstance(families, dict) and families:
                lines.append(f"- By family: {format_family_counts(families)}")
            if isinstance(wrappers, dict) and wrappers:
                lines.append(f"- By wrapper: {format_family_counts(wrappers)}")
            blocked_low_families = discovery_pressure.get("blocked_low_families")
            blocked_low_count = discovery_pressure.get("blocked_low_family_count")
            low_family_count = discovery_pressure.get("low_family_count")
            if (
                isinstance(blocked_low_families, dict)
                and blocked_low_families
                and isinstance(blocked_low_count, int)
                and isinstance(low_family_count, int)
            ):
                lines.append(
                    "- Low-family discovery pressure: "
                    f"blocked={blocked_low_count}/{low_family_count} "
                    f"{format_blocked_low_family_discoveries(blocked_low_families)}"
                )
            if isinstance(recent, list) and recent:
                for idx, row in enumerate(recent[:3], start=1):
                    if not isinstance(row, dict):
                        continue
                    title = row.get("title")
                    area = row.get("area", "unknown")
                    family = row.get("family")
                    wrapper = row.get("wrapper")
                    if not all(
                        isinstance(value, str)
                        for value in (title, area, family, wrapper)
                    ):
                        continue
                    if family != "unknown" and wrapper != "unknown":
                        lines.append(
                            f"- Recent {idx}: `{family}` in `{wrapper}` - {title}"
                        )
                    elif family != "unknown":
                        lines.append(f"- Recent {idx}: `{family}` - {title}")
                    elif area != "unknown":
                        lines.append(f"- Recent {idx}: `{area}` - {title}")
                    else:
                        lines.append(f"- Recent {idx}: {title}")
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
            max_step_count = derive_metrics.get("max_step_count")
            max_step_count_label = (
                max_step_count if max_step_count is not None else "n/a"
            )
            single_step_successes = derive_metrics.get("single_step_successes")
            multi_step_successes = derive_metrics.get("multi_step_successes")
            path_quality = (
                f"- Path quality: mean_step_count={derive_metrics['mean_step_count']:.2f} "
                f"long_path_rate={derive_metrics['long_path_rate']:.2f}"
            )
            if single_step_successes is not None and multi_step_successes is not None:
                path_quality += (
                    f" single_step_successes={single_step_successes} "
                    f"multi_step_successes={multi_step_successes} "
                    f"max_step_count={max_step_count_label}"
                )
            lines.extend(
                [
                    (
                        f"- Expected-status breakdown: derived={derive_metrics['derived']} "
                        f"unsupported={derive_metrics['unsupported']} "
                        f"not_equivalent={derive_metrics['not_equivalent']}"
                    ),
                    path_quality,
                    (
                        f"- Strategy specificity: generic_simplify_expected="
                        f"{derive_metrics.get('generic_simplify_expected', 0)} "
                        f"distinct_expected_strategies="
                        f"{derive_metrics.get('distinct_expected_strategies', 'n/a')}"
                    ),
                    (
                        "- Expected strategy counts: "
                        + format_top_counts(
                            derive_metrics.get("expected_strategy_counts", {})
                        )
                    ),
                    (
                        "- Non-derived expected families: "
                        f"unsupported={format_top_counts(derive_metrics.get('unsupported_by_family', {}))} "
                        f"not_equivalent={format_top_counts(derive_metrics.get('not_equivalent_by_family', {}))}"
                    ),
                    "",
                ]
            )

    derive_shadow_suite = scorecard["suites"].get("derive_shadow_pressure")
    if derive_shadow_suite:
        derive_shadow_metrics = derive_shadow_suite["metrics"]
        lines.extend(
            [
                "## Derive Shadow Pressure",
                "",
                "- Dimension: diagnostic engine-to-derive bridgeability over representative engine equivalence rows.",
                "- Interpretation: exposes where known engine/metamorphic identities are not yet reachable or provable as derive targets; diagnostic, not a support gate.",
            ]
        )
        if "parse_error" in derive_shadow_metrics:
            lines.extend(
                [
                    f"- Parser status: `parse_error={derive_shadow_metrics['parse_error']}`",
                    "",
                ]
            )
        else:
            max_step_count = derive_shadow_metrics.get("max_step_count")
            max_step_count_label = (
                max_step_count if max_step_count is not None else "n/a"
            )
            lines.extend(
                [
                    (
                        f"- Outcomes: sampled={derive_shadow_metrics['sampled']} "
                        f"derived={derive_shadow_metrics['derived']} "
                        f"unsupported={derive_shadow_metrics['unsupported']} "
                        f"not_equivalent={derive_shadow_metrics['not_equivalent']}"
                    ),
                    (
                        f"- Path signal: mean_step_count={derive_shadow_metrics['mean_step_count']:.2f} "
                        f"single_step_successes={derive_shadow_metrics['single_step_successes']} "
                        f"multi_step_successes={derive_shadow_metrics['multi_step_successes']} "
                        f"max_step_count={max_step_count_label}"
                    ),
                    (
                        f"- Strategy specificity: generic_simplify_strategy_successes="
                        f"{derive_shadow_metrics.get('generic_simplify_strategy_successes', 0)} "
                        f"distinct_actual_strategies="
                        f"{derive_shadow_metrics.get('distinct_actual_strategies', 'n/a')}"
                    ),
                    (
                        f"- Embedded family coverage: sampled_families="
                        f"{derive_shadow_metrics.get('embedded_family_sampled_count', 'n/a')} "
                        f"total_families="
                        f"{derive_shadow_metrics.get('embedded_family_total_count', 'n/a')} "
                        f"missing="
                        + (
                            ", ".join(
                                derive_shadow_metrics.get(
                                    "embedded_family_missing", []
                                )
                            )
                            or "none"
                        )
                    ),
                    (
                        "- Multi-step shadow IDs: "
                        + (
                            ", ".join(
                                derive_shadow_metrics.get(
                                    "multi_step_success_ids", []
                                )
                            )
                            or "none"
                        )
                    ),
                    (
                        "- Generic simplify shadow IDs: "
                        + (
                            ", ".join(
                                derive_shadow_metrics.get(
                                    "generic_simplify_strategy_ids", []
                                )
                            )
                            or "none"
                        )
                    ),
                    (
                        "- Actual strategy counts: "
                        + format_top_counts(
                            derive_shadow_metrics.get("actual_strategy_counts", {})
                        )
                    ),
                    (
                        "- Derived family counts: "
                        + format_top_counts(
                            derive_shadow_metrics.get("derived_by_family", {})
                        )
                    ),
                    (
                        "- Problem family counts: "
                        f"unsupported={format_top_counts(derive_shadow_metrics.get('unsupported_by_family', {}))} "
                        f"not_equivalent={format_top_counts(derive_shadow_metrics.get('not_equivalent_by_family', {}))}"
                    ),
                    "",
                ]
            )

    derive_didactic_suite = scorecard["suites"].get("derive_didactic_audit")
    if derive_didactic_suite:
        derive_didactic_metrics = derive_didactic_suite["metrics"]
        lines.extend(
            [
                "## Derive Didactic Trace Audit",
                "",
                "- Dimension: educational quality of derive target-driven traces and web substeps.",
                "- Interpretation: diagnostic trace-quality lane; not a semantic correctness or runtime metric.",
            ]
        )
        if "parse_error" in derive_didactic_metrics:
            lines.extend(
                [
                    f"- Parser status: `parse_error={derive_didactic_metrics['parse_error']}`",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    (
                        f"- Outcomes: cases={derive_didactic_metrics['cases']} "
                        f"flagged={derive_didactic_metrics['flagged_cases']} "
                        f"flagged_rate={derive_didactic_metrics['flagged_rate']:.1%}"
                    ),
                    (
                        f"- Substeps: total_web_substeps={derive_didactic_metrics['total_web_substeps']} "
                        f"mean_step_count={derive_didactic_metrics['mean_step_count']:.2f}"
                    ),
                    (
                        f"- Flags: no_web_substeps={derive_didactic_metrics['no_web_substeps']} "
                        f"no_web_steps={derive_didactic_metrics['no_web_steps']}"
                    ),
                ]
            )
            if "artifact_seconds" in derive_didactic_metrics:
                lines.append(
                    "- Runtime split: "
                    f"artifacts={derive_didactic_metrics['artifact_seconds']:.2f}s "
                    f"cli={derive_didactic_metrics['cli_seconds']:.2f}s "
                    f"report={derive_didactic_metrics['report_seconds']:.2f}s "
                    f"total={derive_didactic_metrics['reported_total_seconds']:.2f}s "
                    f"workers={derive_didactic_metrics['worker_count']}"
                )
            if "artifact_family_hotspots" in derive_didactic_metrics:
                lines.append(
                    "- Runtime family hotspots: "
                    f"artifacts={derive_didactic_metrics['artifact_family_hotspots']} "
                    f"cli={derive_didactic_metrics['cli_family_hotspots']}"
                )
            if "artifact_case_hotspots" in derive_didactic_metrics:
                lines.append(
                    "- Runtime case hotspots: "
                    f"artifacts={derive_didactic_metrics['artifact_case_hotspots']} "
                    f"cli={derive_didactic_metrics['cli_case_hotspots']}"
                )
            lines.append("")

    simplify_didactic_suite = scorecard["suites"].get("simplify_didactic_audit")
    if simplify_didactic_suite:
        simplify_didactic_metrics = simplify_didactic_suite["metrics"]
        lines.extend(
            [
                "## Simplify Didactic Trace Audit",
                "",
                "- Dimension: educational quality of simplify step traces and web substeps.",
                "- Interpretation: diagnostic trace-quality lane; not a semantic correctness or runtime metric.",
            ]
        )
        if "parse_error" in simplify_didactic_metrics:
            lines.extend(
                [
                    f"- Parser status: `parse_error={simplify_didactic_metrics['parse_error']}`",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    (
                        f"- Outcomes: cases={simplify_didactic_metrics['cases']} "
                        f"flagged={simplify_didactic_metrics['flagged_cases']} "
                        f"flagged_rate={simplify_didactic_metrics['flagged_rate']:.1%}"
                    ),
                    (
                        f"- Substeps: total_wire_substeps={simplify_didactic_metrics['total_wire_substeps']} "
                        f"mean_step_count={simplify_didactic_metrics['mean_step_count']:.2f}"
                    ),
                    (
                        f"- Flags: no_wire_substeps={simplify_didactic_metrics['no_wire_substeps']} "
                        f"single_step_no_substeps={simplify_didactic_metrics['single_step_no_substeps']} "
                        f"missing_math_sides={simplify_didactic_metrics['missing_math_sides']}"
                    ),
                    "",
                ]
            )

    calculus_contract_rows = [
        (label, suite)
        for label, suite in (
            ("diff", scorecard["suites"].get("calculus_diff_contract")),
            (
                "diff_command_matrix",
                scorecard["suites"].get("calculus_diff_command_matrix_smoke"),
            ),
            (
                "diff_exhaustive",
                scorecard["suites"].get("calculus_diff_exhaustive_contract"),
            ),
            ("limit", scorecard["suites"].get("calculus_limit_contract")),
            (
                "limit_compact",
                scorecard["suites"].get("calculus_limit_compact_contract"),
            ),
            (
                "limit_presimplify_safe",
                scorecard["suites"].get("calculus_limit_presimplify_contract"),
            ),
            (
                "limit_command_matrix",
                scorecard["suites"].get("calculus_limit_command_matrix_smoke"),
            ),
            (
                "integrate_compact",
                scorecard["suites"].get("calculus_integrate_compact_contract"),
            ),
            (
                "integrate_backend_observability",
                scorecard["suites"].get("calculus_integrate_backend_observability"),
            ),
            (
                "integrate_command_matrix",
                scorecard["suites"].get("calculus_integrate_command_matrix_smoke"),
            ),
            (
                "residual_matrix",
                scorecard["suites"].get("calculus_residual_matrix_smoke"),
            ),
            ("integrate", scorecard["suites"].get("calculus_integrate_contract")),
            (
                "integrate_exhaustive",
                scorecard["suites"].get("calculus_integrate_exhaustive_contract"),
            ),
        )
        if suite
    ]
    if calculus_contract_rows:
        lines.extend(
            [
                "## Calculus Support Matrix Signal",
                "",
                "- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.",
                "- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.",
                "- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.",
            ]
        )
        runtime_guardrail = calculus_runtime_guardrail_fragments(scorecard)
        if runtime_guardrail:
            lines.append(
                "- Calculus runtime guardrail: " + "; ".join(runtime_guardrail)
            )
        for label, suite in calculus_contract_rows:
            metrics = suite["metrics"]
            if "parse_error" in metrics:
                lines.append(
                    f"- `{label}`: parse_error={metrics['parse_error']}"
                )
            elif "total_cases" in metrics:
                line = (
                    f"- `{label}`: passed={metrics['passed']} "
                    f"failed={metrics['failed']} total={metrics['total_cases']} "
                    f"slow={metrics.get('slow', 0)} "
                    f"timeouts={metrics.get('timeouts', 0)}"
                )
                matrix_base_count = metrics.get("matrix_base_count")
                matrix_wrapped_base_count = metrics.get("matrix_wrapped_base_count")
                matrix_standalone_base_count = metrics.get(
                    "matrix_standalone_base_count"
                )
                matrix_wrapper_count = metrics.get("matrix_wrapper_count")
                matrix_wrapped_case_count = metrics.get("matrix_wrapped_case_count")
                matrix_standalone_case_count = metrics.get(
                    "matrix_standalone_case_count"
                )
                if isinstance(matrix_base_count, int):
                    line += f" total_bases={matrix_base_count}"
                if isinstance(matrix_wrapped_base_count, int):
                    line += f" wrapped_bases={matrix_wrapped_base_count}"
                if isinstance(matrix_standalone_base_count, int):
                    line += f" standalone_bases={matrix_standalone_base_count}"
                if isinstance(matrix_wrapper_count, int):
                    line += f" wrappers={matrix_wrapper_count}"
                if isinstance(matrix_wrapped_case_count, int) and isinstance(
                    matrix_standalone_case_count, int
                ):
                    line += (
                        f" wrapped_cases={matrix_wrapped_case_count} "
                        f"standalone_cases={matrix_standalone_case_count}"
                    )
                conditioned_cases = metrics.get("expected_required_condition_case_count")
                distinct_conditions = metrics.get("distinct_expected_required_conditions")
                if isinstance(conditioned_cases, int) and isinstance(
                    distinct_conditions, int
                ):
                    line += (
                        f" conditioned_cases={conditioned_cases} "
                        f"distinct_conditions={distinct_conditions}"
                    )
                limit_supported_cases = metrics.get("limit_supported_case_count")
                limit_residual_cases = metrics.get("limit_residual_case_count")
                limit_warning_expected_cases = metrics.get(
                    "limit_warning_expected_case_count"
                )
                limit_required_display_cases = metrics.get(
                    "limit_required_display_case_count"
                )
                limit_step_checked_cases = metrics.get("limit_step_checked_case_count")
                limit_supported_step_unchecked_cases = metrics.get(
                    "limit_supported_step_unchecked_case_count"
                )
                limit_family_count = metrics.get("limit_family_count")
                diff_supported_cases = metrics.get("diff_supported_case_count")
                diff_residual_cases = metrics.get("diff_residual_case_count")
                diff_warning_expected_cases = metrics.get(
                    "diff_warning_expected_case_count"
                )
                diff_required_display_cases = metrics.get(
                    "diff_required_display_case_count"
                )
                diff_step_checked_cases = metrics.get(
                    "diff_step_checked_case_count"
                )
                diff_supported_step_unchecked_cases = metrics.get(
                    "diff_supported_step_unchecked_case_count"
                )
                diff_family_count = metrics.get("diff_family_count")
                integrate_supported_cases = metrics.get(
                    "integrate_supported_case_count"
                )
                integrate_residual_cases = metrics.get(
                    "integrate_residual_case_count"
                )
                integrate_warning_expected_cases = metrics.get(
                    "integrate_warning_expected_case_count"
                )
                integrate_required_display_cases = metrics.get(
                    "integrate_required_display_case_count"
                )
                integrate_step_checked_cases = metrics.get(
                    "integrate_step_checked_case_count"
                )
                integrate_supported_step_unchecked_cases = metrics.get(
                    "integrate_supported_step_unchecked_case_count"
                )
                integrate_antiderivative_verification_cases = metrics.get(
                    "integrate_antiderivative_verification_case_count"
                )
                integrate_verified_supported_cases = metrics.get(
                    "integrate_verified_supported_case_count"
                )
                integrate_family_count = metrics.get("integrate_family_count")
                backend_attempts = metrics.get("backend_attempts")
                backend_public_accepted = metrics.get("backend_public_accepted")
                backend_unverified_public_acceptances = metrics.get(
                    "backend_unverified_public_acceptances"
                )
                backend_fallback_eligible = metrics.get("backend_fallback_eligible")
                backend_unverified_fallback_acceptances = metrics.get(
                    "backend_unverified_fallback_acceptances"
                )
                backend_verified_count = metrics.get("backend_verified_count")
                backend_failed_or_blocked_count = metrics.get(
                    "backend_failed_or_blocked_count"
                )
                backend_required_condition_count = metrics.get(
                    "backend_required_condition_count"
                )
                backend_budget_exceeded_count = metrics.get(
                    "backend_budget_exceeded_count"
                )
                backend_method_probe_budget_exhausted_count = metrics.get(
                    "backend_method_probe_budget_exhausted_count"
                )
                backend_verification_budget_exceeded_count = metrics.get(
                    "backend_verification_budget_exceeded_count"
                )
                backend_method_probes_used_total = metrics.get(
                    "backend_method_probes_used_total"
                )
                backend_verification_checks_used_total = metrics.get(
                    "backend_verification_checks_used_total"
                )
                backend_verification_elapsed_ms = metrics.get(
                    "backend_verification_elapsed_ms"
                )
                backend_verification_pressure_status = metrics.get(
                    "backend_verification_pressure_status"
                )
                if isinstance(diff_supported_cases, int):
                    line += f" supported_cases={diff_supported_cases}"
                if isinstance(diff_residual_cases, int):
                    line += f" residual_cases={diff_residual_cases}"
                if isinstance(diff_warning_expected_cases, int):
                    line += f" warning_expected={diff_warning_expected_cases}"
                if isinstance(diff_required_display_cases, int):
                    line += f" required_display={diff_required_display_cases}"
                if isinstance(diff_step_checked_cases, int):
                    line += f" step_checked={diff_step_checked_cases}"
                if isinstance(diff_supported_step_unchecked_cases, int):
                    line += (
                        " unchecked_supported_steps="
                        f"{diff_supported_step_unchecked_cases}"
                    )
                if isinstance(diff_family_count, int):
                    line += f" families={diff_family_count}"
                if isinstance(limit_supported_cases, int):
                    line += f" supported_cases={limit_supported_cases}"
                if isinstance(limit_residual_cases, int):
                    line += f" residual_cases={limit_residual_cases}"
                if isinstance(limit_warning_expected_cases, int):
                    line += f" warning_expected={limit_warning_expected_cases}"
                if isinstance(limit_required_display_cases, int):
                    line += f" required_display={limit_required_display_cases}"
                if isinstance(limit_step_checked_cases, int):
                    line += f" step_checked={limit_step_checked_cases}"
                if isinstance(limit_supported_step_unchecked_cases, int):
                    line += (
                        " unchecked_supported_steps="
                        f"{limit_supported_step_unchecked_cases}"
                    )
                if isinstance(limit_family_count, int):
                    line += f" families={limit_family_count}"
                if isinstance(integrate_supported_cases, int):
                    line += f" supported_cases={integrate_supported_cases}"
                if isinstance(integrate_residual_cases, int):
                    line += f" residual_cases={integrate_residual_cases}"
                if isinstance(integrate_warning_expected_cases, int):
                    line += (
                        f" warning_expected={integrate_warning_expected_cases}"
                    )
                if isinstance(integrate_required_display_cases, int):
                    line += (
                        f" required_display={integrate_required_display_cases}"
                    )
                if isinstance(integrate_step_checked_cases, int):
                    line += f" step_checked={integrate_step_checked_cases}"
                if isinstance(integrate_supported_step_unchecked_cases, int):
                    line += (
                        " unchecked_supported_steps="
                        f"{integrate_supported_step_unchecked_cases}"
                    )
                if isinstance(integrate_antiderivative_verification_cases, int):
                    line += (
                        " antiderivative_verified="
                        f"{integrate_antiderivative_verification_cases}"
                    )
                if isinstance(integrate_verified_supported_cases, int):
                    line += (
                        " verified_supported="
                        f"{integrate_verified_supported_cases}"
                    )
                integrate_direct_diff_integrate_cases = metrics.get(
                    "integrate_direct_diff_integrate_case_count"
                )
                if isinstance(integrate_direct_diff_integrate_cases, int):
                    line += (
                        " direct_diff_integrate="
                        f"{integrate_direct_diff_integrate_cases}"
                    )
                integrate_direct_diff_integrate_exact_cases = metrics.get(
                    "integrate_direct_diff_integrate_exact_case_count"
                )
                integrate_direct_diff_integrate_equivalence_cases = metrics.get(
                    "integrate_direct_diff_integrate_equivalence_case_count"
                )
                if isinstance(
                    integrate_direct_diff_integrate_exact_cases, int
                ) and isinstance(integrate_direct_diff_integrate_equivalence_cases, int):
                    line += (
                        " direct_exact="
                        f"{integrate_direct_diff_integrate_exact_cases}"
                        " direct_equiv="
                        f"{integrate_direct_diff_integrate_equivalence_cases}"
                    )
                integrate_direct_diff_integrate_gap_cases = metrics.get(
                    "integrate_direct_diff_integrate_gap_case_count"
                )
                if isinstance(integrate_direct_diff_integrate_gap_cases, int):
                    line += (
                        " direct_gap="
                        f"{integrate_direct_diff_integrate_gap_cases}"
                    )
                if isinstance(integrate_family_count, int):
                    line += f" families={integrate_family_count}"
                if isinstance(backend_attempts, int):
                    line += f" backend_attempts={backend_attempts}"
                if isinstance(backend_public_accepted, int):
                    line += f" backend_public_accepted={backend_public_accepted}"
                if isinstance(backend_unverified_public_acceptances, int):
                    line += (
                        " backend_unverified_public="
                        f"{backend_unverified_public_acceptances}"
                    )
                if isinstance(backend_fallback_eligible, int):
                    line += f" backend_fallback_eligible={backend_fallback_eligible}"
                if isinstance(backend_unverified_fallback_acceptances, int):
                    line += (
                        " backend_unverified_fallback="
                        f"{backend_unverified_fallback_acceptances}"
                    )
                if isinstance(backend_verified_count, int):
                    line += f" backend_verified={backend_verified_count}"
                if isinstance(backend_failed_or_blocked_count, int):
                    line += (
                        " backend_failed_or_blocked="
                        f"{backend_failed_or_blocked_count}"
                    )
                if isinstance(backend_required_condition_count, int):
                    line += (
                        " backend_required_conditions="
                        f"{backend_required_condition_count}"
                    )
                if isinstance(backend_budget_exceeded_count, int):
                    line += f" backend_budget_exceeded={backend_budget_exceeded_count}"
                if isinstance(backend_method_probe_budget_exhausted_count, int):
                    line += (
                        " backend_method_budget_exhausted="
                        f"{backend_method_probe_budget_exhausted_count}"
                    )
                if isinstance(backend_verification_budget_exceeded_count, int):
                    line += (
                        " backend_verification_budget_exceeded="
                        f"{backend_verification_budget_exceeded_count}"
                    )
                if isinstance(backend_method_probes_used_total, int):
                    line += (
                        " backend_method_probes_used="
                        f"{backend_method_probes_used_total}"
                    )
                if isinstance(backend_verification_checks_used_total, int):
                    line += (
                        " backend_verification_checks_used="
                        f"{backend_verification_checks_used_total}"
                    )
                if isinstance(backend_verification_elapsed_ms, (int, float)):
                    line += (
                        " backend_verification_ms="
                        f"{backend_verification_elapsed_ms:.3f}"
                    )
                if isinstance(backend_verification_pressure_status, str):
                    line += (
                        " backend_verification_pressure="
                        f"{backend_verification_pressure_status}"
                    )
                lines.append(line)
                if label == "diff_command_matrix":
                    lines.extend(
                        calculus_runtime_lines(
                            label,
                            metrics,
                            prefix="diff",
                            group_keys=(
                                "family",
                                "calculus_maturity_block",
                                "calculus_block_gate",
                                "domain_regime",
                                "trace_regime",
                                "presentation_regime",
                                "positive_quadratic_policy_cluster",
                                "variable_power_policy_cluster",
                            ),
                        )
                    )
                    slowest_process = phase_runtime_case_fragments(
                        metrics.get("diff_slowest_process_evaluations"),
                        elapsed_key="process_elapsed_seconds",
                    )
                    if slowest_process:
                        lines.append(
                            f"- `{label}` slowest process evaluations: "
                            + ", ".join(slowest_process)
                        )
                    for phase_label, distribution_key, rows_key, elapsed_key in (
                        (
                            "CLI simplify",
                            "diff_cli_simplify_runtime_distribution",
                            "diff_slowest_cli_simplify_evaluations",
                            "cli_simplify_elapsed_seconds",
                        ),
                        (
                            "CLI total",
                            "diff_cli_total_runtime_distribution",
                            "diff_slowest_cli_total_evaluations",
                            "cli_total_elapsed_seconds",
                        ),
                        (
                            "CLI public overhead",
                            "diff_cli_public_overhead_runtime_distribution",
                            "diff_slowest_cli_public_overhead_evaluations",
                            "cli_public_overhead_seconds",
                        ),
                    ):
                        phase_distribution = calculus_runtime_distribution_fragment(
                            metrics.get(distribution_key)
                        )
                        if phase_distribution:
                            lines.append(
                                f"- `{label}` {phase_label} runtime distribution: "
                                + phase_distribution
                            )
                        slowest_phase = phase_runtime_case_fragments(
                            metrics.get(rows_key),
                            elapsed_key=elapsed_key,
                        )
                        if slowest_phase:
                            lines.append(
                                f"- `{label}` slowest {phase_label} evaluations: "
                                + ", ".join(slowest_phase)
                            )
                    exact_square_pairs = diff_exact_square_runtime_pair_fragments(
                        metrics.get("diff_exact_square_runtime_pairs")
                    )
                    if exact_square_pairs:
                        lines.append(
                            f"- `{label}` exact-square runtime pairs: "
                            + ", ".join(exact_square_pairs)
                        )
                    harness_distribution = calculus_runtime_distribution_fragment(
                        metrics.get("diff_harness_check_runtime_distribution")
                    )
                    if harness_distribution:
                        lines.append(
                            f"- `{label}` harness check runtime distribution: "
                            + harness_distribution
                        )
                    slowest_harness = phase_runtime_case_fragments(
                        metrics.get("diff_slowest_harness_checks"),
                        elapsed_key="harness_check_elapsed_seconds",
                    )
                    if slowest_harness:
                        lines.append(
                            f"- `{label}` slowest harness checks: "
                            + ", ".join(slowest_harness)
                        )
                    largest_stdout = payload_case_fragments(
                        metrics.get("diff_largest_stdout_payload_cases"),
                        metric_key="stdout_bytes",
                        unit="B",
                    )
                    if largest_stdout:
                        lines.append(
                            f"- `{label}` largest stdout payload cases: "
                            + ", ".join(largest_stdout)
                        )
                    largest_steps = payload_case_fragments(
                        metrics.get("diff_largest_step_trace_cases"),
                        metric_key="step_text_char_count",
                        unit=" chars",
                    )
                    if largest_steps:
                        lines.append(
                            f"- `{label}` largest step trace cases: "
                            + ", ".join(largest_steps)
                        )
                elif label == "limit_command_matrix":
                    lines.extend(
                        calculus_runtime_lines(
                            label,
                            metrics,
                            prefix="limit",
                            group_keys=(
                                "family",
                                "point_regime",
                                "domain_regime",
                                "trace_regime",
                            ),
                        )
                    )
                    for phase_label, distribution_key, rows_key, elapsed_key in (
                        (
                            "CLI simplify",
                            "limit_cli_simplify_runtime_distribution",
                            "limit_slowest_cli_simplify_evaluations",
                            "cli_simplify_elapsed_seconds",
                        ),
                        (
                            "CLI total",
                            "limit_cli_total_runtime_distribution",
                            "limit_slowest_cli_total_evaluations",
                            "cli_total_elapsed_seconds",
                        ),
                        (
                            "CLI public overhead",
                            "limit_cli_public_overhead_runtime_distribution",
                            "limit_slowest_cli_public_overhead_evaluations",
                            "cli_public_overhead_seconds",
                        ),
                    ):
                        phase_distribution = calculus_runtime_distribution_fragment(
                            metrics.get(distribution_key)
                        )
                        if phase_distribution:
                            lines.append(
                                f"- `{label}` {phase_label} runtime distribution: "
                                + phase_distribution
                            )
                        slowest_phase = phase_runtime_case_fragments(
                            metrics.get(rows_key),
                            elapsed_key=elapsed_key,
                        )
                        if slowest_phase:
                            lines.append(
                                f"- `{label}` slowest {phase_label} evaluations: "
                                + ", ".join(slowest_phase)
                            )
                elif label == "integrate_backend_observability":
                    mode_counts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_mode_counts")
                    )
                    if mode_counts:
                        lines.append(
                            f"- `{label}` backend modes: " + ", ".join(mode_counts)
                        )
                    method_counts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_method_counts")
                    )
                    if method_counts:
                        lines.append(
                            f"- `{label}` backend methods: "
                            + ", ".join(method_counts)
                        )
                    verification_status_counts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_verification_status_counts")
                    )
                    if verification_status_counts:
                        lines.append(
                            f"- `{label}` verification statuses: "
                            + ", ".join(verification_status_counts)
                        )
                    residual_reason_counts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_residual_reason_counts")
                    )
                    if residual_reason_counts:
                        lines.append(
                            f"- `{label}` residual reasons: "
                            + ", ".join(residual_reason_counts)
                        )
                    method_budget = metrics.get(
                        "backend_method_probe_budget_exhausted_count"
                    )
                    verification_budget = metrics.get(
                        "backend_verification_budget_exceeded_count"
                    )
                    method_probes_used = metrics.get("backend_method_probes_used_total")
                    verification_checks_used = metrics.get(
                        "backend_verification_checks_used_total"
                    )
                    if isinstance(method_budget, int) or isinstance(
                        verification_budget, int
                    ):
                        budget_pieces = []
                        if isinstance(method_budget, int):
                            budget_pieces.append(
                                f"method_probe_exhausted={method_budget}"
                            )
                        if isinstance(verification_budget, int):
                            budget_pieces.append(
                                f"verification_exceeded={verification_budget}"
                            )
                        lines.append(
                            f"- `{label}` budget split: "
                            + ", ".join(budget_pieces)
                        )
                    usage_pieces = []
                    if isinstance(method_probes_used, int):
                        usage_pieces.append(f"method_probes_used={method_probes_used}")
                    if isinstance(verification_checks_used, int):
                        usage_pieces.append(
                            f"verification_checks_used={verification_checks_used}"
                        )
                    if usage_pieces:
                        lines.append(
                            f"- `{label}` budget usage: "
                            + ", ".join(usage_pieces)
                        )
                    verification_pressure = backend_verification_pressure_fragment(
                        metrics.get("backend_verification_pressure")
                    )
                    if verification_pressure:
                        lines.append(
                            f"- `{label}` backend verification pressure: "
                            + verification_pressure
                        )
                    method_probe_usage = calculus_matrix_count_map_fragments(
                        metrics.get("backend_method_probe_usage_by_method")
                    )
                    if method_probe_usage:
                        lines.append(
                            f"- `{label}` method-probe usage by method: "
                            + ", ".join(method_probe_usage)
                        )
                    method_probe_attempts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_method_probe_attempt_counts")
                    )
                    if method_probe_attempts:
                        lines.append(
                            f"- `{label}` method-probe attempts: "
                            + ", ".join(method_probe_attempts)
                        )
                    method_probe_candidates = calculus_matrix_count_map_fragments(
                        metrics.get("backend_method_probe_candidate_counts")
                    )
                    if method_probe_candidates:
                        lines.append(
                            f"- `{label}` method-probe candidates: "
                            + ", ".join(method_probe_candidates)
                        )
                    method_probe_no_matches = calculus_matrix_count_map_fragments(
                        metrics.get("backend_method_probe_no_match_counts")
                    )
                    if method_probe_no_matches:
                        lines.append(
                            f"- `{label}` method-probe no-matches: "
                            + ", ".join(method_probe_no_matches)
                        )
                    method_probe_no_match_reasons = (
                        calculus_matrix_count_map_fragments(
                            metrics.get("backend_method_probe_no_match_reason_counts")
                        )
                    )
                    if method_probe_no_match_reasons:
                        lines.append(
                            f"- `{label}` method-probe no-match reasons: "
                            + ", ".join(method_probe_no_match_reasons)
                        )
                    verification_usage = calculus_matrix_count_map_fragments(
                        metrics.get("backend_verification_check_usage_by_method")
                    )
                    if verification_usage:
                        lines.append(
                            f"- `{label}` verification-check usage by method: "
                            + ", ".join(verification_usage)
                        )
                    verification_status_by_method = (
                        calculus_matrix_count_map_fragments(
                            metrics.get("backend_verification_status_by_method")
                        )
                    )
                    if verification_status_by_method:
                        lines.append(
                            f"- `{label}` verification status by method: "
                            + ", ".join(verification_status_by_method)
                        )
                    residual_reason_by_method = calculus_matrix_count_map_fragments(
                        metrics.get("backend_residual_reason_by_method")
                    )
                    if residual_reason_by_method:
                        lines.append(
                            f"- `{label}` residual reason by method: "
                            + ", ".join(residual_reason_by_method)
                        )
                    verification_blocker_counts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_verification_blocker_counts")
                    )
                    if verification_blocker_counts:
                        lines.append(
                            f"- `{label}` verification blockers: "
                            + ", ".join(verification_blocker_counts)
                        )
                    verification_blocker_by_method = (
                        calculus_matrix_count_map_fragments(
                            metrics.get("backend_verification_blocker_by_method")
                        )
                    )
                    if verification_blocker_by_method:
                        lines.append(
                            f"- `{label}` verification blocker by method: "
                            + ", ".join(verification_blocker_by_method)
                        )
                    failure_class_counts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_failure_class_counts")
                    )
                    if failure_class_counts:
                        lines.append(
                            f"- `{label}` failure classes: "
                            + ", ".join(failure_class_counts)
                        )
                    failure_class_by_method = calculus_matrix_count_map_fragments(
                        metrics.get("backend_failure_class_by_method")
                    )
                    if failure_class_by_method:
                        lines.append(
                            f"- `{label}` failure class by method: "
                            + ", ".join(failure_class_by_method)
                        )
                    verification_residual_counts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_verification_residual_counts")
                    )
                    if verification_residual_counts:
                        lines.append(
                            f"- `{label}` verification residuals: "
                            + ", ".join(verification_residual_counts)
                        )
                    verification_residual_by_method = (
                        calculus_matrix_count_map_fragments(
                            metrics.get("backend_verification_residual_by_method")
                        )
                    )
                    if verification_residual_by_method:
                        lines.append(
                            f"- `{label}` verification residual by method: "
                            + ", ".join(verification_residual_by_method)
                        )
                    verification_residual_kind_counts = (
                        calculus_matrix_count_map_fragments(
                            metrics.get("backend_verification_residual_kind_counts")
                        )
                    )
                    if verification_residual_kind_counts:
                        lines.append(
                            f"- `{label}` verification residual kinds: "
                            + ", ".join(verification_residual_kind_counts)
                        )
                    verification_residual_kind_by_method = (
                        calculus_matrix_count_map_fragments(
                            metrics.get("backend_verification_residual_kind_by_method")
                        )
                    )
                    if verification_residual_kind_by_method:
                        lines.append(
                            f"- `{label}` verification residual kind by method: "
                            + ", ".join(verification_residual_kind_by_method)
                        )
                    verification_residual_signature_counts = (
                        calculus_matrix_count_map_fragments(
                            metrics.get("backend_verification_residual_signature_counts")
                        )
                    )
                    if verification_residual_signature_counts:
                        lines.append(
                            f"- `{label}` verification residual signatures: "
                            + ", ".join(verification_residual_signature_counts)
                        )
                    verification_residual_signature_by_method = (
                        calculus_matrix_count_map_fragments(
                            metrics.get(
                                "backend_verification_residual_signature_by_method"
                            )
                        )
                    )
                    if verification_residual_signature_by_method:
                        lines.append(
                            f"- `{label}` verification residual signature by method: "
                            + ", ".join(verification_residual_signature_by_method)
                        )
                    publication_status_counts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_publication_status_counts")
                    )
                    if publication_status_counts:
                        lines.append(
                            f"- `{label}` publication statuses: "
                            + ", ".join(publication_status_counts)
                        )
                    publication_status_by_method = (
                        calculus_matrix_count_map_fragments(
                            metrics.get("backend_publication_status_by_method")
                        )
                    )
                    if publication_status_by_method:
                        lines.append(
                            f"- `{label}` publication status by method: "
                            + ", ".join(publication_status_by_method)
                        )
                    fallback_status_counts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_fallback_status_counts")
                    )
                    if fallback_status_counts:
                        lines.append(
                            f"- `{label}` fallback statuses: "
                            + ", ".join(fallback_status_counts)
                        )
                    fallback_status_by_method = calculus_matrix_count_map_fragments(
                        metrics.get("backend_fallback_status_by_method")
                    )
                    if fallback_status_by_method:
                        lines.append(
                            f"- `{label}` fallback status by method: "
                            + ", ".join(fallback_status_by_method)
                        )
                    trace_level_counts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_trace_level_counts")
                    )
                    if trace_level_counts:
                        lines.append(
                            f"- `{label}` trace levels: "
                            + ", ".join(trace_level_counts)
                        )
                    constant_policy_counts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_constant_policy_counts")
                    )
                    if constant_policy_counts:
                        lines.append(
                            f"- `{label}` constant policies: "
                            + ", ".join(constant_policy_counts)
                        )
                    public_trace_level_counts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_public_trace_level_counts")
                    )
                    if public_trace_level_counts:
                        lines.append(
                            f"- `{label}` public trace levels: "
                            + ", ".join(public_trace_level_counts)
                        )
                    public_constant_policy_counts = (
                        calculus_matrix_count_map_fragments(
                            metrics.get("backend_public_constant_policy_counts")
                        )
                    )
                    if public_constant_policy_counts:
                        lines.append(
                            f"- `{label}` public constant policies: "
                            + ", ".join(public_constant_policy_counts)
                        )
                    fallback_trace_level_counts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_fallback_trace_level_counts")
                    )
                    if fallback_trace_level_counts:
                        lines.append(
                            f"- `{label}` fallback trace levels: "
                            + ", ".join(fallback_trace_level_counts)
                        )
                    fallback_constant_policy_counts = (
                        calculus_matrix_count_map_fragments(
                            metrics.get("backend_fallback_constant_policy_counts")
                        )
                    )
                    if fallback_constant_policy_counts:
                        lines.append(
                            f"- `{label}` fallback constant policies: "
                            + ", ".join(fallback_constant_policy_counts)
                        )
                    assumption_exprs = metrics.get("backend_assumption_exprs")
                    public_assumption_exprs = metrics.get(
                        "backend_public_assumption_exprs"
                    )
                    fallback_assumption_exprs = metrics.get(
                        "backend_fallback_assumption_exprs"
                    )
                    if all(
                        isinstance(value, int)
                        for value in (
                            assumption_exprs,
                            public_assumption_exprs,
                            fallback_assumption_exprs,
                        )
                    ):
                        lines.append(
                            f"- `{label}` assumption exprs: "
                            f"total={assumption_exprs}, "
                            f"public={public_assumption_exprs}, "
                            f"fallback={fallback_assumption_exprs}"
                        )
                    verification_evidence_counts = (
                        calculus_matrix_count_map_fragments(
                            metrics.get("backend_verification_evidence_counts")
                        )
                    )
                    if verification_evidence_counts:
                        lines.append(
                            f"- `{label}` verification evidence: "
                            + ", ".join(verification_evidence_counts)
                        )
                    public_verification_evidence_counts = (
                        calculus_matrix_count_map_fragments(
                            metrics.get("backend_public_verification_evidence_counts")
                        )
                    )
                    if public_verification_evidence_counts:
                        lines.append(
                            f"- `{label}` public verification evidence: "
                            + ", ".join(public_verification_evidence_counts)
                        )
                    fallback_verification_evidence_counts = (
                        calculus_matrix_count_map_fragments(
                            metrics.get("backend_fallback_verification_evidence_counts")
                        )
                    )
                    if fallback_verification_evidence_counts:
                        lines.append(
                            f"- `{label}` fallback verification evidence: "
                            + ", ".join(fallback_verification_evidence_counts)
                        )
                    verification_evidence_by_method = (
                        calculus_matrix_count_map_fragments(
                            metrics.get("backend_verification_evidence_by_method")
                        )
                    )
                    if verification_evidence_by_method:
                        lines.append(
                            f"- `{label}` verification evidence by method: "
                            + ", ".join(verification_evidence_by_method)
                        )
                    public_verification_evidence_by_method = (
                        calculus_matrix_count_map_fragments(
                            metrics.get(
                                "backend_public_verification_evidence_by_method"
                            )
                        )
                    )
                    if public_verification_evidence_by_method:
                        lines.append(
                            f"- `{label}` public verification evidence by method: "
                            + ", ".join(public_verification_evidence_by_method)
                        )
                    fallback_verification_evidence_by_method = (
                        calculus_matrix_count_map_fragments(
                            metrics.get(
                                "backend_fallback_verification_evidence_by_method"
                            )
                        )
                    )
                    if fallback_verification_evidence_by_method:
                        lines.append(
                            f"- `{label}` fallback verification evidence by method: "
                            + ", ".join(fallback_verification_evidence_by_method)
                        )
                    verification_normalization_reason_counts = (
                        calculus_matrix_count_map_fragments(
                            metrics.get(
                                "backend_verification_normalization_reason_counts"
                            )
                        )
                    )
                    if verification_normalization_reason_counts:
                        lines.append(
                            f"- `{label}` verification normalization reasons: "
                            + ", ".join(verification_normalization_reason_counts)
                        )
                    public_verification_normalization_reason_counts = (
                        calculus_matrix_count_map_fragments(
                            metrics.get(
                                "backend_public_verification_normalization_reason_counts"
                            )
                        )
                    )
                    if public_verification_normalization_reason_counts:
                        lines.append(
                            f"- `{label}` public verification normalization reasons: "
                            + ", ".join(
                                public_verification_normalization_reason_counts
                            )
                        )
                    fallback_verification_normalization_reason_counts = (
                        calculus_matrix_count_map_fragments(
                            metrics.get(
                                "backend_fallback_verification_normalization_reason_counts"
                            )
                        )
                    )
                    if fallback_verification_normalization_reason_counts:
                        lines.append(
                            f"- `{label}` fallback verification normalization reasons: "
                            + ", ".join(
                                fallback_verification_normalization_reason_counts
                            )
                        )
                    verification_normalization_reason_by_method = (
                        calculus_matrix_count_map_fragments(
                            metrics.get(
                                "backend_verification_normalization_reason_by_method"
                            )
                        )
                    )
                    if verification_normalization_reason_by_method:
                        lines.append(
                            f"- `{label}` verification normalization reason by method: "
                            + ", ".join(verification_normalization_reason_by_method)
                        )
                    public_verification_normalization_reason_by_method = (
                        calculus_matrix_count_map_fragments(
                            metrics.get(
                                "backend_public_verification_normalization_reason_by_method"
                            )
                        )
                    )
                    if public_verification_normalization_reason_by_method:
                        lines.append(
                            f"- `{label}` public verification normalization reason by method: "
                            + ", ".join(
                                public_verification_normalization_reason_by_method
                            )
                        )
                    fallback_verification_normalization_reason_by_method = (
                        calculus_matrix_count_map_fragments(
                            metrics.get(
                                "backend_fallback_verification_normalization_reason_by_method"
                            )
                        )
                    )
                    if fallback_verification_normalization_reason_by_method:
                        lines.append(
                            f"- `{label}` fallback verification normalization reason by method: "
                            + ", ".join(
                                fallback_verification_normalization_reason_by_method
                            )
                        )
                    verification_normalization_pass_count_counts = (
                        calculus_matrix_count_map_fragments(
                            metrics.get(
                                "backend_verification_normalization_pass_count_counts"
                            )
                        )
                    )
                    if verification_normalization_pass_count_counts:
                        lines.append(
                            f"- `{label}` verification normalization passes: "
                            f"max={metrics.get('backend_max_verification_normalization_passes', 0)}; "
                            + ", ".join(verification_normalization_pass_count_counts)
                        )
                    public_verification_normalization_pass_count_counts = (
                        calculus_matrix_count_map_fragments(
                            metrics.get(
                                "backend_public_verification_normalization_pass_count_counts"
                            )
                        )
                    )
                    if public_verification_normalization_pass_count_counts:
                        lines.append(
                            f"- `{label}` public verification normalization passes: "
                            f"max={metrics.get('backend_public_max_verification_normalization_passes', 0)}; "
                            + ", ".join(
                                public_verification_normalization_pass_count_counts
                            )
                        )
                    fallback_verification_normalization_pass_count_counts = (
                        calculus_matrix_count_map_fragments(
                            metrics.get(
                                "backend_fallback_verification_normalization_pass_count_counts"
                            )
                        )
                    )
                    if fallback_verification_normalization_pass_count_counts:
                        lines.append(
                            f"- `{label}` fallback verification normalization passes: "
                            f"max={metrics.get('backend_fallback_max_verification_normalization_passes', 0)}; "
                            + ", ".join(
                                fallback_verification_normalization_pass_count_counts
                            )
                        )
                    verification_normalization_pass_count_by_method = (
                        calculus_matrix_count_map_fragments(
                            metrics.get(
                                "backend_verification_normalization_pass_count_by_method"
                            )
                        )
                    )
                    if verification_normalization_pass_count_by_method:
                        lines.append(
                            f"- `{label}` verification normalization passes by method: "
                            + ", ".join(verification_normalization_pass_count_by_method)
                        )
                    required_condition_counts = calculus_matrix_count_map_fragments(
                        metrics.get("backend_required_condition_counts")
                    )
                    if required_condition_counts:
                        lines.append(
                            f"- `{label}` required conditions: "
                            + ", ".join(required_condition_counts)
                        )
                elif label == "integrate_command_matrix":
                    lines.extend(
                        calculus_runtime_lines(
                            label,
                            metrics,
                            prefix="integrate",
                            group_keys=(
                                "family",
                                "calculus_maturity_block",
                                "calculus_block_gate",
                                "trace_regime",
                            ),
                        )
                    )
                    for phase_label, distribution_key, rows_key, elapsed_key in (
                        (
                            "CLI simplify",
                            "integrate_cli_simplify_runtime_distribution",
                            "integrate_slowest_cli_simplify_evaluations",
                            "cli_simplify_elapsed_seconds",
                        ),
                        (
                            "CLI total",
                            "integrate_cli_total_runtime_distribution",
                            "integrate_slowest_cli_total_evaluations",
                            "cli_total_elapsed_seconds",
                        ),
                        (
                            "CLI public overhead",
                            "integrate_cli_public_overhead_runtime_distribution",
                            "integrate_slowest_cli_public_overhead_evaluations",
                            "cli_public_overhead_seconds",
                        ),
                    ):
                        phase_distribution = calculus_runtime_distribution_fragment(
                            metrics.get(distribution_key)
                        )
                        if phase_distribution:
                            lines.append(
                                f"- `{label}` {phase_label} runtime distribution: "
                                + phase_distribution
                            )
                        slowest_phase = phase_runtime_case_fragments(
                            metrics.get(rows_key),
                            elapsed_key=elapsed_key,
                        )
                        if slowest_phase:
                            lines.append(
                                f"- `{label}` slowest {phase_label} evaluations: "
                                + ", ".join(slowest_phase)
                            )
                    slowest_integrate = phase_runtime_case_fragments(
                        metrics.get("integrate_slowest_integrate_evaluations"),
                        elapsed_key="integrate_elapsed_seconds",
                    )
                    if slowest_integrate:
                        lines.append(
                            f"- `{label}` slowest integrate evaluations: "
                            + ", ".join(slowest_integrate)
                        )
                    slowest_antiderivative = phase_runtime_case_fragments(
                        metrics.get(
                            "integrate_slowest_antiderivative_verifications"
                        ),
                        elapsed_key="antiderivative_verification_elapsed_seconds",
                    )
                    if slowest_antiderivative:
                        lines.append(
                            f"- `{label}` slowest antiderivative verifications: "
                            + ", ".join(slowest_antiderivative)
                        )
                    largest_stdout = payload_case_fragments(
                        metrics.get("integrate_largest_stdout_payload_cases"),
                        metric_key="stdout_bytes",
                        unit="B",
                    )
                    if largest_stdout:
                        lines.append(
                            f"- `{label}` largest stdout payload cases: "
                            + ", ".join(largest_stdout)
                        )
                    largest_steps = payload_case_fragments(
                        metrics.get("integrate_largest_step_trace_cases"),
                        metric_key="step_text_char_count",
                        unit=" chars",
                    )
                    if largest_steps:
                        lines.append(
                            f"- `{label}` largest step trace cases: "
                            + ", ".join(largest_steps)
                        )
                    verification_modes = verification_mode_runtime_fragments(
                        metrics.get(
                            "integrate_runtime_by_antiderivative_verification_mode"
                        )
                    )
                    if verification_modes:
                        lines.append(
                            f"- `{label}` runtime by antiderivative verification mode: "
                            + ", ".join(verification_modes)
                        )
                    residual_cause_runtime = residual_cause_runtime_fragments(
                        metrics.get("integrate_runtime_by_residual_cause")
                    )
                    if residual_cause_runtime:
                        lines.append(
                            f"- `{label}` runtime by residual cause: "
                            + ", ".join(residual_cause_runtime)
                        )
                    residual_cause_family_runtime = residual_cause_runtime_fragments(
                        metrics.get("integrate_runtime_by_residual_cause_family"),
                        group_key="cause_family",
                    )
                    if residual_cause_family_runtime:
                        lines.append(
                            f"- `{label}` runtime by residual cause-family: "
                            + ", ".join(residual_cause_family_runtime)
                        )
                    residual_phase_by_cause = residual_public_phase_group_fragments(
                        metrics.get("integrate_residual_public_phase_by_cause")
                    )
                    if residual_phase_by_cause:
                        lines.append(
                            f"- `{label}` residual public phase by cause: "
                            + ", ".join(residual_phase_by_cause)
                        )
                    residual_phase_by_cause_family = (
                        residual_public_phase_group_fragments(
                            metrics.get(
                                "integrate_residual_public_phase_by_cause_family"
                            ),
                            group_key="cause_family",
                        )
                    )
                    if residual_phase_by_cause_family:
                        lines.append(
                            f"- `{label}` residual public phase by cause-family: "
                            + ", ".join(residual_phase_by_cause_family)
                        )
                    residual_phase_slowest = residual_public_phase_case_fragments(
                        metrics.get(
                            "integrate_residual_public_phase_slowest_cases"
                        )
                    )
                    if residual_phase_slowest:
                        lines.append(
                            f"- `{label}` residual public phase slowest cases: "
                            + ", ".join(residual_phase_slowest)
                        )
                    residual_shape_summary = (
                        residual_shape_orientation_summary_fragment(
                            metrics.get(
                                "integrate_residual_shape_orientation_summary"
                            )
                        )
                    )
                    if residual_shape_summary:
                        lines.append(
                            f"- `{label}` residual shape/orientation summary: "
                            + residual_shape_summary
                        )
                    residual_shape_orientation = (
                        residual_shape_orientation_probe_fragments(
                            metrics.get(
                                "integrate_residual_shape_orientation_probes"
                            )
                        )
                    )
                    if residual_shape_orientation:
                        lines.append(
                            f"- `{label}` residual shape/orientation probes: "
                            + ", ".join(residual_shape_orientation)
                        )
                argument_regimes = calculus_matrix_count_map_fragments(
                    metrics.get("diff_argument_regime_counts")
                )
                if argument_regimes:
                    lines.append(
                        f"- `{label}` argument regimes: " + ", ".join(argument_regimes)
                    )
                diff_domain_regimes = calculus_matrix_count_map_fragments(
                    metrics.get("diff_domain_regime_counts")
                )
                if diff_domain_regimes:
                    lines.append(
                        f"- `{label}` domain regimes: "
                        + ", ".join(diff_domain_regimes)
                    )
                diff_required_displays = calculus_matrix_count_map_fragments(
                    metrics.get("diff_required_display_counts")
                )
                if diff_required_displays:
                    lines.append(
                        f"- `{label}` required displays: "
                        + ", ".join(diff_required_displays)
                    )
                diff_outcome_counts = calculus_matrix_count_map_fragments(
                    metrics.get("diff_outcome_counts")
                )
                if diff_outcome_counts:
                    lines.append(
                        f"- `{label}` outcomes: " + ", ".join(diff_outcome_counts)
                    )
                diff_residual_cases = calculus_residual_case_fragments(
                    metrics.get("diff_residual_case_names")
                )
                if diff_residual_cases:
                    lines.append(
                        f"- `{label}` residual case IDs: "
                        + ", ".join(diff_residual_cases)
                    )
                for metric_key, regime_label in (
                    (
                        "diff_calculus_maturity_block_counts",
                        "calculus maturity blocks",
                    ),
                    ("diff_calculus_block_gate_counts", "calculus block gates"),
                ):
                    fragments = calculus_matrix_count_map_fragments(
                        metrics.get(metric_key)
                    )
                    if fragments:
                        lines.append(
                            f"- `{label}` {regime_label}: "
                            + ", ".join(fragments)
                        )
                diff_symbolic_radius_clusters = calculus_matrix_count_map_fragments(
                    metrics.get("diff_symbolic_radius_policy_cluster_counts")
                )
                if diff_symbolic_radius_clusters:
                    lines.append(
                        f"- `{label}` symbolic-radius policy clusters: "
                        + ", ".join(diff_symbolic_radius_clusters)
                    )
                diff_symbolic_radius_consolidated = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "diff_symbolic_radius_consolidated_policy_cluster_counts"
                        )
                    )
                )
                if diff_symbolic_radius_consolidated:
                    lines.append(
                        f"- `{label}` symbolic-radius consolidated policy clusters: "
                        + ", ".join(diff_symbolic_radius_consolidated)
                    )
                diff_positive_quadratic_clusters = (
                    calculus_matrix_count_map_fragments(
                        metrics.get("diff_positive_quadratic_policy_cluster_counts")
                    )
                )
                if diff_positive_quadratic_clusters:
                    lines.append(
                        f"- `{label}` positive-quadratic policy clusters: "
                        + ", ".join(diff_positive_quadratic_clusters)
                    )
                diff_positive_quadratic_consolidated = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "diff_positive_quadratic_consolidated_policy_cluster_counts"
                        )
                    )
                )
                if diff_positive_quadratic_consolidated:
                    lines.append(
                        f"- `{label}` positive-quadratic consolidated policy clusters: "
                        + ", ".join(diff_positive_quadratic_consolidated)
                    )
                diff_positive_quadratic_consolidation_candidates = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "diff_positive_quadratic_consolidation_candidate_counts"
                        )
                    )
                )
                if diff_positive_quadratic_consolidation_candidates:
                    lines.append(
                        f"- `{label}` positive-quadratic consolidation candidates: "
                        + ", ".join(diff_positive_quadratic_consolidation_candidates)
                    )
                diff_positive_quadratic_runtime_candidates = (
                    runtime_cluster_candidate_fragments(
                        metrics.get(
                            "diff_positive_quadratic_runtime_candidate_clusters"
                        )
                    )
                )
                if diff_positive_quadratic_runtime_candidates:
                    lines.append(
                        f"- `{label}` positive-quadratic runtime candidate clusters: "
                        + ", ".join(diff_positive_quadratic_runtime_candidates)
                    )
                diff_variable_power_clusters = calculus_matrix_count_map_fragments(
                    metrics.get("diff_variable_power_policy_cluster_counts")
                )
                if diff_variable_power_clusters:
                    lines.append(
                        f"- `{label}` variable-power policy clusters: "
                        + ", ".join(diff_variable_power_clusters)
                    )
                diff_variable_power_runtime_candidates = (
                    runtime_cluster_candidate_fragments(
                        metrics.get("diff_variable_power_runtime_candidate_clusters")
                    )
                )
                if diff_variable_power_runtime_candidates:
                    lines.append(
                        f"- `{label}` variable-power runtime candidate clusters: "
                        + ", ".join(diff_variable_power_runtime_candidates)
                    )
                for metric_key, regime_label in (
                    ("diff_trace_regime_counts", "trace regimes"),
                    ("diff_presentation_regime_counts", "presentation regimes"),
                ):
                    fragments = calculus_matrix_count_map_fragments(
                        metrics.get(metric_key)
                    )
                    if fragments:
                        lines.append(
                            f"- `{label}` {regime_label}: "
                            + ", ".join(fragments)
                        )
                integrate_argument_regimes = calculus_matrix_count_map_fragments(
                    metrics.get("integrate_argument_regime_counts")
                )
                if integrate_argument_regimes:
                    lines.append(
                        f"- `{label}` argument regimes: "
                        + ", ".join(integrate_argument_regimes)
                    )
                integrate_domain_regimes = calculus_matrix_count_map_fragments(
                    metrics.get("integrate_domain_regime_counts")
                )
                if integrate_domain_regimes:
                    lines.append(
                        f"- `{label}` domain regimes: "
                        + ", ".join(integrate_domain_regimes)
                    )
                integrate_required_displays = calculus_matrix_count_map_fragments(
                    metrics.get("integrate_required_display_counts")
                )
                if integrate_required_displays:
                    lines.append(
                        f"- `{label}` required displays: "
                        + ", ".join(integrate_required_displays)
                    )
                integrate_outcome_counts = calculus_matrix_count_map_fragments(
                    metrics.get("integrate_outcome_counts")
                )
                if integrate_outcome_counts:
                    lines.append(
                        f"- `{label}` outcomes: "
                        + ", ".join(integrate_outcome_counts)
                    )
                integrate_residual_cases = calculus_residual_case_fragments(
                    metrics.get("integrate_residual_case_names")
                )
                if integrate_residual_cases:
                    lines.append(
                        f"- `{label}` residual case IDs: "
                        + ", ".join(integrate_residual_cases)
                    )
                integrate_residual_causes = calculus_matrix_count_map_fragments(
                    metrics.get("integrate_residual_cause_counts")
                )
                if integrate_residual_causes:
                    lines.append(
                        f"- `{label}` residual causes: "
                        + ", ".join(integrate_residual_causes)
                    )
                integrate_residual_families = calculus_matrix_count_map_fragments(
                    metrics.get("integrate_residual_family_counts")
                )
                if integrate_residual_families:
                    lines.append(
                        f"- `{label}` residual families: "
                        + ", ".join(integrate_residual_families)
                    )
                integrate_residual_cause_families = (
                    calculus_matrix_count_map_fragments(
                        metrics.get("integrate_residual_cause_family_counts")
                    )
                )
                if integrate_residual_cause_families:
                    lines.append(
                        f"- `{label}` residual cause-family buckets: "
                        + ", ".join(integrate_residual_cause_families)
                    )
                integrate_residual_cases_by_cause = (
                    calculus_residual_cases_by_cause_fragments(
                        metrics.get("integrate_residual_cases_by_cause")
                    )
                )
                if integrate_residual_cases_by_cause:
                    lines.append(
                        f"- `{label}` residual examples by cause: "
                        + "; ".join(integrate_residual_cases_by_cause)
                    )
                integrate_verification_regimes = calculus_matrix_count_map_fragments(
                    metrics.get("integrate_verification_regime_counts")
                )
                if integrate_verification_regimes:
                    lines.append(
                        f"- `{label}` verification regimes: "
                        + ", ".join(integrate_verification_regimes)
                    )
                for metric_key, regime_label in (
                    (
                        "integrate_calculus_maturity_block_counts",
                        "calculus maturity blocks",
                    ),
                    ("integrate_calculus_block_gate_counts", "calculus block gates"),
                ):
                    fragments = calculus_matrix_count_map_fragments(
                        metrics.get(metric_key)
                    )
                    if fragments:
                        lines.append(
                            f"- `{label}` {regime_label}: "
                            + ", ".join(fragments)
                        )
                integrate_policy_clusters = calculus_matrix_count_map_fragments(
                    metrics.get("integrate_trig_hyperbolic_policy_cluster_counts")
                )
                if integrate_policy_clusters:
                    lines.append(
                        f"- `{label}` trig/hyperbolic policy clusters: "
                        + ", ".join(integrate_policy_clusters)
                    )
                integrate_consolidated_policy_clusters = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_trig_hyperbolic_consolidated_policy_cluster_counts"
                        )
                    )
                )
                if integrate_consolidated_policy_clusters:
                    lines.append(
                        f"- `{label}` trig/hyperbolic consolidated policy clusters: "
                        + ", ".join(integrate_consolidated_policy_clusters)
                    )
                integrate_consolidation_candidates = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_trig_hyperbolic_consolidation_candidate_counts"
                        )
                    )
                )
                if integrate_consolidation_candidates:
                    lines.append(
                        f"- `{label}` trig/hyperbolic consolidation candidates: "
                        + ", ".join(integrate_consolidation_candidates)
                    )
                base_integration_policy_clusters = calculus_matrix_count_map_fragments(
                    metrics.get("integrate_base_integration_policy_cluster_counts")
                )
                if base_integration_policy_clusters:
                    lines.append(
                        f"- `{label}` base integration policy clusters: "
                        + ", ".join(base_integration_policy_clusters)
                    )
                base_integration_consolidated_policy_clusters = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_base_integration_consolidated_policy_cluster_counts"
                        )
                    )
                )
                if base_integration_consolidated_policy_clusters:
                    lines.append(
                        f"- `{label}` base integration consolidated policy clusters: "
                        + ", ".join(base_integration_consolidated_policy_clusters)
                    )
                base_integration_consolidation_candidates = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_base_integration_consolidation_candidate_counts"
                        )
                    )
                )
                if base_integration_consolidation_candidates:
                    lines.append(
                        f"- `{label}` base integration consolidation candidates: "
                        + ", ".join(base_integration_consolidation_candidates)
                    )
                direct_block_counts = calculus_matrix_count_map_fragments(
                    metrics.get(
                        "integrate_direct_diff_integrate_calculus_maturity_block_counts"
                    )
                )
                if direct_block_counts:
                    lines.append(
                        f"- `{label}` direct diff(integrate) calculus maturity blocks: "
                        + ", ".join(direct_block_counts)
                    )
                direct_gate_counts = calculus_matrix_count_map_fragments(
                    metrics.get(
                        "integrate_direct_diff_integrate_calculus_block_gate_counts"
                    )
                )
                if direct_gate_counts:
                    lines.append(
                        f"- `{label}` direct diff(integrate) calculus block gates: "
                        + ", ".join(direct_gate_counts)
                    )
                direct_trig_hyperbolic_clusters = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_direct_diff_integrate_trig_hyperbolic_policy_cluster_counts"
                        )
                    )
                )
                if direct_trig_hyperbolic_clusters:
                    lines.append(
                        f"- `{label}` direct diff(integrate) trig/hyperbolic policy clusters: "
                        + ", ".join(direct_trig_hyperbolic_clusters)
                    )
                direct_trig_hyperbolic_shared_clusters = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_direct_diff_integrate_trig_hyperbolic_shared_policy_cluster_counts"
                        )
                    )
                )
                if direct_trig_hyperbolic_shared_clusters:
                    lines.append(
                        f"- `{label}` direct diff(integrate) trig/hyperbolic shared-policy clusters: "
                        + ", ".join(direct_trig_hyperbolic_shared_clusters)
                    )
                direct_base_integration_clusters = calculus_matrix_count_map_fragments(
                    metrics.get(
                        "integrate_direct_diff_integrate_base_integration_policy_cluster_counts"
                    )
                )
                if direct_base_integration_clusters:
                    lines.append(
                        f"- `{label}` direct diff(integrate) base integration policy clusters: "
                        + ", ".join(direct_base_integration_clusters)
                    )
                direct_base_integration_shared_clusters = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_direct_diff_integrate_base_integration_shared_policy_cluster_counts"
                        )
                    )
                )
                if direct_base_integration_shared_clusters:
                    lines.append(
                        f"- `{label}` direct diff(integrate) base integration shared-policy clusters: "
                        + ", ".join(direct_base_integration_shared_clusters)
                    )
                direct_cases = metrics.get("integrate_direct_diff_integrate_case_count")
                direct_exact_cases = metrics.get(
                    "integrate_direct_diff_integrate_exact_case_count"
                )
                direct_equiv_cases = metrics.get(
                    "integrate_direct_diff_integrate_equivalence_case_count"
                )
                if (
                    isinstance(direct_cases, int)
                    and isinstance(direct_exact_cases, int)
                    and isinstance(direct_equiv_cases, int)
                    and direct_cases > 0
                    and direct_exact_cases == direct_cases
                    and direct_equiv_cases == 0
                ):
                    lines.append(
                        f"- `{label}` direct diff(integrate) exactness: "
                        f"all {direct_cases} checks exact; no equivalence-backed fallback"
                    )
                direct_gap_block_counts = calculus_matrix_count_map_fragments(
                    metrics.get(
                        "integrate_direct_diff_integrate_gap_calculus_maturity_block_counts"
                    )
                )
                if direct_gap_block_counts:
                    lines.append(
                        f"- `{label}` direct diff(integrate) gaps by calculus maturity block: "
                        + ", ".join(direct_gap_block_counts)
                    )
                direct_gap_block_examples = calculus_residual_cases_by_cause_fragments(
                    metrics.get(
                        "integrate_direct_diff_integrate_gap_cases_by_calculus_maturity_block"
                    )
                )
                if direct_gap_block_examples:
                    lines.append(
                        f"- `{label}` direct diff(integrate) gap examples by calculus maturity block: "
                        + "; ".join(direct_gap_block_examples)
                    )
                direct_gap_gate_counts = calculus_matrix_count_map_fragments(
                    metrics.get(
                        "integrate_direct_diff_integrate_gap_calculus_block_gate_counts"
                    )
                )
                if direct_gap_gate_counts:
                    lines.append(
                        f"- `{label}` direct diff(integrate) gaps by calculus block gate: "
                        + ", ".join(direct_gap_gate_counts)
                    )
                direct_gap_gate_examples = calculus_residual_cases_by_cause_fragments(
                    metrics.get(
                        "integrate_direct_diff_integrate_gap_cases_by_calculus_block_gate"
                    )
                )
                if direct_gap_gate_examples:
                    lines.append(
                        f"- `{label}` direct diff(integrate) gap examples by calculus block gate: "
                        + "; ".join(direct_gap_gate_examples)
                    )
                direct_gap_trig_hyperbolic_clusters = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_direct_diff_integrate_gap_trig_hyperbolic_policy_cluster_counts"
                        )
                    )
                )
                if direct_gap_trig_hyperbolic_clusters:
                    lines.append(
                        f"- `{label}` direct diff(integrate) gaps by trig/hyperbolic policy cluster: "
                        + ", ".join(direct_gap_trig_hyperbolic_clusters)
                    )
                direct_gap_trig_hyperbolic_examples = (
                    calculus_residual_cases_by_cause_fragments(
                        metrics.get(
                            "integrate_direct_diff_integrate_gap_cases_by_trig_hyperbolic_policy_cluster"
                        )
                    )
                )
                if direct_gap_trig_hyperbolic_examples:
                    lines.append(
                        f"- `{label}` direct diff(integrate) gap examples by trig/hyperbolic policy cluster: "
                        + "; ".join(direct_gap_trig_hyperbolic_examples)
                    )
                direct_gap_base_integration_clusters = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_direct_diff_integrate_gap_base_integration_policy_cluster_counts"
                        )
                    )
                )
                if direct_gap_base_integration_clusters:
                    lines.append(
                        f"- `{label}` direct diff(integrate) gaps by base integration policy cluster: "
                        + ", ".join(direct_gap_base_integration_clusters)
                    )
                direct_gap_base_integration_examples = (
                    calculus_residual_cases_by_cause_fragments(
                        metrics.get(
                            "integrate_direct_diff_integrate_gap_cases_by_base_integration_policy_cluster"
                        )
                    )
                )
                if direct_gap_base_integration_examples:
                    lines.append(
                        f"- `{label}` direct diff(integrate) gap examples by base integration policy cluster: "
                        + "; ".join(direct_gap_base_integration_examples)
                    )
                direct_gap_radical_inverse_clusters = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_direct_diff_integrate_gap_radical_inverse_policy_cluster_counts"
                        )
                    )
                )
                if direct_gap_radical_inverse_clusters:
                    lines.append(
                        f"- `{label}` direct diff(integrate) gaps by radical/inverse policy cluster: "
                        + ", ".join(direct_gap_radical_inverse_clusters)
                    )
                direct_gap_radical_inverse_examples = (
                    calculus_residual_cases_by_cause_fragments(
                        metrics.get(
                            "integrate_direct_diff_integrate_gap_cases_by_radical_inverse_policy_cluster"
                        )
                    )
                )
                if direct_gap_radical_inverse_examples:
                    lines.append(
                        f"- `{label}` direct diff(integrate) gap examples by radical/inverse policy cluster: "
                        + "; ".join(direct_gap_radical_inverse_examples)
                    )
                direct_equiv_block_counts = calculus_matrix_count_map_fragments(
                    metrics.get(
                        "integrate_direct_diff_integrate_equivalence_calculus_maturity_block_counts"
                    )
                )
                if direct_equiv_block_counts:
                    lines.append(
                        f"- `{label}` equivalence-backed diff(integrate) calculus maturity blocks: "
                        + ", ".join(direct_equiv_block_counts)
                    )
                direct_equiv_base_integration_clusters = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_direct_diff_integrate_equivalence_base_integration_policy_cluster_counts"
                        )
                    )
                )
                if direct_equiv_base_integration_clusters:
                    lines.append(
                        f"- `{label}` equivalence-backed diff(integrate) base integration policy clusters: "
                        + ", ".join(direct_equiv_base_integration_clusters)
                    )
                direct_equiv_trig_hyperbolic_clusters = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_direct_diff_integrate_equivalence_trig_hyperbolic_policy_cluster_counts"
                        )
                    )
                )
                if direct_equiv_trig_hyperbolic_clusters:
                    lines.append(
                        f"- `{label}` equivalence-backed diff(integrate) trig/hyperbolic policy clusters: "
                        + ", ".join(direct_equiv_trig_hyperbolic_clusters)
                    )
                radical_inverse_policy_clusters = calculus_matrix_count_map_fragments(
                    metrics.get("integrate_radical_inverse_policy_cluster_counts")
                )
                if radical_inverse_policy_clusters:
                    lines.append(
                        f"- `{label}` radical/inverse policy clusters: "
                        + ", ".join(radical_inverse_policy_clusters)
                    )
                radical_inverse_consolidated_policy_clusters = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_radical_inverse_consolidated_policy_cluster_counts"
                        )
                    )
                )
                if radical_inverse_consolidated_policy_clusters:
                    lines.append(
                        f"- `{label}` radical/inverse consolidated policy clusters: "
                        + ", ".join(radical_inverse_consolidated_policy_clusters)
                    )
                radical_inverse_consolidation_candidates = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_radical_inverse_consolidation_candidate_counts"
                        )
                    )
                )
                if radical_inverse_consolidation_candidates:
                    lines.append(
                        f"- `{label}` radical/inverse consolidation candidates: "
                        + ", ".join(radical_inverse_consolidation_candidates)
                    )
                direct_radical_inverse_clusters = calculus_matrix_count_map_fragments(
                    metrics.get(
                        "integrate_direct_diff_integrate_radical_inverse_policy_cluster_counts"
                    )
                )
                if direct_radical_inverse_clusters:
                    lines.append(
                        f"- `{label}` direct diff(integrate) radical/inverse policy clusters: "
                        + ", ".join(direct_radical_inverse_clusters)
                    )
                direct_radical_inverse_shared_clusters = (
                    calculus_matrix_count_map_fragments(
                        metrics.get(
                            "integrate_direct_diff_integrate_radical_inverse_shared_policy_cluster_counts"
                        )
                    )
                )
                if direct_radical_inverse_shared_clusters:
                    lines.append(
                        f"- `{label}` direct diff(integrate) radical/inverse shared-policy clusters: "
                        + ", ".join(direct_radical_inverse_shared_clusters)
                    )
                for metric_key, regime_label in (
                    ("integrate_trace_regime_counts", "trace regimes"),
                    (
                        "integrate_presentation_regime_counts",
                        "presentation regimes",
                    ),
                ):
                    fragments = calculus_matrix_count_map_fragments(
                        metrics.get(metric_key)
                    )
                    if fragments:
                        lines.append(
                            f"- `{label}` {regime_label}: "
                            + ", ".join(fragments)
                        )
                point_regimes = calculus_matrix_count_map_fragments(
                    metrics.get("limit_point_regime_counts")
                )
                if point_regimes:
                    lines.append(
                        f"- `{label}` point regimes: " + ", ".join(point_regimes)
                    )
                domain_regimes = calculus_matrix_count_map_fragments(
                    metrics.get("limit_domain_regime_counts")
                )
                if domain_regimes:
                    lines.append(
                        f"- `{label}` domain regimes: " + ", ".join(domain_regimes)
                    )
                required_condition_regimes = calculus_matrix_count_map_fragments(
                    metrics.get("limit_required_condition_regime_counts")
                )
                if required_condition_regimes:
                    lines.append(
                        f"- `{label}` required condition regimes: "
                        + ", ".join(required_condition_regimes)
                    )
                required_displays = calculus_matrix_count_map_fragments(
                    metrics.get("limit_required_display_counts")
                )
                if required_displays:
                    lines.append(
                        f"- `{label}` required displays: "
                        + ", ".join(required_displays)
                    )
                outcome_counts = calculus_matrix_count_map_fragments(
                    metrics.get("limit_outcome_counts")
                )
                if outcome_counts:
                    lines.append(
                        f"- `{label}` outcomes: " + ", ".join(outcome_counts)
                    )
                residual_cases = calculus_residual_case_fragments(
                    metrics.get("limit_residual_case_names")
                )
                if residual_cases:
                    lines.append(
                        f"- `{label}` residual case IDs: "
                        + ", ".join(residual_cases)
                    )
                residual_causes = calculus_matrix_count_map_fragments(
                    metrics.get("limit_residual_cause_counts")
                )
                if residual_causes:
                    lines.append(
                        f"- `{label}` residual causes: "
                        + ", ".join(residual_causes)
                    )
                residual_families = calculus_matrix_count_map_fragments(
                    metrics.get("limit_residual_family_counts")
                )
                if residual_families:
                    lines.append(
                        f"- `{label}` residual families: "
                        + ", ".join(residual_families)
                    )
                residual_cause_families = calculus_matrix_count_map_fragments(
                    metrics.get("limit_residual_cause_family_counts")
                )
                if residual_cause_families:
                    lines.append(
                        f"- `{label}` residual cause-family buckets: "
                        + ", ".join(residual_cause_families)
                    )
                residual_cases_by_cause = (
                    calculus_residual_cases_by_cause_fragments(
                        metrics.get("limit_residual_cases_by_cause")
                    )
                )
                if residual_cases_by_cause:
                    lines.append(
                        f"- `{label}` residual examples by cause: "
                        + "; ".join(residual_cases_by_cause)
                    )
                for metric_key, regime_label in (
                    (
                        "limit_calculus_maturity_block_counts",
                        "calculus maturity blocks",
                    ),
                    ("limit_calculus_block_gate_counts", "calculus block gates"),
                ):
                    fragments = calculus_matrix_count_map_fragments(
                        metrics.get(metric_key)
                    )
                    if fragments:
                        lines.append(
                            f"- `{label}` {regime_label}: "
                            + ", ".join(fragments)
                        )
                for metric_key, regime_label in (
                    ("limit_trace_regime_counts", "trace regimes"),
                    ("limit_presentation_regime_counts", "presentation regimes"),
                ):
                    fragments = calculus_matrix_count_map_fragments(
                        metrics.get(metric_key)
                    )
                    if fragments:
                        lines.append(
                            f"- `{label}` {regime_label}: "
                            + ", ".join(fragments)
                        )
                sparse_conditions = sparse_expected_condition_fragments(metrics)
                if sparse_conditions:
                    lines.append(
                        f"- `{label}` sparse expected conditions: "
                        + ", ".join(sparse_conditions)
                    )
                domain_conditions = domain_expected_condition_fragments(metrics)
                if domain_conditions:
                    lines.append(
                        f"- `{label}` domain expected conditions: "
                        + ", ".join(domain_conditions)
                    )
                expected_wrapped_cases = metrics.get(
                    "matrix_expected_wrapped_case_count"
                )
                missing_wrapped_pairs = metrics.get(
                    "matrix_missing_wrapped_pair_count"
                )
                full_wrapper_bases = metrics.get("matrix_full_wrapper_base_count")
                partial_wrapper_bases = metrics.get(
                    "matrix_partial_wrapper_base_count"
                )
                largest_wrapper_gap = metrics.get("matrix_largest_wrapper_gap_count")
                if all(
                    isinstance(value, int)
                    for value in (
                        expected_wrapped_cases,
                        missing_wrapped_pairs,
                        full_wrapper_bases,
                        partial_wrapper_bases,
                        largest_wrapper_gap,
                    )
                ):
                    lines.append(
                        f"- `{label}` wrapper coverage: "
                        f"expected_wrapped_cases={expected_wrapped_cases} "
                        f"missing_wrapped_pairs={missing_wrapped_pairs} "
                        f"full_wrapper_bases={full_wrapper_bases} "
                        f"partial_wrapper_bases={partial_wrapper_bases} "
                        f"largest_gap={largest_wrapper_gap}"
                    )
                wrapper_gap_fragments = matrix_wrapper_gap_fragments(metrics)
                if wrapper_gap_fragments:
                    lines.append(
                        f"- `{label}` largest wrapper gaps: "
                        + ", ".join(wrapper_gap_fragments)
                    )
                problem_case_count = metrics.get("problem_case_count", 0)
                if isinstance(problem_case_count, int) and problem_case_count > 0:
                    problem_fragments = []
                    raw_problem_cases = metrics.get("problem_cases", [])
                    problem_cases = (
                        raw_problem_cases
                        if isinstance(raw_problem_cases, list)
                        else []
                    )
                    for case in problem_cases[:3]:
                        if not isinstance(case, dict):
                            continue
                        name = case.get("name")
                        status = case.get("status")
                        if not isinstance(name, str) or not isinstance(status, str):
                            continue
                        fragment = f"{name} status={status}"
                        error_kind = case.get("error_kind")
                        if isinstance(error_kind, str):
                            fragment += f" kind={error_kind}"
                        problem_fragments.append(fragment)
                    if problem_fragments:
                        lines.append(
                            f"- `{label}` problem cases: "
                            + "; ".join(problem_fragments)
                        )
                    else:
                        lines.append(
                            f"- `{label}` problem cases: count={problem_case_count}"
                        )
            else:
                lines.append(
                    (
                        f"- `{label}`: passed={metrics['passed']} "
                        f"failed={metrics['failed']} ignored={metrics['ignored']} "
                        f"filtered_out={metrics['filtered_out']}"
                    )
                )
                ignored_tests = metrics.get("ignored_tests")
                if ignored_tests:
                    lines.append(
                        f"- `{label}` ignored tests: {format_ignored_tests(ignored_tests)}"
                    )
        lines.append("")

    strict_suite = scorecard["suites"].get("simplify_strict")
    if strict_suite:
        strict_metrics = strict_suite["metrics"]
        proof_shape_mix = strict_metrics.get("proof_shape_mix")
        proved_breakdown = strict_metrics.get("proved_breakdown")
        if "symbolic_closure_rate_percent" in strict_metrics:
            lines.extend(
                [
                    "## Simplify Closure Signal",
                    "",
                    "- Dimension: normal-form convergence vs symbolic residual proof in the unified simplification benchmark.",
                    (
                        "- Closure: "
                        f"symbolic={strict_metrics['symbolic_closure']}/{strict_metrics['effective_combos']} "
                        f"({strict_metrics['symbolic_closure_rate_percent']:.1f}%), "
                        f"NF={strict_metrics['nf_convergent']} "
                        f"({strict_metrics['nf_rate_percent']:.1f}%), "
                        f"proved-only={strict_metrics['proved_symbolic']} "
                        f"({strict_metrics['proved_symbolic_rate_percent']:.1f}%)."
                    ),
                    (
                        "- Residual outcomes: "
                        f"numeric_only={strict_metrics['numeric_only']} "
                        f"({strict_metrics['numeric_only_rate_percent']:.1f}%), "
                        f"inconclusive={strict_metrics['inconclusive']} "
                        f"({strict_metrics['inconclusive_rate_percent']:.1f}%), "
                        f"timeouts={strict_metrics['timeouts']}."
                    ),
                    (
                        "- NF gap: "
                        f"{strict_metrics['normalization_gap']} cases are proved symbolically "
                        "but do not converge to the same normal form."
                    ),
                    "",
                ]
            )
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
                    fragment += (
                        f" elapsed={format_runtime_duration(row['elapsed_seconds'])}"
                    )
                if "avg_case_ms" in row:
                    fragment += f" avg_case_ms={row['avg_case_ms']:.2f}"
                if "simplify_elapsed_seconds" in row:
                    fragment += (
                        " simplify="
                        f"{format_runtime_duration(row['simplify_elapsed_seconds'])}"
                    )
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
                    f"{composition} simplify="
                    f"{format_runtime_duration(row['simplify_elapsed_seconds'])}"
                )
                if "avg_simplify_ms" in row:
                    fragment += f" avg_simplify_ms={row['avg_simplify_ms']:.2f}"
                if "elapsed_seconds" in row:
                    fragment += (
                        f" wall={format_runtime_duration(row['elapsed_seconds'])}"
                    )
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
                    fragment += (
                        f" elapsed={format_runtime_duration(row['elapsed_seconds'])}"
                    )
                if "avg_case_ms" in row:
                    fragment += f" avg_case_ms={row['avg_case_ms']:.2f}"
                if "avg_simplify_ms" in row:
                    fragment += f" avg_simplify_ms={row['avg_simplify_ms']:.2f}"
                window_fragments.append(fragment)
            lines.append(f"- Window slices: {', '.join(window_fragments)}")
        steady_engine_heavy_rows = mixed_metrics.get("steady_engine_heavy_rows")
        if isinstance(steady_engine_heavy_rows, list) and steady_engine_heavy_rows:
            steady_fragments = []
            expression_fragments = []
            for row in steady_engine_heavy_rows[:5]:
                label = f"#{row['case_number']} {row['composition']}"
                if row.get("window_label"):
                    label = f"{row['window_label']} {label}"
                fragment = (
                    f"{label} runs={row['runs']} "
                    f"median_simplify={format_runtime_duration(row['median_simplify_seconds'])} "
                    f"median_wire={format_runtime_duration(row['median_wire_seconds'])} "
                    f"median_wall={format_runtime_duration(row['median_elapsed_seconds'])}"
                )
                steady_fragments.append(fragment)
                expression = row.get("expression")
                if isinstance(expression, str) and len(expression_fragments) < 3:
                    expression_fragments.append(
                        f"{label} expr={compact_pressure_expression(expression)}"
                    )
            lines.append(
                "- Steady-state engine reruns: " + ", ".join(steady_fragments)
            )
            if expression_fragments:
                lines.append(
                    "- Steady-state dominant expressions: "
                    + ", ".join(expression_fragments)
                )
        lines.append("")

        orchestrator_profile = mixed_metrics.get("orchestrator_profile")
        if isinstance(orchestrator_profile, dict):
            profile_slice = mixed_metrics.get("orchestrator_profile_slice")
            totals = orchestrator_profile["totals"]
            lines.extend(
                [
                    "## Mixed Zero Orchestrator Profile",
                    "",
                    "- Purpose: identify hot shortcut groups and expensive no-match traffic under mixed zero pressure.",
                ]
            )
            if isinstance(profile_slice, dict):
                profile_filter = profile_slice.get("filter", "unknown")
                profile_limit = profile_slice.get("limit")
                limit_fragment = (
                    f" (limit {profile_limit})"
                    if isinstance(profile_limit, int)
                    else ""
                )
                lines.append(
                    (
                        "- Profiled slice: "
                        f"{profile_slice['total_cases']} cases"
                        f"{limit_fragment}, "
                        f"{profile_slice['elapsed_seconds']:.2f}s elapsed, "
                        f"filter `{profile_filter}`."
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
            for idx, row in enumerate(
                orchestrator_profile["top_hot_sections"][:3], start=1
            ):
                sample_suffix = orchestrator_profile_sample_suffix(row)
                lines.append(
                    (
                        f"- Hot {idx}: `{row['section']}` "
                        f"{row['total_ms']:.3f}ms over "
                        f"{row['attempts']} attempts "
                        f"(hits {row['hits']}, misses {row['misses']})"
                        f"{sample_suffix}"
                    )
                )
            no_match_sections = orchestrator_profile["top_no_match_cost_sections"][:3]
            if no_match_sections:
                for idx, row in enumerate(no_match_sections, start=1):
                    sample_suffix = orchestrator_profile_sample_suffix(
                        row, prefer_miss=True
                    )
                    lines.append(
                        (
                            f"- No-match hotspot {idx}: `{row['section']}` "
                            f"{row['total_ms']:.3f}ms "
                            f"(misses {row['misses']} of {row['attempts']})"
                            f"{sample_suffix}"
                        )
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
            if "reported_elapsed_per_case_ms" in metrics:
                pieces.append(
                    f"avg_case={metrics['reported_elapsed_per_case_ms']:.3f}ms"
                )
            if "expected_required_condition_case_count" in metrics:
                pieces.append(
                    f"conditioned={metrics['expected_required_condition_case_count']}"
                )
            if "distinct_expected_required_conditions" in metrics:
                pieces.append(
                    f"conditions={metrics['distinct_expected_required_conditions']}"
                )
            if "matrix_base_count" in metrics:
                pieces.append(f"total_bases={metrics['matrix_base_count']}")
            if "matrix_wrapped_base_count" in metrics:
                pieces.append(f"wrapped_bases={metrics['matrix_wrapped_base_count']}")
            if "matrix_standalone_base_count" in metrics:
                pieces.append(
                    f"standalone_bases={metrics['matrix_standalone_base_count']}"
                )
            if "matrix_wrapper_count" in metrics:
                pieces.append(f"wrappers={metrics['matrix_wrapper_count']}")
            if "matrix_missing_wrapped_pair_count" in metrics:
                pieces.append(
                    f"missing_wrapped_pairs={metrics['matrix_missing_wrapped_pair_count']}"
                )
            if "matrix_partial_wrapper_base_count" in metrics:
                pieces.append(
                    f"partial_wrapper_bases={metrics['matrix_partial_wrapper_base_count']}"
                )
            if "diff_supported_case_count" in metrics:
                pieces.append(f"supported={metrics['diff_supported_case_count']}")
            if "diff_residual_case_count" in metrics:
                pieces.append(f"residual={metrics['diff_residual_case_count']}")
            if "diff_warning_expected_case_count" in metrics:
                pieces.append(
                    f"warning_expected={metrics['diff_warning_expected_case_count']}"
                )
            if "diff_required_display_case_count" in metrics:
                pieces.append(
                    f"required_display={metrics['diff_required_display_case_count']}"
                )
            if "diff_step_checked_case_count" in metrics:
                pieces.append(f"step_checked={metrics['diff_step_checked_case_count']}")
            if "diff_supported_step_unchecked_case_count" in metrics:
                pieces.append(
                    "unchecked_supported_steps="
                    f"{metrics['diff_supported_step_unchecked_case_count']}"
                )
            if "diff_family_count" in metrics:
                pieces.append(f"families={metrics['diff_family_count']}")
            if "limit_supported_case_count" in metrics:
                pieces.append(f"supported={metrics['limit_supported_case_count']}")
            if "limit_residual_case_count" in metrics:
                pieces.append(f"residual={metrics['limit_residual_case_count']}")
            if "limit_warning_expected_case_count" in metrics:
                pieces.append(
                    f"warning_expected={metrics['limit_warning_expected_case_count']}"
                )
            if "limit_required_display_case_count" in metrics:
                pieces.append(
                    f"required_display={metrics['limit_required_display_case_count']}"
                )
            if "limit_step_checked_case_count" in metrics:
                pieces.append(f"step_checked={metrics['limit_step_checked_case_count']}")
            if "limit_supported_step_unchecked_case_count" in metrics:
                pieces.append(
                    "unchecked_supported_steps="
                    f"{metrics['limit_supported_step_unchecked_case_count']}"
                )
            if "limit_family_count" in metrics:
                pieces.append(f"families={metrics['limit_family_count']}")
            if "integrate_supported_case_count" in metrics:
                pieces.append(f"supported={metrics['integrate_supported_case_count']}")
            if "integrate_residual_case_count" in metrics:
                pieces.append(f"residual={metrics['integrate_residual_case_count']}")
            if "integrate_warning_expected_case_count" in metrics:
                pieces.append(
                    f"warning_expected={metrics['integrate_warning_expected_case_count']}"
                )
            if "integrate_required_display_case_count" in metrics:
                pieces.append(
                    f"required_display={metrics['integrate_required_display_case_count']}"
                )
            if "integrate_step_checked_case_count" in metrics:
                pieces.append(
                    f"step_checked={metrics['integrate_step_checked_case_count']}"
                )
            if "integrate_supported_step_unchecked_case_count" in metrics:
                pieces.append(
                    "unchecked_supported_steps="
                    f"{metrics['integrate_supported_step_unchecked_case_count']}"
                )
            if "integrate_antiderivative_verification_case_count" in metrics:
                pieces.append(
                    "antiderivative_verified="
                    f"{metrics['integrate_antiderivative_verification_case_count']}"
                )
            if "integrate_verified_supported_case_count" in metrics:
                pieces.append(
                    "verified_supported="
                    f"{metrics['integrate_verified_supported_case_count']}"
                )
            if "integrate_direct_diff_integrate_case_count" in metrics:
                pieces.append(
                    "direct_diff_integrate="
                    f"{metrics['integrate_direct_diff_integrate_case_count']}"
                )
            if (
                "integrate_direct_diff_integrate_exact_case_count" in metrics
                and "integrate_direct_diff_integrate_equivalence_case_count" in metrics
            ):
                pieces.append(
                    "direct_exact="
                    f"{metrics['integrate_direct_diff_integrate_exact_case_count']}"
                )
                pieces.append(
                    "direct_equiv="
                    f"{metrics['integrate_direct_diff_integrate_equivalence_case_count']}"
                )
            if "integrate_direct_diff_integrate_gap_case_count" in metrics:
                pieces.append(
                    "direct_gap="
                    f"{metrics['integrate_direct_diff_integrate_gap_case_count']}"
                )
            if "integrate_family_count" in metrics:
                pieces.append(f"families={metrics['integrate_family_count']}")
            if "backend_attempts" in metrics:
                pieces.append(f"backend_attempts={metrics['backend_attempts']}")
            if "backend_public_accepted" in metrics:
                pieces.append(
                    f"backend_public_accepted={metrics['backend_public_accepted']}"
                )
            if "backend_unverified_public_acceptances" in metrics:
                pieces.append(
                    "backend_unverified_public="
                    f"{metrics['backend_unverified_public_acceptances']}"
                )
            if "backend_fallback_eligible" in metrics:
                pieces.append(
                    f"backend_fallback_eligible={metrics['backend_fallback_eligible']}"
                )
            if "backend_unverified_fallback_acceptances" in metrics:
                pieces.append(
                    "backend_unverified_fallback="
                    f"{metrics['backend_unverified_fallback_acceptances']}"
                )
            if "backend_required_condition_count" in metrics:
                pieces.append(
                    "backend_required_conditions="
                    f"{metrics['backend_required_condition_count']}"
                )
            if "backend_budget_exceeded_count" in metrics:
                pieces.append(
                    "backend_budget_exceeded="
                    f"{metrics['backend_budget_exceeded_count']}"
                )
            if "backend_method_probe_budget_exhausted_count" in metrics:
                pieces.append(
                    "backend_method_budget_exhausted="
                    f"{metrics['backend_method_probe_budget_exhausted_count']}"
                )
            if "backend_verification_budget_exceeded_count" in metrics:
                pieces.append(
                    "backend_verification_budget_exceeded="
                    f"{metrics['backend_verification_budget_exceeded_count']}"
                )
            if "backend_method_probes_used_total" in metrics:
                pieces.append(
                    "backend_method_probes_used="
                    f"{metrics['backend_method_probes_used_total']}"
                )
            if "backend_verification_checks_used_total" in metrics:
                pieces.append(
                    "backend_verification_checks_used="
                    f"{metrics['backend_verification_checks_used_total']}"
                )
            if "backend_verification_elapsed_ms" in metrics:
                pieces.append(
                    "backend_verify="
                    f"{metrics['backend_verification_elapsed_ms']:.3f}ms"
                )
            if "backend_verification_pressure_status" in metrics:
                pieces.append(
                    "backend_verification_pressure="
                    f"{metrics['backend_verification_pressure_status']}"
                )
            summary = " ".join(pieces)
        elif "derived" in metrics:
            pieces = []
            if "sampled" in metrics:
                pieces.append(f"sampled={metrics['sampled']}")
            not_equivalent_label = (
                "expected_not_equivalent"
                if "generic_simplify_expected" in metrics and "sampled" not in metrics
                else "not_equivalent"
            )
            pieces.extend(
                [
                    f"derived={metrics['derived']}",
                    f"unsupported={metrics['unsupported']}",
                    f"{not_equivalent_label}={metrics['not_equivalent']}",
                    f"mean_step_count={metrics['mean_step_count']:.2f}",
                ]
            )
            if "generic_simplify_expected" in metrics:
                pieces.append(
                    f"generic_simplify_expected={metrics['generic_simplify_expected']}"
                )
            if "generic_simplify_strategy_successes" in metrics:
                pieces.append(
                    "generic_simplify_strategy_successes="
                    f"{metrics['generic_simplify_strategy_successes']}"
                )
            if "single_step_successes" in metrics:
                pieces.append(f"single_step={metrics['single_step_successes']}")
            if "multi_step_success_ids" in metrics:
                pieces.append(f"multi_step_ids={len(metrics['multi_step_success_ids'])}")
            if metrics.get("max_step_count") is not None:
                pieces.append(f"max_step={metrics['max_step_count']}")
            if metrics.get("embedded_family_sampled_count") is not None:
                pieces.append(
                    "embedded_families="
                    f"{metrics['embedded_family_sampled_count']}/"
                    f"{metrics['embedded_family_total_count']}"
                )
            summary = " ".join(pieces)
        elif "cargo_status" in metrics:
            pieces = [
                f"passed={metrics['passed']}",
                f"failed={metrics['failed']}",
            ]
            if metrics.get("ignored", 0) > 0:
                pieces.append(f"ignored={metrics['ignored']}")
            if "nf_convergent" in metrics:
                if "symbolic_closure_rate_percent" in metrics:
                    pieces.extend(
                        [
                            f"closure={metrics['symbolic_closure_rate_percent']:.1f}%",
                            f"nf={metrics['nf_convergent']} ({metrics['nf_rate_percent']:.1f}%)",
                            f"proved={metrics['proved_symbolic']} "
                            f"({metrics['proved_symbolic_rate_percent']:.1f}%)",
                            f"timeouts={metrics.get('timeouts', 0)}",
                        ]
                    )
                else:
                    pieces.extend(
                        [
                            f"nf={metrics['nf_convergent']}",
                            f"proved={metrics['proved_symbolic']}",
                            f"timeouts={metrics.get('timeouts', 0)}",
                        ]
                    )
            summary = " ".join(pieces)
        elif "flagged_cases" in metrics:
            if "no_wire_substeps" in metrics:
                summary = (
                    f"cases={metrics['cases']} flagged={metrics['flagged_cases']} "
                    f"no_wire_substeps={metrics['no_wire_substeps']} "
                    f"missing_math_sides={metrics['missing_math_sides']}"
                )
            else:
                pieces = [
                    f"cases={metrics['cases']}",
                    f"flagged={metrics['flagged_cases']}",
                    f"no_web_substeps={metrics['no_web_substeps']}",
                    f"no_web_steps={metrics['no_web_steps']}",
                ]
                if "artifact_seconds" in metrics:
                    pieces.extend(
                        [
                            f"artifacts={metrics['artifact_seconds']:.2f}s",
                            f"cli={metrics['cli_seconds']:.2f}s",
                            f"workers={metrics['worker_count']}",
                        ]
                    )
                if "artifact_family_hotspots" in metrics:
                    pieces.append(
                        f"top_artifact_family={metrics['artifact_family_hotspots'].split(',')[0]}"
                    )
                if "cli_family_hotspots" in metrics:
                    pieces.append(
                        f"top_cli_family={metrics['cli_family_hotspots'].split(',')[0]}"
                    )
                summary = " ".join(pieces)
        else:
            if "symbolic_closure_rate_percent" in metrics:
                summary = (
                    f"closure={metrics['symbolic_closure_rate_percent']:.1f}% "
                    f"nf={metrics['nf_convergent']} ({metrics['nf_rate_percent']:.1f}%) "
                    f"proved={metrics['proved_symbolic']} "
                    f"({metrics['proved_symbolic_rate_percent']:.1f}%) "
                    f"numeric={metrics['numeric_only']} inconclusive={metrics['inconclusive']} "
                    f"timeouts={metrics['timeouts']}"
                )
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
        "orchestrator_profile_filter": (
            args.orchestrator_profile_filter if args.orchestrator_profile else None
        ),
        "orchestrator_profile_limit": (
            args.orchestrator_profile_limit if args.orchestrator_profile else None
        ),
        "generated_discovery": generated_discovery_ledger_metrics(),
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
            profile_command = orchestrator_profile_command(spec, args)
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
                if (
                    spec.parser == "cargo_test_basic"
                    and metrics.get("ignored", 0) > 0
                ):
                    ignored_tests = ignored_cargo_tests_for_suite(spec)
                    if ignored_tests:
                        metrics["ignored_tests"] = ignored_tests
                status = suite_status(spec.name, metrics, returncode)
                if spec.name == "embedded_equivalence_context":
                    metrics["corpus_structure"] = embedded_corpus_structure_metrics()
                    metrics["coverage_saturation"] = (
                        embedded_coverage_saturation_metrics(metrics)
                    )
        measured_elapsed = effective_elapsed_seconds(metrics, elapsed)
        add_elapsed_per_case_metric(metrics, measured_elapsed)
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
        profile_command = orchestrator_profile_command(spec, args)
        if profile_env and profile_command and "parse_error" not in metrics:
            profile_limit = orchestrator_profile_case_limit(spec, args)
            profile_scope = (
                f"limit {profile_limit}"
                if profile_limit is not None
                else "configured pressure windows"
            )
            print()
            print(
                f"=== {spec.name}.orchestrator_profile ===\n"
                f"Profiled observability slice "
                f"({profile_scope}, "
                f"filter {args.orchestrator_profile_filter})"
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
                            "filter": args.orchestrator_profile_filter,
                            "limit": profile_limit,
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

    scorecard["generated_discovery_pressure"] = (
        generated_discovery_pressure_metrics(scorecard)
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
