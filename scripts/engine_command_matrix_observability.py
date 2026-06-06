"""Shared observability checks for command-matrix smoke lanes."""

from __future__ import annotations

from typing import Any


FRAGILE_STDERR_SUBSTRINGS = (
    "WARN",
    "depth_overflow",
    "stack overflow",
    "thread 'main' panicked",
    "panicked at",
    "fatal runtime error",
    "SIGSEGV",
)


def stderr_fragility_error(
    stderr: str,
    *,
    label: str = "stderr",
    forbidden_substrings: tuple[str, ...] = (),
) -> str | None:
    """Return an error when stderr contains high-signal fragility markers."""

    for fragment in (*FRAGILE_STDERR_SUBSTRINGS, *forbidden_substrings):
        if fragment and fragment in stderr:
            return f"{label} fragile substring found: {fragment!r}"
    return None


def runtime_observability_summary(
    results: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
    slowest_limit: int = 5,
    group_limit: int = 5,
) -> dict[str, Any]:
    """Summarize retained per-case runtime signal without keeping all cases."""

    timed_results = [
        result
        for result in results
        if isinstance(result.get("wall_elapsed_seconds"), (int, float))
    ]
    if not timed_results:
        return {}

    summary: dict[str, Any] = {
        "runtime_distribution": runtime_distribution_row(timed_results),
        "runtime_concentration": runtime_concentration_row(timed_results),
        "slowest_cases": slowest_case_rows(timed_results, limit=slowest_limit),
    }
    warm_results = timed_results[1:]
    if warm_results:
        summary["cold_start_case"] = runtime_case_row(timed_results[0])
        summary["warm_runtime_distribution"] = runtime_distribution_row(warm_results)
        summary["warm_runtime_concentration"] = runtime_concentration_row(warm_results)
        summary["warm_slowest_cases"] = slowest_case_rows(
            warm_results,
            limit=slowest_limit,
        )
    for key in group_keys:
        rows = runtime_group_rows(timed_results, group_key=key, limit=group_limit)
        if rows:
            summary[f"runtime_by_{key}"] = rows
        if warm_results:
            warm_rows = runtime_group_rows(
                warm_results,
                group_key=key,
                limit=group_limit,
            )
            if warm_rows:
                summary[f"warm_runtime_by_{key}"] = warm_rows
    return summary


def runtime_distribution_row(results: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed_values = sorted(
        float(result["wall_elapsed_seconds"])
        for result in results
        if isinstance(result.get("wall_elapsed_seconds"), (int, float))
    )
    if not elapsed_values:
        return {}

    total_elapsed = sum(elapsed_values)
    # Nearest-rank percentile keeps the signal stable for small smoke matrices.
    p95_index = max(0, (95 * len(elapsed_values) + 99) // 100 - 1)
    return {
        "timed_case_count": len(elapsed_values),
        "total_elapsed_seconds": round(total_elapsed, 3),
        "avg_case_ms": round(total_elapsed * 1000.0 / len(elapsed_values), 3),
        "p95_case_ms": round(elapsed_values[p95_index] * 1000.0, 3),
        "max_case_ms": round(elapsed_values[-1] * 1000.0, 3),
    }


def runtime_concentration_row(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(
        (
            result
            for result in results
            if isinstance(result.get("wall_elapsed_seconds"), (int, float))
        ),
        key=lambda result: (
            -float(result["wall_elapsed_seconds"]),
            str(result.get("name", "")),
        ),
    )
    total_elapsed = sum(float(result["wall_elapsed_seconds"]) for result in rows)
    if not rows or total_elapsed <= 0.0:
        return {}

    slowest = rows[0]
    slowest_elapsed = float(slowest["wall_elapsed_seconds"])
    top_3_elapsed = sum(float(result["wall_elapsed_seconds"]) for result in rows[:3])
    row: dict[str, Any] = {
        "timed_case_count": len(rows),
        "total_elapsed_seconds": round(total_elapsed, 3),
        "slowest_case": slowest.get("name"),
        "slowest_case_ms": round(slowest_elapsed * 1000.0, 3),
        "slowest_case_share_percent": round(
            slowest_elapsed * 100.0 / total_elapsed,
            1,
        ),
        "top_3_share_percent": round(top_3_elapsed * 100.0 / total_elapsed, 1),
    }
    for source_key, target_key in (
        ("family", "slowest_family"),
        ("argument_regime", "slowest_argument_regime"),
        ("point_regime", "slowest_point_regime"),
        ("domain_regime", "slowest_domain_regime"),
        ("trace_regime", "slowest_trace_regime"),
        ("presentation_regime", "slowest_presentation_regime"),
    ):
        value = slowest.get(source_key)
        if isinstance(value, str):
            row[target_key] = value
    return row


def slowest_case_rows(
    results: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows = sorted(
        results,
        key=lambda result: (
            -float(result.get("wall_elapsed_seconds", 0.0)),
            str(result.get("name", "")),
        ),
    )
    return [runtime_case_row(result) for result in rows[:limit]]


def payload_observability_case_rows(
    results: list[dict[str, Any]],
    *,
    metric_key: str,
    output_key: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    measured_results = [
        result for result in results if isinstance(result.get(metric_key), int)
    ]
    measured_results.sort(
        key=lambda result: (
            -int(result.get(metric_key, 0)),
            str(result.get("name", "")),
        )
    )

    rows: list[dict[str, Any]] = []
    for result in measured_results[:limit]:
        row: dict[str, Any] = {
            "name": result.get("name"),
            output_key: int(result[metric_key]),
            "required_display_count": len(result.get("required_display", [])),
            "expected_step_substring_count": len(
                result.get("expected_step_substrings", [])
            ),
        }
        for key in (
            "family",
            "argument_regime",
            "domain_regime",
            "trace_regime",
            "presentation_regime",
            "calculus_maturity_block",
            "calculus_block_gate",
        ):
            value = result.get(key)
            if isinstance(value, str):
                row[key] = value
        rows.append(row)
    return rows


def payload_observability_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    stdout_rows = payload_observability_case_rows(
        results,
        metric_key="stdout_bytes",
        output_key="stdout_bytes",
    )
    if stdout_rows:
        summary["largest_stdout_payload_cases"] = stdout_rows

    step_rows = payload_observability_case_rows(
        results,
        metric_key="step_text_char_count",
        output_key="step_text_char_count",
    )
    if step_rows:
        summary["largest_step_trace_cases"] = step_rows
    return summary


def runtime_case_row(result: dict[str, Any]) -> dict[str, Any]:
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
        value = result.get(key)
        if isinstance(value, str):
            row[key] = value
    elapsed = result.get("wall_elapsed_seconds")
    if isinstance(elapsed, (int, float)):
        row["wall_elapsed_seconds"] = round(float(elapsed), 3)
    return row


def runtime_group_rows(
    results: list[dict[str, Any]],
    *,
    group_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        group = result.get(group_key)
        if not isinstance(group, str):
            continue
        groups.setdefault(group, []).append(result)

    rows = [
        runtime_group_row(group_key, group, group_results)
        for group, group_results in groups.items()
    ]
    rows.sort(
        key=lambda row: (
            -float(row["total_elapsed_seconds"]),
            -int(row["case_count"]),
            str(row["group"]),
        )
    )
    return rows[:limit]


def runtime_group_row(
    group_key: str,
    group: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    elapsed_values = [
        float(result["wall_elapsed_seconds"])
        for result in results
        if isinstance(result.get("wall_elapsed_seconds"), (int, float))
    ]
    total_elapsed = sum(elapsed_values)
    slowest = max(
        results,
        key=lambda result: float(result.get("wall_elapsed_seconds", 0.0)),
    )
    return {
        "axis": group_key,
        "group": group,
        "case_count": len(elapsed_values),
        "total_elapsed_seconds": round(total_elapsed, 3),
        "avg_case_ms": round(total_elapsed * 1000.0 / len(elapsed_values), 3),
        "max_elapsed_seconds": round(max(elapsed_values), 3),
        "slowest_case": slowest.get("name"),
    }
