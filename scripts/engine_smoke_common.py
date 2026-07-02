"""Shared CLI-runner/JSON plumbing for the engine smoke lanes.

Canonical home for the copy-pasted helpers the command-matrix smokes and the
residual probe carried individually (audit §9.2). Behavior is byte-identical
to the previous copies.

Note: `engine_improvement_scorecard.py` keeps its own `terminate_process_group`
deliberately — it has a different contract (caller-orchestrated signal
escalation via an explicit `sig` parameter) rather than the built-in
TERM→wait→KILL sequence the smokes use.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from typing import Any


def parse_json(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    """Decode one CLI JSON payload; returns (payload, error)."""
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(value, dict):
        return None, "json output is not an object"
    return value, None


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    """TERM the process group, escalate to KILL after a 1s grace period."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=1.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def extract_cli_timings_us(payload: dict[str, Any] | None) -> dict[str, int]:
    """Extract the public timings_us wire block (parse/simplify/total)."""
    if not payload:
        return {}
    raw = payload.get("timings_us")
    if not isinstance(raw, dict):
        return {}

    timings: dict[str, int] = {}
    for key in ("parse_us", "simplify_us", "total_us"):
        value = raw.get(key)
        if isinstance(value, int):
            timings[key] = value
    return timings
