"""Shared observability checks for command-matrix smoke lanes."""

from __future__ import annotations


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
