#!/usr/bin/env python3
"""Smoke checks for engine-improvement Makefile target wiring."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def make_dry_run(target: str) -> str:
    completed = subprocess.run(
        ["make", "-n", target],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


class EngineMakefileTargetsTests(unittest.TestCase):
    def test_fast_target_writes_fast_artifacts(self) -> None:
        output = make_dry_run("engine-fast")

        self.assertIn(
            '--profile fast --output "docs/generated/engine_improvement_scorecard_fast.json" '
            '--markdown-output "docs/generated/engine_improvement_scorecard_fast.md"',
            output,
        )

    def test_fast_embedded_target_writes_fast_embedded_artifacts(self) -> None:
        output = make_dry_run("engine-fast-embedded")

        self.assertIn(
            '--profile fast_embedded --output '
            '"docs/generated/engine_improvement_scorecard_fast_embedded.json" '
            '--markdown-output '
            '"docs/generated/engine_improvement_scorecard_fast_embedded.md"',
            output,
        )

    def test_full_target_writes_full_artifacts(self) -> None:
        output = make_dry_run("engine-scorecard-full")

        self.assertIn(
            '--profile full --output "docs/generated/engine_improvement_scorecard_full.json" '
            '--markdown-output "docs/generated/engine_improvement_scorecard_full.md"',
            output,
        )

    def test_guardrail_target_keeps_canonical_default_artifacts(self) -> None:
        output = make_dry_run("engine-scorecard")

        self.assertIn(
            "python3 ./scripts/engine_improvement_scorecard.py --profile guardrail",
            output,
        )
        self.assertNotIn("--output", output)
        self.assertNotIn("--markdown-output", output)


if __name__ == "__main__":
    unittest.main()
