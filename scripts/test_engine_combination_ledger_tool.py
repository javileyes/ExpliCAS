#!/usr/bin/env python3
"""Tests for engine_combination_ledger_tool.py rotation and indexing."""

from __future__ import annotations

import datetime
import pathlib
import sys
import tempfile
import unittest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import engine_combination_ledger_tool as TOOL

HEADER = """# Engine Combination Ledger

Purpose text.

## Required Fields

- area / status / observed / decision / retained learning

"""

ENTRY_APRIL = """## 2026-04-29 - April entry

- area:
  - orchestrator / april family
- status:
  - `rejected`
- observed:
  - body line april

"""

ENTRY_MAY = """## 2026-05-20 - May entry

- area:
  - calculus / may family
- status:
  - `observe-only`
- observed:
  - body line may

"""

ENTRY_JUNE = """## 2026-06-10 - June entry

- area:
  - calculus / june family
- status:
  - `retained`
- observed:
  - body line june
"""


class LedgerToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = pathlib.Path(self.tmp.name) / "ENGINE_COMBINATION_LEDGER.md"
        self.ledger.write_text(HEADER + ENTRY_APRIL + ENTRY_MAY + ENTRY_JUNE)
        self.today = datetime.date(2026, 6, 10)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_split_preserves_entry_bodies(self) -> None:
        header, entries = TOOL.split_ledger(self.ledger.read_text())
        self.assertIn("Required Fields", header)
        self.assertEqual(
            [entry["date"] for entry in entries],
            ["2026-04-29", "2026-05-20", "2026-06-10"],
        )
        self.assertEqual(entries[0]["body"], ENTRY_APRIL)
        self.assertEqual(entries[2]["body"], ENTRY_JUNE)

    def test_rotate_moves_months_byte_identical_and_keeps_rest(self) -> None:
        code = TOOL.rotate(self.ledger, ["2026-04", "2026-05"], self.today)
        self.assertEqual(code, 0)
        active = self.ledger.read_text()
        self.assertIn("June entry", active)
        self.assertNotIn("April entry", active)
        self.assertNotIn("May entry", active)
        april = TOOL.archive_path(self.ledger, "2026-04").read_text()
        may = TOOL.archive_path(self.ledger, "2026-05").read_text()
        self.assertIn(ENTRY_APRIL, april)
        self.assertIn(ENTRY_MAY, may)
        total = sum(
            len(TOOL.split_ledger(path.read_text())[1])
            for path in [self.ledger, *TOOL.archive_files(self.ledger)]
        )
        self.assertEqual(total, 3)

    def test_rotate_refuses_current_month(self) -> None:
        code = TOOL.rotate(self.ledger, ["2026-06"], self.today)
        self.assertEqual(code, 2)
        self.assertIn("June entry", self.ledger.read_text())

    def test_reindex_is_idempotent_and_lists_archives(self) -> None:
        TOOL.rotate(self.ledger, ["2026-04"], self.today)
        self.assertEqual(TOOL.reindex(self.ledger), 0)
        first = self.ledger.read_text()
        self.assertEqual(TOOL.reindex(self.ledger), 0)
        self.assertEqual(first, self.ledger.read_text())
        self.assertIn("ENGINE_COMBINATION_LEDGER_ARCHIVE_2026_04.md", first)
        self.assertIn("- 2026-06-10 | `retained` | calculus / june family | June entry", first)
        self.assertIn("Active entries: 2", first)
        # index lines stay inside the marker block, before the first entry
        self.assertLess(first.find(TOOL.INDEX_END), first.find("## 2026-05-20"))


if __name__ == "__main__":
    unittest.main()
