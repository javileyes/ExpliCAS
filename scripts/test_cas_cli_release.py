import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "cas_cli_release.py"
SPEC = importlib.util.spec_from_file_location("cas_cli_release", MODULE_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(HELPER)


class CasCliReleaseTests(unittest.TestCase):
    def test_release_cas_cli_needs_rebuild_when_missing_or_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "crates" / "cas_cli" / "src" / "main.rs"
            source.parent.mkdir(parents=True)
            source.write_text("fn main() {}\n", encoding="utf-8")
            cas_cli = root / "target" / "release" / "cas_cli"
            cas_cli.parent.mkdir(parents=True)

            self.assertTrue(HELPER.release_cas_cli_needs_rebuild(cas_cli, root=root))

            cas_cli.write_text("#!/bin/sh\n", encoding="utf-8")
            os.utime(source, (100.0, 100.0))
            os.utime(cas_cli, (200.0, 200.0))
            self.assertFalse(HELPER.release_cas_cli_needs_rebuild(cas_cli, root=root))

            os.utime(source, (300.0, 300.0))
            self.assertTrue(HELPER.release_cas_cli_needs_rebuild(cas_cli, root=root))

    def test_ensure_release_cas_cli_ignores_custom_binary_paths(self) -> None:
        calls = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            custom = root / "custom" / "cas_cli"
            rebuilt = HELPER.ensure_release_cas_cli(
                custom,
                root=root,
                run=lambda *args, **kwargs: calls.append((args, kwargs)),
            )

        self.assertFalse(rebuilt)
        self.assertEqual(calls, [])

    def test_ensure_release_cas_cli_rebuilds_default_binary_when_stale(self) -> None:
        calls = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "crates" / "cas_cli" / "src" / "main.rs"
            source.parent.mkdir(parents=True)
            source.write_text("fn main() {}\n", encoding="utf-8")
            cas_cli = root / "target" / "release" / "cas_cli"
            cas_cli.parent.mkdir(parents=True)
            cas_cli.write_text("#!/bin/sh\n", encoding="utf-8")
            os.utime(cas_cli, (100.0, 100.0))
            os.utime(source, (200.0, 200.0))

            rebuilt = HELPER.ensure_release_cas_cli(
                cas_cli,
                root=root,
                run=lambda *args, **kwargs: calls.append((args, kwargs)),
            )

        self.assertTrue(rebuilt)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], ["cargo", "build", "--release", "-q", "-p", "cas_cli"])
