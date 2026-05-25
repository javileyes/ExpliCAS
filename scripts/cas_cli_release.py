#!/usr/bin/env python3
"""Helpers for keeping the release cas_cli binary fresh in smoke scripts."""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Iterable
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CAS_CLI = ROOT / "target" / "release" / "cas_cli"


def default_cas_cli(root: str | pathlib.Path = ROOT) -> pathlib.Path:
    return pathlib.Path(root) / "target" / "release" / "cas_cli"


def iter_release_input_paths(root: str | pathlib.Path = ROOT) -> Iterable[pathlib.Path]:
    root = pathlib.Path(root)
    for path in (root / "Cargo.toml", root / "Cargo.lock"):
        if path.exists():
            yield path

    crates_dir = root / "crates"
    if not crates_dir.exists():
        return

    yield from sorted(crates_dir.glob("*/Cargo.toml"))
    yield from sorted(crates_dir.glob("*/src/**/*.rs"))


def release_cas_cli_needs_rebuild(
    cas_cli: str | pathlib.Path,
    *,
    root: str | pathlib.Path = ROOT,
    inputs: Iterable[pathlib.Path] | None = None,
) -> bool:
    root = pathlib.Path(root)
    path = pathlib.Path(cas_cli)
    if path.resolve() != default_cas_cli(root).resolve():
        return False

    try:
        binary_mtime = path.stat().st_mtime
    except FileNotFoundError:
        return True

    input_paths = iter_release_input_paths(root) if inputs is None else inputs
    for source in input_paths:
        try:
            if pathlib.Path(source).stat().st_mtime > binary_mtime:
                return True
        except FileNotFoundError:
            continue
    return False


def ensure_release_cas_cli(
    cas_cli: str | pathlib.Path,
    *,
    root: str | pathlib.Path = ROOT,
    run: Any = subprocess.run,
) -> bool:
    if not release_cas_cli_needs_rebuild(cas_cli, root=root):
        return False

    run(
        ["cargo", "build", "--release", "-q", "-p", "cas_cli"],
        cwd=pathlib.Path(root),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return True
