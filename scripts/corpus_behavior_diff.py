#!/usr/bin/env python3
"""corpus_behavior_diff.py — GATE-A pata 2 del plan de auditoría.

Captura el comportamiento del corpus (web/examples.csv, 221 filas) por el CLI
release y lo compara contra una baseline. El contrato del gate (plan
2026-07-29 §2): «diff conductual VACÍO del corpus 221 —
result + steps_count + solve_steps + substeps + warnings». Se añade
required_conditions porque los lotes S-P0 giran precisamente sobre
condiciones que aparecen o dejan de perderse.

Las filas se ANCLAN POR EXPRESIÓN, nunca por índice (lección frente E:
el csv se reordena).

Uso:
  capture <salida.jsonl> [--steps on|off|compact]
      Corre el corpus y escribe una fila JSON por expresión.
  diff <baseline.jsonl> <after.jsonl> [--expect <fichero>]
      Compara. Exit 0 ⟺ diff vacío. Con --expect (un path con una expresión
      por línea, # comenta), implementa la excepción R7 de los lotes S-P0:
      exit 0 ⟺ el conjunto de filas cambiadas es EXACTAMENTE el previsto,
      ni una más ni una menos.

Baseline canónica del ciclo 0:
  docs/generated/quality_audit_baseline/corpus_behavior_baseline.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "cas_cli"
CSV = ROOT / "web" / "examples.csv"

FIELDS = ("ok", "r", "steps", "solve", "subs", "warnings", "req")


def capture(out_path: str, steps: str) -> int:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    n_err = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in rows:
            expr = row["expression"]
            try:
                proc = subprocess.run(
                    [str(BIN), "eval", expr, "--steps", steps, "--format", "json"],
                    capture_output=True, text=True, timeout=120,
                )
                d = json.loads(proc.stdout)
                rec = {
                    "e": expr,
                    "ok": d.get("ok"),
                    "r": d.get("result"),
                    "steps": d.get("steps_count"),
                    "solve": d.get("solve_steps_count"),
                    "subs": d.get("substeps_count"),
                    "warnings": d.get("warnings", []),
                    "req": d.get("required_conditions", []),
                }
            except Exception as exc:  # noqa: BLE001 — un crash del CLI ES señal
                rec = {"e": expr, "ok": "ERR", "r": str(exc)[:80],
                       "steps": None, "solve": None, "subs": None,
                       "warnings": [], "req": []}
                n_err += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"capturadas {len(rows)} filas -> {out_path} (errores de captura: {n_err})")
    return 0


def load(path: str) -> dict[str, dict]:
    recs = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        recs[d["e"]] = d
    return recs


def diff(base_path: str, after_path: str, expect_path: str | None) -> int:
    base, after = load(base_path), load(after_path)
    changed: dict[str, list[str]] = {}
    for e in sorted(set(base) | set(after)):
        if e not in base:
            changed[e] = ["FILA NUEVA (no está en la baseline)"]
            continue
        if e not in after:
            changed[e] = ["FILA AUSENTE (está en la baseline y no en after)"]
            continue
        deltas = []
        for f in FIELDS:
            if base[e].get(f) != after[e].get(f):
                deltas.append(f"{f}: {base[e].get(f)!r} -> {after[e].get(f)!r}")
        if deltas:
            changed[e] = deltas

    if expect_path is None:
        for e, deltas in changed.items():
            print(f"CAMBIADA: {e}")
            for d in deltas:
                print(f"    {d}")
        print(f"\nfilas cambiadas: {len(changed)} / {len(base)}")
        return 0 if not changed else 1

    expected = set()
    for line in open(expect_path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            expected.add(line)
    unexpected = set(changed) - expected
    missing = expected - set(changed)
    for e in sorted(unexpected):
        print(f"CAMBIO NO PREVISTO: {e}")
        for d in changed[e]:
            print(f"    {d}")
    for e in sorted(missing):
        print(f"PREVISTA Y SIN CAMBIO: {e}")
    print(f"\ncambiadas={len(changed)} previstas={len(expected)} "
          f"no_previstas={len(unexpected)} previstas_sin_cambio={len(missing)}")
    return 0 if not unexpected and not missing else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("out")
    c.add_argument("--steps", default="on", choices=["on", "off", "compact"])
    d = sub.add_parser("diff")
    d.add_argument("baseline")
    d.add_argument("after")
    d.add_argument("--expect", default=None,
                   help="fichero con las expresiones cuyo cambio está previsto (R7, lotes S-P0)")
    args = ap.parse_args()
    if args.cmd == "capture":
        return capture(args.out, args.steps)
    return diff(args.baseline, args.after, args.expect)


if __name__ == "__main__":
    sys.exit(main())
