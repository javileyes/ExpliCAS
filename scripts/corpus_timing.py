#!/usr/bin/env python3
"""corpus_timing.py — GATE-B (perf no-regresión) del plan de auditoría.

Mide `simplify_us` (mediana de 3 corridas) por fila del corpus
web/examples.csv con el CLI release, como en la campaña de perf 2ª
(corpus 5.37s→1.64s). Presupuesto del gate: base × 1.25.

⚠️ Medir SIEMPRE con la máquina quieta: sin cargo/make en paralelo
(lección 2026-07-28: la contención fabrica rojos falsos).

Uso:
  run [--steps off]                  Informe: total, mediana, p90, top-20.
  save <baseline.json> [--steps off] Mide y guarda la baseline.
  compare <baseline.json> [--steps off]
      Re-mide y compara. FALLA (exit 1) si el total supera base×1.25 o si
      alguna fila supera su base×1.25 con exceso absoluto > 1000 µs (el
      suelo absoluto evita falsas alarmas por jitter en casos de µs).

Baseline canónica del ciclo 0:
  docs/generated/quality_audit_baseline/corpus_timing_baseline.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "cas_cli"
CSV = ROOT / "web" / "examples.csv"

RUNS = 3
BUDGET_FACTOR = 1.25
ABS_FLOOR_US = 1000


def measure(steps: str) -> dict[str, int]:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    out: dict[str, int] = {}
    for row in rows:
        expr = row["expression"]
        samples = []
        for _ in range(RUNS):
            try:
                proc = subprocess.run(
                    [str(BIN), "eval", expr, "--steps", steps, "--format", "json"],
                    capture_output=True, text=True, timeout=120,
                )
                d = json.loads(proc.stdout)
                samples.append(int(d.get("timings_us", {}).get("simplify_us", -1)))
            except Exception:  # noqa: BLE001
                samples.append(-1)
        good = [s for s in samples if s >= 0]
        out[expr] = int(statistics.median(good)) if good else -1
    return out


def report(times: dict[str, int]) -> None:
    good = sorted(t for t in times.values() if t >= 0)
    total = sum(good)
    print(f"n={len(good)} total={total/1e6:.2f}s "
          f"median={statistics.median(good)/1000:.1f}ms "
          f"p90={good[int(len(good)*0.9)]/1000:.1f}ms")
    print("\nTOP 20 más lentas:")
    for expr, t in sorted(times.items(), key=lambda kv: -kv[1])[:20]:
        print(f"{t/1000:9.1f}ms  {expr[:90]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "save", "compare"):
        p = sub.add_parser(name)
        if name != "run":
            p.add_argument("baseline")
        p.add_argument("--steps", default="off", choices=["off", "on", "compact"])
    args = ap.parse_args()

    times = measure(args.steps)
    if args.cmd == "run":
        report(times)
        return 0
    if args.cmd == "save":
        Path(args.baseline).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"steps": args.steps, "times_us": times},
                  open(args.baseline, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=0)
        report(times)
        print(f"\nbaseline guardada -> {args.baseline}")
        return 0

    base = json.load(open(args.baseline, encoding="utf-8"))
    base_times: dict[str, int] = base["times_us"]
    if base.get("steps") != args.steps:
        print(f"AVISO: baseline medida con steps={base.get('steps')}, "
              f"comparación con steps={args.steps}")
    total_base = sum(t for t in base_times.values() if t >= 0)
    total_now = sum(t for e, t in times.items() if t >= 0 and e in base_times)
    offenders = []
    for e, t in times.items():
        b = base_times.get(e)
        if b is None or b < 0 or t < 0:
            continue
        if t > b * BUDGET_FACTOR and (t - b) > ABS_FLOOR_US:
            offenders.append((t - b, e, b, t))
    offenders.sort(reverse=True)
    for delta, e, b, t in offenders:
        print(f"REGRESIÓN: {b/1000:.1f}ms -> {t/1000:.1f}ms (+{delta/1000:.1f}ms)  {e[:80]}")
    verdict_total = total_now <= total_base * BUDGET_FACTOR
    print(f"\ntotal: base={total_base/1e6:.2f}s ahora={total_now/1e6:.2f}s "
          f"presupuesto={total_base*BUDGET_FACTOR/1e6:.2f}s "
          f"{'OK' if verdict_total else 'EXCEDIDO'}")
    print(f"filas con regresión: {len(offenders)}")
    return 0 if verdict_total and not offenders else 1


if __name__ == "__main__":
    sys.exit(main())
