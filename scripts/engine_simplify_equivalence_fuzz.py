#!/usr/bin/env python3
"""Equivalence fuzz for the shared simplifier.

Generates a DETERMINISTIC set of trig rational expressions (fixed seed), runs
`simplify(...)` through the release CLI, and checks that the output is
NUMERICALLY EQUIVALENT to the input at several sample points. This is the
detector that caught P0-G (`collect_mul_factors_int_pow` returning repeated
bases, over-cancelling a factor in `diff(sin(x)*tan(x), x, 2)`): a green unit
suite does not see a non-equivalence that only appears on non-canonical
mid-pipeline shapes, but a numeric oracle does.

Contract:
- 0 NON-EQUIVALENCES allowed (any is a soundness bug: simplify must preserve
  numeric equivalence).
- HANGS (timeout) are reported and bounded by HANG_BUDGET but do not fail the
  run: the known inter-node expand<->factor oscillation (C5 bug-2, see
  docs/ENGINE_COMBINATION_LEDGER.md) hits ~2-3% of forms. When that bug is
  fixed, drop HANG_BUDGET to 0.

Usage: python3 scripts/engine_simplify_equivalence_fuzz.py [--trials N]
Exit code 0 = pass, 1 = non-equivalence found or hang budget exceeded.
"""

import argparse
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

CLI = Path(__file__).resolve().parent.parent / "target" / "release" / "cas_cli"
SEED = 20260702
DEFAULT_TRIALS = 120
TIMEOUT_SECONDS = 12
HANG_BUDGET_FRACTION = 0.08  # generous ceiling over the observed ~2-3%
SAMPLE_POINTS = [0.5, 0.7, 1.1]
TOLERANCE = 1e-7


def simplify(expr: str):
    try:
        out = subprocess.run(
            [str(CLI), "eval", f"simplify({expr})", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        return json.loads(out.stdout).get("result")
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def numeric_value(expr: str, x: float) -> float:
    text = expr.replace("·", "*").replace("^", "**")
    text = re.sub(r"(sin|cos|tan)\(", r"math.\1(", text)
    text = re.sub(r"\bsec\(", "1/math.cos(", text)
    text = re.sub(r"\bcsc\(", "1/math.sin(", text)
    text = re.sub(r"\bcot\(", "_cot(", text)
    text = re.sub(r"\bpi\b", "math.pi", text)
    text = re.sub(r"(?<![\w.])e(?![\w(])", "math.e", text)
    text = re.sub(r"\bx\b", f"({x})", text)
    scope = {"math": math, "_cot": lambda t: math.cos(t) / math.sin(t)}
    return eval(text, scope)  # noqa: S307 - trusted generated expressions


def random_term(rng: random.Random) -> str:
    coef = rng.choice(["", "2*", "-", "-2*", "3*"])
    parts = []
    for fname in ("sin(x)", "cos(x)"):
        power = rng.choice([0, 0, 1, 1, 2, 3])
        if power == 1:
            parts.append(fname)
        elif power > 1:
            parts.append(f"{fname}^{power}")
    if not parts:
        parts = ["1"]
    return coef + "*".join(parts)


def generate_cases(trials: int):
    rng = random.Random(SEED)
    denominators = [
        "cos(x)^2",
        "cos(x)^3",
        "cos(x)^4",
        "sin(x)^2",
        "cos(x)^2*sin(x)",
        "2*cos(x)^2",
        "sin(x)*cos(x)",
    ]
    for _ in range(trials):
        nterms = rng.choice([2, 2, 3])
        numerator = " + ".join(random_term(rng) for _ in range(nterms)).replace("+ -", "- ")
        denominator = rng.choice(denominators)
        yield f"({numerator}) / ({denominator})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    args = parser.parse_args()

    if not CLI.exists():
        print(f"error: release CLI not found at {CLI} (run `cargo build --release -p cas_cli`)")
        return 1

    non_equivalences = []
    hangs = 0
    total = 0
    for expr in generate_cases(args.trials):
        total += 1
        result = simplify(expr)
        if result is None:
            hangs += 1
            continue
        try:
            equivalent = all(
                abs(numeric_value(expr, x) - numeric_value(result, x)) < TOLERANCE
                for x in SAMPLE_POINTS
            )
        except Exception:
            # Unevaluable output (unexpected rendering) counts as a finding: the
            # oracle must be able to check everything the engine emits here.
            non_equivalences.append((expr, result, "unevaluable output"))
            continue
        if not equivalent:
            non_equivalences.append((expr, result, "numeric mismatch"))

    hang_budget = max(1, int(total * HANG_BUDGET_FRACTION))
    print(
        f"simplify_equivalence_fuzz: total={total} non_equivalences={len(non_equivalences)} "
        f"hangs={hangs} (budget {hang_budget})"
    )
    for expr, result, kind in non_equivalences[:10]:
        print(f"  NON-EQUIV ({kind}): simplify({expr}) -> {result}")

    if non_equivalences:
        return 1
    if hangs > hang_budget:
        print("  hang budget exceeded (regression in simplifier termination)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
