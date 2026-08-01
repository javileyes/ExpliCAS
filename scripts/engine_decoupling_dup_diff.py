#!/usr/bin/env python3
"""¿Las copias de un helper duplicado son iguales o han derivado?

Agrupa todas las definiciones de cada nombre por cuerpo normalizado
(espacios, comentarios y reformateos de rustfmt fuera). Paso previo
OBLIGATORIO antes de fusionar duplicados: fusionar a ciegas cambia el
comportamiento en N sitios (campaña 2026-07-31: `collect_add_terms` tenía
18 definiciones y 15 variantes DISTINTAS — no eran copias; y la deriva cobró
como P0 real, el 7/3). Si las variantes difieren, la divergencia es un
hallazgo en sí misma: puede ser un fix que nunca viajó, o semántica distinta
que merece RENOMBRE, no fusión.

Uso: engine_decoupling_dup_diff.py <nombre_fn> [<nombre_fn> ...] [--root crates]
     [--all] [--diff N M]   (--diff imprime el diff de dos variantes)
"""
import argparse
import difflib
import re
import subprocess
from collections import defaultdict
from pathlib import Path


def norm(body):
    b = re.sub(r"//.*", "", body)
    b = re.sub(r"\s+", " ", b).strip()
    b = re.sub(r"([(\[{])\s+", r"\1", b)
    b = re.sub(r"\s+([)\]}])", r"\1", b)
    b = re.sub(r",([)\]}])", r"\1", b)
    return re.sub(r"\s+\.", ".", b)


def definitions(name, root):
    out = subprocess.run(
        ["grep", "-rn", "--include=*.rs", "-E",
         rf"^\s*(pub(\([a-z()]+\))? )?fn {name}\b", root],
        capture_output=True, text=True).stdout
    defs = []
    for line in out.splitlines():
        path, lineno, _ = line.split(":", 2)
        lines = Path(path).read_text().splitlines(keepends=True)
        i = int(lineno) - 1
        indent = len(lines[i]) - len(lines[i].lstrip())
        close = (" " * indent + "}").rstrip()
        e = i
        while e < len(lines) and lines[e].rstrip("\n") != close:
            e += 1
        defs.append((path, int(lineno), "".join(lines[i:e + 1])))
    return defs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--root", default="crates")
    ap.add_argument("--all", action="store_true",
                    help="lista todas las ubicaciones, no solo 3 por variante")
    ap.add_argument("--diff", nargs=2, type=int, metavar=("N", "M"),
                    help="imprime el diff entre las variantes N y M (1-index)")
    args = ap.parse_args()

    for name in args.names:
        defs = definitions(name, args.root)
        groups = defaultdict(list)
        for path, ln, body in defs:
            groups[norm(body)].append((path, ln, len(body.splitlines()), body))
        ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        print(f"\n=== fn {name}: {len(defs)} definiciones, "
              f"{len(ordered)} variantes distintas ===")
        for i, (_, locs) in enumerate(ordered, 1):
            crates = sorted({loc[0].split("/")[1] if "/" in loc[0] else loc[0]
                             for loc in locs})
            print(f"  variante {i}: {len(locs):2} copias, {locs[0][2]:3} líneas, "
                  f"en: {', '.join(crates)}")
            shown = locs if args.all else locs[:3]
            for p, ln, _, _ in shown:
                print(f"      {p}:{ln}")
            if not args.all and len(locs) > 3:
                print(f"      … y {len(locs) - 3} más")
        if args.diff and len(ordered) >= max(args.diff):
            a = ordered[args.diff[0] - 1][1][0]
            b = ordered[args.diff[1] - 1][1][0]
            print(f"\n--- diff variante {args.diff[0]} ({a[0]}:{a[1]}) "
                  f"vs {args.diff[1]} ({b[0]}:{b[1]}) ---")
            for line in difflib.unified_diff(
                    a[3].splitlines(), b[3].splitlines(), lineterm="", n=1):
                print(f"  {line}")


if __name__ == "__main__":
    main()
