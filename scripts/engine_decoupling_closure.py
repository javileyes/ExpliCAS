#!/usr/bin/env python3
"""Cierre transitivo de helpers por punto de entrada (¿puede viajar una regla?).

Para cada punto de entrada (bloques `define_rule!` por defecto) calcula el
cierre transitivo de helpers del fichero/directorio que necesita, agrupa por
tema y mide solape: si el solape es alto, dispersar los entries obliga a
duplicar o arrastrar maquinaria compartida. Es la herramienta que midió que
`rules/arithmetic` es UN motor de cancelación (trig necesita 151 helpers y
solo 15 son suyos — campaña 2026-07-31).

Uso: engine_decoupling_closure.py <ruta.rs | ruta/dir> [--per-entry]
     [--theme nombre=regex ...] [--exclude mod.rs,tests.rs] [--json out.json]

--per-entry lista cada entry con el tamaño de su cierre y el ranking de
helpers por nº de entries que los alcanzan (la "API mínima que todos usan").
Los temas por defecto son los de rules/arithmetic; para otra costura pásalos.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from engine_decoupling_callgraph import CALL, collect_units

ENTRY = re.compile(r"^define_rule!\(\s*([A-Za-z_0-9]+)?")
DEFAULT_THEMES = [
    ("trigonometry", r"Trig"),
    ("hyperbolic", r"Hyperbolic"),
    ("logarithms", r"Log"),
    ("exponents", r"HalfPower|NumericExponents"),
    ("algebra", r"CubesQuotient"),
]


def entry_blocks(root, exclude):
    """[(nombre, cuerpo)] de los bloques define_rule! del fichero/directorio."""
    root = Path(root)
    files = (sorted(p for p in root.glob("*.rs") if p.name not in exclude)
             if root.is_dir() else [root])
    if root.is_dir() and (root.parent / (root.name + ".rs")).exists():
        files.append(root.parent / (root.name + ".rs"))
    out = []
    for f in files:
        lines = f.read_text().splitlines(keepends=True)
        for i, line in enumerate(lines):
            if not line.startswith("define_rule!"):
                continue
            e = i
            while e < len(lines) and not re.match(r"^\);", lines[e]):
                e += 1
            body = "".join(lines[i:e + 1])
            m = re.search(r"define_rule!\(\s*(?://[^\n]*\n\s*)?([A-Za-z_0-9]+)", body)
            if m:
                out.append((m.group(1), body))
    return out


def closures(entries, helpers):
    """{entry: set(helpers alcanzados transitivamente)}"""
    names = set(helpers)

    def called(text):
        return {c for c in CALL.findall(text) if c in names}

    memo = {}

    def close(seed):
        seen, stack = set(), list(seed)
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(called(helpers[n]) - seen)
        return seen

    for name, body in entries:
        memo[name] = close(called(body))
    return memo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--per-entry", action="store_true")
    ap.add_argument("--theme", nargs="*", default=[])
    ap.add_argument("--exclude", default="mod.rs,tests.rs")
    ap.add_argument("--json")
    args = ap.parse_args()

    exclude = set(args.exclude.split(","))
    units = collect_units(args.target, exclude)
    helpers = {}
    for _, fns in units:
        helpers.update(fns)
    entries = entry_blocks(args.target, exclude)
    themes = ([tuple(t.split("=", 1)) for t in args.theme]
              if args.theme else DEFAULT_THEMES)

    def theme_of(entry):
        for t, pat in themes:
            if re.search(pat, entry):
                return t
        return "(resto)"

    need = closures(entries, helpers)
    print(f"objetivo: {args.target}")
    print(f"helpers nivel superior: {len(helpers)}   entries: {len(entries)}")

    by_theme = defaultdict(set)
    for e, hs in need.items():
        by_theme[theme_of(e)] |= hs
    owner = defaultdict(set)
    for t, hs in by_theme.items():
        for h in hs:
            owner[h].add(t)

    print("\n=== cierre transitivo por tema ===")
    for t, hs in sorted(by_theme.items(), key=lambda kv: -len(kv[1])):
        n_rules = sum(1 for e in need if theme_of(e) == t)
        excl = sum(1 for h in hs if owner[h] == {t})
        print(f"  {t:14} {n_rules:2} entries  necesita {len(hs):4} helpers  "
              f"exclusivos {excl:4}")
    shared = {h for h, ts in owner.items() if len(ts) > 1}
    print(f"\nhelpers alcanzados: {len(owner)}/{len(helpers)}   "
          f"compartidos entre temas: {len(shared)}")

    reach = defaultdict(set)               # helper -> entries que lo alcanzan
    for e, hs in need.items():
        for h in hs:
            reach[h].add(e)
    universal = sorted(reach, key=lambda h: -len(reach[h]))
    if args.per_entry:
        print("\n=== cierre por entry ===")
        for e in sorted(need, key=lambda e: -len(need[e])):
            print(f"  {len(need[e]):4} helpers  {theme_of(e):14} {e}")
        print("\n=== helpers por nº de entries que los alcanzan (API de facto) ===")
        for h in universal[:25]:
            print(f"  {len(reach[h]):3}/{len(entries)} entries  {h[:64]}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "helpers": len(helpers),
            "entries": {e: sorted(hs) for e, hs in need.items()},
            "por_tema": {t: sorted(hs) for t, hs in by_theme.items()},
            "compartidos_entre_temas": sorted(shared),
            "alcance_helpers": {h: len(es) for h, es in reach.items()},
        }, indent=1, ensure_ascii=False))
        print(f"\n[json en {args.json}]")


if __name__ == "__main__":
    main()
