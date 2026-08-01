#!/usr/bin/env python3
"""Grafo de llamadas de un fichero .rs o de un directorio de submódulos.

Mide cuánta estructura real tiene una partición: % de aristas de llamada que
quedan DENTRO del grupo (fichero en modo directorio, familia-por-regex en modo
fichero), flujos entre grupos, y primitivas compartidas candidatas a `support`.
Es la herramienta que midió que el orquestador era una bola (34,1% intra) y
que 41 primitivas pedían módulo propio (campaña 2026-07-31).

Uso: engine_decoupling_callgraph.py <ruta.rs | ruta/dir> [--support-threshold 4]
     [--exclude mod.rs,tests.rs] [--rules nombre=regex ...] [--json out.json]

- Modo directorio: cada .rs del directorio es un grupo; si existe `<dir>.rs`
  (el padre del troceo) se incluye como grupo propio. La resolución de
  llamadas es local-first (una def en el propio fichero gana, como en Rust);
  los nombres con >1 definición en ficheros distintos se cuentan aparte como
  ambiguos en vez de inventarles dueño.
- Modo fichero: los grupos salen de --rules (primer regex que casa, en orden;
  resto → "core"), como en la auditoría original.
- Detector de llamadas sobreaproximado a propósito (nombres >=2 chars,
  métodos `.foo(` incluidos): para métricas y visibilidad, pasarse es inocuo
  y quedarse corto miente (lección L-P3 de la campaña).
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

FN_DEF = re.compile(r"^(?:pub(?:\([a-z()]+\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_0-9]+)")
CALL = re.compile(r"\b([a-z_][a-z_0-9]+)\s*\(")


def top_level_fns(text):
    """{nombre: cuerpo} de las fns con `fn` en columna 0 (nivel superior)."""
    lines = text.splitlines(keepends=True)
    out = {}
    for i, line in enumerate(lines):
        m = FN_DEF.match(line)
        if not m:
            continue
        e = i
        while e < len(lines) and not re.match(r"^\}", lines[e]):
            e += 1
        out[m.group(1)] = "".join(lines[i:e + 1])
    return out


def collect_units(root, exclude):
    """[(grupo, {nombre: cuerpo})] por fichero (modo dir) o único (modo fichero)."""
    root = Path(root)
    units = []
    if root.is_dir():
        files = sorted(p for p in root.glob("*.rs") if p.name not in exclude)
        parent = root.parent / (root.name + ".rs")
        if parent.exists():
            files.append(parent)
        for f in files:
            units.append((f.stem if f != parent else f"{root.name}(padre)",
                          top_level_fns(f.read_text())))
    else:
        units.append((root.stem, top_level_fns(root.read_text())))
    return units


def build_graph(units):
    """Devuelve (defs_por_nombre, aristas [(grupo_org, fn, grupo_dst, callee)], ambiguos)."""
    where = defaultdict(list)          # nombre -> [grupos que lo definen]
    for grp, fns in units:
        for n in fns:
            where[n].append(grp)
    edges = []
    ambiguous = Counter()
    for grp, fns in units:
        local = set(fns)
        for n, body in fns.items():
            for callee in set(CALL.findall(body)):
                if callee == n or callee not in where:
                    continue
                if callee in local:
                    edges.append((grp, n, grp, callee))
                elif len(where[callee]) == 1:
                    edges.append((grp, n, where[callee][0], callee))
                else:
                    ambiguous[callee] += 1
    return where, edges, ambiguous


def classify_by_rules(units, rules):
    """Reagrupa las fns de un único fichero según regex de nombre."""
    fns = units[0][1]

    def cls(n):
        for name, pat in rules:
            if re.search(pat, n):
                return name
        return "core"

    return [(g, {n: b for n, b in fns.items() if cls(n) == g})
            for g in {cls(n) for n in fns}]


def analyze(units, support_threshold):
    where, edges, ambiguous = build_graph(units)
    n_fns = sum(len(f) for _, f in units)
    intra = sum(1 for a, _, b, _ in edges if a == b)
    cross = Counter((a, b) for a, _, b, _ in edges if a != b)
    callers_by_group = defaultdict(set)   # callee -> grupos distintos que la llaman
    callers = defaultdict(set)            # callee -> fns que la llaman
    for a, n, b, callee in edges:
        callers_by_group[callee].add(a)
        callers[callee].add(n)
    spread = {n: len(g) for n, g in callers_by_group.items()}
    support = sorted((n for n, s in spread.items() if s >= support_threshold),
                     key=lambda n: (-spread[n], -len(callers[n])))
    groups = Counter()
    for grp, fns in units:
        groups[grp] += len(fns)
    return {
        "fns": n_fns,
        "grupos": dict(groups),
        "aristas": len(edges),
        "aristas_intra": intra,
        "pct_intra": round(100 * intra / len(edges), 1) if edges else None,
        "aristas_ambiguas": sum(ambiguous.values()),
        "nombres_ambiguos": len(ambiguous),
        "flujos_cross_top": [
            {"de": a, "a": b, "aristas": c} for (a, b), c in cross.most_common(12)
        ],
        "support_threshold": support_threshold,
        "primitivas_compartidas": [
            {"fn": n, "grupos": spread[n], "llamadores": len(callers[n])}
            for n in support
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--support-threshold", type=int, default=4)
    ap.add_argument("--exclude", default="mod.rs,tests.rs")
    ap.add_argument("--rules", nargs="*", default=[],
                    help="nombre=regex para modo fichero")
    ap.add_argument("--json")
    args = ap.parse_args()

    units = collect_units(args.target, set(args.exclude.split(",")))
    if args.rules:
        if len(units) > 1:
            sys.exit("--rules solo tiene sentido en modo fichero")
        units = classify_by_rules(units, [r.split("=", 1) for r in args.rules])
    res = analyze(units, args.support_threshold)

    print(f"objetivo: {args.target}")
    print(f"fns nivel superior: {res['fns']}  en {len(res['grupos'])} grupos")
    print(f"aristas: {res['aristas']}  intra-grupo: {res['aristas_intra']} "
          f"({res['pct_intra']}%)  ambiguas: {res['aristas_ambiguas']} "
          f"({res['nombres_ambiguos']} nombres)")
    print("\n=== flujos ENTRE grupos (top) ===")
    for f in res["flujos_cross_top"]:
        print(f"  {f['de']:28} -> {f['a']:28} {f['aristas']:4}")
    prim = res["primitivas_compartidas"]
    print(f"\n=== primitivas llamadas desde >={args.support_threshold} grupos: "
          f"{len(prim)} ===")
    for p in prim[:15]:
        print(f"  {p['grupos']} grupos, {p['llamadores']:3} llamadores  {p['fn'][:64]}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(f"\n[json en {args.json}]")


if __name__ == "__main__":
    main()
