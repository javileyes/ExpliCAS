#!/usr/bin/env python3
"""Snapshot de las métricas de desacoplo por costura (plan D0-D3 de 2026-08).

Define operativamente las métricas del PLAN_DESACOPLO_2026-08 y las mide en el
árbol actual, para comparar cada peldaño contra el baseline versionado en
docs/generated/decoupling_metrics_baseline.json:

- D1 (rules/arithmetic): entries define_rule!, cierre transitivo por tema
  (exclusivos vs compartidos), % aristas intra-fichero del directorio.
- D2 (orchestrator): % aristas intra-fichero, fan-in del módulo `support`,
  nº de `pub(super)`/`pub(crate)` (superficie interna de facto).
- D3 (cas_math): aristas de import entre familias de módulos (¿qué bloquea
  partir el crate?), líneas y pub fns por crate del workspace.

Uso: engine_decoupling_metrics.py [--seam d1|d2|d3|all] [--json out.json]
El baseline de COMPILACIÓN (cargo build --timings tras touch de cas_math) se
mide aparte — es caro y contamina target/: ver DESACOPLO_D0_BASELINE_2026-08.
"""
import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from engine_decoupling_callgraph import analyze, build_graph, collect_units
from engine_decoupling_closure import DEFAULT_THEMES, closures, entry_blocks

ARITH = "crates/cas_engine/src/rules/arithmetic"
ORCH = "crates/cas_engine/src/orchestrator"
CAS_MATH = "crates/cas_math/src"
EXCLUDE = {"mod.rs", "tests.rs"}

D3_FAMILIES = [
    ("integration", r"integrat|antideriv|rootsum|partial_fraction|risch|hermite"),
    ("differentiation", r"^diff|derivative|differentiat"),
    ("limits", r"limit"),
    ("poly", r"poly|monomial|resultant|groebner|sylvester"),
    ("series", r"series|taylor"),
    ("solve_support", r"solve|isolation|inequal"),
    ("trig", r"trig|angle"),
    ("numeric", r"numeric|const_|bigint|rational|root_forms|sign"),
]


def seam_d1():
    units = collect_units(ARITH, EXCLUDE)
    helpers = {}
    for _, fns in units:
        helpers.update(fns)
    entries = entry_blocks(ARITH, EXCLUDE)
    need = closures(entries, helpers)

    def theme_of(e):
        for t, pat in DEFAULT_THEMES:
            if re.search(pat, e):
                return t
        return "(resto)"

    by_theme = defaultdict(set)
    for e, hs in need.items():
        by_theme[theme_of(e)] |= hs
    owner = defaultdict(set)
    for t, hs in by_theme.items():
        for h in hs:
            owner[h].add(t)
    reach = defaultdict(set)
    for e, hs in need.items():
        for h in hs:
            reach[h].add(e)
    graph = analyze(units, support_threshold=4)
    return {
        "entries": len(entries),
        "helpers": len(helpers),
        "cierre_por_tema": {
            t: {"entries": sum(1 for e in need if theme_of(e) == t),
                "necesita": len(hs),
                "exclusivos": sum(1 for h in hs if owner[h] == {t})}
            for t, hs in sorted(by_theme.items(), key=lambda kv: -len(kv[1]))
        },
        "helpers_compartidos_entre_temas": sum(1 for ts in owner.values() if len(ts) > 1),
        "pct_aristas_intra_fichero": graph["pct_intra"],
        "api_de_facto_top": [
            {"fn": h, "entries": len(reach[h])}
            for h in sorted(reach, key=lambda h: -len(reach[h]))[:15]
        ],
    }


def seam_d2():
    units = collect_units(ORCH, EXCLUDE)
    graph = analyze(units, support_threshold=4)
    _, edges, _ = build_graph(units)
    sup_edges = [(a, n, callee) for a, n, b, callee in edges
                 if b == "support" and a != "support"]
    vis = Counter()
    for f in Path(ORCH).glob("*.rs"):
        text = f.read_text()
        vis["pub(super)"] += len(re.findall(r"\bpub\(super\)", text))
        vis["pub(crate)"] += len(re.findall(r"\bpub\(crate\)", text))
    return {
        "fns": graph["fns"],
        "ficheros": len(graph["grupos"]),
        "pct_aristas_intra_fichero": graph["pct_intra"],
        "support_fan_in": {
            "aristas": len(sup_edges),
            "modulos_que_lo_usan": len({a for a, _, _ in sup_edges}),
            "fns_llamadoras": len({(a, n) for a, n, _ in sup_edges}),
            "primitivas_usadas": len({c for _, _, c in sup_edges}),
        },
        "visibilidad_interna": dict(vis),
        "flujos_cross_top": graph["flujos_cross_top"][:8],
    }


def seam_d3():
    files = sorted(Path(CAS_MATH).rglob("*.rs"))

    def top_module(p):
        rel = p.relative_to(CAS_MATH)
        return rel.parts[0].removesuffix(".rs") if len(rel.parts) == 1 else rel.parts[0]

    def family(mod):
        for fam, pat in D3_FAMILIES:
            if re.search(pat, mod):
                return fam
        return "(resto)"

    cross = Counter()
    mod_out = defaultdict(set)
    for f in files:
        src_fam = family(top_module(f))
        text = f.read_text()
        for dst_mod in set(re.findall(r"\bcrate::([a-z_0-9]+)", text)):
            dst_fam = family(dst_mod)
            if dst_fam != src_fam:
                cross[(src_fam, dst_fam)] += 1
                mod_out[top_module(f)].add(dst_fam)
    bridges = sorted(mod_out.items(), key=lambda kv: -len(kv[1]))[:10]

    crates = {}
    for cargo in sorted(Path("crates").glob("*/Cargo.toml")):
        crate = cargo.parent.name
        n_lines = n_pub = 0
        for f in (cargo.parent / "src").rglob("*.rs"):
            text = f.read_text()
            n_lines += text.count("\n")
            n_pub += len(re.findall(r"^pub fn ", text, re.M))
        crates[crate] = {"lineas_src": n_lines, "pub_fns": n_pub}
    return {
        "familias": [f for f, _ in D3_FAMILIES] + ["(resto)"],
        "imports_cross_familia": [
            {"de": a, "a": b, "ficheros": c} for (a, b), c in cross.most_common(20)
        ],
        "modulos_puente_top": [
            {"modulo": m, "familias_que_importa": len(fs)} for m, fs in bridges
        ],
        "crates": crates,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", choices=["d1", "d2", "d3", "all"], default="all")
    ap.add_argument("--json")
    args = ap.parse_args()

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    snap = {"commit": head}
    if args.seam in ("d1", "all"):
        snap["d1_arithmetic"] = seam_d1()
    if args.seam in ("d2", "all"):
        snap["d2_orchestrator"] = seam_d2()
    if args.seam in ("d3", "all"):
        snap["d3_cas_math"] = seam_d3()

    print(json.dumps(snap, indent=1, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(snap, indent=1, ensure_ascii=False) + "\n")
        print(f"\n[snapshot en {args.json}]")


if __name__ == "__main__":
    main()
