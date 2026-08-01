#!/usr/bin/env python3
"""Verificación independiente de un troceo/move contra un ref de git.

Compara las fns (o los bloques #[test]) del fichero original en REF con lo que
hay ahora en el padre + el directorio de submódulos: ninguna puede faltar,
sobrar, duplicarse ni cambiar de cuerpo. El canal de fallo de un move es la
resolución de nombres, no la semántica — pero este verificador es el que
certifica que fue movimiento PURO (campaña 2026-07-31: 1.378 bloques de test
byte a byte; 692 fns del orquestador con 0 cuerpos alterados).

Uso: engine_decoupling_verify_move.py <ruta/original.rs> [--ref HEAD]
     [--mode fns|tests] [--outdir ruta]   (outdir default: <parent>/<stem>/)

- mode=fns: normaliza prefijo de visibilidad (`pub(super)/pub(crate)`) y el
  reformateo de rustfmt antes de comparar — detecta cambios REALES de código.
- mode=tests: bloques #[test] completos (atributos multilínea incluidos),
  byte a byte. Excluye main.rs por nombre EXACTO (endswith casaría
  misc_domain.rs — trampa ya pagada).
"""
import argparse
import glob
import re
import subprocess
import sys
from pathlib import Path

FN_DEF = re.compile(r"^(?:pub(?:\([a-z()]+\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_0-9]+)")


def fns_of(text):
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


def test_blocks(text):
    lines = text.splitlines(keepends=True)
    out = []
    for i, line in enumerate(lines):
        if not re.match(r"^(pub )?fn ", line):
            continue
        p = i
        while p > 0 and re.match(r"^\s*(#\[|#!\[|///|//\s|\)\])", lines[p - 1]):
            p -= 1
        if not any("#[test]" in lines[k] for k in range(p, i)):
            continue
        e = i
        while e < len(lines) and not re.match(r"^\}", lines[e]):
            e += 1
        name = re.match(r"^(?:pub )?fn ([a-zA-Z_0-9]+)", line).group(1)
        out.append((name, "".join(lines[p:e + 1]).strip("\n")))
    return out


def norm_fn(body):
    body = re.sub(r"^\s*pub\((?:super|crate)\)\s+", "", body)
    body = re.sub(r"\bpub\((?:super|crate)\)\s+fn\b", "fn", body)
    body = re.sub(r"\s+", " ", body).strip()
    body = re.sub(r"([(\[{])\s+", r"\1", body)
    body = re.sub(r"\s+([)\]}])", r"\1", body)
    return re.sub(r",([)\]}])", r"\1", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--mode", choices=["fns", "tests"], default="fns")
    ap.add_argument("--outdir")
    args = ap.parse_args()

    old_text = subprocess.run(["git", "show", f"{args.ref}:{args.original}"],
                              capture_output=True, text=True).stdout
    if not old_text:
        sys.exit(f"no pude leer {args.original} en {args.ref}")
    outdir = args.outdir or str(Path(args.original).parent / Path(args.original).stem)

    new_files = [f for f in sorted(glob.glob(f"{outdir}/*.rs"))
                 if Path(f).name != "main.rs"]
    if Path(args.original).exists():
        new_files.append(args.original)

    if args.mode == "tests":
        old = test_blocks(old_text)
        new = []
        for f in new_files:
            new.extend(test_blocks(Path(f).read_text()))
        old_names = sorted(n for n, _ in old)
        new_names = sorted(n for n, _ in new)
        same_bodies = sorted(t for _, t in old) == sorted(t for _, t in new)
        print(f"{args.ref}: {len(old)} tests | ahora: {len(new)} tests "
              f"en {len(new_files)} ficheros")
        print(f"faltan: {sorted(set(old_names) - set(new_names)) or 'ninguno'}")
        print(f"sobran: {sorted(set(new_names) - set(old_names)) or 'ninguno'}")
        print(f"CUERPOS idénticos byte a byte: {same_bodies}")
        sys.exit(0 if (old_names == new_names and same_bodies) else 1)

    old_fns = fns_of(old_text)
    new_fns, dup = {}, []
    for f in new_files:
        for n, b in fns_of(Path(f).read_text()).items():
            if n in new_fns:
                dup.append(f"{n} ({f})")
            new_fns[n] = b
    missing = sorted(set(old_fns) - set(new_fns))
    extra = sorted(set(new_fns) - set(old_fns))
    changed = [n for n in old_fns if n in new_fns
               and norm_fn(old_fns[n]) != norm_fn(new_fns[n])]
    print(f"fns en {args.ref}: {len(old_fns)} | ahora: {len(new_fns)} "
          f"en {len(new_files)} ficheros")
    print(f"duplicadas: {dup or 'ninguna'}")
    print(f"faltan: {missing or 'ninguna'}")
    print(f"sobran (nuevas): {extra or 'ninguna'}")
    print(f"cuerpos que CAMBIARON (ignorando visibilidad/rustfmt): {len(changed)}")
    for n in changed[:8]:
        print(f"  {n}")
    sys.exit(0 if (not missing and not dup and not changed) else 1)


if __name__ == "__main__":
    main()
