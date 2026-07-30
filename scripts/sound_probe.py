#!/usr/bin/env python3
"""sound_probe.py — oraculo EXACTO para fichas de soundness (R11 del plan).

Compara dos expresiones en puntos RACIONALES exactos (sympy.Rational, jamas
float para el veredicto). Uso principal: comprobar que la simplificacion del
motor es una IDENTIDAD respecto a su entrada.

Subcomandos
-----------
  check   "<input>" [--axes ...]   Corre el CLI del motor sobre <input>, toma
                                   `result` del JSON y compara input vs result.
  compare "<A>" "<B>"              Compara dos expresiones dadas a mano.

Veredictos por punto:
  EQUAL           ambos lados definidos y su diferencia es exactamente 0
  DIFF            ambos lados definidos y la diferencia NO es 0 (exacto)
  DIFF_NUMERIC    diferencia no cero solo certificada a 60 digitos (marcar!)
  SKIP_UNDEF      algun lado indefinido/complejo en value-domain real
  SKIP_UNSUPPORTED  token que no sabemos traducir -> nunca inventamos veredicto

Codigo de salida: 0 = sin discrepancia, 2 = discrepancia, 3 = no concluyente.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from fractions import Fraction

import sympy as sp

CLI = "./target/release/cas_cli"

# ---------------------------------------------------------------- traduccion

# Tokens del motor que NO sabemos traducir con seguridad: si aparecen,
# devolvemos SKIP_UNSUPPORTED en vez de arriesgar un veredicto falso.
UNSUPPORTED = [
    "root_sum", "RootSum", "Piecewise", "piecewise", "Union", "union",
    "__hold", "Integral", "integral(", "Sum(", "Product(", "O(",
    "undefined", "Undefined", "NaN", "nan", "?",
    "and", "or", "<", ">", "=",
]

REPLACEMENTS = [
    ("·", "*"),
    ("×", "*"),
    ("−", "-"),
    ("π", "pi"),
    ("∞", "oo"),
    ("infinity", "oo"),
    ("ln(", "log("),
    ("arcsinh(", "asinh("),
    ("arccosh(", "acosh("),
    ("arctanh(", "atanh("),
    ("arcsin(", "asin("),
    ("arccos(", "acos("),
    ("arctan(", "atan("),
    ("arccot(", "acot("),
    ("arcsec(", "asec("),
    ("arccsc(", "acsc("),
    ("cbrt(", "real_cbrt("),
    ("^", "**"),
]

SUPERSCRIPTS = {"²": "**2", "³": "**3", "⁴": "**4", "⁵": "**5", "⁶": "**6",
                "⁷": "**7", "⁸": "**8", "⁹": "**9", "¹": "**1", "⁰": "**0",
                "⁻": "-"}


def _abs_bars(s: str) -> str:
    """|expr| -> Abs(expr); solo pares balanceados simples."""
    out = []
    depth = 0
    i = 0
    open_stack = []
    while i < len(s):
        c = s[i]
        if c == "|":
            if open_stack and open_stack[-1] == depth:
                out.append(")")
                open_stack.pop()
            else:
                out.append("Abs(")
                open_stack.append(depth)
        else:
            if c in "([":
                depth += 1
            elif c in ")]":
                depth -= 1
            out.append(c)
        i += 1
    if open_stack:
        raise ValueError("barras de valor absoluto desbalanceadas")
    return "".join(out)


def to_sympy(text: str):
    """Traduce la sintaxis del motor a sympy. Lanza ValueError si no es seguro."""
    s = text.strip()
    if not s:
        raise ValueError("expresion vacia")
    for bad in UNSUPPORTED:
        if bad in s:
            raise ValueError(f"token no soportado: {bad!r}")
    for k, v in SUPERSCRIPTS.items():
        s = s.replace(k, v)
    s = _abs_bars(s)
    for a, b in REPLACEMENTS:
        s = s.replace(a, b)
    # sqrt[n]x -> x**(1/n)  (notacion radical del motor)
    s = re.sub(r"sqrt\[(\d+)\]\(?([A-Za-z0-9_.]+)\)?", r"(\2)**(Rational(1,\1))", s)
    if "[" in s or "]" in s:
        raise ValueError("corchetes residuales sin traducir")

    ns = {
        "Abs": sp.Abs, "sign": sp.sign, "log": sp.log, "exp": sp.exp,
        "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "cot": sp.cot, "sec": sp.sec, "csc": sp.csc, "asin": sp.asin,
        "acos": sp.acos, "atan": sp.atan, "acot": sp.acot, "asec": sp.asec,
        "acsc": sp.acsc, "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
        "pi": sp.pi, "E": sp.E, "e": sp.E, "I": sp.I, "i": sp.I,
        "oo": sp.oo, "Rational": sp.Rational, "gamma": sp.gamma,
        "floor": sp.floor, "ceiling": sp.ceiling, "factorial": sp.factorial,
        # cbrt real: rama real del radical cubico (el motor usa la real)
        "real_cbrt": lambda z: sp.sign(z) * sp.Abs(z) ** sp.Rational(1, 3),
    }
    expr = sp.sympify(s, locals=ns, rational=True, convert_xor=False)
    return expr


# ------------------------------------------------------------------- puntos

# Racionales "seguros": mezcla de signos, no enteros pequenos solo, para
# despertar ramas par/impar/negativo (leccion reaudit-post-fixes).
POINT_POOL = [
    Fraction(3, 7), Fraction(-3, 7), Fraction(5, 2), Fraction(-5, 2),
    Fraction(1, 3), Fraction(-1, 3), Fraction(7, 5), Fraction(-7, 5),
    Fraction(2, 1), Fraction(-2, 1), Fraction(11, 13), Fraction(-11, 13),
    Fraction(9, 4), Fraction(-9, 4), Fraction(1, 5), Fraction(-1, 5),
]


def points_for(free_vars, n):
    pts = []
    for k in range(n):
        pt = {}
        for j, v in enumerate(sorted(free_vars, key=str)):
            pt[v] = sp.Rational(POINT_POOL[(k + 3 * j) % len(POINT_POOL)])
        pts.append(pt)
    return pts


def _defined(val, value_domain):
    """True si val es un numero finito valido en el dominio pedido."""
    if val is None:
        return False
    if val.has(sp.zoo) or val.has(sp.nan) or val.has(sp.oo) or val.has(-sp.oo):
        return False
    if value_domain == "real":
        try:
            im = sp.im(sp.simplify(val))
        except Exception:
            return False
        if im is None:
            return False
        if sp.simplify(im) != 0:
            return False
    return True


def compare(a_text, b_text, n_points=6, value_domain="real", verbose=True):
    try:
        A = to_sympy(a_text)
        B = to_sympy(b_text)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"SKIP_UNSUPPORTED: {exc}")
        return "SKIP_UNSUPPORTED", []

    free = sorted(A.free_symbols | B.free_symbols, key=str)
    if not free:
        pts = [{}]
    else:
        pts = points_for(free, n_points)

    rows = []
    for pt in pts:
        try:
            va = A.subs(pt)
            vb = B.subs(pt)
            va = sp.nsimplify(va) if va.free_symbols else sp.simplify(va)
            vb = sp.nsimplify(vb) if vb.free_symbols else sp.simplify(vb)
        except Exception as exc:  # noqa: BLE001
            rows.append(("SKIP_UNSUPPORTED", pt, None, None, str(exc)))
            continue
        da, db = _defined(va, value_domain), _defined(vb, value_domain)
        if not da and not db:
            rows.append(("SKIP_UNDEF", pt, va, vb, ""))
            continue
        if da != db:
            # Un lado definido y el otro no => el dominio CAMBIO. No es ruido:
            # es la senal de una condicion perdida o de un ensanchamiento.
            side = "A" if da else "B"
            rows.append(("DOMAIN_DIFF", pt, va, vb,
                         f"solo {side} definido en value-domain={value_domain}"))
            continue
        d = sp.simplify(va - vb)
        if d == 0:
            rows.append(("EQUAL", pt, va, vb, ""))
            continue
        # segundo intento: sympy .equals usa metodos mixtos
        try:
            if sp.Eq(va, vb).simplify() is sp.true or (va - vb).equals(0):
                rows.append(("EQUAL", pt, va, vb, ""))
                continue
        except Exception:  # noqa: BLE001
            pass
        # certificar la DIFERENCIA: si ambos son racionales exactos, es exacta
        if va.is_rational and vb.is_rational:
            rows.append(("DIFF", pt, va, vb, "racionales exactos"))
        else:
            try:
                gap = abs(sp.N(va - vb, 60))
                if gap > sp.Float("1e-40", 60):
                    rows.append(("DIFF_NUMERIC", pt, va, vb, f"gap~{gap}"))
                else:
                    rows.append(("UNDECIDED", pt, va, vb, f"gap~{gap}"))
            except Exception as exc:  # noqa: BLE001
                rows.append(("UNDECIDED", pt, va, vb, str(exc)))

    if verbose:
        for kind, pt, va, vb, note in rows:
            ptxt = ", ".join(f"{k}={v}" for k, v in pt.items()) or "(sin vars)"
            print(f"  {kind:15} {ptxt:28} A={va}  B={vb}  {note}")
    kinds = {r[0] for r in rows}
    if "DIFF" in kinds:
        return "DIFF", rows
    if "DIFF_NUMERIC" in kinds:
        return "DIFF_NUMERIC", rows
    if "DOMAIN_DIFF" in kinds:
        return "DOMAIN_DIFF", rows
    if kinds == {"SKIP_UNDEF"} or kinds <= {"SKIP_UNDEF", "SKIP_UNSUPPORTED", "UNDECIDED"}:
        return "INCONCLUSIVE", rows
    return "EQUAL", rows


def engine_eval(expr, axes=None):
    cmd = [CLI, "eval", expr, "--format", "json", "--no-pretty"]
    if axes:
        cmd += axes
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"CLI fallo: {out.stderr.strip()[:400]}")
    return json.loads(out.stdout)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check")
    c.add_argument("expr")
    c.add_argument("--n", type=int, default=6)
    c.add_argument("--value-domain", default="real")
    c.add_argument("--axes", nargs=argparse.REMAINDER, default=[])

    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--value-domain", default="real")

    args = ap.parse_args()
    if args.cmd == "check":
        env = engine_eval(args.expr, args.axes)
        print(f"input : {env['input']}")
        print(f"result: {env['result']}")
        if env.get("required_conditions"):
            print(f"required_conditions: {env['required_conditions']}")
        if env.get("warnings"):
            print(f"warnings: {env['warnings']}")
        verdict, _ = compare(args.expr, env["result"], args.n, args.value_domain)
    else:
        verdict, _ = compare(args.a, args.b, args.n, args.value_domain)
    print(f"VERDICT={verdict}")
    sys.exit({"EQUAL": 0, "DIFF": 2, "DIFF_NUMERIC": 2,
              "DOMAIN_DIFF": 2}.get(verdict, 3))


if __name__ == "__main__":
    main()
