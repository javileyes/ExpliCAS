//! Reescritura de PRESENTACIÓN: `x^(p/q)` como `sqrt(x)` / `root(x, q)`.
//!
//! El texto plano del resultado lo imprime `DisplayExpr`, que no tiene perilla de
//! estilo (se construye en ~2000 sitios, así que añadirle un campo no es una
//! opción). En vez de parametrizar el renderizador, se reescribe el ÁRBOL a los
//! nodos que ese renderizador ya sabe imprimir como raíz — el mismo truco que usa
//! la normalización de display, y sin tocar ningún llamante.
//!
//! Es una transformación de DISPLAY sobre un contexto de scratch: el valor no
//! cambia (`x^(1/2)` y `sqrt(x)` son la misma función) y la forma interna sigue
//! siendo la potencia. Reproduce EXACTAMENTE las condiciones de `render_as_root`
//! del renderizador LaTeX (`latex_core`), que es lo que hace que las dos
//! superficies digan lo mismo; si esas condiciones cambian, cambian las dos.
//!
//! Con exponente IMPROPIO (p > q) se extrae la parte entera: `√(x³)` → `x·√x`,
//! `√(5³)` → `5·√5`. Es la «forma simplificada del radical» que enseña la escuela
//! (sacar factores de la raíz) y la MISMA canónica que el motor ya aplica a los
//! surds numéricos (`sqrt(125)` → `5·√5`): sin la extracción, la cantidad idéntica
//! salía en dos formas según qué camino la produjera. La partición vive en
//! [`split_improper_fractional_exponent`], compartida con el LaTeX.

use crate::{Context, Expr, ExprId};
use num_traits::ToPrimitive;

/// Reduce `p/q` por su gcd y devuelve `(k, r, q)` con `p/q = k + r/q`, `0 ≤ r < q`.
///
/// `None` cuando la fracción no es un exponente de raíz utilizable: `p ≤ 0`
/// (los negativos se quedan como potencia, igual que en LaTeX) o `q ≤ 1` tras
/// reducir (exponente entero disfrazado, p.ej. un `Div(6, 2)` sin plegar — que
/// además NO debe presentarse como `√(x⁶)`: esa forma es par-en-el-radicando y
/// vale `|x³|`, no `x³`).
pub(crate) fn split_improper_fractional_exponent(
    numerator: i64,
    denominator: i64,
) -> Option<(i64, i64, i64)> {
    if numerator <= 0 || denominator <= 1 {
        return None;
    }
    let gcd = {
        let (mut a, mut b) = (numerator, denominator);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let (p, q) = (numerator / gcd, denominator / gcd);
    if q <= 1 {
        return None;
    }
    Some((p / q, p % q, q))
}

/// `n^k` exacto para el pliegue del factor extraído con base NUMÉRICA positiva
/// (`2^(5/2)` → `4·√2`, no `2^2·√2`). El tope de `k` es una red anti-explosión:
/// más allá se deja la potencia simbólica, que sigue siendo correcta.
pub(crate) fn fold_positive_rational_power(
    n: &num_rational::BigRational,
    k: i64,
) -> Option<num_rational::BigRational> {
    if !(2..=64).contains(&k) || !num_traits::Signed::is_positive(n) {
        return None;
    }
    let mut acc = n.clone();
    for _ in 1..k {
        acc *= n;
    }
    Some(acc)
}

/// `(numerador, denominador)` de un exponente literal fraccionario, en las MISMAS
/// dos formas que acepta el renderizador LaTeX: `Number` no entero y `Div` de dos
/// enteros. Devuelve `None` para cualquier otra cosa (exponente simbólico, entero).
fn fractional_exponent_parts(ctx: &Context, exponent: ExprId) -> Option<(i64, i64)> {
    match ctx.get(exponent) {
        Expr::Number(n) => {
            if n.is_integer() {
                return None;
            }
            Some((n.numer().to_i64()?, n.denom().to_i64()?))
        }
        Expr::Div(numerator, denominator) => {
            let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*numerator), ctx.get(*denominator))
            else {
                return None;
            };
            if !n.is_integer() || !d.is_integer() {
                return None;
            }
            Some((n.numer().to_i64()?, d.numer().to_i64()?))
        }
        _ => None,
    }
}

/// La llamada de raíz con la ORTOGRAFÍA que el motor ya usa en sus resultados:
/// `sqrt` para índice 2 y `cbrt` para índice 3 (ambas `BuiltinFn`; `cbrt` aparece
/// por ejemplo en `integrate(1/(x^3-2), x)`), `root(r, n)` para el resto. Emitir
/// `root(r, 3)` habría metido una SEGUNDA ortografía para la raíz cúbica, que es
/// justo la clase de defecto que este frente persigue.
fn radical_call(ctx: &mut Context, radicand: ExprId, index: i64) -> ExprId {
    match index {
        2 => {
            let name = ctx.intern_symbol("sqrt");
            ctx.add_raw(Expr::Function(name, vec![radicand]))
        }
        3 => {
            let name = ctx.intern_symbol("cbrt");
            ctx.add_raw(Expr::Function(name, vec![radicand]))
        }
        _ => {
            let name = ctx.intern_symbol("root");
            let index_node = ctx.num(index);
            ctx.add_raw(Expr::Function(name, vec![radicand, index_node]))
        }
    }
}

/// Reescribe cada `Pow(base, p/q)` del árbol como la llamada de raíz equivalente.
///
/// Devuelve el id ORIGINAL cuando nada cambia, así que el caso común no reconstruye
/// nada. La reconstrucción usa `add_raw`: `add` canonicaliza (reordena `Add`/`Mul`
/// por rango) y eso reordenaría términos del resultado que hoy salen en otro orden
/// — un cambio de presentación que este paso no debe hacer.
pub fn rewrite_fractional_powers_as_roots(ctx: &mut Context, id: ExprId) -> ExprId {
    let expr = ctx.get(id).clone();
    match expr {
        Expr::Pow(base, exponent) => {
            let new_base = rewrite_fractional_powers_as_roots(ctx, base);
            let split = fractional_exponent_parts(ctx, exponent)
                .and_then(|(p, q)| split_improper_fractional_exponent(p, q));
            match split {
                // Fracción propia: la raíz tal cual (`x^(2/3)` → `∛(x²)`).
                Some((0, r, q)) => {
                    let radicand = if r == 1 {
                        new_base
                    } else {
                        let power = ctx.num(r);
                        ctx.add_raw(Expr::Pow(new_base, power))
                    };
                    radical_call(ctx, radicand, q)
                }
                // Impropia: extraer la parte entera (`x^(7/2)` → `x³·√x`), salvo
                // base numérica NEGATIVA (la extracción movería el signo fuera de
                // la raíz y cambiaría cómo se agrupa; se queda en potencia).
                Some((k, r, q)) => {
                    let base_is_negative_literal = matches!(
                        ctx.get(new_base),
                        Expr::Number(n) if num_traits::Signed::is_negative(n)
                    );
                    if base_is_negative_literal {
                        if new_base == base {
                            return id;
                        }
                        return ctx.add_raw(Expr::Pow(new_base, exponent));
                    }
                    let factor = if let Expr::Number(n) = ctx.get(new_base) {
                        fold_positive_rational_power(&n.clone(), k).map(Expr::Number)
                    } else {
                        None
                    };
                    let factor = match factor {
                        Some(folded) => ctx.add_raw(folded),
                        None if k == 1 => new_base,
                        None => {
                            let power = ctx.num(k);
                            ctx.add_raw(Expr::Pow(new_base, power))
                        }
                    };
                    let radicand = if r == 1 {
                        new_base
                    } else {
                        let power = ctx.num(r);
                        ctx.add_raw(Expr::Pow(new_base, power))
                    };
                    let radical = radical_call(ctx, radicand, q);
                    ctx.add_raw(Expr::Mul(factor, radical))
                }
                // Exponente no fraccionario, entero disfrazado, o numerador
                // NEGATIVO: el LaTeX tampoco los presenta como raíz
                // (`x^(-1/2)` no se vuelve raíz de recíproco).
                None if new_base == base => id,
                None => ctx.add_raw(Expr::Pow(new_base, exponent)),
            }
        }
        Expr::Add(lhs, rhs) => map_binary(ctx, id, lhs, rhs, Expr::Add),
        Expr::Sub(lhs, rhs) => map_binary(ctx, id, lhs, rhs, Expr::Sub),
        Expr::Mul(lhs, rhs) => map_binary(ctx, id, lhs, rhs, Expr::Mul),
        Expr::Div(lhs, rhs) => map_binary(ctx, id, lhs, rhs, Expr::Div),
        Expr::Neg(inner) => {
            let rewritten = rewrite_fractional_powers_as_roots(ctx, inner);
            if rewritten == inner {
                id
            } else {
                ctx.add_raw(Expr::Neg(rewritten))
            }
        }
        Expr::Hold(inner) => {
            let rewritten = rewrite_fractional_powers_as_roots(ctx, inner);
            if rewritten == inner {
                id
            } else {
                ctx.add_raw(Expr::Hold(rewritten))
            }
        }
        Expr::Function(fn_id, args) => {
            let rewritten: Vec<ExprId> = args
                .iter()
                .map(|arg| rewrite_fractional_powers_as_roots(ctx, *arg))
                .collect();
            if rewritten == args {
                id
            } else {
                ctx.add_raw(Expr::Function(fn_id, rewritten))
            }
        }
        Expr::Matrix { rows, cols, data } => {
            let rewritten: Vec<ExprId> = data
                .iter()
                .map(|cell| rewrite_fractional_powers_as_roots(ctx, *cell))
                .collect();
            if rewritten == data {
                id
            } else {
                ctx.add_raw(Expr::Matrix {
                    rows,
                    cols,
                    data: rewritten,
                })
            }
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => id,
    }
}

fn map_binary(
    ctx: &mut Context,
    id: ExprId,
    lhs: ExprId,
    rhs: ExprId,
    build: fn(ExprId, ExprId) -> Expr,
) -> ExprId {
    let new_lhs = rewrite_fractional_powers_as_roots(ctx, lhs);
    let new_rhs = rewrite_fractional_powers_as_roots(ctx, rhs);
    if new_lhs == lhs && new_rhs == rhs {
        id
    } else {
        ctx.add_raw(build(new_lhs, new_rhs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DisplayExpr;

    fn rendered(source: &str) -> String {
        let mut ctx = Context::new();
        let parsed = cas_parser::parse(source, &mut ctx).expect("parse");
        let rewritten = rewrite_fractional_powers_as_roots(&mut ctx, parsed);
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewritten
            }
        )
    }

    #[test]
    fn fractional_powers_become_radical_calls() {
        assert_eq!(rendered("x^(1/2)"), "sqrt(x)");
        assert_eq!(rendered("x^(1/3)"), "cbrt(x)");
        assert_eq!(rendered("(x + 1)^(1/3)"), "cbrt(x + 1)");
        assert_eq!(rendered("x^(2/3)"), "cbrt(x^2)");
    }

    #[test]
    fn improper_exponents_extract_the_whole_part() {
        // «Sacar factores de la raíz», la forma simplificada escolar — y la misma
        // canónica que el motor ya aplica a los surds numéricos
        // (`sqrt(125)` → `5·sqrt(5)`).
        assert_eq!(rendered("x^(3/2)"), "x * sqrt(x)");
        assert_eq!(rendered("x^(7/2)"), "x^3 * sqrt(x)");
        assert_eq!(rendered("x^(5/3)"), "x * cbrt(x^2)");
        assert_eq!(rendered("x^(7/6)"), "x * root(x, 6)");
        assert_eq!(rendered("(x + 1)^(5/2)"), "(x + 1)^2 * sqrt(x + 1)");
        // Base numérica positiva: el factor se PLIEGA.
        assert_eq!(rendered("5^(3/2)"), "5 * sqrt(5)");
        assert_eq!(rendered("2^(5/2)"), "4 * sqrt(2)");
        assert_eq!(rendered("2^(7/2)"), "8 * sqrt(2)");
    }

    #[test]
    fn negative_literal_base_with_whole_part_stays_a_power() {
        // Extraer movería el signo fuera de la raíz y cambiaría la agrupación.
        let mut ctx = Context::new();
        let minus_two = ctx.num(-2);
        let exponent = ctx.rational(5, 3);
        let pow = ctx.add_raw(Expr::Pow(minus_two, exponent));
        assert_eq!(rewrite_fractional_powers_as_roots(&mut ctx, pow), pow);
    }

    #[test]
    fn disguised_integer_exponent_is_not_presented_as_a_radical() {
        // `Div(6, 2)` sin plegar: `√(x⁶)` valdría `|x³|`, no `x³` — potencia.
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let six = ctx.num(6);
        let two = ctx.num(2);
        let exponent = ctx.add_raw(Expr::Div(six, two));
        let pow = ctx.add_raw(Expr::Pow(x, exponent));
        assert_eq!(rewrite_fractional_powers_as_roots(&mut ctx, pow), pow);
    }

    #[test]
    fn non_root_shapes_are_left_alone() {
        // Exponente entero, simbólico y numerador negativo: el LaTeX tampoco los
        // presenta como raíz, así que el texto no debe inventarse una.
        assert_eq!(rendered("x^2"), "x^2");
        assert_eq!(rendered("x^n"), "x^n");
        assert_eq!(rendered("x^(-1/2)"), "x^(-1 / 2)");
        // Los tests corren en modo ASCII: el símbolo de producto es " * ".
        assert_eq!(rendered("2*x + 1"), "2 * x + 1");
    }

    #[test]
    fn rewrite_reaches_nested_positions() {
        assert_eq!(rendered("sin(x^(1/2))"), "sin(sqrt(x))");
        assert_eq!(rendered("1/(x^(1/3) + 1)"), "1 / (cbrt(x) + 1)");
        assert_eq!(rendered("x^(1/5)"), "root(x, 5)");
        // El ORDEN de los sumandos no lo decide esta reescritura: `DisplayExpr`
        // ordena los términos por su propio criterio al imprimir. Se fija aquí
        // para que quede dicho que la reescritura no lo toca ni lo puede tocar.
        assert_eq!(rendered("y + x^(1/2)"), "sqrt(x) + y");
        assert_eq!(rendered("x^(1/2) + y"), "sqrt(x) + y");
    }

    #[test]
    fn unchanged_tree_keeps_its_identity() {
        let mut ctx = Context::new();
        let parsed = cas_parser::parse("x^2 + sin(y)", &mut ctx).expect("parse");
        assert_eq!(
            rewrite_fractional_powers_as_roots(&mut ctx, parsed),
            parsed,
            "sin potencias fraccionarias no debe reconstruirse nada"
        );
    }
}
