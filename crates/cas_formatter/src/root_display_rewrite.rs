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

use crate::{Context, Expr, ExprId};
use num_traits::ToPrimitive;

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

fn radical_call(ctx: &mut Context, radicand: ExprId, index: i64) -> ExprId {
    if index == 2 {
        let name = ctx.intern_symbol("sqrt");
        return ctx.add_raw(Expr::Function(name, vec![radicand]));
    }
    let name = ctx.intern_symbol("root");
    let index_node = ctx.num(index);
    ctx.add_raw(Expr::Function(name, vec![radicand, index_node]))
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
            match fractional_exponent_parts(ctx, exponent) {
                Some((numerator, denominator)) if denominator > 1 && numerator > 0 => {
                    let radicand = if numerator == 1 {
                        new_base
                    } else {
                        let power = ctx.num(numerator);
                        ctx.add_raw(Expr::Pow(new_base, power))
                    };
                    radical_call(ctx, radicand, denominator)
                }
                // Exponente no fraccionario, o numerador NEGATIVO: el LaTeX tampoco
                // los presenta como raíz (`x^(-1/2)` no se vuelve raíz de recíproco).
                _ if new_base == base => id,
                _ => ctx.add_raw(Expr::Pow(new_base, exponent)),
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
        assert_eq!(rendered("x^(1/3)"), "root(x, 3)");
        assert_eq!(rendered("(x + 1)^(1/3)"), "root(x + 1, 3)");
        assert_eq!(rendered("x^(2/3)"), "root(x^2, 3)");
        assert_eq!(rendered("x^(7/6)"), "root(x^7, 6)");
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
        assert_eq!(rendered("1/(x^(1/3) + 1)"), "1 / (root(x, 3) + 1)");
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
