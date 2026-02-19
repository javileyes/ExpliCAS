//! Root-shape helpers over AST expressions.

use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

/// Extract `(radicand, index)` when `expr` is a root-like form.
///
/// Recognizes:
/// - `sqrt(x)` as `(x, 2)`
/// - `x^(1/k)` where exponent is numeric `1/k`
/// - `x^(1/k)` where exponent is structural `Div(1, k)`
pub fn extract_root_base_and_index(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let sqrt_arg = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    };
    if let Some(arg) = sqrt_arg {
        return Some((arg, ctx.num(2)));
    }

    let (base, exp) = match ctx.get(expr) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };

    if let Some(n) = crate::numeric::as_number(ctx, exp) {
        if !n.is_integer() && n.numer().is_one() {
            let k_expr = ctx.add(Expr::Number(BigRational::from_integer(n.denom().clone())));
            return Some((base, k_expr));
        }
    }

    if let Expr::Div(num_exp, den_exp) = ctx.get(exp) {
        if let Some(n) = crate::numeric::as_number(ctx, *num_exp) {
            if n.is_one() {
                return Some((base, *den_exp));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly_compare::poly_eq;
    use cas_parser::parse;

    #[test]
    fn extract_from_sqrt_function() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x)", &mut ctx).expect("parse");
        let (radicand, index) = extract_root_base_and_index(&mut ctx, expr).expect("root");

        let x = parse("x", &mut ctx).expect("parse x");
        let two = parse("2", &mut ctx).expect("parse 2");
        assert!(poly_eq(&ctx, radicand, x));
        assert!(poly_eq(&ctx, index, two));
    }

    #[test]
    fn extract_from_fractional_power() {
        let mut ctx = Context::new();
        let expr = parse("x^(1/3)", &mut ctx).expect("parse");
        let (radicand, index) = extract_root_base_and_index(&mut ctx, expr).expect("root");

        let x = parse("x", &mut ctx).expect("parse x");
        let three = parse("3", &mut ctx).expect("parse 3");
        assert!(poly_eq(&ctx, radicand, x));
        assert!(poly_eq(&ctx, index, three));
    }

    #[test]
    fn reject_non_unit_numerator_exponent() {
        let mut ctx = Context::new();
        let expr = parse("x^(2/3)", &mut ctx).expect("parse");
        assert!(extract_root_base_and_index(&mut ctx, expr).is_none());
    }
}
