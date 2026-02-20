use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::cmp::Ordering;

/// Extract trig function name and argument from `sin(arg)` / `cos(arg)`.
pub fn extract_sin_cos_fn_arg(ctx: &Context, expr: ExprId) -> Option<(&'static str, ExprId)> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let builtin = ctx.builtin_of(*fn_id);
        if args.len() == 1 {
            return match builtin {
                Some(BuiltinFn::Sin) => Some(("sin", args[0])),
                Some(BuiltinFn::Cos) => Some(("cos", args[0])),
                _ => None,
            };
        }
    }
    None
}

/// Extract linear coefficients if both args are linear multiples of a common base.
/// Returns `(base, coef_a, coef_b)` where `a = coef_a * base`, `b = coef_b * base`.
pub fn extract_linear_coefficients(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
) -> Option<(ExprId, BigRational, BigRational)> {
    let (coef_a, base_a) = extract_coef_and_base(ctx, a);
    let (coef_b, base_b) = extract_coef_and_base(ctx, b);

    if cas_ast::ordering::compare_expr(ctx, base_a, base_b) == Ordering::Equal {
        Some((base_a, coef_a, coef_b))
    } else {
        None
    }
}

/// Extract coefficient and base from an expression.
/// `n*t -> (n, t)`, `t -> (1, t)`, `-t -> (-1, t)`.
pub fn extract_coef_and_base(ctx: &Context, expr: ExprId) -> (BigRational, ExprId) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if let Expr::Number(n) = ctx.get(l) {
                return (n.clone(), r);
            }
            if let Expr::Number(n) = ctx.get(r) {
                return (n.clone(), l);
            }
            (BigRational::from_integer(1.into()), expr)
        }
        Expr::Number(n) => (n.clone(), expr),
        Expr::Neg(inner) => {
            let (inner_coef, inner_base) = extract_coef_and_base(ctx, *inner);
            (-inner_coef, inner_base)
        }
        _ => (BigRational::from_integer(1.into()), expr),
    }
}

/// Build `coef * base`, simplifying coefficients `0`, `1` and `-1`.
pub fn build_coef_times_base(ctx: &mut Context, coef: &BigRational, base: ExprId) -> ExprId {
    if coef.is_zero() {
        ctx.num(0)
    } else if coef.is_one() {
        base
    } else if *coef == -BigRational::one() {
        ctx.add(Expr::Neg(base))
    } else {
        let coef_id = ctx.add(Expr::Number(coef.clone()));
        ctx.add(Expr::Mul(coef_id, base))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn extract_sin_cos_fn_arg_matches_supported_functions() {
        let mut ctx = Context::new();
        let sin_x = parse("sin(x)", &mut ctx).expect("parse sin");
        let tan_x = parse("tan(x)", &mut ctx).expect("parse tan");

        assert_eq!(extract_sin_cos_fn_arg(&ctx, sin_x).map(|(n, _)| n), Some("sin"));
        assert!(extract_sin_cos_fn_arg(&ctx, tan_x).is_none());
    }

    #[test]
    fn extract_linear_coefficients_requires_common_base() {
        let mut ctx = Context::new();
        let a = parse("2*x", &mut ctx).expect("parse a");
        let b = parse("3*x", &mut ctx).expect("parse b");
        let c = parse("4*y", &mut ctx).expect("parse c");

        let same = extract_linear_coefficients(&ctx, a, b);
        let diff = extract_linear_coefficients(&ctx, a, c);

        assert!(same.is_some());
        assert!(diff.is_none());
    }

    #[test]
    fn build_coef_times_base_simplifies_unit_and_zero_coefficients() {
        let mut ctx = Context::new();
        let base = parse("x", &mut ctx).expect("parse base");

        let zero = build_coef_times_base(&mut ctx, &BigRational::from_integer(0.into()), base);
        let one = build_coef_times_base(&mut ctx, &BigRational::from_integer(1.into()), base);
        let minus_one = build_coef_times_base(&mut ctx, &BigRational::from_integer((-1).into()), base);

        let expected_zero = parse("0", &mut ctx).expect("parse 0");
        let expected_one = parse("x", &mut ctx).expect("parse x");
        let expected_minus_one = parse("-x", &mut ctx).expect("parse -x");

        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, zero, expected_zero),
            Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, one, expected_one),
            Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, minus_one, expected_minus_one),
            Ordering::Equal
        );
    }
}
