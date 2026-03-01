//! Support for denominator rationalization patterns with roots.

use crate::root_forms::{extract_cube_root_base, extract_square_root_base};
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootDenRationalizeRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

/// Try to rationalize linear square-root denominator:
/// `num / (sqrt(t) ± c) -> num*(sqrt(t) ∓ c) / (t - c^2)`.
pub fn try_rewrite_rationalize_linear_sqrt_den_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RootDenRationalizeRewrite> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(n, d) => (*n, *d),
        _ => return None,
    };

    let (sqrt_arg, const_part, is_plus) = match ctx.get(denominator) {
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            if let Some(arg) = extract_square_root_base(ctx, l) {
                (arg, r, true)
            } else if let Some(arg) = extract_square_root_base(ctx, r) {
                (arg, l, true)
            } else {
                return None;
            }
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            if let Some(arg) = extract_square_root_base(ctx, l) {
                (arg, r, false)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    if !matches!(ctx.get(const_part), Expr::Number(_) | Expr::Variable(_)) {
        return None;
    }

    let half = ctx.rational(1, 2);
    let sqrt_t = ctx.add(Expr::Pow(sqrt_arg, half));
    let conjugate_num = if is_plus {
        ctx.add(Expr::Sub(sqrt_t, const_part))
    } else {
        ctx.add(Expr::Add(sqrt_t, const_part))
    };
    let new_num = ctx.add(Expr::Mul(numerator, conjugate_num));
    let two = ctx.num(2);
    let c_squared = ctx.add(Expr::Pow(const_part, two));
    let new_den = ctx.add(Expr::Sub(sqrt_arg, c_squared));
    let rewritten = ctx.add(Expr::Div(new_num, new_den));

    Some(RootDenRationalizeRewrite {
        rewritten,
        desc: "Rationalize: multiply by conjugate",
    })
}

/// Try to rationalize sum/difference of two square roots denominator:
/// `k/(sqrt(p) ± sqrt(q)) -> k*(sqrt(p) ∓ sqrt(q))/(p-q)`.
pub fn try_rewrite_rationalize_sum_of_sqrts_den_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RootDenRationalizeRewrite> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(n, d) => (*n, *d),
        _ => return None,
    };

    let (sqrt_p_arg, sqrt_q_arg, is_plus) = match ctx.get(denominator) {
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            (
                extract_square_root_base(ctx, l)?,
                extract_square_root_base(ctx, r)?,
                true,
            )
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            (
                extract_square_root_base(ctx, l)?,
                extract_square_root_base(ctx, r)?,
                false,
            )
        }
        _ => return None,
    };

    let half = ctx.rational(1, 2);
    let sqrt_p = ctx.add(Expr::Pow(sqrt_p_arg, half));
    let half2 = ctx.rational(1, 2);
    let sqrt_q = ctx.add(Expr::Pow(sqrt_q_arg, half2));
    let conjugate = if is_plus {
        ctx.add(Expr::Sub(sqrt_p, sqrt_q))
    } else {
        ctx.add(Expr::Add(sqrt_p, sqrt_q))
    };

    let new_num = ctx.add(Expr::Mul(numerator, conjugate));
    let new_den = ctx.add(Expr::Sub(sqrt_p_arg, sqrt_q_arg));
    let rewritten = ctx.add(Expr::Div(new_num, new_den));

    Some(RootDenRationalizeRewrite {
        rewritten,
        desc: "Rationalize: (sqrt(p)±sqrt(q)) multiply by conjugate",
    })
}

/// Try to rationalize cube-root denominator:
/// `num/(1 ± u^(1/3)) -> num*(1 ∓ u^(1/3) + u^(2/3)) / (1 ± u)`.
pub fn try_rewrite_rationalize_cube_root_den_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RootDenRationalizeRewrite> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(n, d) => (*n, *d),
        _ => return None,
    };

    let (cbrt_base, is_plus) = match ctx.get(denominator) {
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            if is_one(ctx, l) {
                (extract_cube_root_base(ctx, r)?, true)
            } else if is_one(ctx, r) {
                (extract_cube_root_base(ctx, l)?, true)
            } else {
                return None;
            }
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            if is_one(ctx, l) {
                (extract_cube_root_base(ctx, r)?, false)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    let one = ctx.num(1);
    let one_third = ctx.rational(1, 3);
    let two_thirds = ctx.rational(2, 3);
    let r = ctx.add(Expr::Pow(cbrt_base, one_third));
    let r_squared = ctx.add(Expr::Pow(cbrt_base, two_thirds));
    let factor = if is_plus {
        let one_minus_r = ctx.add(Expr::Sub(one, r));
        ctx.add(Expr::Add(one_minus_r, r_squared))
    } else {
        let one_plus_r = ctx.add(Expr::Add(one, r));
        ctx.add(Expr::Add(one_plus_r, r_squared))
    };
    let new_num = ctx.add(Expr::Mul(numerator, factor));
    let new_den = if is_plus {
        let one2 = ctx.num(1);
        ctx.add(Expr::Add(one2, cbrt_base))
    } else {
        let one2 = ctx.num(1);
        ctx.add(Expr::Sub(one2, cbrt_base))
    };
    let rewritten = ctx.add(Expr::Div(new_num, new_den));

    Some(RootDenRationalizeRewrite {
        rewritten,
        desc: "Rationalize: cube root denominator via sum of cubes",
    })
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_rationalize_cube_root_den_expr, try_rewrite_rationalize_linear_sqrt_den_expr,
        try_rewrite_rationalize_sum_of_sqrts_den_expr,
    };
    use cas_ast::{Context, Expr};
    use cas_parser::parse;

    #[test]
    fn rewrites_linear_sqrt_denominator() {
        let mut ctx = Context::new();
        let expr = parse("1/(sqrt(x)+1)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_rationalize_linear_sqrt_den_expr(&mut ctx, expr);
        assert!(rewrite.is_some());
    }

    #[test]
    fn rewrites_sum_of_sqrts_denominator() {
        let mut ctx = Context::new();
        let expr = parse("3/(sqrt(2)+sqrt(5))", &mut ctx).expect("parse");
        let rewrite = try_rewrite_rationalize_sum_of_sqrts_den_expr(&mut ctx, expr);
        assert!(rewrite.is_some());
    }

    #[test]
    fn rewrites_cube_root_denominator() {
        let mut ctx = Context::new();
        let expr = parse("1/(1+x^(1/3))", &mut ctx).expect("parse");
        let rewrite = try_rewrite_rationalize_cube_root_den_expr(&mut ctx, expr);
        assert!(rewrite.is_some());
    }

    #[test]
    fn cube_root_rewrite_returns_division() {
        let mut ctx = Context::new();
        let expr = parse("1/(1-x^(1/3))", &mut ctx).expect("parse");
        let rewrite = try_rewrite_rationalize_cube_root_den_expr(&mut ctx, expr).expect("rewrite");
        assert!(matches!(ctx.get(rewrite.rewritten), Expr::Div(_, _)));
    }
}
