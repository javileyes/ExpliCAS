//! Support for denominator rationalization patterns with roots.

use crate::expr_nary::{AddView, Sign};
use crate::root_forms::{extract_cube_root_base, extract_square_root_base};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootDenRationalizeRewrite {
    pub rewritten: ExprId,
    pub kind: RootDenRationalizeRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootDenRationalizeRewriteKind {
    ShiftedUnitSquareExactQuotient,
    LinearSqrtDen,
    SumOfSqrtsDen,
    CubeRootDen,
}

fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

fn expr_eq(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
}

fn try_extract_linear_two_times_candidate(ctx: &Context, term: ExprId) -> Option<ExprId> {
    let Expr::Mul(left, right) = ctx.get(term) else {
        return None;
    };
    match (ctx.get(*left), ctx.get(*right)) {
        (Expr::Number(n), _) if *n == num_rational::BigRational::from_integer(2.into()) => {
            Some(*right)
        }
        (_, Expr::Number(n)) if *n == num_rational::BigRational::from_integer(2.into()) => {
            Some(*left)
        }
        _ => None,
    }
}

fn square_term_matches_candidate(
    ctx: &mut Context,
    square_term: ExprId,
    candidate: ExprId,
) -> bool {
    let two = ctx.num(2);
    let candidate_sq = ctx.add(Expr::Pow(candidate, two));
    if expr_eq(ctx, square_term, candidate_sq) {
        return true;
    }

    if let Some(radicand) = extract_square_root_base(ctx, candidate) {
        if expr_eq(ctx, square_term, radicand) {
            return true;
        }
    }

    let one = ctx.num(1);
    match ctx.get(candidate).clone() {
        Expr::Div(num, den) if expr_eq(ctx, num, one) => {
            if let Some(radicand) = extract_square_root_base(ctx, den) {
                let reciprocal = ctx.add(Expr::Div(one, radicand));
                if expr_eq(ctx, square_term, reciprocal) {
                    return true;
                }
                let neg_one = ctx.add(Expr::Number(num_rational::BigRational::from_integer(
                    (-1).into(),
                )));
                let inverse_pow = ctx.add(Expr::Pow(radicand, neg_one));
                if expr_eq(ctx, square_term, inverse_pow) {
                    return true;
                }
            }
        }
        Expr::Pow(base, exp) => match ctx.get(exp) {
            Expr::Number(n) if *n == num_rational::BigRational::new((-1).into(), 2.into()) => {
                let reciprocal = ctx.add(Expr::Div(one, base));
                if expr_eq(ctx, square_term, reciprocal) {
                    return true;
                }
                let neg_one = ctx.add(Expr::Number(num_rational::BigRational::from_integer(
                    (-1).into(),
                )));
                let inverse_pow = ctx.add(Expr::Pow(base, neg_one));
                if expr_eq(ctx, square_term, inverse_pow) {
                    return true;
                }
            }
            _ => {}
        },
        _ => {}
    }

    false
}

fn try_extract_square_root_linear_binomial_candidate(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Sign)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    for idx in 0..2 {
        let (candidate, candidate_sign) = terms[idx];
        let (square_term, square_sign) = terms[1 - idx];
        if square_sign != Sign::Pos {
            continue;
        }
        if extract_square_root_base(ctx, candidate).is_none() {
            continue;
        }
        if square_term_matches_candidate(ctx, square_term, candidate) {
            return Some((candidate, candidate_sign));
        }
    }

    None
}

fn matches_shifted_unit_square_candidate(
    ctx: &mut Context,
    expr: ExprId,
    candidate: ExprId,
    linear_sign: Sign,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 3 {
        return false;
    }

    let mut square_term = None;
    let mut linear_term = None;
    let mut constant_one = false;

    for (term, sign) in terms {
        if sign == Sign::Pos && matches!(ctx.get(term), Expr::Number(n) if n.is_one()) {
            if constant_one {
                return false;
            }
            constant_one = true;
            continue;
        }

        if let Some(linear_candidate) = try_extract_linear_two_times_candidate(ctx, term) {
            if linear_term.is_some()
                || sign != linear_sign
                || !expr_eq(ctx, linear_candidate, candidate)
            {
                return false;
            }
            linear_term = Some(term);
            continue;
        }

        if sign == Sign::Pos {
            if square_term.is_some() {
                return false;
            }
            square_term = Some(term);
            continue;
        }

        return false;
    }

    constant_one
        && linear_term.is_some()
        && square_term
            .map(|term| square_term_matches_candidate(ctx, term, candidate))
            .unwrap_or(false)
}

/// Try to close an exact quotient before conjugate rationalization:
/// `(t^2 ± 2t + 1) / (t^2 ± t) -> 1 ± 1/t`
/// when `t` is a square-root-like atom.
pub fn try_rewrite_shifted_unit_square_over_linear_sqrt_den_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RootDenRationalizeRewrite> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(n, d) => (*n, *d),
        _ => return None,
    };

    let (candidate, linear_sign) =
        try_extract_square_root_linear_binomial_candidate(ctx, denominator)?;
    if !matches_shifted_unit_square_candidate(ctx, numerator, candidate, linear_sign) {
        return None;
    }

    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, candidate));
    let rewritten = match linear_sign {
        Sign::Pos => ctx.add(Expr::Add(reciprocal, one)),
        Sign::Neg => ctx.add(Expr::Sub(one, reciprocal)),
    };

    Some(RootDenRationalizeRewrite {
        rewritten,
        kind: RootDenRationalizeRewriteKind::ShiftedUnitSquareExactQuotient,
    })
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
        kind: RootDenRationalizeRewriteKind::LinearSqrtDen,
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
        kind: RootDenRationalizeRewriteKind::SumOfSqrtsDen,
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
        kind: RootDenRationalizeRewriteKind::CubeRootDen,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_rationalize_cube_root_den_expr, try_rewrite_rationalize_linear_sqrt_den_expr,
        try_rewrite_rationalize_sum_of_sqrts_den_expr,
        try_rewrite_shifted_unit_square_over_linear_sqrt_den_expr,
    };
    use cas_ast::ordering::compare_expr;
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

    #[test]
    fn rewrites_shifted_unit_square_over_linear_sqrt_denominator_exactly() {
        let mut ctx = Context::new();
        let expr = parse("(2*sqrt(u) + u + 1)/(sqrt(u) + u)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_shifted_unit_square_over_linear_sqrt_den_expr(&mut ctx, expr)
            .expect("rewrite");
        let expected = parse("1/sqrt(u) + 1", &mut ctx).expect("expected");
        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }
}
