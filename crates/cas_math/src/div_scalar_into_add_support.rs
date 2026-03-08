//! Support for distributing a scalar rational factor into an additive expression
//! by factoring a common coefficient from terms.

use crate::build::mul2_raw;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

/// Rewrite payload for scalar-into-add distribution.
#[derive(Debug, Clone, Copy)]
pub struct DivScalarIntoAddRewrite {
    pub rewritten: ExprId,
    pub kind: DivScalarIntoAddRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivScalarIntoAddRewriteKind {
    AllTermsCancel,
    FactorCommonCoefficientFromSum,
}

/// Try to rewrite `Mul(Number(frac), Add(...))` by factoring a common
/// coefficient in the sum and absorbing it into `frac`.
///
/// This is a structural simplification step used by the engine's
/// `DivScalarIntoAddRule`.
pub fn try_rewrite_div_scalar_into_add_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<DivScalarIntoAddRewrite> {
    // Match Mul(Number(frac), Add(...)) or Mul(Add(...), Number(frac))
    let (frac_val, add_id) = match ctx.get(expr) {
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            match (ctx.get(l), ctx.get(r)) {
                (Expr::Number(n), Expr::Add(_, _)) => (n.clone(), r),
                (Expr::Add(_, _), Expr::Number(n)) => (n.clone(), l),
                _ => return None,
            }
        }
        _ => return None,
    };

    // Only handle non-integer non-zero fractions.
    if frac_val.is_integer() || frac_val.is_zero() {
        return None;
    }

    let mut terms = Vec::new();
    collect_add_terms(ctx, add_id, &mut terms);
    if terms.len() < 2 {
        return None;
    }

    let one_id = ctx.num(1);
    let mut coeffs: Vec<BigRational> = Vec::with_capacity(terms.len());
    let mut rests: Vec<ExprId> = Vec::with_capacity(terms.len());
    for &t in &terms {
        let (c, r) = extract_rational_coeff(ctx, t, one_id);
        coeffs.push(c);
        rests.push(r);
    }

    let mut g = coeffs[0].abs();
    for c in &coeffs[1..] {
        g = rational_gcd(&g, c);
        if g.is_one() {
            return None;
        }
    }
    if g <= BigRational::one() {
        return None;
    }

    let reduced_coeffs: Vec<BigRational> = coeffs.iter().map(|c| c / &g).collect();
    let new_frac = &frac_val * &g;

    let mut new_terms: Vec<ExprId> = Vec::with_capacity(terms.len());
    for (rc, rest) in reduced_coeffs.iter().zip(rests.iter()) {
        let new_term = if rc == &BigRational::from_integer(1.into()) {
            *rest
        } else if rc == &BigRational::from_integer((-1).into()) {
            ctx.add(Expr::Neg(*rest))
        } else if rc.is_zero() {
            continue;
        } else {
            let c_expr = ctx.add(Expr::Number(rc.clone()));
            if matches!(ctx.get(*rest), Expr::Number(n) if n.is_one()) {
                c_expr
            } else {
                mul2_raw(ctx, c_expr, *rest)
            }
        };
        new_terms.push(new_term);
    }

    if new_terms.is_empty() {
        return Some(DivScalarIntoAddRewrite {
            rewritten: ctx.num(0),
            kind: DivScalarIntoAddRewriteKind::AllTermsCancel,
        });
    }

    let mut new_sum = new_terms[0];
    for &t in &new_terms[1..] {
        new_sum = ctx.add(Expr::Add(new_sum, t));
    }

    let rewritten = if new_frac == BigRational::from_integer(1.into()) {
        new_sum
    } else if new_frac == BigRational::from_integer((-1).into()) {
        ctx.add(Expr::Neg(new_sum))
    } else {
        let factor_expr = ctx.add(Expr::Number(new_frac.clone()));
        mul2_raw(ctx, factor_expr, new_sum)
    };

    Some(DivScalarIntoAddRewrite {
        rewritten,
        kind: DivScalarIntoAddRewriteKind::FactorCommonCoefficientFromSum,
    })
}

fn collect_add_terms(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            collect_add_terms(ctx, *l, terms);
            collect_add_terms(ctx, *r, terms);
        }
        _ => terms.push(expr),
    }
}

fn extract_rational_coeff(ctx: &Context, term: ExprId, one_id: ExprId) -> (BigRational, ExprId) {
    match ctx.get(term) {
        Expr::Number(n) => {
            return (n.clone(), one_id);
        }
        Expr::Neg(inner) => {
            let (c, rest) = extract_rational_coeff(ctx, *inner, one_id);
            return (-c, rest);
        }
        Expr::Mul(l, r) => {
            if let Expr::Number(n) = ctx.get(*l) {
                return (n.clone(), *r);
            }
            if let Expr::Number(n) = ctx.get(*r) {
                return (n.clone(), *l);
            }
        }
        _ => {}
    }
    (BigRational::from_integer(1.into()), term)
}

fn rational_gcd(a: &BigRational, b: &BigRational) -> BigRational {
    use num_integer::Integer;
    if a.is_zero() {
        return b.abs();
    }
    if b.is_zero() {
        return a.abs();
    }
    let num_gcd = a.numer().gcd(b.numer());
    let den_lcm = a.denom().lcm(b.denom());
    BigRational::new(num_gcd, den_lcm)
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_div_scalar_into_add_expr;
    use cas_ast::ordering::compare_expr;
    use cas_ast::Context;
    use cas_ast::Expr;
    use cas_parser::parse;
    use std::cmp::Ordering;

    #[test]
    fn rewrites_half_times_even_sum() {
        let mut ctx = Context::new();
        let add = parse("2*x+4*y", &mut ctx).expect("parse");
        let half = ctx.rational(1, 2);
        let expr = ctx.add(Expr::Mul(half, add));
        let out = try_rewrite_div_scalar_into_add_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("x+2*y", &mut ctx).expect("parse expected");
        assert_eq!(compare_expr(&ctx, out.rewritten, expected), Ordering::Equal);
    }

    #[test]
    fn skips_when_no_common_gcd() {
        let mut ctx = Context::new();
        let expr = parse("1/2*(x+y)", &mut ctx).expect("parse");
        let out = try_rewrite_div_scalar_into_add_expr(&mut ctx, expr);
        assert!(out.is_none());
    }
}
