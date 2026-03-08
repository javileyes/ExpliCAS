//! Domain-oriented expression equivalence and implication helpers.

use crate::expr_extract::extract_abs_argument_view;
use cas_ast::{Context, Expr, ExprId};

/// Check if two expressions are equivalent using polynomial comparison.
pub fn exprs_equivalent(ctx: &Context, e1: ExprId, e2: ExprId) -> bool {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    if e1 == e2 {
        return true;
    }

    // Quick check: same variable name
    if let (Expr::Variable(name1), Expr::Variable(name2)) = (ctx.get(e1), ctx.get(e2)) {
        if name1 == name2 {
            return true;
        }
    }

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    if let (Ok(p1), Ok(p2)) = (
        multipoly_from_expr(ctx, e1, &budget),
        multipoly_from_expr(ctx, e2, &budget),
    ) {
        return p1 == p2;
    }

    false
}

/// Check if two expressions are equivalent up to global sign.
///
/// This treats `E` and `-E` as equivalent, which is useful for predicates like
/// `E != 0` where multiplying by `-1` does not change truth value.
pub fn exprs_equivalent_up_to_sign(ctx: &Context, e1: ExprId, e2: ExprId) -> bool {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    if e1 == e2 {
        return true;
    }

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    if let (Ok(p1), Ok(p2)) = (
        multipoly_from_expr(ctx, e1, &budget),
        multipoly_from_expr(ctx, e2, &budget),
    ) {
        return p1 == p2 || p1 == p2.neg();
    }

    false
}

/// Check if `source` is `target^(odd positive integer)`.
pub fn is_odd_power_of(ctx: &Context, source: ExprId, target: ExprId) -> bool {
    if let Expr::Pow(base, exp) = ctx.get(source) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if n.is_integer() {
                let exp_int = n.to_integer();
                let two: num_bigint::BigInt = 2.into();
                let zero: num_bigint::BigInt = 0.into();
                let one: num_bigint::BigInt = 1.into();
                if &exp_int % &two == one && exp_int > zero {
                    return exprs_equivalent(ctx, *base, target);
                }
            }
        }
    }
    false
}

/// Check if `expr` is `base^p` for any non-zero numeric exponent `p`.
pub fn is_power_of_base(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    if let Expr::Pow(pow_base, exp) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            let zero = num_rational::BigRational::from_integer(0.into());
            if *n != zero {
                return exprs_equivalent(ctx, *pow_base, base);
            }
        }
    }
    false
}

/// Check if `source` is `k*target` where `k > 0`.
pub fn is_positive_multiple_of(ctx: &Context, source: ExprId, target: ExprId) -> bool {
    use num_traits::Zero;

    if exprs_equivalent(ctx, source, target) {
        return true;
    }

    if let Expr::Mul(l, r) = ctx.get(source) {
        if let Expr::Number(n) = ctx.get(*l) {
            let zero = num_rational::BigRational::zero();
            if *n > zero && exprs_equivalent(ctx, *r, target) {
                return true;
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            let zero = num_rational::BigRational::zero();
            if *n > zero && exprs_equivalent(ctx, *l, target) {
                return true;
            }
        }
    }
    false
}

/// Check if `expr` is `abs(inner)`.
pub fn is_abs_of(ctx: &Context, expr: ExprId, inner: ExprId) -> bool {
    if let Some(arg) = extract_abs_argument_view(ctx, expr) {
        return exprs_equivalent(ctx, arg, inner);
    }
    false
}

/// Check if a positive product condition is dominated by known positive bases.
pub fn is_product_dominated_by_positives(
    ctx: &Context,
    prod_expr: ExprId,
    known_positives: &[ExprId],
) -> bool {
    let bases = extract_product_bases(ctx, prod_expr);

    if bases.len() < 2 {
        return false;
    }

    for base in &bases {
        let covered = known_positives
            .iter()
            .any(|pos| exprs_equivalent(ctx, *base, *pos));
        if !covered {
            return false;
        }
    }

    true
}

fn extract_product_bases(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut bases = Vec::new();
    collect_product_bases(ctx, expr, &mut bases);
    bases
}

fn collect_product_bases(ctx: &Context, expr: ExprId, bases: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            collect_product_bases(ctx, *l, bases);
            collect_product_bases(ctx, *r, bases);
        }
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() {
                    let exp_int = n.to_integer();
                    let zero: num_bigint::BigInt = 0.into();
                    if exp_int > zero {
                        bases.push(*base);
                        return;
                    }
                }
            }
            bases.push(expr);
        }
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) => {
            bases.push(expr);
        }
        _ => {
            bases.push(expr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;
    use cas_parser::parse;

    #[test]
    fn polynomial_equivalence_detects_reordered_sum() {
        let mut ctx = Context::new();
        let e1 = parse("x+1", &mut ctx).expect("parse");
        let e2 = parse("1+x", &mut ctx).expect("parse");
        assert!(exprs_equivalent(&ctx, e1, e2));
    }

    #[test]
    fn equivalence_up_to_sign_detects_negation() {
        let mut ctx = Context::new();
        let e1 = parse("x+1", &mut ctx).expect("parse");
        let e2 = parse("-(x+1)", &mut ctx).expect("parse");
        let e3 = parse("x+2", &mut ctx).expect("parse");

        assert!(exprs_equivalent_up_to_sign(&ctx, e1, e2));
        assert!(!exprs_equivalent_up_to_sign(&ctx, e1, e3));
    }

    #[test]
    fn odd_power_detection_works() {
        let mut ctx = Context::new();
        let source = parse("b^3", &mut ctx).expect("parse");
        let target = parse("b", &mut ctx).expect("parse");
        let even = parse("b^4", &mut ctx).expect("parse");
        assert!(is_odd_power_of(&ctx, source, target));
        assert!(!is_odd_power_of(&ctx, even, target));
    }

    #[test]
    fn power_of_base_detects_fractional_exponent() {
        let mut ctx = Context::new();
        let base = parse("x", &mut ctx).expect("parse");
        let half = ctx.rational(1, 2);
        let frac_pow = ctx.add(Expr::Pow(base, half));
        let zero_exp = parse("x^0", &mut ctx).expect("parse");
        assert!(is_power_of_base(&ctx, frac_pow, base));
        assert!(!is_power_of_base(&ctx, zero_exp, base));
    }

    #[test]
    fn positive_multiple_detection_works() {
        let mut ctx = Context::new();
        let source = parse("4*x", &mut ctx).expect("parse");
        let target = parse("x", &mut ctx).expect("parse");
        let neg_source = parse("-2*x", &mut ctx).expect("parse");
        assert!(is_positive_multiple_of(&ctx, source, target));
        assert!(!is_positive_multiple_of(&ctx, neg_source, target));
    }

    #[test]
    fn abs_detection_works() {
        let mut ctx = Context::new();
        let abs_expr = parse("abs(x)", &mut ctx).expect("parse");
        let x = parse("x", &mut ctx).expect("parse");
        let y = parse("y", &mut ctx).expect("parse");
        assert!(is_abs_of(&ctx, abs_expr, x));
        assert!(!is_abs_of(&ctx, abs_expr, y));
    }

    #[test]
    fn product_dominance_requires_all_factors() {
        let mut ctx = Context::new();
        let product = parse("a^2*b^3", &mut ctx).expect("parse");
        let a = parse("a", &mut ctx).expect("parse");
        let b = parse("b", &mut ctx).expect("parse");
        assert!(is_product_dominated_by_positives(&ctx, product, &[a, b]));
        assert!(!is_product_dominated_by_positives(&ctx, product, &[a]));
    }
}
