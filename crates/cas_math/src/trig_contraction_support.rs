use crate::expr_destructure::{as_mul, as_neg, as_sub};
use crate::expr_nary::mul_leaves;
use crate::trig_sum_product_support::extract_trig_arg;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use std::cmp::Ordering;

/// Semantic expression equality for trig matching.
pub fn args_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    a == b || compare_expr(ctx, a, b) == Ordering::Equal
}

/// Semantic subtraction: matches `Sub(a, b)`, `Add(a, Neg(b))`, `Add(Neg(b), a)`.
/// Returns `(positive_part, subtracted_part)`.
pub fn semantic_sub(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if let Some((l, r)) = as_sub(ctx, expr) {
        return Some((l, r));
    }

    if let Expr::Add(l, r) = ctx.get(expr) {
        if let Some(inner) = as_neg(ctx, *r) {
            return Some((*l, inner));
        }
        if let Some(inner) = as_neg(ctx, *l) {
            return Some((*r, inner));
        }
    }

    None
}

/// Extract `(sin_arg, cos_arg)` from `sin(x)·cos(y)` or `cos(y)·sin(x)`.
pub fn extract_sin_times_cos(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let (l, r) = as_mul(ctx, expr)?;

    if let Some(sin_arg) = extract_trig_arg(ctx, l, BuiltinFn::Sin.name()) {
        if let Some(cos_arg) = extract_trig_arg(ctx, r, BuiltinFn::Cos.name()) {
            return Some((sin_arg, cos_arg));
        }
    }
    if let Some(cos_arg) = extract_trig_arg(ctx, l, BuiltinFn::Cos.name()) {
        if let Some(sin_arg) = extract_trig_arg(ctx, r, BuiltinFn::Sin.name()) {
            return Some((sin_arg, cos_arg));
        }
    }

    None
}

/// Find argument of a target trig function in factors.
pub fn find_trig_in_factors(
    ctx: &Context,
    factors: &[ExprId],
    target: BuiltinFn,
) -> Option<ExprId> {
    factors
        .iter()
        .copied()
        .find_map(|f| extract_trig_arg(ctx, f, target.name()))
}

/// Extract `(a, b)` from `f(a)·f(b)` where `f` is target trig function.
pub fn extract_same_trig_product(
    ctx: &Context,
    expr: ExprId,
    target: BuiltinFn,
) -> Option<(ExprId, ExprId)> {
    if let Some((l, r)) = as_mul(ctx, expr) {
        if let Some(a) = extract_trig_arg(ctx, l, target.name()) {
            if let Some(b) = extract_trig_arg(ctx, r, target.name()) {
                return Some((a, b));
            }
        }
    }

    let factors = mul_leaves(ctx, expr);
    if factors.len() == 2 {
        if let Some(a) = extract_trig_arg(ctx, factors[0], target.name()) {
            if let Some(b) = extract_trig_arg(ctx, factors[1], target.name()) {
                return Some((a, b));
            }
        }
    }

    None
}

/// Extract `(a, b)` from one term `sin(a)·cos(b)` and another `cos(a)·sin(b)`.
pub fn extract_sin_cos_product_pair(
    ctx: &Context,
    term1: ExprId,
    term2: ExprId,
) -> Option<(ExprId, ExprId)> {
    if let Some((sin_arg1, cos_arg1)) = extract_sin_times_cos(ctx, term1) {
        if let Some((sin_arg2, cos_arg2)) = extract_sin_times_cos(ctx, term2) {
            if args_equal(ctx, sin_arg1, cos_arg2) && args_equal(ctx, cos_arg1, sin_arg2) {
                return Some((sin_arg1, cos_arg1));
            }
        }
    }

    let factors1 = mul_leaves(ctx, term1);
    let factors2 = mul_leaves(ctx, term2);
    if factors1.len() != 2 || factors2.len() != 2 {
        return None;
    }

    let sin1 = find_trig_in_factors(ctx, &factors1, BuiltinFn::Sin);
    let cos1 = find_trig_in_factors(ctx, &factors1, BuiltinFn::Cos);
    let sin2 = find_trig_in_factors(ctx, &factors2, BuiltinFn::Sin);
    let cos2 = find_trig_in_factors(ctx, &factors2, BuiltinFn::Cos);

    if let (Some(sin_arg1), Some(cos_arg1), Some(sin_arg2), Some(cos_arg2)) =
        (sin1, cos1, sin2, cos2)
    {
        if args_equal(ctx, sin_arg1, cos_arg2) && args_equal(ctx, cos_arg1, sin_arg2) {
            return Some((sin_arg1, cos_arg1));
        }
    }

    None
}

/// Extract `(a, b)` from `cos(a)·cos(b) - sin(a)·sin(b)`.
pub fn extract_cos_cos_minus_sin_sin(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, ExprId)> {
    let (cos_a, cos_b) = extract_same_trig_product(ctx, left, BuiltinFn::Cos)?;
    let (sin_a, sin_b) = extract_same_trig_product(ctx, right, BuiltinFn::Sin)?;

    if args_equal(ctx, cos_a, sin_a) && args_equal(ctx, cos_b, sin_b) {
        return Some((cos_a, cos_b));
    }
    if args_equal(ctx, cos_a, sin_b) && args_equal(ctx, cos_b, sin_a) {
        return Some((cos_a, cos_b));
    }
    None
}

/// Extract `(a, b)` from one positive term `cos(a)·cos(b)` and one `sin(a)·sin(b)`.
pub fn extract_cos_cos_and_sin_sin(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, ExprId)> {
    if let Some((cos_a, cos_b)) = extract_same_trig_product(ctx, left, BuiltinFn::Cos) {
        if let Some((sin_a, sin_b)) = extract_same_trig_product(ctx, right, BuiltinFn::Sin) {
            if args_equal(ctx, cos_a, sin_a) && args_equal(ctx, cos_b, sin_b) {
                return Some((cos_a, cos_b));
            }
            if args_equal(ctx, cos_a, sin_b) && args_equal(ctx, cos_b, sin_a) {
                return Some((cos_a, cos_b));
            }
        }
    }

    if let Some((sin_a, sin_b)) = extract_same_trig_product(ctx, left, BuiltinFn::Sin) {
        if let Some((cos_a, cos_b)) = extract_same_trig_product(ctx, right, BuiltinFn::Cos) {
            if args_equal(ctx, cos_a, sin_a) && args_equal(ctx, cos_b, sin_b) {
                return Some((cos_a, cos_b));
            }
            if args_equal(ctx, cos_a, sin_b) && args_equal(ctx, cos_b, sin_a) {
                return Some((cos_a, cos_b));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn semantic_sub_matches_sub_and_add_neg() {
        let mut ctx = Context::new();
        let sub = parse("a-b", &mut ctx).expect("a-b");
        let add_neg = parse("a+(-b)", &mut ctx).expect("a+(-b)");
        let a = parse("a", &mut ctx).expect("a");
        let b = parse("b", &mut ctx).expect("b");

        let (s1_l, s1_r) = semantic_sub(&ctx, sub).expect("sub");
        let (s2_l, s2_r) = semantic_sub(&ctx, add_neg).expect("add-neg");
        assert!(args_equal(&ctx, s1_l, a) && args_equal(&ctx, s1_r, b));
        assert!(args_equal(&ctx, s2_l, a) && args_equal(&ctx, s2_r, b));
    }

    #[test]
    fn extract_sin_times_cos_accepts_both_orders() {
        let mut ctx = Context::new();
        let e1 = parse("sin(a)*cos(b)", &mut ctx).expect("e1");
        let e2 = parse("cos(b)*sin(a)", &mut ctx).expect("e2");
        let a = parse("a", &mut ctx).expect("a");
        let b = parse("b", &mut ctx).expect("b");

        let (a1, b1) = extract_sin_times_cos(&ctx, e1).expect("e1 match");
        let (a2, b2) = extract_sin_times_cos(&ctx, e2).expect("e2 match");
        assert!(args_equal(&ctx, a1, a) && args_equal(&ctx, b1, b));
        assert!(args_equal(&ctx, a2, a) && args_equal(&ctx, b2, b));
    }

    #[test]
    fn pair_and_denominator_extractors_match_expected_forms() {
        let mut ctx = Context::new();
        let term1 = parse("sin(a)*cos(b)", &mut ctx).expect("term1");
        let term2 = parse("cos(a)*sin(b)", &mut ctx).expect("term2");
        let den_l = parse("cos(a)*cos(b)", &mut ctx).expect("den_l");
        let den_r = parse("sin(a)*sin(b)", &mut ctx).expect("den_r");
        let a = parse("a", &mut ctx).expect("a");
        let b = parse("b", &mut ctx).expect("b");

        let (p_a, p_b) = extract_sin_cos_product_pair(&ctx, term1, term2).expect("pair");
        let (m_a, m_b) = extract_cos_cos_minus_sin_sin(&ctx, den_l, den_r).expect("minus");
        let (s_a, s_b) = extract_cos_cos_and_sin_sin(&ctx, den_l, den_r).expect("sum");

        assert!(args_equal(&ctx, p_a, a) && args_equal(&ctx, p_b, b));
        assert!(args_equal(&ctx, m_a, a) && args_equal(&ctx, m_b, b));
        assert!(args_equal(&ctx, s_a, a) && args_equal(&ctx, s_b, b));
    }
}
