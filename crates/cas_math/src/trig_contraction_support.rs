use crate::expr_destructure::{as_add, as_mul, as_neg, as_sub};
use crate::expr_nary::mul_leaves;
use crate::trig_sum_product_support::extract_trig_arg;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
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

/// Extract `(sin_arg, cos_arg)` from `2·sin(t)·cos(t)` in any multiplication arrangement.
pub fn extract_two_sin_cos(ctx: &Context, l: ExprId, r: ExprId) -> Option<(ExprId, ExprId)> {
    let two_rat = BigRational::from_integer(2.into());

    if let Expr::Number(n) = ctx.get(l) {
        if *n == two_rat {
            if let Expr::Mul(a, b) = ctx.get(r) {
                return extract_sin_cos_pair(ctx, *a, *b);
            }
        }
    }

    if let Expr::Number(n) = ctx.get(r) {
        if *n == two_rat {
            if let Expr::Mul(a, b) = ctx.get(l) {
                return extract_sin_cos_pair(ctx, *a, *b);
            }
        }
    }

    if let Expr::Mul(inner_l, inner_r) = ctx.get(l) {
        if let Expr::Number(n) = ctx.get(*inner_l) {
            if *n == two_rat {
                return extract_trig_and_match(ctx, *inner_r, r);
            }
        }
        if let Expr::Number(n) = ctx.get(*inner_r) {
            if *n == two_rat {
                return extract_trig_and_match(ctx, *inner_l, r);
            }
        }
    }

    if let Expr::Mul(inner_l, inner_r) = ctx.get(r) {
        if let Expr::Number(n) = ctx.get(*inner_l) {
            if *n == two_rat {
                return extract_trig_and_match(ctx, *inner_r, l);
            }
        }
        if let Expr::Number(n) = ctx.get(*inner_r) {
            if *n == two_rat {
                return extract_trig_and_match(ctx, *inner_l, l);
            }
        }
    }

    None
}

/// Extract `(sin_arg, cos_arg)` from a pair `sin(t)`/`cos(t)` in either order.
pub fn extract_sin_cos_pair(ctx: &Context, a: ExprId, b: ExprId) -> Option<(ExprId, ExprId)> {
    if let Expr::Function(fn_id_a, args_a) = ctx.get(a) {
        if let Expr::Function(fn_id_b, args_b) = ctx.get(b) {
            if args_a.len() == 1 && args_b.len() == 1 {
                let builtin_a = ctx.builtin_of(*fn_id_a);
                let builtin_b = ctx.builtin_of(*fn_id_b);
                if matches!(builtin_a, Some(BuiltinFn::Sin))
                    && matches!(builtin_b, Some(BuiltinFn::Cos))
                {
                    return Some((args_a[0], args_b[0]));
                }
                if matches!(builtin_a, Some(BuiltinFn::Cos))
                    && matches!(builtin_b, Some(BuiltinFn::Sin))
                {
                    return Some((args_b[0], args_a[0]));
                }
            }
        }
    }
    None
}

/// Extract `(sin_arg, cos_arg)` from two trig nodes by matching `sin` and `cos` in any order.
pub fn extract_trig_and_match(
    ctx: &Context,
    trig1: ExprId,
    trig2: ExprId,
) -> Option<(ExprId, ExprId)> {
    extract_sin_cos_pair(ctx, trig1, trig2)
}

/// Extract `(cos_arg, sin_arg)` from `cos²(t) - sin²(t)`.
pub fn extract_cos2_minus_sin2(ctx: &Context, l: ExprId, r: ExprId) -> Option<(ExprId, ExprId)> {
    let two_rat = BigRational::from_integer(2.into());

    if let Expr::Pow(base_l, exp_l) = ctx.get(l) {
        if let Expr::Number(n) = ctx.get(*exp_l) {
            if *n == two_rat {
                if let Expr::Function(fn_id_l, args_l) = ctx.get(*base_l) {
                    if matches!(ctx.builtin_of(*fn_id_l), Some(BuiltinFn::Cos)) && args_l.len() == 1
                    {
                        if let Expr::Pow(base_r, exp_r) = ctx.get(r) {
                            if let Expr::Number(m) = ctx.get(*exp_r) {
                                if *m == two_rat {
                                    if let Expr::Function(fn_id_r, args_r) = ctx.get(*base_r) {
                                        if matches!(ctx.builtin_of(*fn_id_r), Some(BuiltinFn::Sin))
                                            && args_r.len() == 1
                                        {
                                            return Some((args_l[0], args_r[0]));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

/// Extract `(trig_arg, is_sin, coefficient)` from a term like `±k·sin²(t)` or `±k·cos²(t)`.
pub fn extract_coeff_trig_squared(
    ctx: &Context,
    term: ExprId,
) -> Option<(ExprId, bool, BigRational)> {
    let two_rat = BigRational::from_integer(2.into());

    let (base_term, sign) = if let Expr::Neg(inner) = ctx.get(term) {
        (*inner, BigRational::from_integer((-1).into()))
    } else {
        (term, BigRational::from_integer(1.into()))
    };

    let mut factors = Vec::new();
    let mut stack = vec![base_term];
    while let Some(curr) = stack.pop() {
        if let Expr::Mul(l, r) = ctx.get(curr) {
            stack.push(*l);
            stack.push(*r);
        } else {
            factors.push(curr);
        }
    }

    let mut trig_arg = None;
    let mut is_sin = false;
    let mut trig_idx = None;
    let mut numeric_coeff = sign;

    for (i, &f) in factors.iter().enumerate() {
        if let Expr::Pow(base, exp) = ctx.get(f) {
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n == two_rat {
                    if let Expr::Function(fn_id, args) = ctx.get(*base) {
                        if args.len() == 1 {
                            let builtin = ctx.builtin_of(*fn_id);
                            if matches!(builtin, Some(BuiltinFn::Sin)) {
                                trig_arg = Some(args[0]);
                                is_sin = true;
                                trig_idx = Some(i);
                                break;
                            }
                            if matches!(builtin, Some(BuiltinFn::Cos)) {
                                trig_arg = Some(args[0]);
                                is_sin = false;
                                trig_idx = Some(i);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    let trig_arg = trig_arg?;
    let trig_idx = trig_idx?;

    for (i, &f) in factors.iter().enumerate() {
        if i == trig_idx {
            continue;
        }
        if let Expr::Number(n) = ctx.get(f) {
            numeric_coeff *= n.clone();
        } else {
            return None;
        }
    }

    Some((trig_arg, is_sin, numeric_coeff))
}

/// Match `(sin(a)cos(b) + cos(a)sin(b)) / (cos(a)cos(b) - sin(a)sin(b))`.
/// Returns `(a, b)` if matched.
pub fn match_angle_sum_fraction(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<(ExprId, ExprId)> {
    let (nl, nr) = as_add(ctx, numerator)?;
    let (a, b) = extract_sin_cos_product_pair(ctx, nl, nr)?;

    let (dl, dr) = semantic_sub(ctx, denominator)?;
    let (a2, b2) = extract_cos_cos_minus_sin_sin(ctx, dl, dr)?;

    if args_equal(ctx, a, a2) && args_equal(ctx, b, b2) {
        return Some((a, b));
    }
    if args_equal(ctx, a, b2) && args_equal(ctx, b, a2) {
        return Some((a, b));
    }
    None
}

/// Match `(sin(a)cos(b) - cos(a)sin(b)) / (cos(a)cos(b) + sin(a)sin(b))`.
/// Returns `(a, b)` if matched.
pub fn match_angle_diff_fraction(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<(ExprId, ExprId)> {
    let (nl, nr) = semantic_sub(ctx, numerator)?;
    let (a, b) = extract_sin_cos_product_pair(ctx, nl, nr)?;

    let (dl, dr) = as_add(ctx, denominator)?;
    let (a2, b2) = extract_cos_cos_and_sin_sin(ctx, dl, dr)?;

    if args_equal(ctx, a, a2) && args_equal(ctx, b, b2) {
        return Some((a, b));
    }
    if args_equal(ctx, a, b2) && args_equal(ctx, b, a2) {
        return Some((a, b));
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

    #[test]
    fn double_angle_extractors_match_expected_forms() {
        let mut ctx = Context::new();
        let two_sin_cos = parse("2*sin(t)*cos(t)", &mut ctx).expect("2*sin(t)*cos(t)");
        let cos2_minus_sin2 = parse("cos(t)^2-sin(t)^2", &mut ctx).expect("cos^2-sin^2");
        let t = parse("t", &mut ctx).expect("t");

        let (sin_t, cos_t) = if let Expr::Mul(l, r) = ctx.get(two_sin_cos) {
            extract_two_sin_cos(&ctx, *l, *r).expect("2*sin*cos")
        } else {
            panic!("expected multiplication");
        };
        let (cos_arg, sin_arg) = if let Expr::Sub(l, r) = ctx.get(cos2_minus_sin2) {
            extract_cos2_minus_sin2(&ctx, *l, *r).expect("cos2-sin2")
        } else {
            panic!("expected subtraction");
        };

        assert!(args_equal(&ctx, sin_t, t));
        assert!(args_equal(&ctx, cos_t, t));
        assert!(args_equal(&ctx, cos_arg, t));
        assert!(args_equal(&ctx, sin_arg, t));
    }

    #[test]
    fn coeff_trig_squared_extractor_handles_sign_and_factor() {
        let mut ctx = Context::new();
        let term = parse("-2*sin(x)^2", &mut ctx).expect("-2*sin(x)^2");
        let x = parse("x", &mut ctx).expect("x");
        let neg_two = BigRational::from_integer((-2).into());

        let (arg, is_sin, coeff) = extract_coeff_trig_squared(&ctx, term).expect("coeff*trig^2");
        assert!(args_equal(&ctx, arg, x));
        assert!(is_sin);
        assert_eq!(coeff, neg_two);
    }

    #[test]
    fn matches_angle_sum_fraction_pattern() {
        let mut ctx = Context::new();
        let num = parse("sin(a)*cos(b)+cos(a)*sin(b)", &mut ctx).expect("num");
        let den = parse("cos(a)*cos(b)-sin(a)*sin(b)", &mut ctx).expect("den");
        let a = parse("a", &mut ctx).expect("a");
        let b = parse("b", &mut ctx).expect("b");

        let (ma, mb) = match_angle_sum_fraction(&ctx, num, den).expect("sum match");
        assert!(args_equal(&ctx, ma, a));
        assert!(args_equal(&ctx, mb, b));
    }

    #[test]
    fn matches_angle_diff_fraction_pattern() {
        let mut ctx = Context::new();
        let num = parse("sin(a)*cos(b)-cos(a)*sin(b)", &mut ctx).expect("num");
        let den = parse("cos(a)*cos(b)+sin(a)*sin(b)", &mut ctx).expect("den");
        let a = parse("a", &mut ctx).expect("a");
        let b = parse("b", &mut ctx).expect("b");

        let (ma, mb) = match_angle_diff_fraction(&ctx, num, den).expect("diff match");
        assert!(args_equal(&ctx, ma, a));
        assert!(args_equal(&ctx, mb, b));
    }
}
