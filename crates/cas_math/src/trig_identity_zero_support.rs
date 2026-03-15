use crate::expr_destructure::{as_add, as_mul, as_pow};
use crate::expr_nary::mul_leaves;
use crate::trig_sum_product_support::{args_match_as_multiset, extract_trig_arg};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TanDifferenceIdentityMatch {
    pub tan_diff: ExprId,
    pub tan_a_num: ExprId,
    pub tan_b_num: ExprId,
    pub den_tan_left: ExprId,
    pub den_tan_right: ExprId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdentityZeroRewrite {
    pub kind: IdentityZeroRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityZeroRewriteKind {
    WeierstrassSin,
    WeierstrassCos,
    TanDifference,
    Sin4x,
    SinSumTriple,
    CosTriple,
}

fn extract_n_times_base(ctx: &Context, expr: ExprId, n: i64) -> Option<ExprId> {
    let Expr::Mul(l, r) = ctx.get(expr) else {
        return None;
    };
    let expected = num_rational::BigRational::from_integer(n.into());
    if let Expr::Number(k) = ctx.get(*l) {
        if *k == expected {
            return Some(*r);
        }
    }
    if let Expr::Number(k) = ctx.get(*r) {
        if *k == expected {
            return Some(*l);
        }
    }
    None
}

fn extract_simple_numeric_mul(
    ctx: &Context,
    expr: ExprId,
) -> Option<(num_rational::BigRational, ExprId)> {
    let Expr::Mul(l, r) = ctx.get(expr) else {
        return None;
    };
    if let Expr::Number(k) = ctx.get(*l) {
        return Some((k.clone(), *r));
    }
    if let Expr::Number(k) = ctx.get(*r) {
        return Some((k.clone(), *l));
    }
    None
}

fn extract_numeric_scale_factors(
    ctx: &Context,
    expr: ExprId,
) -> (num_rational::BigRational, Vec<ExprId>) {
    use num_rational::BigRational;

    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (coeff, factors) = extract_numeric_scale_factors(ctx, *inner);
            (-coeff, factors)
        }
        Expr::Mul(_, _) => {
            let mut coeff = BigRational::from_integer(1.into());
            let mut factors = Vec::new();
            for factor in mul_leaves(ctx, expr) {
                if let Expr::Number(n) = ctx.get(factor) {
                    coeff *= n.clone();
                } else {
                    factors.push(factor);
                }
            }
            (coeff, factors)
        }
        Expr::Number(n) => (n.clone(), Vec::new()),
        _ => (BigRational::from_integer(1.into()), vec![expr]),
    }
}

fn factor_multisets_match(ctx: &Context, left: &[ExprId], right: &[ExprId]) -> bool {
    if left.len() != right.len() {
        return false;
    }

    let mut used = vec![false; right.len()];
    for lhs in left {
        let mut matched = false;
        for (idx, rhs) in right.iter().enumerate() {
            if used[idx] {
                continue;
            }
            if compare_expr(ctx, *lhs, *rhs) == std::cmp::Ordering::Equal {
                used[idx] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            return false;
        }
    }
    true
}

fn expr_is_n_times_of(ctx: &Context, expr: ExprId, base: ExprId, n: i64) -> bool {
    if n == 1 {
        return compare_expr(ctx, expr, base) == std::cmp::Ordering::Equal;
    }

    let (expr_coeff, expr_factors) = extract_numeric_scale_factors(ctx, expr);
    let (base_coeff, base_factors) = extract_numeric_scale_factors(ctx, base);
    if factor_multisets_match(ctx, &expr_factors, &base_factors)
        && expr_coeff == base_coeff * num_rational::BigRational::from_integer(n.into())
    {
        return true;
    }

    if let Some(extracted) = extract_n_times_base(ctx, expr, n) {
        return compare_expr(ctx, extracted, base) == std::cmp::Ordering::Equal;
    }

    match (ctx.get(expr), ctx.get(base)) {
        (Expr::Number(en), Expr::Number(bn)) => {
            *en == bn.clone() * num_rational::BigRational::from_integer(n.into())
        }
        (Expr::Mul(_, _), Expr::Mul(_, _)) => {
            let Some((ek, erest)) = extract_simple_numeric_mul(ctx, expr) else {
                return false;
            };
            let Some((bk, brest)) = extract_simple_numeric_mul(ctx, base) else {
                return false;
            };
            compare_expr(ctx, erest, brest) == std::cmp::Ordering::Equal
                && ek == bk * num_rational::BigRational::from_integer(n.into())
        }
        (Expr::Add(el, er), Expr::Add(bl, br)) => {
            (expr_is_n_times_of(ctx, *el, *bl, n) && expr_is_n_times_of(ctx, *er, *br, n))
                || (expr_is_n_times_of(ctx, *el, *br, n) && expr_is_n_times_of(ctx, *er, *bl, n))
        }
        (Expr::Sub(el, er), Expr::Sub(bl, br)) => {
            expr_is_n_times_of(ctx, *el, *bl, n) && expr_is_n_times_of(ctx, *er, *br, n)
        }
        (Expr::Div(enum_, eden), Expr::Div(bnum, bden)) => {
            compare_expr(ctx, *eden, *bden) == std::cmp::Ordering::Equal
                && expr_is_n_times_of(ctx, *enum_, *bnum, n)
        }
        _ => false,
    }
}

fn is_trig_squared_t(ctx: &Context, expr: ExprId, trig: BuiltinFn, t: ExprId) -> bool {
    let Some((base, exp)) = as_pow(ctx, expr) else {
        return false;
    };
    let Expr::Number(n) = ctx.get(exp) else {
        return false;
    };
    if *n != num_rational::BigRational::from_integer(2.into()) {
        return false;
    }

    let Some(arg) = extract_trig_arg(ctx, base, trig.name()) else {
        return false;
    };
    compare_expr(ctx, arg, t) == std::cmp::Ordering::Equal
}

pub fn is_sin_squared_t(ctx: &Context, expr: ExprId, t: ExprId) -> bool {
    is_trig_squared_t(ctx, expr, BuiltinFn::Sin, t)
}

pub fn is_cos_squared_t(ctx: &Context, expr: ExprId, t: ExprId) -> bool {
    is_trig_squared_t(ctx, expr, BuiltinFn::Cos, t)
}

/// Match `1 + tan(a)*tan(b)` (in any operand order) and return the two `tan` nodes.
pub fn extract_one_plus_tan_product_nodes(
    ctx: &Context,
    expr: ExprId,
    a: ExprId,
    b: ExprId,
) -> Option<(ExprId, ExprId)> {
    let (l, r) = as_add(ctx, expr)?;
    let product_part = if let Expr::Number(n) = ctx.get(l) {
        if *n == num_rational::BigRational::from_integer(1.into()) {
            r
        } else {
            return None;
        }
    } else if let Expr::Number(n) = ctx.get(r) {
        if *n == num_rational::BigRational::from_integer(1.into()) {
            l
        } else {
            return None;
        }
    } else {
        return None;
    };

    let (ml, mr) = as_mul(ctx, product_part)?;
    let arg1 = extract_trig_arg(ctx, ml, BuiltinFn::Tan.name())?;
    let arg2 = extract_trig_arg(ctx, mr, BuiltinFn::Tan.name())?;
    if args_match_as_multiset(ctx, arg1, arg2, a, b) {
        Some((ml, mr))
    } else {
        None
    }
}

/// Match `1 + tan(a)*tan(b)` (in any operand order).
pub fn match_one_plus_tan_product(ctx: &Context, expr: ExprId, a: ExprId, b: ExprId) -> bool {
    extract_one_plus_tan_product_nodes(ctx, expr, a, b).is_some()
}

fn extract_sub_like(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if let Expr::Sub(l, r) = ctx.get(expr) {
        return Some((*l, *r));
    }
    if let Expr::Add(l, r) = ctx.get(expr) {
        if let Expr::Neg(inner) = ctx.get(*r) {
            return Some((*l, *inner));
        }
        if let Expr::Neg(inner) = ctx.get(*l) {
            return Some((*r, *inner));
        }
    }
    None
}

fn extract_tan_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Tan) || args.len() != 1 {
        return None;
    }
    Some(args[0])
}

fn match_tan_difference_identity_pair(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<TanDifferenceIdentityMatch> {
    let tan_arg = extract_tan_arg(ctx, lhs)?;
    let (a, b) = extract_sub_like(ctx, tan_arg)?;

    let Expr::Div(num, den) = ctx.get(rhs) else {
        return None;
    };
    let (num, den) = (*num, *den);

    // Keep compatibility with existing engine behavior:
    // accept Sub(tan(a), tan(b)) and Add(tan(a), Neg(tan(b))).
    let (tan_a_num, tan_b_num) = if let Expr::Sub(l, r) = ctx.get(num) {
        (*l, *r)
    } else if let Expr::Add(l, r) = ctx.get(num) {
        if let Expr::Neg(inner) = ctx.get(*r) {
            (*l, *inner)
        } else {
            return None;
        }
    } else {
        return None;
    };

    let tan_a_arg = extract_tan_arg(ctx, tan_a_num)?;
    let tan_b_arg = extract_tan_arg(ctx, tan_b_num)?;
    if compare_expr(ctx, tan_a_arg, a) != std::cmp::Ordering::Equal
        || compare_expr(ctx, tan_b_arg, b) != std::cmp::Ordering::Equal
    {
        return None;
    }

    let (den_tan_left, den_tan_right) = extract_one_plus_tan_product_nodes(ctx, den, a, b)?;
    Some(TanDifferenceIdentityMatch {
        tan_diff: lhs,
        tan_a_num,
        tan_b_num,
        den_tan_left,
        den_tan_right,
    })
}

/// Match full identity cancellation form:
/// `tan(a-b) - (tan(a)-tan(b))/(1+tan(a)*tan(b))` (or swapped sides).
pub fn match_tan_difference_identity_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<TanDifferenceIdentityMatch> {
    let (left, right) = extract_sub_like(ctx, expr)?;
    match_tan_difference_identity_pair(ctx, left, right)
        .or_else(|| match_tan_difference_identity_pair(ctx, right, left))
}

/// Rewrite plan for
/// `tan(a-b) - (tan(a)-tan(b))/(1+tan(a)*tan(b)) -> 0`.
pub fn try_rewrite_tan_difference_identity_zero_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<IdentityZeroRewrite> {
    let _ = match_tan_difference_identity_expr(ctx, expr)?;
    Some(IdentityZeroRewrite {
        kind: IdentityZeroRewriteKind::TanDifference,
    })
}

fn match_sin4x_identity_zero_pair(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(lhs) else {
        return false;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Sin) || args.len() != 1 {
        return false;
    }
    let sin_arg = args[0];

    let t = match ctx.get(sin_arg) {
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if let Expr::Number(n) = ctx.get(l) {
                if *n == num_rational::BigRational::from_integer(4.into()) {
                    r
                } else {
                    return false;
                }
            } else if let Expr::Number(n) = ctx.get(r) {
                if *n == num_rational::BigRational::from_integer(4.into()) {
                    l
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
        _ => return false,
    };

    let factors = mul_leaves(ctx, rhs);
    if factors.len() < 4 {
        return false;
    }

    let mut has_four = false;
    let mut has_sin_t = false;
    let mut has_cos_t = false;
    let mut has_diff_squares = false;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            if *n == num_rational::BigRational::from_integer(4.into()) {
                has_four = true;
                continue;
            }
        }
        if let Expr::Function(fn_name, fn_args) = ctx.get(factor) {
            if ctx.is_builtin(*fn_name, BuiltinFn::Sin)
                && fn_args.len() == 1
                && compare_expr(ctx, fn_args[0], t) == std::cmp::Ordering::Equal
            {
                has_sin_t = true;
                continue;
            }
            if ctx.is_builtin(*fn_name, BuiltinFn::Cos) && fn_args.len() == 1 {
                let arg = fn_args[0];
                if compare_expr(ctx, arg, t) == std::cmp::Ordering::Equal {
                    has_cos_t = true;
                    continue;
                }
                if let Expr::Mul(cl, cr) = ctx.get(arg) {
                    let (cl, cr) = (*cl, *cr);
                    let is_2t = if let Expr::Number(n) = ctx.get(cl) {
                        *n == num_rational::BigRational::from_integer(2.into())
                            && compare_expr(ctx, cr, t) == std::cmp::Ordering::Equal
                    } else if let Expr::Number(n) = ctx.get(cr) {
                        *n == num_rational::BigRational::from_integer(2.into())
                            && compare_expr(ctx, cl, t) == std::cmp::Ordering::Equal
                    } else {
                        false
                    };
                    if is_2t {
                        has_diff_squares = true;
                        continue;
                    }
                }
            }
        }
        if let Expr::Sub(sl, sr) = ctx.get(factor) {
            let (sl, sr) = (*sl, *sr);
            if is_cos_squared_t(ctx, sl, t) && is_sin_squared_t(ctx, sr, t) {
                has_diff_squares = true;
                continue;
            }
            if is_2cos_sq_minus_1(ctx, sl, sr, t) {
                has_diff_squares = true;
                continue;
            }
            if is_1_minus_2sin_sq(ctx, sl, sr, t) {
                has_diff_squares = true;
                continue;
            }
        }
    }

    has_four && has_sin_t && has_cos_t && has_diff_squares
}

fn is_2cos_sq_minus_1(ctx: &Context, lhs: ExprId, rhs: ExprId, t: ExprId) -> bool {
    let Expr::Number(n_rhs) = ctx.get(rhs) else {
        return false;
    };
    if *n_rhs != num_rational::BigRational::from_integer(1.into()) {
        return false;
    }

    let Expr::Mul(ml, mr) = ctx.get(lhs) else {
        return false;
    };
    let (ml, mr) = (*ml, *mr);

    let is_two = |id: ExprId| matches!(ctx.get(id), Expr::Number(n) if *n == num_rational::BigRational::from_integer(2.into()));
    (is_two(ml) && is_cos_squared_t(ctx, mr, t)) || (is_two(mr) && is_cos_squared_t(ctx, ml, t))
}

fn is_1_minus_2sin_sq(ctx: &Context, lhs: ExprId, rhs: ExprId, t: ExprId) -> bool {
    let Expr::Number(n_lhs) = ctx.get(lhs) else {
        return false;
    };
    if *n_lhs != num_rational::BigRational::from_integer(1.into()) {
        return false;
    }

    let Expr::Mul(ml, mr) = ctx.get(rhs) else {
        return false;
    };
    let (ml, mr) = (*ml, *mr);

    let is_two = |id: ExprId| matches!(ctx.get(id), Expr::Number(n) if *n == num_rational::BigRational::from_integer(2.into()));
    (is_two(ml) && is_sin_squared_t(ctx, mr, t)) || (is_two(mr) && is_sin_squared_t(ctx, ml, t))
}

/// Match:
/// `sin(4t) - 4*sin(t)*cos(t)*(cos²(t)-sin²(t))` (or swapped sides),
/// including the equivalent `cos(2t)` factor in place of `(cos²-sin²)`.
pub fn match_sin4x_identity_zero_expr(ctx: &Context, expr: ExprId) -> bool {
    let Some((left, right)) = extract_sub_like(ctx, expr) else {
        return false;
    };
    match_sin4x_identity_zero_pair(ctx, left, right)
        || match_sin4x_identity_zero_pair(ctx, right, left)
}

/// Rewrite plan for
/// `sin(4t) - 4*sin(t)*cos(t)*(cos²(t)-sin²(t)) -> 0`.
pub fn try_rewrite_sin4x_identity_zero_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<IdentityZeroRewrite> {
    if !match_sin4x_identity_zero_expr(ctx, expr) {
        return None;
    }
    Some(IdentityZeroRewrite {
        kind: IdentityZeroRewriteKind::Sin4x,
    })
}

fn match_sin_sum_triple_identity_pair(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    let Expr::Add(l, r) = ctx.get(lhs) else {
        return false;
    };
    let (l, r) = (*l, *r);
    let Some(l_arg) = extract_trig_arg(ctx, l, BuiltinFn::Sin.name()) else {
        return false;
    };
    let Some(r_arg) = extract_trig_arg(ctx, r, BuiltinFn::Sin.name()) else {
        return false;
    };

    let t = if expr_is_n_times_of(ctx, l_arg, r_arg, 3) {
        r_arg
    } else if expr_is_n_times_of(ctx, r_arg, l_arg, 3) {
        l_arg
    } else {
        return false;
    };

    let factors = mul_leaves(ctx, rhs);
    if factors.len() != 3 {
        return false;
    }

    let mut has_two = false;
    let mut has_sin_2t = false;
    let mut has_cos_t = false;
    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            if *n == num_rational::BigRational::from_integer(2.into()) {
                has_two = true;
                continue;
            }
        }
        if let Some(arg) = extract_trig_arg(ctx, factor, BuiltinFn::Sin.name()) {
            if expr_is_n_times_of(ctx, arg, t, 2) {
                has_sin_2t = true;
                continue;
            }
        }
        if let Some(arg) = extract_trig_arg(ctx, factor, BuiltinFn::Cos.name()) {
            if compare_expr(ctx, arg, t) == std::cmp::Ordering::Equal {
                has_cos_t = true;
                continue;
            }
        }
    }

    has_two && has_sin_2t && has_cos_t
}

pub fn match_sin_sum_triple_identity_zero_expr(ctx: &Context, expr: ExprId) -> bool {
    let Some((left, right)) = extract_sub_like(ctx, expr) else {
        return false;
    };
    match_sin_sum_triple_identity_pair(ctx, left, right)
        || match_sin_sum_triple_identity_pair(ctx, right, left)
}

pub fn try_rewrite_sin_sum_triple_identity_zero_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<IdentityZeroRewrite> {
    if !match_sin_sum_triple_identity_zero_expr(ctx, expr) {
        return None;
    }
    Some(IdentityZeroRewrite {
        kind: IdentityZeroRewriteKind::SinSumTriple,
    })
}

fn is_cos_power_t(ctx: &Context, expr: ExprId, t: ExprId, n: i64) -> bool {
    let Some((base, exp)) = as_pow(ctx, expr) else {
        return false;
    };
    let Expr::Number(k) = ctx.get(exp) else {
        return false;
    };
    if *k != num_rational::BigRational::from_integer(n.into()) {
        return false;
    }
    let Some(arg) = extract_trig_arg(ctx, base, BuiltinFn::Cos.name()) else {
        return false;
    };
    compare_expr(ctx, arg, t) == std::cmp::Ordering::Equal
}

fn match_cos_triple_identity_pair(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    let Some(lhs_arg) = extract_trig_arg(ctx, lhs, BuiltinFn::Cos.name()) else {
        return false;
    };
    let Some(t) = extract_n_times_base(ctx, lhs_arg, 3) else {
        return false;
    };

    let Some((left, right)) = extract_sub_like(ctx, rhs) else {
        return false;
    };

    let left_matches = if let Some(term) = extract_n_times_base(ctx, left, 4) {
        is_cos_power_t(ctx, term, t, 3)
    } else {
        false
    };
    let right_matches = if let Some(term) = extract_n_times_base(ctx, right, 3) {
        if let Some(arg) = extract_trig_arg(ctx, term, BuiltinFn::Cos.name()) {
            compare_expr(ctx, arg, t) == std::cmp::Ordering::Equal
        } else {
            false
        }
    } else {
        false
    };

    left_matches && right_matches
}

pub fn match_cos_triple_identity_zero_expr(ctx: &Context, expr: ExprId) -> bool {
    let Some((left, right)) = extract_sub_like(ctx, expr) else {
        return false;
    };
    match_cos_triple_identity_pair(ctx, left, right)
        || match_cos_triple_identity_pair(ctx, right, left)
}

pub fn try_rewrite_cos_triple_identity_zero_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<IdentityZeroRewrite> {
    if !match_cos_triple_identity_zero_expr(ctx, expr) {
        return None;
    }
    Some(IdentityZeroRewrite {
        kind: IdentityZeroRewriteKind::CosTriple,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn detects_sin_and_cos_squared_of_target() {
        let mut ctx = Context::new();
        let t = parse("u+v", &mut ctx).expect("t");
        let sin_sq = parse("sin(u+v)^2", &mut ctx).expect("sin_sq");
        let cos_sq = parse("cos(u+v)^2", &mut ctx).expect("cos_sq");
        let wrong = parse("sin(u)^2", &mut ctx).expect("wrong");

        assert!(is_sin_squared_t(&ctx, sin_sq, t));
        assert!(is_cos_squared_t(&ctx, cos_sq, t));
        assert!(!is_sin_squared_t(&ctx, wrong, t));
    }

    #[test]
    fn matches_one_plus_tan_product_both_orders() {
        let mut ctx = Context::new();
        let a = parse("a", &mut ctx).expect("a");
        let b = parse("b", &mut ctx).expect("b");
        let expr1 = parse("1 + tan(a)*tan(b)", &mut ctx).expect("expr1");
        let expr2 = parse("tan(b)*tan(a) + 1", &mut ctx).expect("expr2");
        let wrong = parse("1 + tan(a)*tan(c)", &mut ctx).expect("wrong");

        assert!(match_one_plus_tan_product(&ctx, expr1, a, b));
        assert!(match_one_plus_tan_product(&ctx, expr2, a, b));
        assert!(!match_one_plus_tan_product(&ctx, wrong, a, b));
    }

    #[test]
    fn matches_tan_difference_identity_full_expression() {
        let mut ctx = Context::new();
        let expr = parse("tan(a-b) - (tan(a)-tan(b))/(1 + tan(a)*tan(b))", &mut ctx).expect("expr");
        let m = match_tan_difference_identity_expr(&ctx, expr).expect("match");

        let tan_diff_expected = parse("tan(a-b)", &mut ctx).expect("tan diff");
        assert_eq!(
            compare_expr(&ctx, m.tan_diff, tan_diff_expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn tan_difference_identity_zero_rewrite_plan_matches() {
        let mut ctx = Context::new();
        let expr = parse("tan(a-b) - (tan(a)-tan(b))/(1 + tan(a)*tan(b))", &mut ctx).expect("expr");
        let rewrite = try_rewrite_tan_difference_identity_zero_expr(&ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, IdentityZeroRewriteKind::TanDifference);
    }

    #[test]
    fn matches_sin4x_identity_zero_variants() {
        let mut ctx = Context::new();
        let expr1 =
            parse("sin(4*t) - 4*sin(t)*cos(t)*(cos(t)^2-sin(t)^2)", &mut ctx).expect("expr1");
        let expr2 = parse("4*sin(t)*cos(t)*cos(2*t) - sin(4*t)", &mut ctx).expect("expr2");
        let expr3 = parse("sin(4*t) - 4*sin(t)*cos(t)*(2*cos(t)^2-1)", &mut ctx).expect("expr3");
        let expr4 = parse("sin(4*t) - 4*sin(t)*cos(t)*(1-2*sin(t)^2)", &mut ctx).expect("expr4");
        let wrong =
            parse("sin(4*t) - 4*sin(t)*cos(t)*(cos(u)^2-sin(u)^2)", &mut ctx).expect("wrong");

        assert!(match_sin4x_identity_zero_expr(&ctx, expr1));
        assert!(match_sin4x_identity_zero_expr(&ctx, expr2));
        assert!(match_sin4x_identity_zero_expr(&ctx, expr3));
        assert!(match_sin4x_identity_zero_expr(&ctx, expr4));
        assert!(!match_sin4x_identity_zero_expr(&ctx, wrong));
    }

    #[test]
    fn sin4x_identity_zero_rewrite_plan_matches() {
        let mut ctx = Context::new();
        let expr = parse("sin(4*t) - 4*sin(t)*cos(t)*(cos(t)^2-sin(t)^2)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_sin4x_identity_zero_expr(&ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, IdentityZeroRewriteKind::Sin4x);
    }

    #[test]
    fn matches_sin_sum_triple_identity_zero_variants() {
        let mut ctx = Context::new();
        let expr1 = parse("sin(t) + sin(3*t) - 2*sin(2*t)*cos(t)", &mut ctx).expect("expr1");
        let expr2 = parse("2*sin(2*t)*cos(t) - (sin(3*t) + sin(t))", &mut ctx).expect("expr2");
        let wrong = parse("sin(t) + sin(3*u) - 2*sin(2*t)*cos(t)", &mut ctx).expect("wrong");

        assert!(match_sin_sum_triple_identity_zero_expr(&ctx, expr1));
        assert!(match_sin_sum_triple_identity_zero_expr(&ctx, expr2));
        assert!(!match_sin_sum_triple_identity_zero_expr(&ctx, wrong));
    }

    #[test]
    fn sin_sum_triple_identity_zero_rewrite_plan_matches() {
        let mut ctx = Context::new();
        let expr = parse("sin(t) + sin(3*t) - 2*sin(2*t)*cos(t)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_sin_sum_triple_identity_zero_expr(&ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, IdentityZeroRewriteKind::SinSumTriple);
    }

    #[test]
    fn matches_sin_sum_triple_identity_zero_with_rational_context() {
        let mut ctx = Context::new();
        let expr = parse(
            "sin((1/(x - 1) + 1/(x + 1))) + sin(3*(1/(x - 1) + 1/(x + 1))) - 2*sin(2*(1/(x - 1) + 1/(x + 1)))*cos((1/(x - 1) + 1/(x + 1)))",
            &mut ctx,
        )
        .expect("expr");
        assert!(match_sin_sum_triple_identity_zero_expr(&ctx, expr));
    }

    #[test]
    fn matches_sin_sum_triple_identity_zero_after_distributing_scalar_over_sum() {
        let mut ctx = Context::new();
        let expr = parse(
            "sin(u^3 + 1) + sin(3*u^3 + 3) - 2*sin(2*u^3 + 2)*cos(u^3 + 1)",
            &mut ctx,
        )
        .expect("expr");
        assert!(match_sin_sum_triple_identity_zero_expr(&ctx, expr));
    }

    #[test]
    fn matches_sin_sum_triple_identity_zero_after_fraction_scalar_pullout() {
        let mut ctx = Context::new();
        let expr = parse(
            "sin((u*2)/(u^2 - 1)) + sin((u*6)/(u^2 - 1)) - 2*sin((u*4)/(u^2 - 1))*cos((u*2)/(u^2 - 1))",
            &mut ctx,
        )
        .expect("expr");
        assert!(match_sin_sum_triple_identity_zero_expr(&ctx, expr));
    }

    #[test]
    fn matches_sin_sum_triple_identity_zero_with_nested_scaled_argument() {
        let mut ctx = Context::new();
        let expr = parse(
            "sin(2*u) + sin(3*(2*u)) - 2*sin(2*(2*u))*cos(2*u)",
            &mut ctx,
        )
        .expect("expr");
        assert!(match_sin_sum_triple_identity_zero_expr(&ctx, expr));
    }

    #[test]
    fn matches_cos_triple_identity_zero_variants() {
        let mut ctx = Context::new();
        let expr1 = parse("cos(3*t) - (4*cos(t)^3 - 3*cos(t))", &mut ctx).expect("expr1");
        let expr2 = parse("(4*cos(t)^3 - 3*cos(t)) - cos(3*t)", &mut ctx).expect("expr2");
        let wrong = parse("cos(3*t) - (4*cos(u)^3 - 3*cos(u))", &mut ctx).expect("wrong");

        assert!(match_cos_triple_identity_zero_expr(&ctx, expr1));
        assert!(match_cos_triple_identity_zero_expr(&ctx, expr2));
        assert!(!match_cos_triple_identity_zero_expr(&ctx, wrong));
    }

    #[test]
    fn cos_triple_identity_zero_rewrite_plan_matches() {
        let mut ctx = Context::new();
        let expr = parse("cos(3*t) - (4*cos(t)^3 - 3*cos(t))", &mut ctx).expect("expr");
        let rewrite = try_rewrite_cos_triple_identity_zero_expr(&ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, IdentityZeroRewriteKind::CosTriple);
    }

    #[test]
    fn matches_cos_triple_identity_zero_with_arctan() {
        let mut ctx = Context::new();
        let expr = parse(
            "cos(3*arctan(x)) - (4*cos(arctan(x))^3 - 3*cos(arctan(x)))",
            &mut ctx,
        )
        .expect("expr");
        assert!(match_cos_triple_identity_zero_expr(&ctx, expr));
    }
}
