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
    pub desc: &'static str,
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
        desc: "tan(a-b) = (tan(a)-tan(b))/(1+tan(a)·tan(b))",
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
        desc: "sin(4t) = 4·sin(t)·cos(t)·(cos²(t)-sin²(t))",
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
        assert_eq!(rewrite.desc, "tan(a-b) = (tan(a)-tan(b))/(1+tan(a)·tan(b))");
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
        assert_eq!(rewrite.desc, "sin(4t) = 4·sin(t)·cos(t)·(cos²(t)-sin²(t))");
    }
}
