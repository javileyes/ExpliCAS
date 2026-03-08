use crate::expr_destructure::{as_add, as_div, as_mul, as_neg, as_sub};
use crate::expr_nary::{add_terms_signed, mul_leaves, Sign};
use crate::pattern_marks::PatternMarks;
use crate::trig_roots_flatten::extract_double_angle_arg;
use crate::trig_sum_product_support::extract_trig_arg;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrigContractionRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SquareDoubleAngleContractionRewrite {
    pub rewritten: ExprId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneralizedSinCosContractionRewrite {
    pub rewritten: ExprId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TanDoubleAngleContractionRewrite {
    pub rewritten: ExprId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalfAngleTangentRewrite {
    pub rewritten: ExprId,
    pub inherited_nonzero: ExprId,
    pub required_nonzero: ExprId,
    pub kind: HalfAngleTangentRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfAngleTangentRewriteKind {
    OneMinusCosOverSin,
    SinOverOnePlusCos,
}

/// Policy guard for double-angle contraction when the `sin(4x)` identity pattern
/// is pre-marked. Keeping this centralized allows engine rules to remain thin.
pub fn should_block_double_angle_contraction_for_marks(marks: Option<&PatternMarks>) -> bool {
    marks.is_some_and(|m| m.has_sin4x_identity_pattern)
}

fn is_tan_squared_of_arg(ctx: &Context, expr: ExprId, tan_arg: ExprId) -> bool {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        let exp_is_2 = matches!(ctx.get(*exp), Expr::Number(n)
            if *n == BigRational::from_integer(2.into()));
        if exp_is_2 {
            if let Expr::Function(fn_id, args) = ctx.get(*base) {
                return matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tan))
                    && args.len() == 1
                    && compare_expr(ctx, args[0], tan_arg) == Ordering::Equal;
            }
        }
    }
    false
}

/// Detect `2*tan(t)/(1-tan(t)^2)` and return `t`.
///
/// Accepted denominator shapes:
/// - `1 - tan(t)^2`
/// - `1 + (-(tan(t)^2))`
pub fn match_tan_double_angle_contraction_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let (num, den) = as_div(ctx, expr)?;

    let tan_arg = if let Expr::Mul(l, r) = ctx.get(num) {
        let (l, r) = (*l, *r);
        let tan_part = if matches!(ctx.get(l), Expr::Number(n) if *n == BigRational::from_integer(2.into()))
        {
            Some(r)
        } else if matches!(ctx.get(r), Expr::Number(n) if *n == BigRational::from_integer(2.into()))
        {
            Some(l)
        } else {
            None
        }?;

        if let Expr::Function(fn_id, args) = ctx.get(tan_part) {
            if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tan)) && args.len() == 1 {
                Some(args[0])
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    }?;

    let den_matches = if let Expr::Sub(one_part, tan2_part) = ctx.get(den) {
        matches!(ctx.get(*one_part), Expr::Number(n)
            if n.is_integer() && *n == BigRational::from_integer(1.into()))
            && is_tan_squared_of_arg(ctx, *tan2_part, tan_arg)
    } else if let Expr::Add(l, r) = ctx.get(den) {
        let (one_part, neg_part) = if matches!(ctx.get(*l), Expr::Number(n)
            if n.is_integer() && *n == BigRational::from_integer(1.into()))
        {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Number(n)
            if n.is_integer() && *n == BigRational::from_integer(1.into()))
        {
            (*r, *l)
        } else {
            return None;
        };
        let _ = one_part;
        if let Expr::Neg(inner) = ctx.get(neg_part) {
            is_tan_squared_of_arg(ctx, *inner, tan_arg)
        } else {
            false
        }
    } else {
        false
    };

    if den_matches {
        Some(tan_arg)
    } else {
        None
    }
}

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

fn extract_trig_squared_factor(ctx: &Context, factor: ExprId) -> Option<(ExprId, bool)> {
    let two_rat = BigRational::from_integer(2.into());
    if let Expr::Pow(base, exp) = ctx.get(factor) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if *n == two_rat {
                if let Expr::Function(fn_id, args) = ctx.get(*base) {
                    if args.len() == 1 {
                        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sin)) {
                            return Some((args[0], true));
                        }
                        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) {
                            return Some((args[0], false));
                        }
                    }
                }
            }
        }
    }
    None
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

/// Rewrite half-angle tangent identities:
/// - `(1 - cos(2x)) / sin(2x) -> tan(x)`
/// - `sin(2x) / (1 + cos(2x)) -> tan(x)`
///
/// Returns the rewritten node plus both non-zero conditions needed by the caller:
/// - `inherited_nonzero`: original denominator
/// - `required_nonzero`: `cos(x)` for `tan(x)` domain
pub fn try_rewrite_half_angle_tangent_div_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HalfAngleTangentRewrite> {
    let (num_id, den_id) = as_div(ctx, expr)?;

    enum Pattern {
        OneMinusCosOverSin { x: ExprId, sin_2x: ExprId },
        SinOverOnePlusCos { x: ExprId, one_plus_cos_2x: ExprId },
    }

    let pattern = 'pattern: {
        let try_extract_cos_2x = |ctx: &Context, id: ExprId| -> Option<(ExprId, bool)> {
            if let Expr::Function(fn_id, args) = ctx.get(id) {
                if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) && args.len() == 1 {
                    return extract_double_angle_arg(ctx, args[0]).map(|x| (x, false));
                }
            }
            if let Expr::Neg(inner) = ctx.get(id) {
                if let Expr::Function(fn_id, args) = ctx.get(*inner) {
                    if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) && args.len() == 1 {
                        return extract_double_angle_arg(ctx, args[0]).map(|x| (x, true));
                    }
                }
            }
            None
        };

        // Pattern 1a: Sub(1, cos(2x))
        if let Expr::Sub(one_id, cos_id) = ctx.get(num_id) {
            if let Expr::Number(n) = ctx.get(*one_id) {
                if n.is_integer() && *n == BigRational::from_integer(1.into()) {
                    if let Some((x, false)) = try_extract_cos_2x(ctx, *cos_id) {
                        if let Expr::Function(den_fn_id, den_args) = ctx.get(den_id) {
                            if matches!(ctx.builtin_of(*den_fn_id), Some(BuiltinFn::Sin))
                                && den_args.len() == 1
                            {
                                if let Some(x2) = extract_double_angle_arg(ctx, den_args[0]) {
                                    if compare_expr(ctx, x, x2) == Ordering::Equal {
                                        break 'pattern Some(Pattern::OneMinusCosOverSin {
                                            x,
                                            sin_2x: den_id,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Pattern 1b: Add(1, Neg(cos(2x))) or swapped
        if let Expr::Add(left, right) = ctx.get(num_id) {
            let try_order = |one: ExprId, neg_cos: ExprId| -> Option<ExprId> {
                if let Expr::Number(n) = ctx.get(one) {
                    if n.is_integer() && *n == BigRational::from_integer(1.into()) {
                        if let Some((x, true)) = try_extract_cos_2x(ctx, neg_cos) {
                            return Some(x);
                        }
                    }
                }
                None
            };

            let x_opt = try_order(*left, *right).or_else(|| try_order(*right, *left));
            if let Some(x) = x_opt {
                if let Expr::Function(den_fn_id, den_args) = ctx.get(den_id) {
                    if matches!(ctx.builtin_of(*den_fn_id), Some(BuiltinFn::Sin))
                        && den_args.len() == 1
                    {
                        if let Some(x2) = extract_double_angle_arg(ctx, den_args[0]) {
                            if compare_expr(ctx, x, x2) == Ordering::Equal {
                                break 'pattern Some(Pattern::OneMinusCosOverSin {
                                    x,
                                    sin_2x: den_id,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Pattern 2: sin(2x)/(1+cos(2x))
        if let Expr::Function(fn_id, args) = ctx.get(num_id) {
            if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sin)) && args.len() == 1 {
                if let Some(x) = extract_double_angle_arg(ctx, args[0]) {
                    if let Expr::Add(left, right) = ctx.get(den_id) {
                        let (one_id, cos_id) = if matches!(ctx.get(*left), Expr::Number(n) if n.is_integer() && *n == BigRational::from_integer(1.into()))
                        {
                            (*left, *right)
                        } else if matches!(ctx.get(*right), Expr::Number(n) if n.is_integer() && *n == BigRational::from_integer(1.into()))
                        {
                            (*right, *left)
                        } else {
                            break 'pattern None;
                        };

                        if let Expr::Number(n) = ctx.get(one_id) {
                            if n.is_integer() && *n == BigRational::from_integer(1.into()) {
                                if let Expr::Function(cos_fn_id, cos_args) = ctx.get(cos_id) {
                                    if matches!(ctx.builtin_of(*cos_fn_id), Some(BuiltinFn::Cos))
                                        && cos_args.len() == 1
                                    {
                                        if let Some(x2) = extract_double_angle_arg(ctx, cos_args[0])
                                        {
                                            if compare_expr(ctx, x, x2) == Ordering::Equal {
                                                break 'pattern Some(Pattern::SinOverOnePlusCos {
                                                    x,
                                                    one_plus_cos_2x: den_id,
                                                });
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
    }?;

    let (x, inherited_nonzero, kind) = match pattern {
        Pattern::OneMinusCosOverSin { x, sin_2x } => {
            (x, sin_2x, HalfAngleTangentRewriteKind::OneMinusCosOverSin)
        }
        Pattern::SinOverOnePlusCos { x, one_plus_cos_2x } => (
            x,
            one_plus_cos_2x,
            HalfAngleTangentRewriteKind::SinOverOnePlusCos,
        ),
    };

    let rewritten = ctx.call_builtin(BuiltinFn::Tan, vec![x]);
    let required_nonzero = ctx.call_builtin(BuiltinFn::Cos, vec![x]);
    Some(HalfAngleTangentRewrite {
        rewritten,
        inherited_nonzero,
        required_nonzero,
        kind,
    })
}

/// Rewrite:
/// - `2*sin(t)*cos(t) -> sin(2t)`
/// - `cos(t)^2 - sin(t)^2 -> cos(2t)`
pub fn try_rewrite_double_angle_contraction_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigContractionRewrite> {
    if let Some((l, r)) = as_mul(ctx, expr) {
        if let Some((sin_arg, cos_arg)) = extract_two_sin_cos(ctx, l, r) {
            if compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal {
                let two = ctx.num(2);
                let double_arg = ctx.add(Expr::Mul(two, sin_arg));
                let sin_2t = ctx.call_builtin(BuiltinFn::Sin, vec![double_arg]);
                return Some(TrigContractionRewrite {
                    rewritten: sin_2t,
                    desc: "2·sin(t)·cos(t) = sin(2t)",
                });
            }
        }
    }

    if let Some((l, r)) = as_sub(ctx, expr) {
        if let Some((cos_arg, sin_arg)) = extract_cos2_minus_sin2(ctx, l, r) {
            if compare_expr(ctx, cos_arg, sin_arg) == Ordering::Equal {
                let two = ctx.num(2);
                let double_arg = ctx.add(Expr::Mul(two, cos_arg));
                let cos_2t = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);
                return Some(TrigContractionRewrite {
                    rewritten: cos_2t,
                    desc: "cos²(t) - sin²(t) = cos(2t)",
                });
            }
        }
    }

    None
}

/// Rewrite:
/// - `sin(t)^2*cos(t)^2 -> sin(2t)^2/4`
pub fn try_rewrite_square_double_angle_contraction_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<SquareDoubleAngleContractionRewrite> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let mut sin_arg = None;
    let mut cos_arg = None;
    for &factor in &factors {
        let (arg, is_sin) = extract_trig_squared_factor(ctx, factor)?;
        if is_sin {
            if sin_arg.is_some() {
                return None;
            }
            sin_arg = Some(arg);
        } else {
            if cos_arg.is_some() {
                return None;
            }
            cos_arg = Some(arg);
        }
    }

    let sin_arg = sin_arg?;
    let cos_arg = cos_arg?;
    if compare_expr(ctx, sin_arg, cos_arg) != Ordering::Equal {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let double_arg = ctx.add(Expr::Mul(two, sin_arg));
    let sin_2t = ctx.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    let sin_2t_sq = ctx.add(Expr::Pow(sin_2t, two));
    let rewritten = ctx.add(Expr::Div(sin_2t_sq, four));
    Some(SquareDoubleAngleContractionRewrite { rewritten })
}

/// Rewrite generalized even-coefficient forms:
/// - `k*sin(t)*cos(t) -> (k/2)*sin(2t)` for even `k >= 4`.
///
/// Preserves additional multiplicative factors by multiplying them back.
pub fn try_rewrite_generalized_sin_cos_contraction_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<GeneralizedSinCosContractionRewrite> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 3 {
        return None;
    }

    let mut coef_idx: Option<usize> = None;
    let mut coef_val: Option<BigRational> = None;
    let mut sin_idx: Option<usize> = None;
    let mut sin_arg: Option<ExprId> = None;
    let mut cos_idx: Option<usize> = None;
    let mut cos_arg: Option<ExprId> = None;

    for (i, &factor) in factors.iter().enumerate() {
        if let Expr::Number(n) = ctx.get(factor) {
            let two = BigRational::from_integer(2.into());
            let four = BigRational::from_integer(4.into());
            if n >= &four && (n / &two).is_integer() {
                coef_idx = Some(i);
                coef_val = Some(n.clone());
                continue;
            }
        }

        if let Expr::Function(fn_id, args) = ctx.get(factor) {
            let builtin = ctx.builtin_of(*fn_id);
            if matches!(builtin, Some(BuiltinFn::Sin)) && args.len() == 1 && sin_idx.is_none() {
                sin_idx = Some(i);
                sin_arg = Some(args[0]);
                continue;
            }
            if matches!(builtin, Some(BuiltinFn::Cos)) && args.len() == 1 && cos_idx.is_none() {
                cos_idx = Some(i);
                cos_arg = Some(args[0]);
                continue;
            }
        }
    }

    let (c_i, c_val, s_i, s_arg, o_i, c_arg) =
        (coef_idx?, coef_val?, sin_idx?, sin_arg?, cos_idx?, cos_arg?);
    if compare_expr(ctx, s_arg, c_arg) != Ordering::Equal {
        return None;
    }

    let two = BigRational::from_integer(2.into());
    let half_coef = c_val / &two;
    let half_coef_expr = ctx.add(Expr::Number(half_coef));
    let two_expr = ctx.num(2);
    let double_arg = ctx.add(Expr::Mul(two_expr, s_arg));
    let sin_2t = ctx.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    let contracted = ctx.add(Expr::Mul(half_coef_expr, sin_2t));

    let mut remaining: Vec<ExprId> = Vec::new();
    for (j, &f) in factors.iter().enumerate() {
        if j != c_i && j != s_i && j != o_i {
            remaining.push(f);
        }
    }

    let rewritten = if remaining.is_empty() {
        contracted
    } else {
        let mut acc = contracted;
        for &f in &remaining {
            acc = ctx.add(Expr::Mul(acc, f));
        }
        acc
    };

    Some(GeneralizedSinCosContractionRewrite { rewritten })
}

/// Rewrite tangent double-angle contraction:
/// - `2*tan(t)/(1 - tan(t)^2) -> tan(2*t)`
pub fn try_rewrite_tan_double_angle_contraction_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TanDoubleAngleContractionRewrite> {
    let tan_arg = match_tan_double_angle_contraction_arg(ctx, expr)?;

    let two = ctx.num(2);
    let double_t = ctx.add(Expr::Mul(two, tan_arg));
    let rewritten = ctx.call_builtin(BuiltinFn::Tan, vec![double_t]);
    Some(TanDoubleAngleContractionRewrite { rewritten })
}

/// Rewrite:
/// - `1 - 2*sin(t)^2 -> cos(2t)`
/// - `2*cos(t)^2 - 1 -> cos(2t)`
pub fn try_rewrite_cos2x_additive_contraction_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigContractionRewrite> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let signed_terms = add_terms_signed(ctx, expr);
    if signed_terms.len() != 2 {
        return None;
    }

    let one_rat = BigRational::from_integer(1.into());
    let two_rat = BigRational::from_integer(2.into());
    let neg_two_rat = BigRational::from_integer((-2).into());

    for (i, &(term_i, sign_i)) in signed_terms.iter().enumerate() {
        let term_val = match ctx.get(term_i) {
            Expr::Number(n) => {
                if sign_i == Sign::Pos {
                    n.clone()
                } else {
                    -n.clone()
                }
            }
            _ => continue,
        };

        let is_pos_one = term_val == one_rat;
        let is_neg_one = term_val == -one_rat.clone();
        if !is_pos_one && !is_neg_one {
            continue;
        }

        for (j, &(term_j, sign_j)) in signed_terms.iter().enumerate() {
            if j == i {
                continue;
            }
            if let Some((trig_arg, trig_is_sin, mut coeff)) =
                extract_coeff_trig_squared(ctx, term_j)
            {
                if sign_j == Sign::Neg {
                    coeff = -coeff;
                }

                let matches = (is_pos_one && trig_is_sin && coeff == neg_two_rat)
                    || (is_neg_one && !trig_is_sin && coeff == two_rat);
                if !matches {
                    continue;
                }

                let two = ctx.num(2);
                let double_arg = ctx.add(Expr::Mul(two, trig_arg));
                let cos_2t = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);
                let desc = if trig_is_sin {
                    "1 - 2·sin²(t) = cos(2t)"
                } else {
                    "2·cos²(t) - 1 = cos(2t)"
                };
                return Some(TrigContractionRewrite {
                    rewritten: cos_2t,
                    desc,
                });
            }
        }
    }

    None
}

/// Rewrite:
/// - `(sin(a)cos(b)+cos(a)sin(b))/(cos(a)cos(b)-sin(a)sin(b)) -> tan(a+b)`
/// - `(sin(a)cos(b)-cos(a)sin(b))/(cos(a)cos(b)+sin(a)sin(b)) -> tan(a-b)`
pub fn try_rewrite_angle_sum_fraction_to_tan_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigContractionRewrite> {
    let (num, den) = as_div(ctx, expr)?;

    if let Some((a, b)) = match_angle_sum_fraction(ctx, num, den) {
        let sum_arg = ctx.add(Expr::Add(a, b));
        let tan_result = ctx.call_builtin(BuiltinFn::Tan, vec![sum_arg]);
        return Some(TrigContractionRewrite {
            rewritten: tan_result,
            desc: "(sin(a)cos(b)+cos(a)sin(b))/(cos(a)cos(b)-sin(a)sin(b)) = tan(a+b)",
        });
    }

    if let Some((a, b)) = match_angle_diff_fraction(ctx, num, den) {
        let diff_arg = ctx.add(Expr::Sub(a, b));
        let tan_result = ctx.call_builtin(BuiltinFn::Tan, vec![diff_arg]);
        return Some(TrigContractionRewrite {
            rewritten: tan_result,
            desc: "(sin(a)cos(b)-cos(a)sin(b))/(cos(a)cos(b)+sin(a)sin(b)) = tan(a-b)",
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn double_angle_contraction_policy_blocks_with_sin4x_mark() {
        let marks = PatternMarks {
            has_sin4x_identity_pattern: true,
            ..PatternMarks::default()
        };
        assert!(should_block_double_angle_contraction_for_marks(Some(
            &marks
        )));
        assert!(!should_block_double_angle_contraction_for_marks(None));
    }

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

    #[test]
    fn rewrites_double_angle_contraction_forms() {
        let mut ctx = Context::new();
        let expr1 = parse("2*sin(t)*cos(t)", &mut ctx).expect("expr1");
        let expected1 = parse("sin(2*t)", &mut ctx).expect("expected1");
        let rw1 = try_rewrite_double_angle_contraction_expr(&mut ctx, expr1).expect("rw1");
        assert_eq!(
            compare_expr(&ctx, rw1.rewritten, expected1),
            Ordering::Equal
        );

        let expr2 = parse("cos(t)^2-sin(t)^2", &mut ctx).expect("expr2");
        let expected2 = parse("cos(2*t)", &mut ctx).expect("expected2");
        let rw2 = try_rewrite_double_angle_contraction_expr(&mut ctx, expr2).expect("rw2");
        assert_eq!(
            compare_expr(&ctx, rw2.rewritten, expected2),
            Ordering::Equal
        );
    }

    #[test]
    fn rewrites_cos2x_additive_contraction_forms() {
        let mut ctx = Context::new();
        let expr1 = parse("1-2*sin(t)^2", &mut ctx).expect("expr1");
        let expected = parse("cos(2*t)", &mut ctx).expect("expected");
        let rw1 = try_rewrite_cos2x_additive_contraction_expr(&mut ctx, expr1).expect("rw1");
        assert_eq!(compare_expr(&ctx, rw1.rewritten, expected), Ordering::Equal);

        let expr2 = parse("2*cos(t)^2-1", &mut ctx).expect("expr2");
        let rw2 = try_rewrite_cos2x_additive_contraction_expr(&mut ctx, expr2).expect("rw2");
        assert_eq!(compare_expr(&ctx, rw2.rewritten, expected), Ordering::Equal);
    }

    #[test]
    fn rewrites_angle_sum_fraction_to_tan() {
        let mut ctx = Context::new();
        let expr = parse(
            "(sin(a)*cos(b)+cos(a)*sin(b))/(cos(a)*cos(b)-sin(a)*sin(b))",
            &mut ctx,
        )
        .expect("expr");
        let expected = parse("tan(a+b)", &mut ctx).expect("expected");
        let rw = try_rewrite_angle_sum_fraction_to_tan_expr(&mut ctx, expr).expect("rw");
        assert_eq!(compare_expr(&ctx, rw.rewritten, expected), Ordering::Equal);
    }

    #[test]
    fn rewrites_half_angle_tangent_pattern_one() {
        let mut ctx = Context::new();
        let expr = parse("(1-cos(2*x))/sin(2*x)", &mut ctx).expect("expr");
        let expected = parse("tan(x)", &mut ctx).expect("expected");
        let rw = try_rewrite_half_angle_tangent_div_expr(&mut ctx, expr).expect("rw");
        assert_eq!(compare_expr(&ctx, rw.rewritten, expected), Ordering::Equal);
    }

    #[test]
    fn rewrites_half_angle_tangent_pattern_two() {
        let mut ctx = Context::new();
        let expr = parse("sin(2*x)/(1+cos(2*x))", &mut ctx).expect("expr");
        let expected = parse("tan(x)", &mut ctx).expect("expected");
        let rw = try_rewrite_half_angle_tangent_div_expr(&mut ctx, expr).expect("rw");
        assert_eq!(compare_expr(&ctx, rw.rewritten, expected), Ordering::Equal);
    }

    #[test]
    fn rewrites_generalized_sin_cos_contraction() {
        let mut ctx = Context::new();
        let expr = parse("6*sin(t)*cos(t)", &mut ctx).expect("expr");
        let expected = parse("3*sin(2*t)", &mut ctx).expect("expected");
        let rw = try_rewrite_generalized_sin_cos_contraction_expr(&mut ctx, expr).expect("rw");
        assert_eq!(compare_expr(&ctx, rw.rewritten, expected), Ordering::Equal);
    }

    #[test]
    fn rewrites_square_double_angle_contraction() {
        let mut ctx = Context::new();
        let expr = parse("sin(t)^2*cos(t)^2", &mut ctx).expect("expr");
        let expected = parse("sin(2*t)^2/4", &mut ctx).expect("expected");
        let rw = try_rewrite_square_double_angle_contraction_expr(&mut ctx, expr).expect("rw");
        assert_eq!(compare_expr(&ctx, rw.rewritten, expected), Ordering::Equal);
    }

    #[test]
    fn rewrites_tan_double_angle_contraction() {
        let mut ctx = Context::new();
        let expr = parse("2*tan(t)/(1-tan(t)^2)", &mut ctx).expect("expr");
        let expected = parse("tan(2*t)", &mut ctx).expect("expected");
        let rw = try_rewrite_tan_double_angle_contraction_expr(&mut ctx, expr).expect("rw");
        assert_eq!(compare_expr(&ctx, rw.rewritten, expected), Ordering::Equal);
    }

    #[test]
    fn matches_tan_double_angle_contraction_arg() {
        let mut ctx = Context::new();
        let expr1 = parse("2*tan(t)/(1-tan(t)^2)", &mut ctx).expect("expr1");
        let expr2 = parse("2*tan(u)/(1+(-tan(u)^2))", &mut ctx).expect("expr2");
        let t = parse("t", &mut ctx).expect("t");
        let u = parse("u", &mut ctx).expect("u");

        let out1 = match_tan_double_angle_contraction_arg(&ctx, expr1).expect("out1");
        let out2 = match_tan_double_angle_contraction_arg(&ctx, expr2).expect("out2");

        assert_eq!(compare_expr(&ctx, out1, t), Ordering::Equal);
        assert_eq!(compare_expr(&ctx, out2, u), Ordering::Equal);
    }
}
