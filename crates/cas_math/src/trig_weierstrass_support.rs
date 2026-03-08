use crate::trig_half_angle_support::{extract_tan_half_angle, is_half_angle};
use crate::trig_identity_zero_support::{IdentityZeroRewrite, IdentityZeroRewriteKind};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::One;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeierstrassContractionRewrite {
    pub rewritten: ExprId,
    pub kind: WeierstrassContractionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeierstrassContractionKind {
    Sin,
    Cos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeierstrassSubstitutionKind {
    Sin,
    Cos,
    Tan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeierstrassSubstitutionRewrite {
    pub rewritten: ExprId,
    pub arg: ExprId,
    pub kind: WeierstrassSubstitutionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReverseWeierstrassSinRewrite {
    pub rewritten: ExprId,
    pub arg: ExprId,
}

/// Extract the full angle `x` from either `tan(x/2)` or `sin(x/2)/cos(x/2)`.
pub fn extract_tan_half_angle_like(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Some(full_angle) = extract_tan_half_angle(ctx, expr) {
        return Some(full_angle);
    }

    if let Expr::Div(sin_id, cos_id) = ctx.get(expr) {
        if let (Expr::Function(sin_fn, sin_args), Expr::Function(cos_fn, cos_args)) =
            (ctx.get(*sin_id), ctx.get(*cos_id))
        {
            if matches!(ctx.builtin_of(*sin_fn), Some(BuiltinFn::Sin))
                && matches!(ctx.builtin_of(*cos_fn), Some(BuiltinFn::Cos))
                && sin_args.len() == 1
                && cos_args.len() == 1
            {
                let sin_arg = sin_args[0];
                let cos_arg = cos_args[0];
                if cas_ast::ordering::compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal {
                    return is_half_angle(ctx, sin_arg);
                }
            }
        }
    }

    None
}

/// Match `2*tan(x/2)` (or `2*sin(x/2)/cos(x/2)`) and return `x`.
pub fn match_two_tan_half(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let two_rat = num_rational::BigRational::from_integer(2.into());
    if let Expr::Mul(l, r) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*l) {
            if *n == two_rat {
                return extract_tan_half_angle_like(ctx, *r);
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if *n == two_rat {
                return extract_tan_half_angle_like(ctx, *l);
            }
        }
    }
    None
}

/// Match `1 + tan(x/2)^2` and return `(x, tan_half_expr)`.
pub fn match_one_plus_tan_half_squared(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if let Expr::Add(l, r) = ctx.get(expr) {
        let two_rat = num_rational::BigRational::from_integer(2.into());
        let (_one_id, pow_id) = if matches!(ctx.get(*l), Expr::Number(n) if n.is_one()) {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Number(n) if n.is_one()) {
            (*r, *l)
        } else {
            return None;
        };

        if let Expr::Pow(base, exp) = ctx.get(pow_id) {
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n == two_rat {
                    if let Some(full_angle) = extract_tan_half_angle_like(ctx, *base) {
                        return Some((full_angle, *base));
                    }
                }
            }
        }
    }
    None
}

/// Match `1 - tan(x/2)^2` (or `1 + (-tan(x/2)^2)`) and return `(x, tan_half_expr)`.
pub fn match_one_minus_tan_half_squared(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let two_rat = num_rational::BigRational::from_integer(2.into());

    if let Expr::Sub(l, r) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n.is_one() {
                if let Expr::Pow(base, exp) = ctx.get(*r) {
                    if let Expr::Number(e) = ctx.get(*exp) {
                        if *e == two_rat {
                            if let Some(full_angle) = extract_tan_half_angle_like(ctx, *base) {
                                return Some((full_angle, *base));
                            }
                        }
                    }
                }
            }
        }
    }

    if let Expr::Add(l, r) = ctx.get(expr) {
        let (_one_id, neg_id) = if matches!(ctx.get(*l), Expr::Number(n) if n.is_one()) {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Number(n) if n.is_one()) {
            (*r, *l)
        } else {
            return None;
        };

        if let Expr::Neg(inner) = ctx.get(neg_id) {
            if let Expr::Pow(base, exp) = ctx.get(*inner) {
                if let Expr::Number(e) = ctx.get(*exp) {
                    if *e == two_rat {
                        if let Some(full_angle) = extract_tan_half_angle_like(ctx, *base) {
                            return Some((full_angle, *base));
                        }
                    }
                }
            }
        }
    }

    None
}

/// Build the Weierstrass substitution image of `sin(x)` using `t = tan(x/2)`.
/// Returns `2t/(1+t^2)`.
pub fn build_weierstrass_sin(ctx: &mut Context, t: ExprId) -> ExprId {
    let two = ctx.num(2);
    let one = ctx.num(1);
    let t_squared = ctx.add(Expr::Pow(t, two));
    let numerator = crate::expr_rewrite::smart_mul(ctx, two, t);
    let denominator = ctx.add(Expr::Add(one, t_squared));
    ctx.add(Expr::Div(numerator, denominator))
}

/// Build the Weierstrass substitution image of `cos(x)` using `t = tan(x/2)`.
/// Returns `(1-t^2)/(1+t^2)`.
pub fn build_weierstrass_cos(ctx: &mut Context, t: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let t_squared = ctx.add(Expr::Pow(t, two));
    let numerator = ctx.add(Expr::Sub(one, t_squared));
    let denominator = ctx.add(Expr::Add(one, t_squared));
    ctx.add(Expr::Div(numerator, denominator))
}

/// Build the Weierstrass substitution image of `tan(x)` using `t = tan(x/2)`.
/// Returns `2t/(1-t^2)`.
pub fn build_weierstrass_tan(ctx: &mut Context, t: ExprId) -> ExprId {
    let two = ctx.num(2);
    let one = ctx.num(1);
    let t_squared = ctx.add(Expr::Pow(t, two));
    let numerator = crate::expr_rewrite::smart_mul(ctx, two, t);
    let denominator = ctx.add(Expr::Sub(one, t_squared));
    ctx.add(Expr::Div(numerator, denominator))
}

/// Rewrite `sin(x)`, `cos(x)`, `tan(x)` via Weierstrass substitution
/// with `t = tan(x/2)` represented as `sin(x/2)/cos(x/2)`.
pub fn try_rewrite_weierstrass_substitution_function_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<WeierstrassSubstitutionRewrite> {
    // Extract data first to avoid borrow conflicts.
    let (fn_id, arg) = match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
        _ => return None,
    };

    let kind = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sin) => WeierstrassSubstitutionKind::Sin,
        Some(BuiltinFn::Cos) => WeierstrassSubstitutionKind::Cos,
        Some(BuiltinFn::Tan) => WeierstrassSubstitutionKind::Tan,
        _ => return None,
    };

    // Build t = tan(x/2) as sin(x/2)/cos(x/2).
    let half = ctx.add(Expr::Number(num_rational::BigRational::new(
        1.into(),
        2.into(),
    )));
    let half_arg = crate::expr_rewrite::smart_mul(ctx, half, arg);
    let sin_half = ctx.call_builtin(BuiltinFn::Sin, vec![half_arg]);
    let cos_half = ctx.call_builtin(BuiltinFn::Cos, vec![half_arg]);
    let t = ctx.add(Expr::Div(sin_half, cos_half));

    let rewritten = match kind {
        WeierstrassSubstitutionKind::Sin => build_weierstrass_sin(ctx, t),
        WeierstrassSubstitutionKind::Cos => build_weierstrass_cos(ctx, t),
        WeierstrassSubstitutionKind::Tan => build_weierstrass_tan(ctx, t),
    };

    Some(WeierstrassSubstitutionRewrite {
        rewritten,
        arg,
        kind,
    })
}

/// Rewrite `2*tan(x/2)/(1+tan(x/2)^2)` (including `sin/cos` ratio form for `tan(x/2)`)
/// back to `sin(x)`.
pub fn try_rewrite_reverse_weierstrass_sin_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ReverseWeierstrassSinRewrite> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };

    let num_angle = match_two_tan_half(ctx, *num)?;
    let (den_angle, _) = match_one_plus_tan_half_squared(ctx, *den)?;
    if cas_ast::ordering::compare_expr(ctx, num_angle, den_angle) != Ordering::Equal {
        return None;
    }

    let rewritten = ctx.call_builtin(BuiltinFn::Sin, vec![num_angle]);
    Some(ReverseWeierstrassSinRewrite {
        rewritten,
        arg: num_angle,
    })
}

/// Detect and contract classic Weierstrass half-angle tangent forms:
/// - `2*tan(x/2) / (1 + tan(x/2)^2) -> sin(x)`
/// - `(1 - tan(x/2)^2) / (1 + tan(x/2)^2) -> cos(x)`
pub fn try_rewrite_weierstrass_contraction_div_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<WeierstrassContractionRewrite> {
    let Expr::Div(num_id, den_id) = ctx.get(expr) else {
        return None;
    };
    let (num_id, den_id) = (*num_id, *den_id);

    if let Some((full_angle, _tan_half)) = match_one_plus_tan_half_squared(ctx, den_id) {
        if let Some(num_angle) = match_two_tan_half(ctx, num_id) {
            if cas_ast::ordering::compare_expr(ctx, full_angle, num_angle) == Ordering::Equal {
                let sin_x = ctx.call_builtin(BuiltinFn::Sin, vec![full_angle]);
                return Some(WeierstrassContractionRewrite {
                    rewritten: sin_x,
                    kind: WeierstrassContractionKind::Sin,
                });
            }
        }
    }

    if let Some((num_angle, _)) = match_one_minus_tan_half_squared(ctx, num_id) {
        if let Some((den_angle, _)) = match_one_plus_tan_half_squared(ctx, den_id) {
            if cas_ast::ordering::compare_expr(ctx, num_angle, den_angle) == Ordering::Equal {
                let cos_x = ctx.call_builtin(BuiltinFn::Cos, vec![num_angle]);
                return Some(WeierstrassContractionRewrite {
                    rewritten: cos_x,
                    kind: WeierstrassContractionKind::Cos,
                });
            }
        }
    }

    None
}

fn extract_sub_like_operands(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
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

fn match_weierstrass_sin_identity_zero_pair(ctx: &Context, sin_side: ExprId, rhs: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(sin_side) else {
        return false;
    };
    if !matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sin)) || args.len() != 1 {
        return false;
    }
    let full_angle = args[0];

    let Expr::Div(num, den) = ctx.get(rhs) else {
        return false;
    };
    let Some(num_angle) = match_two_tan_half(ctx, *num) else {
        return false;
    };
    let Some((den_angle, _)) = match_one_plus_tan_half_squared(ctx, *den) else {
        return false;
    };

    cas_ast::ordering::compare_expr(ctx, full_angle, num_angle) == Ordering::Equal
        && cas_ast::ordering::compare_expr(ctx, full_angle, den_angle) == Ordering::Equal
}

fn match_weierstrass_cos_identity_zero_pair(ctx: &Context, cos_side: ExprId, rhs: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(cos_side) else {
        return false;
    };
    if !matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) || args.len() != 1 {
        return false;
    }
    let full_angle = args[0];

    let Expr::Div(num, den) = ctx.get(rhs) else {
        return false;
    };
    let Some((num_angle, _)) = match_one_minus_tan_half_squared(ctx, *num) else {
        return false;
    };
    let Some((den_angle, _)) = match_one_plus_tan_half_squared(ctx, *den) else {
        return false;
    };

    cas_ast::ordering::compare_expr(ctx, full_angle, num_angle) == Ordering::Equal
        && cas_ast::ordering::compare_expr(ctx, full_angle, den_angle) == Ordering::Equal
}

/// Match:
/// `sin(x) - 2*tan(x/2)/(1+tan(x/2)^2)` (or swapped sides) for identity-zero cancellation.
pub fn match_weierstrass_sin_identity_zero_expr(ctx: &Context, expr: ExprId) -> bool {
    let Some((left, right)) = extract_sub_like_operands(ctx, expr) else {
        return false;
    };
    match_weierstrass_sin_identity_zero_pair(ctx, left, right)
        || match_weierstrass_sin_identity_zero_pair(ctx, right, left)
}

/// Match:
/// `cos(x) - (1-tan(x/2)^2)/(1+tan(x/2)^2)` (or swapped sides) for identity-zero cancellation.
pub fn match_weierstrass_cos_identity_zero_expr(ctx: &Context, expr: ExprId) -> bool {
    let Some((left, right)) = extract_sub_like_operands(ctx, expr) else {
        return false;
    };
    match_weierstrass_cos_identity_zero_pair(ctx, left, right)
        || match_weierstrass_cos_identity_zero_pair(ctx, right, left)
}

/// Rewrite plan for
/// `sin(x) - 2*tan(x/2)/(1+tan(x/2)^2) -> 0`.
pub fn try_rewrite_weierstrass_sin_identity_zero_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<IdentityZeroRewrite> {
    if !match_weierstrass_sin_identity_zero_expr(ctx, expr) {
        return None;
    }
    Some(IdentityZeroRewrite {
        kind: IdentityZeroRewriteKind::WeierstrassSin,
    })
}

/// Rewrite plan for
/// `cos(x) - (1-tan(x/2)^2)/(1+tan(x/2)^2) -> 0`.
pub fn try_rewrite_weierstrass_cos_identity_zero_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<IdentityZeroRewrite> {
    if !match_weierstrass_cos_identity_zero_expr(ctx, expr) {
        return None;
    }
    Some(IdentityZeroRewrite {
        kind: IdentityZeroRewriteKind::WeierstrassCos,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn extract_tan_half_angle_like_matches_tan_and_ratio_forms() {
        let mut ctx = Context::new();
        let tan = parse("tan(x/2)", &mut ctx).expect("tan");
        let ratio = parse("sin(x/2)/cos(x/2)", &mut ctx).expect("ratio");
        let x = parse("x", &mut ctx).expect("x");

        let tan_out = extract_tan_half_angle_like(&ctx, tan).expect("tan out");
        let ratio_out = extract_tan_half_angle_like(&ctx, ratio).expect("ratio out");

        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, tan_out, x),
            Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, ratio_out, x),
            Ordering::Equal
        );
    }

    #[test]
    fn plus_and_minus_tan_squared_matchers_detect_expected_patterns() {
        let mut ctx = Context::new();
        let plus = parse("1 + tan(x/2)^2", &mut ctx).expect("plus");
        let minus = parse("1 - tan(x/2)^2", &mut ctx).expect("minus");

        assert!(match_one_plus_tan_half_squared(&ctx, plus).is_some());
        assert!(match_one_minus_tan_half_squared(&ctx, minus).is_some());
    }

    #[test]
    fn two_tan_half_matcher_handles_tan_and_ratio() {
        let mut ctx = Context::new();
        let tan_form = parse("2*tan(x/2)", &mut ctx).expect("tan form");
        let ratio_form = parse("2*(sin(x/2)/cos(x/2))", &mut ctx).expect("ratio form");

        assert!(match_two_tan_half(&ctx, tan_form).is_some());
        assert!(match_two_tan_half(&ctx, ratio_form).is_some());
    }

    #[test]
    fn weierstrass_builders_emit_expected_forms() {
        let mut ctx = Context::new();
        let t = parse("t", &mut ctx).expect("t");
        let expected_sin = parse("2*t/(1+t^2)", &mut ctx).expect("expected sin");
        let expected_cos = parse("(1-t^2)/(1+t^2)", &mut ctx).expect("expected cos");
        let expected_tan = parse("2*t/(1-t^2)", &mut ctx).expect("expected tan");

        let sin_form = build_weierstrass_sin(&mut ctx, t);
        let cos_form = build_weierstrass_cos(&mut ctx, t);
        let tan_form = build_weierstrass_tan(&mut ctx, t);

        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, sin_form, expected_sin),
            Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, cos_form, expected_cos),
            Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, tan_form, expected_tan),
            Ordering::Equal
        );
    }

    #[test]
    fn rewrites_weierstrass_contraction_to_sin_and_cos() {
        let mut ctx = Context::new();
        let sin_form = parse("2*tan(x/2)/(1+tan(x/2)^2)", &mut ctx).expect("sin form");
        let cos_form = parse("(1-tan(x/2)^2)/(1+tan(x/2)^2)", &mut ctx).expect("cos form");

        let sin_rw = try_rewrite_weierstrass_contraction_div_expr(&mut ctx, sin_form)
            .expect("sin contraction");
        let cos_rw = try_rewrite_weierstrass_contraction_div_expr(&mut ctx, cos_form)
            .expect("cos contraction");

        let expected_sin = parse("sin(x)", &mut ctx).expect("expected sin");
        let expected_cos = parse("cos(x)", &mut ctx).expect("expected cos");

        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, sin_rw.rewritten, expected_sin),
            Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, cos_rw.rewritten, expected_cos),
            Ordering::Equal
        );
    }

    #[test]
    fn rewrite_weierstrass_substitution_function_expr_handles_all_three_trig_functions() {
        let mut ctx = Context::new();
        let sin_expr = parse("sin(x)", &mut ctx).expect("sin");
        let cos_expr = parse("cos(x)", &mut ctx).expect("cos");
        let tan_expr = parse("tan(x)", &mut ctx).expect("tan");

        let sin_rw =
            try_rewrite_weierstrass_substitution_function_expr(&mut ctx, sin_expr).expect("sin rw");
        let cos_rw =
            try_rewrite_weierstrass_substitution_function_expr(&mut ctx, cos_expr).expect("cos rw");
        let tan_rw =
            try_rewrite_weierstrass_substitution_function_expr(&mut ctx, tan_expr).expect("tan rw");

        assert_eq!(sin_rw.kind, WeierstrassSubstitutionKind::Sin);
        assert_eq!(cos_rw.kind, WeierstrassSubstitutionKind::Cos);
        assert_eq!(tan_rw.kind, WeierstrassSubstitutionKind::Tan);
    }

    #[test]
    fn rewrite_reverse_weierstrass_sin_expr_matches_canonical_form() {
        let mut ctx = Context::new();
        let expr = parse("2*tan(x/2)/(1+tan(x/2)^2)", &mut ctx).expect("expr");
        let expected = parse("sin(x)", &mut ctx).expect("expected");

        let rw = try_rewrite_reverse_weierstrass_sin_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rw.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn matches_weierstrass_identity_zero_sin_and_cos_forms() {
        let mut ctx = Context::new();
        let sin_expr = parse("sin(x) - 2*tan(x/2)/(1+tan(x/2)^2)", &mut ctx).expect("sin expr");
        let cos_expr = parse("cos(x) - (1-tan(x/2)^2)/(1+tan(x/2)^2)", &mut ctx).expect("cos expr");

        assert!(match_weierstrass_sin_identity_zero_expr(&ctx, sin_expr));
        assert!(match_weierstrass_cos_identity_zero_expr(&ctx, cos_expr));
    }

    #[test]
    fn weierstrass_identity_zero_rewrite_plans_match() {
        let mut ctx = Context::new();
        let sin_expr = parse("sin(x) - 2*tan(x/2)/(1+tan(x/2)^2)", &mut ctx).expect("sin expr");
        let cos_expr = parse("cos(x) - (1-tan(x/2)^2)/(1+tan(x/2)^2)", &mut ctx).expect("cos expr");

        let sin_rw = try_rewrite_weierstrass_sin_identity_zero_expr(&ctx, sin_expr).expect("sin");
        let cos_rw = try_rewrite_weierstrass_cos_identity_zero_expr(&ctx, cos_expr).expect("cos");
        assert_eq!(sin_rw.kind, IdentityZeroRewriteKind::WeierstrassSin);
        assert_eq!(cos_rw.kind, IdentityZeroRewriteKind::WeierstrassCos);
    }
}
