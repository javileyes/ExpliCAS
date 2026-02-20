use crate::trig_half_angle_support::{extract_tan_half_angle, is_half_angle};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::One;
use std::cmp::Ordering;

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
}
