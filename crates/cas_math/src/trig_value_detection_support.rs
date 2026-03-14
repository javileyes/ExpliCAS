use crate::expr_destructure::as_fn1;
use crate::pattern_marks::PatternMarks;
use crate::pi_helpers::{extract_rational_pi_multiple, is_pi, is_pi_over_n};
use crate::root_forms::extract_numeric_sqrt_radicand;
use crate::trig_canonicalization_support::{
    is_inverse_trig_function_call, try_rewrite_tan_to_sin_cos_function_expr,
    TrigCanonicalRewritePlan,
};
use crate::trig_multi_angle_support::is_multiple_angle;
use crate::trig_tan_triple_support::is_part_of_tan_triple_product_with_ancestors;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

/// Angle key enum for special angles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpecialAngle {
    Zero,         // 0
    Pi,           // π
    PiOver2,      // π/2
    PiOver3,      // π/3
    PiOver4,      // π/4
    PiOver6,      // π/6
    PiOver8,      // π/8 = 22.5°
    PiOver12,     // π/12 = 15°
    ThreePiOver8, // 3π/8 = 67.5°
    FivePiOver12, // 5π/12 = 75°
    PiOver5,      // π/5 = 36° (golden angle)
    TwoPiOver5,   // 2π/5 = 72°
    PiOver10,     // π/10 = 18°
}

impl SpecialAngle {
    pub fn display(&self) -> &'static str {
        match self {
            Self::Zero => "0",
            Self::Pi => "π",
            Self::PiOver2 => "π/2",
            Self::PiOver3 => "π/3",
            Self::PiOver4 => "π/4",
            Self::PiOver6 => "π/6",
            Self::PiOver8 => "π/8",
            Self::PiOver12 => "π/12",
            Self::ThreePiOver8 => "3π/8",
            Self::FivePiOver12 => "5π/12",
            Self::PiOver5 => "π/5",
            Self::TwoPiOver5 => "2π/5",
            Self::PiOver10 => "π/10",
        }
    }
}

/// Detect if an expression represents a special angle.
///
/// Checks for 0, π, π/2, π/3, π/4, π/6, π/8, π/12, 3π/8, 5π/12, π/5, 2π/5, π/10.
pub fn detect_special_angle(ctx: &Context, expr: ExprId) -> Option<SpecialAngle> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_zero() {
            return Some(SpecialAngle::Zero);
        }
    }

    if is_pi(ctx, expr) {
        return Some(SpecialAngle::Pi);
    }

    if is_pi_over_n(ctx, expr, 2) {
        return Some(SpecialAngle::PiOver2);
    }
    if is_pi_over_n(ctx, expr, 3) {
        return Some(SpecialAngle::PiOver3);
    }
    if is_pi_over_n(ctx, expr, 4) {
        return Some(SpecialAngle::PiOver4);
    }
    if is_pi_over_n(ctx, expr, 6) {
        return Some(SpecialAngle::PiOver6);
    }
    if is_pi_over_n(ctx, expr, 8) {
        return Some(SpecialAngle::PiOver8);
    }
    if is_pi_over_n(ctx, expr, 12) {
        return Some(SpecialAngle::PiOver12);
    }
    if is_pi_over_n(ctx, expr, 5) {
        return Some(SpecialAngle::PiOver5);
    }
    if is_pi_over_n(ctx, expr, 10) {
        return Some(SpecialAngle::PiOver10);
    }

    if let Some(k) = extract_rational_pi_multiple(ctx, expr) {
        if k == BigRational::new(3.into(), 8.into()) {
            return Some(SpecialAngle::ThreePiOver8);
        }
        if k == BigRational::new(5.into(), 12.into()) {
            return Some(SpecialAngle::FivePiOver12);
        }
        if k == BigRational::new(2.into(), 5.into()) {
            return Some(SpecialAngle::TwoPiOver5);
        }
    }

    None
}

/// Special input values for inverse trig functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InverseTrigInput {
    Zero,         // 0
    One,          // 1
    Half,         // 1/2
    Sqrt2Over2,   // √2/2
    Sqrt3Over2,   // √3/2
    OneOverSqrt3, // 1/√3 = √3/3
    Sqrt3,        // √3
}

impl InverseTrigInput {
    pub fn display(&self) -> &'static str {
        match self {
            Self::Zero => "0",
            Self::One => "1",
            Self::Half => "1/2",
            Self::Sqrt2Over2 => "√2/2",
            Self::Sqrt3Over2 => "√3/2",
            Self::OneOverSqrt3 => "√3/3",
            Self::Sqrt3 => "√3",
        }
    }
}

/// Detect if an expression is a special input for inverse trig.
pub fn detect_inverse_trig_input(ctx: &Context, expr: ExprId) -> Option<InverseTrigInput> {
    match ctx.get(expr) {
        Expr::Number(n) => {
            if n.is_zero() {
                return Some(InverseTrigInput::Zero);
            }
            if n.is_one() {
                return Some(InverseTrigInput::One);
            }
            if *n == BigRational::new(1.into(), 2.into()) {
                return Some(InverseTrigInput::Half);
            }
            None
        }
        Expr::Div(num, den) => {
            if let Expr::Number(d) = ctx.get(*den) {
                let denom: i64 = d.to_integer().try_into().ok()?;
                if denom == 2 {
                    if let Some(2) = extract_numeric_sqrt_radicand(ctx, *num) {
                        return Some(InverseTrigInput::Sqrt2Over2);
                    }
                }
                if denom == 2 {
                    if let Some(3) = extract_numeric_sqrt_radicand(ctx, *num) {
                        return Some(InverseTrigInput::Sqrt3Over2);
                    }
                }
                if denom == 3 {
                    if let Some(3) = extract_numeric_sqrt_radicand(ctx, *num) {
                        return Some(InverseTrigInput::OneOverSqrt3);
                    }
                }
            }
            if let Expr::Number(n) = ctx.get(*num) {
                if n.is_one() {
                    if let Some(3) = extract_numeric_sqrt_radicand(ctx, *den) {
                        return Some(InverseTrigInput::OneOverSqrt3);
                    }
                }
            }
            None
        }
        Expr::Mul(l, r) => {
            let (num_id, sqrt_id) = if matches!(ctx.get(*l), Expr::Number(_) | Expr::Div(_, _)) {
                (*l, *r)
            } else {
                (*r, *l)
            };

            if let Some(sqrt_base) = extract_numeric_sqrt_radicand(ctx, sqrt_id) {
                if let Expr::Number(n) = ctx.get(num_id) {
                    if sqrt_base == 3 && *n == BigRational::new(1.into(), 3.into()) {
                        return Some(InverseTrigInput::OneOverSqrt3);
                    }
                    if sqrt_base == 2 && *n == BigRational::new(1.into(), 2.into()) {
                        return Some(InverseTrigInput::Sqrt2Over2);
                    }
                    if sqrt_base == 3 && *n == BigRational::new(1.into(), 2.into()) {
                        return Some(InverseTrigInput::Sqrt3Over2);
                    }
                }
            }
            None
        }
        Expr::Pow(_, _) => {
            if let Some(3) = extract_numeric_sqrt_radicand(ctx, expr) {
                return Some(InverseTrigInput::Sqrt3);
            }
            None
        }
        _ => None,
    }
}

/// Whether `tan(arg)` should be preserved (not expanded to sin/cos) due to
/// anti-worsen policy:
/// - multiple-angle argument `n*x` with `|n| > 1`
/// - special-angle argument where table evaluation is preferred
pub fn should_block_tan_to_sin_cos_for_arg(ctx: &Context, arg: ExprId) -> bool {
    is_multiple_angle(ctx, arg) || detect_special_angle(ctx, arg).is_some()
}

/// Whether `tan(expr)` should be preserved due to pre-scanned structural marks.
pub fn should_block_tan_to_sin_cos_for_marks(marks: &PatternMarks, expr: ExprId) -> bool {
    marks.is_pythagorean_protected(expr)
        || marks.is_inverse_trig_protected(expr)
        || marks.is_tan_triple_product_protected(expr)
        || marks.is_tan_double_angle_protected(expr)
        || marks.is_identity_cancellation_protected(expr)
        || marks.has_tan_identity_pattern
}

/// Unified anti-expansion gate for `tan(expr) -> sin(expr)/cos(expr)`.
pub fn should_block_tan_to_sin_cos_expr(
    ctx: &Context,
    expr: ExprId,
    marks: Option<&PatternMarks>,
    immediate_parent: Option<ExprId>,
    ancestors: &[ExprId],
) -> bool {
    if marks.is_some_and(|m| should_block_tan_to_sin_cos_for_marks(m, expr)) {
        return true;
    }
    if immediate_parent.is_some_and(|parent_id| is_inverse_trig_function_call(ctx, parent_id)) {
        return true;
    }
    if is_part_of_tan_triple_product_with_ancestors(ctx, expr, ancestors) {
        return true;
    }
    if let Some(tan_arg) = as_fn1(ctx, expr, "tan") {
        if crate::trig_canonicalization_support::is_inverse_trig_function_call(ctx, tan_arg) {
            return true;
        }
        return should_block_tan_to_sin_cos_for_arg(ctx, tan_arg);
    }
    false
}

/// Plan `tan(x) -> sin(x)/cos(x)` only when unified anti-expansion policy allows it.
pub fn try_plan_tan_to_sin_cos_with_policy(
    ctx: &mut Context,
    expr: ExprId,
    marks: Option<&PatternMarks>,
    immediate_parent: Option<ExprId>,
    ancestors: &[ExprId],
) -> Option<TrigCanonicalRewritePlan> {
    if should_block_tan_to_sin_cos_expr(ctx, expr, marks, immediate_parent, ancestors) {
        return None;
    }
    try_rewrite_tan_to_sin_cos_function_expr(ctx, expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Constant;

    fn sqrt_expr(ctx: &mut Context, n: i64) -> ExprId {
        let half = ctx.rational(1, 2);
        let n = ctx.num(n);
        ctx.add(Expr::Pow(n, half))
    }

    #[test]
    fn detect_special_angle_simple() {
        let mut ctx = Context::new();
        let pi = ctx.add(Expr::Constant(Constant::Pi));
        let six = ctx.num(6);
        let pi_over_six = ctx.add(Expr::Div(pi, six));
        assert_eq!(
            detect_special_angle(&ctx, pi_over_six),
            Some(SpecialAngle::PiOver6)
        );
    }

    #[test]
    fn detect_special_angle_fractional_multiple() {
        let mut ctx = Context::new();
        let three = ctx.num(3);
        let pi = ctx.add(Expr::Constant(Constant::Pi));
        let num = ctx.add(Expr::Mul(three, pi));
        let eight = ctx.num(8);
        let expr = ctx.add(Expr::Div(num, eight));
        assert_eq!(
            detect_special_angle(&ctx, expr),
            Some(SpecialAngle::ThreePiOver8)
        );
    }

    #[test]
    fn detect_inverse_trig_input_forms() {
        let mut ctx = Context::new();
        let sqrt3 = sqrt_expr(&mut ctx, 3);
        let one = ctx.num(1);
        let one_over_sqrt3 = ctx.add(Expr::Div(one, sqrt3));
        assert_eq!(
            detect_inverse_trig_input(&ctx, one_over_sqrt3),
            Some(InverseTrigInput::OneOverSqrt3)
        );

        let sqrt2 = sqrt_expr(&mut ctx, 2);
        let two = ctx.num(2);
        let rationalized = ctx.add(Expr::Div(sqrt2, two));
        assert_eq!(
            detect_inverse_trig_input(&ctx, rationalized),
            Some(InverseTrigInput::Sqrt2Over2)
        );
    }

    #[test]
    fn tan_to_sin_cos_block_policy_matches_expected_args() {
        let mut ctx = Context::new();
        let blocked1 = cas_parser::parse("3*x", &mut ctx).expect("3*x");
        let blocked2 = cas_parser::parse("pi/6", &mut ctx).expect("pi/6");
        let allowed = cas_parser::parse("x+y", &mut ctx).expect("x+y");

        assert!(should_block_tan_to_sin_cos_for_arg(&ctx, blocked1));
        assert!(should_block_tan_to_sin_cos_for_arg(&ctx, blocked2));
        assert!(!should_block_tan_to_sin_cos_for_arg(&ctx, allowed));
    }

    #[test]
    fn tan_to_sin_cos_block_policy_matches_marks() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let mut marks = PatternMarks::new();
        assert!(!should_block_tan_to_sin_cos_for_marks(&marks, x));

        marks.mark_pythagorean(x);
        assert!(should_block_tan_to_sin_cos_for_marks(&marks, x));

        let y = ctx.var("y");
        let mut marks = PatternMarks::new();
        marks.mark_inverse_trig(y);
        assert!(should_block_tan_to_sin_cos_for_marks(&marks, y));

        let z = ctx.var("z");
        let mut marks = PatternMarks::new();
        marks.mark_tan_triple_product(z);
        assert!(should_block_tan_to_sin_cos_for_marks(&marks, z));

        let w = ctx.var("w");
        let mut marks = PatternMarks::new();
        marks.mark_tan_double_angle(w);
        assert!(should_block_tan_to_sin_cos_for_marks(&marks, w));

        let q = ctx.var("q");
        let mut marks = PatternMarks::new();
        marks.mark_identity_cancellation(q);
        assert!(should_block_tan_to_sin_cos_for_marks(&marks, q));

        let k = ctx.var("k");
        let mut marks = PatternMarks::new();
        marks.has_tan_identity_pattern = true;
        assert!(should_block_tan_to_sin_cos_for_marks(&marks, k));
    }

    #[test]
    fn tan_to_sin_cos_unified_block_policy_covers_fallbacks() {
        let mut ctx = Context::new();
        let tan_3x = cas_parser::parse("tan(3*x)", &mut ctx).expect("tan(3*x)");
        assert!(should_block_tan_to_sin_cos_expr(
            &ctx,
            tan_3x,
            None,
            None,
            &[]
        ));

        let tan_x = cas_parser::parse("tan(x)", &mut ctx).expect("tan(x)");
        let parent = cas_parser::parse("arctan(tan(x))", &mut ctx).expect("arctan(tan(x))");
        assert!(should_block_tan_to_sin_cos_expr(
            &ctx,
            tan_x,
            None,
            Some(parent),
            &[]
        ));

        let tan_arctan_x = cas_parser::parse("tan(arctan(x))", &mut ctx).expect("tan(arctan(x))");
        assert!(should_block_tan_to_sin_cos_expr(
            &ctx,
            tan_arctan_x,
            None,
            None,
            &[]
        ));
    }

    #[test]
    fn tan_to_sin_cos_policy_plan_applies_for_regular_argument() {
        let mut ctx = Context::new();
        let tan_x = cas_parser::parse("tan(x)", &mut ctx).expect("tan(x)");
        let plan = try_plan_tan_to_sin_cos_with_policy(&mut ctx, tan_x, None, None, &[])
            .expect("rewrite plan");
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: plan.rewritten
            }
        );
        assert_eq!(rendered, "sin(x) / cos(x)");
    }
}
