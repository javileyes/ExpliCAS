//! Tan half-angle and trigonometric quotient rules.
//!
//! This module contains rules for:
//! - Hyperbolic half-angle identities: cosh²(x/2) = (cosh(x)+1)/2
//! - Generalized sin·cos contraction: k·sin(t)·cos(t) = (k/2)·sin(2t)
//! - Trig quotient simplification: sin/cos → tan
//! - Tan double angle contraction

use crate::define_rule;
use crate::parent_context::ParentContext;
use crate::rule::Rewrite;
use crate::rules::trig_canonicalization::format_trig_canonical_rewrite_desc;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::trig_canonicalization_support::try_rewrite_trig_quotient_div_expr;
use cas_math::trig_contraction_support::{
    try_rewrite_generalized_sin_cos_contraction_expr, try_rewrite_tan_double_angle_contraction_expr,
};
use cas_math::trig_half_angle_support::{
    is_half_angle, try_rewrite_hyperbolic_half_angle_squares_expr,
    try_rewrite_trig_half_angle_squares_expr, HalfAngleSquareRewriteKind,
};
use num_rational::BigRational;
use std::cmp::Ordering;

// =============================================================================

fn format_half_angle_square_desc(kind: HalfAngleSquareRewriteKind) -> &'static str {
    match kind {
        HalfAngleSquareRewriteKind::HyperbolicCosh => "cosh²(x/2) = (cosh(x)+1)/2",
        HalfAngleSquareRewriteKind::HyperbolicSinh => "sinh²(x/2) = (cosh(x)-1)/2",
        HalfAngleSquareRewriteKind::TrigSin => "sin²(x/2) = (1 - cos(x))/2",
        HalfAngleSquareRewriteKind::TrigCos => "cos²(x/2) = (1 + cos(x))/2",
    }
}

fn format_generalized_sin_cos_contraction_desc() -> &'static str {
    "k·sin(t)·cos(t) = (k/2)·sin(2t)"
}

fn format_tan_double_angle_contraction_desc() -> &'static str {
    "2·tan(t)/(1-tan²(t)) = tan(2t)"
}

fn is_two(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == BigRational::from_integer(2.into()))
}

fn extract_half_angle_trig_call(ctx: &Context, expr: ExprId) -> Option<(ExprId, bool)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let is_sin = matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sin));
    let is_cos = matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos));
    if !(is_sin || is_cos) || is_half_angle(ctx, args[0]).is_none() {
        return None;
    }

    Some((args[0], is_sin))
}

fn extract_half_angle_trig_square(ctx: &Context, expr: ExprId) -> Option<(ExprId, bool)> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) if is_two(ctx, *exp) => extract_half_angle_trig_call(ctx, *base),
        Expr::Mul(left, right) => {
            let (left_arg, left_is_sin) = extract_half_angle_trig_call(ctx, *left)?;
            let (right_arg, right_is_sin) = extract_half_angle_trig_call(ctx, *right)?;
            if left_is_sin == right_is_sin
                && compare_expr(ctx, left_arg, right_arg) == Ordering::Equal
            {
                Some((left_arg, left_is_sin))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn expr_contains_structural(ctx: &Context, haystack: ExprId, needle: ExprId) -> bool {
    if haystack == needle || compare_expr(ctx, haystack, needle) == Ordering::Equal {
        return true;
    }

    match ctx.get(haystack) {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right) => {
            expr_contains_structural(ctx, *left, needle)
                || expr_contains_structural(ctx, *right, needle)
        }
        Expr::Pow(base, exp) => {
            expr_contains_structural(ctx, *base, needle)
                || expr_contains_structural(ctx, *exp, needle)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains_structural(ctx, *inner, needle),
        Expr::Function(_, args) => args
            .iter()
            .any(|arg| expr_contains_structural(ctx, *arg, needle)),
        _ => false,
    }
}

fn expr_contains_matching_trig_call(
    ctx: &Context,
    haystack: ExprId,
    expected_fn: BuiltinFn,
    expected_arg: ExprId,
) -> bool {
    match ctx.get(haystack) {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(*fn_id), Some(found) if found == expected_fn)
                && compare_expr(ctx, args[0], expected_arg) == Ordering::Equal =>
        {
            true
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right) => {
            expr_contains_matching_trig_call(ctx, *left, expected_fn, expected_arg)
                || expr_contains_matching_trig_call(ctx, *right, expected_fn, expected_arg)
        }
        Expr::Pow(base, exp) => {
            expr_contains_matching_trig_call(ctx, *base, expected_fn, expected_arg)
                || expr_contains_matching_trig_call(ctx, *exp, expected_fn, expected_arg)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            expr_contains_matching_trig_call(ctx, *inner, expected_fn, expected_arg)
        }
        Expr::Function(_, args) => args
            .iter()
            .any(|arg| expr_contains_matching_trig_call(ctx, *arg, expected_fn, expected_arg)),
        _ => false,
    }
}

fn should_keep_reciprocal_trig_derivative_denominator_square(
    ctx: &Context,
    expr: ExprId,
    parent_ctx: &ParentContext,
) -> bool {
    if !parent_ctx.is_inside_division() {
        return false;
    }

    let Some((half_arg, denominator_is_sin)) = extract_half_angle_trig_square(ctx, expr) else {
        return false;
    };
    let expected_numerator_fn = if denominator_is_sin {
        BuiltinFn::Cos
    } else {
        BuiltinFn::Sin
    };

    parent_ctx.all_ancestors().iter().any(|&ancestor| {
        let Expr::Div(numerator, denominator) = ctx.get(ancestor) else {
            return false;
        };
        expr_contains_structural(ctx, *denominator, expr)
            && expr_contains_matching_trig_call(ctx, *numerator, expected_numerator_fn, half_arg)
    })
}

define_rule!(
    HyperbolicHalfAngleSquaresRule,
    "Hyperbolic Half-Angle Squares",
    |ctx, expr| {
        let rewrite = try_rewrite_hyperbolic_half_angle_squares_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_half_angle_square_desc(rewrite.kind)))
    }
);

// =============================================================================
// TrigHalfAngleSquaresRule: sin(x/2)² → (1 - cos(x))/2, cos(x/2)² → (1 + cos(x))/2
// =============================================================================
// Trig analogue of HyperbolicHalfAngleSquaresRule. Only applies to squared forms
// (no sqrt branching). TRANSFORM-only to prevent oscillation with
// Cos2xAdditiveContractionRule (POST-only, contracts 1-2sin²→cos(2t)).
//
// Also matches Mul(sin(x/2), sin(x/2)) and Mul(cos(x/2), cos(x/2)) for cases
// where the AST uses product form instead of Pow.

pub struct TrigHalfAngleSquaresRule;

impl crate::rule::Rule for TrigHalfAngleSquaresRule {
    fn name(&self) -> &str {
        "Trig Half-Angle Squares"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        if should_keep_reciprocal_trig_derivative_denominator_square(ctx, expr, parent_ctx) {
            return None;
        }

        let rewrite = try_rewrite_trig_half_angle_squares_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_half_angle_square_desc(rewrite.kind)))
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // TRANSFORM only: prevents oscillation with Cos2xAdditiveContractionRule (POST-only)
        crate::phase::PhaseMask::TRANSFORM
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW.union(crate::target_kind::TargetKindSet::MUL))
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

// =============================================================================
// GeneralizedSinCosContractionRule: k*sin(t)*cos(t) → (k/2)*sin(2t) for even k
// =============================================================================
// Extends DoubleAngleContractionRule to handle k*sin*cos where k is even (4, 6, 8, etc.)

define_rule!(
    GeneralizedSinCosContractionRule,
    "Generalized Sin Cos Contraction",
    |ctx, expr| {
        let rewrite = try_rewrite_generalized_sin_cos_contraction_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_generalized_sin_cos_contraction_desc()))
    }
);

// =============================================================================
// TrigQuotientToNamedRule: sin(t)/cos(t) → tan(t), 1/cos(t) → sec(t), etc.
// =============================================================================
// Canonicalize trig quotients to named functions for better normalization.
// This ensures that `sin(u)/cos(u)` and `tan(u)` converge to the same form.

define_rule!(
    TrigQuotientToNamedRule,
    "Trig Quotient to Named Function",
    |ctx, expr| {
        let rewrite = try_rewrite_trig_quotient_div_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_trig_canonical_rewrite_desc(
                rewrite.kind.expect("fixed quotient canonical desc"),
            )),
        )
    }
);

// =============================================================================
// TanDoubleAngleContractionRule: 2*tan(t)/(1 - tan(t)²) → tan(2*t)
// =============================================================================
// This contracts the expanded tan(2t) form back to the double angle form.
// Prevents the engine from creating deeply nested fractions when tan²(t)
// appears in denominators.

define_rule!(
    TanDoubleAngleContractionRule,
    "Tan Double Angle Contraction",
    |ctx, expr| {
        let rewrite = try_rewrite_tan_double_angle_contraction_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_tan_double_angle_contraction_desc()))
    }
);
