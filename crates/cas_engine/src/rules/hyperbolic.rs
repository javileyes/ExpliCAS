use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::hyperbolic_core_support::{
    try_eval_hyperbolic_special_value, try_rewrite_hyperbolic_composition,
};
use cas_math::hyperbolic_identity_support::{
    try_rewrite_hyperbolic_double_angle_sub_chain, try_rewrite_hyperbolic_double_angle_sum,
    try_rewrite_hyperbolic_pythagorean_sub_expr, try_rewrite_hyperbolic_triple_angle,
    try_rewrite_sinh_cosh_to_exp, try_rewrite_sinh_cosh_to_tanh_identity_expr,
    try_rewrite_sinh_double_angle_expansion_identity_expr,
    try_rewrite_tanh_to_sinh_cosh_identity_expr,
};
use cas_math::hyperbolic_negative_support::try_rewrite_hyperbolic_negative_expr;

fn format_hyperbolic_negative_desc(
    kind: cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind,
) -> &'static str {
    match kind {
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::SinhExplicitNeg => {
            "sinh(-x) = -sinh(x)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::CoshExplicitNeg => {
            "cosh(-x) = cosh(x)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::TanhExplicitNeg => {
            "tanh(-x) = -tanh(x)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::AsinhExplicitNeg => {
            "asinh(-x) = -asinh(x)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::AtanhExplicitNeg => {
            "atanh(-x) = -atanh(x)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::SinhCanonicalSub => {
            "sinh(a−b) = −sinh(b−a)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::CoshCanonicalSub => {
            "cosh(a−b) = cosh(b−a)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::TanhCanonicalSub => {
            "tanh(a−b) = −tanh(b−a)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::AsinhCanonicalSub => {
            "asinh(a−b) = −asinh(b−a)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::AtanhCanonicalSub => {
            "atanh(a−b) = −atanh(b−a)"
        }
    }
}

fn format_hyperbolic_special_value_desc(ctx: &Context, expr: ExprId) -> Option<&'static str> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Sinh) => Some("sinh(0) = 0"),
        Some(BuiltinFn::Tanh) => Some("tanh(0) = 0"),
        Some(BuiltinFn::Cosh) => Some("cosh(0) = 1"),
        Some(BuiltinFn::Asinh) => Some("asinh(0) = 0"),
        Some(BuiltinFn::Atanh) => Some("atanh(0) = 0"),
        Some(BuiltinFn::Acosh) => Some("acosh(1) = 0"),
        _ => None,
    }
}

fn format_hyperbolic_composition_desc(ctx: &Context, expr: ExprId) -> Option<&'static str> {
    let Expr::Function(outer_fn, outer_args) = ctx.get(expr) else {
        return None;
    };
    if outer_args.len() != 1 {
        return None;
    }
    let Expr::Function(inner_fn, inner_args) = ctx.get(outer_args[0]) else {
        return None;
    };
    if inner_args.len() != 1 {
        return None;
    }
    match (ctx.builtin_of(*outer_fn), ctx.builtin_of(*inner_fn)) {
        (Some(BuiltinFn::Sinh), Some(BuiltinFn::Asinh)) => Some("sinh(asinh(x)) = x"),
        (Some(BuiltinFn::Cosh), Some(BuiltinFn::Acosh)) => Some("cosh(acosh(x)) = x"),
        (Some(BuiltinFn::Tanh), Some(BuiltinFn::Atanh)) => Some("tanh(atanh(x)) = x"),
        (Some(BuiltinFn::Asinh), Some(BuiltinFn::Sinh)) => Some("asinh(sinh(x)) = x"),
        (Some(BuiltinFn::Acosh), Some(BuiltinFn::Cosh)) => Some("acosh(cosh(x)) = x"),
        (Some(BuiltinFn::Atanh), Some(BuiltinFn::Tanh)) => Some("atanh(tanh(x)) = x"),
        _ => None,
    }
}

// ==================== Hyperbolic Function Rules ====================

// Rule 1: Evaluate hyperbolic functions at special values
define_rule!(
    EvaluateHyperbolicRule,
    "Evaluate Hyperbolic Functions",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_eval_hyperbolic_special_value(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_hyperbolic_special_value_desc(ctx, expr)?))
    }
);

// Rule 2: Composition identities - sinh(asinh(x)) = x, etc.
// HIGH PRIORITY: Must run BEFORE TanhToSinhCoshRule - ensured by registration order
define_rule!(
    HyperbolicCompositionRule,
    "Hyperbolic Composition",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Analytic
    ),
    |ctx, expr, _parent_ctx| {
        let rewrite = try_rewrite_hyperbolic_composition(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_hyperbolic_composition_desc(ctx, expr)?))
    }
);

// Rule 3: Negative argument identities
// Handles both explicit Neg(x) and Sub(a,b) where a < b canonically.
// V2.16: Extended to catch Sub patterns like sinh(1-u²) → -sinh(u²-1).
define_rule!(
    HyperbolicNegativeRule,
    "Hyperbolic Negative Argument",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_hyperbolic_negative_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_hyperbolic_negative_desc(rewrite.kind)))
    }
);

// Rule 4: Hyperbolic Pythagorean identity: cosh²(x) - sinh²(x) = 1
define_rule!(
    HyperbolicPythagoreanRule,
    "Hyperbolic Pythagorean Identity",
    Some(crate::target_kind::TargetKindSet::SUB),
    |ctx, expr| {
        let rewrite = try_rewrite_hyperbolic_pythagorean_sub_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Rule 4b: sinh(x) + cosh(x) = exp(x), cosh(x) - sinh(x) = exp(-x)
// Inverse of RecognizeHyperbolicFromExpRule — collapses hyperbolic sums/diffs to exp.
define_rule!(
    SinhCoshToExpRule,
    "Hyperbolic Sum to Exponential",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    |ctx, expr| {
        let rewrite = try_rewrite_sinh_cosh_to_exp(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Rule 5: Hyperbolic double angle identity: cosh²(x) + sinh²(x) = cosh(2x)
// This direction collapses two squared terms into a single term, reducing complexity.
// The inverse (expansion) is not implemented to avoid loops.
define_rule!(
    HyperbolicDoubleAngleRule,
    "Hyperbolic Double Angle",
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| {
        let rewrite = try_rewrite_hyperbolic_double_angle_sum(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Rule: tanh(x) → sinh(x) / cosh(x)
// This is the hyperbolic analogue of tan(x) → sin(x) / cos(x)
// GUARD: Skip if argument is inverse hyperbolic (let composition rule handle tanh(atanh(x)) → x)
define_rule!(
    TanhToSinhCoshRule,
    "tanh(x) = sinh(x)/cosh(x)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_tanh_to_sinh_cosh_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Rule: sinh(2x) → 2·sinh(x)·cosh(x)
// Expansion of double angle for sinh
define_rule!(
    SinhDoubleAngleExpansionRule,
    "sinh(2x) = 2·sinh(x)·cosh(x)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_sinh_double_angle_expansion_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Rule: cosh(2x) - cosh²(x) - sinh²(x) → 0
// After canonicalization, Sub nodes become Add+Neg, so the actual pattern is:
//   Add-chain containing: cosh(2x), Neg(cosh²(x)), Neg(sinh²(x))
// When found, these three terms cancel to 0 and are removed from the sum.
define_rule!(
    HyperbolicDoubleAngleSubRule,
    "Hyperbolic Double Angle Subtraction",
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| {
        let rewrite = try_rewrite_hyperbolic_double_angle_sub_chain(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Rule: sinh(x) / cosh(x) → tanh(x)
// Contraction rule (inverse of TanhToSinhCoshRule) - safe direction that doesn't break composition tests
define_rule!(
    SinhCoshToTanhRule,
    "sinh(x)/cosh(x) = tanh(x)",
    Some(crate::target_kind::TargetKindSet::DIV),
    |ctx, expr| {
        let rewrite = try_rewrite_sinh_cosh_to_tanh_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// ==================== Recognize Hyperbolic From Exponential ====================

// Rule 5: Recognize hyperbolic functions from exponential definitions
// (e^x + e^(-x))/2 → cosh(x)
// (e^x - e^(-x))/2 → sinh(x)
// (e^(-x) - e^x)/2 → -sinh(x)
define_rule!(
    RecognizeHyperbolicFromExpRule,
    "Recognize Hyperbolic from Exponential",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        let rewrite = cas_math::hyperbolic_identity_support::try_rewrite_recognize_hyperbolic_from_exp(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Rule: Hyperbolic triple angle expansion
// sinh(3x) → 3·sinh(x) + 4·sinh³(x)
// cosh(3x) → 4·cosh³(x) - 3·cosh(x)
// Note: sinh uses + (not -) because sinh is always positive for positive x
define_rule!(
    HyperbolicTripleAngleRule,
    "Hyperbolic Triple Angle Identity",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_hyperbolic_triple_angle(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

/// Register all hyperbolic function rules
pub fn register(simplifier: &mut crate::engine::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateHyperbolicRule));
    simplifier.add_rule(Box::new(HyperbolicCompositionRule));
    simplifier.add_rule(Box::new(HyperbolicNegativeRule));
    simplifier.add_rule(Box::new(HyperbolicPythagoreanRule));
    simplifier.add_rule(Box::new(SinhCoshToExpRule));
    simplifier.add_rule(Box::new(HyperbolicDoubleAngleRule));
    simplifier.add_rule(Box::new(HyperbolicDoubleAngleSubRule));
    // DISABLED: TanhToSinhCoshRule breaks tanh(atanh(x))→x and tanh(-x)→-tanh(x) paths
    // simplifier.add_rule(Box::new(TanhToSinhCoshRule)); // tanh(x) → sinh(x)/cosh(x)
    simplifier.add_rule(Box::new(SinhCoshToTanhRule)); // sinh(x)/cosh(x) → tanh(x) (contraction)
    simplifier.add_rule(Box::new(SinhDoubleAngleExpansionRule)); // sinh(2x) → 2sinh(x)cosh(x)
    simplifier.add_rule(Box::new(RecognizeHyperbolicFromExpRule));
    simplifier.add_rule(Box::new(HyperbolicTripleAngleRule)); // sinh(3x), cosh(3x)
}
