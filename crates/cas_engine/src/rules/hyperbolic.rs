use crate::define_rule;
use crate::rule::Rewrite;
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

#[cfg(test)]
use cas_ast::Context;
#[cfg(test)]
use cas_ast::Expr;

// ==================== Hyperbolic Function Rules ====================

// Rule 1: Evaluate hyperbolic functions at special values
define_rule!(
    EvaluateHyperbolicRule,
    "Evaluate Hyperbolic Functions",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_eval_hyperbolic_special_value(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Rule 2: Composition identities - sinh(asinh(x)) = x, etc.
// HIGH PRIORITY: Must run BEFORE TanhToSinhCoshRule - ensured by registration order
define_rule!(
    HyperbolicCompositionRule,
    "Hyperbolic Composition",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Analytic
    ),
    |ctx, expr, _parent_ctx| {
        let rewrite = try_rewrite_hyperbolic_composition(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
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
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_formatter::DisplayExpr;

    #[test]
    fn test_recognize_cosh_from_exp() {
        // (e^x + e^(-x))/2 -> cosh(x)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let neg_x = ctx.add(Expr::Neg(x));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
        let sum = ctx.add(Expr::Add(exp_x, exp_neg_x));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Div(sum, two));

        let rule = RecognizeHyperbolicFromExpRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_some(), "Should recognize cosh(x)");
        let result = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.unwrap().new_expr
            }
        );
        assert_eq!(result, "cosh(x)");
    }

    #[test]
    fn test_recognize_sinh_from_exp() {
        // (e^x - e^(-x))/2 -> sinh(x)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let neg_x = ctx.add(Expr::Neg(x));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
        let diff = ctx.add(Expr::Sub(exp_x, exp_neg_x));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Div(diff, two));

        let rule = RecognizeHyperbolicFromExpRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_some(), "Should recognize sinh(x)");
        let result = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.unwrap().new_expr
            }
        );
        assert_eq!(result, "sinh(x)");
    }

    #[test]
    fn test_recognize_neg_sinh_from_exp() {
        // (e^(-x) - e^x)/2 -> -sinh(x)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let neg_x = ctx.add(Expr::Neg(x));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
        // Note: order is reversed
        let diff = ctx.add(Expr::Sub(exp_neg_x, exp_x));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Div(diff, two));

        let rule = RecognizeHyperbolicFromExpRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_some(), "Should recognize -sinh(x)");
        let result = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.unwrap().new_expr
            }
        );
        assert!(
            result.contains("sinh") && result.contains("-"),
            "Should be -sinh(x), got: {}",
            result
        );
    }

    #[test]
    fn test_no_match_different_args() {
        // (e^x + e^(-y))/2 should NOT match (different args)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let neg_y = ctx.add(Expr::Neg(y));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_neg_y = ctx.add(Expr::Pow(e2, neg_y));
        let sum = ctx.add(Expr::Add(exp_x, exp_neg_y));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Div(sum, two));

        let rule = RecognizeHyperbolicFromExpRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_none(), "Should NOT match different args");
    }

    #[test]
    fn test_no_match_wrong_divisor() {
        // (e^x + e^(-x))/3 should NOT match (not divided by 2)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let neg_x = ctx.add(Expr::Neg(x));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
        let sum = ctx.add(Expr::Add(exp_x, exp_neg_x));
        let three = ctx.num(3);
        let expr = ctx.add(Expr::Div(sum, three));

        let rule = RecognizeHyperbolicFromExpRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_none(), "Should NOT match divisor != 2");
    }

    #[test]
    fn test_hyperbolic_double_angle_rule() {
        // cosh(x)^2 + sinh(x)^2 -> cosh(2*x)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let cosh_x = ctx.call_builtin(cas_ast::BuiltinFn::Cosh, vec![x]);
        let sinh_x = ctx.call_builtin(cas_ast::BuiltinFn::Sinh, vec![x]);
        let two = ctx.num(2);
        let two2 = ctx.num(2);
        let cosh_sq = ctx.add(Expr::Pow(cosh_x, two));
        let sinh_sq = ctx.add(Expr::Pow(sinh_x, two2));
        let expr = ctx.add(Expr::Add(cosh_sq, sinh_sq));

        let rule = HyperbolicDoubleAngleRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_some(), "Should apply cosh²+sinh² -> cosh(2x)");
        let result = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.unwrap().new_expr
            }
        );
        assert!(
            result.contains("cosh") && result.contains("2"),
            "Should be cosh(2*x), got: {}",
            result
        );
    }
}
