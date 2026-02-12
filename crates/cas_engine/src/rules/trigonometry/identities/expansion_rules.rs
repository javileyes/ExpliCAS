//! Expansion and contraction rules for trigonometric expressions.

use crate::define_rule;
use crate::helpers::extract_double_angle_arg;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{BuiltinFn, Expr, ExprId};

// Import helpers from sibling modules (via re-exports in parent)
use super::{build_avg, build_half_diff, is_multiple_angle, normalize_for_even_fn};

// =============================================================================
// STANDALONE SUM-TO-PRODUCT RULE
// sin(A)+sin(B) → 2·sin((A+B)/2)·cos((A-B)/2)
// sin(A)-sin(B) → 2·cos((A+B)/2)·sin((A-B)/2)
// cos(A)+cos(B) → 2·cos((A+B)/2)·cos((A-B)/2)
// cos(A)-cos(B) → -2·sin((A+B)/2)·sin((A-B)/2)
// =============================================================================
// This rule applies sum-to-product identities to standalone sums/differences
// of trig functions (not inside quotients handled by SinCosSumQuotientRule).
//
// GATING: Only apply when both arguments are rational multiples of π, ensuring
// the transformed expression can be evaluated via trig table lookup (π/4, π/6, etc.)
// This prevents unnecessary expansion of symbolic expressions like sin(a)+sin(b).
//
// MATCHERS: Uses semantic TrigSumMatch (unordered) and TrigDiffMatch (ordered)
// to ensure correct sign handling for difference identities.
define_rule!(
    TrigSumToProductRule,
    "Sum-to-Product Identity",
    |ctx, expr| {
        use crate::helpers::{extract_rational_pi_multiple, match_trig_diff, match_trig_sum};

        // Try all four patterns
        enum Pattern {
            SinSum { arg1: ExprId, arg2: ExprId },
            SinDiff { a: ExprId, b: ExprId }, // ordered!
            CosSum { arg1: ExprId, arg2: ExprId },
            CosDiff { a: ExprId, b: ExprId }, // ordered!
        }

        let pattern = if let Some(m) = match_trig_sum(ctx, expr, "sin") {
            Pattern::SinSum {
                arg1: m.arg1,
                arg2: m.arg2,
            }
        } else if let Some(m) = match_trig_diff(ctx, expr, "sin") {
            Pattern::SinDiff { a: m.a, b: m.b }
        } else if let Some(m) = match_trig_sum(ctx, expr, "cos") {
            Pattern::CosSum {
                arg1: m.arg1,
                arg2: m.arg2,
            }
        } else if let Some(m) = match_trig_diff(ctx, expr, "cos") {
            Pattern::CosDiff { a: m.a, b: m.b }
        } else {
            return None;
        };

        // Extract (A, B) and the function name
        let (arg_a, arg_b, is_diff, fn_name) = match pattern {
            Pattern::SinSum { arg1, arg2 } => (arg1, arg2, false, "sin"),
            Pattern::SinDiff { a, b } => (a, b, true, "sin"),
            Pattern::CosSum { arg1, arg2 } => (arg1, arg2, false, "cos"),
            Pattern::CosDiff { a, b } => (a, b, true, "cos"),
        };

        // GATING: Only apply when BOTH arguments are rational multiples of π
        // This ensures the result can be simplified via trig table lookup
        let pi_a = extract_rational_pi_multiple(ctx, arg_a);
        let pi_b = extract_rational_pi_multiple(ctx, arg_b);
        if pi_a.is_none() || pi_b.is_none() {
            return None; // Don't expand symbolic sums
        }

        // Build avg = (A+B)/2 and half_diff = (A-B)/2
        let avg = build_avg(ctx, arg_a, arg_b);
        let half_diff = build_half_diff(ctx, arg_a, arg_b);
        let two = ctx.num(2);

        let (result, desc) = match (fn_name, is_diff) {
            // sin(A) + sin(B) → 2·sin(avg)·cos(half_diff)
            ("sin", false) => {
                let sin_avg = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![avg]);
                let cos_half = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_diff]);
                let product = smart_mul(ctx, sin_avg, cos_half);
                let result = smart_mul(ctx, two, product);
                (result, "sin(A)+sin(B) = 2·sin((A+B)/2)·cos((A-B)/2)")
            }
            // sin(A) - sin(B) → 2·cos(avg)·sin(half_diff)
            // Note: half_diff preserves order (A-B)/2 for correct sign
            ("sin", true) => {
                let cos_avg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![avg]);
                let sin_half = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![half_diff]);
                let product = smart_mul(ctx, cos_avg, sin_half);
                let result = smart_mul(ctx, two, product);
                (result, "sin(A)-sin(B) = 2·cos((A+B)/2)·sin((A-B)/2)")
            }
            // cos(A) + cos(B) → 2·cos(avg)·cos(half_diff)
            ("cos", false) => {
                // For cos, half_diff sign doesn't matter (even function)
                let half_diff_normalized = normalize_for_even_fn(ctx, half_diff);
                let cos_avg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![avg]);
                let cos_half =
                    ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_diff_normalized]);
                let product = smart_mul(ctx, cos_avg, cos_half);
                let result = smart_mul(ctx, two, product);
                (result, "cos(A)+cos(B) = 2·cos((A+B)/2)·cos((A-B)/2)")
            }
            // cos(A) - cos(B) → -2·sin(avg)·sin(half_diff)
            ("cos", true) => {
                let sin_avg = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![avg]);
                let sin_half = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![half_diff]);
                let product = smart_mul(ctx, sin_avg, sin_half);
                let two_product = smart_mul(ctx, two, product);
                let result = ctx.add(Expr::Neg(two_product));
                (result, "cos(A)-cos(B) = -2·sin((A+B)/2)·sin((A-B)/2)")
            }
            _ => return None,
        };

        Some(Rewrite::new(result).desc(desc))
    }
);

define_rule!(
    DoubleAngleRule,
    "Double Angle Identity",
    |ctx, expr, parent_ctx| {
        // GUARD: Only expand double angles in expand mode.
        // In default simplification, we prefer the contracted form (cos(2t), sin(2t))
        // to avoid oscillation with DoubleAngleContractionRule/Cos2xAdditiveContractionRule.
        // The contracted form is the canonical NF for double-angle trig expressions.
        if !parent_ctx.is_expand_mode() {
            return None;
        }

        // GUARD: Don't expand double angle inside a Div context
        // This prevents sin(2x)/cos(2x) from being "polinomized" to a worse form.
        // Expansion should only happen when it helps simplification, not in canonical quotients.
        if parent_ctx
            .has_ancestor_matching(ctx, |c, id| matches!(c.get(id), cas_ast::Expr::Div(_, _)))
        {
            return None;
        }

        // GUARD: Don't expand when sin(4x) identity pattern is detected
        // This allows Sin4xIdentityZeroRule to see 4*sin*cos*cos(2t) intact
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.has_sin4x_identity_pattern {
                return None;
            }
        }

        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if args.len() == 1 {
                // Check if arg is 2*x or x*2
                // We need to match "2 * x"
                if let Some(inner_var) = extract_double_angle_arg(ctx, args[0]) {
                    // GUARD: Anti-worsen for multiple angles.
                    // Don't expand sin(2*(8x)) = sin(16x) because the inner argument
                    // is already a multiple (8x). This would cause exponential recursion:
                    // sin(16x) → 2sin(8x)cos(8x) → 2·2sin(4x)cos(4x)·... = explosion
                    if is_multiple_angle(ctx, inner_var) {
                        return None;
                    }

                    match ctx.builtin_of(*fn_id) {
                        Some(BuiltinFn::Sin) => {
                            // sin(2x) -> 2sin(x)cos(x)
                            let two = ctx.num(2);
                            let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![inner_var]);
                            let cos_x = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![inner_var]);
                            let sin_cos = smart_mul(ctx, sin_x, cos_x);
                            let new_expr = smart_mul(ctx, two, sin_cos);
                            return Some(Rewrite::new(new_expr).desc("sin(2x) -> 2sin(x)cos(x)"));
                        }
                        Some(BuiltinFn::Cos) => {
                            // cos(2x) -> cos^2(x) - sin^2(x)
                            let two = ctx.num(2);
                            let cos_x = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![inner_var]);
                            let cos2 = ctx.add(Expr::Pow(cos_x, two));

                            let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![inner_var]);
                            let sin2 = ctx.add(Expr::Pow(sin_x, two));

                            let new_expr = ctx.add(Expr::Sub(cos2, sin2));
                            return Some(
                                Rewrite::new(new_expr).desc("cos(2x) -> cos^2(x) - sin^2(x)"),
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use crate::rules::trigonometry::identities::{
        AngleIdentityRule, EvaluateTrigRule, TanToSinCosRule,
    };
    use cas_ast::{Context, DisplayExpr};
    use cas_parser::parse;

    #[test]
    fn test_evaluate_trig_zero() {
        let mut ctx = Context::new();
        let rule = EvaluateTrigRule;

        // sin(0) -> 0
        let expr = parse("sin(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );

        // cos(0) -> 1
        let expr = parse("cos(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "1"
        );

        // tan(0) -> 0
        let expr = parse("tan(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
    }

    #[test]
    fn test_evaluate_trig_identities() {
        let mut ctx = Context::new();
        let rule = EvaluateTrigRule;

        // sin(-x) -> -sin(x)
        let expr = parse("sin(-x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "-sin(x)"
        );

        // cos(-x) -> cos(x)
        let expr = parse("cos(-x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "cos(x)"
        );

        // tan(-x) -> -tan(x)
        let expr = parse("tan(-x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "-tan(x)"
        );
    }

    #[test]
    fn test_trig_identities() {
        let mut ctx = Context::new();
        let rule = AngleIdentityRule;

        // sin(x + y)
        let expr = parse("sin(x + y)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("sin(x)"));

        // cos(x + y) -> cos(x)cos(y) - sin(x)sin(y)
        let expr = parse("cos(x + y)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        let res = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(res.contains("cos(x)"));
        assert!(res.contains("-"));

        // sin(x - y)
        let expr = parse("sin(x - y)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("-"));
    }

    #[test]
    fn test_tan_to_sin_cos() {
        let mut ctx = Context::new();
        let rule = TanToSinCosRule;
        let expr = parse("tan(x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "sin(x) / cos(x)"
        );
    }

    #[test]
    fn test_double_angle() {
        let mut ctx = Context::new();
        let rule = DoubleAngleRule;
        // DoubleAngleRule is now gated behind expand_mode to prevent oscillation
        // with DoubleAngleContractionRule during default simplification.
        let expand_ctx = crate::parent_context::ParentContext::root().with_expand_mode_flag(true);

        // sin(2x)
        let expr = parse("sin(2 * x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr, &expand_ctx).unwrap();
        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        // Check that result contains the key components, regardless of order
        assert!(
            result_str.contains("sin(x)"),
            "Result should contain sin(x), got: {}",
            result_str
        );
        assert!(
            result_str.contains("cos(x)"),
            "Result should contain cos(x), got: {}",
            result_str
        );
        assert!(
            result_str.contains("2") || result_str.contains("* 2") || result_str.contains("2 *"),
            "Result should contain 2, got: {}",
            result_str
        );

        // cos(2x)
        let expr = parse("cos(2 * x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr, &expand_ctx).unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("cos(x)^2 - sin(x)^2"));
    }

    #[test]
    fn test_evaluate_inverse_trig() {
        let mut ctx = Context::new();
        let rule = EvaluateTrigRule;

        // arcsin(0) -> 0
        let expr = parse("arcsin(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );

        // arccos(1) -> 0
        let expr = parse("arccos(1)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );

        // arcsin(1) -> pi/2
        // Note: pi/2 might be formatted as "pi / 2" or similar depending on Display impl
        let expr = parse("arcsin(1)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("pi"));
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("2"));

        // arccos(0) -> pi/2
        let expr = parse("arccos(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("pi"));
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("2"));
    }
}
