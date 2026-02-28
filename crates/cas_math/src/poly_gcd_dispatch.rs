//! Unified polynomial-GCD dispatcher shared across runtime crates.
//!
//! This module centralizes mode selection (`structural/exact/modp/auto`) and
//! soundness gates based on solve goals.

use crate::gcd_exact::{gcd_exact, GcdExactBudget, GcdExactLayer};
use crate::gcd_zippel_modp::ZippelPreset;
use crate::poly_gcd_mode::{GcdGoal, GcdMode};
use crate::poly_gcd_structural::poly_gcd_structural;
use crate::poly_modp_conv::{compute_gcd_modp_expr_with_options, DEFAULT_PRIME};
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;

/// Classification of runtime pre-evaluation action for one GCD argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcdPreEvalDirective {
    /// Evaluate `expand(expr)` eagerly.
    EvaluateExpand { expand_call: ExprId },
    /// Unwrap internal hold wrapper.
    UnwrapHold { inner: ExprId },
    /// Keep expression as-is.
    Keep,
}

/// Classify whether an expression needs runtime pre-evaluation before GCD paths.
///
/// Rules:
/// - `expand(...)` (builtin) -> [`GcdPreEvalDirective::EvaluateExpand`]
/// - `__hold(x)` -> [`GcdPreEvalDirective::UnwrapHold`]
/// - `factor(...)` / `simplify(...)` -> [`GcdPreEvalDirective::Keep`]
pub fn classify_pre_evaluate_for_gcd(ctx: &Context, expr: ExprId) -> GcdPreEvalDirective {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return GcdPreEvalDirective::Keep;
    };

    if matches!(ctx.builtin_of(*fn_id), Some(cas_ast::BuiltinFn::Expand)) {
        return GcdPreEvalDirective::EvaluateExpand { expand_call: expr };
    }

    if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Hold) && !args.is_empty() {
        return GcdPreEvalDirective::UnwrapHold { inner: args[0] };
    }

    let name = ctx.sym_name(*fn_id);
    if name == "factor" || name == "simplify" {
        return GcdPreEvalDirective::Keep;
    }

    GcdPreEvalDirective::Keep
}

/// Compute polynomial GCD using the selected mode.
///
/// `pre_evaluate` lets runtime crates unwrap/evaluate wrappers before exact/modp paths.
/// `render` is used only for human-readable descriptions.
#[allow(clippy::too_many_arguments)]
pub fn compute_poly_gcd_unified_with<FPreEval, FRender>(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    goal: GcdGoal,
    mode: GcdMode,
    modp_preset: Option<ZippelPreset>,
    modp_main_var: Option<usize>,
    mut pre_evaluate: FPreEval,
    mut render: FRender,
) -> (ExprId, String)
where
    FPreEval: FnMut(&mut Context, ExprId) -> ExprId,
    FRender: FnMut(&Context, ExprId) -> String,
{
    match mode {
        GcdMode::Structural => {
            let gcd = poly_gcd_structural(ctx, a, b);
            let desc = format!("poly_gcd({}, {})", render(ctx, a), render(ctx, b));
            (gcd, desc)
        }
        GcdMode::Exact => {
            let eval_a = pre_evaluate(ctx, a);
            let eval_b = pre_evaluate(ctx, b);
            let budget = GcdExactBudget::default();
            let result = gcd_exact(ctx, eval_a, eval_b, &budget);
            let desc = format!(
                "poly_gcd({}, {}, exact) [{}]",
                render(ctx, a),
                render(ctx, b),
                format!("{:?}", result.layer_used).to_lowercase()
            );
            (result.gcd, desc)
        }
        GcdMode::Modp => {
            if goal == GcdGoal::CancelFraction {
                let one = ctx.num(1);
                return (
                    one,
                    "poly_gcd(..., modp) [blocked for soundness]".to_string(),
                );
            }

            let eval_a = pre_evaluate(ctx, a);
            let eval_b = pre_evaluate(ctx, b);
            let preset = modp_preset.unwrap_or(ZippelPreset::Aggressive);
            match compute_gcd_modp_expr_with_options(
                ctx,
                eval_a,
                eval_b,
                DEFAULT_PRIME,
                modp_main_var,
                Some(preset),
            ) {
                Ok(result) => (
                    result,
                    format!(
                        "poly_gcd({}, {}, modp) [{:?}]",
                        render(ctx, a),
                        render(ctx, b),
                        preset
                    ),
                ),
                Err(_) => {
                    let one = ctx.num(1);
                    (one, "poly_gcd(..., modp) [error]".to_string())
                }
            }
        }
        GcdMode::Auto => {
            let structural_gcd = poly_gcd_structural(ctx, a, b);
            let is_one = matches!(ctx.get(structural_gcd), Expr::Number(n) if n.is_one());
            if !is_one {
                return (
                    structural_gcd,
                    format!(
                        "poly_gcd({}, {}, auto) [structural]",
                        render(ctx, a),
                        render(ctx, b)
                    ),
                );
            }

            let eval_a = pre_evaluate(ctx, a);
            let eval_b = pre_evaluate(ctx, b);
            let budget = GcdExactBudget::default();
            let exact_result = gcd_exact(ctx, eval_a, eval_b, &budget);
            if exact_result.layer_used != GcdExactLayer::BudgetExceeded {
                return (
                    exact_result.gcd,
                    format!(
                        "poly_gcd({}, {}, auto) [exact:{:?}]",
                        render(ctx, a),
                        render(ctx, b),
                        exact_result.layer_used
                    ),
                );
            }

            if goal == GcdGoal::CancelFraction {
                let one = ctx.num(1);
                return (
                    one,
                    "poly_gcd(..., auto) [exact exceeded budget, modp blocked for soundness]"
                        .to_string(),
                );
            }

            let preset = modp_preset.unwrap_or(ZippelPreset::Aggressive);
            match compute_gcd_modp_expr_with_options(
                ctx,
                eval_a,
                eval_b,
                DEFAULT_PRIME,
                modp_main_var,
                Some(preset),
            ) {
                Ok(result) => (
                    result,
                    format!(
                        "poly_gcd({}, {}, auto) [modp:{:?} - probabilistic]",
                        render(ctx, a),
                        render(ctx, b),
                        preset
                    ),
                ),
                Err(_) => {
                    let one = ctx.num(1);
                    (one, "poly_gcd(..., auto) [modp error]".to_string())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        classify_pre_evaluate_for_gcd, compute_poly_gcd_unified_with, GcdPreEvalDirective,
    };
    use crate::gcd_zippel_modp::ZippelPreset;
    use crate::poly_gcd_mode::{GcdGoal, GcdMode};
    use cas_ast::{BuiltinFn, Expr};
    use cas_parser::parse;

    #[test]
    fn structural_mode_finds_common_factor() {
        let mut ctx = cas_ast::Context::new();
        let a = parse("(x+1)*(y+2)", &mut ctx).expect("parse a");
        let b = parse("(x+1)*(z+3)", &mut ctx).expect("parse b");
        let (gcd, desc) = compute_poly_gcd_unified_with(
            &mut ctx,
            a,
            b,
            GcdGoal::UserPolyGcd,
            GcdMode::Structural,
            None,
            None,
            |_ctx, id| id,
            |_ctx, id| format!("{id:?}"),
        );
        assert!(desc.contains("poly_gcd("));
        assert!(!matches!(ctx.get(gcd), Expr::Number(_)));
    }

    #[test]
    fn modp_mode_is_blocked_for_cancel_fraction_goal() {
        let mut ctx = cas_ast::Context::new();
        let a = parse("x^2+1", &mut ctx).expect("parse a");
        let b = parse("x+1", &mut ctx).expect("parse b");
        let (gcd, desc) = compute_poly_gcd_unified_with(
            &mut ctx,
            a,
            b,
            GcdGoal::CancelFraction,
            GcdMode::Modp,
            Some(ZippelPreset::Aggressive),
            None,
            |_ctx, id| id,
            |_ctx, id| format!("{id:?}"),
        );
        assert!(matches!(ctx.get(gcd), Expr::Number(_)));
        assert!(desc.contains("blocked for soundness"));
    }

    #[test]
    fn classify_pre_eval_marks_expand_and_hold() {
        let mut ctx = cas_ast::Context::new();
        let inner = parse("x+1", &mut ctx).expect("parse");
        let expand_call = ctx.call_builtin(BuiltinFn::Expand, vec![inner]);
        let hold = ctx.call_builtin(BuiltinFn::Hold, vec![inner]);

        assert!(matches!(
            classify_pre_evaluate_for_gcd(&ctx, expand_call),
            GcdPreEvalDirective::EvaluateExpand { .. }
        ));
        assert_eq!(
            classify_pre_evaluate_for_gcd(&ctx, hold),
            GcdPreEvalDirective::UnwrapHold { inner }
        );
    }

    #[test]
    fn classify_pre_eval_keeps_factor_and_plain() {
        let mut ctx = cas_ast::Context::new();
        let x = parse("x", &mut ctx).expect("parse");
        let factor = ctx.call("factor", vec![x]);
        assert_eq!(
            classify_pre_evaluate_for_gcd(&ctx, factor),
            GcdPreEvalDirective::Keep
        );
        assert_eq!(
            classify_pre_evaluate_for_gcd(&ctx, x),
            GcdPreEvalDirective::Keep
        );
    }
}
