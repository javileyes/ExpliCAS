//! Planning helpers for combine-like-terms rewrite rules.

use crate::collect_focus_support::select_collect_didactic_focus;
use crate::collect_semantics_support::{collect_with_semantics_mode, CollectSemanticsMode};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, ExprId};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct CollectRulePlan {
    pub new_expr: ExprId,
    pub description: String,
    pub local_before: Option<ExprId>,
    pub local_after: Option<ExprId>,
    pub assumption: Option<String>,
}

/// Build a combine-like-terms rewrite plan with semantics and didactic focus.
///
/// Returns `None` when:
/// - collection is blocked by semantics policy,
/// - result is structurally equivalent to input,
/// - no actual cancellation/combination happened (trivial normalization only).
pub fn try_plan_collect_rule_expr(
    ctx: &mut Context,
    expr: ExprId,
    mode: CollectSemanticsMode,
    undefined_risk: bool,
) -> Option<CollectRulePlan> {
    let result = collect_with_semantics_mode(ctx, expr, mode, undefined_risk)?;

    if compare_expr(ctx, result.new_expr, expr) == Ordering::Equal {
        return None;
    }

    // Skip trivial changes that only normalize signs/coefficient shape.
    if result.cancelled.is_empty() && result.combined.is_empty() {
        return None;
    }

    let focus = select_collect_didactic_focus(ctx, &result.cancelled, &result.combined);
    Some(CollectRulePlan {
        new_expr: result.new_expr,
        description: focus.description,
        local_before: focus.before,
        local_after: focus.after,
        assumption: result.assumption,
    })
}

#[cfg(test)]
mod tests {
    use super::try_plan_collect_rule_expr;
    use crate::collect_semantics_support::CollectSemanticsMode;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn builds_plan_for_real_combination() {
        let mut ctx = Context::new();
        let expr = parse("x + x", &mut ctx).expect("parse");
        let plan = try_plan_collect_rule_expr(&mut ctx, expr, CollectSemanticsMode::Generic, false)
            .expect("plan");
        assert_ne!(plan.new_expr, expr);
        assert!(plan.local_before.is_some());
        assert!(plan.local_after.is_some());
        assert_eq!(plan.description, "Combine like terms");
    }

    #[test]
    fn returns_none_when_strict_blocks() {
        let mut ctx = Context::new();
        let expr = parse("x/(x+1) - x/(x+1)", &mut ctx).expect("parse");
        let plan = try_plan_collect_rule_expr(&mut ctx, expr, CollectSemanticsMode::Strict, true);
        assert!(plan.is_none());
    }
}
