//! Planning helpers for combine-like-terms rewrite rules.

use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use cas_math::collect_semantics_support::{collect_with_semantics_mode, CollectSemanticsMode};
use std::cmp::Ordering;

use crate::collect_focus_support::select_collect_didactic_focus;

#[derive(Debug, Clone)]
pub struct CollectRulePlan {
    pub new_expr: ExprId,
    pub description: String,
    pub local_before: Option<ExprId>,
    pub local_after: Option<ExprId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectLikeTermsRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
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
    })
}

/// Generic rewrite planner for collect-like-terms on Add/Sub expressions.
///
/// Returns a rewritten expression only when the result is structurally different.
pub fn try_rewrite_collect_like_terms_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) => {}
        _ => return None,
    }

    if !ctx.is_mul_commutative(expr) {
        return None;
    }

    let result = collect_with_semantics_mode(ctx, expr, CollectSemanticsMode::Generic, false)?;
    if result.new_expr != expr && compare_expr(ctx, result.new_expr, expr) != Ordering::Equal {
        Some(result.new_expr)
    } else {
        None
    }
}

/// Generic rewrite planner for collect-like-terms with canonical description.
pub fn try_rewrite_collect_like_terms_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CollectLikeTermsRewrite> {
    let rewritten = try_rewrite_collect_like_terms_expr(ctx, expr)?;
    Some(CollectLikeTermsRewrite {
        rewritten,
        desc: "Collect like terms",
    })
}

#[cfg(test)]
#[path = "collect_rule_support_tests.rs"]
mod tests;
