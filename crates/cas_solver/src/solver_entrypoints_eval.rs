//! Eval, limit and transform-oriented solver entrypoints.

use crate::{
    Budget, CasError, ConstFoldMode, ConstFoldResult, DisplayEvalSteps, EvalConfig, Operation,
    PassStats, Step,
};
use cas_math::limit_types::{Approach, LimitOptions};

/// Result of symbolic limit evaluation from solver facade.
#[derive(Debug, Clone)]
pub struct LimitResult {
    /// The computed limit expression (or residual `limit(...)` when unresolved).
    pub expr: cas_ast::ExprId,
    /// Steps emitted by limit evaluation (when requested).
    pub steps: Vec<Step>,
    /// Warning emitted when limit cannot be determined safely.
    pub warning: Option<String>,
}

/// Convert raw eval steps to display-ready, cleaned steps.
pub fn to_display_steps(raw_steps: Vec<Step>) -> DisplayEvalSteps {
    let cleaned = cas_solver_core::eval_step_pipeline::clean_eval_steps(
        raw_steps,
        |s: &Step| s.before,
        |s: &Step| s.after,
        |s: &Step| s.before_local(),
        |s: &Step| s.after_local(),
        |s: &Step| s.global_after,
        |s: &mut Step, gb| s.global_before = Some(gb),
    );
    DisplayEvalSteps(cleaned)
}

/// Expand with budget tracking, returning pass stats for charging.
pub fn expand_with_stats(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> (cas_ast::ExprId, PassStats) {
    let nodes_snap = ctx.stats().nodes_created;
    let estimated_terms = cas_math::expand_estimate::estimate_expand_terms(ctx, expr).unwrap_or(0);
    let result = crate::expand(ctx, expr);
    let nodes_delta = ctx.stats().nodes_created.saturating_sub(nodes_snap);

    let stats = PassStats {
        op: Operation::Expand,
        rewrite_count: 0,
        nodes_delta,
        terms_materialized: estimated_terms,
        poly_ops: 0,
        stop_reason: None,
    };

    (result, stats)
}

/// Fold constants under the given semantic config and mode.
pub fn fold_constants(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    cfg: &EvalConfig,
    mode: ConstFoldMode,
    budget: &mut Budget,
) -> Result<ConstFoldResult, CasError> {
    crate::const_fold_local::fold_constants_local(ctx, expr, cfg, mode, budget)
}

/// Evaluate a symbolic limit with the engine's current limit evaluator.
pub fn limit(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    var: cas_ast::ExprId,
    approach: Approach,
    opts: &LimitOptions,
    _budget: &mut Budget,
) -> Result<LimitResult, CasError> {
    let outcome = cas_math::limits_support::eval_limit_at_infinity(ctx, expr, var, approach, opts);
    Ok(LimitResult {
        expr: outcome.expr,
        steps: Vec::new(),
        warning: outcome.warning,
    })
}
