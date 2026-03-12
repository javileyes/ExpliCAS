use crate::{Budget, CasError, Step};
use cas_ast::{Context, ExprId};
use cas_math::limit_types::{Approach, LimitOptions};

/// Result of symbolic limit evaluation from solver facade.
#[derive(Debug, Clone)]
pub struct LimitResult {
    /// The computed limit expression (or residual `limit(...)` when unresolved).
    pub expr: cas_ast::ExprId,
    /// Steps emitted by limit evaluation (when requested).
    #[allow(dead_code)]
    pub steps: Vec<Step>,
    /// Warning emitted when limit cannot be determined safely.
    pub warning: Option<String>,
}

/// Evaluate a symbolic limit with the engine's current limit evaluator.
pub fn limit(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
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
