//! Limit computation engine.
//!
//! Entry point for computing limits with conservative policy.

use cas_ast::{Context, ExprId};
use cas_math::limits_support::eval_limit_at_infinity;

use crate::{Budget, CasError};

use super::{Approach, LimitOptions, LimitResult};

/// Compute the limit of an expression.
///
/// # V1 Scope
/// - Limits to ±∞ for polynomials, rationals, and simple powers
/// - Returns residual `limit(expr, var, approach)` for unresolved limits
///
/// # Example
/// ```ignore
/// let result = limit(
///     ctx,
///     expr,
///     var,
///     Approach::PosInfinity,
///     &LimitOptions::default(),
///     &mut budget,
/// )?;
/// ```
pub fn limit(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: Approach,
    opts: &LimitOptions,
    _budget: &mut Budget,
) -> Result<LimitResult, CasError> {
    let steps = Vec::new();
    let outcome = eval_limit_at_infinity(ctx, expr, var, approach, opts);

    Ok(LimitResult {
        expr: outcome.expr,
        steps,
        warning: outcome.warning,
    })
}

#[cfg(test)]
mod tests;
