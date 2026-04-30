//! Symbolic integration adapter.
//!
//! Core integration logic lives in `cas_math::symbolic_integration_support`.

use cas_ast::{Context, ExprId};

pub(crate) fn integrate(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    cas_math::symbolic_integration_support::integrate_symbolic_expr(ctx, expr, var)
}

pub(crate) fn integrate_required_nonzero_conditions(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Vec<ExprId> {
    cas_math::symbolic_integration_support::integrate_symbolic_required_nonzero_conditions(
        ctx, expr, var,
    )
}

pub(crate) fn integrate_required_positive_conditions(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Vec<ExprId> {
    cas_math::symbolic_integration_support::integrate_symbolic_required_positive_conditions(
        ctx, expr, var,
    )
}

#[cfg(test)]
mod tests;
