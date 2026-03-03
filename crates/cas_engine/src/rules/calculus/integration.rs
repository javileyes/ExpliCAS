//! Symbolic integration adapter.
//!
//! Core integration logic lives in `cas_math::symbolic_integration_support`.

use cas_ast::{Context, ExprId};

pub(crate) fn integrate(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    cas_math::symbolic_integration_support::integrate_symbolic_expr(ctx, expr, var)
}

#[cfg(test)]
mod tests;
