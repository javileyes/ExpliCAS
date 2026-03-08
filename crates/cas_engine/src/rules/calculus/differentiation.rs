//! Symbolic differentiation adapter.
//!
//! Core differentiation logic lives in `cas_math::symbolic_differentiation_support`.

use cas_ast::{Context, ExprId};

pub(crate) fn differentiate(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(ctx, expr, var)
}
