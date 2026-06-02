//! Symbolic integration adapter.
//!
//! Core integration logic lives in `cas_math::symbolic_integration_support`.

use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::{render_integrate_desc_with, NamedVarCall};
use cas_ast::{Context, ExprId};

pub(crate) fn integrate(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    cas_math::symbolic_integration_support::integrate_symbolic_expr(ctx, expr, var)
}

pub(super) fn integrate_rewrite_with_conditions<I>(
    ctx: &mut Context,
    call: &NamedVarCall,
    result: ExprId,
    required_conditions: I,
) -> Rewrite
where
    I: IntoIterator<Item = crate::ImplicitCondition>,
{
    let desc = render_integrate_desc_with(call, |id| {
        format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
    });
    Rewrite::new(result)
        .desc(desc)
        .requires_all(required_conditions)
}

#[cfg(test)]
mod tests;
