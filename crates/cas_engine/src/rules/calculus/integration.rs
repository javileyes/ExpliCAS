//! Symbolic integration adapter.
//!
//! Core integration logic lives in `cas_math::symbolic_integration_support`.

use super::domain_checks::collect_atanh_open_interval_conditions;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::{render_integrate_desc_with, NamedVarCall};
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

pub(super) struct IntegrationRequiredConditions {
    nonzero: Vec<ExprId>,
    positive: Vec<ExprId>,
}

impl IntegrationRequiredConditions {
    pub(super) fn from_target(ctx: &mut Context, target: ExprId, var_name: &str) -> Self {
        Self {
            nonzero: integrate_required_nonzero_conditions(ctx, target, var_name),
            positive: integrate_required_positive_conditions(ctx, target, var_name),
        }
    }

    pub(super) fn extend_atanh_result_conditions_if_source_positive_absent(
        &mut self,
        ctx: &mut Context,
        result: ExprId,
    ) {
        if self.positive.is_empty() {
            self.positive
                .extend(collect_atanh_open_interval_conditions(ctx, result));
        }
    }

    pub(super) fn has_positive(&self) -> bool {
        !self.positive.is_empty()
    }

    pub(super) fn into_implicit_conditions(self) -> impl Iterator<Item = crate::ImplicitCondition> {
        self.nonzero
            .into_iter()
            .map(crate::ImplicitCondition::NonZero)
            .chain(
                self.positive
                    .into_iter()
                    .map(crate::ImplicitCondition::Positive),
            )
    }
}

#[cfg(test)]
mod tests;
