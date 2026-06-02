use super::domain_checks::collect_atanh_open_interval_conditions;
use cas_ast::{Context, ExprId};

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

    pub(super) fn include_conditions_implied_by_result(
        &mut self,
        ctx: &mut Context,
        result: ExprId,
    ) {
        self.include_atanh_open_interval_conditions_if_source_positive_absent(ctx, result);
    }

    fn include_atanh_open_interval_conditions_if_source_positive_absent(
        &mut self,
        ctx: &mut Context,
        result: ExprId,
    ) {
        if !self.positive.is_empty() {
            return;
        }

        self.positive
            .extend(collect_atanh_open_interval_conditions(ctx, result));
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
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::IntegrationRequiredConditions;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    fn rendered_positive_conditions(
        ctx: &Context,
        conditions: IntegrationRequiredConditions,
    ) -> Vec<String> {
        conditions
            .into_implicit_conditions()
            .filter_map(|condition| match condition {
                crate::ImplicitCondition::Positive(expr) => Some(rendered(ctx, expr)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn result_implied_conditions_add_atanh_open_interval_when_source_has_no_positive_condition() {
        let mut ctx = Context::new();
        let target = parse("1/(x+1)", &mut ctx).unwrap();
        let result = parse("atanh(x)", &mut ctx).unwrap();
        let mut conditions = IntegrationRequiredConditions::from_target(&mut ctx, target, "x");

        conditions.include_conditions_implied_by_result(&mut ctx, result);

        assert_eq!(
            rendered_positive_conditions(&ctx, conditions),
            vec!["1 - x^2"]
        );
    }

    #[test]
    fn result_implied_conditions_do_not_duplicate_when_source_already_requires_positive() {
        let mut ctx = Context::new();
        let target = parse("tanh(sqrt(x))/(2*sqrt(x))", &mut ctx).unwrap();
        let result = parse("atanh(y)", &mut ctx).unwrap();
        let mut conditions = IntegrationRequiredConditions::from_target(&mut ctx, target, "x");

        conditions.include_conditions_implied_by_result(&mut ctx, result);

        assert_eq!(rendered_positive_conditions(&ctx, conditions), vec!["x"]);
    }
}
