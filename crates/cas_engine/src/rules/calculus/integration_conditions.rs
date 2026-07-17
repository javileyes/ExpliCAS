use super::domain_checks::collect_atanh_open_interval_conditions;
use super::reciprocal_trig_log_domain::collect_reciprocal_trig_log_denominator_conditions;
use cas_ast::{ConditionPredicate, Context, ExprId};

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
    extra: Vec<crate::ImplicitCondition>,
}

impl IntegrationRequiredConditions {
    pub(super) fn from_target(ctx: &mut Context, target: ExprId, var_name: &str) -> Self {
        Self {
            nonzero: integrate_required_nonzero_conditions(ctx, target, var_name),
            positive: integrate_required_positive_conditions(ctx, target, var_name),
            extra: Vec::new(),
        }
    }

    pub(super) fn include_conditions_implied_by_result(
        &mut self,
        ctx: &mut Context,
        result: ExprId,
    ) {
        self.include_reciprocal_trig_log_denominator_conditions(ctx, result);
        self.include_atanh_open_interval_conditions_if_source_positive_absent(ctx, result);
    }

    pub(super) fn include_backend_conditions(
        &mut self,
        ctx: &mut Context,
        conditions: Vec<ConditionPredicate>,
    ) {
        for condition in conditions {
            match condition {
                ConditionPredicate::NonZero(expr) => self.include_nonzero(ctx, expr),
                ConditionPredicate::Positive(expr) => self.include_positive(ctx, expr),
                ConditionPredicate::NonNegative(expr) => {
                    self.extra.push(crate::ImplicitCondition::NonNegative(expr))
                }
                ConditionPredicate::LowerBound { expr, lower } => self
                    .extra
                    .push(crate::ImplicitCondition::LowerBound(expr, lower)),
                ConditionPredicate::Defined(_)
                | ConditionPredicate::InvTrigPrincipalRange { .. }
                | ConditionPredicate::PrincipalBranch { .. }
                | ConditionPredicate::EqZero(_)
                | ConditionPredicate::EqOne(_) => {}
            }
        }
    }

    pub(super) fn suppress_backend_positive_quadratic_source_condition(
        &mut self,
        ctx: &mut Context,
        target: ExprId,
        var_name: &str,
        backend_conditions: &[ConditionPredicate],
    ) {
        if !backend_conditions
            .iter()
            .any(|condition| matches!(condition, ConditionPredicate::Positive(_)))
        {
            return;
        }
        if !super::integration::is_public_algorithmic_backend_symbolic_positive_quadratic_fallback_shape(
            ctx, target, var_name, 0,
        ) {
            return;
        }
        let Some(denominator) = target_denominator(ctx, target) else {
            return;
        };

        let mut retained = Vec::with_capacity(self.positive.len());
        for condition in self.positive.drain(..) {
            if cas_math::expr_domain::exprs_equivalent(ctx, condition, denominator) {
                continue;
            }
            retained.push(condition);
        }
        self.positive = retained;
    }

    fn include_nonzero(&mut self, ctx: &mut Context, expr: ExprId) {
        if self
            .nonzero
            .iter()
            .any(|existing| cas_math::expr_domain::exprs_equivalent(ctx, *existing, expr))
        {
            return;
        }
        self.nonzero.push(expr);
    }

    fn include_positive(&mut self, ctx: &mut Context, expr: ExprId) {
        if self
            .positive
            .iter()
            .any(|existing| cas_math::expr_domain::exprs_equivalent(ctx, *existing, expr))
        {
            return;
        }
        self.positive.push(expr);
    }

    fn include_reciprocal_trig_log_denominator_conditions(
        &mut self,
        ctx: &mut Context,
        result: ExprId,
    ) {
        for condition in collect_reciprocal_trig_log_denominator_conditions(ctx, result) {
            if !self
                .nonzero
                .iter()
                .any(|existing| cas_math::expr_domain::exprs_equivalent(ctx, *existing, condition))
            {
                self.nonzero.push(condition);
            }
        }
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
            .chain(self.extra)
    }
}

fn target_denominator(ctx: &Context, target: ExprId) -> Option<ExprId> {
    match ctx.get(target) {
        cas_ast::Expr::Div(_, denominator) => Some(*denominator),
        cas_ast::Expr::Neg(inner) => target_denominator(ctx, *inner),
        _ => None,
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

    fn rendered_nonzero_conditions(
        ctx: &Context,
        conditions: IntegrationRequiredConditions,
    ) -> Vec<String> {
        conditions
            .into_implicit_conditions()
            .filter_map(|condition| match condition {
                crate::ImplicitCondition::NonZero(expr) => Some(rendered(ctx, expr)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn result_implied_conditions_add_secant_log_denominator_pole() {
        let mut ctx = Context::new();
        let target = parse("2/(3*cos(2*x+1))", &mut ctx).unwrap();
        let result = parse("ln(abs(tan(2*x+1)+sec(2*x+1)))/3", &mut ctx).unwrap();
        let mut conditions = IntegrationRequiredConditions::from_target(&mut ctx, target, "x");

        conditions.include_conditions_implied_by_result(&mut ctx, result);

        assert_eq!(
            rendered_nonzero_conditions(&ctx, conditions),
            vec!["cos(2 * x + 1)"]
        );
    }

    #[test]
    fn result_implied_conditions_add_cosecant_log_denominator_pole() {
        let mut ctx = Context::new();
        let target = parse("1/(3*sin(2*x+1))", &mut ctx).unwrap();
        let result = parse("ln(abs(csc(2*x+1)-cot(2*x+1)))/6", &mut ctx).unwrap();
        let mut conditions = IntegrationRequiredConditions::from_target(&mut ctx, target, "x");

        conditions.include_conditions_implied_by_result(&mut ctx, result);

        assert_eq!(
            rendered_nonzero_conditions(&ctx, conditions),
            vec!["sin(2 * x + 1)"]
        );
    }

    #[test]
    fn result_implied_conditions_add_cosecant_log_denominator_pole_from_negated_sum() {
        let mut ctx = Context::new();
        let target = parse("1/(3*sin(2*x+1))", &mut ctx).unwrap();
        let result = parse("ln(abs(-cot(2*x+1)+csc(2*x+1)))/6", &mut ctx).unwrap();
        let mut conditions = IntegrationRequiredConditions::from_target(&mut ctx, target, "x");

        conditions.include_conditions_implied_by_result(&mut ctx, result);

        assert_eq!(
            rendered_nonzero_conditions(&ctx, conditions),
            vec!["sin(2 * x + 1)"]
        );
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
