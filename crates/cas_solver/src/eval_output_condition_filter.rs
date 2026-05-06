use cas_api_models::AssumptionDto;
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use std::collections::HashSet;

pub(crate) struct AssumedConditionFilter {
    displays: HashSet<String>,
    positive_exprs: HashSet<String>,
}

impl AssumedConditionFilter {
    pub(crate) fn from_assumptions(assumptions: &[AssumptionDto]) -> Self {
        Self {
            displays: assumptions
                .iter()
                .map(|assumption| assumption.display.clone())
                .collect(),
            positive_exprs: assumptions
                .iter()
                .filter(|assumption| assumption.kind == "positive")
                .map(|assumption| assumption.expr_canonical.clone())
                .collect(),
        }
    }

    pub(crate) fn covers_required_condition(
        &self,
        ctx: &Context,
        cond: &crate::ImplicitCondition,
    ) -> bool {
        if self.displays.contains(&cond.display(ctx)) {
            return true;
        }

        let crate::ImplicitCondition::NonZero(expr_id) = cond else {
            return false;
        };

        self.positive_exprs.contains(&expr_display(ctx, *expr_id))
    }
}

fn expr_display(ctx: &Context, expr_id: cas_ast::ExprId) -> String {
    DisplayExpr {
        context: ctx,
        id: expr_id,
    }
    .to_string()
}
