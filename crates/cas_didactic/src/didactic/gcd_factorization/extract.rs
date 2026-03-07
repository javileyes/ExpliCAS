use cas_ast::{Context, Expr, ExprId};
use cas_solver::Step;

const GCD_PREFIX: &str = "Simplified fraction by GCD: ";

pub(super) fn extract_gcd_str(step: &Step) -> Option<&str> {
    step.description.strip_prefix(GCD_PREFIX)
}

pub(super) fn extract_fraction_operands(
    context: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    match context.get(expr) {
        Expr::Div(numerator, denominator) => Some((*numerator, *denominator)),
        _ => None,
    }
}
