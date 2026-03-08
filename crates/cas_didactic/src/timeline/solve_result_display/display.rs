use super::{conditional, discrete, interval};
use cas_ast::{Context, ExprId, SolutionSet};
use cas_formatter::DisplayExpr;

pub(super) fn display_solution_set(context: &Context, solution_set: &SolutionSet) -> String {
    match solution_set {
        SolutionSet::Empty => "Empty Set".to_string(),
        SolutionSet::AllReals => "All Real Numbers".to_string(),
        SolutionSet::Discrete(exprs) => {
            discrete::display_discrete_solution_set(context, exprs, display_expr)
        }
        SolutionSet::Continuous(interval) => interval::display_interval(context, interval),
        SolutionSet::Union(intervals) => intervals
            .iter()
            .map(|interval| interval::display_interval(context, interval))
            .collect::<Vec<_>>()
            .join(" U "),
        SolutionSet::Residual(expr) => display_expr(context, *expr),
        SolutionSet::Conditional(cases) => {
            conditional::display_conditional_solution_set(context, cases, display_solution_set)
        }
    }
}

fn display_expr(context: &Context, expr: ExprId) -> String {
    format!("{}", DisplayExpr { context, id: expr })
}
