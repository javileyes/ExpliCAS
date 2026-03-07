mod conditional;
mod interval;

use cas_ast::{Context, ExprId, SolutionSet};
use cas_formatter::DisplayExpr;

pub(super) fn format_timeline_solve_result_line(
    context: &Context,
    solution_set: &SolutionSet,
) -> String {
    format!("Result: {}", display_solution_set(context, solution_set))
}

pub(super) fn format_timeline_solve_no_steps_message(
    context: &Context,
    solution_set: &SolutionSet,
) -> String {
    format!(
        "No solving steps to visualize.\n{}",
        format_timeline_solve_result_line(context, solution_set)
    )
}

fn display_solution_set(context: &Context, solution_set: &SolutionSet) -> String {
    match solution_set {
        SolutionSet::Empty => "Empty Set".to_string(),
        SolutionSet::AllReals => "All Real Numbers".to_string(),
        SolutionSet::Discrete(exprs) => display_discrete_solution_set(context, exprs),
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

fn display_discrete_solution_set(context: &Context, exprs: &[ExprId]) -> String {
    let rendered: Vec<String> = exprs
        .iter()
        .map(|expr| display_expr(context, *expr))
        .collect();
    format!("{{ {} }}", rendered.join(", "))
}

fn display_expr(context: &Context, expr: ExprId) -> String {
    format!("{}", DisplayExpr { context, id: expr })
}
