mod conditional;
mod interval;

use cas_ast::{Context, ExprId, SolutionSet};
use cas_formatter::LaTeXExpr;

pub(super) fn render_solution_set_to_latex(
    context: &Context,
    solution_set: &SolutionSet,
) -> String {
    match solution_set {
        SolutionSet::Conditional(_) => {
            conditional::render_conditional_solution_set_to_latex(context, solution_set)
        }
        _ => render_non_nested_solution_set_to_latex(context, solution_set),
    }
}

pub(super) fn render_inner_solution_set_to_latex(
    context: &Context,
    solution_set: &SolutionSet,
) -> String {
    match solution_set {
        SolutionSet::Conditional(_) => r"\text{(nested conditional)}".to_string(),
        _ => render_non_nested_solution_set_to_latex(context, solution_set),
    }
}

fn render_non_nested_solution_set_to_latex(
    context: &Context,
    solution_set: &SolutionSet,
) -> String {
    match solution_set {
        SolutionSet::Empty => r"\emptyset".to_string(),
        SolutionSet::AllReals => r"\mathbb{R}".to_string(),
        SolutionSet::Discrete(exprs) => {
            let elements: Vec<String> = exprs
                .iter()
                .map(|expr| render_expr_to_latex(context, *expr))
                .collect();
            format!(r"\left\{{ {} \right\}}", elements.join(", "))
        }
        SolutionSet::Continuous(interval) => interval::render_interval_to_latex(context, interval),
        SolutionSet::Union(intervals) => {
            let parts: Vec<String> = intervals
                .iter()
                .map(|interval| interval::render_interval_to_latex(context, interval))
                .collect();
            parts.join(r" \cup ")
        }
        SolutionSet::Residual(expr) => render_expr_to_latex(context, *expr),
        SolutionSet::Conditional(_) => unreachable!("conditional sets are routed separately"),
    }
}

fn render_expr_to_latex(context: &Context, expr: ExprId) -> String {
    LaTeXExpr { context, id: expr }.to_latex()
}
