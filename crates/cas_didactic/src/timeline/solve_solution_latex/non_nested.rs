use super::expr::render_expr_to_latex;
use super::interval;
use cas_ast::{Context, SolutionSet};

pub(super) fn render_non_nested_solution_set_to_latex(
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
