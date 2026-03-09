mod conditional;
mod discrete;
mod intervals;
mod residual;

use cas_ast::{Context, SolutionSet};

pub(crate) fn solution_set_to_output_latex(ctx: &Context, solution_set: &SolutionSet) -> String {
    match solution_set {
        SolutionSet::Empty => r"\emptyset".to_string(),
        SolutionSet::AllReals => r"\mathbb{R}".to_string(),
        SolutionSet::Discrete(exprs) => discrete::render_discrete_solution_set(ctx, exprs),
        SolutionSet::Conditional(cases) => conditional::render_conditional_solution_set(ctx, cases),
        SolutionSet::Continuous(interval) => intervals::render_continuous_interval(ctx, interval),
        SolutionSet::Union(intervals) => intervals::render_interval_union(ctx, intervals),
        SolutionSet::Residual(expr) => residual::render_residual_solution(ctx, *expr),
    }
}
