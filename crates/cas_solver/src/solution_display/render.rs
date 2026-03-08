mod conditional;
mod discrete;
mod expr;

use cas_ast::{Context, SolutionSet};

use super::display_interval;

/// Render a full [`SolutionSet`] for REPL/UI textual output.
pub fn display_solution_set(ctx: &Context, set: &SolutionSet) -> String {
    match set {
        SolutionSet::Empty => "Empty Set".to_string(),
        SolutionSet::AllReals => "All Real Numbers".to_string(),
        SolutionSet::Discrete(exprs) => discrete::display_discrete_solution_set(ctx, exprs),
        SolutionSet::Continuous(interval) => display_interval(ctx, interval),
        SolutionSet::Union(intervals) => {
            let s: Vec<String> = intervals.iter().map(|i| display_interval(ctx, i)).collect();
            s.join(" U ")
        }
        SolutionSet::Residual(expr) => expr::display_expr(ctx, *expr),
        SolutionSet::Conditional(cases) => {
            conditional::display_conditional_solution_set(ctx, cases)
        }
    }
}
