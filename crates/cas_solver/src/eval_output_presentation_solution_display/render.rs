use cas_ast::{Context, SolutionSet};
use cas_formatter::DisplayExpr;

use super::conditional::format_conditional_output_solution_set;
use super::interval::format_output_interval;

pub(crate) fn format_output_solution_set(ctx: &Context, solution_set: &SolutionSet) -> String {
    match solution_set {
        SolutionSet::Empty => "No solution".to_string(),
        SolutionSet::AllReals => "All real numbers".to_string(),
        SolutionSet::Discrete(exprs) => {
            if exprs.is_empty() {
                "No solution".to_string()
            } else {
                let sols: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        format!(
                            "{}",
                            DisplayExpr {
                                context: ctx,
                                id: *e
                            }
                        )
                    })
                    .collect();
                format!("{{ {} }}", sols.join(", "))
            }
        }
        SolutionSet::Conditional(cases) => format_conditional_output_solution_set(ctx, cases),
        SolutionSet::Continuous(interval) => {
            format_output_interval(ctx, interval.min, interval.max)
        }
        SolutionSet::Union(intervals) => {
            let parts: Vec<String> = intervals
                .iter()
                .map(|int| format_output_interval(ctx, int.min, int.max))
                .collect();
            parts.join(" U ")
        }
        SolutionSet::Residual(expr) => {
            let expr_str = format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: *expr
                }
            );
            format!("Solve: {expr_str} = 0")
        }
    }
}
