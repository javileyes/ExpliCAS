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
        SolutionSet::Continuous(interval) => format_output_interval(ctx, interval),
        SolutionSet::Union(intervals) => {
            let parts: Vec<String> = intervals
                .iter()
                .map(|int| format_output_interval(ctx, int))
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
            // Scout cycle-3 honesty contract: a residual that is itself a
            // `solve(...)` call is self-describing — it already carries the
            // full relation (operator included), matching the
            // `integrate(...)` residual convention. The old "Solve: … = 0"
            // wrapper both duplicated the framing and appended a dangling
            // "= 0" that misdescribed inequalities.
            if matches!(ctx.get(*expr), cas_ast::Expr::Function(name, _) if ctx.sym_name(*name) == "solve")
            {
                expr_str
            } else {
                format!("Solve: {expr_str} = 0")
            }
        }
        SolutionSet::Periodic { bases, period } => {
            cas_formatter::display_periodic_family(ctx, bases, *period)
        }
        SolutionSet::PeriodicIntervalUnion { windows, period } => {
            cas_formatter::display_periodic_interval_union(ctx, windows, *period)
        }
    }
}
