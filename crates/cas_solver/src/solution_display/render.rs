use cas_ast::{Context, SolutionSet};
use cas_formatter::DisplayExpr;

use super::{display_interval, is_pure_residual_otherwise};

/// Render a full [`SolutionSet`] for REPL/UI textual output.
pub fn display_solution_set(ctx: &Context, set: &SolutionSet) -> String {
    match set {
        SolutionSet::Empty => "Empty Set".to_string(),
        SolutionSet::AllReals => "All Real Numbers".to_string(),
        SolutionSet::Discrete(exprs) => {
            let s: Vec<String> = exprs
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
            format!("{{ {} }}", s.join(", "))
        }
        SolutionSet::Continuous(interval) => display_interval(ctx, interval),
        SolutionSet::Union(intervals) => {
            let s: Vec<String> = intervals.iter().map(|i| display_interval(ctx, i)).collect();
            s.join(" U ")
        }
        SolutionSet::Residual(expr) => {
            format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: *expr
                }
            )
        }
        SolutionSet::Conditional(cases) => {
            let case_strs: Vec<String> = cases
                .iter()
                .filter_map(|case| {
                    if is_pure_residual_otherwise(case) {
                        return None;
                    }
                    let sol_str = display_solution_set(ctx, &case.then.solutions);
                    if case.when.is_otherwise() {
                        Some(format!("  otherwise: {}", sol_str))
                    } else {
                        let cond_str = cas_formatter::condition_set_to_display(&case.when, ctx);
                        Some(format!("  if {}: {}", cond_str, sol_str))
                    }
                })
                .collect();
            if case_strs.len() == 1 {
                case_strs[0].trim().to_string()
            } else {
                format!("Conditional:\n{}", case_strs.join("\n"))
            }
        }
    }
}
