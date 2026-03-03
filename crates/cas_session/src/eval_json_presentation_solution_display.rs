use cas_ast::{Context, SolutionSet};
use cas_formatter::DisplayExpr;

pub(crate) fn format_solution_set_eval_json(ctx: &Context, solution_set: &SolutionSet) -> String {
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
        SolutionSet::Conditional(cases) => {
            let mut parts = Vec::new();
            for case in cases {
                if cas_solver::is_pure_residual_otherwise(case) {
                    continue;
                }
                let cond_str = cas_formatter::condition_set_to_display(&case.when, ctx);
                let inner_str = format_solution_set_eval_json(ctx, &case.then.solutions);
                if case.when.is_empty() {
                    parts.push(format!("{inner_str} otherwise"));
                } else {
                    parts.push(format!("{inner_str} if {cond_str}"));
                }
            }

            if parts.is_empty() && !cases.is_empty() {
                for case in cases {
                    if !case.when.is_empty() {
                        let inner_str = format_solution_set_eval_json(ctx, &case.then.solutions);
                        let cond_str = cas_formatter::condition_set_to_display(&case.when, ctx);
                        return format!("{inner_str} if {cond_str}");
                    }
                }
            }
            parts.join("; ")
        }
        SolutionSet::Continuous(interval) => {
            format!(
                "[{}, {}]",
                DisplayExpr {
                    context: ctx,
                    id: interval.min
                },
                DisplayExpr {
                    context: ctx,
                    id: interval.max
                }
            )
        }
        SolutionSet::Union(intervals) => {
            let parts: Vec<String> = intervals
                .iter()
                .map(|int| {
                    format!(
                        "[{}, {}]",
                        DisplayExpr {
                            context: ctx,
                            id: int.min
                        },
                        DisplayExpr {
                            context: ctx,
                            id: int.max
                        }
                    )
                })
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
