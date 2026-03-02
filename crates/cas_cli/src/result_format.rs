use cas_ast::{BoundType, Case, Context, ExprId, Interval, SolutionSet};
use cas_formatter::DisplayExpr;

/// Display an expression, preferring formatted poly output when available.
pub(crate) fn display_expr_or_poly(ctx: &Context, id: ExprId) -> String {
    if let Some(poly_str) = cas_solver::try_render_poly_result(ctx, id) {
        return poly_str;
    }
    cas_formatter::clean_display_string(&format!("{}", DisplayExpr { context: ctx, id }))
}

/// Normalize the last `Result: ...` line by cleaning display artifacts.
pub(crate) fn clean_result_output_line(lines: &mut [String]) {
    let Some(last) = lines.last_mut() else {
        return;
    };
    let Some(raw_value) = last.strip_prefix("Result: ") else {
        return;
    };
    *last = format!("Result: {}", cas_formatter::clean_display_string(raw_value));
}

/// Returns true when a conditional case is an "otherwise" containing only a
/// residual expression.
pub(crate) fn is_pure_residual_otherwise(case: &Case) -> bool {
    case.when.is_empty() && matches!(&case.then.solutions, SolutionSet::Residual(_))
}

/// Render one interval in a compact textual form.
pub(crate) fn display_interval(ctx: &Context, interval: &Interval) -> String {
    let min_bracket = match interval.min_type {
        BoundType::Open => "(",
        BoundType::Closed => "[",
    };
    let max_bracket = match interval.max_type {
        BoundType::Open => ")",
        BoundType::Closed => "]",
    };

    format!(
        "{}{}, {}{}",
        min_bracket,
        DisplayExpr {
            context: ctx,
            id: interval.min
        },
        DisplayExpr {
            context: ctx,
            id: interval.max
        },
        max_bracket
    )
}

/// Render a full [`SolutionSet`] for REPL/UI textual output.
pub(crate) fn display_solution_set(ctx: &Context, set: &SolutionSet) -> String {
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
        SolutionSet::Residual(expr) => format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: *expr
            }
        ),
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
