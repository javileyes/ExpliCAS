use cas_ast::{BoundType, Case, Context, Interval, SolutionSet};
use cas_formatter::{condition_set_to_display, DisplayExpr};

fn is_pure_residual_otherwise(case: &Case) -> bool {
    case.when.is_empty() && matches!(&case.then.solutions, SolutionSet::Residual(_))
}

fn display_interval(context: &Context, interval: &Interval) -> String {
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
            context,
            id: interval.min
        },
        DisplayExpr {
            context,
            id: interval.max
        },
        max_bracket
    )
}

fn display_solution_set(context: &Context, set: &SolutionSet) -> String {
    match set {
        SolutionSet::Empty => "Empty Set".to_string(),
        SolutionSet::AllReals => "All Real Numbers".to_string(),
        SolutionSet::Discrete(exprs) => {
            let rendered: Vec<String> = exprs
                .iter()
                .map(|e| format!("{}", DisplayExpr { context, id: *e }))
                .collect();
            format!("{{ {} }}", rendered.join(", "))
        }
        SolutionSet::Continuous(interval) => display_interval(context, interval),
        SolutionSet::Union(intervals) => intervals
            .iter()
            .map(|i| display_interval(context, i))
            .collect::<Vec<_>>()
            .join(" U "),
        SolutionSet::Residual(expr) => format!("{}", DisplayExpr { context, id: *expr }),
        SolutionSet::Conditional(cases) => {
            let case_strs: Vec<String> = cases
                .iter()
                .filter_map(|case| {
                    if is_pure_residual_otherwise(case) {
                        return None;
                    }
                    let sol_str = display_solution_set(context, &case.then.solutions);
                    if case.when.is_otherwise() {
                        Some(format!("  otherwise: {}", sol_str))
                    } else {
                        let cond_str = condition_set_to_display(&case.when, context);
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
