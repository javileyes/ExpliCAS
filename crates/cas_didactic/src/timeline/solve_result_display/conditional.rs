use cas_ast::{Case, Context, SolutionSet};
use cas_formatter::condition_set_to_display;

pub(super) fn display_conditional_solution_set<F>(
    context: &Context,
    cases: &[Case],
    display_solution_set: F,
) -> String
where
    F: Fn(&Context, &SolutionSet) -> String,
{
    let case_lines: Vec<String> = cases
        .iter()
        .filter_map(|case| display_conditional_case(context, case, &display_solution_set))
        .collect();

    if case_lines.len() == 1 {
        case_lines[0].trim().to_string()
    } else {
        format!("Conditional:\n{}", case_lines.join("\n"))
    }
}

fn display_conditional_case<F>(
    context: &Context,
    case: &Case,
    display_solution_set: &F,
) -> Option<String>
where
    F: Fn(&Context, &SolutionSet) -> String,
{
    if case.when.is_empty() && matches!(&case.then.solutions, SolutionSet::Residual(_)) {
        return None;
    }

    let solution_str = display_solution_set(context, &case.then.solutions);
    if case.when.is_otherwise() {
        Some(format!("  otherwise: {}", solution_str))
    } else {
        let condition_str = condition_set_to_display(&case.when, context);
        Some(format!("  if {}: {}", condition_str, solution_str))
    }
}
