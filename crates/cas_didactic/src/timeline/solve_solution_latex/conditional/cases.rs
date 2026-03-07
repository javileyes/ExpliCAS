use super::super::render_inner_solution_set_to_latex;
use cas_ast::{Case, Context, SolutionSet};

pub(super) fn collect_conditional_case_lines(context: &Context, cases: &[Case]) -> Vec<String> {
    cases
        .iter()
        .filter_map(|case| render_conditional_case_line(context, case))
        .collect()
}

fn render_conditional_case_line(context: &Context, case: &Case) -> Option<String> {
    if case.when.is_otherwise() && matches!(&case.then.solutions, SolutionSet::Residual(_)) {
        return None;
    }

    let solution_latex = render_inner_solution_set_to_latex(context, &case.then.solutions);
    if case.when.is_otherwise() {
        Some(format!("{} & \\text{{otherwise}}", solution_latex))
    } else {
        let condition_latex = cas_formatter::condition_set_to_latex(&case.when, context);
        Some(format!(
            "{} & \\text{{if }} {}",
            solution_latex, condition_latex
        ))
    }
}
