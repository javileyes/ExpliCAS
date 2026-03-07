mod filter;
mod line;

use super::super::render_inner_solution_set_to_latex;
use cas_ast::{Case, Context};

pub(super) fn collect_conditional_case_lines(context: &Context, cases: &[Case]) -> Vec<String> {
    cases
        .iter()
        .filter(|case| !filter::skip_conditional_case(case))
        .map(|case| {
            line::render_conditional_case_line(context, case, render_inner_solution_set_to_latex)
        })
        .collect()
}
