mod evaluation;
mod reporting;
mod solve;

use crate::SemanticsSetState;

pub(crate) fn set_semantic_axis(
    state: &mut SemanticsSetState,
    axis: &str,
    value: &str,
) -> Option<String> {
    match axis {
        "domain" | "value" | "branch" | "inv_trig" | "const_fold" => {
            evaluation::set_evaluation_axis(state, axis, value)
        }
        "assumptions" | "assume_scope" | "hints" | "requires" => {
            reporting::set_reporting_axis(state, axis, value)
        }
        "solve" => solve::set_solve_axis(value),
        _ => Some(format!(
            "ERROR: Unknown axis '{}'\n\
             Valid axes: domain, value, branch, inv_trig, const_fold, assumptions, assume_scope, hints, solve, requires",
            axis
        )),
    }
}
