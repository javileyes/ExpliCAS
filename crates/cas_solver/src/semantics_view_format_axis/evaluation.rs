mod branch;
mod const_fold;
mod domain;
mod inv_trig;
mod value;

use crate::semantics_view_types::SemanticsViewState;

pub(super) fn format_evaluation_axis_lines(state: &SemanticsViewState, axis: &str) -> Vec<String> {
    match axis {
        "domain" => domain::format_domain_axis_lines(state),
        "value" => value::format_value_axis_lines(state),
        "branch" => branch::format_branch_axis_lines(state),
        "inv_trig" => inv_trig::format_inv_trig_axis_lines(state),
        "const_fold" => const_fold::format_const_fold_axis_lines(state),
        _ => unreachable!("unsupported evaluation axis: {axis}"),
    }
}
