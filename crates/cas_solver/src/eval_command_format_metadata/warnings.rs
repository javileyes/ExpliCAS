use crate::command_api::eval::EvalCommandEvalView;

pub(super) fn format_warning_lines(output: &EvalCommandEvalView) -> Vec<String> {
    crate::assumption_format::format_domain_warning_lines(&output.domain_warnings, true, "⚠ ")
}
