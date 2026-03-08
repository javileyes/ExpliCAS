use crate::eval_command_types::EvalCommandEvalView;

pub(super) fn format_warning_lines(output: &EvalCommandEvalView) -> Vec<String> {
    crate::format_domain_warning_lines(&output.domain_warnings, true, "⚠ ")
}
