#![allow(unused_imports)]

pub use crate::analysis_command_format_equivalence::{
    format_equivalence_result_lines, format_expr_pair_parse_error_message,
};
pub use crate::analysis_command_format_errors::{
    format_explain_command_error_message, format_timeline_command_error_message,
    format_visualize_command_error_message,
};
pub use crate::analysis_command_format_explain::format_explain_gcd_eval_lines;

#[cfg(test)]
mod tests {
    use super::{
        format_equivalence_result_lines, format_explain_command_error_message,
        format_expr_pair_parse_error_message, format_timeline_command_error_message,
        format_visualize_command_error_message,
    };

    #[test]
    fn format_equivalence_result_lines_conditional_includes_requires() {
        let lines =
            format_equivalence_result_lines(&cas_solver::EquivalenceResult::ConditionalTrue {
                requires: vec!["x != 0".to_string()],
            });
        assert!(lines.iter().any(|line| line.contains("conditional")));
        assert!(lines.iter().any(|line| line.contains("x != 0")));
    }

    #[test]
    fn format_timeline_command_error_message_parse_is_human_readable() {
        let err = crate::TimelineCommandEvalError::Simplify(
            crate::TimelineSimplifyEvalError::Parse("bad input".to_string()),
        );
        let msg = format_timeline_command_error_message(&err);
        assert!(msg.contains("Parse error"));
    }

    #[test]
    fn format_explain_command_error_message_parse_is_human_readable() {
        let msg = format_explain_command_error_message(&crate::ExplainCommandEvalError::Parse(
            "oops".to_string(),
        ));
        assert!(msg.contains("Parse error"));
    }

    #[test]
    fn format_visualize_command_error_message_parse_is_human_readable() {
        let msg = format_visualize_command_error_message(&crate::VisualizeEvalError::Parse(
            "bad".to_string(),
        ));
        assert!(msg.contains("Parse error"));
    }

    #[test]
    fn format_expr_pair_parse_error_message_usage_is_human_readable() {
        let msg = format_expr_pair_parse_error_message(
            &crate::ParseExprPairError::MissingDelimiter,
            "equiv",
        );
        assert!(msg.contains("Usage: equiv"));
    }
}
