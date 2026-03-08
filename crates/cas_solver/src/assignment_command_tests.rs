#[cfg(test)]
mod tests {
    use crate::{
        evaluate_assignment_command_message_with, evaluate_let_assignment_command_with,
        format_assignment_command_output_message, AssignmentCommandOutput, AssignmentError,
    };

    #[test]
    fn evaluate_let_assignment_command_with_parse_error() {
        let err = evaluate_let_assignment_command_with("a", |_name, _expr, _lazy| {
            Ok(cas_ast::ExprId::from_raw(10))
        })
        .expect_err("expected parse error");
        assert!(err.contains("Usage: let"));
    }

    #[test]
    fn evaluate_assignment_command_message_with_formats_success() {
        let message = evaluate_assignment_command_message_with(
            "a",
            "x+1",
            false,
            |_name, _expr, _lazy| Ok(cas_ast::ExprId::from_raw(10)),
            |expr| format!("E{}", expr.index()),
        )
        .expect("success");
        assert!(message.contains("a = E10"));
    }

    #[test]
    fn evaluate_assignment_command_message_with_formats_error() {
        let err = evaluate_assignment_command_message_with(
            "1bad",
            "x+1",
            false,
            |_name, _expr, _lazy| Err(AssignmentError::InvalidNameStart),
            |_expr| "ignored".to_string(),
        )
        .expect_err("expected error");
        assert!(err.contains("must start with a letter"));
    }

    #[test]
    fn format_assignment_command_output_message_keeps_lazy_marker() {
        let output = AssignmentCommandOutput {
            name: "f".to_string(),
            expr: cas_ast::ExprId::from_raw(7),
            lazy: true,
        };
        let message = format_assignment_command_output_message(&output, "x + 1");
        assert!(message.contains(":="));
    }
}
