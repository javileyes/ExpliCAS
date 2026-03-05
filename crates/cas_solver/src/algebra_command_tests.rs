#[cfg(test)]
mod tests {
    #[test]
    fn parse_telescope_invocation_input_reads_tail() {
        assert_eq!(
            crate::parse_telescope_invocation_input("telescope 1+cos(x)"),
            Some("1+cos(x)".to_string())
        );
    }

    #[test]
    fn parse_expand_invocation_input_reads_tail() {
        assert_eq!(
            crate::parse_expand_invocation_input("expand (x+1)^2"),
            Some("(x+1)^2".to_string())
        );
    }

    #[test]
    fn parse_expand_log_invocation_input_reads_tail() {
        assert_eq!(
            crate::parse_expand_log_invocation_input("expand_log ln(x*y)"),
            Some("ln(x*y)".to_string())
        );
    }

    #[test]
    fn evaluate_telescope_command_lines_runs() {
        let mut ctx = cas_ast::Context::new();
        let lines =
            crate::evaluate_telescope_command_lines(&mut ctx, "1 + 2*cos(x)").expect("telescope");
        assert!(lines.iter().any(|line| line.contains("Parsed:")));
    }

    #[test]
    fn evaluate_expand_log_command_lines_runs() {
        let mut ctx = cas_ast::Context::new();
        let lines =
            crate::evaluate_expand_log_command_lines(&mut ctx, "ln(x^2*y)").expect("expand_log");
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }

    #[test]
    fn evaluate_expand_wrapped_expression_requires_input() {
        let err = crate::evaluate_expand_wrapped_expression("expand").expect_err("usage");
        assert!(err.contains("Usage: expand"));
    }

    #[test]
    fn evaluate_telescope_invocation_lines_requires_input() {
        let mut ctx = cas_ast::Context::new();
        let err =
            crate::evaluate_telescope_invocation_lines(&mut ctx, "telescope").expect_err("usage");
        assert!(err.contains("Usage: telescope"));
    }

    #[test]
    fn evaluate_expand_log_invocation_lines_requires_input() {
        let mut ctx = cas_ast::Context::new();
        let err =
            crate::evaluate_expand_log_invocation_lines(&mut ctx, "expand_log").expect_err("usage");
        assert!(err.contains("Usage: expand_log"));
    }

    #[test]
    fn evaluate_telescope_invocation_message_joins_lines() {
        let mut ctx = cas_ast::Context::new();
        let message = crate::evaluate_telescope_invocation_message(&mut ctx, "telescope 1+cos(x)")
            .expect("telescope");
        assert!(message.contains("Parsed: 1+cos(x)"));
    }

    #[test]
    fn evaluate_expand_log_invocation_message_returns_result_line() {
        let mut ctx = cas_ast::Context::new();
        let message = crate::evaluate_expand_log_invocation_message(&mut ctx, "expand_log ln(x*y)")
            .expect("expand_log");
        assert!(message.contains("Result:"));
    }
}
