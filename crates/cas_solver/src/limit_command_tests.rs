#[cfg(test)]
mod tests {
    #[test]
    fn evaluate_limit_command_lines_empty_input_returns_usage() {
        let err = crate::command_api::limit::evaluate_limit_command_lines("limit")
            .expect_err("expected usage");
        assert!(err.contains("Usage: limit"));
    }

    #[test]
    fn evaluate_limit_command_lines_accepts_function_style_input() {
        let lines = crate::command_api::limit::evaluate_limit_command_lines(
            "limit((x^2 + 3*x)/(2*x^2 - x), x, inf)",
        )
        .expect("function-style limit");

        assert!(
            lines.iter().any(|line| line.contains("1/2")),
            "expected limit output to contain the computed result, got: {lines:?}"
        );
    }
}
