#[cfg(test)]
mod tests {
    #[test]
    fn evaluate_limit_command_lines_empty_input_returns_usage() {
        let err = crate::evaluate_limit_command_lines("limit").expect_err("expected usage");
        assert!(err.contains("Usage: limit"));
    }
}
