/// Parse REPL `timeline` command shape.
pub fn parse_timeline_command_input(rest: &str) -> crate::TimelineCommandInput {
    if let Some(solve_rest) = rest.strip_prefix("solve ") {
        return crate::TimelineCommandInput::Solve(solve_rest.trim().to_string());
    }

    if let Some(inner) = rest
        .strip_prefix("simplify(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return crate::TimelineCommandInput::Simplify {
            expr: inner.trim().to_string(),
            aggressive: true,
        };
    }

    if let Some(simplify_rest) = rest.strip_prefix("simplify ") {
        return crate::TimelineCommandInput::Simplify {
            expr: simplify_rest.trim().to_string(),
            aggressive: true,
        };
    }

    crate::TimelineCommandInput::Simplify {
        expr: rest.trim().to_string(),
        aggressive: false,
    }
}
