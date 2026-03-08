/// Parse optional solve invocation flags and return (check_enabled, remaining_tail).
pub fn parse_solve_invocation_check(input: &str, default_check_enabled: bool) -> (bool, &str) {
    let trimmed = input.trim();
    if let Some(rest) = trimmed.strip_prefix("--check") {
        (true, rest.trim_start())
    } else {
        (default_check_enabled, trimmed)
    }
}

/// Parse REPL `solve` argument shape.
pub fn parse_solve_command_input(input: &str) -> crate::SolveCommandInput {
    if let Some((eq, var)) = crate::input_parse_common::rsplit_ignoring_parens(input, ',') {
        return crate::SolveCommandInput {
            equation: eq.trim().to_string(),
            variable: Some(var.trim().to_string()),
        };
    }

    if let Some((eq, var)) = crate::input_parse_common::rsplit_ignoring_parens(input, ' ') {
        let eq_trim = eq.trim();
        let var_trim = var.trim();

        let has_operators_after_eq = if let Some(eq_pos) = eq_trim.find('=') {
            let after_eq = &eq_trim[eq_pos + 1..];
            after_eq.contains('+')
                || after_eq.contains('-')
                || after_eq.contains('*')
                || after_eq.contains('/')
                || after_eq.contains('^')
        } else {
            false
        };

        if !var_trim.is_empty()
            && var_trim.chars().all(char::is_alphabetic)
            && !eq_trim.ends_with('=')
            && !has_operators_after_eq
        {
            return crate::SolveCommandInput {
                equation: eq_trim.to_string(),
                variable: Some(var_trim.to_string()),
            };
        }
    }

    crate::SolveCommandInput {
        equation: input.to_string(),
        variable: None,
    }
}
