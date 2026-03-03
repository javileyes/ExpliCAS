//! Session-level helpers for `limit` command parsing/formatting.

const LIMIT_USAGE_MESSAGE: &str = "Usage: limit <expr> [, <var> [, <direction> [, safe]]]\n\
                 Examples:\n\
                   limit x^2                      → infinity (default: x → +∞)\n\
                   limit (x^2+1)/(2*x^2-3), x     → 1/2\n\
                   limit x^3/x^2, x, -infinity    → -infinity\n\
                   limit (x-x)/x, x, infinity, safe → 0 (with pre-simplify)";

fn extract_limit_command_tail(line: &str) -> &str {
    line.strip_prefix("limit").unwrap_or(line).trim()
}

fn format_limit_command_error_message(error: &cas_solver::LimitCommandEvalError) -> String {
    match error {
        cas_solver::LimitCommandEvalError::EmptyInput => LIMIT_USAGE_MESSAGE.to_string(),
        cas_solver::LimitCommandEvalError::Parse(message) => message.clone(),
        cas_solver::LimitCommandEvalError::Limit(message) => {
            format!("Error computing limit: {}", message)
        }
    }
}

fn format_limit_command_eval_lines(output: &cas_solver::LimitCommandEvalOutput) -> Vec<String> {
    let dir_disp = match output.approach {
        cas_solver::Approach::PosInfinity => "+∞",
        cas_solver::Approach::NegInfinity => "-∞",
    };
    let mut lines = vec![format!(
        "lim_{{{}→{}}} = {}",
        output.var, dir_disp, output.result
    )];
    if let Some(warning) = &output.warning {
        lines.push(format!("Warning: {}", warning));
    }
    lines
}

/// Evaluate `limit` command input and return final display lines.
pub fn evaluate_limit_command_lines(line: &str) -> Result<Vec<String>, String> {
    let rest = extract_limit_command_tail(line);
    if rest.is_empty() {
        return Err(LIMIT_USAGE_MESSAGE.to_string());
    }

    let output = cas_solver::evaluate_limit_command_input(rest)
        .map_err(|error| format_limit_command_error_message(&error))?;
    Ok(format_limit_command_eval_lines(&output))
}

#[cfg(test)]
mod tests {
    #[test]
    fn evaluate_limit_command_lines_empty_input_returns_usage() {
        let err = super::evaluate_limit_command_lines("limit").expect_err("expected usage");
        assert!(err.contains("Usage: limit"));
    }
}
