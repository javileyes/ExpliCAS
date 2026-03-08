use super::{eval::evaluate_weierstrass_command_lines, usage::weierstrass_usage_message};

/// Parse `weierstrass ...` invocation and return expression tail.
pub fn parse_weierstrass_invocation_input(line: &str) -> Option<String> {
    let rest = line.strip_prefix("weierstrass").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}

/// Evaluate and format `weierstrass ...` invocation.
pub fn evaluate_weierstrass_invocation_lines(
    simplifier: &mut crate::Simplifier,
    line: &str,
) -> Result<Vec<String>, String> {
    let Some(rest) = parse_weierstrass_invocation_input(line) else {
        return Err(weierstrass_usage_message().to_string());
    };
    evaluate_weierstrass_command_lines(simplifier, &rest)
}

/// Evaluate and format `weierstrass ...` invocation as message text.
pub fn evaluate_weierstrass_invocation_message(
    simplifier: &mut crate::Simplifier,
    line: &str,
) -> Result<String, String> {
    let mut lines = evaluate_weierstrass_invocation_lines(simplifier, line)?;
    crate::clean_result_output_line(&mut lines);
    Ok(lines.join("\n"))
}
