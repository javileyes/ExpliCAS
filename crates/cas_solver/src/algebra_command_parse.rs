const TELESCOPE_USAGE_MESSAGE: &str = "Usage: telescope <expression>\n\
                 Example: telescope 1 + 2*cos(x) + 2*cos(2*x) - sin(5*x/2)/sin(x/2)";
const EXPAND_USAGE_MESSAGE: &str = "Usage: expand <expr>\n\
                 Description: Aggressively expands and distributes polynomials.\n\
                 Example: expand 1/2 * (sqrt(2) - 1) -> sqrt(2)/2 - 1/2";
const EXPAND_LOG_USAGE_MESSAGE: &str = "Usage: expand_log <expr>\n\
                 Description: Expand logarithms using log properties.\n\
                 Transformations:\n\
                   ln(x*y)   -> ln(x) + ln(y)\n\
                   ln(x/y)   -> ln(x) - ln(y)\n\
                   ln(x^n)   -> n * ln(x)\n\
                 Example: expand_log ln(x^2 * y) -> 2*ln(x) + ln(y)";

/// Usage string for `telescope`.
pub fn telescope_usage_message() -> &'static str {
    TELESCOPE_USAGE_MESSAGE
}

/// Usage string for `expand`.
pub fn expand_usage_message() -> &'static str {
    EXPAND_USAGE_MESSAGE
}

/// Usage string for `expand_log`.
pub fn expand_log_usage_message() -> &'static str {
    EXPAND_LOG_USAGE_MESSAGE
}

/// Parse `telescope ...` invocation and return input expression.
pub fn parse_telescope_invocation_input(line: &str) -> Option<String> {
    let rest = line.strip_prefix("telescope").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}

/// Parse `expand ...` invocation and return input expression.
pub fn parse_expand_invocation_input(line: &str) -> Option<String> {
    let rest = line.strip_prefix("expand").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}

/// Parse `expand_log ...` invocation and return input expression.
pub fn parse_expand_log_invocation_input(line: &str) -> Option<String> {
    let rest = line.strip_prefix("expand_log").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}

/// Wrap expression as explicit `expand(...)` call.
pub fn wrap_expand_eval_expression(expr: &str) -> String {
    format!("expand({expr})")
}
