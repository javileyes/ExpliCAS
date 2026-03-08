mod parse;
mod usage;
mod wrap;

/// Usage string for `telescope`.
pub fn telescope_usage_message() -> &'static str {
    usage::telescope_usage_message()
}

/// Usage string for `expand`.
pub fn expand_usage_message() -> &'static str {
    usage::expand_usage_message()
}

/// Usage string for `expand_log`.
pub fn expand_log_usage_message() -> &'static str {
    usage::expand_log_usage_message()
}

/// Parse `telescope ...` invocation and return input expression.
pub fn parse_telescope_invocation_input(line: &str) -> Option<String> {
    parse::parse_telescope_invocation_input(line)
}

/// Parse `expand ...` invocation and return input expression.
pub fn parse_expand_invocation_input(line: &str) -> Option<String> {
    parse::parse_expand_invocation_input(line)
}

/// Parse `expand_log ...` invocation and return input expression.
pub fn parse_expand_log_invocation_input(line: &str) -> Option<String> {
    parse::parse_expand_log_invocation_input(line)
}

/// Wrap expression as explicit `expand(...)` call.
pub fn wrap_expand_eval_expression(expr: &str) -> String {
    wrap::wrap_expand_eval_expression(expr)
}
