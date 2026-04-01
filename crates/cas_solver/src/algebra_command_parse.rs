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

/// Usage string for `collect`.
pub fn collect_usage_message() -> &'static str {
    usage::collect_usage_message()
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

/// Parse `collect ...` invocation and return `(expr, var)`.
pub fn parse_collect_invocation_input(line: &str) -> Option<(String, String)> {
    parse::parse_collect_invocation_input(line)
}

/// Parse `expand_log ...` invocation and return input expression.
pub fn parse_expand_log_invocation_input(line: &str) -> Option<String> {
    parse::parse_expand_log_invocation_input(line)
}

/// Wrap expression as explicit `expand(...)` call.
pub fn wrap_expand_eval_expression(expr: &str) -> String {
    wrap::wrap_expand_eval_expression(expr)
}

/// Wrap `(expr, var)` as explicit `collect(expr, var)` call.
pub fn wrap_collect_eval_expression(expr: &str, var: &str) -> String {
    wrap::wrap_collect_eval_expression(expr, var)
}
