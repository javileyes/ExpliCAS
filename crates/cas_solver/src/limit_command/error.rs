use crate::LimitCommandEvalError;

pub(super) const LIMIT_USAGE_MESSAGE: &str =
    "Usage: limit <expr> [, <var> [, <direction> [, safe]]]\n\
                 Examples:\n\
                   limit x^2                      → infinity (default: x → +∞)\n\
                   limit (x^2+1)/(2*x^2-3), x     → 1/2\n\
                   limit x^3/x^2, x, -infinity    → -infinity\n\
                   limit (x-x)/x, x, infinity, safe → 0 (with pre-simplify)";

pub(super) fn format_limit_command_error_message(error: &LimitCommandEvalError) -> String {
    match error {
        LimitCommandEvalError::EmptyInput => LIMIT_USAGE_MESSAGE.to_string(),
        LimitCommandEvalError::Parse(message) => message.clone(),
        LimitCommandEvalError::Limit(message) => {
            format!("Error computing limit: {}", message)
        }
    }
}
