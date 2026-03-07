use crate::algebra_command_parse::{
    expand_usage_message, parse_expand_invocation_input, wrap_expand_eval_expression,
};

/// Parse and wrap `expand ...` as an explicit `expand(...)` eval input.
pub fn evaluate_expand_wrapped_expression(line: &str) -> Result<String, String> {
    let Some(rest) = parse_expand_invocation_input(line) else {
        return Err(expand_usage_message().to_string());
    };
    Ok(wrap_expand_eval_expression(&rest))
}
