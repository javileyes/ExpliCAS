use crate::algebra_command_parse::{
    collect_usage_message, parse_collect_invocation_input, wrap_collect_eval_expression,
};

/// Parse and wrap `collect ...` as an explicit `collect(expr, var)` eval input.
pub(crate) fn evaluate_collect_wrapped_expression(line: &str) -> Result<String, String> {
    let Some((expr, var)) = parse_collect_invocation_input(line) else {
        return Err(collect_usage_message().to_string());
    };
    Ok(wrap_collect_eval_expression(&expr, &var))
}
