/// Evaluate full `explain ...` invocation and return user-facing message text.
pub fn evaluate_explain_invocation_message(
    ctx: &mut cas_ast::Context,
    line: &str,
) -> Result<String, String> {
    let input = crate::extract_explain_command_tail(line);
    super::message::evaluate_explain_command_message(ctx, input)
        .map_err(|error| crate::format_explain_command_error_message(&error))
}
