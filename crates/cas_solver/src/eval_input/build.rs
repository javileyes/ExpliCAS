mod special;
mod statement;

use cas_api_models::{
    parse_eval_dsolve_command_error, parse_eval_limit_command_error, parse_eval_special_command,
};

use super::PreparedEvalRequest;

/// Build a typed eval request as a solver-owned action enum.
pub(crate) fn build_prepared_eval_request_for_input(
    raw_input: &str,
    ctx: &mut cas_ast::Context,
    auto_store: bool,
) -> Result<PreparedEvalRequest, String> {
    if let Some(command) = parse_eval_special_command(raw_input) {
        return special::build_special_command_request(raw_input, ctx, auto_store, command);
    }
    if let Some(message) = parse_eval_limit_command_error(raw_input) {
        return Err(crate::parse_error_render::parse_error_message(message));
    }
    if let Some(message) = parse_eval_dsolve_command_error(raw_input) {
        return Err(crate::parse_error_render::parse_error_message(message));
    }

    statement::build_statement_request(raw_input, ctx, auto_store)
}
