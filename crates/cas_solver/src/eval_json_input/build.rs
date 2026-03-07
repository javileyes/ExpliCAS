mod special;
mod statement;

use cas_api_models::parse_eval_json_special_command;

use super::types::EvalJsonPreparedRequest;

/// Build eval-json request as a solver-owned action enum.
pub fn build_eval_json_request_for_input(
    raw_input: &str,
    ctx: &mut cas_ast::Context,
    auto_store: bool,
) -> Result<EvalJsonPreparedRequest, String> {
    if let Some(command) = parse_eval_json_special_command(raw_input) {
        return special::build_special_command_request(raw_input, ctx, auto_store, command);
    }

    statement::build_statement_request(raw_input, ctx, auto_store)
}
