use cas_ast::Context;

use crate::linear_system_command_eval::evaluate_linear_system_command_input;
use crate::linear_system_command_format::{
    format_linear_system_command_error_message, format_linear_system_result_message,
};
use crate::linear_system_command_parse::parse_linear_system_invocation_input;

pub(super) fn evaluate_linear_system_command_message(ctx: &mut Context, line: &str) -> String {
    let spec = parse_linear_system_invocation_input(line);
    match evaluate_linear_system_command_input(ctx, &spec) {
        Ok(output) => format_linear_system_result_message(ctx, &output),
        Err(error) => format_linear_system_command_error_message(&error),
    }
}
