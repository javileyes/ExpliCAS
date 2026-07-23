use cas_ast::Context;

use crate::linear_system_command_eval::{
    evaluate_linear_system_command_input, evaluate_linear_system_command_input_with_simplifier,
};
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

/// REPL-parity variant: with the full simplifier, nonlinear 2×2 systems get
/// the same S2 composition shot the wire route has.
pub(super) fn evaluate_linear_system_command_message_with_simplifier(
    simplifier: &mut crate::Simplifier,
    line: &str,
) -> String {
    let spec = parse_linear_system_invocation_input(line);
    match evaluate_linear_system_command_input_with_simplifier(simplifier, &spec) {
        Ok(output) => format_linear_system_result_message(&mut simplifier.context, &output),
        Err(error) => format_linear_system_command_error_message(&error),
    }
}
