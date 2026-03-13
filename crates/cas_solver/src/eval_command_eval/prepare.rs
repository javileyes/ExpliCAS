use cas_formatter::root_style::ParseStyleSignals;

use crate::command_api::eval::EvalCommandError;
use crate::eval_command_request::build_simplify_eval_request_from_statement;

pub(super) fn build_eval_request(
    engine: &mut crate::Engine,
    line: &str,
) -> Result<(ParseStyleSignals, crate::EvalRequest), EvalCommandError> {
    let style_signals = ParseStyleSignals::from_input_string(line);
    let stmt = cas_parser::parse_statement(line, &mut engine.simplifier.context)
        .map_err(EvalCommandError::Parse)?;
    let req = build_simplify_eval_request_from_statement(
        &mut engine.simplifier.context,
        line,
        stmt,
        true,
    );
    Ok((style_signals, req))
}
