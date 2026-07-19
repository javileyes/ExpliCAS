use cas_formatter::root_style::ParseStyleSignals;

use crate::command_api::eval::EvalCommandError;
use crate::eval_command_request::build_simplify_eval_request_from_statement;

pub(super) fn build_eval_request(
    engine: &mut crate::Engine,
    line: &str,
) -> Result<(ParseStyleSignals, crate::EvalRequest), EvalCommandError> {
    let style_signals = ParseStyleSignals::from_input_string(line);
    // Fase 4 (D2): `dsolve(...)` reaches the REPL fallback-eval through the
    // same wire grammar as the CLI eval command, so both entries stay in
    // parity. The generic statement-parse below would die on the ODE's inner
    // `=` with a cryptic token error.
    if let Some(cas_api_models::EvalSpecialCommand::Dsolve {
        equation,
        func,
        var,
        conditions,
    }) = cas_api_models::parse_eval_special_command(line)
    {
        let (parsed, _original_equation) =
            crate::eval_input_special::parse_solve_input_for_eval_request(
                &mut engine.simplifier.context,
                &equation,
            )
            .map_err(|e| EvalCommandError::Eval(format!("Parse error in dsolve equation: {e}")))?;
        return Ok((
            style_signals,
            crate::EvalRequest {
                raw_input: line.to_string(),
                parsed,
                action: crate::EvalAction::Dsolve {
                    func,
                    var,
                    conditions,
                },
                auto_store: true,
            },
        ));
    }
    if let Some(message) = cas_api_models::parse_eval_dsolve_command_error(line) {
        return Err(EvalCommandError::Eval(
            crate::parse_error_render::parse_error_message(message),
        ));
    }
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
