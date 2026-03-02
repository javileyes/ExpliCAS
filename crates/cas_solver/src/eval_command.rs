use cas_ast::ExprId;
use cas_formatter::root_style::ParseStyleSignals;

/// Evaluated payload for REPL `eval` rendering.
#[derive(Debug, Clone)]
pub struct EvalCommandOutput {
    pub resolved_expr: ExprId,
    pub steps: crate::DisplayEvalSteps,
    pub stored_entry_line: Option<String>,
    pub metadata: crate::EvalMetadataLines,
    pub result_line: Option<crate::EvalResultLine>,
}

/// Errors while evaluating REPL `eval` command.
#[derive(Debug, Clone)]
pub enum EvalCommandError {
    Parse(cas_parser::ParseError),
    Eval(String),
}

/// Evaluate full REPL `eval` input and prepare display payload.
pub fn evaluate_eval_command_output<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    line: &str,
    style_signals: &ParseStyleSignals,
    debug_mode: bool,
) -> Result<EvalCommandOutput, EvalCommandError>
where
    S: cas_engine::EvalSession<
        Options = cas_engine::EvalOptions,
        Diagnostics = cas_engine::Diagnostics,
    >,
    S::Store: cas_engine::EvalStore<
        DomainMode = cas_engine::DomainMode,
        RequiredItem = cas_engine::RequiredItem,
        Step = cas_engine::Step,
        Diagnostics = cas_engine::Diagnostics,
    >,
{
    let stmt = cas_parser::parse_statement(line, &mut engine.simplifier.context)
        .map_err(EvalCommandError::Parse)?;

    let req = crate::build_simplify_eval_request_from_statement(
        &mut engine.simplifier.context,
        line,
        stmt,
        true,
    );

    let output = engine
        .eval(session, req)
        .map_err(|e| EvalCommandError::Eval(format!("Error: {}", e)))?;

    let eval_options = session.options().clone();
    let metadata = crate::format_eval_metadata_lines(
        &mut engine.simplifier.context,
        &output,
        crate::EvalMetadataConfig {
            requires_display: eval_options.requires_display,
            debug_mode,
            hints_enabled: eval_options.hints_enabled,
            domain_mode: eval_options.shared.semantics.domain_mode,
            assumption_reporting: eval_options.shared.assumption_reporting,
        },
    );

    let stored_entry_line =
        crate::format_eval_stored_entry_line(&engine.simplifier.context, &output);
    let result_line = crate::format_eval_result_line(
        &engine.simplifier.context,
        output.parsed,
        &output.result,
        style_signals,
    );

    Ok(EvalCommandOutput {
        resolved_expr: output.resolved,
        steps: output.steps,
        stored_entry_line,
        metadata,
        result_line,
    })
}

#[cfg(test)]
mod tests {
    use super::{evaluate_eval_command_output, EvalCommandError};
    use cas_formatter::root_style::ParseStyleSignals;
    use cas_session::SessionState;

    #[test]
    fn evaluate_eval_command_output_success() {
        let mut engine = crate::Engine::new();
        let mut session = SessionState::new();
        let style_signals = ParseStyleSignals::from_input_string("x + x");
        let out =
            evaluate_eval_command_output(&mut engine, &mut session, "x + x", &style_signals, false)
                .expect("eval");

        assert!(out.result_line.is_some());
    }

    #[test]
    fn evaluate_eval_command_output_parse_error() {
        let mut engine = crate::Engine::new();
        let mut session = SessionState::new();
        let style_signals = ParseStyleSignals::from_input_string("x +");
        let err =
            evaluate_eval_command_output(&mut engine, &mut session, "x +", &style_signals, false)
                .expect_err("parse error");

        assert!(matches!(err, EvalCommandError::Parse(_)));
    }
}
