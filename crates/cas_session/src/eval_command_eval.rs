use cas_formatter::root_style::ParseStyleSignals;

use crate::eval_command_format::{
    format_eval_metadata_lines, format_eval_result_line, format_eval_stored_entry_line,
};
use crate::eval_command_request::build_simplify_eval_request_from_statement;
use crate::eval_command_types::{EvalCommandError, EvalCommandEvalView, EvalCommandOutput};

/// Evaluate full REPL `eval` input and prepare display payload.
pub fn evaluate_eval_command_output(
    engine: &mut cas_solver::Engine,
    session: &mut crate::SessionState,
    line: &str,
    debug_mode: bool,
) -> Result<EvalCommandOutput, EvalCommandError> {
    let style_signals = ParseStyleSignals::from_input_string(line);
    let stmt = cas_parser::parse_statement(line, &mut engine.simplifier.context)
        .map_err(EvalCommandError::Parse)?;

    let req = build_simplify_eval_request_from_statement(
        &mut engine.simplifier.context,
        line,
        stmt,
        true,
    );

    let output = engine
        .eval(session, req)
        .map_err(|e| EvalCommandError::Eval(format!("Error: {}", e)))?;
    let output_view = cas_solver::eval_output_view(&output);
    let eval_view = EvalCommandEvalView {
        stored_id: output_view.stored_id,
        parsed: output_view.parsed,
        resolved: output_view.resolved,
        result: output_view.result,
        diagnostics: output_view.diagnostics,
        steps: output_view.steps,
        domain_warnings: output_view.domain_warnings,
        blocked_hints: output_view.blocked_hints,
    };

    let eval_options = session.options().clone();
    let metadata = format_eval_metadata_lines(
        &mut engine.simplifier.context,
        &eval_view,
        eval_options.requires_display,
        debug_mode,
        eval_options.hints_enabled,
        eval_options.shared.semantics.domain_mode,
        eval_options.shared.assumption_reporting,
    );

    let stored_entry_line = format_eval_stored_entry_line(&engine.simplifier.context, &eval_view);
    let result_line = format_eval_result_line(
        &engine.simplifier.context,
        eval_view.parsed,
        &eval_view.result,
        &style_signals,
    );

    Ok(EvalCommandOutput {
        resolved_expr: eval_view.resolved,
        style_signals,
        steps: eval_view.steps,
        stored_entry_line,
        metadata,
        result_line,
    })
}
