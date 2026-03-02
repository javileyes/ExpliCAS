use cas_ast::ExprId;
use cas_formatter::root_style::ParseStyleSignals;

/// Evaluated payload for REPL `eval` rendering.
#[derive(Debug, Clone)]
pub struct EvalCommandOutput {
    pub resolved_expr: ExprId,
    pub style_signals: ParseStyleSignals,
    pub steps: crate::DisplayEvalSteps,
    pub stored_entry_line: Option<String>,
    pub metadata: crate::EvalMetadataLines,
    pub result_line: Option<crate::EvalResultLine>,
}

/// Lightweight message kind for rendering eval output in frontends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalDisplayMessageKind {
    Output,
    Warn,
    Info,
}

/// Frontend-agnostic message line emitted by eval render planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalDisplayMessage {
    pub kind: EvalDisplayMessageKind,
    pub text: String,
}

/// Ordered plan for rendering eval output in a frontend.
#[derive(Debug, Clone)]
pub struct EvalCommandRenderPlan {
    pub pre_messages: Vec<EvalDisplayMessage>,
    pub render_steps: bool,
    pub resolved_expr: ExprId,
    pub style_signals: ParseStyleSignals,
    pub steps: crate::DisplayEvalSteps,
    pub result_message: Option<EvalDisplayMessage>,
    pub result_terminal: bool,
    pub post_messages: Vec<EvalDisplayMessage>,
}

/// Errors while evaluating REPL `eval` command.
#[derive(Debug, Clone)]
pub enum EvalCommandError {
    Parse(cas_parser::ParseError),
    Eval(String),
}

/// Convert an eval output payload into an ordered rendering plan.
pub fn build_eval_command_render_plan(
    output: EvalCommandOutput,
    verbosity_is_none: bool,
) -> EvalCommandRenderPlan {
    let mut pre_messages = Vec::new();
    if let Some(line) = output.stored_entry_line {
        pre_messages.push(EvalDisplayMessage {
            kind: EvalDisplayMessageKind::Output,
            text: line,
        });
    }
    pre_messages.extend(
        output
            .metadata
            .warning_lines
            .into_iter()
            .map(|line| EvalDisplayMessage {
                kind: EvalDisplayMessageKind::Warn,
                text: line,
            }),
    );
    pre_messages.extend(output.metadata.requires_lines.into_iter().map(|line| {
        EvalDisplayMessage {
            kind: EvalDisplayMessageKind::Info,
            text: line,
        }
    }));

    let render_steps = !output.steps.is_empty() || !verbosity_is_none;

    let (result_message, result_terminal) = match output.result_line {
        Some(result) => (
            Some(EvalDisplayMessage {
                kind: EvalDisplayMessageKind::Output,
                text: result.line,
            }),
            result.terminal,
        ),
        None => (None, false),
    };

    let mut post_messages = Vec::new();
    post_messages.extend(
        output
            .metadata
            .hint_lines
            .into_iter()
            .map(|line| EvalDisplayMessage {
                kind: EvalDisplayMessageKind::Info,
                text: line,
            }),
    );
    post_messages.extend(output.metadata.assumption_lines.into_iter().map(|line| {
        EvalDisplayMessage {
            kind: EvalDisplayMessageKind::Info,
            text: line,
        }
    }));

    EvalCommandRenderPlan {
        pre_messages,
        render_steps,
        resolved_expr: output.resolved_expr,
        style_signals: output.style_signals,
        steps: output.steps,
        result_message,
        result_terminal,
        post_messages,
    }
}

/// Evaluate full REPL `eval` input and prepare display payload.
pub fn evaluate_eval_command_output<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    line: &str,
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
    let style_signals = ParseStyleSignals::from_input_string(line);
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
        &style_signals,
    );

    Ok(EvalCommandOutput {
        resolved_expr: output.resolved,
        style_signals,
        steps: output.steps,
        stored_entry_line,
        metadata,
        result_line,
    })
}

/// Evaluate plain text simplification input and return final rendered result.
///
/// This is a thin solver-level orchestration helper for CLI/frontends that
/// want `parse -> eval(simplify) -> render result` with a stateful session.
pub fn evaluate_eval_text_simplify_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    expr: &str,
    auto_store: bool,
) -> Result<String, String>
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
    let parsed = cas_parser::parse(expr, &mut engine.simplifier.context)
        .map_err(|e| format!("Parse error: {}", e))?;
    let req = crate::EvalRequest {
        raw_input: expr.to_string(),
        parsed,
        action: crate::EvalAction::Simplify,
        auto_store,
    };
    let output = engine
        .eval(session, req)
        .map_err(|e| format!("Error: {}", e))?;
    Ok(crate::json::format_eval_result_text(
        &engine.simplifier.context,
        &output.result,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        build_eval_command_render_plan, evaluate_eval_command_output,
        evaluate_eval_text_simplify_with_session, EvalCommandError, EvalCommandOutput,
        EvalDisplayMessageKind,
    };
    use cas_session::SessionState;

    #[test]
    fn evaluate_eval_command_output_success() {
        let mut engine = crate::Engine::new();
        let mut session = SessionState::new();
        let out =
            evaluate_eval_command_output(&mut engine, &mut session, "x + x", false).expect("eval");

        assert!(out.result_line.is_some());
    }

    #[test]
    fn evaluate_eval_command_output_parse_error() {
        let mut engine = crate::Engine::new();
        let mut session = SessionState::new();
        let err = evaluate_eval_command_output(&mut engine, &mut session, "x +", false)
            .expect_err("parse error");

        assert!(matches!(err, EvalCommandError::Parse(_)));
    }

    #[test]
    fn build_eval_command_render_plan_respects_steps_and_terminal_result() {
        let mut ctx = cas_ast::Context::new();
        let expr = ctx.num(2);

        let output = EvalCommandOutput {
            resolved_expr: expr,
            style_signals: cas_formatter::root_style::ParseStyleSignals::default(),
            steps: crate::to_display_steps(Vec::new()),
            stored_entry_line: Some("#1: 2".to_string()),
            metadata: crate::EvalMetadataLines {
                warning_lines: vec!["warn".to_string()],
                requires_lines: vec!["req".to_string()],
                hint_lines: vec!["hint".to_string()],
                assumption_lines: vec!["assume".to_string()],
            },
            result_line: Some(crate::EvalResultLine {
                line: "Result: 2".to_string(),
                terminal: true,
            }),
        };

        let plan = build_eval_command_render_plan(output, true);
        assert!(!plan.render_steps);
        assert!(plan.result_terminal);
        assert_eq!(plan.pre_messages.len(), 3);
        assert_eq!(plan.post_messages.len(), 2);
        assert_eq!(plan.pre_messages[0].kind, EvalDisplayMessageKind::Output);
        assert_eq!(plan.pre_messages[1].kind, EvalDisplayMessageKind::Warn);
        assert_eq!(plan.pre_messages[2].kind, EvalDisplayMessageKind::Info);
    }

    #[test]
    fn evaluate_eval_text_simplify_with_session_returns_rendered_result() {
        let mut engine = crate::Engine::new();
        let mut session = SessionState::new();
        let out = evaluate_eval_text_simplify_with_session(
            &mut engine,
            &mut session,
            "x + x",
            false,
        )
        .expect("eval");

        assert!(out.contains("2 * x"));
    }
}
