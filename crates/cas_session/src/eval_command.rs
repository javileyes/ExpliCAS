use cas_ast::{Expr, ExprId};
use cas_formatter::root_style::ParseStyleSignals;

/// Evaluated payload for REPL `eval` rendering.
#[derive(Debug, Clone)]
pub struct EvalCommandOutput {
    pub resolved_expr: ExprId,
    pub style_signals: ParseStyleSignals,
    pub steps: cas_solver::DisplayEvalSteps,
    pub stored_entry_line: Option<String>,
    pub metadata: EvalMetadataLines,
    pub result_line: Option<EvalResultLine>,
}

/// Formatted final result line for eval output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalResultLine {
    pub line: String,
    /// When true, caller should stop rendering extra metadata sections.
    pub terminal: bool,
}

/// Formatted eval metadata lines grouped by display phase.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvalMetadataLines {
    pub warning_lines: Vec<String>,
    pub requires_lines: Vec<String>,
    pub hint_lines: Vec<String>,
    pub assumption_lines: Vec<String>,
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
    pub steps: cas_solver::DisplayEvalSteps,
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

#[derive(Debug, Clone)]
struct EvalCommandEvalView {
    stored_id: Option<u64>,
    parsed: ExprId,
    resolved: ExprId,
    result: cas_solver::EvalResult,
    diagnostics: cas_solver::Diagnostics,
    steps: cas_solver::DisplayEvalSteps,
    domain_warnings: Vec<cas_solver::DomainWarning>,
    blocked_hints: Vec<cas_solver::BlockedHint>,
}

fn statement_to_expr_id(ctx: &mut cas_ast::Context, stmt: cas_parser::Statement) -> ExprId {
    match stmt {
        cas_parser::Statement::Equation(eq) => ctx.call("Equal", vec![eq.lhs, eq.rhs]),
        cas_parser::Statement::Expression(expr) => expr,
    }
}

fn build_simplify_eval_request_from_statement(
    ctx: &mut cas_ast::Context,
    raw_input: &str,
    stmt: cas_parser::Statement,
    auto_store: bool,
) -> cas_solver::EvalRequest {
    cas_solver::EvalRequest {
        raw_input: raw_input.to_string(),
        parsed: statement_to_expr_id(ctx, stmt),
        action: cas_solver::EvalAction::Simplify,
        auto_store,
    }
}

fn format_eval_result_line(
    context: &cas_ast::Context,
    parsed_expr: ExprId,
    result: &cas_solver::EvalResult,
    style_signals: &ParseStyleSignals,
) -> Option<EvalResultLine> {
    let style_prefs = cas_formatter::StylePreferences::from_expression_with_signals(
        context,
        parsed_expr,
        Some(style_signals),
    );

    match result {
        cas_solver::EvalResult::Expr(res) => {
            if let Expr::Function(name, args) = context.get(*res) {
                if context.is_builtin(*name, cas_ast::BuiltinFn::Equal) && args.len() == 2 {
                    return Some(EvalResultLine {
                        line: format!(
                            "Result: {} = {}",
                            cas_formatter::clean_display_string(&format!(
                                "{}",
                                cas_formatter::DisplayExprStyled::new(
                                    context,
                                    args[0],
                                    &style_prefs
                                )
                            )),
                            cas_formatter::clean_display_string(&format!(
                                "{}",
                                cas_formatter::DisplayExprStyled::new(
                                    context,
                                    args[1],
                                    &style_prefs
                                )
                            )),
                        ),
                        terminal: true,
                    });
                }
            }

            Some(EvalResultLine {
                line: format!(
                    "Result: {}",
                    cas_solver::display_expr_or_poly(context, *res)
                ),
                terminal: false,
            })
        }
        cas_solver::EvalResult::SolutionSet(solution_set) => Some(EvalResultLine {
            line: format!(
                "Result: {}",
                cas_solver::display_solution_set(context, solution_set)
            ),
            terminal: false,
        }),
        cas_solver::EvalResult::Set(_sols) => Some(EvalResultLine {
            line: "Result: Set(...)".to_string(),
            terminal: false,
        }),
        cas_solver::EvalResult::Bool(value) => Some(EvalResultLine {
            line: format!("Result: {}", value),
            terminal: false,
        }),
        cas_solver::EvalResult::None => None,
    }
}

fn format_eval_stored_entry_line(
    context: &cas_ast::Context,
    output: &EvalCommandEvalView,
) -> Option<String> {
    output.stored_id.map(|id| {
        format!(
            "#{id}: {}",
            cas_formatter::DisplayExpr {
                context,
                id: output.parsed
            }
        )
    })
}

fn format_eval_metadata_lines(
    context: &mut cas_ast::Context,
    output: &EvalCommandEvalView,
    requires_display: cas_solver::RequiresDisplayLevel,
    debug_mode: bool,
    hints_enabled: bool,
    domain_mode: cas_solver::DomainMode,
    assumption_reporting: cas_solver::AssumptionReporting,
) -> EvalMetadataLines {
    let warning_lines = crate::format_domain_warning_lines(&output.domain_warnings, true, "⚠ ");

    let result_expr = match &output.result {
        cas_solver::EvalResult::Expr(expr_id) => Some(*expr_id),
        _ => None,
    };
    let mut requires_lines = Vec::new();
    if !output.diagnostics.requires.is_empty() {
        let rendered = crate::format_diagnostics_requires_lines(
            context,
            &output.diagnostics,
            result_expr,
            requires_display,
            debug_mode,
        );
        if !rendered.is_empty() {
            requires_lines.push("ℹ️ Requires:".to_string());
            requires_lines.extend(rendered);
        }
    }

    let hint_lines = if hints_enabled {
        let hints =
            crate::filter_blocked_hints_for_eval(context, output.resolved, &output.blocked_hints);
        if hints.is_empty() {
            Vec::new()
        } else {
            crate::format_eval_blocked_hints_lines(context, &hints, domain_mode)
        }
    } else {
        Vec::new()
    };

    let assumption_lines = if assumption_reporting != cas_solver::AssumptionReporting::Off {
        let assumed_conditions = crate::collect_assumed_conditions_from_steps(&output.steps);
        if assumed_conditions.is_empty() {
            Vec::new()
        } else {
            crate::format_assumed_conditions_report_lines(&assumed_conditions)
        }
    } else {
        Vec::new()
    };

    EvalMetadataLines {
        warning_lines,
        requires_lines,
        hint_lines,
        assumption_lines,
    }
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

/// Evaluate plain text simplification input and return final rendered result.
///
/// This is a thin solver-level orchestration helper for CLI/frontends that
/// want `parse -> eval(simplify) -> render result` with a stateful session.
pub fn evaluate_eval_text_simplify_with_session(
    engine: &mut cas_solver::Engine,
    session: &mut crate::SessionState,
    expr: &str,
    auto_store: bool,
) -> Result<String, String> {
    let parsed = cas_parser::parse(expr, &mut engine.simplifier.context)
        .map_err(|e| format!("Parse error: {}", e))?;
    let req = cas_solver::EvalRequest {
        raw_input: expr.to_string(),
        parsed,
        action: cas_solver::EvalAction::Simplify,
        auto_store,
    };
    let output = engine
        .eval(session, req)
        .map_err(|e| format!("Error: {}", e))?;
    let output_view = cas_solver::eval_output_view(&output);
    Ok(cas_solver::json::format_eval_result_text(
        &engine.simplifier.context,
        &output_view.result,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        build_eval_command_render_plan, evaluate_eval_command_output,
        evaluate_eval_text_simplify_with_session, EvalCommandError, EvalCommandOutput,
        EvalDisplayMessageKind, EvalMetadataLines, EvalResultLine,
    };
    use crate::SessionState;

    #[test]
    fn evaluate_eval_command_output_success() {
        let mut engine = cas_solver::Engine::new();
        let mut session = SessionState::new();
        let out = match evaluate_eval_command_output(&mut engine, &mut session, "x + x", false) {
            Ok(out) => out,
            Err(err) => panic!("eval failed: {err:?}"),
        };

        assert!(out.result_line.is_some());
    }

    #[test]
    fn evaluate_eval_command_output_parse_error() {
        let mut engine = cas_solver::Engine::new();
        let mut session = SessionState::new();
        let err = match evaluate_eval_command_output(&mut engine, &mut session, "x +", false) {
            Ok(_) => panic!("expected parse error"),
            Err(err) => err,
        };

        assert!(matches!(err, EvalCommandError::Parse(_)));
    }

    #[test]
    fn build_eval_command_render_plan_respects_steps_and_terminal_result() {
        let mut ctx = cas_ast::Context::new();
        let expr = ctx.num(2);

        let output = EvalCommandOutput {
            resolved_expr: expr,
            style_signals: cas_formatter::root_style::ParseStyleSignals::default(),
            steps: cas_solver::to_display_steps(Vec::new()),
            stored_entry_line: Some("#1: 2".to_string()),
            metadata: EvalMetadataLines {
                warning_lines: vec!["warn".to_string()],
                requires_lines: vec!["req".to_string()],
                hint_lines: vec!["hint".to_string()],
                assumption_lines: vec!["assume".to_string()],
            },
            result_line: Some(EvalResultLine {
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
        let mut engine = cas_solver::Engine::new();
        let mut session = SessionState::new();
        let out = match evaluate_eval_text_simplify_with_session(
            &mut engine,
            &mut session,
            "x + x",
            false,
        ) {
            Ok(out) => out,
            Err(err) => panic!("eval failed: {err}"),
        };

        assert!(out.contains("2 * x"));
    }
}
