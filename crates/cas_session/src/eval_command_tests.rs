#[cfg(test)]
mod tests {
    use crate::solver_exports::{
        build_eval_command_render_plan, evaluate_eval_command_output,
        evaluate_eval_text_simplify_with_session, EvalCommandError, EvalCommandOutput,
    };
    use crate::{EvalDisplayMessageKind, EvalMetadataLines, EvalResultLine, SessionState};

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
