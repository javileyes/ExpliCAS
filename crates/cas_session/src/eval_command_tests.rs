#[cfg(test)]
mod tests {
    use crate::eval::{EvalDisplayMessageKind, EvalMetadataLines, EvalResultLine};
    use crate::SessionState;
    #[allow(unused_imports)]
    use cas_solver::session_api::{assumptions::*, eval::*, simplifier::*};

    #[test]
    fn evaluate_eval_command_output_success() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();
        let out = match evaluate_eval_command_output(&mut engine, &mut session, "x + x", false) {
            Ok(out) => out,
            Err(err) => panic!("eval failed: {err:?}"),
        };

        assert!(out.result_line.is_some());
    }

    #[test]
    fn evaluate_eval_command_output_parse_error() {
        let mut engine = cas_solver::runtime::Engine::new();
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
            steps: cas_solver::runtime::to_display_steps(Vec::new()),
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
        let mut engine = cas_solver::runtime::Engine::new();
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

    #[test]
    fn evaluate_eval_text_simplify_with_session_uses_user_defined_function() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();
        cas_solver::session_api::bindings::apply_assignment(
            &mut session,
            &mut engine.simplifier,
            "f(x)",
            "x + 1",
            true,
        )
        .expect("function assignment");

        let out =
            evaluate_eval_text_simplify_with_session(&mut engine, &mut session, "f(2)", false)
                .expect("eval succeeds");
        assert_eq!(out, "3");
    }

    #[test]
    fn evaluate_eval_text_simplify_with_session_reports_unknown_function_in_spanish() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();

        let err =
            evaluate_eval_text_simplify_with_session(&mut engine, &mut session, "foo(x)", false)
                .expect_err("unknown function");
        assert_eq!(err, "Error: función [foo] no definida");
    }

    #[test]
    fn evaluate_eval_text_simplify_with_session_accepts_collect_function() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();

        let out = evaluate_eval_text_simplify_with_session(
            &mut engine,
            &mut session,
            "collect(a*x + b*x + c, x)",
            false,
        )
        .expect("collect succeeds");

        assert!(out.contains("a + b"));
        assert!(out.contains("x"));
        assert!(out.contains("c"));
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_accepts_lazy_function_assignment() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            crate::eval::EvalCommandConfig {
                expr: "f(x) := x + 1",
                auto_store: false,
                max_chars: 2000,
                steps_mode: cas_api_models::EvalStepsMode::Off,
                budget_preset: cas_api_models::EvalBudgetPreset::Standard,
                strict: false,
                domain: cas_api_models::EvalDomainMode::Generic,
                context_mode: cas_api_models::EvalContextMode::Auto,
                branch_mode: cas_api_models::EvalBranchMode::Strict,
                expand_policy: cas_api_models::EvalExpandPolicy::Off,
                complex_mode: cas_api_models::EvalComplexMode::Auto,
                const_fold: cas_api_models::EvalConstFoldMode::Off,
                value_domain: cas_api_models::EvalValueDomain::Real,
                complex_branch: cas_api_models::EvalBranchMode::Principal,
                inv_trig: cas_api_models::EvalInvTrigPolicy::Strict,
                assume_scope: cas_api_models::EvalAssumeScope::Real,
            },
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["result"], "x + 1");
    }
}
