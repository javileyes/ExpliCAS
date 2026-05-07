#[cfg(test)]
mod tests {
    use crate::eval::{EvalDisplayMessageKind, EvalMetadataLines, EvalResultLine};
    use crate::SessionState;
    use cas_formatter::{DisplayExpr, LaTeXExpr};
    #[allow(unused_imports)]
    use cas_solver::session_api::{assumptions::*, eval::*, simplifier::*};

    fn standard_eval_config<'a>(expr: &'a str) -> crate::eval::EvalCommandConfig<'a> {
        crate::eval::EvalCommandConfig {
            expr,
            auto_store: false,
            max_chars: 2000,
            time_budget_ms: None,
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
        }
    }

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
    fn evaluate_eval_text_simplify_with_session_reports_diff_requires_explicit_variable() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();

        let err = evaluate_eval_text_simplify_with_session(
            &mut engine,
            &mut session,
            "diff(sin(e^(x^2)))",
            false,
        )
        .expect_err("invalid diff arity");
        assert_eq!(
            err,
            "Error: diff requiere variable explícita: diff(expr, x)"
        );
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
    fn evaluate_eval_text_simplify_with_session_accepts_expand_log_function_name() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();

        let out = evaluate_eval_text_simplify_with_session(
            &mut engine,
            &mut session,
            "expand_log(ln(x*y))",
            false,
        )
        .expect("expand_log succeeds");

        assert!(out.contains("ln(x"));
        assert!(out.contains("y"));
    }

    #[test]
    fn evaluate_eval_text_simplify_with_session_accepts_derive_special_command() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();

        let out = evaluate_eval_text_simplify_with_session(
            &mut engine,
            &mut session,
            "derive x + x, 2*x",
            false,
        )
        .expect("derive succeeds");

        assert!(out.contains("2"));
        assert!(out.contains("x"));
    }

    #[test]
    fn evaluate_eval_text_simplify_with_session_accepts_function_style_derive_special_command() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();

        let out = evaluate_eval_text_simplify_with_session(
            &mut engine,
            &mut session,
            "derive(x + x, 2*x)",
            false,
        )
        .expect("derive succeeds");

        assert!(out.contains("2"));
        assert!(out.contains("x"));
    }

    #[test]
    fn evaluate_eval_text_simplify_with_session_accepts_solve_system_special_command() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();

        let out = evaluate_eval_text_simplify_with_session(
            &mut engine,
            &mut session,
            "solve_system(x+y=3; x-y=1; x; y)",
            false,
        )
        .expect("solve_system succeeds");

        assert_eq!(out, "{ x = 2, y = 1 }");
    }

    #[test]
    fn evaluate_eval_text_simplify_with_session_restores_session_and_simplifier_steps_mode() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();
        session.options_mut().steps_mode = cas_solver::runtime::StepsMode::Compact;
        engine
            .simplifier
            .set_steps_mode(cas_solver::runtime::StepsMode::Compact);

        let out =
            evaluate_eval_text_simplify_with_session(&mut engine, &mut session, "x + x", false)
                .expect("eval succeeds");

        assert_eq!(out, "2 * x");
        assert_eq!(
            session.options().steps_mode,
            cas_solver::runtime::StepsMode::Compact
        );
        assert_eq!(
            engine.simplifier.get_steps_mode(),
            cas_solver::runtime::StepsMode::Compact
        );
    }

    #[test]
    fn evaluate_eval_text_simplify_with_session_handles_triple_sine_plus_rational_against_hyperbolic_pythagorean_regression(
    ) {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();

        let out = evaluate_eval_text_simplify_with_session(
            &mut engine,
            &mut session,
            "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
            false,
        )
        .expect("eval succeeds");

        assert_eq!(out, "0");
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_accepts_lazy_function_assignment() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            crate::eval::EvalCommandConfig {
                expr: "f(x) := x + 1",
                auto_store: false,
                max_chars: 2000,
                time_budget_ms: None,
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

    #[test]
    fn evaluate_eval_command_pretty_with_session_accepts_derive_special_command() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            crate::eval::EvalCommandConfig {
                expr: "derive x + x, 2*x",
                auto_store: false,
                max_chars: 2000,
                time_budget_ms: None,
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
        let result = payload["result"].as_str().expect("result string");
        assert!(result.contains('2'));
        assert!(result.contains('x'));
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_false_equiv_includes_residual_diagnostics() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("equiv(x^2, x)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["result"], "false");

        let diagnostics = payload["equivalence_diagnostics"]
            .as_object()
            .expect("equivalence diagnostics");
        let residual = diagnostics
            .get("residual")
            .and_then(|value| value.as_str())
            .expect("residual");
        assert!(
            residual.contains('x'),
            "expected residual to mention x, got: {residual}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_false_equiv_simplifies_residual_diagnostics() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("equiv((1+x)^5, x^5+5*x^4+10*x^3+10*x^2+5*x)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["result"], "false");

        let diagnostics = payload["equivalence_diagnostics"]
            .as_object()
            .expect("equivalence diagnostics");
        assert_eq!(
            diagnostics.get("residual").and_then(|value| value.as_str()),
            Some("1")
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_surfaces_blocked_diff_domain_hint() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("diff(atanh(sqrt(x^2+2)), x)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["result"], "diff(atanh((x^2 + 2)^(1/2)), x)");

        let blocked = payload["blocked_hints"]
            .as_array()
            .expect("blocked_hints array");
        assert_eq!(blocked.len(), 1);
        assert_eq!(blocked[0]["rule"], "Symbolic Differentiation");
        assert_eq!(
            blocked[0]["tip"],
            "real domain is empty; no real derivative is exposed"
        );
        let condition = blocked[0]["requires"][0]
            .as_str()
            .expect("blocked hint condition");
        assert!(
            condition.contains("-x") && condition.contains("> 0"),
            "expected concrete impossible positive gap, got: {condition}"
        );

        let wire_messages = payload["wire"]["messages"]
            .as_array()
            .expect("wire messages");
        assert!(
            wire_messages.iter().any(|message| message["text"]
                .as_str()
                .is_some_and(|text| text.contains("Blocked: requires -x"))),
            "wire reply should include the blocked hint, got: {wire_messages:?}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_groups_repeated_blocked_hint_messages() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            crate::eval::EvalCommandConfig {
                domain: cas_api_models::EvalDomainMode::Strict,
                ..standard_eval_config("x/x")
            },
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["result"], "x / x");
        assert_eq!(
            payload["blocked_hints"]
                .as_array()
                .expect("blocked_hints array")
                .len(),
            3,
            "structured hints should remain ungrouped for API consumers"
        );

        let wire_messages = payload["wire"]["messages"]
            .as_array()
            .expect("wire messages");
        let blocked_messages: Vec<&str> = wire_messages
            .iter()
            .filter_map(|message| message["text"].as_str())
            .filter(|text| text.contains("Blocked: requires x"))
            .collect();
        assert_eq!(
            blocked_messages.len(),
            1,
            "wire display should group repeated blocked hints: {wire_messages:?}"
        );
        assert!(blocked_messages[0].contains("Cancel Identical Numerator/Denominator"));
        assert!(blocked_messages[0].contains("Simplify Nested Fraction"));
        assert!(blocked_messages[0].contains("Cancel Common Factors"));
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_surfaces_requires_for_conditional_derive_target() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            crate::eval::EvalCommandConfig {
                expr: "derive(a*x + b*x + c, x*(a + b + c/x))",
                auto_store: false,
                max_chars: 2000,
                time_budget_ms: None,
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
        let required = payload["required_display"]
            .as_array()
            .expect("required display array");
        assert!(
            required.iter().any(|item| {
                item.as_str()
                    .is_some_and(|display| display.contains("x") && display.contains("≠ 0"))
            }),
            "expected x != 0 in required_conditions, got: {required:?}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_preserves_derive_operator_in_input_latex() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            crate::eval::EvalCommandConfig {
                expr: "derive(x + x, 2*x)",
                auto_store: false,
                max_chars: 2000,
                time_budget_ms: None,
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
        let input_latex = payload["input_latex"].as_str().expect("input_latex string");
        assert!(
            input_latex.contains("\\operatorname{derive}"),
            "expected input_latex to preserve the derive operator, got: {input_latex}"
        );
        assert!(
            input_latex.contains("{x} + {x}") || input_latex.contains("x + x"),
            "expected input_latex to include the source expression, got: {input_latex}"
        );
        assert!(
            input_latex.contains("2\\cdot x") || input_latex.contains("2 x"),
            "expected input_latex to include the target expression, got: {input_latex}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_keeps_factor_derive_steps_visible() {
        let mut config = standard_eval_config("derive(a^3-b^3, (a-b)*(a^2+a*b+b^2))");
        config.steps_mode = cas_api_models::EvalStepsMode::On;

        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            config,
            |steps, _events, context, _steps_mode| {
                steps
                    .iter()
                    .enumerate()
                    .map(|(index, step)| cas_api_models::StepWire {
                        index: index + 1,
                        rule: step.rule_name.to_string(),
                        rule_latex: String::new(),
                        before: DisplayExpr {
                            context,
                            id: step.before,
                        }
                        .to_string(),
                        after: DisplayExpr {
                            context,
                            id: step.after,
                        }
                        .to_string(),
                        before_latex: LaTeXExpr {
                            context,
                            id: step.before,
                        }
                        .to_latex(),
                        after_latex: LaTeXExpr {
                            context,
                            id: step.after,
                        }
                        .to_latex(),
                        substeps: Vec::new(),
                    })
                    .collect()
            },
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert!(
            payload["steps_count"].as_u64().unwrap_or(0) > 0,
            "expected derive factorization to keep visible eval steps, got: {payload}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_preserves_limit_operator_in_input_latex() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            crate::eval::EvalCommandConfig {
                expr: "limit((x^2 + 3*x)/(2*x^2 - x), x, inf)",
                auto_store: false,
                max_chars: 2000,
                time_budget_ms: None,
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
        assert_eq!(payload["warnings"], serde_json::json!([]));
        let input_latex = payload["input_latex"].as_str().expect("input_latex string");
        assert!(
            input_latex.contains("\\lim_{x \\to \\infty}"),
            "expected input_latex to preserve the limit operator, got: {input_latex}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_returns_finite_limit_residual() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("limit(ln(x), x, -1)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["input"], "limit(ln(x), x, -1)");
        assert_eq!(payload["result"], "limit(ln(x), x, -1)");
        let input_latex = payload["input_latex"].as_str().expect("input_latex string");
        assert!(
            input_latex.contains("\\lim_{x \\to -1}"),
            "expected input_latex to preserve finite limit point, got: {input_latex}"
        );

        let warnings = payload["warnings"].as_array().expect("warnings array");
        assert!(
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Evaluation"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("Finite point limits are not supported safely yet")
                    })
            }),
            "expected finite limit residual warning, got: {warnings:?}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_computes_finite_limit_of_independent_expr() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("limit(ln(y), x, -1)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["result"], "ln(y)");
        assert_eq!(payload["warnings"], serde_json::json!([]));

        let required = payload["required_display"]
            .as_array()
            .expect("required_display array");
        assert!(
            required.iter().any(|condition| condition == "y > 0"),
            "expected independent expression domain condition to remain visible, got: {required:?}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_surfaces_unresolved_limit_warning() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("limit(ln(x), x, -infinity)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["result"], "limit(ln(x), x, -infinity)");

        let warnings = payload["warnings"].as_array().expect("warnings array");
        assert!(
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Evaluation"
                    && warning["assumption"]
                        .as_str()
                        .is_some_and(|message| message.contains("Could not determine limit safely"))
            }),
            "expected unresolved limit warning in eval wire output, got: {warnings:?}"
        );
        assert!(
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Domain Path"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("x -> -infinity")
                            && message.contains("expression requires x > 0")
                    })
            }),
            "expected negative-infinity path/domain warning in eval wire output, got: {warnings:?}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_does_not_warn_for_resolved_negative_infinity_log_path(
    ) {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("limit(ln(-x + 1), x, -infinity)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["result"], "infinity");
        assert_eq!(payload["warnings"], serde_json::json!([]));
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_surfaces_nonnegative_path_conflict_for_limit() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("limit(sqrt(x), x, -infinity)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["result"], "limit(sqrt(x), x, -infinity)");

        let warnings = payload["warnings"].as_array().expect("warnings array");
        assert!(
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Domain Path"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("x -> -infinity")
                            && message.contains("expression requires x ≥ 0")
                    })
            }),
            "expected nonnegative negative-infinity path/domain warning, got: {warnings:?}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_surfaces_negative_infinity_lower_affine_domain_path_conflict_for_limit(
    ) {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("limit(ln(x + 1), x, -infinity)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["result"], "limit(ln(x + 1), x, -infinity)");

        let warnings = payload["warnings"].as_array().expect("warnings array");
        assert!(
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Domain Path"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("x -> -infinity")
                            && message.contains("expression requires x + 1 > 0")
                    })
            }),
            "expected negative-infinity lower affine path/domain warning, got: {warnings:?}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_surfaces_negative_infinity_lower_nonnegative_affine_domain_path_conflict_for_limit(
    ) {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("limit(sqrt(x - 2), x, -infinity)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["result"], "limit(sqrt(x - 2), x, -infinity)");

        let warnings = payload["warnings"].as_array().expect("warnings array");
        assert!(
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Domain Path"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("x -> -infinity")
                            && message.contains("expression requires x - 2 ≥ 0")
                    })
            }),
            "expected negative-infinity lower affine nonnegative path/domain warning, got: {warnings:?}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_surfaces_positive_infinity_upper_domain_path_conflict_for_limit(
    ) {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("limit(ln(1 - x), x, infinity)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["result"], "limit(ln(1 - x), x, infinity)");

        let warnings = payload["warnings"].as_array().expect("warnings array");
        assert!(
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Domain Path"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("x -> infinity")
                            && message.contains("expression requires 1 - x > 0")
                    })
            }),
            "expected positive-infinity upper-domain path warning, got: {warnings:?}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_surfaces_positive_infinity_upper_nonnegative_path_conflict_for_limit(
    ) {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("limit(sqrt(1 - x), x, infinity)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["result"], "limit(sqrt(1 - x), x, infinity)");

        let warnings = payload["warnings"].as_array().expect("warnings array");
        assert!(
            warnings.iter().any(|warning| {
                warning["rule"] == "Limit Domain Path"
                    && warning["assumption"].as_str().is_some_and(|message| {
                        message.contains("x -> infinity")
                            && message.contains("expression requires 1 - x ≥ 0")
                    })
            }),
            "expected positive-infinity upper nonnegative path warning, got: {warnings:?}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_preserves_solve_operator_and_hides_pure_residual_otherwise_in_latex(
    ) {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("solve(Q = Q0 * 2^(-t/T), t)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);

        let input_latex = payload["input_latex"].as_str().expect("input_latex string");
        assert!(
            input_latex.contains("\\operatorname{solve}"),
            "expected input_latex to preserve the solve operator, got: {input_latex}"
        );
        assert!(
            input_latex.contains("Q ="),
            "expected input_latex to include the equation, got: {input_latex}"
        );
        assert!(
            input_latex.contains(", t") || input_latex.contains("{t}"),
            "expected input_latex to include the solve variable, got: {input_latex}"
        );

        let result_latex = payload["result_latex"]
            .as_str()
            .expect("result_latex string");
        assert!(
            result_latex.contains(r"\text{if }"),
            "expected guarded solve output to keep its condition in latex, got: {result_latex}"
        );
        assert!(
            !result_latex.contains(r"\text{otherwise}"),
            "expected pure residual otherwise branch to be hidden in latex, got: {result_latex}"
        );
        assert!(
            !result_latex.contains(r"\text{Solve: }"),
            "expected pure residual otherwise branch not to leak into latex, got: {result_latex}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_solve_steps_on_suppresses_generic_eval_noise() {
        let mut config = standard_eval_config("solve(Q = Q0 * 2^(-t/T), t)");
        config.steps_mode = cas_api_models::EvalStepsMode::On;

        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            config,
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);

        let has_primary_steps = payload
            .get("steps")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|steps| !steps.is_empty());
        assert!(
            !has_primary_steps,
            "expected solve(...) JSON to suppress generic eval noise when solve_steps already explain the solving path, got: {payload}"
        );

        let solve_steps = payload["solve_steps"]
            .as_array()
            .expect("solve_steps array");
        assert!(
            !solve_steps.is_empty(),
            "expected solve(...) JSON to expose solve_steps, got: {payload}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_preserves_fractional_power_input_style_in_latex() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("x^(1/2)*x^(2/3)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        let input_latex = payload["input_latex"].as_str().expect("input_latex string");
        assert!(
            input_latex.contains("{x}^{\\frac{1}{2}}"),
            "expected input_latex to preserve x^(1/2), got: {input_latex}"
        );
        assert!(
            input_latex.contains("{x}^{\\frac{2}{3}}"),
            "expected input_latex to preserve x^(2/3), got: {input_latex}"
        );
        assert!(
            !input_latex.contains("\\sqrt"),
            "expected input_latex to avoid radical notation for exponential-style input, got: {input_latex}"
        );

        let result_latex = payload["result_latex"]
            .as_str()
            .expect("result_latex string");
        assert!(
            result_latex.contains("{x}^{\\frac{7}{6}}"),
            "expected result_latex to preserve exponential-style result, got: {result_latex}"
        );
        assert!(
            !result_latex.contains("\\sqrt"),
            "expected result_latex to avoid radical notation for exponential-style input, got: {result_latex}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_prefers_radical_result_style_for_mixed_root_input() {
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            standard_eval_config("sqrt(x)*x^(2/3)"),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        let input_latex = payload["input_latex"].as_str().expect("input_latex string");
        assert!(
            input_latex.contains("\\sqrt{x}"),
            "expected input_latex to preserve sqrt notation, got: {input_latex}"
        );
        assert!(
            input_latex.contains("{x}^{\\frac{2}{3}}"),
            "expected input_latex to preserve fractional-power notation, got: {input_latex}"
        );

        let result_latex = payload["result_latex"]
            .as_str()
            .expect("result_latex string");
        assert!(
            result_latex.contains("\\sqrt"),
            "expected mixed root input to prefer radical result style, got: {result_latex}"
        );
        assert!(
            !result_latex.contains("{x}^{\\frac{7}{6}}"),
            "expected mixed root input not to preserve exponential result style, got: {result_latex}"
        );
    }

    #[test]
    fn evaluate_eval_command_pretty_with_session_accepts_solve_system_special_command() {
        let mut config = standard_eval_config("solve_system(x+y=3; x-y=1; x; y)");
        config.auto_store = true;
        let json = crate::eval::evaluate_eval_command_pretty_with_session(
            None,
            config,
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["result"], "{ x = 2, y = 1 }");
        assert!(payload["stored_id"].is_null());
        let input_latex = payload["input_latex"].as_str().expect("input_latex string");
        assert!(
            input_latex.contains("\\operatorname{solve\\_system}"),
            "expected input_latex to preserve the solve_system operator, got: {input_latex}"
        );
        let result_latex = payload["result_latex"]
            .as_str()
            .expect("result_latex string");
        assert!(
            result_latex.contains("\\left\\{") && result_latex.contains("x = 2"),
            "expected result_latex to render the solved system, got: {result_latex}"
        );
    }
}
