pub(super) fn push_solution_verification_lines(
    lines: &mut Vec<String>,
    simplifier: &mut crate::Simplifier,
    original_equation: Option<&cas_ast::Equation>,
    output: &crate::EvalOutputView,
    config: crate::SolveCommandRenderConfig,
    var: &str,
) {
    if !config.check_solutions {
        return;
    }

    if let crate::EvalResult::SolutionSet(solution_set) = &output.result {
        if let Some(eq) = original_equation {
            let verify_result = crate::api::verify_solution_set(simplifier, eq, var, solution_set);
            lines.extend(
                crate::solve_verify_display::format_verify_summary_lines_with_hints(
                    &simplifier.context,
                    var,
                    &verify_result,
                    "  ",
                    config.hints_enabled,
                ),
            );
            if config.hints_enabled && (config.debug_mode || config.show_verbose_substeps) {
                push_suppressed_counterexample_hint_notes(
                    lines,
                    &simplifier.context,
                    var,
                    &verify_result,
                    "  ",
                );
            }
        }
    }
}

fn push_suppressed_counterexample_hint_notes(
    lines: &mut Vec<String>,
    ctx: &cas_ast::Context,
    var: &str,
    verify_result: &crate::VerifyResult,
    detail_prefix: &str,
) {
    for (sol_id, status) in &verify_result.solutions {
        if let crate::VerifyStatus::Unverifiable {
            residual,
            counterexample_hint,
            ..
        } = status
        {
            if counterexample_hint.is_none() {
                if let Some(note) =
                    cas_solver_core::verification_runtime_flow::counterexample_hint_suppression_note(
                        ctx, *residual,
                    )
                {
                    let sol_str = cas_formatter::render_expr(ctx, *sol_id);
                    lines.push(format!("{detail_prefix}ℹ {var} = {sol_str}: {note}"));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    fn test_render_config() -> crate::SolveCommandRenderConfig {
        crate::SolveCommandRenderConfig {
            show_steps: false,
            show_verbose_substeps: false,
            requires_display: crate::RequiresDisplayLevel::Essential,
            debug_mode: false,
            hints_enabled: true,
            domain_mode: crate::DomainMode::Generic,
            check_solutions: true,
        }
    }

    fn verbose_render_config() -> crate::SolveCommandRenderConfig {
        crate::SolveCommandRenderConfig {
            show_steps: true,
            show_verbose_substeps: true,
            requires_display: crate::RequiresDisplayLevel::Essential,
            debug_mode: false,
            hints_enabled: true,
            domain_mode: crate::DomainMode::Generic,
            check_solutions: true,
        }
    }

    fn debug_render_config() -> crate::SolveCommandRenderConfig {
        crate::SolveCommandRenderConfig {
            show_steps: false,
            show_verbose_substeps: false,
            requires_display: crate::RequiresDisplayLevel::Essential,
            debug_mode: true,
            hints_enabled: true,
            domain_mode: crate::DomainMode::Generic,
            check_solutions: true,
        }
    }

    fn verbose_no_hints_render_config() -> crate::SolveCommandRenderConfig {
        crate::SolveCommandRenderConfig {
            show_steps: true,
            show_verbose_substeps: true,
            requires_display: crate::RequiresDisplayLevel::Essential,
            debug_mode: false,
            hints_enabled: false,
            domain_mode: crate::DomainMode::Generic,
            check_solutions: true,
        }
    }

    fn minimal_output_view(
        expr: cas_ast::ExprId,
        result: crate::EvalResult,
    ) -> crate::EvalOutputView {
        crate::EvalOutputView {
            stored_id: None,
            parsed: expr,
            resolved: expr,
            result,
            steps: crate::DisplayEvalSteps::default(),
            solve_steps: vec![],
            output_scopes: vec![],
            diagnostics: crate::Diagnostics::default(),
            required_conditions: vec![],
            domain_warnings: vec![],
            blocked_hints: vec![],
            solver_assumptions: vec![],
        }
    }

    #[test]
    fn push_solution_verification_lines_surfaces_counterexample_hint_for_failed_discrete_solution()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("a*x", &mut simplifier.context).unwrap();
        let rhs = parse("1", &mut simplifier.context).unwrap();
        let wrong_solution = parse("1", &mut simplifier.context).unwrap();
        let eq = cas_ast::Equation {
            lhs,
            rhs,
            op: cas_ast::RelOp::Eq,
        };
        let output = minimal_output_view(
            lhs,
            crate::EvalResult::SolutionSet(cas_ast::SolutionSet::Discrete(vec![wrong_solution])),
        );
        let mut lines = Vec::new();

        push_solution_verification_lines(
            &mut lines,
            &mut simplifier,
            Some(&eq),
            &output,
            test_render_config(),
            "x",
        );

        assert_eq!(
            lines,
            vec![
                "⚠ No solutions could be verified".to_string(),
                "  ⚠ x = 1: residual: a - 1".to_string(),
                "  ↳ counterexample hint: a=0 gives residual -1".to_string(),
            ]
        );
    }

    #[test]
    fn push_solution_verification_lines_suppresses_counterexample_hint_for_log_sensitive_residual()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("ln(a*x)", &mut simplifier.context).unwrap();
        let rhs = parse("1", &mut simplifier.context).unwrap();
        let wrong_solution = parse("1", &mut simplifier.context).unwrap();
        let eq = cas_ast::Equation {
            lhs,
            rhs,
            op: cas_ast::RelOp::Eq,
        };
        let output = minimal_output_view(
            lhs,
            crate::EvalResult::SolutionSet(cas_ast::SolutionSet::Discrete(vec![wrong_solution])),
        );
        let mut lines = Vec::new();

        push_solution_verification_lines(
            &mut lines,
            &mut simplifier,
            Some(&eq),
            &output,
            test_render_config(),
            "x",
        );

        assert_eq!(
            lines,
            vec![
                "⚠ No solutions could be verified".to_string(),
                "  ⚠ x = 1: residual: ln(a) - 1".to_string(),
            ]
        );
    }

    #[test]
    fn push_solution_verification_lines_surfaces_suppressed_hint_note_in_verbose_mode() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("ln(a*x)", &mut simplifier.context).unwrap();
        let rhs = parse("1", &mut simplifier.context).unwrap();
        let wrong_solution = parse("1", &mut simplifier.context).unwrap();
        let eq = cas_ast::Equation {
            lhs,
            rhs,
            op: cas_ast::RelOp::Eq,
        };
        let output = minimal_output_view(
            lhs,
            crate::EvalResult::SolutionSet(cas_ast::SolutionSet::Discrete(vec![wrong_solution])),
        );
        let mut lines = Vec::new();

        push_solution_verification_lines(
            &mut lines,
            &mut simplifier,
            Some(&eq),
            &output,
            verbose_render_config(),
            "x",
        );

        assert_eq!(
            lines,
            vec![
                "⚠ No solutions could be verified".to_string(),
                "  ⚠ x = 1: residual: ln(a) - 1".to_string(),
                "  ℹ x = 1: counterexample hint omitted for branch-sensitive residual (`log/ln`)"
                    .to_string(),
            ]
        );
    }

    #[test]
    fn push_solution_verification_lines_hides_counterexample_hint_when_hints_are_disabled() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("a*x", &mut simplifier.context).unwrap();
        let rhs = parse("1", &mut simplifier.context).unwrap();
        let wrong_solution = parse("1", &mut simplifier.context).unwrap();
        let eq = cas_ast::Equation {
            lhs,
            rhs,
            op: cas_ast::RelOp::Eq,
        };
        let output = minimal_output_view(
            lhs,
            crate::EvalResult::SolutionSet(cas_ast::SolutionSet::Discrete(vec![wrong_solution])),
        );
        let mut lines = Vec::new();

        push_solution_verification_lines(
            &mut lines,
            &mut simplifier,
            Some(&eq),
            &output,
            verbose_no_hints_render_config(),
            "x",
        );

        assert_eq!(
            lines,
            vec![
                "⚠ No solutions could be verified".to_string(),
                "  ⚠ x = 1: residual: a - 1".to_string(),
            ]
        );
    }

    #[test]
    fn push_solution_verification_lines_surfaces_sqrt_suppressed_hint_note_in_debug_mode() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("sqrt(a*x)", &mut simplifier.context).unwrap();
        let rhs = parse("1", &mut simplifier.context).unwrap();
        let wrong_solution = parse("1", &mut simplifier.context).unwrap();
        let eq = cas_ast::Equation {
            lhs,
            rhs,
            op: cas_ast::RelOp::Eq,
        };
        let output = minimal_output_view(
            lhs,
            crate::EvalResult::SolutionSet(cas_ast::SolutionSet::Discrete(vec![wrong_solution])),
        );
        let mut lines = Vec::new();

        push_solution_verification_lines(
            &mut lines,
            &mut simplifier,
            Some(&eq),
            &output,
            debug_render_config(),
            "x",
        );

        assert_eq!(
            lines,
            vec![
                "⚠ No solutions could be verified".to_string(),
                "  ⚠ x = 1: residual: a^(1/2) - 1".to_string(),
                "  ℹ x = 1: counterexample hint omitted for branch-sensitive residual (`sqrt`)"
                    .to_string(),
            ]
        );
    }

    #[test]
    fn push_solution_verification_lines_surfaces_inverse_trig_suppressed_hint_note_in_verbose_mode()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("arcsin(a*x)", &mut simplifier.context).unwrap();
        let rhs = parse("1", &mut simplifier.context).unwrap();
        let wrong_solution = parse("1", &mut simplifier.context).unwrap();
        let eq = cas_ast::Equation {
            lhs,
            rhs,
            op: cas_ast::RelOp::Eq,
        };
        let output = minimal_output_view(
            lhs,
            crate::EvalResult::SolutionSet(cas_ast::SolutionSet::Discrete(vec![wrong_solution])),
        );
        let mut lines = Vec::new();

        push_solution_verification_lines(
            &mut lines,
            &mut simplifier,
            Some(&eq),
            &output,
            verbose_render_config(),
            "x",
        );

        assert_eq!(
            lines,
            vec![
                "⚠ No solutions could be verified".to_string(),
                "  ⚠ x = 1: residual: arcsin(a) - 1".to_string(),
                "  ℹ x = 1: counterexample hint omitted for branch-sensitive residual (`inverse trig`)"
                    .to_string(),
            ]
        );
    }

    #[test]
    fn push_solution_verification_lines_hides_suppressed_hint_note_when_hints_are_disabled() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("ln(a*x)", &mut simplifier.context).unwrap();
        let rhs = parse("1", &mut simplifier.context).unwrap();
        let wrong_solution = parse("1", &mut simplifier.context).unwrap();
        let eq = cas_ast::Equation {
            lhs,
            rhs,
            op: cas_ast::RelOp::Eq,
        };
        let output = minimal_output_view(
            lhs,
            crate::EvalResult::SolutionSet(cas_ast::SolutionSet::Discrete(vec![wrong_solution])),
        );
        let mut lines = Vec::new();

        push_solution_verification_lines(
            &mut lines,
            &mut simplifier,
            Some(&eq),
            &output,
            verbose_no_hints_render_config(),
            "x",
        );

        assert_eq!(
            lines,
            vec![
                "⚠ No solutions could be verified".to_string(),
                "  ⚠ x = 1: residual: ln(a) - 1".to_string(),
            ]
        );
    }
}
