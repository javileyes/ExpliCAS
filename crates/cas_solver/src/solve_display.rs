use cas_ast::{Context, ExprId};
use cas_formatter::display_transforms::{DisplayTransformRegistry, ScopeTag, ScopedRenderer};
use cas_formatter::DisplayExpr;

/// Rendering config for `solve` command textual output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveCommandRenderConfig {
    pub show_steps: bool,
    pub show_verbose_substeps: bool,
    pub requires_display: crate::RequiresDisplayLevel,
    pub debug_mode: bool,
    pub hints_enabled: bool,
    pub domain_mode: crate::DomainMode,
    pub check_solutions: bool,
}

/// Visibility flags for solve steps derived from a display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveStepVerbosity {
    pub show_steps: bool,
    pub show_verbose_substeps: bool,
}

/// Map generic display mode (shared across REPL commands) to solve-step visibility.
pub fn solve_step_verbosity_from_display_mode(
    mode: crate::set_command::SetDisplayMode,
) -> SolveStepVerbosity {
    match mode {
        crate::set_command::SetDisplayMode::None => SolveStepVerbosity {
            show_steps: false,
            show_verbose_substeps: false,
        },
        crate::set_command::SetDisplayMode::Succinct
        | crate::set_command::SetDisplayMode::Normal => SolveStepVerbosity {
            show_steps: true,
            show_verbose_substeps: false,
        },
        crate::set_command::SetDisplayMode::Verbose => SolveStepVerbosity {
            show_steps: true,
            show_verbose_substeps: true,
        },
    }
}

/// Render solver steps as textual lines for REPL/UI output.
pub fn format_solve_steps_lines(
    ctx: &Context,
    solve_steps: &[crate::SolveStep],
    output_scopes: &[ScopeTag],
    include_verbose_substeps: bool,
) -> Vec<String> {
    if solve_steps.is_empty() {
        return Vec::new();
    }

    let registry = DisplayTransformRegistry::with_defaults();
    let style = cas_formatter::root_style::StylePreferences::default();
    let renderer = if output_scopes.is_empty() {
        None
    } else {
        Some(ScopedRenderer::new(ctx, output_scopes, &registry, &style))
    };

    let mut lines = Vec::new();
    for (i, step) in solve_steps.iter().enumerate() {
        lines.push(format!("{}. {}", i + 1, step.description));

        let (lhs_str, rhs_str) = if let Some(ref r) = renderer {
            (
                r.render(step.equation_after.lhs),
                r.render(step.equation_after.rhs),
            )
        } else {
            (
                DisplayExpr {
                    context: ctx,
                    id: step.equation_after.lhs,
                }
                .to_string(),
                DisplayExpr {
                    context: ctx,
                    id: step.equation_after.rhs,
                }
                .to_string(),
            )
        };
        lines.push(format!(
            "   -> {} {} {}",
            lhs_str, step.equation_after.op, rhs_str
        ));

        if include_verbose_substeps && !step.substeps.is_empty() {
            for (j, substep) in step.substeps.iter().enumerate() {
                let sub_lhs = DisplayExpr {
                    context: ctx,
                    id: substep.equation_after.lhs,
                }
                .to_string();
                let sub_rhs = DisplayExpr {
                    context: ctx,
                    id: substep.equation_after.rhs,
                }
                .to_string();
                lines.push(format!(
                    "      {}.{}. {}",
                    i + 1,
                    j + 1,
                    substep.description
                ));
                lines.push(format!(
                    "          -> {} {} {}",
                    sub_lhs, substep.equation_after.op, sub_rhs
                ));
            }
        }
    }

    lines
}

/// Render the final solve result line for REPL/UI output.
pub fn format_solve_result_line(
    ctx: &Context,
    result: &crate::EvalResult,
    output_scopes: &[ScopeTag],
) -> String {
    match result {
        crate::EvalResult::SolutionSet(solution_set) => {
            format!("Result: {}", crate::display_solution_set(ctx, solution_set))
        }
        crate::EvalResult::Set(solutions) => {
            let sol_strs: Vec<String> = {
                let registry = DisplayTransformRegistry::with_defaults();
                let style = cas_formatter::root_style::StylePreferences::default();
                let renderer = ScopedRenderer::new(ctx, output_scopes, &registry, &style);
                solutions.iter().map(|id| renderer.render(*id)).collect()
            };
            if sol_strs.is_empty() {
                "Result: No solution".to_string()
            } else {
                format!("Result: {{ {} }}", sol_strs.join(", "))
            }
        }
        _ => format!("Result: {:?}", result),
    }
}

/// Render the final timeline-solve result line for REPL/UI output.
pub fn format_timeline_solve_result_line(
    ctx: &Context,
    solution_set: &cas_ast::SolutionSet,
) -> String {
    format!("Result: {}", crate::display_solution_set(ctx, solution_set))
}

/// Render the message used when timeline solve has no displayable steps.
pub fn format_timeline_solve_no_steps_message(
    ctx: &Context,
    solution_set: &cas_ast::SolutionSet,
) -> String {
    format!(
        "No solving steps to visualize.\n{}",
        format_timeline_solve_result_line(ctx, solution_set)
    )
}

/// Select the expression anchor for `Requires` rendering from eval result payload.
pub fn requires_result_expr_anchor(result: &crate::EvalResult, resolved: ExprId) -> ExprId {
    match result {
        crate::EvalResult::Expr(expr) => *expr,
        crate::EvalResult::Set(values) => *values.first().unwrap_or(&resolved),
        _ => resolved,
    }
}

/// Render full `solve` command output lines from eval payload and rendering flags.
pub fn format_solve_command_eval_lines(
    engine: &mut crate::Engine,
    eval_out: &crate::SolveCommandEvalOutput,
    config: SolveCommandRenderConfig,
) -> Vec<String> {
    use crate::EvalResult;

    let output = &eval_out.output;
    let mut lines: Vec<String> = Vec::new();

    let id_prefix = output
        .stored_id
        .map(|id| format!("#{id}: "))
        .unwrap_or_default();
    lines.push(format!("{}Solving for {}...", id_prefix, eval_out.var));

    lines.extend(crate::format_domain_warning_lines(
        &output.domain_warnings,
        true,
        "⚠ ",
    ));

    let solver_assumption_records =
        crate::assumption_records_from_engine(&output.solver_assumptions);
    if let Some(summary) = crate::format_assumption_records_summary(&solver_assumption_records) {
        lines.push(format!("⚠ Assumptions: {summary}"));
    }

    if config.show_steps && !output.solve_steps.is_empty() {
        lines.push("Steps:".to_string());
        lines.extend(format_solve_steps_lines(
            &engine.simplifier.context,
            &output.solve_steps,
            &output.output_scopes,
            config.show_verbose_substeps,
        ));
    }

    lines.push(format_solve_result_line(
        &engine.simplifier.context,
        &output.result,
        &output.output_scopes,
    ));

    let result_expr_id = requires_result_expr_anchor(&output.result, output.resolved);
    let requires_lines = crate::format_diagnostics_requires_lines(
        &mut engine.simplifier.context,
        &output.diagnostics,
        Some(result_expr_id),
        config.requires_display,
        config.debug_mode,
    );
    if !requires_lines.is_empty() {
        lines.push("ℹ️ Requires:".to_string());
        lines.extend(requires_lines);
    }

    if config.check_solutions {
        if let EvalResult::SolutionSet(ref solution_set) = output.result {
            if let Some(ref eq) = eval_out.original_equation {
                let verify_result = crate::verify_solution_set(
                    &mut engine.simplifier,
                    eq,
                    &eval_out.var,
                    solution_set,
                );
                lines.extend(crate::format_verify_summary_lines(
                    &engine.simplifier.context,
                    &eval_out.var,
                    &verify_result,
                    "  ",
                ));
            }
        }
    }

    let hints = crate::take_blocked_hints();
    lines.extend(crate::format_solve_assumption_and_blocked_sections(
        &engine.simplifier.context,
        &solver_assumption_records,
        &hints,
        crate::SolveAssumptionSectionConfig {
            debug_mode: config.debug_mode,
            hints_enabled: config.hints_enabled,
            domain_mode: config.domain_mode,
        },
    ));

    lines
}

#[cfg(test)]
mod tests {
    use super::{
        format_solve_command_eval_lines, format_solve_result_line,
        format_timeline_solve_no_steps_message, format_timeline_solve_result_line,
        requires_result_expr_anchor, solve_step_verbosity_from_display_mode,
        SolveCommandRenderConfig, SolveStepVerbosity,
    };

    #[test]
    fn format_solve_result_line_handles_empty_legacy_set() {
        let ctx = cas_ast::Context::new();
        let shown = format_solve_result_line(&ctx, &crate::EvalResult::Set(Vec::new()), &[]);
        assert_eq!(shown, "Result: No solution");
    }

    #[test]
    fn requires_result_expr_anchor_prefers_expr_result() {
        let expr = cas_ast::ExprId::from_raw(42);
        let resolved = cas_ast::ExprId::from_raw(7);
        let anchor = requires_result_expr_anchor(&crate::EvalResult::Expr(expr), resolved);
        assert_eq!(anchor, expr);
    }

    #[test]
    fn format_timeline_solve_result_line_renders_result_prefix() {
        let ctx = cas_ast::Context::new();
        let line = format_timeline_solve_result_line(&ctx, &cas_ast::SolutionSet::Empty);
        assert!(line.starts_with("Result: "));
    }

    #[test]
    fn format_timeline_solve_no_steps_message_includes_header() {
        let ctx = cas_ast::Context::new();
        let msg = format_timeline_solve_no_steps_message(&ctx, &cas_ast::SolutionSet::Empty);
        assert!(msg.contains("No solving steps to visualize."));
    }

    #[test]
    fn format_solve_command_eval_lines_contains_header_and_result() {
        let mut engine = crate::Engine::new();
        let mut session = cas_session::SessionState::new();
        let eval_out =
            crate::evaluate_solve_command_input(&mut engine, &mut session, "x + 2 = 5, x", true)
                .expect("solve");

        let lines = format_solve_command_eval_lines(
            &mut engine,
            &eval_out,
            SolveCommandRenderConfig {
                show_steps: false,
                show_verbose_substeps: false,
                requires_display: crate::RequiresDisplayLevel::Essential,
                debug_mode: false,
                hints_enabled: true,
                domain_mode: crate::DomainMode::Generic,
                check_solutions: false,
            },
        );

        assert!(lines.iter().any(|line| line.contains("Solving for x")));
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }

    #[test]
    fn solve_step_verbosity_from_display_mode_maps_all_variants() {
        assert_eq!(
            solve_step_verbosity_from_display_mode(crate::set_command::SetDisplayMode::None),
            SolveStepVerbosity {
                show_steps: false,
                show_verbose_substeps: false
            }
        );
        assert_eq!(
            solve_step_verbosity_from_display_mode(crate::set_command::SetDisplayMode::Succinct),
            SolveStepVerbosity {
                show_steps: true,
                show_verbose_substeps: false
            }
        );
        assert_eq!(
            solve_step_verbosity_from_display_mode(crate::set_command::SetDisplayMode::Normal),
            SolveStepVerbosity {
                show_steps: true,
                show_verbose_substeps: false
            }
        );
        assert_eq!(
            solve_step_verbosity_from_display_mode(crate::set_command::SetDisplayMode::Verbose),
            SolveStepVerbosity {
                show_steps: true,
                show_verbose_substeps: true
            }
        );
    }
}
