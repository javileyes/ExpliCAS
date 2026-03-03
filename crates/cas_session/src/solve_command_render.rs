//! Session-level rendering of solve command evaluation output.

use cas_ast::{Context, ExprId};
use cas_formatter::display_transforms::{DisplayTransformRegistry, ScopeTag, ScopedRenderer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveCommandRenderConfig {
    pub show_steps: bool,
    pub show_verbose_substeps: bool,
    pub requires_display: cas_solver::RequiresDisplayLevel,
    pub debug_mode: bool,
    pub hints_enabled: bool,
    pub domain_mode: cas_solver::DomainMode,
    pub check_solutions: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SolveStepVerbosity {
    show_steps: bool,
    show_verbose_substeps: bool,
}

fn solve_step_verbosity_from_display_mode(mode: crate::SetDisplayMode) -> SolveStepVerbosity {
    match mode {
        crate::SetDisplayMode::None => SolveStepVerbosity {
            show_steps: false,
            show_verbose_substeps: false,
        },
        crate::SetDisplayMode::Succinct | crate::SetDisplayMode::Normal => SolveStepVerbosity {
            show_steps: true,
            show_verbose_substeps: false,
        },
        crate::SetDisplayMode::Verbose => SolveStepVerbosity {
            show_steps: true,
            show_verbose_substeps: true,
        },
    }
}

pub fn solve_render_config_from_eval_options(
    options: &cas_solver::EvalOptions,
    display_mode: crate::SetDisplayMode,
    debug_mode: bool,
) -> SolveCommandRenderConfig {
    let step_verbosity = solve_step_verbosity_from_display_mode(display_mode);
    SolveCommandRenderConfig {
        show_steps: step_verbosity.show_steps,
        show_verbose_substeps: step_verbosity.show_verbose_substeps,
        requires_display: options.requires_display,
        debug_mode,
        hints_enabled: options.hints_enabled,
        domain_mode: options.shared.semantics.domain_mode,
        check_solutions: options.check_solutions,
    }
}

fn format_solve_steps_lines(
    ctx: &Context,
    solve_steps: &[cas_solver::SolveStep],
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
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: step.equation_after.lhs,
                }
                .to_string(),
                cas_formatter::DisplayExpr {
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
                let sub_lhs = cas_formatter::DisplayExpr {
                    context: ctx,
                    id: substep.equation_after.lhs,
                }
                .to_string();
                let sub_rhs = cas_formatter::DisplayExpr {
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

fn format_solve_result_line(
    ctx: &Context,
    result: &cas_solver::EvalResult,
    output_scopes: &[ScopeTag],
) -> String {
    match result {
        cas_solver::EvalResult::SolutionSet(solution_set) => {
            format!(
                "Result: {}",
                cas_solver::display_solution_set(ctx, solution_set)
            )
        }
        cas_solver::EvalResult::Set(solutions) => {
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

fn requires_result_expr_anchor(result: &cas_solver::EvalResult, resolved: ExprId) -> ExprId {
    match result {
        cas_solver::EvalResult::Expr(expr) => *expr,
        cas_solver::EvalResult::Set(values) => *values.first().unwrap_or(&resolved),
        _ => resolved,
    }
}

pub fn format_solve_command_eval_lines(
    simplifier: &mut cas_solver::Simplifier,
    eval_out: &cas_solver::SolveCommandEvalOutput,
    config: SolveCommandRenderConfig,
) -> Vec<String> {
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

    if let Some(summary) = crate::format_assumption_records_summary(&output.solver_assumptions) {
        lines.push(format!("⚠ Assumptions: {summary}"));
    }

    if config.show_steps && !output.solve_steps.is_empty() {
        lines.push("Steps:".to_string());
        lines.extend(format_solve_steps_lines(
            &simplifier.context,
            &output.solve_steps,
            &output.output_scopes,
            config.show_verbose_substeps,
        ));
    }

    lines.push(format_solve_result_line(
        &simplifier.context,
        &output.result,
        &output.output_scopes,
    ));

    let result_expr_id = requires_result_expr_anchor(&output.result, output.resolved);
    let requires_lines = crate::format_diagnostics_requires_lines(
        &mut simplifier.context,
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
        if let cas_solver::EvalResult::SolutionSet(solution_set) = &output.result {
            if let Some(ref eq) = eval_out.original_equation {
                let verify_result =
                    cas_solver::verify_solution_set(simplifier, eq, &eval_out.var, solution_set);
                lines.extend(crate::format_verify_summary_lines(
                    &simplifier.context,
                    &eval_out.var,
                    &verify_result,
                    "  ",
                ));
            }
        }
    }

    let hints = cas_solver::take_blocked_hints();
    lines.extend(crate::format_solve_assumption_and_blocked_sections(
        &simplifier.context,
        &output.solver_assumptions,
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
    #[test]
    fn solve_render_config_from_eval_options_maps_modes() {
        let options = cas_solver::EvalOptions::default();
        let cfg = super::solve_render_config_from_eval_options(
            &options,
            crate::SetDisplayMode::Verbose,
            true,
        );
        assert!(cfg.show_steps);
        assert!(cfg.show_verbose_substeps);
        assert!(cfg.debug_mode);
    }
}
