use crate::solve_command_render_result::{format_solve_result_line, requires_result_expr_anchor};
use crate::solve_command_render_steps::format_solve_steps_lines;
use crate::solve_command_render_types::SolveCommandRenderConfig;

pub fn format_solve_command_eval_lines(
    simplifier: &mut cas_solver::Simplifier,
    eval_out: &crate::SolveCommandEvalOutput,
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
