use crate::{format_solve_result_line, format_solve_steps_lines, requires_result_expr_anchor};

pub fn format_solve_command_eval_lines(
    simplifier: &mut crate::Simplifier,
    var: &str,
    original_equation: Option<&cas_ast::Equation>,
    output: &crate::EvalOutputView,
    config: crate::SolveCommandRenderConfig,
) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();

    let id_prefix = output
        .stored_id
        .map(|id| format!("#{id}: "))
        .unwrap_or_default();
    lines.push(format!("{id_prefix}Solving for {var}..."));

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
        if let crate::EvalResult::SolutionSet(solution_set) = &output.result {
            if let Some(eq) = original_equation {
                let verify_result = crate::verify_solution_set(simplifier, eq, var, solution_set);
                lines.extend(crate::format_verify_summary_lines(
                    &simplifier.context,
                    var,
                    &verify_result,
                    "  ",
                ));
            }
        }
    }

    let hints = crate::take_blocked_hints();
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
