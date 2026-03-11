pub(super) fn push_solve_steps_section(
    lines: &mut Vec<String>,
    simplifier: &crate::Simplifier,
    output: &crate::EvalOutputView,
    config: crate::SolveCommandRenderConfig,
) {
    if config.show_steps && !output.solve_steps.is_empty() {
        lines.push("Steps:".to_string());
        lines.extend(crate::format_solve_steps_lines(
            &simplifier.context,
            &output.solve_steps,
            &output.output_scopes,
            config.show_verbose_substeps,
        ));
    }
}

pub(super) fn push_requires_section(
    lines: &mut Vec<String>,
    simplifier: &mut crate::Simplifier,
    output: &crate::EvalOutputView,
    config: crate::SolveCommandRenderConfig,
) {
    let result_expr_id = crate::requires_result_expr_anchor(&output.result, output.resolved);
    let requires_lines = crate::assumption_format::format_diagnostics_requires_lines(
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
}

pub(super) fn push_assumption_and_blocked_sections(
    lines: &mut Vec<String>,
    simplifier: &crate::Simplifier,
    output: &crate::EvalOutputView,
    config: crate::SolveCommandRenderConfig,
) {
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
}
