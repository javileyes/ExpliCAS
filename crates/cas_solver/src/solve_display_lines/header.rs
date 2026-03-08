pub(super) fn push_solve_header_lines(
    lines: &mut Vec<String>,
    output: &crate::EvalOutputView,
    var: &str,
) {
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
}
