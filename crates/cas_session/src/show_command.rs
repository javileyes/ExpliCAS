//! Session-level orchestration for `show` command.

/// Evaluate `show #id` command and return formatted lines for CLI rendering.
pub fn evaluate_show_command_lines(
    state: &mut crate::SessionState,
    engine: &mut cas_solver::Engine,
    line: &str,
) -> Result<Vec<String>, String> {
    let inspection = crate::inspect_history_entry_input(state, engine, line)
        .map_err(|error| crate::format_inspect_history_entry_error_message(&error))?;

    Ok(crate::format_show_history_command_lines_with_context(
        &inspection,
        &engine.simplifier.context,
        |context, expr_info| {
            crate::format_history_eval_metadata_sections(
                context,
                &expr_info.required_conditions,
                &expr_info.domain_warnings,
                &expr_info.blocked_hints,
            )
        },
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn evaluate_show_command_lines_invalid_id_reports_error() {
        let mut engine = cas_solver::Engine::new();
        let mut session = crate::SessionState::new();
        let err = super::evaluate_show_command_lines(&mut session, &mut engine, "show nope")
            .expect_err("expected invalid id error");
        assert!(err.contains("Invalid entry ID"));
    }
}
