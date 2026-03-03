/// Format current context status line.
pub fn format_context_current_message(mode: cas_solver::ContextMode) -> String {
    let context = match mode {
        cas_solver::ContextMode::Auto => "auto",
        cas_solver::ContextMode::Standard => "standard",
        cas_solver::ContextMode::Solve => "solve",
        cas_solver::ContextMode::IntegratePrep => "integrate",
    };
    format!(
        "Current context: {}\n  (use 'context auto|standard|solve|integrate' to change)",
        context
    )
}

/// Format confirmation message after setting context.
pub fn format_context_set_message(mode: cas_solver::ContextMode) -> String {
    match mode {
        cas_solver::ContextMode::Auto => "Context: auto (infers from expression)".to_string(),
        cas_solver::ContextMode::Standard => {
            "Context: standard (safe simplification only)".to_string()
        }
        cas_solver::ContextMode::Solve => {
            "Context: solve (preserves solver-friendly forms)".to_string()
        }
        cas_solver::ContextMode::IntegratePrep => {
            "Context: integrate-prep\n  ⚠️ Enables transforms for integration (telescoping, product→sum)".to_string()
        }
    }
}

/// Format unknown-context error message.
pub fn format_context_unknown_message(mode: &str) -> String {
    format!(
        "Unknown context: '{}'\nUsage: context [auto | standard | solve | integrate]",
        mode
    )
}
