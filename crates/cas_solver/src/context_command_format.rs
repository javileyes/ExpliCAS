/// Format current context status line.
pub fn format_context_current_message(mode: crate::ContextMode) -> String {
    let context = match mode {
        crate::ContextMode::Auto => "auto",
        crate::ContextMode::Standard => "standard",
        crate::ContextMode::Solve => "solve",
        crate::ContextMode::IntegratePrep => "integrate",
    };
    format!(
        "Current context: {}\n  (use 'context auto|standard|solve|integrate' to change)",
        context
    )
}

/// Format confirmation message after setting context.
pub fn format_context_set_message(mode: crate::ContextMode) -> String {
    match mode {
        crate::ContextMode::Auto => "Context: auto (infers from expression)".to_string(),
        crate::ContextMode::Standard => {
            "Context: standard (safe simplification only)".to_string()
        }
        crate::ContextMode::Solve => {
            "Context: solve (preserves solver-friendly forms)".to_string()
        }
        crate::ContextMode::IntegratePrep => {
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
