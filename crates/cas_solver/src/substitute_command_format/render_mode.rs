use cas_solver_core::substitute_command_types::SubstituteRenderMode;

/// Convert REPL display mode into substitute render mode.
pub fn substitute_render_mode_from_display_mode(
    mode: crate::SetDisplayMode,
) -> SubstituteRenderMode {
    match mode {
        crate::SetDisplayMode::None => SubstituteRenderMode::None,
        crate::SetDisplayMode::Succinct => SubstituteRenderMode::Succinct,
        crate::SetDisplayMode::Normal => SubstituteRenderMode::Normal,
        crate::SetDisplayMode::Verbose => SubstituteRenderMode::Verbose,
    }
}
