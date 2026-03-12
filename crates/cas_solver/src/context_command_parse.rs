use cas_solver_core::context_command_types::ContextCommandInput;

/// Parse raw `context ...` command input.
pub fn parse_context_command_input(line: &str) -> ContextCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1) {
        None => ContextCommandInput::ShowCurrent,
        Some(&"auto") => ContextCommandInput::SetMode(crate::ContextMode::Auto),
        Some(&"standard") => ContextCommandInput::SetMode(crate::ContextMode::Standard),
        Some(&"solve") => ContextCommandInput::SetMode(crate::ContextMode::Solve),
        Some(&"integrate") => ContextCommandInput::SetMode(crate::ContextMode::IntegratePrep),
        Some(other) => ContextCommandInput::UnknownMode((*other).to_string()),
    }
}
