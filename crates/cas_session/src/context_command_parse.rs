use crate::context_command_types::ContextCommandInput;

/// Parse raw `context ...` command input.
pub fn parse_context_command_input(line: &str) -> ContextCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1) {
        None => ContextCommandInput::ShowCurrent,
        Some(&"auto") => ContextCommandInput::SetMode(cas_solver::ContextMode::Auto),
        Some(&"standard") => ContextCommandInput::SetMode(cas_solver::ContextMode::Standard),
        Some(&"solve") => ContextCommandInput::SetMode(cas_solver::ContextMode::Solve),
        Some(&"integrate") => ContextCommandInput::SetMode(cas_solver::ContextMode::IntegratePrep),
        Some(other) => ContextCommandInput::UnknownMode((*other).to_string()),
    }
}
