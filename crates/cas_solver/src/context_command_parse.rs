use cas_api_models::{ContextCommandInput, EvalContextMode};

/// Parse raw `context ...` command input.
pub fn parse_context_command_input(line: &str) -> ContextCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1) {
        None => ContextCommandInput::ShowCurrent,
        Some(&"auto") => ContextCommandInput::SetMode(EvalContextMode::Auto),
        Some(&"standard") => ContextCommandInput::SetMode(EvalContextMode::Standard),
        Some(&"solve") => ContextCommandInput::SetMode(EvalContextMode::Solve),
        Some(&"integrate") => ContextCommandInput::SetMode(EvalContextMode::Integrate),
        Some(other) => ContextCommandInput::UnknownMode((*other).to_string()),
    }
}
