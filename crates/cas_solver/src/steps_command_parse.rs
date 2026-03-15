use cas_api_models::{EvalStepsMode, StepsCommandInput, StepsDisplayMode};

/// Parse raw `steps ...` command input.
pub fn parse_steps_command_input(line: &str) -> StepsCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1) {
        None => StepsCommandInput::ShowCurrent,
        Some(&"on") => StepsCommandInput::SetCollectionMode(EvalStepsMode::On),
        Some(&"off") => StepsCommandInput::SetCollectionMode(EvalStepsMode::Off),
        Some(&"compact") => StepsCommandInput::SetCollectionMode(EvalStepsMode::Compact),
        Some(&"verbose") => StepsCommandInput::SetDisplayMode(StepsDisplayMode::Verbose),
        Some(&"succinct") => StepsCommandInput::SetDisplayMode(StepsDisplayMode::Succinct),
        Some(&"normal") => StepsCommandInput::SetDisplayMode(StepsDisplayMode::Normal),
        Some(&"none") => StepsCommandInput::SetDisplayMode(StepsDisplayMode::None),
        Some(other) => StepsCommandInput::UnknownMode((*other).to_string()),
    }
}
