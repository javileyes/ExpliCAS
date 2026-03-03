use crate::steps_command_types::{StepsCommandInput, StepsDisplayMode};

/// Parse raw `steps ...` command input.
pub fn parse_steps_command_input(line: &str) -> StepsCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1) {
        None => StepsCommandInput::ShowCurrent,
        Some(&"on") => StepsCommandInput::SetCollectionMode(cas_solver::StepsMode::On),
        Some(&"off") => StepsCommandInput::SetCollectionMode(cas_solver::StepsMode::Off),
        Some(&"compact") => StepsCommandInput::SetCollectionMode(cas_solver::StepsMode::Compact),
        Some(&"verbose") => StepsCommandInput::SetDisplayMode(StepsDisplayMode::Verbose),
        Some(&"succinct") => StepsCommandInput::SetDisplayMode(StepsDisplayMode::Succinct),
        Some(&"normal") => StepsCommandInput::SetDisplayMode(StepsDisplayMode::Normal),
        Some(&"none") => StepsCommandInput::SetDisplayMode(StepsDisplayMode::None),
        Some(other) => StepsCommandInput::UnknownMode((*other).to_string()),
    }
}
