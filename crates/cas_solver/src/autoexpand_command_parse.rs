use cas_api_models::{AutoexpandCommandInput, EvalExpandPolicy};

/// Parse raw `autoexpand ...` command input.
pub fn parse_autoexpand_command_input(line: &str) -> AutoexpandCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1) {
        None => AutoexpandCommandInput::ShowCurrent,
        Some(&"on") => AutoexpandCommandInput::SetPolicy(EvalExpandPolicy::Auto),
        Some(&"off") => AutoexpandCommandInput::SetPolicy(EvalExpandPolicy::Off),
        Some(other) => AutoexpandCommandInput::UnknownMode((*other).to_string()),
    }
}
