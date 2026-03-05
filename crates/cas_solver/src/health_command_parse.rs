use crate::health_command_format::health_usage_message;
use crate::health_command_types::{HealthCommandInput, HealthStatusInput};

/// Parse raw `health ...` command input.
pub fn parse_health_command_input(line: &str) -> HealthCommandInput {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() == 1 {
        return HealthCommandInput::ShowLast;
    }

    match parts[1] {
        "on" | "enable" => HealthCommandInput::SetEnabled { enabled: true },
        "off" | "disable" => HealthCommandInput::SetEnabled { enabled: false },
        "reset" | "clear" => HealthCommandInput::Clear,
        "status" => {
            let opts: Vec<&str> = parts.iter().skip(2).copied().collect();
            let list_only = opts.contains(&"--list") || opts.contains(&"-l");
            let mut category = None;
            let mut category_missing_arg = false;

            if let Some(idx) = opts.iter().position(|&x| x == "--category" || x == "-c") {
                if let Some(cat) = opts.get(idx + 1) {
                    category = Some((*cat).to_string());
                } else {
                    category_missing_arg = true;
                }
            }

            HealthCommandInput::Status(HealthStatusInput {
                list_only,
                category,
                category_missing_arg,
            })
        }
        _ => HealthCommandInput::Invalid,
    }
}

/// Parse and validate `health ...` command input.
///
/// Returns a preformatted usage message when command input is invalid.
pub fn evaluate_health_command_input(line: &str) -> Result<HealthCommandInput, String> {
    match parse_health_command_input(line) {
        HealthCommandInput::Invalid => Err(health_usage_message()),
        parsed => Ok(parsed),
    }
}
