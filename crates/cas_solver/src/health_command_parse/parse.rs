use cas_solver_core::health_runtime::HealthCommandInput;

use super::status::parse_health_status_input;

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
        "status" => HealthCommandInput::Status(parse_health_status_input(&parts)),
        _ => HealthCommandInput::Invalid,
    }
}
