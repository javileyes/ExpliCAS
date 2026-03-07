use crate::health_command_format::health_usage_message;
use crate::health_command_types::HealthCommandInput;

use super::parse::parse_health_command_input;

/// Parse and validate `health ...` command input.
///
/// Returns a preformatted usage message when command input is invalid.
pub fn evaluate_health_command_input(line: &str) -> Result<HealthCommandInput, String> {
    match parse_health_command_input(line) {
        HealthCommandInput::Invalid => Err(health_usage_message()),
        parsed => Ok(parsed),
    }
}
