mod list;
mod run;

use crate::health_command_types::HealthStatusInput;

pub fn evaluate_health_status_lines(
    simplifier: &mut crate::Simplifier,
    status: &HealthStatusInput,
) -> Result<Vec<String>, String> {
    if status.list_only {
        return Ok(list::list_health_status_lines());
    }

    run::run_health_status_lines(simplifier, status)
}
