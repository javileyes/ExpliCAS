mod list;
mod run;

use cas_solver_core::health_runtime::HealthStatusInput;

pub fn evaluate_health_status_lines(
    simplifier: &mut crate::Simplifier,
    status: &HealthStatusInput,
) -> Result<Vec<String>, String> {
    if status.list_only {
        return Ok(list::list_health_status_lines());
    }

    run::run_health_status_lines(simplifier, status)
}
