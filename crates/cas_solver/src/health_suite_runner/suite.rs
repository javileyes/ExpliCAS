mod execute;
mod select;

use crate::health_suite_catalog::default_suite;
use crate::Simplifier;
use cas_solver_core::health_category::Category;
use cas_solver_core::health_suite_models::HealthCaseResult;

/// Run entire suite and return all results
#[allow(dead_code)]
pub fn run_suite(simplifier: &mut Simplifier) -> Vec<HealthCaseResult> {
    execute::run_selected_suite(&default_suite(), simplifier)
}

/// Run suite filtered by category
pub fn run_suite_filtered(
    simplifier: &mut Simplifier,
    filter: Option<Category>,
) -> Vec<HealthCaseResult> {
    let filtered = select::select_suite(default_suite(), filter);
    execute::run_selected_suite(&filtered, simplifier)
}
