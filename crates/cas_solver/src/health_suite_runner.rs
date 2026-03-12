mod case;
mod cycle;
mod suite;

use crate::Simplifier;
use cas_solver_core::health_category::Category;
use cas_solver_core::health_suite_models::HealthCaseResult;

#[allow(dead_code)]
pub fn run_suite(simplifier: &mut Simplifier) -> Vec<HealthCaseResult> {
    self::suite::run_suite(simplifier)
}

pub fn run_suite_filtered(
    simplifier: &mut Simplifier,
    filter: Option<Category>,
) -> Vec<HealthCaseResult> {
    self::suite::run_suite_filtered(simplifier, filter)
}
