mod case;
mod cycle;
mod suite;

use crate::health_category::Category;
use crate::health_suite_models::HealthCaseResult;
use crate::Simplifier;

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
