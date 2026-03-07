use crate::health_suite_catalog::default_suite;
use crate::health_suite_types::{Category, HealthCaseResult};
use crate::Simplifier;

use super::case::run_case;

/// Run entire suite and return all results
#[allow(dead_code)]
pub fn run_suite(simplifier: &mut Simplifier) -> Vec<HealthCaseResult> {
    let suite = default_suite();
    suite
        .iter()
        .map(|case| run_case(case, simplifier))
        .collect()
}

/// Run suite filtered by category
pub fn run_suite_filtered(
    simplifier: &mut Simplifier,
    filter: Option<Category>,
) -> Vec<HealthCaseResult> {
    let suite = default_suite();
    let filtered: Vec<_> = match filter {
        Some(cat) => suite.into_iter().filter(|c| c.category == cat).collect(),
        None => suite,
    };
    filtered
        .iter()
        .map(|case| run_case(case, simplifier))
        .collect()
}
