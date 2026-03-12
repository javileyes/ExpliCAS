use crate::Simplifier;
use cas_solver_core::health_suite_models::{HealthCase, HealthCaseResult};

use super::super::case::run_case;

pub(super) fn run_selected_suite(
    suite: &[HealthCase],
    simplifier: &mut Simplifier,
) -> Vec<HealthCaseResult> {
    suite
        .iter()
        .map(|case| run_case(case, simplifier))
        .collect()
}
