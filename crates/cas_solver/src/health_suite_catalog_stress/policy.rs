mod expand;
mod shared;
mod simplify;

use cas_solver_core::health_suite_models::HealthCase;

pub(crate) fn policy_suite() -> Vec<HealthCase> {
    let mut suite = Vec::new();
    suite.extend(simplify::policy_simplify_cases());
    suite.extend(expand::policy_expand_cases());
    suite
}
