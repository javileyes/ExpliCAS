use crate::health_suite_catalog_core::core_suite;
use crate::health_suite_catalog_stress::{policy_suite, stress_suite};
use crate::health_suite_models::HealthCase;

/// The default health suite.
pub fn default_suite() -> Vec<HealthCase> {
    let mut suite = core_suite();
    suite.extend(stress_suite());
    suite.extend(policy_suite());
    suite
}
