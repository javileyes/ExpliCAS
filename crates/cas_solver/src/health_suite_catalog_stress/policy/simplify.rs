use crate::health_suite_models::HealthCase;

use super::shared::policy_case;

pub(super) fn policy_simplify_cases() -> [HealthCase; 2] {
    [
        policy_case(
            "policy_simplify_binomial_no_expand",
            "(x+1)*(x+2)",
            10,
            10,
            2,
        ),
        policy_case(
            "policy_simplify_conjugate_expands",
            "(x-1)*(x+1)",
            30,
            20,
            15,
        ),
    ]
}
