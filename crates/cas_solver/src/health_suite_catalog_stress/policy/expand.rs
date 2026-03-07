use crate::health_suite_types::HealthCase;

use super::shared::policy_case;

pub(super) fn policy_expand_cases() -> [HealthCase; 2] {
    [
        policy_case(
            "policy_expand_binomial_product",
            "expand((x+1)*(x+2))",
            50,
            40,
            30,
        ),
        policy_case(
            "policy_expand_binomial_power",
            "expand((x+1)^6)",
            200,
            300,
            120,
        ),
    ]
}
