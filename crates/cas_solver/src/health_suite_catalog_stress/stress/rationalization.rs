use crate::health_suite_types::HealthCase;

use super::shared::stress_case;

pub(super) fn stress_rationalization_cases() -> [HealthCase; 2] {
    [
        stress_case(
            "rationalize_level15_mixed",
            "(x+1)/(2*(1+sqrt(2))) + 2*(y+3)",
            180,
            220,
            100,
        ),
        stress_case(
            "rationalize_binomial_negative",
            "x/(2*(3-2*sqrt(5)))",
            160,
            220,
            80,
        ),
    ]
}
