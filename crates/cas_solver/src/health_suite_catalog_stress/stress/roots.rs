use crate::health_suite_types::HealthCase;

use super::shared::stress_case;

pub(super) fn stress_root_cases() -> [HealthCase; 1] {
    [stress_case(
        "nested_root_simplify_hard",
        "sqrt(5 + 2*sqrt(6))",
        160,
        200,
        60,
    )]
}
