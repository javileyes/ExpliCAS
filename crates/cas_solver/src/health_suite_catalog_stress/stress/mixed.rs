use cas_solver_core::health_suite_models::HealthCase;

use super::shared::stress_case;

pub(super) fn stress_mixed_cases() -> [HealthCase; 1] {
    [stress_case(
        "fraction_polynomial_combo",
        "x/2 + x/3 + (x+1)^6",
        260,
        500,
        140,
    )]
}
