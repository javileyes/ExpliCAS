use cas_solver_core::health_category::Category;
use cas_solver_core::health_suite_models::{HealthCase, HealthLimits};

pub(super) fn powers_cases() -> Vec<HealthCase> {
    vec![HealthCase {
        name: "power_simplify",
        category: Category::Powers,
        expr: "x^2 * x^3",
        limits: HealthLimits {
            max_total_rewrites: 15,
            max_growth: 20,
            max_transform_rewrites: 5,
            forbid_cycles: true,
        },
    }]
}
