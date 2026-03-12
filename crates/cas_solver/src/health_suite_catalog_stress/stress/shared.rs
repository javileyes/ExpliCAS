use cas_solver_core::health_category::Category;
use cas_solver_core::health_suite_models::{HealthCase, HealthLimits};

pub(super) fn stress_case(
    name: &'static str,
    expr: &'static str,
    max_total_rewrites: usize,
    max_growth: i64,
    max_transform_rewrites: usize,
) -> HealthCase {
    HealthCase {
        name,
        category: Category::Stress,
        expr,
        limits: HealthLimits {
            max_total_rewrites,
            max_growth,
            max_transform_rewrites,
            forbid_cycles: true,
        },
    }
}
