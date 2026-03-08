use crate::health_suite_types::{Category, HealthCase, HealthLimits};

pub(super) fn roots_cases() -> Vec<HealthCase> {
    vec![HealthCase {
        name: "nested_root",
        category: Category::Roots,
        expr: "sqrt(8)",
        limits: HealthLimits {
            max_total_rewrites: 20,
            max_growth: 30,
            max_transform_rewrites: 10,
            forbid_cycles: true,
        },
    }]
}
