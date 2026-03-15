use crate::health_category::Category;
use crate::health_suite_models::{HealthCase, HealthLimits};

pub(super) fn mixed_cases() -> Vec<HealthCase> {
    vec![HealthCase {
        name: "mixed_expression",
        category: Category::Mixed,
        expr: "x/(1+sqrt(2)) + 2*(y+3)",
        limits: HealthLimits {
            max_total_rewrites: 80,
            max_growth: 120,
            max_transform_rewrites: 40,
            forbid_cycles: true,
        },
    }]
}
