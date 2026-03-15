use crate::health_category::Category;
use crate::health_suite_models::{HealthCase, HealthLimits};

pub(super) fn baseline_cases() -> Vec<HealthCase> {
    vec![
        HealthCase {
            name: "simple_noop",
            category: Category::Baseline,
            expr: "x + y",
            limits: HealthLimits {
                max_total_rewrites: 15,
                max_growth: 20,
                max_transform_rewrites: 5,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "constant_fold",
            category: Category::Baseline,
            expr: "2 + 3 * 4",
            limits: HealthLimits {
                max_total_rewrites: 10,
                max_growth: 10,
                max_transform_rewrites: 5,
                forbid_cycles: true,
            },
        },
    ]
}
