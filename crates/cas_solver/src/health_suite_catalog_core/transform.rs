use crate::health_suite_types::{Category, HealthCase, HealthLimits};

pub(super) fn transform_cases() -> Vec<HealthCase> {
    vec![
        HealthCase {
            name: "distribute_basic",
            category: Category::Transform,
            expr: "2*(x+3)",
            limits: HealthLimits {
                max_total_rewrites: 20,
                max_growth: 30,
                max_transform_rewrites: 10,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "distribute_nested",
            category: Category::Transform,
            expr: "3*(x+(y+2))",
            limits: HealthLimits {
                max_total_rewrites: 30,
                max_growth: 50,
                max_transform_rewrites: 15,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "expand_product",
            category: Category::Transform,
            expr: "(x+1)*(x+2)",
            limits: HealthLimits {
                max_total_rewrites: 40,
                max_growth: 60,
                max_transform_rewrites: 20,
                forbid_cycles: true,
            },
        },
    ]
}
