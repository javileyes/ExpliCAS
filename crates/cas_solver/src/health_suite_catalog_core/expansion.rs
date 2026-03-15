use crate::health_category::Category;
use crate::health_suite_models::{HealthCase, HealthLimits};

pub(super) fn expansion_cases() -> Vec<HealthCase> {
    vec![
        HealthCase {
            name: "binomial_small",
            category: Category::Expansion,
            expr: "(x+1)^3",
            limits: HealthLimits {
                max_total_rewrites: 30,
                max_growth: 50,
                max_transform_rewrites: 10,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "binomial_medium",
            category: Category::Expansion,
            expr: "(x+1)^5",
            limits: HealthLimits {
                max_total_rewrites: 50,
                max_growth: 100,
                max_transform_rewrites: 20,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "expand_binomial",
            category: Category::Expansion,
            expr: "expand((x+1)^2)",
            limits: HealthLimits {
                max_total_rewrites: 20,
                max_growth: 30,
                max_transform_rewrites: 5,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "expand_conjugate",
            category: Category::Expansion,
            expr: "expand((x-1)*(x+1))",
            limits: HealthLimits {
                max_total_rewrites: 15,
                max_growth: 20,
                max_transform_rewrites: 3,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "expand_product_chain",
            category: Category::Expansion,
            expr: "expand((x-1)*(x+1)*(x^2+1))",
            limits: HealthLimits {
                max_total_rewrites: 40,
                max_growth: 60,
                max_transform_rewrites: 10,
                forbid_cycles: true,
            },
        },
    ]
}
