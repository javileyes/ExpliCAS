use crate::health_suite_types::{Category, HealthCase, HealthLimits};

pub(super) fn rationalization_cases() -> Vec<HealthCase> {
    vec![
        HealthCase {
            name: "rationalize_simple",
            category: Category::Rationalization,
            expr: "1/sqrt(2)",
            limits: HealthLimits {
                max_total_rewrites: 30,
                max_growth: 40,
                max_transform_rewrites: 10,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "rationalize_binomial",
            category: Category::Rationalization,
            expr: "1/(1+sqrt(2))",
            limits: HealthLimits {
                max_total_rewrites: 50,
                max_growth: 80,
                max_transform_rewrites: 20,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "rationalize_complex",
            category: Category::Rationalization,
            expr: "1/(3-2*sqrt(5))",
            limits: HealthLimits {
                max_total_rewrites: 80,
                max_growth: 150,
                max_transform_rewrites: 30,
                forbid_cycles: true,
            },
        },
    ]
}
