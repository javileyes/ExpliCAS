use crate::health_category::Category;
use crate::health_suite_models::{HealthCase, HealthLimits};

pub(super) fn fractions_cases() -> Vec<HealthCase> {
    vec![
        HealthCase {
            name: "fraction_add",
            category: Category::Fractions,
            expr: "x/2 + x/3",
            limits: HealthLimits {
                max_total_rewrites: 40,
                max_growth: 60,
                max_transform_rewrites: 15,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "fraction_simplify",
            category: Category::Fractions,
            expr: "(x^2-1)/(x-1)",
            limits: HealthLimits {
                max_total_rewrites: 50,
                max_growth: 80,
                max_transform_rewrites: 20,
                forbid_cycles: true,
            },
        },
    ]
}
