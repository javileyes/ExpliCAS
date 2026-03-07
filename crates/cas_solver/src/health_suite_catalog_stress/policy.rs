use crate::health_suite_types::{Category, HealthCase, HealthLimits};

pub(crate) fn policy_suite() -> Vec<HealthCase> {
    vec![
        HealthCase {
            name: "policy_simplify_binomial_no_expand",
            category: Category::Policy,
            expr: "(x+1)*(x+2)",
            limits: HealthLimits {
                max_total_rewrites: 10,
                max_growth: 10,
                max_transform_rewrites: 2,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "policy_simplify_conjugate_expands",
            category: Category::Policy,
            expr: "(x-1)*(x+1)",
            limits: HealthLimits {
                max_total_rewrites: 30,
                max_growth: 20,
                max_transform_rewrites: 15,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "policy_expand_binomial_product",
            category: Category::Policy,
            expr: "expand((x+1)*(x+2))",
            limits: HealthLimits {
                max_total_rewrites: 50,
                max_growth: 40,
                max_transform_rewrites: 30,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "policy_expand_binomial_power",
            category: Category::Policy,
            expr: "expand((x+1)^6)",
            limits: HealthLimits {
                max_total_rewrites: 200,
                max_growth: 300,
                max_transform_rewrites: 120,
                forbid_cycles: true,
            },
        },
    ]
}
