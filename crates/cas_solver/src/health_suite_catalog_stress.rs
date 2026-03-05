use crate::health_suite_types::{Category, HealthCase, HealthLimits};

pub(crate) fn stress_suite() -> Vec<HealthCase> {
    vec![
        // ============ STRESS SUITE (heavy integration tests) ============
        // These cases are designed to exercise growth, multi-phase interactions,
        // and higher workloads. Expected to have higher rewrites and growth.
        HealthCase {
            name: "expand_product_chain",
            category: Category::Stress,
            // Use expand() to force Transform phase activity (bypasses binomial*binomial guard)
            expr: "expand((x+1)*(x+2)*(x+3))",
            limits: HealthLimits {
                max_total_rewrites: 200,
                max_growth: 350,
                max_transform_rewrites: 120,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "binomial_large",
            category: Category::Stress,
            expr: "(x+1)^8",
            limits: HealthLimits {
                max_total_rewrites: 220,
                max_growth: 450,
                max_transform_rewrites: 120,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "distribute_sum",
            category: Category::Stress,
            expr: "3*(x+y+z+w)",
            limits: HealthLimits {
                max_total_rewrites: 160,
                max_growth: 250,
                max_transform_rewrites: 100,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "nested_distribution",
            category: Category::Stress,
            expr: "2*(x + 3*(y+4))",
            limits: HealthLimits {
                max_total_rewrites: 200,
                max_growth: 300,
                max_transform_rewrites: 120,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "rationalize_level15_mixed",
            category: Category::Stress,
            expr: "(x+1)/(2*(1+sqrt(2))) + 2*(y+3)",
            limits: HealthLimits {
                max_total_rewrites: 180,
                max_growth: 220,
                max_transform_rewrites: 100,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "rationalize_binomial_negative",
            category: Category::Stress,
            expr: "x/(2*(3-2*sqrt(5)))",
            limits: HealthLimits {
                max_total_rewrites: 160,
                max_growth: 220,
                max_transform_rewrites: 80,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "nested_root_simplify_hard",
            category: Category::Stress,
            expr: "sqrt(5 + 2*sqrt(6))",
            limits: HealthLimits {
                max_total_rewrites: 160,
                max_growth: 200,
                max_transform_rewrites: 60,
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "fraction_polynomial_combo",
            category: Category::Stress,
            expr: "x/2 + x/3 + (x+1)^6",
            limits: HealthLimits {
                max_total_rewrites: 260,
                max_growth: 500,
                max_transform_rewrites: 140,
                forbid_cycles: true,
            },
        },
    ]
}

pub(crate) fn policy_suite() -> Vec<HealthCase> {
    vec![
        // ============ Policy A+: simplify vs expand behavior ============
        HealthCase {
            name: "policy_simplify_binomial_no_expand",
            category: Category::Policy,
            expr: "(x+1)*(x+2)",
            limits: HealthLimits {
                max_total_rewrites: 10, // Very low - should NOT expand
                max_growth: 10,
                max_transform_rewrites: 2, // Minimal transform activity
                forbid_cycles: true,
            },
        },
        HealthCase {
            name: "policy_simplify_conjugate_expands",
            category: Category::Policy,
            expr: "(x-1)*(x+1)",
            limits: HealthLimits {
                max_total_rewrites: 30, // DoS rule should fire
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
                max_total_rewrites: 50, // Expansion should work
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
                max_total_rewrites: 200, // Binomial expansion is expensive
                max_growth: 300,
                max_transform_rewrites: 120,
                forbid_cycles: true,
            },
        },
    ]
}
