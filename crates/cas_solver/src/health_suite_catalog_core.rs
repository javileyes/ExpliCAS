use crate::health_suite_types::{Category, HealthCase, HealthLimits};

pub(crate) fn core_suite() -> Vec<HealthCase> {
    let mut suite = Vec::new();
    suite.extend(transform_cases());
    suite.extend(expansion_cases());
    suite.extend(fractions_cases());
    suite.extend(rationalization_cases());
    suite.extend(mixed_cases());
    suite.extend(baseline_cases());
    suite.extend(roots_cases());
    suite.extend(powers_cases());
    suite
}

fn transform_cases() -> Vec<HealthCase> {
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

fn expansion_cases() -> Vec<HealthCase> {
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

fn fractions_cases() -> Vec<HealthCase> {
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

fn rationalization_cases() -> Vec<HealthCase> {
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

fn mixed_cases() -> Vec<HealthCase> {
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

fn baseline_cases() -> Vec<HealthCase> {
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

fn roots_cases() -> Vec<HealthCase> {
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

fn powers_cases() -> Vec<HealthCase> {
    vec![HealthCase {
        name: "power_simplify",
        category: Category::Powers,
        expr: "x^2 * x^3",
        limits: HealthLimits {
            max_total_rewrites: 15,
            max_growth: 20,
            max_transform_rewrites: 5,
            forbid_cycles: true,
        },
    }]
}
