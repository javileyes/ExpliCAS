use cas_ast::{Constant, Context, Expr, ExprId};
use cas_parser::parse;
use cas_solver::runtime::Simplifier;
use cas_solver::runtime::SimplifyOptions;
use cas_solver::session_api::analysis::{
    evaluate_derive_command_lines_with_resolver, FullSimplifyDisplayMode,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeriveCase {
    id: String,
    family: String,
    source: String,
    target: String,
    expected_status: String,
    expected_strategy: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IdentityShadowCase {
    id: &'static str,
    family: &'static str,
    source: &'static str,
    target: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IdentityPairShadowSeed {
    family: String,
    source: String,
    target: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EmbeddedEquivalenceShadowSeed {
    family: String,
    source: String,
    target: String,
}

const EMBEDDED_DERIVE_SHADOW_EXCLUDED_FAMILIES: &[&str] = &["calculus_integrate"];

fn load_derive_cases() -> Vec<DeriveCase> {
    include_str!("derive_pairs.csv")
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let parts = split_csv_line(line);
            assert_eq!(parts.len(), 6, "unexpected derive csv columns: {line}");
            let id = parts[0].trim().to_string();
            let family = parts[1].trim().to_string();
            let source = parts[2].trim().to_string();
            let target = parts[3].trim().to_string();
            let expected_status = parts[4].trim().to_string();
            let expected_strategy = {
                let trimmed = parts[5].trim();
                (!trimmed.is_empty()).then(|| trimmed.to_string())
            };
            DeriveCase {
                id,
                family,
                source,
                target,
                expected_status,
                expected_strategy,
            }
        })
        .collect()
}

fn load_identity_pair_shadow_seeds() -> Vec<IdentityPairShadowSeed> {
    let mut current_family = String::from("Uncategorized");
    let mut seeds = Vec::new();

    for line in include_str!("identity_pairs.csv").lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Each row")
                && !label.starts_with("var is")
                && !label.starts_with("Mathematical Identity")
                && !label.starts_with('=')
            {
                current_family = label.to_string();
            }
            continue;
        }

        let parts = split_csv_line(line);
        if parts.len() < 2 {
            continue;
        }
        seeds.push(IdentityPairShadowSeed {
            family: current_family.clone(),
            source: parts[0].trim().to_string(),
            target: parts[1].trim().to_string(),
        });
    }

    seeds
}

fn load_embedded_equivalence_shadow_seeds() -> Vec<EmbeddedEquivalenceShadowSeed> {
    include_str!("../../../docs/embedded_equivalence_context_corpus.csv")
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let parts = split_csv_line(line);
            assert!(
                parts.len() >= 7,
                "unexpected embedded equivalence csv columns: {line}"
            );
            EmbeddedEquivalenceShadowSeed {
                family: parts[4].trim().to_string(),
                source: parts[5].trim().to_string(),
                target: parts[6].trim().to_string(),
            }
        })
        .collect()
}

fn split_csv_line(line: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    for ch in line.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                parts.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    parts.push(current.trim().to_string());
    parts
}

fn assert_shadow_cases_exist_in_identity_pairs(cases: &[IdentityShadowCase]) {
    let seeds = load_identity_pair_shadow_seeds();
    for case in cases {
        assert!(
            seeds.iter().any(|seed| {
                seed.family == case.family && seed.source == case.source && seed.target == case.target
            }),
            "derive shadow case {} must come from identity_pairs.csv: family={} source={} target={}",
            case.id,
            case.family,
            case.source,
            case.target
        );
    }
}

fn assert_shadow_cases_exist_in_embedded_equivalence_corpus(cases: &[IdentityShadowCase]) {
    let seeds = load_embedded_equivalence_shadow_seeds();
    for case in cases {
        assert!(
            seeds.iter().any(|seed| {
                seed.family == case.family && seed.source == case.source && seed.target == case.target
            }),
            "derive shadow case {} must come from embedded_equivalence_context_corpus.csv: family={} source={} target={}",
            case.id,
            case.family,
            case.source,
            case.target
        );
    }
}

fn embedded_shadow_family_coverage(cases: &[IdentityShadowCase]) -> (usize, usize, Vec<String>) {
    let all_families = load_embedded_equivalence_shadow_seeds()
        .into_iter()
        .map(|seed| seed.family)
        .filter(|family| !EMBEDDED_DERIVE_SHADOW_EXCLUDED_FAMILIES.contains(&family.as_str()))
        .collect::<BTreeSet<_>>();
    let sampled_families = cases
        .iter()
        .map(|case| case.family.to_string())
        .filter(|family| !EMBEDDED_DERIVE_SHADOW_EXCLUDED_FAMILIES.contains(&family.as_str()))
        .collect::<BTreeSet<_>>();
    let missing = all_families
        .difference(&sampled_families)
        .cloned()
        .collect::<Vec<_>>();

    (sampled_families.len(), all_families.len(), missing)
}

fn run_case(case: &DeriveCase) -> Vec<String> {
    let mut simplifier = Simplifier::with_default_rules();
    let input = format!("derive {}, {}", case.source, case.target);
    evaluate_derive_command_lines_with_resolver(
        &mut simplifier,
        &input,
        FullSimplifyDisplayMode::Normal,
        SimplifyOptions::default(),
        |_ctx, expr| Ok(expr),
    )
    .expect("derive should evaluate")
}

fn run_identity_shadow_case(case: &IdentityShadowCase) -> Vec<String> {
    let mut simplifier = Simplifier::with_default_rules();
    let input = format!("derive {}, {}", case.source, case.target);
    evaluate_derive_command_lines_with_resolver(
        &mut simplifier,
        &input,
        FullSimplifyDisplayMode::Normal,
        SimplifyOptions::default(),
        |_ctx, expr| Ok(expr),
    )
    .unwrap_or_else(|err| panic!("derive shadow case {} should evaluate: {err}", case.id))
}

fn count_top_level_steps(lines: &[String]) -> usize {
    lines
        .iter()
        .filter(|line| {
            let trimmed = line.trim_start();
            let Some(prefix) = trimmed.split_whitespace().next() else {
                return false;
            };
            prefix.ends_with('.')
                && prefix[..prefix.len().saturating_sub(1)]
                    .chars()
                    .all(|ch| ch.is_ascii_digit())
        })
        .count()
}

fn derive_strategy_from_lines(lines: &[String]) -> Option<String> {
    lines.iter().find_map(|line| {
        line.trim_start()
            .strip_prefix("Strategy:")
            .map(str::trim)
            .filter(|strategy| !strategy.is_empty())
            .map(str::to_string)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignatureMode {
    Conservative,
    Abstract,
}

#[derive(Debug, Default)]
struct SignatureState {
    variables: BTreeMap<String, usize>,
    functions: BTreeMap<(String, usize), usize>,
}

fn pair_shape_signature(case: &DeriveCase, mode: SignatureMode) -> String {
    let mut ctx = Context::new();
    let source = parse(&case.source, &mut ctx)
        .unwrap_or_else(|err| panic!("failed to parse derive source {}: {err}", case.id));
    let target = parse(&case.target, &mut ctx)
        .unwrap_or_else(|err| panic!("failed to parse derive target {}: {err}", case.id));
    let mut state = SignatureState::default();
    format!(
        "{} => {}",
        expr_shape_signature(&ctx, source, &mut state, mode),
        expr_shape_signature(&ctx, target, &mut state, mode)
    )
}

#[derive(Debug, Default)]
struct DeriveOutcomeStats {
    derived: usize,
    unsupported: usize,
    not_equivalent: usize,
    derived_step_total: usize,
    long_path_count: usize,
    generic_simplify_expected_ids: Vec<String>,
    expected_strategy_counts: BTreeMap<String, usize>,
    unsupported_by_family: BTreeMap<String, usize>,
    derived_by_family: BTreeMap<String, usize>,
}

#[derive(Debug, Default)]
struct DeriveShadowPressureStats {
    sampled: usize,
    derived: usize,
    unsupported: usize,
    not_equivalent: usize,
    derived_step_total: usize,
    single_step_successes: usize,
    multi_step_successes: usize,
    multi_step_success_ids: Vec<String>,
    generic_simplify_strategy_successes: usize,
    generic_simplify_strategy_ids: Vec<String>,
    actual_strategy_counts: BTreeMap<String, usize>,
    derived_by_family: BTreeMap<String, usize>,
    unsupported_by_family: BTreeMap<String, usize>,
    not_equivalent_by_family: BTreeMap<String, usize>,
}

const LONG_PATH_THRESHOLD: usize = 4;

const IDENTITY_SHADOW_PRESSURE_CASES: &[IdentityShadowCase] = &[
    IdentityShadowCase {
        id: "identity_tan_pythagorean",
        family: "Pythagorean Identities",
        source: "tan(x)^2 + 1",
        target: "sec(x)^2",
    },
    IdentityShadowCase {
        id: "identity_trig_pythagorean_rewrite",
        family: "Pythagorean Identities",
        source: "sin(x)^2 + cos(x)^2",
        target: "1",
    },
    IdentityShadowCase {
        id: "identity_symbolic_cube_expand",
        family: "Binomial Expansion",
        source: "(x+1)^3",
        target: "x^3 + 3*x^2 + 3*x + 1",
    },
    IdentityShadowCase {
        id: "identity_factor_difference_squares",
        family: "Difference of Squares",
        source: "x^2 - 1",
        target: "(x-1)*(x+1)",
    },
    IdentityShadowCase {
        id: "identity_sqrt_arithmetic",
        family: "Sqrt Arithmetic",
        source: "sqrt(8) + sqrt(2)",
        target: "3*sqrt(2)",
    },
    IdentityShadowCase {
        id: "identity_square_of_square_root_requires_nonnegative",
        family: "Root Rules",
        source: "sqrt(x)^2",
        target: "x",
    },
    IdentityShadowCase {
        id: "identity_choose_numeric_binomial",
        family: "Binomial coefficients (choose function)",
        source: "choose(5,2)",
        target: "10",
    },
    IdentityShadowCase {
        id: "identity_choose_numeric_pascal",
        family: "Pascal's identity: C(n,k) + C(n,k+1) = C(n+1,k+1)",
        source: "choose(4,1) + choose(4,2)",
        target: "choose(5,2)",
    },
    IdentityShadowCase {
        id: "identity_choose_numeric_symmetry",
        family: "Symmetry: C(n,k) = C(n,n-k)",
        source: "choose(6,1)",
        target: "choose(6,5)",
    },
    IdentityShadowCase {
        id: "identity_finite_sum_numeric_linear",
        family: "Sum of first n integers: sum(k, k, 1, n) = n*(n+1)/2",
        source: "sum(k, k, 1, 5)",
        target: "15",
    },
    IdentityShadowCase {
        id: "identity_finite_sum_numeric_squares",
        family: "Sum of squares: sum(k^2, k, 1, n) = n*(n+1)*(2n+1)/6",
        source: "sum(k^2, k, 1, 5)",
        target: "55",
    },
    IdentityShadowCase {
        id: "identity_finite_sum_numeric_cubes",
        family: "Sum of cubes: sum(k^3, k, 1, n) = (n*(n+1)/2)^2",
        source: "sum(k^3, k, 1, 5)",
        target: "225",
    },
    IdentityShadowCase {
        id: "identity_finite_sum_numeric_constant",
        family: "Constant sum",
        source: "sum(5, k, 1, 4)",
        target: "20",
    },
    IdentityShadowCase {
        id: "identity_finite_sum_numeric_constant_shifted_lower",
        family: "Constant sum",
        source: "sum(5, k, 2, 4)",
        target: "15",
    },
    IdentityShadowCase {
        id: "identity_finite_sum_numeric_geometric_power",
        family: "Geometric sum: sum(r^k, k, 0, n) = (r^(n+1) - 1)/(r - 1)",
        source: "sum(2^k, k, 0, 3)",
        target: "15",
    },
    IdentityShadowCase {
        id: "identity_finite_product_numeric_factorial",
        family: "Product = factorial: product(k, k, 1, n) = n!",
        source: "product(k, k, 1, 5)",
        target: "120",
    },
    IdentityShadowCase {
        id: "identity_finite_product_numeric_squares",
        family: "Product of powers: product(k^2, k, 1, n)",
        source: "product(k^2, k, 1, 3)",
        target: "36",
    },
    IdentityShadowCase {
        id: "identity_finite_product_numeric_constant",
        family: "Product of constant: product(c, k, 1, n) = c^n",
        source: "product(2, k, 1, 5)",
        target: "32",
    },
    IdentityShadowCase {
        id: "identity_trig_triple_angle",
        family: "Triple angle",
        source: "sin(3*x)",
        target: "3*sin(x) - 4*sin(x)^3",
    },
    IdentityShadowCase {
        id: "identity_log_change_of_base",
        family: "Change of base",
        source: "ln(x)/ln(2)",
        target: "log(2, x)",
    },
    IdentityShadowCase {
        id: "identity_log_product_root_cleanup",
        family: "Log of complex products",
        source: "ln(sqrt(x)*y)",
        target: "ln(x)/2 + ln(y)",
    },
    IdentityShadowCase {
        id: "identity_exponential_log_product",
        family: "Exponential of logs",
        source: "exp(ln(x) + ln(y))",
        target: "x*y",
    },
    IdentityShadowCase {
        id: "identity_tangent_half_angle",
        family: "Tangent half-angle substitution",
        source: "tan(x/2)",
        target: "sin(x)/(1 + cos(x))",
    },
    IdentityShadowCase {
        id: "identity_arctan_reciprocal_sum",
        family: "arctan compositions",
        source: "arctan(x) + arctan(1/x)",
        target: "pi/2",
    },
    IdentityShadowCase {
        id: "identity_completing_square",
        family: "Completing the square patterns",
        source: "x^2 + 2*x",
        target: "(x+1)^2 - 1",
    },
    IdentityShadowCase {
        id: "identity_symbolic_rationalize",
        family: "Rationalize denominators with conjugates",
        source: "1/(sqrt(x)+1)",
        target: "(sqrt(x)-1)/(x-1)",
    },
    IdentityShadowCase {
        id: "identity_nested_radical_denesting",
        family: "Nested radicals",
        source: "sqrt(6 + 2*sqrt(5))",
        target: "sqrt(5)+1",
    },
    IdentityShadowCase {
        id: "identity_cube_root_rationalization",
        family: "Rationalize with cube roots",
        source: "1/(1+x^(1/3))",
        target: "(1-x^(1/3)+x^(2/3))/(1+x)",
    },
    IdentityShadowCase {
        id: "identity_log_inverse_power_tower",
        family: "Power tower identities",
        source: "x^(ln(y)/ln(x))",
        target: "y",
    },
    IdentityShadowCase {
        id: "identity_fractional_power_merge",
        family: "Fractional exponent combination",
        source: "x^(1/2)*x^(1/3)",
        target: "x^(5/6)",
    },
    IdentityShadowCase {
        id: "identity_odd_half_power_expand",
        family: "Power of roots",
        source: "sqrt(x^3)",
        target: "x*sqrt(x)",
    },
    IdentityShadowCase {
        id: "identity_fraction_cancel_difference_squares",
        family: "Factor-cancel patterns",
        source: "(x^2 - 1)/(x + 1)",
        target: "x - 1",
    },
    IdentityShadowCase {
        id: "identity_fraction_combine_adjacent_unit_denominators",
        family: "Addition of fractions",
        source: "1/x + 1/(x+1)",
        target: "(2*x+1)/(x*(x+1))",
    },
    IdentityShadowCase {
        id: "identity_nested_fraction_reciprocal_inverse",
        family: "Fraction Simplification",
        source: "1/(1/x)",
        target: "x",
    },
    IdentityShadowCase {
        id: "identity_partial_fraction_telescoping_split",
        family: "Partial fractions / telescoping patterns",
        source: "1/(x*(x+1))",
        target: "1/x - 1/(x+1)",
    },
    IdentityShadowCase {
        id: "identity_mixed_fraction_decomposition",
        family: "Mixed fraction decomposition",
        source: "(a*x+b)/(x+c)",
        target: "a + (b-a*c)/(x+c)",
    },
    IdentityShadowCase {
        id: "identity_hyperbolic_exp_decomposition",
        family: "Fundamental exp decomposition",
        source: "sinh(x) + cosh(x)",
        target: "exp(x)",
    },
    IdentityShadowCase {
        id: "identity_hyperbolic_negative_exp_decomposition",
        family: "Fundamental exp decomposition",
        source: "cosh(x) - sinh(x)",
        target: "exp(-x)",
    },
    IdentityShadowCase {
        id: "identity_hyperbolic_negated_negative_exp_decomposition",
        family: "Fundamental exp decomposition",
        source: "sinh(x) - cosh(x)",
        target: "-exp(-x)",
    },
];

const EMBEDDED_EQUIVALENCE_SHADOW_PRESSURE_CASES: &[IdentityShadowCase] = &[
    IdentityShadowCase {
        id: "embedded_collect_linear",
        family: "collect",
        source: "a*x + b*x + c",
        target: "(a + b)*x + c",
    },
    IdentityShadowCase {
        id: "embedded_expand_common_factor_sum",
        family: "expand",
        source: "a*(b+c)",
        target: "a*b + a*c",
    },
    IdentityShadowCase {
        id: "embedded_factor_difference_squares",
        family: "factor",
        source: "a^2 - b^2",
        target: "(a - b)*(a + b)",
    },
    IdentityShadowCase {
        id: "embedded_finite_aggregate_sum_first_integers_symbolic",
        family: "finite_aggregate",
        source: "sum(k, k, 1, n)",
        target: "n*(n+1)/2",
    },
    IdentityShadowCase {
        id: "embedded_finite_telescoping_product_basic",
        family: "finite_telescoping",
        source: "product((k+1)/k, k, 1, n)",
        target: "n+1",
    },
    IdentityShadowCase {
        id: "embedded_fraction_combine_same_denominator_sum",
        family: "fraction_combine",
        source: "a/d + b/d",
        target: "(a+b)/d",
    },
    IdentityShadowCase {
        id: "embedded_fraction_decompose_symbolic_over_shift",
        family: "fraction_decompose",
        source: "(a*x+b)/(x+c)",
        target: "a + (b-a*c)/(x+c)",
    },
    IdentityShadowCase {
        id: "embedded_fraction_expand_simple",
        family: "fraction_expand",
        source: "(a+b)/d",
        target: "a/d + b/d",
    },
    IdentityShadowCase {
        id: "embedded_consecutive_factorial_ratio",
        family: "simplify",
        source: "(n+1)!/n!",
        target: "n+1",
    },
    IdentityShadowCase {
        id: "embedded_factor_out_with_division_quadratic",
        family: "conditional_factor",
        source: "a*x^2 + b*x + c",
        target: "x*(a*x + b + c/x)",
    },
    IdentityShadowCase {
        id: "embedded_integrate_prep_morrie_basic",
        family: "integrate_prep",
        source: "cos(x)*cos(2*x)*cos(4*x)",
        target: "sin(8*x)/(8*sin(x))",
    },
    IdentityShadowCase {
        id: "embedded_calculus_diff_bounded_arcsin_residual",
        family: "calculus_diff",
        source: "diff(arcsin(2*x-1)/2,x)",
        target: "1/(2*sqrt(x)*sqrt(1-x))",
    },
    IdentityShadowCase {
        id: "embedded_log_contract_grouped_power",
        family: "log_contract",
        source: "ln(x^3)+ln(y^2)",
        target: "ln(x^3*y^2)",
    },
    IdentityShadowCase {
        id: "embedded_log_expand_product",
        family: "log_expand",
        source: "ln(x*y)",
        target: "ln(x) + ln(y)",
    },
    IdentityShadowCase {
        id: "embedded_log_exp_inverse_power_alias",
        family: "log_exp_inverse",
        source: "exp(y*log(x))",
        target: "x^y",
    },
    IdentityShadowCase {
        id: "embedded_log10_power_alias",
        family: "log_exp_inverse",
        source: "10^(y*log10(x))",
        target: "x^y",
    },
    IdentityShadowCase {
        id: "embedded_log_inverse_power_unary_natural_alias",
        family: "log_inverse_power",
        source: "x^(log(log(x))/log(x))",
        target: "log(x)",
    },
    IdentityShadowCase {
        id: "embedded_power_merge_symbolic_quotient_powers",
        family: "power_merge",
        source: "x^a/x^b",
        target: "x^(a-b)",
    },
    IdentityShadowCase {
        id: "embedded_nested_fraction_one_over_sum",
        family: "nested_fraction",
        source: "1/(1/a + 1/b)",
        target: "(a*b)/(a+b)",
    },
    IdentityShadowCase {
        id: "embedded_number_theory_choose_pascal",
        family: "number_theory",
        source: "choose(4,1)+choose(4,2)",
        target: "choose(5,2)",
    },
    IdentityShadowCase {
        id: "embedded_polynomial_product_difference_of_squares_quadratic",
        family: "polynomial_product",
        source: "(x^2+a^2)*(x^2-a^2)",
        target: "x^4-a^4",
    },
    IdentityShadowCase {
        id: "embedded_radical_power_odd_half",
        family: "radical_power",
        source: "x^(3/2)",
        target: "abs(x)*sqrt(x)",
    },
    IdentityShadowCase {
        id: "embedded_rationalize_linear_root",
        family: "rationalize",
        source: "1/(sqrt(x)-1)",
        target: "(sqrt(x)+1)/(x-1)",
    },
    IdentityShadowCase {
        id: "embedded_solve_prep_complete_square_symbolic_leading_coeff",
        family: "solve_prep",
        source: "a*x^2 + b*x + c",
        target: "a*(x + b/(2*a))^2 + c - b^2/(4*a)",
    },
    IdentityShadowCase {
        id: "embedded_telescoping_fraction_affine_symbolic_shift_gap",
        family: "telescoping_fraction",
        source: "1/((a*n+b)*(a*n+c))",
        target: "1/(c-b)*(1/(a*n+b) - 1/(a*n+c))",
    },
    IdentityShadowCase {
        id: "embedded_trig_tangent_ratio_expand",
        family: "trig_contract",
        source: "tan(x)",
        target: "sin(x)/cos(x)",
    },
    IdentityShadowCase {
        id: "embedded_trig_expand_double_sin",
        family: "trig_expand",
        source: "sin(2*x)",
        target: "2*sin(x)*cos(x)",
    },
];

fn assert_case_matches_expected_outcome(case: &DeriveCase, stats: &mut DeriveOutcomeStats) {
    let lines = run_case(case);
    match case.expected_status.as_str() {
        "derived" => {
            stats.derived += 1;
            *stats
                .derived_by_family
                .entry(case.family.clone())
                .or_default() += 1;
            let expected_strategy = case
                .expected_strategy
                .as_ref()
                .expect("derived cases must declare strategy");
            *stats
                .expected_strategy_counts
                .entry(expected_strategy.clone())
                .or_default() += 1;
            if expected_strategy == "simplify" {
                stats.generic_simplify_expected_ids.push(case.id.clone());
            }
            assert!(
                lines
                    .iter()
                    .any(|line| line.starts_with("Strategy:") && line.contains(expected_strategy)),
                "case {} should use strategy {}; lines={:?}",
                case.id,
                expected_strategy,
                lines
            );
            assert!(
                lines.iter().any(|line| line.starts_with("Result:")),
                "case {} should emit Result line",
                case.id
            );
            let step_count = count_top_level_steps(&lines);
            stats.derived_step_total += step_count;
            if step_count > LONG_PATH_THRESHOLD {
                stats.long_path_count += 1;
            }
        }
        "equivalent_but_unsupported" => {
            stats.unsupported += 1;
            *stats
                .unsupported_by_family
                .entry(case.family.clone())
                .or_default() += 1;
            assert!(
                lines.iter().any(|line| {
                    line.contains(
                        "Equivalent, but the second expression is not a supported simplification target yet."
                    )
                }),
                "case {} should report equivalent-but-unsupported; lines={:?}",
                case.id,
                lines
            );
        }
        "not_equivalent" => {
            stats.not_equivalent += 1;
            assert!(
                lines.iter().any(|line| {
                    line.contains("Derive unavailable: the two expressions are not equivalent.")
                }),
                "case {} should report not-equivalent; lines={:?}",
                case.id,
                lines
            );
        }
        other => panic!("unexpected expected_status {other}"),
    }
}

fn evaluate_identity_shadow_pressure(cases: &[IdentityShadowCase]) -> DeriveShadowPressureStats {
    let mut stats = DeriveShadowPressureStats::default();

    for case in cases {
        stats.sampled += 1;
        let lines = run_identity_shadow_case(case);
        let step_count = count_top_level_steps(&lines);

        if lines.iter().any(|line| line.starts_with("Result:")) {
            stats.derived += 1;
            stats.derived_step_total += step_count;
            let strategy = derive_strategy_from_lines(&lines).unwrap_or_else(|| {
                panic!(
                    "derive shadow case {} should emit a Strategy line: {:?}",
                    case.id, lines
                )
            });
            if strategy == "simplify" {
                stats.generic_simplify_strategy_successes += 1;
                stats
                    .generic_simplify_strategy_ids
                    .push(case.id.to_string());
            }
            *stats.actual_strategy_counts.entry(strategy).or_default() += 1;
            if step_count <= 1 {
                stats.single_step_successes += 1;
            } else {
                stats.multi_step_successes += 1;
                stats
                    .multi_step_success_ids
                    .push(format!("{}:{step_count}", case.id));
            }
            *stats
                .derived_by_family
                .entry(case.family.to_string())
                .or_default() += 1;
        } else if lines.iter().any(|line| {
            line.contains(
                "Equivalent, but the second expression is not a supported simplification target yet.",
            )
        }) {
            stats.unsupported += 1;
            *stats
                .unsupported_by_family
                .entry(case.family.to_string())
                .or_default() += 1;
        } else if lines.iter().any(|line| {
            line.contains("Derive unavailable: the two expressions are not equivalent.")
                || line.contains(
                    "Derive unavailable: cannot prove that the two expressions are equivalent",
                )
        }) {
            stats.not_equivalent += 1;
            *stats
                .not_equivalent_by_family
                .entry(case.family.to_string())
                .or_default() += 1;
        } else {
            panic!(
                "derive shadow case {} produced unclassified output: {:?}",
                case.id, lines
            );
        }
    }

    stats
}

fn evaluate_cases_follow_expected_outcomes<'a>(
    cases: impl IntoIterator<Item = &'a DeriveCase>,
) -> DeriveOutcomeStats {
    let mut stats = DeriveOutcomeStats::default();
    for case in cases {
        assert_case_matches_expected_outcome(case, &mut stats);
    }
    stats
}

fn derive_case_bucket(case: &DeriveCase) -> &'static str {
    match case.family.as_str() {
        "trig_expand" | "trig_contract" => "trig",
        "simplify" | "log_expand" | "log_contract" | "log_exp_inverse" | "log_inverse_power"
        | "number_theory" | "rationalize" | "radical_power" => "simplify_log",
        "expand" | "fraction_expand" | "fraction_combine" | "fraction_decompose"
        | "nested_fraction" => "expansion_fraction",
        "collect"
        | "conditional_factor"
        | "factor"
        | "finite_aggregate"
        | "finite_telescoping"
        | "integrate_prep"
        | "negative"
        | "polynomial_product"
        | "power_merge"
        | "solve_prep"
        | "telescoping_fraction" => "structural",
        other => panic!("unassigned derive family bucket: {other}"),
    }
}

fn cases_in_bucket<'a>(cases: &'a [DeriveCase], bucket: &str) -> Vec<&'a DeriveCase> {
    cases
        .iter()
        .filter(|case| derive_case_bucket(case) == bucket)
        .collect()
}

fn derive_case_perf_slice(case: &DeriveCase) -> &'static str {
    match case.family.as_str() {
        "trig_expand" | "trig_contract" => "trig",
        "simplify" | "log_expand" | "log_contract" | "log_exp_inverse" | "log_inverse_power"
        | "number_theory" | "rationalize" | "radical_power" => "simplify_log",
        "expand" => "expand",
        "fraction_expand" | "fraction_combine" | "fraction_decompose" | "nested_fraction" => {
            "fractional"
        }
        "collect" | "conditional_factor" | "factor" | "negative" => "factor_collect",
        "finite_aggregate"
        | "finite_telescoping"
        | "integrate_prep"
        | "solve_prep"
        | "telescoping_fraction" => "prep_telescoping",
        "polynomial_product" | "power_merge" => "poly_merge",
        other => panic!("unassigned derive family perf slice: {other}"),
    }
}

fn cases_in_perf_slice<'a>(cases: &'a [DeriveCase], slice: &str) -> Vec<&'a DeriveCase> {
    cases
        .iter()
        .filter(|case| derive_case_perf_slice(case) == slice)
        .collect()
}

fn find_case_by_id<'a>(cases: &'a [DeriveCase], id: &str) -> &'a DeriveCase {
    cases
        .iter()
        .find(|case| case.id == id)
        .unwrap_or_else(|| panic!("missing derive case id in corpus: {id}"))
}

fn cases_by_ids<'a>(cases: &'a [DeriveCase], ids: &[&str]) -> Vec<&'a DeriveCase> {
    ids.iter().map(|id| find_case_by_id(cases, id)).collect()
}

fn expr_shape_signature(
    ctx: &Context,
    expr: ExprId,
    state: &mut SignatureState,
    mode: SignatureMode,
) -> String {
    match ctx.get(expr) {
        Expr::Number(_) => "N".to_string(),
        Expr::Constant(constant) => match constant {
            Constant::Pi => "pi".to_string(),
            Constant::E => "e".to_string(),
            Constant::Infinity => "infinity".to_string(),
            Constant::Undefined => "undefined".to_string(),
            Constant::I => "i".to_string(),
            Constant::Phi => "phi".to_string(),
        },
        Expr::Variable(id) => {
            let name = ctx.sym_name(*id).to_string();
            let next = state.variables.len() + 1;
            let slot = *state.variables.entry(name).or_insert(next);
            format!("v{slot}")
        }
        Expr::Add(_, _) => {
            let mut terms = Vec::new();
            collect_add_terms(ctx, expr, &mut terms);
            let mut signatures = terms
                .into_iter()
                .map(|term| expr_shape_signature(ctx, term, state, mode))
                .collect::<Vec<_>>();
            signatures.sort();
            format!("add({})", signatures.join(","))
        }
        Expr::Mul(_, _) => {
            let mut factors = Vec::new();
            collect_mul_factors(ctx, expr, &mut factors);
            let mut signatures = factors
                .into_iter()
                .map(|factor| expr_shape_signature(ctx, factor, state, mode))
                .collect::<Vec<_>>();
            signatures.sort();
            format!("mul({})", signatures.join(","))
        }
        Expr::Sub(lhs, rhs) => format!(
            "sub({},{})",
            expr_shape_signature(ctx, *lhs, state, mode),
            expr_shape_signature(ctx, *rhs, state, mode)
        ),
        Expr::Div(lhs, rhs) => format!(
            "div({},{})",
            expr_shape_signature(ctx, *lhs, state, mode),
            expr_shape_signature(ctx, *rhs, state, mode)
        ),
        Expr::Pow(base, exponent) => format!(
            "pow({},{})",
            expr_shape_signature(ctx, *base, state, mode),
            expr_shape_signature(ctx, *exponent, state, mode)
        ),
        Expr::Neg(inner) => format!("neg({})", expr_shape_signature(ctx, *inner, state, mode)),
        Expr::Function(id, args) => {
            let fn_name = match mode {
                SignatureMode::Conservative => ctx.sym_name(*id).to_string(),
                SignatureMode::Abstract => {
                    let key = (ctx.sym_name(*id).to_string(), args.len());
                    let next = state.functions.len() + 1;
                    let slot = *state.functions.entry(key).or_insert(next);
                    format!("f{slot}/{}", args.len())
                }
            };
            let arg_signatures = args
                .iter()
                .map(|arg| expr_shape_signature(ctx, *arg, state, mode))
                .collect::<Vec<_>>();
            format!("{fn_name}({})", arg_signatures.join(","))
        }
        Expr::Matrix { rows, cols, data } => {
            let entries = data
                .iter()
                .map(|entry| expr_shape_signature(ctx, *entry, state, mode))
                .collect::<Vec<_>>();
            format!("matrix[{rows}x{cols}]({})", entries.join(","))
        }
        Expr::SessionRef(_) => "session_ref".to_string(),
        Expr::Hold(inner) => format!("hold({})", expr_shape_signature(ctx, *inner, state, mode)),
    }
}

fn collect_add_terms(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(lhs, rhs) => {
            collect_add_terms(ctx, *lhs, out);
            collect_add_terms(ctx, *rhs, out);
        }
        _ => out.push(expr),
    }
}

fn collect_mul_factors(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(lhs, rhs) => {
            collect_mul_factors(ctx, *lhs, out);
            collect_mul_factors(ctx, *rhs, out);
        }
        _ => out.push(expr),
    }
}

fn shape_budget_violations(
    cases: &[DeriveCase],
    mode: SignatureMode,
    max_cluster_size: usize,
) -> Vec<String> {
    let mut clusters: BTreeMap<(String, String, String, String), Vec<String>> = BTreeMap::new();

    for case in cases
        .iter()
        .filter(|case| case.expected_status == "derived")
    {
        let signature = pair_shape_signature(case, mode);
        let strategy = case.expected_strategy.clone().unwrap_or_default();
        clusters
            .entry((
                case.family.clone(),
                case.expected_status.clone(),
                strategy,
                signature,
            ))
            .or_default()
            .push(case.id.clone());
    }

    clusters
        .into_iter()
        .filter_map(|((family, _status, strategy, signature), mut ids)| {
            ids.sort();
            let budget =
                shape_cluster_budget(mode, &family, &strategy, &signature, &ids, max_cluster_size);
            if ids.len() <= budget {
                return None;
            }
            Some(format!(
                "mode={mode:?} family={family} strategy={strategy} cluster_size={} ids={} signature={signature}",
                ids.len(),
                ids.join(",")
            ))
        })
        .collect()
}

fn shape_cluster_budget(
    mode: SignatureMode,
    family: &str,
    strategy: &str,
    signature: &str,
    ids: &[String],
    default_budget: usize,
) -> usize {
    let reciprocal_product_ids = [
        "reciprocal_trig_cos_sec_product_to_one",
        "reciprocal_trig_product_to_one",
        "reciprocal_trig_sin_csc_product_to_one",
    ];
    if mode == SignatureMode::Abstract
        && family == "simplify"
        && strategy == "rewrite trigs"
        && signature == "mul(f1/1(v1),f2/1(v1)) => N"
        && ids.iter().map(String::as_str).eq(reciprocal_product_ids)
    {
        return reciprocal_product_ids.len();
    }

    default_budget
}

#[test]
fn derive_pairs_corpus_is_unique_and_nonempty() {
    let cases = load_derive_cases();
    assert!(!cases.is_empty(), "derive corpus must not be empty");

    let mut ids = BTreeSet::new();
    for case in cases {
        assert!(
            ids.insert(case.id.clone()),
            "duplicate derive case id: {}",
            case.id
        );
        assert!(!case.source.is_empty(), "case {} missing source", case.id);
        assert!(!case.target.is_empty(), "case {} missing target", case.id);
        assert!(
            matches!(
                case.expected_status.as_str(),
                "derived" | "equivalent_but_unsupported" | "not_equivalent"
            ),
            "case {} has unsupported expected_status {}",
            case.id,
            case.expected_status
        );
    }
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses partitioned derive outcome slices; broad corpus summary remains for manual/release sweeps"
)]
fn derive_pairs_follow_expected_outcomes() {
    let cases = load_derive_cases();
    let stats = evaluate_cases_follow_expected_outcomes(&cases);
    assert!(
        stats.generic_simplify_expected_ids.is_empty(),
        "derived cases should not expect generic simplify; use a specific derive strategy: {:?}",
        stats.generic_simplify_expected_ids
    );

    let supported_equivalent_total = stats.derived + stats.unsupported;
    let derive_reachability_rate = if supported_equivalent_total == 0 {
        0.0
    } else {
        stats.derived as f64 / supported_equivalent_total as f64
    };
    let derive_mean_step_count = if stats.derived == 0 {
        0.0
    } else {
        stats.derived_step_total as f64 / stats.derived as f64
    };
    let derive_long_path_rate = if stats.derived == 0 {
        0.0
    } else {
        stats.long_path_count as f64 / stats.derived as f64
    };

    eprintln!(
        "derive corpus summary: derived={} unsupported={} not_equivalent={}",
        stats.derived, stats.unsupported, stats.not_equivalent
    );
    eprintln!(
        "derive stats: reachability_rate={:.3} supported_equiv_rate={:.3} mean_step_count={:.2} long_path_rate={:.3}",
        derive_reachability_rate,
        derive_reachability_rate,
        derive_mean_step_count,
        derive_long_path_rate
    );
    eprintln!(
        "derive strategy specificity: generic_simplify_expected={} distinct_expected_strategies={}",
        stats.generic_simplify_expected_ids.len(),
        stats.expected_strategy_counts.len()
    );
    eprintln!(
        "derive expected-strategy-counts: {:?}",
        stats.expected_strategy_counts
    );
    eprintln!("derive derived-by-family: {:?}", stats.derived_by_family);
    eprintln!(
        "derive unsupported-equivalent-by-family: {:?}",
        stats.unsupported_by_family
    );
}

#[test]
fn derive_pairs_do_not_expect_generic_simplify_for_derived_cases() {
    let cases = load_derive_cases();
    let generic_simplify_ids = cases
        .iter()
        .filter(|case| {
            case.expected_status == "derived"
                && case.expected_strategy.as_deref() == Some("simplify")
        })
        .map(|case| case.id.clone())
        .collect::<Vec<_>>();

    assert!(
        generic_simplify_ids.is_empty(),
        "derived cases should use specific derive strategies instead of generic simplify: {:?}",
        generic_simplify_ids
    );
}

#[test]
fn derive_engine_identity_shadow_pressure_reports_reachability() {
    assert_shadow_cases_exist_in_identity_pairs(IDENTITY_SHADOW_PRESSURE_CASES);
    assert_shadow_cases_exist_in_embedded_equivalence_corpus(
        EMBEDDED_EQUIVALENCE_SHADOW_PRESSURE_CASES,
    );
    let (embedded_sampled_families, embedded_total_families, embedded_missing_families) =
        embedded_shadow_family_coverage(EMBEDDED_EQUIVALENCE_SHADOW_PRESSURE_CASES);
    assert!(
        embedded_missing_families.is_empty(),
        "derive shadow pressure must sample every embedded equivalence family; missing={:?}",
        embedded_missing_families
    );

    let mut shadow_cases = Vec::with_capacity(
        IDENTITY_SHADOW_PRESSURE_CASES.len() + EMBEDDED_EQUIVALENCE_SHADOW_PRESSURE_CASES.len(),
    );
    shadow_cases.extend_from_slice(IDENTITY_SHADOW_PRESSURE_CASES);
    shadow_cases.extend_from_slice(EMBEDDED_EQUIVALENCE_SHADOW_PRESSURE_CASES);

    let stats = evaluate_identity_shadow_pressure(&shadow_cases);
    assert_eq!(
        stats.sampled,
        stats.derived + stats.unsupported + stats.not_equivalent,
        "derive shadow pressure accounting must classify every sampled case"
    );

    let supported_equivalent_total = stats.derived + stats.unsupported;
    let reachability_rate = if supported_equivalent_total == 0 {
        0.0
    } else {
        stats.derived as f64 / supported_equivalent_total as f64
    };
    let mean_step_count = if stats.derived == 0 {
        0.0
    } else {
        stats.derived_step_total as f64 / stats.derived as f64
    };

    eprintln!(
        "derive shadow pressure summary: sampled={} derived={} unsupported={} not_equivalent={}",
        stats.sampled, stats.derived, stats.unsupported, stats.not_equivalent
    );
    eprintln!(
        "derive shadow pressure stats: reachability_rate={:.3} mean_step_count={:.2} single_step_successes={} multi_step_successes={}",
        reachability_rate,
        mean_step_count,
        stats.single_step_successes,
        stats.multi_step_successes
    );
    eprintln!(
        "derive shadow pressure strategy specificity: generic_simplify_strategy_successes={} distinct_actual_strategies={}",
        stats.generic_simplify_strategy_successes,
        stats.actual_strategy_counts.len()
    );
    eprintln!(
        "derive shadow pressure embedded-family coverage: sampled_families={} total_families={} missing={}",
        embedded_sampled_families,
        embedded_total_families,
        if embedded_missing_families.is_empty() {
            "none".to_string()
        } else {
            embedded_missing_families.join(",")
        }
    );
    eprintln!(
        "derive shadow pressure multi-step-ids: {}",
        if stats.multi_step_success_ids.is_empty() {
            "none".to_string()
        } else {
            stats.multi_step_success_ids.join(",")
        }
    );
    eprintln!(
        "derive shadow pressure generic-simplify-ids: {}",
        if stats.generic_simplify_strategy_ids.is_empty() {
            "none".to_string()
        } else {
            stats.generic_simplify_strategy_ids.join(",")
        }
    );
    assert!(
        stats.generic_simplify_strategy_ids.is_empty(),
        "derive shadow pressure should use specific strategies instead of generic simplify: {:?}",
        stats.generic_simplify_strategy_ids
    );
    eprintln!(
        "derive shadow pressure actual-strategy-counts: {:?}",
        stats.actual_strategy_counts
    );
    eprintln!(
        "derive shadow pressure derived-by-family: {:?}",
        stats.derived_by_family
    );
    eprintln!(
        "derive shadow pressure unsupported-equivalent-by-family: {:?}",
        stats.unsupported_by_family
    );
    eprintln!(
        "derive shadow pressure not-equivalent-by-family: {:?}",
        stats.not_equivalent_by_family
    );
}

#[test]
fn derive_pairs_partition_covers_corpus() {
    let cases = load_derive_cases();
    let mut assigned_ids = BTreeSet::new();
    let mut seen_buckets = BTreeSet::new();

    for case in &cases {
        seen_buckets.insert(derive_case_bucket(case));
        assert!(
            assigned_ids.insert(case.id.clone()),
            "derive partition assigned duplicate case id: {}",
            case.id
        );
    }

    assert_eq!(
        assigned_ids.len(),
        cases.len(),
        "derive partition must cover all cases"
    );
    assert_eq!(
        seen_buckets,
        BTreeSet::from(["trig", "simplify_log", "expansion_fraction", "structural",]),
        "derive partition buckets changed unexpectedly"
    );
}

#[test]
fn derive_pairs_perf_slices_cover_corpus() {
    let cases = load_derive_cases();
    let mut assigned_ids = BTreeSet::new();
    let mut seen_slices = BTreeSet::new();

    for case in &cases {
        seen_slices.insert(derive_case_perf_slice(case));
        assert!(
            assigned_ids.insert(case.id.clone()),
            "derive perf slice assigned duplicate case id: {}",
            case.id
        );
    }

    assert_eq!(
        assigned_ids.len(),
        cases.len(),
        "derive perf slices must cover all cases"
    );
    assert_eq!(
        seen_slices,
        BTreeSet::from([
            "trig",
            "simplify_log",
            "expand",
            "fractional",
            "factor_collect",
            "prep_telescoping",
            "poly_merge",
        ]),
        "derive perf slices changed unexpectedly"
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses representative trig derive cases; broad trig sweep remains for manual/release runs"
)]
fn derive_pairs_follow_expected_outcomes_trig() {
    let cases = load_derive_cases();
    let bucket = cases_in_bucket(&cases, "trig");
    assert!(!bucket.is_empty(), "trig derive bucket must not be empty");
    let _ = evaluate_cases_follow_expected_outcomes(bucket);
}

#[test]
fn derive_pairs_follow_expected_outcomes_trig_representatives() {
    let cases = load_derive_cases();
    let reps = cases_by_ids(
        &cases,
        &[
            "expand_trig_product_to_sum_sin_sin",
            "contract_trig_phase_shift_sum_to_shifted_sine",
            "expand_trig_phase_shift_general_shifted_sine_to_sum",
            "contract_trig_tan_quotient_after_arg_simplify",
            "contract_trig_tan_quotient_with_cofactor",
            "contract_trig_tan_quotient_with_additive_passthrough",
            "expand_trig_tan_to_sin_cos",
            "contract_trig_sec_reciprocal",
            "contract_trig_csc_reciprocal",
            "contract_trig_cot_quotient",
            "expand_trig_csc_reciprocal",
            "expand_trig_cot_quotient",
            "expand_trig_double_cos_as_two_cos_sq_minus_one",
            "expand_trig_double_sin_arctan_projection",
            "contract_trig_square_double_angle_sine_cosine_product",
            "expand_trig_scaled_half_angle_sine_square_to_shifted_cosine",
            "expand_trig_tangent_half_angle_substitution_sine",
            "expand_trig_cofunction_sine_minus",
            "contract_trig_half_angle_tangent",
            "expand_trig_sine_eighth_power_reduction",
            "contract_trig_triple_angle_cosine",
            "expand_trig_quadruple_angle_sine_expanded_product",
            "contract_trig_quadruple_angle_sine_expanded_product",
            "expand_trig_quadruple_angle_cosine",
            "expand_trig_angle_diff_sine",
            "contract_trig_cos_diff_sin_diff_quotient",
        ],
    );
    let _ = evaluate_cases_follow_expected_outcomes(reps);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses representative simplify/log derive cases; broad simplify/log sweep remains for manual/release runs"
)]
fn derive_pairs_follow_expected_outcomes_simplify_log() {
    let cases = load_derive_cases();
    let bucket = cases_in_bucket(&cases, "simplify_log");
    assert!(
        !bucket.is_empty(),
        "simplify_log derive bucket must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(bucket);
}

#[test]
fn derive_pairs_follow_expected_outcomes_simplify_log_representatives() {
    let cases = load_derive_cases();
    let reps = cases_by_ids(
        &cases,
        &[
            "combine_like_terms",
            "hyperbolic_pythagorean_identity_with_passthrough",
            "hyperbolic_contract_exp_decomposition",
            "hyperbolic_contract_negative_exp_decomposition",
            "hyperbolic_contract_negated_negative_exp_decomposition",
            "hyperbolic_contract_tanh_quotient",
            "inverse_tan_identity",
            "inverse_trig_arcsin_arccos_complement_sum",
            "arcsin_sin_arctan_safe_composition",
            "perfect_square_root_direct_power_to_abs",
            "square_of_square_root_requires_nonnegative",
            "perfect_square_root_to_abs_with_passthrough",
            "simplify_sqrt_arithmetic_sum",
            "simplify_sqrt_arithmetic_difference",
            "log_inverse_power_tower",
            "log_exp_inverse_ln_exp",
            "log_exp_inverse_ln_exp_power",
            "log_exp_inverse_ln_exp_product",
            "expand_log_product_with_root_cleanup",
            "expand_log_general_base_powered_two_denominator_factors_with_powered_denominator",
            "contract_general_base_logs_to_grouped_power_with_passthrough",
            "rationalize_then_cancel_to_zero",
            "radical_notable_quotient",
            "expand_odd_half_power_after_simplify_with_passthrough",
            "consecutive_factorial_ratio_gap_two",
            "reciprocal_trig_sin_csc_product_to_one",
            "reciprocal_trig_cos_sec_product_to_one",
            "sec_tan_pythagorean_to_one",
            "csc_cot_pythagorean_to_one",
            "contract_exponential_power",
            "collapse_exponential_log_product",
            "collapse_exponential_scaled_log_product",
            "expand_trig_sine_cosine_square_product_reduction",
        ],
    );
    let _ = evaluate_cases_follow_expected_outcomes(reps);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses narrower derive outcome slices for expansion and fraction families"
)]
fn derive_pairs_follow_expected_outcomes_expansion_fraction() {
    let cases = load_derive_cases();
    let bucket = cases_in_bucket(&cases, "expansion_fraction");
    assert!(
        !bucket.is_empty(),
        "expansion_fraction derive bucket must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(bucket);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses representative expand cases; broad expand sweep remains for manual/release runs"
)]
fn derive_pairs_follow_expected_outcomes_expand() {
    let cases = load_derive_cases();
    let slice = cases_in_perf_slice(&cases, "expand");
    assert!(
        !slice.is_empty(),
        "expand derive perf slice must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(slice);
}

#[test]
fn derive_pairs_follow_expected_outcomes_expand_representatives() {
    let cases = load_derive_cases();
    let reps = cases_by_ids(
        &cases,
        &[
            "expand_symbolic_binomial",
            "expand_symbolic_trinomial_square",
            "expand_sophie_germain",
            "expand_hyperbolic_sinh_sum_to_product_exact",
            "expand_trig_product_to_sum_to_cosine_difference_polynomial",
            "expand_then_cancel_to_square",
        ],
    );
    let _ = evaluate_cases_follow_expected_outcomes(reps);
}

#[test]
fn derive_pairs_follow_expected_outcomes_fractional() {
    let cases = load_derive_cases();
    let slice = cases_in_perf_slice(&cases, "fractional");
    assert!(
        !slice.is_empty(),
        "fractional derive perf slice must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(slice);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses narrower derive outcome slices for structural families"
)]
fn derive_pairs_follow_expected_outcomes_structural() {
    let cases = load_derive_cases();
    let bucket = cases_in_bucket(&cases, "structural");
    assert!(
        !bucket.is_empty(),
        "structural derive bucket must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(bucket);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses representative factor/collect derive cases; broad factor/collect sweep remains for manual/release runs"
)]
fn derive_pairs_follow_expected_outcomes_factor_collect() {
    let cases = load_derive_cases();
    let slice = cases_in_perf_slice(&cases, "factor_collect");
    assert!(
        !slice.is_empty(),
        "factor_collect derive perf slice must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(slice);
}

#[test]
fn derive_pairs_follow_expected_outcomes_factor_collect_representatives() {
    let cases = load_derive_cases();
    let reps = cases_by_ids(
        &cases,
        &[
            "collect_multiple_power_groups",
            "factor_out_with_division_quadratic",
            "factor_difference_squares",
            "factor_perfect_square_trinomial_symbolic",
            "factor_sophie_germain",
            "factor_symbolic_binomial_cube",
            "factor_symbolic_sixth_power_difference",
            "factor_full_cyclotomic_sixth_power_difference",
            "non_equivalent_mismatch",
        ],
    );
    let _ = evaluate_cases_follow_expected_outcomes(reps);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses representative prep/telescoping cases; broad sweep remains for manual/release runs"
)]
fn derive_pairs_follow_expected_outcomes_prep_telescoping() {
    let cases = load_derive_cases();
    let slice = cases_in_perf_slice(&cases, "prep_telescoping");
    assert!(
        !slice.is_empty(),
        "prep_telescoping derive perf slice must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(slice);
}

#[test]
fn derive_pairs_follow_expected_outcomes_prep_telescoping_representatives() {
    let cases = load_derive_cases();
    let reps = cases_by_ids(
        &cases,
        &[
            "finite_aggregate_sum_of_squares_symbolic_lower_bound",
            "finite_aggregate_sum_of_cubes_symbolic_lower_bound",
            "finite_aggregate_sum_geometric_power_base_two_symbolic_lower_bound",
            "finite_aggregate_product_first_integers_symbolic_lower_bound",
            "finite_aggregate_product_of_squares_symbolic_lower_bound",
            "finite_aggregate_product_of_cubes_symbolic_lower_bound",
            "finite_telescoping_sum_basic",
            "integrate_prep_morrie_basic",
            "integrate_prep_dirichlet_basic",
            "solve_prep_complete_square_symbolic_negative_linear_coeff",
            "split_telescoping_fraction_affine_symbolic_shift_gap",
            "combine_telescoping_fraction_symbolic_difference_squares_unfactored",
        ],
    );
    let _ = evaluate_cases_follow_expected_outcomes(reps);
}

#[test]
fn derive_pairs_follow_expected_outcomes_poly_merge() {
    let cases = load_derive_cases();
    let slice = cases_in_perf_slice(&cases, "poly_merge");
    assert!(
        !slice.is_empty(),
        "poly_merge derive perf slice must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(slice);
}

#[test]
fn derive_pairs_shape_clusters_stay_within_budget() {
    let cases = load_derive_cases();
    let mut violations = Vec::new();
    violations.extend(shape_budget_violations(
        &cases,
        SignatureMode::Conservative,
        2,
    ));
    violations.extend(shape_budget_violations(&cases, SignatureMode::Abstract, 2));

    assert!(
        violations.is_empty(),
        "derive corpus inflation guardrail failed:\n{}",
        violations.join("\n")
    );
}
