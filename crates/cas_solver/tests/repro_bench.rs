use cas_ast::count_nodes;
use cas_ast::{BoundType, Constant, Equation, Expr, Interval, RelOp, SolutionSet};
use cas_parser::parse;
use cas_solver::api::{verify_solution_set, VerifyStatus, VerifySummary};
use cas_solver::runtime::{Simplifier, StepsMode};
use std::time::Instant;

fn setup_bench(input_str: &str) -> (Simplifier, cas_ast::ExprId) {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input_str, &mut simplifier.context).unwrap();
    (simplifier, expr)
}

fn build_sum_fractions_input(max_shift: usize) -> String {
    let mut s = "1/x".to_string();
    for i in 1..=max_shift {
        s.push_str(&format!(" + 1/(x+{})", i));
    }
    s
}

fn build_verification_case_with_discrete_solutions(
    lhs: &str,
    rhs: &str,
    var: &str,
    solutions: &[&str],
) -> (Simplifier, Equation, String, SolutionSet) {
    let mut simplifier = Simplifier::with_default_rules();
    let equation = Equation {
        lhs: parse(lhs, &mut simplifier.context).expect("lhs parse failed"),
        rhs: parse(rhs, &mut simplifier.context).expect("rhs parse failed"),
        op: RelOp::Eq,
    };
    let solutions = SolutionSet::Discrete(
        solutions
            .iter()
            .map(|expr| parse(expr, &mut simplifier.context).expect("solution parse failed"))
            .collect(),
    );
    (simplifier, equation, var.to_string(), solutions)
}

fn build_verification_case_with_positive_interval(
    lhs: &str,
    rhs: &str,
    var: &str,
) -> (Simplifier, Equation, String, SolutionSet) {
    let mut simplifier = Simplifier::with_default_rules();
    let equation = Equation {
        lhs: parse(lhs, &mut simplifier.context).expect("lhs parse failed"),
        rhs: parse(rhs, &mut simplifier.context).expect("rhs parse failed"),
        op: RelOp::Eq,
    };
    let zero = simplifier.context.num(0);
    let infinity = simplifier.context.add(Expr::Constant(Constant::Infinity));
    let solutions = SolutionSet::Continuous(Interval::open(zero, infinity));
    (simplifier, equation, var.to_string(), solutions)
}

fn build_verification_case_with_nonzero_union(
    lhs: &str,
    rhs: &str,
    var: &str,
) -> (Simplifier, Equation, String, SolutionSet) {
    let mut simplifier = Simplifier::with_default_rules();
    let equation = Equation {
        lhs: parse(lhs, &mut simplifier.context).expect("lhs parse failed"),
        rhs: parse(rhs, &mut simplifier.context).expect("rhs parse failed"),
        op: RelOp::Eq,
    };
    let zero = simplifier.context.num(0);
    let infinity = simplifier.context.add(Expr::Constant(Constant::Infinity));
    let neg_infinity = simplifier.context.add(Expr::Neg(infinity));
    let solutions = SolutionSet::Union(vec![
        Interval {
            min: neg_infinity,
            min_type: BoundType::Open,
            max: zero,
            max_type: BoundType::Open,
        },
        Interval {
            min: zero,
            min_type: BoundType::Open,
            max: infinity,
            max_type: BoundType::Open,
        },
    ]);
    (simplifier, equation, var.to_string(), solutions)
}

fn run_bench<F>(name: &str, iterations: u32, mut f: F)
where
    F: FnMut(),
{
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let duration = start.elapsed();
    println!("{} ({} runs): {:?}", name, iterations, duration);
    println!("Average: {:?}", duration / iterations);
}

#[test]
#[ignore = "benchmark-style repro; run manually when investigating performance"]
fn repro_sum_fractions_10() {
    run_bench("sum_fractions_10", 10, || {
        let s = build_sum_fractions_input(10);
        let (mut simplifier, input) = setup_bench(&s);
        simplifier.simplify(input);
    });
}

#[test]
#[ignore = "benchmark-style repro; run manually when investigating performance"]
fn repro_expand_binomial_power_10() {
    run_bench("expand_binomial_power_10", 100, || {
        let (mut simplifier, input) = setup_bench("(x+1)^10");
        simplifier.simplify(input);
    });
}

#[test]
#[ignore = "benchmark-style repro; run manually when investigating performance"]
fn repro_diff_nested_trig_exp() {
    run_bench("diff_nested_trig_exp", 100, || {
        let (mut simplifier, input) = setup_bench("diff(exp(sin(x^2)), x)");
        simplifier.simplify(input);
    });
}

#[test]
#[ignore = "benchmark-style repro; run manually when investigating verification performance"]
fn repro_verify_quadratic_two_roots_100() {
    run_bench("verify_quadratic_two_roots", 100, || {
        let (mut simplifier, equation, var, solutions) =
            build_verification_case_with_discrete_solutions("x^2 - 5*x + 6", "0", "x", &["2", "3"]);
        let result = verify_solution_set(&mut simplifier, &equation, &var, &solutions);
        assert!(matches!(result.summary, VerifySummary::AllVerified));
    });
}

#[test]
#[ignore = "benchmark-style repro; run manually when investigating verification performance"]
fn repro_verify_failed_discrete_with_hint_100() {
    run_bench("verify_failed_discrete_with_hint", 100, || {
        let (mut simplifier, equation, var, solutions) =
            build_verification_case_with_discrete_solutions("a*x", "1", "x", &["1"]);
        let result = verify_solution_set(&mut simplifier, &equation, &var, &solutions);
        assert!(matches!(result.summary, VerifySummary::NoneVerified));
        assert!(matches!(
            result.solutions.first(),
            Some((
                _,
                VerifyStatus::Unverifiable {
                    counterexample_hint: Some(_),
                    ..
                }
            ))
        ));
    });
}

#[test]
#[ignore = "benchmark-style repro; run manually when investigating verification performance"]
fn repro_verify_failed_discrete_log_hint_suppressed_100() {
    run_bench("verify_failed_discrete_log_hint_suppressed", 100, || {
        let (mut simplifier, equation, var, solutions) =
            build_verification_case_with_discrete_solutions("ln(a*x)", "1", "x", &["1"]);
        let result = verify_solution_set(&mut simplifier, &equation, &var, &solutions);
        assert!(matches!(result.summary, VerifySummary::NoneVerified));
        assert!(matches!(
            result.solutions.first(),
            Some((
                _,
                VerifyStatus::Unverifiable {
                    counterexample_hint: None,
                    ..
                }
            ))
        ));
    });
}

#[test]
#[ignore = "benchmark-style repro; run manually when investigating verification performance"]
fn repro_verify_needs_sampling_positive_interval_100() {
    run_bench("verify_needs_sampling_positive_interval", 100, || {
        let (mut simplifier, equation, var, solutions) =
            build_verification_case_with_positive_interval("0^x", "0", "x");
        let result = verify_solution_set(&mut simplifier, &equation, &var, &solutions);
        assert!(matches!(result.summary, VerifySummary::NeedsSampling));
    });
}

#[test]
#[ignore = "benchmark-style repro; run manually when investigating verification performance"]
fn repro_verify_needs_sampling_nonzero_union_100() {
    run_bench("verify_needs_sampling_nonzero_union", 100, || {
        let (mut simplifier, equation, var, solutions) =
            build_verification_case_with_nonzero_union("x", "0", "x");
        let result = verify_solution_set(&mut simplifier, &equation, &var, &solutions);
        assert!(matches!(result.summary, VerifySummary::NeedsSampling));
    });
}

#[test]
fn repro_sum_fractions_10_structural_regression_guard() {
    let input = build_sum_fractions_input(10);
    let (mut simplifier, expr) = setup_bench(&input);
    let (result, timeline) = simplifier.simplify(expr);
    let rendered = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result,
        }
    );

    assert!(
        timeline.len() <= 32,
        "sum_fractions_10 should stay on the cheap path; got {} steps",
        timeline.len()
    );
    assert!(
        simplifier.context.nodes.len() <= 2_000,
        "sum_fractions_10 should avoid pathological node growth; got {} nodes",
        simplifier.context.nodes.len()
    );
    assert!(
        rendered.len() <= 512,
        "sum_fractions_10 should keep the simplified output compact; got {} chars",
        rendered.len()
    );
}

#[test]
fn verify_failed_discrete_hint_structural_regression_guard() {
    let (mut simplifier, equation, var, solutions) =
        build_verification_case_with_discrete_solutions("a*x", "1", "x", &["1"]);
    let result = verify_solution_set(&mut simplifier, &equation, &var, &solutions);

    assert!(matches!(result.summary, VerifySummary::NoneVerified));
    assert!(
        matches!(
            result.solutions.first(),
            Some((
                _,
                VerifyStatus::Unverifiable {
                    counterexample_hint: Some(_),
                    ..
                }
            ))
        ),
        "expected failed discrete verification with counterexample hint"
    );
    assert!(
        simplifier.context.nodes.len() <= 1_000,
        "failed discrete verification with hint should avoid pathological node growth; got {} nodes",
        simplifier.context.nodes.len()
    );
}

#[test]
fn verify_failed_discrete_hint_steps_on_structural_regression_guard() {
    let (mut simplifier, equation, var, solutions) =
        build_verification_case_with_discrete_solutions("a*x", "1", "x", &["1"]);
    simplifier.set_steps_mode(StepsMode::On);
    let result = verify_solution_set(&mut simplifier, &equation, &var, &solutions);

    assert!(matches!(result.summary, VerifySummary::NoneVerified));
    assert!(
        matches!(
            result.solutions.first(),
            Some((
                _,
                VerifyStatus::Unverifiable {
                    counterexample_hint: Some(_),
                    ..
                }
            ))
        ),
        "expected failed discrete verification with counterexample hint"
    );
    assert_eq!(
        simplifier.get_steps_mode(),
        StepsMode::On,
        "verification should preserve caller steps mode"
    );
    assert!(
        simplifier.context.nodes.len() <= 1_500,
        "failed discrete verification with steps on should avoid pathological node growth; got {} nodes",
        simplifier.context.nodes.len()
    );
}

#[test]
fn verify_failed_discrete_log_hint_suppressed_steps_on_structural_regression_guard() {
    let (mut simplifier, equation, var, solutions) =
        build_verification_case_with_discrete_solutions("ln(a*x)", "1", "x", &["1"]);
    simplifier.set_steps_mode(StepsMode::On);
    let result = verify_solution_set(&mut simplifier, &equation, &var, &solutions);

    assert!(matches!(result.summary, VerifySummary::NoneVerified));
    assert!(
        matches!(
            result.solutions.first(),
            Some((
                _,
                VerifyStatus::Unverifiable {
                    counterexample_hint: None,
                    ..
                }
            ))
        ),
        "expected failed discrete verification with suppressed counterexample hint"
    );
    assert_eq!(
        simplifier.get_steps_mode(),
        StepsMode::On,
        "verification should preserve caller steps mode"
    );
    assert!(
        simplifier.context.nodes.len() <= 1_500,
        "failed discrete log verification with steps on should avoid pathological node growth; got {} nodes",
        simplifier.context.nodes.len()
    );
}

#[test]
fn verify_positive_interval_needs_sampling_steps_on_structural_regression_guard() {
    let (mut simplifier, equation, var, solutions) =
        build_verification_case_with_positive_interval("0^x", "0", "x");
    simplifier.set_steps_mode(StepsMode::On);
    let result = verify_solution_set(&mut simplifier, &equation, &var, &solutions);

    assert!(matches!(result.summary, VerifySummary::NeedsSampling));
    assert_eq!(
        result.guard_description.as_deref(),
        Some("verification requires numeric sampling (solution set matches guard `x > 0`)")
    );
    assert_eq!(
        simplifier.get_steps_mode(),
        StepsMode::On,
        "verification should preserve caller steps mode"
    );
    assert!(
        simplifier.context.nodes.len() <= 1_000,
        "positive-interval verification with steps on should avoid pathological node growth; got {} nodes",
        simplifier.context.nodes.len()
    );
}

#[test]
fn verify_nonzero_union_needs_sampling_structural_regression_guard() {
    let (mut simplifier, equation, var, solutions) =
        build_verification_case_with_nonzero_union("x", "0", "x");
    let result = verify_solution_set(&mut simplifier, &equation, &var, &solutions);

    assert!(matches!(result.summary, VerifySummary::NeedsSampling));
    assert_eq!(
        result.guard_description.as_deref(),
        Some("verification requires numeric sampling (solution set matches guard `x != 0`)")
    );
    assert!(
        simplifier.context.nodes.len() <= 1_000,
        "nonzero-union verification should avoid pathological node growth; got {} nodes",
        simplifier.context.nodes.len()
    );
}

#[test]
fn verify_nonzero_union_needs_sampling_steps_on_structural_regression_guard() {
    let (mut simplifier, equation, var, solutions) =
        build_verification_case_with_nonzero_union("x", "0", "x");
    simplifier.set_steps_mode(StepsMode::On);
    let result = verify_solution_set(&mut simplifier, &equation, &var, &solutions);

    assert!(matches!(result.summary, VerifySummary::NeedsSampling));
    assert_eq!(
        result.guard_description.as_deref(),
        Some("verification requires numeric sampling (solution set matches guard `x != 0`)")
    );
    assert_eq!(
        simplifier.get_steps_mode(),
        StepsMode::On,
        "verification should preserve caller steps mode"
    );
    assert!(
        simplifier.context.nodes.len() <= 1_000,
        "nonzero-union verification with steps on should avoid pathological node growth; got {} nodes",
        simplifier.context.nodes.len()
    );
}

#[test]
fn verify_quadratic_two_roots_steps_on_structural_regression_guard() {
    let (mut simplifier, equation, var, solutions) =
        build_verification_case_with_discrete_solutions("x^2 - 5*x + 6", "0", "x", &["2", "3"]);
    simplifier.set_steps_mode(StepsMode::On);
    let result = verify_solution_set(&mut simplifier, &equation, &var, &solutions);

    assert!(
        matches!(result.summary, VerifySummary::AllVerified),
        "expected quadratic verification to stay fully verified"
    );
    assert_eq!(
        simplifier.get_steps_mode(),
        StepsMode::On,
        "verification should preserve caller steps mode"
    );
    assert_eq!(
        result.solutions.len(),
        2,
        "quadratic verification should preserve both discrete roots"
    );
    assert!(
        simplifier.context.nodes.len() <= 1_000,
        "quadratic verification with steps on should avoid pathological node growth; got {} nodes",
        simplifier.context.nodes.len()
    );
}

#[test]
#[ignore = "diagnostic: rule profile for sum_fractions scaling"]
fn diagnose_sum_fractions_scaling() {
    for n in 2..=10 {
        let input = build_sum_fractions_input(n);
        let (mut simplifier, expr) = setup_bench(&input);
        let start = Instant::now();
        let (result, timeline) = simplifier.simplify(expr);
        let elapsed = start.elapsed();
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
        );
        eprintln!(
            "n={n} elapsed={elapsed:?} steps={} nodes={} result_len={}",
            timeline.len(),
            simplifier.context.nodes.len(),
            rendered.len()
        );
    }
}

#[test]
#[ignore = "diagnostic: rule profile for pathological fraction sum"]
fn diagnose_sum_fractions_7_profile() {
    let input = build_sum_fractions_input(7);
    let (mut simplifier, expr) = setup_bench(&input);
    simplifier.profiler.enable_health();
    let start = Instant::now();
    let (result, timeline) = simplifier.simplify(expr);
    let elapsed = start.elapsed();

    eprintln!(
        "Result: {}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result,
        }
    );
    eprintln!("Elapsed: {elapsed:?}");
    eprintln!("Timeline steps: {}", timeline.len());
    eprintln!("Total nodes created: {}", simplifier.context.nodes.len());
    for (idx, step) in timeline.iter().enumerate() {
        eprintln!(
            "  step#{idx:02} rule={:<32} before_nodes={:>4} after_nodes={:>4}",
            step.rule_name,
            count_nodes(&simplifier.context, step.before),
            count_nodes(&simplifier.context, step.after)
        );
    }
    eprintln!("\n=== RULE PROFILING REPORT ===");
    eprintln!("{}", simplifier.profiler.report());
    eprintln!("\n=== HEALTH REPORT ===");
    eprintln!("{}", simplifier.profiler.health_report());
}

#[test]
#[ignore = "diagnostic: ablation study for pathological fraction sum"]
fn diagnose_sum_fractions_7_rule_ablation() {
    let scenarios: &[(&str, &[&str])] =
        &[("baseline", &[]), ("no_add_fractions", &["Add Fractions"])];

    for (label, disabled) in scenarios {
        let input = build_sum_fractions_input(7);
        let (mut simplifier, expr) = setup_bench(&input);
        for rule in *disabled {
            simplifier.disable_rule(rule);
        }
        let start = Instant::now();
        let (result, timeline) = simplifier.simplify(expr);
        let elapsed = start.elapsed();
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
        );
        eprintln!(
            "{label}: disabled={disabled:?} elapsed={elapsed:?} steps={} nodes={} result_len={}",
            timeline.len(),
            simplifier.context.nodes.len(),
            rendered.len()
        );
    }
}
