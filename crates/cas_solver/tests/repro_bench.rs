use cas_ast::count_nodes;
use cas_parser::parse;
use cas_solver::runtime::Simplifier;
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
fn repro_sum_fractions_10() {
    run_bench("sum_fractions_10", 10, || {
        let s = build_sum_fractions_input(10);
        let (mut simplifier, input) = setup_bench(&s);
        simplifier.simplify(input);
    });
}

#[test]
fn repro_expand_binomial_power_10() {
    run_bench("expand_binomial_power_10", 100, || {
        let (mut simplifier, input) = setup_bench("(x+1)^10");
        simplifier.simplify(input);
    });
}

#[test]
fn repro_diff_nested_trig_exp() {
    run_bench("diff_nested_trig_exp", 100, || {
        let (mut simplifier, input) = setup_bench("diff(exp(sin(x^2)), x)");
        simplifier.simplify(input);
    });
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
