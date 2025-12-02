use cas_engine::Simplifier;
use cas_parser::parse;
use std::time::Instant;

fn setup_bench(input_str: &str) -> (Simplifier, cas_ast::ExprId) {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input_str, &mut simplifier.context).unwrap();
    (simplifier, expr)
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
        let mut s = "1/x".to_string();
        for i in 1..=10 {
            s.push_str(&format!(" + 1/(x+{})", i));
        }
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
