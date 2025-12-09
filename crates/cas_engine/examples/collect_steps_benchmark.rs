/// Test to compare performance with collect_steps enabled vs disabled
/// This demonstrates that step collection has minimal impact on core simplification
use cas_parser::parse;
use std::time::Instant;

fn benchmark_collect_steps_impact() {
    let expressions = [
        "sin(x)^2 + cos(x)^2",
        "(x + 1)^5",
        "1/x + 1/(x+1)",
        "sqrt(x * (x * x^(1/4))^(1/3)) - x^(1/2 + 1/6 + 1/24)",
        "diff(sin(x) * cos(x), x)",
    ];

    let iterations = 100;

    println!("\n=== collect_steps Performance Comparison ===\n");
    println!(
        "{:<50} {:>12} {:>12} {:>10}",
        "Expression", "steps=true", "steps=false", "Overhead"
    );
    println!("{}", "-".repeat(90));

    for expr_str in &expressions {
        // Warmup
        {
            let mut simplifier = cas_engine::Simplifier::with_default_rules();
            let expr = parse(expr_str, &mut simplifier.context).unwrap();
            let _ = simplifier.simplify(expr);
        }

        // Benchmark with collect_steps = true
        let time_with_steps = {
            let start = Instant::now();
            for _ in 0..iterations {
                let mut simplifier = cas_engine::Simplifier::with_default_rules();
                simplifier.collect_steps = true;
                let expr = parse(expr_str, &mut simplifier.context).unwrap();
                let _ = simplifier.simplify(expr);
            }
            start.elapsed()
        };

        // Benchmark with collect_steps = false
        let time_without_steps = {
            let start = Instant::now();
            for _ in 0..iterations {
                let mut simplifier = cas_engine::Simplifier::with_default_rules();
                simplifier.collect_steps = false;
                let expr = parse(expr_str, &mut simplifier.context).unwrap();
                let _ = simplifier.simplify(expr);
            }
            start.elapsed()
        };

        let overhead_pct = if time_without_steps.as_nanos() > 0 {
            ((time_with_steps.as_nanos() as f64 / time_without_steps.as_nanos() as f64) - 1.0)
                * 100.0
        } else {
            0.0
        };

        let truncated = if expr_str.len() > 48 {
            format!("{}...", &expr_str[..45])
        } else {
            expr_str.to_string()
        };

        println!(
            "{:<50} {:>10.2}ms {:>10.2}ms {:>+9.1}%",
            truncated,
            time_with_steps.as_secs_f64() * 1000.0,
            time_without_steps.as_secs_f64() * 1000.0,
            overhead_pct
        );
    }

    println!("\n✓ collect_steps=false should be close to collect_steps=true (low overhead)");
    println!("✓ The step post-processing only runs when collect_steps=true\n");
}

fn main() {
    benchmark_collect_steps_impact();
}
