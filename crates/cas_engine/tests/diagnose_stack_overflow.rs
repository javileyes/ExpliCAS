//! Diagnostic test to identify which rules fire excessively in problematic expressions
//!
//! This test should be run with: cargo test --test diagnose_stack_overflow -- --nocapture

use cas_engine::Simplifier;
use cas_parser::parse;

/// Run the problematic expression with profiler to identify high-frequency rules
#[test]
fn diagnose_rule_firing_counts() {
    // The simplest expression that triggers very deep recursion
    let expr_str = "sin((3 - x + sin(x))^4)";

    eprintln!("=== Diagnosing expression: {} ===", expr_str);
    eprintln!();

    let mut simplifier = Simplifier::with_default_rules();

    // Enable profiler with health metrics
    simplifier.profiler.enable_health();

    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    // Run simplification
    let (result, timeline) = simplifier.simplify(expr);

    // Print result
    eprintln!(
        "Result: {}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result,
        }
    );
    eprintln!();

    // Print timeline step count
    eprintln!("Timeline steps: {}", timeline.len());
    eprintln!();

    // Print profiler report
    eprintln!("=== RULE PROFILING REPORT ===");
    eprintln!("{}", simplifier.profiler.report());

    // Print health report with growth metrics
    eprintln!();
    eprintln!("=== HEALTH REPORT ===");
    eprintln!("{}", simplifier.profiler.health_report());

    // Print total applied
    eprintln!();
    eprintln!(
        "Total rules applied: {}",
        simplifier.profiler.total_applied()
    );

    // Print node counts
    eprintln!("Total nodes created: {}", simplifier.context.nodes.len());
}

/// Run with sin^2 + cos^2 + the problematic term  
#[test]
fn diagnose_pythagorean_plus_sin_pow() {
    let expr_str = "(sin(x)^2 + cos(x)^2) + sin((3 - x + sin(x))^4)^2";

    eprintln!("=== Diagnosing expression: {} ===", expr_str);
    eprintln!();

    let mut simplifier = Simplifier::with_default_rules();
    simplifier.profiler.enable_health();

    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");
    let (result, timeline) = simplifier.simplify(expr);

    eprintln!(
        "Result: {}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result,
        }
    );
    eprintln!("Timeline steps: {}", timeline.len());
    eprintln!();
    eprintln!("=== RULE PROFILING REPORT ===");
    eprintln!("{}", simplifier.profiler.report());
    eprintln!();
    eprintln!(
        "Total rules applied: {}",
        simplifier.profiler.total_applied()
    );
    eprintln!("Total nodes created: {}", simplifier.context.nodes.len());
}
