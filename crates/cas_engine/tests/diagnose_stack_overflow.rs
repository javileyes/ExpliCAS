//! Diagnostic test to identify which function causes stack overflow with depth>=50
//!
//! This test should be run with: cargo test --test diagnose_stack_overflow -- --nocapture

use cas_engine::Simplifier;
use cas_parser::parse;

/// Run the problematic expression with MAX_SIMPLIFY_DEPTH=50 equivalent
/// and capture which function exceeds its recursion guard.
#[test]
#[ignore] // Run manually with: cargo test --test diagnose_stack_overflow -- --ignored --nocapture
fn diagnose_stack_overflow_depth_50() {
    use cas_engine::recursion_guard::{get_all_max_depths, reset_all_guards};

    reset_all_guards();

    // The expression that crashes with depth>=50
    let expr_str = "sin((3 - x + sin(x))^4)^2";

    eprintln!("Testing expression: {}", expr_str);
    eprintln!("This test is designed to panic with a backtrace.");
    eprintln!("If it panics, the backtrace will show which function exceeded its limit.");
    eprintln!();

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    // This will crash if depth>=50 causes issues - check the panic message
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| simplifier.simplify(expr)));

    // Print max depths seen
    eprintln!("\nMaximum depths observed:");
    let mut depths: Vec<_> = get_all_max_depths();
    depths.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by depth descending
    for (label, depth) in depths {
        eprintln!("  {}: {}", label, depth);
    }

    match result {
        Ok(_) => eprintln!("\nSimplification completed without panic."),
        Err(e) => {
            if let Some(s) = e.downcast_ref::<&str>() {
                eprintln!("\nPanicked with: {}", s);
            } else if let Some(s) = e.downcast_ref::<String>() {
                eprintln!("\nPanicked with: {}", s);
            } else {
                eprintln!("\nPanicked with unknown error type");
            }
        }
    }
}
