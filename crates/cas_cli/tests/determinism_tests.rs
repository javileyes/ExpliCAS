//! Determinism Tests
//!
//! These tests verify that the CAS engine produces deterministic output.
//! Non-determinism is a critical bug that breaks CI, benchmarks, and user trust.

use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

/// Test that rationalization produces identical output across 200 runs.
/// This catches HashMap iteration order issues, non-stable sorting, etc.
#[test]
fn test_determinism_rationalize_200x() {
    let expr_str = "1/(sqrt(3) + 1 + sqrt(2))";

    let mut first_result: Option<String> = None;

    for i in 0..200 {
        // Create fresh simplifier each time to catch initialization order issues
        let mut simplifier = Simplifier::with_default_rules();
        let expr = parse(expr_str, &mut simplifier.context).expect("parse failed");
        let (result, _) = simplifier.simplify(expr);
        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );

        if let Some(ref first) = first_result {
            assert_eq!(
                &result_str, first,
                "Non-determinism detected at iteration {}!\nFirst: {}\nNow: {}",
                i, first, result_str
            );
        } else {
            first_result = Some(result_str);
        }
    }
}

/// Test determinism of simplify on various expressions
#[test]
fn test_determinism_simplify_mixed() {
    let expressions = [
        "sqrt(12) + sqrt(27)",
        "(x-1)*(x+1)*(x^2+1)",
        "sin(x)^2 + cos(x)^2",
        "x/(2*(1+sqrt(2)))",
        "(a+b)^2 - (a-b)^2",
    ];

    for expr_str in &expressions {
        let mut first_result: Option<String> = None;

        for i in 0..50 {
            let mut simplifier = Simplifier::with_default_rules();
            let expr = parse(expr_str, &mut simplifier.context).expect("parse failed");
            let (result, _) = simplifier.simplify(expr);
            let result_str = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: result
                }
            );

            if let Some(ref first) = first_result {
                assert_eq!(
                    &result_str, first,
                    "Non-determinism in '{}' at iteration {}!\nFirst: {}\nNow: {}",
                    expr_str, i, first, result_str
                );
            } else {
                first_result = Some(result_str);
            }
        }
    }
}

/// Test that rationalization with explicit command is deterministic
#[test]
fn test_determinism_rationalize_explicit() {
    let expr_str = "rationalize(1/(1 + sqrt(2) + sqrt(3)))";

    let mut first_result: Option<String> = None;

    for i in 0..100 {
        let mut simplifier = Simplifier::with_default_rules();
        let expr = parse(expr_str, &mut simplifier.context).expect("parse failed");
        let (result, _) = simplifier.simplify(expr);
        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );

        if let Some(ref first) = first_result {
            assert_eq!(
                &result_str, first,
                "Non-determinism in rationalize() at iteration {}!\nFirst: {}\nNow: {}",
                i, first, result_str
            );
        } else {
            first_result = Some(result_str);
        }
    }
}

/// Test determinism after accumulated state (simulates REPL session)
/// This reproduces the bug seen in run_cli_tests.sh where the same expression
/// produces different results depending on prior expressions processed.
#[test]
fn test_determinism_repl_accumulated_state() {
    // Expressions to process before the target (simulating REPL history)
    let warmup_exprs = [
        "sqrt(12)",
        "1/sqrt(2)",
        "1/(1 + sqrt(2))",
        "x/(2*(1+sqrt(2)))",
        "1/(3 - 2*sqrt(5))",
    ];

    let target_expr = "1/(sqrt(3) + 1 + sqrt(2))";
    let mut simplifier = Simplifier::with_default_rules();

    // Warmup: process expressions to accumulate context state
    for expr_str in &warmup_exprs {
        if let Ok(expr) = parse(expr_str, &mut simplifier.context) {
            let _ = simplifier.simplify(expr);
        }
    }

    // First run of target expression
    let expr1 = parse(target_expr, &mut simplifier.context).expect("parse failed");
    let (result1, _) = simplifier.simplify(expr1);
    let result1_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result1
        }
    );

    // Process more expressions (same warmup again, simulating REPL activity)
    for expr_str in &warmup_exprs {
        if let Ok(expr) = parse(expr_str, &mut simplifier.context) {
            let _ = simplifier.simplify(expr);
        }
    }

    // Second run of SAME target expression
    let expr2 = parse(target_expr, &mut simplifier.context).expect("parse failed");
    let (result2, _) = simplifier.simplify(expr2);
    let result2_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result2
        }
    );

    // MUST be identical
    assert_eq!(
        result1_str, result2_str,
        "Non-determinism in REPL accumulated state!\nFirst run: {}\nSecond run: {}",
        result1_str, result2_str
    );
}
