//! Contract test: JSON output never contains __hold.
//!
//! This test ensures that __hold wrappers are always stripped before
//! returning results to users via JSON (CLI and FFI).

use cas_engine::phase::ExpandPolicy;
use cas_engine::EvalOptions;
use cas_engine::Simplifier;
use cas_parser::parse;

/// Test that expand() result doesn't leak __hold in JSON/string output
#[test]
fn test_expand_no_hold_leak() {
    let mut ctx = cas_ast::Context::new();
    let mut simplifier = Simplifier::new();
    std::mem::swap(&mut simplifier.context, &mut ctx);

    // This expression historically put __hold in output
    let expr = parse("expand((x+y)^2) - x^2 - y^2", &mut simplifier.context).unwrap();

    // Simplify with default options
    let opts = EvalOptions::default();
    let simplify_opts = opts.to_simplify_options();
    let (result, _steps, _stats) = simplifier.simplify_with_stats(expr, simplify_opts);

    // Format as string (same path as REPL display)
    let result_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    assert!(
        !result_str.contains("__hold"),
        "Result should not contain __hold: {}",
        result_str
    );

    // Should simplify to 2xy
    assert!(
        result_str.contains("2") && result_str.contains("x") && result_str.contains("y"),
        "Expected 2xy, got: {}",
        result_str
    );
}

/// Test that autoexpand result doesn't leak __hold
#[test]
fn test_autoexpand_no_hold_leak() {
    let opts = EvalOptions {
        shared: cas_engine::phase::SharedSemanticConfig {
            expand_policy: ExpandPolicy::Auto,
            ..Default::default()
        },
        ..EvalOptions::default()
    };
    let mut simplifier = Simplifier::with_profile(&opts);

    // This expression triggers autoexpand
    let expr = parse("(x+y)^2 - x^2 - y^2", &mut simplifier.context).unwrap();

    let simplify_opts = opts.to_simplify_options();
    let (result, _steps, _stats) = simplifier.simplify_with_stats(expr, simplify_opts);

    let result_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    assert!(
        !result_str.contains("__hold"),
        "Autoexpand result should not contain __hold: {}",
        result_str
    );
}

/// Test that factor() result doesn't leak __hold
#[test]
fn test_factor_no_hold_leak() {
    let mut simplifier = Simplifier::new();

    // factor() wraps result in __hold to prevent DifferenceOfSquaresRule undo
    let expr = parse("factor(x^2 - 1)", &mut simplifier.context).unwrap();

    let (result, _steps, _stats) = simplifier.simplify_with_stats(expr, Default::default());

    let result_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    assert!(
        !result_str.contains("__hold"),
        "Factor result should not contain __hold: {}",
        result_str
    );
}
