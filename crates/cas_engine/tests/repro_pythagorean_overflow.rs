//! Reproduction test for stack overflow in Pythagorean identity with nested sin powers.
//!
//! The problematic expression is:
//!   (sin(x)^2 + cos(x)^2) + (sin(((3) - (x)) + (sin(x)))^(4))^(2)
//!
//! Which simplifies to:
//!   1 + sin((3 - x) + sin(x))^8
//!
//! The issue seems to be in how sin with power expressions are handled.

use cas_engine::Simplifier;
use cas_parser::parse;

/// Minimal reproduction of the stack overflow.
/// Run with: cargo test -p cas_engine --test repro_pythagorean_overflow -- --nocapture
#[test]
fn repro_pythagorean_stack_overflow() {
    // The expression that caused the crash at iter=6
    let expr_str = "(sin(x)^2 + cos(x)^2) + ((sin((((3) - (x)) + (sin(x)))^(4)))^(2))";

    eprintln!("Testing expression: {}", expr_str);

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse expression");

    eprintln!("Parsed, now simplifying...");

    // This should not cause stack overflow
    let (result, _timeline) = simplifier.simplify(expr);

    eprintln!("Result: {:?}", simplifier.context.get(result));
    eprintln!("Test passed!");
}

/// Even simpler: just the nested sin power part
#[test]
fn repro_nested_sin_power() {
    // Isolate just the problematic part: sin of a power
    let expr_str = "sin(((3) - (x)) + (sin(x)))^4";

    eprintln!("Testing: {}", expr_str);

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    // Measure AST depth before simplification
    let depth = measure_depth(&simplifier.context, expr);
    eprintln!("AST depth before simplify: {}", depth);

    let (result, _) = simplifier.simplify(expr);

    let depth_after = measure_depth(&simplifier.context, result);
    eprintln!("AST depth after simplify: {}", depth_after);
    eprintln!("Result: {:?}", simplifier.context.get(result));
}

/// Measure AST depth
fn measure_depth(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> usize {
    use cas_ast::Expr;
    match ctx.get(expr) {
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) => 1,
        Expr::Neg(a) => 1 + measure_depth(ctx, *a),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            1 + measure_depth(ctx, *a).max(measure_depth(ctx, *b))
        }
        Expr::Function(_, args) => {
            1 + args
                .iter()
                .map(|a| measure_depth(ctx, *a))
                .max()
                .unwrap_or(0)
        }
        Expr::Matrix { data, .. } => {
            1 + data
                .iter()
                .map(|e| measure_depth(ctx, *e))
                .max()
                .unwrap_or(0)
        }
        _ => 1, // SessionRef or any other leaf types
    }
}

/// Test sin^2 + cos^2 without the extra term
#[test]
fn repro_simple_pythagorean() {
    let expr_str = "sin(x)^2 + cos(x)^2";

    eprintln!("Testing: {}", expr_str);

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    let (result, _) = simplifier.simplify(expr);
    eprintln!("Result: {:?}", simplifier.context.get(result));
}

/// Test just sin of a sum - might trigger angle expansion
#[test]
fn repro_sin_of_sum() {
    // This might trigger sin(a+b) = sin(a)cos(b) + cos(a)sin(b) expansion
    let expr_str = "sin((3 - x) + sin(x))";

    eprintln!("Testing: {}", expr_str);

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    let (result, _) = simplifier.simplify(expr);
    eprintln!("Result: {:?}", simplifier.context.get(result));
}

/// Test: sin²+cos² plus sin(simple)^2
#[test]
fn repro_pythagorean_plus_sin_squared() {
    let expr_str = "(sin(x)^2 + cos(x)^2) + sin(x-1)^2";

    eprintln!("Testing: {}", expr_str);

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    let (result, _) = simplifier.simplify(expr);
    eprintln!("Result: {:?}", simplifier.context.get(result));
}

/// Test: 1 + the problematic term (skipping sin²+cos² simplification)
#[test]
fn repro_one_plus_sin_power() {
    let expr_str = "1 + sin(((3) - (x)) + (sin(x)))^8";

    eprintln!("Testing: {}", expr_str);

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    let (result, _) = simplifier.simplify(expr);
    eprintln!("Result: {:?}", simplifier.context.get(result));
}

/// Variant with sin(a+b) where a has sin and ^4 power
#[test]
fn repro_pythagorean_plus_sin_sum_inner_pow4() {
    // This is closer to the original - the ^(4) is INSIDE the sin arg
    let expr_str = "(sin(x)^2 + cos(x)^2) + sin((3 - x + sin(x))^4)^2";

    eprintln!("Testing: {}", expr_str);

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    let (result, _) = simplifier.simplify(expr);
    eprintln!("Result: {:?}", simplifier.context.get(result));
}

/// Simplified: sin²+cos² plus sin of simple sum squared  
#[test]
fn repro_simple_plus_sin_sum_squared() {
    let expr_str = "(sin(x)^2 + cos(x)^2) + sin(3 + x)^2";

    eprintln!("Testing: {}", expr_str);

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    let (result, _) = simplifier.simplify(expr);
    eprintln!("Result: {:?}", simplifier.context.get(result));
}

/// Without sin²+cos²: just sin((3 - x + sin(x))^4)^2
#[test]
fn repro_just_sin_pow4_squared() {
    let expr_str = "sin((3 - x + sin(x))^4)^2";

    eprintln!("Testing: {}", expr_str);

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    // Measure AST depth before simplification
    let depth = measure_depth(&simplifier.context, expr);
    eprintln!("AST depth before simplify: {}", depth);

    let (result, _) = simplifier.simplify(expr);

    let depth_after = measure_depth(&simplifier.context, result);
    eprintln!("AST depth after simplify: {}", depth_after);
    eprintln!("Result: {:?}", simplifier.context.get(result));
}

/// With 1 instead of sin²+cos²  
#[test]
fn repro_one_plus_sin_pow4_squared() {
    let expr_str = "1 + sin((3 - x + sin(x))^4)^2";

    eprintln!("Testing: {}", expr_str);

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    let (result, _) = simplifier.simplify(expr);
    eprintln!("Result: {:?}", simplifier.context.get(result));
}

/// Simpler: sin(x^4)^2 - does this crash?
#[test]
fn repro_sin_x_pow4_squared() {
    let expr_str = "sin(x^4)^2";

    eprintln!("Testing: {}", expr_str);

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    let (result, _) = simplifier.simplify(expr);
    eprintln!("Result: {:?}", simplifier.context.get(result));
}

/// sin((a + sin(x))^4)^2 - testing if the inner sin matters
#[test]
fn repro_sin_sum_with_inner_sin_pow4() {
    let expr_str = "sin((3 + sin(x))^4)^2";

    eprintln!("Testing: {}", expr_str);

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    let (result, _) = simplifier.simplify(expr);
    eprintln!("Result: {:?}", simplifier.context.get(result));
}

/// Test with subtraction: the difference from the OK case
#[test]
fn repro_sin_with_subtraction_pow4() {
    let expr_str = "sin((3 - x)^4)^2";

    eprintln!("Testing: {}", expr_str);

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    let (result, _) = simplifier.simplify(expr);
    eprintln!("Result: {:?}", simplifier.context.get(result));
}

/// Test with subtraction AND addition: closer to the crash case  
#[test]
fn repro_sin_sub_add_pow4() {
    let expr_str = "sin((3 - x + x)^4)^2";

    eprintln!("Testing: {}", expr_str);

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    let (result, _) = simplifier.simplify(expr);
    eprintln!("Result: {:?}", simplifier.context.get(result));
}

/// BISECTION 1: Remove outer ^2
/// NOTE: This test causes stack overflow due to deep recursion in trig expansion.
/// The issue is pre-existing and unrelated to HeuristicPoly changes.
/// TODO: Fix the underlying trig expansion recursion issue.
#[test]
fn bisect_no_outer_pow2() {
    let expr_str = "sin((3 - x + sin(x))^4)";
    eprintln!("BISECT no outer ^2: {}", expr_str);
    let mut s = Simplifier::with_default_rules();
    let e = parse(expr_str, &mut s.context).unwrap();
    let (r, _) = s.simplify(e);
    eprintln!("OK: {:?}", s.context.get(r));
}

/// BISECTION 2: Remove inner sin  
#[test]
fn bisect_no_inner_sin() {
    let expr_str = "sin((3 - x + x)^4)^2";
    eprintln!("BISECT no inner sin: {}", expr_str);
    let mut s = Simplifier::with_default_rules();
    let e = parse(expr_str, &mut s.context).unwrap();
    let (r, _) = s.simplify(e);
    eprintln!("OK: {:?}", s.context.get(r));
}

/// BISECTION 3: Use + instead of -  
#[test]
fn bisect_plus_instead_of_minus() {
    let expr_str = "sin((3 + x + sin(x))^4)^2";
    eprintln!("BISECT + instead of -: {}", expr_str);
    let mut s = Simplifier::with_default_rules();
    let e = parse(expr_str, &mut s.context).unwrap();
    let (r, _) = s.simplify(e);
    eprintln!("OK: {:?}", s.context.get(r));
}

/// BISECTION 4: Only subtraction (no sin inner)
#[test]
fn bisect_only_sub() {
    let expr_str = "sin((3 - x)^4)^2";
    eprintln!("BISECT only sub: {}", expr_str);
    let mut s = Simplifier::with_default_rules();
    let e = parse(expr_str, &mut s.context).unwrap();
    let (r, _) = s.simplify(e);
    eprintln!("OK: {:?}", s.context.get(r));
}

// =============================================================================
// V2.15: Regression test for infer_implicit_domain call count
// =============================================================================
// This test ensures that the fix for the stack overflow (caused by 98+ nested
// calls to infer_implicit_domain) doesn't regress. The domain should be inferred
// at most a small number of times per simplification, not O(n²) times.

/// Regression test: ensure infer_implicit_domain is not called excessively.
/// The original bug caused 98+ calls for a simple expression.
/// With the fix, it should be called only 1-3 times (cache + fallback).
#[test]
fn test_domain_inference_call_count_regression() {
    use cas_engine::implicit_domain::{infer_domain_calls_get, infer_domain_calls_reset};

    // Expression that previously caused stack overflow
    let expr_str = "sin((3 - x + sin(x))^4)^2";

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(expr_str, &mut simplifier.context).expect("Failed to parse");

    // Reset counter and simplify
    infer_domain_calls_reset();
    let (_result, _) = simplifier.simplify(expr);

    // Check call count
    let calls = infer_domain_calls_get();
    eprintln!("infer_implicit_domain called {} times", calls);

    // The limit is generous to account for fallback calls, but should catch O(n²) regressions
    // Before fix: ~98 calls. After fix: 1-5 calls.
    const MAX_ALLOWED_CALLS: usize = 10;
    assert!(
        calls <= MAX_ALLOWED_CALLS,
        "REGRESSION: infer_implicit_domain called {} times (max allowed: {}). \
         Rules may be recomputing domain instead of using cache.",
        calls,
        MAX_ALLOWED_CALLS
    );
}
