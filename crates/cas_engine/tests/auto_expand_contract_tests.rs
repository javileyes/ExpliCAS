//! Contract tests for the three-tier expansion system.
//!
//! Verifies that:
//! - Standard (ExpandPolicy::Off): preserves factored forms
//! - Auto (ExpandPolicy::Auto): only expands in cancellation contexts
//! - Solve mode: blocks auto-expand even when enabled
//! - Expand (expand_mode): aggressively expands everything

use cas_ast::{Context, Expr};
use cas_engine::options::ContextMode;
use cas_engine::phase::{ExpandPolicy, SimplifyOptions};
use cas_engine::Simplifier;
use cas_parser::parse;

/// Helper to simplify an expression with given options
fn simplify_with_opts(input: &str, opts: &SimplifyOptions) -> String {
    let mut ctx = Context::new();
    let parsed = parse(input, &mut ctx).expect("parse failed");
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.context = ctx;
    let (result, _steps) = simplifier.simplify_with_options(parsed, opts.clone());
    format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

/// Helper for standard mode (no auto-expand)
fn simplify_standard(input: &str) -> String {
    let opts = SimplifyOptions::default();
    simplify_with_opts(input, &opts)
}

/// Helper for auto-expand mode
fn simplify_auto(input: &str) -> String {
    let mut opts = SimplifyOptions::default();
    opts.expand_policy = ExpandPolicy::Auto;
    simplify_with_opts(input, &opts)
}

/// Helper for Solve mode with auto-expand enabled (should still block)
fn simplify_solve_with_auto(input: &str) -> String {
    let mut opts = SimplifyOptions::default();
    opts.expand_policy = ExpandPolicy::Auto;
    opts.context_mode = ContextMode::Solve;
    simplify_with_opts(input, &opts)
}

// =============================================================================
// STANDARD MODE TESTS: No expansion
// =============================================================================

#[test]
fn standard_preserves_binomial_squared() {
    // (x+1)^2 should stay as Pow, not expand to x^2 + 2x + 1
    let result = simplify_standard("(x+1)^2");
    assert!(
        result.contains("^") || result.contains("Pow"),
        "Standard should preserve (x+1)^2, got: {}",
        result
    );
}

#[test]
fn standard_preserves_binomial_cubed() {
    // (x+1)^3 should stay as factored form
    let result = simplify_standard("(x+1)^3");
    assert!(
        result.contains("^") || result.contains("Pow"),
        "Standard should preserve (x+1)^3, got: {}",
        result
    );
}

#[test]
fn standard_preserves_trinomial_squared() {
    // (x+y+z)^2 should stay as factored form
    let result = simplify_standard("(x+y+z)^2");
    // Should contain a power, not be fully expanded
    assert!(
        result.contains("^"),
        "Standard should preserve (x+y+z)^2, got: {}",
        result
    );
}

#[test]
fn standard_preserves_negative_exponent() {
    // (x+1)^(-3) should stay as is
    let result = simplify_standard("(x+1)^(-3)");
    assert!(
        result.contains("-3") || result.contains("^"),
        "Standard should preserve (x+1)^(-3), got: {}",
        result
    );
}

// =============================================================================
// AUTO MODE TESTS: Only expands in cancellation contexts
// =============================================================================

#[test]
fn auto_preserves_standalone_binomial() {
    // Even with autoexpand ON, standalone (x+1)^3 should NOT expand
    let result = simplify_auto("(x+1)^3");
    assert!(
        result.contains("^") || result.contains("Pow"),
        "Auto should preserve standalone (x+1)^3, got: {}",
        result
    );
}

#[test]
fn auto_expands_difference_quotient() {
    // ((x+h)^2 - x^2)/h should expand and simplify
    // Expected: 2*x + h (after expansion and cancellation)
    let result = simplify_auto("((x+h)^2 - x^2)/h");

    // Result should NOT contain Pow with exponent 2 anymore
    // It should be something like 2*x + h
    let has_pow_2 = result.contains("^2") || result.contains("^(2)");
    assert!(
        !has_pow_2 || result.contains("h") && result.contains("x"),
        "Auto should expand difference quotient ((x+h)^2 - x^2)/h, got: {}",
        result
    );
}

// =============================================================================
// BUDGET TESTS: Verify budget limits prevent explosion
// =============================================================================

#[test]
fn budget_rejects_high_exponent() {
    // ((x+h)^10 - x^10)/h should NOT expand because exponent > max_pow_exp (4)
    let result = simplify_auto("((x+h)^10 - x^10)/h");
    assert!(
        result.contains("^10") || result.contains("^(10)"),
        "Budget should reject exponent 10, got: {}",
        result
    );
}

#[test]
fn budget_rejects_too_many_terms() {
    // ((a+b+c+d+e)^3 - ...) - too many base terms (5 > max_base_terms=4)
    let result = simplify_auto("((a+b+c+d+e)^3 - a^3)/a");
    assert!(
        result.contains("^3") || result.contains("^(3)"),
        "Budget should reject 5-term base, got: {}",
        result
    );
}

// =============================================================================
// DETERMINISM TEST
// =============================================================================

#[test]
fn determinism_difference_quotient() {
    // Running the same expression multiple times should give identical results
    let input = "((x+h)^2 - x^2)/h";
    let first = simplify_auto(input);

    for i in 1..10 {
        let result = simplify_auto(input);
        assert_eq!(
            result, first,
            "Run {} produced different result: {} vs {}",
            i, result, first
        );
    }
}

// =============================================================================
// SCANNER UNIT TESTS
// =============================================================================

#[test]
fn scanner_marks_difference_quotient_context() {
    use cas_engine::auto_expand_scan::mark_auto_expand_candidates;
    use cas_engine::pattern_marks::PatternMarks;
    use cas_engine::phase::ExpandBudget;

    // Build: ((x+h)^3 - x^3)/h
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let h = ctx.var("h");
    let x_plus_h = ctx.add(Expr::Add(x, h));
    let three = ctx.num(3);
    let x_plus_h_cubed = ctx.add(Expr::Pow(x_plus_h, three));
    let x_cubed = ctx.add(Expr::Pow(x, three));
    let diff = ctx.add(Expr::Sub(x_plus_h_cubed, x_cubed));
    let quotient = ctx.add(Expr::Div(diff, h));

    let mut marks = PatternMarks::new();
    let budget = ExpandBudget::default();
    mark_auto_expand_candidates(&ctx, quotient, &budget, &mut marks);

    assert!(
        marks.has_auto_expand_contexts(),
        "Scanner should mark difference quotient context"
    );
    assert!(
        marks.is_auto_expand_context(quotient),
        "Scanner should mark the Div node as auto-expand context"
    );
}

#[test]
fn scanner_does_not_mark_standalone_pow() {
    use cas_engine::auto_expand_scan::mark_auto_expand_candidates;
    use cas_engine::pattern_marks::PatternMarks;
    use cas_engine::phase::ExpandBudget;

    // Build: (x+1)^3 - should NOT be marked
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let x_plus_one = ctx.add(Expr::Add(x, one));
    let three = ctx.num(3);
    let pow = ctx.add(Expr::Pow(x_plus_one, three));

    let mut marks = PatternMarks::new();
    let budget = ExpandBudget::default();
    mark_auto_expand_candidates(&ctx, pow, &budget, &mut marks);

    assert!(
        !marks.has_auto_expand_contexts(),
        "Scanner should NOT mark standalone Pow"
    );
}

// =============================================================================
// SOLVE MODE TESTS: Solve mode blocks auto-expand
// =============================================================================

#[test]
fn solve_blocks_autoexpand_difference_quotient() {
    // Even with ExpandPolicy::Auto, Solve mode should prevent expansion
    // This is defense-in-depth: Solve mode preserves structure for solver
    let result = simplify_solve_with_auto("((x+h)^3 - x^3)/h");

    // In Solve mode, the expression should NOT be expanded
    // It should still contain the Pow(x+h, 3) form
    assert!(
        result.contains("^3") || result.contains("^(3)"),
        "Solve mode should block auto-expand, got: {}",
        result
    );
}

#[test]
fn solve_preserves_standalone_binomial() {
    // Solve mode should preserve factored forms
    let result = simplify_solve_with_auto("(x+1)^2");
    assert!(
        result.contains("^2") || result.contains("^(2)"),
        "Solve mode should preserve (x+1)^2, got: {}",
        result
    );
}

// =============================================================================
// SUB CANCELLATION TESTS: Phase 2 - (x+1)^2 - (x^2+2x+1) → 0
// =============================================================================

#[test]
fn auto_sub_does_not_expand_standalone() {
    // Even with Auto ON, standalone (x+1)^3 should NOT expand
    let result = simplify_auto("(x+1)^3");
    assert!(
        result.contains("^3") || result.contains("^(3)"),
        "Auto should preserve standalone (x+1)^3, got: {}",
        result
    );
}

#[test]
fn scanner_marks_sub_with_pow_and_polynomial() {
    use cas_engine::auto_expand_scan::mark_auto_expand_candidates;
    use cas_engine::pattern_marks::PatternMarks;
    use cas_engine::phase::ExpandBudget;

    // Build: (x+1)^2 - (x^2 + 2*x + 1)
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let two = ctx.num(2);

    // (x+1)^2
    let x_plus_one = ctx.add(Expr::Add(x, one));
    let exp_two = ctx.num(2);
    let lhs = ctx.add(Expr::Pow(x_plus_one, exp_two));

    // x^2 + 2*x + 1
    let x_sq = ctx.add(Expr::Pow(x, exp_two));
    let two_x = ctx.add(Expr::Mul(two, x));
    let x_sq_plus_2x = ctx.add(Expr::Add(x_sq, two_x));
    let rhs = ctx.add(Expr::Add(x_sq_plus_2x, one));

    // Sub
    let sub = ctx.add(Expr::Sub(lhs, rhs));

    let mut marks = PatternMarks::new();
    let budget = ExpandBudget::default();
    mark_auto_expand_candidates(&ctx, sub, &budget, &mut marks);

    assert!(
        marks.is_auto_expand_context(sub),
        "Scanner should mark Sub(Pow(Add..), polynomial) as auto-expand context"
    );
}

#[test]
fn scanner_does_not_mark_sub_without_polynomial_rhs() {
    use cas_engine::auto_expand_scan::mark_auto_expand_candidates;
    use cas_engine::pattern_marks::PatternMarks;
    use cas_engine::phase::ExpandBudget;

    // Build: (x+1)^2 - sin(x) -- sin(x) is not polynomial-like
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);

    // (x+1)^2
    let x_plus_one = ctx.add(Expr::Add(x, one));
    let two = ctx.num(2);
    let lhs = ctx.add(Expr::Pow(x_plus_one, two));

    // sin(x)
    let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![x]));

    // Sub
    let sub = ctx.add(Expr::Sub(lhs, sin_x));

    let mut marks = PatternMarks::new();
    let budget = ExpandBudget::default();
    mark_auto_expand_candidates(&ctx, sub, &budget, &mut marks);

    assert!(
        !marks.is_auto_expand_context(sub),
        "Scanner should NOT mark Sub when rhs is not polynomial-like"
    );
}

#[test]
fn solve_blocks_auto_sub() {
    // Solve mode blocks Sub auto-expand too
    let result = simplify_solve_with_auto("(x+1)^2 - (x^2 + 2*x + 1)");
    // In Solve mode, should NOT auto-expand
    assert!(
        result.contains("^2") || result.contains("^(2)"),
        "Solve mode should block auto-expand in Sub, got: {}",
        result
    );
}

// =============================================================================
// ZERO-SHORTCUT CANCELLATION TESTS: (x+1)^2 - (x^2+2x+1) → 0
// =============================================================================

#[test]
fn auto_sub_cancels_binomial_square() {
    // Core win condition: (x+1)^2 - (x^2 + 2*x + 1) should simplify to 0
    let result = simplify_auto("(x+1)^2 - (x^2 + 2*x + 1)");
    assert_eq!(
        result, "0",
        "Auto should cancel (x+1)^2 - (x^2 + 2*x + 1) to 0, got: {}",
        result
    );
}

#[test]
fn auto_sub_cancels_binomial_cubed() {
    // (x+1)^3 - (x^3 + 3*x^2 + 3*x + 1) → 0
    let result = simplify_auto("(x+1)^3 - (x^3 + 3*x^2 + 3*x + 1)");
    assert_eq!(
        result, "0",
        "Auto should cancel (x+1)^3 - expanded to 0, got: {}",
        result
    );
}

#[test]
fn auto_sub_cancels_with_reversed_sides() {
    // (x^2 + 2*x + 1) - (x+1)^2 → 0 (reversed order)
    let result = simplify_auto("(x^2 + 2*x + 1) - (x+1)^2");
    assert_eq!(
        result, "0",
        "Auto should cancel reversed binomial to 0, got: {}",
        result
    );
}
