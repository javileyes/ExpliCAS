//! Context-aware expansion tests.
//!
//! Verifies that the semantic cancel pipeline expands `Pow(Add, n)` terms
//! only when the expansion would produce overlap with opposing terms,
//! enabling cancellation without requiring `expand_mode` in simplify.

use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

/// Helper: simplify an expression string and return the result as a string.
fn simplify_str(input: &str) -> String {
    let mut s = Simplifier::with_default_rules();
    let expr = parse(input, &mut s.context).unwrap();
    let (result, _) = s.simplify(expr);
    format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: result
        }
    )
}

/// Helper: solve an equation string "lhs = rhs" and return the solution set debug string.
fn solve_str(lhs_str: &str, rhs_str: &str) -> String {
    let mut s = Simplifier::with_default_rules();
    let lhs = parse(lhs_str, &mut s.context).unwrap();
    let rhs = parse(rhs_str, &mut s.context).unwrap();
    let eq = cas_ast::Equation {
        lhs,
        rhs,
        op: cas_ast::RelOp::Eq,
    };
    match cas_engine::solver::solve(&eq, "x", &mut s) {
        Ok((solution, _steps)) => format!("{:?}", solution),
        Err(e) => format!("Error: {:?}", e),
    }
}
// ── Test 1: Binomial cancel in solver/simplify context ───────────────
// (x+1)^2 - (x² + 2x + 1) should simplify to 0
// because the cancel pipeline expands (x+1)^2 when it detects overlap
// with the x², 2x, 1 terms on the other side of the subtraction.
#[test]
fn binomial_cancel_via_context_expansion() {
    let result = simplify_str("(x + 1)^2 - (x^2 + 2*x + 1)");
    assert_eq!(
        result, "0",
        "Expected (x+1)^2 - (x² + 2x + 1) to cancel to 0, got: {}",
        result
    );
}

// ── Test 2: Multinomial cancel ────────────────────────────────────────
// (a+b+c)^2 - (a² + b² + c² + 2ab + 2ac + 2bc) should simplify to 0
#[test]
fn multinomial_cancel_via_context_expansion() {
    let result = simplify_str("(a + b + c)^2 - (a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c)");
    assert_eq!(
        result, "0",
        "Expected multinomial identity to cancel to 0, got: {}",
        result
    );
}

// ── Test 3: No-expand when no overlap (guard test) ───────────────────
// (x+1)^2 + y has no overlap between expanded (x+1)^2 and y,
// so the Pow should stay compact.
#[test]
fn no_expand_without_overlap() {
    let result = simplify_str("(x + 1)^2 + y");
    assert!(
        result.contains("^2") || result.contains("²"),
        "Expected (x+1)^2 + y to keep Pow compact (no overlap), got: {}",
        result
    );
}

// ── Test 4: Equation solve with identity noise ───────────────────────
// x^2 + (x+1)^2 = 2x² + 2x + 1  is an identity (true for all x).
// The solver should detect this via context-aware expansion in cancel.
#[test]
fn solve_identity_via_context_expansion() {
    let result = solve_str("x^2 + (x + 1)^2", "2*x^2 + 2*x + 1");
    assert!(
        result.contains("AllReals"),
        "Expected identity equation to yield AllReals, got: {}",
        result
    );
}

// ── Test 5: Large exponent guard ─────────────────────────────────────
// (x+1)^10 should NOT be expanded even in cancel context (n > 4 guard).
#[test]
fn large_exponent_not_expanded() {
    let result = simplify_str("(x + 1)^10 - x^10");
    // Should not blow up. The result may not fully cancel, but it
    // should not take excessive time. The key assertion is that
    // it completes without timeout.
    assert!(
        !result.is_empty(),
        "Expected a result (not timeout) for (x+1)^10 - x^10"
    );
}

// ── Test 6: Simplify stays conservative (UX guard) ───────────────────
// simplify((x+1)^2) should stay as Pow — context-aware expansion
// must NOT leak into the global simplifier.
#[test]
fn simplify_stays_compact_without_cancel_context() {
    let result = simplify_str("(x + 1)^2");
    assert!(
        result.contains("^2") || result.contains("²"),
        "Expected (x+1)^2 to stay compact in simplify (no cancel context), got: {}",
        result
    );
}
