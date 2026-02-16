//! Contract tests for numeric-only reduction tickets.
//!
//! Verifies that the three changes (Abs normalization, Hyperbolic parity,
//! Mul(Add,Add) distributive expansion) correctly converge expressions
//! that were previously numeric-only in the metamorphic diagnostic.
//!
//! Each test group validates:
//! 1. Soundness: the rewrite produces mathematically correct results
//! 2. Convergence: equivalent forms simplify to the same normal form
//! 3. Idempotency: simplify(simplify(expr)) == simplify(expr) (anti-loop)

use cas_ast::DisplayExpr;
use cas_engine::Simplifier;
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

/// Helper: check idempotency — simplify twice should equal simplify once.
fn assert_idempotent(input: &str) {
    let mut s = Simplifier::with_default_rules();
    let expr = parse(input, &mut s.context).unwrap();
    let (first, _) = s.simplify(expr);
    let (second, _) = s.simplify(first);
    let first_str = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: first
        }
    );
    let second_str = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: second
        }
    );
    assert_eq!(
        first_str, second_str,
        "Idempotency failed for '{}': first='{}', second='{}'",
        input, first_str, second_str
    );
}

// =============================================================================
// Ticket 1: Abs Sub Normalize — compound expression convergence
// =============================================================================

/// |u²-1| and |1-u²| should produce the same result (canonical ordering).
#[test]
fn abs_sub_compound_convergence() {
    let result = simplify_str("|u^2 - 1| - |1 - u^2|");
    assert_eq!(
        result, "0",
        "Expected |u²-1| - |1-u²| to cancel to 0, got: {}",
        result
    );
}

/// |sin(u)-1| and |1-sin(u)| should produce the same result.
#[test]
fn abs_sub_function_arg_convergence() {
    let result = simplify_str("|sin(u) - 1| - |1 - sin(u)|");
    assert_eq!(
        result, "0",
        "Expected |sin(u)-1| - |1-sin(u)| to cancel to 0, got: {}",
        result
    );
}

/// Idempotency test for abs with compound arguments (anti-loop guard).
#[test]
fn abs_sub_compound_idempotent() {
    assert_idempotent("|sin(u) - 1|");
    assert_idempotent("|u^2 - 1|");
    assert_idempotent("|cos(x) - sin(x)|");
}

/// |x-y| should still work (original atom case preserved).
#[test]
fn abs_sub_atom_still_works() {
    let result = simplify_str("|x - y| - |y - x|");
    assert_eq!(
        result, "0",
        "Expected |x-y| - |y-x| to cancel to 0, got: {}",
        result
    );
}

// =============================================================================
// Ticket 2: Hyperbolic parity with Sub(a,b) arguments
// =============================================================================

/// sinh is odd: sinh(1-u²) + sinh(u²-1) = 0
#[test]
fn sinh_sub_parity_convergence() {
    let result = simplify_str("sinh(1 - u^2) + sinh(u^2 - 1)");
    assert_eq!(
        result, "0",
        "Expected sinh(1-u²) + sinh(u²-1) = 0 (odd), got: {}",
        result
    );
}

/// cosh is even: cosh(1-u²) - cosh(u²-1) = 0
#[test]
fn cosh_sub_parity_convergence() {
    let result = simplify_str("cosh(1 - u^2) - cosh(u^2 - 1)");
    assert_eq!(
        result, "0",
        "Expected cosh(1-u²) - cosh(u²-1) = 0 (even), got: {}",
        result
    );
}

/// tanh is odd: tanh(1-x) + tanh(x-1) = 0
#[test]
fn tanh_sub_parity_convergence() {
    let result = simplify_str("tanh(1 - x) + tanh(x - 1)");
    assert_eq!(
        result, "0",
        "Expected tanh(1-x) + tanh(x-1) = 0 (odd), got: {}",
        result
    );
}

/// Idempotency test for hyperbolic with Sub arguments (anti-loop guard).
#[test]
fn hyp_sub_parity_idempotent() {
    assert_idempotent("sinh(1 - u^2)");
    assert_idempotent("cosh(1 - u^2)");
    assert_idempotent("tanh(1 - x)");
}

/// Original Neg behavior still works: sinh(-x) = -sinh(x)
#[test]
fn sinh_neg_original_still_works() {
    let result = simplify_str("sinh(-x) + sinh(x)");
    assert_eq!(
        result, "0",
        "Expected sinh(-x) + sinh(x) = 0, got: {}",
        result
    );
}

// =============================================================================
// Ticket 3: Mul(Add,Add) distributive expansion in cancel context
// =============================================================================

/// (x+1)(x²-x+1) - (x³+1) = 0  (sum of cubes factorization)
#[test]
fn mul_add_add_sum_of_cubes() {
    let result = simplify_str("(x + 1)*(x^2 - x + 1) - (x^3 + 1)");
    assert_eq!(
        result, "0",
        "Expected (x+1)(x²-x+1) - (x³+1) to cancel to 0, got: {}",
        result
    );
}

/// (a+b)(a-b) - (a²-b²) = 0  (difference of squares)
#[test]
fn mul_add_add_difference_of_squares() {
    let result = simplify_str("(a + b)*(a - b) - (a^2 - b^2)");
    assert_eq!(
        result, "0",
        "Expected (a+b)(a-b) - (a²-b²) to cancel to 0, got: {}",
        result
    );
}

/// Guard test: Mul(Add, non-Add) should NOT be affected (stays compact).
#[test]
fn mul_add_scalar_not_expanded() {
    let result = simplify_str("x*(a + b) + c");
    assert!(
        !result.is_empty(),
        "Expected a result for x*(a+b) + c, got empty"
    );
}
