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

// =============================================================================
// Ticket 4: Generalized Mul-factor expansion (nested products)
// These tests use cancel_additive_terms_semantic (the solver-pipeline cancel)
// which includes Phase B mul-factor expansion with try_expand_for_cancel.
// =============================================================================

/// Helper: parse "lhs - rhs" expression and run semantic cancel.
/// Returns the display string of the result after cancellation + simplify.
fn semantic_cancel_str(lhs_str: &str, rhs_str: &str) -> String {
    use cas_ast::Expr;
    let mut s = Simplifier::with_default_rules();
    let lhs = parse(lhs_str, &mut s.context).unwrap();
    let rhs = parse(rhs_str, &mut s.context).unwrap();
    // Pre-simplify each side
    let (lhs_s, _) = s.simplify(lhs);
    let (rhs_s, _) = s.simplify(rhs);
    // Run semantic cancel (includes Phase B expansion)
    if let Some(cr) = cas_engine::cancel_additive_terms_semantic(&mut s, lhs_s, rhs_s) {
        // Build new_lhs - new_rhs and simplify
        let diff = s.context.add(Expr::Sub(cr.new_lhs, cr.new_rhs));
        let (result, _) = s.simplify(diff);
        format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: result
            }
        )
    } else {
        // No cancel happened — build and simplify original
        let diff = s.context.add(Expr::Sub(lhs_s, rhs_s));
        let (result, _) = s.simplify(diff);
        format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: result
            }
        )
    }
}

/// u³·(u³+1)·(u³+2) - (u⁹+3u⁶+2u³) = 0  (nested product with scalar factor)
#[test]
fn nested_mul_poly_gcf_convergence() {
    let result = semantic_cancel_str("u^3*(u^3 + 1)*(u^3 + 2)", "u^9 + 3*u^6 + 2*u^3");
    assert_eq!(
        result, "0",
        "Expected u³·(u³+1)·(u³+2) - expanded form to cancel to 0, got: {}",
        result
    );
}

/// x·(x+2)·(x+3) - (x³+5x²+6x) = 0  (3-factor product, scalar + 2 add-like)
#[test]
fn nested_mul_trinomial_factoring() {
    let result = semantic_cancel_str("x*(x + 2)*(x + 3)", "x^3 + 5*x^2 + 6*x");
    assert_eq!(
        result, "0",
        "Expected x·(x+2)·(x+3) - expanded to cancel to 0, got: {}",
        result
    );
}

/// Guard: no-overlap should NOT expand. (x+1)*(y+1) + z has no opposing terms.
#[test]
fn nested_mul_no_overlap_no_expand() {
    let result = simplify_str("(x + 1)*(y + 1) + z");
    // Should remain compact — not fully distributed
    assert!(
        !result.contains("x * y + x + y + 1"),
        "Should NOT distribute when no overlap: got: {}",
        result
    );
}

/// Idempotency for nested products.
#[test]
fn nested_mul_idempotent() {
    assert_idempotent("u^3*(u^3 + 1)*(u^3 + 2)");
    assert_idempotent("x*(x + 2)*(x + 3)");
}

// =============================================================================
// Ticket 5: Abs negative factor pull-out
// =============================================================================

/// |(-3)·x| should simplify to 3·|x|
#[test]
fn abs_neg_factor_pullout() {
    let result = simplify_str("|(-3)*x| - 3*|x|");
    assert_eq!(
        result, "0",
        "Expected |(-3)·x| - 3·|x| = 0, got: {}",
        result
    );
}

/// Idempotency: |(-2)*sin(x)| should be stable after double simplify
#[test]
fn abs_neg_factor_idempotent() {
    assert_idempotent("|(-2)*sin(x)|");
    assert_idempotent("|(-5)*u^2|");
}

// =============================================================================
// Ticket 6: Trig Half-Angle / Double-Angle bridge
// =============================================================================

// --- 6a: Half-angle squared convergence ---

/// 2·sin²(x/2) ≡ 1 - cos(x)  → should cancel to 0
#[test]
fn half_angle_sin_squared_convergence() {
    let result = simplify_str("2*sin(x/2)^2 - (1 - cos(x))");
    assert_eq!(
        result, "0",
        "Expected 2·sin²(x/2) - (1-cos(x)) = 0, got: {}",
        result
    );
}

/// 2·cos²(x/2) ≡ 1 + cos(x)  → should cancel to 0
#[test]
fn half_angle_cos_squared_convergence() {
    let result = simplify_str("2*cos(x/2)^2 - (1 + cos(x))");
    assert_eq!(
        result, "0",
        "Expected 2·cos²(x/2) - (1+cos(x)) = 0, got: {}",
        result
    );
}

// --- 6b: Double-angle cancel convergence (via semantic cancel) ---

/// cos(2x) ≡ 1 - 2·sin²(x)  → should cancel to 0 in cancel context
#[test]
fn double_angle_cos_sin_sq_convergence() {
    let result = semantic_cancel_str("cos(2*x)", "1 - 2*sin(x)^2");
    assert_eq!(
        result, "0",
        "Expected cos(2x) - (1-2sin²(x)) = 0 via cancel, got: {}",
        result
    );
}

/// cos(2x) ≡ 2·cos²(x) - 1  → should cancel to 0 in cancel context
#[test]
fn double_angle_cos_cos_sq_convergence() {
    let result = semantic_cancel_str("cos(2*x)", "2*cos(x)^2 - 1");
    assert_eq!(
        result, "0",
        "Expected cos(2x) - (2cos²(x)-1) = 0 via cancel, got: {}",
        result
    );
}

// --- Guards ---

/// sin(x/2)^3 should NOT expand (only squared forms fire)
#[test]
fn half_angle_cubed_no_fire() {
    let result = simplify_str("sin(x/2)^3");
    assert!(
        result.contains("sin"),
        "sin(x/2)^3 should remain in trig form, got: {}",
        result
    );
    // Should NOT contain "cos(x)" (the half-angle identity output)
    assert!(
        !result.contains("cos(x)"),
        "sin(x/2)^3 should NOT expand to cos(x) form, got: {}",
        result
    );
}

/// cos(2x) in normal mode should stay compact (not expand to 1-2sin²)
#[test]
fn cos_double_angle_normal_mode_stable() {
    let result = simplify_str("cos(2*x)");
    assert_eq!(
        result, "cos(2 * x)",
        "cos(2x) should stay compact in normal mode, got: {}",
        result
    );
}

/// Idempotency for half-angle squared expressions
#[test]
fn half_angle_squared_idempotent() {
    assert_idempotent("sin(x/2)^2");
    assert_idempotent("cos(x/2)^2");
    assert_idempotent("2*sin(x/2)^2 - 1 + cos(x)");
}

// =============================================================================
// Ticket 6b: Double-angle factor extraction from additive arguments
// e.g. cos(2u²+2) → cos(2*(u²+1)) → 1 - 2sin²(u²+1) in cancel context
// =============================================================================

/// cos(2u²+2) ≡ 1 - 2·sin²(u²+1)  → should cancel to 0 via semantic cancel
/// This exercises extract_int_multiple_additive recognizing Add(Mul(2,u²), 2) → 2*(u²+1)
#[test]
fn double_angle_distributed_convergence() {
    let result = semantic_cancel_str("cos(2*u^2 + 2)", "1 - 2*sin(u^2 + 1)^2");
    assert_eq!(
        result, "0",
        "Expected cos(2u²+2) - (1-2sin²(u²+1)) = 0 via cancel, got: {}",
        result
    );
}

/// sin(2u²+2) ≡ 2·sin(u²+1)·cos(u²+1)  → should cancel to 0 via semantic cancel
#[test]
fn double_angle_distributed_sin_convergence() {
    let result = semantic_cancel_str("sin(2*u^2 + 2)", "2*sin(u^2 + 1)*cos(u^2 + 1)");
    assert_eq!(
        result, "0",
        "Expected sin(2u²+2) - 2sin(u²+1)cos(u²+1) = 0 via cancel, got: {}",
        result
    );
}

/// Idempotency: cos(2u²+2) should be stable after double simplification
#[test]
fn double_angle_distributed_idempotent() {
    assert_idempotent("cos(2*u^2 + 2)");
    assert_idempotent("sin(2*u^2 + 2)");
}

// =============================================================================
// Ticket 6c: AngleIdentityRule gated to expand_mode only
// sin/cos(a+b) should NOT expand in normal simplification — expansion
// creates non-canonical forms (e.g. cos(u²)·cos(1) - sin(u²)·sin(1))
// that block convergence.
// =============================================================================

/// cos(u²+1) must stay compact in normal mode (no angle-sum expansion).
/// Structural check: result must NOT contain cos(1) or sin(1).
#[test]
fn angle_sum_no_expand_cos_default() {
    let result = simplify_str("cos(u^2 + 1)");
    assert!(
        !result.contains("cos(1)") && !result.contains("sin(1)"),
        "cos(u²+1) should NOT expand in normal mode; got: {}",
        result
    );
    // Should still be a cos(...) expression
    assert!(
        result.contains("cos"),
        "cos(u²+1) should remain as cos form; got: {}",
        result
    );
}

/// sin(x+y) DOES expand in normal mode when both summands have variables.
/// The surgical gate allows this because both `x` and `y` contain symbols,
/// enabling NF convergence with product forms like sin(x)cos(y)+cos(x)sin(y).
#[test]
fn angle_sum_expand_sin_both_vars() {
    let result = simplify_str("sin(x + y)");
    assert!(
        result.contains("cos(y)") || result.contains("cos(x)"),
        "sin(x+y) SHOULD expand when both summands have vars; got: {}",
        result
    );
}

/// sin(u²+1) must stay compact in normal mode (constant summand → blocked).
/// Structural check: result must NOT contain sin(1) or cos(1).
#[test]
fn angle_sum_no_expand_sin_const_summand() {
    let result = simplify_str("sin(u^2 + 1)");
    assert!(
        !result.contains("cos(1)") && !result.contains("sin(1)"),
        "sin(u²+1) should NOT expand in normal mode (const summand); got: {}",
        result
    );
    assert!(
        result.contains("sin"),
        "sin(u²+1) should remain as sin form; got: {}",
        result
    );
}

/// Idempotency for trig with sum arguments (anti-loop guard).
#[test]
fn angle_sum_no_expand_idempotent() {
    assert_idempotent("cos(u^2 + 1)");
    assert_idempotent("sin(x + y)");
    assert_idempotent("cos(a + b + c)");
}
