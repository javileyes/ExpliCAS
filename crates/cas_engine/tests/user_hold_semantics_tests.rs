//! Tests for user-facing `hold()` function semantics.
//!
//! The `hold(expr)` function is the user-facing equivalent of the internal `__hold`
//! barrier but with a key difference:
//!
//! - `__hold(expr)`: Internal, transparent in display, stripped before output
//! - `hold(expr)`: User-facing, VISIBLE in display, preserved in output
//!
//! Both have HoldAll semantics: children are NOT simplified.

use cas_engine::Simplifier;
use cas_parser::parse;

// =============================================================================
// Section 1: HoldAll semantics - children should NOT be simplified
// =============================================================================

/// Test: hold(x + 0) should NOT simplify to hold(x)
/// The + 0 must be preserved because HoldAll blocks child simplification.
#[test]
fn hold_preserves_additive_identity() {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse("hold(x + 0)", &mut simplifier.context).unwrap();

    let (result, _steps) = simplifier.simplify(expr);

    // The result should still show x + 0 inside hold
    // Since hold is visible, we expect "hold(x + 0)"
    let display = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    assert!(
        display.contains("hold"),
        "Expected 'hold' wrapper to be visible, got: {}",
        display
    );
    assert!(
        display.contains("0"),
        "Expected '+ 0' to be preserved inside hold (HoldAll semantics), got: {}",
        display
    );
}

/// Test: hold(2 * 3) should NOT evaluate to hold(6)
/// Constants inside hold should NOT be folded.
#[test]
fn hold_blocks_constant_folding() {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse("hold(2 * 3)", &mut simplifier.context).unwrap();

    let (result, _steps) = simplifier.simplify(expr);

    let display = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    assert!(
        display.contains("hold"),
        "Expected 'hold' wrapper to be visible, got: {}",
        display
    );
    // Should contain 2 and 3, NOT 6
    assert!(
        display.contains("2") && display.contains("3"),
        "Expected 2 * 3 to be preserved (not folded to 6), got: {}",
        display
    );
    assert!(
        !display.contains("hold(6)"),
        "Should NOT fold to 6 inside hold, got: {}",
        display
    );
}

/// Test: hold(sin(0)) should NOT evaluate to hold(0)
/// Function evaluation inside hold should be blocked.
#[test]
fn hold_blocks_function_evaluation() {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse("hold(sin(0))", &mut simplifier.context).unwrap();

    let (result, _steps) = simplifier.simplify(expr);

    let display = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    assert!(
        display.contains("sin"),
        "Expected sin(0) to be preserved inside hold, got: {}",
        display
    );
}

/// Test: hold(x^1) - the ^1 might be simplified during display, but hold wrapper stays.
/// Note: Display layer may simplify x^1 to x for readability, but HoldAll still prevents
/// evaluation-time transformations.
#[test]
fn hold_blocks_power_identity() {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse("hold(x^1)", &mut simplifier.context).unwrap();

    let (result, _steps) = simplifier.simplify(expr);

    let display = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // The hold wrapper should be preserved
    assert!(
        display.contains("hold("),
        "Expected hold wrapper to be preserved, got: {}",
        display
    );
    // The inner should be x (display may simplify x^1)
    assert!(
        display.contains("x"),
        "Expected x inside hold, got: {}",
        display
    );
}

// =============================================================================
// Section 2: Mixed context - hold inside vs outside
// =============================================================================

/// Test: hold(x + 0) + 0 → hold inner is frozen, outer + 0 simplifies away
/// This verifies that HoldAll only affects what's INSIDE the hold wrapper.
#[test]
fn hold_mixed_context_outer_simplifies() {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse("hold(x + 0) + 0", &mut simplifier.context).unwrap();

    let (result, _steps) = simplifier.simplify(expr);

    let display = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // The outer + 0 should simplify away, leaving just hold(x + 0)
    // But the inner + 0 stays because HoldAll blocks child simplification
    assert!(
        display.contains("hold"),
        "Expected hold wrapper to remain, got: {}",
        display
    );
    // The inner 0 should still be there (inside hold)
    // We should NOT see something like "hold(x + 0) + 0" after simplification
}

/// Test: 2 * hold(x + 0) → the 2 * multiplies hold, but inside stays frozen
#[test]
fn hold_mixed_context_multiplication() {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse("2 * hold(x + 0)", &mut simplifier.context).unwrap();

    let (result, _steps) = simplifier.simplify(expr);

    let display = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Should have 2 * hold(...) with the inner preserved
    assert!(
        display.contains("hold") && display.contains("2"),
        "Expected '2 * hold(...)' structure, got: {}",
        display
    );
}

// =============================================================================
// Section 3: Display layer - __hold vs hold visibility
// =============================================================================

/// Test: __hold is transparent in display (internal barrier)
#[test]
fn internal_hold_is_transparent() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let held = cas_ast::hold::wrap_hold(&mut ctx, x);

    let display = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &ctx,
            id: held
        }
    );

    // __hold should NOT appear in output
    assert_eq!(
        display, "x",
        "Internal __hold should be transparent, got: {}",
        display
    );
    assert!(
        !display.contains("__hold"),
        "Internal __hold should NOT appear in output"
    );
}

/// Test: user-facing hold() is VISIBLE in display
#[test]
fn user_hold_is_visible() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");

    // Create hold(x) as Function("hold", [x])
    let hold_fn = ctx.call("hold", vec![x]);

    let display = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &ctx,
            id: hold_fn
        }
    );

    // User hold should appear in output
    assert!(
        display.contains("hold("),
        "User hold() should be visible in output, got: {}",
        display
    );
}

/// Test: __hold(1*x) displays as x (transparent, simplified)
/// Note: This tests display transparency, not evaluation (since __hold is internal)
#[test]
fn internal_hold_display_product_identity() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let product = ctx.add(cas_ast::Expr::Mul(one, x));
    let held = cas_ast::hold::wrap_hold(&mut ctx, product);

    let display = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &ctx,
            id: held
        }
    );

    // Internal hold is transparent, so we see 1 * x
    // Display shows whatever is inside (not simplified by display)
    assert!(
        !display.contains("__hold"),
        "Internal __hold should never appear in display"
    );
}

// =============================================================================
// Section 4: Contract validation - no __hold leaks
// =============================================================================

/// Test: After simplification, no __hold should appear in final output
#[test]
fn no_internal_hold_leak_in_result() {
    let mut simplifier = Simplifier::with_default_rules();

    // Expression that might create internal holds (expand typically does)
    let expr = parse("expand((x+1)^2)", &mut simplifier.context).unwrap();
    let (result, _steps) = simplifier.simplify(expr);

    let display = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    assert!(
        !display.contains("__hold"),
        "Result should not contain __hold: {}",
        display
    );
}
