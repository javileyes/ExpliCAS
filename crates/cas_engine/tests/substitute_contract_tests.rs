//! Contract tests for substitute module (power-aware substitution).
//!
//! These tests verify the "load-bearing" behaviors of substitution:
//! 1. ExactOnly mode only replaces structural matches
//! 2. PowerPattern mode recognizes x^4 = (x^2)^2 when target is x^2
//! 3. Does NOT invent algebra (no product→power conversion)
//! 4. Structural matching only (no commutative matching)

use cas_ast::{Context, Expr};
use cas_engine::substitute::{substitute_power_aware, SubstituteOptions};
use cas_parser::parse;

/// Helper to parse expression
fn parse_expr(ctx: &mut Context, s: &str) -> cas_ast::ExprId {
    parse(s, ctx).expect("parse failed")
}

/// Check if expression contains a variable
fn contains_var(ctx: &Context, expr: cas_ast::ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Variable(name) => name == var,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            contains_var(ctx, *l, var) || contains_var(ctx, *r, var)
        }
        Expr::Neg(e) => contains_var(ctx, *e, var),
        Expr::Function(_, args) => args.iter().any(|a| contains_var(ctx, *a, var)),
        _ => false,
    }
}

// ============================================================================
// Group A — ExactOnly (base stable)
// ============================================================================

/// A1: exact_replaces_literal_target
/// expr: x^2 + 1, target: x^2, repl: y → y + 1
#[test]
fn a1_exact_replaces_literal_target() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x^2 + 1");
    let target = parse_expr(&mut ctx, "x^2");
    let replacement = parse_expr(&mut ctx, "y");

    let result = substitute_power_aware(
        &mut ctx,
        expr,
        target,
        replacement,
        SubstituteOptions::exact(),
    );

    // Should contain y, not x
    assert!(contains_var(&ctx, result, "y"), "A1: Expected y in result");
    assert!(
        !contains_var(&ctx, result, "x"),
        "A1: Expected no x in result"
    );
}

/// A3: exact_does_not_touch_nonmatching_power
/// expr: x^4 + x^2, target: x^2, repl: y → x^4 + y (ExactOnly)
#[test]
fn a3_exact_does_not_touch_nonmatching_power() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x^4 + x^2");
    let target = parse_expr(&mut ctx, "x^2");
    let replacement = parse_expr(&mut ctx, "y");

    let result = substitute_power_aware(
        &mut ctx,
        expr,
        target,
        replacement,
        SubstituteOptions::exact(),
    );

    // x^4 should remain (ExactOnly doesn't recognize power multiples)
    assert!(contains_var(&ctx, result, "x"), "A3: x^4 should remain");
    assert!(contains_var(&ctx, result, "y"), "A3: x^2 should become y");
}

// ============================================================================
// Group B — PowerPattern (main cases)
// ============================================================================

/// B1: powerpattern_polynomial_in_x2
/// expr: x^4 + x^2 + 1, target: x^2, repl: y → y^2 + y + 1
#[test]
fn b1_powerpattern_polynomial_in_x2() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x^4 + x^2 + 1");
    let target = parse_expr(&mut ctx, "x^2");
    let replacement = parse_expr(&mut ctx, "y");

    let result = substitute_power_aware(
        &mut ctx,
        expr,
        target,
        replacement,
        SubstituteOptions::power_aware_no_remainder(),
    );

    // Should contain y, not x
    assert!(contains_var(&ctx, result, "y"), "B1: Expected y in result");
    assert!(
        !contains_var(&ctx, result, "x"),
        "B1: Expected no x in result"
    );
}

/// B4: powerpattern_nontrivial_base_u
/// expr: (x+1)^4 + (x+1)^2 + 7, target: (x+1)^2, repl: y → y^2 + y + 7
#[test]
fn b4_powerpattern_nontrivial_base_u() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "(x+1)^4 + (x+1)^2 + 7");
    let target = parse_expr(&mut ctx, "(x+1)^2");
    let replacement = parse_expr(&mut ctx, "y");

    let result = substitute_power_aware(
        &mut ctx,
        expr,
        target,
        replacement,
        SubstituteOptions::power_aware_no_remainder(),
    );

    // Should contain y, not x
    assert!(contains_var(&ctx, result, "y"), "B4: Expected y in result");
    assert!(
        !contains_var(&ctx, result, "x"),
        "B4: Expected no x in result"
    );
}

/// B5: powerpattern_only_when_divisible
/// expr: x^3 + x^2 + 1, target: x^2, repl: y → x^3 + y + 1
#[test]
fn b5_powerpattern_only_when_divisible() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x^3 + x^2 + 1");
    let target = parse_expr(&mut ctx, "x^2");
    let replacement = parse_expr(&mut ctx, "y");

    let result = substitute_power_aware(
        &mut ctx,
        expr,
        target,
        replacement,
        SubstituteOptions::power_aware_no_remainder(),
    );

    // x^3 should remain (3 % 2 != 0), x^2 becomes y
    assert!(contains_var(&ctx, result, "x"), "B5: x^3 should remain");
    assert!(contains_var(&ctx, result, "y"), "B5: x^2 should become y");
}

// ============================================================================
// Group C — Robustness (no invented algebra)
// ============================================================================

/// C1: does_not_convert_products_to_powers
/// expr: x*x + 1, target: x^2, repl: y → x*x + 1 (unchanged)
#[test]
fn c1_does_not_convert_products_to_powers() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x*x + 1");
    let target = parse_expr(&mut ctx, "x^2");
    let replacement = parse_expr(&mut ctx, "y");

    let result = substitute_power_aware(
        &mut ctx,
        expr,
        target,
        replacement,
        SubstituteOptions::default(),
    );

    // x*x is NOT x^2 structurally, should remain
    assert!(
        contains_var(&ctx, result, "x"),
        "C1: x*x should remain unchanged"
    );
    assert!(
        !contains_var(&ctx, result, "y"),
        "C1: Should not introduce y"
    );
}

/// C3: symbolic_exponent_blocks_powerpattern
/// expr: x^n + x^2, target: x^2, repl: y → x^n + y
#[test]
fn c3_symbolic_exponent_blocks_powerpattern() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x^n + x^2");
    let target = parse_expr(&mut ctx, "x^2");
    let replacement = parse_expr(&mut ctx, "y");

    let result = substitute_power_aware(
        &mut ctx,
        expr,
        target,
        replacement,
        SubstituteOptions::default(),
    );

    // x^n has symbolic exponent, should not be touched
    assert!(contains_var(&ctx, result, "n"), "C3: x^n should remain");
    assert!(contains_var(&ctx, result, "y"), "C3: x^2 should become y");
}

// ============================================================================
// Group B2/B3 — Higher multiples and pow-of-target
// ============================================================================

/// B2: powerpattern_higher_multiple
/// expr: x^6 + x^2, target: x^2, repl: y → y^3 + y
#[test]
fn b2_powerpattern_higher_multiple() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x^6 + x^2");
    let target = parse_expr(&mut ctx, "x^2");
    let replacement = parse_expr(&mut ctx, "y");

    let result = substitute_power_aware(
        &mut ctx,
        expr,
        target,
        replacement,
        SubstituteOptions::power_aware_no_remainder(),
    );

    // x^6 = (x^2)^3 → y^3
    assert!(contains_var(&ctx, result, "y"), "B2: Expected y in result");
    assert!(
        !contains_var(&ctx, result, "x"),
        "B2: Expected no x in result"
    );
}

/// B3: powerpattern_pow_of_target
/// expr: (x^2)^3 + x^2, target: x^2, repl: y → y^3 + y
#[test]
fn b3_powerpattern_pow_of_target() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "(x^2)^3 + x^2");
    let target = parse_expr(&mut ctx, "x^2");
    let replacement = parse_expr(&mut ctx, "y");

    let result = substitute_power_aware(
        &mut ctx,
        expr,
        target,
        replacement,
        SubstituteOptions::default(),
    );

    // (x^2)^3 should have x^2 replaced by y → y^3
    assert!(contains_var(&ctx, result, "y"), "B3: Expected y in result");
    assert!(
        !contains_var(&ctx, result, "x"),
        "B3: Expected no x in result"
    );
}

// ============================================================================
// Group C2 — Commutative matching (NOT supported)
// ============================================================================

/// C2: does_not_match_different_base_structure
/// expr: (x+y)^2 + 1, target: (x-y)^2, repl: t → unchanged (different base)
#[test]
fn c2_does_not_match_different_base_structure() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "(x+y)^2 + 1");
    let target = parse_expr(&mut ctx, "(x-y)^2"); // Different: Sub vs Add
    let replacement = parse_expr(&mut ctx, "t");

    let result = substitute_power_aware(
        &mut ctx,
        expr,
        target,
        replacement,
        SubstituteOptions::default(),
    );

    // Different base structure (Add vs Sub), should not match
    assert!(
        !contains_var(&ctx, result, "t"),
        "C2: Should not match different base structure"
    );
    assert!(
        contains_var(&ctx, result, "x"),
        "C2: Original should remain"
    );
}
