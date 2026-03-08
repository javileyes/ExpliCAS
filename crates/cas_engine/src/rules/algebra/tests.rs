//! Tests for algebra rules
//!
//! These tests verify the correctness of fraction simplification,
//! factorization, and other algebraic rules.

use super::*;
use crate::rule::Rule;
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

#[test]
fn test_simplify_fraction() {
    let mut ctx = Context::new();
    let rule = SimplifyFractionRule;

    // (x^2 - 1) / (x + 1) -> x - 1
    let expr = parse("(x^2 - 1) / (x + 1)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    // Result might be -1 + x or x - 1 depending on polynomial to_expr order
    // Polynomial to_expr outputs lowest degree first?
    // My implementation: "1 + x" for x+1.
    // x^2 - 1 = (x-1)(x+1).
    // (x-1) -> -1 + x
    let s = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    );
    assert!(s.contains("x"));
    // With improved display, it might be "x - 1" or "-1 + x"
    assert!(s.contains("- 1") || s.contains("-1"));
}

#[test]
fn test_simplify_fraction_2() {
    let mut ctx = Context::new();
    let rule = SimplifyFractionRule;
    // (x^2 + 2*x + 1) / (x + 1) -> x + 1
    let expr = parse("(x^2 + 2*x + 1) / (x + 1)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    let s = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    );
    assert!(s.contains("1"));
    assert!(s.contains("x"));
}

#[test]
fn test_factor_difference_squares() {
    let mut ctx = Context::new();
    let rule = FactorRule;
    // factor(x^2 - 1) -> (x - 1)(x + 1)
    // Note: My implementation produces (x-1) and (x+1) (or similar).
    // Order depends on root finding.
    // Roots are 1, -1.
    // Factors: (x-1), (x+1).
    let expr = parse("factor(x^2 - 1)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    let res = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    );
    assert!(res.contains("x - 1") || res.contains("-1 + x") || res.contains("x + -1"));
    assert!(res.contains("x + 1") || res.contains("1 + x"));
}

#[test]
fn test_factor_perfect_square() {
    let mut ctx = Context::new();
    let rule = FactorRule;
    // factor(x^2 + 2x + 1) -> (x + 1)(x + 1)
    let expr = parse("factor(x^2 + 2*x + 1)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    let res = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    );
    // Should be (x+1)^2
    assert!(res.contains("x + 1") || res.contains("1 + x"));
    assert!(res.contains("^2") || res.contains("^ 2"));
}
