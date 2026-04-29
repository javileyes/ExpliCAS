//! Tests for condition normalization in implicit domain
//!
//! These tests ensure that domain conditions are displayed consistently:
//! 1. `1 + x ≠ 0` displays as `x + 1 ≠ 0` (canonical polynomial order)
//! 2. `2 - x ≠ 0` and `x - 2 ≠ 0` are unified (sign normalization)
//! 3. Duplicate equivalent conditions are removed

use cas_ast::{Context, Expr};
use cas_solver::api::{
    normalize_and_dedupe_conditions, normalize_condition, render_conditions_normalized,
    ImplicitCondition,
};

/// Test that `1 + x` displays as `x + 1` (canonical polynomial order)
#[test]
fn test_condition_canonical_order() {
    let mut ctx = Context::new();

    // Build 1 + x (constant first)
    let one = ctx.num(1);
    let x = ctx.var("x");
    let expr = ctx.add(Expr::Add(one, x));

    let cond = ImplicitCondition::NonZero(expr);

    // Display should be in canonical order: x + 1 ≠ 0
    let display = cond.display(&ctx);
    println!("Display: {}", display);

    // DisplayExpr now applies canonical ordering, so it should be x + 1
    assert!(
        display.contains("x + 1") || display.contains("x+1"),
        "Expected 'x + 1 ≠ 0' but got: {}",
        display
    );
}

/// Test sign normalization: negative leading coefficient gets negated
#[test]
fn test_condition_sign_normalization() {
    let mut ctx = Context::new();

    // Build 2 - x (which has leading coeff of x as -1)
    let two = ctx.num(2);
    let x = ctx.var("x");
    let expr = ctx.add(Expr::Sub(two, x));

    let cond = ImplicitCondition::NonZero(expr);
    let normalized = normalize_condition(&mut ctx, &cond);

    // After normalization, leading coeff should be positive
    // 2 - x has poly -x + 2, leading coeff -1, so it becomes x - 2
    let display = normalized.display(&ctx);
    println!("Original: 2 - x ≠ 0");
    println!("Normalized: {}", display);

    assert!(
        display.contains("x - 2") || display.contains("x-2"),
        "Expected 'x - 2 ≠ 0' after sign normalization, got: {}",
        display
    );
}

#[test]
fn test_nonnegative_condition_preserves_polynomial_sign() {
    let mut ctx = Context::new();
    let expr = cas_parser::parse("1 - x^2", &mut ctx).expect("parse");
    let cond = ImplicitCondition::NonNegative(expr);

    let normalized = normalize_condition(&mut ctx, &cond);
    let display = normalized.display(&ctx);

    assert_eq!(display, "1 - x^2 ≥ 0");
}

#[test]
fn test_positive_condition_preserves_polynomial_sign() {
    let mut ctx = Context::new();
    let expr = cas_parser::parse("1 - x^2", &mut ctx).expect("parse");
    let cond = ImplicitCondition::Positive(expr);

    let normalized = normalize_condition(&mut ctx, &cond);
    let display = normalized.display(&ctx);

    assert_eq!(display, "1 - x^2 > 0");
}

/// Test that x - 2 and 2 - x are deduplicated as equivalent
#[test]
fn test_equivalent_conditions_dedupe() {
    let mut ctx = Context::new();

    // Build x - 2
    let x1 = ctx.var("x");
    let two1 = ctx.num(2);
    let expr1 = ctx.add(Expr::Sub(x1, two1));
    let cond1 = ImplicitCondition::NonZero(expr1);

    // Build 2 - x (equivalent to -(x - 2))
    let two2 = ctx.num(2);
    let x2 = ctx.var("x");
    let expr2 = ctx.add(Expr::Sub(two2, x2));
    let cond2 = ImplicitCondition::NonZero(expr2);

    // Dedupe should recognize these as equivalent
    let conditions = vec![cond1.clone(), cond2.clone()];
    let deduped = normalize_and_dedupe_conditions(&mut ctx, &conditions);

    println!("Before dedupe: {:?}", conditions.len());
    println!("After dedupe: {:?}", deduped.len());
    for c in &deduped {
        println!("  - {}", c.display(&ctx));
    }

    assert_eq!(
        deduped.len(),
        1,
        "Expected 1 condition after dedupe, got {}",
        deduped.len()
    );
}

/// Test that non-polynomial conditions (like sin(x)) are handled gracefully
#[test]
fn test_non_polynomial_condition_stable() {
    let mut ctx = Context::new();

    // Build sin(x)
    let x = ctx.var("x");
    let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![x]);

    let cond = ImplicitCondition::NonZero(sin_x);
    let normalized = normalize_condition(&mut ctx, &cond);

    // Should not panic and should return a valid display
    let display = normalized.display(&ctx);
    println!("Non-poly condition: {}", display);

    assert!(
        display.contains("sin"),
        "Display should contain sin(x): {}",
        display
    );
}

/// Test render_conditions_normalized end-to-end
#[test]
fn test_render_normalized_complete() {
    let mut ctx = Context::new();

    // Build x + 1
    let x1 = ctx.var("x");
    let one1 = ctx.num(1);
    let expr1 = ctx.add(Expr::Add(x1, one1));
    let cond1 = ImplicitCondition::NonZero(expr1);

    // Build 1 + x (same, different order)
    let one2 = ctx.num(1);
    let x2 = ctx.var("x");
    let expr2 = ctx.add(Expr::Add(one2, x2));
    let cond2 = ImplicitCondition::NonZero(expr2);

    // Build y (different)
    let y = ctx.var("y");
    let cond3 = ImplicitCondition::Positive(y);

    let conditions = vec![cond1, cond2, cond3];
    let rendered = render_conditions_normalized(&mut ctx, &conditions);

    println!("Rendered conditions:");
    for s in &rendered {
        println!("  - {}", s);
    }

    // Should have 2 conditions: x + 1 ≠ 0 (deduped) and y > 0
    assert_eq!(
        rendered.len(),
        2,
        "Expected 2 conditions, got {:?}",
        rendered
    );
}

#[test]
fn test_nonzero_polynomial_factorizes_to_atomic_guards() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let three = ctx.num(3);
    let x3 = ctx.add(Expr::Pow(x, three));
    let cubic_minus_x = ctx.add(Expr::Sub(x3, x));

    let normalized =
        normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(cubic_minus_x)]);

    let rendered: Vec<_> = normalized.iter().map(|cond| cond.display(&ctx)).collect();
    println!("Rendered atomic guards: {:?}", rendered);

    assert_eq!(rendered.len(), 3);
    assert!(rendered.iter().any(|item| item == "x ≠ 0"));
    assert!(rendered.iter().any(|item| item == "x - 1 ≠ 0"));
    assert!(rendered.iter().any(|item| item == "x + 1 ≠ 0"));
}

#[test]
fn test_nonzero_factorization_dedupes_explicit_subfactor() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let three = ctx.num(3);
    let one = ctx.num(1);
    let x3 = ctx.add(Expr::Pow(x, three));
    let cubic_minus_x = ctx.add(Expr::Sub(x3, x));
    let x_minus_1 = ctx.add(Expr::Sub(x, one));

    let normalized = normalize_and_dedupe_conditions(
        &mut ctx,
        &[
            ImplicitCondition::NonZero(x_minus_1),
            ImplicitCondition::NonZero(cubic_minus_x),
        ],
    );

    let rendered: Vec<_> = normalized.iter().map(|cond| cond.display(&ctx)).collect();
    println!("Rendered deduped guards: {:?}", rendered);

    assert_eq!(rendered.len(), 3);
    assert_eq!(
        rendered
            .iter()
            .filter(|item| item.as_str() == "x - 1 ≠ 0")
            .count(),
        1
    );
    assert!(rendered.iter().any(|item| item == "x ≠ 0"));
    assert!(rendered.iter().any(|item| item == "x + 1 ≠ 0"));
}

#[test]
fn test_positive_base_dominates_symbolic_power_condition() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let x_pow_n = cas_parser::parse("x^n", &mut ctx).expect("parse x^n");

    let normalized = normalize_and_dedupe_conditions(
        &mut ctx,
        &[
            ImplicitCondition::Positive(x_pow_n),
            ImplicitCondition::Positive(x),
        ],
    );

    let rendered: Vec<_> = normalized.iter().map(|cond| cond.display(&ctx)).collect();
    println!("Rendered positive guards: {:?}", rendered);

    assert_eq!(rendered, vec!["x > 0"]);
}
