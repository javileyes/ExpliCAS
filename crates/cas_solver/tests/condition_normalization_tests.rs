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

#[test]
fn test_positive_condition_dedupes_positive_scalar_multiple() {
    let mut ctx = Context::new();
    let unscaled = cas_parser::parse("3 - x^2 - 2*x", &mut ctx).expect("parse unscaled");
    let scaled = cas_parser::parse("3/4 - x^2/4 - x/2", &mut ctx).expect("parse scaled");

    let normalized = normalize_and_dedupe_conditions(
        &mut ctx,
        &[
            ImplicitCondition::Positive(unscaled),
            ImplicitCondition::Positive(scaled),
        ],
    );

    let rendered: Vec<_> = normalized.iter().map(|cond| cond.display(&ctx)).collect();
    assert_eq!(rendered, vec!["3 - x^2 - 2 * x > 0"]);
}

#[test]
fn test_positive_condition_clears_positive_rational_content_for_display() {
    let mut ctx = Context::new();
    let scaled = cas_parser::parse("1 - x^4/4", &mut ctx).expect("parse scaled");

    let rendered = render_conditions_normalized(&mut ctx, &[ImplicitCondition::Positive(scaled)]);

    assert_eq!(rendered, vec!["4 - x^4 > 0"]);
}

#[test]
fn test_positive_condition_dominates_shifted_nonnegative_condition() {
    let mut ctx = Context::new();
    let positive_x = cas_parser::parse("x", &mut ctx).expect("parse x");
    let nonnegative_shift = cas_parser::parse("x + 1", &mut ctx).expect("parse x + 1");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonNegative(nonnegative_shift),
            ImplicitCondition::Positive(positive_x),
        ],
    );

    assert_eq!(rendered, vec!["x > 0"]);
}

#[test]
fn test_positive_condition_dominates_product_nonnegative_condition() {
    let mut ctx = Context::new();
    let positive_x = cas_parser::parse("x", &mut ctx).expect("parse x");
    let product = cas_parser::parse("x * (x + 1)", &mut ctx).expect("parse product");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonNegative(product),
            ImplicitCondition::Positive(positive_x),
        ],
    );

    assert_eq!(rendered, vec!["x > 0"]);
}

#[test]
fn test_positive_denominator_dominates_quotient_nonnegative_condition() {
    let mut ctx = Context::new();
    let nonnegative_x = cas_parser::parse("x", &mut ctx).expect("parse x");
    let positive_denominator = cas_parser::parse("x + 1", &mut ctx).expect("parse x + 1");
    let quotient = cas_parser::parse("x / (x + 1)", &mut ctx).expect("parse quotient");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonNegative(quotient),
            ImplicitCondition::NonNegative(nonnegative_x),
            ImplicitCondition::Positive(positive_denominator),
        ],
    );

    assert_eq!(rendered, vec!["x ≥ 0"]);
}

#[test]
fn test_positive_numerator_reduces_quotient_positive_condition() {
    let mut ctx = Context::new();
    let positive_x = cas_parser::parse("x", &mut ctx).expect("parse x");
    let y = cas_parser::parse("y", &mut ctx).expect("parse y");
    let quotient = cas_parser::parse("x / y", &mut ctx).expect("parse quotient");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::Positive(positive_x),
            ImplicitCondition::Positive(quotient),
            ImplicitCondition::NonZero(y),
        ],
    );

    assert_eq!(rendered, vec!["x > 0", "y > 0"]);
}

#[test]
fn test_positive_even_power_numerator_quotient_expands_to_atomic_domain_guards() {
    let mut ctx = Context::new();
    let quotient = cas_parser::parse("x^2 / y", &mut ctx).expect("parse quotient");

    let rendered = render_conditions_normalized(&mut ctx, &[ImplicitCondition::Positive(quotient)]);

    assert_eq!(rendered, vec!["x ≠ 0", "y > 0"]);
}

#[test]
fn test_positive_multiple_even_power_product_numerator_quotient_expands_to_atomic_domain_guards() {
    let mut ctx = Context::new();
    let quotient = cas_parser::parse("(x^2 * z^2) / y", &mut ctx).expect("parse quotient");

    let rendered = render_conditions_normalized(&mut ctx, &[ImplicitCondition::Positive(quotient)]);

    assert_eq!(rendered, vec!["x ≠ 0", "z ≠ 0", "y > 0"]);
}

#[test]
fn test_positive_even_power_product_quotient_with_nonnegative_shadow_stays_atomic() {
    let mut ctx = Context::new();
    let quotient = cas_parser::parse("(x^2 * z^2) / y", &mut ctx).expect("parse quotient");
    let y = cas_parser::parse("y", &mut ctx).expect("parse y");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::Positive(quotient),
            ImplicitCondition::NonNegative(quotient),
            ImplicitCondition::NonZero(y),
        ],
    );

    assert_eq!(rendered, vec!["x ≠ 0", "z ≠ 0", "y > 0"]);
}

#[test]
fn test_expanded_shifted_square_product_nonzero_is_dominated_by_atomic_guards() {
    let mut ctx = Context::new();
    let composite =
        cas_parser::parse("y^2*z^2 + 2*y^2*z + y^2", &mut ctx).expect("parse composite nonzero");
    let y = cas_parser::parse("y", &mut ctx).expect("parse y");
    let z_plus_one = cas_parser::parse("z + 1", &mut ctx).expect("parse z + 1");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonZero(composite),
            ImplicitCondition::NonZero(y),
            ImplicitCondition::NonZero(z_plus_one),
        ],
    );

    assert_eq!(rendered, vec!["y ≠ 0", "z + 1 ≠ 0"]);
}

#[test]
fn test_expanded_shifted_square_product_nonzero_expands_missing_atomic_guard() {
    let mut ctx = Context::new();
    let composite =
        cas_parser::parse("y^2*z^2 + 2*y^2*z + y^2", &mut ctx).expect("parse composite nonzero");
    let y = cas_parser::parse("y", &mut ctx).expect("parse y");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonZero(composite),
            ImplicitCondition::NonZero(y),
        ],
    );

    assert_eq!(rendered, vec!["y ≠ 0", "z + 1 ≠ 0"]);
}

#[test]
fn test_positive_even_power_denominator_quotient_expands_to_atomic_domain_guards() {
    let mut ctx = Context::new();
    let quotient = cas_parser::parse("x / y^2", &mut ctx).expect("parse quotient");

    let rendered = render_conditions_normalized(&mut ctx, &[ImplicitCondition::Positive(quotient)]);

    assert_eq!(rendered, vec!["x > 0", "y ≠ 0"]);
}

#[test]
fn test_positive_multiple_even_power_denominator_quotient_expands_to_atomic_domain_guards() {
    let mut ctx = Context::new();
    let quotient = cas_parser::parse("x / (y^2 * z^2)", &mut ctx).expect("parse quotient");

    let rendered = render_conditions_normalized(&mut ctx, &[ImplicitCondition::Positive(quotient)]);

    assert_eq!(rendered, vec!["x > 0", "y ≠ 0", "z ≠ 0"]);
}

#[test]
fn test_positive_denominator_reduces_quotient_nonnegative_condition() {
    let mut ctx = Context::new();
    let quotient = cas_parser::parse("(x^2 - 1) / x^2", &mut ctx).expect("parse quotient");
    let denominator_base = cas_parser::parse("x", &mut ctx).expect("parse x");
    let left_boundary = cas_parser::parse("x - 1", &mut ctx).expect("parse x - 1");
    let right_boundary = cas_parser::parse("x + 1", &mut ctx).expect("parse x + 1");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonNegative(quotient),
            ImplicitCondition::NonZero(denominator_base),
            ImplicitCondition::NonZero(left_boundary),
            ImplicitCondition::NonZero(right_boundary),
        ],
    );

    assert_eq!(rendered, vec!["x^2 - 1 > 0"]);
}

#[test]
fn test_factored_expanded_square_denominator_reduces_quotient_nonnegative_condition() {
    let mut ctx = Context::new();
    let quotient =
        cas_parser::parse("(x^2 + 2*x) / (x^2 + 2*x + 1)", &mut ctx).expect("parse quotient");
    let denominator_base = cas_parser::parse("x + 1", &mut ctx).expect("parse denominator base");
    let left_boundary = cas_parser::parse("x", &mut ctx).expect("parse left boundary");
    let right_boundary = cas_parser::parse("x + 2", &mut ctx).expect("parse right boundary");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonNegative(quotient),
            ImplicitCondition::NonZero(denominator_base),
            ImplicitCondition::NonZero(left_boundary),
            ImplicitCondition::NonZero(right_boundary),
        ],
    );

    assert_eq!(rendered, vec!["x^2 + 2 * x > 0"]);
}

#[test]
fn test_factored_rational_nonnegative_condition_is_dropped() {
    let mut ctx = Context::new();
    let expr = cas_parser::parse("1 - 1/(x^2 + 1)^2", &mut ctx).expect("parse expression");

    let rendered = render_conditions_normalized(&mut ctx, &[ImplicitCondition::NonNegative(expr)]);

    assert!(rendered.is_empty(), "unexpected conditions: {rendered:?}");
}

#[test]
fn test_reciprocal_square_gap_nonnegative_condition_is_not_dropped_without_domain_guards() {
    let mut ctx = Context::new();
    let expr = cas_parser::parse("1 - 1/x^2", &mut ctx).expect("parse expression");

    let rendered = render_conditions_normalized(&mut ctx, &[ImplicitCondition::NonNegative(expr)]);

    assert_eq!(rendered, vec!["1 - 1 / x^2 ≥ 0"]);
}

#[test]
fn test_even_power_gap_positive_condition_dominates_base_nonzero_condition() {
    let mut ctx = Context::new();
    let base = cas_parser::parse("x", &mut ctx).expect("parse x");
    let gap = cas_parser::parse("x^2 - 1", &mut ctx).expect("parse gap");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonZero(base),
            ImplicitCondition::Positive(gap),
        ],
    );

    assert_eq!(rendered, vec!["x^2 - 1 > 0"]);
}

#[test]
fn test_scaled_even_power_gap_positive_condition_dominates_base_nonzero_condition() {
    let mut ctx = Context::new();
    let base = cas_parser::parse("x", &mut ctx).expect("parse x");
    let gap = cas_parser::parse("4*x^2 - 1", &mut ctx).expect("parse scaled gap");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonZero(base),
            ImplicitCondition::Positive(gap),
        ],
    );

    assert_eq!(rendered, vec!["4 * x^2 - 1 > 0"]);
}

#[test]
fn test_shifted_even_power_gap_positive_condition_dominates_shifted_base_nonzero_condition() {
    let mut ctx = Context::new();
    let base = cas_parser::parse("x + 1", &mut ctx).expect("parse shifted base");
    let gap = cas_parser::parse("x^2 + 2*x", &mut ctx).expect("parse shifted gap");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonZero(base),
            ImplicitCondition::Positive(gap),
        ],
    );

    assert_eq!(rendered, vec!["x^2 + 2 * x > 0"]);
}

#[test]
fn test_scaled_shifted_even_power_gap_positive_condition_dominates_scaled_base_nonzero_condition() {
    let mut ctx = Context::new();
    let base = cas_parser::parse("2*x + 1", &mut ctx).expect("parse scaled shifted base");
    let gap = cas_parser::parse("x^2 + x", &mut ctx).expect("parse scaled shifted gap");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonZero(base),
            ImplicitCondition::Positive(gap),
        ],
    );

    assert_eq!(rendered, vec!["x^2 + x > 0"]);
}

#[test]
fn test_scaled_factor_guards_promote_nonnegative_gap_to_positive_condition() {
    let mut ctx = Context::new();
    let base = cas_parser::parse("x", &mut ctx).expect("parse x");
    let gap = cas_parser::parse("4*x^2 - 1", &mut ctx).expect("parse scaled gap");
    let left_boundary = cas_parser::parse("2*x - 1", &mut ctx).expect("parse left boundary");
    let right_boundary =
        cas_parser::parse("4*x + 2", &mut ctx).expect("parse scaled right boundary");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonNegative(gap),
            ImplicitCondition::NonZero(base),
            ImplicitCondition::NonZero(left_boundary),
            ImplicitCondition::NonZero(right_boundary),
        ],
    );

    assert_eq!(rendered, vec!["4 * x^2 - 1 > 0"]);
}

#[test]
fn test_negative_scaled_even_power_gap_does_not_dominate_base_nonzero_condition() {
    let mut ctx = Context::new();
    let base = cas_parser::parse("x", &mut ctx).expect("parse x");
    let gap = cas_parser::parse("1 - 4*x^2", &mut ctx).expect("parse negative scaled gap");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonZero(base),
            ImplicitCondition::Positive(gap),
        ],
    );

    assert_eq!(rendered, vec!["x ≠ 0", "1 - 4 * x^2 > 0"]);
}

#[test]
fn test_negative_shifted_even_power_gap_does_not_dominate_shifted_base_nonzero_condition() {
    let mut ctx = Context::new();
    let base = cas_parser::parse("x + 1", &mut ctx).expect("parse shifted base");
    let gap = cas_parser::parse("-x^2 - 2*x", &mut ctx).expect("parse negative shifted gap");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonZero(base),
            ImplicitCondition::Positive(gap),
        ],
    );

    assert_eq!(rendered, vec!["x + 1 ≠ 0", "-x^2 - 2 * x > 0"]);
}

#[test]
fn test_intrinsically_positive_even_power_sum_does_not_dominate_base_nonzero_condition() {
    let mut ctx = Context::new();
    let base = cas_parser::parse("x", &mut ctx).expect("parse x");
    let positive_sum = cas_parser::parse("x^2 + 1", &mut ctx).expect("parse positive sum");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonZero(base),
            ImplicitCondition::Positive(positive_sum),
        ],
    );

    assert_eq!(rendered, vec!["x ≠ 0"]);
}

#[test]
fn test_factored_positive_even_power_condition_reduces_to_base_nonzero_guard() {
    let mut ctx = Context::new();
    let expr = cas_parser::parse("x^8 + 4*x^6 + 6*x^4 + 4*x^2", &mut ctx)
        .expect("parse factored positive even-power condition");

    let rendered = render_conditions_normalized(&mut ctx, &[ImplicitCondition::Positive(expr)]);

    assert_eq!(rendered, vec!["x ≠ 0"]);
}

#[test]
fn test_positive_multiple_even_power_product_reduces_to_atomic_nonzero_guards() {
    let mut ctx = Context::new();
    let expr = cas_parser::parse("x^2 * z^2", &mut ctx)
        .expect("parse positive even-power product condition");

    let rendered = render_conditions_normalized(&mut ctx, &[ImplicitCondition::Positive(expr)]);

    assert_eq!(rendered, vec!["x ≠ 0", "z ≠ 0"]);
}

#[test]
fn test_positive_even_power_shift_gap_reduces_to_base_nonzero_guard() {
    let mut ctx = Context::new();
    let expr = cas_parser::parse("(x^2 + 1)^4 - 1", &mut ctx)
        .expect("parse positive even-power shift gap");

    let rendered = render_conditions_normalized(&mut ctx, &[ImplicitCondition::Positive(expr)]);

    assert_eq!(rendered, vec!["x ≠ 0"]);
}

#[test]
fn test_combined_even_power_positive_keeps_nonzero_replacement_guard() {
    let mut ctx = Context::new();
    let base = cas_parser::parse("x", &mut ctx).expect("parse base");
    let even_power = cas_parser::parse("x^2", &mut ctx).expect("parse even power");

    let rendered = render_conditions_normalized(
        &mut ctx,
        &[
            ImplicitCondition::NonZero(base),
            ImplicitCondition::NonNegative(even_power),
        ],
    );

    assert_eq!(rendered, vec!["x ≠ 0"]);
}

#[test]
fn test_positive_even_power_with_sign_varying_cofactor_keeps_positive_guard() {
    let mut ctx = Context::new();
    let expr = cas_parser::parse("x^2*(x^2 - 1)", &mut ctx)
        .expect("parse sign-varying even-power condition");

    let rendered = render_conditions_normalized(&mut ctx, &[ImplicitCondition::Positive(expr)]);

    assert!(!rendered.iter().any(|condition| condition == "x ≠ 0"));
    assert!(rendered.iter().any(|condition| condition.ends_with("> 0")));
}

#[test]
fn test_positive_condition_keeps_opposite_orientation_distinct() {
    let mut ctx = Context::new();
    let positive_x = cas_parser::parse("x", &mut ctx).expect("parse x");
    let negative_x = cas_parser::parse("-x", &mut ctx).expect("parse -x");

    let normalized = normalize_and_dedupe_conditions(
        &mut ctx,
        &[
            ImplicitCondition::Positive(positive_x),
            ImplicitCondition::Positive(negative_x),
        ],
    );

    let rendered: Vec<_> = normalized.iter().map(|cond| cond.display(&ctx)).collect();
    assert_eq!(rendered, vec!["x > 0", "-x > 0"]);
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
fn test_nonzero_perfect_square_trinomial_collapses_to_base_guard() {
    let mut ctx = Context::new();
    let square = cas_parser::parse("x^2 + 2*x + 1", &mut ctx).expect("parse square");
    let base = cas_parser::parse("x + 1", &mut ctx).expect("parse base");

    let normalized = normalize_and_dedupe_conditions(
        &mut ctx,
        &[
            ImplicitCondition::NonZero(square),
            ImplicitCondition::NonZero(base),
        ],
    );

    let rendered: Vec<_> = normalized.iter().map(|cond| cond.display(&ctx)).collect();
    println!("Rendered perfect-square guards: {:?}", rendered);

    assert_eq!(rendered, vec!["x + 1 ≠ 0"]);
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

#[test]
fn test_positive_base_dominates_powered_nonzero_condition() {
    let mut ctx = Context::new();
    let y = ctx.var("y");
    let inverse_root = cas_parser::parse("y^(-1/2)", &mut ctx).expect("parse y^(-1/2)");

    let normalized = normalize_and_dedupe_conditions(
        &mut ctx,
        &[
            ImplicitCondition::NonZero(inverse_root),
            ImplicitCondition::Positive(y),
        ],
    );

    let rendered: Vec<_> = normalized.iter().map(|cond| cond.display(&ctx)).collect();
    println!("Rendered positive-base powered guards: {:?}", rendered);

    assert_eq!(rendered, vec!["y > 0"]);
}
