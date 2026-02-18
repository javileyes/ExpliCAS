use super::*;
use cas_ast::{Context, Expr};
use cas_math::numeric_eval::{as_rational_const, as_rational_const_depth};
use cas_parser::parse;

#[test]
fn test_is_one() {
    let mut ctx = Context::new();
    let one = ctx.num(1);
    let two = ctx.num(2);
    let zero = ctx.num(0);
    let x = ctx.var("x");

    assert!(is_one(&ctx, one));
    assert!(!is_one(&ctx, two));
    assert!(!is_one(&ctx, zero));
    assert!(!is_one(&ctx, x));
}

#[test]
fn test_is_zero() {
    let mut ctx = Context::new();
    let zero = ctx.num(0);
    let one = ctx.num(1);
    let x = ctx.var("x");

    assert!(is_zero(&ctx, zero));
    assert!(!is_zero(&ctx, one));
    assert!(!is_zero(&ctx, x));
}

#[test]
fn test_get_integer() {
    let mut ctx = Context::new();
    let five = ctx.num(5);
    let half = ctx.add(Expr::Number(num_rational::BigRational::new(
        1.into(),
        2.into(),
    )));
    let x = ctx.var("x");

    assert_eq!(get_integer(&ctx, five), Some(5));
    assert_eq!(get_integer(&ctx, half), None); // Not an integer
    assert_eq!(get_integer(&ctx, x), None);
}

#[test]
fn test_flatten_add() {
    let mut ctx = Context::new();
    let expr = parse("a + b + c", &mut ctx).unwrap();
    let terms = crate::nary::add_leaves(&ctx, expr);
    assert_eq!(terms.len(), 3);
}

#[test]
fn test_flatten_add_sub_chain() {
    let mut ctx = Context::new();
    let expr = parse("a + b - c", &mut ctx).unwrap();
    let terms = flatten_add_sub_chain(&mut ctx, expr);
    // Should have 3 terms: a, b, Neg(c)
    assert_eq!(terms.len(), 3);
}

#[test]
fn test_flatten_mul() {
    let mut ctx = Context::new();
    let expr = parse("a * b * c", &mut ctx).unwrap();
    let factors = crate::nary::mul_leaves(&ctx, expr);
    assert_eq!(factors.len(), 3);
}

#[test]
fn test_flatten_mul_chain_with_neg() {
    let mut ctx = Context::new();
    let expr = parse("-a * b", &mut ctx).unwrap();
    let factors = flatten_mul_chain(&mut ctx, expr);
    // Should have factors including -1
    assert!(factors.len() >= 2);
}

#[test]
fn test_is_pi_over_n() {
    let mut ctx = Context::new();
    let pi_over_2 = build_pi_over_n(&mut ctx, 2);
    let pi_over_4 = build_pi_over_n(&mut ctx, 4);

    assert!(is_pi_over_n(&ctx, pi_over_2, 2));
    assert!(!is_pi_over_n(&ctx, pi_over_2, 4));
    assert!(is_pi_over_n(&ctx, pi_over_4, 4));
}

#[test]
fn test_is_half() {
    let mut ctx = Context::new();
    let half = ctx.add(Expr::Number(num_rational::BigRational::new(
        1.into(),
        2.into(),
    )));
    let one = ctx.num(1);

    assert!(is_half(&ctx, half));
    assert!(!is_half(&ctx, one));
}

#[test]
fn test_is_pi() {
    let mut ctx = Context::new();
    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
    let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let x = ctx.var("x");

    assert!(is_pi(&ctx, pi));
    assert!(!is_pi(&ctx, e));
    assert!(!is_pi(&ctx, x));
}

#[test]
fn test_as_rational_const_number() {
    let mut ctx = Context::new();
    let half = ctx.rational(1, 2);
    let result = as_rational_const(&ctx, half);
    assert_eq!(
        result,
        Some(num_rational::BigRational::new(1.into(), 2.into()))
    );
}

#[test]
fn test_as_rational_const_div() {
    let mut ctx = Context::new();
    let one = ctx.num(1);
    let two = ctx.num(2);
    let div = ctx.add(Expr::Div(one, two));
    let result = as_rational_const(&ctx, div);
    assert_eq!(
        result,
        Some(num_rational::BigRational::new(1.into(), 2.into()))
    );
}

#[test]
fn test_as_rational_const_neg_div() {
    let mut ctx = Context::new();
    let one = ctx.num(1);
    let two = ctx.num(2);
    let div = ctx.add(Expr::Div(one, two));
    let neg = ctx.add(Expr::Neg(div));
    let result = as_rational_const(&ctx, neg);
    assert_eq!(
        result,
        Some(num_rational::BigRational::new((-1).into(), 2.into()))
    );
}

#[test]
fn test_as_rational_const_variable_returns_none() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    assert!(as_rational_const(&ctx, x).is_none());
}

#[test]
fn test_as_rational_const_mul_with_variable_returns_none() {
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let x = ctx.var("x");
    let mul = ctx.add(Expr::Mul(two, x));
    assert!(as_rational_const(&ctx, mul).is_none());
}

#[test]
fn test_as_rational_const_depth_budget() {
    // Build a deeply nested expression: Div(1, Div(1, Div(1, ...Div(1, 2)...)))
    // Note: We can't use Neg(Neg(...)) because Context::add canonicalizes Neg(Neg(x)) -> x
    let mut ctx = Context::new();
    let one = ctx.num(1);
    let two = ctx.num(2);

    // Start with 1/2, then nest 100 levels: 1/(1/(1/(...1/2...)))
    let mut expr = ctx.add(Expr::Div(one, two));
    for _ in 0..100 {
        let one_copy = ctx.num(1);
        expr = ctx.add(Expr::Div(one_copy, expr));
    }

    // With depth=50, should return None (budget exhausted)
    assert!(as_rational_const_depth(&ctx, expr, 50).is_none());

    // With depth=200, should succeed
    // 1/(1/(1/...(1/2)...)) with even nesting = 2, odd nesting = 1/2
    // 100 levels of nesting on 1/2 => result depends on parity
    let result = as_rational_const_depth(&ctx, expr, 200);
    assert!(result.is_some());
    // 100 nestings of 1/x on 1/2: alternates between 2 and 1/2
    // Even count (100) means result is 1/2
    let expected = num_rational::BigRational::new(1.into(), 2.into());
    assert_eq!(result.unwrap(), expected);
}

/// Test that add_raw preserves operand order while add() canonicalizes
#[test]
fn test_add_raw_preserves_mul_order() {
    let mut ctx = Context::new();
    let z = ctx.var("z"); // 'z' > 'a' in ordering
    let a = ctx.var("a");

    // With add(): z * a → a * z (swapped because 'z' > 'a')
    let mul_canonical = ctx.add(Expr::Mul(z, a));
    if let Expr::Mul(l, r) = ctx.get(mul_canonical) {
        // Should be swapped to canonical order: (a, z)
        assert_eq!(*l, a, "add() should swap to put 'a' first");
        assert_eq!(*r, z, "add() should swap to put 'z' second");
    } else {
        panic!("Expected Mul expression");
    }

    // With add_raw(): z * a → z * a (preserved order)
    let mul_raw = ctx.add_raw(Expr::Mul(z, a));
    if let Expr::Mul(l, r) = ctx.get(mul_raw) {
        // Should preserve original order: (z, a)
        assert_eq!(*l, z, "add_raw() should preserve 'z' first");
        assert_eq!(*r, a, "add_raw() should preserve 'a' second");
    } else {
        panic!("Expected Mul expression");
    }
}

/// Test that Matrix*Matrix multiplication preserves order (non-commutative)
#[test]
fn test_matrix_mul_preserves_order() {
    let mut ctx = Context::new();

    // Create two different matrices A and B
    let one = ctx.num(1);
    let two = ctx.num(2);
    let matrix_a = ctx.add(Expr::Matrix {
        rows: 1,
        cols: 1,
        data: vec![one],
    });
    let matrix_b = ctx.add(Expr::Matrix {
        rows: 1,
        cols: 1,
        data: vec![two],
    });

    // A*B should NOT be swapped even though ctx.add canonicalizes
    let mul_ab = ctx.add(Expr::Mul(matrix_a, matrix_b));
    if let Expr::Mul(l, r) = ctx.get(mul_ab) {
        assert_eq!(
            *l, matrix_a,
            "Matrix A*B: A should stay first (non-commutative)"
        );
        assert_eq!(
            *r, matrix_b,
            "Matrix A*B: B should stay second (non-commutative)"
        );
    } else {
        panic!("Expected Mul expression");
    }

    // B*A should also preserve its order
    let mul_ba = ctx.add(Expr::Mul(matrix_b, matrix_a));
    if let Expr::Mul(l, r) = ctx.get(mul_ba) {
        assert_eq!(
            *l, matrix_b,
            "Matrix B*A: B should stay first (non-commutative)"
        );
        assert_eq!(
            *r, matrix_a,
            "Matrix B*A: A should stay second (non-commutative)"
        );
    } else {
        panic!("Expected Mul expression");
    }
}

// =========================================================================
// prove_nonzero contract tests (ground fallback)
// =========================================================================

#[test]
fn prove_nonzero_constants() {
    let mut ctx = Context::new();
    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
    let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let zero = ctx.num(0);
    let x = ctx.var("x");

    assert_eq!(prove_nonzero(&ctx, pi), crate::domain::Proof::Proven);
    assert_eq!(prove_nonzero(&ctx, e), crate::domain::Proof::Proven);
    assert_eq!(prove_nonzero(&ctx, zero), crate::domain::Proof::Disproven);
    assert_eq!(prove_nonzero(&ctx, x), crate::domain::Proof::Unknown);
}

#[test]
fn prove_nonzero_sqrt2() {
    // sqrt(2) = 2^(1/2) — should be provably non-zero
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let half = ctx.rational(1, 2);
    let sqrt2 = ctx.add(Expr::Pow(two, half));

    assert_eq!(prove_nonzero(&ctx, sqrt2), crate::domain::Proof::Proven);
}

#[test]
fn prove_nonzero_pow_neg_half() {
    // 2^(-1/2) — positive base, any exponent → non-zero
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let neg_half = ctx.rational(-1, 2);
    let expr = ctx.add(Expr::Pow(two, neg_half));

    assert_eq!(prove_nonzero(&ctx, expr), crate::domain::Proof::Proven);
}

#[test]
fn prove_nonzero_sqrt_function() {
    // sqrt(2) via Function node — ground fallback should handle
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let sqrt2 = ctx.call("sqrt", vec![two]);

    assert_eq!(prove_nonzero(&ctx, sqrt2), crate::domain::Proof::Proven);
}

#[test]
fn prove_nonzero_cos_pi_over_3() {
    // cos(π/3) = 1/2 ≠ 0 — ground function fallback
    let mut ctx = Context::new();
    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
    let three = ctx.num(3);
    let pi_over_3 = ctx.add(Expr::Div(pi, three));
    let cos_expr = ctx.call("cos", vec![pi_over_3]);

    assert_eq!(prove_nonzero(&ctx, cos_expr), crate::domain::Proof::Proven);
}

#[test]
fn prove_nonzero_cos_pi_over_2_is_zero() {
    // cos(π/2) = 0 — ground fallback should prove Disproven
    let mut ctx = Context::new();
    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
    let two = ctx.num(2);
    let pi_over_2 = ctx.add(Expr::Div(pi, two));
    let cos_expr = ctx.call("cos", vec![pi_over_2]);

    assert_eq!(
        prove_nonzero(&ctx, cos_expr),
        crate::domain::Proof::Disproven
    );
}

#[test]
fn prove_nonzero_variable_stays_unknown() {
    // x/x — should NOT be Proven (x could be 0)
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let x_over_x = ctx.add(Expr::Div(x, x));

    // We're testing prove_nonzero on x/x as a whole expression
    // This should be Unknown because the denominator x is Unknown
    assert_eq!(prove_nonzero(&ctx, x_over_x), crate::domain::Proof::Unknown);
}
