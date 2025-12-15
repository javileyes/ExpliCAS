use cas_ast::DisplayExpr;
use cas_engine::Simplifier;
use cas_parser::parse;

fn simplify_str(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (result, _steps) = simplifier.simplify(expr);
    format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result,
        }
    )
}

// ==================== Evaluation Tests ====================

#[test]
fn test_sinh_zero() {
    assert_eq!(simplify_str("sinh(0)"), "0", "sinh(0) should equal 0");
}

#[test]
fn test_cosh_zero() {
    assert_eq!(simplify_str("cosh(0)"), "1", "cosh(0) should equal 1");
}

#[test]
fn test_tanh_zero() {
    assert_eq!(simplify_str("tanh(0)"), "0", "tanh(0) should equal 0");
}

#[test]
fn test_asinh_zero() {
    assert_eq!(simplify_str("asinh(0)"), "0", "asinh(0) should equal 0");
}

#[test]
fn test_acosh_one() {
    assert_eq!(simplify_str("acosh(1)"), "0", "acosh(1) should equal 0");
}

#[test]
fn test_atanh_zero() {
    assert_eq!(simplify_str("atanh(0)"), "0", "atanh(0) should equal 0");
}

// ==================== Composition Tests ====================

#[test]
fn test_sinh_asinh() {
    assert_eq!(
        simplify_str("sinh(asinh(x))"),
        "x",
        "sinh(asinh(x)) should simplify to x"
    );
}

#[test]
fn test_cosh_acosh() {
    assert_eq!(
        simplify_str("cosh(acosh(y))"),
        "y",
        "cosh(acosh(y)) should simplify to y"
    );
}

#[test]
fn test_tanh_atanh() {
    assert_eq!(
        simplify_str("tanh(atanh(z))"),
        "z",
        "tanh(atanh(z)) should simplify to z"
    );
}

#[test]
fn test_asinh_sinh() {
    assert_eq!(
        simplify_str("asinh(sinh(a))"),
        "a",
        "asinh(sinh(a)) should simplify to a"
    );
}

#[test]
fn test_acosh_cosh() {
    assert_eq!(
        simplify_str("acosh(cosh(b))"),
        "b",
        "acosh(cosh(b)) should simplify to b"
    );
}

#[test]
fn test_atanh_tanh() {
    assert_eq!(
        simplify_str("atanh(tanh(c))"),
        "c",
        "atanh(tanh(c)) should simplify to c"
    );
}

// ==================== Negative Argument Tests ====================

#[test]
fn test_sinh_negative() {
    assert_eq!(
        simplify_str("sinh(-x)"),
        "-sinh(x)",
        "sinh(-x) should equal -sinh(x)"
    );
}

#[test]
fn test_cosh_negative() {
    assert_eq!(
        simplify_str("cosh(-y)"),
        "cosh(y)",
        "cosh(-y) should equal cosh(y)"
    );
}

#[test]
fn test_tanh_negative() {
    assert_eq!(
        simplify_str("tanh(-z)"),
        "-tanh(z)",
        "tanh(-z) should equal -tanh(z)"
    );
}

#[test]
fn test_asinh_negative() {
    assert_eq!(
        simplify_str("asinh(-a)"),
        "-asinh(a)",
        "asinh(-a) should equal -asinh(a)"
    );
}

#[test]
fn test_atanh_negative() {
    assert_eq!(
        simplify_str("atanh(-b)"),
        "-atanh(b)",
        "atanh(-b) should equal -atanh(b)"
    );
}

// ==================== Pythagorean Identity Tests ====================

#[test]
fn test_hyperbolic_pythagorean() {
    let result = simplify_str("cosh(x)^2 - sinh(x)^2");
    assert_eq!(result, "1", "cosh²(x) - sinh²(x) should equal 1");
}

#[test]
fn test_hyperbolic_pythagorean_reverse() {
    let result = simplify_str("sinh(y)^2 - cosh(y)^2");
    assert_eq!(result, "-1", "sinh²(y) - cosh²(y) should equal -1");
}

// ==================== Complex/Nested Tests ====================

#[test]
fn test_nested_hyperbolic() {
    assert_eq!(
        simplify_str("sinh(asinh(cosh(acosh(x))))"),
        "x",
        "Nested composition should simplify to x"
    );
}

#[test]
fn test_hyperbolic_with_negatives() {
    assert_eq!(
        simplify_str("sinh(asinh(-x))"),
        "-x",
        "sinh(asinh(-x)) should simplify to -x"
    );
}

#[test]
fn test_composition_identity_chain() {
    assert_eq!(
        simplify_str("asinh(sinh(asinh(sinh(x))))"),
        "x",
        "Chain of compositions should simplify to x"
    );
}

#[test]
fn test_double_composition() {
    assert_eq!(
        simplify_str("cosh(acosh(cosh(acosh(z))))"),
        "z",
        "Double composition should simplify to z"
    );
}

#[test]
fn test_composition_with_negative() {
    assert_eq!(
        simplify_str("tanh(atanh(-y))"),
        "-y",
        "tanh(atanh(-y)) should simplify to -y"
    );
}

#[test]
fn test_multiple_compositions() {
    let result = simplify_str("asinh(sinh(tanh(atanh(u))))");
    assert_eq!(result, "u", "Multiple compositions should simplify to u");
}

#[test]
fn test_mixed_hyperbolic() {
    // This one might not fully simplify depending on rule order,
    // but at least check it doesn't crash
    let result = simplify_str("sinh(x) + cosh(x)");
    assert!(
        result.contains("sinh") || result.contains("cosh"),
        "Mixed hyperbolic should contain function calls"
    );
}

#[test]
fn test_pythagorean_with_variable() {
    let result = simplify_str("cosh(a+b)^2 - sinh(a+b)^2");
    assert_eq!(
        result, "1",
        "Pythagorean identity should work with complex arguments"
    );
}
