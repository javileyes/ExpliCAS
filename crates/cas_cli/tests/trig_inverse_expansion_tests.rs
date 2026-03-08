use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_solver::Simplifier;

fn simplify_str(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (result, _) = simplifier.simplify(expr);
    format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

// ========== Priority 1: Core Expansion Rules Tests ==========

#[test]
fn test_sin_arctan_numeric() {
    let result = simplify_str("sin(arctan(3))");
    // sin(arctan(3)) = 3/sqrt(1 + 9) = 3/sqrt(10)
    assert!(
        result.contains("3"),
        "Expected 3 in numerator, got: {}",
        result
    );
    assert!(
        result.contains("10"),
        "Expected 10 under sqrt, got: {}",
        result
    );
}

#[test]
fn test_sin_arctan_symbolic() {
    let result = simplify_str("sin(arctan(x))");
    // Should expand to x/sqrt(1+x²)
    assert!(
        result.contains("x") && result.contains("1"),
        "Expected x/√(1+x²), got: {}",
        result
    );
}

#[test]
fn test_cos_arctan_numeric() {
    let result = simplify_str("cos(arctan(4))");
    // cos(arctan(4)) = 1/sqrt(1 + 16) = 1/sqrt(17)
    assert!(
        result.contains("17"),
        "Expected 17 under sqrt, got: {}",
        result
    );
}

#[test]
fn test_cos_arctan_symbolic() {
    let result = simplify_str("cos(arctan(y))");
    // Should expand to 1/sqrt(1+y²)
    assert!(
        result.contains("y") && result.contains("1"),
        "Expected 1/√(1+y²), got: {}",
        result
    );
}

#[test]
fn test_tan_arcsin() {
    let result = simplify_str("tan(arcsin(x))");
    // Should expand to x/sqrt(1-x²)
    assert!(
        result.contains("x"),
        "Expected x in result, got: {}",
        result
    );
}

#[test]
fn test_cot_arcsin() {
    let result = simplify_str("cot(arcsin(x))");
    // Should expand to sqrt(1-x²)/x
    assert!(
        result.contains("x"),
        "Expected x in result, got: {}",
        result
    );
}

// ========== Priority 2: Secondary Expansion Rules Tests ==========

#[test]
fn test_sin_arcsec() {
    let result = simplify_str("sin(arcsec(x))");
    // Should expand to sqrt(x²-1)/x
    assert!(
        result.contains("x"),
        "Expected x in result, got: {}",
        result
    );
}

#[test]
fn test_cos_arcsec() {
    let result = simplify_str("cos(arcsec(x))");
    // Should expand to 1/x
    assert_eq!(result, "1 / x");
}

#[test]
fn test_tan_arccos() {
    let result = simplify_str("tan(arccos(x))");
    // Should expand to sqrt(1-x²)/x
    assert!(
        result.contains("x"),
        "Expected x in result, got: {}",
        result
    );
}

#[test]
fn test_cot_arccos() {
    let result = simplify_str("cot(arccos(x))");
    // Should expand to x/sqrt(1-x²)
    assert!(
        result.contains("x"),
        "Expected x in result, got: {}",
        result
    );
}

// ========== Priority 3: Reciprocal Expansion Rules Tests ==========

#[test]
fn test_sec_arctan() {
    let result = simplify_str("sec(arctan(x))");
    // Should expand to sqrt(1+x²)
    assert!(
        result.contains("x") && result.contains("1"),
        "Expected √(1+x²), got: {}",
        result
    );
}

#[test]
fn test_csc_arctan() {
    let result = simplify_str("csc(arctan(x))");
    // Should expand to sqrt(1+x²)/x
    assert!(
        result.contains("x"),
        "Expected x in result, got: {}",
        result
    );
}

#[test]
fn test_sec_arcsin() {
    let result = simplify_str("sec(arcsin(x))");
    // Should expand to 1/sqrt(1-x²)
    assert!(
        result.contains("x") && result.contains("1"),
        "Expected 1/√(1-x²), got: {}",
        result
    );
}

#[test]
fn test_csc_arcsin() {
    let result = simplify_str("csc(arcsin(x))");
    // Should expand to 1/x
    assert_eq!(result, "1 / x");
}

// ========== Integration Tests ==========

#[test]
fn test_sin_arctan_minus_algebraic() {
    // This is test_51 from mixed_trig_tests
    let result = simplify_str("sin(arctan(x)) - x / sqrt(1 + x^2)");
    // Should simplify to 0
    assert_eq!(
        result, "0",
        "Expected 0 for sin(arctan(x)) - x/√(1+x²), got: {}",
        result
    );
}

#[test]
fn test_cot_arcsin_minus_algebraic() {
    // This is test_53 from mixed_trig_tests
    let result = simplify_str("cot(arcsin(x)) - sqrt(1 - x^2) / x");
    // Should simplify to 0
    assert_eq!(
        result, "0",
        "Expected 0 for cot(arcsin(x)) - √(1-x²)/x, got: {}",
        result
    );
}

#[test]
fn test_nested_expansion() {
    // Test that expansions work in nested contexts
    let result = simplify_str("2 * sin(arctan(x))");
    // Should expand and multiply
    assert!(
        result.contains("x") && result.contains("2"),
        "Expectedexpansion with 2x, got: {}",
        result
    );
}

#[test]
fn test_composition_still_works() {
    // Make sure we didn't break basic compositions
    let result = simplify_str("tan(arctan(x + 5))");
    // Should still simplify to x + 5
    assert_eq!(
        result, "x + 5",
        "Composition broken! Expected x + 5, got: {}",
        result
    );
}

#[test]
fn test_arctan_abbreviation() {
    // Test that atan (abbreviation) also works
    let result = simplify_str("sin(atan(3))");
    assert!(
        result.contains("3") && result.contains("10"),
        "Abbreviation 'atan' not working, got: {}",
        result
    );
}

#[test]
fn test_arcsin_abbreviation() {
    // Test that asin (abbreviation) also works
    let result = simplify_str("csc(asin(x))");
    assert_eq!(
        result, "1 / x",
        "Abbreviation 'asin' not working, got: {}",
        result
    );
}
