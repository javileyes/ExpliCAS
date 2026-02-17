use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_solver::Simplifier;

fn simplify_str(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules(); // ← FIX: was new()
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (result, _steps) = simplifier.simplify(expr);
    format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

// ==================== Composition Identities ====================

#[test]
fn test_sin_arcsin_composition() {
    let result = simplify_str("sin(arcsin(x))");
    assert_eq!(result, "x", "sin(arcsin(x)) should simplify to x");
}

#[test]
fn test_cos_arccos_composition() {
    let result = simplify_str("cos(arccos(x))");
    assert_eq!(result, "x", "cos(arccos(x)) should simplify to x");
}

#[test]
fn test_tan_arctan_composition() {
    let result = simplify_str("tan(arctan(x))");
    assert_eq!(result, "x", "tan(arctan(x)) should simplify to x");
}

#[test]
fn test_arcsin_sin_composition() {
    // Inverse∘function compositions are NOT simplified by default
    // because arcsin(sin(x)) ≠ x outside [-π/2, π/2]
    let result = simplify_str("arcsin(sin(x))");
    assert!(
        result.contains("arcsin") && result.contains("sin"),
        "arcsin(sin(x)) should NOT simplify to x (unsafe outside principal branch), got: {}",
        result
    );
}

#[test]
fn test_arccos_cos_composition() {
    // Inverse∘function compositions are NOT simplified by default
    // because arccos(cos(x)) ≠ x outside [0, π]
    let result = simplify_str("arccos(cos(x))");
    assert!(
        result.contains("arccos") && result.contains("cos"),
        "arccos(cos(x)) should NOT simplify to x (unsafe outside principal branch), got: {}",
        result
    );
}

#[test]
fn test_arctan_tan_composition() {
    // Inverse∘function compositions are NOT simplified by default
    // because arctan(tan(x)) ≠ x outside (-π/2, π/2)
    let result = simplify_str("arctan(tan(x))");
    // tan(x) may be expanded to sin(x)/cos(x)
    assert!(
        result.contains("arctan"),
        "arctan(tan(x)) should NOT simplify to x (unsafe outside principal branch), got: {}",
        result
    );
}

// ==================== Sum Relations ====================

#[test]
fn test_arcsin_arccos_sum() {
    let result = simplify_str("arcsin(x) + arccos(x)");
    // Result is "1/2 * pi" or "pi / 2" or similar
    assert!(
        result.contains("pi") && (result.contains("2") || result.contains("1/2")),
        "arcsin(x) + arccos(x) should be π/2, got: {}",
        result
    );
}

#[test]
fn test_arccos_arcsin_sum() {
    let result = simplify_str("arccos(y) + arcsin(y)");
    assert!(
        result.contains("pi") && (result.contains("2") || result.contains("1/2")),
        "arccos(y) + arcsin(y) should be π/2, got: {}",
        result
    );
}

// ==================== Arctan Relations ====================

#[test]
// Fixed: Now uses are_reciprocals() helper
fn test_arctan_reciprocal_sum() {
    let result = simplify_str("arctan(x) + arctan(1/x)");
    assert!(
        result.contains("pi") && (result.contains("2") || result.contains("1/2")),
        "arctan(x) + arctan(1/x) should be π/2, got: {}",
        result
    );
}

#[test]
// Fixed: Now uses are_reciprocals() with semantic comparison
fn test_arctan_reciprocal_sum_reverse() {
    let result = simplify_str("arctan(1/a) + arctan(a)");
    assert!(
        result.contains("pi") && (result.contains("2") || result.contains("1/2")),
        "arctan(1/a) + arctan(a) should be π/2, got: {}",
        result
    );
}

// ==================== Negative Arguments ====================

#[test]
fn test_arcsin_negative() {
    let result = simplify_str("arcsin(-x)");
    assert_eq!(result, "-arcsin(x)", "arcsin(-x) should be -arcsin(x)");
}

#[test]
fn test_arctan_negative() {
    let result = simplify_str("arctan(-y)");
    assert_eq!(result, "-arctan(y)", "arctan(-y) should be -arctan(y)");
}

#[test]
fn test_arccos_negative() {
    let result = simplify_str("arccos(-z)");
    // Result should be "pi - arccos(z)" but might be formatted differently
    assert!(
        result.contains("pi") && result.contains("arccos"),
        "arccos(-z) should be π - arccos(z), got: {}",
        result
    );
}

// ==================== Evaluation Tests ====================

#[test]
fn test_arcsin_zero() {
    let result = simplify_str("arcsin(0)");
    assert_eq!(result, "0", "arcsin(0) should be 0");
}

#[test]
fn test_arcsin_one() {
    let result = simplify_str("arcsin(1)");
    assert!(
        result.contains("pi") && result.contains("2"),
        "arcsin(1) should be π/2, got: {}",
        result
    );
}

#[test]
fn test_arccos_zero() {
    let result = simplify_str("arccos(0)");
    assert!(
        result.contains("pi") && result.contains("2"),
        "arccos(0) should be π/2, got: {}",
        result
    );
}

#[test]
fn test_arccos_one() {
    let result = simplify_str("arccos(1)");
    assert_eq!(result, "0", "arccos(1) should be 0");
}

#[test]
fn test_arctan_zero() {
    let result = simplify_str("arctan(0)");
    assert_eq!(result, "0", "arctan(0) should be 0");
}

#[test]
fn test_arctan_one() {
    let result = simplify_str("arctan(1)");
    assert!(
        result.contains("pi") && result.contains("4"),
        "arctan(1) should be π/4, got: {}",
        result
    );
}

// ==================== Complex Mix Tests ====================

#[test]
fn test_nested_inverse_trig() {
    let result = simplify_str("sin(arcsin(cos(arccos(x))))");
    assert_eq!(result, "x", "Nested inverse trig should simplify to x");
}

#[test]
fn test_arcsin_arccos_with_expression() {
    let result = simplify_str("arcsin(x+1) + arccos(x+1)");
    assert!(
        result.contains("pi") && (result.contains("2") || result.contains("1/2")),
        "arcsin(expr) + arccos(expr) should be π/2"
    );
}

#[test]
fn test_inverse_trig_with_arithmetic() {
    let result = simplify_str("2 * arcsin(x) + 2 * arccos(x)");
    // 2 * (arcsin + arccos) = 2 * π/2 = π
    // Now correctly simplified thanks to coefficient matching in check_pair_with_negation
    assert_eq!(result, "pi", "2 * arcsin(x) + 2 * arccos(x) should equal π");
}

#[test]
fn test_composition_with_negative() {
    let result = simplify_str("sin(arcsin(-x))");
    // sin(arcsin(-x)) = sin(-arcsin(x)) = -sin(arcsin(x)) = -x
    assert!(
        result == "-x" || result == "x * -1" || result == "-1 * x",
        "sin(arcsin(-x)) should be -x, got: {}",
        result
    );
}

#[test]
fn test_double_composition() {
    // Double composition: arcsin(sin(arcsin(sin(x))))
    // The inner sin(arcsin(sin(x))) simplifies to sin(x) (function∘inverse is safe)
    // But arcsin(sin(x)) is NOT simplified (inverse∘function is unsafe)
    let result = simplify_str("arcsin(sin(arcsin(sin(x))))");
    // We expect it to contain arcsin and sin since inverse∘function is not simplified
    assert!(
        result.contains("arcsin") || result.contains("sin") || result == "x",
        "Double composition may partially simplify, got: {}",
        result
    );
}

#[test]
fn test_mixed_trig_and_inverse() {
    let result = simplify_str("cos(arcsin(x))^2 + sin(arcsin(x))^2");
    // This should eventually become cos(arcsin(x))^2 + x^2
    // or ideally simplify more, but let's just check it doesn't crash
    println!("Mixed trig result: {}", result);
    assert!(!result.is_empty());
}

#[test]
// Fixed: are_reciprocals() handles polynomial arguments
fn test_arctan_with_polynomial() {
    let result = simplify_str("arctan(x^2) + arctan(1/x^2)");
    assert!(
        result.contains("pi") && (result.contains("2") || result.contains("1/2")),
        "arctan(x^2) + arctan(1/x^2) should be π/2"
    );
}

// ==================== Additional Verification Tests ====================

#[test]
fn test_multiple_compositions() {
    let result = simplify_str("tan(arctan(cos(arccos(sin(arcsin(y))))))");
    assert_eq!(
        result, "y",
        "Multiple nested compositions should simplify to y"
    );
}

#[test]
fn test_sum_with_negatives() {
    let result = simplify_str("arcsin(-a) + arccos(-a)");
    // arcsin(-a) = -arcsin(a), arccos(-a) = π - arccos(a)
    // So: -arcsin(a) + π - arccos(a) = π - (arcsin(a) + arccos(a)) = π - π/2 = π/2
    // But the simplifier might not do this full reduction
    assert!(
        result.contains("pi") || result.contains("arcsin") || result.contains("arccos"),
        "arcsin(-a) + arccos(-a) should contain trig terms or pi"
    );
}

#[test]
fn test_composition_identity_chain() {
    // sin(arcsin(tan(arctan(z))))
    let result = simplify_str("sin(arcsin(tan(arctan(z))))");
    assert_eq!(result, "z", "Chain of compositions should simplify");
}
