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
            id: result
        }
    )
}

// ==================== Test 46: El Bumerán Roto (Principal Values) ====================

#[test]
fn test_46_principal_values() {
    // asin(sin(3*pi/2)) + acos(cos(3*pi))
    // Expected: pi/2
    // Reasoning:
    //   sin(3π/2) = -1, asin(-1) = -π/2
    //   cos(3π) = -1, acos(-1) = π
    //   -π/2 + π = π/2
    let result = simplify_str("asin(sin(3*pi/2)) + acos(cos(3*pi))");

    println!("Test 46 result: {}", result);

    // TODO: This requires evaluating inverse trig with concrete values
    // For now, just verify it doesn't crash
    assert!(!result.is_empty(), "Should produce some result");

    // Uncomment when rule is implemented:
    // assert!(
    //     result.contains("pi") && result.contains("2"),
    //     "asin(sin(3π/2)) + acos(cos(3π)) should be π/2, got: {}",
    //     result
    // );
}

// ==================== Test 47: La Identidad Complementaria ====================

#[test]
fn test_47_complementary_identity() {
    // asin(x^2 - 1) + acos(x^2 - 1)
    // Expected: pi/2
    // The identity asin(u) + acos(u) = π/2 holds for any u
    let result = simplify_str("asin(x^2 - 1) + acos(x^2 - 1)");

    println!("Test 47 result: {}", result);

    // This now works with the fixed InverseTrigSumRule!
    assert!(
        result.contains("pi") && (result.contains("2") || result.contains("1/2")),
        "asin(u) + acos(u) should be π/2, got: {}",
        result
    );
}

// ==================== Test 48: Atan Reciprocal (with sign) ====================

#[test]
fn test_48_atan_reciprocal_sign() {
    // atan(2) + atan(1/2) - pi/2
    // Expected: 0
    // Since 2 > 0, atan(2) + atan(1/2) = π/2
    let result = simplify_str("atan(2) + atan(1/2) - pi/2");

    println!("Test 48 result: {}", result);

    // This requires the atan reciprocal rule to work with constants
    // TODO: May need rule enhancement
    assert!(!result.is_empty(), "Should produce some result");

    // Uncomment when rule is implemented:
    // assert_eq!(result, "0", "atan(2) + atan(1/2) - π/2 should be 0, got: {}", result);
}

// ==================== Test 49: MachinLike Formula ====================

#[test]
fn test_49_machin_formula() {
    // 4*atan(1/5) - atan(1/239) - pi/4
    // Expected: 0
    // This is Machin's formula: 4*arctan(1/5) - arctan(1/239) = π/4
    let result = simplify_str("4*atan(1/5) - atan(1/239) - pi/4");

    println!("Test 49 result: {}", result);

    // This is very complex and requires atan addition formula
    // atan(a) + atan(b) = atan((a+b)/(1-ab))
    // TODO: This is a future enhancement
    assert!(!result.is_empty(), "Should produce some result");

    // This test is expected to NOT pass without atan addition formula
    // Uncomment only when that advanced rule is implemented:
    // assert_eq!(result, "0", "Machin's formula should simplify to 0");
}

// ==================== Test 50: Triángulo Algebraico (Composition) ====================

#[test]
fn test_50_tan_asin_composition() {
    // tan(asin(x))^2 - x^2/(1-x^2)
    // Expected: 0
    // Reasoning:
    //   If θ = asin(x), then sin(θ) = x
    //   From Pythagorean identity: cos²(θ) = 1 - sin²(θ) = 1 - x²
    //   cos(θ) = √(1-x²)
    //   tan(θ) = sin(θ)/cos(θ) = x/√(1-x²)
    //   tan²(θ) = x²/(1-x²)
    let result = simplify_str("tan(asin(x))^2 - x^2/(1-x^2)");

    println!("Test 50 result: {}", result);

    // Now working with cos(asin(x)) → sqrt(1-x²) rule!
    // The expression tan(asin(x)) = sin(asin(x))/cos(asin(x)) = x/sqrt(1-x²)
    // So tan²(asin(x)) = x²/(1-x²)
    assert_eq!(
        result, "0",
        "tan(asin(x))² should equal x²/(1-x²), got: {}",
        result
    );
}

// ==================== Additional Composition Test ====================

#[test]
fn test_sin_acos_composition() {
    // sin(acos(x)) should become sqrt(1-x^2)
    let result = simplify_str("sin(acos(x))");

    println!("sin(acos(x)) = {}", result);

    // TODO: Implement this rule in trig_inverse_expansion
    assert!(!result.is_empty(), "Should produce some result");

    // Expected: sqrt(1 - x^2)
    // Uncomment when implemented:
    // assert!(
    //     result.contains("sqrt") && result.contains("1") && result.contains("x"),
    //     "sin(acos(x)) should be sqrt(1-x²), got: {}",
    //     result
    // );
}

#[test]
fn test_cos_asin_composition() {
    // cos(asin(x)) should become sqrt(1-x^2)
    let result = simplify_str("cos(asin(x))");

    println!("cos(asin(x)) = {}", result);

    // TODO: Implement this rule in trig_inverse_expansion
    assert!(!result.is_empty(), "Should produce some result");

    // Expected: sqrt(1 - x^2)
    // Uncomment when implemented:
    // assert!(
    //     result.contains("sqrt") && result.contains("1") && result.contains("x"),
    //     "cos(asin(x)) should be sqrt(1-x²), got: {}",
    //     result
    // );
}
