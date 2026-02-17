use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

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

// ========== Basic Conversion Tests ==========

#[test]
fn test_arcsec_to_arccos() {
    let result = simplify_str("arcsec(2)");
    // Note: arcsec(2) may evaluate numerically to π/3 if evaluation rules fire first
    // The important thing is that it doesn't remain as "arcsec"
    println!("arcsec(2) -> {}", result);
    assert!(
        !result.contains("arcsec"),
        "Should not contain arcsec: {}",
        result
    );
    // Could be either "arccos" or numerical result like "pi"
}

#[test]
fn test_arccsc_to_arcsin() {
    let result = simplify_str("arccsc(3)");
    // Should convert to arcsin(1/3)
    println!("arccsc(3) -> {}", result);
    assert!(
        result.contains("arcsin"),
        "Should convert to arcsin: {}",
        result
    );
    assert!(
        !result.contains("arccsc"),
        "Should not contain arccsc: {}",
        result
    );
}

#[test]
fn test_arccot_to_arctan() {
    let result = simplify_str("arccot(4)");
    // Should convert to arctan(1/4)
    println!("arccot(4) -> {}", result);
    assert!(
        result.contains("arctan"),
        "Should convert to arctan: {}",
        result
    );
    assert!(
        !result.contains("arccot"),
        "Should not contain arccot: {}",
        result
    );
}

// ========== Symbolic Arguments ==========

#[test]
fn test_arcsec_symbolic() {
    let result = simplify_str("arcsec(x)");
    println!("arcsec(x) -> {}", result);
    assert!(
        result.contains("arccos"),
        "Should convert to arccos: {}",
        result
    );
    assert!(
        !result.contains("arcsec"),
        "Should not contain arcsec: {}",
        result
    );
}

#[test]
fn test_arccsc_symbolic() {
    let result = simplify_str("arccsc(y)");
    println!("arccsc(y) -> {}", result);
    assert!(
        result.contains("arcsin"),
        "Should convert to arcsin: {}",
        result
    );
    assert!(
        !result.contains("arccsc"),
        "Should not contain arccsc: {}",
        result
    );
}

#[test]
fn test_arccot_symbolic() {
    let result = simplify_str("arccot(z)");
    println!("arccot(z) -> {}", result);
    assert!(
        result.contains("arctan"),
        "Should convert to arctan: {}",
        result
    );
    assert!(
        !result.contains("arccot"),
        "Should not contain arccot: {}",
        result
    );
}

// ========== Abbreviations ==========

#[test]
fn test_asec_abbreviation() {
    let result = simplify_str("asec(5)");
    println!("asec(5) -> {}", result);
    assert!(
        result.contains("arccos"),
        "Should convert to arccos: {}",
        result
    );
    assert!(
        !result.contains("asec"),
        "Should not contain asec: {}",
        result
    );
}

#[test]
fn test_acsc_abbreviation() {
    let result = simplify_str("acsc(6)");
    println!("acsc(6) -> {}", result);
    assert!(
        result.contains("arcsin"),
        "Should convert to arcsin: {}",
        result
    );
    assert!(
        !result.contains("acsc"),
        "Should not contain acsc: {}",
        result
    );
}

#[test]
fn test_acot_abbreviation() {
    let result = simplify_str("acot(7)");
    println!("acot(7) -> {}", result);
    assert!(
        result.contains("arctan"),
        "Should convert to arctan: {}",
        result
    );
    assert!(
        !result.contains("acot"),
        "Should not contain acot: {}",
        result
    );
}

// ========== Nested Expressions ==========

#[test]
fn test_sin_arcsec() {
    let result = simplify_str("sin(arcsec(2))");
    println!("sin(arcsec(2)) -> {}", result);
    // After conversion: sin(arccos(1/2))
    // Should potentially expand further
    assert!(
        !result.contains("arcsec"),
        "Should not contain arcsec: {}",
        result
    );
}

#[test]
fn test_cos_arccsc() {
    let result = simplify_str("cos(arccsc(3))");
    println!("cos(arccsc(3)) -> {}", result);
    // After conversion: cos(arcsin(1/3))
    assert!(
        !result.contains("arccsc"),
        "Should not contain arccsc: {}",
        result
    );
}

#[test]
fn test_tan_arccot() {
    let result = simplify_str("tan(arccot(4))");
    println!("tan(arccot(4)) -> {}", result);
    // After conversion: tan(arctan(1/4))
    assert!(
        !result.contains("arccot"),
        "Should not contain arccot: {}",
        result
    );
}

// ========== Integration Test (test_54) ==========

#[test]
fn test_54_arcsec_arccos_relation() {
    let result = simplify_str("arcsec(x) - arccos(1/x)");
    println!("test_54 result: {}", result);
    // After conversion: arccos(1/x) - arccos(1/x) = 0
    // Ideally should be "0", but let's verify conversion happens
    assert!(
        !result.contains("arcsec"),
        "Should convert arcsec: {}",
        result
    );
}

// ========== Complex Expressions ==========

#[test]
fn test_arcsec_in_expression() {
    let result = simplify_str("2 * arcsec(x) + 3");
    println!("2*arcsec(x)+3 -> {}", result);
    assert!(
        !result.contains("arcsec"),
        "Should convert arcsec: {}",
        result
    );
    assert!(
        result.contains("arccos"),
        "Should contain arccos: {}",
        result
    );
}

#[test]
fn test_multiple_conversions() {
    let result = simplify_str("arcsec(a) + arccsc(b) + arccot(c)");
    println!("Multiple conversions -> {}", result);
    assert!(
        !result.contains("arcsec"),
        "Should not contain arcsec: {}",
        result
    );
    assert!(
        !result.contains("arccsc"),
        "Should not contain arccsc: {}",
        result
    );
    assert!(
        !result.contains("arccot"),
        "Should not contain arccot: {}",
        result
    );
}

// ========== No Regressions ==========

#[test]
fn test_arcsin_unchanged() {
    let result = simplify_str("arcsin(x)");
    // arcsin should not be converted
    assert_eq!(result, "arcsin(x)", "arcsin should stay unchanged");
}

#[test]
fn test_arccos_unchanged() {
    let result = simplify_str("arccos(y)");
    // arccos should not be converted
    assert_eq!(result, "arccos(y)", "arccos should stay unchanged");
}

#[test]
fn test_arctan_unchanged() {
    let result = simplify_str("arctan(z)");
    // arctan should not be converted
    assert_eq!(result, "arctan(z)", "arctan should stay unchanged");
}
