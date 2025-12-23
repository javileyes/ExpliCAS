//! Tests for trig function name canonicalization
//! Verifies that short/long forms unify to canonical forms

use cas_engine::Simplifier;

fn simplify_str(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let mut ctx = simplifier.context.clone();

    match cas_parser::parse(input, &mut ctx) {
        Ok(expr) => {
            simplifier.context = ctx;
            let (result, _) = simplifier.simplify(expr);
            format!(
                "{}",
                cas_ast::DisplayExpr {
                    context: &simplifier.context,
                    id: result,
                }
            )
        }
        Err(e) => panic!("Parse error: {:?}", e),
    }
}

// ============ Basic Canonicalization ============

#[test]
fn test_asin_to_arcsin() {
    let result = simplify_str("asin(x)");
    assert_eq!(result, "arcsin(x)", "asin should canonicalize to arcsin");
}

#[test]
fn test_acos_to_arccos() {
    let result = simplify_str("acos(x)");
    assert_eq!(result, "arccos(x)", "acos should canonicalize to arccos");
}

#[test]
fn test_atan_to_arctan() {
    let result = simplify_str("atan(x)");
    assert_eq!(result, "arctan(x)", "atan should canonicalize to arctan");
}

#[test]
fn test_asec_to_arcsec() {
    let result = simplify_str("asec(x)");
    // asec converts to arccos(1/x) first
    assert!(
        result.contains("arccos"),
        "asec should eventually use arccos"
    );
}

// ============ Mixed Names Bug Fix ============

#[test]
fn test_mixed_names_cancel() {
    // This was the bug - mixed names didn't cancel
    let result = simplify_str("arcsin(x) - asin(x)");
    assert_eq!(
        result, "0",
        "Mixed forms should cancel after canonicalization"
    );
}

#[test]
fn test_arccos_minus_acos() {
    let result = simplify_str("arccos(x) - acos(x)");
    assert_eq!(result, "0", "arccos - acos should be 0");
}

#[test]
fn test_arctan_minus_atan() {
    let result = simplify_str("arctan(x) - atan(x)");
    assert_eq!(result, "0", "arctan - atan should be 0");
}

// ============ In Context ============

#[test]
fn test_asin_plus_acos() {
    // Both should canonicalize then InverseTrigSumRule applies
    let result = simplify_str("asin(x) + acos(x)");
    assert_eq!(
        result, "1/2 * pi",
        "asin + acos → π/2 after canonicalization"
    );
}

#[test]
fn test_mixed_in_sum() {
    // Mix of short and long should all canonicalize
    let result = simplify_str("asin(x) + arccos(x)");
    assert_eq!(result, "1/2 * pi", "Mixed forms should work in sum");
}

// ============ Idempotency ============

#[test]
fn test_arcsin_stays_arcsin() {
    let result = simplify_str("arcsin(x)");
    assert_eq!(result, "arcsin(x)", "arcsin should stay as arcsin");
}

#[test]
fn test_arccos_stays_arccos() {
    let result = simplify_str("arccos(x)");
    assert_eq!(result, "arccos(x)", "arccos should stay as arccos");
}

// ===== Nested Functions ============

#[test]
fn test_nested_canonicalization() {
    let result = simplify_str("asin(acos(x))");
    assert!(
        result.contains("arcsin") && result.contains("arccos"),
        "Nested should canonicalize both, got: {}",
        result
    );
}
