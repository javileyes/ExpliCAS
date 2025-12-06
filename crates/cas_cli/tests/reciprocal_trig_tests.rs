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
#[ignore] // TODO: Requires better handling of canonicalized pi/4 form
fn test_cot_pi_over_4() {
    let result = simplify_str("cot(pi/4)");
    assert_eq!(result, "1", "cot(π/4) should equal 1");
}

#[test]
#[ignore] // TODO: Requires better handling of canonicalized pi/2 form
fn test_cot_pi_over_2() {
    let result = simplify_str("cot(pi/2)");
    assert_eq!(result, "0", "cot(π/2) should equal 0");
}

#[test]
fn test_sec_zero() {
    assert_eq!(simplify_str("sec(0)"), "1", "sec(0) should equal 1");
}

#[test]
#[ignore] // TODO: Requires better handling of canonicalized pi/2 form
fn test_csc_pi_over_2() {
    let result = simplify_str("csc(pi/2)");
    assert_eq!(result, "1", "csc(π/2) should equal 1");
}

#[test]
fn test_arccot_one() {
    let result = simplify_str("arccot(1)");
    assert!(
        result.contains("pi") && result.contains("4"),
        "arccot(1) should be π/4, got: {}",
        result
    );
}

#[test]
fn test_arccot_zero() {
    let result = simplify_str("arccot(0)");
    assert!(
        result.contains("pi") && result.contains("2"),
        "arccot(0) should be π/2, got: {}",
        result
    );
}

#[test]
fn test_arcsec_one() {
    assert_eq!(simplify_str("arcsec(1)"), "0", "arcsec(1) should equal 0");
}

#[test]
fn test_arccsc_one() {
    let result = simplify_str("arccsc(1)");
    assert!(
        result.contains("pi") && result.contains("2"),
        "arccsc(1) should be π/2, got: {}",
        result
    );
}

// ==================== Composition Tests ====================

#[test]
fn test_cot_arccot() {
    assert_eq!(
        simplify_str("cot(arccot(x))"),
        "x",
        "cot(arccot(x)) should simplify to x"
    );
}

#[test]
fn test_sec_arcsec() {
    assert_eq!(
        simplify_str("sec(arcsec(y))"),
        "y",
        "sec(arcsec(y)) should simplify to y"
    );
}

#[test]
fn test_csc_arccsc() {
    assert_eq!(
        simplify_str("csc(arccsc(z))"),
        "z",
        "csc(arccsc(z)) should simplify to z"
    );
}

#[test]
fn test_arccot_cot() {
    assert_eq!(
        simplify_str("arccot(cot(a))"),
        "a",
        "arccot(cot(a)) should simplify to a"
    );
}

#[test]
fn test_arcsec_sec() {
    assert_eq!(
        simplify_str("arcsec(sec(b))"),
        "b",
        "arcsec(sec(b)) should simplify to b"
    );
}

#[test]
fn test_arccsc_csc() {
    assert_eq!(
        simplify_str("arccsc(csc(c))"),
        "c",
        "arccsc(csc(c)) should simplify to c"
    );
}

// ==================== Negative Argument Tests ====================

#[test]
fn test_cot_negative() {
    assert_eq!(
        simplify_str("cot(-x)"),
        "-cot(x)",
        "cot(-x) should equal -cot(x)"
    );
}

#[test]
fn test_sec_negative() {
    assert_eq!(
        simplify_str("sec(-y)"),
        "sec(y)",
        "sec(-y) should equal sec(y)"
    );
}

#[test]
fn test_csc_negative() {
    assert_eq!(
        simplify_str("csc(-z)"),
        "-csc(z)",
        "csc(-z) should equal -csc(z)"
    );
}

#[test]
fn test_arccot_negative() {
    assert_eq!(
        simplify_str("arccot(-a)"),
        "-arccot(a)",
        "arccot(-a) should equal -arccot(a)"
    );
}

#[test]
fn test_arcsec_negative() {
    let result = simplify_str("arcsec(-b)");
    assert!(
        result.contains("pi") && result.contains("arcsec"),
        "arcsec(-b) should be π - arcsec(b), got: {}",
        result
    );
}

#[test]
fn test_arccsc_negative() {
    assert_eq!(
        simplify_str("arccsc(-c)"),
        "-arccsc(c)",
        "arccsc(-c) should equal -arccsc(c)"
    );
}

// ==================== Complex/Nested Tests ====================

#[test]
fn test_nested_reciprocal() {
    assert_eq!(
        simplify_str("cot(arccot(sec(arcsec(x))))"),
        "x",
        "Nested composition should simplify to x"
    );
}

#[test]
fn test_composition_with_negative() {
    assert_eq!(
        simplify_str("cot(arccot(-y))"),
        "-y",
        "cot(arccot(-y)) should simplify to -y"
    );
}

#[test]
fn test_double_composition() {
    assert_eq!(
        simplify_str("arcsec(sec(arcsec(sec(z))))"),
        "z",
        "Double composition should simplify to z"
    );
}

#[test]
fn test_composition_chain() {
    assert_eq!(
        simplify_str("arccot(cot(arccot(cot(a))))"),
        "a",
        "Chain of compositions should simplify to a"
    );
}

#[test]
fn test_mixed_with_basic_trig() {
    // This one might not fully simplify, just check it doesn't crash
    let result = simplify_str("cot(x) + tan(x)");
    assert!(
        result.contains("cot") || result.contains("tan"),
        "Mixed reciprocal and basic trig should retain function calls"
    );
}

#[test]
fn test_multiple_compositions() {
    assert_eq!(
        simplify_str("csc(arccsc(cot(arccot(u))))"),
        "u",
        "Multiple compositions should simplify to u"
    );
}

#[test]
#[ignore] // TODO: Composition with negative needs further investigation
fn test_nested_with_negatives() {
    assert_eq!(
        simplify_str("sec(arcsec(-x))"),
        "-x",
        "sec(arcsec(-x)) should simplify to -x"
    );
}
