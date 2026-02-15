/// Hardening tests for the rational canonicalization, nested Pow domain
/// safety, and cancel-common-terms rules.
///
/// These tests create a safety perimeter around the consolidation rules
/// so that future refactors can't silently introduce incorrect rewrites
/// like `(x^2)^(1/2) → x` (must stay as sqrt(x^2) or |x|).
use cas_ast::{DisplayExpr, Expr};
use cas_engine::Simplifier;

fn simplify_display(input: &str) -> String {
    let mut simplifier = Simplifier::new();
    simplifier.register_default_rules();
    let expr = cas_parser::parse(input, &mut simplifier.context).expect("parse failed");
    let (result, _steps) = simplifier.simplify(expr);
    DisplayExpr {
        context: &simplifier.context,
        id: result,
    }
    .to_string()
}

// ═══════════════════════════════════════════════════════════════════════
// 1A) Rational canonicality — different source paths, same result
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn rational_canon_pow_div_vs_rational() {
    // Parser produces Div(5,6) for `x^(5/6)` exponent
    // After canonicalization, both must produce identical display
    let a = simplify_display("x^(5/6)");
    let b = {
        let mut simplifier = Simplifier::new();
        simplifier.register_default_rules();
        let x = simplifier.context.var("x");
        let five_sixths = simplifier.context.rational(5, 6);
        let expr = simplifier.context.add(Expr::Pow(x, five_sixths));
        let (result, _) = simplifier.simplify(expr);
        DisplayExpr {
            context: &simplifier.context,
            id: result,
        }
        .to_string()
    };
    assert_eq!(
        a, b,
        "Different construction paths must produce identical display"
    );
}

#[test]
fn rational_canon_negative_denominator() {
    // 10 / (-12) should normalize — no raw 10 or 12 in result
    let result = simplify_display("10/(-12)");
    assert!(
        !result.contains("10") && !result.contains("12"),
        "Should be fully reduced, got: {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 1B) Nested Pow domain safety
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn nested_pow_safe_odd_k_even_q() {
    // (x^3)^(1/2) → x^(3/2) — safe because k=3 is odd
    let result = simplify_display("(x^3)^(1/2)");
    assert!(
        result.contains("3/2") || result.contains("x^(3/2)"),
        "Should fold to x^(3/2), got: {}",
        result
    );
}

#[test]
fn nested_pow_blocked_even_k_even_q() {
    // (x^2)^(1/2) MUST NOT fold to x — would lose |x| semantics
    let result = simplify_display("(x^2)^(1/2)");
    assert!(
        result != "x",
        "DOMAIN BUG: (x^2)^(1/2) must NOT simplify to plain x, got: {}",
        result
    );
}

#[test]
fn nested_pow_safe_integer_exponent() {
    // (sin(x)^2)^3 → sin(x)^6 — safe because outer exponent is integer
    let result = simplify_display("(sin(x)^2)^3");
    assert!(
        result.contains("^6") || result.contains("sin(x)^6"),
        "Should fold to sin(x)^6, got: {}",
        result
    );
}

#[test]
fn nested_pow_safe_odd_q() {
    // (x^2)^(1/3) → x^(2/3) — safe because q=3 is odd
    let result = simplify_display("(x^2)^(1/3)");
    assert!(
        result.contains("2/3") || result.contains("x^(2/3)"),
        "Should fold to x^(2/3), got: {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 2) CancelCommonAdditiveTerms coverage
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn cancel_symmetric_sub_add_add() {
    // (a + b + c) - (b + c) → a
    let result = simplify_display("(a + b + c) - (b + c)");
    assert_eq!(result, "a", "Should cancel b and c, got: {}", result);
}

#[test]
fn cancel_rhs_superset() {
    // b - (a + b) → -a
    let result = simplify_display("b - (a + b)");
    assert!(
        result == "-a" || result == "-(a)" || result.contains("-a") || result.contains("-1*a"),
        "Should reduce to -a, got: {}",
        result
    );
}

#[test]
fn cancel_identity_noise_radical() {
    // The critical benchmark case: (x^2 + x^(5/6) + 1) - x^(5/6) → x^2 + 1
    let result = simplify_display("(x^2 + x^(5/6) + 1) - x^(5/6)");
    assert!(
        !result.contains("5/6"),
        "x^(5/6) terms should cancel, got: {}",
        result
    );
}
