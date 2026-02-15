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
// 3) Architecture: "Sub is NOT stable" — CI enforcement
// ═══════════════════════════════════════════════════════════════════════

/// After canonicalization, `Sub` nodes must be eliminated.
/// `CanonicalizeNegationRule` converts `Sub(a,b) → Add(a, Neg(b))`.
/// If this test fails, someone moved/disabled the rule — review immediately.
#[test]
fn sub_is_not_stable_after_canonicalization() {
    let mut simplifier = Simplifier::new();
    simplifier.register_default_rules();

    // x - y must become Add(x, Neg(y)), no Sub nodes survive
    let expr = cas_parser::parse("x - y", &mut simplifier.context).expect("parse failed");
    let (result, _steps) = simplifier.simplify(expr);

    let sub_count = cas_ast::traversal::count_nodes_matching(&simplifier.context, result, |expr| {
        matches!(expr, Expr::Sub(_, _))
    });
    assert_eq!(
        sub_count, 0,
        "ARCHITECTURE VIOLATION: Sub nodes survived canonicalization. \
         CanonicalizeNegationRule must convert Sub→Add(Neg) before other rules fire. \
         Found {} Sub nodes in simplified form of 'x - y'.",
        sub_count
    );
}

#[test]
fn sub_is_not_stable_complex_expression() {
    let mut simplifier = Simplifier::new();
    simplifier.register_default_rules();

    // More complex case: nested subtractions
    let expr =
        cas_parser::parse("(a - b) + (c - d) - e", &mut simplifier.context).expect("parse failed");
    let (result, _steps) = simplifier.simplify(expr);

    let sub_count = cas_ast::traversal::count_nodes_matching(&simplifier.context, result, |expr| {
        matches!(expr, Expr::Sub(_, _))
    });
    assert_eq!(
        sub_count, 0,
        "ARCHITECTURE VIOLATION: Sub nodes survived in complex expression. \
         Found {} Sub nodes in simplified form of '(a - b) + (c - d) - e'.",
        sub_count
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 4) cancel_common_additive_terms property tests
// ═══════════════════════════════════════════════════════════════════════

/// Idempotency: applying cancel twice produces no further cancellations.
#[test]
fn cancel_idempotent() {
    use cas_ast::Context;

    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");
    let c = ctx.var("c");
    let ab = ctx.add(Expr::Add(a, b));
    let lhs = ctx.add(Expr::Add(ab, c));
    let rhs = ctx.add(Expr::Add(b, c));

    // First cancellation: (a+b+c) vs (b+c) → a vs 0, cancelled=2
    let cr1 = cas_engine::cancel_common_additive_terms(&mut ctx, lhs, rhs)
        .expect("first cancel should fire");
    assert_eq!(cr1.cancelled_count, 2);

    // Second cancellation: a vs 0 → nothing to cancel
    let cr2 = cas_engine::cancel_common_additive_terms(&mut ctx, cr1.new_lhs, cr1.new_rhs);
    assert!(
        cr2.is_none(),
        "Idempotency violation: second cancel should be None, but cancelled {} terms",
        cr2.map(|r| r.cancelled_count).unwrap_or(0)
    );
}

/// Symmetry: swapping LHS↔RHS produces the same cancelled_count.
#[test]
fn cancel_symmetric_count() {
    use cas_ast::Context;

    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");
    let c = ctx.var("c");
    let ab = ctx.add(Expr::Add(a, b));
    let lhs = ctx.add(Expr::Add(ab, c));
    let rhs = ctx.add(Expr::Add(b, c));

    // Forward: (a+b+c) vs (b+c)
    let cr_fwd = cas_engine::cancel_common_additive_terms(&mut ctx, lhs, rhs)
        .expect("forward cancel should fire");

    // Reverse: (b+c) vs (a+b+c)
    // Need fresh expressions (ids are consumed differently), rebuild
    let mut ctx2 = Context::new();
    let a2 = ctx2.var("a");
    let b2 = ctx2.var("b");
    let c2 = ctx2.var("c");
    let ab2 = ctx2.add(Expr::Add(a2, b2));
    let lhs2 = ctx2.add(Expr::Add(ab2, c2));
    let b2r = ctx2.var("b");
    let c2r = ctx2.var("c");
    let rhs2 = ctx2.add(Expr::Add(b2r, c2r));

    let cr_rev = cas_engine::cancel_common_additive_terms(&mut ctx2, rhs2, lhs2)
        .expect("reverse cancel should fire");

    assert_eq!(
        cr_fwd.cancelled_count, cr_rev.cancelled_count,
        "Symmetry violation: forward cancelled {} but reverse cancelled {}",
        cr_fwd.cancelled_count, cr_rev.cancelled_count
    );
}
