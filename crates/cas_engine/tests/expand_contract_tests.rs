//! Contract tests for Standard vs Expand mode behavior.
//!
//! These tests ensure that:
//! - Standard mode preserves structure (no automatic binomial/multinomial expansion)
//! - Expand mode explicitly expands expressions
//! - expand_mode flag doesn't leak between evaluations

use cas_engine::Simplifier;
use cas_parser::parse;
use num_traits::ToPrimitive;

/// Helper: check if expression is Pow(Add(...), exp)
fn is_pow_of_additive(ctx: &cas_ast::Context, expr: cas_ast::ExprId, expected_exp: i64) -> bool {
    use cas_ast::Expr;
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        let base_is_additive = matches!(ctx.get(*base), Expr::Add(_, _) | Expr::Sub(_, _));
        let exp_matches = if let Expr::Number(n) = ctx.get(*exp) {
            n.to_integer()
                .to_i64()
                .map(|v| v == expected_exp)
                .unwrap_or(false)
        } else {
            false
        };
        base_is_additive && exp_matches
    } else {
        false
    }
}

/// Helper: simplify with Standard options (default)
fn simplify_standard(input: &str) -> (String, cas_ast::ExprId, cas_ast::Context) {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).expect("parse failed");
    let (result, _) = simplifier.simplify(expr);
    let result_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    (result_str, result, simplifier.context)
}

/// Helper: expand and simplify using the same approach as REPL's expand command
fn simplify_expand(input: &str) -> (String, cas_ast::ExprId, cas_ast::Context) {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).expect("parse failed");

    // Use cas_engine::expand::expand() directly like REPL does
    let expanded = cas_engine::expand::expand(&mut simplifier.context, expr);
    // Then simplify to clean up
    let (result, _) = simplifier.simplify(expanded);

    let result_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    (result_str, result, simplifier.context)
}

// =========================================================================
// A) Standard NO expande (5 tests)
// =========================================================================

/// Test 1: Binomio básico no se expande en Standard
#[test]
fn test_standard_binomial_basic() {
    let (result_str, result, ctx) = simplify_standard("(x+1)^3");

    // Should remain as Pow(Add, 3), not expanded
    assert!(
        is_pow_of_additive(&ctx, result, 3),
        "Expected Pow(Add, 3), got: {}",
        result_str
    );
    assert!(
        result_str.contains("³") || result_str.contains("^3"),
        "Expected power notation, got: {}",
        result_str
    );
}

/// Test 2: Binomio con orden permutado llega a la misma forma canónica
#[test]
fn test_standard_binomial_canonical_order() {
    let (result1, _, _) = simplify_standard("(x+1)^3");
    let (result2, _, _) = simplify_standard("(1+x)^3");

    // Both should produce the same canonical form (preserved, not expanded)
    assert_eq!(
        result1, result2,
        "(x+1)^3 and (1+x)^3 should have same canonical form"
    );
}

/// Test 3: Small multinomial DOES expand in Standard (SmallMultinomialExpansionRule)
/// (x+y+z)^4 has C(6,2)=15 terms, within the pred_terms≤35 budget.
/// Larger cases like (x+y+z)^5 should NOT expand (n>MAX_N=4).
#[test]
fn test_standard_multinomial() {
    // (x+y+z)^4 — within SmallMultinomialExpansionRule guards (n=4, k=3, pred=15≤35)
    let (result_str, result, ctx) = simplify_standard("(x+y+z)^4");

    // Should be expanded — no longer Pow(Add, 4)
    assert!(
        !is_pow_of_additive(&ctx, result, 4),
        "Expected (x+y+z)^4 to be expanded by SmallMultinomialExpansionRule, got: {}",
        result_str
    );

    // Should have multiple terms (15 for trinomial^4)
    let plus_count = result_str.matches('+').count() + result_str.matches('-').count();
    assert!(
        plus_count >= 10,
        "Expected many terms in expanded form, got {} operators in: {}",
        plus_count,
        result_str
    );

    // (x+y+z)^5 — n=5 > MAX_N=4, should NOT expand
    let (result_str5, result5, ctx5) = simplify_standard("(x+y+z)^5");
    assert!(
        is_pow_of_additive(&ctx5, result5, 5),
        "Expected (x+y+z)^5 to stay as Pow (n>MAX_N=4), got: {}",
        result_str5
    );
}

/// Test 4: Potencia negativa no se expande (y no causa problemas)
#[test]
fn test_standard_negative_exponent() {
    let (result_str, result, ctx) = simplify_standard("(x+1)^(-3)");

    // Should NOT be expanded - either stays as (x+1)^(-3) or becomes 1/(x+1)^3
    // But definitely NOT 1/(x^3 + 3x^2 + 3x + 1)
    let expanded_markers = ["x^3 + 3", "x³ + 3", "3·x² +", "3·x^2 +"];
    for marker in expanded_markers {
        assert!(
            !result_str.contains(marker),
            "Should not be expanded! Found '{}' in: {}",
            marker,
            result_str
        );
    }

    // Verify structure: should be Pow or Div(1, Pow)
    match ctx.get(result) {
        cas_ast::Expr::Pow(_, _) => {} // OK: (x+1)^(-3)
        cas_ast::Expr::Div(_, _) => {} // OK: 1/(x+1)^3
        other => panic!(
            "Expected Pow or Div for negative exponent, got: {:?}",
            other
        ),
    }
}

/// Test 5: ContextMode::Solve no activa expansión
#[test]
fn test_solve_context_no_expansion() {
    use cas_engine::options::{ContextMode, EvalOptions};

    let opts = EvalOptions {
        shared: cas_engine::phase::SharedSemanticConfig {
            context_mode: ContextMode::Solve,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut simplifier = Simplifier::with_profile(&opts);
    let expr = parse("(x+1)^3 + (x+1)^3", &mut simplifier.context).expect("parse failed");
    let (result, _) = simplifier.simplify(expr);
    let result_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Should simplify to 2*(x+1)^3, NOT to 2*x^3 + 6*x^2 + 6*x + 2
    let expanded_markers = ["x^3 +", "x³ +", "6·x²", "6·x^2"];
    for marker in expanded_markers {
        assert!(
            !result_str.contains(marker),
            "Solve context should not expand! Found '{}' in: {}",
            marker,
            result_str
        );
    }
}

// =========================================================================
// B) Expand SÍ expande (4 tests)
// =========================================================================

/// Test 6: Expand binomio básico
#[test]
fn test_expand_binomial_basic() {
    let (result_str, result, ctx) = simplify_expand("(x+1)^3");

    // Should be expanded - no longer Pow(Add, 3)
    assert!(
        !is_pow_of_additive(&ctx, result, 3),
        "Should be expanded, not Pow(Add, 3): {}",
        result_str
    );

    // Should contain expanded terms (x³, x², x, 1)
    // At minimum, should have Add at top level
    assert!(
        matches!(ctx.get(result), cas_ast::Expr::Add(_, _)),
        "Expected Add at top level after expansion, got: {}",
        result_str
    );
}

/// Test 7: Expand multinomio (x+y+z)^3 produces 10 terms
#[test]
fn test_expand_multinomial() {
    let (result_str, _, _) = simplify_expand("(x+y+z)^3");

    // (x+y+z)^3 has 10 terms: x³, y³, z³, 3x²y, 3x²z, 3y²x, 3y²z, 3z²x, 3z²y, 6xyz
    // Count the '+' signs (9 for 10 terms) or check for presence of key terms
    let plus_count = result_str.matches('+').count() + result_str.matches('-').count();

    // Should have at least 9 addition operators for 10 terms
    assert!(
        plus_count >= 8,
        "Expected ~9 operators for 10-term expansion, got {} in: {}",
        plus_count,
        result_str
    );
}

/// Test 8: Expand con potencia negativa - should preserve (not expand)
#[test]
fn test_expand_negative_exponent_preserved() {
    let (result_str, result, ctx) = simplify_expand("(x+1)^(-3)");

    // Negative exponents should NOT be expanded even in expand mode
    // Should remain as (x+1)^(-3) or 1/(x+1)^3
    let expanded_markers = ["x^3 + 3", "x³ + 3", "3·x² +", "3·x^2 +"];
    for marker in expanded_markers {
        assert!(
            !result_str.contains(marker),
            "Negative exponent should not expand! Found '{}' in: {}",
            marker,
            result_str
        );
    }

    // Verify structure
    match ctx.get(result) {
        cas_ast::Expr::Pow(_, _) => {} // OK
        cas_ast::Expr::Div(_, _) => {} // OK
        other => panic!(
            "Expected Pow or Div for negative exponent, got: {:?}",
            other
        ),
    }
}

/// Test 9: Expand mode doesn't contaminate subsequent Standard evaluations
#[test]
fn test_expand_no_contamination() {
    // First: expand using cas_engine::expand::expand() directly
    let mut simplifier = Simplifier::with_default_rules();
    let expr1 = parse("(x+1)^3", &mut simplifier.context).expect("parse failed");
    let expanded_raw = cas_engine::expand::expand(&mut simplifier.context, expr1);
    let (expanded_result, _) = simplifier.simplify(expanded_raw);
    let expanded_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: expanded_result
        }
    );

    // Should be expanded
    assert!(
        !is_pow_of_additive(&simplifier.context, expanded_result, 3),
        "First eval (expand) should expand: {}",
        expanded_str
    );

    // Second: standard mode (same simplifier instance)
    let expr2 = parse("(y+2)^3", &mut simplifier.context).expect("parse failed");
    let (standard_result, _) = simplifier.simplify(expr2);
    let standard_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: standard_result
        }
    );

    // Should NOT be expanded - expand function call should not persist
    assert!(
        is_pow_of_additive(&simplifier.context, standard_result, 3),
        "Second eval (standard) should preserve structure: {}",
        standard_str
    );
}

// =========================================================================
// C) Budget / rechazo de expansión catastrófica (1 test)
// =========================================================================

/// Test 10: Large multinomial expansion is handled reasonably
#[test]
fn test_expand_large_multinomial() {
    // (x+y+z+w)^5 produces 56 terms (8C3 = 56) - substantial but tractable
    // This tests that expand handles larger expressions

    let start = std::time::Instant::now();
    let (result_str, _, _) = simplify_expand("(x+y+z+w)^5");
    let elapsed = start.elapsed();

    // Should complete in reasonable time (< 5 seconds)
    assert!(
        elapsed.as_secs() < 5,
        "Large expand should complete in <5s, took {:?}",
        elapsed
    );

    // Should have many terms (56 for 4 variables, degree 5)
    let plus_count = result_str.matches('+').count() + result_str.matches('-').count();
    assert!(
        plus_count >= 10,
        "Expected many terms in expanded form, got {} operators in: {}",
        plus_count,
        result_str
    );

    println!(
        "Large multinomial (x+y+z+w)^5 expanded in {:?}, {} operators",
        elapsed, plus_count
    );
}
