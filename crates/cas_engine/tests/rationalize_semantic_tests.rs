//! Semantic validation tests for rationalization.
//!
//! These tests verify that rationalized expressions are mathematically equivalent
//! to the originals by evaluating both with numeric substitutions.

use cas_ast::Context;
use cas_engine::Simplifier;

/// Parse an expression and evaluate it numerically with x=val.
fn parse_and_eval(input: &str, x_val: f64) -> f64 {
    let mut ctx = Context::new();
    let expr = cas_parser::parse(input, &mut ctx).expect("parse failed");
    eval_expr(&ctx, expr, x_val)
}

/// Evaluate an expression numerically with x=val.
fn eval_expr(ctx: &Context, id: cas_ast::ExprId, x_val: f64) -> f64 {
    use cas_ast::Expr;
    match ctx.get(id) {
        Expr::Number(n) => {
            let numer: i64 = n.numer().try_into().unwrap_or(0);
            let denom: i64 = n.denom().try_into().unwrap_or(1);
            numer as f64 / denom as f64
        }
        Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == "x" => x_val,
        Expr::Variable(_) => 1.0, // Other variables treated as 1.0
        Expr::Add(l, r) => eval_expr(ctx, *l, x_val) + eval_expr(ctx, *r, x_val),
        Expr::Sub(l, r) => eval_expr(ctx, *l, x_val) - eval_expr(ctx, *r, x_val),
        Expr::Mul(l, r) => eval_expr(ctx, *l, x_val) * eval_expr(ctx, *r, x_val),
        Expr::Div(l, r) => eval_expr(ctx, *l, x_val) / eval_expr(ctx, *r, x_val),
        Expr::Neg(inner) => -eval_expr(ctx, *inner, x_val),
        Expr::Pow(base, exp) => eval_expr(ctx, *base, x_val).powf(eval_expr(ctx, *exp, x_val)),
        Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
            eval_expr(ctx, args[0], x_val).sqrt()
        }
        _ => f64::NAN,
    }
}

/// Simplify and evaluate numerically with x=val.
fn simplify_and_eval(input: &str, x_val: f64) -> f64 {
    let mut simplifier = Simplifier::new();
    let expr = cas_parser::parse(input, &mut simplifier.context).expect("parse failed");
    let (simplified, _) = simplifier.simplify(expr);
    eval_expr(&simplifier.context, simplified, x_val)
}

/// Verify that simplification is semantically correct (result ≈ original).
fn assert_semantic_equivalence(input: &str, x_val: f64) {
    let original = parse_and_eval(input, x_val);
    let simplified = simplify_and_eval(input, x_val);

    let diff = (original - simplified).abs();
    let tolerance = 1e-10;

    assert!(
        diff < tolerance || (original.is_nan() && simplified.is_nan()),
        "Semantic mismatch for '{}' at x={}: original={}, simplified={}, diff={}",
        input,
        x_val,
        original,
        simplified,
        diff
    );
}

// ===== Level 1.5 Rationalization Tests =====

#[test]
fn test_semantic_level_1_5_basic() {
    // x/(2*(1+√2)) should simplify and remain equivalent
    assert_semantic_equivalence("x/(2*(1+sqrt(2)))", 7.0);
    assert_semantic_equivalence("x/(2*(1+sqrt(2)))", 1.0);
    assert_semantic_equivalence("x/(2*(1+sqrt(2)))", 13.5);
}

#[test]
fn test_semantic_level_1_5_with_subtraction() {
    // x/(2*(3-2√5)) with sign handling
    assert_semantic_equivalence("x/(2*(3-2*sqrt(5)))", 7.0);
    assert_semantic_equivalence("x/(2*(3-2*sqrt(5)))", 1.0);
}

#[test]
fn test_semantic_same_surd_squared() {
    // 1/((1+√2)²) → 3-2√2
    assert_semantic_equivalence("1/((1+sqrt(2))*(1+sqrt(2)))", 1.0);
}

#[test]
fn test_semantic_binomial_basic() {
    // Basic Level 1 rationalization
    assert_semantic_equivalence("1/(1+sqrt(2))", 1.0);
    assert_semantic_equivalence("1/(3-2*sqrt(5))", 1.0);
    assert_semantic_equivalence("x/(1+sqrt(3))", 5.0);
}

#[test]
fn test_semantic_single_surd() {
    // Level 0: 1/√n
    assert_semantic_equivalence("1/sqrt(2)", 1.0);
    assert_semantic_equivalence("x/sqrt(5)", 7.0);
}

#[test]
fn test_semantic_numeric_factor() {
    // 1/(3*(1+√2))
    assert_semantic_equivalence("1/(3*(1+sqrt(2)))", 1.0);
}

#[test]
fn test_multi_surd_unchanged() {
    // Multi-surd should NOT be auto-rationalized, but still be correct
    assert_semantic_equivalence("1/((1+sqrt(2))*(1+sqrt(3)))", 1.0);
}
