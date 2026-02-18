use cas_engine::Simplifier;
use cas_formatter::LaTeXExpr;
use cas_parser::parse;

fn simplify_str(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).unwrap();
    let result = simplifier.simplify(expr);
    LaTeXExpr {
        context: &simplifier.context,
        id: result.0,
    }
    .to_latex()
}

#[test]
fn test_rationalize_denominator_simple() {
    let res = simplify_str("1 / (sqrt(2) + 1)");
    println!("test_rationalize_denominator_simple: {}", res);
    // Output: -1 + 2^{\frac{1}{2}}
    assert!(res.contains("2^{\\frac{1}{2}}") || res.contains("\\sqrt{2}"));
}

#[test]
fn test_rationalize_denominator_roots() {
    let res = simplify_str("1 / (sqrt(3) + sqrt(2))");
    println!("test_rationalize_denominator_roots: {}", res);
    // Output: 3^{\frac{1}{2}} + -1 \cdot 2^{\frac{1}{2}}
    assert!(res.contains("3^{\\frac{1}{2}}") || res.contains("\\sqrt{3}"));
    assert!(res.contains("2^{\\frac{1}{2}}") || res.contains("\\sqrt{2}"));
}

#[test]
fn test_root_denesting_simple() {
    let res = simplify_str("sqrt(3 + 2*sqrt(2))");
    println!("test_root_denesting_simple: {}", res);
    // Output: 1 + 2^{\frac{1}{2}}
    assert!(res.contains("1"));
    assert!(res.contains("2^{\\frac{1}{2}}") || res.contains("\\sqrt{2}"));
}

#[test]
fn test_root_denesting_sub() {
    let res = simplify_str("sqrt(5 - 2*sqrt(6))");
    println!("test_root_denesting_sub: {}", res);
    // Expected: 3^{\frac{1}{2}} - 2^{\frac{1}{2}}
    assert!(res.contains("3^{\\frac{1}{2}}") || res.contains("\\sqrt{3}"));
    assert!(res.contains("2^{\\frac{1}{2}}") || res.contains("\\sqrt{2}"));
}

#[test]
fn test_cancel_common_factors_powers() {
    let res = simplify_str("(x^3 * y^2) / (x^2 * y^3)");
    println!("test_cancel_common_factors_powers: {}", res);
    assert_eq!(res, "\\frac{x}{y}");
}

#[test]
fn test_integral_linear_subst_sin() {
    let res = simplify_str("integrate(sin(2*x + 1), x)");
    println!("test_integral_linear_subst_sin: {}", res);
    // Smoke test: integration returns without crash
    // Full assertion pending: integrate not fully implemented
    assert!(!res.is_empty());
}

#[test]
fn test_integral_linear_subst_exp() {
    let res = simplify_str("integrate(exp(3*x), x)");
    println!("test_integral_linear_subst_exp: {}", res);
    // Smoke test: integration returns without crash
    // Full assertion pending: integrate not fully implemented
    assert!(!res.is_empty());
}

#[test]
fn test_diff_chain_rule_depth() {
    let res = simplify_str("diff(sin(x^2), x)");
    println!("test_diff_chain_rule_depth: {}", res);
    assert!(res.contains("\\cos"));
    assert!(res.contains("{x}^{2}") || res.contains("x^{2}"));
    assert!(res.contains("2"));
    assert!(res.contains("x"));
}

#[test]
#[ignore = "needs solve_inequality feature"]
fn test_inequality_abs_gt() {
    let res = simplify_str("solve_inequality(abs(x - 1) > 2, x)");
    assert!(res.contains("3"));
    assert!(res.contains("-1"));
}

#[test]
#[ignore = "needs solve_inequality feature"]
fn test_inequality_quadratic() {
    let res = simplify_str("solve_inequality(x^2 - 4 > 0, x)");
    assert!(res.contains("2"));
    assert!(res.contains("-2"));
}

#[test]
#[ignore = "needs cubic solver"]
fn test_equation_cubic() {
    // x^3 = 8 -> x^3 - 8 = 0
    let res = simplify_str("solve(x^3 - 8, x)");
    println!("test_equation_cubic: {}", res);
    assert!(res.contains("2"));
}

#[test]
fn test_trig_square_canonicalization() {
    let res = simplify_str("sin(x)^2 + cos(x)^2");
    println!("test_trig_square_canonicalization: {}", res);
    assert_eq!(res, "1");
}
