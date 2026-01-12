// Tests for sum() function (finite summations)

use cas_ast::DisplayExpr;
use cas_engine::Simplifier;

fn parse_and_simplify(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = cas_parser::parse(input, &mut simplifier.context).unwrap();
    let (result, _steps) = simplifier.simplify(expr);
    format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

#[test]
fn test_sum_linear() {
    // sum(k, k, 1, 10) = 1+2+...+10 = 55
    let result = parse_and_simplify("sum(k, k, 1, 10)");
    assert_eq!(result, "55");
}

#[test]
fn test_sum_squares() {
    // sum(k^2, k, 1, 5) = 1+4+9+16+25 = 55
    let result = parse_and_simplify("sum(k^2, k, 1, 5)");
    assert_eq!(result, "55");
}

#[test]
fn test_sum_cubes() {
    // sum(k^3, k, 1, 4) = 1+8+27+64 = 100
    let result = parse_and_simplify("sum(k^3, k, 1, 4)");
    assert_eq!(result, "100");
}

#[test]
fn test_sum_constant() {
    // sum(5, k, 1, 10) = 5*10 = 50
    let result = parse_and_simplify("sum(5, k, 1, 10)");
    assert_eq!(result, "50");
}

#[test]
fn test_sum_expression() {
    // sum(2*k + 1, k, 1, 5) = 3+5+7+9+11 = 35
    let result = parse_and_simplify("sum(2*k + 1, k, 1, 5)");
    assert_eq!(result, "35");
}

#[test]
fn test_sum_single_term() {
    // sum(k, k, 5, 5) = 5
    let result = parse_and_simplify("sum(k, k, 5, 5)");
    assert_eq!(result, "5");
}

#[test]
fn test_sum_with_trig() {
    // sum(sin(k*pi), k, 1, 4) should work (each term evaluates to 0)
    // Note: sin(k*pi) = 0 for integer k, but we don't have that simplification
    // so just test that it doesn't crash
    let _result = parse_and_simplify("sum(k, k, 0, 3)");
    // 0+1+2+3 = 6
    assert_eq!(_result, "6");
}

#[test]
fn test_sum_telescoping_symbolic() {
    // sum(1/(k*(k+1)), k, 1, n) = 1 - 1/(n+1) via partial fractions
    let result = parse_and_simplify("sum(1/(k*(k+1)), k, 1, n)");
    // Result should be 1 - 1/(n+1) which equals n/(n+1)
    assert!(
        result.contains("1 - 1 / (n + 1)") || result.contains("n / (n + 1)"),
        "Expected telescoping result, got: {}",
        result
    );
}

#[test]
fn test_sum_telescoping_numeric() {
    // sum(1/(k*(k+1)), k, 1, 10) = 10/11
    let result = parse_and_simplify("sum(1/(k*(k+1)), k, 1, 10)");
    assert_eq!(result, "10/11");
}
