// Tests for product() function (finite products/productorios)

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
fn test_product_factorial() {
    // product(k, k, 1, 5) = 5! = 120
    let result = parse_and_simplify("product(k, k, 1, 5)");
    assert_eq!(result, "120");
}

#[test]
fn test_product_factorial_4() {
    // product(k, k, 1, 4) = 4! = 24
    let result = parse_and_simplify("product(k, k, 1, 4)");
    assert_eq!(result, "24");
}

#[test]
fn test_product_telescoping_symbolic() {
    // product((k+1)/k, k, 1, n) = n+1 via telescoping
    let result = parse_and_simplify("product((k+1)/k, k, 1, n)");
    // Result should be 1+n or n+1
    assert!(
        result.contains("1 + n") || result.contains("n + 1"),
        "Expected telescoping result n+1, got: {}",
        result
    );
}

#[test]
fn test_product_telescoping_numeric() {
    // product((k+1)/k, k, 1, 10) = 11 via telescoping
    let result = parse_and_simplify("product((k+1)/k, k, 1, 10)");
    assert_eq!(result, "11");
}

#[test]
fn test_product_constant() {
    // product(2, k, 1, 4) = 2^4 = 16
    let result = parse_and_simplify("product(2, k, 1, 4)");
    assert_eq!(result, "16");
}

#[test]
fn test_product_squares() {
    // product(k^2, k, 1, 3) = 1*4*9 = 36
    let result = parse_and_simplify("product(k^2, k, 1, 3)");
    assert_eq!(result, "36");
}
