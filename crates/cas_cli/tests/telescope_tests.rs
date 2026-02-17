// Tests for Dirichlet kernel identity and telescoping strategy

use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;
use num_traits::Zero;

fn parse_and_simplify(input: &str) -> (String, bool) {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = cas_parser::parse(input, &mut simplifier.context).unwrap();
    let (result, _steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    let is_zero = match simplifier.context.get(result) {
        cas_ast::Expr::Number(n) => n.is_zero(),
        _ => false,
    };

    (result_str, is_zero)
}

#[test]
fn test_dirichlet_kernel_n1() {
    // D_1(x) = 1 + 2*cos(x) = sin(3x/2)/sin(x/2)
    let (result, is_zero) = parse_and_simplify("1 + 2*cos(x) - sin(3*x/2)/sin(x/2)");
    assert!(is_zero, "Expected 0, got: {}", result);
}

#[test]
fn test_dirichlet_kernel_n2() {
    // D_2(x) = 1 + 2*cos(x) + 2*cos(2x) = sin(5x/2)/sin(x/2)
    let (result, is_zero) = parse_and_simplify("1 + 2*cos(x) + 2*cos(2*x) - sin(5*x/2)/sin(x/2)");
    assert!(is_zero, "Expected 0, got: {}", result);
}

#[test]
fn test_dirichlet_kernel_n3() {
    // D_3(x) = 1 + 2*cos(x) + 2*cos(2x) + 2*cos(3x) = sin(7x/2)/sin(x/2)
    let (result, is_zero) =
        parse_and_simplify("1 + 2*cos(x) + 2*cos(2*x) + 2*cos(3*x) - sin(7*x/2)/sin(x/2)");
    assert!(is_zero, "Expected 0, got: {}", result);
}

#[test]
fn test_dirichlet_kernel_different_order() {
    // Same as n=2 but terms in different order
    let (result, is_zero) = parse_and_simplify("2*cos(2*x) + 1 - sin(5*x/2)/sin(x/2) + 2*cos(x)");
    assert!(is_zero, "Expected 0, got: {}", result);
}

#[test]
fn test_telescope_command_basic() {
    // Test the telescoping module directly
    let mut ctx = cas_ast::Context::new();
    let expr =
        cas_parser::parse("1 + 2*cos(x) + 2*cos(2*x) - sin(5*x/2)/sin(x/2)", &mut ctx).unwrap();

    let result = cas_engine::telescoping::telescope(&mut ctx, expr);
    assert!(result.success, "Telescoping proof should succeed");
}
