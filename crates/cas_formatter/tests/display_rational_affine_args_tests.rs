use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

fn render(input: &str) -> String {
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("parse expression");
    format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr
        }
    )
}

#[test]
fn elementary_function_args_prefer_quotient_for_unit_fraction_scaled_sum() {
    assert_eq!(render("sin(1/2*(3*x+2))"), "sin((3 * x + 2) / 2)");
    assert_eq!(render("cos(1/2*(2-3*x))"), "cos((2 - 3 * x) / 2)");
    assert_eq!(render("sec(1/2*(3*x+2))"), "sec((3 * x + 2) / 2)");
    assert_eq!(render("csc(1/2*(2-3*x))"), "csc((2 - 3 * x) / 2)");
    assert_eq!(render("cot(1/2*(2-3*x))"), "cot((2 - 3 * x) / 2)");
    assert_eq!(render("sinh(1/2*(3*x+2))"), "sinh((3 * x + 2) / 2)");
    assert_eq!(render("cosh(1/2*(2-3*x))"), "cosh((2 - 3 * x) / 2)");
}

#[test]
fn power_exponent_prefers_quotient_for_unit_fraction_scaled_sum() {
    assert_eq!(render("e^(1/2*(3*x+2))"), "e^((3 * x + 2) / 2)");
}

#[test]
fn inverse_function_args_share_the_same_quotient_presentation() {
    assert_eq!(render("arcsin(1/2*(x+1))"), "arcsin((x + 1) / 2)");
}

#[test]
fn inverse_function_args_prefer_quotient_for_unit_fraction_scaled_terms() {
    assert_eq!(render("arcsin(1/2*x^2)"), "arcsin(x^2 / 2)");
    assert_eq!(render("acosh(1/2*x^2)"), "acosh(x^2 / 2)");
    assert_eq!(render("arctan(1/2*sqrt(x))"), "arctan(sqrt(x) / 2)");
}

#[test]
fn numeric_half_powers_prefer_sqrt_notation() {
    assert_eq!(render("5^(1/2)"), "sqrt(5)");
    assert_eq!(render("(x^2+1)^(1/2)"), "(x^2 + 1)^(1 / 2)");
}

#[test]
fn function_args_prefer_sqrt_for_half_power_arguments() {
    assert_eq!(render("arctan((4*x+1)^(1/2))"), "arctan(sqrt(4 * x + 1))");
    assert_eq!(render("sin((x^2+1)^(1/2))"), "sin(sqrt(x^2 + 1))");
    assert_eq!(render("(x^2+1)^(1/2)"), "(x^2 + 1)^(1 / 2)");
}

#[test]
fn trig_products_prefer_degree_order_for_larger_polynomial_cofactors() {
    assert_eq!(
        render("(3*x^2 - x^4 - 4)*cos(2*x+1)"),
        "(-x^4 + 3 * x^2 - 4) * cos(2 * x + 1)"
    );
    assert_eq!(
        render("sin(1-2*x)*(3*x^5 + 45*x - 15*x^3)"),
        "(3 * x^5 - 15 * x^3 + 45 * x) * sin(1 - 2 * x)"
    );
}

#[test]
fn trig_products_keep_two_term_orientation_for_common_domain_shapes() {
    assert_eq!(render("(1-x^2)*sin(x)"), "sin(x) * (1 - x^2)");
    assert_eq!(render("(2-x^2)*cos(x)"), "cos(x) * (2 - x^2)");
}
