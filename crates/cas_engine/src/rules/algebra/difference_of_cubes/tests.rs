use super::*;
use crate::rule::Rule;
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use num_bigint::BigInt;
use num_rational::BigRational;

#[test]
fn test_cancel_cube_root_difference_basic() {
    let mut ctx = Context::new();

    // Build: (x - 27) / (x^(2/3) + 3*x^(1/3) + 9)
    let x = ctx.var("x");
    let c27 = ctx.num(27);
    let c3 = ctx.num(3);
    let c9 = ctx.num(9);

    // Numerator: x - 27 as Add(x, Neg(27))
    let neg_27 = ctx.add(Expr::Neg(c27));
    let num = ctx.add(Expr::Add(x, neg_27));

    // Exponents
    let one_third = ctx.add(Expr::Number(BigRational::new(
        BigInt::from(1),
        BigInt::from(3),
    )));
    let two_thirds = ctx.add(Expr::Number(BigRational::new(
        BigInt::from(2),
        BigInt::from(3),
    )));

    // Denominator terms
    let x_2_3 = ctx.add(Expr::Pow(x, two_thirds));
    let x_1_3 = ctx.add(Expr::Pow(x, one_third));
    let term_mid = ctx.add(Expr::Mul(c3, x_1_3));

    // Den: x^(2/3) + 3*x^(1/3) + 9
    let den_partial = ctx.add(Expr::Add(x_2_3, term_mid));
    let den = ctx.add(Expr::Add(den_partial, c9));

    // Full expression
    let expr = ctx.add(Expr::Div(num, den));

    let rule = CancelCubeRootDifferenceRule;
    let result = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(result.is_some(), "Rule should match this pattern");

    let rewrite = result.unwrap();
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    );

    // Should be x^(1/3) - 3 or equivalent
    println!("Result: {}", result_str);
    assert!(
        result_str.contains("x") && result_str.contains("1/3"),
        "Result should contain cube root of x: got {}",
        result_str
    );
}
