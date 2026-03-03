use super::complex::{extract_gaussian, GaussianRational, ImaginaryPowerRule};
use crate::rule::Rule;
use cas_ast::{Constant, Context, Expr};
use cas_formatter::DisplayExpr;
use num_rational::BigRational;
use num_traits::{One, Zero};

fn complex_ctx() -> crate::parent_context::ParentContext {
    crate::parent_context::ParentContext::root()
        .with_value_domain(crate::semantics::ValueDomain::ComplexEnabled)
}

#[test]
fn test_i_squared() {
    let mut ctx = Context::new();
    let rule = ImaginaryPowerRule;
    let i = ctx.add(Expr::Constant(Constant::I));
    let two = ctx.num(2);
    let i_squared = ctx.add(Expr::Pow(i, two));

    let rewrite = rule.apply(&mut ctx, i_squared, &complex_ctx()).unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "-1"
    );
}

#[test]
fn test_i_cubed() {
    let mut ctx = Context::new();
    let rule = ImaginaryPowerRule;
    let i = ctx.add(Expr::Constant(Constant::I));
    let three = ctx.num(3);
    let i_cubed = ctx.add(Expr::Pow(i, three));

    let rewrite = rule.apply(&mut ctx, i_cubed, &complex_ctx()).unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "-i"
    );
}

#[test]
fn test_i_fourth() {
    let mut ctx = Context::new();
    let rule = ImaginaryPowerRule;
    let i = ctx.add(Expr::Constant(Constant::I));
    let four = ctx.num(4);
    let i_fourth = ctx.add(Expr::Pow(i, four));

    let rewrite = rule.apply(&mut ctx, i_fourth, &complex_ctx()).unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "1"
    );
}

#[test]
fn test_i_large_power() {
    let mut ctx = Context::new();
    let rule = ImaginaryPowerRule;
    let i = ctx.add(Expr::Constant(Constant::I));
    let seventeen = ctx.num(17);
    let i_17 = ctx.add(Expr::Pow(i, seventeen));

    let rewrite = rule.apply(&mut ctx, i_17, &complex_ctx()).unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "i"
    );
}

#[test]
fn test_extract_gaussian_number() {
    let mut ctx = Context::new();
    let three = ctx.num(3);
    let g = extract_gaussian(&ctx, three).unwrap();
    assert_eq!(g.real, BigRational::from_integer(3.into()));
    assert!(g.imag.is_zero());
}

#[test]
fn test_extract_gaussian_i() {
    let mut ctx = Context::new();
    let i = ctx.add(Expr::Constant(Constant::I));
    let g = extract_gaussian(&ctx, i).unwrap();
    assert!(g.real.is_zero());
    assert!(g.imag.is_one());
}

#[test]
fn test_gaussian_to_expr() {
    let mut ctx = Context::new();
    let g = GaussianRational::new(
        BigRational::from_integer(3.into()),
        BigRational::from_integer(2.into()),
    );
    let expr = g.to_expr(&mut ctx);
    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr
        }
    );
    assert_eq!(display, "3 + 2 * i");
}
