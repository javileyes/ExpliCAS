//! Complex-number rewrite rules.
//!
//! Rule matching remains in engine, while rewrite math lives in `cas_math`.

use crate::define_rule;
use crate::rule::Rewrite;
pub use cas_math::complex_support::{extract_gaussian, GaussianRational};
use cas_math::complex_support::{
    try_rewrite_gaussian_add_expr, try_rewrite_gaussian_div_expr, try_rewrite_gaussian_mul_expr,
    try_rewrite_i_squared_mul_expr, try_rewrite_imaginary_power_expr,
    try_rewrite_sqrt_negative_expr,
};

define_rule!(
    ImaginaryPowerRule,
    "Imaginary Power",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_imaginary_power_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(ISquaredMulRule, "i * i = -1", |ctx, expr, parent_ctx| {
    if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
        return None;
    }

    let rewritten = try_rewrite_i_squared_mul_expr(ctx, expr)?;
    Some(Rewrite::new(rewritten).desc("i · i = -1"))
});

define_rule!(
    GaussianMulRule,
    "Gaussian Multiplication",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_gaussian_mul_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    GaussianAddRule,
    "Gaussian Addition",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_gaussian_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    GaussianDivRule,
    "Gaussian Division",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_gaussian_div_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    SqrtNegativeRule,
    "Square Root of Negative",
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_sqrt_negative_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(ImaginaryPowerRule));
    simplifier.add_rule(Box::new(ISquaredMulRule));
    simplifier.add_rule(Box::new(GaussianMulRule));
    simplifier.add_rule(Box::new(GaussianAddRule));
    simplifier.add_rule(Box::new(GaussianDivRule));
    simplifier.add_rule(Box::new(SqrtNegativeRule));
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
