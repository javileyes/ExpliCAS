//! Polynomial-scale helpers for symbolic integration routes.

use crate::polynomial::Polynomial;
use crate::symbolic_integration_log_support::ln_abs;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::Zero;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PolynomialSubstitutionKernel {
    Exp,
    Sin,
    Cos,
    Sinh,
    Cosh,
    Tanh,
    Tan,
    Cot,
    Sec,
    Csc,
}

pub(crate) fn polynomial_substitution_kernel(
    ctx: &Context,
    expr: ExprId,
) -> Option<(PolynomialSubstitutionKernel, ExprId)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let kernel = match ctx.builtin_of(*fn_id)? {
                BuiltinFn::Exp => PolynomialSubstitutionKernel::Exp,
                BuiltinFn::Sin => PolynomialSubstitutionKernel::Sin,
                BuiltinFn::Cos => PolynomialSubstitutionKernel::Cos,
                BuiltinFn::Sinh => PolynomialSubstitutionKernel::Sinh,
                BuiltinFn::Cosh => PolynomialSubstitutionKernel::Cosh,
                BuiltinFn::Tanh => PolynomialSubstitutionKernel::Tanh,
                BuiltinFn::Tan => PolynomialSubstitutionKernel::Tan,
                BuiltinFn::Cot => PolynomialSubstitutionKernel::Cot,
                BuiltinFn::Sec => PolynomialSubstitutionKernel::Sec,
                BuiltinFn::Csc => PolynomialSubstitutionKernel::Csc,
                _ => return None,
            };
            Some((kernel, args[0]))
        }
        Expr::Pow(base, exp) if matches!(ctx.get(*base), Expr::Constant(Constant::E)) => {
            Some((PolynomialSubstitutionKernel::Exp, *exp))
        }
        _ => None,
    }
}

pub(crate) fn elementary_polynomial_substitution_kernel_antiderivative(
    ctx: &mut Context,
    kernel: PolynomialSubstitutionKernel,
    arg: ExprId,
    kernel_factor: ExprId,
) -> Option<ExprId> {
    match kernel {
        PolynomialSubstitutionKernel::Exp => Some(kernel_factor),
        PolynomialSubstitutionKernel::Sin => {
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            Some(ctx.add(Expr::Neg(cos_arg)))
        }
        PolynomialSubstitutionKernel::Cos => Some(ctx.call_builtin(BuiltinFn::Sin, vec![arg])),
        PolynomialSubstitutionKernel::Sinh => Some(ctx.call_builtin(BuiltinFn::Cosh, vec![arg])),
        PolynomialSubstitutionKernel::Cosh => Some(ctx.call_builtin(BuiltinFn::Sinh, vec![arg])),
        PolynomialSubstitutionKernel::Tanh => {
            let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
            Some(ln_abs(ctx, cosh_arg))
        }
        PolynomialSubstitutionKernel::Tan => {
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            let log_abs = ln_abs(ctx, cos_arg);
            Some(ctx.add(Expr::Neg(log_abs)))
        }
        PolynomialSubstitutionKernel::Cot => {
            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
            Some(ln_abs(ctx, sin_arg))
        }
        PolynomialSubstitutionKernel::Sec | PolynomialSubstitutionKernel::Csc => None,
    }
}

pub(crate) fn constant_polynomial_ratio(
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<BigRational> {
    if denominator.is_zero() {
        return None;
    }

    let pivot = denominator
        .coeffs
        .iter()
        .position(|coeff| !coeff.is_zero())?;
    let numerator_pivot = numerator
        .coeffs
        .get(pivot)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let scale = numerator_pivot / denominator.coeffs[pivot].clone();
    let len = numerator.coeffs.len().max(denominator.coeffs.len());

    for idx in 0..len {
        let left = numerator
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let right = denominator
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero)
            * scale.clone();
        if left != right {
            return None;
        }
    }

    Some(scale)
}

#[cfg(test)]
mod tests {
    use super::{
        constant_polynomial_ratio, elementary_polynomial_substitution_kernel_antiderivative,
        polynomial_substitution_kernel, PolynomialSubstitutionKernel,
    };
    use crate::polynomial::Polynomial;
    use cas_ast::{BuiltinFn, Constant, Context, Expr};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;
    use num_rational::BigRational;

    fn rational(value: i64) -> BigRational {
        BigRational::new(value.into(), 1.into())
    }

    fn polynomial(coeffs: &[i64]) -> Polynomial {
        Polynomial::new(
            coeffs.iter().map(|value| rational(*value)).collect(),
            "x".into(),
        )
    }

    fn rendered(ctx: &Context, id: cas_ast::ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn detects_constant_polynomial_ratio() {
        let numerator = polynomial(&[2, 4, 6]);
        let denominator = polynomial(&[1, 2, 3]);

        assert_eq!(
            constant_polynomial_ratio(&numerator, &denominator),
            Some(rational(2))
        );
    }

    #[test]
    fn rejects_non_proportional_polynomials() {
        let numerator = polynomial(&[2, 4, 7]);
        let denominator = polynomial(&[1, 2, 3]);

        assert!(constant_polynomial_ratio(&numerator, &denominator).is_none());
    }

    #[test]
    fn detects_function_polynomial_substitution_kernel() {
        let mut ctx = Context::new();
        let arg = parse("x^2 + 1", &mut ctx).unwrap();
        let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);

        assert_eq!(
            polynomial_substitution_kernel(&ctx, sin_arg),
            Some((PolynomialSubstitutionKernel::Sin, arg))
        );
    }

    #[test]
    fn detects_e_power_polynomial_substitution_kernel_as_exp() {
        let mut ctx = Context::new();
        let arg = parse("x^2 + 1", &mut ctx).unwrap();
        let e = ctx.add(Expr::Constant(Constant::E));
        let exp_arg = ctx.add(Expr::Pow(e, arg));

        assert_eq!(
            polynomial_substitution_kernel(&ctx, exp_arg),
            Some((PolynomialSubstitutionKernel::Exp, arg))
        );
    }

    #[test]
    fn builds_elementary_polynomial_substitution_kernel_antiderivative() {
        let mut ctx = Context::new();
        let arg = parse("x", &mut ctx).unwrap();
        let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);

        let antiderivative = elementary_polynomial_substitution_kernel_antiderivative(
            &mut ctx,
            PolynomialSubstitutionKernel::Sin,
            arg,
            sin_arg,
        )
        .unwrap();

        assert_eq!(rendered(&ctx, antiderivative), "-cos(x)");
    }

    #[test]
    fn builds_log_elementary_polynomial_substitution_kernel_antiderivative() {
        let mut ctx = Context::new();
        let arg = parse("x", &mut ctx).unwrap();
        let tan_arg = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);

        let antiderivative = elementary_polynomial_substitution_kernel_antiderivative(
            &mut ctx,
            PolynomialSubstitutionKernel::Tan,
            arg,
            tan_arg,
        )
        .unwrap();

        assert_eq!(rendered(&ctx, antiderivative), "-ln(|cos(x)|)");
    }

    #[test]
    fn leaves_reciprocal_trig_polynomial_substitution_kernel_local() {
        let mut ctx = Context::new();
        let arg = parse("x", &mut ctx).unwrap();
        let sec_arg = ctx.call_builtin(BuiltinFn::Sec, vec![arg]);

        assert!(elementary_polynomial_substitution_kernel_antiderivative(
            &mut ctx,
            PolynomialSubstitutionKernel::Sec,
            arg,
            sec_arg,
        )
        .is_none());
    }
}
