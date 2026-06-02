use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::One;

use super::polynomial_support::split_polynomial_content_for_calculus_presentation;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_for_calculus_presentation, signed_rational_const_for_calculus_presentation,
};

pub(super) fn sqrt_elementary_function_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    #[derive(Clone, Copy)]
    enum SqrtElementaryDerivativeShape {
        Function(BuiltinFn),
        DenominatorSquare(BuiltinFn),
        OnePlusArgSquare,
        OneMinusArgSquare,
        SqrtOneMinusArgSquare,
        SqrtOnePlusArgSquare,
        SqrtArgMinusOneTimesArgPlusOne,
        Log,
        LogConstantBase(i64),
    }

    let radicand = extract_square_root_base(ctx, target)?;
    let Expr::Function(fn_id, args) = ctx.get(radicand).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let (shape, sign) = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sin) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Cos),
            BigRational::one(),
        ),
        Some(BuiltinFn::Cos) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Sin),
            -BigRational::one(),
        ),
        Some(BuiltinFn::Exp) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Exp),
            BigRational::one(),
        ),
        Some(BuiltinFn::Tan) => (
            SqrtElementaryDerivativeShape::DenominatorSquare(BuiltinFn::Cos),
            BigRational::one(),
        ),
        Some(BuiltinFn::Tanh) => (
            SqrtElementaryDerivativeShape::DenominatorSquare(BuiltinFn::Cosh),
            BigRational::one(),
        ),
        Some(BuiltinFn::Cot) => (
            SqrtElementaryDerivativeShape::DenominatorSquare(BuiltinFn::Sin),
            -BigRational::one(),
        ),
        Some(BuiltinFn::Atan | BuiltinFn::Arctan) => (
            SqrtElementaryDerivativeShape::OnePlusArgSquare,
            BigRational::one(),
        ),
        Some(BuiltinFn::Atanh) => (
            SqrtElementaryDerivativeShape::OneMinusArgSquare,
            BigRational::one(),
        ),
        Some(BuiltinFn::Asin | BuiltinFn::Arcsin) => (
            SqrtElementaryDerivativeShape::SqrtOneMinusArgSquare,
            BigRational::one(),
        ),
        Some(BuiltinFn::Acos | BuiltinFn::Arccos) => (
            SqrtElementaryDerivativeShape::SqrtOneMinusArgSquare,
            -BigRational::one(),
        ),
        Some(BuiltinFn::Ln) => (SqrtElementaryDerivativeShape::Log, BigRational::one()),
        Some(BuiltinFn::Log2) => (
            SqrtElementaryDerivativeShape::LogConstantBase(2),
            BigRational::one(),
        ),
        Some(BuiltinFn::Log10) => (
            SqrtElementaryDerivativeShape::LogConstantBase(10),
            BigRational::one(),
        ),
        Some(BuiltinFn::Asinh) => (
            SqrtElementaryDerivativeShape::SqrtOnePlusArgSquare,
            BigRational::one(),
        ),
        Some(BuiltinFn::Acosh) => (
            SqrtElementaryDerivativeShape::SqrtArgMinusOneTimesArgPlusOne,
            BigRational::one(),
        ),
        Some(BuiltinFn::Sinh) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Cosh),
            BigRational::one(),
        ),
        Some(BuiltinFn::Cosh) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Sinh),
            BigRational::one(),
        ),
        _ => return None,
    };

    let arg_poly = Polynomial::from_expr(ctx, args[0], var_name).ok()?;
    let derivative_poly = arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (mut derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let mut coefficient = sign * derivative_content * BigRational::new(1.into(), 2.into());
    if let Some(core_value) = signed_rational_const_for_calculus_presentation(ctx, derivative_core)
    {
        coefficient *= core_value;
        derivative_core = ctx.num(1);
    }
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let derivative_function = match shape {
        SqrtElementaryDerivativeShape::Function(derivative_fn) => {
            Some(ctx.call_builtin(derivative_fn, vec![args[0]]))
        }
        SqrtElementaryDerivativeShape::DenominatorSquare(_) => None,
        SqrtElementaryDerivativeShape::OnePlusArgSquare => None,
        SqrtElementaryDerivativeShape::OneMinusArgSquare => None,
        SqrtElementaryDerivativeShape::SqrtOneMinusArgSquare => None,
        SqrtElementaryDerivativeShape::SqrtOnePlusArgSquare => None,
        SqrtElementaryDerivativeShape::SqrtArgMinusOneTimesArgPlusOne => None,
        SqrtElementaryDerivativeShape::Log => None,
        SqrtElementaryDerivativeShape::LogConstantBase(_) => None,
    };
    let derivative_core_is_one = cas_ast::views::as_rational_const(ctx, derivative_core, 8)
        .is_some_and(|value| value.is_one());
    let numerator_core = match derivative_function {
        Some(derivative_function) if derivative_core_is_one => derivative_function,
        Some(derivative_function) => {
            cas_math::expr_nary::build_balanced_mul(ctx, &[derivative_core, derivative_function])
        }
        None => derivative_core,
    };
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut denominator_factors = Vec::new();
    if denominator_coeff != BigRational::one() {
        denominator_factors.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::Log) {
        denominator_factors.push(args[0]);
    }
    if let SqrtElementaryDerivativeShape::LogConstantBase(base) = shape {
        denominator_factors.push(args[0]);
        let base_expr = ctx.num(base);
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Ln, vec![base_expr]));
    }
    if let SqrtElementaryDerivativeShape::DenominatorSquare(denominator_fn) = shape {
        let denominator_arg = ctx.call_builtin(denominator_fn, vec![args[0]]);
        let two = ctx.num(2);
        denominator_factors.push(ctx.add(Expr::Pow(denominator_arg, two)));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::OnePlusArgSquare) {
        let two = ctx.num(2);
        let one = ctx.num(1);
        let arg_square = ctx.add(Expr::Pow(args[0], two));
        denominator_factors.push(ctx.add(Expr::Add(arg_square, one)));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::OneMinusArgSquare) {
        let two = ctx.num(2);
        let one = ctx.num(1);
        let arg_square = ctx.add(Expr::Pow(args[0], two));
        let neg_arg_square = ctx.add(Expr::Neg(arg_square));
        denominator_factors.push(ctx.add(Expr::Add(one, neg_arg_square)));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::SqrtOneMinusArgSquare) {
        let two = ctx.num(2);
        let one = ctx.num(1);
        let arg_square = ctx.add(Expr::Pow(args[0], two));
        let neg_arg_square = ctx.add(Expr::Neg(arg_square));
        let radicand = ctx.add(Expr::Add(one, neg_arg_square));
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::SqrtOnePlusArgSquare) {
        let two = ctx.num(2);
        let one = ctx.num(1);
        let arg_square = ctx.add(Expr::Pow(args[0], two));
        let radicand = ctx.add(Expr::Add(arg_square, one));
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]));
    }
    if matches!(
        shape,
        SqrtElementaryDerivativeShape::SqrtArgMinusOneTimesArgPlusOne
    ) {
        let one_poly = Polynomial::one(var_name.to_string());
        let arg_minus_one = arg_poly.sub(&one_poly).to_expr(ctx);
        let arg_plus_one = arg_poly.add(&one_poly).to_expr(ctx);
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![arg_minus_one]));
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![arg_plus_one]));
    }
    denominator_factors.push(sqrt_radicand);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);

    Some(ctx.add_raw(Expr::Div(numerator, denominator)))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::sqrt_elementary_function_derivative_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn sqrt_elementary_function_derivative_presentation_handles_sine_radicand() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(sin(x))", &mut ctx).unwrap();
        let compact = sqrt_elementary_function_derivative_presentation(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("sqrt elementary derivative should be recognized"));

        assert_eq!(rendered(&ctx, compact), "cos(x) / (2 * sqrt(sin(x)))");
    }
}
