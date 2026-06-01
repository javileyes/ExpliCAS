use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Zero};

use super::differentiation::differentiate;
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_for_calculus_presentation, signed_numerator_for_calculus_presentation,
    signed_rational_const_for_calculus_presentation, split_numeric_scale_single_core,
};
use super::sqrt_small_additive_derivative_presentation::sqrt_small_additive_elementary_derivative_presentation;
use super::{
    bounded_sin_cos_shift_margin_for_calculus_presentation,
    compact_double_angle_sine_products_for_calculus_presentation,
    compact_numeric_mul_factors_for_calculus_presentation,
    compact_small_power_exponents_for_calculus_presentation,
    distribute_half_over_additive_numerator_for_calculus_presentation,
    log_over_sqrt_polynomial_derivative_presentation,
    polynomial_over_sqrt_polynomial_derivative_presentation,
    reciprocal_positive_shifted_sqrt_derivative,
    reciprocal_sqrt_polynomial_product_derivative_presentation,
    signed_elementary_sqrt_polynomial_derivative_presentation,
    sqrt_of_polynomial_quotient_derivative_presentation,
    sqrt_over_log_polynomial_derivative_presentation, sqrt_over_positive_shifted_sqrt_derivative,
    sqrt_polynomial_quotient_derivative_presentation, sqrt_shifted_exp_derivative_presentation,
    sqrt_shifted_ln_derivative_presentation,
};

pub(super) fn sqrt_derivative_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) = reciprocal_positive_shifted_sqrt_derivative(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some((compact, _)) = sqrt_over_positive_shifted_sqrt_derivative(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        reciprocal_sqrt_polynomial_product_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = log_over_sqrt_polynomial_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) = sqrt_over_log_polynomial_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        polynomial_over_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = sqrt_polynomial_quotient_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        sqrt_of_polynomial_quotient_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = sqrt_shifted_exp_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) = sqrt_shifted_ln_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some((compact, _, _)) =
        sqrt_small_additive_elementary_derivative_presentation(ctx, target, var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some(compact) = sqrt_elementary_function_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        sqrt_reciprocal_trig_function_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        polynomial_times_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = sqrt_polynomial_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    signed_elementary_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
}

pub(super) fn sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn sqrt_bounded_trig_positive_shift_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    bounded_sin_cos_shift_margin_for_calculus_presentation(ctx, radicand)?;
    let presentation_radicand =
        compact_double_angle_sine_products_for_calculus_presentation(ctx, radicand)
            .filter(|candidate| {
                bounded_sin_cos_shift_margin_for_calculus_presentation(ctx, *candidate).is_some()
            })
            .unwrap_or(radicand);

    let derivative = differentiate(ctx, presentation_radicand, var_name)?;
    let derivative = compact_small_power_exponents_for_calculus_presentation(ctx, derivative);
    let derivative = compact_numeric_mul_factors_for_calculus_presentation(ctx, derivative);
    if cas_ast::views::as_rational_const(ctx, derivative, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }
    let (derivative_scale, derivative_core) = split_numeric_scale_single_core(ctx, derivative)
        .unwrap_or((BigRational::one(), derivative));
    let coefficient = derivative_scale * BigRational::new(1.into(), 2.into());
    let distributed_numerator = if coefficient == BigRational::new(1.into(), 2.into()) {
        distribute_half_over_additive_numerator_for_calculus_presentation(ctx, derivative_core)
    } else {
        None
    };
    let (numerator, denominator_coeff) = if let Some(numerator) = distributed_numerator {
        (numerator, BigRational::one())
    } else {
        let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
        (
            scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core),
            denominator_coeff,
        )
    };
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![presentation_radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

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

pub(super) fn sqrt_reciprocal_trig_function_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    let Expr::Function(fn_id, args) = ctx.get(radicand).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let (derivative_fn, sign) = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sec) => (BuiltinFn::Tan, BigRational::one()),
        Some(BuiltinFn::Csc) => (BuiltinFn::Cot, -BigRational::one()),
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

    let derivative_core_is_one = cas_ast::views::as_rational_const(ctx, derivative_core, 8)
        .is_some_and(|value| value.is_one());
    let trig_factor = ctx.call_builtin(derivative_fn, vec![args[0]]);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut numerator_factors = Vec::new();
    if !derivative_core_is_one {
        numerator_factors.push(derivative_core);
    }
    numerator_factors.push(trig_factor);
    numerator_factors.push(sqrt_radicand);
    let numerator_core = cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors);
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    if denominator_coeff == BigRational::one() {
        return Some(numerator);
    }

    let denominator = rational_const_for_calculus_presentation(ctx, denominator_coeff);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn polynomial_times_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Mul(_, _) = ctx.get(target) else {
        return None;
    };

    let mut polynomial_factors = Vec::new();
    let mut radicand = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, target) {
        if let Some(factor_radicand) = extract_square_root_base(ctx, factor) {
            if radicand.replace(factor_radicand).is_some() {
                return None;
            }
        } else {
            polynomial_factors.push(factor);
        }
    }

    let radicand = radicand?;
    if polynomial_factors.is_empty() {
        return None;
    }

    let polynomial_expr = if polynomial_factors.len() == 1 {
        polynomial_factors[0]
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &polynomial_factors)
    };
    let multiplier_poly =
        polynomial_radicand_for_calculus_presentation(ctx, polynomial_expr, var_name)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let multiplier_derivative = multiplier_poly.derivative();
    let radicand_derivative = radicand_poly.derivative();

    let two_poly = Polynomial::new(
        vec![BigRational::from_integer(2.into())],
        var_name.to_string(),
    );
    let numerator_poly = multiplier_derivative
        .mul(&radicand_poly)
        .mul(&two_poly)
        .add(&multiplier_poly.mul(&radicand_derivative));
    if numerator_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let raw_numerator = numerator_poly.to_expr(ctx);
    let (numerator_core, numerator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_numerator);
    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(numerator_content * BigRational::new(1.into(), 2.into())))?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::sqrt_bounded_trig_positive_shift_derivative_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn bounded_trig_positive_shift_sqrt_derivative_presentation_accepts_multi_function_sum() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(cos(x)+2*sin(x)*cos(x)+4)", &mut ctx).unwrap();
        let compact = sqrt_bounded_trig_positive_shift_derivative_presentation(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("positive shifted bounded trig root should be recognized"));

        assert_eq!(
            rendered(&ctx, compact),
            "(cos(2 * x) - 1/2 * sin(x)) / sqrt(sin(2 * x) + cos(x) + 4)"
        );
    }
}
