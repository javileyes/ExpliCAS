use super::derivative_result_scaling_presentation::scale_compact_derivative_by_rational;
use super::inverse_tangent_root_args::arctan_sqrt_radicand_arg;
use super::polynomial_support::{
    nonzero_affine_variable_derivative, polynomial_is_strictly_positive_everywhere,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::{unwrap_internal_hold_for_calculus, variable_named};
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    rational_scaled_single_factor, signed_numerator_for_calculus_presentation,
    split_numeric_scale_single_core,
};
use crate::symbolic_calculus_call_support::try_extract_diff_call;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

fn positive_scaled_variable_factor(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    if variable_named(ctx, target, var_name) {
        return Some(BigRational::one());
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    if factors.len() < 2 {
        return None;
    }

    let mut scale = BigRational::one();
    let mut saw_variable = false;
    for factor in factors {
        if variable_named(ctx, factor, var_name) {
            if saw_variable {
                return None;
            }
            saw_variable = true;
            continue;
        }

        let value = cas_ast::views::as_rational_const(ctx, factor, 8)?;
        if !value.is_positive() {
            return None;
        }
        scale *= value;
    }

    saw_variable.then_some(scale)
}

pub(super) fn arctan_sqrt_scaled_variable_arg(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, BigRational)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let scale = positive_scaled_variable_factor(ctx, radicand, var_name)
        .or_else(|| nonzero_affine_variable_derivative(ctx, radicand, var_name))?;
    Some((radicand, scale))
}

pub(super) fn arctan_sqrt_affine_partition_quotient_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };

    let numerator_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
    if numerator_poly.degree() > 1 || denominator_poly.degree() > 1 {
        return None;
    }

    let partition_sum = numerator_poly.add(&denominator_poly);
    if partition_sum.degree() != 0 {
        return None;
    }
    let partition_total = partition_sum
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !partition_total.is_positive() {
        return None;
    }

    let wronskian = numerator_poly
        .derivative()
        .mul(&denominator_poly)
        .sub(&numerator_poly.mul(&denominator_poly.derivative()));
    if wronskian.degree() != 0 {
        return None;
    }
    let wronskian_value = wronskian
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if wronskian_value.is_zero() {
        return Some(ctx.num(0));
    }

    let coefficient =
        derivative_sign * wronskian_value / (BigRational::from_integer(2.into()) * partition_total);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let sqrt_num = ctx.call_builtin(BuiltinFn::Sqrt, vec![num]);
    let sqrt_den = ctx.call_builtin(BuiltinFn::Sqrt, vec![den]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_num, sqrt_den]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn arctan_sqrt_polynomial_quotient_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };

    let numerator_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
    if numerator_poly.degree() == 0
        || denominator_poly.degree() == 0
        || numerator_poly.degree() > 2
        || denominator_poly.degree() > 2
    {
        return None;
    }

    let sum_poly = numerator_poly.add(&denominator_poly);
    if sum_poly.degree() == 0 || sum_poly.degree() > 2 {
        return None;
    }

    let wronskian = numerator_poly
        .derivative()
        .mul(&denominator_poly)
        .sub(&numerator_poly.mul(&denominator_poly.derivative()));
    if wronskian.degree() > 1 {
        return None;
    }
    if wronskian.is_zero() {
        return Some(ctx.num(0));
    }

    let sum_expr = sum_poly.to_expr(ctx);
    let (sum_core, sum_content) = split_polynomial_content_for_calculus_presentation(ctx, sum_expr);
    if sum_content.is_zero() {
        return None;
    }

    let wronskian_expr = wronskian.to_expr(ctx);
    let (wronskian_core, wronskian_content) =
        split_polynomial_content_for_calculus_presentation(ctx, wronskian_expr);
    if wronskian_content.is_zero() {
        return None;
    }

    let coefficient =
        derivative_sign * wronskian_content / (BigRational::from_integer(2.into()) * sum_content);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, wronskian_core);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        cas_math::expr_nary::build_balanced_mul(ctx, &[sum_core, den, sqrt_radicand])
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[denominator_scale, sum_core, den, sqrt_radicand],
        )
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn arctan_sqrt_positive_polynomial_quotient_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };

    let numerator_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let sum_poly = numerator_poly.add(&denominator_poly);

    if !polynomial_is_strictly_positive_everywhere(&numerator_poly)
        || !polynomial_is_strictly_positive_everywhere(&denominator_poly)
        || !polynomial_is_strictly_positive_everywhere(&sum_poly)
    {
        return None;
    }

    let compact = arctan_sqrt_polynomial_quotient_derivative_presentation(
        ctx,
        target,
        var_name,
        BigRational::one(),
    )?;
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(crate) fn arctan_sqrt_positive_polynomial_quotient_derivative_for_diff_call(
    ctx: &mut Context,
    source: ExprId,
) -> Option<ExprId> {
    let call = try_extract_diff_call(ctx, source)?;
    let target = unwrap_internal_hold_for_calculus(ctx, call.target);
    arctan_sqrt_positive_polynomial_quotient_derivative_shortcut(ctx, target, &call.var_name)
}

pub(super) fn arctan_sqrt_variable_over_positive_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Arctan | BuiltinFn::Atan)
        )
    {
        return None;
    }

    let (argument_scale, numerator, denominator) = match ctx.get(args[0]).clone() {
        Expr::Div(numerator, denominator) => (BigRational::one(), numerator, denominator),
        _ => {
            let (argument_scale, quotient_core) = split_numeric_scale_single_core(ctx, args[0])?;
            if !argument_scale.is_positive() {
                return None;
            }
            let Expr::Div(numerator, denominator) = ctx.get(quotient_core).clone() else {
                return None;
            };
            (argument_scale, numerator, denominator)
        }
    };
    let (numerator_scale, numerator_core) = split_numeric_scale_single_core(ctx, numerator)?;
    let numerator_scale = argument_scale * numerator_scale;
    if !numerator_scale.is_positive() {
        return None;
    }
    let radicand = extract_square_root_base(ctx, numerator_core)?;
    let radicand_scale = positive_scaled_variable_factor(ctx, radicand, var_name)?;
    let numerator_derivative_scale = numerator_scale.clone() * radicand_scale.clone();
    let denominator_variable_scale =
        numerator_scale.clone() * numerator_scale * radicand_scale.clone();

    let denominator_poly = Polynomial::from_expr(ctx, denominator, var_name).ok()?;
    if denominator_poly.degree() != 1 {
        return None;
    }
    let denominator_constant = denominator_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let denominator_slope = denominator_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !denominator_slope.is_positive() || denominator_constant.is_negative() {
        return None;
    }

    let var_poly = Polynomial::new(
        vec![BigRational::zero(), BigRational::one()],
        var_name.to_string(),
    );
    let scaled_var_poly = Polynomial::new(
        vec![BigRational::zero(), denominator_variable_scale],
        var_name.to_string(),
    );
    let scale_poly = Polynomial::new(vec![numerator_derivative_scale], var_name.to_string());
    let two = Polynomial::new(
        vec![BigRational::from_integer(2.into())],
        var_name.to_string(),
    );
    let mut numerator_poly = denominator_poly
        .sub(&var_poly.mul(&denominator_poly.derivative()).mul(&two))
        .mul(&scale_poly);
    if numerator_poly.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }
    let mut denominator_coeff = BigRational::from_integer(2.into());
    let numerator_content = numerator_poly.content();
    if numerator_content.is_positive() && numerator_content.denom().is_one() {
        let common_integer = numerator_content
            .numer()
            .gcd(denominator_coeff.numer())
            .abs();
        if common_integer > BigInt::from(1) {
            let common = BigRational::from_integer(common_integer);
            numerator_poly = numerator_poly.div_scalar(&common);
            denominator_coeff /= common;
        }
    }

    let denominator_sum_poly = denominator_poly
        .mul(&denominator_poly)
        .add(&scaled_var_poly);
    let denominator_sum = denominator_sum_poly.to_expr(ctx);
    let numerator_expr = numerator_poly.to_expr(ctx);
    let sqrt_var = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut denominator_factors = Vec::new();
    if !denominator_coeff.is_one() {
        denominator_factors.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_factors.push(sqrt_var);
    denominator_factors.push(denominator_sum);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    let result = ctx.add(Expr::Div(numerator_expr, denominator));
    let var = ctx.var(var_name);

    Some((
        cas_ast::hold::wrap_hold(ctx, result),
        vec![crate::ImplicitCondition::Positive(var)],
    ))
}

pub(super) fn constant_scaled_arctan_sqrt_variable_over_positive_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if let Expr::Div(inner, outer_den) = ctx.get(target).clone() {
        let denominator_scale = cas_ast::views::as_rational_const(ctx, outer_den, 8)?;
        if denominator_scale.is_zero() {
            return None;
        }
        let (derivative, required_conditions) =
            arctan_sqrt_variable_over_positive_affine_derivative_presentation(
                ctx, inner, var_name,
            )?;
        let scaled = scale_compact_derivative_by_rational(
            ctx,
            derivative,
            BigRational::one() / denominator_scale,
        );
        return Some((cas_ast::hold::wrap_hold(ctx, scaled), required_conditions));
    }

    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let (derivative, required_conditions) =
        arctan_sqrt_variable_over_positive_affine_derivative_presentation(ctx, inner, var_name)?;
    let scaled = scale_compact_derivative_by_rational(ctx, derivative, scale);
    Some((cas_ast::hold::wrap_hold(ctx, scaled), required_conditions))
}
