use super::polynomial_support::{
    nonzero_affine_variable_derivative, polynomial_radicand_for_calculus_presentation,
    rational_polynomial_content_for_calculus_presentation,
    scale_polynomial_for_calculus_presentation,
};
use super::result_presentation::{
    divide_compact_derivative_by_constant_factor,
    reciprocal_constant_denominator_for_calculus_presentation,
    remove_unit_mul_factors_for_calculus_presentation,
};
use super::scalar_presentation::{
    add_one_for_calculus_presentation, add_rational_for_calculus_presentation,
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_for_calculus_presentation, signed_numerator_for_calculus_presentation,
};
use super::sqrt_product_presentation::shared_positive_content_sqrt_product_for_calculus_presentation;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Zero};

pub(super) fn acosh_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Acosh) {
        return None;
    }

    let arg = args[0];
    if let Some(compact) = acosh_fractional_affine_derivative_presentation(ctx, arg, var_name) {
        return Some(compact);
    }

    let slope = nonzero_affine_variable_derivative(ctx, arg, var_name)?;
    let one = ctx.num(1);
    let numerator = scale_expr_for_calculus_presentation(ctx, slope, one);
    let lower_branch = add_rational_for_calculus_presentation(ctx, arg, -BigRational::one());
    let upper_branch = add_one_for_calculus_presentation(ctx, arg);
    let sqrt_lower = ctx.call_builtin(BuiltinFn::Sqrt, vec![lower_branch]);
    let sqrt_upper = ctx.call_builtin(BuiltinFn::Sqrt, vec![upper_branch]);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_lower, sqrt_upper]);
    let result = ctx.add(Expr::Div(numerator, denominator));

    Some((
        result,
        vec![crate::ImplicitCondition::Positive(lower_branch)],
    ))
}

fn acosh_fractional_affine_derivative_presentation(
    ctx: &mut Context,
    arg: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, arg, var_name)?;
    if arg_poly.degree() != 1 {
        return None;
    }

    let arg_content = rational_polynomial_content_for_calculus_presentation(&arg_poly);
    if arg_content.is_zero() || arg_content.is_integer() {
        return None;
    }

    let primitive_arg_poly = arg_poly.div_scalar(&arg_content);
    let derivative_poly = primitive_arg_poly.derivative();
    if derivative_poly.is_zero() || derivative_poly.degree() != 0 {
        return None;
    }

    let content_num = BigRational::from_integer(arg_content.numer().clone());
    let content_den = BigRational::from_integer(arg_content.denom().clone());
    let scaled_arg_poly =
        scale_polynomial_for_calculus_presentation(&primitive_arg_poly, &content_num);
    let denominator_gap =
        Polynomial::new(vec![content_den.clone()], primitive_arg_poly.var.clone());
    let lower_poly = scaled_arg_poly.sub(&denominator_gap);
    let upper_poly = scaled_arg_poly.add(&denominator_gap);
    let lower_branch = lower_poly.to_expr(ctx);
    let upper_branch = upper_poly.to_expr(ctx);

    let derivative_coeff = derivative_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let one = ctx.num(1);
    let mut numerator_coeff = derivative_coeff * content_num;
    let mut denominator_coeff = BigRational::one();
    let denominator = if let Some((primitive_product, shared_content)) =
        shared_positive_content_sqrt_product_for_calculus_presentation(
            ctx,
            lower_branch,
            upper_branch,
        ) {
        (numerator_coeff, denominator_coeff) =
            nonzero_rational_parts(&(numerator_coeff / shared_content))?;
        primitive_product
    } else {
        let sqrt_lower = ctx.call_builtin(BuiltinFn::Sqrt, vec![lower_branch]);
        let sqrt_upper = ctx.call_builtin(BuiltinFn::Sqrt, vec![upper_branch]);
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_lower, sqrt_upper])
    };
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);
    let denominator = if denominator_coeff == BigRational::one() {
        denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, denominator])
    };
    let result = ctx.add(Expr::Div(numerator, denominator));

    Some((
        result,
        vec![crate::ImplicitCondition::Positive(lower_branch)],
    ))
}

pub(super) fn constant_scaled_acosh_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if let Expr::Div(inner, outer_den) = ctx.get(target).clone() {
        if contains_named_var(ctx, outer_den, var_name) {
            return None;
        }

        let inner = remove_unit_mul_factors_for_calculus_presentation(ctx, inner);
        let (inner_derivative, required_conditions) =
            acosh_affine_derivative_presentation(ctx, inner, var_name)?;
        let result = divide_compact_derivative_by_constant_factor(ctx, inner_derivative, outer_den);
        let result = cas_ast::hold::wrap_hold(ctx, result);
        return Some((result, required_conditions));
    }

    let Expr::Mul(_, _) = ctx.get(target).clone() else {
        return None;
    };

    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    for idx in 0..factors.len() {
        let inner = factors[idx];
        let mut constant_factors = factors.clone();
        constant_factors.remove(idx);
        let [constant_factor] = constant_factors.as_slice() else {
            continue;
        };
        let Some(outer_den) = reciprocal_constant_denominator_for_calculus_presentation(
            ctx,
            *constant_factor,
            var_name,
        ) else {
            continue;
        };
        let Some((inner_derivative, required_conditions)) =
            acosh_affine_derivative_presentation(ctx, inner, var_name)
        else {
            continue;
        };
        let result = divide_compact_derivative_by_constant_factor(ctx, inner_derivative, outer_den);
        let result = cas_ast::hold::wrap_hold(ctx, result);
        return Some((result, required_conditions));
    }

    None
}
