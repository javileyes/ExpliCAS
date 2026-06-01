use super::differentiation::differentiate;
use super::gap_presentation::{
    compact_squared_affine_gap_for_calculus_presentation, primitive_positive_gap,
    reciprocal_positive_rational, squared_expr_for_compact_gap_presentation,
};
use super::polynomial_support::{
    polynomial_derivative_expr_for_calculus_presentation,
    polynomial_radicand_for_calculus_presentation,
    rational_polynomial_content_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::squared_expr;
use super::scalar_presentation::{
    exact_positive_rational_sqrt_for_calculus_presentation, fold_numeric_mul_constants_for_hold,
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_by_sqrt_positive_rational_for_calculus_presentation,
    scale_expr_for_calculus_presentation, signed_numerator_for_calculus_presentation,
};
use super::surd_quotient_args::{
    atanh_arg_over_sqrt_parts, sqrt_scaled_arg_over_sqrt_parts_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn unit_interval_bounded_inverse_trig_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let mut outer_coeff = BigRational::one();
    let mut inverse_trig = None;
    let mut target = target;

    if let Expr::Neg(inner) = ctx.get(target).clone() {
        outer_coeff = -outer_coeff;
        target = inner;
    }

    if let Expr::Div(numerator, denominator) = ctx.get(target).clone() {
        let denominator = cas_ast::views::as_rational_const(ctx, denominator, 8)?;
        if denominator.is_zero() {
            return None;
        }
        outer_coeff /= denominator;
        target = numerator;
    }

    for factor in cas_math::expr_nary::mul_leaves(ctx, target) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            outer_coeff *= value;
            continue;
        }

        let Expr::Function(fn_id, args) = ctx.get(factor).clone() else {
            return None;
        };
        let derivative_sign = match ctx.builtin_of(fn_id) {
            Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => BigRational::one(),
            Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -BigRational::one(),
            _ => return None,
        };
        if args.len() != 1 || inverse_trig.is_some() {
            return None;
        }
        inverse_trig = Some((derivative_sign, args[0]));
    }

    let (derivative_sign, arg) = inverse_trig?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var_name).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }
    let offset = arg_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let two = BigRational::from_integer(2.into());
    let unit_interval_arg = (offset == -BigRational::one() && slope == two)
        || (offset == BigRational::one() && slope == -two);
    if !unit_interval_arg {
        return None;
    }

    let coefficient = outer_coeff * derivative_sign * slope / BigRational::from_integer(2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let one = ctx.num(1);
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);
    let var = ctx.var(var_name);
    let sqrt_var = ctx.call_builtin(BuiltinFn::Sqrt, vec![var]);
    let one = ctx.num(1);
    let var = ctx.var(var_name);
    let one_minus_var = ctx.add(Expr::Sub(one, var));
    let sqrt_one_minus_var = ctx.call_builtin(BuiltinFn::Sqrt, vec![one_minus_var]);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_var, sqrt_one_minus_var]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn bounded_inverse_trig_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let derivative_sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => BigRational::one(),
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let arg = args[0];
    let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, arg, var_name)?;
    let arg_content = rational_polynomial_content_for_calculus_presentation(&arg_poly);
    if arg_content.is_zero() {
        return None;
    }
    let primitive_arg_poly = if arg_content.is_one() {
        arg_poly
    } else {
        arg_poly.div_scalar(&arg_content)
    };
    let derivative_poly = primitive_arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let content_num = BigRational::from_integer(arg_content.numer().clone());
    let numerator = scale_expr_for_calculus_presentation(
        ctx,
        derivative_sign * derivative_content * content_num,
        derivative_core,
    );

    let primitive_arg = primitive_arg_poly.to_expr(ctx);
    let primitive_arg_sq = squared_expr(ctx, primitive_arg);
    let raw_gap = if arg_content.is_one() {
        let one = ctx.num(1);
        ctx.add(Expr::Sub(one, primitive_arg_sq))
    } else {
        let content_num_sq =
            BigRational::from_integer(arg_content.numer().clone() * arg_content.numer().clone());
        let content_den_sq =
            BigRational::from_integer(arg_content.denom().clone() * arg_content.denom().clone());
        let scaled_arg_sq =
            scale_expr_for_calculus_presentation(ctx, content_num_sq, primitive_arg_sq);
        let den_sq = rational_const_for_calculus_presentation(ctx, content_den_sq);
        ctx.add(Expr::Sub(den_sq, scaled_arg_sq))
    };
    let (gap, gap_content) = primitive_positive_gap(ctx, raw_gap);
    let numerator = scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
        ctx,
        reciprocal_positive_rational(&gap_content),
        numerator,
    );
    let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn bounded_inverse_trig_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };

    let sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => 1,
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -1,
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let (num, radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])
        .or_else(|| sqrt_scaled_arg_over_sqrt_parts_for_calculus_presentation(ctx, args[0]))?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let d_num = polynomial_derivative_expr_for_calculus_presentation(ctx, num, var_name)
        .or_else(|| differentiate(ctx, num, var_name))?;
    let (d_num_core, d_num_content) =
        split_polynomial_content_for_calculus_presentation(ctx, d_num);
    let numerator_content = if sign < 0 {
        -d_num_content
    } else {
        d_num_content
    };
    let radicand_numer = BigRational::from_integer(radicand_value.numer().clone());
    let radicand_denom = BigRational::from_integer(radicand_value.denom().clone());
    let numerator = if radicand_denom.is_one() {
        signed_numerator_for_calculus_presentation(ctx, numerator_content.clone(), d_num_core)
    } else if let Some(sqrt_content) =
        exact_positive_rational_sqrt_for_calculus_presentation(&radicand_denom)
    {
        signed_numerator_for_calculus_presentation(
            ctx,
            numerator_content * sqrt_content,
            d_num_core,
        )
    } else {
        let base_numerator =
            signed_numerator_for_calculus_presentation(ctx, numerator_content, d_num_core);
        scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
            ctx,
            radicand_denom.clone(),
            base_numerator,
        )
    };
    let num_square = squared_expr_for_compact_gap_presentation(ctx, num);
    let scaled_num_square = if radicand_denom.is_one() {
        num_square
    } else {
        scale_expr_for_calculus_presentation(ctx, radicand_denom, num_square)
    };
    let compact_numer = rational_const_for_calculus_presentation(ctx, radicand_numer);
    let gap = ctx.add(Expr::Sub(compact_numer, scaled_num_square));
    let gap = compact_squared_affine_gap_for_calculus_presentation(ctx, gap, var_name);
    let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let compact = ctx.add(Expr::Div(numerator, denominator));
    let compact = fold_numeric_mul_constants_for_hold(ctx, compact);

    Some(compact)
}
