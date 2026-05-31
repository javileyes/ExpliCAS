use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::{
    add_rational_for_calculus_presentation, nonzero_rational_parts,
    rational_const_for_calculus_presentation, signed_numerator_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::root_forms::{extract_square_root_base, try_rewrite_simplify_square_root_expr};
use num_rational::BigRational;
use num_traits::{One, Zero};

pub(super) fn inverse_reciprocal_trig_sqrt_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => BigRational::one(),
        Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if radicand_poly.degree() != 1 {
        return None;
    }

    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let derivative_core_factor = cas_ast::views::as_rational_const(ctx, derivative_core, 8)?;
    if derivative_core_factor.is_zero() {
        return None;
    }

    let coefficient =
        sign * derivative_content * derivative_core_factor * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let one = ctx.num(1);
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);

    let gap = add_rational_for_calculus_presentation(ctx, radicand, -BigRational::one());
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[radicand, sqrt_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn inverse_reciprocal_trig_sqrt_quadratic_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => BigRational::one(),
        Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if radicand_poly.degree() != 2 {
        return None;
    }

    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let gap = add_rational_for_calculus_presentation(ctx, radicand, -BigRational::one());
    let raw_sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let sqrt_gap = try_rewrite_simplify_square_root_expr(ctx, raw_sqrt_gap)
        .map(|rewrite| rewrite.rewritten)
        .unwrap_or(raw_sqrt_gap);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[radicand, sqrt_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}
