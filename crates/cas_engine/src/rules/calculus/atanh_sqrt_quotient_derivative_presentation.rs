use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn atanh_sqrt_affine_quotient_positive_gap_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Atanh) {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };

    let numerator_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
    if numerator_poly.degree() != 1 || denominator_poly.degree() != 1 {
        return None;
    }

    let gap_poly = denominator_poly.sub(&numerator_poly);
    if gap_poly.degree() != 0 {
        return None;
    }
    let gap_value = gap_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !gap_value.is_positive() {
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

    let coefficient = wronskian_value / (BigRational::from_integer(2.into()) * gap_value);
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
