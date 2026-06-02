use super::polynomial_support::{
    polynomial_derivative_expr_for_calculus_presentation,
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation, square_of_strictly_positive_quadratic_arg,
    strictly_positive_quadratic_on_reals,
};
use super::scalar_presentation::{
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

fn positive_rational_denominator_scaled_base(
    ctx: &mut Context,
    base: ExprId,
    var_name: &str,
) -> Option<(ExprId, BigRational, ExprId)> {
    if let Expr::Div(numerator, denominator) = ctx.get(base) {
        let scale = cas_ast::views::as_rational_const(ctx, *denominator, 8)?;
        if !scale.is_positive() {
            return None;
        }

        let numerator_poly =
            polynomial_radicand_for_calculus_presentation(ctx, *numerator, var_name)?;
        return strictly_positive_quadratic_on_reals(&numerator_poly)
            .then_some((*numerator, scale, base));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, base);
    if factors.len() < 2 {
        return None;
    }

    let mut scale = BigRational::one();
    let mut non_rational_factor = None;
    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            if !value.is_positive() {
                return None;
            }
            scale *= value;
            continue;
        }

        if non_rational_factor.replace(factor).is_some() {
            return None;
        }
    }

    if scale.is_one() {
        return None;
    }
    let core = non_rational_factor?;
    let core_poly = polynomial_radicand_for_calculus_presentation(ctx, core, var_name)?;
    if !strictly_positive_quadratic_on_reals(&core_poly) {
        return None;
    }

    let compact_scaled_base = if scale.numer() == &BigInt::one() {
        let denominator = rational_const_for_calculus_presentation(
            ctx,
            BigRational::from_integer(scale.denom().clone()),
        );
        ctx.add(Expr::Div(core, denominator))
    } else {
        scale_expr_for_calculus_presentation(ctx, scale.clone(), core)
    };

    Some((core, BigRational::one() / scale, compact_scaled_base))
}

pub(super) fn inverse_reciprocal_trig_positive_quadratic_square_presentation(
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

    let base = square_of_strictly_positive_quadratic_arg(ctx, args[0], var_name)?;
    let derivative = polynomial_derivative_expr_for_calculus_presentation(ctx, base, var_name)?;
    if cas_ast::views::as_rational_const(ctx, derivative, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let (denominator_base, numerator_scale, gap_base) = positive_rational_denominator_scaled_base(
        ctx, base, var_name,
    )
    .unwrap_or((base, BigRational::one(), base));
    let numerator = signed_numerator_for_calculus_presentation(
        ctx,
        sign * derivative_content * BigRational::from_integer(2.into()) * numerator_scale,
        derivative_core,
    );

    let four = ctx.num(4);
    let base_fourth = ctx.add(Expr::Pow(gap_base, four));
    let one = ctx.num(1);
    let gap = ctx.add(Expr::Sub(base_fourth, one));
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_base, sqrt_gap]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}
