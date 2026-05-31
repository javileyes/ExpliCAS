//! Source-side hyperbolic by-parts integrand detection.

use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::Zero;

pub(super) fn linear_hyperbolic_by_parts_integrand_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    crate::rule::steps_enabled()
        && cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_hyperbolic_linear_target(
            ctx, expr, var_name,
        )
        && linear_hyperbolic_integer_slope_for_calculus_presentation(ctx, expr, var_name)
}

pub(super) fn repeated_hyperbolic_by_parts_integrand_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_hyperbolic_linear_target(
        ctx, expr, var_name,
    )
}

fn linear_hyperbolic_integer_slope_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    for factor in cas_math::expr_nary::MulView::from_expr(ctx, expr).factors {
        let factor = cas_ast::hold::unwrap_internal_hold(ctx, factor);
        let Expr::Function(fn_id, args) = ctx.get(factor).clone() else {
            continue;
        };
        if args.len() != 1
            || !matches!(
                ctx.builtin_of(fn_id),
                Some(BuiltinFn::Sinh | BuiltinFn::Cosh)
            )
        {
            continue;
        }
        let Ok(arg_poly) = Polynomial::from_expr(ctx, args[0], var_name) else {
            return false;
        };
        if arg_poly.degree() != 1 {
            return false;
        }
        let slope = arg_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        return !slope.is_zero() && slope.is_integer();
    }
    false
}
