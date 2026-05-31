//! Source-side arctan-polynomial integrand detection for calculus shortcuts.

use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;

pub(super) fn polynomial_times_arctan_affine_integrand_for_diff_shortcut(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let terms = cas_math::expr_nary::AddView::from_expr(ctx, expr).terms;
    !terms.is_empty()
        && terms.into_iter().all(|(term, _)| {
            polynomial_times_arctan_affine_term_for_diff_shortcut(ctx, term, var_name)
        })
}

fn polynomial_times_arctan_affine_term_for_diff_shortcut(
    ctx: &Context,
    term: ExprId,
    var_name: &str,
) -> bool {
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    match ctx.get(term).clone() {
        Expr::Neg(inner) => {
            return polynomial_times_arctan_affine_term_for_diff_shortcut(ctx, inner, var_name);
        }
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, den, 8).is_some() => {
            return polynomial_times_arctan_affine_term_for_diff_shortcut(ctx, num, var_name);
        }
        _ => {}
    }

    let mut arctan_arg = None;
    let mut polynomial_factor = Polynomial::one(var_name.to_string());
    for factor in cas_math::expr_nary::MulView::from_expr(ctx, term).factors {
        let factor = cas_ast::hold::unwrap_internal_hold(ctx, factor);
        if let Expr::Function(fn_id, args) = ctx.get(factor).clone() {
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(fn_id),
                    Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                )
            {
                if arctan_arg.replace(args[0]).is_some() {
                    return false;
                }
                continue;
            }
        }

        let Ok(factor_poly) = Polynomial::from_expr(ctx, factor, var_name) else {
            return false;
        };
        polynomial_factor = polynomial_factor.mul(&factor_poly);
    }

    let Some(arg) = arctan_arg else {
        return false;
    };
    let Ok(arg_poly) = Polynomial::from_expr(ctx, arg, var_name) else {
        return false;
    };
    arg_poly.degree() == 1 && !arg_poly.derivative().is_zero() && !polynomial_factor.is_zero()
}
