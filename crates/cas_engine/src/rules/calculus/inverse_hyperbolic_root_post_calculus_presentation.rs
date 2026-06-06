use cas_ast::{BuiltinFn, Context, Expr, ExprId};

use super::acosh_over_sqrt_derivative_presentation::{
    acosh_polynomial_over_sqrt_derivative_presentation,
    constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation,
};
use super::acosh_sqrt_derivative_presentation::acosh_sqrt_family_derivative_presentation;
use super::asinh_sqrt_derivative_presentation::asinh_sqrt_family_derivative_presentation;
use super::atanh_sqrt_derivative_presentation::atanh_sqrt_family_derivative_presentation;
use super::inverse_tangent_scaled_root_derivative_presentation::atanh_sqrt_over_symbolic_constant_derivative_presentation;

pub(super) fn inverse_hyperbolic_root_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    match direct_inverse_hyperbolic_builtin(ctx, target) {
        Some(BuiltinFn::Asinh) => {
            return asinh_sqrt_family_derivative_presentation(ctx, target, var_name);
        }
        Some(BuiltinFn::Atanh) => {
            if let Some(compact) =
                atanh_sqrt_over_symbolic_constant_derivative_presentation(ctx, target, var_name)
            {
                return Some(compact);
            }
            return atanh_sqrt_family_derivative_presentation(ctx, target, var_name);
        }
        Some(BuiltinFn::Acosh) => {
            if let Some(compact) = acosh_sqrt_family_derivative_presentation(ctx, target, var_name)
            {
                return Some(compact);
            }
            if let Some((compact, _)) =
                acosh_polynomial_over_sqrt_derivative_presentation(ctx, target, var_name)
            {
                return Some(compact);
            }

            let (compact, _) = constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation(
                ctx, target, var_name,
            )?;
            return Some(compact);
        }
        _ => {}
    }

    if let Some(compact) = asinh_sqrt_family_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        atanh_sqrt_over_symbolic_constant_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = atanh_sqrt_family_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) = acosh_sqrt_family_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some((compact, _)) =
        acosh_polynomial_over_sqrt_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }

    let (compact, _) =
        constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation(ctx, target, var_name)?;
    Some(compact)
}

fn direct_inverse_hyperbolic_builtin(ctx: &Context, target: ExprId) -> Option<BuiltinFn> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    match ctx.builtin_of(*fn_id)? {
        BuiltinFn::Asinh => Some(BuiltinFn::Asinh),
        BuiltinFn::Atanh => Some(BuiltinFn::Atanh),
        BuiltinFn::Acosh => Some(BuiltinFn::Acosh),
        _ => None,
    }
}
