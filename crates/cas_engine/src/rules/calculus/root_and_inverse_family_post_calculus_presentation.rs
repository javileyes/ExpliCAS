use cas_ast::{BuiltinFn, Context, Expr, ExprId};

use super::affine_inverse_family_post_calculus_presentation::affine_inverse_family_post_calculus_presentation;
use super::bounded_inverse_root_quotient_post_calculus_presentation::bounded_inverse_root_quotient_post_calculus_presentation;
use super::constant_scaled_inverse_trig_root_post_calculus_presentation::constant_scaled_inverse_trig_root_post_calculus_presentation;
use super::inverse_hyperbolic_root_post_calculus_presentation::inverse_hyperbolic_root_post_calculus_presentation;
use super::inverse_reciprocal_trig_positive_quadratic_derivative_presentation::inverse_reciprocal_trig_post_calculus_presentation;
use super::inverse_surd_quotient_derivative_presentation::inverse_surd_quotient_post_calculus_presentation;
use super::inverse_tangent_scaled_root_quotient_post_calculus_presentation::inverse_tangent_scaled_root_quotient_post_calculus_presentation;
use super::sqrt_derivative_post_calculus_presentation::sqrt_derivative_post_calculus_presentation;

pub(super) fn root_and_inverse_family_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) = sqrt_derivative_post_calculus_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_inverse_trig_root_post_calculus_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if inverse_hyperbolic_root_source(target, ctx) {
        if let Some(compact) =
            inverse_hyperbolic_root_post_calculus_presentation(ctx, target, var_name)
        {
            return Some(compact);
        }
    }
    if let Some(compact) = affine_inverse_family_post_calculus_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) = inverse_surd_quotient_post_calculus_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        bounded_inverse_root_quotient_post_calculus_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_tangent_scaled_root_quotient_post_calculus_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = inverse_hyperbolic_root_post_calculus_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }

    inverse_reciprocal_trig_post_calculus_presentation(ctx, target, var_name)
}

fn inverse_hyperbolic_root_source(target: ExprId, ctx: &Context) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    args.len() == 1
        && matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Asinh | BuiltinFn::Atanh | BuiltinFn::Acosh)
        )
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::root_and_inverse_family_post_calculus_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn root_inverse_dispatch_prefers_hyperbolic_root_family_for_hyperbolic_sources() {
        let mut ctx = Context::new();
        let atanh_target = parse("atanh(sqrt(4*x+4)/a)", &mut ctx).unwrap();
        let arctan_target = parse("arctan(sqrt(2*x+2)/(2*a))", &mut ctx).unwrap();

        let atanh_compact =
            root_and_inverse_family_post_calculus_presentation(&mut ctx, atanh_target, "x")
                .unwrap();
        assert_eq!(
            rendered(&ctx, atanh_compact),
            "a / (sqrt(x + 1) * (a^2 - 4 * x - 4))"
        );

        let arctan_compact =
            root_and_inverse_family_post_calculus_presentation(&mut ctx, arctan_target, "x")
                .unwrap();
        assert_eq!(
            rendered(&ctx, arctan_compact),
            "a / (sqrt(2 * x + 2) * (2 * a^2 + x + 1))"
        );
    }
}
