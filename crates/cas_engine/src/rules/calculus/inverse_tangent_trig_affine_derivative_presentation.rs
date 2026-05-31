use super::scalar_presentation::signed_numerator_for_calculus_presentation;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Zero};

pub(super) fn inverse_tangent_direct_trig_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let outer_sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Atan | BuiltinFn::Arctan) => BigRational::one(),
        Some(BuiltinFn::Acot | BuiltinFn::Arccot) => -BigRational::one(),
        _ => return None,
    };

    let trig_arg = args[0];
    let Expr::Function(trig_fn_id, trig_args) = ctx.get(trig_arg).clone() else {
        return None;
    };
    if trig_args.len() != 1
        || !matches!(
            ctx.builtin_of(trig_fn_id),
            Some(BuiltinFn::Sin | BuiltinFn::Cos)
        )
    {
        return None;
    }
    let inner_poly = Polynomial::from_expr(ctx, trig_args[0], var_name).ok()?;
    if inner_poly.degree() != 1 {
        return None;
    }

    let inner_derivative = inner_poly.derivative().to_expr(ctx);
    let mut scale = cas_ast::views::as_rational_const(ctx, inner_derivative, 8)?;
    if scale.is_zero() {
        return None;
    }
    scale *= outer_sign;
    let numerator_core = match ctx.builtin_of(trig_fn_id) {
        Some(BuiltinFn::Sin) => ctx.call_builtin(BuiltinFn::Cos, vec![trig_args[0]]),
        Some(BuiltinFn::Cos) => {
            scale = -scale;
            ctx.call_builtin(BuiltinFn::Sin, vec![trig_args[0]])
        }
        _ => return None,
    };
    let numerator = signed_numerator_for_calculus_presentation(ctx, scale, numerator_core);

    let two = ctx.num(2);
    let trig_square = ctx.add(Expr::Pow(trig_arg, two));
    let one = ctx.num(1);
    let denominator = ctx.add(Expr::Add(trig_square, one));
    let compact = ctx.add(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}
