use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::{render_diff_desc_with, NamedVarCall};
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::polynomial::Polynomial;

use super::arctan_sqrt_additive_derivative_presentation::{
    arctan_sqrt_additive_tan_polynomial_derivative_presentation,
    arctan_sqrt_additive_trig_polynomial_derivative_presentation,
    arctan_sqrt_small_additive_elementary_derivative_presentation,
};
use super::domain_checks::{
    append_positive_required_conditions, reciprocal_trig_and_log_diff_required_conditions,
};
use super::sqrt_small_additive_derivative_presentation::sqrt_small_additive_elementary_derivative_presentation;
use super::{
    sqrt_additive_tan_polynomial_derivative_presentation,
    sqrt_additive_trig_polynomial_derivative_presentation,
};

pub(super) fn sign_polynomial_diff_result(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(BuiltinFn::Sign) {
        return None;
    }

    let arg = cas_ast::hold::unwrap_internal_hold(ctx, args[0]);
    if !contains_named_var(ctx, arg, var_name) {
        return None;
    }

    let polynomial = Polynomial::from_expr(ctx, arg, var_name).ok()?;
    let zero = ctx.num(0);
    if polynomial.is_zero() || polynomial.degree() == 0 {
        return Some((zero, Vec::new()));
    }

    Some((zero, vec![crate::ImplicitCondition::NonZero(arg)]))
}

pub(super) fn undefined_diff_rewrite(ctx: &mut Context, call: &NamedVarCall) -> Rewrite {
    let undefined = ctx.add(Expr::Constant(Constant::Undefined));
    diff_rewrite_with_conditions(
        ctx,
        call,
        undefined,
        std::iter::empty::<crate::ImplicitCondition>(),
    )
}

pub(super) fn diff_rewrite_with_conditions<I>(
    ctx: &mut Context,
    call: &NamedVarCall,
    result: ExprId,
    required_conditions: I,
) -> Rewrite
where
    I: IntoIterator<Item = crate::ImplicitCondition>,
{
    let desc = render_diff_desc_with(call, |id| {
        format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
    });
    Rewrite::new(result)
        .desc(desc)
        .requires_all(required_conditions)
}

pub(super) fn arctan_sqrt_additive_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    if let Some((result, required_positive, required_conditions)) =
        arctan_sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        let mut shortcut_required_conditions = Vec::new();
        append_positive_required_conditions(
            &mut shortcut_required_conditions,
            required_positive,
            required_conditions,
        );
        return Some(diff_rewrite_with_conditions(
            ctx,
            call,
            result,
            shortcut_required_conditions,
        ));
    }

    if let Some((result, required_positive, required_conditions)) =
        arctan_sqrt_additive_tan_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        let mut shortcut_required_conditions = Vec::new();
        append_positive_required_conditions(
            &mut shortcut_required_conditions,
            required_positive,
            required_conditions,
        );
        let required_conditions =
            reciprocal_trig_and_log_diff_required_conditions(ctx, target, &call.var_name)
                .into_iter()
                .chain(shortcut_required_conditions);
        return Some(diff_rewrite_with_conditions(
            ctx,
            call,
            result,
            required_conditions,
        ));
    }

    if let Some((result, required_positive, required_conditions)) =
        arctan_sqrt_small_additive_elementary_derivative_presentation(ctx, target, &call.var_name)
    {
        let mut shortcut_required_conditions = Vec::new();
        append_positive_required_conditions(
            &mut shortcut_required_conditions,
            required_positive,
            required_conditions,
        );
        return Some(diff_rewrite_with_conditions(
            ctx,
            call,
            result,
            shortcut_required_conditions,
        ));
    }

    None
}

pub(super) fn sqrt_additive_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    shortcut_required_conditions: &mut Vec<crate::ImplicitCondition>,
) -> Option<ExprId> {
    if let Some((result, required_positive, required_conditions)) =
        sqrt_additive_tan_polynomial_derivative_presentation(ctx, target, var_name)
    {
        append_positive_required_conditions(
            shortcut_required_conditions,
            required_positive,
            required_conditions,
        );
        return Some(result);
    }

    if let Some((result, required_positive, required_conditions)) =
        sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, var_name)
    {
        append_positive_required_conditions(
            shortcut_required_conditions,
            required_positive,
            required_conditions,
        );
        return Some(result);
    }

    let (result, required_positive, required_conditions) =
        sqrt_small_additive_elementary_derivative_presentation(ctx, target, var_name)?;
    append_positive_required_conditions(
        shortcut_required_conditions,
        required_positive,
        required_conditions,
    );
    Some(result)
}
