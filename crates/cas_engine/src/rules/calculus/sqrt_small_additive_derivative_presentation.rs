use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::One;

use super::domain_checks::positive_required_conditions_first;
use super::presentation_compaction::{
    split_signed_numeric_scale_single_core_for_calculus_presentation,
    unary_variable_builtin_arg_for_calculus_presentation,
};
use super::presentation_utils::{
    sqrt_raw_for_calculus_presentation, unwrap_internal_hold_for_calculus,
};
use super::scalar_presentation::scale_expr_for_calculus_presentation;
use super::{
    compact_numeric_mul_factors_for_calculus_presentation,
    polynomial_radicand_for_calculus_presentation,
    scaled_ln_variable_arg_for_calculus_presentation,
    scaled_sqrt_variable_term_for_calculus_presentation,
};

pub(super) fn sqrt_small_additive_elementary_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = extract_square_root_base(ctx, target)?;
    let (radicand_derivative, derivative_denominator, required_conditions) =
        small_additive_elementary_radicand_derivative_for_calculus_presentation(
            ctx, radicand, var_name,
        )?;

    let two = ctx.num(2);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let mut denominator_factors = vec![two];
    if let Some(derivative_denominator) = derivative_denominator {
        denominator_factors.push(derivative_denominator);
    }
    denominator_factors.push(sqrt_radicand);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    let denominator = compact_numeric_mul_factors_for_calculus_presentation(ctx, denominator);
    let compact = ctx.add_raw(Expr::Div(radicand_derivative, denominator));
    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        radicand,
        required_conditions,
    ))
}

pub(super) fn small_additive_elementary_radicand_derivative_for_calculus_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    var_name: &str,
) -> Option<(ExprId, Option<ExprId>, Vec<crate::ImplicitCondition>)> {
    small_additive_elementary_common_derivative_for_calculus_presentation(ctx, radicand, var_name)
}

fn small_additive_elementary_common_derivative_for_calculus_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    var_name: &str,
) -> Option<(ExprId, Option<ExprId>, Vec<crate::ImplicitCondition>)> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, radicand);
    if terms.len() < 2 || terms.len() > 4 {
        return None;
    }

    let var = ctx.var(var_name);
    let two = ctx.num(2);
    let sqrt_var = sqrt_raw_for_calculus_presentation(ctx, var);
    let common_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[two, var, sqrt_var]);
    let common_denominator =
        compact_numeric_mul_factors_for_calculus_presentation(ctx, common_denominator);
    let common_scale = cas_math::expr_nary::build_balanced_mul(ctx, &[two, var, sqrt_var]);

    let mut saw_ln_var = false;
    let mut saw_sqrt_var = false;
    let mut numerator_terms = Vec::new();
    let mut required_conditions = Vec::new();
    for (term, sign) in terms {
        let signed_term = if sign == cas_math::expr_nary::Sign::Neg {
            ctx.add(Expr::Neg(term))
        } else {
            term
        };

        if let Some((scale, arg)) =
            scaled_ln_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if arg != var {
                return None;
            }
            saw_ln_var = true;
            required_conditions.push(crate::ImplicitCondition::Positive(arg));
            let two_sqrt = cas_math::expr_nary::build_balanced_mul(ctx, &[two, sqrt_var]);
            numerator_terms.push(scale_expr_for_calculus_presentation(ctx, scale, two_sqrt));
            continue;
        }

        if let Some((scale, sqrt_arg)) =
            scaled_sqrt_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if sqrt_arg != var {
                return None;
            }
            saw_sqrt_var = true;
            required_conditions.push(crate::ImplicitCondition::Positive(sqrt_arg));
            numerator_terms.push(scale_expr_for_calculus_presentation(ctx, scale, var));
            continue;
        }

        if let Some(derivative) = scaled_exp_trig_variable_term_derivative_for_calculus_presentation(
            ctx,
            signed_term,
            var_name,
        ) {
            let term = ctx.add_raw(Expr::Mul(common_scale, derivative));
            numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
                ctx, term,
            ));
            continue;
        }

        if cas_ast::views::as_rational_const(ctx, signed_term, 8).is_some() {
            continue;
        }

        let poly = polynomial_radicand_for_calculus_presentation(ctx, signed_term, var_name)?;
        if poly.degree() > 3 || poly.coeffs.len() > 5 {
            return None;
        }
        let derivative = poly.derivative();
        if !derivative.is_zero() {
            let derivative = derivative.to_expr(ctx);
            let term = ctx.add_raw(Expr::Mul(common_scale, derivative));
            numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
                ctx, term,
            ));
        }
    }

    if !(saw_ln_var && saw_sqrt_var) || numerator_terms.is_empty() {
        return None;
    }
    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    Some((numerator, Some(common_denominator), required_conditions))
}

fn scaled_exp_trig_variable_term_derivative_for_calculus_presentation(
    ctx: &mut Context,
    term: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, term)
        .unwrap_or((BigRational::one(), term));
    let exp_arg = match ctx.get(core).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Exp) =>
        {
            args[0]
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => exp,
        _ => return None,
    };

    let (trig_derivative, sign) = match ctx.get(exp_arg).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Sin) =>
        {
            let var = unary_variable_builtin_arg_for_calculus_presentation(
                ctx,
                exp_arg,
                var_name,
                BuiltinFn::Sin,
            )?;
            (
                ctx.call_builtin(BuiltinFn::Cos, vec![var]),
                BigRational::one(),
            )
        }
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Cos) =>
        {
            let var = unary_variable_builtin_arg_for_calculus_presentation(
                ctx,
                exp_arg,
                var_name,
                BuiltinFn::Cos,
            )?;
            (
                ctx.call_builtin(BuiltinFn::Sin, vec![var]),
                -BigRational::one(),
            )
        }
        _ => return None,
    };

    let product = cas_math::expr_nary::build_balanced_mul(ctx, &[trig_derivative, core]);
    Some(scale_expr_for_calculus_presentation(
        ctx,
        scale * sign,
        product,
    ))
}

pub(crate) fn sqrt_small_additive_elementary_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, required_conditions) =
        sqrt_small_additive_elementary_derivative_presentation(ctx, target, var_name)?;
    let required_conditions =
        positive_required_conditions_first(required_positive, required_conditions);
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}
