//! Derivative presentation routes for reciprocal trigonometric primitives.
//!
//! This module owns compact `sec/csc` derivative presentation, including
//! affine arguments, scaled powers of `tan/cot`, and shifted sqrt-chain source
//! routes with explicit pole and radicand conditions.

use super::diff_rule_support::diff_rewrite_with_conditions;
use super::polynomial_support::nonzero_affine_variable_derivative;
use super::presentation_utils::{is_calculus_presentation_one, squared_expr};
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use super::shifted_sqrt_args::shifted_sqrt_arg_radicand_and_sign;
use super::signed_factor_presentation::signed_mul_leaves_for_calculus_presentation;
use super::sqrt_chain_factor_presentation::sqrt_chain_linear_derivative_coeff;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};

pub(super) fn reciprocal_trig_shifted_sqrt_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let (result, required_conditions) =
        reciprocal_trig_shifted_sqrt_derivative_presentation(ctx, target, &call.var_name)?;
    Some(diff_rewrite_with_conditions(
        ctx,
        call,
        result,
        required_conditions,
    ))
}

pub(crate) fn reciprocal_trig_shifted_sqrt_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let mut coeff = BigRational::one();
    let mut scale_factors = Vec::new();
    let mut reciprocal_part = None;

    for factor in signed_mul_leaves_for_calculus_presentation(ctx, target) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            coeff *= value;
            continue;
        }

        if let Some((builtin, arg, table_sign, pole_builtin, reciprocal_scale)) =
            shifted_sqrt_reciprocal_trig_factor(ctx, factor)
        {
            if reciprocal_part
                .replace((builtin, arg, table_sign, pole_builtin))
                .is_some()
            {
                return None;
            }
            if let Some(scale_factor) = reciprocal_scale {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, scale_factor, 8) {
                    coeff *= value;
                } else {
                    if contains_named_var(ctx, scale_factor, var_name) {
                        return None;
                    }
                    scale_factors.push(scale_factor);
                }
            }
            continue;
        }

        if contains_named_var(ctx, factor, var_name) {
            return None;
        }
        scale_factors.push(factor);
    }

    let (builtin, arg, table_sign, pole_builtin) = reciprocal_part?;
    let (radicand, orientation) = shifted_sqrt_arg_radicand_and_sign(ctx, arg, var_name)?;
    let chain_coeff = orientation * sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
    coeff *= table_sign * chain_coeff;

    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coeff)?;
    let first = ctx.call_builtin(builtin, vec![arg]);
    let second_builtin = match builtin {
        BuiltinFn::Sec => BuiltinFn::Tan,
        BuiltinFn::Csc => BuiltinFn::Cot,
        _ => return None,
    };
    let second = ctx.call_builtin(second_builtin, vec![arg]);
    let table_core = cas_math::expr_nary::build_balanced_mul(ctx, &[first, second]);
    let table_core = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, table_core);

    let numerator = if scale_factors.is_empty() {
        table_core
    } else {
        scale_factors.push(table_core);
        cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors)
    };

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, sqrt_radicand])
    };
    let result = ctx.add(Expr::Div(numerator, denominator));
    let pole = ctx.call_builtin(pole_builtin, vec![arg]);

    Some((
        cas_ast::hold::wrap_hold(ctx, result),
        vec![
            crate::ImplicitCondition::Positive(radicand),
            crate::ImplicitCondition::NonZero(pole),
        ],
    ))
}

pub(super) fn reciprocal_trig_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (arg, first, second, sign) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() != 1 {
                return None;
            }
            let (first, second, sign) = match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Sec) => (BuiltinFn::Sec, BuiltinFn::Tan, BigRational::one()),
                Some(BuiltinFn::Csc) => (BuiltinFn::Csc, BuiltinFn::Cot, -BigRational::one()),
                _ => return None,
            };
            (args[0], first, second, sign)
        }
        Expr::Div(numerator, denominator)
            if cas_ast::views::as_rational_const(ctx, numerator, 8)
                .is_some_and(|value| value.is_one()) =>
        {
            let Expr::Function(den_fn_id, den_args) = ctx.get(denominator).clone() else {
                return None;
            };
            if den_args.len() != 1 {
                return None;
            }
            let (first, second, sign) = match ctx.builtin_of(den_fn_id) {
                Some(BuiltinFn::Cos) => (BuiltinFn::Sec, BuiltinFn::Tan, BigRational::one()),
                Some(BuiltinFn::Sin) => (BuiltinFn::Csc, BuiltinFn::Cot, -BigRational::one()),
                _ => return None,
            };
            (den_args[0], first, second, sign)
        }
        _ => return None,
    };
    let slope = nonzero_affine_variable_derivative(ctx, arg, var_name)?;
    let coeff = sign * slope;

    let first_arg = ctx.call_builtin(first, vec![arg]);
    let second_arg = ctx.call_builtin(second, vec![arg]);
    let table_core = cas_math::expr_nary::build_balanced_mul(ctx, &[first_arg, second_arg]);
    Some(signed_numerator_for_calculus_presentation(
        ctx, coeff, table_core,
    ))
}

pub(super) fn scaled_reciprocal_trig_power_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(target).clone() else {
        return None;
    };
    let denominator_scale = cas_ast::views::as_rational_const(ctx, denominator, 8)?;
    if denominator_scale.is_zero() {
        return None;
    }

    let (outer_sign, core) = match ctx.get(numerator).clone() {
        Expr::Neg(inner) => (-BigRational::one(), inner),
        _ => (BigRational::one(), numerator),
    };
    let Expr::Pow(base, exp) = ctx.get(core).clone() else {
        return None;
    };
    let power = cas_ast::views::as_rational_const(ctx, exp, 8)?;
    if !power.is_integer() {
        return None;
    }
    let power = power.to_integer().to_i64()?;
    if !(2..=6).contains(&power) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    let (reciprocal_square_builtin, derivative_sign) = match builtin {
        BuiltinFn::Tan => (BuiltinFn::Sec, BigRational::one()),
        BuiltinFn::Cot => (BuiltinFn::Csc, -BigRational::one()),
        _ => return None,
    };
    let arg = args[0];
    let slope = nonzero_affine_variable_derivative(ctx, arg, var_name)?;

    let base_power = if power == 2 {
        ctx.call_builtin(builtin, vec![arg])
    } else {
        let base_call = ctx.call_builtin(builtin, vec![arg]);
        let next_power = ctx.num(power - 1);
        ctx.add(Expr::Pow(base_call, next_power))
    };
    let reciprocal = ctx.call_builtin(reciprocal_square_builtin, vec![arg]);
    let reciprocal_square = squared_expr(ctx, reciprocal);
    let table_core = cas_math::expr_nary::build_balanced_mul(ctx, &[base_power, reciprocal_square]);
    let coeff = outer_sign * derivative_sign * slope * BigRational::from_integer(power.into())
        / denominator_scale;

    Some(signed_numerator_for_calculus_presentation(
        ctx, coeff, table_core,
    ))
}

pub(super) fn direct_reciprocal_trig_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) =
        scaled_reciprocal_trig_power_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }

    reciprocal_trig_affine_derivative_presentation(ctx, target, var_name)
}

fn shifted_sqrt_reciprocal_trig_factor(
    ctx: &Context,
    factor: ExprId,
) -> Option<(BuiltinFn, ExprId, BigRational, BuiltinFn, Option<ExprId>)> {
    match ctx.get(factor) {
        Expr::Function(fn_id, args) if args.len() == 1 => match ctx.builtin_of(*fn_id)? {
            BuiltinFn::Sec => Some((
                BuiltinFn::Sec,
                args[0],
                BigRational::one(),
                BuiltinFn::Cos,
                None,
            )),
            BuiltinFn::Csc => Some((
                BuiltinFn::Csc,
                args[0],
                -BigRational::one(),
                BuiltinFn::Sin,
                None,
            )),
            _ => None,
        },
        Expr::Div(numerator, denominator) => {
            let Expr::Function(fn_id, args) = ctx.get(*denominator) else {
                return None;
            };
            if args.len() != 1 {
                return None;
            }
            let (builtin, table_sign, pole_builtin) = match ctx.builtin_of(*fn_id)? {
                BuiltinFn::Cos => (BuiltinFn::Sec, BigRational::one(), BuiltinFn::Cos),
                BuiltinFn::Sin => (BuiltinFn::Csc, -BigRational::one(), BuiltinFn::Sin),
                _ => return None,
            };
            let scale = (!is_calculus_presentation_one(ctx, *numerator)).then_some(*numerator);
            Some((builtin, args[0], table_sign, pole_builtin, scale))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    fn rendered_required_conditions(ctx: &Context, rewrite: &Rewrite) -> Vec<String> {
        rewrite
            .required_conditions
            .iter()
            .map(|condition| condition.display(ctx))
            .collect()
    }

    #[test]
    fn shifted_sqrt_reciprocal_trig_rewrite_preserves_result_and_conditions() {
        let mut ctx = Context::new();
        let target = parse("sec(sqrt(x)+1)", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };

        let rewrite =
            reciprocal_trig_shifted_sqrt_derivative_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            "sec(sqrt(x) + 1) * tan(sqrt(x) + 1) / (2 * sqrt(x))"
        );
        assert_eq!(
            rendered_required_conditions(&ctx, &rewrite),
            vec!["x > 0", "cos(sqrt(x) + 1) ≠ 0"]
        );
    }
}
