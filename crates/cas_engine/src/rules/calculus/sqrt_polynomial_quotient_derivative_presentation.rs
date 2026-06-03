//! Compact derivative presentation for square-root polynomial quotient routes.
//!
//! This module owns the presentation and domain wrappers for
//! `sqrt(p(x)) / q(x)` and `sqrt(p(x) / q(x))`. It deliberately keeps the
//! shared polynomial-power helpers live in a small presentation-policy module
//! because sibling extracted families use the same denominator-power policy.

use super::diff_rule_support::finalize_diff_rewrite_with_conditions;
use super::polynomial_power_presentation::{
    expanded_affine_square_for_calculus_presentation, polynomial_power_for_calculus_presentation,
    positive_integer_polynomial_power_for_calculus_presentation,
};
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::calculus_sqrt_like_radicand;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use super::scaled_sqrt_args::scaled_square_root_radicand_for_calculus_presentation;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

pub(super) fn sqrt_polynomial_quotient_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(target).clone() else {
        return None;
    };
    let (numerator_scale, radicand) =
        scaled_square_root_radicand_for_calculus_presentation(ctx, numerator_expr)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let denominator_poly =
        polynomial_radicand_for_calculus_presentation(ctx, denominator_expr, var_name)?;
    if denominator_poly.is_zero() {
        return None;
    }

    let radicand_derivative = radicand_poly.derivative();
    let denominator_derivative = denominator_poly.derivative();
    let mut numerator_poly = denominator_poly.mul(&radicand_derivative).sub(
        &radicand_poly
            .mul(&denominator_derivative)
            .mul(&cas_math::polynomial::Polynomial::new(
                vec![BigRational::from_integer(2.into())],
                var_name.to_string(),
            )),
    );
    if numerator_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let mut presentation_denominator_power = {
        let two = ctx.num(2);
        ctx.add(Expr::Pow(denominator_expr, two))
    };
    let denominator_power_factor = positive_integer_polynomial_power_for_calculus_presentation(
        ctx,
        denominator_expr,
        var_name,
    )
    .or_else(|| expanded_affine_square_for_calculus_presentation(ctx, denominator_expr, var_name));
    if let Some((base, exponent, base_poly)) = denominator_power_factor {
        if exponent > 1 {
            let cancellable = polynomial_power_for_calculus_presentation(&base_poly, exponent - 1);
            if let Ok((quotient, remainder)) = numerator_poly.div_rem(&cancellable) {
                if remainder.is_zero() {
                    numerator_poly = quotient;
                    let denominator_exponent = ctx.num((exponent + 1) as i64);
                    presentation_denominator_power = ctx.add(Expr::Pow(base, denominator_exponent));
                }
            }
        }
    } else if let Ok((quotient, remainder)) = numerator_poly.div_rem(&denominator_poly) {
        if remainder.is_zero() {
            numerator_poly = quotient;
            presentation_denominator_power = denominator_expr;
        }
    }

    let raw_numerator = numerator_poly.to_expr(ctx);
    let (numerator_core, numerator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_numerator);
    let coefficient = numerator_scale * numerator_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[sqrt_radicand, presentation_denominator_power],
    );
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn sqrt_of_polynomial_quotient_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let quotient = calculus_sqrt_like_radicand(ctx, target)?;
    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(quotient).clone() else {
        return None;
    };
    let numerator_poly =
        polynomial_radicand_for_calculus_presentation(ctx, numerator_expr, var_name)?;
    let denominator_poly =
        polynomial_radicand_for_calculus_presentation(ctx, denominator_expr, var_name)?;
    if denominator_poly.is_zero() {
        return None;
    }

    let mut numerator_result_poly = denominator_poly
        .mul(&numerator_poly.derivative())
        .sub(&numerator_poly.mul(&denominator_poly.derivative()));
    if numerator_result_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let mut presentation_denominator_power = {
        let two = ctx.num(2);
        ctx.add(Expr::Pow(denominator_expr, two))
    };
    let denominator_power_factor = positive_integer_polynomial_power_for_calculus_presentation(
        ctx,
        denominator_expr,
        var_name,
    )
    .or_else(|| expanded_affine_square_for_calculus_presentation(ctx, denominator_expr, var_name));
    if let Some((base, exponent, base_poly)) = denominator_power_factor {
        let mut remaining_exponent = 2 * exponent;
        while remaining_exponent > 0 {
            let Ok((quotient_poly, remainder)) = numerator_result_poly.div_rem(&base_poly) else {
                break;
            };
            if !remainder.is_zero() {
                break;
            }
            numerator_result_poly = quotient_poly;
            remaining_exponent -= 1;
        }
        presentation_denominator_power = match remaining_exponent {
            0 => ctx.num(1),
            1 => base,
            _ => {
                let exponent = ctx.num(remaining_exponent as i64);
                ctx.add(Expr::Pow(base, exponent))
            }
        };
    } else if let Ok((quotient_poly, remainder)) = numerator_result_poly.div_rem(&denominator_poly)
    {
        if remainder.is_zero() {
            numerator_result_poly = quotient_poly;
            presentation_denominator_power = denominator_expr;
        }
    }

    let raw_numerator = numerator_result_poly.to_expr(ctx);
    let (numerator_core, numerator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_numerator);
    let coefficient = numerator_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_quotient = ctx.call_builtin(BuiltinFn::Sqrt, vec![quotient]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[sqrt_quotient, presentation_denominator_power],
    );
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn sqrt_of_polynomial_quotient_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let (result, required_conditions) =
        sqrt_of_polynomial_quotient_derivative_presentation_with_domain(
            ctx,
            target,
            &call.var_name,
        )?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        required_conditions,
    ))
}

pub(crate) fn sqrt_polynomial_quotient_has_powered_expanded_affine_square_denominator(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(target).clone() else {
        return false;
    };
    if calculus_sqrt_like_radicand(ctx, numerator_expr).is_none() {
        return false;
    }

    let denominator_expr = cas_ast::hold::unwrap_internal_hold(ctx, denominator_expr);
    let Expr::Pow(base, exp) = ctx.get(denominator_expr).clone() else {
        return false;
    };
    let Some(exponent) = cas_ast::views::as_rational_const(ctx, exp, 8) else {
        return false;
    };
    if !exponent.is_integer() || exponent < BigRational::from_integer(2.into()) {
        return false;
    }

    expanded_affine_square_for_calculus_presentation(ctx, base, var_name).is_some()
}

pub(crate) fn sqrt_polynomial_quotient_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Div(numerator_expr, _) = ctx.get(target).clone() else {
        return None;
    };
    let (_, radicand) = scaled_square_root_radicand_for_calculus_presentation(ctx, numerator_expr)?;
    let result = sqrt_polynomial_quotient_derivative_presentation(ctx, target, var_name)?;
    Some((result, vec![crate::ImplicitCondition::Positive(radicand)]))
}

pub(crate) fn sqrt_of_polynomial_quotient_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let quotient = calculus_sqrt_like_radicand(ctx, target)?;
    let result = sqrt_of_polynomial_quotient_derivative_presentation(ctx, target, var_name)?;
    Some((result, vec![crate::ImplicitCondition::Positive(quotient)]))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::sqrt_of_polynomial_quotient_derivative_rewrite;
    use crate::symbolic_calculus_call_support::NamedVarCall;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn sqrt_of_polynomial_quotient_rewrite_preserves_finalized_conditions() {
        let mut ctx = Context::new();
        let target = parse("sqrt((x+1)/(x+2))", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let rewrite =
            sqrt_of_polynomial_quotient_derivative_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            "1 / (2 * sqrt((x + 1) / (x + 2)) * (x + 2)^2)"
        );
        assert_eq!(rewrite.required_conditions.len(), 1);
        assert_eq!(
            rewrite.required_conditions[0].display(&ctx),
            "(x + 1) / (x + 2) > 0"
        );
    }
}
