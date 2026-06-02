//! Direct derivative presentation for `ln(sqrt(polynomial) + polynomial)`.
//!
//! This module owns the polynomial direct log-root route and the companion
//! negative-gap detector used by post-calculus sqrt/log presentation.

use super::gap_presentation::squared_expr_for_compact_gap_presentation;
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::{
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
    sqrt_positive_rational_expr_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

pub(super) fn ln_sqrt_negative_polynomial_gap_target(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return false;
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return false;
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, args[0]);
    let mut radicand = None;
    let mut polynomial_term_poly = Polynomial::zero(var_name.to_string());
    for (term, sign) in terms {
        if let Some(term_radicand) = extract_square_root_base(ctx, term) {
            if sign == cas_math::expr_nary::Sign::Neg || radicand.replace(term_radicand).is_some() {
                return false;
            }
            continue;
        }

        let Some(mut term_poly) =
            polynomial_radicand_for_calculus_presentation(ctx, term, var_name)
        else {
            return false;
        };
        if sign == cas_math::expr_nary::Sign::Neg {
            term_poly = term_poly.neg();
        }
        polynomial_term_poly = polynomial_term_poly.add(&term_poly);
    }

    let Some(radicand) = radicand else {
        return false;
    };
    if polynomial_term_poly.is_zero() || polynomial_term_poly.derivative().is_zero() {
        return false;
    }
    let Some(radicand_poly) =
        polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)
    else {
        return false;
    };
    let square_gap = polynomial_term_poly
        .mul(&polynomial_term_poly)
        .sub(&radicand_poly);
    square_gap.degree() == 0
        && square_gap
            .coeffs
            .first()
            .is_some_and(|value| value.is_positive())
}

pub(super) fn ln_sqrt_plus_polynomial_direct_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, args[0]);
    let mut radicand = None;
    let mut polynomial_term_poly = Polynomial::zero(var_name.to_string());
    for (term, sign) in terms {
        if let Some(term_radicand) = extract_square_root_base(ctx, term) {
            if sign == cas_math::expr_nary::Sign::Neg || radicand.replace(term_radicand).is_some() {
                return None;
            }
            continue;
        }

        let mut term_poly = polynomial_radicand_for_calculus_presentation(ctx, term, var_name)?;
        if sign == cas_math::expr_nary::Sign::Neg {
            term_poly = term_poly.neg();
        }
        polynomial_term_poly = polynomial_term_poly.add(&term_poly);
    }

    let radicand = radicand?;
    if polynomial_term_poly.is_zero() {
        return None;
    }
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let radicand_derivative_poly = radicand_poly.derivative();
    let polynomial_derivative_poly = polynomial_term_poly.derivative();
    if radicand_derivative_poly.is_zero() && polynomial_derivative_poly.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let square_gap = polynomial_term_poly
        .mul(&polynomial_term_poly)
        .sub(&radicand_poly);
    if square_gap.degree() == 0 {
        if let Some(gap_value) = square_gap.coeffs.first() {
            if gap_value.is_positive() {
                let polynomial_derivative_poly = polynomial_term_poly.derivative();
                if polynomial_derivative_poly.is_zero() {
                    return Some((ctx.num(0), Vec::new()));
                }

                let derivative = polynomial_derivative_poly.to_expr(ctx);
                let (derivative_core, derivative_content) =
                    split_polynomial_content_for_calculus_presentation(ctx, derivative);
                let numerator =
                    scale_expr_for_calculus_presentation(ctx, derivative_content, derivative_core);
                let square_arg_poly = if polynomial_term_poly.leading_coeff().is_negative() {
                    polynomial_term_poly.neg()
                } else {
                    polynomial_term_poly.clone()
                };
                let polynomial_term = polynomial_term_poly.to_expr(ctx);
                let square_arg = square_arg_poly.to_expr(ctx);
                let polynomial_term_sq = squared_expr_for_compact_gap_presentation(ctx, square_arg);
                let gap_expr = rational_const_for_calculus_presentation(ctx, gap_value.clone());
                let compact_radicand = ctx.add(Expr::Sub(polynomial_term_sq, gap_expr));
                let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![compact_radicand]);
                let denominator = cas_ast::hold::wrap_hold(ctx, denominator);
                let compact = ctx.add(Expr::Div(numerator, denominator));

                let branch_boundary =
                    sqrt_positive_rational_expr_for_calculus_presentation(ctx, gap_value.clone());
                let branch_gap = ctx.add(Expr::Sub(polynomial_term, branch_boundary));
                return Some((
                    ctx.add(Expr::Hold(compact)),
                    vec![crate::ImplicitCondition::Positive(branch_gap)],
                ));
            }
        }
    }

    if let Some(scale) = nonzero_polynomial_scale_factor(&polynomial_term_poly, &radicand_poly) {
        let required_conditions = if scale.is_negative() {
            let scale_square = &scale * &scale;
            let scale_square_expr = rational_const_for_calculus_presentation(ctx, scale_square);
            let scaled_radicand = ctx.add(Expr::Mul(scale_square_expr, radicand));
            let one = ctx.num(1);
            let upper_boundary = ctx.add(Expr::Sub(one, scaled_radicand));
            vec![
                crate::ImplicitCondition::Positive(radicand),
                crate::ImplicitCondition::Positive(upper_boundary),
            ]
        } else {
            Vec::new()
        };
        let radicand_derivative = radicand_derivative_poly.to_expr(ctx);
        let scaled_sqrt = scale_expr_for_calculus_presentation(ctx, scale.clone(), sqrt_radicand);
        let one = ctx.num(1);
        let denominator_tail = ctx.add(Expr::Add(one, scaled_sqrt));
        let leading = ctx.add(Expr::Div(radicand_derivative, radicand));
        let two = ctx.num(2);
        let correction_denominator =
            cas_math::expr_nary::build_balanced_mul(ctx, &[two, radicand, denominator_tail]);
        let correction = ctx.add(Expr::Div(radicand_derivative, correction_denominator));
        let compact = ctx.add(Expr::Sub(leading, correction));
        return Some((cas_ast::hold::wrap_hold(ctx, compact), required_conditions));
    }

    None
}

fn nonzero_polynomial_scale_factor(scaled: &Polynomial, base: &Polynomial) -> Option<BigRational> {
    if base.is_zero() || scaled.is_zero() || scaled.var != base.var {
        return None;
    }
    let max_len = scaled.coeffs.len().max(base.coeffs.len());
    let mut scale = None;
    for index in 0..max_len {
        let scaled_coeff = scaled
            .coeffs
            .get(index)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let base_coeff = base
            .coeffs
            .get(index)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if base_coeff.is_zero() {
            if !scaled_coeff.is_zero() {
                return None;
            }
            continue;
        }
        let local_scale = scaled_coeff / base_coeff;
        if let Some(existing) = &scale {
            if existing != &local_scale {
                return None;
            }
        } else {
            scale = Some(local_scale);
        }
    }

    scale.filter(|value| !value.is_zero())
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::ln_sqrt_plus_polynomial_direct_derivative_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn ln_sqrt_affine_gap_derivative_keeps_compact_radicand() {
        let mut ctx = Context::new();
        let expr = parse("ln(sqrt((2*x+1)^2-4)+(2*x+1))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            ln_sqrt_plus_polynomial_direct_derivative_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "2 / sqrt((2 * x + 1)^2 - 4)");
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(required_conditions[0].display(&ctx), "x > 1/2");

        let expr = parse("ln(sqrt((2*x+1)^2-4)-(2*x+1))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            ln_sqrt_plus_polynomial_direct_derivative_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "-2 / sqrt((2 * x + 1)^2 - 4)");
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(required_conditions[0].display(&ctx), "x < -3/2");
    }
}
