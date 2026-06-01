//! Post-calculus derivative presentation for shifted logarithmic root routes.
//!
//! This module owns narrow logarithmic/root derivative presentation shortcuts.
//! It preserves the call order and domain policy from `calculus/mod.rs`; only
//! the log/root-specific parsers and presentation builders live here.

use super::differentiation::differentiate;
use super::gap_presentation::squared_expr_for_compact_gap_presentation;
use super::polynomial_support::{
    polynomial_is_strictly_positive_everywhere, polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::{
    add_rational_for_calculus_presentation, nonzero_rational_parts,
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
    split_numeric_scale_single_core, sqrt_positive_rational_expr_for_calculus_presentation,
};
use super::shifted_sqrt_args::supported_sqrt_shift_constant_parts;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn sqrt_shifted_ln_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    let (ln_arg, ln_scale, base_ln_factor, shift) =
        scaled_ln_plus_positive_rational_shift(ctx, radicand)?;
    if !ln_scale.is_positive() || !shift.is_positive() {
        return None;
    }

    let arg_poly = Polynomial::from_expr(ctx, ln_arg, var_name).ok()?;
    let derivative_poly = arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = ln_scale * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut denominator_factors = Vec::new();
    if !denominator_coeff.is_one() {
        denominator_factors.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_factors.push(ln_arg);
    if let Some(base_ln_factor) = base_ln_factor {
        denominator_factors.push(base_ln_factor);
    }
    denominator_factors.push(sqrt_radicand);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn ln_sqrt_polynomial_gap_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
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
            if sign == cas_math::expr_nary::Sign::Neg {
                return None;
            }
            if radicand.is_some() {
                return None;
            }
            radicand = Some(term_radicand);
        } else {
            let mut term_poly = polynomial_radicand_for_calculus_presentation(ctx, term, var_name)?;
            if sign == cas_math::expr_nary::Sign::Neg {
                term_poly = term_poly.neg();
            }
            polynomial_term_poly = polynomial_term_poly.add(&term_poly);
        }
    }

    let radicand = radicand?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let positive_gap = radicand_poly.sub(&polynomial_term_poly.mul(&polynomial_term_poly));
    if positive_gap.degree() != 0
        || positive_gap
            .coeffs
            .first()
            .is_none_or(|value| !value.is_positive())
    {
        return None;
    }

    let derivative_poly = polynomial_term_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let numerator = scale_expr_for_calculus_presentation(ctx, derivative_content, derivative_core);
    let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn ln_sqrt_shift_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let (sqrt_arg, shift) = supported_sqrt_shift_constant_parts(ctx, args[0])?;
    let radicand = extract_square_root_base(ctx, sqrt_arg)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let shifted_sqrt = add_rational_for_calculus_presentation(ctx, sqrt_radicand, shift);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, shifted_sqrt]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(
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

    let (sqrt_arg, shift) = supported_sqrt_shift_constant_parts(ctx, args[0])?;
    if !shift.is_positive() {
        return None;
    }
    let radicand = extract_square_root_base(ctx, sqrt_arg)?;
    if polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name).is_some() {
        return None;
    }

    let derivative = differentiate(ctx, radicand, var_name)?;
    let derivative =
        remove_unit_log_e_factor_for_calculus_presentation(ctx, derivative).unwrap_or(derivative);
    if cas_ast::views::as_rational_const(ctx, derivative, 8).is_some_and(|value| value.is_zero()) {
        return Some((ctx.num(0), Vec::new()));
    }
    let (derivative_scale, derivative_core) = split_numeric_scale_single_core(ctx, derivative)
        .unwrap_or((BigRational::one(), derivative));
    let coefficient = derivative_scale * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let shifted_sqrt = add_rational_for_calculus_presentation(ctx, sqrt_radicand, shift);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, shifted_sqrt]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        vec![crate::ImplicitCondition::Positive(radicand)],
    ))
}

pub(super) fn log_root_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) = ln_sqrt_shift_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some((compact, _)) =
        ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        ln_sum_of_equal_derivative_roots_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }

    ln_sqrt_polynomial_gap_derivative_presentation(ctx, target, var_name)
}

pub(super) fn ln_sum_of_equal_derivative_roots_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (compact, _) = ln_sum_of_equal_derivative_roots_derivative_presentation_with_domain(
        ctx, target, var_name,
    )?;
    Some(compact)
}

pub(crate) fn ln_sum_of_equal_derivative_roots_derivative_presentation_with_domain(
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
    if terms.len() != 2 {
        return None;
    }

    let mut radicands = Vec::with_capacity(2);
    for (term, sign) in terms {
        if sign == cas_math::expr_nary::Sign::Neg {
            return None;
        }
        radicands.push(extract_square_root_base(ctx, term)?);
    }

    let left_poly = polynomial_radicand_for_calculus_presentation(ctx, radicands[0], var_name)?;
    let right_poly = polynomial_radicand_for_calculus_presentation(ctx, radicands[1], var_name)?;
    let left_positive_everywhere = polynomial_is_strictly_positive_everywhere(&left_poly);
    let right_positive_everywhere = polynomial_is_strictly_positive_everywhere(&right_poly);
    let affine_pair = left_poly.degree() <= 1 && right_poly.degree() <= 1;
    let strictly_positive_quadratic_pair = left_poly.degree() <= 2
        && right_poly.degree() <= 2
        && left_positive_everywhere
        && right_positive_everywhere;
    if !affine_pair && !strictly_positive_quadratic_pair {
        return None;
    }
    let derivative_poly = left_poly.derivative();
    if derivative_poly != right_poly.derivative() {
        return None;
    }
    if derivative_poly.is_zero() {
        let required_conditions =
            positive_radicand_conditions_for_equal_derivative_roots_presentation(
                radicands[0],
                left_positive_everywhere,
                radicands[1],
                right_positive_everywhere,
            );
        return Some((ctx.num(0), required_conditions));
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let left_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicands[0]]);
    let right_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicands[1]]);
    let left_sqrt = cas_ast::hold::wrap_hold(ctx, left_sqrt);
    let right_sqrt = cas_ast::hold::wrap_hold(ctx, right_sqrt);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[left_sqrt, right_sqrt]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    let compact = ctx.add(Expr::Div(numerator, denominator));
    let required_conditions = positive_radicand_conditions_for_equal_derivative_roots_presentation(
        radicands[0],
        left_positive_everywhere,
        radicands[1],
        right_positive_everywhere,
    );
    Some((cas_ast::hold::wrap_hold(ctx, compact), required_conditions))
}

fn positive_radicand_conditions_for_equal_derivative_roots_presentation(
    left_radicand: ExprId,
    left_positive_everywhere: bool,
    right_radicand: ExprId,
    right_positive_everywhere: bool,
) -> Vec<crate::ImplicitCondition> {
    let mut conditions = Vec::with_capacity(2);
    if !left_positive_everywhere {
        conditions.push(crate::ImplicitCondition::Positive(left_radicand));
    }
    if !right_positive_everywhere {
        conditions.push(crate::ImplicitCondition::Positive(right_radicand));
    }
    conditions
}

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

fn remove_unit_log_e_factor_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    if is_ln_e_for_calculus_presentation(ctx, expr) {
        return Some(ctx.num(1));
    }
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let compact = remove_unit_log_e_factor_for_calculus_presentation(ctx, inner)?;
        return Some(ctx.add(Expr::Neg(compact)));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }
    let retained = factors
        .into_iter()
        .filter(|factor| !is_ln_e_for_calculus_presentation(ctx, *factor))
        .collect::<Vec<_>>();
    if retained.is_empty() {
        return Some(ctx.num(1));
    }
    if retained.len() == cas_math::expr_nary::mul_leaves(ctx, expr).len() {
        return None;
    }
    Some(cas_math::expr_nary::build_balanced_mul(ctx, &retained))
}

fn is_ln_e_for_calculus_presentation(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Ln) => {
            matches!(ctx.get(args[0]), Expr::Constant(Constant::E))
        }
        _ => false,
    }
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

fn scaled_ln_plus_positive_rational_shift(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational, Option<ExprId>, BigRational)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let (left, right) = match ctx.get(expr) {
        Expr::Add(left, right) => (*left, *right),
        _ => return None,
    };

    if let Some((arg, scale, base_ln_factor)) = scaled_ln_term_arg(ctx, left) {
        if let Some(shift) = cas_ast::views::as_rational_const(ctx, right, 8) {
            return Some((arg, scale, base_ln_factor, shift));
        }
    }
    if let Some((arg, scale, base_ln_factor)) = scaled_ln_term_arg(ctx, right) {
        if let Some(shift) = cas_ast::views::as_rational_const(ctx, left, 8) {
            return Some((arg, scale, base_ln_factor, shift));
        }
    }

    None
}

fn scaled_ln_term_arg(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational, Option<ExprId>)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some((arg, base_ln_factor)) = shifted_root_log_term_arg(ctx, expr) {
        return Some((arg, BigRational::one(), base_ln_factor));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let mut scale = BigRational::one();
    let mut ln_arg = None;
    let mut base_ln_factor = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }
        if let Some((arg, factor_base_ln)) = shifted_root_log_term_arg(ctx, factor) {
            if ln_arg.replace(arg).is_none() && base_ln_factor.replace(factor_base_ln).is_none() {
                continue;
            }
        }
        return None;
    }

    Some((ln_arg?, scale, base_ln_factor?))
}

fn shifted_root_log_term_arg(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, Option<ExprId>)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Ln) => Some((args[0], None)),
        Some(BuiltinFn::Log2) => {
            let two = ctx.num(2);
            Some((args[0], Some(ctx.call_builtin(BuiltinFn::Ln, vec![two]))))
        }
        Some(BuiltinFn::Log10) => {
            let ten = ctx.num(10);
            Some((args[0], Some(ctx.call_builtin(BuiltinFn::Ln, vec![ten]))))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::ln_sum_of_equal_derivative_roots_derivative_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn ln_sum_of_equal_derivative_roots_presentation_accepts_scaled_affines() {
        let mut ctx = Context::new();
        let expr = parse("ln(sqrt(2*x+1)+sqrt(2*x+3))", &mut ctx).unwrap();
        let compact = ln_sum_of_equal_derivative_roots_derivative_presentation(&mut ctx, expr, "x")
            .unwrap_or_else(|| {
                panic!("scaled affine equal-derivative root sum should be recognized")
            });

        assert_eq!(
            rendered(&ctx, compact),
            "1 / (sqrt(2 * x + 1) * sqrt(2 * x + 3))"
        );
    }
}
