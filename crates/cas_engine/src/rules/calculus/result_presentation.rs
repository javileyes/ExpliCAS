use super::fractional_denominator_power_integrand_preservation::fractional_denominator_power_substitution_integrand_for_calculus_presentation;
use super::integration::integrate_required_positive_conditions;
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::power_result_presentation::{
    compact_half_power_sum_root_product_for_integration_presentation,
    compact_negative_half_power_product_for_calculus_presentation,
    compact_negative_half_power_result_for_integration_presentation,
    compact_negative_three_half_power_result_for_integration_presentation,
    compact_positive_half_power_result_for_integration_presentation,
};
use super::presentation_utils::{
    calculus_sqrt_like_radicand, same_sqrt_like_argument, sqrt_raw_for_calculus_presentation,
    unwrap_internal_hold_for_calculus, variable_named,
};
pub(super) use super::rationalized_sqrt_result_presentation::compact_acosh_surd_width_arg_for_integration_presentation;
use super::rationalized_sqrt_result_presentation::compact_rationalized_sqrt_denominator_quotient_for_calculus_presentation;
use super::scalar_presentation::{
    fold_numeric_mul_constants_for_hold, negate_calculus_presentation, nonzero_rational_parts,
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
    scale_fraction_for_calculus_presentation, signed_rational_const_for_calculus_presentation,
};
use super::sqrt_hyperbolic_log_integrand_presentation::compact_direct_sqrt_hyperbolic_log_derivative_integrand;
use super::sqrt_trig_log_integrand_presentation::compact_sqrt_trig_log_derivative_integrand;
use super::trig_result_presentation::{
    compact_trig_odd_power_reduction_primitive_for_integration_presentation,
    compact_trig_square_reduction_primitive_for_integration_presentation,
};
use crate::symbolic_calculus_call_support::{try_extract_integrate_call, NamedVarCall};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::polynomial::Polynomial;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn compact_division_by_positive_denominator_content_for_calculus_presentation(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<ExprId> {
    let numerator_value = signed_rational_const_for_calculus_presentation(ctx, num)?;
    let (den_core, den_content) = split_polynomial_content_for_calculus_presentation(ctx, den);
    if !den_content.is_positive() || den_content.is_one() || den_core == den {
        return None;
    }

    let scaled_numerator = numerator_value / den_content;
    let numerator = rational_const_for_calculus_presentation(
        ctx,
        BigRational::from_integer(scaled_numerator.numer().clone()),
    );
    if scaled_numerator.denom().is_one() {
        return Some(ctx.add(Expr::Div(numerator, den_core)));
    }

    let denominator_scale = BigRational::from_integer(scaled_numerator.denom().clone());
    let denominator = scale_expr_for_calculus_presentation(ctx, denominator_scale, den_core);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn cancel_denominator_content_with_numerator_for_calculus_presentation(
    ctx: &mut Context,
    numerator_coeff: BigRational,
    denominator_coeff: BigRational,
    denominator_factor: ExprId,
) -> (BigRational, BigRational, ExprId) {
    let (primitive_factor, factor_content) =
        split_polynomial_content_for_calculus_presentation(ctx, denominator_factor);
    if factor_content.is_zero() || factor_content.is_one() {
        return (numerator_coeff, denominator_coeff, denominator_factor);
    }

    let adjusted_numerator = numerator_coeff.clone() / factor_content.clone();
    if !adjusted_numerator.is_integer() {
        let adjusted_denominator = denominator_coeff * factor_content;
        let (numerator_coeff, denominator_coeff) =
            nonzero_rational_parts(&(numerator_coeff / adjusted_denominator))
                .unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
        return (numerator_coeff, denominator_coeff, primitive_factor);
    }

    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(adjusted_numerator / denominator_coeff))
            .unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
    (numerator_coeff, denominator_coeff, primitive_factor)
}

pub(super) fn cancel_positive_denominator_content_with_numerator_for_calculus_presentation(
    ctx: &mut Context,
    numerator_coeff: BigRational,
    denominator_coeff: BigRational,
    denominator_factor: ExprId,
) -> (BigRational, BigRational, ExprId) {
    let (primitive_factor, factor_content) =
        split_polynomial_content_for_calculus_presentation(ctx, denominator_factor);
    if factor_content.is_zero() || !factor_content.is_positive() || factor_content.is_one() {
        return (numerator_coeff, denominator_coeff, denominator_factor);
    }

    let adjusted_numerator = numerator_coeff.clone() / factor_content.clone();
    if !adjusted_numerator.is_integer() {
        let adjusted_denominator = denominator_coeff * factor_content;
        let (numerator_coeff, denominator_coeff) =
            nonzero_rational_parts(&(numerator_coeff / adjusted_denominator))
                .unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
        return (numerator_coeff, denominator_coeff, primitive_factor);
    }

    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(adjusted_numerator / denominator_coeff))
            .unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
    (numerator_coeff, denominator_coeff, primitive_factor)
}

pub(super) fn remove_unit_mul_factors_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> ExprId {
    let Expr::Mul(_, _) = ctx.get(expr) else {
        return expr;
    };

    let mut non_unit_factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if cas_ast::views::as_rational_const(ctx, factor, 8) == Some(BigRational::one()) {
            continue;
        }
        non_unit_factors.push(factor);
    }

    match non_unit_factors.as_slice() {
        [single] => *single,
        _ => expr,
    }
}

pub(super) fn scale_compact_derivative_by_rational(
    ctx: &mut Context,
    derivative: ExprId,
    scale: BigRational,
) -> ExprId {
    if scale.is_one() {
        return derivative;
    }

    let derivative = unwrap_internal_hold_for_calculus(ctx, derivative);
    let scaled = if let Expr::Div(num, den) = ctx.get(derivative).clone() {
        let (num, den) = scale_fraction_for_calculus_presentation(ctx, num, den, scale);
        if let Some(compact) =
            compact_division_by_positive_denominator_content_for_calculus_presentation(
                ctx, num, den,
            )
        {
            return compact;
        }
        ctx.add(Expr::Div(num, den))
    } else {
        scale_expr_for_calculus_presentation(ctx, scale, derivative)
    };

    fold_numeric_mul_constants_for_hold(ctx, scaled)
}

pub(super) fn reciprocal_constant_denominator_for_calculus_presentation(
    ctx: &mut Context,
    factor: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if contains_named_var(ctx, factor, var_name) {
        return None;
    }

    match ctx.get(factor).clone() {
        Expr::Number(value) if value.numer() == &BigInt::from(1) && !value.is_zero() => {
            Some(ctx.add(Expr::Number(BigRational::from_integer(
                value.denom().clone(),
            ))))
        }
        Expr::Div(numerator, denominator) => {
            let numerator_value = cas_ast::views::as_rational_const(ctx, numerator, 8)?;
            if numerator_value == BigRational::one() {
                Some(denominator)
            } else {
                None
            }
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, exp, 8)
                == Some(BigRational::new((-1).into(), 1.into())) =>
        {
            Some(base)
        }
        _ => None,
    }
}

fn arctan_arg_matches_for_calculus_presentation(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    if compare_expr(ctx, left, right) == std::cmp::Ordering::Equal {
        return true;
    }

    let Ok(left_poly) = Polynomial::from_expr(ctx, left, var_name) else {
        return false;
    };
    let Ok(right_poly) = Polynomial::from_expr(ctx, right, var_name) else {
        return false;
    };
    left_poly == right_poly
}

fn extract_arctan_term_for_calculus_presentation(
    ctx: &mut Context,
    term: ExprId,
) -> Option<(ExprId, ExprId)> {
    let term = unwrap_internal_hold_for_calculus(ctx, term);
    if let Expr::Neg(inner) = ctx.get(term).clone() {
        let (arg, coeff) = extract_arctan_term_for_calculus_presentation(ctx, inner)?;
        let coeff = ctx.add(Expr::Neg(coeff));
        return Some((arg, coeff));
    }
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let (arg, coeff) = extract_arctan_term_for_calculus_presentation(ctx, num)?;
        let coeff = ctx.add(Expr::Div(coeff, den));
        return Some((arg, coeff));
    }

    let factors = cas_math::expr_nary::MulView::from_expr(ctx, term).factors;
    let mut arctan_arg = None;
    let mut coefficient_factors = Vec::new();

    for factor in factors {
        let factor = unwrap_internal_hold_for_calculus(ctx, factor);
        match ctx.get(factor).clone() {
            Expr::Function(fn_id, args)
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(fn_id),
                        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                    ) =>
            {
                if arctan_arg.is_some() {
                    return None;
                }
                arctan_arg = Some(args[0]);
            }
            Expr::Div(num, den) => {
                let Expr::Function(fn_id, args) = ctx.get(num).clone() else {
                    coefficient_factors.push(factor);
                    continue;
                };
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(fn_id),
                        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                    )
                {
                    if arctan_arg.is_some() {
                        return None;
                    }
                    arctan_arg = Some(args[0]);
                    let one = ctx.num(1);
                    coefficient_factors.push(ctx.add(Expr::Div(one, den)));
                } else {
                    coefficient_factors.push(factor);
                }
            }
            Expr::Neg(inner) => {
                let inner = unwrap_internal_hold_for_calculus(ctx, inner);
                let Expr::Function(fn_id, args) = ctx.get(inner).clone() else {
                    coefficient_factors.push(factor);
                    continue;
                };
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(fn_id),
                        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                    )
                {
                    if arctan_arg.is_some() {
                        return None;
                    }
                    arctan_arg = Some(args[0]);
                    let minus_one = ctx.num(-1);
                    coefficient_factors.push(minus_one);
                } else {
                    coefficient_factors.push(factor);
                }
            }
            _ => coefficient_factors.push(factor),
        }
    }

    let arg = arctan_arg?;
    let coeff = if coefficient_factors.is_empty() {
        ctx.num(1)
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &coefficient_factors)
    };
    Some((arg, coeff))
}

fn negate_term_for_calculus_presentation(ctx: &mut Context, term: ExprId) -> ExprId {
    let term = unwrap_internal_hold_for_calculus(ctx, term);
    if let Expr::Number(value) = ctx.get(term).clone() {
        return ctx.add(Expr::Number(-value));
    }
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let num = negate_term_for_calculus_presentation(ctx, num);
        return ctx.add(Expr::Div(num, den));
    }

    let factors = cas_math::expr_nary::MulView::from_expr(ctx, term).factors;
    if factors.len() > 1 {
        let mut replaced = false;
        let mut negated_factors = Vec::with_capacity(factors.len());
        for factor in factors {
            if !replaced {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    negated_factors.push(ctx.add(Expr::Number(-value)));
                    replaced = true;
                    continue;
                }
            }
            negated_factors.push(factor);
        }
        if replaced {
            return cas_math::expr_nary::build_balanced_mul(ctx, &negated_factors);
        }
    }

    ctx.add(Expr::Neg(term))
}

struct LnTermForCalculusPresentation {
    arg: ExprId,
    arg_poly: Polynomial,
    coefficient: BigRational,
}

fn extract_ln_term_for_calculus_presentation(
    ctx: &mut Context,
    term: ExprId,
    var_name: &str,
) -> Option<LnTermForCalculusPresentation> {
    let term = unwrap_internal_hold_for_calculus(ctx, term);
    if let Expr::Neg(inner) = ctx.get(term).clone() {
        let mut extracted = extract_ln_term_for_calculus_presentation(ctx, inner, var_name)?;
        extracted.coefficient = -extracted.coefficient;
        return Some(extracted);
    }
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let denominator = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if denominator.is_zero() {
            return None;
        }
        let mut extracted = extract_ln_term_for_calculus_presentation(ctx, num, var_name)?;
        extracted.coefficient /= denominator;
        return Some(extracted);
    }

    let mut ln_arg = None;
    let mut coefficient = BigRational::one();
    for factor in cas_math::expr_nary::MulView::from_expr(ctx, term).factors {
        let factor = unwrap_internal_hold_for_calculus(ctx, factor);
        if let Expr::Function(fn_id, args) = ctx.get(factor).clone() {
            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) && args.len() == 1 {
                if ln_arg.replace(args[0]).is_some() {
                    return None;
                }
                continue;
            }
        }

        let factor_value = cas_ast::views::as_rational_const(ctx, factor, 8)?;
        coefficient *= factor_value;
    }

    let arg = ln_arg?;
    Some(LnTermForCalculusPresentation {
        arg,
        arg_poly: Polynomial::from_expr(ctx, arg, var_name).ok()?,
        coefficient,
    })
}

fn build_scaled_ln_for_calculus_presentation(
    ctx: &mut Context,
    coefficient: &BigRational,
    arg: ExprId,
) -> Option<ExprId> {
    if coefficient.is_zero() {
        return None;
    }

    let ln = ctx.call_builtin(BuiltinFn::Ln, vec![arg]);
    if coefficient.is_one() {
        return Some(ln);
    }
    if *coefficient == -BigRational::one() {
        return Some(ctx.add(Expr::Neg(ln)));
    }

    let coefficient = ctx.add(Expr::Number(coefficient.clone()));
    Some(ctx.add(Expr::Mul(coefficient, ln)))
}

fn ln_polynomial_coefficient_degree_for_calculus_presentation(
    ctx: &mut Context,
    term: ExprId,
    var_name: &str,
) -> Option<usize> {
    let term = unwrap_internal_hold_for_calculus(ctx, term);
    if let Expr::Neg(inner) = ctx.get(term).clone() {
        return ln_polynomial_coefficient_degree_for_calculus_presentation(ctx, inner, var_name);
    }

    let mut ln_seen = false;
    let mut coefficient_factors = Vec::new();
    for factor in cas_math::expr_nary::MulView::from_expr(ctx, term).factors {
        let factor = unwrap_internal_hold_for_calculus(ctx, factor);
        if let Expr::Function(fn_id, args) = ctx.get(factor).clone() {
            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) && args.len() == 1 {
                if ln_seen {
                    return None;
                }
                ln_seen = true;
                continue;
            }
        }
        coefficient_factors.push(factor);
    }
    if !ln_seen {
        return None;
    }

    let coefficient = if coefficient_factors.is_empty() {
        ctx.num(1)
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &coefficient_factors)
    };
    Some(
        Polynomial::from_expr(ctx, coefficient, var_name)
            .ok()?
            .degree(),
    )
}

fn compact_arctan_presentation_other_terms(
    ctx: &mut Context,
    terms: Vec<ExprId>,
    var_name: &str,
) -> Vec<ExprId> {
    let mut polynomial_sum = Polynomial::zero(var_name.to_string());
    let mut ln_groups: Vec<LnTermForCalculusPresentation> = Vec::new();
    let mut passthrough = Vec::new();

    for term in terms {
        if let Some(ln_term) = extract_ln_term_for_calculus_presentation(ctx, term, var_name) {
            if let Some(existing) = ln_groups
                .iter_mut()
                .find(|existing| existing.arg_poly == ln_term.arg_poly)
            {
                existing.coefficient += ln_term.coefficient;
            } else {
                ln_groups.push(ln_term);
            }
            continue;
        }

        if let Ok(poly) = Polynomial::from_expr(ctx, term, var_name) {
            polynomial_sum = polynomial_sum.add(&poly);
            continue;
        }

        passthrough.push(term);
    }

    let mut out = Vec::new();
    for ln_term in ln_groups {
        if let Some(term) =
            build_scaled_ln_for_calculus_presentation(ctx, &ln_term.coefficient, ln_term.arg)
        {
            out.push(term);
        }
    }
    if !polynomial_sum.is_zero() {
        out.push(polynomial_sum.to_expr(ctx));
    }
    out.extend(passthrough);
    out
}

fn contains_nontrivial_arctan_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let mut stack = vec![cas_ast::hold::unwrap_internal_hold(ctx, expr)];
    while let Some(current) = stack.pop() {
        match ctx.get(cas_ast::hold::unwrap_internal_hold(ctx, current)) {
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(fn_id, args) => {
                if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Arctan)
                    && args.len() == 1
                    && !variable_named(ctx, args[0], var_name)
                {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            _ => {}
        }
    }
    false
}

pub(super) fn flatten_subtracting_additive_group_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    let Expr::Sub(left, right) = ctx.get(expr).clone() else {
        return None;
    };
    let right = unwrap_internal_hold_for_calculus(ctx, right);
    if !matches!(ctx.get(right), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }
    if !contains_nontrivial_arctan_for_calculus_presentation(ctx, right, var_name) {
        return None;
    }
    if ln_polynomial_coefficient_degree_for_calculus_presentation(ctx, left, var_name)? > 5 {
        return None;
    }

    let mut additive_terms = Vec::new();
    collect_additive_terms_for_arctan_calculus_presentation(
        ctx,
        left,
        cas_math::expr_nary::Sign::Pos,
        &mut additive_terms,
    );
    collect_additive_terms_for_arctan_calculus_presentation(
        ctx,
        right,
        cas_math::expr_nary::Sign::Neg,
        &mut additive_terms,
    );
    if additive_terms.len() < 3 {
        return None;
    }

    let terms = additive_terms
        .into_iter()
        .map(|(term, sign)| {
            let signed = match sign {
                cas_math::expr_nary::Sign::Pos => term,
                cas_math::expr_nary::Sign::Neg => negate_term_for_calculus_presentation(ctx, term),
            };
            fold_numeric_mul_constants_for_hold(ctx, signed)
        })
        .collect::<Vec<_>>();

    Some(cas_math::expr_nary::build_balanced_add(ctx, &terms))
}

pub(super) fn compact_arctan_additive_terms_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);

    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            if cas_ast::views::as_rational_const(ctx, left, 8).is_some() {
                let compact =
                    compact_arctan_additive_terms_for_calculus_presentation(ctx, right, var_name)?;
                let compact = cas_ast::hold::wrap_hold(ctx, compact);
                return Some(ctx.add(Expr::Mul(left, compact)));
            }
            if cas_ast::views::as_rational_const(ctx, right, 8).is_some() {
                let compact =
                    compact_arctan_additive_terms_for_calculus_presentation(ctx, left, var_name)?;
                let compact = cas_ast::hold::wrap_hold(ctx, compact);
                return Some(ctx.add(Expr::Mul(compact, right)));
            }
        }
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, den, 8).is_some() => {
            let compact =
                compact_arctan_additive_terms_for_calculus_presentation(ctx, num, var_name)?;
            let compact = cas_ast::hold::wrap_hold(ctx, compact);
            return Some(ctx.add(Expr::Div(compact, den)));
        }
        _ => {}
    }

    let mut additive_terms = Vec::new();
    collect_additive_terms_for_arctan_calculus_presentation(
        ctx,
        expr,
        cas_math::expr_nary::Sign::Pos,
        &mut additive_terms,
    );
    if additive_terms.len() < 2 {
        return None;
    }

    let mut arctan_arg = None;
    let mut arctan_coefficients = Vec::new();
    let mut other_terms = Vec::new();
    let mut arctan_term_count = 0usize;

    for (term, sign) in additive_terms {
        if let Some((arg, coeff)) = extract_arctan_term_for_calculus_presentation(ctx, term) {
            if let Some(existing_arg) = arctan_arg {
                if !arctan_arg_matches_for_calculus_presentation(ctx, existing_arg, arg, var_name) {
                    return None;
                }
            } else {
                arctan_arg = Some(arg);
            }
            arctan_term_count += 1;
            let coeff = match sign {
                cas_math::expr_nary::Sign::Pos => coeff,
                cas_math::expr_nary::Sign::Neg => negate_term_for_calculus_presentation(ctx, coeff),
            };
            let coeff = fold_numeric_mul_constants_for_hold(ctx, coeff);
            arctan_coefficients.push(coeff);
        } else {
            let signed_term = match sign {
                cas_math::expr_nary::Sign::Pos => term,
                cas_math::expr_nary::Sign::Neg => negate_term_for_calculus_presentation(ctx, term),
            };
            let signed_term = fold_numeric_mul_constants_for_hold(ctx, signed_term);
            other_terms.push(signed_term);
        }
    }

    if arctan_term_count < 2 {
        return None;
    }

    let arg = arctan_arg?;
    if Polynomial::from_expr(ctx, arg, var_name).is_err() {
        return None;
    };
    let coeff = cas_math::expr_nary::build_balanced_add(ctx, &arctan_coefficients);
    let coeff = cas_ast::hold::wrap_hold(ctx, coeff);
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
    let arctan_term = ctx.add(Expr::Mul(coeff, arctan));

    let mut terms = vec![arctan_term];
    terms.extend(compact_arctan_presentation_other_terms(
        ctx,
        other_terms,
        var_name,
    ));
    Some(cas_math::expr_nary::build_balanced_add(ctx, &terms))
}

fn collect_additive_terms_for_arctan_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    sign: cas_math::expr_nary::Sign,
    out: &mut Vec<(ExprId, cas_math::expr_nary::Sign)>,
) {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            collect_additive_terms_for_arctan_calculus_presentation(ctx, left, sign, out);
            collect_additive_terms_for_arctan_calculus_presentation(ctx, right, sign, out);
        }
        Expr::Sub(left, right) => {
            collect_additive_terms_for_arctan_calculus_presentation(ctx, left, sign, out);
            collect_additive_terms_for_arctan_calculus_presentation(ctx, right, sign.negate(), out);
        }
        Expr::Neg(inner) => {
            collect_additive_terms_for_arctan_calculus_presentation(ctx, inner, sign.negate(), out);
        }
        _ => out.push((expr, sign)),
    }
}

pub(super) fn divide_compact_derivative_by_constant_factor(
    ctx: &mut Context,
    derivative: ExprId,
    outer_den: ExprId,
) -> ExprId {
    let derivative = unwrap_internal_hold_for_calculus(ctx, derivative);
    if let Expr::Div(num, den) = ctx.get(derivative).clone() {
        if let Some(cancelled_num) = remove_matching_sqrt_like_product_factor(ctx, num, outer_den) {
            return ctx.add(Expr::Div(cancelled_num, den));
        }

        let combined_den = cas_math::expr_nary::build_balanced_mul(ctx, &[outer_den, den]);
        return ctx.add(Expr::Div(num, combined_den));
    }

    ctx.add(Expr::Div(derivative, outer_den))
}

fn remove_matching_sqrt_like_product_factor(
    ctx: &mut Context,
    expr: ExprId,
    factor: ExprId,
) -> Option<ExprId> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let cancelled = remove_matching_sqrt_like_product_factor(ctx, inner, factor)?;
        return Some(negate_calculus_presentation(ctx, cancelled));
    }

    if same_sqrt_like_argument(ctx, expr, factor) {
        return Some(ctx.num(1));
    }

    let mut factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    for idx in 0..factors.len() {
        if same_sqrt_like_argument(ctx, factors[idx], factor) {
            factors.remove(idx);
            return Some(match factors.as_slice() {
                [] => ctx.num(1),
                [single] => *single,
                _ => cas_math::expr_nary::build_balanced_mul(ctx, &factors),
            });
        }
    }

    None
}

struct ArctanAffineByPartsTerm {
    arg: ExprId,
    arg_poly: Polynomial,
    cofactor_poly: Polynomial,
}

struct LnAffineByPartsTerm {
    arg_poly: Polynomial,
    coefficient: BigRational,
}

fn apply_additive_sign_to_poly(poly: Polynomial, sign: cas_math::expr_nary::Sign) -> Polynomial {
    match sign {
        cas_math::expr_nary::Sign::Pos => poly,
        cas_math::expr_nary::Sign::Neg => poly.neg(),
    }
}

fn arctan_affine_by_parts_arctan_term(
    ctx: &Context,
    term: ExprId,
    sign: cas_math::expr_nary::Sign,
    var_name: &str,
) -> Option<ArctanAffineByPartsTerm> {
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let denominator = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if denominator.is_zero() {
            return None;
        }
        let mut term = arctan_affine_by_parts_arctan_term(ctx, num, sign, var_name)?;
        term.cofactor_poly = term.cofactor_poly.div_scalar(&denominator);
        return Some(term);
    }

    let factors = cas_math::expr_nary::MulView::from_expr(ctx, term).factors;
    let mut arctan_arg = None;
    let mut cofactor_poly = Polynomial::one(var_name.to_string());

    for factor in factors {
        let factor = cas_ast::hold::unwrap_internal_hold(ctx, factor);
        if let Expr::Function(fn_id, args) = ctx.get(factor) {
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                )
            {
                if arctan_arg.replace(args[0]).is_some() {
                    return None;
                }
                continue;
            }
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
        cofactor_poly = cofactor_poly.mul(&factor_poly);
    }

    let arg = arctan_arg?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var_name).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }

    Some(ArctanAffineByPartsTerm {
        arg,
        arg_poly,
        cofactor_poly: apply_additive_sign_to_poly(cofactor_poly, sign),
    })
}

fn arctan_affine_by_parts_ln_term(
    ctx: &Context,
    term: ExprId,
    sign: cas_math::expr_nary::Sign,
    var_name: &str,
) -> Option<LnAffineByPartsTerm> {
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let denominator = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if denominator.is_zero() {
            return None;
        }
        let mut term = arctan_affine_by_parts_ln_term(ctx, num, sign, var_name)?;
        term.coefficient /= denominator;
        return Some(term);
    }

    let factors = cas_math::expr_nary::MulView::from_expr(ctx, term).factors;
    let mut ln_arg = None;
    let mut coefficient_poly = Polynomial::one(var_name.to_string());

    for factor in factors {
        let factor = cas_ast::hold::unwrap_internal_hold(ctx, factor);
        if let Expr::Function(fn_id, args) = ctx.get(factor) {
            if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Ln) && args.len() == 1 {
                if ln_arg.replace(args[0]).is_some() {
                    return None;
                }
                continue;
            }
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
        coefficient_poly = coefficient_poly.mul(&factor_poly);
    }

    let ln_arg = ln_arg?;
    let coefficient_poly = apply_additive_sign_to_poly(coefficient_poly, sign);
    if coefficient_poly.degree() != 0 {
        return None;
    }

    Some(LnAffineByPartsTerm {
        arg_poly: Polynomial::from_expr(ctx, ln_arg, var_name).ok()?,
        coefficient: coefficient_poly
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero),
    })
}

fn scale_polynomial(poly: &Polynomial, scale: &BigRational) -> Polynomial {
    Polynomial::new(
        poly.coeffs.iter().map(|coeff| coeff * scale).collect(),
        poly.var.clone(),
    )
}

fn polynomial_arctan_product(ctx: &mut Context, poly: &Polynomial, arg: ExprId) -> ExprId {
    if poly.is_zero() {
        return ctx.num(0);
    }

    let one = Polynomial::one(poly.var.clone());
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
    if *poly == one {
        return arctan;
    }
    if *poly == one.neg() {
        return ctx.add(Expr::Neg(arctan));
    }

    let poly_expr = poly.to_expr(ctx);
    ctx.add(Expr::Mul(poly_expr, arctan))
}

pub(super) fn arctan_affine_by_parts_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let target = unwrap_internal_hold_for_calculus(ctx, target);
    match ctx.get(target).clone() {
        Expr::Mul(left, right) => {
            if cas_ast::views::as_rational_const(ctx, left, 8).is_some() {
                let derivative = arctan_affine_by_parts_compact_derivative(ctx, right, var_name)?;
                let scaled = ctx.add(Expr::Mul(left, derivative));
                return Some(fold_numeric_mul_constants_for_hold(ctx, scaled));
            }
            if cas_ast::views::as_rational_const(ctx, right, 8).is_some() {
                let derivative = arctan_affine_by_parts_compact_derivative(ctx, left, var_name)?;
                let scaled = ctx.add(Expr::Mul(derivative, right));
                return Some(fold_numeric_mul_constants_for_hold(ctx, scaled));
            }
        }
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, den, 8).is_some() => {
            let derivative = arctan_affine_by_parts_compact_derivative(ctx, num, var_name)?;
            let scaled = ctx.add(Expr::Div(derivative, den));
            return Some(fold_numeric_mul_constants_for_hold(ctx, scaled));
        }
        _ => {}
    }

    let terms = cas_math::expr_nary::AddView::from_expr(ctx, target).terms;
    if terms.len() < 2 {
        return None;
    }

    let mut arctan_term: Option<ArctanAffineByPartsTerm> = None;
    let mut ln_term: Option<LnAffineByPartsTerm> = None;
    let mut remainder_poly = Polynomial::zero(var_name.to_string());

    for (term, sign) in terms {
        if let Some(term) = arctan_affine_by_parts_arctan_term(ctx, term, sign, var_name) {
            if let Some(existing) = &mut arctan_term {
                if existing.arg_poly != term.arg_poly {
                    return None;
                }
                existing.cofactor_poly = existing.cofactor_poly.add(&term.cofactor_poly);
            } else {
                arctan_term = Some(term);
            }
            continue;
        }

        if let Some(term) = arctan_affine_by_parts_ln_term(ctx, term, sign, var_name) {
            if let Some(existing) = &mut ln_term {
                if existing.arg_poly != term.arg_poly {
                    return None;
                }
                existing.coefficient += term.coefficient;
            } else {
                ln_term = Some(term);
            }
            continue;
        }

        let term_poly = Polynomial::from_expr(ctx, term, var_name).ok()?;
        remainder_poly = remainder_poly.add(&apply_additive_sign_to_poly(term_poly, sign));
    }

    let arctan_term = arctan_term?;
    let ln_term = ln_term?;
    let derivative_poly = arctan_term.arg_poly.derivative();
    if derivative_poly.degree() != 0 || derivative_poly.is_zero() {
        return None;
    }
    let linear_coeff = derivative_poly.coeffs.first()?.clone();
    if linear_coeff.is_zero() {
        return None;
    }

    let expected_ln_arg_poly = arctan_term
        .arg_poly
        .mul(&arctan_term.arg_poly)
        .add(&Polynomial::one(var_name.to_string()));
    if ln_term.arg_poly != expected_ln_arg_poly {
        return None;
    }

    let rational_numerator = scale_polynomial(&arctan_term.cofactor_poly, &linear_coeff)
        .add(&scale_polynomial(
            &arctan_term.arg_poly,
            &(BigRational::from_integer(2.into()) * &ln_term.coefficient * &linear_coeff),
        ))
        .add(&remainder_poly.derivative().mul(&expected_ln_arg_poly));
    if !rational_numerator.is_zero() {
        return None;
    }

    let arctan_cofactor_derivative = arctan_term.cofactor_poly.derivative();
    Some(polynomial_arctan_product(
        ctx,
        &arctan_cofactor_derivative,
        arctan_term.arg,
    ))
}

pub(super) fn apply_integration_final_presentation(
    ctx: &mut Context,
    mut result: ExprId,
    var_name: &str,
) -> ExprId {
    if let Some(compact) = compact_acosh_surd_width_arg_for_integration_presentation(ctx, result) {
        result = compact;
    }
    compact_integer_affine_inverse_args_for_integration_presentation(ctx, result, var_name)
}

pub(super) fn try_integrate_post_calculus_presentation(
    ctx: &mut Context,
    call: &NamedVarCall,
    result: ExprId,
) -> Option<ExprId> {
    if cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_arctan_affine_target(
        ctx,
        call.target,
        &call.var_name,
    ) {
        if let Some(compact) =
            compact_arctan_additive_terms_for_calculus_presentation(ctx, result, &call.var_name)
        {
            return Some(compact);
        }
    }
    if fractional_denominator_power_substitution_integrand_for_calculus_presentation(
        ctx,
        call.target,
        &call.var_name,
    ) {
        let allow_conditional_positive_quadratic =
            !integrate_required_positive_conditions(ctx, call.target, &call.var_name).is_empty();
        if let Some(compact) = compact_negative_three_half_power_result_for_integration_presentation(
            ctx,
            result,
            &call.var_name,
            allow_conditional_positive_quadratic,
        ) {
            return Some(compact);
        }
        if let Some(compact) =
            compact_negative_half_power_result_for_integration_presentation(ctx, result)
        {
            return Some(compact);
        }
    }
    if let Some(compact) =
        compact_positive_half_power_result_for_integration_presentation(ctx, result)
    {
        return Some(compact);
    }
    if let Some(compact) = compact_acosh_surd_width_arg_for_integration_presentation(ctx, result) {
        return Some(compact);
    }
    None
}

pub(super) fn try_diff_integral_source_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    result: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    try_extract_integrate_call(ctx, target)?;
    compact_sqrt_trig_log_derivative_integrand(ctx, result, var_name)
        .or_else(|| compact_direct_sqrt_hyperbolic_log_derivative_integrand(ctx, result, var_name))
}

pub(crate) fn try_calculus_result_presentation(
    ctx: &mut Context,
    result: ExprId,
) -> Option<ExprId> {
    let result = unwrap_internal_hold_for_calculus(ctx, result);
    if matches!(ctx.get(result), Expr::Constant(Constant::Undefined)) {
        return None;
    }

    compact_sqrt_trig_log_derivative_integrand(ctx, result, "x")
        .or_else(|| compact_direct_sqrt_hyperbolic_log_derivative_integrand(ctx, result, "x"))
        .or_else(|| {
            compact_rationalized_sqrt_denominator_quotient_for_calculus_presentation(ctx, result)
        })
        .or_else(|| compact_negative_half_power_product_for_calculus_presentation(ctx, result))
        .or_else(|| {
            has_compactable_ln_abs_cosh_sqrt(ctx, result, "x").then(|| {
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, result, "x")
            })
        })
        .or_else(|| {
            compact_half_power_sum_root_product_for_integration_presentation(ctx, result, "x")
        })
        .or_else(|| {
            compact_trig_square_reduction_primitive_for_integration_presentation(ctx, result, "x")
        })
        .or_else(|| {
            compact_trig_odd_power_reduction_primitive_for_integration_presentation(ctx, result)
        })
        .or_else(|| compact_acosh_surd_width_arg_for_integration_presentation(ctx, result))
        .or_else(|| compact_arctan_additive_terms_for_calculus_presentation(ctx, result, "x"))
}

fn compact_integer_affine_inverse_args_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(fn_id),
                    Some(
                        BuiltinFn::Arcsin
                            | BuiltinFn::Asin
                            | BuiltinFn::Arccos
                            | BuiltinFn::Acos
                            | BuiltinFn::Acosh
                    )
                ) =>
        {
            let arg = args[0];
            let Some(builtin) = ctx.builtin_of(fn_id) else {
                return expr;
            };
            let compact_arg = Polynomial::from_expr(ctx, arg, var_name)
                .ok()
                .filter(|poly| {
                    poly.degree() == 1
                        && (poly.coeffs.iter().all(|c| c.is_integer())
                            || (builtin == BuiltinFn::Acosh
                                && poly
                                    .coeffs
                                    .first()
                                    .is_some_and(|constant| constant.is_one())))
                })
                .map(|poly| poly.to_expr(ctx))
                .unwrap_or(arg);
            ctx.add(Expr::Function(fn_id, vec![compact_arg]))
        }
        Expr::Neg(inner) => {
            let inner = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, inner, var_name,
            );
            ctx.add(Expr::Neg(inner))
        }
        Expr::Add(left, right) => {
            let left = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(left, right) => {
            let left = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Div(left, right))
        }
        _ => expr,
    }
}

pub(super) fn compact_shifted_sqrt_argument_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(radicand) = calculus_sqrt_like_radicand(ctx, expr) {
        if !contains_named_var(ctx, radicand, var_name) {
            return None;
        }
        polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
        return Some(sqrt_raw_for_calculus_presentation(ctx, radicand));
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => compact_sqrt_shift_with_constant_side(ctx, left, right, var_name)
            .map(|(left, right)| ctx.add_raw(Expr::Add(left, right))),
        Expr::Sub(left, right) => compact_sqrt_shift_with_constant_side(ctx, left, right, var_name)
            .map(|(left, right)| ctx.add_raw(Expr::Sub(left, right))),
        _ => None,
    }
}

fn compact_sqrt_shift_with_constant_side(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    if !contains_named_var(ctx, right, var_name) {
        if let Some(compact_left) =
            compact_shifted_sqrt_argument_for_integration_presentation(ctx, left, var_name)
        {
            return Some((compact_left, right));
        }
    }

    if !contains_named_var(ctx, left, var_name) {
        if let Some(compact_right) =
            compact_shifted_sqrt_argument_for_integration_presentation(ctx, right, var_name)
        {
            return Some((left, compact_right));
        }
    }

    None
}

pub(super) fn compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(num, den) => {
            let num =
                compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(ctx, num, var_name);
            let den = compact_sqrt_hyperbolic_call_for_integration_presentation(ctx, den, var_name)
                .unwrap_or_else(|| {
                    compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                        ctx, den, var_name,
                    )
                });
            ctx.add(Expr::Div(num, den))
        }
        Expr::Neg(inner) => {
            let inner = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, inner, var_name,
            );
            ctx.add(Expr::Neg(inner))
        }
        _ => compact_sqrt_hyperbolic_call_for_integration_presentation(ctx, expr, var_name)
            .unwrap_or(expr),
    }
}

pub(super) fn has_compactable_sqrt_hyperbolic_reciprocal_result(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if compact_sqrt_hyperbolic_call_for_integration_presentation(ctx, expr, var_name).is_some() {
        return true;
    }
    match ctx.get(expr).clone() {
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, left, var_name)
                || has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, right, var_name)
        }
        Expr::Div(num, den) => {
            has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, num, var_name)
                || has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, den, var_name)
        }
        Expr::Neg(inner) => has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, inner, var_name),
        _ => false,
    }
}

fn compact_sqrt_hyperbolic_call_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(builtin, BuiltinFn::Sinh | BuiltinFn::Cosh | BuiltinFn::Tanh) || args.len() != 1 {
        return None;
    }

    let compact_arg =
        compact_shifted_sqrt_argument_for_integration_presentation(ctx, args[0], var_name)?;
    Some(ctx.call_builtin(builtin, vec![compact_arg]))
}

pub(super) fn compact_positive_cosh_log_abs_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some(compact) = compact_ln_abs_cosh_sqrt(ctx, expr, var_name) {
        return compact;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(num, den) => {
            let num =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, num, var_name);
            let den =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, den, var_name);
            ctx.add(Expr::Div(num, den))
        }
        Expr::Neg(inner) => {
            let inner =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, inner, var_name);
            ctx.add(Expr::Neg(inner))
        }
        _ => expr,
    }
}

fn compact_ln_abs_cosh_sqrt(ctx: &mut Context, expr: ExprId, var_name: &str) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let (hyperbolic_expr, wrapped_in_abs) = match ctx.get(args[0]).clone() {
        Expr::Function(abs_fn, abs_args)
            if ctx.builtin_of(abs_fn) == Some(BuiltinFn::Abs) && abs_args.len() == 1 =>
        {
            (abs_args[0], true)
        }
        Expr::Function(hyperbolic_fn, hyperbolic_args)
            if matches!(
                ctx.builtin_of(hyperbolic_fn),
                Some(BuiltinFn::Cosh | BuiltinFn::Sinh)
            ) && hyperbolic_args.len() == 1 =>
        {
            (args[0], false)
        }
        _ => return None,
    };
    let Expr::Function(hyperbolic_fn, hyperbolic_args) = ctx.get(hyperbolic_expr).clone() else {
        return None;
    };
    let hyperbolic_builtin = ctx.builtin_of(hyperbolic_fn)?;
    if !matches!(hyperbolic_builtin, BuiltinFn::Cosh | BuiltinFn::Sinh)
        || hyperbolic_args.len() != 1
    {
        return None;
    }

    let compact_arg = compact_shifted_sqrt_argument_for_integration_presentation(
        ctx,
        hyperbolic_args[0],
        var_name,
    )?;
    let hyperbolic_expr = ctx.call_builtin(hyperbolic_builtin, vec![compact_arg]);
    let log_arg = if wrapped_in_abs && hyperbolic_builtin != BuiltinFn::Cosh {
        ctx.call_builtin(BuiltinFn::Abs, vec![hyperbolic_expr])
    } else {
        hyperbolic_expr
    };
    Some(ctx.call_builtin(BuiltinFn::Ln, vec![log_arg]))
}

pub(super) fn has_compactable_ln_abs_cosh_sqrt(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if compact_ln_abs_cosh_sqrt(ctx, expr, var_name).is_some() {
        return true;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            has_compactable_ln_abs_cosh_sqrt(ctx, left, var_name)
                || has_compactable_ln_abs_cosh_sqrt(ctx, right, var_name)
        }
        Expr::Div(num, den) => {
            has_compactable_ln_abs_cosh_sqrt(ctx, num, var_name)
                || has_compactable_ln_abs_cosh_sqrt(ctx, den, var_name)
        }
        Expr::Neg(inner) => has_compactable_ln_abs_cosh_sqrt(ctx, inner, var_name),
        _ => false,
    }
}

pub(super) fn compact_sqrt_trig_log_abs_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some(compact) = compact_ln_abs_trig_sqrt(ctx, expr, var_name) {
        return compact;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_sqrt_trig_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_sqrt_trig_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_sqrt_trig_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(num, den) => {
            let num = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, num, var_name);
            let den = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, den, var_name);
            ctx.add(Expr::Div(num, den))
        }
        Expr::Neg(inner) => {
            let inner =
                compact_sqrt_trig_log_abs_for_integration_presentation(ctx, inner, var_name);
            ctx.add(Expr::Neg(inner))
        }
        _ => expr,
    }
}

fn compact_ln_abs_trig_sqrt(ctx: &mut Context, expr: ExprId, var_name: &str) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let Expr::Function(abs_fn, abs_args) = ctx.get(args[0]).clone() else {
        return None;
    };
    if ctx.builtin_of(abs_fn) != Some(BuiltinFn::Abs) || abs_args.len() != 1 {
        return None;
    }

    let Expr::Function(trig_fn, trig_args) = ctx.get(abs_args[0]).clone() else {
        return None;
    };
    let trig_builtin = ctx.builtin_of(trig_fn)?;
    if !matches!(trig_builtin, BuiltinFn::Sin | BuiltinFn::Cos) || trig_args.len() != 1 {
        return None;
    }

    let radicand = calculus_sqrt_like_radicand(ctx, trig_args[0])?;
    if !contains_named_var(ctx, radicand, var_name) {
        return None;
    }
    polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let trig_expr = ctx.call_builtin(trig_builtin, vec![sqrt_radicand]);
    let abs_expr = ctx.call_builtin(BuiltinFn::Abs, vec![trig_expr]);
    Some(ctx.call_builtin(BuiltinFn::Ln, vec![abs_expr]))
}

pub(super) fn has_compactable_ln_abs_trig_sqrt(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if compact_ln_abs_trig_sqrt(ctx, expr, var_name).is_some() {
        return true;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            has_compactable_ln_abs_trig_sqrt(ctx, left, var_name)
                || has_compactable_ln_abs_trig_sqrt(ctx, right, var_name)
        }
        Expr::Div(num, den) => {
            has_compactable_ln_abs_trig_sqrt(ctx, num, var_name)
                || has_compactable_ln_abs_trig_sqrt(ctx, den, var_name)
        }
        Expr::Neg(inner) => has_compactable_ln_abs_trig_sqrt(ctx, inner, var_name),
        _ => false,
    }
}
