//! Fraction addition rules.
//!
//! This module contains rules for adding terms with fractions:
//! - `FoldAddIntoFractionRule`: k + p/q → (k·q + p)/q
//! - `AddFractionsRule`: a/b + c/d → (ad+bc)/bd

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Expr};
use cas_math::expr_classify::is_trig_function;
use cas_math::fraction_add_rewrite_support::{
    plan_add_fraction_rewrite_with, AddFractionRewriteInput,
};
use cas_math::fraction_add_rule_support::{
    try_plan_fold_add_into_fraction_rewrite, try_plan_sub_term_matches_denom_rewrite,
    try_plan_symmetric_reciprocal_sum_rewrite,
};
use cas_math::fraction_pair_guard_support::{
    should_block_add_fraction_pair, should_block_sub_fraction_pair, AddFractionPairGuardInput,
    SubFractionPairGuardInput,
};
use cas_math::fraction_pair_support::extract_fraction_pair;
use cas_math::fraction_sub_rewrite_support::plan_sub_fraction_rewrite_with;

fn collect_signed_mul_factors(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    sign: &mut i8,
    out: &mut Vec<cas_ast::ExprId>,
) {
    match ctx.get(expr).clone() {
        Expr::Mul(lhs, rhs) => {
            collect_signed_mul_factors(ctx, lhs, sign, out);
            collect_signed_mul_factors(ctx, rhs, sign, out);
        }
        Expr::Neg(inner) => {
            *sign *= -1;
            collect_signed_mul_factors(ctx, inner, sign, out);
        }
        Expr::Number(n) if n < num_rational::BigRational::from_integer(0.into()) => {
            *sign *= -1;
            out.push(ctx.add(Expr::Number(-n)));
        }
        _ => out.push(expr),
    }
}

fn sqrt_or_half_power_base(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let expr = match ctx.get(expr) {
        Expr::Hold(inner) => *inner,
        _ => expr,
    };
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Sqrt) =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exp) => match ctx.get(*exp) {
            Expr::Number(n) if *n == num_rational::BigRational::new(1.into(), 2.into()) => {
                Some(*base)
            }
            Expr::Div(num, den)
                if matches!(ctx.get(*num), Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into()))
                    && matches!(ctx.get(*den), Expr::Number(n) if *n == num_rational::BigRational::from_integer(2.into())) =>
            {
                Some(*base)
            }
            _ => None,
        },
        _ => None,
    }
}

fn factors_match_for_denominator(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    if compare_expr(ctx, lhs, rhs) == std::cmp::Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs, rhs)
    {
        return true;
    }

    let lhs_normalized = cas_math::canonical_forms::normalize_core(ctx, lhs);
    let rhs_normalized = cas_math::canonical_forms::normalize_core(ctx, rhs);
    if compare_expr(ctx, lhs_normalized, rhs_normalized) == std::cmp::Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs_normalized, rhs_normalized)
    {
        return true;
    }

    if power_factors_match_for_denominator(ctx, lhs, rhs)
        || additive_factors_match_for_denominator(ctx, lhs, rhs)
    {
        return true;
    }

    let Some(lhs_base) = sqrt_or_half_power_base(ctx, lhs) else {
        return false;
    };
    let Some(rhs_base) = sqrt_or_half_power_base(ctx, rhs) else {
        return false;
    };
    compare_expr(ctx, lhs_base, rhs_base) == std::cmp::Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs_base, rhs_base)
}

fn power_factors_match_for_denominator(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let (Expr::Pow(lhs_base, lhs_exp), Expr::Pow(rhs_base, rhs_exp)) =
        (ctx.get(lhs).clone(), ctx.get(rhs).clone())
    else {
        return false;
    };

    let exponents_match = compare_expr(ctx, lhs_exp, rhs_exp) == std::cmp::Ordering::Equal
        || cas_ast::views::as_rational_const(ctx, lhs_exp, 8)
            .zip(cas_ast::views::as_rational_const(ctx, rhs_exp, 8))
            .is_some_and(|(lhs_value, rhs_value)| lhs_value == rhs_value);
    exponents_match && factors_match_for_denominator(ctx, lhs_base, rhs_base)
}

fn additive_factors_match_for_denominator(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let lhs_is_additive = matches!(
        ctx.get(lhs),
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_)
    );
    let rhs_is_additive = matches!(
        ctx.get(rhs),
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_)
    );
    if !lhs_is_additive || !rhs_is_additive {
        return false;
    }

    let mut lhs_terms = Vec::new();
    let mut rhs_terms = Vec::new();
    collect_signed_add_terms(ctx, lhs, 1, &mut lhs_terms);
    collect_signed_add_terms(ctx, rhs, 1, &mut rhs_terms);
    if lhs_terms.len() != rhs_terms.len() {
        return false;
    }

    let mut rhs_used = vec![false; rhs_terms.len()];
    'lhs: for (lhs_sign, lhs_term) in lhs_terms {
        for (rhs_index, (rhs_sign, rhs_term)) in rhs_terms.iter().copied().enumerate() {
            if rhs_used[rhs_index] || lhs_sign != rhs_sign {
                continue;
            }
            if exprs_equal_up_to_mul_factor_order_and_sign(ctx, lhs_term, rhs_term) {
                rhs_used[rhs_index] = true;
                continue 'lhs;
            }
        }
        return false;
    }

    true
}

fn exprs_equal_up_to_mul_factor_order_and_sign(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let mut lhs_sign = 1_i8;
    let mut rhs_sign = 1_i8;
    let mut lhs_factors = Vec::new();
    let mut rhs_factors = Vec::new();

    collect_signed_mul_factors(ctx, lhs, &mut lhs_sign, &mut lhs_factors);
    collect_signed_mul_factors(ctx, rhs, &mut rhs_sign, &mut rhs_factors);

    if lhs_sign != rhs_sign || lhs_factors.len() != rhs_factors.len() {
        return false;
    }

    let mut rhs_used = vec![false; rhs_factors.len()];
    'lhs: for lhs_factor in lhs_factors {
        for (rhs_index, rhs_factor) in rhs_factors.iter().copied().enumerate() {
            if rhs_used[rhs_index] {
                continue;
            }
            if factors_match_for_denominator(ctx, lhs_factor, rhs_factor) {
                rhs_used[rhs_index] = true;
                continue 'lhs;
            }
        }
        return false;
    }

    true
}

fn split_numeric_content_and_non_numeric_factors(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> (i8, num_rational::BigRational, Vec<cas_ast::ExprId>) {
    let mut sign = 1_i8;
    let mut factors = Vec::new();
    collect_signed_mul_factors(ctx, expr, &mut sign, &mut factors);

    let mut numeric = num_rational::BigRational::from_integer(1.into());
    let mut non_numeric = Vec::new();
    for factor in factors {
        match ctx.get(factor).clone() {
            Expr::Number(n) => numeric *= n,
            _ => non_numeric.push(factor),
        }
    }

    (sign, numeric, non_numeric)
}

fn factor_multisets_match(
    ctx: &mut cas_ast::Context,
    lhs_factors: &[cas_ast::ExprId],
    rhs_factors: &[cas_ast::ExprId],
) -> bool {
    if lhs_factors.len() != rhs_factors.len() {
        return false;
    }

    let mut rhs_used = vec![false; rhs_factors.len()];
    'lhs: for lhs_factor in lhs_factors.iter().copied() {
        for (rhs_index, rhs_factor) in rhs_factors.iter().copied().enumerate() {
            if rhs_used[rhs_index] {
                continue;
            }
            if factors_match_for_denominator(ctx, lhs_factor, rhs_factor) {
                rhs_used[rhs_index] = true;
                continue 'lhs;
            }
        }
        return false;
    }

    true
}

fn rational_power_factor(
    ctx: &mut cas_ast::Context,
    factor: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, num_rational::BigRational)> {
    let factor = cas_ast::hold::unwrap_internal_hold(ctx, factor);
    if let Some(base) = sqrt_or_half_power_base(ctx, factor) {
        return Some((base, num_rational::BigRational::new(1.into(), 2.into())));
    }

    match ctx.get(factor).clone() {
        Expr::Pow(base, exp) => {
            let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
            Some((base, exponent))
        }
        _ => Some((factor, num_rational::BigRational::from_integer(1.into()))),
    }
}

fn single_base_power_product(
    ctx: &mut cas_ast::Context,
    factors: &[cas_ast::ExprId],
) -> Option<(cas_ast::ExprId, num_rational::BigRational)> {
    let mut iter = factors.iter().copied();
    let first = iter.next()?;
    let (base, mut exponent) = rational_power_factor(ctx, first)?;
    for factor in iter {
        let (factor_base, factor_exponent) = rational_power_factor(ctx, factor)?;
        if !factors_match_for_denominator(ctx, base, factor_base) {
            return None;
        }
        exponent += factor_exponent;
    }
    Some((base, exponent))
}

fn factor_multisets_match_with_combined_power_family(
    ctx: &mut cas_ast::Context,
    lhs_factors: &[cas_ast::ExprId],
    rhs_factors: &[cas_ast::ExprId],
) -> bool {
    if factor_multisets_match(ctx, lhs_factors, rhs_factors) {
        return true;
    }

    let mut rhs_used = vec![false; rhs_factors.len()];
    let mut lhs_unmatched = Vec::new();
    for lhs_factor in lhs_factors.iter().copied() {
        let mut matched = false;
        for (rhs_index, rhs_factor) in rhs_factors.iter().copied().enumerate() {
            if rhs_used[rhs_index] {
                continue;
            }
            if factors_match_for_denominator(ctx, lhs_factor, rhs_factor) {
                rhs_used[rhs_index] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            lhs_unmatched.push(lhs_factor);
        }
    }

    let rhs_unmatched = rhs_factors
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, factor)| (!rhs_used[index]).then_some(factor))
        .collect::<Vec<_>>();
    if lhs_unmatched.is_empty() || rhs_unmatched.is_empty() {
        return false;
    }

    let Some((lhs_base, lhs_exponent)) = single_base_power_product(ctx, &lhs_unmatched) else {
        return false;
    };
    let Some((rhs_base, rhs_exponent)) = single_base_power_product(ctx, &rhs_unmatched) else {
        return false;
    };

    lhs_exponent == rhs_exponent && factors_match_for_denominator(ctx, lhs_base, rhs_base)
}

fn collect_signed_add_terms(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    sign: i8,
    out: &mut Vec<(i8, cas_ast::ExprId)>,
) {
    let expr = match ctx.get(expr) {
        Expr::Hold(inner) => *inner,
        _ => expr,
    };
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            collect_signed_add_terms(ctx, lhs, sign, out);
            collect_signed_add_terms(ctx, rhs, sign, out);
        }
        Expr::Sub(lhs, rhs) => {
            collect_signed_add_terms(ctx, lhs, sign, out);
            collect_signed_add_terms(ctx, rhs, -sign, out);
        }
        Expr::Neg(inner) => collect_signed_add_terms(ctx, inner, -sign, out),
        Expr::Mul(lhs, rhs) => {
            let lhs_is_additive = matches!(
                ctx.get(lhs),
                Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_)
            );
            let rhs_is_additive = matches!(
                ctx.get(rhs),
                Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_)
            );
            if lhs_is_additive && !rhs_is_additive {
                let mut lhs_terms = Vec::new();
                collect_signed_add_terms(ctx, lhs, sign, &mut lhs_terms);
                for (term_sign, term) in lhs_terms {
                    let product = ctx.add(Expr::Mul(term, rhs));
                    out.push((term_sign, product));
                }
            } else if rhs_is_additive && !lhs_is_additive {
                let mut rhs_terms = Vec::new();
                collect_signed_add_terms(ctx, rhs, sign, &mut rhs_terms);
                for (term_sign, term) in rhs_terms {
                    let product = ctx.add(Expr::Mul(lhs, term));
                    out.push((term_sign, product));
                }
            } else {
                out.push((sign, expr));
            }
        }
        _ => out.push((sign, expr)),
    }
}

fn rational_scale_between_additive_terms(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> Option<num_rational::BigRational> {
    let mut lhs_terms = Vec::new();
    let mut rhs_terms = Vec::new();
    collect_signed_add_terms(ctx, lhs, 1, &mut lhs_terms);
    collect_signed_add_terms(ctx, rhs, 1, &mut rhs_terms);
    if lhs_terms.len() != rhs_terms.len() {
        return None;
    }

    let mut rhs_used = vec![false; rhs_terms.len()];
    let mut ratio: Option<num_rational::BigRational> = None;
    for (lhs_outer_sign, lhs_term) in lhs_terms {
        let (lhs_term_sign, lhs_content, lhs_factors) =
            split_numeric_content_and_non_numeric_factors(ctx, lhs_term);
        let lhs_coeff = if lhs_outer_sign * lhs_term_sign < 0 {
            -lhs_content
        } else {
            lhs_content
        };
        if lhs_coeff == num_rational::BigRational::from_integer(0.into()) {
            return None;
        }

        let mut matched = false;
        for (rhs_index, (rhs_outer_sign, rhs_term)) in rhs_terms.iter().copied().enumerate() {
            if rhs_used[rhs_index] {
                continue;
            }
            let (rhs_term_sign, rhs_content, rhs_factors) =
                split_numeric_content_and_non_numeric_factors(ctx, rhs_term);
            let factors_match =
                factor_multisets_match_with_combined_power_family(ctx, &lhs_factors, &rhs_factors)
                    || {
                        let lhs_core = rebuild_mul_factors(ctx, &lhs_factors);
                        let rhs_core = rebuild_mul_factors(ctx, &rhs_factors);
                        exprs_equivalent_after_core_normalization(ctx, lhs_core, rhs_core)
                    };
            if !factors_match {
                continue;
            }

            let rhs_coeff = if rhs_outer_sign * rhs_term_sign < 0 {
                -rhs_content
            } else {
                rhs_content
            };
            let term_ratio = rhs_coeff / lhs_coeff.clone();
            if let Some(existing_ratio) = &ratio {
                if existing_ratio != &term_ratio {
                    return None;
                }
            } else {
                ratio = Some(term_ratio);
            }
            rhs_used[rhs_index] = true;
            matched = true;
            break;
        }
        if !matched {
            return None;
        }
    }

    ratio
}

fn factor_scale_ratio_if_rational_multiple(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> Option<num_rational::BigRational> {
    if factors_match_for_denominator(ctx, lhs, rhs) {
        return Some(num_rational::BigRational::from_integer(1.into()));
    }

    if let Some(ratio) = rational_scale_between_additive_terms(ctx, lhs, rhs) {
        return Some(ratio);
    }

    let lhs_normalized = cas_math::canonical_forms::normalize_core(ctx, lhs);
    let rhs_normalized = cas_math::canonical_forms::normalize_core(ctx, rhs);
    rational_scale_between_additive_terms(ctx, lhs_normalized, rhs_normalized)
}

fn factor_multisets_scale_ratio_with_at_most_one_scaled_factor(
    ctx: &mut cas_ast::Context,
    lhs_factors: &[cas_ast::ExprId],
    rhs_factors: &[cas_ast::ExprId],
) -> Option<num_rational::BigRational> {
    if lhs_factors.len() != rhs_factors.len() {
        return None;
    }

    let mut rhs_used = vec![false; rhs_factors.len()];
    let mut scaled_ratio: Option<num_rational::BigRational> = None;
    'lhs: for lhs_factor in lhs_factors.iter().copied() {
        for (rhs_index, rhs_factor) in rhs_factors.iter().copied().enumerate() {
            if rhs_used[rhs_index] {
                continue;
            }
            if factors_match_for_denominator(ctx, lhs_factor, rhs_factor) {
                rhs_used[rhs_index] = true;
                continue 'lhs;
            }
        }

        if scaled_ratio.is_some() {
            return None;
        }

        for (rhs_index, rhs_factor) in rhs_factors.iter().copied().enumerate() {
            if rhs_used[rhs_index] {
                continue;
            }
            let Some(ratio) = factor_scale_ratio_if_rational_multiple(ctx, lhs_factor, rhs_factor)
            else {
                continue;
            };
            if ratio == num_rational::BigRational::from_integer(1.into()) {
                rhs_used[rhs_index] = true;
                continue 'lhs;
            }
            rhs_used[rhs_index] = true;
            scaled_ratio = Some(ratio);
            continue 'lhs;
        }

        return None;
    }

    Some(scaled_ratio.unwrap_or_else(|| num_rational::BigRational::from_integer(1.into())))
}

fn denominator_factor_products_scale_ratio(
    ctx: &mut cas_ast::Context,
    lhs_factors: &[cas_ast::ExprId],
    rhs_factors: &[cas_ast::ExprId],
) -> Option<num_rational::BigRational> {
    if let Some(ratio) =
        factor_multisets_scale_ratio_with_at_most_one_scaled_factor(ctx, lhs_factors, rhs_factors)
    {
        return Some(ratio);
    }

    let lhs_product = rebuild_mul_factors(ctx, lhs_factors);
    let rhs_product = rebuild_mul_factors(ctx, rhs_factors);
    if factors_match_for_denominator(ctx, lhs_product, rhs_product) {
        return Some(num_rational::BigRational::from_integer(1.into()));
    }
    if let Some(ratio) = rational_scale_between_additive_terms(ctx, lhs_product, rhs_product) {
        return Some(ratio);
    }

    let lhs_normalized = cas_math::canonical_forms::normalize_core(ctx, lhs_product);
    let rhs_normalized = cas_math::canonical_forms::normalize_core(ctx, rhs_product);
    if factors_match_for_denominator(ctx, lhs_normalized, rhs_normalized) {
        return Some(num_rational::BigRational::from_integer(1.into()));
    }
    rational_scale_between_additive_terms(ctx, lhs_normalized, rhs_normalized)
}

fn factor_multisets_match_with_one_scaled_factor(
    ctx: &mut cas_ast::Context,
    lhs_factors: &[cas_ast::ExprId],
    rhs_factors: &[cas_ast::ExprId],
    rhs_over_lhs_scale: &num_rational::BigRational,
) -> bool {
    if *rhs_over_lhs_scale == num_rational::BigRational::from_integer(1.into()) {
        return factor_multisets_match(ctx, lhs_factors, rhs_factors);
    }
    if lhs_factors.len() != rhs_factors.len() {
        return false;
    }

    let mut rhs_used = vec![false; rhs_factors.len()];
    let mut used_scaled_factor = false;
    'lhs: for lhs_factor in lhs_factors.iter().copied() {
        for (rhs_index, rhs_factor) in rhs_factors.iter().copied().enumerate() {
            if rhs_used[rhs_index] {
                continue;
            }
            if factors_match_for_denominator(ctx, lhs_factor, rhs_factor) {
                rhs_used[rhs_index] = true;
                continue 'lhs;
            }
        }

        if used_scaled_factor {
            return false;
        }

        for (rhs_index, rhs_factor) in rhs_factors.iter().copied().enumerate() {
            if rhs_used[rhs_index] {
                continue;
            }
            if factor_scale_ratio_if_rational_multiple(ctx, lhs_factor, rhs_factor)
                == Some(rhs_over_lhs_scale.clone())
            {
                rhs_used[rhs_index] = true;
                used_scaled_factor = true;
                continue 'lhs;
            }
        }

        return false;
    }

    used_scaled_factor
}

fn scale_factor_by_numeric_content(
    ctx: &mut cas_ast::Context,
    factor: cas_ast::ExprId,
    content: &num_rational::BigRational,
) -> cas_ast::ExprId {
    if *content == num_rational::BigRational::from_integer(1.into()) {
        return factor;
    }

    let numeric = ctx.add(Expr::Number(content.clone()));
    ctx.add(Expr::Mul(numeric, factor))
}

fn exprs_equivalent_after_core_normalization(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    if compare_expr(ctx, lhs, rhs) == std::cmp::Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs, rhs)
    {
        return true;
    }

    let lhs_normalized = cas_math::canonical_forms::normalize_core(ctx, lhs);
    let rhs_normalized = cas_math::canonical_forms::normalize_core(ctx, rhs);
    compare_expr(ctx, lhs_normalized, rhs_normalized) == std::cmp::Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs_normalized, rhs_normalized)
        || cas_math::poly_compare::poly_eq(ctx, lhs_normalized, rhs_normalized)
}

fn exprs_equal_up_to_single_numeric_content_factor(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let (lhs_sign, lhs_content, lhs_factors) =
        split_numeric_content_and_non_numeric_factors(ctx, lhs);
    let (rhs_sign, rhs_content, rhs_factors) =
        split_numeric_content_and_non_numeric_factors(ctx, rhs);
    if lhs_sign != rhs_sign {
        return false;
    }

    let mut rhs_used = vec![false; rhs_factors.len()];
    let mut unmatched_lhs = Vec::new();
    for lhs_factor in lhs_factors {
        let mut matched = false;
        for (rhs_index, rhs_factor) in rhs_factors.iter().copied().enumerate() {
            if rhs_used[rhs_index] {
                continue;
            }
            if factors_match_for_denominator(ctx, lhs_factor, rhs_factor) {
                rhs_used[rhs_index] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            unmatched_lhs.push(lhs_factor);
        }
    }

    let unmatched_rhs: Vec<_> = rhs_factors
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, factor)| (!rhs_used[index]).then_some(factor))
        .collect();

    if unmatched_lhs.is_empty() && unmatched_rhs.is_empty() {
        return lhs_content == rhs_content;
    }

    if unmatched_lhs.len() != 1 || unmatched_rhs.len() != 1 {
        return false;
    }

    let lhs_scaled = scale_factor_by_numeric_content(ctx, unmatched_lhs[0], &lhs_content);
    let rhs_scaled = scale_factor_by_numeric_content(ctx, unmatched_rhs[0], &rhs_content);
    exprs_equivalent_after_core_normalization(ctx, lhs_scaled, rhs_scaled)
}

fn exprs_match_same_denominator(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let lhs = cas_ast::hold::unwrap_internal_hold(ctx, lhs);
    let rhs = cas_ast::hold::unwrap_internal_hold(ctx, rhs);
    compare_expr(ctx, lhs, rhs) == std::cmp::Ordering::Equal
        || exprs_equal_up_to_mul_factor_order_and_sign(ctx, lhs, rhs)
        || exprs_equal_up_to_single_numeric_content_factor(ctx, lhs, rhs)
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs, rhs)
}

fn rebuild_mul_factors(ctx: &mut cas_ast::Context, factors: &[cas_ast::ExprId]) -> cas_ast::ExprId {
    let mut iter = factors.iter().copied();
    let Some(first) = iter.next() else {
        return ctx.num(1);
    };
    iter.fold(first, |acc, factor| ctx.add(Expr::Mul(acc, factor)))
}

fn normalize_fraction_numerator_sign(
    ctx: &mut cas_ast::Context,
    numerator: cas_ast::ExprId,
    sign: i8,
) -> (cas_ast::ExprId, i8) {
    let numerator = cas_ast::hold::unwrap_internal_hold(ctx, numerator);
    match ctx.get(numerator).clone() {
        Expr::Neg(inner) => (inner, -sign),
        Expr::Number(n) if n < num_rational::BigRational::from_integer(0.into()) => {
            (ctx.add(Expr::Number(-n)), -sign)
        }
        _ => {
            let mut factor_sign = 1_i8;
            let mut factors = Vec::new();
            collect_signed_mul_factors(ctx, numerator, &mut factor_sign, &mut factors);
            if factor_sign < 0 {
                (rebuild_mul_factors(ctx, &factors), -sign)
            } else {
                (numerator, sign)
            }
        }
    }
}

fn remove_redundant_fraction_sign(
    ctx: &mut cas_ast::Context,
    numerator: cas_ast::ExprId,
    sign: i8,
) -> (cas_ast::ExprId, i8) {
    let numerator = cas_ast::hold::unwrap_internal_hold(ctx, numerator);
    if sign < 0 {
        if let Expr::Neg(inner) = ctx.get(numerator).clone() {
            return (inner, sign);
        }
    }
    (numerator, sign)
}

fn scaled_fraction_value_parts(
    ctx: &mut cas_ast::Context,
    numerator: cas_ast::ExprId,
    denominator: cas_ast::ExprId,
    sign: i8,
) -> Option<(
    i8,
    num_rational::BigRational,
    Vec<cas_ast::ExprId>,
    Vec<cas_ast::ExprId>,
)> {
    let (numerator, sign) = normalize_fraction_numerator_sign(ctx, numerator, sign);
    let (num_sign, num_content, num_factors) =
        split_numeric_content_and_non_numeric_factors(ctx, numerator);
    let (den_sign, den_content, den_factors) =
        split_numeric_content_and_non_numeric_factors(ctx, denominator);
    if den_content == num_rational::BigRational::from_integer(0.into()) {
        return None;
    }

    Some((
        sign * num_sign * den_sign,
        num_content / den_content,
        num_factors,
        den_factors,
    ))
}

struct ScaledFractionPartsRef<'a> {
    value_sign: i8,
    scale: &'a num_rational::BigRational,
    numerator_factors: &'a [cas_ast::ExprId],
    denominator_factors: &'a [cas_ast::ExprId],
}

fn scaled_fraction_parts_match(
    ctx: &mut cas_ast::Context,
    lhs: ScaledFractionPartsRef<'_>,
    rhs: ScaledFractionPartsRef<'_>,
    opposite_values: bool,
) -> bool {
    if *rhs.scale == num_rational::BigRational::from_integer(0.into()) {
        return false;
    }

    let Some(denominator_rhs_over_lhs_scale) = denominator_factor_products_scale_ratio(
        ctx,
        lhs.denominator_factors,
        rhs.denominator_factors,
    ) else {
        return false;
    };

    let mut required_rhs_over_lhs_scale = lhs.scale.clone() / rhs.scale.clone();
    if lhs.value_sign != rhs.value_sign {
        required_rhs_over_lhs_scale = -required_rhs_over_lhs_scale;
    }
    if opposite_values {
        required_rhs_over_lhs_scale = -required_rhs_over_lhs_scale;
    }
    required_rhs_over_lhs_scale *= denominator_rhs_over_lhs_scale;

    if required_rhs_over_lhs_scale == num_rational::BigRational::from_integer(1.into()) {
        return factor_multisets_match(ctx, lhs.numerator_factors, rhs.numerator_factors);
    }

    factor_multisets_match_with_one_scaled_factor(
        ctx,
        lhs.numerator_factors,
        rhs.numerator_factors,
        &required_rhs_over_lhs_scale,
    )
}

fn fractions_match_same_scaled_value(
    ctx: &mut cas_ast::Context,
    n1: cas_ast::ExprId,
    d1: cas_ast::ExprId,
    sign1: i8,
    n2: cas_ast::ExprId,
    d2: cas_ast::ExprId,
    sign2: i8,
) -> bool {
    let Some((value_sign1, scale1, numerator_factors1, denominator_factors1)) =
        scaled_fraction_value_parts(ctx, n1, d1, sign1)
    else {
        return false;
    };
    let Some((value_sign2, scale2, numerator_factors2, denominator_factors2)) =
        scaled_fraction_value_parts(ctx, n2, d2, sign2)
    else {
        return false;
    };

    scaled_fraction_parts_match(
        ctx,
        ScaledFractionPartsRef {
            value_sign: value_sign1,
            scale: &scale1,
            numerator_factors: &numerator_factors1,
            denominator_factors: &denominator_factors1,
        },
        ScaledFractionPartsRef {
            value_sign: value_sign2,
            scale: &scale2,
            numerator_factors: &numerator_factors2,
            denominator_factors: &denominator_factors2,
        },
        false,
    )
}

fn fractions_match_opposite_scaled_value(
    ctx: &mut cas_ast::Context,
    n1: cas_ast::ExprId,
    d1: cas_ast::ExprId,
    sign1: i8,
    n2: cas_ast::ExprId,
    d2: cas_ast::ExprId,
    sign2: i8,
) -> bool {
    let Some((value_sign1, scale1, numerator_factors1, denominator_factors1)) =
        scaled_fraction_value_parts(ctx, n1, d1, sign1)
    else {
        return false;
    };
    let Some((value_sign2, scale2, numerator_factors2, denominator_factors2)) =
        scaled_fraction_value_parts(ctx, n2, d2, sign2)
    else {
        return false;
    };

    scaled_fraction_parts_match(
        ctx,
        ScaledFractionPartsRef {
            value_sign: value_sign1,
            scale: &scale1,
            numerator_factors: &numerator_factors1,
            denominator_factors: &denominator_factors1,
        },
        ScaledFractionPartsRef {
            value_sign: value_sign2,
            scale: &scale2,
            numerator_factors: &numerator_factors2,
            denominator_factors: &denominator_factors2,
        },
        true,
    )
}

fn should_defer_exact_opposite_fraction_pair_to_additive_cancellation(
    ctx: &mut cas_ast::Context,
    n1: cas_ast::ExprId,
    d1: cas_ast::ExprId,
    sign1: i8,
    n2: cas_ast::ExprId,
    d2: cas_ast::ExprId,
    sign2: i8,
) -> bool {
    let (n1, sign1) = remove_redundant_fraction_sign(ctx, n1, sign1);
    let (n2, sign2) = remove_redundant_fraction_sign(ctx, n2, sign2);
    let (n1, sign1) = normalize_fraction_numerator_sign(ctx, n1, sign1);
    let (n2, sign2) = normalize_fraction_numerator_sign(ctx, n2, sign2);

    (sign1 != sign2
        && compare_expr(ctx, n1, n2) == std::cmp::Ordering::Equal
        && exprs_match_same_denominator(ctx, d1, d2))
        || fractions_match_opposite_scaled_value(ctx, n1, d1, sign1, n2, d2, sign2)
}

fn fractions_match_same_value(
    ctx: &mut cas_ast::Context,
    n1: cas_ast::ExprId,
    d1: cas_ast::ExprId,
    sign1: i8,
    n2: cas_ast::ExprId,
    d2: cas_ast::ExprId,
    sign2: i8,
) -> bool {
    let (n1, sign1) = remove_redundant_fraction_sign(ctx, n1, sign1);
    let (n2, sign2) = remove_redundant_fraction_sign(ctx, n2, sign2);
    let (n1, sign1) = normalize_fraction_numerator_sign(ctx, n1, sign1);
    let (n2, sign2) = normalize_fraction_numerator_sign(ctx, n2, sign2);

    (sign1 == sign2
        && compare_expr(ctx, n1, n2) == std::cmp::Ordering::Equal
        && exprs_match_same_denominator(ctx, d1, d2))
        || fractions_match_same_scaled_value(ctx, n1, d1, sign1, n2, d2, sign2)
}

fn format_add_fraction_desc(
    kind: cas_math::fraction_add_rewrite_support::AddFractionRewriteKind,
) -> &'static str {
    match kind {
        cas_math::fraction_add_rewrite_support::AddFractionRewriteKind::ZeroNumerator => {
            "Add fractions: numerator cancels to 0"
        }
        cas_math::fraction_add_rewrite_support::AddFractionRewriteKind::NumericDenominators => {
            "Add numeric fractions"
        }
        cas_math::fraction_add_rewrite_support::AddFractionRewriteKind::General => {
            "Add fractions: a/b + c/d -> (ad+bc)/bd"
        }
    }
}

fn format_sub_fraction_desc(
    kind: cas_math::fraction_sub_rewrite_support::SubFractionRewriteKind,
) -> &'static str {
    match kind {
        cas_math::fraction_sub_rewrite_support::SubFractionRewriteKind::ZeroNumerator => {
            "Subtract fractions: numerator cancels to 0"
        }
        cas_math::fraction_sub_rewrite_support::SubFractionRewriteKind::NumericDenominators => {
            "Subtract numeric fractions"
        }
        cas_math::fraction_sub_rewrite_support::SubFractionRewriteKind::General => {
            "Subtract fractions: a/b - c/d -> (ad-bc)/bd"
        }
    }
}

fn format_fold_add_into_fraction_desc(swapped: bool) -> &'static str {
    if swapped {
        "Common denominator: p/q + k → (p + k·q)/q"
    } else {
        "Common denominator: k + p/q → (k·q + p)/q"
    }
}

fn format_sub_term_matches_denom_desc() -> &'static str {
    "Common denominator: a - b/a → (a² - b)/a"
}

fn collapse_exact_zero_radical_fraction_result(
    ctx: &mut cas_ast::Context,
    source_expr: cas_ast::ExprId,
    rewritten: cas_ast::ExprId,
    description: &'static str,
) -> Option<Rewrite> {
    let child_rewrite =
        crate::rules::arithmetic::try_build_exact_zero_radical_numerator_const_division_rewrite(
            ctx, rewritten,
        )?;
    Some(
        Rewrite::with_local(
            child_rewrite.new_expr,
            description,
            source_expr,
            child_rewrite.new_expr,
        )
        .requires_all(child_rewrite.required_conditions)
        .assume_all(child_rewrite.assumption_events),
    )
}

define_rule!(
    CancelOppositeFractionsRule,
    "Cancel Opposite Fractions",
    priority: 499,
    |ctx, expr, parent_ctx| {
        let (l, r) = cas_math::expr_destructure::as_add(ctx, expr)?;
        let parts = extract_fraction_pair(ctx, l, r);
        if !parts.is_frac1
            || !parts.is_frac2
            || !should_defer_exact_opposite_fraction_pair_to_additive_cancellation(
                ctx,
                parts.n1,
                parts.d1,
                parts.sign1,
                parts.n2,
                parts.d2,
                parts.sign2,
            )
        {
            return None;
        }

        let decision = crate::oracle_allows_with_hint(
            ctx,
            parent_ctx.domain_mode(),
            parent_ctx.value_domain(),
            &crate::Predicate::NonZero(parts.d1),
            "Cancel exact additive fraction pair",
        );
        if !decision.allow {
            return None;
        }

        let zero = ctx.num(0);
        let mut rewrite = Rewrite::with_local(zero, "Cancel exact additive pairs", expr, zero);
        if decision.assumption.is_some()
            && crate::helpers::prove_nonzero(ctx, parts.d1) != crate::Proof::Proven
        {
            rewrite = rewrite.assume_all(vec![crate::AssumptionEvent::nonzero(ctx, parts.d1)]);
        }
        Some(rewrite)
    }
);

define_rule!(
    CancelEqualFractionsDifferenceRule,
    "Cancel Equal Fractions Difference",
    priority: 499,
    |ctx, expr, parent_ctx| {
        let (l, r) = cas_math::expr_destructure::as_sub(ctx, expr)?;
        let parts = extract_fraction_pair(ctx, l, r);
        if !parts.is_frac1
            || !parts.is_frac2
            || !fractions_match_same_value(
                ctx,
                parts.n1,
                parts.d1,
                parts.sign1,
                parts.n2,
                parts.d2,
                parts.sign2,
            )
        {
            return None;
        }

        let decision = crate::oracle_allows_with_hint(
            ctx,
            parent_ctx.domain_mode(),
            parent_ctx.value_domain(),
            &crate::Predicate::NonZero(parts.d1),
            "Cancel equal fraction difference",
        );
        if !decision.allow {
            return None;
        }

        let zero = ctx.num(0);
        let mut rewrite =
            Rewrite::with_local(zero, "Cancel equal fractions", expr, zero);
        if decision.assumption.is_some()
            && crate::helpers::prove_nonzero(ctx, parts.d1) != crate::Proof::Proven
        {
            rewrite = rewrite.assume_all(vec![crate::AssumptionEvent::nonzero(ctx, parts.d1)]);
        }
        Some(rewrite)
    }
);

// =============================================================================
// Fold Add Into Fraction: k + p/q → (k·q + p)/q
// =============================================================================
//
// This rule combines a simple term with a fraction into a single fraction.
// Unlike AddFractionsRule, this always fires when k is "simple enough"
// (Number, Variable, or simple polynomial) to produce canonical rational form.
//
// Examples:
// - 1 + (x+1)/(2x+1) → (3x+2)/(2x+1)
// - x + 1/y → (x·y + 1)/y
// - 2 + 3/x → (2x + 3)/x
//
// Guards:
// - Skip if inside trig arguments (preserve sin(a + pi/9) structure)
// - Skip if k contains functions (preserve arctan(x) + 1/y structure)

define_rule!(
    FoldAddIntoFractionRule,
    "Common Denominator",
    |ctx, expr, parent_ctx| {
        // Match Add(l, r) where one is a fraction and the other is not
        let (l, r) = cas_math::expr_destructure::as_add(ctx, expr)?;

        // Avoid destroying a larger exact complete-the-square cancellation scope.
        if parent_ctx.has_ancestor_matching(ctx, |c, node_id| {
            crate::rules::arithmetic::maybe_solve_prep_exact_additive_candidate(c, node_id)
        }) {
            return None;
        }

        // Guard: Skip if inside trig function argument
        let inside_trig = parent_ctx.has_ancestor_matching(ctx, |c, node_id| {
            matches!(c.get(node_id), Expr::Function(fn_id, _) if is_trig_function(c, *fn_id))
        });
        // Guard: Skip if this expression is inside a fraction (numerator OR denominator)
        // Let SimplifyComplexFraction handle nested cases properly
        // This prevents preemptive simplification of 1 + x/(x+1) when it's in a complex fraction
        let inside_fraction = parent_ctx
            .has_ancestor_matching(ctx, |c, node_id| matches!(c.get(node_id), Expr::Div(_, _)));

        let plan =
            try_plan_fold_add_into_fraction_rewrite(ctx, expr, l, r, inside_trig, inside_fraction)?;
        Some(Rewrite::new(plan.rewritten).desc(format_fold_add_into_fraction_desc(plan.swapped)))
    }
);

// =============================================================================
// SubTermMatchesDenomRule: a - b/a → (a² - b)/a
// =============================================================================
//
// When the denominator of a subtracted fraction matches the other term,
// combine them into a single fraction. This pattern always reduces nesting
// and is essential for trig simplification:
//   cos(x) - sin²(x)/cos(x) → (cos²(x) - sin²(x))/cos(x) → cos(2x)/cos(x)
//
// This rule complements FoldAddIntoFractionRule (which handles Add only)
// by specifically targeting the Sub case where the denominator matches.
//
// Guard: Skip inside trig arguments and inside fractions (same as FoldAddIntoFraction).

define_rule!(
    SubTermMatchesDenomRule,
    "Combine Same Denominator Sub",
    |ctx, expr, parent_ctx| {
        // Guard: Skip if inside trig function argument
        let inside_trig = parent_ctx.has_ancestor_matching(ctx, |c, node_id| {
            matches!(c.get(node_id), Expr::Function(fn_id, _) if is_trig_function(c, *fn_id))
        });
        let plan = try_plan_sub_term_matches_denom_rewrite(ctx, expr, inside_trig)?;
        Some(Rewrite::new(plan.rewritten).desc(format_sub_term_matches_denom_desc()))
    }
);

define_rule!(
    SymmetricReciprocalSumRule,
    "Combine Symmetric Reciprocals",
    |ctx, expr| {
        let plan = try_plan_symmetric_reciprocal_sum_rewrite(ctx, expr)?;
        Some(
            Rewrite::new(plan.rewritten)
                .desc("Common denominator: 1/(a-1) + 1/(a+1) -> 2*a/(a^2 - 1)"),
        )
    }
);

define_rule!(
    AddFractionsRule,
    "Add Fractions",
    |ctx, expr, parent_ctx| {
        // Use zero-clone destructuring
        let (l, r) = cas_math::expr_destructure::as_add(ctx, expr)?;

        if parent_ctx.has_ancestor_matching(ctx, |c, node_id| {
            crate::rules::arithmetic::maybe_solve_prep_exact_additive_candidate(c, node_id)
        }) {
            return None;
        }

        let parts = extract_fraction_pair(ctx, l, r);
        let (n1, d1, is_frac1) = (parts.n1, parts.d1, parts.is_frac1);
        let (n2, d2, is_frac2) = (parts.n2, parts.d2, parts.is_frac2);
        if is_frac1
            && is_frac2
            && should_defer_exact_opposite_fraction_pair_to_additive_cancellation(
                ctx,
                n1,
                d1,
                parts.sign1,
                n2,
                d2,
                parts.sign2,
            )
        {
            return None;
        }

        if should_block_add_fraction_pair(
            ctx,
            AddFractionPairGuardInput {
                l,
                r,
                n1,
                d1,
                is_frac1,
                n2,
                d2,
                is_frac2,
            },
        ) {
            return None;
        }

        // V2.15.8: Detect same-sign fractions for growth allowance
        // (same_sign = both positive or both negative; opposite = one +, one -)
        let same_sign = parts.sign1 == parts.sign2;

        // Context-aware gating: avoid combining symbol + pi-const inside trig functions.
        let inside_trig = parent_ctx.has_ancestor_matching(ctx, |c, node_id| {
            matches!(c.get(node_id), Expr::Function(fn_id, _) if is_trig_function(c, *fn_id))
        });

        let plan = plan_add_fraction_rewrite_with(
            ctx,
            AddFractionRewriteInput {
                expr,
                l,
                r,
                n1,
                d1,
                n2,
                d2,
                same_sign,
                inside_trig,
            },
            crate::expand::expand,
        )?;
        let description = format_add_fraction_desc(plan.kind);
        if let Some(rewrite) =
            collapse_exact_zero_radical_fraction_result(ctx, expr, plan.rewritten, description)
        {
            return Some(rewrite);
        }
        Some(Rewrite::new(plan.rewritten).desc(description))
    }
);

// =============================================================================
// SubFractionsRule: a/b - c/d → (a·d - c·b) / (b·d)
// =============================================================================
//
// Combines two fractions being subtracted into a single fraction.
// The resulting numerator goes through normal simplification which can prove
// it equals 0 when the fractions were algebraically equal (e.g., different
// representations of the same rational expression).
//
// This handles cases that SubSelfToZeroRule misses because the two fractions
// have structurally different (but algebraically equivalent) numerators/denominators
// from independent simplification paths.
//
// Example: ((u+1)·(u·x+1)+u)/(u·(u+1)) - (u²x+ux+2u+1)/(u²+u)
//        → (cross_product) / (common_den) → 0/den → 0
//
// Guards:
// - Both sides must be fractions (direct Div or FractionParts)
// - Skip if inside trig arguments (preserve sin(a - pi/9) structure)
// - Skip function-containing expressions mixed with constant fractions
// - Same complexity heuristics as AddFractionsRule

define_rule!(
    SubFractionsRule,
    "Subtract Fractions",
    |ctx, expr, parent_ctx| {
        let (l, r) = cas_math::expr_destructure::as_sub(ctx, expr)?;

        let parts = extract_fraction_pair(ctx, l, r);
        let (n1, d1, is_frac1) = (parts.n1, parts.d1, parts.is_frac1);
        let (n2, d2, is_frac2) = (parts.n2, parts.d2, parts.is_frac2);

        // Guard: Skip if inside trig function argument
        let inside_trig = parent_ctx.has_ancestor_matching(ctx, |c, node_id| {
            matches!(c.get(node_id), Expr::Function(fn_id, _) if is_trig_function(c, *fn_id))
        });
        if is_frac1 && is_frac2 && compare_expr(ctx, d1, d2) == std::cmp::Ordering::Equal {
            let residual = ctx.add(Expr::Sub(n1, n2));
            if crate::rules::arithmetic::maybe_solve_prep_exact_additive_candidate(ctx, residual) {
                return None;
            }
        }
        if should_block_sub_fraction_pair(
            ctx,
            SubFractionPairGuardInput {
                l,
                r,
                n1,
                d1,
                is_frac1,
                n2,
                d2,
                is_frac2,
                inside_trig,
            },
        ) {
            return None;
        }

        let plan = plan_sub_fraction_rewrite_with(ctx, n1, n2, d1, d2, crate::expand::expand);
        let description = format_sub_fraction_desc(plan.kind);
        if let Some(rewrite) =
            collapse_exact_zero_radical_fraction_result(ctx, expr, plan.rewritten, description)
        {
            return Some(rewrite);
        }
        Some(Rewrite::new(plan.rewritten).desc(description))
    }
);

#[cfg(test)]
mod tests {
    use super::{
        should_defer_exact_opposite_fraction_pair_to_additive_cancellation,
        CancelEqualFractionsDifferenceRule, CancelOppositeFractionsRule, SubFractionsRule,
    };
    use crate::parent_context::ParentContext;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_math::fraction_pair_support::extract_fraction_pair;
    use cas_parser::parse;

    #[test]
    fn sub_fractions_rule_skips_same_denominator_complete_square_difference() {
        let mut ctx = Context::new();
        let expr = parse(
            "((a*x^2 + b*x + c)/q) - ((a*(x + b/(2*a))^2 + c - b^2/(4*a))/q)",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let rewrite = SubFractionsRule.apply(&mut ctx, expr, &ParentContext::root());
        assert!(rewrite.is_none());
    }

    #[test]
    fn detects_opposite_fraction_pair_with_sqrt_and_half_power_denominator() {
        let mut ctx = Context::new();
        let expr = parse(
            "-2/(sqrt(2*x+3)*(8*x+13)) + 2/((2*x+3)^(1/2)*(8*x+13))",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));
        let cas_ast::Expr::Add(lhs, rhs) = ctx.get(expr).clone() else {
            panic!("expected add");
        };
        let pair = extract_fraction_pair(&mut ctx, lhs, rhs);

        assert!(pair.is_frac1);
        assert!(pair.is_frac2);
        assert!(
            should_defer_exact_opposite_fraction_pair_to_additive_cancellation(
                &mut ctx, pair.n1, pair.d1, pair.sign1, pair.n2, pair.d2, pair.sign2,
            )
        );

        let rewrite = CancelOppositeFractionsRule.apply(&mut ctx, expr, &ParentContext::root());
        assert!(rewrite.is_some());
    }

    #[test]
    fn detects_equal_fraction_difference_with_absorbed_numeric_denominator_content() {
        let mut ctx = Context::new();
        let expr = parse(
            "3/(2*sqrt(2*x+1)*(9*x+5)) - 3/(sqrt(2*x+1)*(18*x+10))",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let rewrite =
            CancelEqualFractionsDifferenceRule.apply(&mut ctx, expr, &ParentContext::root());
        assert!(rewrite.is_some());
    }

    #[test]
    fn detects_equal_fraction_difference_with_common_rational_scale() {
        let mut ctx = Context::new();
        let expr = parse(
            "3/(2*sqrt(2*x+1)*(9*x+5)) - 6/(4*sqrt(2*x+1)*(9*x+5))",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let rewrite =
            CancelEqualFractionsDifferenceRule.apply(&mut ctx, expr, &ParentContext::root());
        assert!(rewrite.is_some());
    }

    #[test]
    fn detects_opposite_fraction_pair_with_common_rational_scale() {
        let mut ctx = Context::new();
        let expr = parse(
            "-3/(2*sqrt(2*x+1)*(9*x+5)) + 6/(4*sqrt(2*x+1)*(9*x+5))",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let rewrite = CancelOppositeFractionsRule.apply(&mut ctx, expr, &ParentContext::root());
        assert!(rewrite.is_some());
    }

    #[test]
    fn detects_equal_fraction_difference_with_scaled_linear_numerator_factor() {
        let mut ctx = Context::new();
        let expr = parse(
            "(3*x+1)/(2*sqrt(x)*(x*(x+1)^2+1)) - (6*x+2)/(4*sqrt(x)*(x*(x+1)^2+1))",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let rewrite =
            CancelEqualFractionsDifferenceRule.apply(&mut ctx, expr, &ParentContext::root());
        assert!(rewrite.is_some());
    }

    #[test]
    fn detects_opposite_fraction_pair_with_scaled_linear_numerator_factor() {
        let mut ctx = Context::new();
        let expr = parse(
            "-(3*x+1)/(2*sqrt(x)*(x*(x+1)^2+1)) + (6*x+2)/(4*sqrt(x)*(x*(x+1)^2+1))",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let rewrite = CancelOppositeFractionsRule.apply(&mut ctx, expr, &ParentContext::root());
        assert!(rewrite.is_some());
    }

    #[test]
    fn detects_opposite_fraction_pair_with_distributed_negative_scaled_linear_factor() {
        let mut ctx = Context::new();
        let expr = parse(
            "(-3*x-1)/(2*sqrt(x)*(x*(x+1)^2+1)) + (6*x+2)/(4*sqrt(x)*(x*(x+1)^2+1))",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let rewrite = CancelOppositeFractionsRule.apply(&mut ctx, expr, &ParentContext::root());
        assert!(rewrite.is_some());
    }

    #[test]
    fn detects_opposite_fraction_pair_with_half_power_scaled_linear_factor() {
        let mut ctx = Context::new();
        let expr = parse(
            "-(3*x+1)/(2*sqrt(x)*(x*(x+1)^2+1)) + (6*x+2)/(4*x^(1/2)*(x*(x+1)^2+1))",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let rewrite = CancelOppositeFractionsRule.apply(&mut ctx, expr, &ParentContext::root());
        assert!(rewrite.is_some());
    }

    #[test]
    fn detects_opposite_fraction_pair_with_distributed_scaled_denominator_product() {
        let mut ctx = Context::new();
        let expr = parse(
            "-(3*x+1)/(2*sqrt(x)*(x*(x+1)^2+1)) + \
                (6*x+2)/(4*x^(1/2)+4*x^(3/2)*(x+1)^2)",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let rewrite = CancelOppositeFractionsRule.apply(&mut ctx, expr, &ParentContext::root());
        assert!(rewrite.is_some());
    }

    #[test]
    fn detects_opposite_fraction_pair_with_shifted_sqrt_half_power_denominator() {
        let mut ctx = Context::new();
        let expr = parse(
            "-(2*sqrt(x)-1)/(2*x*sqrt(x)*(sqrt(x)-1)^2) + \
                (2*x^(1/2)-1)/(2*x^(3/2)*(x^(1/2)-1)^2)",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let rewrite = CancelOppositeFractionsRule.apply(&mut ctx, expr, &ParentContext::root());
        assert!(rewrite.is_some());
    }
}
