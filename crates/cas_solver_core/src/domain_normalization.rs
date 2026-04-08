//! Condition normalization, deduplication, and dominance rules.

use crate::domain_condition::ImplicitCondition;
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_domain::{
    exprs_equivalent, exprs_equivalent_up_to_sign, is_abs_of, is_odd_power_of,
    is_positive_power_of_base, is_product_dominated_by_positives,
};
use cas_math::expr_extract::{extract_abs_argument_view, extract_sqrt_argument_view};
use cas_math::expr_nary::build_balanced_mul;
use cas_math::expr_normalization::{
    extract_even_positive_power_base, normalize_condition_expr as normalize_condition_expr_math,
};
use cas_math::factor::factor;
use cas_math::numeric_eval::as_rational_const;
use cas_math::prove_sign::{prove_nonnegative_depth_with, prove_positive_depth_with};
use cas_math::tri_proof::TriProof;
use num_traits::{One, Signed};

const DISPLAY_SIGN_PROOF_DEPTH: usize = 12;

/// Normalize an expression for display in conditions.
pub fn normalize_condition_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    normalize_condition_expr_math(ctx, expr)
}

/// Normalize a condition for display (applies normalization to the inner expression).
pub fn normalize_condition(ctx: &mut Context, cond: &ImplicitCondition) -> ImplicitCondition {
    // Positive(x^even) -> NonZero(x) because x^(2k) > 0 <=> x != 0 for reals.
    if let ImplicitCondition::Positive(e) = cond {
        if let Some(base) = extract_even_positive_power_base(ctx, *e) {
            let normalized_base = normalize_condition_expr(ctx, base);
            return ImplicitCondition::NonZero(normalized_base);
        }
    }

    let normalized_expr = match cond {
        ImplicitCondition::NonNegative(e) => normalize_condition_expr(ctx, *e),
        ImplicitCondition::Positive(e) => normalize_condition_expr(ctx, *e),
        ImplicitCondition::NonZero(e) => {
            let stripped = strip_nonzero_scalar_factors_for_display(ctx, *e);
            normalize_condition_expr(ctx, stripped)
        }
    };

    match cond {
        ImplicitCondition::NonNegative(_) => ImplicitCondition::NonNegative(normalized_expr),
        ImplicitCondition::Positive(_) => ImplicitCondition::Positive(normalized_expr),
        ImplicitCondition::NonZero(_) => ImplicitCondition::NonZero(normalized_expr),
    }
}

fn strip_nonzero_scalar_factors_for_display(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => strip_nonzero_scalar_factors_for_display(ctx, inner),
        Expr::Mul(_, _) => {
            let mut symbolic_factors = Vec::new();
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if as_rational_const(ctx, factor).is_none() {
                    symbolic_factors.push(strip_nonzero_scalar_factors_for_display(ctx, factor));
                }
            }

            if symbolic_factors.is_empty() {
                expr
            } else {
                build_balanced_mul(ctx, &symbolic_factors)
            }
        }
        Expr::Div(num, den) => {
            let numerator_is_numeric = as_rational_const(ctx, num).is_some();
            let denominator_is_numeric = as_rational_const(ctx, den).is_some();
            match (numerator_is_numeric, denominator_is_numeric) {
                (true, false) => strip_nonzero_scalar_factors_for_display(ctx, den),
                (false, true) => strip_nonzero_scalar_factors_for_display(ctx, num),
                _ => expr,
            }
        }
        _ => expr,
    }
}

fn is_intrinsically_positive_real(ctx: &Context, expr: ExprId) -> bool {
    prove_positive_depth_with(
        ctx,
        expr,
        DISPLAY_SIGN_PROOF_DEPTH,
        true,
        |_inner_ctx, _inner_expr, _inner_depth| TriProof::Unknown,
    )
    .is_proven()
}

fn is_intrinsically_nonnegative_real(ctx: &Context, expr: ExprId) -> bool {
    prove_nonnegative_depth_with(
        ctx,
        expr,
        DISPLAY_SIGN_PROOF_DEPTH,
        true,
        |_inner_ctx, _inner_expr, _inner_depth| TriProof::Unknown,
    )
    .is_proven()
}

fn is_intrinsically_nonnegative_real_after_factoring(ctx: &mut Context, expr: ExprId) -> bool {
    if is_intrinsically_nonnegative_real(ctx, expr) {
        return true;
    }

    let factored = factor(ctx, expr);
    factored != expr && is_intrinsically_nonnegative_real(ctx, factored)
}

fn extract_sqrt_like_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Some(arg) = extract_sqrt_argument_view(ctx, expr) {
        return Some(arg);
    }

    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let half = num_rational::BigRational::new(1.into(), 2.into());
    (as_rational_const(ctx, *exp).as_ref() == Some(&half)).then_some(*base)
}

fn exprs_equivalent_or_same_sqrt_like_base(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    exprs_equivalent_up_to_sign(ctx, left, right)
        || match (
            extract_sqrt_like_base(ctx, left),
            extract_sqrt_like_base(ctx, right),
        ) {
            (Some(left_base), Some(right_base)) => exprs_equivalent(ctx, left_base, right_base),
            _ => false,
        }
}

fn positive_target_from_nonzero_and_nonnegative(
    ctx: &Context,
    nonzero_expr: ExprId,
    nonnegative_expr: ExprId,
) -> Option<ExprId> {
    let nonzero_core = extract_abs_argument_view(ctx, nonzero_expr).unwrap_or(nonzero_expr);

    if exprs_equivalent_up_to_sign(ctx, nonzero_core, nonnegative_expr) {
        return Some(nonnegative_expr);
    }

    let nonzero_base = extract_sqrt_like_base(ctx, nonzero_core)?;
    if exprs_equivalent(ctx, nonzero_base, nonnegative_expr) {
        return Some(nonnegative_expr);
    }

    None
}

fn combine_nonzero_nonnegative_into_positive(
    ctx: &Context,
    conditions: &mut Vec<ImplicitCondition>,
) {
    let mut to_remove: Vec<usize> = Vec::new();
    let mut to_add: Vec<ImplicitCondition> = Vec::new();

    for (i, cond) in conditions.iter().enumerate() {
        let ImplicitCondition::NonZero(nz_expr) = cond else {
            continue;
        };

        for (j, other) in conditions.iter().enumerate() {
            if i == j {
                continue;
            }

            let ImplicitCondition::NonNegative(nn_expr) = other else {
                continue;
            };

            let Some(target) =
                positive_target_from_nonzero_and_nonnegative(ctx, *nz_expr, *nn_expr)
            else {
                continue;
            };

            to_remove.push(i);
            to_remove.push(j);

            let positive = ImplicitCondition::Positive(target);
            let already_present = conditions
                .iter()
                .any(|existing| conditions_equivalent(ctx, existing, &positive))
                || to_add
                    .iter()
                    .any(|existing| conditions_equivalent(ctx, existing, &positive));
            if !already_present {
                to_add.push(positive);
            }
            break;
        }
    }

    to_remove.sort();
    to_remove.dedup();
    for idx in to_remove.into_iter().rev() {
        conditions.remove(idx);
    }
    conditions.extend(to_add);
}

fn has_nonzero_display_condition(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
) -> bool {
    conditions.iter().enumerate().any(|(idx, cond)| {
        idx != skip_index
            && matches!(
                cond,
                ImplicitCondition::NonZero(other)
                    if exprs_equivalent_or_same_sqrt_like_base(ctx, *other, expr)
            )
    })
}

fn has_nonnegative_display_condition(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
) -> bool {
    conditions.iter().enumerate().any(|(idx, cond)| {
        idx != skip_index
            && matches!(
                cond,
                ImplicitCondition::NonNegative(other) | ImplicitCondition::Positive(other)
                    if exprs_equivalent_or_same_sqrt_like_base(ctx, *other, expr)
            )
    })
}

fn has_positive_display_condition(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
) -> bool {
    conditions.iter().enumerate().any(|(idx, cond)| {
        idx != skip_index
            && matches!(
                cond,
                ImplicitCondition::Positive(other)
                    if exprs_equivalent_or_same_sqrt_like_base(ctx, *other, expr)
            )
    })
}

fn is_nonnegative_under_display_conditions(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
    depth: usize,
) -> bool {
    if depth == 0 {
        return false;
    }

    if has_nonnegative_display_condition(ctx, conditions, skip_index, expr)
        || is_intrinsically_nonnegative_real(ctx, expr)
    {
        return true;
    }

    if let Some(base) = extract_sqrt_like_base(ctx, expr) {
        return is_nonnegative_under_display_conditions(
            ctx,
            conditions,
            skip_index,
            base,
            depth - 1,
        );
    }

    false
}

fn is_positive_under_display_conditions(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
    depth: usize,
) -> bool {
    if depth == 0 {
        return false;
    }

    if has_positive_display_condition(ctx, conditions, skip_index, expr)
        || is_intrinsically_positive_real(ctx, expr)
        || (has_nonzero_display_condition(ctx, conditions, skip_index, expr)
            && is_nonnegative_under_display_conditions(
                ctx,
                conditions,
                skip_index,
                expr,
                depth - 1,
            ))
    {
        return true;
    }

    if let Some(base) = extract_even_positive_power_base(ctx, expr) {
        if has_nonzero_display_condition(ctx, conditions, skip_index, base) {
            return true;
        }
    }

    if let Some(base) = extract_sqrt_like_base(ctx, expr) {
        if is_positive_under_display_conditions(ctx, conditions, skip_index, base, depth - 1) {
            return true;
        }
    }

    if let Some(arg) = extract_abs_argument_view(ctx, expr) {
        if has_nonzero_display_condition(ctx, conditions, skip_index, arg)
            || is_positive_under_display_conditions(ctx, conditions, skip_index, arg, depth - 1)
        {
            return true;
        }
    }

    match ctx.get(expr) {
        Expr::Add(left, right) => {
            (is_positive_under_display_conditions(ctx, conditions, skip_index, *left, depth - 1)
                && is_nonnegative_under_display_conditions(
                    ctx,
                    conditions,
                    skip_index,
                    *right,
                    depth - 1,
                ))
                || (is_positive_under_display_conditions(
                    ctx,
                    conditions,
                    skip_index,
                    *right,
                    depth - 1,
                ) && is_nonnegative_under_display_conditions(
                    ctx,
                    conditions,
                    skip_index,
                    *left,
                    depth - 1,
                ))
        }
        Expr::Mul(left, right) | Expr::Div(left, right) => {
            is_positive_under_display_conditions(ctx, conditions, skip_index, *left, depth - 1)
                && is_positive_under_display_conditions(
                    ctx,
                    conditions,
                    skip_index,
                    *right,
                    depth - 1,
                )
        }
        _ => false,
    }
}

/// Check if two conditions are equivalent (same type and expression-equivalent up to sign).
pub fn conditions_equivalent(
    ctx: &Context,
    c1: &ImplicitCondition,
    c2: &ImplicitCondition,
) -> bool {
    let (e1, e2) = match (c1, c2) {
        (ImplicitCondition::NonNegative(a), ImplicitCondition::NonNegative(b)) => (*a, *b),
        (ImplicitCondition::Positive(a), ImplicitCondition::Positive(b)) => (*a, *b),
        (ImplicitCondition::NonZero(a), ImplicitCondition::NonZero(b)) => (*a, *b),
        _ => return false,
    };

    exprs_equivalent_up_to_sign(ctx, e1, e2)
}

/// Normalize and deduplicate a list of conditions for display.
pub fn normalize_and_dedupe_conditions(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
) -> Vec<ImplicitCondition> {
    let mut result: Vec<ImplicitCondition> = Vec::new();

    for cond in conditions {
        for normalized in expand_condition_for_display(ctx, cond) {
            let is_duplicate = result
                .iter()
                .any(|existing| conditions_equivalent(ctx, existing, &normalized));

            if !is_duplicate {
                result.push(normalized);
            }
        }
    }

    combine_nonzero_nonnegative_into_positive(ctx, &mut result);
    apply_dominance_rules(ctx, &mut result);
    result
}

fn expand_condition_for_display(
    ctx: &mut Context,
    cond: &ImplicitCondition,
) -> Vec<ImplicitCondition> {
    match cond {
        ImplicitCondition::NonZero(expr) => expand_nonzero_condition_for_display(ctx, *expr),
        ImplicitCondition::Positive(expr) => {
            if let Some(arg) = extract_abs_argument_view(ctx, *expr) {
                return expand_nonzero_condition_for_display(ctx, arg);
            }
            vec![normalize_condition(ctx, cond)]
        }
        _ => vec![normalize_condition(ctx, cond)],
    }
}

fn expand_nonzero_condition_for_display(ctx: &mut Context, expr: ExprId) -> Vec<ImplicitCondition> {
    let core_expr = extract_abs_argument_view(ctx, expr).unwrap_or(expr);
    let stripped_expr = strip_nonzero_scalar_factors_for_display(ctx, core_expr);
    let normalized_expr = normalize_condition_expr(ctx, stripped_expr);

    if let Some(base) = extract_even_positive_power_base(ctx, normalized_expr) {
        return vec![ImplicitCondition::NonZero(normalize_condition_expr(
            ctx, base,
        ))];
    }

    let factored = factor(ctx, normalized_expr);
    let mut atomic_factors = Vec::new();
    collect_nonzero_atomic_factors(ctx, factored, &mut atomic_factors);

    if atomic_factors.len() <= 1 {
        return vec![ImplicitCondition::NonZero(normalized_expr)];
    }

    let mut expanded = Vec::new();
    for factor_expr in atomic_factors {
        let normalized_factor = normalize_condition_expr(ctx, factor_expr);
        let cond = ImplicitCondition::NonZero(normalized_factor);
        if !expanded
            .iter()
            .any(|existing| conditions_equivalent(ctx, existing, &cond))
        {
            expanded.push(cond);
        }
    }

    if expanded.is_empty() {
        vec![ImplicitCondition::NonZero(normalized_expr)]
    } else {
        expanded
    }
}

fn collect_nonzero_atomic_factors(ctx: &Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            collect_nonzero_atomic_factors(ctx, *l, factors);
            collect_nonzero_atomic_factors(ctx, *r, factors);
        }
        Expr::Div(num, den) => {
            collect_nonzero_atomic_factors(ctx, *num, factors);
            collect_nonzero_atomic_factors(ctx, *den, factors);
        }
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() && n.is_positive() {
                    factors.push(*base);
                    return;
                }
            }
            factors.push(expr);
        }
        Expr::Number(_) => {}
        _ => factors.push(expr),
    }
}

fn is_factorial_of_arg(ctx: &Context, expr: ExprId, arg: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.sym_name(*fn_id), "fact" | "factorial")
                && exprs_equivalent(ctx, args[0], arg)
    )
}

fn as_unit_reciprocal_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(num, den) if as_rational_const(ctx, *num).is_some_and(|n| n.is_one()) => {
            Some(*den)
        }
        _ => None,
    }
}

fn extract_unit_reciprocal_sum_denominators(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    let left_den = as_unit_reciprocal_denominator(ctx, *left)?;
    let right_den = as_unit_reciprocal_denominator(ctx, *right)?;
    Some((left_den, right_den))
}

fn is_sum_of_terms(ctx: &Context, expr: ExprId, a: ExprId, b: ExprId) -> bool {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return false;
    };

    (exprs_equivalent_up_to_sign(ctx, *left, a) && exprs_equivalent_up_to_sign(ctx, *right, b))
        || (exprs_equivalent_up_to_sign(ctx, *left, b)
            && exprs_equivalent_up_to_sign(ctx, *right, a))
}

fn reciprocal_sum_nonzero_is_dominated(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
) -> bool {
    let Some((left_den, right_den)) = extract_unit_reciprocal_sum_denominators(ctx, expr) else {
        return false;
    };

    let mut has_left = false;
    let mut has_right = false;
    let mut has_sum = false;

    for (idx, condition) in conditions.iter().enumerate() {
        if idx == skip_index {
            continue;
        }
        let ImplicitCondition::NonZero(other_expr) = condition else {
            continue;
        };

        if exprs_equivalent_up_to_sign(ctx, *other_expr, left_den) {
            has_left = true;
        } else if exprs_equivalent_up_to_sign(ctx, *other_expr, right_den) {
            has_right = true;
        } else if is_sum_of_terms(ctx, *other_expr, left_den, right_den) {
            has_sum = true;
        }
    }

    has_left && has_right && has_sum
}

fn apply_dominance_rules(ctx: &mut Context, conditions: &mut Vec<ImplicitCondition>) {
    let mut to_remove: Vec<usize> = Vec::new();

    for (i, cond) in conditions.iter().enumerate() {
        for (j, other) in conditions.iter().enumerate() {
            if i == j {
                continue;
            }

            match (cond, other) {
                (ImplicitCondition::NonZero(nz_expr), ImplicitCondition::Positive(pos_expr)) => {
                    if exprs_equivalent(ctx, *nz_expr, *pos_expr)
                        || is_abs_of(ctx, *nz_expr, *pos_expr)
                        || is_abs_of(ctx, *pos_expr, *nz_expr)
                    {
                        to_remove.push(i);
                        break;
                    }
                }
                (
                    ImplicitCondition::Positive(derived_expr),
                    ImplicitCondition::Positive(pos_expr),
                ) => {
                    if is_abs_of(ctx, *derived_expr, *pos_expr)
                        || is_positive_power_of_base(ctx, *derived_expr, *pos_expr)
                    {
                        to_remove.push(i);
                        break;
                    }
                }
                (
                    ImplicitCondition::NonNegative(nn_expr),
                    ImplicitCondition::Positive(pos_expr),
                ) => {
                    if exprs_equivalent(ctx, *nn_expr, *pos_expr)
                        || is_odd_power_of(ctx, *nn_expr, *pos_expr)
                    {
                        to_remove.push(i);
                        break;
                    }
                }
                (
                    ImplicitCondition::NonNegative(derived_expr),
                    ImplicitCondition::NonNegative(base_expr),
                ) => {
                    if is_odd_power_of(ctx, *derived_expr, *base_expr) {
                        to_remove.push(i);
                        break;
                    }
                }
                (ImplicitCondition::NonZero(nz_expr), ImplicitCondition::NonNegative(nn_expr)) => {
                    if is_factorial_of_arg(ctx, *nz_expr, *nn_expr) {
                        to_remove.push(i);
                        break;
                    }
                }
                _ => {}
            }
        }

        if !to_remove.contains(&i) {
            if let ImplicitCondition::Positive(pos_expr) = cond {
                if is_positive_under_display_conditions(
                    ctx,
                    conditions,
                    i,
                    *pos_expr,
                    DISPLAY_SIGN_PROOF_DEPTH,
                ) {
                    to_remove.push(i);
                    continue;
                }
            }

            if let ImplicitCondition::NonNegative(nn_expr) = cond {
                if is_intrinsically_nonnegative_real_after_factoring(ctx, *nn_expr) {
                    to_remove.push(i);
                    continue;
                }
            }

            if let ImplicitCondition::NonZero(nz_expr) = cond {
                if is_positive_under_display_conditions(
                    ctx,
                    conditions,
                    i,
                    *nz_expr,
                    DISPLAY_SIGN_PROOF_DEPTH,
                ) {
                    to_remove.push(i);
                    continue;
                }

                if reciprocal_sum_nonzero_is_dominated(ctx, conditions, i, *nz_expr) {
                    to_remove.push(i);
                    continue;
                }
            }

            if let ImplicitCondition::Positive(prod_expr) = cond {
                let other_positive_exprs: Vec<ExprId> = conditions
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, condition)| {
                        if idx == i {
                            return None;
                        }
                        match condition {
                            ImplicitCondition::Positive(e) => Some(*e),
                            _ => None,
                        }
                    })
                    .collect();

                if is_product_dominated_by_positives(ctx, *prod_expr, &other_positive_exprs) {
                    to_remove.push(i);
                    continue;
                }

                let factored = factor(ctx, *prod_expr);
                if factored != *prod_expr
                    && is_product_dominated_by_positives(ctx, factored, &other_positive_exprs)
                {
                    to_remove.push(i);
                }
            }
        }
    }

    to_remove.sort();
    to_remove.dedup();
    for i in to_remove.into_iter().rev() {
        conditions.remove(i);
    }
}

/// Render conditions for display, applying normalization and deduplication.
pub fn render_conditions_normalized(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
) -> Vec<String> {
    let normalized = normalize_and_dedupe_conditions(ctx, conditions);
    normalized.iter().map(|c| c.display(ctx)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn nonnegative_factorial_argument_dominates_factorial_nonzero_display_condition() {
        let mut ctx = Context::new();
        let n = parse("n", &mut ctx).expect("parse n");
        let fact_n = parse("n!", &mut ctx).expect("parse n!");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonNegative(n),
                ImplicitCondition::NonZero(fact_n),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::NonNegative(n)]);
    }

    #[test]
    fn nonzero_constant_multiple_normalizes_to_base_condition() {
        let mut ctx = Context::new();
        let scaled = parse("2*a", &mut ctx).expect("parse scaled");
        let base = parse("a", &mut ctx).expect("parse base");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(scaled),
                ImplicitCondition::NonZero(base),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::NonZero(base)]);
    }

    #[test]
    fn nonzero_fractional_multiple_normalizes_to_base_condition() {
        let mut ctx = Context::new();
        let scaled = parse("a/2", &mut ctx).expect("parse scaled");
        let base = parse("a", &mut ctx).expect("parse base");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(scaled),
                ImplicitCondition::NonZero(base),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::NonZero(base)]);
    }

    #[test]
    fn positive_abs_dominates_base_nonzero_condition() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse x");
        let abs_x = parse("abs(x)", &mut ctx).expect("parse abs(x)");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(x),
                ImplicitCondition::Positive(abs_x),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::NonZero(x)]);
    }

    #[test]
    fn atomic_positive_factors_dominate_composite_log_argument_positive_condition() {
        let mut ctx = Context::new();
        let t = parse("t", &mut ctx).expect("parse t");
        let y = parse("y", &mut ctx).expect("parse y");
        let z = parse("z", &mut ctx).expect("parse z");
        let x = parse("x", &mut ctx).expect("parse x");
        let abs_x = parse("abs(x)", &mut ctx).expect("parse abs(x)");
        let composite = parse("y*x^2/(t*z)", &mut ctx).expect("parse composite");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(t),
                ImplicitCondition::Positive(y),
                ImplicitCondition::Positive(composite),
                ImplicitCondition::Positive(z),
                ImplicitCondition::Positive(abs_x),
            ],
        );

        assert_eq!(
            normalized,
            vec![
                ImplicitCondition::Positive(t),
                ImplicitCondition::Positive(y),
                ImplicitCondition::Positive(z),
                ImplicitCondition::NonZero(x),
            ]
        );
    }

    #[test]
    fn factored_positive_factors_dominate_unfactored_difference_of_squares_condition() {
        let mut ctx = Context::new();
        let x_plus_y = parse("x + y", &mut ctx).expect("parse x + y");
        let x_minus_y = parse("x - y", &mut ctx).expect("parse x - y");
        let composite = parse("x^2 - y^2", &mut ctx).expect("parse composite");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(x_plus_y),
                ImplicitCondition::Positive(x_minus_y),
                ImplicitCondition::Positive(composite),
            ],
        );

        assert_eq!(normalized.len(), 2);
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x_plus_y))
        }));
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x_minus_y))
        }));
    }

    #[test]
    fn reciprocal_sum_nonzero_is_dominated_by_atomic_denominators_and_sum() {
        let mut ctx = Context::new();
        let reciprocal_sum = parse("1/a + 1/b", &mut ctx).expect("parse reciprocal sum");
        let sum = parse("a + b", &mut ctx).expect("parse sum");
        let a = parse("a", &mut ctx).expect("parse a");
        let b = parse("b", &mut ctx).expect("parse b");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(reciprocal_sum),
                ImplicitCondition::NonZero(sum),
                ImplicitCondition::NonZero(a),
                ImplicitCondition::NonZero(b),
            ],
        );

        assert_eq!(normalized.len(), 3);
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(sum)) }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(a)) }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(b)) }));
    }

    #[test]
    fn intrinsically_positive_nonzero_display_condition_is_dropped() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x^2 + 1)^3", &mut ctx).expect("parse expr");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(expr)]);

        assert!(normalized.is_empty());
    }

    #[test]
    fn positive_sum_under_other_display_conditions_drops_composite_nonzero_condition() {
        let mut ctx = Context::new();
        let reciprocal_sum = parse("1/sqrt(u) + 1", &mut ctx).expect("parse reciprocal sum");
        let transformed_den =
            parse("u^(1/2) + u", &mut ctx).expect("parse transformed denominator");
        let sqrt_u = parse("sqrt(u)", &mut ctx).expect("parse sqrt(u)");
        let u = parse("u", &mut ctx).expect("parse u");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(reciprocal_sum),
                ImplicitCondition::NonZero(sqrt_u),
                ImplicitCondition::NonNegative(u),
                ImplicitCondition::NonZero(transformed_den),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::Positive(u)]);
    }

    #[test]
    fn intrinsically_nonnegative_display_condition_is_dropped() {
        let mut ctx = Context::new();
        let expr = parse("2*sqrt(x^2 + 1) + x^2 + 2", &mut ctx).expect("parse expr");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonNegative(expr)]);

        assert!(normalized.is_empty());
    }

    #[test]
    fn factored_perfect_square_nonnegative_display_condition_is_dropped() {
        let mut ctx = Context::new();
        let expr = parse("a^2 + 2*a*b + b^2", &mut ctx).expect("parse expr");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonNegative(expr)]);

        assert!(normalized.is_empty());
    }

    #[test]
    fn nonzero_sqrt_and_nonnegative_base_combine_into_positive_base() {
        let mut ctx = Context::new();
        let sqrt_u = parse("sqrt(u)", &mut ctx).expect("parse sqrt(u)");
        let u = parse("u", &mut ctx).expect("parse u");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(sqrt_u),
                ImplicitCondition::NonNegative(u),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::Positive(u)]);
    }

    #[test]
    fn nonzero_abs_sqrt_and_nonnegative_base_combine_into_positive_base() {
        let mut ctx = Context::new();
        let abs_sqrt_u = parse("abs(sqrt(u))", &mut ctx).expect("parse abs(sqrt(u))");
        let u = parse("u", &mut ctx).expect("parse u");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(abs_sqrt_u),
                ImplicitCondition::NonNegative(u),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::Positive(u)]);
    }

    #[test]
    fn positive_abs_sqrt_is_dominated_by_positive_base() {
        let mut ctx = Context::new();
        let abs_sqrt_u = parse("abs(sqrt(u))", &mut ctx).expect("parse abs(sqrt(u))");
        let u = parse("u", &mut ctx).expect("parse u");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(abs_sqrt_u),
                ImplicitCondition::Positive(u),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::Positive(u)]);
    }

    #[test]
    fn nonzero_abs_product_expands_to_atomic_factors() {
        let mut ctx = Context::new();
        let abs_product = parse("abs((x-1)*(x+1))", &mut ctx).expect("parse abs product");
        let x_minus_1 = parse("x - 1", &mut ctx).expect("parse x - 1");
        let x_plus_1 = parse("x + 1", &mut ctx).expect("parse x + 1");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(abs_product)]);

        assert_eq!(normalized.len(), 2);
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_minus_1))
        }));
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_plus_1))
        }));
    }

    #[test]
    fn positive_abs_product_expands_to_atomic_factors() {
        let mut ctx = Context::new();
        let abs_product = parse("abs(x*y)", &mut ctx).expect("parse abs product");
        let x = parse("x", &mut ctx).expect("parse x");
        let y = parse("y", &mut ctx).expect("parse y");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::Positive(abs_product)]);

        assert_eq!(normalized.len(), 2);
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x)) }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(y)) }));
    }

    #[test]
    fn nonzero_abs_quotient_expands_to_atomic_numerator_and_denominator() {
        let mut ctx = Context::new();
        let abs_quotient = parse("abs(x/(x+1))", &mut ctx).expect("parse abs quotient");
        let x = parse("x", &mut ctx).expect("parse x");
        let x_plus_1 = parse("x + 1", &mut ctx).expect("parse x + 1");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(abs_quotient)]);

        assert_eq!(normalized.len(), 2);
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x)) }));
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_plus_1))
        }));
    }

    #[test]
    fn positive_abs_quotient_expands_to_atomic_numerator_and_denominator() {
        let mut ctx = Context::new();
        let abs_quotient = parse("abs((x-1)/(x+1))", &mut ctx).expect("parse abs quotient");
        let x_minus_1 = parse("x - 1", &mut ctx).expect("parse x - 1");
        let x_plus_1 = parse("x + 1", &mut ctx).expect("parse x + 1");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::Positive(abs_quotient)]);

        assert_eq!(normalized.len(), 2);
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_minus_1))
        }));
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_plus_1))
        }));
    }

    #[test]
    fn nonnegative_odd_power_is_dominated_by_nonnegative_base() {
        let mut ctx = Context::new();
        let x_cubed = parse("x^3", &mut ctx).expect("parse x^3");
        let x = parse("x", &mut ctx).expect("parse x");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonNegative(x_cubed),
                ImplicitCondition::NonNegative(x),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::NonNegative(x)]);
    }
}
