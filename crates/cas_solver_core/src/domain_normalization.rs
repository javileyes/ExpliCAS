//! Condition normalization, deduplication, and dominance rules.

use crate::domain_condition::ImplicitCondition;
use cas_ast::{Context, ExprId};
use cas_math::expr_domain::{
    exprs_equivalent, exprs_equivalent_up_to_sign, is_abs_of, is_power_of_base,
    is_product_dominated_by_positives,
};
use cas_math::expr_normalization::{
    extract_even_positive_power_base, normalize_condition_expr as normalize_condition_expr_math,
};

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
        ImplicitCondition::NonZero(e) => normalize_condition_expr(ctx, *e),
    };

    match cond {
        ImplicitCondition::NonNegative(_) => ImplicitCondition::NonNegative(normalized_expr),
        ImplicitCondition::Positive(_) => ImplicitCondition::Positive(normalized_expr),
        ImplicitCondition::NonZero(_) => ImplicitCondition::NonZero(normalized_expr),
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
        let normalized = normalize_condition(ctx, cond);
        let is_duplicate = result
            .iter()
            .any(|existing| conditions_equivalent(ctx, existing, &normalized));

        if !is_duplicate {
            result.push(normalized);
        }
    }

    apply_dominance_rules(ctx, &mut result);
    result
}

fn apply_dominance_rules(ctx: &Context, conditions: &mut Vec<ImplicitCondition>) {
    let positive_exprs: Vec<ExprId> = conditions
        .iter()
        .filter_map(|c| {
            if let ImplicitCondition::Positive(e) = c {
                Some(*e)
            } else {
                None
            }
        })
        .collect();

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
                    {
                        to_remove.push(i);
                        break;
                    }
                }
                (ImplicitCondition::Positive(abs_expr), ImplicitCondition::Positive(pos_expr)) => {
                    if is_abs_of(ctx, *abs_expr, *pos_expr)
                        || is_power_of_base(ctx, *abs_expr, *pos_expr)
                    {
                        to_remove.push(i);
                        break;
                    }
                }
                (
                    ImplicitCondition::NonNegative(nn_expr),
                    ImplicitCondition::Positive(pos_expr),
                ) => {
                    if exprs_equivalent(ctx, *nn_expr, *pos_expr) {
                        to_remove.push(i);
                        break;
                    }
                }
                _ => {}
            }
        }

        if !to_remove.contains(&i) {
            if let ImplicitCondition::Positive(prod_expr) = cond {
                if is_product_dominated_by_positives(ctx, *prod_expr, &positive_exprs) {
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
