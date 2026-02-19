//! Condition normalization, deduplication, and dominance rules.

use super::ImplicitCondition;
use cas_ast::{Context, ExprId};
use cas_math::expr_domain::{
    exprs_equivalent, is_abs_of, is_power_of_base, is_product_dominated_by_positives,
};
use cas_math::expr_normalization::{
    extract_even_positive_power_base, normalize_condition_expr as normalize_condition_expr_math,
};

// =============================================================================
// Condition Normalization (Canonical Form + Sign Normalization + Dedupe)
// =============================================================================

/// Normalize an expression for display in conditions.
///
/// This ensures that equivalent expressions like `1 + x` and `x + 1`
/// display consistently (as `x + 1` in polynomial order).
///
/// Strategy:
/// 1. Try polynomial conversion → get canonical polynomial form
/// 2. Normalize sign: if leading coefficient is negative, negate the whole expression
/// 3. Fallback: use DisplayExpr which already applies term ordering
///
/// The normalized ExprId is only used for DISPLAY - it does not affect the
/// underlying AST or comparisons.
pub fn normalize_condition_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    normalize_condition_expr_math(ctx, expr)
}

/// Normalize a condition for display (applies normalization to the inner expression).
///
/// Special cases:
/// - `Positive(x^(even))` → `NonZero(x)` because x^(2k) > 0 ⟺ x ≠ 0 for reals
pub fn normalize_condition(ctx: &mut Context, cond: &ImplicitCondition) -> ImplicitCondition {
    // Special case: Positive(x^even) → NonZero(x)
    // Because x^(2k) > 0 is equivalent to x ≠ 0 for real numbers
    if let ImplicitCondition::Positive(e) = cond {
        if let Some(base) = extract_even_positive_power_base(ctx, *e) {
            let normalized_base = normalize_condition_expr(ctx, base);
            return ImplicitCondition::NonZero(normalized_base);
        }
    }

    // Standard normalization
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

/// Check if two conditions are equivalent (same type and polynomial-equivalent expressions).
pub(crate) fn conditions_equivalent(
    ctx: &Context,
    c1: &ImplicitCondition,
    c2: &ImplicitCondition,
) -> bool {
    use cas_math::multipoly::{multipoly_from_expr, PolyBudget};

    // Must be same condition type
    let (e1, e2) = match (c1, c2) {
        (ImplicitCondition::NonNegative(a), ImplicitCondition::NonNegative(b)) => (*a, *b),
        (ImplicitCondition::Positive(a), ImplicitCondition::Positive(b)) => (*a, *b),
        (ImplicitCondition::NonZero(a), ImplicitCondition::NonZero(b)) => (*a, *b),
        _ => return false,
    };

    // Same ExprId = definitely equivalent
    if e1 == e2 {
        return true;
    }

    // Try polynomial comparison
    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    if let (Ok(p1), Ok(p2)) = (
        multipoly_from_expr(ctx, e1, &budget),
        multipoly_from_expr(ctx, e2, &budget),
    ) {
        // For NonZero, E ≠ 0 is equivalent to -E ≠ 0
        // So check if p1 == p2 OR p1 == -p2
        if p1 == p2 {
            return true;
        }

        // Check negation equivalence (E ≠ 0 ⟺ -E ≠ 0)
        // Also applies to Positive (E > 0 ⟺ -E < 0, but for reals we handle this)
        let p2_neg = p2.neg();
        if p1 == p2_neg {
            return true;
        }
    }

    false
}

/// Normalize and deduplicate a list of conditions for display.
///
/// This function:
/// 1. Normalizes each condition (canonical form + sign normalization)
/// 2. Deduplicates conditions by polynomial equivalence
/// 3. Removes dominated conditions via dominance rules:
///    - x > 0 or x < 0 dominates x ≠ 0
///    - x > 0 dominates x ≥ 0 and |x| > 0
/// 4. Preserves order (stable dedupe - keeps first occurrence)
///
/// Used when rendering "Requires:" in timeline and REPL.
pub fn normalize_and_dedupe_conditions(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
) -> Vec<ImplicitCondition> {
    let mut result: Vec<ImplicitCondition> = Vec::new();

    for cond in conditions {
        let normalized = normalize_condition(ctx, cond);

        // Check if we already have an equivalent condition
        let is_duplicate = result
            .iter()
            .any(|existing| conditions_equivalent(ctx, existing, &normalized));

        if !is_duplicate {
            result.push(normalized);
        }
    }

    // Apply dominance rules to remove redundant conditions
    apply_dominance_rules(ctx, &mut result);

    result
}

/// Apply dominance rules to remove redundant conditions.
///
/// Rules:
/// 1. x > 0 dominates x ≠ 0 (remove x ≠ 0)
/// 2. x > 0 dominates |x| > 0 (remove |x| > 0)  
/// 3. x > 0 dominates x ≥ 0 (remove x ≥ 0)
/// 4. x > 0 dominates x^n > 0 for any positive n (remove x^n > 0)
/// 5. If all factors of a product are known positive, remove product > 0
fn apply_dominance_rules(ctx: &Context, conditions: &mut Vec<ImplicitCondition>) {
    // First pass: collect all known positive expressions (for product dominance)
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

    // Collect indices to remove
    let mut to_remove: Vec<usize> = Vec::new();

    for (i, cond) in conditions.iter().enumerate() {
        // Check if this condition is dominated by another
        for (j, other) in conditions.iter().enumerate() {
            if i == j {
                continue;
            }

            match (cond, other) {
                // x ≠ 0 dominated by x > 0
                (ImplicitCondition::NonZero(nz_expr), ImplicitCondition::Positive(pos_expr)) => {
                    if exprs_equivalent(ctx, *nz_expr, *pos_expr) {
                        to_remove.push(i);
                        break;
                    }
                    // Also check if nz_expr is |x| and pos_expr is x
                    if is_abs_of(ctx, *nz_expr, *pos_expr) {
                        to_remove.push(i);
                        break;
                    }
                }
                // |x| > 0 dominated by x > 0
                (ImplicitCondition::Positive(abs_expr), ImplicitCondition::Positive(pos_expr)) => {
                    // Check if abs_expr is |pos_expr|
                    if is_abs_of(ctx, *abs_expr, *pos_expr) {
                        to_remove.push(i);
                        break;
                    }
                    // NEW: x^n > 0 dominated by x > 0 (for any positive n)
                    if is_power_of_base(ctx, *abs_expr, *pos_expr) {
                        to_remove.push(i);
                        break;
                    }
                }
                // x ≥ 0 dominated by x > 0
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

        // NEW: Check product dominance - if cond is Positive(product) and all factors are known positive
        if !to_remove.contains(&i) {
            if let ImplicitCondition::Positive(prod_expr) = cond {
                if is_product_dominated_by_positives(ctx, *prod_expr, &positive_exprs) {
                    to_remove.push(i);
                }
            }
        }
    }

    // Remove dominated conditions in reverse order to preserve indices
    to_remove.sort();
    to_remove.dedup();
    for i in to_remove.into_iter().rev() {
        conditions.remove(i);
    }
}

/// Render conditions for display, applying normalization and deduplication.
///
/// This is the main entry point for rendering "Requires:" lists.
pub fn render_conditions_normalized(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
) -> Vec<String> {
    let normalized = normalize_and_dedupe_conditions(ctx, conditions);
    normalized.iter().map(|c| c.display(ctx)).collect()
}
