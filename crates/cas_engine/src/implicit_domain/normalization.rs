//! Condition normalization, deduplication, and dominance rules.

use super::ImplicitCondition;
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_extract::extract_abs_argument_view;

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
    use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};

    // Small budget - we're just normalizing for display, not doing heavy algebra
    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    // Try polynomial normalization
    if let Ok(poly) = multipoly_from_expr(ctx, expr, &budget) {
        // Check leading coefficient sign
        let needs_negation = if let Some((_coeff, _mono)) = poly.leading_term_lex() {
            // Leading coefficient is negative if coeff < 0
            _coeff < &num_rational::BigRational::from_integer(0.into())
        } else {
            false
        };

        // If leading coeff is negative, negate the polynomial
        let normalized_poly = if needs_negation { poly.neg() } else { poly };

        // Convert back to expression
        return multipoly_to_expr(&normalized_poly, ctx);
    }

    // Fallback: expression is not polynomial
    // Check if it's a Neg node and unwrap it
    if let Expr::Neg(inner) = ctx.get(expr) {
        // -E ≠ 0 is equivalent to E ≠ 0, prefer non-negated form
        return *inner;
    }

    // Otherwise return original (DisplayExpr will apply term ordering)
    expr
}

/// Normalize a condition for display (applies normalization to the inner expression).
///
/// Special cases:
/// - `Positive(x^(even))` → `NonZero(x)` because x^(2k) > 0 ⟺ x ≠ 0 for reals
pub fn normalize_condition(ctx: &mut Context, cond: &ImplicitCondition) -> ImplicitCondition {
    // Special case: Positive(x^even) → NonZero(x)
    // Because x^(2k) > 0 is equivalent to x ≠ 0 for real numbers
    if let ImplicitCondition::Positive(e) = cond {
        if let Some((base, normalized_to_nonzero)) =
            try_convert_even_power_positive_to_nonzero(ctx, *e)
        {
            if normalized_to_nonzero {
                let normalized_base = normalize_condition_expr(ctx, base);
                return ImplicitCondition::NonZero(normalized_base);
            }
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

/// Check if expression is x^(even positive integer) and should be converted to NonZero(x).
/// Returns Some((base, true)) if conversion should happen, None otherwise.
fn try_convert_even_power_positive_to_nonzero(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, bool)> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if n.is_integer() {
                let exp_int = n.to_integer();
                let two: num_bigint::BigInt = 2.into();
                let zero: num_bigint::BigInt = 0.into();
                // Check: even AND positive (not zero)
                if &exp_int % &two == zero && exp_int > zero {
                    return Some((*base, true));
                }
            }
        }
    }
    None
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

/// Check if two expressions are equivalent using polynomial comparison.
pub(crate) fn exprs_equivalent(ctx: &Context, e1: ExprId, e2: ExprId) -> bool {
    use cas_math::multipoly::{multipoly_from_expr, PolyBudget};

    if e1 == e2 {
        return true;
    }

    // Quick check: same variable name
    if let (Expr::Variable(name1), Expr::Variable(name2)) = (ctx.get(e1), ctx.get(e2)) {
        if name1 == name2 {
            return true;
        }
    }

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    if let (Ok(p1), Ok(p2)) = (
        multipoly_from_expr(ctx, e1, &budget),
        multipoly_from_expr(ctx, e2, &budget),
    ) {
        return p1 == p2;
    }

    false
}

/// Check if `source` is `target^(odd positive integer)`.
/// E.g., is_odd_power_of(ctx, b^3, b) = true
pub(crate) fn is_odd_power_of(ctx: &Context, source: ExprId, target: ExprId) -> bool {
    if let Expr::Pow(base, exp) = ctx.get(source) {
        // Check if exp is an odd positive integer
        if let Expr::Number(n) = ctx.get(*exp) {
            if n.is_integer() {
                let exp_int = n.to_integer();
                let two: num_bigint::BigInt = 2.into();
                let zero: num_bigint::BigInt = 0.into();
                let one: num_bigint::BigInt = 1.into();
                // Odd: exp % 2 == 1, positive: exp > 0
                if &exp_int % &two == one && exp_int > zero {
                    // Check if base equals target
                    return exprs_equivalent(ctx, *base, target);
                }
            }
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

/// Check if expr is base^n for some positive integer n
pub(crate) fn is_power_of_base(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    if let Expr::Pow(pow_base, exp) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            // V2.15.9: In RealOnly, base > 0 → base^p > 0 for ANY real p (not just integers)
            // This includes x^(1/2), x^(3/4), etc.
            // The only exception is p = 0 (base^0 = 1, not related to base positivity)
            let zero = num_rational::BigRational::from_integer(0.into());
            if *n != zero {
                return exprs_equivalent(ctx, *pow_base, base);
            }
        }
    }
    false
}

/// Check if source is k*target where k > 0.
/// Used to deduce x ≥ 0 from k*x ≥ 0 when k is positive.
/// e.g., is_positive_multiple_of(ctx, 4*x, x) = true
pub(crate) fn is_positive_multiple_of(ctx: &Context, source: ExprId, target: ExprId) -> bool {
    use num_traits::Zero;

    // First check direct equivalence
    if exprs_equivalent(ctx, source, target) {
        return true;
    }

    // Check if source is Mul(k, target) or Mul(target, k) where k > 0
    if let Expr::Mul(l, r) = ctx.get(source) {
        // Check k * target
        if let Expr::Number(n) = ctx.get(*l) {
            let zero = num_rational::BigRational::zero();
            if *n > zero && exprs_equivalent(ctx, *r, target) {
                return true;
            }
        }
        // Check target * k
        if let Expr::Number(n) = ctx.get(*r) {
            let zero = num_rational::BigRational::zero();
            if *n > zero && exprs_equivalent(ctx, *l, target) {
                return true;
            }
        }
    }
    false
}

/// Check if a product expression is dominated by known positive expressions.
/// E.g., a^2 * b^3 > 0 is dominated if we have both a > 0 and b > 0
/// Only applies to actual products (≥2 factors) to avoid removing single conditions.
fn is_product_dominated_by_positives(
    ctx: &Context,
    prod_expr: ExprId,
    known_positives: &[ExprId],
) -> bool {
    // Collect all base variables from the product
    let bases = extract_product_bases(ctx, prod_expr);

    // Only dominate actual products with ≥2 factors
    // Single bases are handled by is_power_of_base or direct equivalence
    if bases.len() < 2 {
        return false;
    }

    // Check if all bases are known positive
    for base in &bases {
        let base_is_covered = known_positives
            .iter()
            .any(|pos| exprs_equivalent(ctx, *base, *pos));
        if !base_is_covered {
            return false;
        }
    }

    true
}

/// Extract base expressions from a product like a^2 * b^3
/// Returns [a, b] for a^2 * b^3
fn extract_product_bases(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut bases = Vec::new();
    collect_product_bases(ctx, expr, &mut bases);
    bases
}

fn collect_product_bases(ctx: &Context, expr: ExprId, bases: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            collect_product_bases(ctx, *l, bases);
            collect_product_bases(ctx, *r, bases);
        }
        Expr::Pow(base, exp) => {
            // Check if exponent is positive - if so, add the base
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() {
                    let exp_int = n.to_integer();
                    let zero: num_bigint::BigInt = 0.into();
                    if exp_int > zero {
                        bases.push(*base);
                        return;
                    }
                }
            }
            // Fallback: treat the whole thing as a base
            bases.push(expr);
        }
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) => {
            bases.push(expr);
        }
        _ => {
            // Complex expression - treat as single base
            bases.push(expr);
        }
    }
}

/// Check if expr is Abs(inner_expr), i.e., |inner_expr|
fn is_abs_of(ctx: &Context, expr: ExprId, inner: ExprId) -> bool {
    if let Some(arg) = extract_abs_argument_view(ctx, expr) {
        return exprs_equivalent(ctx, arg, inner);
    }
    false
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
