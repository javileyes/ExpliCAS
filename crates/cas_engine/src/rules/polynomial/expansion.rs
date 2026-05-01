//! Polynomial expansion rules: binomial expansion, auto-expand, identity detection,
//! and heuristic polynomial normalization.

use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use cas_math::expansion_rule_support::{
    is_auto_sub_cancel_zero, try_auto_expand_pow_sum_expr, try_expand_binomial_pow_expr,
    try_expand_small_multinomial_expr, AutoExpandPowSumPolicy, AutoSubCancelPolicy,
    SmallMultinomialPolicy,
};
use cas_math::{numeric::as_i64, polynomial::Polynomial};
use num_traits::Signed;

// =============================================================================
// BinomialExpansionRule
// =============================================================================

/// BinomialExpansionRule: (a + b)^n → expanded polynomial
/// ONLY expands true binomials (exactly 2 terms).
/// Multinomial expansion (>2 terms) is NOT done by default to avoid explosion.
/// Use explicit expand() mode for multinomial expansion.
/// Implements Rule directly to access ParentContext
pub struct BinomialExpansionRule;

impl crate::rule::Rule for BinomialExpansionRule {
    fn name(&self) -> &str {
        "Binomial Expansion"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Skip if expression is in canonical (elegant) form
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }

        // GUARD: Don't expand if this expression is protected as a sqrt-square base
        // This is set by pre-scan when this expr is inside sqrt(u²) or sqrt(u*u)
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sqrt_square_protected(expr) {
                // Protected from expansion - let sqrt(u²) → |u| shortcut fire instead
                return None;
            }
        }

        // Only expand binomials in explicit expand mode.
        // In Standard mode, preserve structures like `(x+1)^3`.
        if !parent_ctx.is_expand_mode() {
            return None;
        }

        let plan = try_expand_binomial_pow_expr(ctx, expr, 2, 20)?;
        Some(
            Rewrite::new(plan.expanded)
                .desc_lazy(|| format!("Expand binomial power ^{}", plan.exponent)),
        )
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }
}

// =============================================================================
// AutoExpandPowSumRule
// =============================================================================

/// Rule for auto-expanding cheap power-of-sum expressions within budget limits.
/// Unlike BinomialExpansionRule, this is opt-in and checks budget constraints.
///
/// Only triggers when `parent_ctx.is_auto_expand()` is true.
/// Respects budget limits: max_pow_exp, max_base_terms, max_generated_terms, max_vars.
pub struct AutoExpandPowSumRule;

impl crate::rule::Rule for AutoExpandPowSumRule {
    fn name(&self) -> &str {
        "Auto Expand Power Sum"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Expand if: global auto-expand mode OR inside a marked cancellation context
        // (e.g., difference quotient like ((x+h)^n - x^n)/h)
        let in_expand_context = parent_ctx.in_auto_expand_context();
        if !(parent_ctx.is_auto_expand() || in_expand_context) {
            return None;
        }

        if should_preserve_compact_reciprocal_polynomial_power_denominator(ctx, expr, parent_ctx) {
            return None;
        }

        if should_preserve_compact_power_gap_in_quotient(ctx, expr, parent_ctx) {
            return None;
        }

        // Get budget - use default if in context but no explicit budget set
        let default_budget = crate::phase::ExpandBudget::default();
        let budget = parent_ctx.auto_expand_budget().unwrap_or(&default_budget);

        // Skip if expression is in canonical form
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }

        let policy = AutoExpandPowSumPolicy {
            max_pow_exp: budget.max_pow_exp,
            max_base_terms: budget.max_base_terms,
            max_generated_terms: budget.max_generated_terms,
            max_vars: budget.max_vars,
        };
        let plan = try_auto_expand_pow_sum_expr(ctx, expr, policy)?;

        if plan.num_terms == 2 {
            Some(
                Rewrite::new(plan.expanded)
                    .desc_lazy(|| format!("Auto-expand (a+b)^{}", plan.exponent)),
            )
        } else {
            Some(Rewrite::new(plan.expanded).desc_lazy(|| {
                format!(
                    "Auto-expand ({}-term sum)^{}",
                    plan.num_terms, plan.exponent
                )
            }))
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // Include RATIONALIZE so auto-expand can clean up after rationalization
        // e.g., 1/(1+√2+√3) → ... → (1+√2)² - 3 → needs auto-expand to become 2√2
        crate::phase::PhaseMask::CORE
            | crate::phase::PhaseMask::TRANSFORM
            | crate::phase::PhaseMask::RATIONALIZE
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        // Auto-expand steps are didactically important: users should see the expansion
        crate::step::ImportanceLevel::Medium
    }
}

fn should_preserve_compact_power_gap_in_quotient(
    ctx: &Context,
    expr: ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    if !is_under_division(ctx, parent_ctx) || !is_high_additive_power(ctx, expr) {
        return false;
    }

    let Some(parent) = parent_ctx.immediate_parent() else {
        return false;
    };
    let Expr::Sub(left, right) = ctx.get(parent) else {
        return false;
    };

    *right == expr && matches!(ctx.get(*left), Expr::Number(n) if n.is_positive())
}

fn is_under_division(ctx: &Context, parent_ctx: &crate::parent_context::ParentContext) -> bool {
    parent_ctx.has_ancestor_matching(ctx, |c, ancestor| {
        matches!(c.get(ancestor), Expr::Div(_, _))
    })
}

fn is_high_additive_power(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return false;
    };
    let Expr::Number(exp_num) = ctx.get(*exp) else {
        return false;
    };
    if !exp_num.is_integer() || exp_num.to_integer() < 4.into() {
        return false;
    }

    matches!(ctx.get(*base), Expr::Add(_, _) | Expr::Sub(_, _))
        && additive_leaf_count_up_to(ctx, *base, 2).is_some()
}

fn additive_leaf_count_up_to(ctx: &Context, expr: ExprId, limit: usize) -> Option<usize> {
    if limit == 0 {
        return None;
    }

    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            let left_count = additive_leaf_count_up_to(ctx, *left, limit)?;
            let remaining = limit.checked_sub(left_count)?;
            let right_count = additive_leaf_count_up_to(ctx, *right, remaining)?;
            let total = left_count + right_count;
            (total <= limit).then_some(total)
        }
        _ => Some(1),
    }
}

// =============================================================================
// SmallMultinomialExpansionRule
// =============================================================================

/// SmallMultinomialExpansionRule: (a + b + c + ...)^n → expanded polynomial
///
/// Fires in **default simplification** (not just expand mode) for small, safe
/// multinomials.  The primary gate is `pred_terms = C(n+k-1, k-1) ≤ 35`,
/// which caps the output size regardless of `n` or `k` individually.
///
/// Guards (pre-expansion):
/// - n ∈ [2, 4]  (MAX_N)
/// - k ∈ [3, 6]  (need ≥3 terms; ≤6 as secondary cap)
/// - pred_terms ≤ 35  (primary output cap)
/// - count_all_nodes(base) ≤ 25  (base complexity cap)
///
/// Guard (post-expansion):
/// - count_all_nodes(expanded) ≤ 300  (output complexity cap)
///   Catches cases where pred_terms is small but each term carries heavy
///   subexpressions that get replicated.
///
/// Uses `budget_exempt` since its own guards (pred_terms, n, k, base_nodes,
/// output_nodes) are stricter than the global anti-worsen budget.
pub struct SmallMultinomialExpansionRule;

impl crate::rule::Rule for SmallMultinomialExpansionRule {
    fn name(&self) -> &str {
        "Small Multinomial Expansion"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        if parent_ctx.context_mode() == crate::options::ContextMode::IntegratePrep {
            return None;
        }

        let integrate_fn = ctx.intern_symbol("integrate");
        if parent_ctx.has_ancestor_matching(ctx, |c, ancestor| {
            matches!(
                c.get(ancestor),
                Expr::Function(fn_id, _) if *fn_id == integrate_fn
            )
        }) {
            return None;
        }

        if should_preserve_compact_low_degree_power_product(ctx, expr, parent_ctx) {
            return None;
        }

        if should_preserve_compact_low_degree_power_shared_sum(ctx, expr, parent_ctx) {
            return None;
        }

        if should_preserve_compact_reciprocal_polynomial_power_denominator(ctx, expr, parent_ctx) {
            return None;
        }

        let plan = try_expand_small_multinomial_expr(ctx, expr, SmallMultinomialPolicy::default())?;
        let k = plan.term_count;
        let n = plan.exponent;
        let expanded = plan.expanded;

        let k_copy = k;
        let n_copy = n;
        Some(
            Rewrite::new(expanded)
                .desc_lazy(move || format!("Expand ({}-term sum)^{}", k_copy, n_copy))
                .budget_exempt(),
        )
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }
}

fn should_preserve_compact_low_degree_power_product(
    ctx: &Context,
    expr: ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    if parent_ctx.is_expand_mode()
        || parent_ctx.is_auto_expand()
        || parent_ctx.in_auto_expand_context()
    {
        return false;
    }

    let Some((_power, poly)) = low_degree_multiterm_polynomial_power(ctx, expr) else {
        return false;
    };
    let var = poly.var.as_str();

    parent_ctx.has_ancestor_matching(ctx, |c, ancestor| {
        matches!(c.get(ancestor), Expr::Mul(_, _))
            && cas_math::expr_nary::mul_leaves(c, ancestor)
                .into_iter()
                .any(|factor| {
                    factor != expr && is_nonconstant_linear_polynomial_factor(c, factor, var)
                })
    })
}

fn should_preserve_compact_low_degree_power_shared_sum(
    ctx: &Context,
    expr: ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    if parent_ctx.is_expand_mode()
        || parent_ctx.is_auto_expand()
        || parent_ctx.in_auto_expand_context()
    {
        return false;
    }

    let Some((_power, poly)) = low_degree_multiterm_polynomial_power(ctx, expr) else {
        return false;
    };
    let var = poly.var.as_str();

    parent_ctx.has_ancestor_matching(ctx, |c, ancestor| {
        if !matches!(c.get(ancestor), Expr::Add(_, _)) {
            return false;
        }

        let terms = cas_math::expr_nary::add_leaves(c, ancestor);
        if !(2..=4).contains(&terms.len()) {
            return false;
        }

        let mut terms_with_power = 0usize;
        let mut term_with_linear_partner = false;
        for term in terms {
            let factors = cas_math::expr_nary::mul_leaves(c, term);
            if !factors
                .iter()
                .any(|factor| same_low_degree_polynomial_power(c, expr, *factor))
            {
                continue;
            }

            terms_with_power += 1;
            if factors.iter().any(|factor| {
                !same_low_degree_polynomial_power(c, expr, *factor)
                    && is_nonconstant_linear_polynomial_factor(c, *factor, var)
            }) {
                term_with_linear_partner = true;
            }
        }

        terms_with_power >= 2 && term_with_linear_partner
    })
}

fn low_degree_multiterm_polynomial_power(ctx: &Context, expr: ExprId) -> Option<(i64, Polynomial)> {
    let Expr::Pow(base, _) = ctx.get(expr) else {
        return None;
    };
    if additive_leaf_count_up_to(ctx, *base, 2).is_some() {
        return None;
    }

    low_degree_polynomial_power(ctx, expr)
}

fn is_nonconstant_linear_polynomial_factor(ctx: &Context, factor: ExprId, var: &str) -> bool {
    matches!(Polynomial::from_expr(ctx, factor, var), Ok(poly) if !poly.is_zero() && poly.degree() == 1)
        || symbolic_degree_with_small_literal_exponents(ctx, factor, var, 1) == Some(1)
}

fn symbolic_degree_with_small_literal_exponents(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    max_degree: usize,
) -> Option<usize> {
    let degree = match ctx.get(expr) {
        Expr::Number(_) | Expr::Constant(_) => 0,
        Expr::Variable(sym_id) => {
            if ctx.sym_name(*sym_id) == var {
                1
            } else {
                return None;
            }
        }
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            let left_degree =
                symbolic_degree_with_small_literal_exponents(ctx, *left, var, max_degree)?;
            let right_degree =
                symbolic_degree_with_small_literal_exponents(ctx, *right, var, max_degree)?;
            left_degree.max(right_degree)
        }
        Expr::Mul(left, right) => {
            let left_degree =
                symbolic_degree_with_small_literal_exponents(ctx, *left, var, max_degree)?;
            let right_degree =
                symbolic_degree_with_small_literal_exponents(ctx, *right, var, max_degree)?;
            left_degree.checked_add(right_degree)?
        }
        Expr::Div(left, right) => {
            let numerator_degree =
                symbolic_degree_with_small_literal_exponents(ctx, *left, var, max_degree)?;
            let denominator_degree =
                symbolic_degree_with_small_literal_exponents(ctx, *right, var, max_degree)?;
            if denominator_degree != 0 {
                return None;
            }
            numerator_degree
        }
        Expr::Pow(base, exp) => {
            let exponent = small_literal_integer_value(ctx, *exp)?;
            if exponent < 0 {
                return None;
            }
            let base_degree =
                symbolic_degree_with_small_literal_exponents(ctx, *base, var, max_degree)?;
            base_degree.checked_mul(exponent as usize)?
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            symbolic_degree_with_small_literal_exponents(ctx, *inner, var, max_degree)?
        }
        _ => return None,
    };

    (degree <= max_degree).then_some(degree)
}

fn small_literal_integer_value(ctx: &Context, expr: ExprId) -> Option<i64> {
    match ctx.get(expr) {
        Expr::Number(_) => as_i64(ctx, expr),
        Expr::Neg(inner) => small_literal_integer_value(ctx, *inner)?.checked_neg(),
        Expr::Add(left, right) => small_literal_integer_value(ctx, *left)?
            .checked_add(small_literal_integer_value(ctx, *right)?),
        Expr::Sub(left, right) => small_literal_integer_value(ctx, *left)?
            .checked_sub(small_literal_integer_value(ctx, *right)?),
        Expr::Mul(left, right) => small_literal_integer_value(ctx, *left)?
            .checked_mul(small_literal_integer_value(ctx, *right)?),
        Expr::Div(left, right) => {
            let numerator = small_literal_integer_value(ctx, *left)?;
            let denominator = small_literal_integer_value(ctx, *right)?;
            if denominator == 0 || numerator % denominator != 0 {
                return None;
            }
            Some(numerator / denominator)
        }
        _ => None,
    }
}

fn should_preserve_compact_reciprocal_polynomial_power_denominator(
    ctx: &Context,
    expr: ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    if parent_ctx.is_expand_mode() || parent_ctx.is_auto_expand() {
        return false;
    }

    let has_matching_div_denominator = parent_ctx.has_ancestor_matching(ctx, |c, ancestor| {
        matches!(
            c.get(ancestor),
            Expr::Div(_, den) if denominator_contains_matching_low_degree_power(c, expr, *den)
        )
    });
    if !has_matching_div_denominator {
        return false;
    }

    low_degree_polynomial_power(ctx, expr).is_some()
}

fn denominator_contains_matching_low_degree_power(
    ctx: &Context,
    expr: ExprId,
    denominator: ExprId,
) -> bool {
    same_low_degree_polynomial_power(ctx, expr, denominator)
        || cas_math::expr_nary::mul_leaves(ctx, denominator)
            .into_iter()
            .any(|factor| same_low_degree_polynomial_power(ctx, expr, factor))
}

fn same_low_degree_polynomial_power(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    let Some((left_power, left_poly)) = low_degree_polynomial_power(ctx, left) else {
        return false;
    };
    let Some((right_power, right_poly)) = low_degree_polynomial_power(ctx, right) else {
        return false;
    };

    left_power == right_power && left_poly == right_poly
}

fn low_degree_polynomial_power(ctx: &Context, expr: ExprId) -> Option<(i64, Polynomial)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let power = as_i64(ctx, *exp)?;
    if !(2..=8).contains(&power) {
        return None;
    }

    let variables = cas_ast::collect_variables(ctx, *base);
    let var = variables.iter().next()?;
    if variables.len() != 1 {
        return None;
    }

    Polynomial::from_expr(ctx, *base, var.as_str())
        .ok()
        .filter(|poly| (1..=2).contains(&poly.degree()))
        .map(|poly| (power, poly))
}

// =============================================================================
// AutoExpandSubCancelRule
// =============================================================================

/// AutoExpandSubCancelRule: Zero-shortcut for Sub(Pow(Add..), polynomial)
/// Priority 95 (higher than AutoExpandPowSumRule at 50)
pub struct AutoExpandSubCancelRule;

impl crate::rule::Rule for AutoExpandSubCancelRule {
    fn name(&self) -> &str {
        "AutoExpandSubCancelRule"
    }

    fn priority(&self) -> i32 {
        95
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::TRANSFORM
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only trigger if in auto-expand context (marked by scanner)
        // Use _for_expr to also check if current node is marked (not just ancestors)
        if !parent_ctx.in_auto_expand_context_for_expr(expr) {
            return None;
        }

        // Must be Sub or Add to be a cancellation candidate
        let is_sub_or_add = matches!(ctx.get(expr), Expr::Sub(_, _) | Expr::Add(_, _));
        if !is_sub_or_add {
            return None;
        }

        if is_auto_sub_cancel_zero(ctx, expr, AutoSubCancelPolicy::default()) {
            let zero = ctx.num(0);
            return Some(Rewrite::new(zero).desc("Polynomial equality: expressions cancel to 0"));
        }

        None
    }
}
