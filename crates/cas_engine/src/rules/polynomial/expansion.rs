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
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
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
