//! Polynomial identity detection, heuristic normalization, and small binomial expansion rules.
//!
//! These rules are extracted from `expansion.rs` to keep the module focused on
//! core binomial/multinomial expansion logic.

use crate::phase::PhaseMask;
use crate::polynomial_identity_support::{
    try_prove_polynomial_identity_zero_expr, PolynomialIdentityProofKind,
};
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use cas_math::expansion_rule_support::{
    try_expand_small_pow_sum_expr, try_heuristic_poly_normalize_add_expr,
    HeuristicPolyNormalizePolicy, SmallPowExpandPolicy,
};

// =============================================================================
// PolynomialIdentityZeroRule
// =============================================================================

/// PolynomialIdentityZeroRule: Always-on polynomial identity detector
/// Converts expressions to MultiPoly form and checks if result is 0.
/// Priority 90 (lower than AutoExpandSubCancelRule at 95 to avoid duplicate work)
pub struct PolynomialIdentityZeroRule;

impl crate::rule::Rule for PolynomialIdentityZeroRule {
    fn name(&self) -> &str {
        "Polynomial Identity"
    }

    fn priority(&self) -> i32 {
        90 // Lower than AutoExpandSubCancelRule (95) to avoid duplicate work
    }

    fn allowed_phases(&self) -> PhaseMask {
        // Also run in POST so that polynomial identities exposed AFTER Rationalize
        // (e.g., (x+1)^6 - expanded_form left behind when 1/√5 - √5/5 → 0)
        // get cancelled. TRANSFORM alone can't see these because the non-polynomial
        // terms haven't been removed yet at that point.
        PhaseMask::TRANSFORM | PhaseMask::POST
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Skip in Solve mode - preserve structure for equation solving
        if parent_ctx.is_solve_context() {
            return None;
        }

        let plan = try_prove_polynomial_identity_zero_expr(ctx, expr)?;
        let zero = ctx.num(0);
        let desc = match plan.kind {
            PolynomialIdentityProofKind::Direct => "Polynomial identity: normalize and cancel to 0",
            PolynomialIdentityProofKind::OpaqueSubstitution => {
                "Polynomial identity (opaque substitution): cancel to 0"
            }
        };
        Some(Rewrite::new(zero).desc(desc).poly_proof(plan.proof_data))
    }
}

// =============================================================================
// HeuristicPolyNormalizeAddRule
// =============================================================================

/// HeuristicPolyNormalizeAddRule: Poly-normalize sums with binomial powers
/// Priority 42 (after ExpandSmallBinomialPowRule at 40, before others)
pub struct HeuristicPolyNormalizeAddRule;

impl crate::rule::Rule for HeuristicPolyNormalizeAddRule {
    fn name(&self) -> &str {
        "Heuristic Poly Normalize"
    }

    fn priority(&self) -> i32 {
        100 // Very high priority - must process Add BEFORE children Pow are expanded
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::TRANSFORM
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only trigger when heuristic_poly is On
        use crate::options::HeuristicPoly;
        if parent_ctx.heuristic_poly() != HeuristicPoly::On {
            return None;
        }

        // Skip in Solve mode
        if parent_ctx.is_solve_context() {
            return None;
        }

        // Must be Add or Sub
        if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
            return None;
        }

        let new_expr = try_heuristic_poly_normalize_add_expr(
            ctx,
            expr,
            HeuristicPolyNormalizePolicy::default(),
        )?;

        Some(
            Rewrite::new(new_expr)
                .desc("Expand and combine polynomial terms (heuristic)")
                .local(expr, new_expr),
        )
    }
}

// =============================================================================
// ExpandSmallBinomialPowRule
// =============================================================================
/// ExpandSmallBinomialPowRule: Always-on expansion for small binomial/trinomial powers
/// Priority 40 (before AutoExpandPowSumRule at 50, after most algebraic rules)
pub struct ExpandSmallBinomialPowRule;

impl crate::rule::Rule for ExpandSmallBinomialPowRule {
    fn name(&self) -> &str {
        "Expand Small Power"
    }

    fn priority(&self) -> i32 {
        40 // Before AutoExpandPowSumRule (50), after basic algebra
    }

    fn allowed_phases(&self) -> PhaseMask {
        // Only in TRANSFORM phase to avoid interfering with early simplification
        PhaseMask::TRANSFORM
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // V2.15.9: Check autoexpand_binomials mode (Off/On)
        use crate::options::AutoExpandBinomials;
        match parent_ctx.autoexpand_binomials() {
            AutoExpandBinomials::Off => return None, // Never expand standalone
            AutoExpandBinomials::On => {
                // Always expand (subject to budget checks below)
            }
        }

        // Skip in Solve mode - preserve structure for equation solving
        if parent_ctx.is_solve_context() {
            return None;
        }

        // Skip if already in auto-expand context (let AutoExpandPowSumRule handle)
        if parent_ctx.in_auto_expand_context()
            && parent_ctx.autoexpand_binomials() != AutoExpandBinomials::On
        {
            return None;
        }

        // Very restrictive budget for automatic expansion in generic mode:
        // - max_exp: 6 (binomial (x+1)^6 = 7 terms, trinomial (a+b+c)^4 = 15 terms)
        // - max_base_terms: 3 (binomial or trinomial only)
        // - max_vars: 2 (keeps output manageable)
        // - max_output_terms: 20 (strict limit to prevent bloat)
        let expanded = try_expand_small_pow_sum_expr(ctx, expr, SmallPowExpandPolicy::default())?;

        Some(
            Rewrite::new(expanded)
                .desc("Expand binomial/trinomial power")
                .local(expr, expanded),
        )
    }
}
