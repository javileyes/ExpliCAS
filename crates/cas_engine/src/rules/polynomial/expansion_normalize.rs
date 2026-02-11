//! Polynomial identity detection, heuristic normalization, and small binomial expansion rules.
//!
//! These rules are extracted from `expansion.rs` to keep the module focused on
//! core binomial/multinomial expansion logic.

use crate::multipoly::{MultiPoly, PolyBudget};
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_traits::Signed;

use super::expansion::AutoExpandSubCancelRule;
use crate::multinomial_expand::{try_expand_multinomial_direct, MultinomialExpandBudget};

// =============================================================================
// PolynomialIdentityZeroRule
// =============================================================================

/// PolynomialIdentityZeroRule: Always-on polynomial identity detector
/// Converts expressions to MultiPoly form and checks if result is 0.
/// Priority 90 (lower than AutoExpandSubCancelRule at 95 to avoid duplicate work)
pub struct PolynomialIdentityZeroRule;

impl PolynomialIdentityZeroRule {
    /// Budget limits for polynomial conversion
    /// V2.15.8: Increased max_pow_exp to 6 for binomial identities like (x+1)^5 - expansion = 0
    fn poly_budget() -> PolyBudget {
        PolyBudget {
            max_terms: 50,       // Max monomials in result
            max_total_degree: 6, // Max total degree (covers up to n=6)
            max_pow_exp: 6,      // Max exponent in Pow nodes
        }
    }

    /// Quick check: does expression look polynomial-like and worth checking?
    /// Avoids expensive conversion for obviously non-polynomial expressions.
    fn is_polynomial_candidate(ctx: &Context, expr: ExprId) -> bool {
        Self::is_polynomial_candidate_inner(ctx, expr, 0)
    }

    fn is_polynomial_candidate_inner(ctx: &Context, expr: ExprId, depth: usize) -> bool {
        if depth > 30 {
            return false; // Too deep
        }

        match ctx.get(expr) {
            Expr::Number(_) | Expr::Variable(_) => true,
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                Self::is_polynomial_candidate_inner(ctx, *l, depth + 1)
                    && Self::is_polynomial_candidate_inner(ctx, *r, depth + 1)
            }
            Expr::Neg(inner) => Self::is_polynomial_candidate_inner(ctx, *inner, depth + 1),
            Expr::Pow(base, exp) => {
                // Only integer exponents, and check base
                if let Expr::Number(n) = ctx.get(*exp) {
                    if n.is_integer() && !n.is_negative() {
                        use num_traits::ToPrimitive;
                        if let Some(e) = n.to_integer().to_u32() {
                            if e <= 6 {
                                // V2.15.8: Extended budget for binomial identities
                                return Self::is_polynomial_candidate_inner(ctx, *base, depth + 1);
                            }
                        }
                    }
                }
                false
            }
            _ => false, // Functions, Division, etc. are not polynomial
        }
    }

    /// Convert expression to MultiPoly (reusing AutoExpandSubCancelRule's method)
    fn expr_to_multipoly(
        ctx: &Context,
        id: ExprId,
        vars: &mut Vec<String>,
        budget: &PolyBudget,
    ) -> Option<MultiPoly> {
        AutoExpandSubCancelRule::expr_to_multipoly(ctx, id, vars, budget)
    }
}

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

        // Must be Add or Sub to be a cancellation candidate
        let is_sub_or_add = matches!(ctx.get(expr), Expr::Sub(_, _) | Expr::Add(_, _));
        if !is_sub_or_add {
            return None;
        }

        // Quick node count check (avoid expensive conversion for huge expressions)
        let node_count = cas_ast::count_nodes(ctx, expr);
        if node_count > 100 {
            return None; // Too big, skip
        }

        // Quick polynomial-like check
        if !Self::is_polynomial_candidate(ctx, expr) {
            return None;
        }

        // Try to convert to MultiPoly
        let budget = Self::poly_budget();
        let mut vars = Vec::new();
        let poly = Self::expr_to_multipoly(ctx, expr, &mut vars, &budget)?;

        // Check variable count
        if vars.len() > 4 {
            return None; // Too many variables
        }

        // If the result is zero, we have a polynomial identity!
        if poly.is_zero() {
            let zero = ctx.num(0);

            // Split terms into positive and negative to show LHS/RHS normal forms
            // For an expression like A + B - C - D, we show:
            //   LHS (positive): A + B expanded
            //   RHS (negative): C + D expanded
            let (positive_terms, negative_terms) = {
                let mut pos = Vec::new();
                let mut neg = Vec::new();

                // Collect all additive terms
                fn collect_terms(
                    ctx: &Context,
                    e: ExprId,
                    pos: &mut Vec<ExprId>,
                    neg: &mut Vec<ExprId>,
                ) {
                    match ctx.get(e) {
                        Expr::Add(a, b) => {
                            collect_terms(ctx, *a, pos, neg);
                            collect_terms(ctx, *b, pos, neg);
                        }
                        Expr::Sub(a, b) => {
                            collect_terms(ctx, *a, pos, neg);
                            // b is subtracted, so it goes to negative
                            neg.push(*b);
                        }
                        Expr::Neg(inner) => {
                            neg.push(*inner);
                        }
                        _ => {
                            pos.push(e);
                        }
                    }
                }
                collect_terms(ctx, expr, &mut pos, &mut neg);
                (pos, neg)
            };

            // Build proof data with LHS/RHS if we have both positive and negative terms
            let proof_data = if !positive_terms.is_empty() && !negative_terms.is_empty() {
                // Build polys for positive sum (LHS) and negative sum (RHS)
                let mut lhs_poly = crate::multipoly::MultiPoly::zero(vars.clone());
                let mut rhs_poly = crate::multipoly::MultiPoly::zero(vars.clone());

                // Sum positive terms - use the same vars we already collected
                for &term in &positive_terms {
                    let mut _term_vars = vars.clone();
                    if let Some(term_poly) =
                        Self::expr_to_multipoly(ctx, term, &mut _term_vars, &budget)
                    {
                        // If same vars, can add directly
                        if term_poly.vars == lhs_poly.vars {
                            if let Ok(sum) = lhs_poly.add(&term_poly) {
                                lhs_poly = sum;
                            }
                        }
                    }
                }

                // Sum negative terms (these are the RHS that was subtracted)
                for &term in &negative_terms {
                    let mut _term_vars = vars.clone();
                    if let Some(term_poly) =
                        Self::expr_to_multipoly(ctx, term, &mut _term_vars, &budget)
                    {
                        if term_poly.vars == rhs_poly.vars {
                            if let Ok(sum) = rhs_poly.add(&term_poly) {
                                rhs_poly = sum;
                            }
                        }
                    }
                }

                crate::multipoly_display::PolynomialProofData::from_identity(
                    ctx,
                    &lhs_poly,
                    &rhs_poly,
                    vars.clone(),
                )
            } else {
                // No clear LHS/RHS split
                crate::multipoly_display::PolynomialProofData {
                    monomials: 0,
                    degree: 0,
                    vars: vars.clone(),
                    normal_form_expr: Some(zero),
                    lhs_stats: None,
                    rhs_stats: None,
                }
            };

            return Some(
                Rewrite::new(zero)
                    .desc("Polynomial identity: normalize and cancel to 0")
                    .poly_proof(proof_data),
            );
        }

        None
    }
}

// =============================================================================
// HeuristicPolyNormalizeAddRule
// =============================================================================

/// HeuristicPolyNormalizeAddRule: Poly-normalize sums with binomial powers
/// Priority 42 (after ExpandSmallBinomialPowRule at 40, before others)
pub struct HeuristicPolyNormalizeAddRule;

impl HeuristicPolyNormalizeAddRule {
    /// Check if expression contains Pow(Add, n) with 2 ≤ n ≤ 6
    fn contains_pow_add(ctx: &Context, expr: ExprId) -> bool {
        Self::contains_pow_add_inner(ctx, expr, 0)
    }

    fn contains_pow_add_inner(ctx: &Context, expr: ExprId, depth: usize) -> bool {
        if depth > 20 {
            return false;
        }
        match ctx.get(expr) {
            Expr::Pow(base, exp) => {
                // Check if this is Pow(Add, n) with 2 ≤ n ≤ 6 AND base is polynomial-like
                if matches!(ctx.get(*base), Expr::Add(_, _)) {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        if n.is_integer() && !n.is_negative() {
                            use num_traits::ToPrimitive;
                            if let Some(e) = n.to_integer().to_u32() {
                                // Must be polynomial-like base (no functions like sqrt, sin)
                                if (2..=6).contains(&e)
                                    && crate::auto_expand_scan::looks_polynomial_like(ctx, *base)
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }
                Self::contains_pow_add_inner(ctx, *base, depth + 1)
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                Self::contains_pow_add_inner(ctx, *l, depth + 1)
                    || Self::contains_pow_add_inner(ctx, *r, depth + 1)
            }
            Expr::Neg(inner) | Expr::Hold(inner) => {
                Self::contains_pow_add_inner(ctx, *inner, depth + 1)
            }
            Expr::Div(l, r) => {
                Self::contains_pow_add_inner(ctx, *l, depth + 1)
                    || Self::contains_pow_add_inner(ctx, *r, depth + 1)
            }
            Expr::Function(_, args) => args
                .iter()
                .any(|a| Self::contains_pow_add_inner(ctx, *a, depth + 1)),
            Expr::Matrix { data, .. } => data
                .iter()
                .any(|e| Self::contains_pow_add_inner(ctx, *e, depth + 1)),
            // Leaves
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
        }
    }
}

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

        // Must contain at least one Pow(Add, n) with 2 ≤ n ≤ 6
        // We process the ORIGINAL Add before children are expanded by ExpandSmallBinomialPowRule
        if !Self::contains_pow_add(ctx, expr) {
            return None;
        }

        // Quick size check
        let node_count = cas_ast::count_nodes(ctx, expr);
        if node_count > 80 {
            return None;
        }

        // Try to convert to MultiPoly (this expands and combines terms)
        let budget = PolyBudget {
            max_terms: 40,
            max_total_degree: 6,
            max_pow_exp: 6,
        };

        let mut vars = Vec::new();
        let poly = AutoExpandSubCancelRule::expr_to_multipoly(ctx, expr, &mut vars, &budget)?;

        // Check if result is reasonable
        if poly.terms.len() > 30 || vars.len() > 3 {
            return None;
        }

        // If polynomial is zero, let PolynomialIdentityZeroRule handle it
        if poly.is_zero() {
            return None;
        }

        // Convert back to expression using multipoly_to_expr (produces flattened Add)
        let new_expr = crate::multipoly::multipoly_to_expr(&poly, ctx);

        // Don't rewrite to same expression
        if new_expr == expr {
            return None;
        }

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

        // Pattern: Pow(base, exp)
        let (base, exp) = crate::helpers::as_pow(ctx, expr)?;

        // Very restrictive budget for automatic expansion in generic mode
        // - max_exp: 6 (binomial (x+1)^6 = 7 terms, trinomial (a+b+c)^4 = 15 terms)
        // - max_base_terms: 3 (binomial or trinomial only)
        // - max_vars: 2 (keeps output manageable)
        // - max_output_terms: 20 (strict limit to prevent bloat)
        let budget = MultinomialExpandBudget {
            max_exp: 6,
            max_base_terms: 3,
            max_vars: 2,
            max_output_terms: 20,
        };

        // try_expand_multinomial_direct already:
        // 1. Checks exponent is small positive integer
        // 2. Extracts linear terms (fails if base has functions/div)
        // 3. Estimates output terms and checks budget
        // 4. Wraps result in __hold() for anti-cycle protection
        let expanded = try_expand_multinomial_direct(ctx, base, exp, &budget)?;

        Some(
            Rewrite::new(expanded)
                .desc("Expand binomial/trinomial power")
                .local(expr, expanded),
        )
    }
}
