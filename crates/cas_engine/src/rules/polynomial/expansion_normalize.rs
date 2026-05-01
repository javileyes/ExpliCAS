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
    try_normalize_small_polynomial_product_expr, HeuristicPolyNormalizePolicy,
    PolynomialProductNormalizePolicy, SmallPowExpandPolicy,
};
use cas_math::{numeric::as_i64, polynomial::Polynomial};
use num_traits::Signed;

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
            PolynomialIdentityProofKind::OpaqueRootRelation => {
                "Polynomial identity (opaque root relation): cancel to 0"
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

        if is_compact_high_power_gap(ctx, expr) {
            return None;
        }

        if is_compact_low_degree_power_shared_sum(ctx, expr) {
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

fn is_compact_high_power_gap(ctx: &Context, expr: ExprId) -> bool {
    let Some(power) = positive_const_minus_expr_rhs(ctx, expr) else {
        return false;
    };
    is_high_additive_power(ctx, power)
}

fn is_compact_low_degree_power_shared_sum(ctx: &Context, expr: ExprId) -> bool {
    let terms = cas_math::expr_nary::add_leaves(ctx, expr);
    if !(2..=4).contains(&terms.len()) {
        return false;
    }

    for term in terms.iter().copied() {
        for factor in cas_math::expr_nary::mul_leaves(ctx, term) {
            let Some((_power, poly)) = low_degree_multiterm_polynomial_power(ctx, factor) else {
                continue;
            };
            let var = poly.var.as_str();

            let mut terms_with_power = 0usize;
            let mut term_with_linear_partner = false;
            for candidate_term in terms.iter().copied() {
                let factors = cas_math::expr_nary::mul_leaves(ctx, candidate_term);
                if !factors
                    .iter()
                    .any(|candidate| same_low_degree_polynomial_power(ctx, factor, *candidate))
                {
                    continue;
                }

                terms_with_power += 1;
                if factors.iter().any(|candidate| {
                    !same_low_degree_polynomial_power(ctx, factor, *candidate)
                        && is_nonconstant_linear_polynomial_factor(ctx, *candidate, var)
                }) {
                    term_with_linear_partner = true;
                }
            }

            if terms_with_power >= 2 && term_with_linear_partner {
                return true;
            }
        }
    }

    false
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

fn positive_const_minus_expr_rhs(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(left, right) if matches!(ctx.get(*left), Expr::Number(n) if n.is_positive()) => {
            Some(*right)
        }
        Expr::Add(left, right) => {
            if matches!(ctx.get(*left), Expr::Number(n) if n.is_positive()) {
                negated_expr(ctx, *right)
            } else if matches!(ctx.get(*right), Expr::Number(n) if n.is_positive()) {
                negated_expr(ctx, *left)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn negated_expr(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Neg(inner) = ctx.get(expr) else {
        return None;
    };
    Some(*inner)
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

/// Normalize small univariate polynomial products through coefficient
/// multiplication, avoiding explicit distributive blow-ups.
pub struct PolynomialProductNormalizeRule;

impl crate::rule::Rule for PolynomialProductNormalizeRule {
    fn name(&self) -> &str {
        "Polynomial Product Normalize"
    }

    fn priority(&self) -> i32 {
        92
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::TRANSFORM
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        if parent_ctx.is_solve_context() {
            return None;
        }

        let rewritten = try_normalize_small_polynomial_product_expr(
            ctx,
            expr,
            PolynomialProductNormalizePolicy::default(),
        )?;

        Some(
            Rewrite::new(rewritten)
                .desc("Expand and combine polynomial product")
                .local(expr, rewritten),
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
