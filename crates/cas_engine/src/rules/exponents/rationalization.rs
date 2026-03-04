use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, ExprId};

// =============================================================================
// RationalizeLinearSqrtDenRule: 1/(sqrt(t)+c) → (sqrt(t)-c)/(t-c²)
// =============================================================================
// Rationalizes denominators with linear sqrt terms by multiplying by conjugate.
// This is a canonical transformation that eliminates radicals from denominators.
//
// Examples:
//   1/(sqrt(2)+1) → (sqrt(2)-1)/1 = sqrt(2)-1
//   1/(sqrt(3)+1) → (sqrt(3)-1)/2
//   1/(sqrt(u)+1) → (sqrt(u)-1)/(u-1)
//   2/(sqrt(3)-1) → 2*(sqrt(3)+1)/2 = sqrt(3)+1
//
// Guard: Only apply when result is simpler (no radicals in denominator)
// =============================================================================
define_rule!(
    RationalizeLinearSqrtDenRule,
    "Rationalize Linear Sqrt Denominator",
    |ctx, expr| {
        let rewrite =
            cas_math::root_den_rationalize_support::try_rewrite_rationalize_linear_sqrt_den_expr(
                ctx, expr,
            )?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// RationalizeSumOfSqrtsDenRule: k/(sqrt(p)+sqrt(q)) → k*(sqrt(p)-sqrt(q))/(p-q)
// =============================================================================
// Rationalizes denominators with sum of two square roots.
//
// Examples:
//   3/(sqrt(2)+sqrt(3)) → 3*(sqrt(2)-sqrt(3))/(2-3) = -3*(sqrt(2)-sqrt(3))
//   1/(sqrt(5)+sqrt(2)) → (sqrt(5)-sqrt(2))/3
// =============================================================================
define_rule!(
    RationalizeSumOfSqrtsDenRule,
    "Rationalize Sum of Sqrts Denominator",
    |ctx, expr| {
        let rewrite =
            cas_math::root_den_rationalize_support::try_rewrite_rationalize_sum_of_sqrts_den_expr(
                ctx, expr,
            )?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// CubeRootDenRationalizeRule: k/(1+u^(1/3)) → k*(1-u^(1/3)+u^(2/3))/(1+u)
// =============================================================================
// Uses the sum of cubes identity: 1 + r³ = (1 + r)(1 - r + r²)
// So: 1/(1+r) = (1-r+r²)/(1+r³)
// With r = u^(1/3), r³ = u
//
// Similarly for difference: 1 - r³ = (1 - r)(1 + r + r²)
// So: 1/(1-r) = (1+r+r²)/(1-r³)
//
// Examples:
//   1/(1+u^(1/3)) → (1-u^(1/3)+u^(2/3))/(1+u)
//   1/(1-u^(1/3)) → (1+u^(1/3)+u^(2/3))/(1-u)
// =============================================================================
define_rule!(
    CubeRootDenRationalizeRule,
    "Rationalize Cube Root Denominator",
    |ctx, expr| {
        let rewrite =
            cas_math::root_den_rationalize_support::try_rewrite_rationalize_cube_root_den_expr(
                ctx, expr,
            )?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// RootMergeMulRule: sqrt(a) * sqrt(b) → sqrt(a*b)
// =============================================================================
// Merges products of square roots into a single root.
// This is valid for non-negative real a and b.
//
// Examples:
//   sqrt(u) * sqrt(b) → sqrt(u*b)
//   u^(1/2) * b^(1/2) → (u*b)^(1/2)
//
// Requires: a ≥ 0 and b ≥ 0 (or they are squared terms)
// =============================================================================
pub struct RootMergeMulRule;

impl crate::rule::Rule for RootMergeMulRule {
    fn name(&self) -> &str {
        "Merge Sqrt Product"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let strict_mode = matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict);
        let vd = parent_ctx.value_domain();
        let rewrite = cas_math::root_power_canonical_support::try_rewrite_root_merge_mul_expr_with(
            ctx,
            expr,
            strict_mode,
            |core_ctx, inner| crate::helpers::prove_nonnegative_core(core_ctx, inner, vd),
        )?;

        let mut out = Rewrite::new(rewrite.rewritten).desc("√a · √b = √(a·b)");
        if rewrite.assume_left_nonnegative {
            out = out.assume(crate::AssumptionEvent::nonnegative(ctx, rewrite.left_base));
        }
        if rewrite.assume_right_nonnegative {
            out = out.assume(crate::AssumptionEvent::nonnegative(ctx, rewrite.right_base));
        }
        Some(out)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

// =============================================================================
// RootMergeDivRule: sqrt(a) / sqrt(b) → sqrt(a/b)
// =============================================================================
// Merges quotients of square roots into a single root.
// This is valid for non-negative real a and positive b.
//
// Examples:
//   sqrt(u) / sqrt(b) → sqrt(u/b)
//   u^(1/2) / b^(1/2) → (u/b)^(1/2)
//
// Requires: a ≥ 0 and b > 0
// =============================================================================
pub struct RootMergeDivRule;

impl crate::rule::Rule for RootMergeDivRule {
    fn name(&self) -> &str {
        "Merge Sqrt Quotient"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let strict_mode = matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict);
        let vd = parent_ctx.value_domain();
        let rewrite = cas_math::root_power_canonical_support::try_rewrite_root_merge_div_expr_with(
            ctx,
            expr,
            strict_mode,
            |core_ctx, inner| crate::helpers::prove_nonnegative_core(core_ctx, inner, vd),
            |core_ctx, inner| crate::helpers::prove_positive_core(core_ctx, inner, vd),
        )?;

        let mut out = Rewrite::new(rewrite.rewritten).desc("√a / √b = √(a/b)");
        if rewrite.assume_num_nonnegative {
            out = out.assume(crate::AssumptionEvent::nonnegative(ctx, rewrite.num_base));
        }
        if rewrite.assume_den_positive {
            out = out.assume(crate::AssumptionEvent::positive(ctx, rewrite.den_base));
        }
        Some(out)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

// =============================================================================
// PowPowCancelReciprocalRule: (u^y)^(1/y) → u
// =============================================================================
// Cancels reciprocal exponents in nested powers.
// This is valid for u > 0 and y ≠ 0 in real domain.
//
// Examples:
//   (u^y)^(1/y) → u
//   (x^n)^(1/n) → x
//
// Requires: u > 0 (base), y ≠ 0 (exponent)
// =============================================================================
pub struct PowPowCancelReciprocalRule;

impl crate::rule::Rule for PowPowCancelReciprocalRule {
    fn name(&self) -> &str {
        "Cancel Reciprocal Exponents"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let strict_mode = matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict);
        let vd = parent_ctx.value_domain();
        let rewrite =
            cas_math::root_power_canonical_support::try_rewrite_powpow_cancel_reciprocal_expr_with(
                ctx,
                expr,
                strict_mode,
                |core_ctx, inner| crate::helpers::prove_positive_core(core_ctx, inner, vd),
                crate::helpers::prove_nonzero_core,
            )?;

        let mut out = Rewrite::new(rewrite.rewritten).desc("(u^y)^(1/y) = u");
        if rewrite.assume_base_positive {
            out = out.assume(crate::AssumptionEvent::positive(ctx, rewrite.base));
        }
        if rewrite.assume_exp_nonzero {
            out = out.assume(crate::AssumptionEvent::nonzero(ctx, rewrite.inner_exp));
        }
        Some(out)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

// =============================================================================
// ReciprocalSqrtCanonRule: Canonicalize reciprocal sqrt forms to Pow(x, -1/2)
// =============================================================================
// Ensures all representations of "1/√x" converge to a single canonical AST:
//
//   Pattern 1: 1/√x      = Div(1, Pow(x, 1/2))     → Pow(x, -1/2)
//   Pattern 2: √x/x      = Div(Pow(x, 1/2), x)     → Pow(x, -1/2)
//   Pattern 3: √(x^(-1)) = Pow(Pow(x,-1), 1/2)     → already handled by PowerPowerRule
//
// GUARD: Only applied when the base contains symbols (variables).
// Pure numeric bases (e.g., 1/√2) are left as-is to avoid creating Pow(2, -1/2)
// forms that Strict-mode verification cannot fold back to √2/2.
//
// This is sound in RealOnly: all forms require x > 0, same definability domain.
// No cycle risk: NegativeExponentNormalizationRule only fires on INTEGER negative
// exponents, and -1/2 is not integer.
// =============================================================================

pub struct ReciprocalSqrtCanonRule;

impl crate::rule::Rule for ReciprocalSqrtCanonRule {
    fn name(&self) -> &str {
        "Canonicalize Reciprocal Sqrt"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let rewrite =
            cas_math::reciprocal_sqrt_canon_support::try_rewrite_reciprocal_sqrt_canon_expr(
                ctx, expr,
            )?;
        Some(crate::rule::Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Low
    }
}
