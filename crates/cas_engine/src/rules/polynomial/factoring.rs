//! Polynomial factoring rules: common factor extraction from sums.

use crate::rule::Rewrite;
use cas_ast::{Context, ExprId};

/// HeuristicExtractCommonFactorAddRule: Extract common base factors from sums
pub struct HeuristicExtractCommonFactorAddRule;

impl crate::rule::Rule for HeuristicExtractCommonFactorAddRule {
    fn name(&self) -> &str {
        "Heuristic Extract Common Factor"
    }

    fn priority(&self) -> i32 {
        110 // Higher than HeuristicPolyNormalizeAddRule (100) to try factorization first
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::TRANSFORM
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

        let new_expr =
            cas_math::factoring_add_support::try_extract_common_factor_add_expr(ctx, expr)?;

        Some(
            Rewrite::new(new_expr)
                .desc("Extract common polynomial factor")
                .local(expr, new_expr),
        )
    }
}

// =============================================================================
// ExtractCommonMulFactorRule — extract common multiplicative factors from n-ary sums
// =============================================================================

/// ExtractCommonMulFactorRule: Extract common multiplicative factors from sums.
///
/// Pattern: `f·a + f·b + f·c` → `f·(a + b + c)`
///
/// Unlike `HeuristicExtractCommonFactorAddRule` (which only handles 2-term sums
/// with polynomial bases), this rule handles:
/// - N-ary sums (any number of terms)
/// - Any expression type as common factor (functions, powers, symbols, etc.)
///
/// This fixes cross-product normal form divergence in metamorphic Mul tests,
/// where DistributeRule distributes `f(x)·(a+b+c)` into individual terms
/// but there's no rule to factor `f(x)` back out.
pub struct ExtractCommonMulFactorRule;

impl crate::rule::Rule for ExtractCommonMulFactorRule {
    fn name(&self) -> &str {
        "Extract Common Multiplicative Factor"
    }

    fn priority(&self) -> i32 {
        108 // Slightly below HeuristicExtractCommonFactorAddRule (110),
            // but above HeuristicPolyNormalizeAddRule (100)
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // POST only: prevents Distribute↔Extract infinite loop.
        // DistributeRule runs in CORE|TRANSFORM|RATIONALIZE (no POST),
        // so phase separation breaks the oscillation cycle.
        // This mirrors FactorCommonIntegerFromAdd which also runs in POST.
        crate::phase::PhaseMask::POST
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
        // Skip in Solve mode — solver needs specific polynomial forms
        if parent_ctx.is_solve_context() {
            return None;
        }

        let new_expr = cas_math::factoring_mul_support::try_extract_common_mul_factor_expr(
            ctx,
            expr,
            cas_math::factoring_mul_support::ExtractCommonMulFactorPolicy::default(),
        )?;

        Some(
            Rewrite::new(new_expr)
                .desc("Factor out common multiplicative factor from sum")
                .local(expr, new_expr),
        )
    }
}
