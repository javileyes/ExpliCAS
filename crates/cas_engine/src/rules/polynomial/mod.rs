//! Polynomial manipulation rules: distribution, annihilation, combining like terms,
//! expansion, and factoring.
//!
//! This module is split into submodules:
//! - `expansion`: Binomial/multinomial expansion, auto-expand, polynomial identity detection
//! - `factoring`: Heuristic common factor extraction

mod expansion;
mod expansion_normalize;
mod factoring;

pub use expansion::{
    AutoExpandPowSumRule, AutoExpandSubCancelRule, BinomialExpansionRule,
    SmallMultinomialExpansionRule,
};
pub use expansion_normalize::{
    ExpandSmallBinomialPowRule, HeuristicPolyNormalizeAddRule, PolynomialIdentityZeroRule,
};
pub use factoring::{ExtractCommonMulFactorRule, HeuristicExtractCommonFactorAddRule};

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::Expr;
use cas_math::annihilation_support::{
    should_rewrite_annihilation_to_zero_with_mode_flags, AnnihilationRewriteKind,
};
use cas_math::cube_identity_support::try_rewrite_sum_diff_cubes_product_expr;
use cas_math::distribution_division_support::try_rewrite_div_distribution_simplifying_expr;
use cas_math::distribution_rule_support::try_rewrite_mul_distribution_legacy_expr;
use cas_math::expr_destructure::as_mul;

// ── Sum/Difference of Cubes Contraction Rule ────────────────────────────
//
// Pre-order rule: (X + c)·(X² - c·X + c²) → X³ + c³
//                 (X - c)·(X² + c·X + c²) → X³ - c³
//
// This fires BEFORE DistributeRule to prevent suboptimal splitting of the
// trinomial factor. Works for any base X (polynomial, transcendental, etc.)
define_rule!(
    SumDiffCubesContractionRule,
    "Sum/Difference of Cubes Contraction",
    None,
    PhaseMask::CORE,
    |ctx, expr| {
        let rewrite = try_rewrite_sum_diff_cubes_product_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc("Sum/Difference of cubes")
                .local(expr, rewrite.rewritten),
        )
    }
);

// ── Sqrt Perfect-Square Trinomial Rule ───────────────────────────────────
//
// sqrt(A² + 2·A·B + B²) → |A + B|
//
// Detects perfect-square trinomials inside sqrt and simplifies directly.
// Works for any sub-expressions A, B (polynomial, transcendental, etc.)
//
// Example: sqrt(sin²(u) + 2·sin(u) + 1) → |sin(u) + 1|
//
// We support two forms:
//   (a) A² + 2·A·c + c²  where c is a Number (most common from CSV)
//   (b) Fully symbolic: both A² and B² are Pow(_, 2) nodes

define_rule!(
    SqrtPerfectSquareRule,
    "Sqrt Perfect Square",
    None,
    PhaseMask::CORE,
    |ctx, expr| {
        let rewrite =
            cas_math::perfect_square_support::try_rewrite_sqrt_perfect_square_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc("sqrt(A^2 ± 2AB + B^2) = |A ± B|")
                .local(expr, rewrite.rewritten),
        )
    }
);

// DistributeRule: Runs in CORE, TRANSFORM, RATIONALIZE but NOT in POST
// This prevents Factor↔Distribute infinite loops (FactorCommonIntegerFromAdd runs in POST)
define_rule!(
    DistributeRule,
    "Distributive Property",
    None,
    // NO POST: evita ciclo con FactorCommonIntegerFromAdd (ver test_factor_distribute_no_loop)
    PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE,
    |ctx, expr, parent_ctx| {
        use crate::semantics::NormalFormGoal;

        // GATE: Don't distribute when goal is Collected or Factored
        // This prevents undoing the effect of collect() or factor() commands
        match parent_ctx.goal() {
            NormalFormGoal::Collected | NormalFormGoal::Factored => return None,
            _ => {}
        }

        // Don't distribute if expression is in canonical form (e.g., inside abs() or sqrt())
        // This protects patterns like abs((x-2)(x+2)) from expanding
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }

        // GUARD: Block distribution when sin(4x) identity pattern is detected
        // This allows Sin4xIdentityZeroRule to see 4*sin(t)*cos(t)*(cos²-sin²) as a single product
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.has_sin4x_identity_pattern {
                return None;
            }
        }
        // Use zero-clone destructuring pattern
        let (l, r) = as_mul(ctx, expr)?;

        // GUARD: Skip distribution when a factor is 1.
        // 1*(a+b) -> 1*a + 1*b is a visual no-op (MulOne is applied in rendering),
        // and produces confusing "Before/After identical" steps.
        if cas_math::expr_predicates::is_one_expr(ctx, l)
            || cas_math::expr_predicates::is_one_expr(ctx, r)
        {
            return None;
        }

        // Multiplicative distribution uses cas_math helper that preserves
        // historical guard ordering and semantics.
        let parent_mul_terms =
            parent_ctx
                .immediate_parent()
                .and_then(|parent_id| match ctx.get(parent_id) {
                    Expr::Mul(pl, pr) => Some((*pl, *pr)),
                    _ => None,
                });
        if let Some(rewrite) = try_rewrite_mul_distribution_legacy_expr(ctx, expr, parent_mul_terms)
        {
            return Some(
                Rewrite::new(rewrite.rewritten)
                    .desc("Distribute")
                    .local(expr, rewrite.rewritten),
            );
        }

        if let Some(rewrite) = try_rewrite_div_distribution_simplifying_expr(ctx, expr) {
            return Some(
                Rewrite::new(rewrite.rewritten)
                    .desc("Distribute division (simplifying)")
                    .local(expr, rewrite.rewritten),
            );
        }
        None
    }
);

// AnnihilationRule: Detects and cancels terms like x - x or __hold(sum) - sum
// Domain Mode Policy: Like AddInverseRule, we must respect domain_mode
// because if `x` can be undefined (e.g., a/(a-1) when a=1), then x - x
// is undefined, not 0.
// - Strict: only if no term contains potentially-undefined subexpressions
// - Assume: always apply (educational mode assumption: all expressions are defined)
// - Generic: same as Assume
define_rule!(AnnihilationRule, "Annihilation", |ctx, expr, parent_ctx| {
    if let Some(kind) = should_rewrite_annihilation_to_zero_with_mode_flags(
        ctx,
        expr,
        matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
        matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
        crate::collect::has_undefined_risk,
    ) {
        let zero = ctx.num(0);
        let desc = match kind {
            AnnihilationRewriteKind::TwoTerm => "x - x = 0",
            AnnihilationRewriteKind::HoldSum => "__hold(sum) - sum = 0",
        };
        return Some(Rewrite::new(zero).desc(desc));
    }
    None
});

// CombineLikeTermsRule: Collects like terms in Add/Mul expressions
// Now uses collect_with_semantics for domain_mode awareness:
// - Strict: refuses to cancel terms with undefined risk (e.g., x/(x+1) - x/(x+1))
// - Assume: cancels with domain_assumption warning
// - Generic: cancels unconditionally
define_rule!(
    CombineLikeTermsRule,
    "Combine Like Terms",
    |ctx, expr, parent_ctx| {
        // Only try to collect if it's an Add or Mul
        if matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Mul(_, _)) {
            let plan = crate::collect::plan_collect_rule_rewrite(ctx, expr, parent_ctx)?;
            let mut rewrite = Rewrite::new(plan.new_expr).desc(plan.description);
            if let (Some(before), Some(after)) = (plan.local_before, plan.local_after) {
                rewrite = rewrite.local(before, after);
            }
            Some(rewrite)
        } else {
            None
        }
    }
);

/// BinomialExpansionRule: (a + b)^n → expanded polynomial
/// ONLY expands true binomials (exactly 2 terms).
/// Multinomial expansion (>2 terms) is NOT done by default to avoid explosion.
/// Use explicit expand() mode for multinomial expansion.
/// Implements Rule directly to access ParentContext
pub fn register(simplifier: &mut crate::Simplifier) {
    // Register cube identity contraction BEFORE distribution to prevent suboptimal splits
    simplifier.add_rule(Box::new(SumDiffCubesContractionRule));
    // Sqrt perfect-square trinomial: sqrt(A²+2AB+B²) → |A+B|
    simplifier.add_rule(Box::new(SqrtPerfectSquareRule));
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(BinomialExpansionRule));
    simplifier.add_rule(Box::new(SmallMultinomialExpansionRule));
    // V2.15.8: ExpandSmallBinomialPowRule - controlled by autoexpand_binomials flag
    // Enable via REPL: set autoexpand_binomials on
    simplifier.add_rule(Box::new(ExpandSmallBinomialPowRule));
    simplifier.add_rule(Box::new(AutoExpandPowSumRule));
    simplifier.add_rule(Box::new(AutoExpandSubCancelRule));
    simplifier.add_rule(Box::new(PolynomialIdentityZeroRule));
    // V2.15.8: HeuristicPolyNormalizeAddRule - poly-normalize Add/Sub in Heuristic mode
    // V2.15.9: HeuristicExtractCommonFactorAddRule - extract common factors first (priority 110)
    simplifier.add_rule(Box::new(HeuristicExtractCommonFactorAddRule));
    // V2.16: ExtractCommonMulFactorRule - extract common multiplicative factors from n-ary sums
    // Fixes cross-product NF divergence in metamorphic Mul tests (priority 108)
    simplifier.add_rule(Box::new(ExtractCommonMulFactorRule));
    simplifier.add_rule(Box::new(HeuristicPolyNormalizeAddRule));
}

#[cfg(test)]
mod tests;
