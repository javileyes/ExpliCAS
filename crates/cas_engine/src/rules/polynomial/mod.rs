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
use crate::nary::{build_balanced_add, AddView, Sign};
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Expr, ExprId};
use cas_math::annihilation_support::{
    should_rewrite_annihilation_to_zero_with, AnnihilationRewriteKind,
};
use cas_math::cube_identity_support::try_rewrite_sum_diff_cubes_product_expr;
use cas_math::distribution_guard_support::estimate_division_distribution_simplification_reduction;
use cas_math::distribution_rule_support::try_rewrite_mul_distribution_legacy_expr;
use cas_math::expr_destructure::{as_div, as_mul};

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
                .desc(rewrite.desc)
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
                .desc(rewrite.desc)
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
                    .desc(rewrite.desc)
                    .local(expr, rewrite.rewritten),
            );
        }

        // Handle Division Distribution: (a + b) / c -> a/c + b/c
        // Using AddView for shape-independent n-ary handling
        if let Some((numer, denom)) = as_div(ctx, expr) {
            // N-ARY: Use AddView for shape-independent handling of sums
            // This correctly handles ((a+b)+c), (a+(b+c)), and balanced trees
            let num_view = AddView::from_expr(ctx, numer);

            // Check if it's actually a sum (more than 1 term)
            if num_view.terms.len() > 1 {
                // Calculate total reduction potential
                let mut total_reduction: usize = 0;
                let mut any_simplifies = false;

                for &(term, _sign) in &num_view.terms {
                    let red =
                        estimate_division_distribution_simplification_reduction(ctx, term, denom);
                    if red > 0 {
                        any_simplifies = true;
                        total_reduction += red;
                    }
                }

                // Only distribute if at least one term simplifies
                if any_simplifies {
                    // Build new terms: each term divided by denominator
                    let new_terms: Vec<ExprId> = num_view
                        .terms
                        .iter()
                        .map(|&(term, sign)| {
                            let div_term = ctx.add(Expr::Div(term, denom));
                            match sign {
                                Sign::Pos => div_term,
                                Sign::Neg => ctx.add(Expr::Neg(div_term)),
                            }
                        })
                        .collect();

                    // Rebuild as balanced sum
                    let new_expr = build_balanced_add(ctx, &new_terms);

                    // Check complexity to prevent cycles with AddFractionsRule
                    let old_complexity = cas_ast::count_nodes(ctx, expr);
                    let new_complexity = cas_ast::count_nodes(ctx, new_expr);

                    // Allow if predicted complexity (after simplification) is not worse
                    if new_complexity <= old_complexity + total_reduction {
                        return Some(
                            Rewrite::new(new_expr)
                                .desc("Distribute division (simplifying)")
                                .local(expr, new_expr),
                        );
                    }
                }
            }
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
    let strict_domain = parent_ctx.domain_mode() == crate::DomainMode::Strict;
    if let Some(kind) =
        should_rewrite_annihilation_to_zero_with(ctx, expr, strict_domain, |core_ctx, term| {
            crate::collect::has_undefined_risk(core_ctx, term)
        })
    {
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
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;

    #[test]
    fn test_distribute() {
        let mut ctx = Context::new();
        let rule = DistributeRule;
        // x^2 * (x + 3) - use x^2 (not an integer) so guard doesn't block
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x_sq = ctx.add(Expr::Pow(x, two));
        let add = ctx.add(Expr::Add(x, three));
        let expr = ctx.add(Expr::Mul(x_sq, add));

        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // Should be (x^2 * x) + (x^2 * 3) before further simplification
        // Note: x^2*x -> x^3 happens in a later pass, not in DistributeRule
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "x^2 * x + x^2 * 3" // Canonical: polynomial order (x terms before constants)
        );
    }

    #[test]
    fn test_annihilation() {
        let mut ctx = Context::new();
        let rule = AnnihilationRule;
        let x = ctx.var("x");
        let expr = ctx.add(Expr::Sub(x, x));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
    }

    #[test]
    fn test_annihilation_hold_sum_pattern() {
        let mut ctx = Context::new();
        let rule = AnnihilationRule;
        let x = ctx.var("x");
        let y = ctx.var("y");
        let sum = ctx.add(Expr::Add(x, y));
        let held = cas_ast::hold::wrap_hold(&mut ctx, sum);
        let neg_x = ctx.add(Expr::Neg(x));
        let neg_y = ctx.add(Expr::Neg(y));
        let rhs = ctx.add(Expr::Add(neg_x, neg_y));
        let expr = ctx.add(Expr::Add(held, rhs));

        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
    }

    #[test]
    fn test_combine_like_terms() {
        let mut ctx = Context::new();
        let rule = CombineLikeTermsRule;

        // 2x + 3x -> 5x
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let term1 = ctx.add(Expr::Mul(two, x));
        let term2 = ctx.add(Expr::Mul(three, x));
        let expr = ctx.add(Expr::Add(term1, term2));

        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "5 * x"
        );

        // x + 2x -> 3x
        let term1 = x;
        let term2 = ctx.add(Expr::Mul(two, x));
        let expr2 = ctx.add(Expr::Add(term1, term2));
        let rewrite2 = rule
            .apply(
                &mut ctx,
                expr2,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite2.new_expr
                }
            ),
            "3 * x"
        );

        // ln(x) + ln(x) -> 2 * ln(x)
        let ln_x = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![x]);
        let expr3 = ctx.add(Expr::Add(ln_x, ln_x));
        let rewrite3 = rule
            .apply(
                &mut ctx,
                expr3,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // ln(x) is log(e, x), prints as ln(x)
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite3.new_expr
                }
            ),
            "2 * ln(x)"
        );
    }

    #[test]
    fn test_polynomial_identity_zero_rule() {
        // Test: (a+b)^2 - (a^2 + 2ab + b^2) = 0
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");

        // (a+b)^2
        let a_plus_b = ctx.add(Expr::Add(a, b));
        let two = ctx.num(2);
        let a_plus_b_sq = ctx.add(Expr::Pow(a_plus_b, two));

        // a^2 + 2ab + b^2
        let a_sq = ctx.add(Expr::Pow(a, two));
        let b_sq = ctx.add(Expr::Pow(b, two));
        let ab = ctx.add(Expr::Mul(a, b));
        let two_ab = ctx.add(Expr::Mul(two, ab));
        let sum1 = ctx.add(Expr::Add(a_sq, two_ab));
        let rhs = ctx.add(Expr::Add(sum1, b_sq));

        // (a+b)^2 - (a^2 + 2ab + b^2)
        let expr = ctx.add(Expr::Sub(a_plus_b_sq, rhs));

        let rule = PolynomialIdentityZeroRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        // Should simplify to 0
        assert!(rewrite.is_some(), "Polynomial identity should be detected");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.unwrap().new_expr
                }
            ),
            "0"
        );
    }

    #[test]
    fn test_polynomial_identity_zero_rule_non_identity() {
        // Test: (a+b)^2 - a^2 ≠ 0 (not an identity)
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");

        // (a+b)^2
        let a_plus_b = ctx.add(Expr::Add(a, b));
        let two = ctx.num(2);
        let a_plus_b_sq = ctx.add(Expr::Pow(a_plus_b, two));

        // a^2
        let a_sq = ctx.add(Expr::Pow(a, two));

        // (a+b)^2 - a^2
        let expr = ctx.add(Expr::Sub(a_plus_b_sq, a_sq));

        let rule = PolynomialIdentityZeroRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        // Should NOT return a rewrite (not an identity to 0)
        assert!(rewrite.is_none(), "Non-identity should not trigger rule");
    }
}
