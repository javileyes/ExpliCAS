use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::root_forms::{
    ExtractPerfectPowerFromRadicandKind, RootDenestingRewriteKind, SimplifySquareRootRewriteKind,
};

fn format_root_denesting_desc(kind: RootDenestingRewriteKind) -> &'static str {
    match kind {
        RootDenestingRewriteKind::DenestSquareRoot => "Denest square root",
    }
}

fn format_extract_perfect_power_desc(kind: ExtractPerfectPowerFromRadicandKind) -> &'static str {
    match kind {
        ExtractPerfectPowerFromRadicandKind::ExtractPerfectSquare => {
            "Extract perfect square from under radical"
        }
    }
}

fn format_simplify_square_root_desc(kind: SimplifySquareRootRewriteKind) -> &'static str {
    match kind {
        SimplifySquareRootRewriteKind::PerfectSquare => "Simplify perfect square root",
        SimplifySquareRootRewriteKind::SquareRootFactors => "Simplify square root factors",
    }
}

define_rule!(RootDenestingRule, "Root Denesting", |ctx, expr| {
    let rewrite = cas_math::root_forms::try_rewrite_root_denesting_expr(ctx, expr)?;
    Some(
        Rewrite::new(rewrite.rewritten)
            .desc(format_root_denesting_desc(rewrite.kind))
            .local(expr, rewrite.rewritten),
    )
});

define_rule!(
    SimplifySquareRootRule,
    "Simplify Square Root",
    |ctx, expr| {
        let rewrite = cas_math::root_forms::try_rewrite_simplify_square_root_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_simplify_square_root_desc(rewrite.kind)))
    }
);

// Rule: (k²·a)^(1/2) → k·a^(1/2)  (and more generally (k^n·a)^(1/n) → k·a^(1/n))
// Extracts the largest perfect-square numeric factor from under a square root.
// Examples:
//   sqrt(4u) → 2·√u
//   sqrt(9u) → 3·√u
//   sqrt(8u) → 2·√(2u)
//   (4u)^(1/2) → 2·u^(1/2)
define_rule!(
    ExtractPerfectSquareFromRadicandRule,
    "Extract Perfect Square from Radicand",
    Some(crate::target_kind::TargetKindSet::POW),
    |ctx, expr| {
        let rewrite =
            cas_math::root_forms::try_rewrite_extract_perfect_power_from_radicand_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_extract_perfect_power_desc(rewrite.kind)))
    }
);
