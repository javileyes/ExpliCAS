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
        SimplifySquareRootRewriteKind::AdditiveCommonFactor => {
            "Extract common square factor from additive radicand"
        }
        SimplifySquareRootRewriteKind::QuotientOfSquares => {
            "Simplify square root of quotient of squares"
        }
    }
}

define_rule!(
    RootDenestingRule,
    "Root Denesting",
    Some(crate::target_kind::TargetKindSet::FUNCTION | crate::target_kind::TargetKindSet::POW),
    |ctx, expr| {
        let rewrite = cas_math::root_forms::try_rewrite_root_denesting_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(format_root_denesting_desc(rewrite.kind))
                .local(expr, rewrite.rewritten),
        )
    }
);

define_rule!(
    SimplifySquareRootRule,
    "Simplify Square Root",
    Some(crate::target_kind::TargetKindSet::FUNCTION | crate::target_kind::TargetKindSet::POW),
    |ctx, expr, parent_ctx| {
        // REAL-ONLY: every arm of this helper emits an |·| form of a symbolic
        // square (`√(X²) → |X|`), a real-domain identity — over ℂ,
        // `√(i²) = √(-1) = i ≠ 1 = |i|`. Numeric folds (`√4 → 2`) have other
        // owners and keep working in complex mode.
        if parent_ctx.value_domain() != crate::semantics::ValueDomain::RealOnly {
            return None;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use crate::Simplifier;
    use cas_ast::target_kind::TargetKind;
    use cas_ast::Expr;
    use cas_parser::parse;
    use num_rational::BigRational;

    #[test]
    fn root_denesting_rule_targets_function_and_pow_only() {
        let target_types = RootDenestingRule
            .target_types()
            .expect("RootDenestingRule should be structurally targeted");
        assert!(target_types.contains(TargetKind::Function));
        assert!(target_types.contains(TargetKind::Pow));
        assert!(!target_types.contains(TargetKind::Add));
        assert!(!target_types.contains(TargetKind::Mul));
    }

    #[test]
    fn root_denesting_rule_applies_to_function_form() {
        let mut simplifier = Simplifier::with_default_rules();
        let expr = parse("sqrt(3 + 2*sqrt(2))", &mut simplifier.context).expect("parse");
        let rule = RootDenestingRule;

        let rewrite = rule.apply(
            &mut simplifier.context,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(
            rewrite.is_some(),
            "RootDenestingRule should still match sqrt(...) function form"
        );
    }

    #[test]
    fn root_denesting_rule_applies_to_pow_half_form() {
        let mut simplifier = Simplifier::with_default_rules();
        let base = parse("3 + 2*sqrt(2)", &mut simplifier.context).expect("parse");
        let half = simplifier
            .context
            .add(Expr::Number(BigRational::new(1.into(), 2.into())));
        let expr = simplifier.context.add(Expr::Pow(base, half));
        let rule = RootDenestingRule;

        let rewrite = rule.apply(
            &mut simplifier.context,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(
            rewrite.is_some(),
            "RootDenestingRule should still match Pow(_, 1/2)"
        );
    }

    #[test]
    fn simplify_square_root_rule_targets_function_and_pow_only() {
        let target_types = SimplifySquareRootRule
            .target_types()
            .expect("SimplifySquareRootRule should be structurally targeted");
        assert!(target_types.contains(TargetKind::Function));
        assert!(target_types.contains(TargetKind::Pow));
        assert!(!target_types.contains(TargetKind::Add));
        assert!(!target_types.contains(TargetKind::Mul));
    }

    #[test]
    fn simplifier_applies_simplify_square_root_rule_to_function_form() {
        let mut simplifier = Simplifier::with_default_rules();
        let expr = parse("sqrt(x^2 + 2*x + 1)", &mut simplifier.context).expect("parse");

        let (_result, steps) = simplifier.simplify(expr);

        let applied = steps
            .iter()
            .any(|step| step.rule_name.starts_with("Simplify Square Root"));
        assert!(
            applied,
            "SimplifySquareRootRule should apply from the default simplifier"
        );
    }

    #[test]
    fn simplify_square_root_rule_applies_to_pow_half_form() {
        let mut simplifier = Simplifier::with_default_rules();
        let base = parse("x^2 + 2*x + 1", &mut simplifier.context).expect("parse");
        let half = simplifier
            .context
            .add(Expr::Number(BigRational::new(1.into(), 2.into())));
        let expr = simplifier.context.add(Expr::Pow(base, half));
        let rule = SimplifySquareRootRule;

        let rewrite = rule.apply(
            &mut simplifier.context,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(
            rewrite.is_some(),
            "SimplifySquareRootRule should still match Pow(_, 1/2)"
        );
    }
}
