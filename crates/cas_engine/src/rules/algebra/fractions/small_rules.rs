//! Light rationalization rules for simple surd denominators.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::div_scalar_into_add_support::{
    try_rewrite_div_scalar_into_add_expr, DivScalarIntoAddRewriteKind,
};
use cas_math::rationalize_single_surd_support::try_rewrite_rationalize_single_surd_expr;

// ========== Light Rationalization for Single Numeric Surd Denominators ==========
// Transforms: num / (k * √n) → (num * √n) / (k * n)
// Only applies when:
// - denominator contains exactly one numeric square root
// - base of the root is a positive integer
// - no variables inside the radical

define_rule!(
    RationalizeSingleSurdRule,
    "Rationalize Single Surd",
    Some(crate::target_kind::TargetKindSet::DIV),
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        let rewritten = try_rewrite_rationalize_single_surd_expr(ctx, expr)?;
        let desc = format_rationalize_single_surd_desc(ctx, rewritten);
        Some(Rewrite::new(rewritten.rewritten).desc(desc))
    }
);

// ========== Distribute Numeric Fraction Into Sum ==========
// After canonicalization, (c₁·A + c₂·B) / d  becomes  Mul(Number(1/d), Add(c₁·A, c₂·B)).
// This rule matches that canonicalized form and distributes:
//   Mul(1/d, Add(c₁·A, c₂·B))  →  (c₁/d)·A + (c₂/d)·B
// when all resulting coefficients are integers (or when GCD simplification is possible).
//
// Examples (after canonicalization):
//   1/2 * (2x + 4y)      →  x + 2y
//   1/3 * (6a + 3b)      →  2a + b
//   1/2 * (2·√3 + 2·x²)  →  √3 + x²
//
// Only applies when:
// - outer factor is a non-integer rational Number(p/q) with q > 1
// - inner expression is Add
// - all term coefficients are divisible by q/gcd(all_coeffs*p, q)
// - distributing actually simplifies (doesn't just rearrange)

define_rule!(
    DivScalarIntoAddRule,
    "Distribute Division Into Sum",
    Some(crate::target_kind::TargetKindSet::MUL),
    PhaseMask::CORE | PhaseMask::POST,
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        let rewrite = try_rewrite_div_scalar_into_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_div_scalar_into_add_desc(rewrite.kind)))
    }
);

fn format_div_scalar_into_add_desc(kind: DivScalarIntoAddRewriteKind) -> &'static str {
    match kind {
        DivScalarIntoAddRewriteKind::AllTermsCancel => "All terms cancel",
        DivScalarIntoAddRewriteKind::FactorCommonCoefficientFromSum => {
            "Factor common coefficient from sum"
        }
    }
}

fn format_rationalize_single_surd_desc(
    ctx: &cas_ast::Context,
    rewrite: cas_math::rationalize_single_surd_support::RationalizeSingleSurdRewrite,
) -> String {
    format!(
        "{} / {} -> {} / {}",
        cas_formatter::render_expr(ctx, rewrite.num),
        cas_formatter::render_expr(ctx, rewrite.den),
        cas_formatter::render_expr(ctx, rewrite.new_num),
        cas_formatter::render_expr(ctx, rewrite.new_den)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use crate::Simplifier;
    use cas_ast::target_kind::TargetKind;
    use cas_parser::parse;

    #[test]
    fn rationalize_single_surd_rule_targets_div_only() {
        let target_types = RationalizeSingleSurdRule
            .target_types()
            .expect("RationalizeSingleSurdRule should be structurally targeted");
        assert!(target_types.contains(TargetKind::Div));
        assert!(!target_types.contains(TargetKind::Mul));
    }

    #[test]
    fn simplifier_applies_rationalize_single_surd_rule() {
        let mut simplifier = Simplifier::with_default_rules();
        let expr = parse("x/sqrt(3)", &mut simplifier.context).expect("parse");

        let (_result, steps) = simplifier.simplify(expr);

        let applied = steps
            .iter()
            .any(|step| step.rule_name.starts_with("Rationalize Single Surd"));
        assert!(
            applied,
            "RationalizeSingleSurdRule should apply from the default simplifier"
        );
    }

    #[test]
    fn div_scalar_into_add_rule_targets_mul_only() {
        let target_types = DivScalarIntoAddRule
            .target_types()
            .expect("DivScalarIntoAddRule should be structurally targeted");
        assert!(target_types.contains(TargetKind::Mul));
        assert!(!target_types.contains(TargetKind::Add));
    }

    #[test]
    fn simplifier_applies_div_scalar_into_add_rule() {
        let mut simplifier = Simplifier::with_default_rules();
        let expr = parse("1/2*(2*x + 4*y)", &mut simplifier.context).expect("parse");

        let (_result, steps) = simplifier.simplify(expr);

        let applied = steps
            .iter()
            .any(|step| step.rule_name.starts_with("Distribute Division Into Sum"));
        assert!(
            applied,
            "DivScalarIntoAddRule should apply from the default simplifier"
        );
    }
}
