//! Trig values and specialized identity rules.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_canonicalization_support::{
    try_rewrite_csc_cot_pythagorean_identity_expr, try_rewrite_sec_tan_pythagorean_identity_expr,
    try_rewrite_trig_quotient_div_expr,
};
use cas_math::trig_tan_triple_support::try_rewrite_tan_triple_product_mul_expr;
use cas_math::trig_value_detection_support::try_plan_tan_to_sin_cos_with_policy;

// =============================================================================
// TRIPLE TANGENT PRODUCT IDENTITY
// tan(u) · tan(π/3 - u) · tan(π/3 + u) = tan(3u)
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
struct TanTripleDidacticSubstep {
    title: &'static str,
    details: Vec<String>,
}

fn tan_triple_product_desc() -> &'static str {
    "tan(u)·tan(π/3+u)·tan(π/3−u) = tan(3u)"
}

fn tan_triple_product_substeps(u_display: &str) -> Vec<TanTripleDidacticSubstep> {
    vec![
        TanTripleDidacticSubstep {
            title: "Normalizar argumentos",
            details: vec![
                "π/3 − u se representa como −u + π/3 para comparar como u + const".to_string(),
            ],
        },
        TanTripleDidacticSubstep {
            title: "Reconocer patrón",
            details: vec![
                format!("Sea u = {}", u_display),
                "Factores: tan(u), tan(u + π/3), tan(π/3 − u)".to_string(),
            ],
        },
        TanTripleDidacticSubstep {
            title: "Aplicar identidad",
            details: vec!["tan(u)·tan(u + π/3)·tan(π/3 − u) = tan(3u)".to_string()],
        },
    ]
}

/// Matches tan(u)·tan(π/3+u)·tan(π/3-u) and simplifies to tan(3u).
/// Must run BEFORE TanToSinCosRule to prevent expansion.
pub struct TanTripleProductRule;

impl crate::rule::Rule for TanTripleProductRule {
    fn name(&self) -> &str {
        "Triple Tangent Product (π/3)"
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::TRANSFORM
    }

    fn soundness(&self) -> crate::rule::SoundnessLabel {
        // This rule introduces requires (cos ≠ 0) for the tangent definitions
        crate::rule::SoundnessLabel::EquivalenceUnderIntroducedRequires
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let rewrite = try_rewrite_tan_triple_product_mul_expr(ctx, expr)?;
        let u_str = cas_formatter::DisplayExpr {
            context: ctx,
            id: rewrite.u,
        }
        .to_string();

        let mut out = Rewrite::new(rewrite.rewritten).desc(tan_triple_product_desc());
        for sub in tan_triple_product_substeps(&u_str) {
            out = out.substep(sub.title, sub.details);
        }
        out = out.requires(crate::ImplicitCondition::NonZero(rewrite.nonzero_cos_u));
        out = out.requires(crate::ImplicitCondition::NonZero(
            rewrite.nonzero_cos_u_plus_pi_over_3,
        ));
        out = out.requires(crate::ImplicitCondition::NonZero(
            rewrite.nonzero_cos_pi_over_3_minus_u,
        ));
        Some(out)
    }
}

/// Convert tan(x) to sin(x)/cos(x) UNLESS it's part of a Pythagorean pattern
pub struct TanToSinCosRule;

impl crate::rule::Rule for TanToSinCosRule {
    fn name(&self) -> &str {
        "Tan to Sin/Cos"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let plan = try_plan_tan_to_sin_cos_with_policy(
            ctx,
            expr,
            parent_ctx.pattern_marks(),
            parent_ctx.immediate_parent(),
            parent_ctx.all_ancestors(),
        )?;
        Some(crate::rule::Rewrite::new(plan.rewritten).desc(plan.desc))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // Exclude PostCleanup to avoid cycle with TrigQuotientRule
        // TanToSinCos expands for algebra, TrigQuotient reconverts to canonical form
        // NOTE: CORE is included because some tests (e.g., test_tangent_sum) need tan→sin/cos expansion
        // TanTripleProductRule is registered BEFORE this rule and will handle triple product patterns
        crate::phase::PhaseMask::CORE
            | crate::phase::PhaseMask::TRANSFORM
            | crate::phase::PhaseMask::RATIONALIZE
    }
}

/// Convert trig quotients to their canonical function forms:
/// - sin(x)/cos(x) → tan(x)
/// - cos(x)/sin(x) → cot(x)
/// - 1/sin(x) → csc(x)
/// - 1/cos(x) → sec(x)
/// - 1/tan(x) → cot(x)
pub struct TrigQuotientRule;

impl crate::rule::Rule for TrigQuotientRule {
    fn name(&self) -> &str {
        "Trig Quotient"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let plan = try_rewrite_trig_quotient_div_expr(ctx, expr)?;
        Some(crate::rule::Rewrite::new(plan.rewritten).desc(plan.desc))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // Only run in PostCleanup to avoid cycle with TanToSinCosRule
        crate::phase::PhaseMask::POST
    }
}

// Secant-Tangent Pythagorean Identity: sec²(x) - tan²(x) = 1
// Also recognizes factored form: (sec(x) + tan(x)) * (sec(x) - tan(x)) = 1
define_rule!(
    SecTanPythagoreanRule,
    "Secant-Tangent Pythagorean Identity",
    |ctx, expr| {
        let rewrite = try_rewrite_sec_tan_pythagorean_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Cosecant-Cotangent Pythagorean Identity: csc²(x) - cot²(x) = 1
// NOTE: Subtraction is normalized to Add(a, Neg(b))
define_rule!(
    CscCotPythagoreanRule,
    "Cosecant-Cotangent Pythagorean Identity",
    |ctx, expr| {
        let rewrite = try_rewrite_csc_cot_pythagorean_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

#[cfg(test)]
mod tests {
    use super::{tan_triple_product_desc, tan_triple_product_substeps};

    #[test]
    fn tan_triple_didactic_builder_contains_pattern_and_identity() {
        let substeps = tan_triple_product_substeps("x");
        assert_eq!(
            tan_triple_product_desc(),
            "tan(u)·tan(π/3+u)·tan(π/3−u) = tan(3u)"
        );
        assert_eq!(substeps.len(), 3);
        assert!(substeps[1].details.iter().any(|d| d.contains("Sea u = x")));
    }
}
