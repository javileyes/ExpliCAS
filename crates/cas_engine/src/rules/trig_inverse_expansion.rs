use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_identity_zero_support::match_cos_triple_identity_zero_expr;
use cas_math::trig_inverse_expansion_support::{
    try_rewrite_trig_inverse_composition_expr, TrigInverseExpansionKind,
};

// ========== Unified Rule ==========

define_rule!(
    TrigInverseExpansionRule,
    "Trig of Inverse Trig Expansion",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Analytic
    ),
    |ctx, expr, _parent_ctx| {
        if _parent_ctx.has_ancestor_matching(ctx, |ctx, ancestor| {
            match_cos_triple_identity_zero_expr(ctx, ancestor)
        }) {
            return None;
        }
        let rewrite = try_rewrite_trig_inverse_composition_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_trig_inverse_expansion_desc(
            rewrite.kind,
        )))
    }
);

fn format_trig_inverse_expansion_desc(kind: TrigInverseExpansionKind) -> &'static str {
    match kind {
        TrigInverseExpansionKind::SinArctan => "sin(arctan(x)) → x/√(1+x²)",
        TrigInverseExpansionKind::CosArctan => "cos(arctan(x)) → 1/√(1+x²)",
        TrigInverseExpansionKind::TanArcsin => "tan(arcsin(x)) → x/√(1-x²)",
        TrigInverseExpansionKind::CotArcsin => "cot(arcsin(x)) → √(1-x²)/x",
        TrigInverseExpansionKind::CosArcsin => "cos(arcsin(x)) → √(1-x²)",
        TrigInverseExpansionKind::SinArccos => "sin(arccos(x)) → √(1-x²)",
        TrigInverseExpansionKind::SinArcsec => "sin(arcsec(x)) → √(x²-1)/x",
        TrigInverseExpansionKind::CosArcsec => "cos(arcsec(x)) → 1/x",
        TrigInverseExpansionKind::TanArccos => "tan(arccos(x)) → √(1-x²)/x",
        TrigInverseExpansionKind::CotArccos => "cot(arccos(x)) → x/√(1-x²)",
        TrigInverseExpansionKind::SecArctan => "sec(arctan(x)) → √(1+x²)",
        TrigInverseExpansionKind::CscArctan => "csc(arctan(x)) → √(1+x²)/x",
        TrigInverseExpansionKind::SecArcsin => "sec(arcsin(x)) → 1/√(1-x²)",
        TrigInverseExpansionKind::CscArcsin => "csc(arcsin(x)) → 1/x",
    }
}

// ========== Registration ==========

pub fn register(simplifier: &mut crate::engine::Simplifier) {
    simplifier.add_rule(Box::new(TrigInverseExpansionRule));
}
