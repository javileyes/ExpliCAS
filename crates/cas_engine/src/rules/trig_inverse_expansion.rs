use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_inverse_expansion_support::try_rewrite_trig_inverse_composition_expr;

// ========== Unified Rule ==========

define_rule!(
    TrigInverseExpansionRule,
    "Trig of Inverse Trig Expansion",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Analytic
    ),
    |ctx, expr, _parent_ctx| {
        let rewrite = try_rewrite_trig_inverse_composition_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// ========== Registration ==========

pub fn register(simplifier: &mut crate::engine::Simplifier) {
    simplifier.add_rule(Box::new(TrigInverseExpansionRule));
}
