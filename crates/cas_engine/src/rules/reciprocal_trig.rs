use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_reciprocal_eval_support::{
    try_rewrite_eval_reciprocal_trig_expr, try_rewrite_negative_reciprocal_trig_expr,
    try_rewrite_reciprocal_trig_composition_expr,
};

define_rule!(
    EvaluateReciprocalTrigRule,
    "Evaluate Reciprocal Trig Functions",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_eval_reciprocal_trig_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    ReciprocalTrigCompositionRule,
    "Reciprocal Trig Composition",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_reciprocal_trig_composition_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    ReciprocalTrigNegativeRule,
    "Reciprocal Trig Negative Argument",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_negative_reciprocal_trig_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// ==================== Registration ====================

pub fn register(simplifier: &mut crate::engine::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateReciprocalTrigRule));
    simplifier.add_rule(Box::new(ReciprocalTrigCompositionRule));
    simplifier.add_rule(Box::new(ReciprocalTrigNegativeRule));
}
