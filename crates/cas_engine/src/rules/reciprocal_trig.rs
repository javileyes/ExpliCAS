use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_reciprocal_eval_support::{
    try_rewrite_eval_reciprocal_trig_expr, try_rewrite_negative_reciprocal_trig_expr,
    try_rewrite_reciprocal_trig_composition_expr, ReciprocalTrigCompositionKind,
    ReciprocalTrigEvalKind, ReciprocalTrigNegativeKind,
};

fn format_reciprocal_trig_eval_desc(kind: ReciprocalTrigEvalKind) -> &'static str {
    match kind {
        ReciprocalTrigEvalKind::CotPiOver4 => "cot(π/4) = 1",
        ReciprocalTrigEvalKind::CotPiOver2 => "cot(π/2) = 0",
        ReciprocalTrigEvalKind::SecZero => "sec(0) = 1",
        ReciprocalTrigEvalKind::CscPiOver2 => "csc(π/2) = 1",
        ReciprocalTrigEvalKind::ArccotOne => "arccot(1) = π/4",
        ReciprocalTrigEvalKind::ArccotZero => "arccot(0) = π/2",
        ReciprocalTrigEvalKind::ArcsecOne => "arcsec(1) = 0",
        ReciprocalTrigEvalKind::ArccscOne => "arccsc(1) = π/2",
    }
}

fn format_reciprocal_trig_negative_desc(kind: ReciprocalTrigNegativeKind) -> &'static str {
    match kind {
        ReciprocalTrigNegativeKind::Cot => "cot(-x) = -cot(x)",
        ReciprocalTrigNegativeKind::Sec => "sec(-x) = sec(x)",
        ReciprocalTrigNegativeKind::Csc => "csc(-x) = -csc(x)",
        ReciprocalTrigNegativeKind::Arccot => "arccot(-x) = -arccot(x)",
        ReciprocalTrigNegativeKind::Arcsec => "arcsec(-x) = π - arcsec(x)",
        ReciprocalTrigNegativeKind::Arccsc => "arccsc(-x) = -arccsc(x)",
    }
}

fn format_reciprocal_trig_composition_desc(kind: ReciprocalTrigCompositionKind) -> &'static str {
    match kind {
        ReciprocalTrigCompositionKind::CotArccot => "cot(arccot(x)) = x",
        ReciprocalTrigCompositionKind::SecArcsec => "sec(arcsec(x)) = x",
        ReciprocalTrigCompositionKind::CscArccsc => "csc(arccsc(x)) = x",
        ReciprocalTrigCompositionKind::ArccotCot => "arccot(cot(x)) = x",
        ReciprocalTrigCompositionKind::ArcsecSec => "arcsec(sec(x)) = x",
        ReciprocalTrigCompositionKind::ArccscCsc => "arccsc(csc(x)) = x",
    }
}

define_rule!(
    EvaluateReciprocalTrigRule,
    "Evaluate Reciprocal Trig Functions",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_eval_reciprocal_trig_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_reciprocal_trig_eval_desc(rewrite.kind)))
    }
);

define_rule!(
    ReciprocalTrigCompositionRule,
    "Reciprocal Trig Composition",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_reciprocal_trig_composition_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(format_reciprocal_trig_composition_desc(rewrite.kind)),
        )
    }
);

define_rule!(
    ReciprocalTrigNegativeRule,
    "Reciprocal Trig Negative Argument",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_negative_reciprocal_trig_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(format_reciprocal_trig_negative_desc(rewrite.kind)),
        )
    }
);

// ==================== Registration ====================

pub fn register(simplifier: &mut crate::engine::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateReciprocalTrigRule));
    simplifier.add_rule(Box::new(ReciprocalTrigCompositionRule));
    simplifier.add_rule(Box::new(ReciprocalTrigNegativeRule));
}
