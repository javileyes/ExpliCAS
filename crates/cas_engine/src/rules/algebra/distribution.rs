use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::distribution_simple_support::try_rewrite_simple_mul_distribution_expr;
use cas_math::expand_call_support::{
    decide_expand_call_rewrite_conservative, decide_expand_call_rewrite_default,
    try_plan_conservative_implicit_expand_expr, ExpandCallDecision,
};

// ExpandRule: only runs in Transform phase
define_rule!(
    ExpandRule,
    "Expand Polynomial",
    None,
    PhaseMask::TRANSFORM,
    |ctx, expr| {
        if let Some(decision) = decide_expand_call_rewrite_default(ctx, expr) {
            match decision {
                ExpandCallDecision::Rewrite(rewrite) => {
                    return Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc));
                }
                ExpandCallDecision::LeaveUnevaluatedTooLarge {
                    estimated_terms,
                    limit,
                } => {
                    tracing::warn!(
                        estimated_terms = estimated_terms,
                        limit = limit,
                        "expand() aborted: estimated {} terms exceeds limit {}. \
                         Use poly_mul_modp() for large polynomial operations.",
                        estimated_terms,
                        limit
                    );
                    // Return None → leaves expand(...) unevaluated
                    return None;
                }
            }
        }
        None
    }
);

// ConservativeExpandRule: only runs in Transform phase
define_rule!(
    ConservativeExpandRule,
    "Conservative Expand",
    None,
    PhaseMask::TRANSFORM,
    |ctx, expr| {
        if let Some(ExpandCallDecision::Rewrite(rewrite)) =
            decide_expand_call_rewrite_conservative(ctx, expr)
        {
            return Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc));
        }

        // Implicit expansion (e.g. (x+1)^2), only if not complexity-worsening.
        if let Some(new_expr) = try_plan_conservative_implicit_expand_expr(ctx, expr) {
            return Some(Rewrite::new(new_expr).desc("Conservative Expansion"));
        }
        None
    }
);

// DistributeRule: only runs in Transform phase
define_rule!(
    DistributeRule,
    "Distributive Property (Simple)",
    None,
    PhaseMask::TRANSFORM,
    |ctx, expr| {
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }
        if let Some(rewrite) = try_rewrite_simple_mul_distribution_expr(ctx, expr) {
            return Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc));
        }
        None
    }
);
