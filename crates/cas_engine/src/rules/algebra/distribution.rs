use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::Expr;
use cas_math::distribution_simple_support::try_rewrite_simple_mul_distribution_expr;
use cas_math::expand_call_support::{
    decide_expand_call_rewrite_with_policy, try_plan_conservative_implicit_expand_expr,
    ExpandCallDecision, ExpandCallPolicy, ExpandCallRewriteKind,
};

fn format_expand_call_desc(kind: ExpandCallRewriteKind) -> &'static str {
    match kind {
        ExpandCallRewriteKind::ModpFastPath => "expand() [mod-p fast path]",
        ExpandCallRewriteKind::Expand => "expand()",
        ExpandCallRewriteKind::ExpandAtom => "expand(atom)",
    }
}

// ExpandRule: only runs in Transform phase
define_rule!(
    ExpandRule,
    "Expand Polynomial",
    None,
    PhaseMask::TRANSFORM,
    |ctx, expr| {
        if let Some(decision) =
            decide_expand_call_rewrite_with_policy(ctx, expr, ExpandCallPolicy::default())
        {
            match decision {
                ExpandCallDecision::Rewrite(rewrite) => {
                    return Some(
                        Rewrite::new(rewrite.rewritten).desc(format_expand_call_desc(rewrite.kind)),
                    );
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
        if let Some(ExpandCallDecision::Rewrite(rewrite)) = decide_expand_call_rewrite_with_policy(
            ctx,
            expr,
            ExpandCallPolicy {
                max_materialize_terms: u64::MAX,
                modp_threshold: u64::MAX,
            },
        ) {
            return Some(
                Rewrite::new(rewrite.rewritten).desc(format_expand_call_desc(rewrite.kind)),
            );
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
            let desc = match ctx.get(expr) {
                Expr::Mul(_, right)
                    if matches!(ctx.get(*right), Expr::Add(_, _) | Expr::Sub(_, _)) =>
                {
                    "Distribute (RHS)"
                }
                Expr::Mul(left, _)
                    if matches!(ctx.get(*left), Expr::Add(_, _) | Expr::Sub(_, _)) =>
                {
                    "Distribute (LHS)"
                }
                _ => "Distribute",
            };
            return Some(Rewrite::new(rewrite.rewritten).desc(desc));
        }
        None
    }
);
