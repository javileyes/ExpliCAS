//! Support for `expand(...)` function-call rewrites.

use crate::expand_estimate::estimate_expand_terms;
use crate::expand_ops::expand;
use crate::poly_store::expand_expr_modp_materialized_hold;
use cas_ast::hold::strip_all_holds;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use std::cmp::Ordering;

/// Default limit for materializing expanded AST terms.
pub const DEFAULT_EXPAND_MAX_MATERIALIZE_TERMS: u64 = 200_000;
/// Default threshold for switching to mod-p expansion.
pub const DEFAULT_EXPAND_MODP_THRESHOLD: u64 = 1_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpandCallPolicy {
    pub max_materialize_terms: u64,
    pub modp_threshold: u64,
}

impl Default for ExpandCallPolicy {
    fn default() -> Self {
        Self {
            max_materialize_terms: DEFAULT_EXPAND_MAX_MATERIALIZE_TERMS,
            modp_threshold: DEFAULT_EXPAND_MODP_THRESHOLD,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpandCallRewrite {
    pub rewritten: ExprId,
    pub kind: ExpandCallRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandCallRewriteKind {
    ModpFastPath,
    Expand,
    ExpandAtom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandCallDecision {
    Rewrite(ExpandCallRewrite),
    LeaveUnevaluatedTooLarge { estimated_terms: u64, limit: u64 },
}

/// Plan implicit conservative expansion for non-`expand(...)` expressions.
///
/// Mirrors the conservative policy:
/// - expand structurally
/// - strip `__hold`
/// - accept only if complexity does not increase
/// - reject pure ordering no-op rewrites
pub fn try_plan_conservative_implicit_expand_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expanded_raw = expand(ctx, expr);
    let rewritten = strip_all_holds(ctx, expanded_raw);
    if rewritten == expr {
        return None;
    }

    let old_count = cas_ast::count_nodes(ctx, expr);
    let new_count = cas_ast::count_nodes(ctx, rewritten);
    if new_count > old_count {
        return None;
    }
    if compare_expr(ctx, rewritten, expr) == Ordering::Equal {
        return None;
    }

    Some(rewritten)
}

/// Decide rewrite for function call `expand(arg)`.
///
/// Returns `None` when `expr` is not an `expand(...)` call.
pub fn decide_expand_call_rewrite_with_policy(
    ctx: &mut Context,
    expr: ExprId,
    policy: ExpandCallPolicy,
) -> Option<ExpandCallDecision> {
    let arg = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Expand)) && args.len() == 1 =>
        {
            args[0]
        }
        _ => return None,
    };

    let estimated_terms = estimate_expand_terms(ctx, arg);
    if let Some(est) = estimated_terms {
        if est > policy.max_materialize_terms {
            return Some(ExpandCallDecision::LeaveUnevaluatedTooLarge {
                estimated_terms: est,
                limit: policy.max_materialize_terms,
            });
        }
    }

    if estimated_terms.unwrap_or(0) > policy.modp_threshold {
        if let Some(result) = expand_expr_modp_materialized_hold(ctx, arg) {
            let rewritten = strip_all_holds(ctx, result);
            return Some(ExpandCallDecision::Rewrite(ExpandCallRewrite {
                rewritten,
                kind: ExpandCallRewriteKind::ModpFastPath,
            }));
        }
    }

    let expanded = expand(ctx, arg);
    let rewritten = strip_all_holds(ctx, expanded);
    if rewritten != expr {
        Some(ExpandCallDecision::Rewrite(ExpandCallRewrite {
            rewritten,
            kind: ExpandCallRewriteKind::Expand,
        }))
    } else {
        Some(ExpandCallDecision::Rewrite(ExpandCallRewrite {
            rewritten: arg,
            kind: ExpandCallRewriteKind::ExpandAtom,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        decide_expand_call_rewrite_with_policy, try_plan_conservative_implicit_expand_expr,
        ExpandCallDecision, ExpandCallPolicy, ExpandCallRewriteKind,
        DEFAULT_EXPAND_MAX_MATERIALIZE_TERMS, DEFAULT_EXPAND_MODP_THRESHOLD,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn rewrites_expand_atom_to_inner() {
        let mut ctx = Context::new();
        let expr = parse("expand(x)", &mut ctx).expect("parse");
        let decision = decide_expand_call_rewrite_with_policy(
            &mut ctx,
            expr,
            ExpandCallPolicy {
                max_materialize_terms: u64::MAX,
                modp_threshold: u64::MAX,
            },
        )
        .expect("decision");
        match decision {
            ExpandCallDecision::Rewrite(rewrite) => {
                assert_eq!(rewrite.kind, ExpandCallRewriteKind::Expand);
                assert_eq!(cas_formatter::render_expr(&ctx, rewrite.rewritten), "x");
            }
            other => panic!("expected rewrite, got {other:?}"),
        }
    }

    #[test]
    fn reports_too_large_when_limit_exceeded() {
        let mut ctx = Context::new();
        let expr = parse("expand((x+1)^8)", &mut ctx).expect("parse");
        let decision = decide_expand_call_rewrite_with_policy(
            &mut ctx,
            expr,
            ExpandCallPolicy {
                max_materialize_terms: 1,
                modp_threshold: u64::MAX,
            },
        )
        .expect("decision");
        assert!(matches!(
            decision,
            ExpandCallDecision::LeaveUnevaluatedTooLarge { .. }
        ));
    }

    #[test]
    fn conservative_implicit_expand_accepts_non_worsening_rewrite() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse");
        let expr = cas_ast::hold::wrap_hold(&mut ctx, x);
        let rewritten = try_plan_conservative_implicit_expand_expr(&mut ctx, expr).expect("plan");
        assert_eq!(cas_formatter::render_expr(&ctx, rewritten), "x");
    }

    #[test]
    fn default_policy_constants_match_default_impl() {
        let default = ExpandCallPolicy::default();
        assert_eq!(
            default.max_materialize_terms,
            DEFAULT_EXPAND_MAX_MATERIALIZE_TERMS
        );
        assert_eq!(default.modp_threshold, DEFAULT_EXPAND_MODP_THRESHOLD);
    }

    #[test]
    fn explicit_default_and_conservative_policies_both_rewrite() {
        let mut ctx = Context::new();
        let expr = parse("expand((x+1)^8)", &mut ctx).expect("parse");
        let defaulted =
            decide_expand_call_rewrite_with_policy(&mut ctx, expr, ExpandCallPolicy::default())
                .expect("default");
        let conservative = decide_expand_call_rewrite_with_policy(
            &mut ctx,
            expr,
            ExpandCallPolicy {
                max_materialize_terms: u64::MAX,
                modp_threshold: u64::MAX,
            },
        )
        .expect("conservative");
        assert!(matches!(defaulted, ExpandCallDecision::Rewrite(_)));
        assert!(matches!(conservative, ExpandCallDecision::Rewrite(_)));
    }
}
