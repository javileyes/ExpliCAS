//! Equation-level additive cancellation helpers shared across engine/solver.

use cas_ast::{Context, ExprId};
use cas_math::cancel_semantic_support::{
    try_cancel_additive_terms_semantic_with_state, SemanticCancelConfig,
};
use cas_math::cancel_support::{try_cancel_common_additive_terms_expr, CancelCommonAdditivePlan};

/// Result of equation-level additive term cancellation.
pub type CancelResult = CancelCommonAdditivePlan;

/// Cancel common additive terms between two expression trees.
pub fn cancel_common_additive_terms(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<CancelResult> {
    try_cancel_common_additive_terms_expr(ctx, lhs, rhs)
}

/// Shared semantic fallback for equation-level additive cancellation.
///
/// Uses the migration-wide guardrails (`MAX_TERMS=12`, `MAX_NODES=200`) and
/// delegates strategy details (candidate simplify, expansion, strict proof)
/// to callbacks supplied by the caller.
#[allow(clippy::too_many_arguments)]
pub fn cancel_additive_terms_semantic_with_state<
    S,
    FGetContext,
    FGetContextMut,
    FSimplifyCandidate,
    FProveEqual,
>(
    state: &mut S,
    lhs: ExprId,
    rhs: ExprId,
    get_context: FGetContext,
    get_context_mut: FGetContextMut,
    simplify_candidate: FSimplifyCandidate,
    fallback_expand: fn(&mut Context, ExprId) -> ExprId,
    prove_equal: FProveEqual,
) -> Option<CancelResult>
where
    FGetContext: FnMut(&S) -> &Context,
    FGetContextMut: FnMut(&mut S) -> &mut Context,
    FSimplifyCandidate: FnMut(&mut S, ExprId) -> ExprId,
    FProveEqual: FnMut(&mut S, ExprId, ExprId) -> bool,
{
    let result = try_cancel_additive_terms_semantic_with_state(
        state,
        lhs,
        rhs,
        SemanticCancelConfig {
            max_terms: 12,
            max_nodes: 200,
        },
        get_context,
        get_context_mut,
        simplify_candidate,
        fallback_expand,
        prove_equal,
    )?;

    Some(CancelResult {
        new_lhs: result.new_lhs,
        new_rhs: result.new_rhs,
        cancelled_count: result.cancelled_count,
    })
}

/// Runtime-oriented semantic cancellation helper with canonical option policy.
///
/// Applies the same 2-phase scheme across engine/solver wrappers:
/// - candidate generation with default generic semantics,
/// - strict proof per candidate pair via `(lt - rt) -> 0`.
pub fn cancel_additive_terms_semantic_runtime_with_state<
    S,
    FGetContext,
    FGetContextMut,
    FSimplifyWithOptions,
>(
    state: &mut S,
    lhs: ExprId,
    rhs: ExprId,
    get_context: FGetContext,
    get_context_mut: FGetContextMut,
    simplify_with_options: FSimplifyWithOptions,
    fallback_expand: fn(&mut Context, ExprId) -> ExprId,
) -> Option<CancelResult>
where
    FGetContext: Fn(&S) -> &Context,
    FGetContextMut: Fn(&mut S) -> &mut Context,
    FSimplifyWithOptions: Fn(&mut S, ExprId, crate::simplify_options::SimplifyOptions) -> ExprId,
{
    use cas_ast::Expr;
    use num_traits::Zero;

    let candidate_opts = crate::simplify_options::SimplifyOptions {
        collect_steps: false,
        ..Default::default()
    };
    let strict_proof_opts = crate::simplify_options::SimplifyOptions {
        shared: crate::simplify_options::SharedSemanticConfig {
            semantics: crate::eval_config::EvalConfig::strict(),
            ..Default::default()
        },
        collect_steps: false,
        ..Default::default()
    };

    cancel_additive_terms_semantic_with_state(
        state,
        lhs,
        rhs,
        |s| get_context(s),
        |s| get_context_mut(s),
        |s, term| simplify_with_options(s, term, candidate_opts.clone()),
        fallback_expand,
        |s, lt, rt| {
            let diff = get_context_mut(s).add(Expr::Sub(lt, rt));
            let simplified_diff = simplify_with_options(s, diff, strict_proof_opts.clone());
            matches!(get_context(s).get(simplified_diff), Expr::Number(n) if n.is_zero())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{
        cancel_additive_terms_semantic_runtime_with_state,
        cancel_additive_terms_semantic_with_state,
    };
    use cas_ast::ordering::compare_expr;
    use cas_ast::Expr;
    use num_traits::Zero;
    use std::cmp::Ordering;

    fn no_expand(_ctx: &mut cas_ast::Context, expr: cas_ast::ExprId) -> cas_ast::ExprId {
        expr
    }

    #[test]
    fn semantic_cancel_wrapper_cancels_structural_overlap() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Add(x, one));
        let rhs = x;

        let out = cancel_additive_terms_semantic_with_state(
            &mut ctx,
            lhs,
            rhs,
            |s| s,
            |s| s,
            |_s, term| term,
            no_expand,
            |s, lt, rt| compare_expr(s, lt, rt) == Ordering::Equal,
        )
        .expect("expected cancellable overlap");

        assert!(out.cancelled_count >= 1);
        assert!(
            matches!(ctx.get(out.new_rhs), Expr::Number(n) if n.is_zero()),
            "rhs should reduce to zero after cancellation"
        );
    }

    #[test]
    fn semantic_cancel_wrapper_returns_none_without_overlap() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Add(x, one));
        let rhs = y;

        let out = cancel_additive_terms_semantic_with_state(
            &mut ctx,
            lhs,
            rhs,
            |s| s,
            |s| s,
            |_s, term| term,
            no_expand,
            |_s, _lt, _rt| false,
        );

        assert!(out.is_none());
    }

    #[test]
    fn semantic_cancel_runtime_helper_cancels_overlap() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let lhs = ctx.add(cas_ast::Expr::Add(x, one));
        let rhs = x;

        let out = cancel_additive_terms_semantic_runtime_with_state(
            &mut ctx,
            lhs,
            rhs,
            |s| s,
            |s| s,
            |state, term, _opts| match state.get(term) {
                cas_ast::Expr::Sub(a, b) if compare_expr(state, *a, *b) == Ordering::Equal => {
                    state.num(0)
                }
                _ => term,
            },
            no_expand,
        )
        .expect("expected cancellable overlap");

        assert!(out.cancelled_count >= 1);
        assert!(
            matches!(ctx.get(out.new_rhs), cas_ast::Expr::Number(n) if n.is_zero()),
            "rhs should reduce to zero after cancellation"
        );
    }
}
