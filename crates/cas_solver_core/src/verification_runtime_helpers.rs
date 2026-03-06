//! Shared runtime helpers for verification wrappers.
//!
//! This module centralizes small glue helpers used by runtime crates
//! (`cas_engine`, `cas_solver`) so they keep less duplicated orchestration code.

use crate::domain_mode::DomainMode;
use crate::eval_config::EvalConfig;
use cas_ast::{Context, ExprId};

/// Conservative phase-budget caps used by verification/ground-eval folds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConservativePhaseBudgetCaps {
    pub core_iters: usize,
    pub transform_iters: usize,
    pub rationalize_iters: usize,
    pub post_iters: usize,
    pub max_total_rewrites: usize,
}

/// Build the verification eval config for a given domain mode.
#[inline]
pub fn eval_config_for_domain(domain_mode: DomainMode) -> EvalConfig {
    crate::conservative_eval_config::eval_config_for_domain(domain_mode)
}

/// Build eval config used by conservative numeric folding passes.
#[inline]
pub fn conservative_numeric_fold_eval_config() -> EvalConfig {
    crate::conservative_eval_config::conservative_numeric_fold_eval_config()
}

/// Conservative phase-budget caps shared by runtime wrappers.
#[inline]
pub fn conservative_phase_budget_caps() -> ConservativePhaseBudgetCaps {
    ConservativePhaseBudgetCaps {
        core_iters: crate::phase_budget_defaults::CONSERVATIVE_CORE_ITERS,
        transform_iters: crate::phase_budget_defaults::CONSERVATIVE_TRANSFORM_ITERS,
        rationalize_iters: crate::phase_budget_defaults::CONSERVATIVE_RATIONALIZE_ITERS,
        post_iters: crate::phase_budget_defaults::CONSERVATIVE_POST_ITERS,
        max_total_rewrites: crate::phase_budget_defaults::CONSERVATIVE_MAX_TOTAL_REWRITES,
    }
}

/// Fold numeric islands under default limits with a caller-provided guard and
/// candidate evaluator.
pub fn fold_numeric_islands_with_guard_and_default_limits<Guard, FEnterGuard, FEvaluateCandidate>(
    ctx: &mut Context,
    root: ExprId,
    enter_guard: FEnterGuard,
    evaluate_candidate: FEvaluateCandidate,
) -> ExprId
where
    FEnterGuard: FnMut() -> Option<Guard>,
    FEvaluateCandidate: FnMut(&Context, ExprId) -> Option<(Context, ExprId)>,
{
    crate::verification_numeric_islands::fold_numeric_islands_guarded_with_default_limits_and_candidate_evaluator(
        ctx,
        root,
        enter_guard,
        evaluate_candidate,
    )
}

/// Fold numeric islands using the standard ground-eval guard and default limits.
///
/// Runtime crates can provide only a candidate evaluator and avoid duplicating
/// the guard wiring.
pub fn fold_numeric_islands_with_default_guard_and_candidate<FEvaluateCandidate>(
    ctx: &mut Context,
    root: ExprId,
    evaluate_candidate: FEvaluateCandidate,
) -> ExprId
where
    FEvaluateCandidate: FnMut(&Context, ExprId) -> Option<(Context, ExprId)>,
{
    fold_numeric_islands_with_guard_and_default_limits(
        ctx,
        root,
        cas_math::ground_eval_guard::GroundEvalGuard::enter,
        evaluate_candidate,
    )
}

/// Fold numeric islands using default guard/limits and the conservative
/// simplify-option preset.
///
/// The evaluator receives `(source_context, expr_id, conservative_options)`.
pub fn fold_numeric_islands_with_default_guard_and_conservative_options<FEvaluateCandidate>(
    ctx: &mut Context,
    root: ExprId,
    mut evaluate_candidate: FEvaluateCandidate,
) -> ExprId
where
    FEvaluateCandidate: FnMut(
        &Context,
        ExprId,
        &crate::simplify_options::SimplifyOptions,
    ) -> Option<(Context, ExprId)>,
{
    let fold_opts = crate::conservative_eval_config::conservative_numeric_fold_options();
    fold_numeric_islands_with_default_guard_and_candidate(ctx, root, |src_ctx, id| {
        evaluate_candidate(src_ctx, id, &fold_opts)
    })
}

#[cfg(test)]
mod tests {
    use super::{
        conservative_phase_budget_caps, fold_numeric_islands_with_guard_and_default_limits,
    };

    struct DummyGuard;

    #[test]
    fn fold_wrapper_respects_guard_block() {
        let mut ctx = cas_ast::Context::new();
        let root = ctx.num(3);
        let out = fold_numeric_islands_with_guard_and_default_limits::<DummyGuard, _, _>(
            &mut ctx,
            root,
            || None,
            |_src_ctx, _id| None,
        );
        assert_eq!(out, root);
    }

    #[test]
    fn conservative_caps_match_phase_defaults() {
        let caps = conservative_phase_budget_caps();
        assert_eq!(
            caps.core_iters,
            crate::phase_budget_defaults::CONSERVATIVE_CORE_ITERS
        );
        assert_eq!(
            caps.transform_iters,
            crate::phase_budget_defaults::CONSERVATIVE_TRANSFORM_ITERS
        );
        assert_eq!(
            caps.rationalize_iters,
            crate::phase_budget_defaults::CONSERVATIVE_RATIONALIZE_ITERS
        );
        assert_eq!(
            caps.post_iters,
            crate::phase_budget_defaults::CONSERVATIVE_POST_ITERS
        );
        assert_eq!(
            caps.max_total_rewrites,
            crate::phase_budget_defaults::CONSERVATIVE_MAX_TOTAL_REWRITES
        );
    }
}
