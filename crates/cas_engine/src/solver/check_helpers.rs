use cas_ast::{Context, ExprId};
use cas_math::ground_eval_guard::GroundEvalGuard;
use cas_solver_core::domain_mode::DomainMode;

use crate::engine::Simplifier;

pub(crate) fn simplify_options_for_domain(domain_mode: DomainMode) -> crate::SimplifyOptions {
    crate::SimplifyOptions {
        shared: crate::SharedSemanticConfig {
            semantics: crate::EvalConfig {
                domain_mode,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

pub(crate) fn fold_numeric_islands(ctx: &mut Context, root: ExprId) -> ExprId {
    let fold_opts = crate::SimplifyOptions {
        collect_steps: false,
        expand_mode: false,
        shared: crate::SharedSemanticConfig {
            semantics: crate::EvalConfig {
                domain_mode: crate::DomainMode::Generic,
                value_domain: crate::ValueDomain::RealOnly,
                ..Default::default()
            },
            ..Default::default()
        },
        budgets: crate::PhaseBudgets {
            core_iters: 4,
            transform_iters: 2,
            rationalize_iters: 0,
            post_iters: 2,
            max_total_rewrites: 50,
        },
        ..Default::default()
    };

    cas_solver_core::verification_numeric_islands::fold_numeric_islands_guarded_with_default_limits_and_candidate_evaluator(
        ctx,
        root,
        GroundEvalGuard::enter,
        |src_ctx, id| {
            let mut tmp = Simplifier::with_context(src_ctx.clone());
            tmp.set_collect_steps(false);
            let (result, _, _) = tmp.simplify_with_stats(id, fold_opts.clone());
            Some((tmp.context, result))
        },
    )
}
