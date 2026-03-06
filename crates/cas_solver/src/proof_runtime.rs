//! Local predicate-proof runtime for solver facade.

use cas_ast::{Context, ExprId};

use crate::{Proof, ValueDomain};

pub(crate) fn prove_nonzero(ctx: &Context, expr: ExprId) -> Proof {
    cas_solver_core::predicate_proofs::prove_nonzero_with_default_depth_with_runtime_evaluator(
        ctx,
        expr,
        crate::runtime_ground_eval::ground_eval_candidate,
    )
}

pub(crate) fn prove_positive(ctx: &Context, expr: ExprId, value_domain: ValueDomain) -> Proof {
    cas_solver_core::predicate_proofs::prove_positive_with_default_depth_with_runtime_evaluator(
        ctx,
        expr,
        value_domain,
        crate::runtime_ground_eval::ground_eval_candidate,
    )
}

pub(crate) fn prove_nonnegative(ctx: &Context, expr: ExprId, value_domain: ValueDomain) -> Proof {
    cas_solver_core::predicate_proofs::prove_nonnegative_with_default_depth_with_runtime_evaluator(
        ctx,
        expr,
        value_domain,
        crate::runtime_ground_eval::ground_eval_candidate,
    )
}
