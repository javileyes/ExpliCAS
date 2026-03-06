use crate::{Simplifier, SolveCtx};
use cas_ast::Equation;
use cas_solver_core::solve_analysis::PreflightContext;

pub(crate) fn build_solve_preflight_state(
    simplifier: &Simplifier,
    eq: &Equation,
    var: &str,
    value_domain: crate::ValueDomain,
    parent_ctx: &SolveCtx,
) -> PreflightContext<SolveCtx> {
    cas_solver_core::solve_runtime_pipeline_preflight_context_bound_runtime::build_runtime_solve_preflight_state_with_default_domain_inference_and_derivation(
        simplifier,
        eq,
        var,
        value_domain,
        parent_ctx,
        |state| &state.context,
        crate::infer_implicit_domain,
        crate::derive_requires_from_equation,
    )
}
