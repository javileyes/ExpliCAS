pub mod check;
pub(crate) mod isolation;
pub(crate) mod solve_core;
mod types;
pub use cas_solver_core::isolation_utils::contains_var;
pub use cas_solver_core::solve_budget::SolveBudget;
pub use cas_solver_core::verify_stats;
pub(crate) use types::medium_step;
pub(crate) use types::SolveDomainEnv;
pub use types::{
    DisplaySolveSteps, SolveCtx, SolveDiagnostics, SolveStep, SolveSubStep, SolverOptions,
};

pub use self::solve_core::{solve, solve_with_display_steps};

/// Classify whether a logarithmic solve step (`base^x = rhs`) is valid.
pub(crate) fn classify_log_solve(
    ctx: &cas_ast::Context,
    base: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
    opts: &SolverOptions,
    env: &SolveDomainEnv,
) -> cas_solver_core::log_domain::LogSolveDecision {
    cas_solver_core::log_domain::classify_log_solve_with_external_prover(
        ctx,
        base,
        rhs,
        opts.value_domain == crate::semantics::ValueDomain::RealOnly,
        opts.core_domain_mode(),
        env.has_positive(base),
        env.has_positive(rhs),
        |core_ctx, expr| crate::helpers::prove_positive(core_ctx, expr, opts.value_domain),
        |proof| {
            cas_solver_core::external_proof::map_external_proof_status_with(
                proof,
                |value| {
                    matches!(
                        value,
                        crate::domain::Proof::Proven | crate::domain::Proof::ProvenImplicit
                    )
                },
                |value| matches!(value, crate::domain::Proof::Disproven),
            )
        },
    )
}

/// Map engine proof classification to solver-core non-zero status.
pub(crate) fn prove_nonzero_status(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_solver_core::linear_solution::NonZeroStatus {
    cas_solver_core::external_proof::map_external_nonzero_status_with(
        crate::helpers::prove_nonzero(ctx, expr),
        |proof| {
            matches!(
                proof,
                crate::domain::Proof::Proven | crate::domain::Proof::ProvenImplicit
            )
        },
        |proof| matches!(proof, crate::domain::Proof::Disproven),
    )
}

#[cfg(test)]
mod tests;
