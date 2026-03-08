//! Log-domain strategy runtime wrappers extracted from `solve_runtime_flow_strategy`.

use cas_ast::ExprId;

/// Classify a logarithmic solve decision using:
/// - domain environment (`base > 0`, `rhs > 0`) facts,
/// - runtime value-domain configuration,
/// - runtime positive prover callback returning [`crate::domain_proof::Proof`].
pub fn classify_log_solve_with_domain_env_and_runtime_positive_prover<R, FProvePositive>(
    ctx: &cas_ast::Context,
    base: ExprId,
    rhs: ExprId,
    value_domain: crate::value_domain::ValueDomain,
    mode: crate::log_domain::DomainModeKind,
    env: &crate::domain_env::SolveDomainEnv<R>,
    mut prove_positive: FProvePositive,
) -> crate::log_domain::LogSolveDecision
where
    R: crate::domain_env::RequiredDomainSet,
    FProvePositive: FnMut(
        &cas_ast::Context,
        ExprId,
        crate::value_domain::ValueDomain,
    ) -> crate::domain_proof::Proof,
{
    crate::log_domain::classify_log_solve_with_env_and_tri_prover(
        ctx,
        base,
        rhs,
        value_domain == crate::value_domain::ValueDomain::RealOnly,
        mode,
        env,
        |core_ctx, expr| {
            crate::predicate_proofs::prove_positive_core_with(
                core_ctx,
                expr,
                value_domain,
                |proof_ctx, proof_expr, vd| prove_positive(proof_ctx, proof_expr, vd),
            )
        },
    )
}

/// Map one logarithmic assumption into a runtime assumption-event sink.
pub fn note_log_assumption_with_runtime_sink<FSink>(
    ctx: &cas_ast::Context,
    base: ExprId,
    rhs: ExprId,
    assumption: crate::log_domain::LogAssumption,
    mut sink: FSink,
) where
    FSink: FnMut(crate::assumption_model::AssumptionEvent),
{
    sink(crate::assumption_model::assumption_event_from_log_assumption(ctx, assumption, base, rhs));
}

/// Map one blocked logarithmic hint into a runtime blocked-hint sink.
pub fn note_log_blocked_hint_with_runtime_sink<FSink>(
    ctx: &cas_ast::Context,
    hint: crate::solve_outcome::LogBlockedHintRecord,
    mut sink: FSink,
) where
    FSink: FnMut(crate::blocked_hint::BlockedHint),
{
    sink(crate::assumption_model::map_log_blocked_hint(ctx, hint));
}
