//! Shared predicate-proof helpers bound to a host runtime simplifier contract.

use cas_ast::{Context, ExprId};

/// Factory contract for short-lived simplifiers used by proof/ground-eval flows.
pub trait RuntimeProofSimplifierFactory {
    fn runtime_proof_with_context(ctx: Context) -> Self;
    fn runtime_proof_set_collect_steps(&mut self, collect: bool);
    fn runtime_proof_simplify_with_options_expr(
        &mut self,
        expr: ExprId,
        opts: crate::simplify_options::SimplifyOptions,
    ) -> ExprId;
    fn runtime_proof_into_context(self) -> Context;
}

/// Evaluate one ground candidate using a proof-simplifier factory.
pub fn ground_eval_candidate_with_runtime_proof_simplifier<SState>(
    source_ctx: &Context,
    source_expr: ExprId,
    opts: &crate::simplify_options::SimplifyOptions,
) -> Option<(Context, ExprId)>
where
    SState: RuntimeProofSimplifierFactory,
{
    ground_eval_candidate_with_runtime_simplifier_contract(
        source_ctx,
        source_expr,
        opts,
        SState::runtime_proof_with_context,
        SState::runtime_proof_set_collect_steps,
        SState::runtime_proof_simplify_with_options_expr,
        SState::runtime_proof_into_context,
    )
}

/// Evaluate one ground candidate using a host-provided simplifier contract.
pub fn ground_eval_candidate_with_runtime_simplifier_contract<
    SState,
    FBuildSimplifier,
    FSetCollectSteps,
    FSimplifyExprWithOptions,
    FIntoContext,
>(
    source_ctx: &Context,
    source_expr: ExprId,
    opts: &crate::simplify_options::SimplifyOptions,
    build_simplifier: FBuildSimplifier,
    set_collect_steps: FSetCollectSteps,
    simplify_expr_with_options: FSimplifyExprWithOptions,
    into_context: FIntoContext,
) -> Option<(Context, ExprId)>
where
    FBuildSimplifier: FnOnce(Context) -> SState,
    FSetCollectSteps: FnMut(&mut SState, bool),
    FSimplifyExprWithOptions:
        FnMut(&mut SState, ExprId, crate::simplify_options::SimplifyOptions) -> ExprId,
    FIntoContext: FnOnce(SState) -> Context,
{
    crate::ground_eval_runtime::ground_eval_candidate_with_runtime_simplifier_with_state(
        source_ctx,
        source_expr,
        opts,
        build_simplifier,
        set_collect_steps,
        simplify_expr_with_options,
        into_context,
    )
}

/// Prove non-zero using a proof-simplifier factory.
pub fn prove_nonzero_with_runtime_proof_simplifier<SState>(
    ctx: &Context,
    expr: ExprId,
) -> crate::domain_proof::Proof
where
    SState: RuntimeProofSimplifierFactory,
{
    prove_nonzero_with_runtime_simplifier_contract(
        ctx,
        expr,
        SState::runtime_proof_with_context,
        SState::runtime_proof_set_collect_steps,
        SState::runtime_proof_simplify_with_options_expr,
        SState::runtime_proof_into_context,
    )
}

/// Prove non-zero using the host runtime simplifier contract.
pub fn prove_nonzero_with_runtime_simplifier_contract<
    SState,
    FBuildSimplifier,
    FSetCollectSteps,
    FSimplifyExprWithOptions,
    FIntoContext,
>(
    ctx: &Context,
    expr: ExprId,
    build_simplifier: FBuildSimplifier,
    set_collect_steps: FSetCollectSteps,
    simplify_expr_with_options: FSimplifyExprWithOptions,
    into_context: FIntoContext,
) -> crate::domain_proof::Proof
where
    FBuildSimplifier: Fn(Context) -> SState + Copy,
    FSetCollectSteps: Fn(&mut SState, bool) + Copy,
    FSimplifyExprWithOptions:
        Fn(&mut SState, ExprId, crate::simplify_options::SimplifyOptions) -> ExprId + Copy,
    FIntoContext: Fn(SState) -> Context + Copy,
{
    crate::predicate_proofs::prove_nonzero_with_default_depth_with_runtime_evaluator(
        ctx,
        expr,
        |source_ctx, source_expr, opts| {
            ground_eval_candidate_with_runtime_simplifier_contract(
                source_ctx,
                source_expr,
                opts,
                build_simplifier,
                set_collect_steps,
                simplify_expr_with_options,
                into_context,
            )
        },
    )
}

/// Prove positivity using a proof-simplifier factory.
pub fn prove_positive_with_runtime_proof_simplifier<SState>(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::value_domain::ValueDomain,
) -> crate::domain_proof::Proof
where
    SState: RuntimeProofSimplifierFactory,
{
    prove_positive_with_runtime_simplifier_contract(
        ctx,
        expr,
        value_domain,
        SState::runtime_proof_with_context,
        SState::runtime_proof_set_collect_steps,
        SState::runtime_proof_simplify_with_options_expr,
        SState::runtime_proof_into_context,
    )
}

/// Prove positivity using the host runtime simplifier contract.
pub fn prove_positive_with_runtime_simplifier_contract<
    SState,
    FBuildSimplifier,
    FSetCollectSteps,
    FSimplifyExprWithOptions,
    FIntoContext,
>(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::value_domain::ValueDomain,
    build_simplifier: FBuildSimplifier,
    set_collect_steps: FSetCollectSteps,
    simplify_expr_with_options: FSimplifyExprWithOptions,
    into_context: FIntoContext,
) -> crate::domain_proof::Proof
where
    FBuildSimplifier: Fn(Context) -> SState + Copy,
    FSetCollectSteps: Fn(&mut SState, bool) + Copy,
    FSimplifyExprWithOptions:
        Fn(&mut SState, ExprId, crate::simplify_options::SimplifyOptions) -> ExprId + Copy,
    FIntoContext: Fn(SState) -> Context + Copy,
{
    crate::predicate_proofs::prove_positive_with_default_depth_with_runtime_evaluator(
        ctx,
        expr,
        value_domain,
        |source_ctx, source_expr, opts| {
            ground_eval_candidate_with_runtime_simplifier_contract(
                source_ctx,
                source_expr,
                opts,
                build_simplifier,
                set_collect_steps,
                simplify_expr_with_options,
                into_context,
            )
        },
    )
}

/// Prove non-negativity using a proof-simplifier factory.
pub fn prove_nonnegative_with_runtime_proof_simplifier<SState>(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::value_domain::ValueDomain,
) -> crate::domain_proof::Proof
where
    SState: RuntimeProofSimplifierFactory,
{
    prove_nonnegative_with_runtime_simplifier_contract(
        ctx,
        expr,
        value_domain,
        SState::runtime_proof_with_context,
        SState::runtime_proof_set_collect_steps,
        SState::runtime_proof_simplify_with_options_expr,
        SState::runtime_proof_into_context,
    )
}

/// Prove non-negativity using the host runtime simplifier contract.
pub fn prove_nonnegative_with_runtime_simplifier_contract<
    SState,
    FBuildSimplifier,
    FSetCollectSteps,
    FSimplifyExprWithOptions,
    FIntoContext,
>(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::value_domain::ValueDomain,
    build_simplifier: FBuildSimplifier,
    set_collect_steps: FSetCollectSteps,
    simplify_expr_with_options: FSimplifyExprWithOptions,
    into_context: FIntoContext,
) -> crate::domain_proof::Proof
where
    FBuildSimplifier: Fn(Context) -> SState + Copy,
    FSetCollectSteps: Fn(&mut SState, bool) + Copy,
    FSimplifyExprWithOptions:
        Fn(&mut SState, ExprId, crate::simplify_options::SimplifyOptions) -> ExprId + Copy,
    FIntoContext: Fn(SState) -> Context + Copy,
{
    crate::predicate_proofs::prove_nonnegative_with_default_depth_with_runtime_evaluator(
        ctx,
        expr,
        value_domain,
        |source_ctx, source_expr, opts| {
            ground_eval_candidate_with_runtime_simplifier_contract(
                source_ctx,
                source_expr,
                opts,
                build_simplifier,
                set_collect_steps,
                simplify_expr_with_options,
                into_context,
            )
        },
    )
}
