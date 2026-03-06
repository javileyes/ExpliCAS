//! Shared strategy-apply wrapper bound to the concrete runtime solve context.

use cas_ast::{Equation, ExprId, SolutionSet};

/// Apply one strategy using the shared runtime solve context and core solver
/// options, while runtime crates provide the state kernels and recursive
/// solve/isolation callbacks.
#[allow(clippy::too_many_arguments)]
pub fn apply_strategy_with_runtime_ctx_and_options_and_state<
    T,
    FCollectSteps,
    FContextRef,
    FContextMut,
    FSetCollect,
    FSimplifyExpr,
    FExpandExpr,
    FRenderExpr,
    FRenderExprFromCtx,
    FSolveEquation,
    FIsolateEquation,
    FProvePositive,
>(
    state: &mut T,
    kind: crate::strategy_order::SolveStrategyKind,
    equation: &Equation,
    var: &str,
    opts: crate::solver_options::SolverOptions,
    ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    collect_steps: FCollectSteps,
    context_ref: FContextRef,
    context_mut: FContextMut,
    set_collecting: FSetCollect,
    simplify_expr: FSimplifyExpr,
    expand_expr: FExpandExpr,
    render_expr: FRenderExpr,
    render_expr_from_ctx: FRenderExprFromCtx,
    solve_equation: FSolveEquation,
    isolate_equation: FIsolateEquation,
    mut prove_positive: FProvePositive,
) -> Option<
    Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
        ),
        crate::error_model::CasError,
    >,
>
where
    FCollectSteps: FnMut(&mut T) -> bool,
    FContextRef: Fn(&mut T) -> &cas_ast::Context + Clone,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FSetCollect: FnMut(&mut T, bool),
    FSimplifyExpr: FnMut(&mut T, ExprId) -> ExprId,
    FExpandExpr: FnMut(&mut T, ExprId) -> ExprId,
    FRenderExpr: FnMut(&mut T, ExprId) -> String,
    FRenderExprFromCtx: Fn(&cas_ast::Context, ExprId) -> String,
    FSolveEquation: FnMut(
        &mut T,
        &Equation,
        &str,
    ) -> Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
        ),
        crate::error_model::CasError,
    >,
    FIsolateEquation: FnMut(
        &mut T,
        &Equation,
        &str,
    ) -> Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
        ),
        crate::error_model::CasError,
    >,
    FProvePositive: FnMut(
        &cas_ast::Context,
        ExprId,
        crate::value_domain::ValueDomain,
    ) -> crate::domain_proof::Proof,
{
    let value_domain = opts.value_domain;
    let mode = opts.core_domain_mode();
    let context_ref_for_assumptions = context_ref.clone();

    crate::solve_runtime_pipeline_strategy_apply_runtime::apply_strategy_with_default_mappers_and_state(
        state,
        kind,
        equation,
        var,
        value_domain == crate::value_domain::ValueDomain::RealOnly,
        mode,
        opts.wildcard_scope(),
        collect_steps,
        context_ref,
        context_mut,
        set_collecting,
        simplify_expr,
        expand_expr,
        render_expr,
        render_expr_from_ctx,
        solve_equation,
        isolate_equation,
        |core_ctx, base, other_side| {
            crate::solve_runtime_flow::classify_log_solve_with_domain_env_and_runtime_positive_prover(
                core_ctx,
                base,
                other_side,
                value_domain,
                mode,
                &ctx.domain_env,
                &mut prove_positive,
            )
        },
        |state, record| {
            crate::solve_runtime_flow::note_log_assumption_with_runtime_sink(
                context_ref_for_assumptions(state),
                record.base,
                record.other_side,
                record.assumption,
                |event| ctx.note_assumption(event),
            );
        },
        |_state| {
            ctx.emit_scope(cas_formatter::display_transforms::ScopeTag::Rule(
                "QuadraticFormula",
            ));
        },
    )
}
