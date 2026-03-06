//! Shared isolation-dispatch wrapper bound to the concrete runtime solve
//! context and solver options.

use cas_ast::symbol::SymbolId;
use cas_ast::{Equation, ExprId, RelOp, SolutionSet};

/// Dispatch isolation using the shared runtime solve context and core solver
/// options while runtime crates provide only the state kernels and recursive
/// solve/isolation callbacks.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_isolation_with_runtime_ctx_and_options_and_state<
    T,
    FContextRef,
    FContextMut,
    FSimplifyExpr,
    FCollectSteps,
    FProveNonzeroStatus,
    FRenderExpr,
    FSolveSplitCaseWithVar,
    FIsKnownNegative,
    FIsolateEquationWithVar,
    FClearBlockedHints,
    FSimplifyWithTactic,
    FProvePositive,
    FRegisterBlockedHint,
    FSimplifyWithTrace,
    FSymName,
>(
    state: &mut T,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    opts: crate::solver_options::SolverOptions,
    ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    context_ref: FContextRef,
    context_mut: FContextMut,
    simplify_expr: FSimplifyExpr,
    collect_steps: FCollectSteps,
    prove_nonzero_status: FProveNonzeroStatus,
    render_expr: FRenderExpr,
    solve_split_case_with_var: FSolveSplitCaseWithVar,
    is_known_negative: FIsKnownNegative,
    isolate_equation_with_var: FIsolateEquationWithVar,
    clear_blocked_hints: FClearBlockedHints,
    simplify_with_tactic: FSimplifyWithTactic,
    mut prove_positive: FProvePositive,
    register_blocked_hint: FRegisterBlockedHint,
    simplify_with_trace: FSimplifyWithTrace,
    sym_name: FSymName,
) -> Result<
    (
        SolutionSet,
        Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
    ),
    crate::error_model::CasError,
>
where
    FContextRef: Fn(&mut T) -> &cas_ast::Context + Clone,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FSimplifyExpr: FnMut(&mut T, ExprId) -> ExprId,
    FCollectSteps: FnMut(&mut T) -> bool,
    FProveNonzeroStatus: FnMut(&mut T, ExprId) -> crate::linear_solution::NonZeroStatus,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FSolveSplitCaseWithVar: FnMut(
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
    FIsKnownNegative: FnMut(&mut T, ExprId) -> bool,
    FIsolateEquationWithVar: FnMut(
        &mut T,
        Equation,
        &str,
    ) -> Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
        ),
        crate::error_model::CasError,
    >,
    FClearBlockedHints: FnMut(&mut T),
    FSimplifyWithTactic: FnMut(&mut T, ExprId, &crate::simplify_options::SimplifyOptions) -> ExprId,
    FProvePositive: FnMut(
        &cas_ast::Context,
        ExprId,
        crate::value_domain::ValueDomain,
    ) -> crate::domain_proof::Proof,
    FRegisterBlockedHint: FnMut(crate::blocked_hint::BlockedHint),
    FSimplifyWithTrace: FnMut(&mut T, ExprId) -> (ExprId, Vec<(String, ExprId)>),
    FSymName: FnMut(&mut T, SymbolId) -> String,
{
    let value_domain = opts.value_domain;
    let mode = opts.core_domain_mode();
    let tactic_options =
        crate::simplify_options::SimplifyOptions::for_solve_tactic(opts.domain_mode);
    let context_ref_for_classify = context_ref.clone();
    let collect_steps = std::cell::RefCell::new(collect_steps);
    let simplify_expr = std::cell::RefCell::new(simplify_expr);
    let isolate_equation_with_var = std::cell::RefCell::new(isolate_equation_with_var);
    let clear_blocked_hints = std::cell::RefCell::new(clear_blocked_hints);
    let simplify_with_tactic = std::cell::RefCell::new(simplify_with_tactic);
    let register_blocked_hint = std::cell::RefCell::new(register_blocked_hint);
    let simplify_with_trace = std::cell::RefCell::new(simplify_with_trace);
    let sym_name = std::cell::RefCell::new(sym_name);

    crate::solve_runtime_isolation_dispatch_runtime::dispatch_isolation_with_default_routes_and_mappers_with_state(
        state,
        lhs,
        rhs,
        op,
        var,
        context_ref,
        context_mut,
        |state, expr| (simplify_expr.borrow_mut())(state, expr),
        |state| (collect_steps.borrow_mut())(state),
        |state, expr| (simplify_expr.borrow_mut())(state, expr),
        prove_nonzero_status,
        render_expr,
        |state| (collect_steps.borrow_mut())(state),
        solve_split_case_with_var,
        is_known_negative,
        |state, equation, solve_var| {
            (isolate_equation_with_var.borrow_mut())(state, equation, solve_var)
        },
        mode,
        opts.wildcard_scope(),
        value_domain == crate::value_domain::ValueDomain::RealOnly,
        opts.budget,
        |state| (collect_steps.borrow_mut())(state),
        tactic_options,
        |state, expr| (simplify_expr.borrow_mut())(state, expr),
        |state| (clear_blocked_hints.borrow_mut())(state),
        |state, expr, solve_tactic_opts| {
            (simplify_with_tactic.borrow_mut())(state, expr, solve_tactic_opts)
        },
        |state, tactic_base, tactic_rhs| {
            crate::solve_runtime_flow::classify_log_solve_with_domain_env_and_runtime_positive_prover(
                context_ref_for_classify(state),
                tactic_base,
                tactic_rhs,
                value_domain,
                mode,
                &ctx.domain_env,
                &mut prove_positive,
            )
        },
        crate::solve_runtime_mapping::map_isolation_error,
        |core_ctx, base, eq_rhs, assumption| {
            crate::solve_runtime_flow::note_log_assumption_with_runtime_sink(
                core_ctx,
                base,
                eq_rhs,
                assumption,
                |event| ctx.note_assumption(event),
            );
        },
        |core_ctx, hint| {
            crate::solve_runtime_flow::note_log_blocked_hint_with_runtime_sink(
                core_ctx,
                hint,
                |blocked_hint| (register_blocked_hint.borrow_mut())(blocked_hint),
            );
        },
        crate::solve_runtime_mapping::map_unsupported_in_real_domain_error,
        |state| (collect_steps.borrow_mut())(state),
        |state, rhs_expr| (simplify_with_trace.borrow_mut())(state, rhs_expr),
        |state, fn_symbol| (sym_name.borrow_mut())(state, fn_symbol),
        |state, equation, solve_var| {
            (isolate_equation_with_var.borrow_mut())(state, equation, solve_var)
        },
        |state| (collect_steps.borrow_mut())(state),
    )
}
