use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_with_ctx_and_options;
use crate::solver::{
    context_render_expr, medium_step, simplifier_collect_steps, simplifier_context,
    simplifier_context_mut, simplifier_is_known_negative, simplifier_prove_nonzero_status,
    simplifier_simplify_expr, SolveCtx, SolveStep, SolverOptions,
};
use cas_ast::symbol::SymbolId;
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_arithmetic::{
    execute_add_isolation_pipeline_with_default_factored_linear_collect_and_unified_step_mapper_with_state,
    execute_div_isolation_pipeline_with_default_reciprocal_fallback_and_unified_step_mapper_with_state,
    execute_mul_isolation_pipeline_with_default_additive_linear_collect_and_unified_step_mapper_with_state,
    execute_sub_isolation_pipeline_with_default_plan_and_unified_step_mapper_with_state,
};
use cas_solver_core::isolation_dispatch::execute_isolation_dispatch_with_default_isolated_and_negated_entries_and_default_linear_collect_kernels_for_var_and_unified_step_mapper_with_state;
use cas_solver_core::isolation_functions::execute_function_isolation_with_default_kernels_and_unified_step_mapper_for_var_with_state;
use cas_solver_core::isolation_power::execute_pow_isolation_with_kernel_config_and_unified_step_mapper_for_var_with_state;
use cas_solver_core::solve_analysis::ensure_recursion_depth_within_limit_or_error;
use cas_solver_core::solve_budget::MAX_SOLVE_RECURSION_DEPTH;

type PowIsolationConfig =
    cas_solver_core::strategy_options::PowIsolationRuntimeConfig<crate::SimplifyOptions>;

#[allow(clippy::too_many_arguments)]
pub(crate) fn isolate(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    ensure_recursion_depth_within_limit_or_error(ctx.depth(), MAX_SOLVE_RECURSION_DEPTH, || {
        CasError::SolverError("Maximum solver recursion depth exceeded in isolation.".to_string())
    })?;

    dispatch_isolation(lhs, rhs, op, var, simplifier, opts, ctx)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_isolation(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    execute_isolation_dispatch_with_default_isolated_and_negated_entries_and_default_linear_collect_kernels_for_var_and_unified_step_mapper_with_state(
        simplifier,
        lhs,
        rhs,
        op.clone(),
        var,
        simplifier_context,
        simplifier_context_mut,
        simplifier_simplify_expr,
        simplifier_collect_steps,
        simplifier_simplify_expr,
        simplifier_prove_nonzero_status,
        context_render_expr,
        |simplifier, left, right| {
            isolate_add(
                lhs,
                left,
                right,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, left, right| {
            isolate_sub(
                left,
                right,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, left, right| {
            isolate_mul(
                lhs,
                left,
                right,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, left, right| {
            isolate_div(
                lhs,
                left,
                right,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, base, exponent| {
            isolate_pow(
                lhs,
                base,
                exponent,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, fn_id, args| {
            isolate_function(
                fn_id,
                args,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        simplifier_collect_steps,
        |simplifier, equation, solve_var| {
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op,
                solve_var,
                simplifier,
                opts,
                ctx,
            )
        },
        medium_step,
        |_simplifier, lhs_expr| {
            CasError::IsolationError(
                var.to_string(),
                format!("Cannot isolate from {:?}", lhs_expr),
            )
        },
    )
}

/// Handle isolation for `Function(fn_id, args)`: abs, log, ln, exp, sqrt, trig
#[allow(clippy::too_many_arguments)]
fn isolate_function(
    fn_id: SymbolId,
    args: Vec<ExprId>,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_items = simplifier.collect_steps();
    execute_function_isolation_with_default_kernels_and_unified_step_mapper_for_var_with_state(
        simplifier,
        fn_id,
        &args,
        rhs,
        op,
        var,
        include_items,
        steps,
        simplifier_context,
        simplifier_context_mut,
        context_render_expr,
        |simplifier, lhs_expr, rhs_expr, inner_op| {
            isolate(lhs_expr, rhs_expr, inner_op, var, simplifier, opts, ctx)
        },
        |simplifier, rhs_expr| {
            let (simplified_rhs, sim_steps) = simplifier.simplify(rhs_expr);
            let entries = sim_steps
                .into_iter()
                .map(|step| (step.description, step.after))
                .collect::<Vec<_>>();
            (simplified_rhs, entries)
        },
        medium_step,
        |_simplifier, missing_var| CasError::VariableNotFound(missing_var.to_string()),
        |simplifier, unsupported_fn_id, arity, unsupported_var| {
            CasError::IsolationError(
                unsupported_var.to_string(),
                format!(
                    "Cannot invert function '{}' with {} arguments",
                    simplifier.context.sym_name(unsupported_fn_id),
                    arity
                ),
            )
        },
        |_simplifier, fn_name| CasError::UnknownFunction(fn_name.to_string()),
    )
}

/// Handle isolation for `Add(l, r)`: `(A + B) = RHS`
#[allow(clippy::too_many_arguments)]
fn isolate_add(
    lhs: ExprId,
    l: ExprId,
    r: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_item = simplifier.collect_steps();
    execute_add_isolation_pipeline_with_default_factored_linear_collect_and_unified_step_mapper_with_state(
        simplifier,
        lhs,
        l,
        r,
        rhs,
        op,
        var,
        include_item,
        steps,
        simplifier_context,
        simplifier_context_mut,
        context_render_expr,
        simplifier_simplify_expr,
        simplifier_prove_nonzero_status,
        |simplifier, equation| {
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op,
                var,
                simplifier,
                opts,
                ctx,
            )
        },
        medium_step,
    )
}

/// Handle isolation for `Sub(l, r)`: `(A - B) = RHS`
#[allow(clippy::too_many_arguments)]
fn isolate_sub(
    l: ExprId,
    r: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_item = simplifier.collect_steps();
    execute_sub_isolation_pipeline_with_default_plan_and_unified_step_mapper_with_state(
        simplifier,
        l,
        r,
        rhs,
        op,
        var,
        include_item,
        steps,
        simplifier_context,
        simplifier_context_mut,
        context_render_expr,
        simplifier_simplify_expr,
        |simplifier, equation| {
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op,
                var,
                simplifier,
                opts,
                ctx,
            )
        },
        medium_step,
    )
}

/// Handle isolation for `Mul(l, r)`: `A * B = RHS`
#[allow(clippy::too_many_arguments)]
fn isolate_mul(
    lhs: ExprId,
    l: ExprId,
    r: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_item = simplifier.collect_steps();
    execute_mul_isolation_pipeline_with_default_additive_linear_collect_and_unified_step_mapper_with_state(
        simplifier,
        lhs,
        l,
        r,
        rhs,
        op,
        var,
        include_item,
        steps,
        simplifier_context,
        simplifier_context_mut,
        |simplifier, equation| solve_with_ctx_and_options(equation, var, simplifier, opts, ctx),
        simplifier_simplify_expr,
        simplifier_prove_nonzero_status,
        simplifier_is_known_negative,
        context_render_expr,
        |simplifier, equation| {
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op,
                var,
                simplifier,
                opts,
                ctx,
            )
        },
        medium_step,
    )
}

/// Handle isolation for `Div(l, r)`: `A / B = RHS`
#[allow(clippy::too_many_arguments)]
fn isolate_div(
    lhs: ExprId,
    l: ExprId,
    r: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_numerator_items = simplifier.collect_steps();
    let include_denominator_items = simplifier.collect_steps();
    execute_div_isolation_pipeline_with_default_reciprocal_fallback_and_unified_step_mapper_with_state(
        simplifier,
        lhs,
        l,
        r,
        rhs,
        op,
        var,
        include_numerator_items,
        include_denominator_items,
        steps,
        simplifier_context,
        simplifier_context_mut,
        simplifier_is_known_negative,
        context_render_expr,
        simplifier_simplify_expr,
        |simplifier, equation| {
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )
        },
        simplifier_prove_nonzero_status,
        medium_step,
    )
}

/// Handle isolation for `Pow(b, e)`: `B^E = RHS`
#[allow(clippy::too_many_arguments)]
fn isolate_pow(
    lhs: ExprId,
    b: ExprId,
    e: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let config = build_pow_isolation_config(simplifier, opts);
    execute_isolation_pow(
        lhs, b, e, rhs, op, var, simplifier, opts, steps, ctx, config,
    )
}

fn build_pow_isolation_config(
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> PowIsolationConfig {
    cas_solver_core::strategy_options::pow_runtime_config_with(
        opts.core_domain_mode(),
        opts.wildcard_scope(),
        opts.value_domain == crate::ValueDomain::RealOnly,
        opts.budget,
        || simplifier.collect_steps(),
        || crate::SimplifyOptions::for_solve_tactic(opts.domain_mode),
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_isolation_pow(
    lhs: ExprId,
    b: ExprId,
    e: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &SolveCtx,
    config: PowIsolationConfig,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    execute_pow_isolation_with_kernel_config_and_unified_step_mapper_for_var_with_state(
        simplifier,
        lhs,
        b,
        e,
        rhs,
        op,
        var,
        config.kernel,
        steps,
        simplifier_context,
        simplifier_context_mut,
        context_render_expr,
        |simplifier, iso_lhs, iso_rhs, iso_op| {
            isolate(iso_lhs, iso_rhs, iso_op, var, simplifier, opts, ctx)
        },
        simplifier_simplify_expr,
        |_simplifier| crate::clear_blocked_hints(),
        |simplifier, expr| {
            simplifier
                .simplify_with_options(expr, config.tactic_opts.clone())
                .0
        },
        simplifier_collect_steps,
        |simplifier, tactic_base, tactic_rhs| {
            cas_solver_core::log_domain::classify_log_solve_with_env_and_tri_prover(
                &simplifier.context,
                tactic_base,
                tactic_rhs,
                opts.value_domain == crate::ValueDomain::RealOnly,
                opts.core_domain_mode(),
                &ctx.domain_env,
                |core_ctx, expr| {
                    cas_solver_core::predicate_proofs::prove_positive_core_with(
                        core_ctx,
                        expr,
                        opts.value_domain,
                        crate::helpers::prove_positive,
                    )
                },
            )
        },
        |simplifier, shortcut_rhs, shortcut_op| {
            isolate(e, shortcut_rhs, shortcut_op, var, simplifier, opts, ctx)
        },
        medium_step,
        |local_var, message| CasError::IsolationError(local_var.to_string(), message.to_string()),
        |core_ctx, assumption| {
            ctx.note_assumption(
                cas_solver_core::assumption_model::assumption_event_from_log_assumption(
                    core_ctx, assumption, b, rhs,
                ),
            );
        },
        |simplifier, equation| {
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )
            .ok()
            .map(|(solutions, _)| solutions)
        },
        |core_ctx, hint| {
            crate::register_blocked_hint(cas_solver_core::assumption_model::map_log_blocked_hint(
                core_ctx, hint,
            ));
        },
        |message| CasError::UnsupportedInRealDomain(message.to_string()),
        |simplifier, equation| {
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )
        },
    )
}
