use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{medium_step, render_expr as solver_render_expr, SolveStep, SolverOptions};
use crate::SimplifyOptions;
use cas_ast::{Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_power::{
    execute_pow_base_isolation_with_default_action_with_state,
    execute_pow_exponent_shortcuts_and_guards_with_state,
    execute_pow_exponent_tactic_then_log_pipeline_with_state,
    execute_pow_isolation_route_for_var_with_state,
};
use cas_solver_core::solve_outcome::{
    classify_pow_exponent_base_flags, ensure_pow_exponent_rhs_without_variable,
    plan_pow_exponent_log_isolation_step_with,
    plan_pow_exponent_log_unsupported_execution_from_decision_with,
    plan_pow_exponent_shortcut_action_from_inputs,
    solve_solve_tactic_normalization_pipeline_with_item,
};

use super::isolate;

/// Handle isolation for `Pow(b, e)`: `B^E = RHS`
#[allow(clippy::too_many_arguments)]
pub(super) fn isolate_pow(
    lhs: ExprId,
    b: ExprId,
    e: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    execute_pow_isolation_route_for_var_with_state(
        simplifier,
        |simplifier| &simplifier.context,
        b,
        var,
        |simplifier| {
            // Variable in base: B^E = RHS
            isolate_pow_base(
                lhs,
                b,
                e,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                steps.clone(),
                ctx,
            )
        },
        |simplifier| {
            // Variable in exponent: B^E = RHS → E = log_B(RHS)
            isolate_pow_exponent(
                lhs,
                b,
                e,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                steps.clone(),
                ctx,
            )
        },
    )
}

/// Handle `B^E = RHS` when variable is in `B` (the base)
#[allow(clippy::too_many_arguments)]
fn isolate_pow_base(
    _lhs: ExprId,
    b: ExprId,
    e: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_item = simplifier.collect_steps();
    execute_pow_base_isolation_with_default_action_with_state(
        simplifier,
        b,
        e,
        rhs,
        op,
        include_item,
        steps,
        |simplifier| &mut simplifier.context,
        solver_render_expr,
        |simplifier, iso_lhs, iso_rhs, iso_op| {
            isolate(iso_lhs, iso_rhs, iso_op, var, simplifier, opts, ctx)
        },
        |item| medium_step(item.description().to_string(), item.equation),
    )
}

/// Handle `B^E = RHS` when variable is in `E` (the exponent) — logarithmic isolation
#[allow(clippy::too_many_arguments)]
fn isolate_pow_exponent(
    lhs: ExprId,
    b: ExprId,
    e: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    mut steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_shortcut_item = simplifier.collect_steps();
    let include_base_one_item = simplifier.collect_steps();
    if let Some((solution_set, merged_steps)) =
        execute_pow_exponent_shortcuts_and_guards_with_state(
            simplifier,
            lhs,
            b,
            e,
            rhs,
            op.clone(),
            var,
            opts.budget.max_branches >= 2,
            include_shortcut_item,
            include_base_one_item,
            &mut steps,
            |simplifier, base| classify_pow_exponent_base_flags(&simplifier.context, base),
            |simplifier, id| simplifier.context.get(id).clone(),
            |simplifier,
             base_id,
             rel_op,
             bases_equal,
             rhs_pow_base_equal,
             base_is_zero,
             base_is_numeric,
             can_branch| {
                plan_pow_exponent_shortcut_action_from_inputs(
                    &mut simplifier.context,
                    base_id,
                    rel_op,
                    bases_equal,
                    rhs_pow_base_equal,
                    base_is_zero,
                    base_is_numeric,
                    can_branch,
                )
            },
            |simplifier, left, right| {
                let diff = simplifier.context.add(Expr::Sub(left, right));
                let reduced = simplifier.simplify(diff).0;
                cas_solver_core::isolation_utils::is_numeric_zero(&simplifier.context, reduced)
            },
            |simplifier, expr| solver_render_expr(&simplifier.context, expr),
            |simplifier, shortcut_rhs, shortcut_op| {
                isolate(e, shortcut_rhs, shortcut_op, var, simplifier, opts, ctx)
            },
            |item| medium_step(item.description().to_string(), item.equation),
            |simplifier, local_rhs, local_var| {
                ensure_pow_exponent_rhs_without_variable(&simplifier.context, local_rhs, local_var)
                    .map_err(|message| {
                        CasError::IsolationError(local_var.to_string(), message.to_string())
                    })
            },
            |simplifier, base, local_lhs, local_rhs, local_op, include_item, existing_steps| {
                cas_solver_core::solve_outcome::execute_power_base_one_shortcut_pipeline_with_item_for_pow_and_finalize_with_existing_steps_with(
                &simplifier.context,
                base,
                local_lhs,
                local_rhs,
                local_op,
                include_item,
                existing_steps,
                solver_render_expr,
                |item| medium_step(item.description().to_string(), item.equation),
            )
            },
        )?
    {
        return Ok((solution_set, merged_steps));
    }

    // ================================================================
    // SOLVE TACTIC: Pre-simplify base/rhs with Analytic rules in Assume mode
    // ================================================================
    let solve_tactic_enabled = opts.domain_mode == crate::domain::DomainMode::Assume
        && opts.value_domain == crate::semantics::ValueDomain::RealOnly;
    let tactic_opts = SimplifyOptions::for_solve_tactic(opts.domain_mode);
    let mode = opts.core_domain_mode();
    let wildcard_scope = opts.wildcard_scope();
    let include_terminal_items = simplifier.collect_steps();
    let include_unsupported_items = simplifier.collect_steps();
    let include_log_item = simplifier.collect_steps();
    execute_pow_exponent_tactic_then_log_pipeline_with_state(
        simplifier,
        lhs,
        b,
        e,
        rhs,
        op,
        var,
        solve_tactic_enabled,
        mode,
        wildcard_scope,
        opts.budget.can_branch(),
        include_terminal_items,
        include_unsupported_items,
        include_log_item,
        steps,
        |_simplifier| crate::domain::clear_blocked_hints(),
        |simplifier, expr| {
            simplifier
                .simplify_with_options(expr, tactic_opts.clone())
                .0
        },
        |simplifier, sim_base, sim_exp, sim_rhs, sim_op| {
            let include_item = simplifier.collect_steps();
            solve_solve_tactic_normalization_pipeline_with_item(
                &mut simplifier.context,
                sim_base,
                sim_exp,
                sim_rhs,
                sim_op,
                include_item,
                |item| medium_step(item.description().to_string(), item.equation),
            )
        },
        |simplifier, tactic_base, tactic_rhs| {
            crate::solver::classify_log_solve(
                &simplifier.context,
                tactic_base,
                tactic_rhs,
                &opts,
                &ctx.domain_env,
            )
        },
        |simplifier,
         decision,
         can_branch,
         local_lhs,
         local_rhs,
         var_name,
         local_exp,
         local_base,
         raw_rhs,
         local_op,
         source_equation| {
            plan_pow_exponent_log_unsupported_execution_from_decision_with(
                &mut simplifier.context,
                decision,
                can_branch,
                local_lhs,
                local_rhs,
                var_name,
                local_exp,
                local_base,
                raw_rhs,
                local_op,
                source_equation,
                solver_render_expr,
            )
        },
        |simplifier| simplifier.context.clone(),
        |item| medium_step(item.description().to_string(), item.equation),
        |core_ctx, assumption| {
            let event = crate::solver::assumption_event_from_log_assumption_targets(
                core_ctx, assumption, b, rhs,
            );
            ctx.note_assumption(event);
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
            let blocked_hint =
                crate::solver::domain_blocked_hint_from_log_blocked_hint(core_ctx, hint);
            crate::domain::register_blocked_hint(blocked_hint);
        },
        |message| CasError::UnsupportedInRealDomain(message.to_string()),
        |simplifier, local_exp, local_base, local_rhs, local_op| {
            plan_pow_exponent_log_isolation_step_with(
                &mut simplifier.context,
                local_exp,
                local_base,
                local_rhs,
                local_op,
                None,
                solver_render_expr,
            )
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
        },
        |item| medium_step(item.description().to_string(), item.equation),
    )
}
