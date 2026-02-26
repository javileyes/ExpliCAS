use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{medium_step, render_expr as solver_render_expr, SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::solve_outcome::{
    classify_pow_exponent_base_flags, derive_pow_isolation_route,
    ensure_pow_exponent_rhs_without_variable,
    execute_and_resolve_pow_exponent_log_terminal_then_post_pipeline_with_existing_steps_mut,
    execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut_and_each_assumption,
    execute_pow_base_isolation_pipeline_with_item_and_merge_with_existing_steps,
    execute_pow_exponent_log_isolation_pipeline_with_plan_and_merge_with_existing_steps_with,
    execute_pow_exponent_shortcut_pipeline_with_item_and_finalize_with_existing_steps_with,
    execute_pow_exponent_solve_tactic_normalization_with,
    execute_power_base_one_shortcut_pipeline_with_item_for_pow_and_finalize_with_existing_steps_with,
    plan_pow_exponent_log_isolation_step_with,
    plan_pow_exponent_log_unsupported_execution_from_decision_with,
    shortcut_bases_equivalent_by_difference_with,
    solve_solve_tactic_normalization_pipeline_with_item, PowIsolationRoute,
};

use super::isolate;

fn bases_are_equivalent(simplifier: &mut Simplifier, base: ExprId, candidate: ExprId) -> bool {
    let runtime_cell = std::cell::RefCell::new(simplifier);
    shortcut_bases_equivalent_by_difference_with(
        base,
        candidate,
        |lhs, rhs| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            simplifier_ref.context.add(Expr::Sub(lhs, rhs))
        },
        |expr| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            simplifier_ref.simplify(expr).0
        },
        |expr| {
            let simplifier_ref = runtime_cell.borrow();
            cas_solver_core::isolation_utils::is_numeric_zero(&simplifier_ref.context, expr)
        },
    )
}

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
    match derive_pow_isolation_route(&simplifier.context, b, var) {
        PowIsolationRoute::VariableInBase => {
            // Variable in base: B^E = RHS
            isolate_pow_base(lhs, b, e, rhs, op, var, simplifier, opts, steps, ctx)
        }
        PowIsolationRoute::VariableInExponent => {
            // Variable in exponent: B^E = RHS → E = log_B(RHS)
            isolate_pow_exponent(lhs, b, e, rhs, op, var, simplifier, opts, steps, ctx)
        }
    }
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
    let action = cas_solver_core::solve_outcome::build_pow_base_isolation_action_with(
        &mut simplifier.context,
        b,
        e,
        rhs,
        op,
        solver_render_expr,
    );
    let include_item = simplifier.collect_steps();
    let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
    execute_pow_base_isolation_pipeline_with_item_and_merge_with_existing_steps(
        include_item,
        steps,
        action,
        |iso_lhs, iso_rhs, iso_op| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            isolate(iso_lhs, iso_rhs, iso_op, var, *simplifier_ref, opts, ctx)
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
    // ================================================================
    // POWER EQUALS BASE SHORTCUT: base^x = base
    // ================================================================
    let (base_is_zero, base_is_numeric) = classify_pow_exponent_base_flags(&simplifier.context, b);
    let include_item = simplifier.collect_steps();
    let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
    if let Some((solution_set, merged_steps)) =
        execute_pow_exponent_shortcut_pipeline_with_item_and_finalize_with_existing_steps_with(
            e,
            b,
            rhs,
            op.clone(),
            var,
            base_is_zero,
            base_is_numeric,
            opts.budget.max_branches >= 2,
            include_item,
            &mut steps,
            |expr| {
                let simplifier_ref = runtime_cell.borrow();
                simplifier_ref.context.get(expr).clone()
            },
            |base, rel_op, bases_equal, rhs_pow_base_equal, is_zero, is_numeric, can_branch| {
                let mut simplifier_ref = runtime_cell.borrow_mut();
                cas_solver_core::solve_outcome::plan_pow_exponent_shortcut_action_from_inputs(
                    &mut simplifier_ref.context,
                    base,
                    rel_op,
                    bases_equal,
                    rhs_pow_base_equal,
                    is_zero,
                    is_numeric,
                    can_branch,
                )
            },
            |lhs_base, rhs_base| {
                let mut simplifier_ref = runtime_cell.borrow_mut();
                bases_are_equivalent(*simplifier_ref, lhs_base, rhs_base)
            },
            |expr| {
                let simplifier_ref = runtime_cell.borrow();
                solver_render_expr(&simplifier_ref.context, expr)
            },
            |shortcut_rhs, shortcut_op| {
                let mut simplifier_ref = runtime_cell.borrow_mut();
                isolate(
                    e,
                    shortcut_rhs,
                    shortcut_op,
                    var,
                    *simplifier_ref,
                    opts,
                    ctx,
                )
            },
            |item| medium_step(item.description().to_string(), item.equation),
        )?
    {
        return Ok((solution_set, merged_steps));
    }

    // SAFETY GUARD: If RHS contains the variable, we cannot invert with log.
    ensure_pow_exponent_rhs_without_variable(&simplifier.context, rhs, var)
        .map_err(|message| CasError::IsolationError(var.to_string(), message.to_string()))?;

    // ================================================================
    // DOMAIN GUARDS for log operation (RealOnly mode)
    // ================================================================
    // GUARD 1: Handle base = 1 special case
    let include_item = simplifier.collect_steps();
    if let Some(solved_shortcut) =
        execute_power_base_one_shortcut_pipeline_with_item_for_pow_and_finalize_with_existing_steps_with(
            &simplifier.context,
            b,
            lhs,
            rhs,
            op.clone(),
            include_item,
            &mut steps,
            solver_render_expr,
            |item| medium_step(item.description().to_string(), item.equation),
        )
    {
        return Ok(solved_shortcut);
    }

    // ================================================================
    use crate::solver::classify_log_solve;

    // ================================================================
    // SOLVE TACTIC: Pre-simplify base/rhs with Analytic rules in Assume mode
    // ================================================================
    use crate::SimplifyOptions;
    let solve_tactic_enabled = opts.domain_mode == crate::domain::DomainMode::Assume
        && opts.value_domain == crate::semantics::ValueDomain::RealOnly;
    let tactic_opts = SimplifyOptions::for_solve_tactic(opts.domain_mode);
    let (tactic_base, tactic_rhs, tactic_steps) = {
        let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
        execute_pow_exponent_solve_tactic_normalization_with(
            b,
            e,
            rhs,
            op.clone(),
            solve_tactic_enabled,
            crate::domain::clear_blocked_hints,
            |expr| {
                let mut simplifier_ref = runtime_cell.borrow_mut();
                simplifier_ref
                    .simplify_with_options(expr, tactic_opts.clone())
                    .0
            },
            |sim_base, exponent, sim_rhs, rel_op| {
                let mut simplifier_ref = runtime_cell.borrow_mut();
                let include_item = simplifier_ref.collect_steps();
                solve_solve_tactic_normalization_pipeline_with_item(
                    &mut simplifier_ref.context,
                    sim_base,
                    exponent,
                    sim_rhs,
                    rel_op,
                    include_item,
                    |item| medium_step(item.description().to_string(), item.equation),
                )
            },
        )
    };
    steps.extend(tactic_steps);

    let decision = classify_log_solve(
        &simplifier.context,
        tactic_base,
        tactic_rhs,
        &opts,
        &ctx.domain_env,
    );

    let mode = opts.core_domain_mode();
    let wildcard_scope = opts.wildcard_scope();

    let source_equation = Equation {
        lhs,
        rhs,
        op: op.clone(),
    };
    let include_terminal_items = simplifier.collect_steps();
    let include_unsupported_items = simplifier.collect_steps();
    let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
    if let Some((solution_set, steps)) =
        execute_and_resolve_pow_exponent_log_terminal_then_post_pipeline_with_existing_steps_mut(
            |existing_steps| {
                let mut simplifier_ref = runtime_cell.borrow_mut();
                execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut_and_each_assumption(
                    &mut simplifier_ref.context,
                    &decision,
                    mode,
                    wildcard_scope,
                    lhs,
                    rhs,
                    var,
                    source_equation.clone(),
                    " (residual)",
                    include_terminal_items,
                    existing_steps,
                    |item| medium_step(item.description().to_string(), item.equation),
                    |core_ctx, assumption| {
                        let event = crate::solver::assumption_event_from_log_assumption_targets(
                            core_ctx, assumption, b, rhs,
                        );
                        ctx.note_assumption(event);
                    },
                )
            },
            include_unsupported_items,
            &mut steps,
            || {
                let mut simplifier_ref = runtime_cell.borrow_mut();
                plan_pow_exponent_log_unsupported_execution_from_decision_with(
                    &mut simplifier_ref.context,
                    &decision,
                    opts.budget.can_branch(),
                    lhs,
                    rhs,
                    var,
                    e,
                    b,
                    rhs,
                    op.clone(),
                    source_equation.clone(),
                    solver_render_expr,
                )
            },
            |equation| {
                let mut simplifier_ref = runtime_cell.borrow_mut();
                isolate(
                    equation.lhs,
                    equation.rhs,
                    equation.op.clone(),
                    var,
                    *simplifier_ref,
                    opts,
                    ctx,
                )
                .ok()
                .map(|(solutions, _)| solutions)
            },
            |item| medium_step(item.description().to_string(), item.equation),
            |hint| {
                let simplifier_ref = runtime_cell.borrow();
                let blocked_hint = crate::solver::domain_blocked_hint_from_log_blocked_hint(
                    &simplifier_ref.context,
                    hint,
                );
                crate::domain::register_blocked_hint(blocked_hint);
            },
        )
        .map_err(|message| CasError::UnsupportedInRealDomain(message.to_string()))?
    {
        return Ok((solution_set, steps));
    }
    // ================================================================
    // End of domain guards
    // ================================================================

    let rewrite = plan_pow_exponent_log_isolation_step_with(
        &mut simplifier.context,
        e,
        b,
        rhs,
        op,
        None,
        solver_render_expr,
    );
    let include_item = simplifier.collect_steps();
    let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
    execute_pow_exponent_log_isolation_pipeline_with_plan_and_merge_with_existing_steps_with(
        include_item,
        steps,
        rewrite,
        |equation| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op.clone(),
                var,
                *simplifier_ref,
                opts,
                ctx,
            )
        },
        |item| medium_step(item.description().to_string(), item.equation),
    )
}
