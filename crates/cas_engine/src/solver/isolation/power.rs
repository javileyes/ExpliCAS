use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{medium_step, render_expr as solver_render_expr, SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::solve_outcome::{
    build_pow_exponent_shortcut_execution_plan, classify_pow_exponent_base_flags,
    derive_pow_isolation_route, detect_pow_exponent_shortcut_inputs,
    ensure_pow_exponent_rhs_without_variable,
    execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut_and_each_assumption,
    execute_pow_base_isolation_pipeline_with_item_and_merge_with_existing_steps,
    execute_pow_exponent_log_isolation_pipeline_with_plan_and_merge_with_existing_steps_with,
    execute_pow_exponent_log_unsupported_pipeline_from_decision_with,
    execute_power_base_one_shortcut_pipeline_with_item_for_pow_and_finalize_with_existing_steps_with,
    finalize_pow_exponent_shortcut_pipeline_with_existing_steps, map_pow_exponent_shortcut_with,
    plan_pow_exponent_log_isolation_step_with,
    plan_pow_exponent_log_unsupported_execution_from_decision_with,
    plan_pow_exponent_shortcut_action_from_inputs, solve_pow_exponent_shortcut_pipeline_with_item,
    solve_solve_tactic_normalization_pipeline_with_item, PowIsolationRoute,
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
    execute_pow_base_isolation_pipeline_with_item_and_merge_with_existing_steps(
        include_item,
        steps,
        action,
        |iso_lhs, iso_rhs, iso_op| isolate(iso_lhs, iso_rhs, iso_op, var, simplifier, opts, ctx),
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
    let rhs_expr = simplifier.context.get(rhs).clone();
    let (bases_equal, rhs_pow_base_equal) =
        detect_pow_exponent_shortcut_inputs(rhs, &rhs_expr, |candidate| {
            let diff = simplifier.context.add(Expr::Sub(b, candidate));
            let reduced = simplifier.simplify(diff).0;
            cas_solver_core::isolation_utils::is_numeric_zero(&simplifier.context, reduced)
        });
    let shortcut_action = plan_pow_exponent_shortcut_action_from_inputs(
        &mut simplifier.context,
        b,
        op.clone(),
        bases_equal,
        rhs_pow_base_equal,
        base_is_zero,
        base_is_numeric,
        opts.budget.max_branches >= 2,
    );
    let shortcut_plan = build_pow_exponent_shortcut_execution_plan(shortcut_action);
    let shortcut_execution =
        map_pow_exponent_shortcut_with(shortcut_plan, e, b, rhs, op.clone(), var, |expr| {
            solver_render_expr(&simplifier.context, expr)
        });
    let shortcut_solved = solve_pow_exponent_shortcut_pipeline_with_item(
        shortcut_execution,
        include_item,
        |shortcut_rhs, shortcut_op| {
            isolate(e, shortcut_rhs, shortcut_op, var, simplifier, opts, ctx)
        },
        |item| medium_step(item.description().to_string(), item.equation),
    )?;
    if let Some((solution_set, merged_steps)) =
        finalize_pow_exponent_shortcut_pipeline_with_existing_steps(shortcut_solved, &mut steps)
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
    let (tactic_base, tactic_rhs, tactic_steps) = if solve_tactic_enabled {
        crate::domain::clear_blocked_hints();
        let sim_base = simplifier.simplify_with_options(b, tactic_opts.clone()).0;
        let sim_rhs = simplifier.simplify_with_options(rhs, tactic_opts.clone()).0;
        crate::domain::clear_blocked_hints();

        let steps = if sim_base != b || sim_rhs != rhs {
            let include_item = simplifier.collect_steps();
            solve_solve_tactic_normalization_pipeline_with_item(
                &mut simplifier.context,
                sim_base,
                e,
                sim_rhs,
                op.clone(),
                include_item,
                |item| medium_step(item.description().to_string(), item.equation),
            )
        } else {
            Vec::new()
        };
        (sim_base, sim_rhs, steps)
    } else {
        (b, rhs, Vec::new())
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
    let unsupported_execution = plan_pow_exponent_log_unsupported_execution_from_decision_with(
        &mut simplifier.context,
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
    );
    let include_terminal_items = simplifier.collect_steps();
    let include_unsupported_items = simplifier.collect_steps();
    let terminal_result =
        execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut_and_each_assumption(
            &mut simplifier.context,
            &decision,
            mode,
            wildcard_scope,
            lhs,
            rhs,
            var,
            source_equation.clone(),
            " (residual)",
            include_terminal_items,
            &mut steps,
            |item| medium_step(item.description().to_string(), item.equation),
            |core_ctx, assumption| {
                let event =
                    crate::solver::assumption_event_from_log_assumption_targets(core_ctx, assumption, b, rhs);
                ctx.note_assumption(event);
            },
        );
    match terminal_result {
        cas_solver_core::solve_outcome::LogDecisionTerminalResult::Terminal {
            solution_set,
            steps,
        } => return Ok((solution_set, steps)),
        cas_solver_core::solve_outcome::LogDecisionTerminalResult::NeedsComplex { message } => {
            return Err(CasError::UnsupportedInRealDomain(message.to_string()))
        }
        cas_solver_core::solve_outcome::LogDecisionTerminalResult::Continue => {}
    }

    if let Some(unsupported_solved) =
        execute_pow_exponent_log_unsupported_pipeline_from_decision_with(
            include_unsupported_items,
            move || unsupported_execution,
            |equation| {
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
            |item| medium_step(item.description().to_string(), item.equation),
        )
    {
        for hint in unsupported_solved.blocked_hints {
            let blocked_hint =
                crate::solver::domain_blocked_hint_from_log_blocked_hint(&simplifier.context, hint);
            crate::domain::register_blocked_hint(blocked_hint);
        }
        steps.extend(unsupported_solved.steps);
        return Ok((unsupported_solved.solution_set, steps));
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
    execute_pow_exponent_log_isolation_pipeline_with_plan_and_merge_with_existing_steps_with(
        include_item,
        steps,
        rewrite,
        |equation| {
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
