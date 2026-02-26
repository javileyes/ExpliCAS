use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{medium_step, render_expr as solver_render_expr, SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::log_domain::{decision_assumptions, LogSolveDecision};
use cas_solver_core::solve_outcome::{
    classify_pow_exponent_base_flags, derive_pow_isolation_route,
    execute_log_terminal_outcome_pipeline_with_item,
    execute_pow_base_isolation_pipeline_with_item_and_merge_with_existing_steps_with,
    execute_pow_exponent_log_isolation_pipeline_with_item_with,
    execute_pow_exponent_log_unsupported_pipeline_from_decision_with,
    execute_pow_exponent_shortcut_pipeline_with_item_with,
    execute_power_base_one_shortcut_pipeline_with_item_for_pow_with,
    finalize_pow_exponent_log_unsupported_pipeline_with_existing_steps,
    finalize_pow_exponent_shortcut_pipeline_with_existing_steps,
    merge_solved_with_existing_steps_append, merge_solved_with_existing_steps_prepend,
    plan_pow_exponent_log_isolation_step_with,
    plan_pow_exponent_log_unsupported_execution_from_decision_with,
    pow_exponent_rhs_contains_variable, shortcut_bases_equivalent_by_difference_with,
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
    let include_item = simplifier.collect_steps();
    let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
    execute_pow_base_isolation_pipeline_with_item_and_merge_with_existing_steps_with(
        include_item,
        steps,
        || {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            cas_solver_core::solve_outcome::build_pow_base_isolation_action_with(
                &mut simplifier_ref.context,
                b,
                e,
                rhs,
                op,
                solver_render_expr,
            )
        },
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
    let shortcut_solved = {
        let include_item = simplifier.collect_steps();
        let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
        execute_pow_exponent_shortcut_pipeline_with_item_with(
            e,
            b,
            rhs,
            op.clone(),
            var,
            base_is_zero,
            base_is_numeric,
            opts.budget.max_branches >= 2,
            include_item,
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
    };

    if let Some((solution_set, merged_steps)) =
        finalize_pow_exponent_shortcut_pipeline_with_existing_steps(shortcut_solved, &mut steps)
    {
        return Ok((solution_set, merged_steps));
    }

    // SAFETY GUARD: If RHS contains the variable, we cannot invert with log.
    if pow_exponent_rhs_contains_variable(&simplifier.context, rhs, var) {
        return Err(CasError::IsolationError(
            var.to_string(),
            "Cannot isolate exponential: variable appears on both sides".to_string(),
        ));
    }

    // ================================================================
    // DOMAIN GUARDS for log operation (RealOnly mode)
    // ================================================================
    // GUARD 1: Handle base = 1 special case
    let include_item = simplifier.collect_steps();
    if let Some(solved_shortcut) = execute_power_base_one_shortcut_pipeline_with_item_for_pow_with(
        &simplifier.context,
        b,
        lhs,
        rhs,
        op.clone(),
        include_item,
        solver_render_expr,
        |item| medium_step(item.description().to_string(), item.equation),
    ) {
        return Ok(merge_solved_with_existing_steps_append(
            (solved_shortcut.solution_set, solved_shortcut.steps),
            steps,
        ));
    }

    // ================================================================
    use crate::solver::classify_log_solve;

    // ================================================================
    // SOLVE TACTIC: Pre-simplify base/rhs with Analytic rules in Assume mode
    // ================================================================
    let (tactic_base, tactic_rhs) = if opts.domain_mode == crate::domain::DomainMode::Assume
        && opts.value_domain == crate::semantics::ValueDomain::RealOnly
    {
        use crate::SimplifyOptions;
        let tactic_opts = SimplifyOptions::for_solve_tactic(opts.domain_mode);

        crate::domain::clear_blocked_hints();

        let (sim_base, _) = simplifier.simplify_with_options(b, tactic_opts.clone());
        let (sim_rhs, _) = simplifier.simplify_with_options(rhs, tactic_opts);

        crate::domain::clear_blocked_hints();

        // Add educational step if tactic transformed something.
        if sim_base != b || sim_rhs != rhs {
            let include_item = simplifier.collect_steps();
            let tactic_steps = solve_solve_tactic_normalization_pipeline_with_item(
                &mut simplifier.context,
                sim_base,
                e,
                sim_rhs,
                op.clone(),
                include_item,
                |item| medium_step(item.description().to_string(), item.equation),
            );
            steps.extend(tactic_steps);
        }

        (sim_base, sim_rhs)
    } else {
        (b, rhs)
    };

    let decision = classify_log_solve(
        &simplifier.context,
        tactic_base,
        tactic_rhs,
        &opts,
        &ctx.domain_env,
    );

    let mode = opts.core_domain_mode();
    let wildcard_scope = opts.wildcard_scope();

    let include_item = simplifier.collect_steps();
    if let Some(solved_terminal) = execute_log_terminal_outcome_pipeline_with_item(
        &mut simplifier.context,
        &decision,
        mode,
        wildcard_scope,
        lhs,
        rhs,
        var,
        Equation {
            lhs,
            rhs,
            op: op.clone(),
        },
        " (residual)",
        include_item,
        |item| medium_step(item.description().to_string(), item.equation),
    ) {
        return Ok(merge_solved_with_existing_steps_append(
            (solved_terminal.solution_set, solved_terminal.steps),
            steps,
        ));
    }

    for assumption in decision_assumptions(&decision).iter().copied() {
        let event = crate::assumptions::AssumptionEvent::from_log_assumption(
            assumption,
            &simplifier.context,
            b,
            rhs,
        );
        ctx.note_assumption(event);
    }

    if let LogSolveDecision::NeedsComplex(msg) = &decision {
        return Err(CasError::UnsupportedInRealDomain((*msg).to_string()));
    }
    if matches!(decision, LogSolveDecision::EmptySet(_)) {
        unreachable!("handled by terminal action");
    }

    let source_equation = Equation {
        lhs,
        rhs,
        op: op.clone(),
    };
    let include_items = simplifier.collect_steps();
    let unsupported_solved = {
        let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
        execute_pow_exponent_log_unsupported_pipeline_from_decision_with(
            include_items,
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
                    source_equation,
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
        )
    };
    if let Some((solution_set, merged_steps)) =
        finalize_pow_exponent_log_unsupported_pipeline_with_existing_steps(
            unsupported_solved,
            &mut steps,
            |hint| {
                let event = crate::assumptions::AssumptionEvent::from_log_assumption(
                    hint.assumption,
                    &simplifier.context,
                    b,
                    rhs,
                );
                crate::domain::register_blocked_hint(crate::domain::BlockedHint {
                    key: event.key,
                    expr_id: hint.expr_id,
                    rule: hint.rule.to_string(),
                    suggestion: hint.suggestion,
                });
            },
        )
    {
        return Ok((solution_set, merged_steps));
    }
    // ================================================================
    // End of domain guards
    // ================================================================

    let include_item = simplifier.collect_steps();
    let solved_log = {
        let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
        execute_pow_exponent_log_isolation_pipeline_with_item_with(
            include_item,
            || {
                let mut simplifier_ref = runtime_cell.borrow_mut();
                plan_pow_exponent_log_isolation_step_with(
                    &mut simplifier_ref.context,
                    e,
                    b,
                    rhs,
                    op,
                    None,
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
            },
            |item| medium_step(item.description().to_string(), item.equation),
        )?
    };
    Ok(merge_solved_with_existing_steps_prepend(
        (solved_log.solution_set, solved_log.steps),
        steps,
    ))
}
