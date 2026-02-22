use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::{
    contains_var, is_even_integer_expr, is_known_negative, is_numeric_one, is_numeric_zero,
};
use cas_solver_core::log_domain::{decision_assumptions, LogSolveDecision};
use cas_solver_core::solve_outcome::{
    build_pow_exponent_log_isolation_step_with, build_pow_exponent_shortcut_execution_plan,
    build_solve_tactic_normalization_step, conditional_solution_message,
    detect_pow_exponent_shortcut_inputs, guarded_or_residual, map_pow_base_isolation_plan_with,
    map_pow_exponent_shortcut_with, plan_pow_base_isolation,
    plan_pow_exponent_shortcut_action_from_inputs, residual_budget_exhausted_message,
    residual_message, resolve_log_terminal_outcome, resolve_log_unsupported_outcome,
    resolve_power_base_one_shortcut_with, terminal_outcome_message, LogUnsupportedOutcome,
    PowBaseIsolationEngineAction, PowExponentShortcutEngineAction,
};

use super::{isolate, prepend_steps};

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
    if contains_var(&simplifier.context, b, var) {
        // Variable in base: B^E = RHS
        isolate_pow_base(lhs, b, e, rhs, op, var, simplifier, opts, steps, ctx)
    } else {
        // Variable in exponent: B^E = RHS → E = log_B(RHS)
        isolate_pow_exponent(lhs, b, e, rhs, op, var, simplifier, opts, steps, ctx)
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
    mut steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let exponent_is_even = is_even_integer_expr(&simplifier.context, e);
    let rhs_is_known_negative = is_known_negative(&simplifier.context, rhs);
    let exponent_is_known_negative = is_known_negative(&simplifier.context, e);
    let plan = plan_pow_base_isolation(
        &mut simplifier.context,
        b,
        e,
        rhs,
        op.clone(),
        exponent_is_even,
        rhs_is_known_negative,
        exponent_is_known_negative,
    );

    let action = map_pow_base_isolation_plan_with(plan, b, e, rhs, op, |id| {
        format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id
            }
        )
    });

    match action {
        PowBaseIsolationEngineAction::ReturnSolutionSet { solutions, step } => {
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: step.description,
                    equation_after: step.equation_after,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            Ok((solutions, steps))
        }
        PowBaseIsolationEngineAction::IsolateBase {
            rhs: target_rhs,
            op: target_op,
            step,
        } => {
            let lhs_after = step.equation_after.lhs;
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: step.description,
                    equation_after: step.equation_after,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            let results = isolate(lhs_after, target_rhs, target_op, var, simplifier, opts, ctx)?;
            prepend_steps(results, steps)
        }
    }
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
    let rhs_expr = simplifier.context.get(rhs).clone();
    let (bases_equal, rhs_pow_base_equal) =
        detect_pow_exponent_shortcut_inputs(rhs, &rhs_expr, |candidate| {
            cas_solver_core::solve_outcome::shortcut_bases_equivalent_with(
                b,
                candidate,
                |left, right| {
                    let diff = simplifier.context.add(Expr::Sub(left, right));
                    let (sim_diff, _) = simplifier.simplify(diff);
                    is_numeric_zero(&simplifier.context, sim_diff)
                },
            )
        });
    let base_is_zero = is_numeric_zero(&simplifier.context, b);
    let base_is_numeric = matches!(simplifier.context.get(b), Expr::Number(_));

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
    let shortcut_engine_action =
        map_pow_exponent_shortcut_with(shortcut_plan, e, b, rhs, op.clone(), var, |id| {
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id
                }
            )
        });

    match shortcut_engine_action {
        PowExponentShortcutEngineAction::Continue => {}
        PowExponentShortcutEngineAction::IsolateExponent {
            rhs: target_rhs,
            op: target_op,
            step,
        } => {
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: step.description,
                    equation_after: step.equation_after,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            let results = isolate(e, target_rhs, target_op, var, simplifier, opts, ctx)?;
            return prepend_steps(results, steps);
        }
        PowExponentShortcutEngineAction::ReturnSolutionSet { solutions, step } => {
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: step.description,
                    equation_after: step.equation_after,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            return Ok((solutions, steps));
        }
    }

    // SAFETY GUARD: If RHS contains the variable, we cannot invert with log.
    if contains_var(&simplifier.context, rhs, var) {
        return Err(CasError::IsolationError(
            var.to_string(),
            "Cannot isolate exponential: variable appears on both sides".to_string(),
        ));
    }

    // ================================================================
    // DOMAIN GUARDS for log operation (RealOnly mode)
    // ================================================================
    // GUARD 1: Handle base = 1 special case
    if let Some(outcome) = resolve_power_base_one_shortcut_with(
        is_numeric_one(&simplifier.context, b),
        is_numeric_one(&simplifier.context, rhs),
        lhs,
        rhs,
        op.clone(),
        |id| {
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id
                }
            )
        },
    ) {
        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: outcome.step.description,
                equation_after: outcome.step.equation_after,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        return Ok((outcome.solutions, steps));
    }

    // ================================================================
    use crate::solver::domain_guards::classify_log_solve;

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

        // Add educational step if tactic transformed something
        if (sim_base != b || sim_rhs != rhs) && simplifier.collect_steps() {
            let normalize_step = build_solve_tactic_normalization_step(Equation {
                lhs: simplifier.context.add(Expr::Pow(sim_base, e)),
                rhs: sim_rhs,
                op: op.clone(),
            });
            steps.push(SolveStep {
                description: normalize_step.description,
                equation_after: normalize_step.equation_after,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
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

    let mode = crate::solver::domain_guards::to_core_domain_mode(opts.domain_mode);
    let wildcard_scope = opts.assume_scope == crate::semantics::AssumeScope::Wildcard;

    if let Some(outcome) = resolve_log_terminal_outcome(
        &mut simplifier.context,
        &decision,
        mode,
        wildcard_scope,
        lhs,
        rhs,
        var,
    ) {
        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: terminal_outcome_message(&outcome, " (residual)"),
                equation_after: Equation { lhs, rhs, op },
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        return Ok((outcome.solutions, steps));
    }

    for assumption in decision_assumptions(&decision).iter().copied() {
        let event = crate::assumptions::AssumptionEvent::from_log_assumption(
            assumption,
            &simplifier.context,
            b,
            rhs,
        );
        crate::solver::note_assumption(event);
    }

    if let LogSolveDecision::NeedsComplex(msg) = &decision {
        return Err(CasError::UnsupportedInRealDomain((*msg).to_string()));
    }
    if matches!(decision, LogSolveDecision::EmptySet(_)) {
        unreachable!("handled by terminal action");
    }

    if let Some(unsupported_outcome) = resolve_log_unsupported_outcome(
        &mut simplifier.context,
        &decision,
        opts.budget.can_branch(),
        lhs,
        rhs,
        var,
        b,
        rhs,
    ) {
        match unsupported_outcome {
            LogUnsupportedOutcome::ResidualBudgetExhausted {
                message: msg,
                solutions,
            } => {
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: residual_budget_exhausted_message(msg),
                        equation_after: Equation { lhs, rhs, op },
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
                return Ok((solutions, steps));
            }
            LogUnsupportedOutcome::Guarded {
                message: msg,
                missing_conditions,
                guard,
                residual,
            } => {
                // Register blocked hints for pedagogical feedback
                for condition in missing_conditions {
                    let event = crate::assumptions::AssumptionEvent::from_log_assumption(
                        *condition,
                        &simplifier.context,
                        b,
                        rhs,
                    );
                    let expr_id =
                        cas_solver_core::log_domain::assumption_target_expr(*condition, b, rhs);
                    crate::domain::register_blocked_hint(crate::domain::BlockedHint {
                        key: event.key,
                        expr_id,
                        rule: "Take log of both sides".to_string(),
                        suggestion: "use `semantics set domain assume`",
                    });
                }

                // Execute solver under guard
                let base_desc = format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: b
                    }
                );
                let log_step = build_pow_exponent_log_isolation_step_with(
                    &mut simplifier.context,
                    e,
                    b,
                    rhs,
                    op.clone(),
                    Some(msg),
                    |_| base_desc.clone(),
                );
                let new_eq = log_step.equation_after.clone();
                let new_rhs = new_eq.rhs;
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: log_step.description,
                        equation_after: log_step.equation_after,
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }

                let guarded_result = isolate(
                    new_eq.lhs,
                    new_rhs,
                    new_eq.op.clone(),
                    var,
                    simplifier,
                    opts,
                    ctx,
                );

                let guarded_solutions = match guarded_result {
                    Ok((guarded_solutions, _)) => {
                        if simplifier.collect_steps() {
                            steps.push(SolveStep {
                                description: conditional_solution_message(msg),
                                equation_after: Equation { lhs, rhs, op },
                                importance: crate::step::ImportanceLevel::Medium,
                                substeps: vec![],
                            });
                        }
                        Some(guarded_solutions)
                    }
                    Err(_) => {
                        if simplifier.collect_steps() {
                            steps.push(SolveStep {
                                description: residual_message(msg),
                                equation_after: Equation { lhs, rhs, op },
                                importance: crate::step::ImportanceLevel::Medium,
                                substeps: vec![],
                            });
                        }
                        None
                    }
                };

                return Ok((
                    guarded_or_residual(Some(guard), guarded_solutions, residual),
                    steps,
                ));
            }
        }
    }
    // ================================================================
    // End of domain guards
    // ================================================================

    let base_desc = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: b
        }
    );
    let log_step = build_pow_exponent_log_isolation_step_with(
        &mut simplifier.context,
        e,
        b,
        rhs,
        op,
        None,
        |_| base_desc.clone(),
    );
    let new_eq = log_step.equation_after.clone();
    if simplifier.collect_steps() {
        steps.push(SolveStep {
            description: log_step.description,
            equation_after: log_step.equation_after,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }
    let results = isolate(
        new_eq.lhs, new_eq.rhs, new_eq.op, var, simplifier, opts, ctx,
    )?;
    prepend_steps(results, steps)
}
