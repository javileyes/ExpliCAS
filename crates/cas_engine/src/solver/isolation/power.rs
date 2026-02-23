use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::is_numeric_zero;
use cas_solver_core::log_domain::{decision_assumptions, LogSolveDecision};
use cas_solver_core::solve_outcome::{
    build_pow_base_isolation_action_with, build_terminal_outcome_item, derive_pow_isolation_route,
    execute_pow_exponent_log_unsupported_with, execute_pow_exponent_shortcut_with_runtime,
    first_pow_base_isolation_execution_item, first_pow_exponent_log_isolation_execution_item,
    first_pow_exponent_shortcut_execution_item, first_power_base_one_shortcut_execution_item,
    first_term_isolation_rewrite_execution_item, plan_pow_exponent_log_isolation_step_with,
    plan_pow_exponent_log_unsupported_execution_from_decision_with,
    plan_solve_tactic_normalization_step, pow_exponent_rhs_contains_variable,
    resolve_log_terminal_outcome, resolve_power_base_one_shortcut_for_pow_with,
    solve_pow_base_isolation_action_with, solve_pow_exponent_log_isolation_rewrite_with,
    solve_pow_exponent_shortcut_action_with, PowBaseIsolationSolved,
    PowExponentLogUnsupportedSolvedExecution, PowExponentShortcutRuntime,
    PowExponentShortcutSolved, PowIsolationRoute,
};

use super::{isolate, prepend_steps};

struct EnginePowShortcutRuntime<'a> {
    simplifier: &'a mut Simplifier,
}

impl PowExponentShortcutRuntime for EnginePowShortcutRuntime<'_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn bases_equivalent(&mut self, base: ExprId, candidate: ExprId) -> bool {
        cas_solver_core::solve_outcome::shortcut_bases_equivalent_with(
            base,
            candidate,
            |left, right| {
                let diff = self.simplifier.context.add(Expr::Sub(left, right));
                let (sim_diff, _) = self.simplifier.simplify(diff);
                is_numeric_zero(&self.simplifier.context, sim_diff)
            },
        )
    }

    fn render_expr(&mut self, expr: ExprId) -> String {
        format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &self.simplifier.context,
                id: expr
            }
        )
    }
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
    mut steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let action =
        build_pow_base_isolation_action_with(&mut simplifier.context, b, e, rhs, op, |ctx, id| {
            format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
        });
    let execution_item = simplifier
        .collect_steps()
        .then(|| first_pow_base_isolation_execution_item(&action))
        .flatten();
    let solved_base =
        solve_pow_base_isolation_action_with(action, |lhs_after, target_rhs, target_op| {
            isolate(lhs_after, target_rhs, target_op, var, simplifier, opts, ctx)
        })?;

    if let Some(item) = execution_item {
        steps.push(SolveStep {
            description: item.description().to_string(),
            equation_after: item.equation,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    match solved_base {
        PowBaseIsolationSolved::ReturnedSolutionSet(solutions) => Ok((solutions, steps)),
        PowBaseIsolationSolved::Isolated(results) => prepend_steps(results, steps),
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
    let base_is_zero = is_numeric_zero(&simplifier.context, b);
    let base_is_numeric = matches!(simplifier.context.get(b), Expr::Number(_));
    let shortcut_engine_action = {
        let mut runtime = EnginePowShortcutRuntime { simplifier };
        execute_pow_exponent_shortcut_with_runtime(
            &mut runtime,
            e,
            b,
            rhs,
            op.clone(),
            var,
            base_is_zero,
            base_is_numeric,
            opts.budget.max_branches >= 2,
        )
    };
    let shortcut_item = simplifier
        .collect_steps()
        .then(|| first_pow_exponent_shortcut_execution_item(&shortcut_engine_action))
        .flatten();
    let shortcut_solved = solve_pow_exponent_shortcut_action_with(
        shortcut_engine_action,
        |target_rhs, target_op| isolate(e, target_rhs, target_op, var, simplifier, opts, ctx),
    )?;

    match shortcut_solved {
        PowExponentShortcutSolved::Continue => {}
        PowExponentShortcutSolved::Isolated(results) => {
            if let Some(item) = shortcut_item.as_ref() {
                steps.push(SolveStep {
                    description: item.description().to_string(),
                    equation_after: item.equation.clone(),
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            return prepend_steps(results, steps);
        }
        PowExponentShortcutSolved::ReturnedSolutionSet(solutions) => {
            if let Some(item) = shortcut_item.as_ref() {
                steps.push(SolveStep {
                    description: item.description().to_string(),
                    equation_after: item.equation.clone(),
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            return Ok((solutions, steps));
        }
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
    if let Some(outcome) = resolve_power_base_one_shortcut_for_pow_with(
        &simplifier.context,
        b,
        lhs,
        rhs,
        op.clone(),
        |ctx, id| format!("{}", cas_formatter::DisplayExpr { context: ctx, id }),
    ) {
        let execution_item = simplifier
            .collect_steps()
            .then(|| first_power_base_one_shortcut_execution_item(&outcome))
            .flatten();
        if let Some(item) = execution_item {
            steps.push(SolveStep {
                description: item.description().to_string(),
                equation_after: item.equation,
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
            let normalize_plan = plan_solve_tactic_normalization_step(
                &mut simplifier.context,
                sim_base,
                e,
                sim_rhs,
                op.clone(),
            );
            if let Some(item) = first_term_isolation_rewrite_execution_item(&normalize_plan) {
                steps.push(SolveStep {
                    description: item.description().to_string(),
                    equation_after: item.equation,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
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
            let item = build_terminal_outcome_item(
                &outcome,
                Equation {
                    lhs,
                    rhs,
                    op: op.clone(),
                },
                " (residual)",
            );
            steps.push(SolveStep {
                description: item.description().to_string(),
                equation_after: item.equation,
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

    let source_equation = Equation {
        lhs,
        rhs,
        op: op.clone(),
    };
    if let Some(unsupported_execution) =
        plan_pow_exponent_log_unsupported_execution_from_decision_with(
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
            source_equation,
            |core_ctx, id| {
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: core_ctx,
                        id
                    }
                )
            },
        )
    {
        let unsupported_solved =
            execute_pow_exponent_log_unsupported_with(unsupported_execution, |equation| {
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
            });
        match unsupported_solved {
            PowExponentLogUnsupportedSolvedExecution::Residual { item, solutions } => {
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: item.description().to_string(),
                        equation_after: item.equation,
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
                return Ok((solutions, steps));
            }
            PowExponentLogUnsupportedSolvedExecution::Guarded {
                blocked_hints,
                rewrite_item,
                followup_item,
                solutions,
            } => {
                // Register blocked hints for pedagogical feedback
                for hint in blocked_hints {
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
                }

                if simplifier.collect_steps() {
                    if let Some(item) = rewrite_item {
                        steps.push(SolveStep {
                            description: item.description().to_string(),
                            equation_after: item.equation,
                            importance: crate::step::ImportanceLevel::Medium,
                            substeps: vec![],
                        });
                    }
                    steps.push(SolveStep {
                        description: followup_item.description().to_string(),
                        equation_after: followup_item.equation,
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }

                return Ok((solutions, steps));
            }
        }
    }
    // ================================================================
    // End of domain guards
    // ================================================================

    let log_plan = plan_pow_exponent_log_isolation_step_with(
        &mut simplifier.context,
        e,
        b,
        rhs,
        op,
        None,
        |core_ctx, id| {
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: core_ctx,
                    id
                }
            )
        },
    );
    let solved_log = solve_pow_exponent_log_isolation_rewrite_with(log_plan, |equation| {
        isolate(
            equation.lhs,
            equation.rhs,
            equation.op.clone(),
            var,
            simplifier,
            opts,
            ctx,
        )
    })?;
    let log_item = simplifier
        .collect_steps()
        .then(|| first_pow_exponent_log_isolation_execution_item(&solved_log.rewrite))
        .flatten();
    if let Some(item) = log_item {
        steps.push(SolveStep {
            description: item.description().to_string(),
            equation_after: item.equation,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }
    let results = solved_log.solved;
    prepend_steps(results, steps)
}
