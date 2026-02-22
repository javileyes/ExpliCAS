use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::{
    apply_sign_flip, contains_var, is_even_integer_expr, is_known_negative, is_numeric_one,
    is_numeric_zero,
};
use cas_solver_core::log_domain::{
    classify_log_unsupported_route, decision_assumptions, LogSolveDecision, LogUnsupportedRoute,
};
use cas_solver_core::solve_outcome::{
    classify_pow_base_isolation_route, classify_pow_exponent_shortcut,
    classify_power_base_one_shortcut, detect_pow_exponent_shortcut_inputs,
    even_power_negative_rhs_outcome, guarded_or_residual, power_base_one_shortcut_solutions,
    power_equals_base_symbolic_outcome, residual_expression, resolve_log_terminal_outcome,
    terminal_outcome_message, PowBaseIsolationRoute, PowExponentShortcut, PowerBaseOneShortcut,
    PowerEqualsBaseRoute,
};

use super::{isolate, prepend_steps};

fn are_shortcut_bases_equivalent(
    simplifier: &mut Simplifier,
    base: ExprId,
    candidate: ExprId,
) -> bool {
    if base == candidate {
        true
    } else {
        let diff = simplifier.context.add(Expr::Sub(base, candidate));
        let (sim_diff, _) = simplifier.simplify(diff);
        is_numeric_zero(&simplifier.context, sim_diff)
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
    let route = classify_pow_base_isolation_route(
        is_even_integer_expr(&simplifier.context, e),
        is_known_negative(&simplifier.context, rhs),
        is_known_negative(&simplifier.context, e),
    );

    match route {
        PowBaseIsolationRoute::EvenExponentNegativeRhsImpossible => {
            let result = even_power_negative_rhs_outcome(op.clone());
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: format!(
                        "Even power cannot be negative ({} {} {})",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: b
                        },
                        op,
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: rhs
                        }
                    ),
                    equation_after: Equation {
                        lhs: simplifier.context.add(Expr::Pow(b, e)),
                        rhs,
                        op,
                    },
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            Ok((result, steps))
        }
        PowBaseIsolationRoute::EvenExponentUseAbsRoot
        | PowBaseIsolationRoute::GeneralRoot { .. } => {
            let use_abs = matches!(route, PowBaseIsolationRoute::EvenExponentUseAbsRoot);
            let new_eq = cas_solver_core::rational_power::build_root_isolation_equation(
                &mut simplifier.context,
                b,
                e,
                rhs,
                op.clone(),
                use_abs,
            );
            let new_rhs = new_eq.rhs;
            if simplifier.collect_steps() {
                let description = if use_abs {
                    format!(
                        "Take {}-th root of both sides (even root implies absolute value)",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: e
                        }
                    )
                } else {
                    format!(
                        "Take {}-th root of both sides",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: e
                        }
                    )
                };
                steps.push(SolveStep {
                    description,
                    equation_after: new_eq.clone(),
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            let new_op = match route {
                PowBaseIsolationRoute::GeneralRoot {
                    flip_inequality_for_negative_exponent,
                } => apply_sign_flip(op, flip_inequality_for_negative_exponent),
                _ => op,
            };
            let results = isolate(new_eq.lhs, new_rhs, new_op, var, simplifier, opts, ctx)?;
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
            are_shortcut_bases_equivalent(simplifier, b, candidate)
        });

    let shortcut = classify_pow_exponent_shortcut(
        op.clone(),
        bases_equal,
        rhs_pow_base_equal,
        is_numeric_zero(&simplifier.context, b),
        matches!(simplifier.context.get(b), Expr::Number(_)),
        opts.budget.max_branches >= 2,
    );

    match shortcut {
        PowExponentShortcut::PowerEqualsBase(route) => match route {
            PowerEqualsBaseRoute::ExponentGreaterThanZero => {
                let zero = simplifier.context.num(0);
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: format!(
                            "Power Equals Base Shortcut: 0^{} = 0 ⟹ {} > 0 (0^0 undefined, 0^t for t<0 undefined)",
                            var, var
                        ),
                        equation_after: Equation {
                            lhs: e,
                            rhs: zero,
                            op: RelOp::Gt,
                        },
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
                let results = isolate(e, zero, RelOp::Gt, var, simplifier, opts, ctx)?;
                return prepend_steps(results, steps);
            }
            PowerEqualsBaseRoute::ExponentEqualsOneNumericBase
            | PowerEqualsBaseRoute::ExponentEqualsOneNoBranchBudget => {
                let one = simplifier.context.num(1);
                if simplifier.collect_steps() {
                    let description = match route {
                        PowerEqualsBaseRoute::ExponentEqualsOneNumericBase => format!(
                            "Power Equals Base Shortcut: {}^{} = {} ⟹ {} = 1 (B^1 = B always holds)",
                            cas_formatter::DisplayExpr {
                                context: &simplifier.context,
                                id: b
                            },
                            var,
                            cas_formatter::DisplayExpr {
                                context: &simplifier.context,
                                id: rhs
                            },
                            var
                        ),
                        PowerEqualsBaseRoute::ExponentEqualsOneNoBranchBudget => format!(
                            "Power Equals Base: {}^{} = {} ⟹ {} = 1 (assuming base ≠ 0, 1)",
                            cas_formatter::DisplayExpr {
                                context: &simplifier.context,
                                id: b
                            },
                            var,
                            cas_formatter::DisplayExpr {
                                context: &simplifier.context,
                                id: rhs
                            },
                            var
                        ),
                        _ => unreachable!("route match is exhaustive above"),
                    };
                    steps.push(SolveStep {
                        description,
                        equation_after: Equation {
                            lhs: e,
                            rhs: one,
                            op: op.clone(),
                        },
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
                let results = isolate(e, one, op, var, simplifier, opts, ctx)?;
                return prepend_steps(results, steps);
            }
            PowerEqualsBaseRoute::SymbolicCaseSplit => {
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: format!(
                            "Power Equals Base with symbolic base '{}': case split → a=1: AllReals, a=0: x>0, otherwise: x=1",
                            cas_formatter::DisplayExpr {
                                context: &simplifier.context,
                                id: b
                            }
                        ),
                        equation_after: Equation {
                            lhs: e,
                            rhs: b,
                            op: op.clone(),
                        },
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
                let conditional = power_equals_base_symbolic_outcome(&mut simplifier.context, b);
                return Ok((conditional, steps));
            }
        },
        PowExponentShortcut::EqualPowBases { rhs_exp } => {
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: format!(
                        "Pattern: {}^{} = {}^{} → {} = {} (equal bases imply equal exponents when base ≠ 0, 1)",
                        cas_formatter::DisplayExpr { context: &simplifier.context, id: b },
                        var,
                        cas_formatter::DisplayExpr { context: &simplifier.context, id: b },
                        cas_formatter::DisplayExpr { context: &simplifier.context, id: rhs_exp },
                        var,
                        cas_formatter::DisplayExpr { context: &simplifier.context, id: rhs_exp }
                    ),
                    equation_after: Equation {
                        lhs: e,
                        rhs: rhs_exp,
                        op: op.clone(),
                    },
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            let results = isolate(e, rhs_exp, op, var, simplifier, opts, ctx)?;
            return prepend_steps(results, steps);
        }
        PowExponentShortcut::None => {}
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
    let base_one_shortcut = classify_power_base_one_shortcut(
        is_numeric_one(&simplifier.context, b),
        is_numeric_one(&simplifier.context, rhs),
    );
    if let Some(result) = power_base_one_shortcut_solutions(base_one_shortcut) {
        if simplifier.collect_steps() {
            let desc = match base_one_shortcut {
                PowerBaseOneShortcut::AllReals => {
                    "1^x = 1 for all x → any real number is a solution".to_string()
                }
                PowerBaseOneShortcut::Empty => format!(
                    "1^x = 1 for all x, but RHS = {} ≠ 1 → no solution",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: rhs
                    }
                ),
                PowerBaseOneShortcut::NotApplicable => unreachable!("shortcut applied above"),
            };
            steps.push(SolveStep {
                description: desc,
                equation_after: Equation { lhs, rhs, op },
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        return Ok((result, steps));
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
            steps.push(SolveStep {
                description:
                    "Applied SolveTactic normalization (Assume mode) to enable logarithm isolation"
                        .to_string(),
                equation_after: Equation {
                    lhs: simplifier.context.add(Expr::Pow(sim_base, e)),
                    rhs: sim_rhs,
                    op: op.clone(),
                },
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

    match classify_log_unsupported_route(&decision, opts.budget.can_branch()) {
        LogUnsupportedRoute::NotUnsupported => {}
        LogUnsupportedRoute::ResidualBudgetExhausted { message: msg } => {
            let residual = residual_expression(&mut simplifier.context, lhs, rhs, var);
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: format!("{} (residual, budget exhausted)", msg),
                    equation_after: Equation { lhs, rhs, op },
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            return Ok((guarded_or_residual(None, None, residual), steps));
        }
        LogUnsupportedRoute::Guarded {
            message: msg,
            missing_conditions,
        } => {
            let residual = residual_expression(&mut simplifier.context, lhs, rhs, var);
            // Build guard set from missing conditions
            let guard = cas_solver_core::log_domain::assumptions_to_condition_set(
                missing_conditions,
                b,
                rhs,
            );

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
            let new_eq = cas_solver_core::rational_power::build_exponent_log_isolation_equation(
                &mut simplifier.context,
                e,
                b,
                rhs,
                op.clone(),
            );
            let new_rhs = new_eq.rhs;
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: format!(
                        "Take log base {} of both sides (under guard: {})",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: b
                        },
                        msg
                    ),
                    equation_after: new_eq.clone(),
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
                            description: format!("Conditional solution: {}", msg),
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
                            description: format!("{} (residual)", msg),
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
    // ================================================================
    // End of domain guards
    // ================================================================

    let new_eq = cas_solver_core::rational_power::build_exponent_log_isolation_equation(
        &mut simplifier.context,
        e,
        b,
        rhs,
        op,
    );
    if simplifier.collect_steps() {
        steps.push(SolveStep {
            description: format!(
                "Take log base {} of both sides",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: b
                }
            ),
            equation_after: new_eq.clone(),
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }
    let results = isolate(
        new_eq.lhs, new_eq.rhs, new_eq.op, var, simplifier, opts, ctx,
    )?;
    prepend_steps(results, steps)
}
