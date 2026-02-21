use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::{
    contains_var, flip_inequality, is_known_negative, is_numeric_zero, mk_residual_solve,
};
use cas_solver_core::log_domain::{LogAssumption, LogSolveDecision};
use cas_solver_core::solution_set::get_number;
use cas_solver_core::solve_outcome::even_power_negative_rhs_outcome;

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
    // Check if exponent is an even integer
    let is_even = if let Some(n) = get_number(&simplifier.context, e) {
        n.is_integer() && (n.to_integer() % 2 == 0.into())
    } else {
        false
    };

    if is_even {
        // Check if RHS is negative
        if is_known_negative(&simplifier.context, rhs) {
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
            return Ok((result, steps));
        }

        // B^E = RHS -> |B| = RHS^(1/E)
        let one = simplifier.context.num(1);
        let inv_exp = simplifier.context.add(Expr::Div(one, e));
        let new_rhs = simplifier.context.add(Expr::Pow(rhs, inv_exp));

        // Construct |B|
        let abs_b = simplifier.context.call("abs", vec![b]);

        let new_eq = Equation {
            lhs: abs_b,
            rhs: new_rhs,
            op: op.clone(),
        };
        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: format!(
                    "Take {}-th root of both sides (even root implies absolute value)",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: e
                    }
                ),
                equation_after: new_eq,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }

        let results = isolate(abs_b, new_rhs, op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
    } else {
        // B = RHS^(1/E)
        let one = simplifier.context.num(1);
        let inv_exp = simplifier.context.add(Expr::Div(one, e));
        let new_rhs = simplifier.context.add(Expr::Pow(rhs, inv_exp));
        let new_eq = Equation {
            lhs: b,
            rhs: new_rhs,
            op: op.clone(),
        };
        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: format!(
                    "Take {}-th root of both sides",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: e
                    }
                ),
                equation_after: new_eq,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }

        // Check if exponent is negative to flip inequality
        let mut new_op = op;
        if is_known_negative(&simplifier.context, e) {
            new_op = flip_inequality(new_op);
        }

        let results = isolate(b, new_rhs, new_op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
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
    let bases_equal = {
        if b == rhs {
            true
        } else {
            let diff = simplifier.context.add(Expr::Sub(b, rhs));
            let (sim_diff, _) = simplifier.simplify(diff);
            is_numeric_zero(&simplifier.context, sim_diff)
        }
    };

    if bases_equal && op == RelOp::Eq {
        let base_is_zero = is_numeric_zero(&simplifier.context, b);

        if base_is_zero {
            // 0^x = 0 ⟹ x > 0
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
        } else {
            // base^x = base ⟹ x = 1 (when base ≠ 0, ≠ 1)
            let base_is_numeric = matches!(simplifier.context.get(b), Expr::Number(_));
            let one = simplifier.context.num(1);
            let zero = simplifier.context.num(0);

            if base_is_numeric {
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: format!(
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

            // ================================================================
            // V2.0 Phase 2B: Symbolic base → Case splits
            // ================================================================
            if opts.budget.max_branches < 2 {
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: format!(
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

            // Build the three cases
            use cas_ast::{
                BoundType, Case, ConditionPredicate, ConditionSet, Interval, SolveResult,
            };

            // Case 1: a = 1 → AllReals
            let case_one_guard = ConditionSet::single(ConditionPredicate::EqOne(b));
            let case_one_result = SolveResult::solved(SolutionSet::AllReals);
            let case_one = Case::with_result(case_one_guard, case_one_result);

            // Case 2: a = 0 → x > 0
            let case_zero_guard = ConditionSet::single(ConditionPredicate::EqZero(b));
            let pos_infinity = simplifier.context.var("infinity");
            let interval_x_positive = Interval {
                min: zero,
                min_type: BoundType::Open,
                max: pos_infinity,
                max_type: BoundType::Open,
            };
            let case_zero_result =
                SolveResult::solved(SolutionSet::Continuous(interval_x_positive));
            let case_zero = Case::with_result(case_zero_guard, case_zero_result);

            // Case 3: otherwise → x = 1
            let case_default_guard = ConditionSet::empty();
            let case_default_result = SolveResult::solved(SolutionSet::Discrete(vec![one]));
            let case_default = Case::with_result(case_default_guard, case_default_result);

            // Pedagogical step
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

            let conditional = SolutionSet::Conditional(vec![case_one, case_zero, case_default]);
            return Ok((conditional, steps));
        }
    }

    // ================================================================
    // SPECIAL CASE: base^x = base^n → x = n
    // ================================================================
    if let Expr::Pow(rhs_base, rhs_exp) = simplifier.context.get(rhs).clone() {
        let pow_bases_equal = {
            if b == rhs_base {
                true
            } else {
                let diff = simplifier.context.add(Expr::Sub(b, rhs_base));
                let (sim_diff, _) = simplifier.simplify(diff);
                is_numeric_zero(&simplifier.context, sim_diff)
            }
        };

        if pow_bases_equal {
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: format!(
                        "Pattern: {}^{} = {}^{} → {} = {} (equal bases imply equal exponents when base ≠ 0, 1)",
                        cas_formatter::DisplayExpr { context: &simplifier.context, id: b },
                        var,
                        cas_formatter::DisplayExpr { context: &simplifier.context, id: rhs_base },
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
    use cas_math::expr_predicates::is_one_expr as is_one;

    // GUARD 1: Handle base = 1 special case
    if is_one(&simplifier.context, b) {
        let result = if is_one(&simplifier.context, rhs) {
            SolutionSet::AllReals
        } else {
            SolutionSet::Empty
        };
        if simplifier.collect_steps() {
            let desc = if is_one(&simplifier.context, rhs) {
                "1^x = 1 for all x → any real number is a solution".to_string()
            } else {
                format!(
                    "1^x = 1 for all x, but RHS = {} ≠ 1 → no solution",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: rhs
                    }
                )
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

    match decision {
        LogSolveDecision::Ok => {
            // Base>0 and RHS>0 proven - safe to proceed
        }
        LogSolveDecision::OkWithAssumptions(assumptions) => {
            for assumption in assumptions {
                let event = crate::assumptions::AssumptionEvent::from_log_assumption(
                    assumption,
                    &simplifier.context,
                    b,
                    rhs,
                );
                crate::solver::note_assumption(event);
            }
        }
        LogSolveDecision::EmptySet(msg) => {
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: msg.to_string(),
                    equation_after: Equation { lhs, rhs, op },
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            return Ok((SolutionSet::Empty, steps));
        }
        LogSolveDecision::NeedsComplex(msg) => {
            use crate::semantics::AssumeScope;
            if opts.domain_mode == crate::domain::DomainMode::Assume
                && opts.assume_scope == AssumeScope::Wildcard
            {
                let residual = mk_residual_solve(&mut simplifier.context, lhs, rhs, var);
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: format!("{} (residual)", msg),
                        equation_after: Equation { lhs, rhs, op },
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
                return Ok((SolutionSet::Residual(residual), steps));
            }
            return Err(CasError::UnsupportedInRealDomain(msg.to_string()));
        }
        LogSolveDecision::Unsupported(msg, missing_conditions) => {
            if !opts.budget.can_branch() {
                let residual = mk_residual_solve(&mut simplifier.context, lhs, rhs, var);
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: format!("{} (residual, budget exhausted)", msg),
                        equation_after: Equation { lhs, rhs, op },
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
                return Ok((SolutionSet::Residual(residual), steps));
            }

            // Build guard set from missing conditions
            let guard = cas_solver_core::log_domain::assumptions_to_condition_set(
                &missing_conditions,
                b,
                rhs,
            );

            // Register blocked hints for pedagogical feedback
            for condition in &missing_conditions {
                let event = crate::assumptions::AssumptionEvent::from_log_assumption(
                    *condition,
                    &simplifier.context,
                    b,
                    rhs,
                );
                let expr_id = match condition {
                    LogAssumption::PositiveBase => b,
                    LogAssumption::PositiveRhs => rhs,
                };
                crate::domain::register_blocked_hint(crate::domain::BlockedHint {
                    key: event.key,
                    expr_id,
                    rule: "Take log of both sides".to_string(),
                    suggestion: "use `semantics set domain assume`",
                });
            }

            // Execute solver under guard
            let new_rhs = simplifier.context.call("log", vec![b, rhs]);
            let new_eq = Equation {
                lhs: e,
                rhs: new_rhs,
                op: op.clone(),
            };

            let mut guarded_steps = steps.clone();
            if simplifier.collect_steps() {
                guarded_steps.push(SolveStep {
                    description: format!(
                        "Take log base {} of both sides (under guard: {})",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: b
                        },
                        msg
                    ),
                    equation_after: new_eq,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            let guarded_result = isolate(e, new_rhs, op.clone(), var, simplifier, opts, ctx);

            let residual = mk_residual_solve(&mut simplifier.context, lhs, rhs, var);

            match guarded_result {
                Ok((guarded_solutions, mut solve_steps)) => {
                    guarded_steps.append(&mut solve_steps);

                    let cases = vec![
                        cas_ast::Case::new(guard, guarded_solutions),
                        cas_ast::Case::new(
                            cas_ast::ConditionSet::empty(),
                            SolutionSet::Residual(residual),
                        ),
                    ];

                    if simplifier.collect_steps() {
                        steps.push(SolveStep {
                            description: format!("Conditional solution: {}", msg),
                            equation_after: Equation { lhs, rhs, op },
                            importance: crate::step::ImportanceLevel::Medium,
                            substeps: vec![],
                        });
                    }

                    return Ok((SolutionSet::Conditional(cases), steps));
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
                    return Ok((SolutionSet::Residual(residual), steps));
                }
            }
        }
    }
    // ================================================================
    // End of domain guards
    // ================================================================

    let new_rhs = simplifier.context.call("log", vec![b, rhs]);
    let new_eq = Equation {
        lhs: e,
        rhs: new_rhs,
        op: op.clone(),
    };
    if simplifier.collect_steps() {
        steps.push(SolveStep {
            description: format!(
                "Take log base {} of both sides",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: b
                }
            ),
            equation_after: new_eq,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }
    let results = isolate(e, new_rhs, op, var, simplifier, opts, ctx)?;
    prepend_steps(results, steps)
}
