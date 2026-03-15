use cas_ast::{
    BoundType, Case, ConditionPredicate, ConditionSet, Constant, Context, Equation, Expr, RelOp,
    SolutionSet,
};
use cas_formatter::DisplayExpr;
use cas_math::tri_proof::TriProof;
use cas_parser::parse;
use cas_solver::api::{solve, verify_solution_set, Proof as EngineProof, VerifySummary};
use cas_solver::command_api::solve::{evaluate_solve_command_lines_with_session, SolveDisplayMode};
use cas_solver::runtime::{
    DomainMode, EvalOptions, Simplifier, SolveDomainEnv, SolverOptions, StatelessEvalSession,
    StepsMode, ValueDomain,
};

// Helper to make equation from strings
fn make_eq(ctx: &mut Context, lhs: &str, rhs: &str) -> Equation {
    Equation {
        lhs: parse(lhs, ctx).unwrap(),
        rhs: parse(rhs, ctx).unwrap(),
        op: RelOp::Eq,
    }
}

#[test]
fn test_solve_linear() {
    // x + 2 = 5 -> x = 3
    let mut simplifier = Simplifier::new();
    let eq = make_eq(&mut simplifier.context, "x + 2", "5");
    simplifier.set_collect_steps(true);
    let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

    if let SolutionSet::Discrete(solutions) = result {
        assert_eq!(solutions.len(), 1);
        let s = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: solutions[0]
            }
        );
        assert_eq!(s, "3");
    } else {
        panic!("Expected Discrete solution");
    }
}

#[test]
fn test_solve_mul() {
    // 2 * x = 6 -> x = 6 / 2
    let mut simplifier = Simplifier::with_default_rules();
    let eq = make_eq(&mut simplifier.context, "2 * x", "6");
    simplifier.set_collect_steps(true);
    let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

    if let SolutionSet::Discrete(solutions) = result {
        assert_eq!(solutions.len(), 1);
        let s = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: solutions[0]
            }
        );
        assert_eq!(s, "3");
    } else {
        panic!("Expected Discrete solution");
    }
}

#[test]
fn test_solve_pow() {
    // x^2 = 4 -> x = 4^(1/2)
    let mut simplifier = Simplifier::new();
    let eq = make_eq(&mut simplifier.context, "x^2", "4");
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::exponents::EvaluatePowerRule,
    ));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::canonicalization::CanonicalizeNegationRule,
    ));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::arithmetic::CombineConstantsRule,
    ));
    simplifier.set_collect_steps(true);
    let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

    if let SolutionSet::Discrete(mut solutions) = result {
        assert_eq!(solutions.len(), 2);
        // Sort to ensure order
        solutions.sort_by(|a, b| {
            let sa = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: *a
                }
            );
            let sb = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: *b
                }
            );
            sa.cmp(&sb)
        });

        let s1 = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: solutions[0]
            }
        );
        let s2 = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: solutions[1]
            }
        );

        // We want to eventually see "-2" and "2".
        assert_eq!(s1, "-2");
        assert_eq!(s2, "2");
    } else {
        panic!("Expected Discrete solution");
    }
}

#[test]
fn test_solve_abs() {
    // |x| = 5 -> x=5, x=-5
    let mut simplifier = Simplifier::new();
    let eq = make_eq(&mut simplifier.context, "|x|", "5");
    let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

    if let SolutionSet::Discrete(solutions) = result {
        assert_eq!(solutions.len(), 2);
        // Order might vary
        let s: Vec<String> = solutions
            .iter()
            .map(|e| {
                format!(
                    "{}",
                    DisplayExpr {
                        context: &simplifier.context,
                        id: *e
                    }
                )
            })
            .collect();
        assert!(s.contains(&"5".to_string()));
        assert!(s.contains(&"-5".to_string()));
    } else {
        panic!("Expected Discrete solution");
    }
}

#[test]
fn test_solve_inequality_flip() {
    // -2x < 10 -> x > -5
    let mut simplifier = Simplifier::with_default_rules();
    let eq = Equation {
        lhs: parse("-2*x", &mut simplifier.context).unwrap(),
        rhs: parse("10", &mut simplifier.context).unwrap(),
        op: RelOp::Lt,
    };
    let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

    if let SolutionSet::Continuous(interval) = result {
        // (-5, inf)
        let s_min = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: interval.min
            }
        );
        let s_max = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: interval.max
            }
        );
        assert!(
            s_min == "-5" || s_min == "10 / -2",
            "Expected -5 or canonical form 10 / -2, got: {}",
            s_min
        );
        assert_eq!(interval.min_type, BoundType::Open);
        assert_eq!(s_max, "infinity");
    } else {
        panic!("Expected Continuous solution, got {:?}", result);
    }
}

#[test]
fn test_solve_abs_inequality() {
    // |x| < 5 -> (-5, 5)
    let mut simplifier = Simplifier::new();
    let eq = Equation {
        lhs: parse("|x|", &mut simplifier.context).unwrap(),
        rhs: parse("5", &mut simplifier.context).unwrap(),
        op: RelOp::Lt,
    };
    let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

    if let SolutionSet::Continuous(interval) = result {
        let s_min = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: interval.min
            }
        );
        let s_max = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: interval.max
            }
        );
        assert_eq!(s_min, "-5");
        assert_eq!(s_max, "5");
    } else {
        panic!("Expected Continuous solution, got {:?}", result);
    }
}

#[test]
fn test_verify_solution_set_restores_exact_steps_mode_after_hidden_simplify() {
    let mut simplifier = Simplifier::with_default_rules();
    let eq = make_eq(&mut simplifier.context, "x^2 - 5*x + 6", "0");
    let solutions = SolutionSet::Discrete(vec![
        parse("2", &mut simplifier.context).unwrap(),
        parse("3", &mut simplifier.context).unwrap(),
    ]);

    simplifier.set_steps_mode(StepsMode::Compact);
    let result = verify_solution_set(&mut simplifier, &eq, "x", &solutions);

    assert!(matches!(result.summary, VerifySummary::AllVerified));
    assert_eq!(simplifier.get_steps_mode(), StepsMode::Compact);
}

#[test]
fn test_verify_solution_set_non_discrete_positive_interval_needs_sampling_with_guard_hint() {
    let mut simplifier = Simplifier::with_default_rules();
    let eq = make_eq(&mut simplifier.context, "ln(x)", "ln(x)");
    let zero = simplifier.context.num(0);
    let inf = simplifier.context.add(Expr::Constant(Constant::Infinity));
    let solutions = SolutionSet::Continuous(cas_ast::Interval {
        min: zero,
        min_type: BoundType::Open,
        max: inf,
        max_type: BoundType::Open,
    });

    let result = verify_solution_set(&mut simplifier, &eq, "x", &solutions);

    assert!(matches!(result.summary, VerifySummary::NeedsSampling));
    assert_eq!(
        result.guard_description.as_deref(),
        Some("verification requires numeric sampling (solution set matches guard `x > 0`)")
    );
}

#[test]
fn test_verify_solution_set_negative_interval_needs_sampling_with_guard_hint() {
    let mut simplifier = Simplifier::with_default_rules();
    let eq = make_eq(&mut simplifier.context, "x", "x");
    let zero = simplifier.context.num(0);
    let inf = simplifier.context.add(Expr::Constant(Constant::Infinity));
    let neg_inf = simplifier.context.add(Expr::Neg(inf));
    let solutions = SolutionSet::Continuous(cas_ast::Interval {
        min: neg_inf,
        min_type: BoundType::Open,
        max: zero,
        max_type: BoundType::Open,
    });

    let result = verify_solution_set(&mut simplifier, &eq, "x", &solutions);

    assert!(matches!(result.summary, VerifySummary::NeedsSampling));
    assert_eq!(
        result.guard_description.as_deref(),
        Some("verification requires numeric sampling (solution set matches guard `x < 0`)")
    );
}

#[test]
fn test_verify_solution_set_nonpositive_interval_needs_sampling_with_guard_hint() {
    let mut simplifier = Simplifier::with_default_rules();
    let eq = make_eq(&mut simplifier.context, "x", "x");
    let zero = simplifier.context.num(0);
    let inf = simplifier.context.add(Expr::Constant(Constant::Infinity));
    let neg_inf = simplifier.context.add(Expr::Neg(inf));
    let solutions = SolutionSet::Continuous(cas_ast::Interval {
        min: neg_inf,
        min_type: BoundType::Open,
        max: zero,
        max_type: BoundType::Closed,
    });

    let result = verify_solution_set(&mut simplifier, &eq, "x", &solutions);

    assert!(matches!(result.summary, VerifySummary::NeedsSampling));
    assert_eq!(
        result.guard_description.as_deref(),
        Some("verification requires numeric sampling (solution set matches guard `x <= 0`)")
    );
}

#[test]
fn test_verify_solution_set_nonzero_union_needs_sampling_with_guard_hint() {
    let mut simplifier = Simplifier::with_default_rules();
    let eq = make_eq(&mut simplifier.context, "x/x", "1");
    let zero = simplifier.context.num(0);
    let inf = simplifier.context.add(Expr::Constant(Constant::Infinity));
    let neg_inf = simplifier.context.add(Expr::Neg(inf));
    let solutions = SolutionSet::Union(vec![
        cas_ast::Interval {
            min: neg_inf,
            min_type: BoundType::Open,
            max: zero,
            max_type: BoundType::Open,
        },
        cas_ast::Interval {
            min: zero,
            min_type: BoundType::Open,
            max: inf,
            max_type: BoundType::Open,
        },
    ]);

    let result = verify_solution_set(&mut simplifier, &eq, "x", &solutions);

    assert!(matches!(result.summary, VerifySummary::NeedsSampling));
    assert_eq!(
        result.guard_description.as_deref(),
        Some("verification requires numeric sampling (solution set matches guard `x != 0`)")
    );
}

#[test]
fn test_verify_solution_set_guarded_non_discrete_conditional_is_verified_under_guard() {
    let mut simplifier = Simplifier::with_default_rules();
    let eq = make_eq(&mut simplifier.context, "x/x", "1");
    let x = parse("x", &mut simplifier.context).unwrap();
    let solutions = SolutionSet::Conditional(vec![
        Case::new(
            ConditionSet::single(ConditionPredicate::NonZero(x)),
            SolutionSet::AllReals,
        ),
        Case::new(ConditionSet::empty(), SolutionSet::Empty),
    ]);

    let result = verify_solution_set(&mut simplifier, &eq, "x", &solutions);

    assert!(matches!(result.summary, VerifySummary::VerifiedUnderGuard));
    assert_eq!(
        result.guard_description.as_deref(),
        Some("verified symbolically under guard (1 guarded non-discrete branch)")
    );
}

#[test]
fn test_verify_solution_set_plain_all_reals_stays_not_checkable() {
    let mut simplifier = Simplifier::with_default_rules();
    let eq = make_eq(&mut simplifier.context, "x", "x");
    let solutions = SolutionSet::AllReals;

    let result = verify_solution_set(&mut simplifier, &eq, "x", &solutions);

    assert!(matches!(result.summary, VerifySummary::NotCheckable));
    assert_eq!(
        result.guard_description.as_deref(),
        Some("not checkable (infinite set: all reals)")
    );
}

#[test]
fn test_verify_solution_set_failed_discrete_solution_can_surface_counterexample_hint() {
    let mut simplifier = Simplifier::with_default_rules();
    let eq = make_eq(&mut simplifier.context, "a*x", "1");
    let wrong_solution = parse("1", &mut simplifier.context).unwrap();
    let solutions = SolutionSet::Discrete(vec![wrong_solution]);

    let result = verify_solution_set(&mut simplifier, &eq, "x", &solutions);

    assert!(matches!(result.summary, VerifySummary::NoneVerified));
    match &result.solutions[0].1 {
        cas_solver::api::VerifyStatus::Unverifiable {
            counterexample_hint,
            ..
        } => {
            assert_eq!(
                counterexample_hint.as_deref(),
                Some("counterexample hint: a=0 gives residual -1")
            );
        }
        other => panic!("expected unverifiable status, got {other:?}"),
    }
}

#[test]
fn test_verify_solution_set_suppresses_counterexample_hint_for_log_sensitive_residual() {
    let mut simplifier = Simplifier::with_default_rules();
    let eq = make_eq(&mut simplifier.context, "ln(a*x)", "1");
    let wrong_solution = parse("1", &mut simplifier.context).unwrap();
    let solutions = SolutionSet::Discrete(vec![wrong_solution]);

    let result = verify_solution_set(&mut simplifier, &eq, "x", &solutions);

    assert!(matches!(result.summary, VerifySummary::NoneVerified));
    match &result.solutions[0].1 {
        cas_solver::api::VerifyStatus::Unverifiable {
            counterexample_hint,
            ..
        } => {
            assert_eq!(counterexample_hint, &None);
        }
        other => panic!("expected unverifiable status, got {other:?}"),
    }
}

#[test]
fn test_solve_check_session_render_surfaces_verified_under_guard_line() {
    let mut simplifier = Simplifier::with_default_rules();
    let mut eval_options = EvalOptions::solve();
    eval_options.budget.max_branches = 2;
    eval_options.shared.semantics.domain_mode = DomainMode::Strict;
    let mut session = StatelessEvalSession::new(eval_options.clone());

    let lines = evaluate_solve_command_lines_with_session(
        &mut simplifier,
        &mut session,
        "solve --check a^x = a, x",
        &eval_options,
        SolveDisplayMode::None,
        false,
    )
    .expect("solve command should render");

    assert!(
        lines.iter().any(|line| {
            line.contains("some non-discrete branches require numeric sampling")
                || line.contains("remain not checkable")
        }),
        "lines: {:?}",
        lines
    );
}

#[test]
fn test_solve_check_session_render_surfaces_all_verified_summary() {
    let mut simplifier = Simplifier::with_default_rules();
    let eval_options = EvalOptions::solve();
    let mut session = StatelessEvalSession::new(eval_options.clone());

    let lines = evaluate_solve_command_lines_with_session(
        &mut simplifier,
        &mut session,
        "solve --check x^2 = 1, x",
        &eval_options,
        SolveDisplayMode::None,
        false,
    )
    .expect("solve command should render");

    assert!(
        lines
            .iter()
            .any(|line| { line.contains("✓ All solutions verified") }),
        "lines: {:?}",
        lines
    );
}

#[test]
fn test_solve_check_session_render_surfaces_all_reals_not_checkable_note() {
    let mut simplifier = Simplifier::with_default_rules();
    let eval_options = EvalOptions::solve();
    let mut session = StatelessEvalSession::new(eval_options.clone());

    let lines = evaluate_solve_command_lines_with_session(
        &mut simplifier,
        &mut session,
        "solve --check x = x, x",
        &eval_options,
        SolveDisplayMode::None,
        false,
    )
    .expect("solve command should render");

    assert!(
        lines
            .iter()
            .any(|line| { line.contains("not checkable (infinite set: all reals)") }),
        "lines: {:?}",
        lines
    );
}

#[test]
fn test_solve_check_session_render_surfaces_needs_sampling_guard_hint_for_interval() {
    let mut simplifier = Simplifier::with_default_rules();
    let eval_options = EvalOptions::solve();
    let mut session = StatelessEvalSession::new(eval_options.clone());

    let lines = evaluate_solve_command_lines_with_session(
        &mut simplifier,
        &mut session,
        "solve --check 0^x = 0, x",
        &eval_options,
        SolveDisplayMode::None,
        false,
    )
    .expect("solve command should render");

    assert!(
        lines.iter().any(|line| {
            line.contains("verification requires numeric sampling") && line.contains("x > 0")
        }),
        "lines: {:?}",
        lines
    );
}

#[test]
fn test_solve_check_session_render_surfaces_needs_sampling_guard_hint_for_union() {
    let mut simplifier = Simplifier::with_default_rules();
    let eval_options = EvalOptions::solve();
    let mut session = StatelessEvalSession::new(eval_options.clone());

    let lines = evaluate_solve_command_lines_with_session(
        &mut simplifier,
        &mut session,
        "solve --check x != 0, x",
        &eval_options,
        SolveDisplayMode::None,
        false,
    )
    .expect("solve command should render");

    assert!(
        lines.iter().any(|line| {
            line.contains("verification requires numeric sampling") && line.contains("x != 0")
        }),
        "lines: {:?}",
        lines
    );
}

#[test]
fn test_solve_check_session_render_surfaces_needs_sampling_guard_hint_for_negative_interval() {
    let mut simplifier = Simplifier::with_default_rules();
    let eval_options = EvalOptions::solve();
    let mut session = StatelessEvalSession::new(eval_options.clone());

    let lines = evaluate_solve_command_lines_with_session(
        &mut simplifier,
        &mut session,
        "solve --check x < 0, x",
        &eval_options,
        SolveDisplayMode::None,
        false,
    )
    .expect("solve command should render");

    assert!(
        lines.iter().any(|line| {
            line.contains("verification requires numeric sampling") && line.contains("x < 0")
        }),
        "lines: {:?}",
        lines
    );
}

#[test]
fn test_solve_check_session_render_surfaces_needs_sampling_guard_hint_for_nonpositive_interval() {
    let mut simplifier = Simplifier::with_default_rules();
    let eval_options = EvalOptions::solve();
    let mut session = StatelessEvalSession::new(eval_options.clone());

    let lines = evaluate_solve_command_lines_with_session(
        &mut simplifier,
        &mut session,
        "solve --check x <= 0, x",
        &eval_options,
        SolveDisplayMode::None,
        false,
    )
    .expect("solve command should render");

    assert!(
        lines.iter().any(|line| {
            line.contains("verification requires numeric sampling") && line.contains("x <= 0")
        }),
        "lines: {:?}",
        lines
    );
}

#[test]
fn test_solve_check_session_render_surfaces_non_discrete_note_for_mixed_conditional_solution() {
    let mut simplifier = Simplifier::with_default_rules();
    let mut eval_options = EvalOptions::solve();
    eval_options.budget.max_branches = 2;
    eval_options.shared.semantics.domain_mode = DomainMode::Strict;
    let mut session = StatelessEvalSession::new(eval_options.clone());

    let lines = evaluate_solve_command_lines_with_session(
        &mut simplifier,
        &mut session,
        "solve --check a^x = a, x",
        &eval_options,
        SolveDisplayMode::None,
        false,
    )
    .expect("solve command should render");

    assert!(
        lines
            .iter()
            .any(|line| { line.contains("✓ x = 1 verified") }),
        "lines: {:?}",
        lines
    );
}

fn classify_for_test(
    ctx: &Context,
    base: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
    mode: DomainMode,
) -> cas_solver_core::log_domain::LogSolveDecision {
    let opts = SolverOptions {
        domain_mode: mode,
        ..Default::default()
    };
    cas_solver_core::log_domain::classify_log_solve_with_env_and_tri_prover(
        ctx,
        base,
        rhs,
        opts.value_domain == ValueDomain::RealOnly,
        cas_solver_core::log_domain::domain_mode_kind_from_flags(
            matches!(mode, DomainMode::Assume),
            matches!(mode, DomainMode::Strict),
        ),
        &SolveDomainEnv::default(),
        |core_ctx, expr| match cas_solver::api::prove_positive(core_ctx, expr, opts.value_domain) {
            EngineProof::Proven | EngineProof::ProvenImplicit => TriProof::Proven,
            EngineProof::Disproven => TriProof::Disproven,
            EngineProof::Unknown => TriProof::Unknown,
        },
    )
}

#[test]
fn test_log_solve_both_proven_positive_ok() {
    let mut ctx = Context::new();
    let base = ctx.num(2);
    let rhs = ctx.num(8);
    let decision = classify_for_test(&ctx, base, rhs, DomainMode::Generic);
    assert!(matches!(
        decision,
        cas_solver_core::log_domain::LogSolveDecision::Ok
    ));
}

#[test]
fn test_log_solve_negative_rhs_empty() {
    let mut ctx = Context::new();
    let base = ctx.num(2);
    let rhs = ctx.num(-5);
    let decision = classify_for_test(&ctx, base, rhs, DomainMode::Generic);
    assert!(matches!(
        decision,
        cas_solver_core::log_domain::LogSolveDecision::EmptySet(_)
    ));
}

#[test]
fn test_log_solve_negative_base_needs_complex() {
    let mut ctx = Context::new();
    let base = ctx.num(-2);
    let rhs = ctx.num(5);
    let decision = classify_for_test(&ctx, base, rhs, DomainMode::Generic);
    assert!(matches!(
        decision,
        cas_solver_core::log_domain::LogSolveDecision::NeedsComplex(_)
    ));
}

#[test]
fn test_log_solve_assume_unknown_rhs_emits_assumption() {
    let mut ctx = Context::new();
    let base = ctx.num(2);
    let rhs = ctx.var("y");
    let decision = classify_for_test(&ctx, base, rhs, DomainMode::Assume);
    match decision {
        cas_solver_core::log_domain::LogSolveDecision::OkWithAssumptions(assumptions) => {
            assert!(assumptions.contains(&cas_solver_core::log_domain::LogAssumption::PositiveRhs));
        }
        _ => panic!("Expected OkWithAssumptions, got {:?}", decision),
    }
}

#[test]
fn test_log_solve_generic_unknown_rhs_unsupported() {
    let mut ctx = Context::new();
    let base = ctx.num(2);
    let rhs = ctx.var("y");
    let decision = classify_for_test(&ctx, base, rhs, DomainMode::Generic);
    assert!(matches!(
        decision,
        cas_solver_core::log_domain::LogSolveDecision::Unsupported(_, _)
    ));
}

#[test]
fn test_log_solve_neg_expr_rhs_empty() {
    let mut ctx = Context::new();
    let base = ctx.num(2);
    let five = ctx.num(5);
    let rhs = ctx.add(Expr::Neg(five));
    let decision = classify_for_test(&ctx, base, rhs, DomainMode::Generic);
    assert!(
        matches!(
            decision,
            cas_solver_core::log_domain::LogSolveDecision::EmptySet(_)
        ),
        "Expected EmptySet for Neg(5), got {:?}",
        decision
    );
}
