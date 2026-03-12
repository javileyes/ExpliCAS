use cas_ast::{BoundType, Context, Equation, Expr, RelOp, SolutionSet};
use cas_formatter::DisplayExpr;
use cas_math::tri_proof::TriProof;
use cas_parser::parse;
use cas_solver::api::{solve, verify_solution_set, Proof as EngineProof, VerifySummary};
use cas_solver::runtime::{
    DomainMode, Simplifier, SolveDomainEnv, SolverOptions, StepsMode, ValueDomain,
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
