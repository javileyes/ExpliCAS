use cas_solver::runtime::rules::algebra::{
    AddFractionsRule, DifferenceOfSquaresRule, ExpandRule, FactorBasedLCDRule,
    FactorDifferenceSquaresRule, FactorRule, PullConstantFromFractionRule, SimplifyFractionRule,
    SimplifyMulDivRule,
};
use cas_solver::runtime::rules::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule};
use cas_solver::runtime::rules::calculus::{DiffRule, IntegrateRule};
use cas_solver::runtime::rules::canonicalization::{
    CanonicalizeAddRule, CanonicalizeDivRule, CanonicalizeMulRule, CanonicalizeNegationRule,
    CanonicalizeRootRule,
};
use cas_solver::runtime::rules::exponents::{
    EvaluatePowerRule, IdentityPowerRule, NegativeExponentNormalizationRule, PowerPowerRule,
    PowerProductRule, PowerQuotientRule, ProductPowerRule,
};
use cas_solver::runtime::rules::functions::EvaluateAbsRule;
use cas_solver::runtime::rules::grouping::CollectRule;
use cas_solver::runtime::rules::logarithms::{
    EvaluateLogRule, ExponentialLogRule, LogAbsSimplifyRule, LogContractionRule, LogExpInverseRule,
    SplitLogExponentsRule,
};
use cas_solver::runtime::rules::number_theory::NumberTheoryRule;
use cas_solver::runtime::rules::polynomial::{
    AnnihilationRule, CombineLikeTermsRule, DistributeRule,
};
use cas_solver::runtime::rules::trigonometry::{
    EvaluateTrigRule, PythagoreanIdentityRule, RecursiveTrigExpansionRule, TanToSinCosRule,
};
use cas_solver::runtime::{Simplifier, SolverOptions};

use cas_ast::{BoundType, Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_solver::api::{solve, solve_with_display_steps};
use num_traits::Zero;

fn create_full_simplifier() -> Simplifier {
    let mut simplifier = Simplifier::new();
    simplifier.allow_numerical_verification = false; // Enforce symbolic equivalence
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));
    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeDivRule));
    simplifier.add_rule(Box::new(CanonicalizeRootRule));
    simplifier.add_rule(Box::new(EvaluateAbsRule));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::functions::AbsSquaredRule,
    ));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::functions::AbsPositiveSimplifyRule,
    )); // V2.14.20: |x| → x when x > 0 (POST phase)
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::trigonometry::AngleIdentityRule,
    ));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::trigonometry::AngleConsistencyRule,
    ));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::trigonometry::TrigPythagoreanSimplifyRule,
    ));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::trigonometry::DoubleAngleRule,
    ));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::trigonometry::TrigHalfAngleSquaresRule,
    ));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::trigonometry::TrigPhaseShiftRule,
    ));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::trigonometry::TrigEvenPowerDifferenceRule,
    ));
    simplifier.add_rule(Box::new(RecursiveTrigExpansionRule));
    // NOTE: CanonicalizeTrigSquareRule DISABLED - conflicts with TrigPythagoreanSimplifyRule
    // causing infinite loops (cos² → 1-sin² → cos² → ...)
    // This matches the default simplifier configuration.
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(EvaluateLogRule));
    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(LogExpInverseRule)); // ln(e^x) → x
    simplifier.add_rule(Box::new(LogAbsSimplifyRule)); // ln(|x|) → ln(x) when x > 0 - MUST be before LogContractionRule
    simplifier.add_rule(Box::new(LogContractionRule)); // ln(a) + ln(b) → ln(a*b)
    simplifier.add_rule(Box::new(SplitLogExponentsRule));
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(PowerProductRule));
    simplifier.add_rule(Box::new(PowerQuotientRule));
    simplifier.add_rule(Box::new(IdentityPowerRule));
    simplifier.add_rule(Box::new(NegativeExponentNormalizationRule));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::exponents::NegativeBasePowerRule,
    ));
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(ExpandRule));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::polynomial::BinomialExpansionRule,
    ));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::algebra::NestedFractionRule,
    ));
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(AddFractionsRule));
    simplifier.add_rule(Box::new(SimplifyMulDivRule));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::algebra::RationalizeDenominatorRule,
    ));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::algebra::CancelCommonFactorsRule,
    ));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::algebra::SimplifySquareRootRule,
    ));
    simplifier.add_rule(Box::new(PullConstantFromFractionRule));
    simplifier.add_rule(Box::new(FactorBasedLCDRule));
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(CollectRule));
    simplifier.add_rule(Box::new(FactorDifferenceSquaresRule));
    simplifier.add_rule(Box::new(DifferenceOfSquaresRule)); // Policy A+: (a-b)(a+b) → a²-b²

    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::arithmetic::MulZeroRule,
    ));

    simplifier.add_rule(Box::new(IntegrateRule));
    simplifier.add_rule(Box::new(DiffRule));
    simplifier.add_rule(Box::new(NumberTheoryRule));
    simplifier.add_rule(Box::new(
        cas_solver::runtime::rules::arithmetic::DivZeroRule,
    ));
    simplifier
}

fn assert_equivalent(s: &mut Simplifier, expr1: ExprId, expr2: ExprId) {
    let (sim1, _) = s.simplify(expr1);
    let (sim2, _) = s.simplify(expr2);

    // Check if sim1 == sim2 directly first
    if s.are_equivalent(sim1, sim2) {
        return;
    }

    // Try simplifying difference
    let diff = s.context.add(Expr::Sub(sim1, sim2));
    let (sim_diff, _) = s.simplify(diff);

    // Check if difference is 0
    if let Expr::Number(n) = s.context.get(sim_diff) {
        if n.is_zero() {
            return;
        }
    }

    panic!(
        "Expressions not equivalent.\nExpr1: {}\nSim1: {}\nExpr2: {}\nSim2: {}\nDiff: {}",
        DisplayExpr {
            context: &s.context,
            id: expr1
        },
        DisplayExpr {
            context: &s.context,
            id: sim1
        },
        DisplayExpr {
            context: &s.context,
            id: expr2
        },
        DisplayExpr {
            context: &s.context,
            id: sim2
        },
        DisplayExpr {
            context: &s.context,
            id: sim_diff
        }
    );
}

// --- Level 1: Algebraic and Rational Simplification ---

#[test]
fn test_rational_simplification_invisible() {
    // simplify((x^3 - 1) / (x - 1)) -> x^2 + x + 1
    // Requires difference of cubics or polynomial division.
    let mut simplifier = create_full_simplifier();
    let input = parse("(x^3 - 1) / (x - 1)", &mut simplifier.context).unwrap();
    let expected = parse("x^2 + x + 1", &mut simplifier.context).unwrap();
    assert_equivalent(&mut simplifier, input, expected);
}

#[test]
fn test_exponential_expansion() {
    // expand((x + 1)^5)
    let mut simplifier = create_full_simplifier();
    let input = parse("(x + 1)^5", &mut simplifier.context).unwrap();
    let expected = parse(
        "x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1",
        &mut simplifier.context,
    )
    .unwrap();

    // Use expand() since we're testing binomial expansion
    let (expanded, _) = simplifier.expand(input);
    assert!(
        simplifier.are_equivalent(expanded, expected),
        "Binomial expansion failed: (x+1)^5 should expand to x^5 + 5x^4 + 10x^3 + 10x^2 + 5x + 1"
    );
}

#[test]
fn test_nested_fraction() {
    // simplify((1 + 1/x) / (1 - 1/x)) -> (x + 1) / (x - 1)
    let mut simplifier = create_full_simplifier();
    let input = parse("(1 + 1/x) / (1 - 1/x)", &mut simplifier.context).unwrap();
    let expected = parse("(x + 1) / (x - 1)", &mut simplifier.context).unwrap();
    assert_equivalent(&mut simplifier, input, expected);
}

// --- Level 2: Transcendental and Properties ---

#[test]
fn test_trig_identity_hidden() {
    let input = "sin(x)^4 - cos(x)^4 - (sin(x)^2 - cos(x)^2)";
    let mut simplifier = create_full_simplifier();
    // No manual rules needed anymore

    let expr = parse(input, &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    assert_eq!(result_str, "0", "Failed on: {}", input);
}

#[test]
fn test_algebraic_labyrinth() {
    use cas_session::SessionState;
    use cas_solver::runtime::{Engine, EvalAction, EvalRequest, EvalResult};

    // ln(e^3) + (sin(x) + cos(x))^2 - sin(2*x) - (x^3 - 8)/(x - 2) + x^2 + 2*x
    // Expected: 0
    let input = "ln(e^3) + (sin(x) + cos(x))^2 - sin(2*x) - (x^3 - 8)/(x - 2) + x^2 + 2*x";
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    let parsed = parse(input, &mut engine.simplifier.context).unwrap();
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };
    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_str = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr
            }
        ),
        other => panic!(
            "Unexpected eval result for algebraic labyrinth: {:?}",
            other
        ),
    };
    assert_eq!(result_str, "0", "Failed on: {}", input);
}

#[test]
fn test_log_power_trap() {
    // simplify(ln(e^(x^2 + 1))) -> x^2 + 1
    let mut simplifier = create_full_simplifier();
    let input = parse("ln(e^(x^2 + 1))", &mut simplifier.context).unwrap();
    let expected = parse("x^2 + 1", &mut simplifier.context).unwrap();
    assert_equivalent(&mut simplifier, input, expected);
}

#[test]
fn test_log_cancellation() {
    // simplify(e^(ln(x) + ln(y))) -> x * y
    // Requires Positive(x), Positive(y) - only works in Assume mode (V1.3 contract)
    let mut simplifier = create_full_simplifier();
    let input = parse("e^(ln(x) + ln(y))", &mut simplifier.context).unwrap();
    let expected = parse("x * y", &mut simplifier.context).unwrap();

    // Use Assume mode since Generic now blocks Analytic conditions (Positive)
    let opts = cas_solver::runtime::SimplifyOptions {
        shared: cas_solver::runtime::SharedSemanticConfig {
            semantics: cas_solver::runtime::EvalConfig {
                domain_mode: cas_solver::runtime::DomainMode::Assume,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let (sim_input, _) = simplifier.simplify_with_options(input, opts.clone());
    let (sim_expected, _) = simplifier.simplify_with_options(expected, opts);

    assert!(
        simplifier.are_equivalent(sim_input, sim_expected),
        "Log cancellation should work in Assume mode"
    );
}

// --- Level 3: The Solver ---

#[test]
fn test_hidden_quadratic_solve() {
    // solve e^(2*x) - 3*e^x + 2 = 0
    // u = e^x -> u^2 - 3u + 2 = 0 -> (u-2)(u-1)=0 -> u=2, u=1
    // e^x = 2 -> x = ln(2)
    // e^x = 1 -> x = 0
    let mut simplifier = create_full_simplifier();
    let lhs = parse("e^(2*x) - 3*e^x + 2", &mut simplifier.context).unwrap();
    let rhs = parse("0", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Discrete(solutions) = result {
        // Expect 2 solutions
        assert_eq!(
            solutions.len(),
            2,
            "Expected 2 solutions, got {:?}",
            solutions
        );
        // Check for 0 and ln(2)
        let has_zero = solutions.iter().any(|s| {
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: *s
                }
            ) == "0"
        });
        let has_ln2 = solutions.iter().any(|s| {
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: *s
                }
            ) == "ln(2)"
        });
        assert!(has_zero, "Missing solution x=0");
        assert!(has_ln2, "Missing solution x=ln(2)");
    } else {
        panic!("Expected Discrete solution, got {:?}", result);
    }
}

#[test]
fn test_nested_abs_solve() {
    // solve ||x - 1| - 2| = 1
    // |x-1| - 2 = 1  OR  |x-1| - 2 = -1
    // |x-1| = 3      OR  |x-1| = 1
    // x-1=3 -> x=4   OR  x-1=1 -> x=2
    // x-1=-3 -> x=-2 OR  x-1=-1 -> x=0
    // Solutions: 4, -2, 2, 0
    let mut simplifier = create_full_simplifier();
    let lhs = parse("||x - 1| - 2|", &mut simplifier.context).unwrap();
    let rhs = parse("1", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Discrete(solutions) = result {
        assert_eq!(
            solutions.len(),
            4,
            "Expected 4 solutions, got {:?}",
            solutions
        );
        let s_strs: Vec<String> = solutions
            .iter()
            .map(|s| {
                format!(
                    "{}",
                    DisplayExpr {
                        context: &simplifier.context,
                        id: *s
                    }
                )
            })
            .collect();
        assert!(s_strs.contains(&"4".to_string()));
        assert!(s_strs.contains(&"-2".to_string()));
        assert!(s_strs.contains(&"2".to_string()));
        assert!(s_strs.contains(&"0".to_string()));
    } else {
        panic!("Expected Discrete solution, got {:?}", result);
    }
}

#[test]
fn test_rational_inequality_signs() {
    // solve (x - 1) / (x + 2) >= 0
    // Critical points: 1 (zero), -2 (pole)
    // Intervals: (-inf, -2), (-2, 1), (1, inf)
    // Test -3: (-4)/(-1) = 4 > 0 -> True
    // Test 0: (-1)/(2) = -0.5 < 0 -> False
    // Test 2: (1)/(4) = 0.25 > 0 -> True
    // Result: (-inf, -2) U [1, inf)
    // Note: -2 is OPEN because it's a pole. 1 is CLOSED because >= 0.

    let mut simplifier = create_full_simplifier();
    let lhs = parse("(x - 1) / (x + 2)", &mut simplifier.context).unwrap();
    let rhs = parse("0", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Geq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Union(intervals) = result {
        assert_eq!(intervals.len(), 2);
        // (-inf, -2)
        let i1 = &intervals[0];
        let (min1, _) = simplifier.simplify(i1.min);
        let (max1, _) = simplifier.simplify(i1.max);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: min1
                }
            ),
            "-infinity"
        );
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: max1
                }
            ),
            "-2"
        );
        assert_eq!(i1.max_type, BoundType::Open, "Pole at -2 should be Open");

        // [1, inf)
        let i2 = &intervals[1];
        let (min2, _) = simplifier.simplify(i2.min);
        let (max2, _) = simplifier.simplify(i2.max);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: min2
                }
            ),
            "1"
        );
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: max2
                }
            ),
            "infinity"
        );
        assert_eq!(
            i2.min_type,
            BoundType::Closed,
            "Zero at 1 should be Closed for >="
        );
    } else {
        panic!("Expected Union solution, got {:?}", result);
    }
}

#[test]
fn test_quadratic_abs_inequality() {
    // solve |x^2 - 1| < 3
    // -3 < x^2 - 1 < 3
    // x^2 - 1 > -3 -> x^2 > -2 (Always true)
    // x^2 - 1 < 3 -> x^2 < 4 -> (-2, 2)
    // Intersection: (-2, 2)

    let mut simplifier = create_full_simplifier();
    let lhs = parse("|x^2 - 1|", &mut simplifier.context).unwrap();
    let rhs = parse("3", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Lt,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Continuous(interval) = result {
        let (min, _) = simplifier.simplify(interval.min);
        let (max, _) = simplifier.simplify(interval.max);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: min
                }
            ),
            "-2"
        );
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: max
                }
            ),
            "2"
        );
    } else {
        panic!("Expected Continuous solution, got {:?}", result);
    }
}

#[test]
fn test_zero_equivalence_suite() {
    let cases = vec![
        "1/(x-1) - 2/(x^2-1) - 1/(x+1)",
        "(tan(x)*cos(x))^2 + cos(x)^2 - 1",
        "sqrt(x^2 + 2*x + 1) - abs(x + 1)",
        "2*ln(sqrt(e^x)) - x + ln(1)",
        "(x+2)^3 - (x^3 + 6*x^2 + 12*x + 8)",
    ];

    for input in cases {
        let mut simplifier = create_full_simplifier();
        // All necessary rules are now in create_full_simplifier

        // Use expand() method which SHOULD expand binomials
        let expr = parse(input, &mut simplifier.context).unwrap();
        let (simplified, _) = simplifier.expand(expr);

        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: simplified
            }
        );
        assert_eq!(result_str, "0", "Failed on: {}", input);
    }
}

// --- Round 2: Advanced Torture Tests ---

#[test]
fn test_torture_6_conjugate() {
    // 1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)
    // Expected: 0
    // NOTE: This requires Assume mode because sqrt(x)^2 = x requires x >= 0
    // The identity uses x - 1 = (sqrt(x) - 1)(sqrt(x) + 1)
    let input = "1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)";
    let mut simplifier = create_full_simplifier();

    let expr = parse(input, &mut simplifier.context).unwrap();

    // Use Assume mode to allow sqrt(x)^2 -> x
    let opts = cas_solver::runtime::SimplifyOptions {
        shared: cas_solver::runtime::SharedSemanticConfig {
            semantics: cas_solver::runtime::EvalConfig {
                domain_mode: cas_solver::runtime::DomainMode::Assume,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let (simplified, _, _) = simplifier.simplify_with_stats(expr, opts);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    assert_eq!(result_str, "0", "Failed on: {}", input);
}

#[test]
fn test_torture_7_log_chain() {
    // (ln(x) / ln(10)) * (ln(10) / ln(x)) - 1
    // Expected: 0
    let input = "(ln(x) / ln(10)) * (ln(10) / ln(x)) - 1";
    let mut simplifier = create_full_simplifier();
    // Rules are now in create_full_simplifier

    let expr = parse(input, &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    assert_eq!(result_str, "0", "Failed on: {}", input);
}

#[test]
fn test_torture_8_sophie_germain() {
    // (x^2 + 2*y^2 + 2*x*y) * (x^2 + 2*y^2 - 2*x*y) - (x^4 + 4*y^4)
    // Expected: 0
    let input_str = "(x^2 + 2*y^2 + 2*x*y) * (x^2 + 2*y^2 - 2*x*y) - (x^4 + 4*y^4)";
    let mut simplifier = create_full_simplifier();
    // Rules are now in create_full_simplifier

    let input = format!("expand({})", input_str);
    let expr = parse(&input, &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    assert_eq!(result_str, "0", "Failed on: {}", input);
}

#[test]
fn test_torture_9_angle_compound() {
    // sin(x + y) - (sin(x)*cos(y) + cos(x)*sin(y))
    // Expected: 0
    // Note: angle-sum expansion is expand-mode only (Ticket 6c),
    // so we use expand() to verify the identity.
    let input = "sin(x + y) - (sin(x)*cos(y) + cos(x)*sin(y))";
    let mut simplifier = create_full_simplifier();

    let expr = parse(input, &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.expand(expr);
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    assert_eq!(result_str, "0", "Failed on: {}", input);
}

#[test]
fn test_torture_10_ghost_solution() {
    // solve sqrt(2*x + 3) = x
    // Expected: x = 3 (reject x = -1)
    let mut simplifier = create_full_simplifier();
    let lhs = parse("sqrt(2*x + 3)", &mut simplifier.context).unwrap();
    let rhs = parse("x", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Discrete(solutions) = result {
        assert_eq!(
            solutions.len(),
            1,
            "Should have eliminated extraneous root x=-1. Got: {:?}",
            solutions
                .iter()
                .map(|s| format!(
                    "{}",
                    DisplayExpr {
                        context: &simplifier.context,
                        id: *s
                    }
                ))
                .collect::<Vec<_>>()
        );
        let sol_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: solutions[0]
            }
        );
        assert_eq!(sol_str, "3");
    } else {
        panic!("Expected Discrete solution, got {:?}", result);
    }
}

#[test]
fn test_torture_24_difference_quotient() {
    let mut simplifier = create_full_simplifier();
    // ((x + h)^3 - x^3) / h - (3*x^2 + 3*x*h + h^2)
    // Should simplify to 0
    // Use expand() method which now correctly propagates expand_mode
    let input = "((x + h)^3 - x^3) / h - (3*x^2 + 3*x*h + h^2)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.expand(expr);

    // Check if result is 0
    let out = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(out, "0", "Difference Quotient failed to simplify to 0");
}

#[test]
fn test_torture_11_polynomial_stress() {
    // 11. La "Cascada de Cuadrados"
    // (x - 1) * (x + 1) * (x^2 + 1) * (x^4 + 1) - (x^8 - 1)

    // Create simplifier matching Repl::new order exactly
    let mut simplifier = Simplifier::new();
    // Always enabled core rules
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));
    let mut simplifier = create_full_simplifier();
    // No manual rules needed anymore

    let expr = parse(
        "(x - 1) * (x + 1) * (x^2 + 1) * (x^4 + 1) - (x^8 - 1)",
        &mut simplifier.context,
    )
    .unwrap();
    let (simplified, _) = simplifier.simplify(expr);

    let zero = simplifier.context.num(0);
    assert!(
        simplifier.are_equivalent(simplified, zero),
        "Polynomial stress test failed"
    );
}

#[test]
fn test_torture_12_solver_singularity() {
    // 12. El "Agujero en la Gráfica"
    // (x^2 - 1) / (x - 1) = 2
    // Sound outcomes:
    // - Empty / Discrete(empty)
    // - Conditional
    // - Discrete {1} only if the original denominator guard survives in
    //   required_conditions.
    let mut simplifier = create_full_simplifier();
    let stmt =
        cas_parser::parse_statement("(x^2 - 1) / (x - 1) = 2", &mut simplifier.context).unwrap();

    if let cas_parser::Statement::Equation(eq) = stmt {
        let result = solve_with_display_steps(&eq, "x", &mut simplifier, SolverOptions::default());

        match result {
            Ok((SolutionSet::Empty, _, _)) => (),
            Ok((SolutionSet::Conditional(cases), _, _)) => {
                assert!(
                    !cases.is_empty(),
                    "Conditional solve should carry at least one guarded case"
                );
            }
            Ok((SolutionSet::Discrete(sols), _, diagnostics)) => {
                let required: Vec<String> = diagnostics
                    .required
                    .iter()
                    .map(|cond| cond.display(&simplifier.context))
                    .collect();
                let has_original_den_guard = required
                    .iter()
                    .any(|r| r.contains("x - 1") && (r.contains("≠ 0") || r.contains("!= 0")));

                let one = simplifier.context.num(1);
                if sols.is_empty() {
                    return;
                }
                for sol in sols {
                    if simplifier.are_equivalent(sol, one) {
                        assert!(
                            has_original_den_guard,
                            "UNSOUND: solver returned x=1 without preserving x - 1 != 0. required={required:?}"
                        );
                        return;
                    }
                }
                panic!(
                    "Resultado discreto inesperado para singularidad: {:?}",
                    required
                );
            }
            _ => panic!("Resultado inesperado para singularidad: {:?}", result),
        }
    } else {
        panic!("Failed to parse equation");
    }
}

#[test]
fn test_torture_13_abs_squared() {
    // 13. La "Identidad Absoluta Oculta"
    // abs(x)^2 - x^2 -> 0
    let mut simplifier = create_full_simplifier();
    let expr = parse("abs(x)^2 - x^2", &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);

    let zero = simplifier.context.num(0);
    assert!(
        simplifier.are_equivalent(simplified, zero),
        "Abs squared identity failed"
    );
}

#[test]
fn test_torture_14_rational_telescoping() {
    // 14. La "Suma Telescópica Racional"
    // 1/(x*(x+1)) + 1/((x+1)*(x+2)) - 2/(x*(x+2)) -> 0
    let mut simplifier = create_full_simplifier();
    let expr = parse(
        "1/(x*(x+1)) + 1/((x+1)*(x+2)) - 2/(x*(x+2))",
        &mut simplifier.context,
    )
    .unwrap();
    let (simplified, _) = simplifier.simplify(expr);

    let zero = simplifier.context.num(0);
    assert!(
        simplifier.are_equivalent(simplified, zero),
        "Rational telescoping failed"
    );
}

#[test]
fn test_torture_15_trig_shift() {
    // 15. El "Cambio de Fase Trigonométrico"
    // sin(x + pi/2) - cos(x) -> 0
    let mut simplifier = create_full_simplifier();
    // Note: pi/2 might need to be parsed carefully or constructed if parser doesn't support 'pi' directly as constant in this context
    // Assuming parser handles 'pi'
    let expr = parse("sin(x + pi/2) - cos(x)", &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);

    let zero = simplifier.context.num(0);
    assert!(
        simplifier.are_equivalent(simplified, zero),
        "Trig shift failed"
    );
}

#[test]
fn test_torture_18_product_rule() {
    // 18. Regla del Producto
    // diff(x * sin(x), x) -> sin(x) + x * cos(x)
    let mut simplifier = create_full_simplifier();
    // DiffRule is now in create_full_simplifier

    let expr = parse("diff(x * sin(x), x)", &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);

    // Expected: sin(x) + x*cos(x)
    // Or x*cos(x) + sin(x)
    let expected_str = "sin(x) + x * cos(x)";
    let expected = parse(expected_str, &mut simplifier.context).unwrap();

    assert!(
        simplifier.are_equivalent(simplified, expected),
        "Product Rule failed"
    );
}

#[test]
fn test_torture_25_half_angle() {
    let mut s = create_full_simplifier();

    // sin(x) / (1 + cos(x)) - tan(x/2) -> 0
    let expr = parse("sin(x) / (1 + cos(x)) - tan(x/2)", &mut s.context).unwrap();
    let (res, _) = s.simplify(expr);

    let res_str = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: res
        }
    );
    assert_eq!(res_str, "0", "Half-Angle Identity failed to simplify to 0");
}

#[test]
fn test_torture_26_lagrange_identity() {
    // 26. La "Identidad de Lagrange"
    // (a^2 + b^2) * (c^2 + d^2) - (a*c + b*d)^2 - (a*d - b*c)^2
    // Expected: 0
    let mut simplifier = create_full_simplifier();
    // Requires expand() to trigger expansion
    let input_str = "expand((a^2 + b^2) * (c^2 + d^2) - (a*c + b*d)^2 - (a*d - b*c)^2)";

    let expr = parse(input_str, &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    assert_eq!(result_str, "0", "Lagrange Identity failed to simplify to 0");
}

#[test]
fn test_torture_27_hyperbolic_masquerade() {
    let mut simplifier = create_full_simplifier();
    // ((e^x + e^(-x))/2)^2 - ((e^x - e^(-x))/2)^2 - 1
    // Use expand() method which now correctly propagates expand_mode
    let input = "((e^x + e^(-x))/2)^2 - ((e^x - e^(-x))/2)^2 - 1";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.expand(expr);
    let result = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    assert_eq!(result, "0");
}

#[test]
fn test_torture_28_tangent_sum() {
    let mut simplifier = create_full_simplifier();
    // sin(x + y) / (cos(x) * cos(y)) - (tan(x) + tan(y))
    let input = "sin(x + y) / (cos(x) * cos(y)) - (tan(x) + tan(y))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    println!("Parsed AST: {:?}", simplifier.context.get(expr));
    let (simplified, _) = simplifier.simplify(expr);
    let result = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    assert_eq!(result, "0");
}

#[test]
fn test_log_sqrt_simplification() {
    let mut simplifier = create_full_simplifier();

    // ln(sqrt(x^2 + 2*x + 1)) - ln(x + 1) -> 0
    // Requires:
    // 1. sqrt(x^2+2x+1) -> |x+1| (SimplifySquareRootRule)
    // 2. ln(|x+1|) -> ln(x+1) (LogAbsSimplifyRule with domain assumption x+1>0)
    // 3. ln(x+1) - ln(x+1) -> 0 (AnnihilationRule or CombineLikeTermsRule)
    //
    // NOTE: This requires DomainMode::Assume because we cannot prove x+1 > 0
    // for a symbolic variable 'x'. The assumption is tracked via assumption_events.
    let input = "ln(sqrt(x^2 + 2*x + 1)) - ln(x + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let opts = cas_solver::runtime::SimplifyOptions {
        shared: cas_solver::runtime::SharedSemanticConfig {
            semantics: cas_solver::runtime::EvalConfig {
                domain_mode: cas_solver::runtime::DomainMode::Assume,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let (res, _, _) = simplifier.simplify_with_stats(expr, opts);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_collapsed_root_square_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // sqrt((sqrt(x^2 + 1))^2 + 2*sqrt(x^2 + 1) + 1) - |sqrt(x^2 + 1) + 1| -> 0
    //
    // The inner (sqrt(...))^2 collapses early to x^2 + 1, so the root matcher
    // must still recognize the resulting shape:
    //   sqrt((x^2 + 1) + 2*sqrt(x^2 + 1) + 1) = |sqrt(x^2 + 1) + 1|
    let input = "sqrt((sqrt(x^2 + 1))^2 + 2*sqrt(x^2 + 1) + 1) - abs(sqrt(x^2 + 1) + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_nested_reciprocal_sqrt_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // 1 / (1 / sqrt(x^2 + 1)) - sqrt(x^2 + 1) -> 0
    //
    // The inner reciprocal sqrt canonicalizes to (x^2 + 1)^(-1/2). The fraction
    // cleanup path must still recover the positive half-power while preserving
    // the original nonzero-domain requirement on the base.
    let input = "1/(1/(sqrt(x^2 + 1))) - sqrt(x^2 + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_negative_reciprocal_root_cubic_factoring_identity_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // (1/sqrt(u))^3 + 1 - ((1/sqrt(u) + 1) * (((u+1)/u) - 1/sqrt(u))) -> 0
    //
    // The runtime often reaches the mixed form ((u+1)/u) - u^(-1/2) instead of
    // a clean polynomial in t = u^(-1/2). The opaque-root proof prepass should
    // now split (u+1)/u into 1 + 1/u and close the same cubic identity.
    let input = "(1/sqrt(u))^3 + 1 - (((1/sqrt(u))+1)*(((u+1)/u) - (1/sqrt(u))))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_additive_common_square_factor_under_sqrt() {
    let mut simplifier = Simplifier::with_default_rules();

    // sqrt(4*x^2 + 4) - 2*sqrt(x^2 + 1) -> 0
    //
    // This is the collapsed form that appears after (sqrt(x^2+1))^2 -> x^2+1.
    // The sqrt simplifier should now extract the common square factor directly
    // from the additive radicand.
    let input = "sqrt(4*x^2 + 4) - 2*sqrt(x^2 + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_abs_additive_common_factor_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // abs(2*(x + pi)) - 2*abs(x + pi) -> 0
    //
    // The inner product may already be expanded to 2*x + 2*pi. The abs
    // simplifier should still recover the common positive factor.
    let input = "abs(2*(x + pi)) - 2*abs(x + pi)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_abs_global_negative_sum_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // abs(-(x^3 + 1)) - abs(x^3 + 1) -> 0
    //
    // After normalization this often appears as a fully negative Add chain
    // rather than a neat Neg wrapper. The abs simplifier should still factor
    // out the global sign.
    let input = "abs(-(x^3 + 1)) - abs(x^3 + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_abs_expanded_positive_factor_shortcut() {
    let mut simplifier = Simplifier::with_default_rules();

    // abs(2*(u^2 - 1)) - 2*abs(u^2 - 1) -> 0
    //
    // The runtime used to distribute inside abs and stop at |2*u^2 - 2|. The
    // top-level abs shortcut should now rescue the positive factor first.
    let input = "abs(2*(u^2 - 1)) - 2*abs(u^2 - 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_abs_global_negative_linear_sum_shortcut() {
    let mut simplifier = Simplifier::with_default_rules();

    // abs(-(2*u + 3)) - abs(2*u + 3) -> 0
    //
    // The runtime should evaluate the outer negation at the abs root before
    // expanding to |-2*u - 3|.
    let input = "abs(-(2*u + 3)) - abs(2*u + 3)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_abs_rational_difference_normalization() {
    let mut simplifier = Simplifier::with_default_rules();

    // abs((x/(x+1)) - 1) - abs(1/(x+1)) -> 0
    //
    // The left side simplifies through a negative reciprocal. Abs should drop
    // that global sign and converge to the same normalized residual as the RHS.
    let input = "abs((x/(x + 1)) - 1) - abs(1/(x + 1))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_abs_scalar_left_rational_difference_normalization() {
    use cas_session::SessionState;
    use cas_solver::runtime::{Engine, EvalAction, EvalRequest, EvalResult};

    let input = "abs(1 - (x/(x + 1))) - abs(1/(x + 1))";
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    let parsed = parse(input, &mut engine.simplifier.context).unwrap();
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };
    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_str = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr
            }
        ),
        other => panic!(
            "Unexpected eval result for abs scalar-left normalization: {:?}",
            other
        ),
    };
    assert_eq!(result_str, "0");
}

#[test]
fn test_abs_scalar_left_shifted_rational_difference_normalization() {
    let mut simplifier = Simplifier::with_default_rules();

    // abs(1 - (x-1)/(x+1)) - 2*abs(1/(x+1)) -> 0
    //
    // This is the same mirror issue, but with an extra factor 2 exposed after
    // rational normalization. Abs should not leave the scalar-left form behind.
    let input = "abs(1 - ((x - 1)/(x + 1))) - 2*abs(1/(x + 1))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_abs_odd_power_magnitude_canonicalization() {
    let mut simplifier = Simplifier::with_default_rules();

    // abs(x)^3 - x^2*abs(x) -> 0
    //
    // This canonical form is useful for downstream polynomial identities under
    // substitution because it exposes the even-power part as a plain factor
    // while preserving one magnitude atom.
    let input = "abs(x)^3 - x^2*abs(x)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_abs_quotient_inner_sign_normalization() {
    let mut simplifier = Simplifier::with_default_rules();

    // abs((1-x)/(x+1)) - abs((x-1)/(x+1)) -> 0
    //
    // Inside abs, swapping a preferred sub-like numerator only changes a
    // global sign. The runtime should normalize that quotient shape.
    let input = "abs((1 - x)/(x + 1)) - abs((x - 1)/(x + 1))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_abs_quotient_inner_denominator_sign_normalization() {
    let mut simplifier = Simplifier::with_default_rules();

    // abs(x/(1-x^2)) - abs(x/(x^2-1)) -> 0
    //
    // This is the same quotient-local sign issue, but in the denominator.
    let input = "abs(x/(1 - x^2)) - abs(x/(x^2 - 1))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_abs_difference_of_squares_quotient_cancellation() {
    let mut simplifier = Simplifier::with_default_rules();

    // ((abs(x))^2 - 4)/(abs(x) + 2) - (abs(x) - 2) -> 0
    //
    // In real mode, x^2 and abs(x)^2 represent the same square factor. The
    // standard runtime should still expose the difference-of-squares cancel.
    let input = "((abs(x))^2 - 4)/(abs(x) + 2) - (abs(x) - 2)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_abs_cube_quotient_exact_cancellation() {
    let mut simplifier = Simplifier::with_default_rules();

    // ((abs(u))^3 - 1)/(abs(u) - 1) - (abs(u)^2 + abs(u) + 1) -> 0
    //
    // Keep the odd abs power intact inside this quotient so the exact
    // t^3 - 1 over t - 1 path can fire before odd-power canonicalization.
    let input = "((abs(u))^3 - 1)/(abs(u) - 1) - (abs(u)^2 + abs(u) + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_arctan_reciprocal_difference_of_squares_cancellation() {
    let mut simplifier = Simplifier::with_default_rules();

    // ((arctan(x)) - 1)/((arctan(x))^2 - 1) - 1/(arctan(x) + 1) -> 0
    let input = "((arctan(x)) - 1)/((arctan(x))^2 - 1) - 1/(arctan(x) + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_arcsin_reciprocal_difference_of_squares_cancellation() {
    let mut simplifier = Simplifier::with_default_rules();

    // ((arcsin(x)) - 1)/((arcsin(x))^2 - 1) - 1/(arcsin(x) + 1) -> 0
    let input = "((arcsin(x)) - 1)/((arcsin(x))^2 - 1) - 1/(arcsin(x) + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_rational_context_sec_tan_pythagorean_identity() {
    let mut simplifier = Simplifier::with_default_rules();

    // sec(1/(x-1)+1/(x+1))^2 - tan(1/(x-1)+1/(x+1))^2 -> 1
    let input = "sec((1/(x - 1) + 1/(x + 1)))^2 - tan((1/(x - 1) + 1/(x + 1)))^2 - 1";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_sqrt_of_rational_perfect_square_quotient() {
    let mut simplifier = Simplifier::with_default_rules();

    // sqrt((x/(x+1))^2 + 2*(x/(x+1)) + 1) - abs((2*x+1)/(x+1)) -> 0
    let input = "sqrt((x/(x + 1))^2 + 2*(x/(x + 1)) + 1) - abs((2*x + 1)/(x + 1))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_sqrt_of_expanded_rational_perfect_square_quotient() {
    let mut simplifier = Simplifier::with_default_rules();

    // sqrt((1/x + 1/(x+1))^2 + 2*(1/x + 1/(x+1)) + 1) - abs((1/x + 1/(x+1)) + 1) -> 0
    let input = "sqrt((1/x + 1/(x + 1))^2 + 2*(1/x + 1/(x + 1)) + 1) - abs((1/x + 1/(x + 1)) + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_sqrt_of_expanded_symmetric_rational_perfect_square_quotient() {
    let mut simplifier = Simplifier::with_default_rules();

    // sqrt((1/(x-1) + 1/(x+1))^2 + 2*(1/(x-1) + 1/(x+1)) + 1) - abs((1/(x-1) + 1/(x+1)) + 1) -> 0
    let input = "sqrt((1/(x - 1) + 1/(x + 1))^2 + 2*(1/(x - 1) + 1/(x + 1)) + 1) - abs((1/(x - 1) + 1/(x + 1)) + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_sqrt_of_root_ctx_shifted_unit_square() {
    let mut simplifier = Simplifier::with_default_rules();

    // sqrt((1/sqrt(u))^2 + 2*(1/sqrt(u)) + 1) - |1/sqrt(u) + 1| -> 0
    //
    // The standard runtime used to simplify the squared term first and lose the
    // exact t^2 + 2t + 1 shape for t = 1/sqrt(u). The root shortcut should now
    // close this directly to the exact abs(...) form.
    let input = "sqrt((1/sqrt(u))^2 + 2*(1/sqrt(u)) + 1) - abs(1/sqrt(u) + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_sqrt_of_root_ctx_scaled_square() {
    let mut simplifier = Simplifier::with_default_rules();

    // sqrt(4*(1/sqrt(u))^2) - |2/sqrt(u)| -> 0
    //
    // After extracting the numeric square factor, the inner sqrt should still
    // close to abs(1/sqrt(u)) in the same standard runtime path.
    let input = "sqrt(4*(1/sqrt(u))^2) - abs(2/sqrt(u))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_root_ctx_shifted_unit_square_exact_quotient() {
    let mut simplifier = Simplifier::with_default_rules();

    // ((1/sqrt(u))^2 + 2*(1/sqrt(u)) + 1)/((1/sqrt(u)) + 1) - ((1/sqrt(u)) + 1) -> 0
    //
    // The standard runtime should prefer the exact quotient before conjugate
    // rationalization, otherwise this falls into a domain-equivalent residual.
    let input = "(((1/sqrt(u))^2 + 2*(1/sqrt(u)) + 1)/((1/sqrt(u)) + 1)) - ((1/sqrt(u)) + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_sqrt_collapsed_abs_square_trinomial_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // sqrt(abs(x)^2 + 2*abs(x) + 1) - (abs(x) + 1) -> 0
    //
    // The middle term carries abs(x) instead of an explicit sqrt. The perfect
    // square matcher should still recover the collapsed square and simplify.
    let input = "sqrt(abs(x)^2 + 2*abs(x) + 1) - (abs(x) + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_sqrt_symbolic_phase_shift_trinomial_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // sqrt((u + pi)^2 + 2*(u + pi) + 1) - abs(u + pi + 1) -> 0
    //
    // The standard runtime expands (u + pi)^2 first. The square-root rewrite
    // still needs to recover the symbolic phase shift from the expanded monic
    // quadratic and close to the exact abs(...) form.
    let input = "sqrt((u + pi)^2 + 2*(u + pi) + 1) - abs(u + pi + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_sqrt_exp_shifted_unit_square_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // sqrt(exp(u)^2 + 2*exp(u) + 1) - (exp(u) + 1) -> 0
    //
    // The square term usually arrives as e^(2*u), not as an explicit (e^u)^2.
    // The standard root shortcut should still recover the shifted unit square,
    // produce abs(e^u + 1), and let nonnegativity close it to e^u + 1.
    let input = "sqrt(exp(u)^2 + 2*exp(u) + 1) - (exp(u) + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_cos_triple_arctan_identity_zero() {
    let mut simplifier = Simplifier::with_default_rules();

    // cos(3*arctan(u)) - (4*cos(arctan(u))^3 - 3*cos(arctan(u))) -> 0
    //
    // The exact triple-angle identity should win before the inverse-atan
    // composition rule expands the individual cosine children into radical
    // denominators.
    let input = "cos(3*(arctan(u))) - (4*cos((arctan(u)))^3 - 3*cos((arctan(u))))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_ln_reciprocal_exp_inverse_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // ln(exp(-u)) + u -> 0
    //
    // After canonicalizing exp(-u) to 1/e^u, the log-exp inverse matcher should
    // still recognize the reciprocal exponential and close exactly to -u.
    let input = "ln(exp((-u))) + u";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_ln_reciprocal_even_power_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    let input = "ln(1/(u^2)) + ln(u^2)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_ln_reciprocal_even_power_function_base_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    let input = "ln(1/(sin(u)^2)) + ln(sin(u)^2)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_reciprocal_sec_tan_pythagorean_identity() {
    let mut simplifier = Simplifier::with_default_rules();

    // 1/cos(x)^2 - tan(x)^2 - 1 -> 0
    //
    // The direct Pythagorean sec/tan rule should also recognize the reciprocal
    // secant form after canonicalization has lowered sec(x)^2 to 1/cos(x)^2.
    let input = "1/cos(x)^2 - tan(x)^2 - 1";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_builtin_sqrt_collapsed_root_quotient_cancellation() {
    let mut simplifier = Simplifier::with_default_rules();

    // ((sqrt(x^2+1))^2 + 2*sqrt(x^2+1))/(sqrt(x^2+1)+2) - sqrt(x^2+1) -> 0
    //
    // The exact-quotient path should treat builtin sqrt the same way it already
    // treats canonical ( ... )^(1/2) roots.
    let input = "((sqrt(x^2 + 1))^2 + 2*sqrt(x^2 + 1))/(sqrt(x^2 + 1) + 2) - sqrt(x^2 + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_builtin_sqrt_canonical_root_cube_quotient_cancellation() {
    let mut simplifier = Simplifier::with_default_rules();

    // ((sqrt(u^2+1))^3 - 1)/(sqrt(u^2+1)-1) - (sqrt(u^2+1)^2 + sqrt(u^2+1) + 1) -> 0
    //
    // The exact-quotient path must also see canonical rational-power variants
    // like (u^2+1)^(3/2), not only explicit sqrt(...) nodes on both sides.
    let input =
        "((sqrt(u^2 + 1))^3 - 1)/(sqrt(u^2 + 1) - 1) - (sqrt(u^2 + 1)^2 + sqrt(u^2 + 1) + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_tan_arctan_of_structural_sqrt_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // tan(arctan(sqrt(x^2 + 1))) - sqrt(x^2 + 1) -> 0
    //
    // TanToSinCos must not expand the outer tan before the direct inverse-trig
    // composition rule has a chance to fire at the root.
    let input = "tan(arctan(sqrt(x^2 + 1))) - sqrt(x^2 + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_sin_sum_triple_identity_after_distributing_scalar_over_sum() {
    let mut simplifier = Simplifier::with_default_rules();

    // sin(u^3 + 1) + sin(3*(u^3 + 1)) - 2*sin(2*(u^3 + 1))*cos(u^3 + 1) -> 0
    //
    // After distributive normalization, the identity matcher must still
    // recognize t, 2t and 3t when the scalars have been pushed into the sum.
    let input = "sin(u^3 + 1) + sin(3*(u^3 + 1)) - 2*sin(2*(u^3 + 1))*cos(u^3 + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_sin_sum_triple_identity_with_rational_context() {
    let mut simplifier = Simplifier::with_default_rules();

    // sin(r) + sin(3*r) - 2*sin(2*r)*cos(r) -> 0 with r = 1/(x-1)+1/(x+1)
    //
    // The identity should still close after fraction addition/pull-constant
    // rewrites reshape r, 2r and 3r into equivalent rational forms.
    let input = "sin((1/(x - 1) + 1/(x + 1))) + sin(3*(1/(x - 1) + 1/(x + 1))) - 2*sin(2*(1/(x - 1) + 1/(x + 1)))*cos((1/(x - 1) + 1/(x + 1)))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_sin_sum_triple_identity_with_nested_scaled_argument() {
    let mut simplifier = Simplifier::with_default_rules();

    // sin(2*u) + sin(3*(2*u)) - 2*sin(2*(2*u))*cos(2*u) -> 0
    //
    // The identity matcher should accumulate numeric scale across nested
    // multiplicative chains, not only direct 3*t / 2*t wrappers.
    let input = "sin(2*u) + sin(3*(2*u)) - 2*sin(2*(2*u))*cos(2*u)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_collapsed_root_gcf_identity_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // sqrt(x^2 + 1) * (sqrt(x^2 + 1) + 1) - ((sqrt(x^2 + 1))^2 + sqrt(x^2 + 1)) -> 0
    //
    // After early square collapse, the residual becomes a polynomial identity
    // modulo t^2 = x^2 + 1 with t = sqrt(x^2 + 1). PolynomialIdentityZeroRule
    // should now close that identity symbolically.
    let input = "sqrt(x^2 + 1)*(sqrt(x^2 + 1) + 1) - ((sqrt(x^2 + 1))^2 + sqrt(x^2 + 1))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_collapsed_root_trinomial_factoring_identity_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // ((sqrt(x^2 + 1))+2)*((sqrt(x^2 + 1))+3) - ((sqrt(x^2 + 1))^2 + 5*sqrt(x^2 + 1) + 6) -> 0
    //
    // One side keeps t^2, the other side arrives collapsed as x^2 + 1, so the
    // proof must use the same opaque-root relation as the simpler GCF case.
    let input =
        "((sqrt(x^2 + 1))+2)*((sqrt(x^2 + 1))+3) - ((sqrt(x^2 + 1))^2 + 5*sqrt(x^2 + 1) + 6)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_collapsed_root_cubic_factoring_identity_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // (sqrt(x^2 + 1))^3 + 1 - ((sqrt(x^2 + 1) + 1) * ((sqrt(x^2 + 1))^2 - sqrt(x^2 + 1) + 1)) -> 0
    //
    // The residual mixes t^3 with a collapsed t^2 -> x^2 + 1 term. The proof
    // should reuse the same root atom t = sqrt(x^2 + 1) and interpret
    // (x^2 + 1)^(3/2) as t^3 internally instead of inventing a second opaque atom.
    let input =
        "(sqrt(x^2 + 1))^3 + 1 - (((sqrt(x^2 + 1))+1)*((sqrt(x^2 + 1))^2 - sqrt(x^2 + 1) + 1))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_poly_high_cubic_factoring_identity_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // (u^3)^3 + 1 - ((u^3 + 1) * ((u^3)^2 - u^3 + 1)) -> 0
    //
    // This is the same algebraic identity as t^3 + 1 = (t + 1)(t^2 - t + 1)
    // with t = u^3. The polynomial identity detector should now admit the
    // resulting degree-9 univariate normal form directly.
    let input = "(u^3)^3 + 1 - (((u^3)+1)*((u^3)^2 - (u^3) + 1))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_rational_atom_binomial_identity_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // ((u/(u+1))+1)^4 - ((u/(u+1))^4 + 4*(u/(u+1))^3 + 6*(u/(u+1))^2 + 4*(u/(u+1)) + 1) -> 0
    //
    // The bottom-up simplifier used to normalize children into a bulky rational
    // residual before PolynomialIdentityZeroRule had a chance to see the exact
    // binomial identity in the opaque rational atom t = u/(u+1).
    let input =
        "((u/(u + 1))+1)^4 - ((u/(u + 1))^4 + 4*(u/(u + 1))^3 + 6*(u/(u + 1))^2 + 4*(u/(u + 1)) + 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_poly_high_degree_eighteen_factoring_identity_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // (u^3)^6 - 1 -> ((u^3)^2 + u^3 + 1) * ((u^3)^2 - u^3 + 1) * (u^3 + 1) * (u^3 - 1)
    //
    // The raw metamorphic canary still contains this exact t^6 - 1 factorization
    // with t = u^3. The direct polynomial proof should now handle the resulting
    // degree-18 univariate identity without falling back to numeric-only.
    let input =
        "(u^3)^6 - 1 - (((u^3)^2 + (u^3) + 1)*((u^3)^2 - (u^3) + 1)*((u^3) + 1)*((u^3) - 1))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

#[test]
fn test_trig_power_simplification() {
    // Use default_rules which has all necessary rules for this test
    // create_full_simplifier has a custom subset that doesn't fully simplify
    let mut simplifier = Simplifier::with_default_rules();

    // 8 * sin(x)^4 - (3 - 4*cos(2*x) + cos(4*x)) → 0
    // Uses TrigEvenPowerDifferenceRule: sin⁴ - cos⁴ → sin² - cos²
    let input = "8 * sin(x)^4 - (3 - 4*cos(2*x) + cos(4*x))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    assert_eq!(output, "0");
}

// ============================================================================
// Policy A+ Tests: simplify vs expand behavior
// See POLICY.md for full documentation
// ============================================================================

#[test]
fn test_policy_a_simplify_preserves_binomial_products() {
    // Policy A+: simplify() does NOT expand (x+1)(x+2)
    let mut simplifier = create_full_simplifier();
    let expr = parse("(x+1)*(x+2)", &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    // Should remain as product, not expanded to x^2 + 3x + 2
    // Note: canonical ordering may reorder to (1+x) and (2+x)
    assert!(
        !output.contains("x^(2)") && output.contains("*"),
        "simplify should preserve product form (not expand), got: {}",
        output
    );
}

#[test]
fn test_policy_a_simplify_applies_difference_of_squares() {
    // Policy A+: simplify() DOES apply diff of squares (x-1)(x+1) → x²-1
    let mut simplifier = create_full_simplifier();
    let expr = parse("(x-1)*(x+1)", &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    // Should reduce to x^2 - 1 (not remain as product)
    assert!(
        output.contains("x^(2)") || output.contains("x^2"),
        "simplify should reduce (x-1)(x+1), got: {}",
        output
    );
    assert!(
        !output.contains("(x + 1)") && !output.contains("(x - 1)"),
        "simplify should NOT preserve (x-1)(x+1) as product, got: {}",
        output
    );
}

#[test]
fn test_policy_a_expand_expands_binomial_products() {
    // Policy A+: expand() DOES expand (x+1)(x+2) → x² + 3x + 2
    let mut simplifier = create_full_simplifier();
    let expr = parse("expand((x+1)*(x+2))", &mut simplifier.context).unwrap();
    let (expanded, _) = simplifier.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: expanded
        }
    );
    // Should expand to polynomial form (contain x^2 term)
    assert!(
        output.contains("x^(2)") || output.contains("x^2"),
        "expand should produce polynomial with x², got: {}",
        output
    );
    // Should not remain as product
    assert!(
        !output.contains("(1 + x)") && !output.contains("(2 + x)"),
        "expand should NOT preserve product form, got: {}",
        output
    );
}

#[test]
fn test_sin_double_angle_additive_shift_equivalence() {
    let mut simplifier = create_full_simplifier();
    let input = parse("sin(2*x + 2*pi)", &mut simplifier.context).unwrap();
    let expected = parse("2*sin(x + pi)*cos(x + pi)", &mut simplifier.context).unwrap();
    assert_equivalent(&mut simplifier, input, expected);
}

#[test]
fn test_trig_half_angle_square_rational_context_equivalence() {
    let mut simplifier = create_full_simplifier();
    let input = parse("2*sin((1/u + 1/(u+1))/2)^2", &mut simplifier.context).unwrap();
    let expected = parse("1 - cos((1/u + 1/(u+1)))", &mut simplifier.context).unwrap();
    assert_equivalent(&mut simplifier, input, expected);
}

#[test]
fn test_exp_negative_symbolic_exponent_reciprocal_equivalence() {
    let mut simplifier = create_full_simplifier();
    let input = parse("exp(-(arctan(u)))", &mut simplifier.context).unwrap();
    let expected = parse("1/exp(arctan(u))", &mut simplifier.context).unwrap();
    assert_equivalent(&mut simplifier, input, expected);
}
