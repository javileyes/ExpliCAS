use cas_engine::rules::algebra::{
    AddFractionsRule, DifferenceOfSquaresRule, ExpandRule, FactorBasedLCDRule,
    FactorDifferenceSquaresRule, FactorRule, PullConstantFromFractionRule, SimplifyFractionRule,
    SimplifyMulDivRule,
};
use cas_engine::rules::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule};
use cas_engine::rules::calculus::{DiffRule, IntegrateRule};
use cas_engine::rules::canonicalization::{
    CanonicalizeAddRule, CanonicalizeDivRule, CanonicalizeMulRule, CanonicalizeNegationRule,
    CanonicalizeRootRule,
};
use cas_engine::rules::exponents::{
    EvaluatePowerRule, IdentityPowerRule, PowerPowerRule, PowerProductRule, PowerQuotientRule,
    ProductPowerRule,
};
use cas_engine::rules::functions::EvaluateAbsRule;
use cas_engine::rules::grouping::CollectRule;
use cas_engine::rules::logarithms::{EvaluateLogRule, ExponentialLogRule, SplitLogExponentsRule};
use cas_engine::rules::number_theory::NumberTheoryRule;
use cas_engine::rules::polynomial::{AnnihilationRule, CombineLikeTermsRule, DistributeRule};
use cas_engine::rules::trigonometry::{
    EvaluateTrigRule, PythagoreanIdentityRule, RecursiveTrigExpansionRule, TanToSinCosRule,
};
use cas_engine::Simplifier;

use cas_ast::{BoundType, DisplayExpr, Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_engine::solver::solve;
use cas_parser::parse;
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
    simplifier.add_rule(Box::new(cas_engine::rules::functions::AbsSquaredRule));
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(cas_engine::rules::trigonometry::AngleIdentityRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    simplifier.add_rule(Box::new(
        cas_engine::rules::trigonometry::AngleConsistencyRule,
    ));
    simplifier.add_rule(Box::new(
        cas_engine::rules::trigonometry::TrigPythagoreanSimplifyRule,
    ));
    simplifier.add_rule(Box::new(cas_engine::rules::trigonometry::DoubleAngleRule));
    simplifier.add_rule(Box::new(
        cas_engine::rules::trigonometry::TrigPhaseShiftRule,
    ));
    simplifier.add_rule(Box::new(
        cas_engine::rules::trigonometry::TrigEvenPowerDifferenceRule,
    ));
    simplifier.add_rule(Box::new(RecursiveTrigExpansionRule));
    // NOTE: CanonicalizeTrigSquareRule DISABLED - conflicts with TrigPythagoreanSimplifyRule
    // causing infinite loops (cos² → 1-sin² → cos² → ...)
    // This matches the default simplifier configuration.
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(EvaluateLogRule));
    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(SplitLogExponentsRule));
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(PowerProductRule));
    simplifier.add_rule(Box::new(PowerQuotientRule));
    simplifier.add_rule(Box::new(IdentityPowerRule));
    simplifier.add_rule(Box::new(
        cas_engine::rules::exponents::NegativeBasePowerRule,
    ));
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(ExpandRule));
    simplifier.add_rule(Box::new(
        cas_engine::rules::polynomial::BinomialExpansionRule,
    ));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(cas_engine::rules::algebra::NestedFractionRule));
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(AddFractionsRule));
    simplifier.add_rule(Box::new(SimplifyMulDivRule));
    simplifier.add_rule(Box::new(
        cas_engine::rules::algebra::RationalizeDenominatorRule,
    ));
    simplifier.add_rule(Box::new(
        cas_engine::rules::algebra::CancelCommonFactorsRule,
    ));
    simplifier.add_rule(Box::new(cas_engine::rules::algebra::SimplifySquareRootRule));
    simplifier.add_rule(Box::new(PullConstantFromFractionRule));
    simplifier.add_rule(Box::new(FactorBasedLCDRule));
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(CollectRule));
    simplifier.add_rule(Box::new(FactorDifferenceSquaresRule));
    simplifier.add_rule(Box::new(DifferenceOfSquaresRule)); // Policy A+: (a-b)(a+b) → a²-b²

    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(cas_engine::rules::arithmetic::MulZeroRule));

    simplifier.add_rule(Box::new(IntegrateRule));
    simplifier.add_rule(Box::new(DiffRule));
    simplifier.add_rule(Box::new(NumberTheoryRule));
    simplifier.add_rule(Box::new(cas_engine::rules::arithmetic::DivZeroRule));
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
    // ln(e^3) + (sin(x) + cos(x))^2 - sin(2*x) - (x^3 - 8)/(x - 2) + x^2 + 2*x
    // Expected: 0
    let input = "ln(e^3) + (sin(x) + cos(x))^2 - sin(2*x) - (x^3 - 8)/(x - 2) + x^2 + 2*x";
    let mut simplifier = create_full_simplifier();

    // All necessary rules are now in create_full_simplifier()
    // - DoubleAngleRule for sin(2x)
    // - BinomialExpansionRule for (sin+cos)^2
    // - DistributeRule for negative sign
    // - Log/Trig/Poly rules

    // Use expand() method which now correctly propagates expand_mode
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
    let mut simplifier = create_full_simplifier();
    let input = parse("e^(ln(x) + ln(y))", &mut simplifier.context).unwrap();
    let expected = parse("x * y", &mut simplifier.context).unwrap();
    assert_equivalent(&mut simplifier, input, expected);
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
    let input = "1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)";
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
    let input = "sin(x + y) - (sin(x)*cos(y) + cos(x)*sin(y))";
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
    // Should be No Solution because x=1 makes denominator zero.
    let mut simplifier = create_full_simplifier();
    let stmt =
        cas_parser::parse_statement("(x^2 - 1) / (x - 1) = 2", &mut simplifier.context).unwrap();

    if let cas_parser::Statement::Equation(eq) = stmt {
        let result = cas_engine::solver::solve(&eq, "x", &mut simplifier);

        match result {
            Ok((SolutionSet::Empty, _)) => (), // Correct
            Ok((SolutionSet::Discrete(sols), _)) => {
                // If it returns x=1, check if it's valid (it shouldn't be)
                let one = simplifier.context.num(1);
                for sol in sols {
                    if simplifier.are_equivalent(sol, one) {
                        panic!("FALLO GRAVE: El solver devolvió x=1, que indefine la ecuación original (división por cero).");
                    }
                }
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
    // 2. ln(|x+1|) -> ln(x+1) (EvaluateLogRule with domain assumption)
    // 3. ln(x+1) - ln(x+1) -> 0 (AnnihilationRule or CombineLikeTermsRule)
    let input = "ln(sqrt(x^2 + 2*x + 1)) - ln(x + 1)";
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
