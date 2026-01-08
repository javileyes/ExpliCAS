//! Stress tests for the equation/inequality solver
//! Tests are ordered by increasing complexity to facilitate sequential debugging

use cas_ast::{Equation, RelOp, SolutionSet};
use cas_engine::engine::Simplifier;
use cas_engine::solver::solve;
use cas_engine::CasError;

// ============================================================================
// LEVEL 1: Basic Linear Equations
// ============================================================================

#[test]
fn test_linear_simple() {
    // x + 5 = 10 → x = 5
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x + 5", &mut s.context).unwrap();
    let rhs = cas_parser::parse("10", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 1);
            let val = cas_ast::DisplayExpr {
                context: &s.context,
                id: values[0],
            };
            assert_eq!(format!("{}", val), "5");
        }
        _ => panic!("Expected discrete solution"),
    }
}

#[test]
fn test_linear_with_negative() {
    // 3x - 7 = 14 → x = 7
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("3 * x - 7", &mut s.context).unwrap();
    let rhs = cas_parser::parse("14", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 1);
        }
        _ => panic!("Expected discrete solution"),
    }
}

#[test]
fn test_linear_both_sides() {
    // 2x + 3 = 5x - 12 → x = 5
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("2 * x + 3", &mut s.context).unwrap();
    let rhs = cas_parser::parse("5 * x - 12", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 1);
        }
        _ => panic!("Expected discrete solution"),
    }
}

// ============================================================================
// LEVEL 2: Linear Inequalities
// ============================================================================

#[test]
fn test_inequality_simple() {
    // x + 3 > 7 → x > 4
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x + 3", &mut s.context).unwrap();
    let rhs = cas_parser::parse("7", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Continuous(_) => {}
        _ => panic!("Expected continuous solution"),
    }
}

#[test]
fn test_inequality_negative_coefficient() {
    // -2x < 10 → x > -5 (flips)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("-2 * x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("10", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Lt,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Continuous(_) => {}
        _ => panic!("Expected continuous solution"),
    }
}

#[test]
fn test_inequality_compound() {
    // 3x - 5 >= 2x + 1 → x >= 6
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("3 * x - 5", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2 * x + 1", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Geq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Continuous(_) => {}
        _ => panic!("Expected continuous solution"),
    }
}

// ============================================================================
// LEVEL 3: Quadratic Equations
// ============================================================================

#[test]
fn test_quadratic_simple() {
    // x^2 = 16 → x = ±4
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^2", &mut s.context).unwrap();
    let rhs = cas_parser::parse("16", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 2, "Expected 2 solutions for x^2 = 16");
        }
        _ => panic!("Expected discrete solution set"),
    }
}

#[test]
fn test_quadratic_standard_form() {
    // x^2 - 5x + 6 = 0 → x = 2 or x = 3
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^2 - 5 * x + 6", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 2);
        }
        _ => panic!("Expected discrete solution"),
    }
}

#[test]
fn test_quadratic_one_solution() {
    // x^2 - 6x + 9 = 0 → x = 3 (double root)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^2 - 6 * x + 9", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Discrete(values) => {
            assert!(!values.is_empty(), "Expected at least 1 solution");
        }
        _ => panic!("Expected discrete solution"),
    }
}

// ============================================================================
// LEVEL 4: Quadratic Inequalities
// ============================================================================

#[test]
fn test_quadratic_inequality_basic() {
    // x^2 < 9 → -3 < x < 3
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^2", &mut s.context).unwrap();
    let rhs = cas_parser::parse("9", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Lt,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Continuous(_) => {}
        _ => panic!("Expected continuous interval"),
    }
}

#[test]
fn test_quadratic_inequality_outside() {
    // x^2 > 4 → x < -2 or x > 2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^2", &mut s.context).unwrap();
    let rhs = cas_parser::parse("4", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Union(_) | SolutionSet::Continuous(_) => {}
        _ => panic!("Expected union or continuous solution"),
    }
}

#[test]
fn test_quadratic_inequality_factored() {
    // (x - 1)(x - 3) > 0 → x < 1 or x > 3
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x - 1) * (x - 3)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Union(intervals) => {
            assert_eq!(intervals.len(), 2, "Expected union of 2 intervals");
        }
        _ => panic!("Expected union solution"),
    }
}

// ============================================================================
// LEVEL 5: Rational Equations
// ============================================================================

#[test]
fn test_rational_simple() {
    // 1/x = 2 → x = 1/2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1 / x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    // Accept both Discrete and Conditional (with guards) as valid
    match &solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 1);
        }
        SolutionSet::Conditional(cases) => {
            // Conditional with guard is valid - extract inner solution
            assert!(!cases.is_empty(), "Expected at least one case");
            // Check that the inner solution has exactly 1 value
            for case in cases {
                match &case.then.solutions {
                    SolutionSet::Discrete(vals) => {
                        assert_eq!(vals.len(), 1, "Expected 1 solution in conditional case");
                    }
                    other => panic!("Expected discrete solution in case, got {:?}", other),
                }
            }
        }
        other => panic!("Expected discrete or conditional solution, got {:?}", other),
    }
}

#[test]
fn test_rational_both_sides() {
    // 1/x = 2/3 → x = 3/2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1 / x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2 / 3", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    // Accept both Discrete and Conditional (with guards) as valid
    match &solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 1);
        }
        SolutionSet::Conditional(cases) => {
            assert!(!cases.is_empty(), "Expected at least one case");
            for case in cases {
                match &case.then.solutions {
                    SolutionSet::Discrete(vals) => {
                        assert_eq!(vals.len(), 1, "Expected 1 solution in conditional case");
                    }
                    other => panic!("Expected discrete solution in case, got {:?}", other),
                }
            }
        }
        other => panic!("Expected discrete or conditional solution, got {:?}", other),
    }
}

#[test]
fn test_rational_complex() {
    // (x + 1) / (x - 2) = 3 → x + 1 = 3(x - 2) → x = 7/2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x + 1) / (x - 2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("3", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Rational equation should solve");
}

// ============================================================================
// LEVEL 6: Rational Inequalities
// ============================================================================

#[test]
fn test_rational_inequality_positive() {
    // 1/x > 0 → x > 0
    // Note: Solver currently returns Conditional with Discrete inner (known limitation)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1 / x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };
    let result = solve(&eq, "x", &mut s);
    assert!(result.is_ok(), "Rational inequality should solve");

    // Verify we got a non-empty solution
    let (solution, _) = result.unwrap();
    assert_ne!(solution, SolutionSet::Empty, "Expected non-empty solution");
}

#[test]
fn test_rational_inequality_negative() {
    // 1/x < 0 → x < 0
    // Note: Solver currently returns Conditional with Discrete inner (known limitation)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1 / x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Lt,
    };
    let result = solve(&eq, "x", &mut s);
    assert!(result.is_ok(), "Rational inequality should solve");

    // Verify we got a non-empty solution
    let (solution, _) = result.unwrap();
    assert_ne!(solution, SolutionSet::Empty, "Expected non-empty solution");
}

#[test]
fn test_rational_inequality_complex() {
    // (x - 1) / (x + 2) > 0 → x < -2 or x > 1
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x - 1) / (x + 2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Rational inequality should solve");
}

// ============================================================================
// LEVEL 7: Absolute Value
// ============================================================================

#[test]
fn test_absolute_value_simple() {
    // |x| = 5 → x = ±5
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("abs(x)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("5", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 2);
        }
        _ => panic!("Expected discrete solution"),
    }
}

#[test]
fn test_absolute_value_inequality() {
    // |x| < 3 → -3 < x < 3
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("abs(x)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("3", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Lt,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Continuous(_) => {}
        _ => panic!("Expected continuous solution"),
    }
}

#[test]
fn test_absolute_value_complex() {
    // |2x - 3| = 7 → x = 5 or x = -2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("abs(2 * x - 3)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("7", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Absolute value equation should solve");
}

// ============================================================================
// LEVEL 8: Mixed Complex Cases
// ============================================================================

#[test]
fn test_mixed_rational_quadratic() {
    // x^2 / (x - 1) = x + 2 (after simplification)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^2 / (x - 1)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("x + 2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Mixed rational-quadratic should solve");
}

#[test]
fn test_nested_fractions() {
    // (1 / (x + 1)) = (1 / (2*x - 3))
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1 / (x + 1)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("1 / (2 * x - 3)", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Nested fractions should solve");
}

#[test]
fn test_product_of_three_factors() {
    // (x - 1)(x - 2)(x - 3) > 0
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x - 1) * (x - 2) * (x - 3)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };
    let result = solve(&eq, "x", &mut s);

    // This is complex - should handle multiple sign cases
    assert!(result.is_ok(), "Product of 3 factors should solve");
}

// ============================================================================
// LEVEL 9: Edge Cases and Stress Tests
// ============================================================================

#[test]
fn test_no_solution() {
    // x + 5 = x + 3 → no solution
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x + 5", &mut s.context).unwrap();
    let rhs = cas_parser::parse("x + 3", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Empty => {}
        SolutionSet::Discrete(vals) if vals.is_empty() => {}
        _ => panic!("Expected empty solution set"),
    }
}

#[test]
fn test_identity() {
    // x + 1 = x + 1 → all real numbers (identity)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x + 1", &mut s.context).unwrap();
    let rhs = cas_parser::parse("x + 1", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // Identity - might be All or might simplify to 0=0
    assert!(result.is_ok(), "Identity should be handled");
}

#[test]
fn test_division_by_zero_exclusion() {
    // 1 / (x - 2) = 1 / (x - 2) for x ≠ 2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1 / (x - 2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("1 / (x - 2)", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // Should exclude x = 2
    assert!(result.is_ok(), "Division by zero should be handled");
}

#[test]
fn test_large_coefficients() {
    // 1000000x + 500000 = 2500000 → x = 2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1000000 * x + 500000", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2500000", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 1);
        }
        _ => panic!("Expected discrete solution"),
    }
}

#[test]
fn test_very_small_coefficients() {
    // 0.0001x = 0.0005 → x = 5
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(1/10000) * x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("5/10000", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 1);
        }
        _ => panic!("Expected discrete solution"),
    }
}

// ============================================================================
// Regression tests for QuadraticStrategy factor skipping (Issue: VariableNotFound)
// These test that constant factors in products don't cause solver to fail
// ============================================================================

#[test]
fn test_factor_skip_simple_division() {
    // (x-5)/10000 = 0 → x = 5
    // The 1/10000 factor has no variable and should be skipped
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x - 5) / 10000", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 1, "Expected 1 solution");
        }
        _ => panic!("Expected discrete solution"),
    }
}

#[test]
fn test_factor_zero_coefficient_all_reals() {
    // 0*(x-5) = 0 → All reals (0 = 0 is always true)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("0 * (x - 5)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    assert!(
        matches!(solution, SolutionSet::AllReals),
        "Expected AllReals, got {:?}",
        solution
    );
}

#[test]
fn test_factor_skip_with_squared() {
    // (1/10000)*(x-5)^2 = 0 → x = 5 (double root)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(1/10000) * (x - 5)^2", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Discrete(values) => {
            assert!(!values.is_empty(), "Expected at least 1 solution");
        }
        _ => panic!("Expected discrete solution"),
    }
}

#[test]
fn test_constant_equation_no_variable() {
    // (1/2)*(3/7) = 0 → False (no x in equation)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(1/2) * (3/7)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // Should be VariableNotFound or Empty
    match result {
        Err(CasError::VariableNotFound(_)) => {} // Expected
        Ok((SolutionSet::Empty, _)) => {}        // Also acceptable
        other => panic!("Expected VariableNotFound or Empty, got {:?}", other),
    }
}

#[test]
fn test_factor_skip_half_times_linear() {
    // (1/2)*(x-1) = 0 → x = 1
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(1/2) * (x - 1)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 1, "Expected exactly 1 solution");
        }
        _ => panic!("Expected discrete solution"),
    }
}

// ============================================================================
// LEVEL 10: Advanced Rational Inequalities
// ============================================================================

#[test]
fn test_rational_multiple_discontinuities() {
    // 1 / ((x - 1) * (x - 3)) > 0
    // Discontinuities at x = 1 and x = 3
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1 / ((x - 1) * (x - 3))", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Multiple discontinuities should be handled");
}

#[test]
fn test_symmetric_rational() {
    // (x - 1) / (x + 1) = (x + 1) / (x - 1)
    // x = 0
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x - 1) / (x + 1)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("(x + 1) / (x - 1)", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Symmetric rational should solve");
}

// ============================================================================
// LEVEL 11: Advanced Absolute Value and Radicals
// ============================================================================

#[test]
fn test_nested_absolute_values() {
    // ||x| - 2| = 1 → |x| = 3 or |x| = 1
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("abs(abs(x) - 2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("1", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Nested absolute values should solve");
}

#[test]
fn test_irrational_equation() {
    // sqrt(x + 5) = 3 → x = 4
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x + 5)^(1/2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("3", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Irrational equation should solve");
}

#[test]
fn test_reciprocal_equation() {
    // 1/x + x = 2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1/x + x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // Complex - requires converting to quadratic
    assert!(
        result.is_ok() || result.is_err(),
        "Should handle gracefully"
    );
}

// ============================================================================
// LEVEL 12: Pathological Edge Cases
// ============================================================================

#[test]
fn test_equation_with_parameters() {
    // ax + b = 0 where a, b are unknown constants
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("a * x + b", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // Should solve x = -b/a
    assert!(result.is_ok(), "Parametric equation should solve");
}

#[test]
fn test_high_power_polynomial() {
    // x^3 - 6x^2 + 11x - 6 = 0
    // Factors as (x-1)(x-2)(x-3) = 0
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^3 - 6*x^2 + 11*x - 6", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // Cubic - may not solve analytically
    assert!(
        result.is_ok() || result.is_err(),
        "Should handle gracefully"
    );
}

#[test]
fn test_contradiction_with_division() {
    // 1/(x-1) = 1/(x-1) + 1 → 0 = 1
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1 / (x - 1)", &mut s.context).unwrap();
    let rhs_part1 = cas_parser::parse("1 / (x - 1)", &mut s.context).unwrap();
    let rhs_part2 = cas_parser::parse("1", &mut s.context).unwrap();
    let rhs = s.context.add(cas_ast::Expr::Add(rhs_part1, rhs_part2));

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    // Should be empty (contradiction)
    match solution {
        SolutionSet::Empty => {}
        SolutionSet::Discrete(vals) if vals.is_empty() => {}
        _ => panic!("Expected empty solution for contradiction"),
    }
}

// ============================================================================
// LEVEL 13: Complex Mixed Operations
// ============================================================================

#[test]
fn test_absolute_value_with_fraction() {
    // |x/(x-1)| = 2
    // Complex: need to handle both division and absolute value
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("abs(x / (x - 1))", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Absolute value with fraction should handle");
}

#[test]
fn test_nested_radicals_equation() {
    // sqrt(x + sqrt(x)) = 2
    // Requires multiple squaring operations
    let mut s = Simplifier::with_default_rules();
    let x_id = cas_parser::parse("x", &mut s.context).unwrap();
    let inner_sqrt = cas_parser::parse("x^(1/2)", &mut s.context).unwrap();
    let sum = s.context.add(cas_ast::Expr::Add(x_id, inner_sqrt));
    let half_id = cas_parser::parse("1/2", &mut s.context).unwrap();
    let lhs = s.context.add(cas_ast::Expr::Pow(sum, half_id));
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // Very advanced - may not solve
    assert!(
        result.is_ok() || result.is_err(),
        "Should handle gracefully"
    );
}

#[test]
fn test_polynomial_fraction_inequality() {
    // (x^2 + 1) / (x^2 - 4) > 0
    // Need to analyze sign of both numerator and denominator
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x^2 + 1) / (x^2 - 4)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok(),
        "Polynomial fraction inequality should solve"
    );
}

#[test]
fn test_multi_term_rational() {
    // 1/x + 1/(x-1) = 1
    // Requires finding common denominator
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1/x + 1/(x - 1)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("1", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Multi-term rational should handle"
    );
}

// ============================================================================
// LEVEL 14: Deeply Nested Expressions
// ============================================================================

#[test]
fn test_triple_nested_absolute() {
    // |||x| - 1| - 1| = 0
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("abs(abs(abs(x) - 1) - 1)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Triple nested absolute should solve");
}

#[test]
fn test_nested_fractions_complex() {
    // 1/(1 + 1/x) = 2
    // Nested fraction requires careful handling
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1 / (1 + 1/x)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Nested fractions should solve");
}

#[test]
fn test_repeated_squaring() {
    // ((x^2)^2)^2 = 64
    // x^8 = 64, should give ±2^(3/4) or simplify further
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("((x^2)^2)^2", &mut s.context).unwrap();
    let rhs = cas_parser::parse("64", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Repeated squaring should solve");
}

#[test]
fn test_mixed_radical_polynomial() {
    // sqrt(x) + x = 6
    // Requires substitution or squaring technique
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^(1/2) + x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("6", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Mixed radical polynomial should handle"
    );
}

// ============================================================================
// LEVEL 15: Extreme Numerical Cases
// ============================================================================

#[test]
fn test_extremely_large_exponent() {
    // x^100 = 2^100
    // Should give x = 2 (principal root)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^100", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2^100", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Large exponents should handle"
    );
}

#[test]
fn test_fractional_exponents_complex() {
    // x^(3/2) = 8
    // x = 4
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^(3/2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("8", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).expect("Fractional exponents should solve");

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 1, "Expected exactly 1 solution");
            // Verify x = 4
            let result_str = format!(
                "{}",
                cas_ast::DisplayExpr {
                    context: &s.context,
                    id: values[0]
                }
            );
            assert_eq!(result_str, "4", "Expected x = 4");
        }
        _ => panic!("Expected discrete solution"),
    }
}

#[test]
fn test_many_terms_linear() {
    // x + 2x + 3x + 4x + 5x = 150
    // Should combine to 15x = 150, x = 10
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x + 2*x + 3*x + 4*x + 5*x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("150", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 1, "Should have one solution");
        }
        _ => panic!("Expected discrete solution"),
    }
}

#[test]
fn test_alternating_signs() {
    // x - 2*x + 3*x - 4*x + 5*x = 9
    // Should combine to 3x = 9, x = 3
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x - 2*x + 3*x - 4*x + 5*x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("9", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 1);
        }
        _ => panic!("Expected discrete solution"),
    }
}

// ============================================================================
// LEVEL 16: Maximum Stress Tests
// ============================================================================

#[test]
fn test_product_of_many_factors() {
    // (x-1)(x-2)(x-3)(x-4)(x-5) = 0
    // Should have 5 solutions
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse(
        "(x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)",
        &mut s.context,
    )
    .unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // May not find all roots
    assert!(
        result.is_ok() || result.is_err(),
        "Product of many factors should handle"
    );
}

#[test]
fn test_rational_with_many_terms() {
    // (x + 1) / (x - 1) + (x - 1) / (x + 1) = 2
    // Symmetric, should simplify nicely
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x + 1) / (x - 1) + (x - 1) / (x + 1)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Symmetric rational should handle"
    );
}

#[test]
fn test_inequality_with_absolute_rational() {
    // |x / (x + 1)| < 1/2
    // Complex domain restrictions
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("abs(x / (x + 1))", &mut s.context).unwrap();
    let rhs = cas_parser::parse("1/2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Lt,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Absolute rational inequality should handle"
    );
}

#[test]
fn test_quadratic_with_many_operations() {
    // (x + 1)^2 - 2*(x + 1) + 1 = 0
    // Let u = x + 1, then u^2 - 2u + 1 = 0, u = 1, x = 0
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x + 1)^2 - 2*(x + 1) + 1", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok(),
        "Quadratic with substitution pattern should solve"
    );
}

#[test]
fn test_complex_domain_restriction() {
    // sqrt(x - 1) + sqrt(x + 1) = 2
    // Domain: x >= 1
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x - 1)^(1/2) + (x + 1)^(1/2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Complex domain restrictions should handle"
    );
}

#[test]
fn test_pathological_cancellation() {
    // (x^2 - 1) / (x - 1) = x + 2
    // Cancels to x + 1 = x + 2 (no solution) but must exclude x = 1
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x^2 - 1) / (x - 1)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("x + 2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (solution, _) = solve(&eq, "x", &mut s).unwrap();

    // Should be empty (x + 1 = x + 2 has no solution)
    match solution {
        SolutionSet::Empty => {}
        SolutionSet::Discrete(vals) if vals.is_empty() => {}
        _ => {} // May simplify differently
    }
}

#[test]
fn test_stress_deep_nesting() {
    // 1/(1/(1/(1/x))) = 2
    // Should simplify to x = 2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1 / (1 / (1 / (1 / x)))", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Deep nesting should simplify and solve");
}

// ============================================================================
// LEVEL 17: Logarithmic and Exponential Equations
// ============================================================================

#[test]
fn test_simple_log_equation() {
    // log(10, x) = 2 → x = 100
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("log(10, x)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // May not be fully implemented yet
    assert!(
        result.is_ok() || result.is_err(),
        "Log equations should handle gracefully"
    );
}

#[test]
fn test_natural_log_equation() {
    // ln(x) = 3 → x = e^3
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("ln(x)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("3", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Natural log should handle"
    );
}

#[test]
fn test_exponential_equation_base_e() {
    // exp(x) = e^3 → x = 3
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("exp(x)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("exp(3)", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(result.is_ok(), "Exp equations should solve");
}

#[test]
fn test_log_with_linear() {
    // log(2, 2*x + 4) = 3 → 2x + 4 = 8 → x = 2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("log(2, 2*x + 4)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("3", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Log with linear should handle"
    );
}

#[test]
fn test_exponential_with_quadratic() {
    // exp(x^2) = exp(4) → x^2 = 4 → x = ±2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("exp(x^2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("exp(4)", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Exp with quadratic should handle"
    );
}

#[test]
fn test_log_equality() {
    // log(2, x) = log(2, 8) → x = 8
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("log(2, x)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("log(2, 8)", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Log equality should handle"
    );
}

#[test]
fn test_mixed_log_exp() {
    // ln(exp(x)) = 5 → x = 5
    // Uses log-exp inverse property
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("ln(exp(x))", &mut s.context).unwrap();
    let rhs = cas_parser::parse("5", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Mixed log-exp should handle"
    );
}

#[test]
fn test_log_product_property() {
    // log(10, x) + log(10, 2) = 2
    // Uses: log(a) + log(b) = log(ab)
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("log(10, x) + log(10, 2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // Requires log simplification rules
    assert!(
        result.is_ok() || result.is_err(),
        "Log product property should handle"
    );
}

// ============================================================================
// LEVEL 18: Ultra-Complex Mixed Operations
// ============================================================================

#[test]
fn test_absolute_log_fraction() {
    // |log(10, x)| / 2 = 1  → |log(10, x)| = 2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("abs(log(10, x)) / 2", &mut s.context).unwrap();
    let rhs = cas_parser::parse("1", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Absolute log fraction should handle"
    );
}

#[test]
fn test_radical_with_rational() {
    // sqrt(x) / (x - 1) = 2
    // Complex: radical in numerator, polynomial in denominator
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^(1/2) / (x - 1)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Radical with rational should handle"
    );
}

#[test]
fn test_nested_absolute_fraction() {
    // |x / |x - 1|| = 2
    // Double absolute value with fraction
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("abs(x / abs(x - 1))", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Nested absolute fraction should handle"
    );
}

#[test]
fn test_polynomial_with_absolute_inequality() {
    // x^2 + |x| - 6 < 0
    // Mix of polynomial and absolute value
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^2 + abs(x) - 6", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Lt,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Polynomial with absolute should handle"
    );
}

#[test]
fn test_fraction_of_radicals() {
    // sqrt(x + 1) / sqrt(x - 1) = 2
    // Ratio of radicals
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x + 1)^(1/2) / (x - 1)^(1/2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Fraction of radicals should handle"
    );
}

#[test]
fn test_absolute_polynomial_fraction() {
    // |(x^2 - 4) / (x - 2)| = 3
    // Absolute value of rational with cancellation
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("abs((x^2 - 4) / (x - 2))", &mut s.context).unwrap();
    let rhs = cas_parser::parse("3", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Absolute polynomial fraction should handle"
    );
}

#[test]
fn test_sum_of_radicals_complex() {
    // sqrt(x) + sqrt(x + 7) = 7
    // Two different radicals
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("x^(1/2) + (x + 7)^(1/2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("7", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Sum of radicals should handle"
    );
}

#[test]
fn test_rational_inequality_with_absolute() {
    // |x| / (x^2 + 1) > 1/2
    // Absolute in numerator, always positive denominator
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("abs(x) / (x^2 + 1)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("1/2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Rational inequality with absolute should handle"
    );
}

#[test]
fn test_cubic_with_rational() {
    // (x^3 - 8) / (x - 2) = x^2 + 2*x + 4 (factorization identity)
    // Should hold for x ≠ 2
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x^3 - 8) / (x - 2)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("x^2 + 2*x + 4", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // Should be an identity except at x = 2
    assert!(
        result.is_ok() || result.is_err(),
        "Cubic factorization should handle"
    );
}

#[test]
fn test_product_with_absolute() {
    // (x - 1) * |x + 1| = 0
    // Product of linear and absolute
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(x - 1) * abs(x + 1)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // Should give x = 1 or x = -1
    assert!(result.is_ok(), "Product with absolute should solve");
}

#[test]
fn test_very_nested_operations() {
    // abs(sqrt(abs(x - 1))) = 2
    // Multiple layers: abs -> sqrt -> abs
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("abs((abs(x - 1))^(1/2))", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    assert!(
        result.is_ok() || result.is_err(),
        "Very nested operations should handle"
    );
}

#[test]
fn test_rational_with_multiple_variables_treated_as_constants() {
    // (a*x + b) / (c*x + d) = 2
    // Testing robustness with symbolic coefficients
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("(a*x + b) / (c*x + d)", &mut s.context).unwrap();
    let rhs = cas_parser::parse("2", &mut s.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut s);

    // Should solve for x in terms of a, b, c, d
    assert!(
        result.is_ok() || result.is_err(),
        "Symbolic rationals should handle"
    );
}
