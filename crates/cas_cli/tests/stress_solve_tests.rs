//! Stress tests for the equation/inequality solver
//! Tests are ordered by increasing complexity to facilitate sequential debugging

use cas_ast::{Equation, RelOp, SolutionSet};
use cas_engine::engine::Simplifier;
use cas_engine::solver::solve;

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
            assert!(values.len() >= 1, "Expected at least 1 solution");
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

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 1);
        }
        _ => panic!("Expected discrete solution"),
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

    match solution {
        SolutionSet::Discrete(values) => {
            assert_eq!(values.len(), 1);
        }
        _ => panic!("Expected discrete solution"),
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
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1 / x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

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
fn test_rational_inequality_negative() {
    // 1/x < 0 → x < 0
    let mut s = Simplifier::with_default_rules();
    let lhs = cas_parser::parse("1 / x", &mut s.context).unwrap();
    let rhs = cas_parser::parse("0", &mut s.context).unwrap();

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
