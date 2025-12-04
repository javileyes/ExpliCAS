use cas_ast::{BoundType, DisplayExpr, Equation, RelOp, SolutionSet};
use cas_engine::solver::solve;
use cas_engine::Simplifier;
use cas_parser::parse;

/// Crea un simplifier con las reglas por defecto
fn create_simplifier() -> Simplifier {
    Simplifier::with_default_rules()
}

// ========================================
// ECUACIONES CON VALOR ABSOLUTO
// ========================================

#[test]
fn test_abs_equation_basic() {
    // |x| = 5 -> {-5, 5}
    let mut simplifier = create_simplifier();
    let lhs = parse("|x|", &mut simplifier.context).unwrap();
    let rhs = parse("5", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Discrete(ref solutions) = result {
        assert_eq!(solutions.len(), 2);
        let sol_strs: Vec<String> = solutions
            .iter()
            .map(|&s| {
                format!(
                    "{}",
                    DisplayExpr {
                        context: &simplifier.context,
                        id: s
                    }
                )
            })
            .collect();
        assert!(sol_strs.contains(&"5".to_string()));
        assert!(sol_strs.contains(&"-5".to_string()));
    } else {
        panic!("Expected discrete solution set, got {:?}", result);
    }
}

#[test]
fn test_abs_equation_transformed() {
    // |x - 2| = 3 -> {5, -1}
    let mut simplifier = create_simplifier();
    let lhs = parse("|x - 2|", &mut simplifier.context).unwrap();
    let rhs = parse("3", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Discrete(ref solutions) = result {
        assert_eq!(solutions.len(), 2);
        let sol_strs: Vec<String> = solutions
            .iter()
            .map(|&s| {
                format!(
                    "{}",
                    DisplayExpr {
                        context: &simplifier.context,
                        id: s
                    }
                )
            })
            .collect();
        assert!(sol_strs.contains(&"5".to_string()));
        assert!(sol_strs.contains(&"-1".to_string()));
    } else {
        panic!("Expected discrete solution set");
    }
}

#[test]
fn test_abs_equation_scaled() {
    // |2*x| = 6 -> {3, -3}
    let mut simplifier = create_simplifier();
    let lhs = parse("|2*x|", &mut simplifier.context).unwrap();
    let rhs = parse("6", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Discrete(ref solutions) = result {
        assert_eq!(solutions.len(), 2);
        let sol_strs: Vec<String> = solutions
            .iter()
            .map(|&s| {
                format!(
                    "{}",
                    DisplayExpr {
                        context: &simplifier.context,
                        id: s
                    }
                )
            })
            .collect();
        assert!(sol_strs.contains(&"3".to_string()));
        assert!(sol_strs.contains(&"-3".to_string()));
    } else {
        panic!("Expected discrete solution set");
    }
}

// ========================================
// INECUACIONES CON VALOR ABSOLUTO
// ========================================

#[test]
fn test_abs_inequality_less_than() {
    // |x| < 5 -> (-5, 5)
    let mut simplifier = create_simplifier();
    let lhs = parse("|x|", &mut simplifier.context).unwrap();
    let rhs = parse("5", &mut simplifier.context).unwrap();
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
            "-5"
        );
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: max
                }
            ),
            "5"
        );
        assert_eq!(interval.min_type, BoundType::Open);
        assert_eq!(interval.max_type, BoundType::Open);
    } else {
        panic!("Expected continuous solution");
    }
}

#[test]
fn test_abs_inequality_less_equal() {
    // |x| <= 3 -> [-3, 3]
    let mut simplifier = create_simplifier();
    let lhs = parse("|x|", &mut simplifier.context).unwrap();
    let rhs = parse("3", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Leq,
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
            "-3"
        );
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: max
                }
            ),
            "3"
        );
        assert_eq!(interval.min_type, BoundType::Closed);
        assert_eq!(interval.max_type, BoundType::Closed);
    } else {
        panic!("Expected continuous solution");
    }
}

#[test]
fn test_abs_inequality_greater_than() {
    // |x| > 2 -> (-∞, -2) U (2, ∞)
    let mut simplifier = create_simplifier();
    let lhs = parse("|x|", &mut simplifier.context).unwrap();
    let rhs = parse("2", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Union(intervals) = result {
        assert_eq!(intervals.len(), 2);
        // First interval: (-∞, -2)
        let (max1, _) = simplifier.simplify(intervals[0].max);
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
        // Second interval: (2, ∞)
        let (min2, _) = simplifier.simplify(intervals[1].min);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: min2
                }
            ),
            "2"
        );
    } else {
        panic!("Expected union of intervals, got {:?}", result);
    }
}

#[test]
fn test_abs_inequality_transformed() {
    // |x - 3| < 2 -> (1, 5)
    let mut simplifier = create_simplifier();
    let lhs = parse("|x - 3|", &mut simplifier.context).unwrap();
    let rhs = parse("2", &mut simplifier.context).unwrap();
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
            "1"
        );
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: max
                }
            ),
            "5"
        );
    } else {
        panic!("Expected continuous solution");
    }
}

// ========================================
// ECUACIONES LINEALES CON VARIABLES EN AMBOS LADOS
// ========================================

#[test]
fn test_linear_eq_both_sides_simple() {
    // 2*x - 5 = x + 3 -> x = 8
    let mut simplifier = create_simplifier();
    let lhs = parse("2*x - 5", &mut simplifier.context).unwrap();
    let rhs = parse("x + 3", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Discrete(ref solutions) = result {
        assert_eq!(solutions.len(), 1);
        let (sol, _) = simplifier.simplify(solutions[0]);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: sol
                }
            ),
            "8"
        );
    } else {
        panic!("Expected discrete solution");
    }
}

#[test]
fn test_linear_eq_both_sides_complex() {
    // 3*x + 2 = x - 4 -> x = -3
    let mut simplifier = create_simplifier();
    let lhs = parse("3*x + 2", &mut simplifier.context).unwrap();
    let rhs = parse("x - 4", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Discrete(ref solutions) = result {
        assert_eq!(solutions.len(), 1);
        let (sol, _) = simplifier.simplify(solutions[0]);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: sol
                }
            ),
            "-3"
        );
    } else {
        panic!("Expected discrete solution");
    }
}

#[test]
fn test_linear_eq_both_sides_coefficients() {
    // 5*x - 3 = 2*x + 6 -> x = 3
    let mut simplifier = create_simplifier();
    let lhs = parse("5*x - 3", &mut simplifier.context).unwrap();
    let rhs = parse("2*x + 6", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Discrete(ref solutions) = result {
        assert_eq!(solutions.len(), 1);
        let (sol, _) = simplifier.simplify(solutions[0]);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: sol
                }
            ),
            "3"
        );
    } else {
        panic!("Expected discrete solution");
    }
}

// ========================================
// INECUACIONES LINEALES CON VARIABLES EN AMBOS LADOS
// ========================================

#[test]
fn test_linear_ineq_both_sides_gt() {
    // 2*x > x + 3 -> x > 3
    let mut simplifier = create_simplifier();
    let lhs = parse("2*x", &mut simplifier.context).unwrap();
    let rhs = parse("x + 3", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Continuous(interval) = result {
        let (min, _) = simplifier.simplify(interval.min);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: min
                }
            ),
            "3"
        );
        assert_eq!(interval.min_type, BoundType::Open);
    } else {
        panic!("Expected continuous solution, got {:?}", result);
    }
}

#[test]
fn test_linear_ineq_both_sides_lt() {
    // 3*x + 1 < 2*x + 5 -> x < 4
    let mut simplifier = create_simplifier();
    let lhs = parse("3*x + 1", &mut simplifier.context).unwrap();
    let rhs = parse("2*x + 5", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Lt,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Continuous(interval) = result {
        let (max, _) = simplifier.simplify(interval.max);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: max
                }
            ),
            "4"
        );
        assert_eq!(interval.max_type, BoundType::Open);
    } else {
        panic!("Expected continuous solution, got {:?}", result);
    }
}

#[test]
fn test_linear_ineq_both_sides_geq() {
    // 4*x - 2 >= x + 7 -> x >= 3
    let mut simplifier = create_simplifier();
    let lhs = parse("4*x - 2", &mut simplifier.context).unwrap();
    let rhs = parse("x + 7", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Geq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Continuous(interval) = result {
        let (min, _) = simplifier.simplify(interval.min);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: min
                }
            ),
            "3"
        );
        assert_eq!(interval.min_type, BoundType::Closed);
    } else {
        panic!("Expected continuous solution, got {:?}", result);
    }
}

// ========================================
// ECUACIONES CUADRÁTICAS
// ========================================

#[test]
fn test_quadratic_simple() {
    // x^2 = 9 -> {-3, 3}
    let mut simplifier = create_simplifier();
    let lhs = parse("x^2", &mut simplifier.context).unwrap();
    let rhs = parse("9", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Discrete(ref solutions) = result {
        assert_eq!(solutions.len(), 2);
        let sol_strs: Vec<String> = solutions
            .iter()
            .map(|&s| {
                let (simp, _) = simplifier.simplify(s);
                format!(
                    "{}",
                    DisplayExpr {
                        context: &simplifier.context,
                        id: simp
                    }
                )
            })
            .collect();
        assert!(sol_strs.contains(&"3".to_string()));
        assert!(sol_strs.contains(&"-3".to_string()));
    } else {
        panic!("Expected discrete solution set");
    }
}

#[test]
fn test_quadratic_factored() {
    // x^2 + x = 0 -> {0, -1}
    let mut simplifier = create_simplifier();
    let lhs = parse("x^2 + x", &mut simplifier.context).unwrap();
    let rhs = parse("0", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Discrete(ref solutions) = result {
        assert_eq!(solutions.len(), 2);
        let sol_strs: Vec<String> = solutions
            .iter()
            .map(|&s| {
                let (simp, _) = simplifier.simplify(s);
                format!(
                    "{}",
                    DisplayExpr {
                        context: &simplifier.context,
                        id: simp
                    }
                )
            })
            .collect();
        assert!(sol_strs.contains(&"0".to_string()));
        assert!(sol_strs.contains(&"-1".to_string()));
    } else {
        panic!("Expected discrete solution set");
    }
}

#[test]
fn test_quadratic_standard_form() {
    // x^2 - 5*x + 6 = 0 -> {2, 3}
    let mut simplifier = create_simplifier();
    let lhs = parse("x^2 - 5*x + 6", &mut simplifier.context).unwrap();
    let rhs = parse("0", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Discrete(ref solutions) = result {
        assert_eq!(solutions.len(), 2);
        let sol_strs: Vec<String> = solutions
            .iter()
            .map(|&s| {
                let (simp, _) = simplifier.simplify(s);
                format!(
                    "{}",
                    DisplayExpr {
                        context: &simplifier.context,
                        id: simp
                    }
                )
            })
            .collect();
        assert!(sol_strs.contains(&"2".to_string()));
        assert!(sol_strs.contains(&"3".to_string()));
    } else {
        panic!("Expected discrete solution set");
    }
}

// ========================================
// INECUACIONES CUADRÁTICAS
// ========================================

#[test]
fn test_quadratic_ineq_simple() {
    // x^2 < 4 -> (-2, 2)
    let mut simplifier = create_simplifier();
    let lhs = parse("x^2", &mut simplifier.context).unwrap();
    let rhs = parse("4", &mut simplifier.context).unwrap();
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
        assert_eq!(interval.min_type, BoundType::Open);
        assert_eq!(interval.max_type, BoundType::Open);
    } else {
        panic!("Expected continuous solution");
    }
}

#[test]
fn test_quadratic_ineq_greater() {
    // x^2 > 9 -> (-∞, -3) U (3, ∞)
    let mut simplifier = create_simplifier();
    let lhs = parse("x^2", &mut simplifier.context).unwrap();
    let rhs = parse("9", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Union(intervals) = result {
        assert_eq!(intervals.len(), 2);
        let (max1, _) = simplifier.simplify(intervals[0].max);
        let (min2, _) = simplifier.simplify(intervals[1].min);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: max1
                }
            ),
            "-3"
        );
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: min2
                }
            ),
            "3"
        );
    } else {
        panic!("Expected union of intervals, got {:?}", result);
    }
}

#[test]
fn test_quadratic_ineq_factored() {
    // (x - 1)*(x - 3) > 0 -> (-∞, 1) U (3, ∞)
    let mut simplifier = create_simplifier();
    let lhs = parse("(x - 1)*(x - 3)", &mut simplifier.context).unwrap();
    let rhs = parse("0", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Union(intervals) = result {
        assert_eq!(intervals.len(), 2);
        let (max1, _) = simplifier.simplify(intervals[0].max);
        let (min2, _) = simplifier.simplify(intervals[1].min);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: max1
                }
            ),
            "1"
        );
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: min2
                }
            ),
            "3"
        );
    } else {
        panic!("Expected union of intervals, got {:?}", result);
    }
}

// ========================================
// INECUACIONES LINEALES SIMPLES
// ========================================

#[test]
fn test_ineq_flip_sign() {
    // -2*x > 10 -> x < -5
    let mut simplifier = create_simplifier();
    let lhs = parse("-2*x", &mut simplifier.context).unwrap();
    let rhs = parse("10", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Continuous(interval) = result {
        let (max, _) = simplifier.simplify(interval.max);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: max
                }
            ),
            "-5"
        );
        assert_eq!(interval.max_type, BoundType::Open);
    } else {
        panic!("Expected continuous solution");
    }
}

#[test]
fn test_ineq_negative_both_sides() {
    // -x < 5 -> x > -5
    let mut simplifier = create_simplifier();
    let lhs = parse("-x", &mut simplifier.context).unwrap();
    let rhs = parse("5", &mut simplifier.context).unwrap();
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Lt,
    };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Continuous(interval) = result {
        let (min, _) = simplifier.simplify(interval.min);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: min
                }
            ),
            "-5"
        );
        assert_eq!(interval.min_type, BoundType::Open);
    } else {
        panic!("Expected continuous solution");
    }
}
