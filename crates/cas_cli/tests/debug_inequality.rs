use cas_ast::{Equation, RelOp, SolutionSet};
use cas_parser::parse;
use cas_solver::engine::Simplifier;
use cas_solver::solve;

#[test]
fn test_rational_inequality_1_over_x() {
    let mut simplifier = Simplifier::with_default_rules();

    // 1/x > 0 (from line 692-710 of output)
    let lhs = parse("1/x", &mut simplifier.context).unwrap();
    let rhs = parse("0", &mut simplifier.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };

    let result = solve(&eq, "x", &mut simplifier);
    println!("Result for 1/x > 0: {:?}", result);

    assert!(result.is_ok());
    let (solution, steps) = result.unwrap();

    println!("Solution: {:?}", solution);
    for step in steps {
        println!("Step: {}", step.description);
    }

    // Should be (0, infinity), not contain "undefined"
    match solution {
        SolutionSet::Continuous(interval) => {
            let min_str = format!("{:?}", interval.min);
            let max_str = format!("{:?}", interval.max);
            println!("Min: {}, Max: {}", min_str, max_str);
            assert!(!min_str.contains("undefined"));
            assert!(!max_str.contains("undefined"));
        }
        _ => println!("Got solution type: {:?}", solution),
    }
}

#[test]
fn test_linear_equation_with_fractions() {
    let mut simplifier = Simplifier::with_default_rules();

    // 1/3*x + 1/2*x = 5 (from line 338)
    let lhs = parse("1/3*x + 1/2*x", &mut simplifier.context).unwrap();
    let rhs = parse("5", &mut simplifier.context).unwrap();

    println!(
        "LHS: {}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: lhs
        }
    );
    println!(
        "RHS: {}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: rhs
        }
    );

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let result = solve(&eq, "x", &mut simplifier);
    println!("Result for 1/3*x + 1/2*x = 5: {:?}", result);

    assert!(result.is_ok());
    let (solution, steps) = result.unwrap();

    println!("Solution: {:?}", solution);
    for (i, step) in steps.iter().enumerate() {
        println!("Step {}: {}", i + 1, step.description);
        println!(
            "  After: {} {} {}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: step.equation_after.lhs
            },
            step.equation_after.op,
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: step.equation_after.rhs
            }
        );
    }

    // Should be {6}, not empty
    match solution {
        SolutionSet::Discrete(sols) => {
            if !sols.is_empty() {
                println!("Solutions:");
                for sol in &sols {
                    println!(
                        "  x = {}",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: *sol
                        }
                    );
                }
            }
            assert!(!sols.is_empty(), "Solution should not be empty!");
            println!("Number of solutions: {}", sols.len());
        }
        SolutionSet::Empty => panic!("Got empty set, should have solution x = 6"),
        _ => println!("Got solution type: {:?}", solution),
    }
}
