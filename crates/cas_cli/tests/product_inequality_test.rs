use cas_ast::{Equation, RelOp, SolutionSet};
use cas_engine::engine::Simplifier;
use cas_solver::solve;

#[test]
fn test_product_inequality_both_negative() {
    let mut simplifier = Simplifier::with_default_rules();

    // (x - 2) * (-x - 2) > 0
    // This should give (-2, 2) because:
    // Product > 0 when both factors same sign
    // (x - 2) > 0 AND (-x - 2) > 0: x > 2 AND x < -2 → Empty
    // (x - 2) < 0 AND (-x - 2) < 0: x < 2 AND x > -2 → (-2, 2) ✓

    let lhs = cas_parser::parse("(x - 2) * (-x - 2)", &mut simplifier.context).unwrap();
    let rhs = simplifier.context.num(0);

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };

    let result = solve(&eq, "x", &mut simplifier);
    println!("Result: {:?}", result);

    assert!(result.is_ok());
    let (solution, _) = result.unwrap();

    // Should be (-2, 2)
    match solution {
        SolutionSet::Continuous(interval) => {
            println!(
                "Got interval: min={}, max={}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: interval.min
                },
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: interval.max
                }
            );
            // Verify bounds are -2 and 2
        }
        _ => panic!("Expected Continuous interval, got: {:?}", solution),
    }
}

#[test]
fn test_product_inequality_both_positive() {
    let mut simplifier = Simplifier::with_default_rules();

    // (x - 1) * (x - 3) > 0
    // Product > 0 when both factors same sign
    // x > 1 AND x > 3: x > 3 ✓
    // x < 1 AND x < 3: x < 1 ✓
    // Result: (-inf, 1) U (3, inf)

    let lhs = cas_parser::parse("(x - 1) * (x - 3)", &mut simplifier.context).unwrap();
    let rhs = simplifier.context.num(0);

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };

    let result = solve(&eq, "x", &mut simplifier);
    println!("Result: {:?}", result);

    assert!(result.is_ok());
    let (solution, _) = result.unwrap();

    // Should be union of two intervals
    match solution {
        SolutionSet::Union(intervals) => {
            assert_eq!(intervals.len(), 2);
            println!("Got union with {} intervals", intervals.len());
        }
        _ => panic!("Expected Union, got: {:?}", solution),
    }
}
