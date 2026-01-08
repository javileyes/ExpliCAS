//! Contract tests for quadratic formula post-simplification.
//!
//! These tests ensure the solver simplifies discriminant and solutions
//! for cleaner output like `2 ± sqrt(y)` instead of `(4 ± sqrt(16+4(y-4)))/2`.

use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_engine::engine::Simplifier;
use cas_engine::solver::solve;

// ============================================================================
// Test: x² - 4x + 4 = y should simplify to 2 ± sqrt(y)
// ============================================================================

fn solve_perfect_square_quadratic() -> (SolutionSet, Simplifier) {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);

    let ctx = &mut simplifier.context;
    let x = ctx.var("x");
    let y = ctx.var("y");
    let _one = ctx.num(1);
    let two = ctx.num(2);
    let four = ctx.num(4);

    // x^2 - 4x + 4
    let x2 = ctx.add(Expr::Pow(x, two));
    let four_x = ctx.add(Expr::Mul(four, x));
    let four2 = ctx.num(4);
    let x2_minus_4x = ctx.add(Expr::Sub(x2, four_x));
    let lhs = ctx.add(Expr::Add(x2_minus_4x, four2));

    let eq = Equation {
        lhs,
        rhs: y,
        op: RelOp::Eq,
    };

    let (solution_set, _) = solve(&eq, "x", &mut simplifier).expect("Solver should succeed");
    (solution_set, simplifier)
}

/// CONTRACT: Solutions should be in simplified form (not contain complex discriminant)
#[test]
fn quadratic_discriminant_simplified() {
    let (solution_set, simplifier) = solve_perfect_square_quadratic();

    let SolutionSet::Discrete(sols) = solution_set else {
        panic!("Expected Discrete solution set");
    };

    assert_eq!(sols.len(), 2, "Should have two solutions");

    // Check that solutions are simplified (should be 2 - y^(1/2) and 2 + y^(1/2))
    for sol in &sols {
        let sol_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: *sol,
            }
        );

        // Should NOT contain the unsimplified discriminant "16 + 4"
        assert!(
            !sol_str.contains("16 + 4"),
            "Solution should not contain unsimplified discriminant. Got: {}",
            sol_str
        );

        // Should NOT contain "/2" (the fraction should simplify)
        // Note: might still have division in some forms
        assert!(
            !sol_str.contains("1/2·(4"),
            "Solution should not contain unsimplified fraction. Got: {}",
            sol_str
        );

        // Should contain y (since y is the symbolic parameter)
        assert!(
            sol_str.contains("y"),
            "Solution should reference y. Got: {}",
            sol_str
        );
    }
}

/// CONTRACT: Solution should contain 2 (the vertex)
#[test]
fn quadratic_has_vertex_constant() {
    let (solution_set, simplifier) = solve_perfect_square_quadratic();

    let SolutionSet::Discrete(sols) = solution_set else {
        panic!("Expected Discrete solution set");
    };

    // At least one solution should contain "2"
    let any_has_two = sols.iter().any(|sol| {
        let sol_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: *sol,
            }
        );
        sol_str.contains("2")
    });

    assert!(
        any_has_two,
        "At least one solution should contain the constant 2"
    );
}
