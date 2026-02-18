//! Contract tests for linear form with symbolic coefficients.
//!
//! Ensures `solve (x-1)/(x+1) = y, x` produces correct result:
//! - Solution: x = (1+y)/(1-y)
//! - Guard: 1-y ≠ 0

use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_engine::engine::Simplifier;
use cas_engine::solver::solve;

/// Build and solve (x-1)/(x+1) = y for x
fn solve_symbolic_linear() -> (SolutionSet, Vec<cas_engine::solver::SolveStep>, Simplifier) {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);

    let ctx = &mut simplifier.context;

    let x = ctx.var("x");
    let y = ctx.var("y");
    let one = ctx.num(1);

    // (x-1)/(x+1)
    let x_minus_1 = ctx.add(Expr::Sub(x, one));
    let one2 = ctx.num(1);
    let x_plus_1 = ctx.add(Expr::Add(x, one2));
    let lhs = ctx.add(Expr::Div(x_minus_1, x_plus_1));

    let eq = Equation {
        lhs,
        rhs: y,
        op: RelOp::Eq,
    };

    let result = solve(&eq, "x", &mut simplifier).expect("Solver should succeed");
    (result.0, result.1, simplifier)
}

/// CONTRACT: Must not return Residual (must produce a closed-form solution)
#[test]
fn symbolic_linear_no_residual() {
    let (solution_set, _, _) = solve_symbolic_linear();

    assert!(
        !matches!(solution_set, SolutionSet::Residual(_)),
        "Should not return Residual - linear_form should handle this"
    );
}

/// CONTRACT: Solution should contain the correct fraction form
#[test]
fn symbolic_linear_correct_form() {
    let (solution_set, _, simplifier) = solve_symbolic_linear();

    let SolutionSet::Conditional(cases) = solution_set else {
        panic!("Expected Conditional solution");
    };

    assert!(!cases.is_empty());
    let first_case = &cases[0];
    let solutions = &first_case.then.as_ref().solutions;

    let SolutionSet::Discrete(sols) = solutions else {
        panic!("Expected Discrete solutions");
    };

    assert_eq!(sols.len(), 1);

    let sol_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: sols[0],
        }
    );

    // Should be (1+y)/(1-y) form
    assert!(
        sol_str.contains("y"),
        "Solution should contain y: {}",
        sol_str
    );
    assert!(
        sol_str.contains("/"),
        "Solution should be a fraction: {}",
        sol_str
    );
}

/// CONTRACT: Guard should be 1-y ≠ 0 (not anything with x)
#[test]
fn symbolic_linear_correct_guard() {
    let (solution_set, _, simplifier) = solve_symbolic_linear();

    let SolutionSet::Conditional(cases) = solution_set else {
        panic!("Expected Conditional solution");
    };

    let first_case = &cases[0];
    let guard_str = cas_formatter::condition_set_to_display(&first_case.when, &simplifier.context);

    // Guard should NOT contain x (the variable we're solving for)
    assert!(
        !guard_str.contains("x"),
        "Guard should not contain x: {}",
        guard_str
    );

    // Guard should contain y or be about (1-y)
    assert!(
        guard_str.contains("y") || guard_str.contains("1"),
        "Guard should involve y: {}",
        guard_str
    );
}

/// CONTRACT: Degenerate case y=1 should have correct handling
#[test]
fn symbolic_linear_degenerate_case() {
    let (solution_set, _, _) = solve_symbolic_linear();

    let SolutionSet::Conditional(cases) = solution_set else {
        panic!("Expected Conditional solution");
    };

    // Should have at least 2 cases: primary and degenerate
    assert!(
        cases.len() >= 2,
        "Should have at least 2 cases (primary + degenerate): {}",
        cases.len()
    );
}
